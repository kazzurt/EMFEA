"""WR340, 18-inch air-filled PEC straight guide, analytical TE10 wave ports.

Raw two-port S matrix; fixed mesh generated using a 4 GHz design frequency.
All E/H samples correspond to 1 W incident at port 1 and no incident port-2 wave.
"""
from __future__ import annotations
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="sweep")
    parser.add_argument("--mesh-mm", type=float, default=9)
    parser.add_argument("--frequencies-ghz", help="Comma-separated discrete frequencies; default 2:0.05:4 GHz")
    args = parser.parse_args()
    if args.case in (".", "..") or Path(args.case).name != args.case:
        parser.error("Use a new, simple case name")
    if args.mesh_mm <= 0 or Path(args.case).name != args.case:
        parser.error("Use a positive mesh size and a simple case name")
    project = Path(__file__).resolve().parents[1]
    result = project / "results" / args.case
    result.mkdir(parents=True, exist_ok=False)  # preserve previous runs
    geometry = project / "geometry"
    geometry.mkdir(exist_ok=True)
    os.environ["EMERGE_BASE_PATH"] = str(result)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "4"
    libdir = Path(sys.prefix) / "Library" / "bin"
    os.environ["EMERGE_PARDISO_PATH"] = str(libdir / "mkl_rt.3.dll")
    dll_handle = os.add_dll_directory(str(libdir))
    os.chdir(result)

    import numpy as np
    from scipy.constants import c, mu_0
    import psutil
    import emerge as em
    import gmsh

    frequencies = (np.array([float(v)*1e9 for v in args.frequencies_ghz.split(",")])
                   if args.frequencies_ghz else np.linspace(2e9, 4e9, 41))
    frequencies = np.unique(np.round(frequencies))
    if np.any((frequencies < 2e9) | (frequencies > 4e9)):
        parser.error("This benchmark's mesh and port validation are scoped to 2–4 GHz")
    a, b, length = .08636, .04318, .4572
    started = time.perf_counter()
    model = em.Simulation("wr340-18inch-" + args.case, save_file=True)
    model.settings.mw_cap_sp_single = False
    model.settings.mw_cap_sp_col = False
    model.settings.mw_recip_sp = False
    model.set_solver(em.EMSolver.PARDISO)
    model.mw.set_frequency(4e9)  # Fixed design mesh sized for the top of the band.
    guide = em.geo.Box(a, b, length, name="WR340 air volume")
    guide.set_material(em.Material(er=1., ur=1., tand=0., cond=0., name="Ideal lossless air"))
    model.commit_geometry()
    cs1 = em.CoordinateSystem((1,0,0), (0,-1,0), (0,0,-1), np.array([a/2,b/2,0.]))
    cs2 = em.CoordinateSystem((1,0,0), (0,1,0), (0,0,1), np.array([a/2,b/2,length]))
    p1 = model.mw.bc.RectangularWaveguide(guide.bottom, 1, mode=(1,0), mode_type="TE", dims=(a,b), cs=cs1, power=1.)
    p2 = model.mw.bc.RectangularWaveguide(guide.top, 2, mode=(1,0), mode_type="TE", dims=(a,b), cs=cs2, power=1.)
    p1.align_modes(em.YAX)
    p2.align_modes(em.YAX)
    model.mw.bc.PEC(guide.outside("bottom", "top"))
    model.mesher.set_global_max_size(args.mesh_mm/1000)
    model.generate_mesh()
    gmsh.write(str(result / "waveguide.msh"))
    step = geometry / "wr340-18inch-air.step"
    model.export(str(step))
    step_text = step.read_text(encoding="utf-8")
    unit = "SI_UNIT(.MILLI.,.METRE.)"
    if step_text.count(unit) != 1:
        raise RuntimeError("Unexpected STEP unit declaration")
    step.write_text(step_text.replace(unit, "SI_UNIT($,.METRE.)"), encoding="utf-8")
    model.mw.set_frequency(frequencies)
    stop_monitor = threading.Event()
    def monitor():
        process = psutil.Process()
        while not stop_monitor.wait(30):
            print(f"RUNTIME elapsed={time.perf_counter()-started:.1f}s RSS={process.memory_info().rss/2**30:.3f}GiB CPU={sum(process.cpu_times()[:2]):.1f}s", flush=True)
    threading.Thread(target=monitor, daemon=True).start()
    print(f"BENCHMARK solving {len(frequencies)} frequencies, raw two-port S, explicit PARDISO", flush=True)
    solve_started = time.perf_counter()
    try:
        data = model.mw.run_sweep(parallel=False, frequency_groups=1)
    finally:
        stop_monitor.set()
    solve_seconds = time.perf_counter()-solve_started
    model.save()
    scalars = sorted(list(data.scalar), key=lambda item: item.freq)
    freqs = np.array([item.freq for item in scalars], dtype=float)
    smat = np.array([[[item.S(i,j) for j in (1,2)] for i in (1,2)] for item in scalars])
    if not np.isfinite(smat).all():
        raise RuntimeError("Non-finite S parameters")
    beta = np.sqrt((2*np.pi*freqs/c)**2 - (np.pi/a)**2)
    analytic_s21 = np.exp(-1j*beta*length)
    phase_error = np.angle(smat[:,1,0] / analytic_s21, deg=True)
    power_sum = np.abs(smat[:,0,0])**2 + np.abs(smat[:,1,0])**2
    np.savez_compressed(result / "sparameters.npz", frequencies_hz=freqs, S=smat,
                        S21_analytic=analytic_s21, beta_rad_per_m=beta)
    rows = []
    for i, freq in enumerate(freqs):
        row = {"frequency_ghz": float(freq/1e9)}
        for label, r, col in (("S11",0,0),("S21",1,0),("S12",0,1),("S22",1,1)):
            value = smat[i,r,col]
            row.update({label+"_real": float(value.real), label+"_imag": float(value.imag),
                        label+"_db": float(20*np.log10(max(abs(value),1e-15)))})
        row["S21_phase_error_deg"] = float(phase_error[i])
        row["output_power_sum_port1"] = float(power_sum[i])
        rows.append(row)
    with (result / "sparameters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "configuration": vars(args), "dimensions_m": {"a":a,"b":b,"length":length},
        "mode": "TE10, standard axes; dominant propagating mode confirmed by user",
        "material": {"epsilon_r":1,"mu_r":1,"loss_tangent":0,"conductivity_s_per_m":0},
        "walls": "PEC sidewalls", "ports": "Analytical rectangular TE10 wave-port conditions at both ends",
        "termination_scope": "Matched to selected TE10; not a general multimode absorbing boundary",
        "incident_power_w": 1., "field_excitation": {"port1":1,"port2":0},
        "reference_planes_m": [0.,length], "mesh_design_frequency_hz":4e9,
        "frequencies_hz": freqs.tolist(), "tetrahedra":int(list(data.field)[0].mesh.tets.shape[1]),
        "solver":"Intel MKL PARDISO; serial frequency dispatch, one matrix group at a time",
        "raw_s_parameters":True, "solve_seconds":solve_seconds,
        "peak_process_working_set_gib":getattr(psutil.Process().memory_info(),"peak_wset",0)/2**30,
        "max_S11_db":float(max(r["S11_db"] for r in rows)),
        "max_abs_power_balance_error":float(np.max(np.abs(power_sum-1))),
        "max_abs_reciprocity_error":float(np.max(np.abs(smat[:,1,0]-smat[:,0,1]))),
        "max_abs_S21_phase_error_deg":float(np.max(np.abs(phase_error))),
        "cutoffs_ghz":{"TE10":c/(2*a)/1e9,"TE20_TE01":c/a/1e9,"TE11_TM11":c/2*np.sqrt(a**-2+b**-2)/1e9},
        "python":platform.python_version(),
        "packages":{name:importlib.metadata.version(name) for name in ("emerge","gmsh","numpy","scipy","pyvista")},
    }
    target = [field for field in data.field if abs(field.freq-2.45e9)<1]
    if target:
        field = target[0]
        field.excite_port(1)
        x = np.linspace(a*1e-6,a*(1-1e-6),33)
        y = np.linspace(b*1e-6,b*(1-1e-6),21)
        z = np.linspace(length*1e-6,length*(1-1e-6),201)
        X,Y,Z = np.meshgrid(x,y,z,indexing="ij")
        field.interpolate(X,Y,Z)
        E = np.stack([field.Ex,field.Ey,field.Ez],axis=-1)
        H = np.stack([field.Hx,field.Hy,field.Hz],axis=-1)
        if not np.isfinite(E).all() or not np.isfinite(H).all():
            raise RuntimeError("Unexpected nonfinite field sample inside rectangular guide")
        Savg = .5*np.real(np.cross(E,np.conjugate(H)))
        net_power = np.trapezoid(np.trapezoid(Savg[...,2],x=x,axis=0),x=y,axis=0)
        np.savez_compressed(result / "fields_2p45GHz.npz", x=x,y=y,z=z,E=E,H=H,Savg=Savg,
                            frequency_hz=2.45e9,incident_power_w=1.,net_power_z_w=net_power)
        summary["field_sampling_shape"] = list(X.shape)
        summary["sampled_net_power_z_w_min_max"] = [float(net_power.min()),float(net_power.max())]
        summary["sampled_E_magnitude_peak_v_per_m"] = float(np.linalg.norm(E,axis=-1).max())
        beta0 = np.sqrt((2*np.pi*2.45e9/c)**2 - (np.pi/a)**2)
        impedance = 2*np.pi*2.45e9*mu_0/beta0
        summary["analytic_at_2p45GHz"] = {"beta_rad_per_m":float(beta0),"guide_wavelength_m":float(2*np.pi/beta0),
                                           "TE_impedance_ohm":float(impedance),"E_peak_v_per_m_for_1W":float(np.sqrt(4*impedance/(a*b)))}
    summary["elapsed_seconds_including_exports"] = time.perf_counter()-started
    (result / "summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)
    print("BENCHMARK complete",flush=True)


if __name__ == "__main__":
    main()
