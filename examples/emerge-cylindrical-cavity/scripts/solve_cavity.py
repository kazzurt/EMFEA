"""Reproducible EMerge eigenmode benchmark: 100 mm diameter x 100 mm air/PEC cavity.

Run using the dedicated Python environment via run-cavity.ps1.
All lengths passed to EMerge/Gmsh are in meters. Eigenmode amplitudes are arbitrary.
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
import time
import threading


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="coarse")
    parser.add_argument("--mesh-mm", type=float, default=10.0)
    parser.add_argument("--curve-points", type=int, default=48)
    parser.add_argument("--length-mm", type=float, default=100.0)
    parser.add_argument("--diameter-mm", type=float, default=100.0)
    parser.add_argument("--modes", type=int, default=10)
    parser.add_argument("--search-ghz", type=float, default=3.0)
    parser.add_argument("--targeted-arpack", action="store_true", help="Benchmark-specific reduced eigenvalue oversampling")
    args = parser.parse_args()
    if args.case in (".", "..") or Path(args.case).name != args.case:
        parser.error("Use a new, simple case name")
    if args.targeted_arpack and (args.length_mm != 100 or args.diameter_mm != 100 or args.search_ghz != 3 or args.modes != 10):
        parser.error("--targeted-arpack is restricted to this 100x100mm, 3GHz, ten-mode benchmark")
    project = Path(__file__).resolve().parents[1]
    result_dir = project / "results" / args.case
    result_dir.mkdir(parents=True, exist_ok=False)  # preserve previous runs
    geometry_dir = project / "geometry"
    geometry_dir.mkdir(exist_ok=True)
    os.environ["EMERGE_BASE_PATH"] = str(result_dir)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.chdir(result_dir)

    import numpy as np
    from scipy.constants import c
    from scipy.special import jn_zeros, jnp_zeros
    from scipy.optimize import linear_sum_assignment
    import psutil
    import emerge as em
    import gmsh

    start = time.perf_counter()
    radius, length = args.diameter_mm / 2000, args.length_mm / 1000
    model = em.Simulation("air-pec-cylinder-" + args.case, save_file=True)
    model.settings.mw_cap_sp_single = False
    model.settings.mw_cap_sp_col = False
    model.settings.mw_recip_sp = False
    air = em.Material(er=1.0, ur=1.0, tand=0.0, cond=0.0, name="Ideal lossless air")
    cavity = em.geo.Cylinder(radius=radius, height=length, cs=em.GCS).set_material(air)
    model.commit_geometry()
    model.mw.set_frequency(args.search_ghz * 1e9)
    model.mesher.set_global_max_size(args.mesh_mm / 1000)
    model.mesher.set_curved_boundary_meshing(args.curve_points)
    model.generate_mesh()
    # EMerge assigned every exterior face PEC during commit_geometry().
    step_path = geometry_dir / "air-pec-cylinder.step"
    model.export(str(step_path))
    # Gmsh's STEP writer labels our meter-valued coordinates as millimeters.
    # Correct the unit declaration, preserving every CAD coordinate exactly.
    step_text = step_path.read_text(encoding="utf-8")
    millimeter_unit = "SI_UNIT(.MILLI.,.METRE.)"
    if step_text.count(millimeter_unit) != 1:
        raise RuntimeError("Unexpected STEP unit declaration; inspect before export use")
    step_path.write_text(step_text.replace(millimeter_unit, "SI_UNIT($,.METRE.)"), encoding="utf-8")
    gmsh.write(str(result_dir / "cavity.msh"))

    print("BENCHMARK: solving eigenmodes", flush=True)
    adapter = None
    if args.targeted_arpack:
        from targeted_arpack import BENCHMARK_ID, TargetedCylinderARPACK
        adapter = TargetedCylinderARPACK(benchmark_id=BENCHMARK_ID)
        model.mw.solveroutine.set_solver(adapter)
    monitor_done = threading.Event()
    def monitor():
        process = psutil.Process()
        while not monitor_done.wait(30):
            print(f"RUNTIME: elapsed={time.perf_counter()-start:.0f}s, RSS={process.memory_info().rss/1024**3:.2f}GiB, CPU={sum(process.cpu_times()[:2]):.0f}s", flush=True)
    threading.Thread(target=monitor, daemon=True).start()
    data = model.mw.eigenmode(args.search_ghz * 1e9, nmodes=args.modes)
    monitor_done.set()
    # v3.0.0b6 eigenmode() does not set the modified flag used by save().
    # Explicitly mark our completed solution; no installed-package patch needed.
    model.state.set_modified()
    model.save()
    fields = sorted(list(data.field), key=lambda field: field.freq.real)
    frequencies = np.array([field.freq for field in fields], dtype=complex)

    analytic = []
    for family in ("TE", "TM"):
        for m in range(5):
            roots = jnp_zeros(m, 3) if family == "TE" else jn_zeros(m, 3)
            for n, root in enumerate(roots, 1):
                for p in range(1 if family == "TE" else 0, 5):
                    frequency = c / (2*np.pi) * np.sqrt((root/radius)**2+(p*np.pi/length)**2)
                    for partner in range(2 if m else 1):
                        analytic.append({"family": f"{family}{m}{n}{p}", "partner": partner+1, "frequency_hz": float(frequency)})
    analytic.sort(key=lambda item: item["frequency_hz"])
    rows, cols = linear_sum_assignment(np.abs(frequencies.real[:, None] - np.array([a["frequency_hz"] for a in analytic])[None, :]))
    assignment = dict(zip(rows, cols))
    table = []
    for index, frequency in enumerate(frequencies):
        ref = analytic[assignment[index]]
        row = {"mode": index+1, "analytic_family": ref["family"], "partner": ref["partner"],
               "frequency_ghz": float(frequency.real/1e9), "imag_frequency_hz": float(frequency.imag),
               "analytic_ghz": ref["frequency_hz"]/1e9,
               "error_percent": float(100*(frequency.real/ref["frequency_hz"]-1))}
        table.append(row)
    with (result_dir / "eigenmodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    summary = {"configuration": vars(args), "material": {"epsilon_r": 1, "mu_r": 1, "loss_tangent": 0},
               "walls": "PEC on every exterior face", "ports": "none (closed eigenmode study)",
               "ideal_q": "infinite", "amplitudes": "arbitrary eigenvector normalization",
               "mode_identification": "minimum-distance assignment to analytic frequencies; field verification separate",
               "python": platform.python_version(),
               "packages": {name: importlib.metadata.version(name) for name in ("emerge", "emsutil", "gmsh", "numpy", "scipy", "pyvista", "vtk")},
               "elapsed_seconds": time.perf_counter()-start,
               "eigensolver": "TargetedCylinderARPACK-k14-ncv56" if adapter else "EMerge default ARPACK-k30-ncv120",
               "solver_diagnostics": adapter.last_report if adapter else None,
               "peak_process_working_set_gib": getattr(psutil.Process().memory_info(), "peak_wset", 0)/1024**3,
               "process_rss_gib": psutil.Process().memory_info().rss/1024**3,
               "system_total_ram_gib": psutil.virtual_memory().total/1024**3,
               "modes": table}
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    np.savez_compressed(result_dir / "eigenfrequencies.npz", frequencies=frequencies)
    # A regular interior grid makes the solved vector fields portable without
    # depending on EMerge's prerelease model-state serialization.
    grid_x = np.linspace(-0.98*radius, 0.98*radius, 35)
    grid_y = np.linspace(-0.98*radius, 0.98*radius, 35)
    grid_z = np.linspace(0.01*length, 0.99*length, 51)
    x, y, z = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    electric, magnetic = [], []
    for index, field in enumerate(fields):
        print(f"BENCHMARK: sampling mode {index+1}/{len(fields)}", flush=True)
        field.interpolate(x, y, z)
        electric.append(np.stack([field.Ex, field.Ey, field.Ez], axis=-1))
        magnetic.append(np.stack([field.Hx, field.Hy, field.Hz], axis=-1))
    np.savez_compressed(result_dir / "sampled_fields.npz", x=grid_x, y=grid_y, z=grid_z,
                        E=np.stack(electric), H=np.stack(magnetic), frequencies=frequencies)
    summary["sampled_field_grid"] = list(x.shape)
    summary["tetrahedra"] = int(fields[0].mesh.tets.shape[1])
    summary["elapsed_seconds_including_sampling"] = time.perf_counter()-start
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("BENCHMARK: complete", flush=True)


if __name__ == "__main__":
    main()
