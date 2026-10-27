"""Independent analytical frequency, conductor-Q, field-shape and refinement checks."""
from pathlib import Path
import json
import numpy as np
from scipy.constants import c, mu_0
from scipy.special import jv, jvp, jnp_zeros

root = Path(__file__).resolve().parents[1]
a, length, sigma = .0525, .070, 58e6
x = float(jnp_zeros(1, 1)[0])
kc, kz = x/a, 2*np.pi/length
f = c/(2*np.pi)*np.sqrt(kc*kc+kz*kz)
rs = np.sqrt(np.pi*f*mu_0/sigma)
# For Hz = J1(kc*r)*cos(phi)*sin(kz*z), J1(x)^2 cancels in Q.
radial = a*a/2*(1-1/(x*x))
volume = np.pi*length/2*radial*(1+(kz/kc)**2)
sidewall = np.pi*a*length/2*(1+(kz*a/(x*x))**2)
endcaps = 2*np.pi*(kz/kc)**2*radial
geometry_factor = 2*np.pi*f*mu_0*volume/(sidewall+endcaps)
reference = dict(frequency_hz=f, unloaded_q=geometry_factor/rs,
                 geometry_factor_ohms=geometry_factor, surface_resistance_ohms=rs,
                 skin_depth_m=1/(sigma*rs), sidewall_loss_fraction=sidewall/(sidewall+endcaps),
                 formula="Q0 = omega*mu0*integral_V(|H|^2) / (Rs*integral_S(|Ht|^2))",
                 assumptions="TE112 PEC field with first-order smooth copper surface loss, 58 MS/m, mu_r=1")

def analytic_h_basis(xx, yy, zz, rotation):
    radius = np.hypot(xx, yy)
    phi = np.arctan2(yy, xx)
    angle = phi-rotation
    j = jv(1, kc*radius)
    jp = jvp(1, kc*radius)
    j_over_r = np.divide(j, radius, out=np.full_like(radius, kc/2), where=radius > 1e-12)
    hr = kz/kc*jp*np.cos(angle)*np.cos(kz*zz)
    hp = -kz/(kc*kc)*j_over_r*np.sin(angle)*np.cos(kz*zz)
    hz = j*np.cos(angle)*np.sin(kz*zz)
    return np.stack((hr*np.cos(phi)-hp*np.sin(phi), hr*np.sin(phi)+hp*np.cos(phi), hz), axis=-1)

report = {"analytic_reference": reference, "cases": {}}
for case in ("coarse", "refined"):
    folder = root / "results" / case
    summary = json.loads((folder / "summary.json").read_text())
    with np.load(folder / "sampled_fields.npz") as data:
        xyz = np.meshgrid(data["x"], data["y"], data["z"], indexing="ij")
        basis = [analytic_h_basis(*xyz, angle) for angle in (0, np.pi/2)]
        checked = []
        for index, row in enumerate(summary["modes"]):
            if row["analytic_family"] != "TE112":
                continue
            h, e = data["H"][index], data["E"][index]
            valid = np.isfinite(h).all(axis=-1) & np.isfinite(e).all(axis=-1)
            design = np.column_stack([b[valid].ravel() for b in basis])
            measured = h[valid].ravel()
            coeff = np.linalg.lstsq(design, measured, rcond=None)[0]
            residual = np.linalg.norm(measured-design@coeff)/np.linalg.norm(measured)
            ez_fraction = np.sum(np.abs(e[valid, 2])**2)/np.sum(np.abs(e[valid])**2)
            q = row["unloaded_q"]
            checked.append(dict(mode=index+1, frequency_ghz=row["frequency_ghz"], unloaded_q=q,
                                q_error_percent=100*(q/reference["unloaded_q"]-1),
                                frequency_error_percent=row["error_percent"],
                                magnetic_shape_relative_l2_error=float(residual),
                                electric_axial_energy_fraction=float(ez_fraction)))
        assert len(checked) == 2, "Both TE112 partners must be retained"
        assert all(row["unloaded_q"] > 0 for row in checked)
        assert all(abs(row["q_error_percent"]) < 1 for row in checked)
        assert all(abs(row["frequency_error_percent"]) < .1 for row in checked)
        assert all(row["magnetic_shape_relative_l2_error"] < .03 for row in checked)
        report["cases"][case] = dict(tetrahedra=summary["tetrahedra"], modes=checked,
                                     numerical_partner_splitting_hz=1e9*abs(checked[0]["frequency_ghz"]-checked[1]["frequency_ghz"]))

coarse, refined = [report["cases"][case]["modes"] for case in ("coarse", "refined")]
report["refinement"] = {
    "mean_frequency_change_hz": 1e9*(np.mean([r["frequency_ghz"] for r in refined])-np.mean([r["frequency_ghz"] for r in coarse])),
    "mean_q_change_percent": 100*(np.mean([r["unloaded_q"] for r in refined])/np.mean([r["unloaded_q"] for r in coarse])-1),
    "scope": "Two meshes; practical geometry-plus-field refinement check, not an asymptotic error bound. Six nearby eigenmodes retained, not a complete spectrum."
}
target = root / "validation"
target.mkdir(exist_ok=True)
(target / "analytical-and-mesh-validation.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
