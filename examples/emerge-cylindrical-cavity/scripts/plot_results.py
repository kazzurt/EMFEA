"""Plot saved cavity eigenfields and compare two meshes with analytic frequencies.

Run after solve_cavity.py has completed both coarse and refined cases. This only
reads the solver results; figures and validation tables can be rebuilt without
EMerge. Eigenvectors have arbitrary amplitude and phase. No driven powers or
S-parameters are inferred from these closed-cavity eigenmode results.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.special import j0, jn_zeros


BLUE = "#205b93"
ORANGE = "#c66b28"
INK = "#25364a"


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for name in ("mode", "partner"):
            row[name] = int(row[name])
        for name in ("frequency_ghz", "analytic_ghz", "error_percent"):
            row[name] = float(row[name])
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def magnitude(field: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.abs(field)**2, axis=-1))


def phase_reference(field: np.ndarray) -> np.ndarray:
    """Choose one RF phase by making the largest electric entry real positive."""
    peak = field.ravel()[np.nanargmax(np.abs(field))]
    return field * np.exp(-1j*np.angle(peak))


def axial_fraction(field: np.ndarray) -> float:
    return float(np.nansum(np.abs(field[..., 2])**2) /
                 np.nansum(np.abs(field)**2))


def envelope_error(field: np.ndarray, z: np.ndarray, length: float,
                   p: int) -> float:
    """Relative L2 mismatch of transverse-integrated |E|^2 and sin^2(p*pi*z/L).

    For p=0 use the constant TM010 envelope. A fitted amplitude removes the
    arbitrary eigenvector normalization. This is a sampled field diagnostic,
    not an independent rigorous FEM error estimator.
    """
    measured = np.nansum(np.abs(field)**2, axis=(0, 1, 3))
    expected = np.sin(p*np.pi*z/length)**2 if p else np.ones_like(z)
    fitted = expected * (np.dot(measured, expected) / np.dot(expected, expected))
    return float(np.linalg.norm(measured-fitted) / np.linalg.norm(measured))


def format_axis(axis: plt.Axes, xlabel: str, ylabel: str) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=INK)


def compare_meshes(project: Path, coarse_name: str, fine_name: str,
                   figures: Path, validation: Path) -> tuple[list[dict], dict]:
    coarse = read_rows(project / "results" / coarse_name / "eigenmodes.csv")
    fine = read_rows(project / "results" / fine_name / "eigenmodes.csv")
    single_mesh = coarse_name == fine_name
    fine_label = "Same mesh (preview)" if single_mesh else "Refined"
    coarse_map = {(row["analytic_family"], row["partner"]): row for row in coarse}
    comparison = []
    for row in fine:
        key = (row["analytic_family"], row["partner"])
        other = coarse_map[key]
        comparison.append({
            "analytic_family": key[0], "partner": key[1],
            "analytic_ghz": row["analytic_ghz"],
            "coarse_ghz": other["frequency_ghz"],
            "refined_ghz": row["frequency_ghz"],
            "coarse_error_percent": other["error_percent"],
            "refined_error_percent": row["error_percent"],
            "mesh_change_percent_of_analytic":
                100*(row["frequency_ghz"]-other["frequency_ghz"])/row["analytic_ghz"],
        })
    write_rows(validation / "eigenfrequency-comparison.csv", comparison)

    # Within an ideal degenerate family the solver chooses its own basis. The
    # frequency-ordered partners below do not imply tracked physical orientation.
    labels = [r["analytic_family"] + (f" ({r['partner']})" if
              sum(q["analytic_family"] == r["analytic_family"] for q in fine)>1 else "")
              for r in fine]
    indexes = np.arange(len(fine))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    analytic = np.array([r["analytic_ghz"] for r in comparison])
    ax.plot(analytic, [r["coarse_ghz"] for r in comparison], "o",
            ms=6, color=ORANGE, label="Coarse")
    ax.plot(analytic, [r["refined_ghz"] for r in comparison], "x",
            ms=6, color=BLUE, label=fine_label)
    lim = [analytic.min()-.08, analytic.max()+.08]
    ax.plot(lim, lim, color="#81909f", lw=1, zorder=0, label="Exact agreement")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Eigenfrequencies")
    format_axis(ax, "Analytic frequency (GHz)", "FEA frequency (GHz)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.15)
    ax = axes[1]
    ax.bar(indexes-.19, [r["coarse_error_percent"] for r in comparison],
           width=.36, color=ORANGE, label="Coarse")
    ax.bar(indexes+.19, [r["refined_error_percent"] for r in comparison],
           width=.36, color=BLUE, label=fine_label)
    ax.axhline(0, color="#8897a6", lw=.8)
    ax.set_xticks(indexes, labels, rotation=50, ha="right", fontsize=9)
    ax.set_title("Signed error relative to analytic solution")
    format_axis(ax, "Frequency-assigned mode", "Frequency error (%)")
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    title = "single-mesh preview" if single_mesh else "mesh refinement check"
    fig.suptitle(f"Air / PEC cylinder  |  {title}", fontsize=15, color=INK)
    fig.savefig(figures / "eigenfrequency-convergence.png", dpi=190)
    plt.close(fig)

    families = {}
    for name in dict.fromkeys(row["analytic_family"] for row in fine):
        subset = [r for r in fine if r["analytic_family"] == name]
        freq = np.array([r["frequency_ghz"] for r in subset])
        families[name] = {
            "analytic_ghz": subset[0]["analytic_ghz"],
            "refined_mean_ghz": float(freq.mean()),
            "refined_max_abs_error_percent": max(abs(r["error_percent"]) for r in subset),
            "numerical_partner_splitting_hz": float(np.ptp(freq)*1e9) if len(freq)>1 else None,
        }
    metrics = {
        "coarse_case": coarse_name, "refined_case": fine_name,
        "refinement_completed": not single_mesh,
        "coarse_max_abs_frequency_error_percent": max(abs(r["error_percent"]) for r in coarse),
        "refined_max_abs_frequency_error_percent": max(abs(r["error_percent"]) for r in fine),
        "all_frequency_errors_reduced": all(abs(r["refined_error_percent"]) <
                                            abs(r["coarse_error_percent"]) for r in comparison),
        "families": families,
        "identification_note": "Labels initially assigned by frequency; field diagnostics are separate. "
                               "Degenerate partner orientation is arbitrary and not tracked between meshes.",
    }
    return fine, metrics


def plot_fields(project: Path, case: str, rows: list[dict], figures: Path,
                validation: Path, metrics: dict) -> None:
    result_dir = project / "results" / case
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    radius = summary["configuration"]["diameter_mm"] / 2000
    length = summary["configuration"]["length_mm"] / 1000
    with np.load(result_dir / "sampled_fields.npz") as saved:
        x, y, z = (saved[k].copy() for k in ("x", "y", "z"))
        electric, magnetic, frequencies = (saved[k].copy() for k in ("E", "H", "frequencies"))
    expected = (len(rows), x.size, y.size, z.size, 3)
    if electric.shape != expected or magnetic.shape != expected:
        raise ValueError(f"Expected E and H shape {expected}; got {electric.shape}, {magnetic.shape}")
    if not np.allclose(frequencies.real/1e9, [r["frequency_ghz"] for r in rows],
                       rtol=1e-10, atol=1e-10):
        raise ValueError("Field frequency order does not match eigenmodes.csv")
    if not all(np.any(np.isfinite(electric[i])) and np.any(np.isfinite(magnetic[i]))
               for i in range(len(rows))):
        raise ValueError("A sampled eigenfield has no finite values")

    diagnostics = []
    for i, row in enumerate(rows):
        diag = {
            "mode": row["mode"], "analytic_family": row["analytic_family"],
            "partner": row["partner"], "frequency_ghz": row["frequency_ghz"],
            "electric_axial_energy_fraction": axial_fraction(electric[i]),
            "magnetic_axial_energy_fraction": axial_fraction(magnetic[i]),
        }
        diagnostics.append(diag)
    write_rows(validation / "sampled-field-diagnostics.csv", diagnostics)
    metrics["sampled_field_grid"] = list(expected[1:4])
    metrics["field_diagnostic_note"] = (
        "Axial fractions use equal-volume sums on an interior Cartesian grid, excluding NaNs outside "
        "the cylinder. They are sampled diagnostics, not exact element integrals. "
        "TE means E_z=0; TM means H_z=0, with z along the cylinder axis.")

    iy = int(np.argmin(abs(y)))
    iz = int(np.argmin(abs(z-length/4)))
    it = next(i for i, row in enumerate(rows) if row["analytic_family"] == "TE112")
    E, H = electric[it], magnetic[it]
    emagnitude, hmagnitude = magnitude(E), magnitude(H)
    enorm = emagnitude / np.nanmax(emagnitude)
    hnorm = hmagnitude / np.nanmax(hmagnitude)
    real_E = phase_reference(E).real / np.nanmax(emagnitude)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.1), layout="constrained")
    specifications = [
        (enorm[:, iy, :].T, x*1000, z*1000, "Electric magnitude |E|", "z (mm)"),
        (enorm[:, :, iz].T, x*1000, y*1000, "Electric magnitude + RF snapshot", "y (mm)"),
        (hnorm[:, iy, :].T, x*1000, z*1000, "Magnetic magnitude |H|", "z (mm)"),
    ]
    for ax, (values, horizontal, vertical, title, ylabel) in zip(axes, specifications):
        image = ax.pcolormesh(horizontal, vertical, values, cmap="magma", vmin=0, vmax=1,
                              shading="auto", rasterized=True)
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_aspect("equal")
        format_axis(ax, "x (mm)", ylabel)
        colorbar = fig.colorbar(image, ax=ax, shrink=.74, pad=.025, ticks=[0, .5, 1])
        colorbar.set_label("Normalized magnitude", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    for ax in (axes[0], axes[2]):
        ax.set_xlim(-radius*1000, radius*1000)
        ax.set_ylim(0, length*1000)
        ax.text(.02, .98, f"y = {y[iy]*1000:.1f} mm", transform=ax.transAxes,
                va="top", color="white", fontsize=9)
    axes[0].axhline(z[iz]*1000, ls="--", lw=.8, color="white", alpha=.75)
    axes[1].add_patch(Circle((0, 0), radius*1000, fill=False, lw=1, color=INK))
    axes[1].set_xlim(-radius*1000, radius*1000)
    axes[1].set_ylim(-radius*1000, radius*1000)
    axes[1].text(.02, .98, f"z = {z[iz]*1000:.2f} mm", transform=axes[1].transAxes,
                 va="top", color=INK, fontsize=9)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    valid = xx**2+yy**2 < (.9*radius)**2
    sl = (slice(2, -2, 4), slice(2, -2, 4))
    u = np.where(valid, real_E[:, :, iz, 0], np.nan)[sl]
    v = np.where(valid, real_E[:, :, iz, 1], np.nan)[sl]
    axes[1].quiver(xx[sl]*1000, yy[sl]*1000, u, v, color="#ddf7fc", angles="xy",
                   scale_units="xy", scale=.075, width=.005, pivot="mid")
    fig.suptitle(f"TE112 eigenfield  |  {rows[it]['frequency_ghz']:.6f} GHz\n"
                 "One numerical partner; arbitrary eigenvector amplitude and orientation",
                 fontsize=14, color=INK)
    fig.savefig(figures / "te112-fields.png", dpi=190)
    plt.close(fig)
    metrics["TE112_field"] = {
        **diagnostics[it],
        "axial_electric_energy_envelope_relative_l2_error": envelope_error(E, z, length, 2),
        "transverse_plot_z_mm": float(z[iz]*1000),
        "phase_reference": "Largest complex electric entry rotated to real positive; arrows are Re(E).",
        "interpretation": "Two axial electric antinodes and a central node are expected. "
                          "A single eigenvector does not demonstrate driven rotation.",
    }

    im = next(i for i, row in enumerate(rows) if row["analytic_family"] == "TM010")
    E = electric[im]
    phased = phase_reference(E)
    mid = int(np.argmin(abs(z-length/2)))
    exact = j0(jn_zeros(0, 1)[0]*np.abs(x)/radius)
    measured = phased[:, iy, mid, 2].real
    amplitude = np.dot(measured, exact) / np.dot(exact, exact)
    radial_error = float(np.linalg.norm(measured-amplitude*exact)/np.linalg.norm(measured))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    image = axes[0].pcolormesh(x*1000, z*1000,
                             (magnitude(E)/np.nanmax(magnitude(E)))[:, iy, :].T,
                             shading="auto", cmap="magma", vmin=0, vmax=1)
    axes[0].set_aspect("equal")
    axes[0].set_xlim(-radius*1000, radius*1000)
    axes[0].set_ylim(0, length*1000)
    axes[0].set_title("Electric magnitude |E|, y = 0")
    format_axis(axes[0], "x (mm)", "z (mm)")
    fig.colorbar(image, ax=axes[0], shrink=.85, label="Normalized magnitude")
    axes[1].plot(x*1000, exact, color=INK, lw=1.7, label="Analytic J0 profile")
    axes[1].plot(x*1000, measured/amplitude, "o", ms=4, mfc="none", color=BLUE,
                 label="FEA, phase aligned")
    axes[1].set_title(f"Axial E component at z = {z[mid]*1000:.1f} mm")
    format_axis(axes[1], "x (mm)", "E_z / fitted analytic amplitude")
    axes[1].set_ylim(-.03, 1.08)
    axes[1].grid(alpha=.18)
    axes[1].legend(frameon=False, fontsize=9, loc="lower center")
    fig.suptitle(f"TM010 field check  |  {rows[im]['frequency_ghz']:.6f} GHz",
                 fontsize=15, color=INK)
    fig.savefig(figures / "tm010-field-validation.png", dpi=190)
    plt.close(fig)
    metrics["TM010_field"] = {
        **diagnostics[im], "radial_profile_relative_l2_error": radial_error,
        "axial_electric_energy_envelope_relative_l2_error": envelope_error(E, z, length, 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", default="coarse")
    parser.add_argument("--refined", default="refined")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    figures = project / "figures"
    validation = project / "results" / "validation"
    figures.mkdir(exist_ok=True)
    validation.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": INK, "text.color": INK,
                         "axes.edgecolor": "#9aa8b5", "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    rows, metrics = compare_meshes(project, args.coarse, args.refined, figures, validation)
    plot_fields(project, args.refined, rows, figures, validation, metrics)
    (validation / "validation-summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Figures: {figures}")
    print(f"Validation: {validation}")


if __name__ == "__main__":
    main()
