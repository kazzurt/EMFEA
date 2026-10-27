"""Plot raw WR340 S parameters and the actual 1 W TE10 driven fields.

Default inputs are results/sweep and results/coarse. No solver is run. The
straight, uniform, PEC guide has analytical S11=0 and S21=exp(-j*beta*L).
These analytical ports terminate the selected TE10 mode, not every mode that
can propagate above the higher-mode cutoffs. No S-parameter clipping is applied.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.constants import c, mu_0


INK = "#24374a"
BLUE = "#165f9d"
ORANGE = "#bd6d29"
DISPLAY_FLOOR_DB = -180.0


def db(value: np.ndarray) -> np.ndarray:
    """Display floor only; complex data and validation metrics remain unchanged."""
    return 20*np.log10(np.maximum(np.abs(value), 10**(DISPLAY_FLOOR_DB/20)))


def scalar(value) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def load_case(project: Path, name: str) -> dict:
    directory = project / "results" / name
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with np.load(directory / "sparameters.npz") as saved:
        frequencies = saved["frequencies_hz"].copy()
        S = saved["S"].copy()
        analytic = saved["S21_analytic"].copy()
        beta = saved["beta_rad_per_m"].copy()
    if S.shape != (len(frequencies), 2, 2) or analytic.shape != frequencies.shape:
        raise ValueError(f"Unexpected S-parameter shape in {name}")
    if not np.isfinite(S).all() or not np.all(np.diff(frequencies)>0):
        raise ValueError(f"Nonfinite or nonordered S-parameter data in {name}")
    dims = summary["dimensions_m"]
    expected_beta = np.sqrt((2*np.pi*frequencies/c)**2-(np.pi/dims["a"])**2)
    if not np.allclose(beta, expected_beta, rtol=1e-12, atol=1e-12):
        raise ValueError(f"Saved beta disagrees with WR340 analytical TE10 dispersion: {name}")
    if not np.allclose(analytic, np.exp(-1j*expected_beta*dims["length"]), atol=1e-12):
        raise ValueError(f"Saved analytical S21 disagrees with port reference planes: {name}")
    return {"name": name, "directory": directory, "summary": summary,
            "f": frequencies, "S": S, "analytic": analytic, "beta": beta,
            "phase_error": np.angle(S[:, 1, 0]/analytic, deg=True),
            "power_balance_error": np.abs(S[:, 0, 0])**2+np.abs(S[:, 1, 0])**2-1}


def case_metrics(case: dict, cutoff: float) -> dict:
    S, f = case["S"], case["f"]
    worst = int(np.argmax(np.abs(S[:, 0, 0])))
    result = {
        "case": case["name"], "sample_count": len(f),
        "frequency_min_ghz": float(f.min()/1e9), "frequency_max_ghz": float(f.max()/1e9),
        "mesh_max_mm": case["summary"]["configuration"]["mesh_mm"],
        "tetrahedra": case["summary"]["tetrahedra"],
        "raw_s_parameters": case["summary"]["raw_s_parameters"],
        "worst_S11_magnitude": float(np.abs(S[worst, 0, 0])),
        "worst_S11_db": scalar(20*np.log10(np.abs(S[worst, 0, 0]))),
        "worst_S11_frequency_ghz": float(f[worst]/1e9),
        "max_abs_S21_magnitude_error_db": float(np.max(np.abs(db(S[:, 1, 0])))),
        "max_abs_S21_phase_error_deg": float(np.max(np.abs(case["phase_error"]))),
        "max_abs_port_power_balance_error": float(np.max(np.abs(case["power_balance_error"]))),
        "max_abs_reciprocity_error": float(np.max(np.abs(S[:, 1, 0]-S[:, 0, 1]))),
        "S11_points_below_display_floor": int(np.count_nonzero(np.abs(S[:, 0, 0]) < 10**(DISPLAY_FLOOR_DB/20))),
    }
    for label, mask in (("below_first_higher_mode_cutoff", f<cutoff),
                        ("at_or_above_first_higher_mode_cutoff", f>=cutoff)):
        result[label] = {"sample_count": int(mask.sum())}
        if mask.any():
            result[label].update({
                "worst_S11_db": scalar(20*np.log10(np.max(np.abs(S[mask, 0, 0])))),
                "max_abs_S21_phase_error_deg": float(np.max(np.abs(case["phase_error"][mask]))),
                "max_abs_port_power_balance_error": float(np.max(np.abs(case["power_balance_error"][mask]))),
            })
    return result


def axis_style(ax: plt.Axes, ylabel: str, xlabel: str | None = None) -> None:
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.16)
    ax.set_axisbelow(True)
    ax.ticklabel_format(useOffset=False, axis="y")


def shade_cutoffs(ax: plt.Axes, cutoffs: dict, labels: bool = False) -> None:
    first, second = cutoffs["TE20_TE01"], cutoffs["TE11_TM11"]
    ax.axvspan(first, 4.0, color="#d9dce0", alpha=.5, zorder=0)
    ax.axvline(first, color="#7d8995", ls="--", lw=1, zorder=1)
    ax.axvline(second, color="#9aa4ae", ls=":", lw=1, zorder=1)
    ax.set_xlim(2.0, 4.0)
    ax.set_xticks(np.arange(2.0, 4.01, .25))
    if labels:
        for value, title in ((first, "TE20 / TE01 cutoff"), (second, "TE11 / TM11 cutoff")):
            ax.text(value+.014, .055, title, rotation=90, transform=ax.get_xaxis_transform(),
                    fontsize=8, color="#4e5e6d", ha="left", va="bottom")


def case_label(case: dict) -> str:
    return f"{case['name'].capitalize()}: hmax = {case['summary']['configuration']['mesh_mm']:g} mm"


def plot_sparameters(main: dict, comparison: dict | None, cutoffs: dict,
                     figures: Path, validation: Path) -> None:
    datasets = [(main, BLUE, "o")]
    if comparison:
        datasets.insert(0, (comparison, ORANGE, "x"))
    fig, ax = plt.subplots(figsize=(11.8, 5.4))
    fig.subplots_adjust(left=.085, right=.98, bottom=.23, top=.80)
    shade_cutoffs(ax, cutoffs, labels=True)
    for case, color, marker in datasets:
        ax.plot(case["f"]/1e9, db(case["S"][:, 0, 0]), color=color, marker=marker,
                ms=3.8, lw=1.2, label=case_label(case))
    axis_style(ax, "S11 magnitude (dB)", "Frequency (GHz)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("WR340 reflection  |  18-inch straight air / PEC guide", fontsize=15, y=.96)
    ax.set_title(f"TE10 excitation and termination; {len(main['f'])} solved frequencies in {main['name']}",
                 fontsize=11, pad=13)
    fig.text(.085, .09, "Ideal uniform guide: S11 = 0 (−∞ dB). Finite plotted reflection is a numerical residual.", fontsize=10)
    fig.text(.085, .045, "Markers are solved frequencies; lines only connect them. Gray: higher modes can propagate; ports terminate TE10 only.", fontsize=9, color="#566779")
    fig.savefig(figures / "s11-reflection.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11.8, 8), sharex=True)
    fig.subplots_adjust(left=.115, right=.98, bottom=.14, top=.89, hspace=.24)
    for ax in axes:
        shade_cutoffs(ax, cutoffs)
        ax.axhline(0, color="#778795", lw=.9, ls="--", zorder=1)
    for case, color, marker in datasets:
        for ax, values in zip(axes, (db(case["S"][:, 1, 0]), case["phase_error"],
                                    100*case["power_balance_error"])):
            ax.plot(case["f"]/1e9, values, color=color, marker=marker, ms=3.5, lw=1.1,
                    label=case_label(case))
    axis_style(axes[0], "S21 magnitude (dB)")
    axis_style(axes[1], "S21 phase error (deg)")
    axis_style(axes[2], "Port power error (%)", "Frequency (GHz)")
    axes[0].legend(frameon=False, loc="lower left", fontsize=9)
    axes[0].set_title("Dashed zero lines are the analytical values", fontsize=10, loc="left")
    fig.suptitle("Transmission and power checks  |  analytical S21 = exp(−jβL)", fontsize=15, y=.975)
    fig.text(.115, .055, "Phase error = arg(S21 / S21,analytic). Port power error = 100 × (|S11|² + |S21|² − 1).", fontsize=9)
    fig.text(.115, .025, "These are raw TE10-channel results. Above 3.4714 GHz the boundary is not a general multimode termination.", fontsize=9, color="#566779")
    fig.savefig(figures / "transmission-validation.png", dpi=190)
    plt.close(fig)

    rows = []
    comparison_index = {} if comparison is None else {float(f): i for i, f in enumerate(comparison["f"])}
    for i, frequency in enumerate(main["f"]):
        row = {"frequency_ghz": float(frequency/1e9),
               "main_S11_db": float(db(main["S"][i, 0, 0])),
               "main_S21_db": float(db(main["S"][i, 1, 0])),
               "main_S21_phase_error_deg": float(main["phase_error"][i]),
               "main_port_power_balance_error": float(main["power_balance_error"][i])}
        if comparison:
            j = comparison_index.get(float(frequency))
            row.update({"comparison_S11_db": None if j is None else float(db(comparison["S"][j, 0, 0])),
                        "comparison_S21_db": None if j is None else float(db(comparison["S"][j, 1, 0])),
                        "comparison_S21_phase_error_deg": None if j is None else float(comparison["phase_error"][j]),
                        "comparison_port_power_balance_error": None if j is None else float(comparison["power_balance_error"][j])})
        rows.append(row)
    with (validation / "sparameter-validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_fields(main: dict, figures: Path, validation: Path) -> dict:
    path = main["directory"] / "fields_2p45GHz.npz"
    with np.load(path) as saved:
        x, y, z = (saved[key].copy() for key in ("x", "y", "z"))
        E, H, Savg = (saved[key].copy() for key in ("E", "H", "Savg"))
        power = saved["net_power_z_w"].copy()
        freq, incident = float(saved["frequency_hz"]), float(saved["incident_power_w"])
    expected = (len(x), len(y), len(z), 3)
    if E.shape != expected or H.shape != expected or Savg.shape != expected:
        raise ValueError(f"Field array dimensions do not match coordinate vectors: {expected}")
    if not all(np.isfinite(a).all() for a in (E, H, Savg, power)):
        raise ValueError("Nonfinite sampled field values")
    recalculated_S = .5*np.real(np.cross(E, np.conjugate(H)))
    if not np.allclose(Savg, recalculated_S, atol=1e-10, rtol=1e-11):
        raise ValueError("Saved Savg is not the time-averaged Poynting vector")
    integrated = np.trapezoid(np.trapezoid(Savg[..., 2], x=x, axis=0), x=y, axis=0)
    if not np.allclose(power, integrated, atol=1e-10, rtol=1e-10):
        raise ValueError("Saved cross-section power does not match the field integral")
    dims = main["summary"]["dimensions_m"]
    a, b, length = dims["a"], dims["b"], dims["length"]
    beta = np.sqrt((2*np.pi*freq/c)**2-(np.pi/a)**2)
    impedance = 2*np.pi*freq*mu_0/beta
    E0 = np.sqrt(4*incident*impedance/(a*b))
    iy, ix, iz = (int(np.argmin(abs(v-mid))) for v, mid in
                  ((y, b/2), (x, a/2), (z, length/2)))
    normE = np.linalg.norm(E, axis=-1)
    peak = float(np.max(np.abs(E[:, iy, :, 1].real)))
    field_views = [
        (E[:, iy, :, 1].real, "Re(Ey), phasor phase 0°", "RdBu_r", "V/m",
         TwoSlopeNorm(vmin=-peak, vcenter=0, vmax=peak)),
        (normE[:, iy, :], "Electric magnitude |E|", "viridis", "V/m", None),
        (Savg[:, iy, :, 2], "Time-averaged axial power flow ⟨Sz⟩", "inferno", "W/m²", None),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12.2, 7.6))
    fig.subplots_adjust(left=.075, right=.90, bottom=.12, top=.87, hspace=.40)
    for ax, (values, title, cmap, units, norm) in zip(axes, field_views):
        im = ax.pcolormesh(z*1000, x*1000, values, shading="auto", cmap=cmap, norm=norm)
        ax.set_xlim(0, length*1000)
        ax.set_ylim(0, a*1000)
        ax.set_aspect("equal", adjustable="box")
        ax.set_yticks([0, a*500, a*1000])
        ax.set_ylabel("x (mm)")
        ax.set_title(title, fontsize=11, loc="left", pad=5)
        cbar = fig.colorbar(im, ax=ax, pad=.015, fraction=.025)
        cbar.set_label(units)
        cbar.ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("z along guide (mm)")
    axes[-1].annotate("Energy travels +z", xy=(.95, .18), xytext=(.64, .18),
                      xycoords="axes fraction", color="white", fontsize=9,
                      va="center", arrowprops={"arrowstyle": "->", "color": "white", "lw":1.3})
    fig.suptitle(f"WR340 driven TE10 field  |  {freq/1e9:.2f} GHz, {incident:g} W incident",
                 fontsize=15, y=.975)
    fig.text(.075, .92, f"Actual {main['name']} field samples; longitudinal plane y = {y[iy]*1000:.2f} mm",
             fontsize=10, color="#526577")
    fig.text(.075, .028, "SI amplitudes retained. ⟨S⟩ = ½ Re(E × H*); the real-E panel uses the solved excitation phase.", fontsize=9, color="#526577")
    fig.savefig(figures / "te10-fields-2p45GHz.png", dpi=190)
    plt.close(fig)

    exact_profile = E0*np.sin(np.pi*x/a)
    measured_profile = np.abs(E[:, iy, iz, 1])
    profile_error = np.linalg.norm(measured_profile-exact_profile)/np.linalg.norm(exact_profile)
    phases = np.unwrap(np.angle(E[ix, iy, :, 1]))
    measured_beta = -float(np.polyfit(z, phases, 1)[0])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.subplots_adjust(left=.08, right=.98, top=.81, bottom=.20, wspace=.28)
    axes[0].plot(x*1000, exact_profile, color=INK, lw=1.6, label="Analytical TE10, 1 W")
    axes[0].plot(x*1000, measured_profile, "o", ms=3.6, mfc="white", color=BLUE, label="FEA samples")
    axes[0].set_title(f"Transverse profile at z = {z[iz]*1000:.1f} mm", fontsize=11)
    axis_style(axes[0], "|Ey| (V/m)", "x across broad dimension (mm)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].axhline(incident, color=INK, lw=1.3, ls="--", label="Incident power")
    axes[1].plot(z*1000, power, "o-", ms=1.8, lw=.7, color=BLUE, label="Integral of sampled ⟨Sz⟩")
    axes[1].set_title("Net power through each cross section", fontsize=11)
    axis_style(axes[1], "Power (W)", "z along guide (mm)")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("TE10 field checks at 2.45 GHz  |  physical 1 W normalization", fontsize=14, y=.97)
    fig.text(.08, .055, "Field interpolation and Cartesian-grid quadrature add error; this power integral is a separate check from port S parameters.", fontsize=9, color="#526577")
    fig.savefig(figures / "te10-field-validation.png", dpi=190)
    plt.close(fig)
    with (validation / "sampled-power-2p45GHz.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["z_mm", "sampled_net_power_w", "relative_to_incident_error_percent"])
        writer.writerows(zip(z*1000, power, 100*(power/incident-1)))
    return {
        "source": str(path.relative_to(main["directory"].parents[1])).replace("\\", "/"),
        "frequency_ghz": freq/1e9, "incident_power_w": incident,
        "grid_shape": list(expected[:3]), "longitudinal_slice_y_mm": float(y[iy]*1000),
        "analytical_TE_impedance_ohm": float(impedance),
        "analytical_guide_wavelength_mm": float(2*np.pi/beta*1000),
        "analytical_peak_E_v_per_m": float(E0),
        "sampled_peak_E_magnitude_v_per_m": float(normE.max()),
        "sampled_peak_E_relative_difference_percent": float(100*(normE.max()/E0-1)),
        "transverse_Ey_magnitude_profile_relative_l2_error": float(profile_error),
        "estimated_beta_from_centerline_phase_rad_per_m": measured_beta,
        "analytical_beta_rad_per_m": float(beta),
        "estimated_beta_relative_error_percent": float(100*(measured_beta/beta-1)),
        "electric_energy_fraction_in_Ey": float(np.sum(np.abs(E[..., 1])**2)/np.sum(np.abs(E)**2)),
        "sampled_net_power_min_w": float(power.min()), "sampled_net_power_max_w": float(power.max()),
        "sampled_net_power_mean_w": float(power.mean()),
        "max_sampled_power_error_relative_to_incident_percent": float(100*np.max(np.abs(power/incident-1))),
        "interpretation": "Physical SI amplitudes preserved. Profile and integrated-power diagnostics use "
                          "sampled fields; they are not exact finite-element integrals or a global field-error bound.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="sweep")
    parser.add_argument("--comparison", default="coarse", help="Comparison case, or 'none'")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    figures, validation = project / "figures", project / "results" / "validation"
    figures.mkdir(exist_ok=True)
    validation.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": INK, "text.color": INK,
                         "axes.edgecolor": "#a4aeba", "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    main_case = load_case(project, args.case)
    comparison = None if args.comparison.lower() == "none" else load_case(project, args.comparison)
    dims = main_case["summary"]["dimensions_m"]
    if comparison and comparison["summary"]["dimensions_m"] != dims:
        raise ValueError("Cannot compare cases with different guide dimensions")
    a, b = dims["a"], dims["b"]
    cutoffs = {"TE10": c/(2*a)/1e9, "TE20_TE01": c/a/1e9,
               "TE11_TM11": c/2*np.sqrt(a**-2+b**-2)/1e9}
    plot_sparameters(main_case, comparison, cutoffs, figures, validation)
    metrics = {
        "dimensions_m": dims, "cutoffs_ghz": cutoffs,
        "main": case_metrics(main_case, cutoffs["TE20_TE01"]*1e9),
        "comparison": None if comparison is None else case_metrics(comparison, cutoffs["TE20_TE01"]*1e9),
        "analytical_reference": "Uniform, air-filled PEC guide: S11=0 (-infinity dB), S21=exp(-j*beta*L).",
        "termination_scope": "TE10 matched at both reference planes; not a general multimode absorbing boundary. "
                             "Higher modes can propagate above their cutoffs. An ideal uniform guide does not convert TE10 into them.",
        "plot_sampling": "Each S-parameter marker is a solved frequency; straight segments connect existing samples only.",
        "display_floor_db": DISPLAY_FLOOR_DB,
        "fields_2p45GHz": plot_fields(main_case, figures, validation),
    }
    (validation / "validation-summary.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Figures: {figures}")
    print(f"Validation: {validation}")


if __name__ == "__main__":
    main()
