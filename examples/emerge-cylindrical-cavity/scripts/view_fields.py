"""Orbitable native 3D field viewer; uses saved FEM samples, no solve required."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pyvista as pv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="refined")
    parser.add_argument("--mode", type=int, default=7, help="1-based, sorted by frequency")
    parser.add_argument("--field", choices=["E", "H", "Savg"], default="E")
    parser.add_argument("--phase", type=float, default=0, help="Phase in degrees, applied to arrows")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    result = project / "results" / args.case
    meta = json.loads((result / "summary.json").read_text())
    data = np.load(result / "sampled_fields.npz")
    if not 1 <= args.mode <= len(data["frequencies"]):
        parser.error("Mode is outside the saved range")
    mode = args.mode - 1
    E, H = data["E"][mode], data["H"][mode]
    radius = meta["configuration"]["diameter_mm"] / 2
    length = meta["configuration"]["length_mm"]
    # Fix only the arbitrary eigenvector phase, keeping relative E/H phase intact.
    gauge = np.exp(-1j*np.angle(E.reshape(-1)[np.nanargmax(np.abs(E))]))
    if args.field == "Savg":
        vector = .5*np.real(np.cross(E, H.conj()))
        subtitle = "Time-average power flow; ideally zero for a lossless standing eigenmode"
    else:
        vector = (E if args.field == "E" else H) * gauge
        subtitle = "Color: " + args.field + " phasor magnitude; arrows: instantaneous real field"
    magnitude = np.linalg.norm(vector, axis=-1)
    scale = np.nanmax(magnitude)
    coords = np.meshgrid(data["x"]*1000, data["y"]*1000, data["z"]*1000, indexing="ij")
    grid = pv.StructuredGrid(*coords)
    grid["Relative magnitude"] = (magnitude / (scale or 1)).ravel(order="F")
    plotter = pv.Plotter(off_screen=args.screenshot is not None, window_size=(1400, 1050))
    plotter.set_background("#edf1f5")
    cylinder = pv.Cylinder(center=(0,0,length/2), direction=(0,0,1), radius=radius, height=length, resolution=144)
    plotter.add_mesh(cylinder, color="#8b9cae", opacity=.08, smooth_shading=True)
    for normal, origin in [("x", (0,0,length/2)), ("y", (0,0,length/2)), ("z", (0,0,length/4))]:
        plotter.add_mesh(grid.slice(normal=normal, origin=origin), scalars="Relative magnitude",
                         cmap="inferno", clim=[0,1], nan_opacity=0, show_scalar_bar=normal=="z",
                         scalar_bar_args={"title": "Relative magnitude", "vertical": True,
                                          "position_x": .86, "position_y": .18, "height": .58,
                                          "width": .09, "color": "#142a3b", "title_font_size": 14})
    # Sparse arrows on an electric-field antinode section; phase shifts all components.
    k = int(np.argmin(np.abs(data["z"] - length/4000)))
    pos = np.stack([axis[::3, ::3, k] for axis in coords], axis=-1).reshape(-1,3)
    phase = np.exp(1j*np.deg2rad(args.phase))
    arrows = np.real(vector[::3, ::3, k] * phase).reshape(-1,3) / (scale or 1)
    good = np.isfinite(arrows).all(axis=1) & (np.linalg.norm(arrows,axis=1) > .03)
    if np.any(good):
        cloud = pv.PolyData(pos[good])
        cloud["arrows"] = arrows[good]
        plotter.add_mesh(cloud.glyph(orient="arrows", scale="arrows", factor=radius*.19), color="#41c7ed")
    row = meta["modes"][mode]
    plotter.add_text(f"Air / PEC cylinder, {2*radius:g} mm diameter x {length:g} mm length\n"
                     f"Mode {args.mode}: {row['analytic_family']} family, {row['frequency_ghz']:.6f} GHz\n"
                     f"{subtitle}\nArbitrary eigenmode amplitude; arrows at phase {args.phase:g} deg",
                     font_size=13, color="#142a3b", position="upper_left")
    plotter.add_axes(xlabel="x", ylabel="y", zlabel="z")
    plotter.show_grid(xtitle="x (mm)", ytitle="y (mm)", ztitle="z (mm)", color="#34495e")
    plotter.camera_position = [(185,-230,195), (0,0,length/2), (0,0,1)]
    plotter.camera.zoom(.9)
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(args.screenshot))
    else:
        plotter.show()


if __name__ == "__main__":
    main()
