"""Native orbitable viewer of saved WR340 fields at 2.45 GHz; no solver runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pyvista as pv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="sweep")
    parser.add_argument("--field", choices=["EyReal","E","H","Savg"], default="EyReal")
    parser.add_argument("--phase", type=float, default=0)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    folder = project / "results" / args.case
    summary = json.loads((folder / "summary.json").read_text())
    data = np.load(folder / "fields_2p45GHz.npz")
    x,y,z = (data[k]*1000 for k in ("x","y","z"))
    X,Y,Z = np.meshgrid(x,y,z,indexing="ij")
    E,H,Savg = data["E"],data["H"],data["Savg"]
    phase = np.exp(1j*np.deg2rad(args.phase))
    if args.field == "EyReal":
        values = np.real(E[...,1]*phase)
        vector = np.real(E*phase)
        label,cmap = "Real Ey (V/m)","coolwarm"
        peak = float(np.abs(values).max())
        limits = [-peak,peak]
    elif args.field == "Savg":
        values = np.linalg.norm(Savg,axis=-1)
        vector = Savg
        label,cmap = "Mean power flow (W/m2)","viridis"
        limits = [0,float(values.max())]
    else:
        phasor = E if args.field == "E" else H
        values = np.linalg.norm(phasor,axis=-1)
        vector = np.real(phasor*phase)
        label = "Electric magnitude (V/m)" if args.field == "E" else "Magnetic magnitude (A/m)"
        cmap,limits = "inferno",[0,float(values.max())]
    grid = pv.StructuredGrid(X,Y,Z)
    grid[label] = values.ravel(order="F")
    dims = summary["dimensions_m"]
    a,b,L = [dims[k]*1000 for k in ("a","b","length")]
    plotter = pv.Plotter(off_screen=args.screenshot is not None,window_size=(1500,900))
    plotter.set_background("#eff3f7")
    plotter.add_mesh(pv.Box(bounds=(0,a,0,b,0,L)),style="wireframe",color="#6b7d8e",line_width=1)
    for normal,origin in [("y",(a/2,b/2,L/2))]:
        section = grid.slice(normal=normal,origin=origin)
        plotter.add_mesh(section,scalars=label,cmap=cmap,clim=limits,show_scalar_bar=normal=="y",
                         scalar_bar_args={"title":label,"vertical":True,"position_x":.03,"position_y":.2,
                                          "height":.55,"width":.06,"color":"#182f43","title_font_size":13})
    iy = len(y)//2
    points = np.stack([v[::4,iy,::9] for v in (X,Y,Z)],axis=-1).reshape(-1,3)
    vectors = vector[::4,iy,::9].reshape(-1,3)
    scale = np.linalg.norm(vector,axis=-1).max()
    vectors = vectors/(scale or 1)
    keep = np.linalg.norm(vectors,axis=1) > .08
    cloud = pv.PolyData(points[keep])
    cloud["vectors"] = vectors[keep]
    if np.any(keep):
        plotter.add_mesh(cloud.glyph(orient="vectors",scale="vectors",factor=8),color="#22b8c5",show_scalar_bar=False)
    plotter.add_point_labels(np.array([[a/2,b/2,0],[a/2,b/2,L]]),
                            ["Port 1: 1 W incident","Port 2: matched TE10"],
                            font_size=14,text_color="#182f43",shape_color="white",shape_opacity=.8,
                            always_visible=True,point_size=4)
    plotter.add_text("WR340, 18-inch straight PEC waveguide\n"
                     "2.45 GHz TE10, 1 W incident at port 1, propagation +z\n"
                     f"{label}, RF phase {args.phase:g} deg\n"
                     "Saved FEA results; orbit, zoom and pan with the mouse",
                     font_size=14,color="#182f43",position="upper_left")
    plotter.add_axes()
    plotter.show_grid(xtitle="x (mm)",ytitle="y (mm)",ztitle="z (mm)",color="#42566a",
                      n_xlabels=2,n_ylabels=2,n_zlabels=4,font_size=12)
    plotter.camera_position=[(-2.7*a,9*b,.65*L),(a/2,b/2,L/2),(0,1,0)]
    plotter.reset_camera()
    plotter.camera.zoom(1.2)
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True,exist_ok=True)
        plotter.show(screenshot=str(args.screenshot))
    else:
        plotter.show()


if __name__ == "__main__":
    main()
