# EMFEA

Native Windows electromagnetic FEM workflows built around **EMerge 3.0.0b6**.
This repository contains our editable solver drivers, analytical benchmarks,
field visualization and plotting scripts. EMerge remains an upstream dependency;
this is not a fork or a new Maxwell finite-element kernel.

## Setup

Install Python 3.12 x64 and uv:

```powershell
.\setup.ps1 -Python C:\path\to\python.exe
```

The lockfile recreates a separate RF environment. Do not install its VTK into the
FEMview environment. For interactive postprocessing set `FEMPOST_RF_PYTHON` in
FEMview to this checkout's `.venv/Scripts/python.exe`.

## Reproducible examples

```powershell
# 100 mm diameter x 100 mm air/PEC cylinder; analytical eigenmode comparison.
.\.venv\Scripts\python.exe examples/emerge-cylindrical-cavity/scripts/solve_cavity.py --case coarse
# 18 inch WR340 guide, analytical TE10 ports, raw S matrix and power checks.
.\.venv\Scripts\python.exe examples/emerge-wr340-waveguide/scripts/solve_waveguide.py --case sweep --mesh-mm 9
```

These commands solve and can consume minutes/GB of memory. Each writes geometry
and results under its example directory. Use a fresh `--case` per run. Plot with
the adjacent `plot_results.py` and inspect with `view_fields.py` (see `--help`).
Lengths passed to EMerge are metres. Complex fields retain phase; time-average
Poynting is 0.5 Re(E x H*). Raw S clipping/column scaling/reciprocity corrections
are disabled. The waveguide uses serial PARDISO with explicit DLL discovery.

The optional targeted ARPACK adapter is restricted to the documented 100x100 mm,
3 GHz, ten-mode cylinder benchmark. Its speed is not a general solver guarantee.
WR340's TE10-only termination above higher-mode cutoff is not a multimode-port
validation. These scripts do not establish arbitrary STEP/material/multi-feed
model accuracy. Solver-specific acceptance evidence belongs to each retained run.

## Publication scope

Source and pinned setup were extracted from the completed cylinder/waveguide
benchmarks. Private Falcon Space designs, full saved fields, environments and
caches remain with their original projects. They are not needed to regenerate the
synthetic benchmarks. See `docs/PUBLICATION.md` and `THIRD_PARTY_NOTICES.md`.

## Copper-cavity extension

`examples/copper-cavity/scripts/solve_cavity.py` retains the later 105 mm ID x
70 mm copper-wall cylinder benchmark (58 MS/m). `validate_cavity.py` compares
TE112 frequency and Q against a separate analytical calculation. It expects the
coarse/refined case names used by that workflow. Eigenmode wall impedance is
evaluated at the assembly reference frequency; this is a local frozen-impedance
approximation, not an exact nonlinear dispersive eigenproblem. No private project
CAD or results are bundled.
