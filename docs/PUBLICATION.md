# Initial repository publication — 2026-10-27

Repository: EMFEA. Source extracted without changing original installations
or raw engineering measurements. Excludes private CAD, full simulation results,
virtual environments, user profiles and secrets. A local SHA-256 extraction
manifest is retained by the publication task. The public source is an initial
preview; domain-specific numerical acceptance remains separate from software tests.

## Verification

Fresh Python 3.12.11 / EMerge 3.0.0b6 environment recreated from the lockfile; dependency check passed. Executed the WR340 12 mm-mesh single-frequency 2.45 GHz smoke case, verified analytic phase error <1 degree and power-balance error <0.5%. Both cavity CLIs passed help parsing; full eigenmode/refinement benchmarks were not rerun for this source publication. All Python source parsed successfully. Existing run directories are now refused to preserve results.

## Hosted verification

Code `58b91d4`: [successful GitHub Actions run](https://github.com/kazzurt/EMFEA/actions/runs/40593054844). Windows hosted clean installation and analytical WR340 smoke check passed.
