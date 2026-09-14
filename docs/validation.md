# Validation and reproducibility

Phase 0 evidence is recorded in the [project-migration archive](archive/project_migration/2026-09/README.md), including the [REFACTOR_BASELINE.md](archive/project_migration/2026-09/REFACTOR_BASELINE.md) and [REFACTOR_INVENTORY.csv](archive/project_migration/2026-09/REFACTOR_INVENTORY.csv). Compact machine-readable anchors are under `validation/reference/20260802_150233/`. They contain source hashes, calibration values, rectification `P/Q` values, depth quantiles, SGBM settings, and the refractive audit status.

Run the deterministic checks with:

```powershell
python -m pytest -q tests
underwater calibration validate
underwater calibration generate
```

The tests intentionally cover pure geometry and generated artifacts without requiring a ZED SDK or external reconstruction executable. Full SVO replay, ORB-SLAM3, COLMAP, and Metashape statuses remain `NOT EXECUTED` unless those external dependencies are actually present and invoked.

The alpha-invariance reference compares the same raw correspondences through alpha 0 and alpha 1 rectification. It is a coordinate/behavior check, not a new calibration or physical-depth claim. Refractive sensitivity tables using guessed values are historical hypothetical diagnostics only.

The minimum experiment needed to close the depth-scale question is still the one documented in the archived depth-scale report: measured port/housing geometry plus an independent metric target at known distance, acquired in the same configuration.
