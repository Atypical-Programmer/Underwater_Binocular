# COLMAP integration

The full neural workflow is `underwater sfm aliked-colmap`; it performs raw SVO image extraction, real LightGlue ALIKED features, real Kornia AdaLAM matches, custom COLMAP database insertion, and—unless explicitly skipped—external geometric verification and mapping. See [../../docs/runbooks/aliked_adalam_colmap.md](../../docs/runbooks/aliked_adalam_colmap.md).

The lower-level importable process boundary is `underwater_binocular.reconstruction.colmap_runner`. It builds explicit `matches_importer`, `mapper`, and `model_converter` commands with calibration either frozen (default) or intentionally refined.

```powershell
underwater sfm colmap --calibration calibration/profiles/zed2i_37395692_custom.yaml
```

The low-level command only prepares configuration when no remainder command is supplied. The neural workflow requires an external COLMAP executable for `matches_importer` and `mapper`; `--skip-colmap` is an explicit database-only stop point and is recorded as `NOT_EXECUTED`.
