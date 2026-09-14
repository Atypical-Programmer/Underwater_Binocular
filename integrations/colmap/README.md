# COLMAP integration

The importable process boundary is `underwater_binocular.reconstruction.colmap_runner`. It creates explicit camera metadata and can build a mapper command with calibration either frozen (default) or intentionally refined.

```powershell
underwater sfm colmap --calibration calibration/profiles/zed2i_37395692_custom.yaml
```

The command only prepares configuration when no remainder command is supplied. An external COLMAP executable is required for reconstruction and was not invoked in the refactor verification.
