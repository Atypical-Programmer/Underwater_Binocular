# Reconstruction integrations

The package provides small, testable feature/matching and COLMAP database boundaries. `integrations/colmap/` documents the external process contract and makes calibration refinement explicit. The default mode freezes focal length, principal point, and extra parameters; refinement must be requested deliberately.

`reconstruction.metashape` converts COLMAP-style world-to-camera poses to local camera-to-world references and writes explicit `georeferenced=false` and scale-source metadata. `integrations/metashape/` contains import guidance. These coordinates are not GPS/geographic coordinates, and arbitrary SfM scale must not be presented as physical underwater scale.

No external COLMAP or Metashape run was executed as part of this refactor. See [REFACTOR_REGRESSION_REPORT.md](../REFACTOR_REGRESSION_REPORT.md).
