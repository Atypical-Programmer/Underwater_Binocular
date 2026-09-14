# Refactor change log

This file records intentional behavior changes made after the package
migration. Existing operational depth, calibration, tracking, and ordinary
incremental-COLMAP paths remain unchanged unless noted below.

## 2026-09-14 — calibrated stereo path for near-planar SfM

| Old path | New path | Behavior status | Regression status | Deleted? | Reason |
|---|---|---|---|---|---|
| `underwater sfm aliked-colmap` → external incremental `mapper` for every scene | `--calibrated-stereo-planar` → canonical ZED stereo triangulation, adjacent-frame 3-D motion, COLMAP-compatible model conversion | Intentional opt-in extension; ordinary mapper remains the default CLI behavior | PASS on dataset `20260802_150233`: 2,000/2,000 registered images, 80,072 points3D, 2.130112 px mean reprojection error, external `model_analyzer` PASS | No | The near-planar scene produced valid synchronized stereo matches, but homography-dominated incremental initialization failed to register later images. The explicit calibrated baseline supplies the missing pose/scale constraint without rejecting planar geometry. |

## 2026-09-14 — converter output-directory fix

| Old path | New path | Behavior status | Regression status | Deleted? | Reason |
|---|---|---|---|---|---|
| Calibrated-stereo branch invoked `model_converter` with a not-yet-created binary output directory | Create only `colmap/sparse/calibrated_stereo_planar/` before conversion | Bug fix; no scientific behavior change | PASS: formal rerun exited 0 and analyzer read the binary model | No | COLMAP's converter requires the output directory to exist on this Windows build. |

The calibrated-stereo result is metric relative to the canonical stereo
baseline. That is a scale-source statement, not an independent validation of
absolute underwater physical accuracy.
