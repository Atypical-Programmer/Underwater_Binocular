# Refactor baseline

This file freezes the parent repository state before the package migration. It is a migration regression anchor, not a physical ground-truth claim.

## Repository

| Field | Value |
|---|---|
| Repository | `Atypical-Programmer/Underwater_Binocular` |
| Working directory | `D:\\Underwater\\BinocularCamera` (local workspace only; not a runtime default) |
| HEAD SHA | `17238044580a6bb9f0cb7b7b914871f1531602d8` |
| Branch | `main` |
| HEAD date | `2026-09-13T22:12:31-04:00` |
| Baseline date | `2026-09-13` |
| Parent worktree status | clean before this baseline artifact was added |
| Nested ORB-SLAM3 checkout | present and independently dirty; preserved without rewriting it |

## Inventory counts at baseline

The parent Git index contained 53 Python files, 15 Markdown files, 27 YAML/YML files, 6 JSON files, and 3 PowerShell files. Ten tracked files were under `output/`. The root also contained ignored local recordings, videos, CSV/PNG/JSON exports, and a large ignored `output/` result tree; those local artifacts were inspected and were not deleted.

The detailed migration inventory is [REFACTOR_INVENTORY.csv](REFACTOR_INVENTORY.csv). It covers each root Python entry point, tracked documentation/configuration/evidence, the SLAM subtree, tracked audit outputs, and aggregate rows for ignored local results that must remain untouched.

## Tracked audit outputs present before migration

| Path | Bytes | SHA256 / status |
|---|---:|---|
| `output/20260802_150233_final_underwater_audit/correspondence_quality.csv` | 9,781,479 | tracked; source evidence retained in Git history |
| `output/20260802_150233_final_underwater_audit/native_refractive_depth_check.json` | 4,810 | tracked; source evidence retained in Git history |
| `output/20260802_150233_final_underwater_audit/native_vs_custom_calibration.json` | 31,683 | tracked; source evidence retained in Git history |
| `output/20260802_150233_final_underwater_audit/ray_field_comparison.csv` | 20,798 | tracked; source evidence retained in Git history |
| `output/20260802_150233_final_underwater_audit/rig_refractive_sensitivity.csv` | 133,365 | tracked; source evidence retained in Git history |
| `output/20260802_150233_final_underwater_audit/triangulation_quality.csv` | 2,966,476 | tracked; source evidence retained in Git history |
| `output/20260802_150233_flat_port_refractive_audit/alpha_invariance_check.json` | 342,887 | tracked; source evidence retained in Git history |
| `output/20260802_150233_flat_port_refractive_audit/refractive_correspondence_check.json` | 1,876,592 | tracked; source evidence retained in Git history |
| `output/20260802_150233_flat_port_refractive_audit/refractive_sensitivity.csv` | 314,211 | tracked; source evidence retained in Git history |
| `output/20260802_150233_refractive_depth_audit/refractive_depth_audit.json` | 297,709 | tracked; source evidence retained in Git history |

No tracked result was removed before the reference extraction stage. Unknown or ignored local outputs, including the root SVO, root videos, and large local point-cloud/depth directories, remain in place.

## Calibration source files

| Path | Role | SHA256 |
|---|---|---|
| `Calibration/标定结果/camera_intrinsics.yaml` | original calibration source evidence | `DDA48DDEDE9E6FC779AD5196DFB91763B5FC4EDCACF02140F265A2244D09BADC` |
| `Calibration/标定结果/stereo_extrinsics.yaml` | original stereo source evidence | `2843479EDB23C576E4F6633BD1863BC67BA83AEB0570A281FFB63827CC0F6DAD` |
| `Calibration/标定结果/calib_full_params.xlsx` | calibration workbook evidence | not hashed into computation baseline |
| `Calibration/标定结果/Camera-Centric.pdf` | calibration report evidence | not hashed into computation baseline |
| `Calibration/标定结果/重投影误差.pdf` | reprojection report evidence | not hashed into computation baseline |
| `Calibration/zed_custom_opencv.yml` | parent derived OpenCV calibration | `596D63D6198303E1F65E2A19BE52BEF700D52E1BEE5AD8166A8CD21D1F060098` |

The parent custom calibration values used by the production result were:

```text
resolution                 = 1920 x 1080
left K                     = fx 1443.326338, fy 1441.541002, cx 967.033878, cy 540.469238
right K                    = fx 1449.830254, fy 1448.097563, cx 975.331674, cy 514.232843
left D (first five)        = [0.3151847, -0.7495814, 0.002129304, 0.005759798, 6.688496]
right D (first five)       = [0.3023032, -0.1692313, 0.001685104, 0.01069086, 1.944058]
R left-to-right            = [[0.999968008, 0.0054886443, 0.00581871018],
                               [-0.00548930265, 0.999984929, 0.0000971791651],
                               [-0.0058180891, -0.000129116717, 0.999983066]]
t left-to-right           = [-0.1224352, 0.0001676, 0.0178539] m
|Tx|                       = 0.1224352 m
||t||                      = 0.1237302227994842 m
reprojection error         = 0.225048 px
rotation determinant       = approximately 0.9999999994305349
rotation orthogonality err = approximately 8.812907070776532e-10 (max abs)
```

## Main pipelines at baseline

* Custom ZED SDK depth: `convert_zed_custom_depth_to_video.py` using `MEASURE.DEPTH`, `DEPTH_MODE.NEURAL`, custom OpenCV calibration, `METER`, and `CAMERA` reference frame.
* Native/SVO inspection and sampling: `read_svo2.py`, `sample_native_svo_depth.py`, `compare_svo_and_calibration.py`.
* OpenCV stereo diagnostic: `strict_compare_stereo_depth.py`, `regenerate_sgbm_depth_histogram.py`, and rectification experiments.
* Calibration regeneration/runtime verification: `rerun_custom_calibration.py`.
* Refractive geometry/audit: `refractive_geometry.py`, `rig_refractive_geometry.py`, and diagnostic scripts.
* Tracking/point clouds: `benchmark_tracking.py`, `export_pointcloud.py`, `export_gen1_pointcloud.py`, and `SLAM/` runners.
* Feature/matching/SfM: `run_aliked_adalam_colmap.py` and the COLMAP/Metashape conversion scripts.

## Known operational numerical references

These values are expected operational outputs from the parent implementation. They are not independent physical ground truth.

| Reference | Parent value |
|---|---:|
| SVO | `20260802_150233.svo2` |
| camera | ZED 2i, serial `37395692` |
| SDK | `5.4.1` |
| total SVO frames | `35,855` |
| full custom SDK depth exported frames | `35,855` |
| center pixel | `(960, 540)` |
| center pixel valid frames | `35,706 / 35,855` (`0.9958443731697113`) |
| center pixel median | `2.204124689102173 m` |
| center 5x5 median | `2.2040793895721436 m` |
| mean valid ratio across frames | `0.9135921064255291` |
| display range | `1.5–3.5 m` |
| confidence / texture thresholds | `30 / 100` |
| native raw baseline norm | `0.1198962253549308 m` |
| custom baseline norm | `0.1237302227994842 m` |
| refractive candidate | `n=1.333`, diagnostic only; not a production correction |
| audit correspondence counts | 46,024 candidates; Set A 1,958; Set B 1,574 |

## Baseline policy

* `output/` is historical evidence at baseline. It may only leave the active tree after summary/reference extraction and archive documentation.
* `outputs/` is the future disposable run root and must be Git-ignored.
* Unknown local results are not deleted.
* Git history is not rewritten.
* Any behavior change discovered during migration must be recorded in `REFACTOR_CHANGELOG.md` and must stop deletion until explained.
