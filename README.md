# Underwater Binocular

Maintainable calibration, depth, geometry, tracking, and reconstruction tools for a ZED 2i stereo rig.

The repository's current scientific conclusion is deliberately conservative:

- the custom-calibrated ZED SDK depth product is an operational empirical result, with a stable center estimate of about 2.2 m for dataset `20260802_150233`;
- that result is not independent physical ground truth and must not be multiplied by `1.333` automatically;
- the refractive correction `R1` is `NOT_IDENTIFIABLE` until the housing/port geometry, refractive indices, interface planes, and independent metric ground truth are measured.

## Quick start

```powershell
conda env create -f environment.yml
conda activate underwater-binocular
python -m pip install -e ".[dev]"

underwater calibration validate
underwater calibration generate
python -m pytest -q tests
```

For the ALIKED + AdaLAM workflow, install the optional SfM dependencies too:

```powershell
python -m pip install -e ".[dev,sfm]"
```

The ZED SDK is a vendor installation, not a PyPI dependency. Set `ZED_SDK_ROOT_DIR` when running SDK-backed commands. Set `UNDERWATER_SVO_PATH` for the local recording, or pass `--svo` explicitly. No machine-specific path is embedded in a tracked config.

## Repository map

```text
calibration/
  profiles/                 canonical computational calibration
  generated/                derived ZED/OpenCV, ORB-SLAM3, and COLMAP files
  source/                   original calibration evidence and reports
configs/                    dataset, depth, and local-machine templates
src/underwater_binocular/   importable package and public CLI implementation
integrations/               ORB-SLAM3, COLMAP, and Metashape boundaries
experiments/                diagnostic and research-only workflows
third_party/                preserved ORB-SLAM3 checkout
validation/reference/       compact behavioral regression anchors
tests/                      unit, integration, and reference regression tests
docs/                       architecture, operation, and historical notes
outputs/                    ignored run products
```

## Important contracts

Package-internal translations are metres. The stereo transform is always:

```text
X_right = R_left_to_right @ X_left + t_left_to_right
```

Raw distorted images are `RAW_UNRECTIFIED`; OpenCV/ZED rectified images are `RECTIFIED`. The ZED session is opened once through `io.zed.ZedSession`, with explicit native/custom calibration policy. The recommended tracking mode is `native`, which uses calibration embedded in the SVO and does not set `optional_opencv_calibration_file`; `custom` is an explicit comparison mode that passes the derived OpenCV FileStorage calibration and verifies runtime calibration. Depth consumers receive the common `DepthFrame` model, whether the engine is ZED `MEASURE.DEPTH` or OpenCV StereoSGBM.

Read [docs/architecture.md](docs/architecture.md) for the topology, [docs/depth_pipeline.md](docs/depth_pipeline.md) for depth semantics, [docs/validation.md](docs/validation.md) for reproducibility and scientific guardrails, and [OUTPUT_CONSOLIDATION_REPORT.md](OUTPUT_CONSOLIDATION_REPORT.md) for the current local output inventory and run status.

## Capabilities

| Capability | Recommended command | Status | Main dependency |
|---|---|---|---|
| Calibration | `underwater calibration validate` | Stable | core package |
| ZED SDK depth | `underwater depth export ...` | Integration; requires vendor SDK/SVO | `pyzed.sl` |
| StereoSGBM depth | `underwater depth export --engine sgbm ...` | Integration; requires SVO/image input | OpenCV + ZED I/O |
| ZED positional tracking | `underwater tracking zed ... --calibration-mode native` | Integration; sequential replay requires vendor SDK/SVO | `pyzed.sl` |
| ALIKED + AdaLAM + COLMAP | `underwater sfm aliked-colmap ...` | Integration; real feature/match smoke tested, full external workflow environment-dependent | torch, LightGlue, Kornia, COLMAP |
| ORB-SLAM3 | `integrations/orbslam3/run_orbslam3_svo2.ps1` | Integration; requires native build/toolchain | ORB-SLAM3, ZED SDK, CUDA |
| Metashape export | `underwater_binocular.reconstruction.metashape` | Library only | external Metashape GUI |
| Calibration/rectification diagnostics | `underwater diagnostic ...` | Experimental/diagnostic | core + OpenCV |

## Common workflows

The versioned dataset config intentionally has no machine-specific SVO path. Set `UNDERWATER_SVO_PATH` or pass `--svo`.

```powershell
# ZED GEN_1 sequential tracking
.\scripts\run_zed_tracking.ps1 -Mode GEN_1 -CalibrationMode native

# ZED GEN_3 sequential tracking smoke
.\scripts\run_zed_tracking.ps1 -Mode GEN_3 -MaxFrames 1000 -CalibrationMode native

# 50-frame ALIKED + AdaLAM smoke; include both camera sides and stop before COLMAP
.\scripts\run_aliked_adalam_colmap.ps1 -Frames 50 -IncludeRight -SkipColmap

# Longer ALIKED + AdaLAM + COLMAP run with frozen calibration
.\scripts\run_aliked_adalam_colmap.ps1 -Frames 1000 -IncludeRight

# Dedicated 1000-frame left/right run; add -ColmapExecutable if COLMAP is not on PATH
.\scripts\run_aliked_adalam_colmap_1000.ps1 -ColmapExecutable "D:\Underwater\Software\colmap-x64-windows-cuda\bin\colmap.exe"

# Custom calibration comparison (explicit opt-in)
.\scripts\run_zed_tracking.ps1 -Mode GEN_1 -CalibrationMode custom -Profile calibration/profiles/zed2i_37395692_custom.yaml

# Inspect local output state; pruning is dry-run unless explicitly selected
underwater outputs inventory
underwater outputs prune --dry-run

# Core tests
python -m pytest -q
```

The detailed procedures are in [docs/runbooks/zed_tracking.md](docs/runbooks/zed_tracking.md), [docs/runbooks/aliked_adalam_colmap.md](docs/runbooks/aliked_adalam_colmap.md), [docs/runbooks/depth_export.md](docs/runbooks/depth_export.md), and [docs/runbooks/orbslam3.md](docs/runbooks/orbslam3.md).

## Data and version-control policy

Large SVO recordings, videos, dense depth arrays, point clouds, build caches, and run outputs stay local and ignored. Active runs belong under `outputs/`; reproducible disposable intermediates belong under `cache/`. Small JSON references under `validation/reference/20260802_150233/` preserve the numbers and provenance needed for regression without pretending that they are ground truth.

The preserved third-party checkout has its own nested Git metadata and pre-existing local modifications. See [third_party/ORB_SLAM3_PROJECT_PATCHES.md](third_party/ORB_SLAM3_PROJECT_PATCHES.md).
