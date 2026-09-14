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

Raw distorted images are `RAW_UNRECTIFIED`; OpenCV/ZED rectified images are `RECTIFIED`. The ZED session is opened once through `io.zed.ZedSession`, with custom calibration verification after opening. Depth consumers receive the common `DepthFrame` model, whether the engine is ZED `MEASURE.DEPTH` or OpenCV StereoSGBM.

Read [docs/architecture.md](docs/architecture.md) for the topology, [docs/depth_pipeline.md](docs/depth_pipeline.md) for depth semantics, and [docs/validation.md](docs/validation.md) for reproducibility and scientific guardrails.

## Data and version-control policy

Large SVO recordings, videos, dense depth arrays, point clouds, build caches, and run outputs stay local and ignored. Existing local files are not removed by the refactor. Small JSON references under `validation/reference/20260802_150233/` preserve the numbers and provenance needed for regression without pretending that they are ground truth.

The preserved third-party checkout has its own nested Git metadata and pre-existing local modifications. See [third_party/ORB_SLAM3_PROJECT_PATCHES.md](third_party/ORB_SLAM3_PROJECT_PATCHES.md).
