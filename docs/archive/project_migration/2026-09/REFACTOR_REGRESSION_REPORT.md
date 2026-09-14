# Refactor regression report

## Executed

- `python -m pytest -q tests`: 22 passed in the OpenCV-capable environment.
- `underwater calibration validate --profile calibration/profiles/zed2i_37395692_custom.yaml`: PASS.
- `underwater calibration generate --profile calibration/profiles/zed2i_37395692_custom.yaml`: PASS; generated ZED YAML round-trips through OpenCV `FileStorage`.
- `integrations/orbslam3/prepare_orbslam3_stereo.py --root . --output calibration/generated/orbslam3_stereo.yaml --scale 0.5`: PASS; wrapper smoke-tested without invoking ORB-SLAM3.
- `python -m compileall -q src tests integrations`: PASS.
- `ruff check src tests`: PASS.
- Legacy pure refractive and rig tests: 19 passed before they were relocated to the package test suite.

## Not executed

- Full SVO/ZED replay: NOT EXECUTED in this refactor verification; it requires the vendor `pyzed.sl` binding and the local recording.
- ORB-SLAM3 build/replay: NOT EXECUTED; it requires the local C++ toolchain, CUDA, ZED SDK, vocabulary, and preserved third-party checkout.
- COLMAP reconstruction: NOT EXECUTED; no external COLMAP executable was invoked.
- Metashape import: NOT EXECUTED; it is an external GUI workflow.

External workflows are never reported as PASS merely because their configuration can be generated.
