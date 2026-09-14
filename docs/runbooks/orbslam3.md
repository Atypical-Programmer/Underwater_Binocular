# ORB-SLAM3 runbook

## Purpose and status

ORB-SLAM3 is an external native integration under `integrations/orbslam3/` using the preserved source checkout at `third_party/ORB_SLAM3/`. It is separate from the ZED SDK tracking command and from the ALIKED/AdaLAM reconstruction workflow.

## Prerequisites

- ZED SDK and local SVO/SVO2;
- CUDA toolkit, CMake/MSVC, and the integration's vcpkg dependencies;
- the ORBvoc vocabulary under `third_party/ORB_SLAM3/Vocabulary/`;
- a built `svo2_stereo.exe`.

Set `ZED_SDK_ROOT_DIR`, `CUDA_TOOLKIT_ROOT_DIR`, and `UNDERWATER_SVO_PATH`, or pass the wrapper parameters. Calibration settings are regenerated from the canonical profile.

## Build and replay

```powershell
.\integrations\orbslam3\build_orbslam3.ps1
.\integrations\orbslam3\run_orbslam3_svo2.ps1 `
  -MaxFrames 300 `
  -Output outputs\orbslam3_smoke
```

Use `-MaxFrames 0` for a full sequential replay. `-SampleCount` is for explicit resampling/export workflows; it does not turn a non-sequential run into a valid tracking trajectory.

## Outputs and interpretation

Inspect tracking state, valid pose ratio, continuous segments, trajectory jumps, and map points together. ORB-SLAM3 poses are local reconstruction coordinates and are not automatically physical underwater ground truth or independently metric. Metashape CSV conversion is an export helper; external Metashape import/optimization is a separate step.

The native build/replay is an external workflow and is not reported as PASS unless it was actually run on the current machine.
