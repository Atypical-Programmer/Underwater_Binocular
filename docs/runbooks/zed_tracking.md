# ZED positional tracking runbook

## Purpose

`underwater tracking zed` performs a real ZED SDK positional-tracking replay and writes pose/state/provenance files. The tracking implementation uses one `ZedSession` per replay. `GEN_1` and `GEN_3` in `--mode BOTH` are two independent SVO opens from the beginning; they never share an evolving tracker state.

Tracking success is not the same as a physically accurate trajectory. Always inspect state counts, valid ratio, continuity, lost intervals, and large translation jumps together.

## Prerequisites

- A vendor-supported ZED SDK with the matching `pyzed.sl` binding.
- A local `.svo`/`.svo2` recording.
- The canonical profile `calibration/profiles/zed2i_37395692_custom.yaml`.
- A dataset config such as `configs/datasets/20260802_150233.yaml`.

The dataset config can read `UNDERWATER_SVO_PATH`, or an explicit `--svo` can override it. On Windows, set `ZED_SDK_ROOT_DIR` if the vendor DLLs are not already discoverable.

## Calibration modes

The recommended mode is `native`. It opens the SVO without setting
`optional_opencv_calibration_file`, so the SDK uses calibration embedded in
the recording. Runtime metadata records the embedded calibration and mode.

Use `custom` only for an explicit comparison or a controlled custom-calibration
experiment. It requires the canonical profile, passes the derived OpenCV
calibration file to the SDK, and verifies the SDK's raw runtime calibration
against the profile. A profile is rejected in native mode rather than inferred
or silently applied.

## Quick start

```powershell
$env:UNDERWATER_SVO_PATH = '<absolute-path-to-recording.svo2>'

underwater tracking zed `
  --dataset configs/datasets/20260802_150233.yaml `
  --mode GEN_1 `
  --calibration-mode native

underwater tracking zed `
  --dataset configs/datasets/20260802_150233.yaml `
  --mode GEN_3 `
  --calibration-mode native `
  --max-frames 300
```

Equivalent thin wrappers are:

```powershell
.\scripts\run_zed_tracking.ps1 -Mode GEN_1
.\scripts\run_zed_tracking.ps1 -Mode GEN_3 -MaxFrames 1000
.\scripts\run_zed_tracking.ps1 -Mode BOTH -MaxFrames 300
.\scripts\run_zed_tracking.ps1 -Mode GEN_1 -CalibrationMode custom -Profile calibration/profiles/zed2i_37395692_custom.yaml
```

Use `--start-frame`, `--end-frame`, and `--max-frames` to select a range. Seeking is permitted only before tracking starts; the range itself is then consumed consecutively with one `grab()` per frame.

## Tracking configuration

The package keeps one shared `TrackingConfig` with explicit settings for:

```text
mode
enable_area_memory
enable_imu_fusion
enable_pose_smoothing
set_gravity_as_origin
set_floor_as_origin
set_as_static
```

`GEN_1` and `GEN_3` are selected through `sl.POSITIONAL_TRACKING_MODE`. If the installed SDK does not expose the requested mode, the command fails with an informative error; it never silently changes `GEN_3` to `GEN_1`.

## Outputs

The default output is:

```text
outputs/<dataset>/tracking/<run-id>/
    run.json
    summary.json
    trajectory.csv
    tracking_status.csv
```

For `--mode BOTH`, each independent replay is written below `gen_1/` and `gen_3/`, with a root `comparison.csv`, `run.json`, and `summary.json`.

`trajectory.csv` contains one row for every grabbed frame:

```text
frame_index,svo_position,timestamp_ns,tracking_state,pose_valid,
pose_confidence,tx_m,ty_m,tz_m,qx,qy,qz,qw
```

Translations are metres. The pose reference is ZED `WORLD`, the coordinate system is `RIGHT_HANDED_Y_UP`, and the quaternion order is explicitly `x,y,z,w`. Invalid poses remain represented by `pose_valid=0` and `NaN` pose values; no values are fabricated.

`summary.json` reports total attempted/grabbed frames, valid pose frames and ratio, searching/lost/off counts, the longest valid segment, frame range, mode, and trajectory statistics. Path length is accumulated only across adjacent valid poses and never across a lost segment.

## Quality checks and failures

Look for:

- `valid_pose_ratio` and `tracking_state_counts`;
- `longest_valid_segment_frames`;
- `step_max_m` and the p95 step value;
- large discontinuities in `trajectory.csv`;
- a `status` of `completed` in `run.json`.

Common failures are missing `pyzed.sl`, missing SVO, missing custom calibration in custom mode, unsupported tracking mode, SDK calibration mismatch, and non-success SDK status codes. These fail loudly and leave a failed `run.json` when the run directory has already been created.

## Reproducibility and limitations

`run.json` records the dataset config, SVO identity, SDK version, camera identity when available, calibration mode/source, and—only for custom runs—the profile path/hash and runtime verification. It also records tracking settings, frame range, coordinate convention, software version, and Git SHA. The world origin is supplied by the ZED tracker and the trajectory is local; it is not georeferenced or independently proven to be metric underwater truth.
