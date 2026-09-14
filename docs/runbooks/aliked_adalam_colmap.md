# ALIKED + AdaLAM + COLMAP runbook

## Purpose

This is the package's real neural feature/matching workflow. It migrates the legacy implementation found in `experiments/archive/legacy/run_aliked_adalam_colmap.py` without renaming an ORB/BFMatcher pipeline:

```text
SVO raw images
  → deterministic image manifest
  → LightGlue ALIKED
  → Kornia AdaLAM
  → bounded pair list and raw match list
  → COLMAP database
  → optional COLMAP geometric verification and mapper
```

`ALIKED ≠ ORB` and `AdaLAM ≠ BFMatcher`. Missing optional dependencies are errors; the workflow never silently falls back to ORB or BFMatcher.

## Dependencies

Install the package and SfM extra in the project environment:

```powershell
python -m pip install -e ".[dev,sfm]"
```

The workflow requires:

- OpenCV for image decoding and PNG writing;
- PyTorch for ALIKED/AdaLAM tensors;
- the LightGlue package, including `lightglue.ALIKED` and its model weights;
- Kornia with `kornia.feature.match_adalam`;
- a native COLMAP executable for geometric verification and mapping.

The `--profile` argument names the canonical project profile. For the ZED
SVO-open step, the runner regenerates its OpenCV `FileStorage` derivative at
`calibration/generated/zed_custom_opencv.yml`; the canonical profile YAML is
not passed directly to `optional_opencv_calibration_file`.

The `--device cuda` default fails if CUDA is unavailable. Use `--device cpu` explicitly for a CPU run; there is no automatic device fallback. COLMAP and the ZED SDK are external programs and are not pretend-installed by PyPI metadata.

## Quick smoke and longer run

```powershell
# 50 selected frames, both camera sides, stop after the custom DB/match stage
.\scripts\run_aliked_adalam_colmap.ps1 `
  -Frames 50 `
  -IncludeRight `
  -SkipColmap

# 50 frames through COLMAP if the executable is available
.\scripts\run_aliked_adalam_colmap.ps1 `
  -Frames 50 `
  -IncludeRight

# Longer example with frozen canonical calibration (default)
.\scripts\run_aliked_adalam_colmap.ps1 `
  -Frames 1000 `
  -IncludeRight

# Ready-to-run 1000-frame left/right reconstruction with exposed tuning parameters
.\scripts\run_aliked_adalam_colmap_1000.ps1 `
  -ColmapExecutable "D:\Underwater\Software\colmap-x64-windows-cuda\bin\colmap.exe"
```

The dedicated `run_aliked_adalam_colmap_1000.ps1` script defaults to 1000
uniformly sampled source positions, both camera sides, and a new
`outputs/20260802_150233/sfm/sfm__custom__aliked_adalam__1000f__PRIMARY`
directory. It sets `-StereoWindow 40` by default. For the full-SVO uniform
sampling, adjacent selected positions are about 35–36 source frames apart,
so this adds neighboring left/right time pairs while retaining the exact
synchronous pair. Use `-StereoWindow 0` to restore strict synchronization. If
using the first consecutive 1000 source frames, use a much smaller stereo
window or the cross-camera pair count grows quickly. Use
`-StartFrame 0 -EndFrame 999` for that consecutive selection. All major ALIKED,
pair-selection, and COLMAP thresholds are script parameters; run
`Get-Help .\scripts\run_aliked_adalam_colmap_1000.ps1` for the built-in examples.

Direct CLI form:

```powershell
underwater sfm aliked-colmap `
  --dataset configs/datasets/20260802_150233.yaml `
  --num-frames 50 `
  --include-right `
  --device cuda `
  --temporal-window 5 `
  --freeze-calibration
```

Set `UNDERWATER_SVO_PATH` or pass `--svo`. `--profile` can override the dataset profile explicitly. `--no-freeze-calibration` is the deliberate opt-in to COLMAP calibration refinement.

## Image selection and pair policy

The default image view is `RAW_UNRECTIFIED`, so the original-resolution keypoint coordinates match the canonical raw camera intrinsics and distortion model. The extractor selects deterministic SVO positions, retaining both endpoints when sampling, and writes:

```text
images/left/left_<source-frame>.png
images/right/right_<source-frame>.png
```

The manifest preserves source frame, side, timestamp, file hash, and image name. Same-side temporal pairs are bounded by `--temporal-window`; optional `--extra-stride` adds longer same-side pairs; `--stereo-window 0` adds synchronized left/right pairs. This is a bounded graph, not an accidental all-pairs O(N²) graph. `matching/pairs.txt` records every candidate pair and `matching/matches_raw.txt` records only pairs meeting `--min-raw-matches`.

## ALIKED and feature cache

The package controls the long-edge resize before calling `ALIKED.extract(..., resize=None)`. The returned processed-image coordinates are restored to original image pixels by an explicit x/y scale transform. The `FeatureSet` rejects any coordinate space other than `original_image_pixels`.

Feature files and `features/cache_manifest.json` include model, resize, keypoint limit, threshold, NMS radius, source image hash, side/frame identity, coordinate space, and package versions. A changed image or feature configuration invalidates the corresponding cache entry. `--resume` only reuses matching provenance.

## AdaLAM

The matcher calls `kornia.feature.match_adalam` directly with ALIKED descriptors, keypoints converted to LAFs, and each image's `(height, width)`. It validates match shapes and index ranges, removes non-finite/out-of-range values and duplicate one-to-one keypoint assignments, and records the number removed. An AdaLAM exception stops the run; no BFMatcher fallback exists.

## COLMAP database and mapper

The pipeline writes canonical-profile camera parameters into separate left/right cameras. The default mode is `FULL_OPENCV` with `calibration_mode=frozen`; mapper flags disable focal, principal-point, and extra-parameter refinement. `--no-freeze-calibration` marks the run `refined` and enables those flags.

For a small external compatibility audit using already extracted images, run
`scripts/run_colmap_compatibility_smoke.py` with 20–50 synchronized pairs. The
audit records the COLMAP executable, importer database statistics, verified
two-view geometries, mapper model counts, and model-converter output in the
run summary. If no executable is available, record `NOT EXECUTED` and retain
unique historical databases/models.

Custom ALIKED float descriptors remain in `features/*.npz`. COLMAP's legacy SIFT-shaped descriptor column is an inert uint8 storage slot because raw AdaLAM matches are imported explicitly; COLMAP native feature extraction is never run and cannot overwrite the custom feature stage. The database contains the intended custom keypoints and, after `matches_importer`, custom raw matches and verified two-view geometries.

When `--skip-colmap` is passed, the command stops before any external executable and inserts the accepted custom matches directly into `colmap/database.db`; the summary says `NOT_EXECUTED` for COLMAP. Without that flag, the command runs `matches_importer`, then `mapper`, then `model_converter`, with logs under `colmap/logs/`. A missing executable or mapper failure is an explicit error.

## Output structure

```text
outputs/<dataset>/sfm/<run-id>/
    run.json
    summary.json
    image_manifest.json
    images/
        left/
        right/
    features/
        *.npz
        cache_manifest.json
        summary.json
    matching/
        pairs.txt
        matches_raw.txt
        summary.json
    colmap/
        camera.json
        database.db
        logs/
        sparse/
        model_summary.json
    export/
```

`run.json` records dataset/SVO identity, frame selection, left/right usage, ALIKED and AdaLAM configuration, pair policy, calibration path/hash and frozen/refined mode, COLMAP executable, dependency versions, CUDA/device, and Git SHA. `summary.json` reports feature counts, candidate/matched pairs, match statistics, and real mapper counts when the mapper ran; values such as registered images and points are `null` when COLMAP was intentionally skipped.

## Scale and scientific limitations

Ordinary COLMAP SfM output is a local reconstruction with arbitrary scale unless a valid external scale or rig constraint is actually applied. Simultaneous left/right image input alone does not prove physical underwater metric scale. These outputs cannot establish that the custom SDK depth near 2.2 m is ground truth, and they do not justify a blanket `×1.333` correction.

## Troubleshooting

If ZED reports `INVALID CALIBRATION FILE`, verify that the canonical profile
was loaded and that `calibration/generated/zed_custom_opencv.yml` is the file
being used; the canonical profile YAML is an internal schema, not a ZED
OpenCV `FileStorage` file. The command also distinguishes missing
SVO/dataset/calibration, missing torch/LightGlue/Kornia, CUDA unavailable,
empty features, malformed AdaLAM matches, invalid camera parameters, missing
COLMAP, and mapper failure. Inspect `run.json`, `features/summary.json`,
`matching/summary.json`, and `colmap/logs/` in that order.
