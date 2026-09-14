# ALIKED Matcher Comparison

Dataset: `20260802_150233`
Selection: source frames `0–999`, 1,000 frames, synchronized left/right images
Generated: `2026-09-14`

## Experimental control

The two runs use the same:

- 2,000 input images and the same image-record hashes;
- ALIKED `aliked-n16`, resize `1024`, 800 keypoints/image;
- 14,948 candidate pairs (`temporal_window=5`, `extra_stride=10`, `stereo_window=1`);
- custom `FULL_OPENCV` camera calibration with frozen parameters;
- `calibrated_stereo_planar` mapping and the same stereo thresholds;
- no HDF5/INS pose constraint.

The 2,000 cached ALIKED feature files have identical SHA-256 hashes in both
runs. Therefore the intended experimental variable is the descriptor matcher:
AdaLAM versus LightGlue.

## Results

| Metric | AdaLAM | LightGlue | LightGlue relative to AdaLAM |
|---|---:|---:|---:|
| Candidate pairs | 14,948 | 14,948 | same |
| Accepted pairs | 14,948 | 14,948 | same |
| Raw matches | 9,931,728 | 10,240,533 | +3.109% |
| Mean matches/pair | 664.419 | 685.077 | +3.109% |
| Verified geometries | 14,948 | 14,948 | same |
| Registered images | 2,000 | 2,000 | same |
| 3D points | **80,072** | 54,507 | −31.928% |
| Mean stereo reprojection error | **2.1301 px** | 2.2416 px | +5.234% |
| Median stereo reprojection error | **1.5877 px** | 1.7122 px | +7.842% |
| Mean track length | 14.686 | **21.579** | +46.930% |
| Minimum stereo points/frame | 514 | 510 | — |
| Mean stereo points/frame | 587.987 | 588.098 | +0.019% |
| Mean consecutive-motion inliers | 508.423 | **534.125** | +5.055% |
| Mean motion inlier rate | **99.803%** | 99.751% | −0.052 percentage points |
| Mean median motion error | **6.570 mm** | 6.783 mm | +3.237% |

The reprojection-error values are the stereo-point error fields written by the
custom calibrated-stereo planar writer; this path does not run a global
incremental COLMAP bundle adjustment.

## Interpretation

For this dataset and this reconstruction path, LightGlue produces slightly
more pairwise matches and substantially longer tracks, but those matches merge
into fewer final 3D points and have slightly larger stereo reprojection error.
AdaLAM is currently the stronger result if the priority is point count and
lower reprojection error. LightGlue remains a useful comparison because its
motion inlier count and track length are higher; visual inspection of point
distribution and trajectory smoothness is still needed before declaring a
universal winner.

## Artifacts

- AdaLAM: `outputs/20260802_150233/sfm/sfm__custom__aliked_adalam__1000f__PRIMARY/`
- LightGlue: `outputs/20260802_150233/sfm/sfm__custom__aliked_lightglue__1000f__PRIMARY/`
- LightGlue run metadata: `outputs/20260802_150233/sfm/sfm__custom__aliked_lightglue__1000f__PRIMARY/run.json`
- LightGlue model: `outputs/20260802_150233/sfm/sfm__custom__aliked_lightglue__1000f__PRIMARY/colmap/sparse/calibrated_stereo_planar/`
