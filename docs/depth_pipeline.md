# Depth pipeline

There are two explicit engines:

1. `ZedSdkDepthEngine` consumes the verified ZED `MEASURE.DEPTH` product in metres. This is the production path for the custom-calibrated SDK run.
2. `SgbmDepthEngine` consumes a raw image pair, rectifies it with the custom profile, runs OpenCV StereoSGBM, divides the fixed-point disparity by `16.0`, and computes `Z = f_rectified * baseline / disparity`.

Both engines return `DepthFrame`, whose invalid samples are represented by a false mask and `NaN` depth values. Statistics distinguish center-pixel depth, center-window median, valid ratios, and invalid counts. Visualization accepts frames rather than reopening a camera or reimplementing depth semantics.

The recorded SDK reference has 35,855 frames and center-pixel median `2.204124689102173 m`; this is an operational custom-calibrated result, not independent physical ground truth. The recorded SGBM diagnostic uses half-resolution rectification, 192 disparities, block size 5, uniqueness 8, speckle window 100, and speckle range 2.

Example configuration-only commands:

```powershell
underwater depth export --dataset configs/datasets/20260802_150233.yaml --engine zed-neural --format summary
underwater depth export --dataset configs/datasets/20260802_150233.yaml --engine sgbm --format summary
```

These commands require the SVO path and relevant runtime dependencies. Missing SDK, SVO, or external tools produce an explicit error or `NOT EXECUTED` documentation status; they are not simulated as successful runs.
