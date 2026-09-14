# Depth export runbook

## Purpose

Depth export is the production/diagnostic depth entry point. It keeps custom ZED SDK depth and OpenCV StereoSGBM as separate engines and preserves raw/rectified image semantics.

## Prerequisites

Both engines read a dataset through the centralized ZED session. The ZED SDK, matching `pyzed.sl` binding, local SVO, OpenCV, and the canonical calibration profile are required for SDK-backed replay. The dataset config can use `UNDERWATER_SVO_PATH` or an explicit `--svo`.

## Commands

```powershell
underwater depth export `
  --dataset configs/datasets/20260802_150233.yaml `
  --engine zed-neural `
  --format summary

underwater depth export `
  --dataset configs/datasets/20260802_150233.yaml `
  --engine sgbm `
  --format summary
```

Use `--config configs/depth/zed_neural.yaml` or `--config configs/depth/sgbm.yaml` when a non-default engine configuration is needed. `--format mp4` produces a visualization in the ignored output area; visualization failure must not be interpreted as a depth-engine validation result.

## Interpretation

The custom-calibrated ZED SDK result around 2.204 m for dataset `20260802_150233` is an operational empirical product and a useful regression anchor. It is not independently validated physical underwater ground truth. SGBM disparity is fixed-point output divided by 16 and uses the custom rectification/calibration branch; it must not be mixed with native ZED rectified geometry.

No default refractive multiplier is applied. Absolute underwater accuracy requires independent metric ground truth and a measured port/interface model.

## Outputs and failures

Run metadata should identify dataset, SVO, calibration, engine, parameters, SDK/software versions, and frame statistics. Missing SVO, unsupported SDK, calibration mismatch, invalid configuration, and SDK errors fail loudly. Large videos/depth arrays remain local and ignored; keep compact references under `validation/reference/` when a result is intended for regression.
