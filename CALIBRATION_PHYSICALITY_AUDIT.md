# Calibration Physicality Audit

Audit date: 2026-09-12
Input: `20260802_150233.svo2`, 1920x1080, ZED 2i, serial 37395692, SDK 5.4.1.

This document separates three things that must not be conflated:

1. a native raw camera model available from the SVO;
2. an external custom empirical pinhole model;
3. a physically parameterized refractive rig model.

The numerical source of truth is [`native_vs_custom_calibration.json`](output/20260802_150233_final_underwater_audit/native_vs_custom_calibration.json).

## Direct result

The custom focal-length ratio is numerically close to water-index scaling, but that is not a physicality proof. The raw models have different K/D/R/T, the rectified models have different virtual coordinate systems, and the custom calibration provenance does not record the medium, housing, target square size, or metric ground truth.

Therefore:

- native raw K/D/T is the **best available physical/dry candidate**, not proven dry or air-calibrated;
- custom K/D/T is an **empirical underwater pinhole candidate**, not proven to be a physical flat-port model;
- native raw plus explicit refraction is **not identifiable** until the installed housing is measured;
- custom `fB/d` is not to be multiplied by 1.333 a second time;
- custom `fB/d` is not yet an independently validated absolute underwater distance.

## Raw intrinsic comparison

| source | side | fx (px) | fy (px) | cx (px) | cy (px) | distortion |
|---|---:|---:|---:|---:|---:|---|
| SVO native raw | left | 1068.000000 | 1067.770020 | 957.400024 | 539.591003 | 12-coefficient RAD_TAN metadata |
| SVO native raw | right | 1067.829956 | 1067.709961 | 952.570007 | 511.776001 | 12-coefficient RAD_TAN metadata |
| custom YAML raw | left | 1443.326338 | 1441.541002 | 967.033878 | 540.469238 | 5-coefficient OpenCV RAD_TAN-like model |
| custom YAML raw | right | 1449.830254 | 1448.097563 | 975.331674 | 514.232843 | 5-coefficient OpenCV RAD_TAN-like model |

The custom/raw focal ratios are:

| side | fx ratio | fy ratio |
|---|---:|---:|
| left | 1.351429 | 1.350048 |
| right | 1.357735 | 1.356265 |

At `n_water/n_air = 1.333`, the custom-minus-`n`-times-native discrepancy is approximately +1.28% to +1.86% across these raw focal entries. For `n=1.330..1.340`, the same comparison moves across a few percent. This is compatible with an effective underwater fit, but it cannot distinguish a physical refractive calibration from an empirical fit or another systematic change.

## Rectified focal lengths are virtual projection parameters

The SVO native rectified reference has `f = 1078.944092 px` and `B = 0.119896226 m`.

The custom OpenCV rectification uses the custom K/D/R/T and `CALIB_ZERO_DISPARITY`:

| rectification | f=P1[0,0] (px) | B=abs(P2[0,3]/P2[0,0]) (m) | principal point |
|---|---:|---:|---|
| custom alpha=0 | 3635.497172 | 0.123730223 | left/right equal within 1e-6 px |
| custom alpha=1 | 1423.952587 | 0.123730223 | left/right equal within 1e-6 px |

The alpha=0 and alpha=1 focal values differ by design. The disparity is also measured in the corresponding virtual rectified image, so `f_custom` must only be paired with its own `d_custom`. The alpha-invariance check in the prior audit used the same raw correspondences in both virtual image systems and passed to floating-point precision; that validates the OpenCV geometry identity, not absolute metric accuracy.

At alpha=1 the custom/native rectified focal ratio is `1.319765`, while the raw focal ratios above are about `1.35`. Neither ratio proves that the calibration has exactly absorbed `n=1.333`.

## Unified extrinsic convention and baseline audit

All comparisons below use:

```text
X_right = R_left_to_right X_left + T_left_to_right
```

The SVO SDK raw transform is read as right-to-left and converted by:

```text
R_left_to_right = R_right_to_left^T
T_left_to_right = -R_right_to_left^T T_right_to_left
```

| quantity | native raw after conversion | custom YAML |
|---|---:|---:|
| `T_x` | -119.888519 mm | -122.435200 mm |
| `T_y` | +0.979333 mm | +0.167600 mm |
| `T_z` | -0.942721 mm | +17.853900 mm |
| `||T||` | 119.896225 mm | 123.730223 mm |
| baseline-axis tilt | 0.649615 deg | 8.296940 deg |

The three commonly quoted baseline numbers are not interchangeable:

- nominal hardware context: **120.000 mm**;
- custom YAML declared `baseline`: **122.4352 mm**, equal to `abs(T_x)`;
- custom full translation norm and custom rectified baseline: **123.730223 mm**.

The custom-minus-native translation difference after convention unification is:

```text
ΔT = [-2.546681, -0.811733, +18.796621] mm
||ΔT|| = 18.985717 mm
```

The converted rotation difference is `0.576303 deg`, with rotation vector `[0.00143785, 0.00995437, 0.00011891] rad`.

The `17.8539 mm` custom `T_z` is not evidence of a literal physical camera-center separation. It could be an effective/systematic term from a non-central underwater fit, a calibration artifact, or a convention/provenance mismatch. A rigid mechanical survey or known metric target is required to choose among those explanations.

## Provenance and metric scale

The repository search covered the README, detailed report, YAML/OpenCV files, calibration workbook, PDFs, scripts, visible tracked history, and calibration-related filenames/comments. The workbook contains 35 image-pair extrinsics and reprojection statistics, but no recorded target square size or independent distance reference.

| field | repository result |
|---|---|
| calibration medium | UNKNOWN |
| housing used | UNKNOWN |
| same housing as SVO recording | UNKNOWN |
| flat port / dome / shared ports | UNKNOWN |
| target type | UNKNOWN |
| target square size / metric scale | UNKNOWN; `ABSOLUTE_SCALE_NOT_IDENTIFIABLE` |
| calibration image count | 35 pairs in the workbook |
| calibration capture software/version | UNKNOWN; processing environment partially known (ZED SDK 5.4.1/OpenCV/Python) |
| model | native 12-coefficient RAD_TAN metadata versus custom 5-coefficient OpenCV model; physical provenance UNKNOWN |

The custom reprojection error and the observed depth stability are internal consistency evidence. They are not independent absolute-distance ground truth.

The final JSON also records a sampled radial distortion audit for native and custom K/D (16 azimuths over the image-corner radius). It reports negative radial steps, non-finite projections, and a possible foldover warning. This checks the implemented mapping over sampled rays; it is not a proof that a Brown/RAD_TAN fit is a physical non-central flat-port model.

The 35 calibration poses in the workbook are useful calibration-data clues, but without a target scale and known pose distances they are not metric ground truth.

## Required interpretation

The repository currently supports the statement “custom calibration gives an operational depth near 2.2–2.25 m for this scene.” It does not support either “this is definitely physical underwater distance” or “multiply it by 1.333.”
