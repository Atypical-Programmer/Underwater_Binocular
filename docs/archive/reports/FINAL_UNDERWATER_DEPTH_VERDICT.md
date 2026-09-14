# Final Underwater Depth Verdict

Audit date: 2026-09-12.  Input SVO: `20260802_150233.svo2`.

> **THE REPOSITORY ALONE CANNOT DETERMINE ABSOLUTE UNDERWATER DEPTH.**

## First-screen answers

### Q1. Is the current approximately 2.2 m already physically validated underwater metric depth?

**No.** It is an internally consistent operational result from the external custom empirical pinhole calibration. No independent metric ground truth or complete physical calibration provenance is present.

### Q2. Should the current 2.2 m be multiplied by 1.333?

**No, not on the current evidence.** Applying `×1.333` to custom `fB/d` is only a control hypothesis. If the custom calibration absorbed an underwater/housing scale, it double-counts; if it did not, a global multiplier is still not a complete flat-port model.

### Q3. Is approximately 2.93 m established by the repository?

**No.** `2.2×1.333≈2.93 m` is a candidate calculation, not a measured result.

### Q4. What is the best available physical candidate?

The SVO embedded native raw K/D/R/T is the **best available dry/physical candidate**, with raw baseline norm `119.896225 mm`. The repository does not prove its calibration medium or that it represents the installed underwater housing.

### Q5. What is the custom calibration?

It is a **custom empirical underwater pinhole candidate**: raw left/right focal lengths around 1443/1450 px, custom `T=[-122.4352, 0.1676, 17.8539] mm`, and rectified alpha=1 `f=1423.952587 px`, `B=123.730223 mm`. Its physical provenance is UNKNOWN.

### Q6. Can the explicit refractive model produce an absolute R1 depth now?

**No.** The rig-level model is implemented and synthetic-tested, but the installed housing/port geometry, refractive indices, interface distances, and normals are missing. R1 is `NOT_IDENTIFIABLE` and emits no definitive depth.

### Q7. What has been verified?

- native/custom `fB/d` and OpenCV Q identities pass at floating-point precision;
- alpha=0/1 custom rectification is internally depth-invariant for the same raw correspondences;
- native/custom calibration and baseline conventions are logged separately;
- 19 synthetic camera-local/rig-level geometry tests pass;
- final SVO audit: 20 scattered frames, 46,024 candidate matches, Set A 1,958 and Set B 1,574 selected-quality correspondences;
- triangulation CSV reports positive depth, rank, ray angle, ray gap, condition number, Z, and range separately.

These are implementation and repeatability checks, **not absolute accuracy validation**.

### Q8. What is the minimum next experiment?

Use a metrically known target at at least three ranges and five positions per range, with the same installed housing, measured port geometry, recorded medium, and independent range ground truth. The exact protocol is in [`MINIMUM_EXPERIMENT_TO_CLOSE_DEPTH_SCALE.md`](MINIMUM_EXPERIMENT_TO_CLOSE_DEPTH_SCALE.md).

## Frozen model classification

| model | definition | classification |
|---|---|---|
| A | native raw K/D/T pinhole | best available physical/dry candidate; medium still UNKNOWN |
| B / R2 | custom raw K/D/T, own rectification, no extra Snell | valid operational empirical candidate; absolute accuracy unverified |
| R1 | native raw K/D/T + measured rig-level air/glass/water geometry | physically correct target model, currently NOT_IDENTIFIABLE |
| R3 | custom effective K/T + an additional Snell multiplier | legacy double-counting control only; not a physical estimate |

The old custom-plus-Snell sensitivity file is explicitly labelled `LEGACY_HYPOTHETICAL_CUSTOM_PLUS_SNELL` and `NOT_A_PHYSICAL_DEPTH_ESTIMATE`.

## Terminal summary

```text
calibration medium: UNKNOWN
housing used: UNKNOWN
same housing as recording: UNKNOWN
port type/shared geometry: UNKNOWN
target type: UNKNOWN
target square size / metric scale: UNKNOWN
calibration image count: 35 pairs recorded in workbook
software: ZED SDK 5.4.1 and OpenCV/Python processing known; capture/calibration software provenance UNKNOWN
intrinsic/stereo model: native 12-coefficient RAD_TAN metadata; custom 5-coefficient OpenCV RAD_TAN-like model
independent metric GT: NOT AVAILABLE
absolute scale: ABSOLUTE_SCALE_NOT_IDENTIFIABLE_FROM_REPOSITORY_EVIDENCE
physical refractive parameters: NOT AVAILABLE
```

## Final verdict

**Case B — data insufficient to close absolute depth scale.** The defensible operational policy is: keep using custom R2 `f_custom·B_custom/d_custom` only where its empirical behavior is useful, report it as unvalidated absolute metric depth, and do not apply an automatic extra `×1.333`. Close the question with the minimum external experiment, not with another unsupported scalar correction.

Supporting artifacts:

- [`CALIBRATION_PHYSICALITY_AUDIT.md`](CALIBRATION_PHYSICALITY_AUDIT.md)
- [`DEPTH_HYPOTHESIS_FALSIFICATION.md`](DEPTH_HYPOTHESIS_FALSIFICATION.md)
- [`REFRACTIVE_RIG_MODEL_AUDIT.md`](REFRACTIVE_RIG_MODEL_AUDIT.md)
- [`output/20260802_150233_final_underwater_audit/native_vs_custom_calibration.json`](output/20260802_150233_final_underwater_audit/native_vs_custom_calibration.json)
- [`output/20260802_150233_final_underwater_audit/correspondence_quality.csv`](output/20260802_150233_final_underwater_audit/correspondence_quality.csv)
- [`output/20260802_150233_final_underwater_audit/triangulation_quality.csv`](output/20260802_150233_final_underwater_audit/triangulation_quality.csv)
- [`output/20260802_150233_final_underwater_audit/rig_refractive_sensitivity.csv`](output/20260802_150233_final_underwater_audit/rig_refractive_sensitivity.csv)
