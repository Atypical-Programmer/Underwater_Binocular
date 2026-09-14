# Minimum Experiment to Close Underwater Depth Scale

The repository is in Case B: the remaining uncertainty is an external physical measurement, not another algebraic rearrangement of the existing files.

## Minimum data collection

Use the same camera, housing, port, focus, resolution, and recording settings as the target SVO. Use a rigid, textured, metrically known target whose square/marker dimensions are measured and recorded.

Acquire at least:

1. **three independent target ranges** spanning the intended working range (for example approximately 1.5 m, 2.2 m, and 3.0 m, adjusted to fit the tank); and
2. **five lateral/vertical positions at each range**: center, left, right, top, and bottom; and
3. **30 synchronized stereo frames per pose**, with a separate held-out subset for validation.

The ground-truth range must be defined explicitly. The preferred definition is the Euclidean distance from the measured left outer-port reference point to the target reference plane/point, with the same reference point used in the model output. Also record optical-axis `Z` separately; `Z` and Euclidean range are not interchangeable.

## Record the missing physical metadata

- air/water medium, temperature and salinity or the refractive-index source;
- housing make/model and a photograph or mechanical drawing;
- flat-port versus dome;
- shared versus separate ports;
- inner-interface distance for each camera;
- glass thickness and glass refractive index;
- port plane normals and a left/right camera pose survey in one rig frame;
- calibration target type, square size, and software/version;
- whether calibration was captured in air or underwater and whether the housing was installed;
- raw image resolution and every resize/rectification parameter.

If a dome is used, replace the flat-plane model with measured spherical port geometry; do not fit a plane by default.

## Analysis protocol

Run the frozen candidates on the same raw images:

- **A:** native raw K/D/T pinhole;
- **R1:** native raw K/D/T plus the measured rig-level refractive model;
- **B/R2:** custom K/D/T without an extra Snell multiplier;
- **R3:** custom plus Snell only as a labelled control, never as the final physical model.

Use model-agnostic correspondences for the first pass, then report custom-consistent correspondences separately. Report positive depth, rank, ray angle, ray gap, condition number, `Z`, and Euclidean range. Split the poses so that at least one distance/position is not used to fit the calibration.

For every pose compute:

```text
absolute_error = estimate - measured_ground_truth
relative_error = absolute_error / measured_ground_truth
```

Pre-register the acceptance threshold before looking at the results. A closure requires one candidate to pass the same threshold across all three ranges and all five positions, with no unexplained field-dependent or distance-dependent bias. The threshold must be chosen from the application requirement and the ground-truth instrument uncertainty, not from the current 2.2 m result.

## Closure criteria

Case A can be closed only if the measured experiment independently validates the model's absolute metric range and identifies the remaining field/depth residuals. If no candidate meets the pre-registered criterion, retain Case B and report the measured bias rather than applying a global 1.333 multiplier.
