# Rig-Level Refractive Model Audit

`rig_refractive_geometry.py` is an audit-only implementation of a shared-rig flat-port model. It is not used by the production depth, SGBM, ZED, or ORB-SLAM3 paths.

## Coordinate convention

Each camera uses optical coordinates `x=right, y=down, z=forward`. A camera pose is represented in one rig frame by:

```text
X_rig = R_rig_from_camera X_camera + C_rig
```

The default diagnostic rig frame is the left camera frame. A water ray begins at the camera, crosses an inner air/glass plane, crosses an outer glass/water plane, and then propagates in water. Water rays from both cameras are triangulated from their actual outer-interface origins; they are not forced to pass through the camera centers.

## Port geometry modes

| mode | implementation | status |
|---|---|---|
| shared flat | one common inner and outer plane in the rig frame | supported when measured parameters are supplied |
| separate flat | independent inner/outer planes for left and right cameras in the same rig frame | supported when both ports are measured |
| dome | spherical interface required | explicitly invalid for this flat-plane tracer |

The repository does not identify which mode is installed. A dome is never silently approximated as a plane.

## Physical parameters required for R1

The final R1 model requires `n_air`, `n_glass`, `n_water`, camera-to-inner-interface distance, glass thickness, both plane normals, and the camera poses in the common rig frame. These values are not present in the calibration YAML, workbook, PDFs, SVO metadata, or visible calibration history.

The existing camera-local model remains useful for a unit-level geometry sanity check, but it is not enough to claim shared housing geometry. The new rig-level model closes that abstraction gap without inventing the missing measurements.

## Synthetic validation

`test_rig_refractive_geometry.py` covers:

- no refraction / dry pinhole recovery;
- zero glass-thickness limiting case represented by coincident planes;
- normal incidence;
- symmetric target;
- off-axis target;
- shared versus separate port planes;
- dry pinhole round trip;
- dome rejection.

Together with the original `test_refractive_geometry.py`, the suite ran **19 tests, all passed**. The forward projector maps a water point through the two interfaces to a raw pixel, and the inverse tracer plus closest-point triangulation recovers the point in the synthetic cases.

## Current status

R1 is `NOT_IDENTIFIABLE_WITHOUT_MEASURED_PORT_PARAMETERS`. The generated `rig_refractive_sensitivity.csv` is a parameter sweep under an explicitly assumed shared, parallel flat port. It is labelled `HYPOTHETICAL_R1_SENSITIVITY_ONLY` and must not be reported as a recovered underwater depth.
