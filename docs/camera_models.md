# Camera and geometry models

The package-wide stereo convention is:

```text
X_right = R_left_to_right @ X_left + t_left_to_right
```

The inverse used by an external consumer is `R.T` and `-R.T @ t`. The convention is encoded in the profile and validated on load.

`geometry.stereo` contains pure frame transforms, baseline computation, and camera-center helpers. `geometry.rectification` wraps OpenCV `stereoRectify` and exposes `P1`, `P2`, `Q`, maps, focal length, baseline, ROI, input size, output size, and alpha. `geometry.triangulation` distinguishes optical-axis Z depth from a generic ray midpoint.

The refractive modules are diagnostic geometry only. A `FlatPortModel` with missing port distances, refractive indices, normals, or housing information remains `unknown` and raises `RefractiveModelNotIdentifiable` when used for definitive tracing. Dome geometry is never silently represented as a flat port.

The existing custom profile is an empirical operational calibration. The repository does not claim that its parameters are a physical underwater model.
