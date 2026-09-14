# Architecture

The package is organized around narrow boundaries:

```text
dataset config -> io.zed.ZedSession -> DepthFrame
                         |                    |
                         v                    v
                 tracking.zed          depth statistics/visualization

canonical calibration -> validation -> generated external configs
                     |                  |
                     v                  v
              geometry/rectification   ORB-SLAM3 / COLMAP / Metashape
```

`io.zed` is the only module that owns the vendor SDK handle. It imports `pyzed.sl` lazily, opens one SVO session, applies the explicit custom calibration, and compares the runtime raw/rectified calibration and resolution against the expected profile. Geometry and tests do not import the SDK.

All package-internal translations are metres. Camera intrinsics are pixels. The only intentional millimetre boundary is the generated ZED/OpenCV file, whose legacy format stores `T` in millimetres.

The image semantic is carried explicitly through the pipeline. A raw distorted pair must be labelled `RAW_UNRECTIFIED`; rectification produces `RECTIFIED`. ZED SDK `MEASURE.DEPTH` is a depth product in the selected camera frame, not an automatically interchangeable Euclidean range.

External processes are adapters. Calibration generation is deterministic; COLMAP refinement is an explicit flag; ORB-SLAM3 and Metashape outputs are labelled local frames with their scale source.
