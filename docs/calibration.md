# Calibration

The computational source of truth is [calibration/profiles/zed2i_37395692_custom.yaml](../calibration/profiles/zed2i_37395692_custom.yaml). It is explicit about camera identity, resolution, distortion, stereo convention, translation units, and provenance. The original YAML, spreadsheet, and PDF evidence is retained under `calibration/source/2026_underwater_calibration/`.

Validate and generate derived artifacts with:

```powershell
underwater calibration validate
underwater calibration generate
```

Generated files under `calibration/generated/` are disposable derivatives. They contain a source profile hash and must not be hand-edited:

- `zed_custom_opencv.yml`: ZED/OpenCV `FileStorage` representation; translation is converted from metres to millimetres at this boundary.
- `orbslam3_stereo.yaml`: scaled camera values and the explicit inverse `T_c1_c2` expected by the ORB-SLAM3 integration.
- `colmap_camera.json`: explicit camera model and parameters, frozen by default.

The profile's translation is `[-0.1224352, 0.0001676, 0.0178539] m`. Its Euclidean baseline is `0.1237302227994842 m`; `abs(Tx)` is a separate value and is not substituted for the baseline.

After a ZED session opens, the runtime raw/rectified `K`, `D`, `R/T`, and resolution are compared with the profile. A mismatch is a hard failure. SDK self-calibration is disabled by the production session policy.
