# Refactor change log

This file records behavior and topology changes made during the repository refactor. Numerical algorithms and calibration values are unchanged unless a change is listed here.

## 2026-09-13

- Canonicalized the checked-in custom stereo calibration as `calibration/profiles/zed2i_37395692_custom.yaml`; package translations are metres, while the ZED/OpenCV boundary writes millimetres as required by the external file format.
- Derived ZED, ORB-SLAM3, and COLMAP calibration files are generated from the canonical profile and include a source hash.
- Made raw/rectified ZED view selection explicit: `RAW_UNRECTIFIED` retrieves `LEFT_UNRECTIFIED`/`RIGHT_UNRECTIFIED`; `RECTIFIED` retrieves `LEFT`/`RIGHT`.
- Corrected scaled OpenCV rectification to pass the original intrinsic matrices with `newImageSize`; this preserves the prior alpha=0 half-resolution focal-length reference instead of scaling the matrices twice.
- Corrected generated OpenCV YAML ordering so `cv2.FileStorage` can read the generated file.
- Made SDK calibration verification reject missing/incomplete distortion metadata explicitly.
- Moved active code and integration helpers out of the repository root. Historical scripts and reports remain available in the documented archive locations or Git history.
- Added the `DepthFrame` interface and the `/16.0` StereoSGBM fixed-point conversion as explicit, tested behavior.
- Made depth video/CSV export incremental so a full recording does not buffer full-resolution depth arrays in memory.

No ZED depth, SGBM matching, ORB-SLAM3, COLMAP, refractive, or tracking algorithm was intentionally retuned.
