# Tracking and SLAM

`tracking.zed` owns the shared GEN_1/GEN_3 positional-tracking configuration and uses the already-open `ZedSession`. It does not open a second camera or SVO handle.

The ORB-SLAM3 integration lives under `integrations/orbslam3/` and uses the preserved source checkout at `third_party/ORB_SLAM3/`. Its calibration settings are generated from the canonical profile. The `T_c1_c2` matrix is explicitly documented as the inverse of the package left-to-right OpenCV transform.

Build and replay require the vendor SDK, CUDA, CMake/MSVC, vcpkg dependencies, vocabulary, and local SVO. Paths are supplied with parameters or environment variables; no developer-specific path is a tracked default.

ORB-SLAM3 poses are local reconstruction coordinates. A successful process launch does not prove tracking quality. Inspect frame status, valid-pose ratio, trajectory continuity, and map-point diagnostics together.
