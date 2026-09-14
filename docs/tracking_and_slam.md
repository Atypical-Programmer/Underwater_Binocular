# Tracking and SLAM

`tracking.zed` owns the shared GEN_1/GEN_3 positional-tracking configuration and uses the already-open `ZedSession`. The public `underwater tracking zed` command performs the complete sequential replay, pose serialization, state accounting, and provenance write. It does not open a second camera or SVO handle inside one replay.

Use [runbooks/zed_tracking.md](runbooks/zed_tracking.md) for the Windows commands. `--mode BOTH` opens the SVO twice from the beginning and runs GEN_1 and GEN_3 independently; it never changes tracking mode on a stateful session.

The ORB-SLAM3 integration lives under `integrations/orbslam3/` and uses the preserved source checkout at `third_party/ORB_SLAM3/`. Its calibration settings are generated from the canonical profile. The `T_c1_c2` matrix is explicitly documented as the inverse of the package left-to-right OpenCV transform.

Build and replay require the vendor SDK, CUDA, CMake/MSVC, vcpkg dependencies, vocabulary, and local SVO. Paths are supplied with parameters or environment variables; no developer-specific path is a tracked default.

ORB-SLAM3 poses are local reconstruction coordinates. A successful process launch does not prove tracking quality. Inspect frame status, valid-pose ratio, trajectory continuity, and map-point diagnostics together.
