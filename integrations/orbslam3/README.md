# ORB-SLAM3 integration

This directory contains the CMake/vcpkg build boundary and SVO2 runner for the preserved ORB-SLAM3 source at `../../third_party/ORB_SLAM3/`.

Prerequisites are the ORB-SLAM3 vocabulary, ZED SDK/CUDA, CMake/MSVC, and the local vcpkg dependencies. Set `ZED_SDK_ROOT_DIR`, `CUDA_TOOLKIT_ROOT_DIR`, and `UNDERWATER_SVO_PATH`, or pass explicit script parameters. Run:

```powershell
./build_orbslam3.ps1
./run_orbslam3_svo2.ps1 -MaxFrames 300 -Output outputs/orbslam3_smoke
```

The runner regenerates the settings file from the canonical profile. Full replay was not executed during this refactor verification.
