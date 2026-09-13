# 水下双目相机数据处理与定位

最终深度审计结论见 [`FINAL_UNDERWATER_DEPTH_VERDICT.md`](FINAL_UNDERWATER_DEPTH_VERDICT.md)：当前仓库不能单独确定 absolute underwater metric depth；custom 标定的约 2.2 m 不应自动再乘 1.333。完整物理性、假设证伪和最小闭环实验见 [`CALIBRATION_PHYSICALITY_AUDIT.md`](CALIBRATION_PHYSICALITY_AUDIT.md)、[`DEPTH_HYPOTHESIS_FALSIFICATION.md`](DEPTH_HYPOTHESIS_FALSIFICATION.md) 和 [`MINIMUM_EXPERIMENT_TO_CLOSE_DEPTH_SCALE.md`](MINIMUM_EXPERIMENT_TO_CLOSE_DEPTH_SCALE.md)。

本项目用于离线处理 ZED 2i 双目相机录制的 SVO2 数据，重点验证独立 OpenCV 标定对深度、双目几何、位姿和三维重建结果的影响。项目同时提供 ZED 原生深度、OpenCV StereoSGBM、ZED GEN_1/GEN_3 tracking、ORB-SLAM3 以及 Agisoft Metashape 导出流程。

## 当前样例

工作区中的样例录制为 `20260802_150233.svo2`。根据现有导出元数据，它来自 ZED 2i，分辨率为 `1920x1080`，帧率为 `30 FPS`，共 `35855` 帧；使用的 ZED SDK 版本为 `5.4.1`。这些数据文件只用于本地复现实验，不建议直接提交到 Git 仓库。

## 处理流程

```text
SVO2
 ├─ read_svo2.py                         ZED 图像、NEURAL 深度、WORLD 位姿采样
 ├─ regenerate_depth_histogram.py        自定义 ZED 标定下的深度统计
 ├─ regenerate_sgbm_depth_histogram.py   OpenCV 标定 + StereoSGBM 深度统计
 ├─ benchmark_tracking.py                GEN_1 / GEN_3 tracking 对比
 ├─ export_pointcloud.py                 ZED 位姿 + 深度融合 WORLD RGB 点云
 ├─ export_gen1_pointcloud.py            GEN_1 位姿融合点云
 ├─ rerun_custom_calibration.py          自定义标定 + GEN_1 + 点云
 └─ SLAM/                                ORB-SLAM3 双目定位与 Metashape 导出
```

自定义标定和 ORB-SLAM3 流程将 SVO2 作为图像/时间戳来源，几何参数来自 `Calibration/`，不会把 SVO 内嵌标定直接用于生成的 ORB-SLAM3 设置或 OpenCV 双目结果。

## 目录说明

| 路径 | 作用 |
| --- | --- |
| `Calibration/zed_custom_opencv.yml` | ZED 可读取的自定义 OpenCV 标定文件 |
| `Calibration/标定结果/camera_intrinsics.yaml` | 左右相机内参和畸变参数 |
| `Calibration/标定结果/stereo_extrinsics.yaml` | 双目旋转矩阵 `R`、平移向量 `T` |
| `Calibration/标定结果/` | 标定报告、重投影误差和立体匹配检查结果 |
| `SLAM/CMakeLists.txt`、`SLAM/vcpkg.json` | ORB-SLAM3 的 Windows 构建配置和依赖声明 |
| `SLAM/config/20260802_150233_stereo.yaml` | 由独立标定参数生成的 ORB-SLAM3 双目设置 |
| `SLAM/*.ps1`、`SLAM/*.py` | 构建、运行、采样和可视化脚本 |
| `output/` | 所有实验生成物，默认不纳入版本控制 |

## 环境要求

当前脚本按 Windows + PowerShell 编写，主要依赖如下：

- ZED SDK 及与其版本匹配的 Python API `pyzed.sl`；
- Python 3，`numpy`、`opencv-python` 和 `matplotlib`；
- ORB-SLAM3 构建所需的 Visual Studio 2019、CMake、CUDA；
- `SLAM/vcpkg.json` 中声明的 Eigen3、Boost.Serialization、OpenSSL、OpenCV 4 和 Pangolin；
- `SLAM/ORB_SLAM3/Vocabulary/ORBvoc.txt`。

项目目前没有 `requirements.txt`。建议使用已经配置好 ZED Python API 的独立 Conda 环境运行脚本，并确认 `python` 指向该环境，而不是系统 Python。

构建脚本默认使用以下本机路径；如果本机安装位置不同，请先修改 `SLAM/build_orbslam3.ps1`、`SLAM/run_orbslam3_svo2.ps1` 和 `SLAM/CMakeLists.txt`：

```text
C:\Program Files (x86)\ZED SDK.old
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
C:\Users\10179\.conda\envs\zed\python.exe
```

## 快速开始

以下命令均在项目根目录执行。先将 `$python` 替换为可导入 `pyzed.sl` 的 Python 解释器：

```powershell
$python = "C:\path\to\zed\python.exe"
```

### 1. 采样导出图像和深度

`read_svo2.py` 会顺序打开 SVO2，并均匀采样左/右图像、NEURAL 深度、深度预览和 WORLD 位姿。`--max-frames 0` 表示导出全部帧；先用较小数量做冒烟测试：

```powershell
& $python .\read_svo2.py `
  .\20260802_150233.svo2 `
  --max-frames 100 `
  --output-dir .\output\sample100 `
  --no-display
```

主要输出包括：

```text
output/sample100/
├── left/                # 左目图像
├── right/               # 右目图像
├── depth_raw/           # float32、单位为米的深度
├── depth_preview/       # 深度预览图
├── pose_world.csv       # WORLD 坐标系位姿
└── metadata.json
```

### 2. 生成左右拼接视频

默认使用 ZED 已校正的 `LEFT | RIGHT` 图像；需要原始图像时加 `--unrectified`。

```powershell
& $python .\convert_svo2_to_video.py `
  .\20260802_150233.svo2 `
  --output .\output\preview.mp4 `
  --max-frames 300 `
  --scale 0.5 `
  --overwrite
```

### 3. 对比两种深度实现

ZED 自定义标定深度：

```powershell
& $python .\regenerate_depth_histogram.py `
  .\20260802_150233.svo2 `
  --calibration .\Calibration\zed_custom_opencv.yml `
  --max-frames 20 `
  --output-dir .\output\custom_depth_smoke `
  --overwrite
```

OpenCV StereoSGBM 深度：

```powershell
& $python .\regenerate_sgbm_depth_histogram.py `
  .\20260802_150233.svo2 `
  --intrinsics .\Calibration\标定结果\camera_intrinsics.yaml `
  --extrinsics .\Calibration\标定结果\stereo_extrinsics.yaml `
  --max-frames 20 `
  --output-dir .\output\sgbm_depth_smoke `
  --overwrite
```

两条流程都会输出中心像素/中心窗口深度 CSV、直方图和摘要 JSON。不要在完整录制上随意设置 `--save-depth-maps-every 1`，这会产生非常大的 float32 深度文件；脚本默认只保留统计结果和少量预览。

用于隔离比较 ZED 内嵌标定与自定义标定：

```powershell
& $python .\strict_compare_stereo_depth.py `
  .\20260802_150233.svo2 `
  --frames 20 `
  --output-dir .\output\strict_compare_20 `
  --overwrite

& $python .\compare_svo_and_calibration.py `
  .\20260802_150233.svo2 `
  --output .\output\calibration_comparison.json
```

### 4. tracking 和点云

先顺序回放 SVO2，比较 GEN_1 和 GEN_3 tracking：

```powershell
& $python .\benchmark_tracking.py `
  .\20260802_150233.svo2 `
  --mode both `
  --max-source-frames 300 `
  --output-dir .\output\tracking_smoke
```

使用采样深度和顺序位姿导出 WORLD RGB 点云：

```powershell
& $python .\export_pointcloud.py `
  .\20260802_150233.svo2 `
  --source-dir .\output\20260802_150233_sample1000 `
  --output-dir .\output\pointcloud1000 `
  --max-frames 1000
```

`export_gen1_pointcloud.py` 可将 `benchmark_tracking.py` 产生的 GEN_1 位姿用于同一批深度帧。完整参数请直接查看：

```powershell
& $python .\export_pointcloud.py --help
& $python .\export_gen1_pointcloud.py --help
& $python .\rerun_custom_calibration.py --help
```

### 5. 构建并运行 ORB-SLAM3

`SLAM/build_orbslam3.ps1` 会用 CMake + vcpkg 构建自定义的 `svo2_stereo.exe`。程序读取 SVO2 的未校正左右图像，使用 `Calibration/` 生成的双目设置进行 ORB-SLAM3 几何处理。

```powershell
Set-Location .
& .\SLAM\build_orbslam3.ps1

# 先跑少量帧验证环境
& .\SLAM\run_orbslam3_svo2.ps1 `
  -MaxFrames 300 `
  -Output output\orbslam3_smoke `
  -ImageScale 0.5
```

完整回放：

```powershell
& .\SLAM\run_orbslam3_svo2.ps1 `
  -MaxFrames 0 `
  -Output output\20260802_150233_orbslam3_stereo `
  -ImageScale 0.5
```

运行结果通常包括 `CameraTrajectory.txt`、`KeyFrameTrajectory.txt`、`tracking_log.csv`、`map_points_xyz.csv`、`run_metadata.txt` 和 `run_summary.txt`。输出目录已有文件时脚本可能拒绝覆盖，请使用新的目录名。

针对连续有效 tracking 段、Metashape 相机参考和稀疏彩色地图，可运行：

```powershell
& .\SLAM\run_orbslam3_resampled.ps1 `
  -ImageScale 0.5 `
  -MinSegmentFrames 3000 `
  -SkipInteractiveViewer
```

该流程会先分析最长连续有效段，再进行稳定段回放，最后生成 `sample1500/`、`sample3000/`、Metashape YPR CSV 以及 `stable_map/`。若要查看轨迹：

```powershell
& $python .\SLAM\visualize_camera_frames.py
& $python .\SLAM\visualize_camera_frames_interactive.py
```

### 6. 转换 Metashape 位姿

已有 ZED 或 ORB-SLAM3 轨迹时，也可以单独转换：

```powershell
& $python .\convert_pose_for_metashape.py --help
& $python .\convert_orbslam3_for_metashape.py --help
```

导出的坐标是局部坐标系，不是带 GPS 的地理坐标。导入 Metashape 前请阅读生成目录中的 `metashape_import_instructions.txt`；不要未经确认套用 ZED 坐标轴翻转。

## 标定更新约定

修改 `Calibration/标定结果/camera_intrinsics.yaml` 或 `stereo_extrinsics.yaml` 后，重新生成 ORB-SLAM3 设置文件：

```powershell
& $python .\SLAM\prepare_orbslam3_stereo.py `
  --root . `
  --output .\SLAM\config\20260802_150233_stereo.yaml `
  --scale 1.0
```

`--scale` 必须与实际送入 ORB-SLAM3 的图像缩放一致。改变标定或缩放后，应重新运行对应的深度/SLAM 流程，不要混用旧的 `output/` 结果。

## 版本控制说明

根目录当前是实验工作区，包含大量本地数据和编译缓存。根 `.gitignore` 已按以下原则配置：

- 保留 Python/C++ 源码、YAML 标定参数、CMake 配置和脚本；
- 忽略 `*.svo2`、`output/`、深度数组、点云、视频和各类编译产物；
- 忽略本地 vcpkg/Pangolin checkout 及 `vcpkg_installed/`；
- ORB-SLAM3 源码目录没有整体忽略，因为其中包含本项目的自定义改动；上游的示例数据集、evaluation 数据和 ORBvoc 压缩包单独忽略。

当前发布仓库会直接包含 `SLAM/ORB_SLAM3` 的源码和本项目改动，因此 clone 外层仓库后不会得到一个空的 gitlink。工作区中原有的嵌套 Git 元数据仅保留在本地，不作为外层仓库历史提交；Pangolin/vcpkg 仍作为可复现的外部依赖管理。

## 常见问题

### 找不到 `pyzed.sl` 或 DLL

确认使用的是安装了 ZED Python API 的解释器，并设置 `ZED_SDK_ROOT_DIR`，例如：

```powershell
$env:ZED_SDK_ROOT_DIR = "C:\Program Files (x86)\ZED SDK.old"
```

同时检查 CUDA、ZED SDK 和 Python API 的版本兼容性。

### 输出文件过大

完整 1920x1080 深度图和彩色点云都很占空间。先使用 `--max-frames 20/300` 做冒烟测试，再执行完整回放；深度图只在确有需要时开启 `--save-depth-maps-every`。

### ORB-SLAM3 运行失败或有效位姿很少

确认 `ORBvoc.txt`、SVO2、标定文件和生成的 `SLAM/config/20260802_150233_stereo.yaml` 都存在，并检查 `run_summary.txt` 中的 `pose_valid_ratio`。水下低纹理、反光、浑浊度和连续帧不足都会影响 tracking，不能只根据点云文件是否生成来判断结果质量。

## 许可

本项目根目录尚未单独声明许可证。`SLAM/ORB_SLAM3` 和其中的第三方组件请分别遵守各自目录中的许可证文件；发布或再分发前请补充本项目的许可证与数据使用说明。
