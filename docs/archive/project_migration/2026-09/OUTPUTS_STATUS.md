# `output/` 运行情况汇总

> 盘点时间：2026-09-14
> 盘点范围：仓库根目录下当前实际存在的 `output/` 目录及其 JSON、CSV、PLY、MP4、YAML 等产物。
> 判定方法：根据文件是否存在、`summary.json`/`metadata.json` 中的帧计数、验证字段和输出结构推断；本报告没有重新启动任何外部运行。

## 先说结论

当前仓库中没有 `outputs/`（复数）目录，实际存在并被盘点的是 `output/`（单数）。`output/` 是本机生成结果目录，已被 `.gitignore` 忽略，不应直接上传其中的大型原始产物。

总体情况如下：

- `output/` 下有 33 个一级目录、8,833 个递归文件，总大小约 32.56 GB（30.33 GiB）。其中最大的单项是 `20260802_150233_sample1000`，约 19.47 GB。
- 自定义 ZED SDK 深度已经完成整段 SVO 顺序回放：35,855/35,855 帧；中心区域的稳定运行结果约为 **2.204 m**。它是 operational custom-calibrated depth，不是独立物理 ground truth。
- 自定义标定的 SGBM 结果也有完整 1,000 帧版本；半分辨率版本中心窗口有效率 94.2%，中位数约 2.253 m。全分辨率和半分辨率结果不能脱离各自的标定/整流几何直接混用。
- native 与 custom 使用了不同的内参、整流焦距和基线，严格对比目录是诊断证据，不是把两个分支的 disparity 直接相减后的结论。
- native ZED tracking 的整段回放状态稳定；custom tracking/point-cloud 目录虽然帧计数大多完成，但出现明显的大步长轨迹异常，不能直接当作可信的物理轨迹。
- `custom_sdk_pointcloud_full` 的内部文件一致性验证通过，是当前点云结果中 provenance 最完整的一项；但它仍然没有独立的尺度或水下物理精度验证。
- ORB-SLAM3 有完整 diagnostic/full、stereo、custom 10% sampled 运行，也有 smoke 运行；`stable_map` 子目录的日志行数与元数据不一致，应视为未验证的中间产物。
- `sample1000` 下的 ALIKED + AdaLAM + COLMAP 结果已经生成多组特征、匹配、数据库、模型目录和部分 Metashape 导出文件。目录只能证明这些文件生成过，不能证明当前机器上已经完成 Metashape GUI 导入。
- 折射审计目前是 smoke/diagnostic 级别：使用了 `n=1.333` 的假设，但缺少玻璃厚度、玻璃折射率和独立 metric ground truth；不能据此推出统一的 `×1.333` 修正。

## 目录状态说明

| 标记 | 含义 |
|---|---|
| ✅ | 输出结构和回放/处理计数显示该运行已完成；不等于物理精度已验证 |
| ⚠️ | 已运行但属于 smoke、抽样、性能测试、参数实验，或结果存在需要复核的异常 |
| ⛔ | 空目录、缺少关键 summary，或内部文件明显不一致 |
| ℹ️ | 转码、预览、配置或元数据等辅助产物，不是独立算法运行 |

## 一级目录清单

文件数是递归统计值；大型目录的大小只在备注中标出。

| 一级目录 | 文件数 | 状态 | 当前含义 |
|---|---:|---|---|
| `_audit_smoke` | 1 | ⚠️ | 折射深度审计 smoke JSON |
| `_orb_uniform_random_smoke` | 6 | ⚠️ | ORB-SLAM3 随机均匀抽样 smoke，3 帧，不是整段回放 |
| `_orb_uniform10_smoke` | 6 | ⚠️ | ORB-SLAM3 均匀抽样 smoke，3 帧，不是整段回放 |
| `_sgbm_depth_perf` | 11 | ✅ | 全分辨率、100 帧 SGBM 性能/有效率测试 |
| `_sgbm_depth_perf_half` | 11 | ✅ | 半分辨率、100 帧 SGBM 性能/有效率测试 |
| `_zed_custom_neural_20` | 0 | ⛔ | 空目录，没有可读运行产物 |
| `20260802_150233_custom_depth_histogram` | 7 | ✅ | 自定义 calibration + ZED NEURAL 深度整段直方图 |
| `20260802_150233_custom_gen1_area_test` | 0 | ⛔ | 空目录，没有可读运行产物 |
| `20260802_150233_custom_gen1_area_test2` | 6 | ⚠️ | 自定义 GEN_1、约 1,000 帧 area-memory 测试 |
| `20260802_150233_custom_gen1_full` | 10 | ⚠️ | 自定义 GEN_1 整段回放和大点云；轨迹有大步长异常 |
| `20260802_150233_custom_gen1_sample10pct` | 6 | ⚠️ | 整段回放、10% 位置做 mapping 的大点云变体 |
| `20260802_150233_custom_sdk_pointcloud_full` | 6 | ✅ | 自定义 SDK 深度 + 点云整段回放，内部验证通过 |
| `20260802_150233_custom_sgbm_depth_20` | 31 | ⚠️ | 自定义标定、全分辨率、20 帧 SGBM 诊断；有效率偏低 |
| `20260802_150233_final_underwater_audit` | 0 | ⛔ | 空目录，没有最终审计文件 |
| `20260802_150233_flat_port_refractive_audit` | 0 | ⛔ | 空目录，没有平面端口审计文件 |
| `20260802_150233_native_svo_depth_samples` | 2 | ⚠️ | native SVO 整流视图上的少量 SGBM 深度样本 |
| `20260802_150233_orbslam3_resampled` | 12 | ⚠️ | 同时包含完整 diagnostic run 和不一致的 `stable_map` |
| `20260802_150233_orbslam3_smoke_new` | 6 | ⚠️ | ORB-SLAM3 500 帧 smoke |
| `20260802_150233_orbslam3_stereo` | 11 | ✅ | 半分辨率 stereo ORB-SLAM3 整段回放；有效状态约 91.9% |
| `20260802_150233_orbslam3_uniform10pct_custom` | 10 | ⚠️ | 自定义标定、10% 均匀抽样的 ORB-SLAM3 运行 |
| `20260802_150233_pointcloud1000` | 3 | ⚠️ | 旧版/较少 provenance 的 1,000 点云采样变体 |
| `20260802_150233_pointcloud1000_gen1` | 5 | ⚠️ | 旧版 GEN_1 1,000 点云采样变体 |
| `20260802_150233_refractive_depth_audit` | 0 | ⛔ | 空目录，没有当前审计产物 |
| `20260802_150233_refractive_depth_audit_v2_smoke` | 1 | ⚠️ | 折射审计 v2 smoke JSON |
| `20260802_150233_sample1000` | 8,603 | ⚠️ | 1,000 对同步图像及多组 ALIKED/AdaLAM/COLMAP 实验，约 19.47 GB |
| `20260802_150233_sgbm_depth_histogram_1000` | 11 | ✅ | 自定义标定、半分辨率、1,000 帧 SGBM 统计 |
| `20260802_150233_sgbm_depth_histogram_half` | 4 | ⚠️ | 只有整流/预览等中间文件，没有 `summary.json` |
| `20260802_150233_strict_calibration_compare_20` | 45 | ✅ | native/custom 各 20 帧的严格几何分支对比 |
| `20260802_150233_tracking_ab` | 0 | ⛔ | 空目录，没有 A/B tracking 结果 |
| `20260802_150233_tracking_ab_performance` | 2 | ✅ | PERFORMANCE + GEN_1 整段 tracking |
| `20260802_150233_tracking_custom_gen1_neural` | 3 | ⚠️ | 自定义 calibration + NEURAL + GEN_1 整段 tracking；有异常大步长 |
| `20260802_150233_tracking_custom_gen1_smoke` | 3 | ⚠️ | 自定义 tracking 100 帧 smoke |
| `20260802_150233_tracking_gen1_neural` | 3 | ✅ | native/embedded calibration + NEURAL + GEN_1 整段 tracking |

## 根目录辅助文件

这些文件位于 `output/` 根部，不代表额外的完整算法运行：

| 文件 | 类型 | 说明 |
|---|---|---|
| `_audit_calibration_comparison.json` | JSON | calibration 对比审计副本 |
| `_audit_orbslam3_half.yaml` | YAML | ORB-SLAM3 半分辨率配置/辅助文件 |
| `_convert_svo2_perf.mp4` + `.json` | MP4/JSON | SVO 转视频性能结果及元数据 |
| `20260802_150233_calibration_comparison.json` | JSON | calibration 对比诊断 |
| `20260802_150233_left_right.mp4` + `.json` | MP4/JSON | 左右目视频转换结果及元数据，视频约 2.7 GB |
| `20260802_150233_left_preview_frame000000.jpg` | JPEG | 单帧预览 |

## 深度运行

### 自定义 ZED SDK 深度

`20260802_150233_custom_depth_histogram/summary.json` 显示这是当前最完整的 custom SDK depth 整段结果：

| 项目 | 数值 |
|---|---:|
| 回放帧数 | 35,855 / 35,855 |
| 引擎 | ZED SDK NEURAL |
| 单位/参考系 | METER / CAMERA |
| 中心像素有效 | 35,706 / 35,855 = 99.584% |
| 中心像素深度中位数 | 2.2041 m |
| 中心 5×5 有效 | 35,723 / 35,855 = 99.632% |
| 中心 5×5 深度中位数 | 2.2041 m |

这里的 2.2041 m 应称为 **stable custom-calibrated operational depth**，可作为行为回归参考；没有独立水下尺、声学测距或其他 metric ground truth 时，不能称为真实水下距离。

### OpenCV SGBM

| 目录 | 分辨率/帧数 | 中心有效率 | 中心窗口有效率 | 中位数 | 运行判断 |
|---|---|---:|---:|---:|---|
| `custom_sgbm_depth_20` | 1920×1080 / 20 | 15.0% | 45.0% | 中心 2.1717 m；窗口 2.1869 m | ⚠️ 小样本且有效率低 |
| `sgbm_depth_histogram_1000` | 960×540 / 1,000 | 82.8% | 94.2% | 中心 2.2521 m；窗口 2.2530 m | ✅ 统计产物完整，但仍是诊断结果 |
| `_sgbm_depth_perf` | 1920×1080 / 100 | 36.0% | 56.0% | 中心 2.2414 m；窗口 2.2390 m | ⚠️ 性能/有效率测试 |
| `_sgbm_depth_perf_half` | 960×540 / 100 | 75.0% | 91.0% | 中心 2.2438 m；窗口 2.2449 m | ⚠️ 性能/有效率测试 |

半分辨率性能目录记录的耗时约为 7.88 s，全分辨率约为 59 s；同时半分辨率有效率明显更高。`sgbm_depth_histogram_half` 只有 `rectification.json` 和少量预览/CSV，没有深度汇总，因此不算完整运行。

`native_svo_depth_samples` 是 native 嵌入标定和 native rectified view 的少量样本，中心区域中位数约 2.0413 m。它和 custom SGBM/SDK 结果属于不同几何分支，不应直接做数值拼接。

## Calibration 与严格对比

`calibration_comparison.json` 和 `_audit_calibration_comparison.json` 给出的关键值：

| 项目 | native | custom |
|---|---:|---:|
| raw 左目焦距（约） | 1068 px | 1443 px |
| raw 右目焦距（约） | 1067.8 px | 1449 px |
| rectified focal（full/half） | 1078.944 px | 3635.497 px / 2140.972 px |
| baseline norm | 0.119896 m | 0.123730 m |
| custom `abs(Tx)` | — | 122.4352 mm |
| custom baseline 相对 120 mm 误差 | — | 2.0293% |
| custom stereo reprojection error | — | 0.2250 px |

因此 native 和 custom 的 disparity/depth 结果必须在各自的图像、整流矩阵、焦距和 baseline 下解释。`strict_calibration_compare_20` 遵循了“两个独立 fresh SVO handle、不混用 disparity”的诊断协议，产物完整，但它验证的是分支隔离和几何差异，不是绝对尺度真值。

## Tracking 运行

| 目录 | 配置 | 回放 | 有效状态 | 路径长度 | 最大步长 | 判断 |
|---|---|---:|---:|---:|---:|---|
| `tracking_gen1_neural` | NEURAL + GEN_1，IMU，area memory off | 35,855/35,855 | 35,855，100% | 232.306 m | 0.228 m | ✅ native 整段结果稳定 |
| `tracking_ab_performance` | PERFORMANCE + GEN_1，IMU，area memory off | 35,855/35,855 | 35,855，100% | 239.696 m | 0.411 m | ✅ 完整性能变体 |
| `tracking_custom_gen1_neural` | custom + NEURAL + GEN_1 | 35,855/35,855 | 35,855，100% | 985.484 m | 8.187 m | ⚠️ 名义上完整，但轨迹存在明显跳变 |
| `tracking_custom_gen1_smoke` | custom + NEURAL + GEN_1 | 100/100 | 100，100% | 0.801 m | 0.0115 m | ⚠️ 仅 smoke，不能代表整段表现 |

custom tracking 的“有效状态 100%”只表示 SDK 返回了有效 pose 行，不代表轨迹没有离群点；最大步长和总路径长度已经显示出需要进一步复核的异常。

## 点云与 Area Memory

| 目录 | 回放/采样 | 点数 | PLY 大小 | 其他状态 | 判断 |
|---|---|---:|---:|---|---|
| `custom_sdk_pointcloud_full` | 35,855 帧回放；1,000 个 mapping position | 75,885,506 | 1,138,282,772 bytes（约 1.138 GB） | mapping 与 PLY 点数一致；pose CSV 完整；custom file、K/R/T、SDK override 验证均通过 | ✅ 当前内部一致性最好的点云结果 |
| `custom_gen1_full` | 35,855/35,855；1,000 个 mapping position | 75,885,506 | 约 1.138 GB | tracking path 1,317.402 m，最大步长 128.145 m | ⚠️ 文件完成，但轨迹/融合结果需复核 |
| `custom_gen1_sample10pct` | 全段回放；3,586 个 sampled position（约 10%） | 106,620,743 | 约 1.599 GB | 与 full 变体有相同的大步长轨迹特征 | ⚠️ 抽样 mapping，不能当独立验证 |
| `custom_gen1_area_test2` | 前 1,000 帧 | 194,429 | 约 2.916 MB | 10 个 mapped frame；area map 保存成功 | ⚠️ 短时 area-memory 测试 |
| `pointcloud1000` | 旧版采样 | 82,892,278 | 1,243,384,352 bytes | 关键 SDK/camera/depth provenance 字段为空 | ⚠️ 历史/低 provenance 变体 |
| `pointcloud1000_gen1` | 旧版 GEN_1 采样 | 82,892,278 | 约 1.243 GB | 与上一项高度相似，关键 provenance 仍不完整 | ⚠️ 历史/重复变体 |

点云坐标是本地坐标系下的融合结果，当前目录没有独立地面真值或地理配准信息。`custom_sdk_pointcloud_full` 的 verification PASS 是文件布局、点数映射、配置和 SDK override 的内部一致性检查，不是水下尺度精度 PASS。

## ORB-SLAM3

| 目录/子目录 | 处理量 | 有效 pose | 是否到 SVO 末尾 | 判断 |
|---|---:|---:|---|---|
| `_orb_uniform_random_smoke` | 3 帧 | 3/3 | 否，抽样 smoke | ⚠️ |
| `_orb_uniform10_smoke` | 3 帧 | 2/3 = 66.7% | 否，抽样 smoke | ⚠️ |
| `orbslam3_smoke_new` | 500 帧 | 500/500 | 否，max-frame smoke | ⚠️ |
| `orbslam3_resampled/diagnostic_full` | 35,855 帧 | 35,855/35,855 | 是 | ✅ 完整 diagnostic run；连续段为 0–35,854 |
| `orbslam3_resampled/stable_map` | 元数据声称 30,248 帧 | 当前 tracking log 只有 8,802 个数据行 | 不可确认 | ⛔ 元数据与日志不一致 |
| `orbslam3_stereo` | 35,855 帧 | 32,935/35,855 = 91.86% | 是 | ✅ 完整 stereo run，但存在无效/碎片状态 |
| `orbslam3_uniform10pct_custom` | 3,586 个均匀抽样位置 | 3,235/3,586 = 90.21% | 随机访问，不适用 | ⚠️ custom 10% sampled run |

`orbslam3_stereo` 的最长连续有效段为 0–29,235（29,236 帧）；同时生成了 state-2 pose 的 Metashape YPR CSV。该 CSV 是转换导出，不代表已经由 Metashape 软件完成导入或优化。

## ALIKED / AdaLAM / COLMAP

### 输入数据

`20260802_150233_sample1000/metadata.json` 记录：SVO 共 35,855 帧，均匀抽取 1,000 个同步位置，生成左右目共 2,000 张图像，并附带原始 float32 depth。该目录目前约 8,603 个文件、19.47 GB。

### 各实验变体

| 运行目录 | 图像数 | 特征数 | 候选/接受 pair | raw accepted matches | verified ≥15 | Camera model | Metashape 导出 | 状态 |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `_smoke_stereo_full_opencv` | 6 | 4,800 | 7 / 7 | 4,257 | 7 | FULL_OPENCV | 0 | ⚠️ smoke |
| `sfm_aliked_adalam_100` | 100 | 78,853 | 485 / 464 | 130,284 | 464 | FULL_OPENCV | 0 | ✅ 实验变体 |
| `sfm_aliked_adalam_100_full_opencv_refined` | 100 | 78,853 | 99 / 99 | 51,805 | 99 | FULL_OPENCV | 1 | ✅ |
| `sfm_aliked_adalam_100_pinhole_window1` | 100 | 78,853 | 99 / 99 | 51,805 | 99 | PINHOLE | 0 | ⚠️ 相机模型实验 |
| `sfm_aliked_adalam_100_v2` | 100 | 78,853 | 485 / 463 | 125,858 | 463 | FULL_OPENCV | 0 | ✅ 实验变体 |
| `sfm_aliked_adalam_100_window1` | 100 | 78,853 | 99 / 99 | 51,805 | 99 | FULL_OPENCV | 0 | ⚠️ 参数实验 |
| `sfm_aliked_adalam_200_full_opencv_refined` | 200 | 157,801 | 298 / 298 | 168,201 | 298 | FULL_OPENCV | 1 | ✅ |
| `sfm_aliked_adalam_1000_full_opencv_refined` | 1,000 | 606,794 | 999 / 835 | 387,677 | 835 | FULL_OPENCV | 0 | ✅ 多模型输出 |
| `sfm_aliked_adalam_2000_full_opencv_refined` | 2,000 | 1,212,518 | 12,950 / 9,567 | 2,644,629 | 9,153 | FULL_OPENCV | 1 | ✅ 大型多模型输出 |

这些目录已经生成了特征、匹配、数据库和 COLMAP 模型相关文件，但属于多组参数/规模实验，不存在一个仅凭目录名就能认定的“最终最优模型”。当前结果还应注意：

- 常规变体使用 `FULL_OPENCV`；`100_pinhole_window1` 是专门的 `PINHOLE` 对照实验。
- 特征提取使用 ALIKED（n16、resize 1024、最多 800 keypoints，CUDA 可用）。
- 自定义 ALIKED descriptor 保存在 NPZ；COLMAP 侧的 custom descriptor 是零值 `uint8` 占位格式。因此不能把 COLMAP 数据库中的 descriptor 当作原始 ALIKED descriptor 质量的独立证明。
- “Metashape 导出”只表示生成了供 Metashape 使用的 CSV/manifest/model 文件；没有证据表明外部 Metashape GUI 导入、对齐和优化已经完成。

## Calibration / 折射审计

| 目录/文件 | 当前状态 | 结论边界 |
|---|---|---|
| `calibration_comparison.json`、`_audit_calibration_comparison.json` | ✅ | 记录 native/custom K、R、T、baseline、reprojection 等几何差异 |
| `strict_calibration_compare_20/comparison.json` | ✅ | native/custom 分支各 20 帧、独立 handle、避免混合 disparity |
| `_audit_smoke/refractive_depth_audit.json` | ⚠️ | 4 帧级别折射审计 smoke；包含 correspondence、triangulation 和跨分支诊断 |
| `refractive_depth_audit_v2_smoke/refractive_depth_audit.json` | ⚠️ | 同一前 N 帧、独立 handle 的 v2 smoke |
| `final_underwater_audit` | ⛔ | 空目录 |
| `flat_port_refractive_audit` | ⛔ | 空目录 |
| `refractive_depth_audit` | ⛔ | 空目录 |

v2 smoke 中 native/custom 的 Q consistency 都标记为 PASS，但 JSON 自己已经说明这是由同一套 `fB/d` 与 Q 生成关系得到的 implementation identity check，不是独立 SDK-Q 或物理折射验证。当前还缺少 flat-port 的玻璃厚度、玻璃折射率、相机到端口距离和独立距离真值。

## 重要解释与限制

1. **2.204 m 的含义**：它是当前 custom-calibrated ZED SDK depth 的稳定 operational 输出，适合做 regression/行为参考；不是已经独立测量确认的水下真实距离。
2. **不能直接乘 1.333**：折射介质、端口几何、相机标定介质、SDK 输出定义和测量 Z 轴都会影响关系；当前目录没有足够证据支持统一比例修正。
3. **native/custom 不可直接横比**：两者的 raw/rectified 图像几何不同，必须使用各自的 K、R、T、Q 和 disparity 定义。
4. **tracking 的有效 pose 不等于可信轨迹**：custom full 结果的最大步长达到 128.145 m（点云融合元数据）或 8.187 m（tracking 统计），需要先排查坐标变换、尺度和离群 pose。
5. **旧结果的 provenance 不完全一致**：部分早期 `metadata.json` 的 SDK、camera、depth 字段为空；部分 JSON 仍保存重构前的 `D:\Underwater\...`、`Calibration/...` 或 `SLAM/...` 历史路径。路径字符串是历史记录，不代表当前代码仍使用这些路径。
6. **目录存在不等于当前运行仍在执行**：本报告没有检测后台进程，也没有重新运行 ZED SDK、COLMAP、ORB-SLAM3 或 Metashape。

## 建议的后续处理

- 把 `20260802_150233_custom_depth_histogram` 的约 2.204 m 作为 operational depth regression reference，并在 `validation/reference/` 中保存小型 summary，而不是上传整段 output。
- 把 `custom_sdk_pointcloud_full` 作为当前点云内部一致性参考，同时单独记录“未做独立物理精度验证”。
- 对 custom tracking/point-cloud 的大步长先做坐标系、尺度、时间戳和 pose 离群检查，再决定是否用于科学结论。
- 将 `orbslam3_resampled/stable_map` 标为 stale/incomplete，除非重新生成与元数据一致的 tracking log。
- 对 `sample1000` 的 COLMAP 实验按“参数实验”归档；若要选最终模型，应另做统一评价和明确的模型选择依据。
- 不要在没有确认和备份前删除本机 `output/`；当前报告只读盘点，没有移动或删除任何输出。

## Git 状态

本报告位于仓库根目录；实际输出目录仍保持本机 ignored 状态。本次盘点本身是只读操作，没有重跑、移动或删除任何输出，也没有把大型 `output/` 产物加入 Git。该 Markdown 汇总可单独随代码上传。
