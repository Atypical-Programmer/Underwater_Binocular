# 畸变矫正与中心/边缘深度核查

核查日期：2026-09-13
输入：`20260802_150233.svo2`，ZED 2i，1920×1080，35,855 帧
标定：`Calibration/zed_custom_opencv.yml`

## 结论

当前“中心深度大于边缘深度”的现象，不能简单判定为颜色映射写反，也不能简单判定为“生成深度图时完全没有做畸变矫正”。代码中确实存在两条明确且互不混用的路径：

1. Native 路径读取 ZED SDK 已经 rectified 的 `VIEW.LEFT/RIGHT`，没有再次 `remap`。
2. Custom 路径读取 `VIEW.LEFT_UNRECTIFIED/RIGHT_UNRECTIFIED`，从 `Calibration/` 读取 K/D/R/T，调用一次 `stereoRectify`，再各调用一次 `remap`，然后重新做 SGBM。

因此，“没有调用矫正”不是当前实现的准确描述。

但是，native 嵌入标定与 `Calibration/` 的水下等效标定不同，native 结果很可能包含防水罩条件下的径向系统误差。自定义标定在可靠的前段数据中确实把趋势改成了“中心略近、边缘略远”。这个结果支持“原生模型不完全适合当前水下光学条件”，但还不能证明自定义结果在全视频和绝对物理尺度上已经正确。

## 直接数据证据

### Native ZED SDK

在第 0 帧直接读取 SDK `MEASURE.DEPTH` 时：

| 位置 | 深度 |
|---|---:|
| 图像中心 | 约 2.042 m |
| 左侧边缘 | 约 1.391 m |
| 右侧边缘 | 约 1.380 m |

中心明显比边缘远。对应的 disparity 绝对值约为：中心 63.36 px、左边缘 93.00 px、右边缘 93.74 px；使用 SDK rectified `f=1078.944 px` 和 `B=0.119896 m` 时满足 `Z=fB/|d|`。所以 native 视频中 `blue=near、red=far` 的颜色关系没有反转。

### Custom 标定，一次矫正后重新 SGBM

以下是 `alpha=0`、默认 SGBM、图像输出坐标中心 `(960,540)` 的结果。每个数是对应半径区域的有效深度中位数：

| SVO 帧 | 中心 | 边缘 | 边缘-中心 |
|---:|---:|---:|---:|
| 0 | 2.216 | 2.242 | +0.026 |
| 5,000 | 2.319 | 2.321 | +0.001 |
| 10,000 | 2.065 | 2.108 | +0.044 |
| 15,000 | 2.172 | 2.228 | +0.056 |
| 20,000 | 2.468 | 2.528 | +0.060 |
| 25,000 | 2.178 | 2.228 | +0.049 |
| 30,000 | 2.170 | 2.135 | -0.035 |
| 35,000 | 2.310 | 1.855 | -0.456 |
| 35,854 | 2.500 | 1.867 | -0.633 |

在前 6 个抽样点，中心全部不大于边缘；后 3 个抽样点反向。

## 100 帧细采样与左右一致性

为了排除单向 SGBM 误匹配，又对 `alpha=0` 自定义路径进行了 100 个均匀 SVO 位置采样。每帧分别计算 left-to-right 和 right-to-left disparity，只保留左右 disparity 误差不超过 1.5 px 的像素。

结果：

| 筛选范围 | 有效帧 | 中心 < 边缘 | 中心 > 边缘 | 近似相等 |
|---|---:|---:|---:|---:|
| 全部 100 帧 | 100 | 65 | 33 | 2 |
| LR 有效率 ≥ 10% | 60 | 53 | 6 | 1 |
| LR 有效率 ≥ 15% | 48 | 44 | 3 | 1 |
| 帧号 < 27,000 | 75 | 62 | 12 | 1 |
| 帧号 ≥ 27,000 | 25 | 3 | 22 | 0 |

后 25 帧的 LR 一致有效率中位数只有约 2.6%，而前 27,000 帧约为 17.9%。后段原始图像纹理明显变弱，边缘中位数主要由极少量匹配像素决定，因此后段的“边缘更近”不适合用来判断标定是否正确。

LR 过滤没有把前段的“中心略近、边缘略远”消掉，说明该趋势不是单纯由单向 SGBM 随机错误造成的；但它也没有让全段视频都变成该趋势。

另外，对同一 `alpha=0` custom pipeline 把 `numDisparities` 从 256 提高到 512 做了第二次 100 帧检查：有效率 ≥10% 的 44 帧中，37 帧满足中心 < 边缘，7 帧相反。较大的搜索范围改善了部分近距离 disparity 接近上限的问题，但后段低纹理区域仍然不稳定。因此 256 像素上限是后段异常的一个放大因素，不是全部根因。

## 最容易混淆的地方：图像中心不是必然的光轴中心

对当前 custom 标定，OpenCV `stereoRectify` 的输出参数为：

| `alpha` | `f_rect` | `P1.cx` | `P1.cy` | 说明 |
|---:|---:|---:|---:|---|
| 0 | 3635.497 px | 1212.686 px | 532.640 px | 保留完整视场，虚拟主点明显右移 |
| 0.5 | 2529.725 px | 1212.686 px | 532.640 px | 中间裁剪/缩放 |
| 1 | 1423.953 px | 1212.686 px | 532.640 px | 保留黑边，实际有效 ROI 较窄 |

因此，`alpha=0` 下输出图像中心 `(960,540)` 并不是 rectified P1 的光轴中心 `(1212.686,532.640)`。如果以 `(960,540)` 统计，看到的“中心/边缘”比较混合了虚拟投影平移和真实场景视线；如果以 P1 主点为中心统计，9 个抽样点的 custom profile 并没有稳定满足 `center < edge`。

把 P1 主点强行平移到 `(960,540)` 是一个虚拟相机投影变换，不是重新获得了新的物理标定；它可以改变输出图像中每个像素对应的原始光线位置，不能单独作为“校正正确”的证据。

## 标定模型差异

`Calibration/` 的 raw 参数为：

- 左相机 `fx=1443.326 px`、`fy=1441.541 px`；
- 右相机 `fx=1449.830 px`、`fy=1448.098 px`；
- 左右主点分别约为 `(967.034,540.469)` 和 `(975.332,514.233)`；
- 畸变为 OpenCV 前 5 项 `[k1,k2,p1,p2,k3]`；
- `R` 在 `zed_custom_opencv.yml` 中是 3 元 Rodrigues 向量，已先用 `cv2.Rodrigues` 转成 3×3 矩阵；
- `T=[-122.4352,0.1676,17.8539] mm`，使用其 3D 范数得到 rectified baseline `123.730223 mm`。

SVO native raw 参数约为 `fx=1068 px`，而 native rectified 参数约为 `f=1078.944 px`。两者的 D 也不同。native 的 `VIEW.LEFT/RIGHT` 已由 SDK 用其嵌入模型矫正；custom 路径则明确使用 raw 图像和 `Calibration/` 的模型。两条路径没有把 native disparity 与 custom `f/B` 混配。

## 当前判断

可以确认的部分：

- native 深度的颜色和数值方向没有写反；
- native 路径不是漏掉了一次显式 `remap`，它读取的是 SDK rectified view；
- custom 路径确实从 raw 图像开始，并使用了 `Calibration/` 的 K/D/R/T 做一次完整矫正；
- custom 标定在可靠的前段数据中明显减弱了 native 的“中心远、边缘近”径向偏差，并多数时候得到“中心近、边缘远”；
- 后段反向结果与有效匹配率极低同时出现，不能作为全局标定结论。

还不能确认的部分：

- 不能仅凭“人眼认为中心近”证明所有帧、所有像素都应满足 `center < edge`；实际场景可能不是平面，也可能存在相机姿态/目标形状变化；
- 不能仅凭这组 profile 证明 `Calibration/` 是绝对正确的水下物理模型；
- 不能仅凭 profile 把剩余误差全部归因于畸变；还可能有弱纹理、重复纹理、反光、非刚体水体或场景真实深度变化。

## 可复现实验文件

- [`rectification_profile_experiment.py`](rectification_profile_experiment.py)：24 个 custom rectification/SGBM 组合；每个组合独立打开 SVO。
- [`RECTIFICATION_PROFILE_EXPERIMENT.json`](RECTIFICATION_PROFILE_EXPERIMENT.json)：9 个均匀位置的完整结果。
- [`rectification_lr_consistency_experiment.py`](rectification_lr_consistency_experiment.py)：双向 SGBM 左右一致性检查。
- [`RECTIFICATION_LR_CONSISTENCY_EXPERIMENT.json`](RECTIFICATION_LR_CONSISTENCY_EXPERIMENT.json)：6 个投影组合的结果。
- [`RECTIFICATION_LR_CONSISTENCY_ALPHA0_100.json`](RECTIFICATION_LR_CONSISTENCY_ALPHA0_100.json)：`alpha=0` 的 100 帧细采样结果。
- [`RECTIFICATION_LR_CONSISTENCY_ALPHA0_100_N512.json`](RECTIFICATION_LR_CONSISTENCY_ALPHA0_100_N512.json)：同一方案、`numDisparities=512` 的复核结果。
- [`UNDERWATER_DEPTH_ACCURACY_AUDIT.md`](UNDERWATER_DEPTH_ACCURACY_AUDIT.md)：已有的参数、单位、baseline 和深度链路审计。

最终工程结论：当前最合理的说法是“原生 SVO 标定与当前水下成像条件存在明显模型差异；自定义 Calibration + 一次 raw-image rectification 在前段可靠数据上修正了径向趋势，但全视频没有得到可无条件信任的中心/边缘 profile”。在拿到同一防水罩下的已知平面/已知距离目标之前，不应仅凭颜色图把某一套 profile 宣称为绝对真值。
