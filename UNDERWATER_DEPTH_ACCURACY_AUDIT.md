# Underwater Stereo Depth Accuracy Audit

> 当前生产输出（2026-09-13）：最终完整 depth map 视频已经改为“ZED SDK `MEASURE.DEPTH` + `Calibration/zed_custom_opencv.yml`”。本次共导出 35,855/35,855 帧，原生 SVO 标定没有参与。完整配置、输出文件、中心深度统计、视频帧数验证和旧结果清理记录见 [`CURRENT_DEPTH_PROCESSING_REPORT_20260913.md`](CURRENT_DEPTH_PROCESSING_REPORT_20260913.md)。

> Final verdict and terminal summary: [`FINAL_UNDERWATER_DEPTH_VERDICT.md`](FINAL_UNDERWATER_DEPTH_VERDICT.md). The repository alone cannot determine absolute underwater depth; the final R1 rig model remains not identifiable without measured housing parameters.

审计日期：2026-09-12
审计对象：`20260802_150233.svo2`、native ZED 标定、`Calibration/` 下的 custom OpenCV 标定，以及仓库中的 SGBM、ZED depth 和 ORB-SLAM3 链路。

## 结论先行

本次审计的最终判定是 **B2：当前结果在内部是自洽的，但 calibration provenance unknown、没有独立绝对真值，不能宣称绝对测距精度已经验证**。

1. 仓库中约 `2.20–2.25 m` 的结果来自 custom 标定下的经验性针孔/立体模型；它与 custom SGBM 和 ZED `MEASURE.DEPTH` 的输出相互吻合。
2. 没有找到水下已知距离、量尺/激光、带已知尺度的目标、相机到目标的独立测量，或者带水槽/舷窗/玻璃信息的标定记录。因此，不能从现有仓库证明 `2.20–2.25 m` 是绝对物理距离，也不能证明它已经正确吸收了折射。
3. **不应把当前的 `2.20–2.25 m` 自动再乘 `1.333`。** `n·Z` 是本审计中明确计算的候选模型，不是由现有证据确认的修正。若 `2.2 m` 已经是水下/舷窗条件下的有效标定结果，再乘一次会明显过校正；若它是空气针孔距离，也不能仅凭一个全局 `n` 得到严格的平板舷窗模型。
4. OpenCV `P/Q`、`fB/d`、disparity `/16` 和“只做一次 rectification”的实现检查均通过；这些通过项只能证明内部计算链路一致，不能替代绝对真值。
5. 当前最稳妥的工程表述是：**继续使用 custom 标定的 `Z3 = f_custom·B_custom/d_custom` 作为未经绝对真值验证的 operational depth；同时把绝对精度标为未定，并禁止默认乘 `n=1.333`。**

第二轮审计把 verdict 细分为 **B2**：内部几何链路通过，但 custom calibration 的介质/housing provenance 本身也是 unknown，且没有独立绝对 GT。

### Executive questions

- **Q1：当前代码实现的 `fB/d` 是否正确？——CONDITIONAL YES。** 在同一套 rectified 坐标、同一 `P/Q`/baseline 单位和 `/16` disparity 下，custom `fB/d` 与 OpenCV Q 一致，alpha=0/1 对同一 raw correspondence 也保持深度不变；这证明实现几何自洽，不证明标定物理尺度正确。
- **Q2：当前 custom depth `2.2 m` 是否应乘 `1.333`？——NOT JUSTIFIED。** `2.2×1.333≈2.93 m` 只是 H1 候选；既没有 measured flat-port 参数，也没有证据支持一个全局乘法。
- **Q3：`2.2 m` 是否已被证明是绝对物理距离？——NO。** 仓库没有 independent metric GT，且 calibration medium/housing provenance unknown。

### Calibration-provenance decision tree

```text
IF custom calibration was performed underwater with the same housing
and a correctly scaled metric target:
    extra ×1.333 is almost certainly double-counting;
    remaining question is non-central/refractive model bias.
ELSE IF custom calibration was performed in air:
    current 2.2 m is not validated underwater metric depth;
    a global ×1.333 is still not a rigorous flat-port solution.
ELSE:
    provenance unknown; H0/H1 cannot be closed; treat current result as B2/H2.
```

本报告不把仓库已有的“2.2 m 已经包含折射”等结论当作前提，而是重新追踪代码、标定参数、SVO 图像和 100 帧实验结果。

## 1. 审计范围、证据和可复现输出

### 1.1 直接检查的主要文件

- 图像和 SDK 链路：`read_svo2.py`、`regenerate_depth_histogram.py`。
- 独立 OpenCV/SGBM 链路：`regenerate_sgbm_depth_histogram.py`、`strict_compare_stereo_depth.py`、`compare_svo_and_calibration.py`。
- 标定生成与方向探测：`rerun_custom_calibration.py`。
- SLAM 链路：`SLAM/prepare_orbslam3_stereo.py`、`SLAM/config/20260802_150233_stereo.yaml`、`SLAM/ORB_SLAM3/src/Settings.cc`、`SLAM/ORB_SLAM3/Examples/Stereo/svo2_stereo.cc`。
- 标定来源：
  `Calibration/zed_custom_opencv.yml`、
  `Calibration/标定结果/camera_intrinsics.yaml`、
  `Calibration/标定结果/stereo_extrinsics.yaml`、
  `Calibration/标定结果/calib_full_params.xlsx`、
  `Calibration/标定结果/Camera-Centric.pdf`、
  `Calibration/标定结果/重投影误差.pdf`。
- 本次新增的独立诊断：`debug_refractive_depth_check.py`。

### 1.2 本次实验

诊断脚本分别打开两个新的 SVO handle，处理相同的前 100 帧，但在两个完全独立的图像坐标系中计算 disparity：

```powershell
$env:ZED_SDK_ROOT_DIR='C:\Program Files (x86)\ZED SDK.old'
& 'C:\Users\10179\.conda\envs\zed\python.exe' .\debug_refractive_depth_check.py `
  .\20260802_150233.svo2 `
  --frames 100 `
  --rectify-alpha 1 `
  --output-dir .\output\20260802_150233_refractive_depth_audit `
  --overwrite
```

完整 JSON 结果：[`output/20260802_150233_refractive_depth_audit/refractive_depth_audit.json`](output/20260802_150233_refractive_depth_audit/refractive_depth_audit.json)。

脚本明确计算四种候选量：

```text
Z1 = Z_native            = f_native · B_native / d_native
Z2 = Z_native · n        = 1.333 · Z1
Z3 = Z_custom            = f_custom · B_custom / d_custom
Z4 = Z_custom · n        = 1.333 · Z3
```

其中 `n=1.333` 只是用户问题中提出的水折射率候选值；脚本没有把它作为生产修正写入任何现有流程。

## 2. 从 raw image 到 depth 的实际数据流

| 分支 | 输入图像 | 标定/rectification | disparity | depth |
|---|---|---|---|---|
| Native `Z1` | ZED `VIEW.LEFT`、`VIEW.RIGHT` | SVO/SDK 原生 rectified 图像和原生 rectified 投影参数；没有再次 `remap` | OpenCV StereoSGBM，原始 fixed-point disparity 除以 `16.0` | `f_native_rect·B_native/d_native` |
| Native `Z2` | 与 `Z1` 相同 | 与 `Z1` 相同 | 与 `Z1` 相同 | `1.333·Z1`，仅作为候选模型 |
| Custom `Z3` | ZED `VIEW.LEFT_UNRECTIFIED`、`VIEW.RIGHT_UNRECTIFIED` | 读取 `Calibration/` 的 K、D、R、T，调用一次 OpenCV `stereoRectify`，再各做一次 `cv2.remap` | 同一组 raw unrectified 图像经 custom rectification 后的 SGBM，除以 `16.0` | `f_custom_rect·B_custom_rect/d_custom` |
| Custom `Z4` | 与 `Z3` 相同 | 与 `Z3` 相同 | 与 `Z3` 相同 | `1.333·Z3`，仅作为候选模型 |

关键代码证据：

- `strict_compare_stereo_depth.py:343-400` 的 native 分支直接取 `VIEW.LEFT/RIGHT`；`strict_compare_stereo_depth.py:439-546` 的 custom 分支取 unrectified 图像、做 custom `remap` 和 SGBM。
- `regenerate_sgbm_depth_histogram.py:217-280` 构造 OpenCV rectification；`regenerate_sgbm_depth_histogram.py:812-835` 取 unrectified 图像并把 SGBM 输出除以 `16.0`。
- `debug_refractive_depth_check.py:609-713` 和 `:715-830` 对两条链路做了同样的显式记录，遇到 SDK 错误、尺寸错误或无效参数会直接失败，不会静默切换到另一种图像源。
- `read_svo2.py:350-359` 与 `regenerate_depth_histogram.py:517-525` 的 `MEASURE.DEPTH` 是 ZED SDK 深度链路，不应与独立 OpenCV SGBM disparity 混为同一观测量。

因此，审计中没有发生“native disparity 配 custom `f`/`B`”或“custom disparity 配 native `f`/`B`”的混用。

## 3. 标定参数和来源审计

### 3.1 SVO native 参数

从 SVO embedded calibration 读到：

| 参数 | Left raw | Right raw |
|---|---:|---:|
| `fx` px | 1068.0000 | 1067.8300 |
| `fy` px | 1067.7700 | 1067.7100 |
| `cx` px | 957.4000 | 952.5700 |
| `cy` px | 539.5910 | 511.7760 |
| distortion | ZED RAD_TAN，12 项 | ZED RAD_TAN，12 项 |

SVO 原生 rectified 参数为：

- `f_native_rect = 1078.944091796875 px`，`fx=fy`；
- principal point 约为 `(955.63647, 524.44782)` px；
- `B_native = 0.11989622563 m`；
- rectified distortion 为零/由 SDK 的 remap 隐含处理。

### 3.2 Custom 标定参数

`camera_intrinsics.yaml`、`zed_custom_opencv.yml` 和 `calib_full_params.xlsx` 的数值一致到输出精度。raw custom K 为：

| 参数 | Left | Right |
|---|---:|---:|
| `fx` px | 1443.326338 | 1449.830254 |
| `fy` px | 1441.541002 | 1448.097563 |
| `cx` px | 967.033878 | 975.331674 |
| `cy` px | 540.469238 | 514.232843 |
| distortion | `[0.3151847,-0.7495814,0.002129304,0.005759798,6.688496]` | `[0.3023032,-0.1692313,0.001685104,0.01069086,1.944058]` |

custom 左右相对位姿为：

```text
R = [[ 0.999968008,  0.00548864430, 0.00581871018],
     [-0.00548930265, 0.999984929, 0.0000971791651],
     [-0.00581808910,-0.000129116717,0.999983066]]

T = [-122.4352, 0.1676, 17.8539] mm
```

`stereo_extrinsics.yaml` 显式写有：

- `baseline: 122.4352 # unit mm`；
- `reprojection_error: 0.225048 # px`；
- `scale_error_percent: 2.0293 # relative to 120mm nominal baseline`。

`calib_full_params.xlsx` 给出的高精度 baseline 是 `122.4351935822 mm`，35 对标定图像的 mean reprojection error 是 `0.2250480131 px`。两个 PDF 只支持这些自标定结果：`重投影误差.pdf` 给出整体约 `0.23 pixels`，`Camera-Centric.pdf` 是 35 个标定板位姿的相对/相机中心可视化。

这几份文件没有提供：标定板物理方格边长的可追溯记录、相机到平板舷窗距离 `h`、玻璃厚度、玻璃折射率、相机是在水中还是空气中完成标定、拍摄对象的独立绝对距离。因此它们不能单独构成水下绝对尺度真值。

### 3.3 “custom 焦距约为 native 的 1.33 倍”不能直接等同于折射率

custom/raw 与 native/raw 的焦距比为：

- Left `fx`: `1.351429`；Left `fy`: `1.350048`；
- Right `fx`: `1.357735`；Right `fy`: `1.356265`。

这只是同一分辨率下两套不同标定的参数比。两套 principal point 也不同，distortion 模型/系数形式和数值也明显不同；rectification 后还会重新改变有效焦距。因此不能依据“接近 1.33”推导出 `f_custom = n·f_air`，也不能依据这个比值证明 2.2 m 已或未包含折射。

## 4. 深度公式、单位和 baseline 审计

### 4.1 实际公式

对同一套 rectified 坐标、同一单位的投影参数，代码使用：

```text
Z = f_rect[px] · B_rect[m] / d[px]
```

custom OpenCV `stereoRectify` 的 `T` 和 `P/Q` 在本仓库中以 mm 输入/输出；因此 custom 分支将 `abs(P2[0,3]/P2[0,0])` 从 mm 转成 m。SGBM 的原始 `int16` disparity 明确除以 `16.0`，不是把整数值直接代入。

### 4.2 Baseline 对照

| 定义 | 数值 |
|---|---:|
| YAML 注释中的 nominal baseline | `120.000000 mm` |
| calibration 声明值/`abs(Tx)` | `122.435200 mm` |
| `||T||` | `123.730222800 mm` |
| custom rectified `abs(P2[0,3]/P2[0,0])` | `123.730222800 mm` |
| `Tz` | `17.853900 mm` |

本报告以后严格使用以下名称，避免把 `abs(Tx)` 泛称为 baseline：

```text
B_nominal  = 120.000000 mm       # 文件注释/官方产品 nominal context
B_x        = |Tx| = 122.435200 mm # 标定文件声明的 x 分量
B_3D       = ||T|| = 123.730223 mm
B_rectified= |P2[0,3]/P2[0,0]| = 123.730223 mm
```

ZED 2i 产品资料的 nominal baseline 也是约 `120 mm`（[Stereolabs ZED 2i datasheet](https://support.stereolabs.com/hc/en-us/article_attachments/27901419901463)）；这只是产品规格背景，不是本台相机/当前 housing 的 independent metric GT。

相对差异：

- 声明值相对 nominal：`+2.029333%`，与 YAML 的 `scale_error_percent` 一致；
- `||T||`/P2 baseline 相对 nominal：`+3.108519%`；
- `||T||`/P2 baseline 相对声明的 `122.4352 mm`：`+1.057721%`；
- `Tz/||T|| = 0.144297`，source translation 的 x 轴与 norm 并不相同。

`Tz/B_3D≈14.43%` 对应 source translation 的约 `8.30°` 偏离 x 轴。官方 ZED 2i 资料只给约 `120 mm` 产品 baseline，没有给当前设备安装姿态、housing 光学轴或 `Tz` 的允许范围；因此本值的当前分类是 **physically plausible in magnitude but unidentifiable as a mechanical/housing quantity**，不能说是折射吸收，也没有足够证据判为机械异常。

实际 `P2` 检查为：

- alpha=1：`f_custom_rect=1423.952587176348 px`，`B=0.123730222799 m`；
- alpha=0：`f_custom_rect=3635.497171961014 px`，`B=0.123730222799 m`；
- 两种 alpha 下 `P2[0,3] ≈ -f·B`，残差约 `10^-11` mm 量级。

所以本审计的 custom `Z3` 使用的是 `P2` 定义的 rectified horizontal baseline，即 `123.730223 mm`，并且将其转为 `0.123730223 m`。不能把 `120 mm`、`122.4352 mm` 和 `123.7302 mm` 无说明地互换。

### 4.3 R/T 方向风险

标定文件注释写的是“Right cam -> Left cam”，而 `zed_custom_opencv.yml` 中的 R/T 被直接用于 OpenCV `stereoRectify`。`rerun_custom_calibration.py` 对 direct/inverse 两个候选做了 ZED 导入和正视差探测：direct 候选能够被 ZED 接受并产生正 disparity；inverse 候选在 ZED 打开时为 `INVALID CALIBRATION FILE`。这证明当前 operational 文件路径是可运行的，但不等同于独立证明 exporter 注释和 OpenCV/ZED 语义完全一致。

SLAM 路径在 `SLAM/prepare_orbslam3_stereo.py:103-106` 显式写入：

```text
R_orb = R_source.T
T_orb = -R_source.T @ (T_source_mm / 1000)
```

ORB-SLAM3 的 `SLAM/ORB_SLAM3/src/Settings.cc:485-518` 又对 `Tlr_` 求 inverse 后做内部 rectification，并使用 `b_=||Tlr_.translation()||`、`bf_=b_·P1[0,0]`。这个双向转换在实现上是有记录的，但仍建议把源文件的坐标语义改成明确的“OpenCV stereoRectify 输入约定”，并用独立已知靶标最终闭环验证方向。

## 5. 100 帧四模型实验

以下统计来自前 100 帧；每个区域先对每帧有效像素取中位数，再对帧中位数统计。`mean/std/P05/P95` 因而描述帧间结果，不是全图每一个像素的 pooled distribution。

### 5.1 native/custom optical-center neighborhoods（不是同一物理 ray）

native 和 custom 分别使用各自 rectified 坐标中的 optical-center neighborhood；这两个 neighborhood 不是同一个 raw pixel、不是同一个 physical ray。该表只用于观察数量级和模型关系，不能当作 GT 对照，也不应称作 corresponding optical center。

| 模型 | median (m) | mean (m) | std (m) | P05 (m) | P95 (m) | 有效帧 |
|---|---:|---:|---:|---:|---:|---:|
| `Z1` native | 2.055393 | 2.040308 | 0.151021 | 2.021271 | 2.090794 | 100 |
| `Z2` native×1.333 | 2.739839 | 2.719731 | 0.201311 | 2.694354 | 2.787028 | 100 |
| `Z3` custom | 2.252478 | 2.248115 | 0.018066 | 2.219144 | 2.277224 | 100 |
| `Z4` custom×1.333 | 3.002553 | 2.996737 | 0.024081 | 2.958119 | 3.035540 | 100 |

同一帧对应区域的比值统计：

| 比值 | median | mean | P05 | P95 |
|---|---:|---:|---:|---:|
| `Z3/Z2` | 0.820802 | 0.842930 | 0.805713 | 0.836006 |
| `Z3/Z1` | 1.094129 | — | — | — |

上述首轮表的数值来自 100 帧统计；第二轮修正了 frame-ID 对齐后，image-center、central-40% 和两个 optical-center neighborhoods 的 `Z3/Z2` 中位数分别为 `0.801468`、`0.833315`、`0.820802`，有效配对帧数分别为 `95`、`100`、`100`。`Z3/Z2` 明显不是 1；但由于两套 rectified 图像坐标、disparity 和有效光线不同，这只能说明“不能拿 native disparity 直接解释 custom 结果”，不能单独证明 custom 或 native 哪一套是绝对正确的。

### 5.2 同一输出图像中心与中央 40% 区域

同一输出坐标中心的中位数为：

| 区域 | `Z1` | `Z2` | `Z3` | `Z4` | `Z3/Z2` |
|---|---:|---:|---:|---:|---:|
| image center | 2.061535 | 2.748026 | 2.198889 | 2.931119 | 0.803365 |
| central 40% | 2.007547 | 2.676060 | 2.230202 | 2.972859 | 0.833315 |

同一输出中心的 custom 有效帧为 95/100；中央 40% 的 custom 结果有效帧为 100/100。中央 40% 的帧间统计为：

- `Z1`: mean `2.005219 m`，std `0.011285 m`，P05–P95 `1.990174–2.021271 m`；
- `Z2`: mean `2.672956 m`，std `0.015043 m`；
- `Z3`: mean `2.228578 m`，std `0.005795 m`，P05–P95 `2.217919–2.235508 m`；
- `Z4`: mean `2.970694 m`，std `0.007725 m`。

这说明 custom 2.2 m 结果在本视频的重复帧上很稳定，但“稳定”是 precision/重复性信息，不是 accuracy/绝对正确性信息。

### 5.3 `fB` 量级对照

alpha=1 时：

- `f_custom/f_native = 1.319765`；
- `B_custom/B_native = 1.031978`；
- `(f_custom·B_custom)/(f_native·B_native) = 1.361968`。

因此 native 与 custom 不仅是“同一个 disparity 乘不同焦距”。两条链路得到的 disparity 分布也不同：100 帧全有效 disparity 样本的中位数约为 native `69 px`、custom `79.0625 px`。这些 disparity 来自不同 rectified 坐标，必须分别和各自的 `f/B` 配对。

## 6. Q 与 `fB/d` 的一致性

`debug_refractive_depth_check.py:450-503` 对相同 disparity 同时计算：

1. 直接公式 `f·B/d`；
2. `cv2.reprojectImageTo3D(disparity, Q)[2]`。

custom Q 结果由 mm 除以 `1000` 转为 m；native diagnostic Q 的长度单位是 m。第二轮把两者严格分开报告，100 帧各抽取 `500,000` 个样本：

| 分支 | Q 来源 | 样本数 | 最大绝对差 (m) | 最大相对差 | 状态 |
|---|---|---:|---:|---:|---|
| native | 由直接读取的 SDK rectified P-like 参数生成的 diagnostic Q | 500,000 | `7.62939453e-06` | `1.1916429e-07` | PASS（非独立 SDK-Q 验证） |
| custom | OpenCV `stereoRectify` 生成的 Q | 500,000 | `7.75146485e-06` | `1.1878576e-07` | PASS（独立实现一致性检查） |

native 的 Q 是诊断根据直接读取的 SDK rectified 投影参数构造的；当前 SDK API/记录中没有直接返回 native Q，因此 native 对比不能称为独立验证 ZED SDK Q。custom Q 是 OpenCV `stereoRectify` 的直接输出，验证强度更高。两者都支持“当前 Q/公式没有单位或符号级别的明显 bug”，都不支持“标定绝对尺度已被验证”。

## 7. disparity、resize、rectification 和 SLAM 复核

### 7.1 disparity fixed-point 和符号

所有本次 SGBM 分支都使用：

```python
disparity = matcher.compute(left, right).astype(np.float32) / 16.0
```

并采用正视差 `d = u_left - u_right`。独立特征对应检查（frame 0）得到：

- native：手工对应 disparity 中位数约 `69 px`，SGBM 在同一批 keypoint 上约 `68.6875 px`，两者中位差约 `0.0125 px`；
- custom：手工对应 disparity 中位数约 `79.19995 px`，SGBM 约 `78.875 px`，两者中位差约 `0.0625 px`。

这只是 disparity 实现检查；特征点跨多个深度，不能拿其 depth 中位数作为绝对 GT。

### 7.2 是否重复 rectification

- native 实验取 SDK 已 rectified `VIEW.LEFT/RIGHT`，没有再调用 OpenCV `remap`；
- custom 实验取 `VIEW.*_UNRECTIFIED`，只调用一套 custom map；
- `regenerate_sgbm_depth_histogram.py` 的 alpha=1 输出与本审计的 custom alpha=1 参数一致；
- ZED `MEASURE.DEPTH` 另走 SDK pipeline，不与 SGBM 图像混用。

因此，当前审计没有发现“对已经 rectified 图像再次套 custom map”的重复校正路径。需要注意的是，`strict_compare_stereo_depth.py`/历史输出使用过 alpha=0，而本次 100 帧对照使用 alpha=1；这会改变 `f_rect`、principal point 和有效 ROI，不能跨 alpha 直接比较 depth 数值。

custom SGBM 与 ZED custom `MEASURE.DEPTH` 不是两个 independent GT：两者共享 custom calibration 和底层 stereo images/几何。它们的约 2.2 m agreement 只能归入 Level 1/2 implementation/geometric consistency，不能升级为 Level 4 physical validation。

### 7.3 resize/scale

- custom 100 帧审计在完整 `1920×1080` 上运行，未做隐藏 resize；
- alpha=1 custom：ROI left `[498,261,992,539]`，`f=1423.9526 px`；
- alpha=0 full：`f=3635.4972 px`；
- 已有 half-resolution alpha=0 输出为 `960×540`、`f=2140.9715 px`，属于另一套输出坐标系；
- `debug_refractive_depth_check.py` 对 `--scale != 1` 直接报错，避免把未经同步缩放的 K/disparity 混进审计。

### 7.5 同一 correspondence 的 alpha=0/1 invariance

第二轮新诊断对相同 raw correspondence 分别调用 alpha=0/1 的 `undistortPoints(R,P)`，再计算各自的 `d` 和 `fB/d`，没有比较两个 alpha 下不同的 SGBM 像素。20 个分散帧（frame 0 + 19 个分散帧）全部有高质量匹配，共得到 863 个 correspondence：

- alpha=0：`f=3635.497171961014 px`，`B=0.123730222799 m`；
- alpha=1：`f=1423.952587176348 px`，`B=0.123730222799 m`；
- `|Z_alpha0-Z_alpha1|/Z_alpha1` P95 `4.99e-15`，最大 `9.35e-15`，status `PASS`。

因此 alpha=0/1 的 focal length 数值不能单独拿来判断 depth scale 变了约 2.5 倍；disparity 在同一 physical correspondence 上同步变化。详情见 [`alpha_invariance_check.json`](output/20260802_150233_flat_port_refractive_audit/alpha_invariance_check.json)。

### 7.4 ORB-SLAM3

`SLAM/ORB_SLAM3/Examples/Stereo/svo2_stereo.cc:170-196` 取 `LEFT_UNRECTIFIED_BGR/RIGHT_UNRECTIFIED_BGR`，按命令行 `image_scale` resize。`Settings.cc:485-518` 使用源 K/D/T 做内部 rectification，并将 `bf` 更新为 `||T||·P1[0,0]`。`SLAM/prepare_orbslam3_stereo.py:92-155` 负责将 raw K 按 scale 缩放、将 R/T 写成 ORB 的相对位姿。

这条链路的 pose valid ratio 只能说明跟踪/几何重复性；没有绝对位姿真值，不能用现有 ORB 输出证明深度尺度正确。历史 metadata 还显示某次完整 ORB 运行使用了 `image_scale=0.5`，所以当前可编辑 config 文件不能单独代表历史 run 的实际尺度。

## 8. 折射模型审计：为什么不能简单“乘 1.333”

### 8.1 当前代码实际实现的只是两个候选分支

`Z2` 和 `Z4` 是把针孔 depth 统一乘以 `n=1.333`。它们有助于回答“如果强行乘 n，数值会变成什么”：

- native 约 `2.01–2.06 m` 会变成约 `2.67–2.75 m`；
- custom 约 `2.20–2.23 m` 会变成约 `2.93–2.97 m`。

这不是平板舷窗的完整光线模型。平板接口的严格模型还依赖相机中心到接口的距离 `h`、玻璃厚度、玻璃折射率、相机/接口两侧介质以及入射角；一个全局 `n` 只可能是特定几何下的近轴近似或经验缩放。

### 8.2 对称单界面 Snell sanity check（不是 general per-pixel model）

脚本还报告了一个不带 `h` 的角度项诊断：

```text
factor = sqrt(n^2 + (n^2 - 1) * (d/(2f))^2)
Z_symmetric_single_interface_candidate = (fB/d) * factor
```

在 `n=1.333`、本视频有效 disparity 的范围内，该对称单界面 sanity 项相对简单 `n·(fB/d)` 的额外差异为：

- native：中位约 `0.02235%`，P05 约 `0.01694%`，P95 约 `0.1339%`；
- custom：中位约 `0.01685%`，P05 约 `0.00872%`，P95 约 `0.1335%`。

这只说明在当前 disparity/focal 数值下，所写的角度项相对于“纯乘 n”很小；它没有解决未知的 `h`、玻璃和标定介质问题，也不能把 `n·Z` 变成已验证的绝对距离。

### 8.3 现有数据对 H0/H1 的支持程度

- H0：“2.2 m 已经是正确的水下物理深度”：现有数据不足以确认，因为没有独立 GT。
- H1：“2.2 m 必须再乘 1.333”：现有数据也不支持；custom 标定的 2.2 m 是一套自洽的经验结果，且严格折射模型不是一个无条件全局乘法。
- H2：“全局 2.2 m 和全局 2.93 m 都不一定是物理精确值；当前 2.2 m 是 empirical effective-pinhole depth”：在 provenance unknown、flat-port 参数缺失和无 GT 的条件下，当前证据最接近 H2/B2。
- 可确认的事实：“custom pipeline 输出约 2.2 m，且重复性好；native、custom 使用不同的 rectified geometry；两者的 `fB/d` 结果不能直接当作同一个观测量。”

### 8.4 第二轮 per-pixel refractive diagnostic

`refractive_geometry.py` 已实现 air→glass→water 的 vector Snell、plane intersection、左右水中 ray closest-point triangulation 及 `ray_gap`。`debug_flat_port_refractive_model.py` 使用 raw unrectified pixel correspondence；它不把 ordinary rectified SGBM disparity 直接解释成 refractive depth。

当前输出：[`refractive_correspondence_check.json`](output/20260802_150233_flat_port_refractive_audit/refractive_correspondence_check.json)、[`refractive_sensitivity.csv`](output/20260802_150233_flat_port_refractive_audit/refractive_sensitivity.csv) 和 [`REFRACTIVE_MODEL_DIAGNOSTICS.md`](REFRACTIVE_MODEL_DIAGNOSTICS.md)。由于 `n_air/n_glass/n_water/h/glass_thickness/plane_normal` 都没有实测来源，JSON 的 status 是 `not_identifiable_without_port_parameters`，所有 `refractive_depth` 都是 `null`。

### 8.5 Evidence hierarchy

```text
Level 1  Implementation consistency: source views, /16, units, sign, rectification count, P/Q identities
Level 2  Geometric self-consistency: feature disparity, alpha invariance, P/Q, custom/ZED agreement
Level 3  Physical model consistency: vector Snell, flat-port geometry, calibration medium/housing provenance
Level 4  Absolute metric validation: independently measured underwater GT
```

本仓库目前 Level 1/2 证据较强，Level 3 只有 model framework、没有真实 housing 参数，Level 4 缺失。因此不能把 implementation PASS 改写成 absolute accuracy VERIFIED。

## 9. 独立 GT、精度和误差预算

### 9.1 GT 搜索结果

仓库内找到的是：

- 35 对标定图像的重投影误差和相机中心/标定板位姿；
- SVO 的 native camera model；
- custom 标定 K/D/R/T；
- ZED SDK depth、SGBM depth 和 ORB-SLAM3 输出。

没有找到独立的：

- 水下已知距离标靶；
- 标定板/物体到相机或舷窗的实测距离；
- 量尺、激光、声学测距或带尺度的外部跟踪；
- 已知真实尺寸与位姿的水槽记录；
- calibration capture 的水/空气/舷窗状态元数据。

因此目前只能给出内部一致性和重复性，不能给出“绝对误差 ±x cm”或判断 2.2 m 是否包含全部折射误差。

### 9.2 可量化的误差项

| 误差项 | 当前能量化的证据 | 对结论的含义 |
|---|---|---|
| disparity fixed-point | `/16`；一个 LSB 为 `0.0625 px`。在 `d≈69–79 px` 时约 `0.08–0.09%` 的保守一 LSB 相对量级 | 不是当前最大的未知项；实际匹配误差仍需由 GT 评估 |
| SGBM/手工 disparity | native 中位差 `0.0125 px`，custom `0.0625 px` | 支持实现没有明显 scale/sign 错误，不是绝对精度证明 |
| rectification/Q 一致性 | Q 与 `fB/d` 最大相对差 `1.19e-7` | 公式、Q、单位转换内部一致 |
| calibration reprojection | mean `0.225 px`，35 对图像 | 说明自标定重投影拟合较好；不等于水下 metric accuracy |
| baseline 选择 | `120`、`122.4352`、`123.7302 mm` 三个定义相差最多约 `3.11%`（相对 nominal） | 必须固定定义；当前代码已使用 P2/norm 定义，但 nominal 不是 GT |
| focal/有效模型 | custom/native raw focal 比约 `1.35`，rectified `fB` 比约 `1.362` | 可能是有效模型/介质/分辨率/标定条件差异；不能单独解释为 n |
| refractive/port geometry | `h`、玻璃厚度和玻璃 n 缺失 | 这是主要不可识别的系统误差来源之一 |
| absolute scale | 无独立 GT | 绝对误差上界无法从仓库数据闭合 |
| temporal/area repeatability | custom central 40% std 约 `5.8 mm`，alpha=1 100 帧 | 只代表本场景/本匹配条件的重复性，不代表 bias |

这些项不能在缺少 GT 时机械地 RSS 成一个“总误差”；尤其 calibration bias、折射 bias 和 baseline 语义误差是系统项，不是独立零均值噪声。

## 10. 最终判定和验收状态

本报告采用以下四级含义：

- **A**：有独立水下绝对 GT，误差和折射模型已闭环验证；
- **B1**：水下同 housing calibration provenance 已确认，但绝对 GT 缺失；
- **B2**：calibration provenance 本身 unknown，内部几何链路通过，但绝对 GT 缺失；
- **C**：有证据证明当前结果需一个明确、已验证的折射修正；
- **D**：存在已确认的实现错误（混用 disparity、重复 rectification、错误单位/符号等）主导结果。

当前为 **B2**，不是 A/C/D：

| 检查项 | 状态 | 结论 |
|---|---|---|
| raw → calibration → rectification → disparity → depth 可追踪 | PASS | 两个分支的图像源、参数和单位已逐段核对 |
| native/custom disparity 是否混用 | PASS | 各用自己的 rectified disparity 和 `f/B` |
| `/16`、符号、正深度 | PASS | SGBM fixed-point 和正视差均显式检查 |
| cross-branch frame alignment | PASS | `frame_observations` 保存 frame index，ratio 用 inner join；不再用 positional `zip()` |
| 重复 rectification | PASS | native 不再 remap，custom 只 remap 一次 |
| native rectified K/P principal points | PASS | SDK 直接读取的左/右 `fx/fy/cy/cx` 检查通过，当前 offset 为 0 |
| custom P1/P2 principal points | PASS | `CALIB_ZERO_DISPARITY` 下 `cx/cy` offset 均 `<1e-6 px` |
| P/Q 与 `fB/d` | PASS（custom）；PARTIAL（native） | custom max relative `1.19e-7`；native Q 是 diagnostic-generated，非独立 SDK-Q 验证 |
| alpha=0/1 correspondence invariance | PASS | 863 对，relative difference P95 `4.99e-15` |
| per-pixel flat-port geometry | PARTIAL | vector Snell/triangulation framework 已实现；真实 port 参数 unknown，不生成 refractive depth |
| R/T 数值与 baseline 单位 | PARTIAL | 数值/单位已核对；source 语义仍应由独立 GT 闭环 |
| 折射模型是否已确认 | FAIL/OPEN | 缺少 h、玻璃参数、介质记录和独立水下验证 |
| calibration medium/housing provenance | FAIL/OPEN | air/water/housing/target scale 仍 UNKNOWN |
| absolute ground truth | FAIL/OPEN | 仓库未找到 |
| 当前 `2.2 m` 是否应乘 `1.333` | 不支持自动乘 | 不能默认乘；也不能宣称绝对已验证 |
| 是否修改生产行为 | PASS | 只新增独立诊断和本报告，未修改现有生产脚本行为 |

## 11. 建议的闭环实验

要把 B2 提升为 A，最小实验应同时保留原始记录：

1. 在与实际使用完全相同的相机、舷窗、介质和分辨率下，放置带已知尺寸/已知距离的平面靶标，至少覆盖近、中、远三个距离和多个视场位置。
2. 记录相机光心到舷窗的 `h`、玻璃厚度、玻璃折射率、两侧介质和温度；明确标定图像是在空气、浸水还是隔着舷窗采集。
3. 用外部量尺、激光、声学或带尺度的机械导轨给出 target-to-camera 的 independent GT；不能把同一套棋盘标定残差当作 GT。
4. 对同一批 raw frames 同时运行 native、custom、`Z1–Z4`，按 ROI、距离和入射角报告 median/mean/P05/P95、bias、RMSE 和置信区间。
5. 以独立 GT 决定是保留 custom empirical calibration、拟合含舷窗参数的 refractive model，还是修正 R/T/baseline；在此之前不把 `n=1.333` 写进生产公式。

## 12. 复现入口

主要命令：

```powershell
# 四模型、Q/fB、一致性、Snell 候选和 100 帧统计
$env:ZED_SDK_ROOT_DIR='C:\Program Files (x86)\ZED SDK.old'
& 'C:\Users\10179\.conda\envs\zed\python.exe' .\debug_refractive_depth_check.py .\20260802_150233.svo2 --frames 100 --rectify-alpha 1 --output-dir .\output\20260802_150233_refractive_depth_audit --overwrite

# 已有的 native/custom alpha=0 直接比较工具
& 'C:\Users\10179\.conda\envs\zed\python.exe' .\strict_compare_stereo_depth.py .\20260802_150233.svo2 --frames 20

# 独立 custom SGBM；注意 alpha 和 scale 必须随结果一起记录
& 'C:\Users\10179\.conda\envs\zed\python.exe' .\regenerate_sgbm_depth_histogram.py .\20260802_150233.svo2 --frames 1000 --rectify-alpha 1

# raw correspondence + physical flat-port framework + alpha invariance + sensitivity
& 'C:\Users\10179\.conda\envs\zed\python.exe' .\debug_flat_port_refractive_model.py .\20260802_150233.svo2 --frames 20 --output-dir .\output\20260802_150233_flat_port_refractive_audit --overwrite

# vector-Snell, triangulation, P/Q, alpha and frame-ID unit tests
& 'C:\Users\10179\.conda\envs\zed\python.exe' -m unittest -v .\test_refractive_geometry.py
```

本审计新增的诊断脚本只负责读取 SVO/标定并写入诊断 JSON，不替换、导入或静默修改现有生产流程。
