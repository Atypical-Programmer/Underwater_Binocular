# 20260802_150233 水下双目标定与深度结果详细报告

生成日期：2026-09-12

## 1. 报告范围

本报告汇总以下数据和实验：

- 输入 SVO：20260802_150233.svo2
- 相机：ZED 2i，Serial Number 37395692
- ZED SDK：5.4.1
- 分辨率：1920 × 1080
- 记录帧率：30 FPS
- SVO 总帧数：35855
- 外部标定目录：Calibration/
- 处理方法：OpenCV stereoRectify + StereoSGBM
- 最新统计：前 100 帧，Calibration 水下等效模型，rectify alpha=1

本报告的核心目标是区分：

1. SVO 内嵌原始相机参数；
2. Calibration 中的原始水下等效参数；
3. rectify 后的虚拟投影参数；
4. disparity 的像素尺度；
5. 水下折射率是否需要再次显式乘入深度；
6. 当前深度结果的可信程度和剩余疑问。

## 2. 最终结论

### 2.1 关于 Calibration 是否为水下等效标定

现有证据强烈支持这个判断：

~~~text
SVO 原生左 fx       = 1068.000 px
Calibration 左 fx   = 1443.326 px
比例                = 1.3514

SVO 原生右 fx       = 1067.830 px
Calibration 右 fx   = 1449.830 px
比例                = 1.3577
~~~

普通水的折射率约为 1.33。Calibration 的焦距比 SVO 原生焦距大约 35%，与水下平面防水罩产生的有效焦距变化非常接近。

Calibration/标定结果/立体匹配/ 中的结果图显示为水下场景。文件本身虽然没有明确记录水体折射率、玻璃折射率或相机到防水罩距离，但把 Calibration 理解为水下等效成像模型，是目前最合理的解释。

### 2.2 关于是否需要乘 1.33

如果使用 SVO 原生空气模型：

~~~text
Z_air = f_air × B / disparity
Z_water ≈ 1.33 × Z_air
~~~

如果使用已经在水下防水罩条件下拟合出的等效 Calibration：

~~~text
Z_water = f_effective × B / disparity
~~~

不能两者同时使用。当前最终实验采用第二种方法，因此没有额外乘 1.33。

### 2.3 最新 100 帧深度结果

本次使用：

~~~text
原始图像：VIEW.LEFT_UNRECTIFIED / VIEW.RIGHT_UNRECTIFIED
标定：Calibration/标定结果/camera_intrinsics.yaml
      Calibration/标定结果/stereo_extrinsics.yaml
rectify：OpenCV stereoRectify，alpha=1
匹配：OpenCV StereoSGBM
深度：Z=f_rectified×B/disparity
额外 n 修正：没有
~~~

参数：

~~~text
rectified fx = 1423.952587 px
baseline     = 0.123730223 m
~~~

图像几何中心 (960,540)：

~~~text
单像素有效：58/100
中位深度：  2.201465 m
平均深度：  2.204258 m
标准差：    0.036445 m

5×5 中值有效：83/100
5×5 中位深度：2.202325 m
5×5 平均深度：2.205748 m
5×5 标准差：  0.049513 m
~~~

Rectified 光学主点约为 (1213,533)：

~~~text
单像素有效：97/100
中位深度：  2.249781 m
平均深度：  2.246818 m
标准差：    0.021063 m

5×5 中值有效：98/100
5×5 中位深度：2.249781 m
5×5 平均深度：2.246983 m
5×5 标准差：  0.020341 m
~~~

因此当前 Calibration 模型的中心深度约为 2.20～2.25 m。

## 3. 文件和数据说明

### 3.1 SVO

~~~text
文件：20260802_150233.svo2
相机：ZED 2i
序列号：37395692
分辨率：1920×1080
帧率：30 FPS
帧数：35855
~~~

SVO 可提供：

~~~text
VIEW.LEFT / VIEW.RIGHT
    ZED SDK 原生 rectified 图像

VIEW.LEFT_UNRECTIFIED / VIEW.RIGHT_UNRECTIFIED
    原始未校正图像
~~~

### 3.2 Calibration

~~~text
Calibration/标定结果/camera_intrinsics.yaml
Calibration/标定结果/stereo_extrinsics.yaml
Calibration/zed_custom_opencv.yml
~~~

camera_intrinsics.yaml 包含左右相机的原始 K 和 D。stereo_extrinsics.yaml 包含 R/T、baseline、重投影误差和尺度误差。zed_custom_opencv.yml 是对应的 ZED/OpenCV 格式。

原始 Calibration 文件不直接固定 rectified P1/P2。P1/P2 依赖：

- 输入尺寸；
- 输出尺寸；
- R/T；
- 畸变参数；
- rectify alpha；
- CALIB_ZERO_DISPARITY 设置。

因此 P1/P2 是运行时生成的参数。

## 4. SVO 原生参数

### 4.1 原始 K

左相机：

~~~text
fx = 1068.000000
fy = 1067.770020
cx =  957.400024
cy =  539.591003
~~~

右相机：

~~~text
fx = 1067.829956
fy = 1067.709961
cx =  952.570007
cy =  511.776001
~~~

### 4.2 原始畸变 D

SVO SDK 报告的镜头模型是 RAD_TAN，包含 12 个参数。

左相机：

~~~text
[-1.5635299683, 2.9511098862, 0.0000625819, -0.0003552990,
  0.0564838015, -1.4778499603, 2.8006000519, 0.2278030068,
  0, 0, 0, 0]
~~~

右相机：

~~~text
[-1.4983400106, 2.9091899395, -0.0001598470, -0.0007205130,
  0.0318650010, -1.4111700058, 2.7601499557, 0.2018609941,
  0, 0, 0, 0]
~~~

### 4.3 原始 stereo transform

~~~text
[[ 0.9999756813, -0.0056094090,  0.0041440600,  0.1198950037],
 [ 0.0056157154,  0.9999830723, -0.0015116895, -0.0003074820],
 [-0.0041355100,  0.0015349246,  0.9999902844,  0.0004454080],
 [ 0,            0,             0,             1           ]]
~~~

单位设为 METER 时：

~~~text
X ≈ 0.119895 m
Y ≈ -0.000307 m
Z ≈ 0.000445 m
||T|| ≈ 0.119896225 m
~~~

### 4.4 SVO 原生 rectified 参数

~~~text
fx_rectified = 1078.944092 px
fy_rectified = 1078.944092 px
cx_rectified =  955.636475 px
cy_rectified =  524.447815 px
distortion   = 0
model        = PINHOLE
baseline     = 0.119896225 m
~~~

可理解为：

~~~text
P1_native ≈
[[1078.944092, 0, 955.636475, 0],
 [0, 1078.944092, 524.447815, 0],
 [0, 0, 1, 0]]
~~~

SVO 的 VIEW.LEFT 和 VIEW.RIGHT 已经是这个 native rectified 坐标系中的图像。

## 5. Calibration 参数

### 5.1 原始 K

左相机：

~~~text
fx = 1443.326338
fy = 1441.541002
cx =  967.033878
cy =  540.469238
~~~

右相机：

~~~text
fx = 1449.830254
fy = 1448.097563
cx =  975.331674
cy =  514.232843
~~~

### 5.2 和 SVO 原生参数的比较

左相机：

| 参数 | SVO | Calibration | 差值 | 比例 |
|---|---:|---:|---:|---:|
| fx | 1068.000 | 1443.326 | +375.326 | 1.3514 |
| fy | 1067.770 | 1441.541 | +373.771 | 1.3500 |
| cx | 957.400 | 967.034 | +9.634 | 1.0101 |
| cy | 539.591 | 540.469 | +0.878 | 1.0016 |

右相机：

| 参数 | SVO | Calibration | 差值 | 比例 |
|---|---:|---:|---:|---:|
| fx | 1067.830 | 1449.830 | +382.000 | 1.3577 |
| fy | 1067.710 | 1448.098 | +380.388 | 1.3563 |
| cx | 952.570 | 975.332 | +22.762 | 1.0239 |
| cy | 511.776 | 514.233 | +2.457 | 1.0048 |

这个结果说明：

- 分辨率没有变化；
- cx/cy 变化较小；
- fx/fy 变化显著；
- Calibration 的主要差别是有效焦距和畸变模型，而不是图像尺寸。

### 5.3 Calibration D

Calibration 使用 OpenCV 5 参数模型：

~~~text
[k1, k2, p1, p2, k3]
~~~

左：

~~~text
[0.3151847, -0.7495814, 0.002129304, 0.005759798, 6.688496]
~~~

右：

~~~text
[0.3023032, -0.1692313, 0.001685104, 0.010690860, 1.944058]
~~~

与 SVO 原生 D 相比：

~~~text
SVO：12 参数 RAD_TAN
Calibration：5 参数 OpenCV 模型
~~~

因此两个 D 数组不能逐项直接比较。数值差异可能来自：

- 不同参数化；
- 不同畸变模型；
- 水下与空气成像条件不同；
- 外部防水罩影响；
- 标定样本和拟合方式不同。

### 5.4 Calibration R/T

~~~text
R =
[[ 0.999968008,  0.0054886443,  0.00581871018],
 [-0.005489303,  0.999984929,  0.00009717917],
 [-0.005818089, -0.000129117,  0.999983066]]
~~~

~~~text
T = [-122.4352, 0.1676, 17.8539] mm
~~~

文件中的信息：

~~~text
声明 baseline = 122.4352 mm
完整 T 范数   = 123.730223 mm
重投影误差    = 0.225048 px
scale error   = 2.0293%
~~~

本次 rectification 使用完整 T 的范数：

~~~text
B_custom = 0.123730223 m
~~~

与 SVO baseline 相比，约大 3.20%。

## 6. 水下折射理论

### 6.1 空气中的标准模型

~~~text
Z_air = f_air × B / d
~~~

### 6.2 平面防水罩近轴模型

设：

~~~text
n = n_water / n_air ≈ 1.33
~~~

在相机位于空气中、外部为水、忽略光心到界面距离的近轴条件下：

~~~text
Z_water ≈ n × f_air × B / d
~~~

### 6.3 水下等效焦距

定义：

~~~text
f_effective ≈ n × f_air
~~~

则：

~~~text
Z_water ≈ f_effective × B / d
~~~

Calibration 的焦距约为 SVO 原始焦距的 1.35 倍，和这个关系一致。因此 Calibration 可以直接承担 f_effective 的作用。

### 6.4 带 flat-port 距离 h 的模型

如果光心到防水罩界面的距离为 h，更精确的模型为：

~~~text
Z_water =
(fB/d - h)
× sqrt(n² + (n²-1) × (d/(2f))²)
~~~

近轴时：

~~~text
Z_water ≈ n × (fB/d - h)
~~~

当前项目没有可靠的 h、玻璃厚度和玻璃折射率，所以没有使用这个完整模型。

### 6.5 双重计算风险

如果 Calibration 已经是在水下条件下标定的：

~~~text
Calibration depth = f_effective × B / d
~~~

这时再乘 1.33 会变成近似：

~~~text
n × f_effective × B / d
≈ n² × f_air × B / d
~~~

因此会把折射效应重复计算。

## 7. Rectification 和主点

### 7.1 为什么需要 rectify

原始左右图像中，同一个物理点可能为：

~~~text
左图： (u_left, v_left)
右图： (u_right, v_right)
~~~

由于左右相机的旋转、baseline 方向和镜头畸变，可能出现：

~~~text
v_left != v_right
~~~

这样匹配不能只在同一水平行进行。

Rectification 根据：

~~~text
K_left, D_left, K_right, D_right, R, T
~~~

生成：

~~~text
R1, R2, P1, P2, Q
~~~

把左右图像变换到新的虚拟相机坐标系，使：

~~~text
v_left_rectified ≈ v_right_rectified
~~~

从而可以使用：

~~~text
d = u_left_rectified - u_right_rectified
Z = fx_rectified × B / d
~~~

### 7.2 原始主点变化不大

Calibration 与 SVO 的原始主点差异：

~~~text
左 cx：约 9.6 px
右 cx：约 22.8 px
左 cy：约 0.9 px
右 cy：约 2.5 px
~~~

这支持“水体主要改变有效焦距和畸变，原始主点只小幅变化”的判断。

### 7.3 Rectified 主点可以明显变化

Calibration alpha=1 的 rectified P1 为：

~~~text
P1 =
[[1423.952587, 0, 1212.686325, 0],
 [0, 1423.952587, 532.639713, 0],
 [0, 0, 1, 0]]
~~~

这里的 1212.686 不是原始镜头的物理主点，而是 rectified 后虚拟相机的主点。

### 7.4 造成大主点变化的主要原因

Calibration 的 T：

~~~text
Tx = -122.4352 mm
Ty =    0.1676 mm
Tz =   17.8539 mm
~~~

基线相对于 X 轴的倾角约为：

~~~text
atan(17.8539 / 122.4352) ≈ 8.30°
~~~

Rectification 需要把带有 Z 分量的 baseline 旋转为水平。旋转虚拟相机后，虚拟光轴在输出图像中的交点会发生移动。

偏移量数量级可以估计为：

~~~text
fx × tan(8.30°)
≈ 1424 × 0.146
≈ 208 px
~~~

因此 rectified cx 移动到约 1213 是可能的。这不是水体直接把原始物理主点移动了 200 多像素。

### 7.5 图像中心不等于 rectified 光学中心

当前 Calibration alpha=1：

~~~text
图像几何中心       = (960, 540)
rectified 光学主点 = (1213, 533)
~~~

因此本报告同时统计了两个位置：

- 图像几何中心：用户通常所说的图像中心；
- rectified 主点：虚拟 rectified 相机的光学轴中心。

## 8. disparity 尺度分析

### 8.1 SVO 原生 fB

~~~text
f_native = 1078.944092 px
B_native = 0.119896225 m

fB_native ≈ 129.361324 px·m
~~~

### 8.2 Calibration alpha=0 fB

~~~text
f_alpha0 = 3635.497172 px
B_custom = 0.123730223 m

fB_alpha0 ≈ 449.820875 px·m
~~~

比例：

~~~text
449.820875 / 129.361324 ≈ 3.477
~~~

因此 alpha=0 时 disparity 变为约 3.5 倍，是其 rectified 虚拟投影尺度造成的。

### 8.3 Calibration alpha=1 fB

~~~text
f_alpha1 = 1423.952587 px
B_custom = 0.123730223 m

fB_alpha1 ≈ 176.185971 px·m
~~~

与 SVO 原生相比：

~~~text
176.185971 / 129.361324 ≈ 1.362
~~~

这接近原始焦距约 35% 的变化。

### 8.4 为什么图像尺寸相同但 disparity 不同

图像尺寸只描述数组大小：

~~~text
1920 × 1080
~~~

它不保证：

- 每像素对应的角度相同；
- cx/cy 相同；
- fx/fy 相同；
- 畸变相同；
- rectification 旋转相同；
- 输出图像使用相同的 P1/P2。

Disparity 是像素坐标差，因此会随虚拟投影坐标系改变。

## 9. 实验分支与结果

### 9.1 SVO 原生 20 帧

流程：

~~~text
新建 SVO handle
不加载外部标定
读取 VIEW.LEFT / VIEW.RIGHT
使用 SVO 原生 rectified 图像
SGBM
使用 SVO 原生 rectified fx 和 baseline
~~~

结果：

| 项目 | 数值 |
|---|---:|
| 中心 disparity 有效 | 13/20 |
| 中心 disparity 中位数 | 63.3125 px |
| 中心深度中位数 | 2.0432 m |
| 中心 5×5 中位数 | 2.0503 m |
| 中心 5×5 有效 | 18/20 |

### 9.2 Calibration alpha=0 20 帧

~~~text
rectified fx = 3635.497172 px
baseline     = 0.123730223 m
~~~

结果：

| 项目 | 数值 |
|---|---:|
| 中心 disparity 有效 | 3/20 |
| 中心 disparity 中位数 | 207.125 px |
| 中心深度中位数 | 2.1717 m |
| 中心 5×5 中位数 | 2.1869 m |
| 中心 5×5 有效 | 9/20 |

该分支主要用于观察 alpha=0 的放大效果，不作为最终稳定结果。

### 9.3 Calibration alpha=1 20 帧

~~~text
rectified fx = 1423.952587 px
baseline     = 0.123730223 m
~~~

结果：

| 项目 | 数值 |
|---|---:|
| 中心 disparity 有效 | 5/20 |
| 中心 disparity 中位数 | 80.0000 px |
| 中心深度中位数 | 2.2023 m |
| 中心 5×5 中位数 | 2.1929 m |
| 中心 5×5 有效 | 11/20 |

### 9.4 Calibration alpha=1 100 帧

这是当前报告推荐的短序列结果。

图像中心：

~~~text
单像素 valid = 58/100
单像素 median = 2.201465 m
单像素 mean   = 2.204258 m
单像素 std    = 0.036445 m
单像素 p05    = 2.154195 m
单像素 p95    = 2.264033 m
单像素 min    = 2.122723 m
单像素 max    = 2.308743 m

5×5 valid = 83/100
5×5 median = 2.202325 m
5×5 mean   = 2.205748 m
5×5 std    = 0.049513 m
5×5 p05    = 2.156829 m
5×5 p95    = 2.284051 m
~~~

Rectified 主点：

~~~text
单像素 valid = 97/100
单像素 median = 2.249781 m
单像素 mean   = 2.246818 m
单像素 std    = 0.021063 m

5×5 valid = 98/100
5×5 median = 2.249781 m
5×5 mean   = 2.246983 m
5×5 std    = 0.020341 m
~~~

### 9.5 ZED SDK custom calibration + NEURAL 全序列

曾经使用 ZED SDK 的 NEURAL 模式、custom OpenCV calibration 处理完整 35855 帧。这不是独立手工 SGBM，因此只作为交叉检查。

结果：

~~~text
中心单像素 valid = 35706/35855
中心 median = 2.2041 m
中心 mean   = 2.2084 m

中心 5×5 valid = 35723/35855
中心 median = 2.2041 m
中心 mean   = 2.2084 m
~~~

它和最新手工 SGBM 的 2.20～2.25 m 结果一致，说明 2.2 m 不是某一个 SGBM 帧的偶然结果。

## 10. 当前结果的可信度

### 10.1 有效匹配率

100 帧 alpha=1：

~~~text
图像中心单像素：58%
图像中心 5×5：83%
rectified 主点单像素：97%
rectified 主点 5×5：98%
~~~

因此更推荐使用 rectified 主点附近 5×5 中值，而不是单个图像中心像素。

### 10.2 结果不是对真实距离的绝对证明

当前结果只能说明：

- 参数读取正常；
- rectification 过程明确；
- disparity 和 P1/B 的搭配一致；
- Calibration 模型输出稳定在约 2.2～2.25 m。

它还不能单独证明真实物理距离就是 2.2 m，也不能证明真实距离一定不是 3 m。

### 10.3 可能造成 2.2 m 与预期 3 m 差异的因素

1. Calibration 与 SVO 场景实际成像条件并不完全一致；
2. 外部标定的水下等效模型仍然是近似针孔模型；
3. flat-port 距离 h、玻璃厚度和玻璃折射率没有建模；
4. Calibration 的 5 参数畸变模型不足；
5. SGBM 在水下纹理中发生匹配错误；
6. 中心区域可能没有对应到用户认为的目标表面；
7. Z 深度和相机到目标的欧氏距离定义不同；
8. Calibration 的 T 有明显 Z 分量，rectification 后的虚拟视线和原图中心不一致；
9. 两套标定采用的畸变模型参数化不同。

## 11. 推荐工程方案

### 11.1 推荐方案：水下等效 Calibration

~~~text
1. 读取原始未校正左右图像
2. 读取 Calibration K/D/R/T
3. 固定 image_size、newImageSize 和 alpha
4. 生成并保存 R1/R2/P1/P2/Q
5. remap 左右图像
6. 对 rectified 图像运行 SGBM
7. 使用 P1[0,0] 和对应 baseline
8. 计算 Z=P1[0,0]×B/disparity
9. 不再乘 1.33
~~~

当前采用的 alpha=1 比 alpha=0 更适合观察原始水下等效焦距，因为 alpha=0 会显著放大虚拟投影。

### 11.2 只有使用空气模型时才显式乘 n

如果确认使用的是 SVO 空气模型：

~~~text
Z_air = f_air × B / d
Z_water ≈ 1.33 × Z_air
~~~

宽视场和近距离目标应使用逐像素 Snell 修正，而不是简单地把整张深度图乘一个常数。

### 11.3 不建议的混合方式

~~~text
Calibration 水下等效 K
+ Calibration disparity
+ 再乘 1.33
~~~

或者：

~~~text
SVO native rectified disparity
+ Calibration raw fx
~~~

或者：

~~~text
alpha=0 disparity
+ alpha=1 的 rectified fx
~~~

这些方式都会混合不同的像素坐标系或重复计算折射。

## 12. 最终验证实验

要判定实际目标到底是 2.2 m 还是约 3 m，建议放置一个已知距离的水下目标或标定板，至少测试：

~~~text
1.5 m
2.0 m
2.5 m
3.0 m
~~~

每个距离都比较：

~~~text
SVO 原生空气模型
SVO 原生结果乘 1.33
Calibration 水下等效模型
包含 h 的 flat-port 模型
~~~

同时记录：

- 距离是从防水罩界面测量，还是从相机光心测量；
- 左右角点的重投影误差；
- rectify 后的垂直极线残差；
- 中心和边缘区域的深度误差；
- disparity 有效比例；
- 目标表面是否平整且具有足够纹理。

只有通过已知距离目标，才能决定应该采用：

~~~text
Calibration 直接深度
或
SVO 空气深度再乘水下折射修正
~~~

## 13. 输出文件

最新 100 帧结果：

- output/20260802_150233_underwater_effective_depth_100/summary.json
- output/20260802_150233_underwater_effective_depth_100/center_depth.csv
- output/20260802_150233_underwater_effective_depth_100/center_depth_histogram.png
- output/20260802_150233_underwater_effective_depth_100/center_window_depth_histogram.png
- output/20260802_150233_underwater_effective_depth_100/rectification.json
- output/20260802_150233_underwater_effective_depth_100/depth_maps/depth_000000.npy 至 depth_000099.npy

参数对比：

- output/20260802_150233_calibration_comparison.json

20 帧严格隔离对比：

- output/20260802_150233_strict_calibration_compare_20/

相关脚本：

- regenerate_sgbm_depth_histogram.py
- strict_compare_stereo_depth.py
- compare_svo_and_calibration.py

## 14. 一句话总结

最合理的当前解释是：SVO 内嵌参数接近空气/工厂模型，Calibration 参数很可能已经吸收了水下防水罩造成的有效焦距变化；因此使用 Calibration 重新 rectify 后，应直接用对应的 rectified P1/P2 计算深度，不再额外乘 1.33。当前 100 帧得到的图像中心深度约 2.20 m，rectified 光学主点附近约 2.25 m。若实际应为约 3 m，需要通过已知水下距离目标、flat-port 参数和极线/重投影误差进一步验证，而不能仅靠再次乘折射率得出。
