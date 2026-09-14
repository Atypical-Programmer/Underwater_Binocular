# Refractive Model Diagnostics

> This second-round diagnostic is preserved for reproducibility. The final frozen-model conclusion and rig-level audit are in [`FINAL_UNDERWATER_DEPTH_VERDICT.md`](FINAL_UNDERWATER_DEPTH_VERDICT.md) and [`REFRACTIVE_RIG_MODEL_AUDIT.md`](REFRACTIVE_RIG_MODEL_AUDIT.md).

本文件说明第二轮新增的物理几何诊断。它是独立的、只读的 audit tool，不修改 ZED、SGBM 或 ORB-SLAM3 production pipeline。

## 当前结论

本次运行使用 20 个分散 SVO 帧，包含 frame 0 和另外 19 个分散帧：

```text
[0, 1887, 3774, 5661, 7548, 9435, 11322, 13209, 15096, 16983, 18871, 20758, 22645, 24532, 26419, 28306, 30193, 32080, 33967, 35854]
```

20 个帧全部得到高质量匹配，共 2378 个通过 ORB ratio test、rectified RANSAC 和垂直极线误差筛选的 raw correspondences。输出结果为：

```text
refractive_correspondence_check.json: status = not_identifiable_without_port_parameters
alpha_invariance_check.json:          status = PASS, 863 high-quality correspondences
refractive_sensitivity.csv:            1296 rows, LEGACY_HYPOTHETICAL_CUSTOM_PLUS_SNELL
```

没有把任何默认的 `h`、玻璃厚度或玻璃折射率写入物理重建，因此所有 correspondence 的 `refractive_depth` 保持 `null`。sensitivity CSV 只是参数敏感性分析，不能当作真实水下深度。

## 坐标系与变换约定

每个相机 optical frame 使用：

```text
x: right
y: down
z: forward along optical axis
```

`refractive_geometry.py` 的 stereo common frame 是左相机 frame。它存储：

```text
X_left = R_right_to_left · X_right + T_right_to_left
```

custom YAML 的 R/T 按 OpenCV 立体标定输入解释为：

```text
X_right = R_left_to_right · X_left + T_left_to_right
```

诊断中明确取其 inverse 作为 refractive common-frame transform；没有用正负号猜测或静默 fallback。ZED raw SDK transform 则按 SDK 读取结果记录为 right-camera-to-left-camera，并保留原始矩阵供复核。

## Per-pixel flat-port ray model

物理模块支持三介质平行接口：

```text
air -> glass -> water
```

`FlatPortModel` 的参数为：

```text
n_air
n_glass
n_water
camera_to_inner_interface_m
glass_thickness_m
plane_normal_camera
```

`plane_normal_camera` 默认 `[0, 0, 1]` 只是坐标约定，不代表真实 housing 法向量。

给定 raw distorted pixel `(u,v)` 后，模块按以下步骤工作：

1. 用该相机的 K/D 调用 inverse distortion，恢复 air-side normalized ray；rectified pixel 不会被直接伪装成 physical optical ray。
2. 求 air ray 与 inner glass plane 的交点 `P_inner`。
3. 用 vector Snell 得到 air→glass direction。
4. 求 glass ray 与 outer glass/water plane 的交点 `P_outer`。
5. 用 vector Snell 得到 glass→water direction。
6. 输出水中 ray 的真实 origin `P_outer` 和 direction；水中 ray 一般不再通过原 camera center。
7. 左右两条水中 ray 在左相机 frame 中求 closest-point least-squares triangulation。

三种距离/深度分别输出：

```text
z_left_camera_m
euclidean_distance_from_left_camera_m
distance_from_left_outer_port_m
```

另外输出 `ray_gap_m = ||X_L-X_R||`、左右 ray origin/direction 和 left-ray view angle。skew rays 不会被强行当作精确相交。

## Vector Snell 实现范围

`refract_vector()` 的 normal 必须明确指向 medium 1→medium 2，incident direction 必须朝向 interface；它会：

- normalize incident、normal 和 transmitted vectors；
- 检查 `cos(theta)` 的符号，不自动翻转 normal；
- 对 total internal reflection 抛出明确异常；
- 支持任意正的 `n1/n2`。

当前旧诊断中的 disparity-only 公式已经改名为
`symmetric_single_interface_snell_sanity_check`。它只是假设左右 ray 对称、目标接近 stereo symmetric plane，并从 `d/(2f)` 推一个单侧角度的 sanity approximation；它不是 general per-pixel refractive model，也不是 flat-port ray tracing。

## 参数 unknown 的硬规则

若任意 housing 参数缺失，模型状态为 `unknown`，并抛出 `RefractiveModelNotIdentifiable`，不能生成 definitive refractive depth。

命令行允许传参，但必须声明 provenance：

```powershell
# 只有在所有值都有真实测量记录后，才允许声明 measured。
& python .\debug_flat_port_refractive_model.py .\20260802_150233.svo2 `
  --n-air 1.000 `
  --n-glass 1.500 `
  --n-water 1.333 `
  --port-distance 0.010 `
  --glass-thickness 0.005 `
  --plane-normal 0 0 1 `
  --port-parameters-provenance measured
```

如果只是在做假设实验，应使用 `assumed`；脚本仍不会把结果标记为真实 physical reconstruction。当前仓库没有测量值，因此默认运行没有提供这些参数。

## Correspondence diagnostic

`debug_flat_port_refractive_model.py` 对 frame 0 和另外至少 9 个分散帧执行（当前 run 为 frame 0 + 19 个分散帧）：

1. raw `VIEW.LEFT_UNRECTIFIED/RIGHT_UNRECTIFIED`；
2. ORB descriptor matching + 0.75 ratio test；
3. 用 custom alpha=1 virtual rectified coordinates 做 Fundamental RANSAC；
4. 过滤 `|v_L-v_R| <= 2 px`、正 disparity 和 finite coordinates；
5. 对每个 raw pixel pair 分别计算 native pinhole、custom pinhole；
6. 只有 measured flat-port 参数完整时才计算 refractive triangulation。

这里的 native/custom depth 是两套 raw K/D/R/T 的 pinhole triangulation，属于 implementation/geometric comparison，不是 independent GT。custom SGBM 和 ZED custom depth 也共享相同 custom calibration/底层图像，二者一致只能说明实现自洽，不能说明物理 bias 为零。

## Alpha invariance

同一 raw correspondence 被分别映射到 custom alpha=0 和 alpha=1 的 virtual rectified coordinates，然后分别使用各自的 `f_rect`、`B_rect` 和 disparity：

```text
Z0 = f0 · B0 / d0
Z1 = f1 · B1 / d1
```

实际结果：

- alpha=0 `f=3635.497171961014 px`；
- alpha=1 `f=1423.952587176348 px`；
- 两者 `B=0.123730222799 m`；
- 高质量 correspondence：863；
- `|Z0-Z1|/Z1` P95：约 `4.92e-15`；最大约 `8.48e-15`。

因此 alpha 改变 focal length 不能单独解释成 depth scale 改变；同一 physical raw correspondence 的 metric `fB/d` 保持不变。两种 alpha 的 ROI/FOV 仍不同，不能混用像素坐标或 disparity。

## Sensitivity CSV

`refractive_sensitivity.csv` 使用下列明确标记为 hypothetical 的网格：

```text
n_water: 1.330, 1.333, 1.340
n_glass: 1.47, 1.50, 1.52
h:       5, 10, 20, 30 mm
thick:   3, 5, 8, 10 mm
```

按 central/mid-FOV/near-edge 与 far/mid/near-disparity 分组。它用于回答“未知 housing 参数可以造成多大空间/尺度变化”，不用于选择真实参数，也不用于替代 GT。

## Unit tests

```text
test_snell_normal_incidence
test_snell_known_angle
test_snell_refractive_index_identity
test_snell_rejects_total_internal_reflection
test_ray_plane_intersection
test_two_ray_triangulation_exact_intersection
test_two_ray_triangulation_skew
test_unknown_flat_port_never_produces_definitive_ray
test_custom_p_q_identity
test_alpha_depth_invariance
test_frame_alignment
```

运行：

```powershell
& 'C:\Users\10179\.conda\envs\zed\python.exe' -m unittest -v .\test_refractive_geometry.py
```

## Calibration provenance and official baseline context

本轮再次搜索 README、标定 YAML/YML、XLSX、PDF、H5 文件名/元数据可见内容和 Git 历史，仍没有找到 custom calibration 的介质、housing、flat-port/dome、玻璃参数或 target square size 记录；状态保持 `UNKNOWN`。

官方 ZED 2i 资料把产品 nominal baseline 写为约 `120 mm`（[ZED 2i datasheet](https://support.stereolabs.com/hc/en-us/article_attachments/27901419901463)）。这只能说明仓库中的 `120 mm` nominal 注释与产品级规格量级一致，不能验证本台相机、当前 housing 或 custom 标定的绝对尺度。
