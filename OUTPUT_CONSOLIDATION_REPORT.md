# 当前输出与运行状态汇总

数据集：`20260802_150233`
统计时间（UTC）：2026-09-14T20:24:47.8472940Z

这份报告是当前输出树的唯一入口。运行产物本身保持本地忽略；可提交到 Git 的是本报告、清单、摘要和验证锚点。历史报告已移到 [`docs/archive/project_migration/2026-09/`](docs/archive/project_migration/2026-09/)，旧版 `output/` 已完成迁移、清理并删除。

## 一眼结论

- 旧 `output/`：**不存在**。原始 SVO、校准源文件和必要的历史摘要没有被删除。
- 当前正式结果在 `outputs/20260802_150233/`；其中 `PRIMARY`、`BASELINE`、`REFERENCE`、`ANOMALOUS`、`STALE` 的命名用于区分科学角色，不把异常结果伪装成主结果。
- ZED 原生校准是追踪的推荐默认模式；custom 校准仍保留为显式对照模式。
- ALIKED + AdaLAM + 外部 COLMAP 的 20 对/40 图像兼容性审计为 **PASS**。首次审计暴露的 `FULL_OPENCV` 参数数量错误已修复为 COLMAP 所需的 12 参数。
- custom ZED SDK 深度的中心估计约为 2.204 m，是可复现的运行产品，不是独立的水下物理真值；不自动乘以 `1.333`。
- 折射参数 `R1` 仍为 `NOT_IDENTIFIABLE`，因为缺少端口/壳体几何、介质折射率、界面平面和独立 metric ground truth。
- 近似平面场景的正式 SfM 已切换到显式标定双目路径：`2000/2000` 图像注册、`80,072` 个 3D 点；平面几何不再被当作失败条件。

## 容量与清理结果

| 范围 | 文件数 | 字节数 | 说明 |
|---|---:|---:|---|
| 清理前旧 `output/` | 8,833 | 32,562,536,067 | 只读基线，约 30.33 GiB |
| 明确删除的文件/文件集 | 6,572 | 25,486,542,015 | 不含目录重复计数；六个空目录也已删除 |
| 当前正式 `outputs/20260802_150233/` | 4,227 | 8,917,414,876 | 当前结果与兼容性审计 |
| 本地历史归档 `outputs/_archive/legacy-2026-09-14/` | 250 | 556,075,999 | 保留摘要、轨迹、元数据和必要对照；被 `.gitignore` 忽略 |
| 当前 `outputs/` 合计 | 4,477 | 9,473,490,875 | 正式结果 + 本地历史归档 |
| 可复用图像 `cache/20260802_150233/` | 2,000 | 5,207,272,369 | sample1000 左右图像缓存；可重建、不可提交 |
| `outputs/` + `cache/` 本地合计 | 6,477 | 14,680,763,244 | 仍在本机的运行相关数据 |

删除字节是对 [`OUTPUT_DELETE_MANIFEST.md`](OUTPUT_DELETE_MANIFEST.md) 中各文件/文件集的求和，不包含父目录递归大小，因此没有重复计算。归档不是“释放空间”，而是把仍有证据价值的历史结果从旧目录重新组织；可删除性由清单和 README 约束。

清理前的完整只读基线在 [`docs/archive/project_migration/2026-09/OUTPUT_CONSOLIDATION_BASELINE.md`](docs/archive/project_migration/2026-09/OUTPUT_CONSOLIDATION_BASELINE.md)。归档路径、原始大小和科学用途在 [`OUTPUT_ARCHIVE_MANIFEST.md`](OUTPUT_ARCHIVE_MANIFEST.md)；sample1000 的拆分在 [`SAMPLE1000_CLEANUP_MANIFEST.md`](SAMPLE1000_CLEANUP_MANIFEST.md)。

## 当前正式运行产物

正式树：`outputs/20260802_150233/`。每个正式运行目录包含 `run.json`，并使用统一的 `status`、`result_status`、`purpose`、`retain_policy`、`dataset` 字段。

| 运行目录 | 当前状态 | 关键结果 | 科学角色 |
|---|---|---|---|
| `depth/depth__custom__zed_sdk__full35855__reference` | `complete` | ZED SDK 5.4.1；35,855/35,855 帧；中心有效率 99.5844%；中心中位数 2.20412 m；5×5 窗口中位数 2.20408 m | custom SDK 深度的运行参考；不是独立真值 |
| `depth/depth__custom__sgbm_halfres__1000f__reference` | `complete` | 独立 StereoSGBM；1,000 帧；中心中位数 2.25210 m、有效率 82.8%；窗口中位数 2.25299 m、有效率 94.2% | 诊断/交叉参考 |
| `tracking/tracking__native__gen1_neural__full35855__baseline` | `complete` | native；GEN_1；35,855/35,855 帧为 `OK`；路径 232.3056 m；终点位移 85.9515 m；step p95 0.02135 m | 推荐的 native 追踪基线 |
| `pointcloud/pointcloud__custom__zed_sdk__1000mapped__reference` | `complete` | 75,885,506 点；PLY 1,138,282,772 bytes；1,000 个 mapping 帧；校准/PLY/映射计数校验通过 | custom SDK 点云参考；保留原始运行中的轨迹质量警示 |
| `slam/slam__orbslam3__diagnostic_full__35855f__baseline` | `complete` | 35,855 帧；35,855 有效位姿；到 SVO 末尾；已保存地图点 | ORB-SLAM3 全量诊断基线 |
| `slam/slam__orbslam3__stereo_halfres__35855f__baseline` | `complete` | 960×540；35,855 帧；32,935 有效位姿（91.8561%） | ORB-SLAM3 半分辨率对照 |
| `sfm/sfm__custom__aliked_adalam__1000f__PRIMARY` | `completed` | 1,000 源帧、2,000 图；1,600,000 keypoints；14,948 候选/接受对；标定双目平面模型 2,000/2,000 registered images；80,072 points3D；平均重投影误差 2.130112 px | 视觉时序运动对照；尺度来自 canonical stereo baseline，绝对水下物理精度仍需独立验证 |
| `sfm/sfm__custom__aliked_lightglue__1000f__PRIMARY` | `completed` | 1,000 源帧、2,000 图；1,600,000 keypoints；14,948 候选/接受对；标定双目平面模型 2,000/2,000 registered images；54,507 points3D；平均重投影误差 2.241593 px | LightGlue matcher 对照；控制变量与 AdaLAM PRIMARY 相同；详见 `OUTPUT_MATCHER_COMPARISON.md` |
| `sfm/sfm__custom__aliked_adalam__2000f__PRIMARY` | `complete` | 2,000 图；1,212,518 keypoints（历史摘要）；12,950 候选对；9,567 接受对；1,448 registered images；136,707 points3D | 历史普通增量 SfM 对照，任意局部尺度 |
| `sfm/sfm__custom__aliked_adalam__100f__reference` | `complete` | 100 图；99 接受/验证对；100 registered images；12,887 points3D | 次级复现参考 |
| `sfm/sfm__custom__aliked_adalam_colmap__20f__diagnostic` | `completed` | 40 图；32,000 keypoints；128 对；128 verified geometries；2 registered images；575 points3D；模型已转 TXT | 外部 COLMAP 兼容性审计，非生产重建 |

### Metashape 导入包

上述 5 个 SFM project 均已生成 `export/metashape/`，其中包含
`metashape_cameras.xml`、YPR/OPK 参考 CSV、`points3D.ply`、图像清单和
原始 COLMAP 文本模型副本。XML 均由本机 Metashape Professional 1.7.4
生成；2 个小 project（100f reference、20f diagnostic）通过了 headless
round-trip，相机数较大的 3 个 project 因 Metashape 1.7.4 的 headless
回读限制只进行 XML 生成检查，manifest 中已明确记录。

> 注：历史普通增量主/次级 SfM 的 local scale 未经过外部 metric 约束；当前 `1000f__PRIMARY` 使用 canonical stereo baseline，因此仅对该模型记录“相对标定的 metric scale”。

> `sfm/sfm__custom__aliked_adalam__1000f__H5_CONSTRAINED` 已按用户要求移入 Windows 回收站，因重建结果无效而不再作为当前输出保留。`PRIMARY`、H5 原始文件以及 H5 约束代码仍保留。

正式结果的摘要和轨迹文件仍保留历史 lineage 路径；这些路径明确指向旧 `output/` 来源，便于审计，不代表旧目录仍存在。

## 外部 COLMAP 兼容性审计

审计目录：`outputs/20260802_150233/sfm/sfm__custom__aliked_adalam_colmap__20f__diagnostic/`
跟踪参考：[`validation/reference/20260802_150233/colmap_compatibility_reference.json`](validation/reference/20260802_150233/colmap_compatibility_reference.json)

| 项目 | 结果 |
|---|---|
| 外部 COLMAP | 3.11.1，CUDA build，commit `682ea9a` |
| 输入 | 20 个同步左右图像对，共 40 张；来自本地 cache，不重新打开 SVO |
| ALIKED | `lightglue.ALIKED`，`aliked-n16`，800 keypoints/image，32,000 total |
| AdaLAM | 128 candidate / 128 accepted pairs；128 geometries 通过 ≥15 行阈值 |
| `matches_importer` | PASS |
| `mapper` | PASS；2 registered images，575 points3D |
| `model_converter` | PASS；成功写出文本模型 |
| 修复 | `FULL_OPENCV` 相机参数从错误的 9 项补齐为 COLMAP 要求的 12 项 |

首次运行确实失败于 COLMAP 的 `num_params == 12` 检查；修复后重跑通过。该审计只证明接口/数据库/外部程序边界可用，不代表 20 对结果具有生产级覆盖率或 metric scale。

## 归档内容与保留策略

- `PRIMARY`：主结果，例如 2,000 图 SfM。
- `BASELINE`：可复现的全量或标准参考，例如 native GEN_1、ORB-SLAM3 全量/半分辨率。
- `REFERENCE`：用于数值交叉检查的 depth、SGBM、100 图 SfM 和 custom SDK 点云。
- `ANOMALOUS`：已识别为异常的 custom GEN_1 点云/追踪对照，只保留摘要、轨迹、校准和元数据，不让它成为默认结果。
- `STALE` / `INCONSISTENT`：例如 ORB-SLAM3 `stable_map`，声明帧数与实际日志不一致，只保留审计证据。
- `cache/`：仅保留 sample1000 的 1,000 左图 + 1,000 右图作为可重建输入缓存；`.gitignore` 已明确忽略整个目录。

历史输出已按类别归档，并为入口和主要项目保留 README：本地 `outputs/_archive/legacy-2026-09-14/README.md` 以及已提交的 [`docs/archive/legacy_outputs/`](docs/archive/legacy_outputs/)。归档中的大 PLY、area、数据库和 dense cache 已按删除清单处理；删除前已抽取摘要、元数据或 SHA256。

## 校准、深度与折射解释

- tracking API 现在有明确的 `native` / `custom` 边界，默认 `native`。native 不设置 `optional_opencv_calibration_file`，使用 SVO 嵌入校准；custom 才加载 canonical profile 并执行运行时校验。
- native/custom 原始内参确实是不同模型：custom 左右 `fx` 约为 native 的 1.3514× / 1.3577×，baseline norm 差约 0.003834 m。这个差异不能直接当成折射尺度，也不能把两个坐标变换未经约定对齐后相减。
- custom SDK 深度 2.204 m 与独立 SGBM 2.252 m 的差异只构成交叉参考；没有独立 metric ground truth 时不宣称任何一个是真值。
- `R1`：`NOT_IDENTIFIABLE`。历史 `custom + Snell` 网格只作假设敏感性诊断，不能被解释为物理深度估计。

紧凑验证锚点集中在 [`validation/reference/20260802_150233/`](validation/reference/20260802_150233/)，包括 calibration、depth、SGBM、tracking、SfM、ORB-SLAM3、COLMAP compatibility 和 refractive audit。

## 代码与安全边界改动

- 新增 `underwater outputs inventory`：输出 JSON/CSV/Markdown 清单，记录大小、文件数、数据集、类别、状态、用途、保留策略、摘要/run 元数据、二进制提示和建议动作。
- 新增 `underwater outputs prune --dry-run`：默认不删除；`--apply` 必须显式指定 `--category empty` 或精确 JSON manifest，并拒绝递归删除非空目录。
- stereo matching 将同步左右帧配对从无界的左×右扫描改为排序帧窗口查找；专用 1000 帧脚本默认 `StereoWindow=40`，保留 pair policy 和回归测试。
- 新增显式 `calibrated_stereo_planar` 映射：用 canonical 左右外参三角化同步双目，再用相邻帧 3-D 刚体运动写出完整 COLMAP 模型；不会调用普通 incremental `mapper`。
- 修复 Windows COLMAP `model_converter` 的目标目录创建问题；新的模型位于 `colmap/sparse/calibrated_stereo_planar/`。
- `.gitignore` 现在忽略 `/output/`、`/outputs/`、`/cache/`、大型二进制和 inventory 生成物；不会把本地运行数据上传到 GitHub。
- 原始 SVO `20260802_150233.svo2`、canonical calibration/profile、`calibration/source/` 和现有 `third_party/ORB_SLAM3` checkout 均未删除；后者的既有嵌套工作区改动也未触碰。

## 本次失败结果清理

已将当前主运行目录中明确失败、且已被新模型替代的普通 mapper 产物移入
Windows 回收站（可恢复）：`colmap/sparse/0/`、`colmap/sparse/0_text/`、
对应的 `mapper.log`、`model_converter.log`，以及第一次失败的
`integration_calibrated_stereo_run.log`。成功的
`calibrated_stereo_planar/`、`calibrated_stereo_planar_text/`、匹配数据库、
特征缓存和历史归档均保留。

## 最终 24 项审计

| # | 审计项 | 状态与证据 |
|---:|---|---|
| 1 | tracking native/custom 分离 | PASS；默认 native，custom 显式 opt-in；`test_zed_boundary.py` |
| 2 | stereo pair 性能 | PASS；排序窗口查找；1000 帧脚本默认 `StereoWindow=40`；`test_reconstruction_workflow.py` |
| 3 | COLMAP 兼容 | PASS；外部 3.11.1 importer/converter/analyzer 全链路；标定双目模型 2,000/2,000 |
| 4 | 新输出布局 | PASS；`outputs/<dataset>/<category>/<run-id>`；旧 `output/` 不存在 |
| 5 | 只读基线 | PASS；8,833 files / 32,562,536,067 bytes 已归档 |
| 6 | inventory/prune | PASS；清单命令可运行，prune 默认 dry-run 且无 selector 不允许 apply |
| 7 | 归档命名与 README | PASS；按类别/算法/范围/状态命名并保留入口 README |
| 8 | archive/delete manifests | PASS；迁移与删除路径、用途、摘要、hash/状态已记录 |
| 9 | 空目录/视频/预览清理 | PASS；视频 sidecar 保留，冗余视频/preview 删除，旧根删除 |
| 10 | depth/SGBM | PASS；正式 reference + tracked quantile refs |
| 11 | calibration comparison | PASS；native/custom compact ref，等 hash 的重复 JSON 已去重 |
| 12 | tracking 结果 | PASS；native baseline/performance、custom anomaly/smoke 均有角色标记 |
| 13 | pointcloud 结果 | PASS；authoritative/custom、low-provenance、anomalous 分类与 PLY hash 已记录 |
| 14 | ORB-SLAM3 | PASS；full/stereo baseline；stale map 标为 inconsistent；sampled PLY 删除 |
| 15 | sample1000 | PASS；2,000 图进入 cache；2,000 图 PRIMARY、100 图 reference；弱/不完整/ablation 归档或删除 |
| 16 | 折射诊断 | PASS；R1 明确为 `NOT_IDENTIFIABLE`，不伪造结论 |
| 17 | cache 忽略策略 | PASS；`/cache/` 在 `.gitignore` |
| 18 | validation refs | PASS；12 个紧凑 reference 文件/README 已跟踪 |
| 19 | 统一 run metadata | PASS；正式 run.json 有 status/result/purpose/retain/dataset 等字段 |
| 20 | 自动检查 | PASS；全量 pytest 40 项、ruff、compileall 均通过 |
| 21 | 当前 ZED 新 replay | `NOT EXECUTED`；当前 conda 环境无 `pyzed.sl`；历史 ZED 结果只作为已标注 baseline/reference |
| 22 | 原始输入安全 | PASS；SVO、校准源和第三方 checkout 未删除 |
| 23 | 技术债 | 已记录：vendor SDK 依赖、COLMAP 审计仅 20 对、SfM 任意尺度、R1 缺少物理参数 |
| 24 | GitHub | 本报告及代码已纳入主分支；远程仓库：<https://github.com/Atypical-Programmer/Underwater_Binocular> |

## 可复核命令

```powershell
underwater outputs inventory
underwater outputs prune --dry-run
conda run --no-capture-output -n cv python -m pytest -q
conda run --no-capture-output -n cv ruff check src tests scripts/run_colmap_compatibility_smoke.py
```

最后一次正式的 ZED SVO replay 需要安装并正确发现厂商 SDK；在此之前，native/custom 的边界由单元测试、元数据约束和已存在的历史运行结果共同保护。
