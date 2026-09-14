# Output delete manifest

This manifest was prepared before cleanup. A row is added for every file or file set/directory removed from the legacy `output/` tree. Deletion is permitted only after the replacement/reference and extracted summary are present and the source is rechecked.

## Execution record

Cleanup completed on 2026-09-14 (UTC). The explicit deletion set removed **6,572 files** totaling **25,486,542,015 bytes**; six named empty directories and the now-empty `output/` container were also removed. The byte total is the sum of file rows/sets below and intentionally excludes directory rows, so no recursive size is double-counted. All source containers were rechecked empty before removal. The original SVO, calibration source files, and the retained archive/formal outputs were not deleted.

| Original path | Size bytes | Reason | Replacement/reference | Summary extracted | SHA256 if relevant | Deleted |
|---|---:|---|---|---|---|---|
| `output/_zed_custom_neural_20` | 0 | known empty directory | none | none | n/a | yes |
| `output/20260802_150233_custom_gen1_area_test` | 0 | known empty directory | none | none | n/a | yes |
| `output/20260802_150233_final_underwater_audit` | 0 | known empty directory | diagnostics archive/index | none | n/a | yes |
| `output/20260802_150233_flat_port_refractive_audit` | 0 | known empty directory | diagnostics archive/index | none | n/a | yes |
| `output/20260802_150233_refractive_depth_audit` | 0 | known empty directory | diagnostics archive/index | none | n/a | yes |
| `output/20260802_150233_tracking_ab` | 0 | known empty directory | tracking archive/index | none | n/a | yes |
| `output/20260802_150233_left_right.mp4` | 2726690581 | SVO and sidecar metadata exist; no unique scientific result | root `20260802_150233.svo2`, depth/tracking/SfM references | sidecar copied to diagnostics archive | not hashed; unique video not required | yes |
| `output/_convert_svo2_perf.mp4` | 7615829 | performance-only conversion video | sidecar metadata and SVO | sidecar copied to diagnostics archive | not hashed; performance-only | yes |
| `output/20260802_150233_left_preview_frame000000.jpg` | 371756 | no tracked/reference use found | none | not needed | n/a | yes |
| `output/_audit_calibration_comparison.json` | 14225 | byte-identical duplicate | tracked calibration comparison reference and other source copy | yes | `80d8a9729922c17db70e02d4bf58fcd65d423baff018c5f702307da9a9b9a2e0` | yes |
| `output/20260802_150233_sample1000/depth_raw` | 8294528000 | raw float32 depth derivative; source SVO/summary retained | `depth_reference.json` | yes | n/a | yes |
| `output/20260802_150233_sample1000/depth_preview` | 449064345 | reproducible preview cache | `depth_reference.json` and archived plots | yes | n/a | yes |
| `output/20260802_150233_sample1000/sfm_aliked_adalam_2000_full_opencv_refined/features` | 590987722 | feature cache; reproducible from images/config | selected sparse model and SfM reference | yes | n/a | yes |
| `output/20260802_150233_sample1000/sfm_aliked_adalam_2000_full_opencv_refined/database.db` | 221360128 | intermediate COLMAP database after selected model retained | selected sparse model and `sfm_reference.json` | yes | n/a | yes |
| `output/20260802_150233_sample1000/sfm_aliked_adalam_2000_full_opencv_refined/matches_raw.txt` | 20593924 | reproducible correspondence export | selected sparse model and SfM reference | yes | n/a | yes |
| `output/20260802_150233_sample1000/sfm_aliked_adalam_2000_full_opencv_refined/redundant_sparse_variants` | 159092434 | redundant mapper variants | selected `sparse/` model | yes | n/a | yes |
| `output/20260802_150233_sample1000/sfm_aliked_adalam_100_full_opencv_refined/features` | 38403728 | feature cache after secondary sparse model retained | secondary sparse model/reference | yes | n/a | yes |
| `output/20260802_150233_sample1000/sfm_aliked_adalam_100_full_opencv_refined/database.db` | 11931648 | intermediate database after secondary model retained | secondary sparse model/reference | yes | n/a | yes |
| `output/20260802_150233_sample1000/sfm_aliked_adalam_100_full_opencv_refined/matches_raw.txt` | 399778 | reproducible correspondence export | secondary sparse model/reference | yes | n/a | yes |
| `output/20260802_150233_sample1000/sfm_aliked_adalam_200_full_opencv_refined` | 2288758686 | weak experiment: only 3 registered images/579 points3D | archived summary and selected primary reference | yes | n/a | yes |
| `output/20260802_150233_sample1000/sfm_aliked_adalam_1000_full_opencv_refined` | 502271144 | incomplete: no run summary or sparse model | archived feature/match metrics and `sfm_reference.json` | yes | n/a | yes |
| `output/20260802_150233_sample1000/sfm_aliked_adalam_100` | 53506268 | ablation derivative after summary extraction | SfM archive/index | yes | n/a | yes |
| `output/20260802_150233_sample1000/sfm_aliked_adalam_100_v2` | 53324142 | ablation derivative after summary extraction | SfM archive/index | yes | n/a | yes |
| `output/20260802_150233_sample1000/sfm_aliked_adalam_100_window1` | 51051219 | ablation derivative after summary extraction | SfM archive/index | yes | n/a | yes |
| `output/20260802_150233_sample1000/sfm_aliked_adalam_100_pinhole_window1` | 58708813 | ablation derivative after summary extraction | SfM archive/index | yes | n/a | yes |
| `output/20260802_150233_sample1000/colmap_sparse/database.db` | 870834176 | duplicate/intermediate database; alternate sparse files archived | selected primary sparse model and `sfm_reference.json` | yes | n/a | yes |
| `output/20260802_150233_custom_gen1_full/scene_world_rgb.ply` | 1138282772 | anomalous custom point-cloud derivative | archived pose/metadata and custom SDK point-cloud reference | yes | `df1cd49562d0ceb870f34a45926632cbd3dd41571a0ff20981946096eb8922a6` | yes |
| `output/20260802_150233_custom_gen1_full/area_map_custom.area` | 621505937 | large derivative after verified PLY/pose/mapping metadata retained | custom SDK point-cloud reference | yes | n/a | yes |
| `output/20260802_150233_custom_gen1_sample10pct/scene_world_rgb.ply` | 1599311328 | anomalous sampled custom point-cloud derivative | archived pose/metadata and custom SDK point-cloud reference | yes | `ff6fd99c8948d897e090177fdaadc0c382ee41d2286dbd28f95f2d52d4a042fd` | yes |
| `output/20260802_150233_custom_gen1_sample10pct/area_map_custom.area` | 649453184 | large derivative after summary/metadata retention | custom SDK point-cloud reference | yes | n/a | yes |
| `output/20260802_150233_custom_sdk_pointcloud_full/area_map_custom.area` | 647223457 | large intermediate; authoritative PLY/pose/mapping/metadata retained | active custom SDK point-cloud reference | yes | n/a | yes |
| `output/20260802_150233_pointcloud1000/scene_world_rgb.ply` | 1243384352 | low-provenance comparison; not byte-identical to other same-size candidate | archived metadata/pose and custom SDK reference | yes | `9dea149d48ccea8f8f655bf90f4668101e1965f8341ea0faf387ab4bc15f606d` | yes |
| `output/20260802_150233_pointcloud1000_gen1/scene_world_rgb.ply` | 1243384352 | low-provenance comparison; not byte-identical to other same-size candidate | archived metadata/pose and custom SDK reference | yes | `c325028904de808b9d63f5b4500972801e4b58be65171c06c7f99cc3ab7dccd4` | yes |
| `output/20260802_150233_orbslam3_uniform10pct_custom/pointcloud/scene_world_rgb_orbslam3_pose.ply` | 1446816407 | large sampled derivative; metadata retained | ORB-SLAM3 full/stereo baselines and archive README | yes | `8f2e279a2444b3727331ff615b90cea71261a8903e2c8b803ae9d881ec3c5745` | yes |
| `output/20260802_150233_custom_sgbm_depth_20/*.npy` | 165890560 | 20-frame dense diagnostic cache after summary/rectification retention | SGBM reference | yes | n/a | yes |
| `output/20260802_150233_strict_calibration_compare_20/*.npy` | 331781120 | 40-frame dense comparison cache after comparison summary retention | calibration comparison reference | yes | n/a | yes |
| `output/` | 0 after migration | legacy root must disappear after all contents are verified/moved | `outputs/`, `cache/`, archive, tracked references | yes | n/a | yes |
