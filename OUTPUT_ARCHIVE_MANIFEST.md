# Output archive manifest

This manifest records historical material moved from `output/` into the local ignored archive or formal `outputs/` tree. The migration was completed on 2026-09-14; all listed moves were verified and `output/` was then removed after it became empty.

| Original path | New path | Original size bytes | Category | Reason/scientific role | Replacement/reference | README |
|---|---|---:|---|---|---|---|
| `output/20260802_150233_custom_depth_histogram` | `outputs/20260802_150233/depth/depth__custom__zed_sdk__full35855__reference` + local archive | 4755349 | depth | authoritative operational custom SDK histogram | `depth_reference.json` | depth archive README |
| `output/20260802_150233_sgbm_depth_histogram_1000` | `outputs/20260802_150233/depth/depth__custom__sgbm_halfres__1000f__reference` + local archive | 2714677 | depth | independent SGBM reference | `sgbm_reference.json` | depth archive README |
| `output/20260802_150233_tracking_gen1_neural` | `outputs/20260802_150233/tracking/tracking__native__gen1_neural__full35855__baseline` | 12123994 | tracking | historical native GEN_1 baseline | `tracking_native_reference.json` | active baseline README |
| `output/20260802_150233_tracking_ab_performance` | `outputs/_archive/legacy-2026-09-14/tracking/tracking__native__gen1_performance__full35855__ablation` | 12154393 | tracking | native performance ablation | tracking archive index | entry README |
| `output/20260802_150233_tracking_custom_gen1_neural` | `outputs/_archive/legacy-2026-09-14/tracking/tracking__custom__gen1_neural__full35855__ANOMALOUS` | 12093142 | tracking | anomalous custom calibration comparison | native tracking reference | entry README |
| `output/20260802_150233_tracking_custom_gen1_smoke` | `outputs/_archive/legacy-2026-09-14/tracking/tracking__custom__gen1_neural__smoke__diagnostic` | 39327 | tracking | small custom smoke evidence | tests/runbook | entry README |
| `output/20260802_150233_custom_sdk_pointcloud_full` | `outputs/20260802_150233/pointcloud/pointcloud__custom__zed_sdk__1000mapped__reference` | 1797695272 | pointcloud | verified authoritative SDK PLY/pose/mapping metadata | calibration and pointcloud references | active README |
| `output/20260802_150233_custom_gen1_full` | `outputs/_archive/legacy-2026-09-14/pointcloud/pointcloud__custom__gen1__full__ANOMALOUS` | 1779284558 | pointcloud | anomalous custom GEN_1 mapping comparison | custom SDK pointcloud reference | entry README |
| `output/20260802_150233_custom_gen1_sample10pct` | `outputs/_archive/legacy-2026-09-14/pointcloud/pointcloud__custom__gen1__sample10pct__ANOMALOUS` | 2250262172 | pointcloud | anomalous sampled comparison | custom SDK pointcloud reference | entry README |
| `output/20260802_150233_pointcloud1000` | `outputs/_archive/legacy-2026-09-14/pointcloud/pointcloud__low_provenance__1000__comparison/pointcloud1000` | 1243720793 | pointcloud | low-provenance comparison metadata/pose | custom SDK pointcloud reference | entry README |
| `output/20260802_150233_pointcloud1000_gen1` | `outputs/_archive/legacy-2026-09-14/pointcloud/pointcloud__low_provenance__1000__comparison/pointcloud1000_gen1` | 1243841917 | pointcloud | low-provenance comparison metadata/pose | custom SDK pointcloud reference | entry README |
| `output/20260802_150233_orbslam3_resampled/diagnostic_full` | `outputs/20260802_150233/slam/slam__orbslam3__diagnostic_full__35855f__baseline` | 15575545 | slam | full diagnostic baseline | `orbslam3_reference.json` | active README |
| `output/20260802_150233_orbslam3_resampled/stable_map` | `outputs/_archive/legacy-2026-09-14/slam/slam__orbslam3__stable_map__INCONSISTENT__stale` | 428733 | slam | stale/inconsistent audit evidence | `orbslam3_reference.json` | entry README |
| `output/20260802_150233_orbslam3_stereo` | `outputs/20260802_150233/slam/slam__orbslam3__stereo_halfres__35855f__baseline` | 10560038 | slam | half-resolution stereo baseline | `orbslam3_reference.json` | active README |
| `output/20260802_150233_orbslam3_uniform10pct_custom` | `outputs/_archive/legacy-2026-09-14/slam/slam__orbslam3__sampled_custom__diagnostic` | 1447493856 | slam | sampled diagnostic metadata; large PLY explicitly deleted | ORB-SLAM3 references | entry README |
| `output/20260802_150233_sample1000` | `outputs/_archive/legacy-2026-09-14/sfm/sfm__custom__aliked_adalam__sample1000__ablation` + formal primary/secondary + cache | 19471490853 | sfm | split mixed legacy bundle by scientific role | `sfm_reference.json`, `SAMPLE1000_CLEANUP_MANIFEST.md` | entry README |
| `output/20260802_150233_calibration_comparison.json` | `outputs/_archive/legacy-2026-09-14/diagnostics/calibration__comparison__20260802_150233__reference` | 14225 | diagnostics | native/custom calibration evidence | calibration comparison reference | entry README |
| `outputs/20260802_150233/sfm/sfm__custom__aliked_adalam_colmap__20f__diagnostic` | `outputs/20260802_150233/sfm/sfm__custom__aliked_adalam_colmap__20f__diagnostic` | 134176571 | sfm/compatibility | real external COLMAP compatibility audit; retained as formal diagnostic evidence | `colmap_compatibility_reference.json` | active output README |

## Execution status

- Migration completed: 2026-09-14 (UTC).
- `output/` was rechecked as empty and removed; the legacy root no longer exists.
- Files retained locally are either formal `outputs/` results, the ignored `outputs/_archive/legacy-2026-09-14/`, or the ignored sampled-image `cache/`.
- The compatibility audit row is a formal output created during validation rather than a move from the legacy tree; it is included here so the complete output topology has one index.
