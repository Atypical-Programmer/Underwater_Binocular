# Migration regression reference: `20260802_150233`

This directory contains compact, versioned summaries extracted before the architecture migration. It deliberately excludes the SVO, videos, full depth maps, large correspondence tables, and point clouds.

These files are behavioral regression references, not independent physical ground truth. In particular, the approximately `2.204 m` custom SDK center depth is an expected operational output for this dataset and calibration, not a proven underwater distance.

## Sources

- Dataset: `20260802_150233.svo2`, ZED 2i serial `37395692`, 1920x1080, 30 FPS, 35,855 frames.
- Original calibration evidence: now preserved under `calibration/source/2026_underwater_calibration/`; original pre-migration paths and hashes are retained inside `calibration_reference.json` for provenance.
- Production metadata: ignored local `20260802_150233_custom_sdk_depth_all_metadata.json`, with hash recorded in `sdk_runtime_reference.json`.
- SGBM reference: ignored local `output/20260802_150233_sgbm_depth_histogram_1000/summary.json` and rectification metadata, with hashes recorded in `sgbm_reference.json`.
- Refractive audit evidence: the ten tracked files from the baseline audit directories. Their sizes and SHA256 hashes are recorded in `refractive_audit_reference.json`.

## Interpretation guardrails

- The internal stereo convention is `X_right = R_left_to_right @ X_left + t_left_to_right`.
- Internal physical lengths in the package are metres; source calibration millimetres are converted once at the boundary.
- `Z` is the left-camera optical-axis coordinate. It is not Euclidean range and is not automatically an interface-to-target distance.
- No blanket `x1.333` production correction is represented here.
- R1 rig-level refractive depth remains `NOT_IDENTIFIABLE` because housing/port geometry and independent metric ground truth are missing.

## Reference files

| File | Purpose |
|---|---|
| `calibration_reference.json` | custom source values, units, transform convention, and validation diagnostics |
| `sdk_runtime_reference.json` | native SDK runtime calibration plus custom production runtime/depth metadata |
| `depth_reference.json` | full custom SDK operational center-depth statistics |
| `sgbm_reference.json` | independent OpenCV SGBM/rectification statistics and parameters |
| `calibration_comparison_reference.json` | compact native-vs-custom calibration comparison and duplicate hash |
| `tracking_native_reference.json` | historical native GEN_1 full replay baseline |
| `orbslam3_reference.json` | ORB-SLAM3 baseline runs and stale inconsistent run |
| `sfm_reference.json` | primary/secondary/incomplete ALIKED + AdaLAM + COLMAP results |
| `rectification_reference.json` | native SDK and custom alpha=0/alpha=1 P/Q identities |
| `refractive_audit_reference.json` | compact audit status, counts, hashes, and legacy sensitivity guardrails |
