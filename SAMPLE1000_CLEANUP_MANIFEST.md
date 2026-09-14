# `sample1000` cleanup manifest

The legacy `output/20260802_150233_sample1000` bundle contains 19,471,490,853 bytes and 8,603 files. It mixes sampled images, raw float32 depth, feature caches, databases, sparse models, and multiple ablations. It is split by role before deletion.

| Subpath | Size/files | Classification | Action | Replacement/reference | Reason |
|---|---:|---|---|---|---|
| `left/`, `right/` | 5,207,272,369 / 1,000 + 1,000 | sampled image cache | move to `cache/20260802_150233/sampled_images/legacy_uniform1000/` | SVO + metadata | reproducible from SVO; keep temporarily as disposable cache |
| `depth_raw/` | 8,294,528,000 / 1,000 | raw float32 depth cache | DELETE after source/SVO/metadata recheck | custom SDK depth reference | dense derivative; no physical-GT role |
| `depth_preview/` | 449,064,345 / 1,000 | preview cache | DELETE after summary/reference | depth reference JSON and plots | reproducible visualization |
| `sfm_aliked_adalam_2000_full_opencv_refined/` | 1,111,426,959 / 2,091 | PRIMARY SfM | retain selected sparse model/metadata; archive/delete redundant caches | `validation/reference/20260802_150233/sfm_reference.json` | 1,448 registered / 136,707 points3D |
| `sfm_aliked_adalam_100_full_opencv_refined/` | 63,686,704 / 127 | secondary reference | archive compact model/summary; delete redundant caches | SfM reference JSON | 100 registered / 12,887 points3D |
| `sfm_aliked_adalam_200_full_opencv_refined/` | 2,288,758,686 / 854 | failed/weak experiment | archive summary; delete bulky derivative | SfM reference JSON | only 3 registered / 579 points3D |
| `sfm_aliked_adalam_1000_full_opencv_refined/` | 502,271,144 / 1,023 | incomplete experiment | archive summary/feature metrics; delete bulky derivative | SfM reference JSON | no run summary or sparse model |
| `sfm_aliked_adalam_100`, `_v2`, `_window1`, `_pinhole_window1` | see inventory | ablations | archive summaries; delete bulky derivative | SfM reference JSON | retain only representative results |
| `_smoke_stereo_full_opencv/` | 3,650,896 / 25 | smoke | archive summary | diagnostics archive | not a production result |
| `colmap_sparse/` | 1,333,971,171 / 9 | legacy duplicate sparse export | archive compact model summary; delete duplicate database/model | primary SfM reference | superseded by selected formal primary |
| `metadata.json`, `pose_world.csv` | 270,137 / 2 | provenance | archive | source SVO and reference docs | retain provenance while bundle is split |

The 2,000-frame branch is selected as PRIMARY because it is the only large run with a completed sparse model and broad coverage. No deletion from this bundle is allowed until the delete manifest contains the exact source path, size, reason, replacement, extracted summary, and hash where relevant.
