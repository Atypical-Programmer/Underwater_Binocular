# Workflow completion report

## 1. What was missing

Before this workflow-completion pass:

- the package had only a `start_tracking()` helper, not a user-facing ZED tracking replay, standard outputs, or GEN_1/GEN_3 comparison workflow;
- the package reconstruction modules exposed ORB + BFMatcher only, while the legacy ALIKED + AdaLAM + COLMAP implementation was still under `experiments/archive/legacy/`;
- there were no package CLI or thin PowerShell entries for these two workflows;
- README and runbooks did not clearly separate implemented code from external workflows that had not been executed.

## 2. What was implemented

- Added a sequential ZED tracking runner with one `ZedSession` per replay, pre-tracking initial seek, explicit `POSITIONAL_TRACKING_MODE.GEN_1/GEN_3` selection, loud unsupported-mode errors, resource cleanup, pose/state CSVs, summary statistics, and provenance metadata.
- Added real `lightglue.ALIKED` extraction with explicit resize-coordinate restoration and configuration/image-hash-aware feature caching.
- Added real `kornia.feature.match_adalam` matching with index validation, duplicate filtering, bounded temporal/stereo pair generation, `pairs.txt`, and raw match summaries.
- Added canonical-profile COLMAP camera/database integration, frozen calibration by default, explicit refinement opt-in, external `matches_importer`/`mapper`/`model_converter` calls, logs, and model summaries.
- Added README capability/status information, four runbooks, two thin PowerShell wrappers, unit tests, and this report.
- Fixed only mechanical ruff issues in two existing ORB-SLAM3 visualization integrations; the existing package architecture and calibration semantics were preserved.

## 3. CLI commands added

```powershell
underwater tracking zed --help
underwater sfm aliked-colmap --help
```

Examples:

```powershell
# GEN_1
underwater tracking zed `
  --dataset configs/datasets/20260802_150233.yaml `
  --mode GEN_1

# GEN_3
underwater tracking zed `
  --dataset configs/datasets/20260802_150233.yaml `
  --mode GEN_3 `
  --max-frames 1000

# quick ALIKED + AdaLAM + COLMAP smoke test, stopping before external COLMAP
underwater sfm aliked-colmap `
  --dataset configs/datasets/20260802_150233.yaml `
  --num-frames 50 `
  --include-right `
  --device cuda `
  --skip-colmap

# longer ALIKED + AdaLAM + COLMAP run
underwater sfm aliked-colmap `
  --dataset configs/datasets/20260802_150233.yaml `
  --num-frames 1000 `
  --include-right `
  --device cuda `
  --freeze-calibration
```

The dataset config uses `UNDERWATER_SVO_PATH` unless `--svo` is supplied.

## 4. PowerShell wrappers added

- `scripts/run_zed_tracking.ps1`: thin wrapper for `underwater tracking zed`; supports `-Mode GEN_1/GEN_3/BOTH`, frame range, SVO, output, and Area Memory.
- `scripts/run_aliked_adalam_colmap.ps1`: thin wrapper for `underwater sfm aliked-colmap`; supports frame count, left/right selection, CPU/CUDA, explicit COLMAP skip, output, and calibration refinement.

## 5. Exact ZED tracking workflow

```text
dataset config + canonical profile
  → resolve SVO/profile and provenance
  → open one ZedSession
  → optional initial seek before tracking
  → enable explicit GEN_1 or GEN_3
  → sequential grab()
  → one get_position() per grabbed frame
  → serialize pose/state
  → compute valid/lost/searching/off/segment/path statistics
  → close the session in a context manager
```

`--mode BOTH` repeats that whole sequence with a newly opened SVO from the beginning for each mode. It never changes modes on one stateful session.

## 6. Exact ALIKED + AdaLAM + COLMAP workflow

```text
dataset/SVO
  → one ZED image extraction session (raw unrectified views)
  → deterministic left/right image names + image_manifest.json
  → ALIKED feature extraction/cache
  → bounded temporal/synchronized stereo pairs
  → Kornia AdaLAM matches
  → custom keypoints/camera rows in COLMAP database
  → matches_importer geometric verification (unless --skip-colmap)
  → mapper with frozen calibration by default
  → model_converter + model_summary.json
```

The ALIKED command does not import ORB, SIFT, BFMatcher, or LightGlue matching as a fallback. `--skip-colmap` is an explicit stop point and is recorded as `NOT_EXECUTED`.

## 7. Tests executed

In the local `cv` conda environment:

```text
python -m pytest -q        → 31 passed
ruff check src tests integrations → All checks passed
python -m compileall -q src tests integrations → passed
```

Additional real neural smoke using six existing local PNGs (three left and three right):

```text
ALIKED: 6 images, 600 keypoints, cache metadata written
AdaLAM: 7 candidate pairs, 7 pairs with matches
cache resume: 6 hits, 0 misses
```

This smoke exercised actual LightGlue ALIKED and Kornia AdaLAM inference. It did not require ZED or COLMAP because it used already extracted PNGs and stopped before external mapping.

## 8. External workflows actually executed

- Package CLI parser/help: **EXECUTED**.
- Unit/integration/regression tests: **EXECUTED**, 31 passed.
- ALIKED inference on six local PNGs: **EXECUTED**.
- AdaLAM inference on the same seven bounded pairs: **EXECUTED**.
- ALIKED feature cache resume: **EXECUTED**.
- COLMAP database plumbing with custom keypoints/matches: **EXECUTED** in the real neural smoke and unit tests; the smoke produced 2 cameras, 6 images, 7 match pairs, and 468 raw correspondences.

## 9. External workflows NOT executed

- ZED GEN_1 real SVO replay: **NOT EXECUTED in this pass**; the available environments did not expose the vendor `pyzed.sl` module, although the SVO file exists locally.
- ZED GEN_3 real SVO replay: **NOT EXECUTED in this pass** for the same reason.
- Full SVO image extraction through the new workflow: **NOT EXECUTED**.
- COLMAP `matches_importer`: **NOT EXECUTED in this pass**.
- COLMAP `mapper`: **NOT EXECUTED in this pass**.
- Full 50/1000-frame SVO reconstruction: **NOT EXECUTED in this pass**.
- Metashape GUI import/optimization: **NOT EXECUTED**; it is an external GUI workflow.
- ORB-SLAM3 native build/replay: **NOT EXECUTED in this pass**.

The existing ignored `output/` directory contains historical real runs, including ALIKED/AdaLAM/COLMAP and ORB-SLAM3 outputs. Those files are evidence of earlier executions, not evidence that the new package entry points were rerun in this pass.

## 10. Dependencies required by each workflow

| Workflow | ZED SDK/SVO | CUDA | COLMAP |
|---|---|---|---|
| ZED tracking | required | not inherently required by the tracking code; depends on SDK/depth mode | no |
| ALIKED + AdaLAM feature/match smoke | no, when using existing images | required for the documented default `--device cuda`; CPU is supported explicitly | no when using `--skip-colmap` |
| Full ALIKED + AdaLAM from SVO | required | default CUDA, or explicit CPU | no until mapper stage |
| COLMAP verification/mapper | no additional ZED requirement after images exist | COLMAP executable may be CPU or CUDA build; this runner disables COLMAP GPU matching/BA | required |
| ORB-SLAM3 | required | required by the current native integration | no |

## 11. Scientific and workflow limitations

- The custom SDK depth near 2.204 m remains an operational empirical result, not independent physical ground truth.
- COLMAP reconstruction scale is arbitrary/local unless an actual valid scale constraint is supplied. Left/right input alone does not prove metric underwater scale.
- Tracking state validity does not prove trajectory physical accuracy; large jumps must be reviewed.
- No blanket `×1.333` correction is introduced or justified.
- The new ALIKED workflow uses raw unrectified images with canonical custom calibration. A rectified-image workflow would require a matching derived rectified camera model and is rejected explicitly rather than silently mixing geometries.
- Large images, databases, feature caches, videos, PLY files, SVOs, and generated outputs remain local and ignored.

## 12. Final answers

**Is ZED SDK tracking now end-to-end runnable?**

Yes, the package has an end-to-end runnable CLI and wrapper when the vendor `pyzed.sl` binding, SVO, and calibration are available. A real GEN_1/GEN_3 replay through the new entry point was not executed in this environment.

**Is ALIKED + AdaLAM genuinely implemented, rather than ORB/BFMatcher under another name?**

Yes. The new workflow calls `lightglue.ALIKED` and `kornia.feature.match_adalam`; missing dependencies fail loudly and there is no fallback. Both were exercised on six local images in the real neural smoke.

**Can COLMAP reconstruction be run directly from the documented command?**

Yes, after installing the SfM dependencies and providing a native COLMAP executable plus SVO/SDK inputs. The command writes the database, imports AdaLAM matches, runs mapper, and converts the model. Those external stages were not rerun in this pass.

**Which parts have actually been validated on the real dataset?**

The existing ignored output directory contains historical full SVO/ZED, depth, tracking, point-cloud, ORB-SLAM3, ALIKED/AdaLAM/COLMAP runs. In this pass, package tests used calibration/reference fixtures and the neural smoke used existing real dataset PNGs; the new ZED SVO extraction and COLMAP mapper were not rerun.

**Does any reconstruction result currently provide independently validated physical underwater metric scale?**

No. Current reconstruction outputs are local/arbitrary-scale results unless an independent valid scale constraint is demonstrated. They do not independently validate the physical underwater metric depth.
