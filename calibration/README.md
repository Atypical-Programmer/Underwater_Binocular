# Calibration directory

`profiles/` contains the canonical computational calibration. `generated/` contains deterministic external derivatives and is regenerated from the profile. `source/` retains original calibration evidence and is not loaded implicitly by the production pipeline.

Use `underwater calibration validate` and `underwater calibration generate`; do not edit generated files by hand.
