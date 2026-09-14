# Refactor summary

The repository now has one importable Python package, one canonical calibration profile, explicit generated-file boundaries, a centralized ZED session, typed depth frames, pure geometry modules, separated integrations, and compact behavioral references.

The refactor preserves the prior scientific interpretation. The custom SDK depth run is useful and operational, but it does not independently establish absolute underwater metric depth. The flat-port `R1` correction remains guarded until the missing physical parameters and independent metric validation are supplied.

See [REFACTOR_BASELINE.md](REFACTOR_BASELINE.md), [REFACTOR_CHANGELOG.md](REFACTOR_CHANGELOG.md), and [REFACTOR_REGRESSION_REPORT.md](REFACTOR_REGRESSION_REPORT.md) for evidence and explicit changes.
