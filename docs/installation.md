# Installation

The core package targets Python 3.10 or newer and uses NumPy, OpenCV, and PyYAML. Create the portable environment with:

```powershell
conda env create -f environment.yml
conda activate underwater-binocular
python -m pip install -e ".[dev]"
```

The ZED SDK and `pyzed.sl` binding are installed separately by the vendor. Set `ZED_SDK_ROOT_DIR` to that installation when running SDK-backed workflows. The package does not invent a fallback installation path.

Optional SfM dependencies are intentionally separate:

```powershell
python -m pip install -e ".[dev,sfm]"
```

For a machine-local recording, copy `configs/local.example.yaml` to an ignored local file and set `UNDERWATER_SVO_PATH`, or pass `--svo` on the command line. Do not commit absolute paths, recordings, dense arrays, or generated run outputs.

Smoke checks:

```powershell
underwater --help
underwater calibration validate
underwater calibration generate
python -m pytest -q tests
```
