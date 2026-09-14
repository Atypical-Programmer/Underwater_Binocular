[CmdletBinding()]
param(
    [int]$Frames = 50,
    [string]$Dataset = "configs/datasets/20260802_150233.yaml",
    [string]$Output,
    [string]$Svo,
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [switch]$IncludeRight,
    [switch]$SkipColmap,
    [switch]$RefineCalibration
)

$cliArgs = @(
    "sfm", "aliked-colmap",
    "--dataset", $Dataset,
    "--num-frames", $Frames,
    "--device", $Device
)

if ($IncludeRight) {
    $cliArgs += "--include-right"
}
if ($SkipColmap) {
    $cliArgs += "--skip-colmap"
}
if ($RefineCalibration) {
    $cliArgs += "--no-freeze-calibration"
}
if ($PSBoundParameters.ContainsKey("Output")) {
    $cliArgs += @("--output", $Output)
}
if ($PSBoundParameters.ContainsKey("Svo")) {
    $cliArgs += @("--svo", $Svo)
}

& underwater @cliArgs
exit $LASTEXITCODE
