<#
.SYNOPSIS
    Run the 1000-frame left/right ALIKED + LightGlue + COLMAP comparison.

.DESCRIPTION
    This is the LightGlue counterpart to run_aliked_adalam_colmap_1000.ps1.
    It keeps the same first 1000 source frames, ALIKED configuration, bounded
    pair graph, custom FULL_OPENCV calibration, frozen calibration mode, and
    calibrated stereo planar mapping as the AdaLAM comparison run. Only the
    descriptor matcher changes. HDF5/INS poses are deliberately not supplied
    so the result compares LightGlue with the visual AdaLAM PRIMARY run.

    Relative paths are resolved by the Python command from the repository root.

.EXAMPLE
    .\scripts\run_aliked_lightglue_colmap_1000.ps1 `
        -StartFrame 0 -EndFrame 999 -Frames 1000 -StereoWindow 1

.EXAMPLE
    # Reuse the same experiment but stop after feature/match database creation.
    .\scripts\run_aliked_lightglue_colmap_1000.ps1 -SkipColmap
#>
[CmdletBinding()]
param(
    [string]$Dataset = "configs/datasets/20260802_150233.yaml",
    [string]$Svo = "20260802_150233.svo2",
    [string]$Profile = "calibration/profiles/zed2i_37395692_custom.yaml",
    [string]$Output = "outputs/20260802_150233/sfm/sfm__custom__aliked_lightglue__1000f__PRIMARY",
    [int]$Frames = 1000,
    [int]$StartFrame = 0,
    [int]$EndFrame = 999,
    [int]$FrameStep = 1,
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [string]$AlikedModel = "aliked-n16",
    [int]$Resize = 1024,
    [int]$MaxKeypoints = 800,
    [double]$DetectionThreshold = 0.2,
    [int]$NmsRadius = 2,
    [int]$TemporalWindow = 5,
    [int]$ExtraStride = 10,
    [int]$StereoWindow = 1,
    [int]$MinRawMatches = 20,
    [int]$MaxMatchesPerPair = 0,
    [ValidateSet("PINHOLE", "OPENCV", "FULL_OPENCV")]
    [string]$CameraModel = "FULL_OPENCV",
    [int]$ColmapThreads = 8,
    [int]$MinGeometricInliers = 15,
    [double]$MaxGeometricError = 4.0,
    [int]$MinModelSize = 10,
    [int]$MapperMinNumMatches = 15,
    [int]$InitMinNumInliers = 50,
    [double]$InitMinTriAngle = 2.0,
    [bool]$CalibratedStereoPlanar = $true,
    [double]$StereoMaxReprojectionError = 8.0,
    [double]$StereoMotionRansacThresholdM = 0.12,
    [double]$LightGlueFilterThreshold = 0.1,
    [double]$LightGlueDepthConfidence = 0.95,
    [double]$LightGlueWidthConfidence = 0.99,
    [string]$ColmapExecutable,
    [switch]$LeftOnly,
    [switch]$RefineCalibration,
    [switch]$SkipColmap,
    [switch]$NoResume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if (-not (Get-Command underwater -ErrorAction SilentlyContinue)) {
    throw "The 'underwater' command was not found. Activate the project environment and install the package first."
}

$cliArgs = @(
    "sfm", "aliked-colmap",
    "--dataset", $Dataset,
    "--num-frames", $Frames,
    "--start-frame", $StartFrame,
    "--frame-step", $FrameStep,
    "--device", $Device,
    "--aliked-model", $AlikedModel,
    "--resize", $Resize,
    "--max-keypoints", $MaxKeypoints,
    "--detection-threshold", $DetectionThreshold,
    "--nms-radius", $NmsRadius,
    "--matcher", "lightglue",
    "--lightglue-filter-threshold", $LightGlueFilterThreshold,
    "--lightglue-depth-confidence", $LightGlueDepthConfidence,
    "--lightglue-width-confidence", $LightGlueWidthConfidence,
    "--temporal-window", $TemporalWindow,
    "--extra-stride", $ExtraStride,
    "--stereo-window", $StereoWindow,
    "--min-raw-matches", $MinRawMatches,
    "--max-matches-per-pair", $MaxMatchesPerPair,
    "--camera-model", $CameraModel,
    "--colmap-threads", $ColmapThreads,
    "--min-geometric-inliers", $MinGeometricInliers,
    "--max-geometric-error", $MaxGeometricError,
    "--min-model-size", $MinModelSize,
    "--mapper-min-num-matches", $MapperMinNumMatches,
    "--init-min-num-inliers", $InitMinNumInliers,
    "--init-min-tri-angle", $InitMinTriAngle,
    "--stereo-max-reprojection-error", $StereoMaxReprojectionError,
    "--stereo-motion-ransac-threshold-m", $StereoMotionRansacThresholdM,
    "--output", $Output
)

if ($CalibratedStereoPlanar) {
    $cliArgs += "--calibrated-stereo-planar"
}
if (-not [string]::IsNullOrWhiteSpace($Svo)) {
    $cliArgs += @("--svo", $Svo)
}
if (-not [string]::IsNullOrWhiteSpace($Profile)) {
    $cliArgs += @("--profile", $Profile)
}
if (-not [string]::IsNullOrWhiteSpace($ColmapExecutable)) {
    $cliArgs += @("--colmap-executable", $ColmapExecutable)
} elseif (-not [string]::IsNullOrWhiteSpace($env:COLMAP_EXECUTABLE)) {
    $cliArgs += @("--colmap-executable", $env:COLMAP_EXECUTABLE)
}
if ($EndFrame -ge 0) {
    $cliArgs += @("--end-frame", $EndFrame)
}
if (-not $LeftOnly) {
    $cliArgs += "--include-right"
}
if ($RefineCalibration) {
    $cliArgs += "--no-freeze-calibration"
} else {
    $cliArgs += "--freeze-calibration"
}
if ($SkipColmap) {
    $cliArgs += "--skip-colmap"
}
if ($NoResume) {
    $cliArgs += "--no-resume"
} else {
    $cliArgs += "--resume"
}

Write-Host "Repository: $repoRoot"
Write-Host "Running: underwater $($cliArgs -join ' ')"

$exitCode = 1
Push-Location -LiteralPath $repoRoot
try {
    & underwater @cliArgs
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $exitCode
