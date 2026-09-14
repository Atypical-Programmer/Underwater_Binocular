<#
.SYNOPSIS
    Run the full 1000-frame left/right ALIKED + AdaLAM + COLMAP workflow.

.DESCRIPTION
    By default, 1000 synchronized source-frame positions are sampled uniformly
    over the configured SVO, so --include-right produces 2000 images. The
    defaults mirror the validated large SfM setup: ALIKED aliked-n16,
    1024-pixel resize, 800 keypoints, temporal window 5, extra stride 10,
    synchronized stereo pairs, frozen FULL_OPENCV calibration, and COLMAP.

    Run this script from any directory. Relative paths are resolved from the
    repository root (the parent directory of this script).

.EXAMPLE
    .\scripts\run_aliked_adalam_colmap_1000.ps1 `
        -ColmapExecutable "D:\Underwater\Software\colmap-x64-windows-cuda\bin\colmap.exe"

.EXAMPLE
    # Reconstruct the first consecutive 1000 source frames instead of a
    # uniform sample across the complete SVO.
    .\scripts\run_aliked_adalam_colmap_1000.ps1 `
        -StartFrame 0 -EndFrame 999 -Frames 1000

.EXAMPLE
    # Build features/matches only and stop before invoking external COLMAP.
    .\scripts\run_aliked_adalam_colmap_1000.ps1 -SkipColmap
#>
[CmdletBinding()]
param(
    [string]$Dataset = "configs/datasets/20260802_150233.yaml",
    [string]$Svo = "20260802_150233.svo2",
    [string]$Profile = "calibration/profiles/zed2i_37395692_custom.yaml",
    [string]$Output = "outputs/20260802_150233/sfm/sfm__custom__aliked_adalam__1000f__PRIMARY",
    [int]$Frames = 1000,
    [int]$StartFrame = 0,
    [int]$EndFrame = -1,
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
    [int]$StereoWindow = 0,
    [int]$MinRawMatches = 20,
    [int]$MaxMatchesPerPair = 0,
    [ValidateSet("PINHOLE", "OPENCV", "FULL_OPENCV")]
    [string]$CameraModel = "FULL_OPENCV",
    [string]$ColmapExecutable,
    [int]$ColmapThreads = 8,
    [int]$MinGeometricInliers = 15,
    [double]$MaxGeometricError = 4.0,
    [int]$MinModelSize = 10,
    [int]$MapperMinNumMatches = 15,
    [int]$InitMinNumInliers = 50,
    [double]$InitMinTriAngle = 2.0,
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
    "--init-min-tri-angle", $InitMinTriAngle
)

if (-not [string]::IsNullOrWhiteSpace($Svo)) {
    $cliArgs += @("--svo", $Svo)
}
if (-not [string]::IsNullOrWhiteSpace($Profile)) {
    $cliArgs += @("--profile", $Profile)
}
if (-not [string]::IsNullOrWhiteSpace($Output)) {
    $cliArgs += @("--output", $Output)
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
