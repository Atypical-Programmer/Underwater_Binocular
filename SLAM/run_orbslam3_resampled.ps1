param(
    [string]$Output = 'output\20260802_150233_orbslam3_resampled',
    [double]$ImageScale = 1.0,
    [int]$MinSegmentFrames = 3000,
    [switch]$SkipInteractiveViewer
)

$ErrorActionPreference = 'Stop'

$slamRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspace = Split-Path -Parent $slamRoot
$zedRoot = 'C:\Program Files (x86)\ZED SDK.old'
$python = 'C:\Users\10179\.conda\envs\zed\python.exe'
$svo = Join-Path $workspace '20260802_150233.svo2'
$calibrationRoot = Join-Path $workspace 'Calibration'
$buildScript = Join-Path $slamRoot 'build_orbslam3.ps1'
$runner = Join-Path $slamRoot 'run_orbslam3_svo2.ps1'
$organizer = Join-Path $slamRoot 'prepare_orbslam3_resampled.py'
$viewer = Join-Path $slamRoot 'visualize_camera_frames_interactive.py'
$outputPath = if ([System.IO.Path]::IsPathRooted($Output)) {
    [System.IO.Path]::GetFullPath($Output)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $workspace $Output))
}
$diagnosticDir = Join-Path $outputPath 'diagnostic_full'
$stableDir = Join-Path $outputPath 'stable_map'
$planPath = Join-Path $diagnosticDir 'segment_plan.json'
$stablePlanPath = Join-Path $stableDir 'segment_plan_actual.json'
$planForBuild = $planPath
$imageScaleText = $ImageScale.ToString([System.Globalization.CultureInfo]::InvariantCulture)

if ($ImageScale -le 0.0 -or $ImageScale -gt 1.0) {
    throw 'ImageScale must be in the interval (0, 1].'
}
if ($MinSegmentFrames -le 0) {
    throw 'MinSegmentFrames must be positive.'
}
foreach ($requiredPath in @($svo, $calibrationRoot, $python, $buildScript, $runner, $organizer)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path was not found: $requiredPath"
    }
}

# Do not mix a partial previous export into an exact-count result.  The
# caller can choose another output path or remove this explicitly named result
# directory before starting a fresh run.
if (Test-Path -LiteralPath $outputPath) {
    $existingFiles = @(Get-ChildItem -LiteralPath $outputPath -File -Recurse -ErrorAction SilentlyContinue)
    if ($existingFiles.Count -gt 0) {
        throw "Output already contains files; refusing to overwrite: $outputPath"
    }
} else {
    New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
}

$env:ZED_SDK_ROOT_DIR = $zedRoot
Set-Location $workspace

Write-Host '=== 1/5 Build ORB-SLAM3 Release ===' -ForegroundColor Cyan
& $buildScript
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host '=== 2/5 Full diagnostic replay at scale 1.0 ===' -ForegroundColor Cyan
& $runner -MaxFrames 0 -Output $diagnosticDir -ImageScale $imageScaleText
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host '=== 3/5 Analyze longest continuous valid map segment ===' -ForegroundColor Cyan
& $python $organizer analyze `
    '--tracking-log' (Join-Path $diagnosticDir 'tracking_log.csv') `
    '--output-plan' $planPath `
    '--min-required' ([string]$MinSegmentFrames)
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$planData = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
$segmentStart = [Int64]$planData.selected_segment.start_frame
$segmentEnd = [Int64]$planData.selected_segment.end_frame
$segmentLength = [Int64]$planData.selected_segment.length
if ($segmentLength -lt $MinSegmentFrames) {
    throw "The longest valid segment has only $segmentLength frames; 3000-frame export is refused."
}
$stableMaxFrames = $segmentEnd + 1
Write-Host ("Selected source segment {0}..{1} ({2} frames); bounded replay ends after source frame {1}." -f $segmentStart, $segmentEnd, $segmentLength) -ForegroundColor Green

Write-Host '=== 4/5 Bounded stable-map replay at scale 1.0 ===' -ForegroundColor Cyan
& $runner -MaxFrames $stableMaxFrames -Output $stableDir -ImageScale $imageScaleText
$stableExitCode = $LASTEXITCODE

# A long full-resolution replay can encounter a numerical failure after a
# perfectly usable prefix.  Never fill the missing poses: inspect the
# completed rows, require at least MinSegmentFrames, then rerun only through
# that verified prefix so ORB-SLAM3 can flush its map and trajectories.
$stableTrackingPath = Join-Path $stableDir 'tracking_log.csv'
$stableTrajectoryPath = Join-Path $stableDir 'CameraTrajectory.txt'
$stableMapPointsPath = Join-Path $stableDir 'map_points_xyz.csv'
$stableHasArtifacts = (Test-Path -LiteralPath $stableTrajectoryPath) -and
    (Test-Path -LiteralPath $stableMapPointsPath)
if (Test-Path -LiteralPath $stableTrackingPath) {
    & $python $organizer analyze `
        '--tracking-log' $stableTrackingPath `
        '--output-plan' $stablePlanPath `
        '--min-required' ([string]$MinSegmentFrames)
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not analyze the partial stable-map tracking log.'
    }
    $stablePlanData = Get-Content -LiteralPath $stablePlanPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $stableSegmentLength = [Int64]$stablePlanData.selected_segment.length
    $stableSegmentEnd = [Int64]$stablePlanData.selected_segment.end_frame
    if ($stableSegmentLength -lt $MinSegmentFrames) {
        throw "Stable replay failed and its longest valid prefix has only $stableSegmentLength frames; refusing 3000-frame export."
    }
    $planForBuild = $stablePlanPath
    if (($stableExitCode -ne 0) -or (-not $stableHasArtifacts) -or ($stableSegmentEnd -lt $segmentEnd)) {
        $repairMaxFrames = $stableSegmentEnd + 1
        Write-Warning ("Stable replay did not finish the diagnostic segment; repairing with a bounded replay through source frame {0}." -f $stableSegmentEnd)
        & $runner -MaxFrames $repairMaxFrames -Output $stableDir -ImageScale $imageScaleText
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        $stableHasArtifacts = (Test-Path -LiteralPath $stableTrajectoryPath) -and
            (Test-Path -LiteralPath $stableMapPointsPath)
        if (-not $stableHasArtifacts) {
            throw 'The repaired stable replay finished without trajectory and MapPoint exports.'
        }
        & $python $organizer analyze `
            '--tracking-log' $stableTrackingPath `
            '--output-plan' $stablePlanPath `
            '--min-required' ([string]$MinSegmentFrames)
        if ($LASTEXITCODE -ne 0) {
            throw 'Could not analyze the repaired stable-map tracking log.'
        }
        $stablePlanData = Get-Content -LiteralPath $stablePlanPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $stableSegmentLength = [Int64]$stablePlanData.selected_segment.length
        if ($stableSegmentLength -lt $MinSegmentFrames) {
            throw "The repaired stable replay has only $stableSegmentLength valid frames; refusing 3000-frame export."
        }
        $planForBuild = $stablePlanPath
    }
} elseif ($stableExitCode -ne 0) {
    exit $stableExitCode
}
if (-not $stableHasArtifacts) {
    throw 'Stable replay did not produce the required CameraTrajectory.txt and map_points_xyz.csv.'
}

Write-Host '=== 5/5 Extract exact samples, Metashape YPR, and colored sparse map ===' -ForegroundColor Cyan
& $python $organizer build `
    '--stable-dir' $stableDir `
    '--plan' $planForBuild `
    '--svo' $svo `
    '--calibration-root' $calibrationRoot `
    '--output-root' $outputPath
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not $SkipInteractiveViewer) {
    $interactiveOutput = Join-Path $outputPath 'camera_frames_interactive.html'
    & $python $viewer `
        '--trajectory' (Join-Path $stableDir 'CameraTrajectory.txt') `
        '--tracking-log' (Join-Path $stableDir 'tracking_log.csv') `
        '--output' $interactiveOutput `
        '--sample-count' '120'
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "Completed ORB-SLAM3 resampled export: $outputPath" -ForegroundColor Green
exit 0
