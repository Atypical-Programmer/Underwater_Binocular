param(
    [UInt64]$MaxFrames = 0,
    [string]$Output = 'output\20260802_150233_orbslam3_stereo',
    [double]$ImageScale = 1.0
)

$ErrorActionPreference = 'Stop'

$slamRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspace = Split-Path -Parent $slamRoot
$zedRoot = 'C:\Program Files (x86)\ZED SDK.old'
$cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$python = 'C:\Users\10179\.conda\envs\zed\python.exe'
$vocab = Join-Path $slamRoot 'ORB_SLAM3\Vocabulary\ORBvoc.txt'
$settings = Join-Path $slamRoot 'config\20260802_150233_stereo.yaml'
$svo = Join-Path $workspace '20260802_150233.svo2'
$imageScaleText = $ImageScale.ToString([System.Globalization.CultureInfo]::InvariantCulture)

if (-not (Test-Path -LiteralPath $svo)) {
    throw "SVO2 file was not found: $svo"
}
if (-not (Test-Path -LiteralPath $vocab)) {
    throw "ORB vocabulary was not found: $vocab"
}
if (-not (Test-Path -LiteralPath $settings)) {
    throw "ORB-SLAM3 settings were not found: $settings"
}

if (Test-Path -LiteralPath $python) {
    & $python (Join-Path $slamRoot 'prepare_orbslam3_stereo.py') '--root' $workspace '--output' $settings '--scale' $imageScaleText
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$exe = Get-ChildItem -LiteralPath (Join-Path $slamRoot 'build') -Filter 'svo2_stereo.exe' -File -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match '\\Release\\' } |
    Select-Object -First 1
if ($null -eq $exe) {
    throw 'svo2_stereo.exe was not found. Run SLAM\build_orbslam3.ps1 first.'
}

$outputPath = if ([System.IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $workspace $Output }
$vcpkgBuildRoot = Join-Path $slamRoot 'build\vcpkg_installed\x64-windows'
$vcpkgDirectRoot = Join-Path $slamRoot 'vcpkg_installed\x64-windows'
$env:Path = "$zedRoot\bin;$cudaRoot\bin;$(Join-Path $vcpkgBuildRoot 'bin');$(Join-Path $vcpkgBuildRoot 'debug\bin');$(Join-Path $vcpkgDirectRoot 'bin');$(Join-Path $vcpkgDirectRoot 'debug\bin');$env:Path"

Set-Location $workspace
& $exe.FullName $vocab $settings $svo $outputPath ([string]$MaxFrames) $imageScaleText
exit $LASTEXITCODE
