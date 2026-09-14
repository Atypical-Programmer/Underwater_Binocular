param(
    [UInt64]$MaxFrames = 0,
    [string]$Output = 'outputs\20260802_150233_orbslam3_stereo',
    [double]$ImageScale = 1.0,
    [UInt64]$SampleCount = 0,
    [string]$SvoPath = $env:UNDERWATER_SVO_PATH,
    [string]$PythonExecutable = 'python',
    [string]$ZedSdkRoot = $env:ZED_SDK_ROOT_DIR,
    [string]$CudaRoot = $env:CUDA_TOOLKIT_ROOT_DIR
)

$ErrorActionPreference = 'Stop'
$integrationRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = (Resolve-Path (Join-Path $integrationRoot '..\..')).Path
$zedRoot = $ZedSdkRoot
$cudaRoot = $CudaRoot
$vocab = Join-Path $repositoryRoot 'third_party\ORB_SLAM3\Vocabulary\ORBvoc.txt'
$settings = Join-Path $repositoryRoot 'calibration\generated\orbslam3_stereo.yaml'
$imageScaleText = $ImageScale.ToString([System.Globalization.CultureInfo]::InvariantCulture)

if ([string]::IsNullOrWhiteSpace($SvoPath)) {
    throw 'Set UNDERWATER_SVO_PATH or pass -SvoPath to the local SVO2 recording.'
}
$svo = (Resolve-Path -LiteralPath $SvoPath -ErrorAction Stop).Path
if (-not (Test-Path -LiteralPath $vocab)) {
    throw "ORB vocabulary was not found: $vocab"
}
if (-not (Test-Path -LiteralPath $zedRoot)) {
    throw 'Set ZED_SDK_ROOT_DIR to the installed ZED SDK root.'
}
if (-not (Test-Path -LiteralPath $cudaRoot)) {
    throw 'Set CUDA_TOOLKIT_ROOT_DIR to the installed CUDA root.'
}

& $PythonExecutable '-m' 'underwater_binocular.cli' 'calibration' 'generate' '--profile' `
    (Join-Path $repositoryRoot 'calibration\profiles\zed2i_37395692_custom.yaml') `
    '--output-dir' (Join-Path $repositoryRoot 'calibration\generated') '--orb-scale' $imageScaleText
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$exe = Get-ChildItem -LiteralPath (Join-Path $integrationRoot 'build') -Filter 'svo2_stereo.exe' -File -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match '\\Release\\' } |
    Select-Object -First 1
if ($null -eq $exe) {
    throw 'svo2_stereo.exe was not found. Run build_orbslam3.ps1 first.'
}

if ($SampleCount -lt 0) {
    throw 'SampleCount must be non-negative; 0 means every source frame.'
}
$outputPath = if ([System.IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $repositoryRoot $Output }
$vcpkgBuildRoot = Join-Path $integrationRoot 'build\vcpkg_installed\x64-windows'
$vcpkgDirectRoot = Join-Path $integrationRoot 'vcpkg_installed\x64-windows'
$pathParts = @(
    (Join-Path $zedRoot 'bin'),
    (Join-Path $cudaRoot 'bin'),
    (Join-Path $vcpkgBuildRoot 'bin'),
    (Join-Path $vcpkgBuildRoot 'debug\bin'),
    (Join-Path $vcpkgDirectRoot 'bin'),
    (Join-Path $vcpkgDirectRoot 'debug\bin'),
    $env:Path
)
$env:Path = $pathParts -join ';'

Push-Location $repositoryRoot
try {
    & $exe.FullName $vocab $settings $svo $outputPath ([string]$MaxFrames) $imageScaleText ([string]$SampleCount)
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
