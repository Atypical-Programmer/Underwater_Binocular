param(
    [string]$ZedSdkRoot = $env:ZED_SDK_ROOT_DIR,
    [string]$CudaRoot = $env:CUDA_TOOLKIT_ROOT_DIR,
    [string]$CMakeExecutable = 'cmake',
    [string]$Generator = 'Visual Studio 17 2022'
)

$ErrorActionPreference = 'Stop'
$integrationRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = (Resolve-Path (Join-Path $integrationRoot '..\..')).Path
$toolchain = Join-Path $integrationRoot 'vcpkg\scripts\buildsystems\vcpkg.cmake'
$build = Join-Path $integrationRoot 'build'

if ([string]::IsNullOrWhiteSpace($ZedSdkRoot) -or -not (Test-Path -LiteralPath $ZedSdkRoot)) {
    throw 'Set ZED_SDK_ROOT_DIR to the installed ZED SDK root.'
}
if ([string]::IsNullOrWhiteSpace($CudaRoot) -or -not (Test-Path -LiteralPath $CudaRoot)) {
    throw 'Set CUDA_TOOLKIT_ROOT_DIR to the installed CUDA root.'
}
if (-not (Get-Command $CMakeExecutable -ErrorAction SilentlyContinue)) {
    throw "CMake was not found: $CMakeExecutable"
}
if (-not (Test-Path -LiteralPath $toolchain)) {
    throw "vcpkg toolchain was not found: $toolchain"
}

& $CMakeExecutable '-S' $integrationRoot '-B' $build '-G' $Generator '-A' 'x64' `
    "-DCMAKE_TOOLCHAIN_FILE=$toolchain" '-DVCPKG_TARGET_TRIPLET=x64-windows' `
    "-DZED_SDK_ROOT_DIR=$ZedSdkRoot" "-DCUDA_TOOLKIT_ROOT_DIR=$CudaRoot" `
    "-DUNDERWATER_REPOSITORY_ROOT=$repositoryRoot"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $CMakeExecutable '--build' $build '--config' 'Release' '--target' 'svo2_stereo' '--parallel' '2'
exit $LASTEXITCODE
