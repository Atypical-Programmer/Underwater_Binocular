$ErrorActionPreference = 'Stop'

$slamRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$cmake = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$toolchain = Join-Path $slamRoot 'vcpkg\scripts\buildsystems\vcpkg.cmake'
$build = Join-Path $slamRoot 'build'

if (-not (Test-Path -LiteralPath $cmake)) {
    throw "CMake was not found: $cmake"
}
if (-not (Test-Path -LiteralPath $toolchain)) {
    throw "vcpkg toolchain was not found: $toolchain"
}

& $cmake '-S' $slamRoot '-B' $build '-G' 'Visual Studio 16 2019' '-A' 'x64' `
    "-DCMAKE_TOOLCHAIN_FILE=$toolchain" '-DVCPKG_TARGET_TRIPLET=x64-windows'
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $cmake '--build' $build '--config' 'Release' '--target' 'svo2_stereo' '--parallel' '2'
exit $LASTEXITCODE
