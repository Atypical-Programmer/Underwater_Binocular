[CmdletBinding()]
param(
    [ValidateSet("GEN_1", "GEN_3", "BOTH")]
    [string]$Mode = "GEN_1",
    [string]$Dataset = "configs/datasets/20260802_150233.yaml",
    [int]$StartFrame = 0,
    [int]$MaxFrames = 0,
    [int]$EndFrame,
    [string]$Output,
    [string]$Svo,
    [switch]$AreaMemory
)

$cliArgs = @(
    "tracking", "zed",
    "--dataset", $Dataset,
    "--mode", $Mode,
    "--start-frame", $StartFrame,
    "--max-frames", $MaxFrames
)

if ($PSBoundParameters.ContainsKey("EndFrame")) {
    $cliArgs += @("--end-frame", $EndFrame)
}
if ($PSBoundParameters.ContainsKey("Output")) {
    $cliArgs += @("--output", $Output)
}
if ($PSBoundParameters.ContainsKey("Svo")) {
    $cliArgs += @("--svo", $Svo)
}
if ($AreaMemory) {
    $cliArgs += "--area-memory"
}

& underwater @cliArgs
exit $LASTEXITCODE
