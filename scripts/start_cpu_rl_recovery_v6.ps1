param(
    [int]$Iterations = 80,
    [int]$SegmentIterations = 1,
    [double]$MaxWorkerPrivateGiB = 2.10,
    [string]$ResumeCheckpoint = ""
)

$ErrorActionPreference = "Stop"
$launcher = Join-Path $PSScriptRoot "start_cpu_rl_recovery_v3.ps1"
$arguments = @{
    Iterations = $Iterations
    SegmentIterations = $SegmentIterations
    MaxWorkerPrivateGiB = $MaxWorkerPrivateGiB
    Config = "configs/rl/cpu_recovery_v6.json"
    OutputDir = "artifacts/cpu-rl-recovery-v6"
}
if ($ResumeCheckpoint) {
    $arguments.ResumeCheckpoint = $ResumeCheckpoint
}
& $launcher @arguments
