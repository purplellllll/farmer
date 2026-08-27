param(
    [int]$Iterations = 50,
    [int]$SegmentIterations = 1,
    [double]$MaxWorkerPrivateGiB = 4.0,
    [string]$ResumeCheckpoint = "artifacts/cpu-rl-recovery-v3b/checkpoints/iteration_000001.pt"
)

$launcher = Join-Path $PSScriptRoot "start_cpu_rl_recovery_v3.ps1"
& $launcher `
    -Iterations $Iterations `
    -SegmentIterations $SegmentIterations `
    -MaxWorkerPrivateGiB $MaxWorkerPrivateGiB `
    -Config "configs/rl/cpu_recovery_v4.json" `
    -OutputDir "artifacts/cpu-rl-recovery-v4" `
    -ResumeCheckpoint $ResumeCheckpoint
