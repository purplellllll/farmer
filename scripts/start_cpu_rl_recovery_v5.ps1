param(
    [int]$Iterations = 50,
    [int]$SegmentIterations = 1,
    [double]$MaxWorkerPrivateGiB = 4.0,
    [string]$ResumeCheckpoint = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $ResumeCheckpoint) {
    $gpuLatestPath = Join-Path $repoRoot "artifacts/local-4060-recovery-v5/latest.json"
    $gpuLatest = Get-Content -LiteralPath $gpuLatestPath -Raw | ConvertFrom-Json
    $ResumeCheckpoint = $gpuLatest.checkpoint
}

$launcher = Join-Path $PSScriptRoot "start_cpu_rl_recovery_v3.ps1"
& $launcher `
    -Iterations $Iterations `
    -SegmentIterations $SegmentIterations `
    -MaxWorkerPrivateGiB $MaxWorkerPrivateGiB `
    -Config "configs/rl/cpu_recovery_v5.json" `
    -OutputDir "artifacts/cpu-rl-recovery-v5" `
    -ResumeCheckpoint $ResumeCheckpoint
