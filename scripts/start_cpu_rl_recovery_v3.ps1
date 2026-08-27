param(
    [int]$Iterations = 50,
    [int]$SegmentIterations = 1,
    [double]$MaxWorkerPrivateGiB = 4.0,
    [string]$Config = "configs/rl/cpu_recovery_v3.json",
    [string]$BcCheckpoint = "checkpoints/starter_bc_v5.pt",
    [string]$OutputDir = "artifacts/cpu-rl-recovery-v3",
    [string]$ResumeCheckpoint = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $PSScriptRoot "run_cpu_rl_recovery_segments.ps1"
$configPath = (Resolve-Path (Join-Path $repoRoot $Config)).Path
$bcPath = (Resolve-Path (Join-Path $repoRoot $BcCheckpoint)).Path
$resolvedOutput = [IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDir))
$resumePath = ""
if ($ResumeCheckpoint) {
    $resumePath = (Resolve-Path (Join-Path $repoRoot $ResumeCheckpoint)).Path
}
$logDir = Join-Path $resolvedOutput "process-logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$pwsh = (Get-Process -Id $PID).Path
$arguments = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $runner,
    "-Iterations", "$Iterations",
    "-SegmentIterations", "$SegmentIterations",
    "-MaxWorkerPrivateGiB", "$MaxWorkerPrivateGiB",
    "-Config", $configPath,
    "-BcCheckpoint", $bcPath,
    "-OutputDir", $resolvedOutput
)
if ($resumePath) {
    $arguments += @("-ResumeCheckpoint", $resumePath)
}
$stdout = Join-Path $logDir "supervisor.stdout.log"
$stderr = Join-Path $logDir "supervisor.stderr.log"
$statePath = Join-Path $resolvedOutput "process.json"
$supervisor = Start-Process -FilePath $pwsh -ArgumentList $arguments -PassThru -WindowStyle Hidden `
    -WorkingDirectory $repoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$supervisor.PriorityClass = "BelowNormal"

if (-not (Test-Path -LiteralPath $statePath)) {
    [ordered]@{
        schema_version = "farmer-cpu-rl-process/v2"
        status = "starting"
        supervisor_pid = $supervisor.Id
        priority = "BelowNormal"
        cpu_threads = 4
        target_iterations = $Iterations
        segment_iterations = $SegmentIterations
        max_worker_private_gib = $MaxWorkerPrivateGiB
        config = $configPath
        bc_checkpoint = $bcPath
        output_dir = $resolvedOutput
        stdout = $stdout
        stderr = $stderr
        started_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -Encoding UTF8 $statePath
}

Get-Content -LiteralPath $statePath -Raw
