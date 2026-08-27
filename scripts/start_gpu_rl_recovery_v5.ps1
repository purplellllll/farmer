param(
    [int]$Iterations = 600,
    [string]$Config = "configs/rl/local_4060_recovery_v5.json",
    [string]$BcCheckpoint = "checkpoints/starter_bc_v5.pt",
    [string]$Resume = "",
    [string]$OutputDir = "artifacts/local-4060-recovery-v5"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv/Scripts/python.exe"
$configPath = (Resolve-Path (Join-Path $repoRoot $Config)).Path
$resolvedOutput = [IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDir))
$logDir = Join-Path $resolvedOutput "process-logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:PYTHONPATH = Join-Path $repoRoot "src"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$arguments = @(
    "-m", "farmer_rl.train", "native-self-play",
    "--config", $configPath,
    "--iterations", "$Iterations",
    "--output", $resolvedOutput
)
$bcPath = $null
$resumePath = $null
if ($Resume) {
    $resumePath = (Resolve-Path (Join-Path $repoRoot $Resume)).Path
    $arguments += @("--resume", $resumePath)
    # Older PPO checkpoints do not contain the frozen BC anchor. Passing the
    # original BC checkpoint makes those resumes backward compatible; newer
    # checkpoints are self-contained and simply ignore this fallback.
    $bcPath = (Resolve-Path (Join-Path $repoRoot $BcCheckpoint)).Path
    $arguments += @("--bc-checkpoint", $bcPath)
} else {
    $bcPath = (Resolve-Path (Join-Path $repoRoot $BcCheckpoint)).Path
    $arguments += @("--bc-checkpoint", $bcPath)
}
$logStem = if ($Resume) {
    "resume_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
} else {
    "initial"
}
$stdout = Join-Path $logDir ("{0}.stdout.log" -f $logStem)
$stderr = Join-Path $logDir ("{0}.stderr.log" -f $logStem)
$launcher = Start-Process -FilePath $python -ArgumentList $arguments -PassThru -WindowStyle Hidden `
    -WorkingDirectory $repoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$launcher.PriorityClass = "BelowNormal"
Start-Sleep -Milliseconds 750
$worker = Get-CimInstance Win32_Process |
    Where-Object { $_.ParentProcessId -eq $launcher.Id -and $_.Name -eq "python.exe" } |
    Select-Object -First 1
if ($worker) {
    (Get-Process -Id $worker.ProcessId).PriorityClass = "BelowNormal"
}
$manifest = [ordered]@{
    schema_version = "farmer-gpu-rl-process/v3"
    launcher_pid = $launcher.Id
    worker_pid = if ($worker) { $worker.ProcessId } else { $null }
    device = "cuda"
    priority = "BelowNormal"
    iterations = $Iterations
    config = $configPath
    bc_checkpoint = $bcPath
    resume_checkpoint = $resumePath
    output_dir = $resolvedOutput
    stdout = $stdout
    stderr = $stderr
    started_at = (Get-Date).ToString("o")
}
$manifestPath = Join-Path $resolvedOutput "process.json"
$manifest | ConvertTo-Json | Set-Content -Encoding UTF8 $manifestPath
$manifest | ConvertTo-Json
