param(
    [int]$Iterations = 600,
    [string]$Config = "configs/rl/local_4060_recovery_v2.json",
    [string]$BcCheckpoint = "checkpoints/starter_bc_v5.pt",
    [string]$Resume = "",
    [string]$OutputDir = "artifacts/local-4060-recovery-v2"
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
} else {
    $bcPath = (Resolve-Path (Join-Path $repoRoot $BcCheckpoint)).Path
    $arguments += @("--bc-checkpoint", $bcPath)
}
$stdout = Join-Path $logDir "stdout.log"
$stderr = Join-Path $logDir "stderr.log"
$launcher = Start-Process -FilePath $python -ArgumentList $arguments -PassThru -WindowStyle Hidden `
    -WorkingDirectory $repoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr
Start-Sleep -Milliseconds 750
$worker = Get-CimInstance Win32_Process |
    Where-Object { $_.ParentProcessId -eq $launcher.Id -and $_.Name -eq "python.exe" } |
    Select-Object -First 1
$manifest = [ordered]@{
    schema_version = "farmer-gpu-rl-process/v2"
    launcher_pid = $launcher.Id
    worker_pid = if ($worker) { $worker.ProcessId } else { $null }
    device = "cuda"
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
