param(
    [Parameter(Mandatory = $true)][int]$Iterations,
    [Parameter(Mandatory = $true)][int]$SegmentIterations,
    [Parameter(Mandatory = $true)][double]$MaxWorkerPrivateGiB,
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$BcCheckpoint,
    [Parameter(Mandatory = $true)][string]$OutputDir
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv/Scripts/python.exe"
$configPath = (Resolve-Path $Config).Path
$bcPath = (Resolve-Path $BcCheckpoint).Path
$resolvedOutput = [IO.Path]::GetFullPath($OutputDir)
$logDir = Join-Path $resolvedOutput "process-logs"
$statePath = Join-Path $resolvedOutput "process.json"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:PYTHONPATH = Join-Path $repoRoot "src"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"

$completed = 0
while ($completed -lt $Iterations) {
    $segment = [Math]::Min($SegmentIterations, $Iterations - $completed)
    $latestPath = Join-Path $resolvedOutput "latest.json"
    $resumePath = $null
    if (Test-Path -LiteralPath $latestPath) {
        $latest = Get-Content -LiteralPath $latestPath -Raw | ConvertFrom-Json
        $resumePath = (Resolve-Path $latest.checkpoint).Path
    }

    $arguments = @(
        "-m", "farmer_rl.train", "native-self-play",
        "--config", $configPath,
        "--iterations", "$segment",
        "--output", $resolvedOutput
    )
    if ($resumePath) {
        $arguments += @("--resume", $resumePath)
    } else {
        $arguments += @("--bc-checkpoint", $bcPath)
    }

    $segmentNumber = [int]($completed / $SegmentIterations) + 1
    $stdout = Join-Path $logDir ("segment_{0:D3}.stdout.log" -f $segmentNumber)
    $stderr = Join-Path $logDir ("segment_{0:D3}.stderr.log" -f $segmentNumber)
    $launcher = Start-Process -FilePath $python -ArgumentList $arguments -PassThru -WindowStyle Hidden `
        -WorkingDirectory $repoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $launcher.PriorityClass = "BelowNormal"
    Start-Sleep -Milliseconds 750
    $worker = Get-CimInstance Win32_Process |
        Where-Object { $_.ParentProcessId -eq $launcher.Id -and $_.Name -eq "python.exe" } |
        Select-Object -First 1
    if ($worker) {
        try { (Get-Process -Id $worker.ProcessId).PriorityClass = "BelowNormal" } catch {}
    }

    [ordered]@{
        schema_version = "farmer-cpu-rl-process/v2"
        status = "running"
        supervisor_pid = $PID
        launcher_pid = $launcher.Id
        worker_pid = if ($worker) { $worker.ProcessId } else { $null }
        priority = "BelowNormal"
        cpu_threads = 4
        max_worker_private_gib = $MaxWorkerPrivateGiB
        target_iterations = $Iterations
        completed_before_segment = $completed
        segment_iterations = $segment
        config = $configPath
        bc_checkpoint = if ($resumePath) { $null } else { $bcPath }
        resume_checkpoint = $resumePath
        output_dir = $resolvedOutput
        stdout = $stdout
        stderr = $stderr
        updated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -Encoding UTF8 $statePath

    $memoryGuardTriggered = $false
    while (-not $launcher.HasExited) {
        Start-Sleep -Seconds 5
        $launcher.Refresh()
        if ($worker) {
            $workerProcess = Get-Process -Id $worker.ProcessId -ErrorAction SilentlyContinue
            if ($workerProcess -and $workerProcess.PrivateMemorySize64 -gt $MaxWorkerPrivateGiB * 1GB) {
                $memoryGuardTriggered = $true
                Stop-Process -Id $worker.ProcessId -Force -ErrorAction SilentlyContinue
                Stop-Process -Id $launcher.Id -Force -ErrorAction SilentlyContinue
                break
            }
        }
    }
    if ($memoryGuardTriggered) {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        $state.status = "failed_memory_guard"
        $state | Add-Member -NotePropertyName max_worker_private_gib -NotePropertyValue $MaxWorkerPrivateGiB -Force
        $state | Add-Member -NotePropertyName updated_at -NotePropertyValue (Get-Date).ToString("o") -Force
        $state | ConvertTo-Json | Set-Content -Encoding UTF8 $statePath
        exit 137
    }
    if ($launcher.ExitCode -ne 0) {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        $state.status = "failed"
        $state | Add-Member -NotePropertyName exit_code -NotePropertyValue $launcher.ExitCode -Force
        $state | Add-Member -NotePropertyName updated_at -NotePropertyValue (Get-Date).ToString("o") -Force
        $state | ConvertTo-Json | Set-Content -Encoding UTF8 $statePath
        exit $launcher.ExitCode
    }
    $completed += $segment
}

$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
$state.status = "completed"
$state | Add-Member -NotePropertyName completed_iterations -NotePropertyValue $Iterations -Force
$state | Add-Member -NotePropertyName updated_at -NotePropertyValue (Get-Date).ToString("o") -Force
$state | ConvertTo-Json | Set-Content -Encoding UTF8 $statePath
