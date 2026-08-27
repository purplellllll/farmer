param(
    [Parameter(Mandatory = $true)][int]$Iterations,
    [Parameter(Mandatory = $true)][int]$SegmentIterations,
    [Parameter(Mandatory = $true)][double]$MaxWorkerPrivateGiB,
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$BcCheckpoint,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$ResumeCheckpoint = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv/Scripts/python.exe"
$sourceConfigPath = (Resolve-Path $Config).Path
$bcPath = (Resolve-Path $BcCheckpoint).Path
$resolvedOutput = [IO.Path]::GetFullPath($OutputDir)
$logDir = Join-Path $resolvedOutput "process-logs"
$statePath = Join-Path $resolvedOutput "process.json"
$runtimeConfigPath = Join-Path $resolvedOutput "runtime_config.json"
$interventionPath = Join-Path $resolvedOutput "collapse_interventions.jsonl"
$rewardRedesignPath = Join-Path $resolvedOutput "reward_redesigns.jsonl"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (-not (Test-Path -LiteralPath $runtimeConfigPath)) {
    Copy-Item -LiteralPath $sourceConfigPath -Destination $runtimeConfigPath
}
# Existing recovery directories keep their tuned runtime config.  Merge only
# newly introduced reward-redesign fields so a previously paused run can
# continue from its latest checkpoint without discarding prior interventions.
$sourceConfig = Get-Content -LiteralPath $sourceConfigPath -Raw | ConvertFrom-Json
$runtimeConfig = Get-Content -LiteralPath $runtimeConfigPath -Raw | ConvertFrom-Json
$runtimeChanged = $false
if ($null -eq $runtimeConfig.cpu_recovery.reward_redesign) {
    $runtimeConfig.cpu_recovery | Add-Member -NotePropertyName reward_redesign -NotePropertyValue $sourceConfig.cpu_recovery.reward_redesign -Force
    $runtimeChanged = $true
}
if ($null -eq $runtimeConfig.training.shaping_profile) {
    $runtimeConfig.training | Add-Member -NotePropertyName shaping_profile -NotePropertyValue $sourceConfig.training.shaping_profile -Force
    $runtimeChanged = $true
}
if ($null -eq $runtimeConfig.training.shaping_scale) {
    $runtimeConfig.training | Add-Member -NotePropertyName shaping_scale -NotePropertyValue $sourceConfig.training.shaping_scale -Force
    $runtimeChanged = $true
}
if ($runtimeChanged) {
    $runtimeConfig | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $runtimeConfigPath
}
if ($ResumeCheckpoint) {
    $initialResumePath = (Resolve-Path $ResumeCheckpoint).Path
} else {
    $initialResumePath = $null
}

$env:PYTHONPATH = Join-Path $repoRoot "src"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"

$completed = 0
$interventionCount = 0
$rewardRedesignCount = 0
$lastInterventionIteration = 0
if (Test-Path -LiteralPath $statePath) {
    $existingState = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ($null -ne $existingState.completed_iterations) {
        $completed = [int]$existingState.completed_iterations
    } elseif ($null -ne $existingState.completed_before_segment) {
        $completed = [int]$existingState.completed_before_segment
    }
    if ($null -ne $existingState.intervention_count) {
        $interventionCount = [int]$existingState.intervention_count
    }
    if ($null -ne $existingState.reward_redesign_count) {
        $rewardRedesignCount = [int]$existingState.reward_redesign_count
    }
    if ($null -ne $existingState.last_intervention_iteration) {
        $lastInterventionIteration = [int]$existingState.last_intervention_iteration
    }
}

function Get-RecentMetricWindow {
    param([int]$Count, [int]$AfterIteration)
    $metricsPath = Join-Path $resolvedOutput "metrics.jsonl"
    if (-not (Test-Path -LiteralPath $metricsPath)) { return @() }
    $metrics = @(
        Get-Content -LiteralPath $metricsPath |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            ForEach-Object { $_ | ConvertFrom-Json } |
            Where-Object { [int]$_.iteration -gt $AfterIteration } |
            Select-Object -Last $Count
    )
    return $metrics
}

function Test-PolicyCollapse {
    param([object[]]$Metrics)
    $config = Get-Content -LiteralPath $runtimeConfigPath -Raw | ConvertFrom-Json
    $monitor = $config.cpu_recovery.collapse_monitor
    $window = [int]$monitor.window_iterations
    if ($Metrics.Count -lt $window) {
        return [pscustomobject]@{ detected = $false; reason = "insufficient_window"; metrics = $Metrics }
    }
    $winRate = ($Metrics | Measure-Object -Property learner_win_rate -Average).Average
    $meanScoreDifference = ($Metrics | Measure-Object -Property mean_score_difference -Average).Average
    $meanKl = ($Metrics | Measure-Object -Property kl -Average).Average
    $meanEntropy = ($Metrics | Measure-Object -Property entropy -Average).Average
    $earlyStops = @($Metrics | Where-Object { $_.kl_early_stop }).Count
    $targetKl = [double]$config.native.target_kl
    $detected = (
        $winRate -le [double]$monitor.max_window_win_rate -and
        $meanScoreDifference -le [double]$monitor.max_mean_score_difference -and
        (
            $earlyStops -ge [int]$monitor.min_kl_early_stops -or
            $meanKl -ge ($targetKl * [double]$monitor.kl_multiple) -or
            $meanEntropy -le [double]$monitor.min_mean_entropy
        )
    )
    return [pscustomobject]@{
        detected = $detected
        reason = if ($detected) { "window_win_score_and_update_signal" } else { "thresholds_not_all_met" }
        metrics = $Metrics
        mean_win_rate = $winRate
        mean_score_difference = $meanScoreDifference
        mean_kl = $meanKl
        mean_entropy = $meanEntropy
        kl_early_stops = $earlyStops
    }
}

function Invoke-CollapseIntervention {
    param([int]$Number, [object]$Assessment)
    $config = Get-Content -LiteralPath $runtimeConfigPath -Raw | ConvertFrom-Json
    $recovery = $config.cpu_recovery.recovery
    $before = [ordered]@{
        lr = [double]$config.training.lr
        clip_param = [double]$config.training.clip_param
        terminal_score_coeff = [double]$config.training.terminal_score_coeff
        update_epochs = [int]$config.native.update_epochs
        entropy_coeff = [double]$config.native.entropy_coeff
        target_kl = [double]$config.native.target_kl
        scripted_opponent_probability = [double]$config.native.scripted_opponent_probability
    }
    $interventionBackup = "runtime_config.before_reward_{0:D2}_intervention_{1:D2}.json" -f $rewardRedesignCount, $Number
    Copy-Item -LiteralPath $runtimeConfigPath -Destination (Join-Path $resolvedOutput $interventionBackup)
    $config.training.lr = [math]::Max([double]$recovery.min_lr, [double]$config.training.lr * [double]$recovery.lr_multiplier)
    $config.training.clip_param = [math]::Max([double]$recovery.min_clip_param, [double]$config.training.clip_param * [double]$recovery.clip_multiplier)
    $config.training.terminal_score_coeff = [math]::Min([double]$recovery.max_terminal_score_coeff, [double]$config.training.terminal_score_coeff + [double]$recovery.terminal_score_increment)
    $config.native.update_epochs = [int]$recovery.update_epochs
    $config.native.entropy_coeff = [math]::Min([double]$recovery.max_entropy_coeff, [double]$config.native.entropy_coeff * [double]$recovery.entropy_multiplier)
    $config.native.target_kl = [math]::Min([double]$config.native.target_kl, [double]$recovery.target_kl)
    $config.native.scripted_opponent_probability = [math]::Max([double]$config.native.scripted_opponent_probability, [double]$recovery.min_scripted_opponent_probability)
    $config | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $runtimeConfigPath
    $after = [ordered]@{
        lr = [double]$config.training.lr
        clip_param = [double]$config.training.clip_param
        terminal_score_coeff = [double]$config.training.terminal_score_coeff
        update_epochs = [int]$config.native.update_epochs
        entropy_coeff = [double]$config.native.entropy_coeff
        target_kl = [double]$config.native.target_kl
        scripted_opponent_probability = [double]$config.native.scripted_opponent_probability
    }
    [ordered]@{
        timestamp = (Get-Date).ToString("o")
        intervention = $Number
        reward_stage = $rewardRedesignCount
        reason = $Assessment.reason
        window_metrics = @($Assessment.metrics)
        aggregate = [ordered]@{
            mean_win_rate = $Assessment.mean_win_rate
            mean_score_difference = $Assessment.mean_score_difference
            mean_kl = $Assessment.mean_kl
            mean_entropy = $Assessment.mean_entropy
            kl_early_stops = $Assessment.kl_early_stops
        }
        before = $before
        after = $after
    } | ConvertTo-Json -Depth 10 -Compress | Add-Content -Encoding UTF8 $interventionPath
}

function Invoke-RewardRedesign {
    param([int]$Number, [object]$Assessment)
    $config = Get-Content -LiteralPath $runtimeConfigPath -Raw | ConvertFrom-Json
    $profiles = @($config.cpu_recovery.reward_redesign.profiles)
    if ($profiles.Count -eq 0) {
        throw "cpu_recovery.reward_redesign.profiles must contain at least one profile"
    }
    $profile = $profiles[($Number - 1) % $profiles.Count]
    $before = [ordered]@{
        shaping_profile = $config.training.shaping_profile
        shaping_scale = $config.training.shaping_scale
        lr = [double]$config.training.lr
        clip_param = [double]$config.training.clip_param
        terminal_score_coeff = [double]$config.training.terminal_score_coeff
        entropy_coeff = [double]$config.native.entropy_coeff
        target_kl = [double]$config.native.target_kl
        scripted_opponent_probability = [double]$config.native.scripted_opponent_probability
    }
    Copy-Item -LiteralPath $runtimeConfigPath -Destination (Join-Path $resolvedOutput ("runtime_config.before_reward_redesign_{0:D2}.json" -f $Number))
    $config.training | Add-Member -NotePropertyName shaping_profile -NotePropertyValue ([string]$profile.shaping_profile) -Force
    $config.training | Add-Member -NotePropertyName shaping_scale -NotePropertyValue ([double]$profile.shaping_scale) -Force
    $config.training.lr = [double]$profile.lr
    $config.training.clip_param = [double]$profile.clip_param
    $config.training.terminal_score_coeff = [double]$profile.terminal_score_coeff
    $config.native.update_epochs = [int]$profile.update_epochs
    $config.native.entropy_coeff = [double]$profile.entropy_coeff
    $config.native.target_kl = [double]$profile.target_kl
    $config.native.scripted_opponent_probability = [double]$profile.scripted_opponent_probability
    $config | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $runtimeConfigPath
    $after = [ordered]@{
        shaping_profile = [string]$config.training.shaping_profile
        shaping_scale = [double]$config.training.shaping_scale
        lr = [double]$config.training.lr
        clip_param = [double]$config.training.clip_param
        terminal_score_coeff = [double]$config.training.terminal_score_coeff
        entropy_coeff = [double]$config.native.entropy_coeff
        target_kl = [double]$config.native.target_kl
        scripted_opponent_probability = [double]$config.native.scripted_opponent_probability
    }
    [ordered]@{
        timestamp = (Get-Date).ToString("o")
        reward_redesign = $Number
        profile_index = (($Number - 1) % $profiles.Count)
        reason = $Assessment.reason
        window_metrics = @($Assessment.metrics)
        aggregate = [ordered]@{
            mean_win_rate = $Assessment.mean_win_rate
            mean_score_difference = $Assessment.mean_score_difference
            mean_kl = $Assessment.mean_kl
            mean_entropy = $Assessment.mean_entropy
            kl_early_stops = $Assessment.kl_early_stops
        }
        before = $before
        after = $after
        action = "restart_from_latest_checkpoint"
    } | ConvertTo-Json -Depth 10 -Compress | Add-Content -Encoding UTF8 $rewardRedesignPath
}

function Invoke-CollapseResponse {
    param([object]$Assessment, [object[]]$Window)
    $config = Get-Content -LiteralPath $runtimeConfigPath -Raw | ConvertFrom-Json
    $script:lastInterventionIteration = [int]$Window[-1].iteration
    $maxInterventions = [int]$config.cpu_recovery.max_interventions
    if ($script:interventionCount -lt $maxInterventions) {
        $script:interventionCount += 1
        Invoke-CollapseIntervention -Number $script:interventionCount -Assessment $Assessment
        return
    }
    # Reaching the intervention ceiling is a reward-stage transition, never an
    # automatic pause.  The next segment process loads the latest checkpoint
    # with the redesigned shaping profile and reset intervention budget.
    $script:rewardRedesignCount += 1
    Invoke-RewardRedesign -Number $script:rewardRedesignCount -Assessment $Assessment
    $script:interventionCount = 0
}

function Invoke-PreflightCollapseAssessment {
    $config = Get-Content -LiteralPath $runtimeConfigPath -Raw | ConvertFrom-Json
    $window = Get-RecentMetricWindow -Count ([int]$config.cpu_recovery.collapse_monitor.window_iterations) -AfterIteration $lastInterventionIteration
    $assessment = Test-PolicyCollapse -Metrics $window
    if (-not $assessment.detected) { return }
    Invoke-CollapseResponse -Assessment $assessment -Window $window
}

Invoke-PreflightCollapseAssessment

while ($completed -lt $Iterations) {
    $segment = [Math]::Min($SegmentIterations, $Iterations - $completed)
    $latestPath = Join-Path $resolvedOutput "latest.json"
    $resumePath = $null
    if (Test-Path -LiteralPath $latestPath) {
        $latest = Get-Content -LiteralPath $latestPath -Raw | ConvertFrom-Json
        $resumePath = (Resolve-Path $latest.checkpoint).Path
    } elseif ($initialResumePath) {
        $resumePath = $initialResumePath
    }

    $arguments = @(
        "-m", "farmer_rl.train", "native-self-play",
        "--config", $runtimeConfigPath,
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
        schema_version = "farmer-cpu-rl-process/v3"
        status = "running"
        supervisor_pid = $PID
        launcher_pid = $launcher.Id
        worker_pid = if ($worker) { $worker.ProcessId } else { $null }
        priority = "BelowNormal"
        cpu_threads = 4
        max_worker_private_gib = $MaxWorkerPrivateGiB
        target_iterations = $Iterations
        completed_before_segment = $completed
        completed_iterations = $completed
        segment_iterations = $segment
        source_config = $sourceConfigPath
        config = $runtimeConfigPath
        bc_checkpoint = if ($resumePath) { $null } else { $bcPath }
        resume_checkpoint = $resumePath
        intervention_count = $interventionCount
        reward_redesign_count = $rewardRedesignCount
        last_intervention_iteration = $lastInterventionIteration
        shaping_profile = (Get-Content -LiteralPath $runtimeConfigPath -Raw | ConvertFrom-Json).training.shaping_profile
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
    # The Process object caches ExitCode.  Refresh after HasExited becomes true,
    # otherwise a normal zero exit can appear as $null and be treated as failure.
    $launcher.Refresh()
    $exitCode = $launcher.ExitCode
    if ($memoryGuardTriggered) {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        $state.status = "failed_memory_guard"
        $state | Add-Member -NotePropertyName max_worker_private_gib -NotePropertyValue $MaxWorkerPrivateGiB -Force
        $state | Add-Member -NotePropertyName updated_at -NotePropertyValue (Get-Date).ToString("o") -Force
        $state | ConvertTo-Json | Set-Content -Encoding UTF8 $statePath
        exit 137
    }
    if ($null -eq $exitCode -or $exitCode -ne 0) {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        $state.status = "failed"
        $state | Add-Member -NotePropertyName exit_code -NotePropertyValue $exitCode -Force
        $state | Add-Member -NotePropertyName updated_at -NotePropertyValue (Get-Date).ToString("o") -Force
        $state | ConvertTo-Json | Set-Content -Encoding UTF8 $statePath
        if ($null -eq $exitCode) { exit 1 }
        exit $exitCode
    }
    $completed += $segment

    $runtimeConfig = Get-Content -LiteralPath $runtimeConfigPath -Raw | ConvertFrom-Json
    $monitor = $runtimeConfig.cpu_recovery.collapse_monitor
    $window = Get-RecentMetricWindow -Count ([int]$monitor.window_iterations) -AfterIteration $lastInterventionIteration
    $assessment = Test-PolicyCollapse -Metrics $window
    if ($assessment.detected) {
        Invoke-CollapseResponse -Assessment $assessment -Window $window
    }
}

$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
$state.status = "completed"
$state | Add-Member -NotePropertyName completed_iterations -NotePropertyValue $Iterations -Force
$state | Add-Member -NotePropertyName intervention_count -NotePropertyValue $interventionCount -Force
$state | Add-Member -NotePropertyName reward_redesign_count -NotePropertyValue $rewardRedesignCount -Force
$state | Add-Member -NotePropertyName last_intervention_iteration -NotePropertyValue $lastInterventionIteration -Force
$state | Add-Member -NotePropertyName shaping_profile -NotePropertyValue ((Get-Content -LiteralPath $runtimeConfigPath -Raw | ConvertFrom-Json).training.shaping_profile) -Force
$state | Add-Member -NotePropertyName updated_at -NotePropertyValue (Get-Date).ToString("o") -Force
$state | ConvertTo-Json | Set-Content -Encoding UTF8 $statePath
