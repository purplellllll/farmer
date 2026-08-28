# Farmer project handoff for the next Codex session

Last refreshed: **2026-08-28 10:15 Asia/Shanghai**

Workspace: `F:\Kaggle\farm\farmer`

Branch: `codex/eval-cpu-rl-s6e7`

This document is the source of truth for continuing the current Kaggriculture
work in a new conversation.  Runtime values below are a snapshot: always read
the live process files and metric tails before acting.

## 1. User objective and current direction

The user wants a competitive Kaggriculture agent built around a large,
multi-layer residual Transformer Actor–Critic, alongside an eight-model tabular
ensemble and a reproducible local match evaluator.  The current active work is
RL recovery:

- keep the 8-layer Transformer rather than shrinking the policy;
- correct long-horizon credit assignment and policy collapse;
- keep GPU and CPU training running concurrently when system resources permit;
- detect collapse and redesign the reward instead of automatically pausing;
- judge checkpoints with paired seeds and both seats in the local evaluator;
- keep code, rationale and evaluation framework synchronized to GitHub, but do
  not commit large checkpoints, logs, replay data or unlicensed third-party
  training data.

The eight-model ensemble is not actively training in this snapshot.  Its main
analysis is in `docs/ensemble/s6e7_second_place_analysis.md` and
`docs/ensemble/model_selection.md`.  The latest documented blend does not make
all eight estimators materially active: several weights are zero, and the
research artifact is far over the deployment budget, chiefly because of
ExtraTrees.  Do not call it a valid eight-model Kaggle submission without a new
manifest, package-size gate and end-to-end submission check.

## 2. What was diagnosed before CPU v6

CPU v5 completed iterations 115–164 without NaN, engine failure or target-KL
stops, but none of its three reward profiles produced a scripted-opponent win.
Its final ten iterations contained nine scripted losses and one snapshot win,
with mean score difference about -299.1.  Entropy remained roughly
0.016–0.018 and approximate KL remained near zero.  Changing only potential
weights therefore did not repair the strategy.

Local evaluator artifacts also exposed the concrete collapse: one failed RL
checkpoint issued repeated seed purchases with duplicate market-order rate
0.90 and spent roughly 800 early cash.  The old potential treated seeds,
harvested inventory and cash too similarly, so harvesting and selling offered
almost no intermediate credit.  One 143-step game per CPU update added very
high seed/seat variance.

GPU v5 is numerically stable but strategically weak against the official
starter.  Historical GPU checkpoints occasionally recorded a single scripted
win, but recent windows are back at zero scripted wins.  Snapshot wins are not
evidence of strength because the pool still contains only one weak lineage.

## 3. CPU v6 changes

Implementation and rationale:

- `configs/rl/cpu_recovery_v6.json`
- `docs/rl/cpu_recovery_v6.md`
- `scripts/start_cpu_rl_recovery_v6.ps1`
- `scripts/run_cpu_rl_recovery_segments.ps1`
- `src/farmer_rl/model.py`
- `src/farmer_rl/native_ppo.py`

The model remains an 8-layer, `d_model=256`, 8-head residual Transformer with
6,566,977 parameters.  The critic gradient reaching the shared encoder is
scaled to 0.25, while the critic head receives its full gradient.  The value
loss coefficient is 0.2.  This is checkpoint-compatible and approximates the
policy/value objective isolation motivated by PPG without allocating a second
full Transformer.

`cashflow_cycle_v4` discounts assets by distance from realized cash:

| Component | Potential weight |
| --- | ---: |
| cash | 1.00 |
| harvested product inventory | 0.80 |
| seed inventory | 0.70 |
| carried animal | 0.70 |
| placed animal capital | 0.65 |
| planted crop capital | 0.60 |
| animal output | 0.50 |
| field crop output | 0.35 |

This makes `harvest -> shed -> sell -> cash` progressively valuable.  Shaping
still uses `c * (gamma * Phi(next) - Phi(current))` with zero terminal
potential, so it does not introduce a repeatable event-reward cycle.
`realized_cash_v5` is the more conservative fallback profile.

Important v6 PPO parameters:

- start from `checkpoints/starter_bc_v5.pt`, not collapsed CPU v5;
- `episodeSteps=144`, `train_batch_steps=286`: two complete games per update;
- seats alternate within each two-game batch;
- 2 PPO epochs, minibatch 8, learning rate `1e-6`;
- `gamma=0.999`, `gae_lambda=0.98`, clip 0.05;
- entropy coefficient 0.004, gradient clip 0.25, target KL 0.005;
- frozen BC anchor coefficient 0.5;
- 100% official scripted starter at the initial stage;
- collapse window 4 iterations; one conservative intervention, then switch
  reward profile and continue instead of auto-pausing.

The generic CPU supervisor now always passes the BC checkpoint on resume.  It
is used as a fallback for legacy PPO checkpoints that lack the frozen reference
model; self-contained new checkpoints simply use their saved reference.

Tests passed:

```text
python -m unittest tests.rl.test_model tests.rl.test_core -v
26 tests, all passed
```

## 4. Live training snapshot

### CPU v6

Process file: `artifacts/cpu-rl-recovery-v6/process.json`

Metrics: `artifacts/cpu-rl-recovery-v6/metrics.jsonl`

Latest checkpoint manifest: `artifacts/cpu-rl-recovery-v6/latest.json`

At handoff:

- status `running`, target 80 iterations;
- supervisor PID 43508;
- iteration 1 completed; iteration 2 worker PID 21148 was active;
- active profile `cashflow_cycle_v4`;
- worker memory observed around 1.78 GiB at the snapshot;
- iteration 1: 286 steps, two scripted losses, mean score difference -317;
- iteration 1 wall time 382.46 seconds, so 80 iterations are roughly 8.5
  hours under the current simultaneous GPU load;
- KL `6.88e-6`, max KL `4.94e-4`, entropy `0.01922`, no early stop;
- reference KL `5.77e-8`, confirming that the BC anchor is active.

The first launch used a 1.85 GiB worker guard and was safely stopped during
the initial AdamW allocation before any checkpoint was written.  It was
restarted with a 2.10 GiB guard and completed iteration 1.  This was a memory
safety stop, not a policy or numerical failure.

### GPU v5

Process file: `artifacts/local-4060-recovery-v5/process.json`

Metrics: `artifacts/local-4060-recovery-v5/metrics.jsonl`

Latest checkpoint manifest: `artifacts/local-4060-recovery-v5/latest.json`

At handoff:

- launcher PID 28556, worker PID 42544, both active;
- latest completed iteration 295; the current run targets iteration 600;
- iteration 295: overall win rate 0.21875, but scripted win rate 0/24;
- snapshot win rate 0.875 and mean score difference -176.47;
- last-five mean score difference about -201.28 and last-five scripted win
  rate 0;
- CUDA peak allocated memory reported by the trainer: about 1.835 GiB;
- no numerical failure or KL early stop.

Do not interpret snapshot win rate as progress until scripted and held-out
paired-seat evaluation also improve.

### Resource warning

At 10:15 the machine had only **0.93 GiB free physical RAM out of 15.78 GiB**.
Do not start another trainer, evaluator, notebook or large checkpoint load while
both jobs are alive.  The CPU supervisor already uses `BelowNormal` priority
and a 2.10 GiB worker guard.  If it hits the guard again, reduce minibatch size
from 8 to 6 or 4 and resume; do not raise the guard further without first
stopping another large workload.

## 5. First actions in the next conversation

Run these read-only checks before changing anything:

```powershell
Set-Location F:\Kaggle\farm\farmer

Get-Content artifacts/cpu-rl-recovery-v6/process.json -Raw
Get-Content artifacts/cpu-rl-recovery-v6/metrics.jsonl -Tail 8
Get-Content artifacts/cpu-rl-recovery-v6/collapse_interventions.jsonl -Tail 5 -ErrorAction SilentlyContinue
Get-Content artifacts/cpu-rl-recovery-v6/reward_redesigns.jsonl -Tail 5 -ErrorAction SilentlyContinue

Get-Content artifacts/local-4060-recovery-v5/metrics.jsonl -Tail 5

Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'cpu-rl-recovery-v6|local-4060-recovery-v5' } |
  Select-Object ProcessId, ParentProcessId, Name, CommandLine

Get-CimInstance Win32_OperatingSystem |
  Select-Object @{n='FreeGiB';e={[math]::Round($_.FreePhysicalMemory/1MB,2)}}
```

Then:

1. If both trainers are alive and logs are advancing, leave them running.
2. At CPU iteration 4, inspect the first collapse window.  A zero win rate and
   poor score can trigger the configured conservative intervention because
   entropy is already near the 0.02 threshold.
3. After at least 4–8 CPU iterations, compare mean score, scripted outcomes,
   entropy, KL, reference KL and value loss by reward stage.  One iteration is
   not enough to judge v6.
4. Do not run a full local tournament while free memory is below roughly
   2.5 GiB.  When resources are available, evaluate selected checkpoints with
   paired seeds and both seats against fixed scripted opponents, not only the
   self-play snapshot.
5. If v6 remains at zero scripted wins after the automatic
   `realized_cash_v5` stage, the next intervention should target the action
   representation/teacher data, not another arbitrary reward-weight cycle.

## 6. Recovery commands if a process has died

Only run these after verifying the corresponding process is absent.

CPU v6 resumes from its own `latest.json`; the supervisor reads completed
iterations from the existing process state:

```powershell
$cpuLatest = Get-Content artifacts/cpu-rl-recovery-v6/latest.json -Raw | ConvertFrom-Json
.\scripts\start_cpu_rl_recovery_v6.ps1 `
  -Iterations 80 `
  -SegmentIterations 1 `
  -MaxWorkerPrivateGiB 2.10 `
  -ResumeCheckpoint $cpuLatest.checkpoint
```

For GPU v5, `--iterations` means additional iterations.  Calculate only the
remaining amount up to 600:

```powershell
$gpuLatest = Get-Content artifacts/local-4060-recovery-v5/latest.json -Raw | ConvertFrom-Json
$gpuRemaining = 600 - [int]$gpuLatest.iteration
$gpuResumeRelative = Resolve-Path -Relative $gpuLatest.checkpoint
if ($gpuRemaining -gt 0) {
  .\scripts\start_gpu_rl_recovery_v5.ps1 `
    -Iterations $gpuRemaining `
    -Resume $gpuResumeRelative
}
```

Never start either command if its old worker is still alive.

## 7. Git and GitHub state

Local commit containing CPU v6: `79aad9b`

Remote GitHub commit with the same selected files:
`5cb8ad8bf977566e2602ed7ee481841e24fd6744`

Remote branch: `codex/eval-cpu-rl-s6e7`

The remote was updated through the GitHub integration because native HTTPS
push had previously failed.  The local tracking display can therefore say
`ahead 3` even though equivalent changes exist remotely under different commit
SHAs.  Do not force-push merely to make that counter disappear.  Check the
actual remote tree before publishing later work.

Training artifacts under `artifacts/` and checkpoint binaries are intentionally
not committed.  Any new source/document changes should be staged by explicit
path, committed locally, and mirrored to the existing remote branch without
creating a duplicate branch or PR.

## 8. Research basis already used

- PPO: https://arxiv.org/abs/1707.06347
- GAE: https://arxiv.org/abs/1506.02438
- Potential-based shaping:
  https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf
- PPG: https://proceedings.mlr.press/v139/cobbe21a.html
- RL with demonstrations: https://arxiv.org/abs/1709.10087
- AlphaStar league training:
  https://www.nature.com/articles/s41586-019-1724-z

The main conclusions were: use multiple minibatch epochs on a less noisy
on-policy batch; retain demonstration anchoring; attenuate policy/value
representation interference; keep shaping potential-based; and delay league
diversity until the policy can pass a fixed scripted curriculum gate.

## 9. Copy this into the new conversation

```text
请接手 F:\Kaggle\farm\farmer 项目。先完整读取
F:\Kaggle\farm\farmer\docs\HANDOFF_NEXT_SESSION.md，然后只读检查 CPU v6、GPU v5
的 process.json、metrics.jsonl、最新 checkpoint、相关 PID 和系统可用内存。若两个训练正常，
不要重启或并行启动新任务；汇报从交接时间之后新增的迭代、胜率、scripted 胜率、平均分差、
entropy、KL、reference KL、内存和任何 collapse/reward-redesign 事件。若进程已退出，先从日志
判定是正常完成、内存保护还是训练错误，再按交接文档中的恢复规则处理。不要把 snapshot 胜率
当作真实进步，后续选模必须使用本地评测框架的固定对手、paired seeds 和双席位结果。
```
