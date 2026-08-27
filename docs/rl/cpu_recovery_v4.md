# CPU PPO recovery v4

CPU recovery v4 resumes the successful first `v3b` checkpoint into the new,
independent `artifacts/cpu-rl-recovery-v4` directory. It never overwrites v3b.

The v3b worker had completed its first segment and saved its checkpoint. The
supervisor then treated a cached null `Process.ExitCode` as an error. v4 calls
`Refresh()` after `HasExited`, reads the refreshed exit code, and only fails for
a null or nonzero value.

Each worker handles one segmented PPO iteration and is restarted by the
supervisor. The worker is limited to four CPU threads and stopped if its private
memory exceeds 4 GiB. A complete 144-step environment episode yields 143 action
records, so `train_batch_steps` is 143: this makes a segment one complete game
instead of inadvertently collecting two games.

The supervisor writes an immutable copy of the source config as
`runtime_config.json` in the artifact directory. It examines consecutive
post-intervention windows of three completed iterations. A collapse requires all
of: mean learner win rate at most 2%, mean score difference at most -100, and
either at least two KL early-stops, mean KL above 1.5 times target, or mean
entropy below 0.03. A one-game result cannot independently trigger a reward or
policy change.

At most two interventions are allowed. Each one preserves a before-config copy
and appends evidence plus before/after values to `collapse_interventions.jsonl`.
The following resumed worker uses: lower learning rate and clip range, one PPO
epoch, tighter target KL, higher entropy coefficient, a stronger bounded final
score margin reward, and at least 90% scripted-baseline opponents. The checkpoint
keeps optimizer state, while native PPO reapplies the current runtime learning
rate on resume. This was the historical v4 policy. The shared supervisor now
follows the v5 behavior documented in `cpu_recovery_v5.md`: after two failed
interventions it changes reward stage and continues from the latest checkpoint
instead of automatically pausing.

Launch:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_cpu_rl_recovery_v4.ps1 `
  -Iterations 50 -SegmentIterations 1 -MaxWorkerPrivateGiB 4.0
```

## Recorded v4 result

The resumed run completed checkpoints 2--7. The initial three-iteration
window was five losses with a mean score difference of -227.7 and mean entropy
0.0223, so intervention 1 was applied. It made the update stable and shortened
single-game segments to about 93 seconds, but the following three iterations
were again all losses, with mean score difference -100 and mean entropy 0.0228.
Intervention 2 was recorded, but the CPU job was then deliberately paused before
spending a third evaluation window. The preserved terminal checkpoint is
`artifacts/cpu-rl-recovery-v4/checkpoints/iteration_000007.pt`; exact evidence is
in the artifact directory's `collapse_interventions.jsonl` and
`collapse_conclusion.json`.
