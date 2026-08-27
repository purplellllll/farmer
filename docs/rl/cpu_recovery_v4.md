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
rate on resume. After two failed interventions the run pauses as
`paused_policy_collapse` rather than restarting indefinitely.

Launch:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_cpu_rl_recovery_v4.ps1 `
  -Iterations 50 -SegmentIterations 1 -MaxWorkerPrivateGiB 4.0
```
