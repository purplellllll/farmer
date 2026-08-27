# CPU PPO recovery v5: reward redesign without auto-pause

CPU recovery v5 resumes the local 4060 recovery policy in a segmented CPU
worker. Each segment is one complete 144-step game and resumes from the newest
CPU checkpoint. The worker remains bounded to four CPU threads and a 4 GiB
private-memory guard.

## Collapse response

The supervisor evaluates non-overlapping three-iteration windows. A collapse
still requires weak game results plus a policy-update signal; a single loss is
never sufficient. The response is now staged and does not automatically pause:

1. Apply up to two conservative PPO interventions (learning rate, clip range,
   entropy, target KL, terminal score coefficient, and scripted-opponent mix).
2. If collapse persists, choose the next configured reward profile, reset the
   intervention budget, and start the next worker from the latest checkpoint.
3. Cycle through the configured reward stages if later windows collapse again.

Hyperparameter interventions are appended to `collapse_interventions.jsonl`.
Reward transitions are appended to `reward_redesigns.jsonl`, including the
triggering window, before/after config, and restart action. `process.json`
records `reward_redesign_count`, `intervention_count`, and the active shaping
profile. Only an actual worker failure or the independent memory guard stops the
job; policy collapse does not create `paused_policy_collapse`.

## Reward profiles

All profiles keep a zero terminal potential. Therefore intermediate shaping
redistributes temporal credit while the terminal win/loss and bounded score
margin remain the optimization objective.

- `liquidation_v1`: cash, seeds, and carried/shed inventory. This is retained
  only as the initial compatibility profile.
- `production_cycle_v2`: preserves seed value after planting, values crop
  maturity and visible yield, preserves the value of placed animals, and values
  animal output. This repairs the old incentive to avoid productive investment.
- `harvest_market_v3`: moves more credit toward harvest-ready, market-priced
  output when the broader production-cycle signal is insufficient.

The configured third stage returns to a more conservative
`production_cycle_v2` scale with stronger terminal and exploration settings.
Further collapse events cycle the stages and continue training rather than
pausing.

Launch or resume:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_cpu_rl_recovery_v5.ps1 `
  -Iterations 50 -SegmentIterations 1 -MaxWorkerPrivateGiB 4.0
```

