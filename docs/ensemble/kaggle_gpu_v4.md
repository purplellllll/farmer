# Kaggle GPU complete-action ensemble v4

The v3 ensemble successfully loaded all eight estimators, but its 1.0 OOF score
was not meaningful evidence of game strength.  The 5,752 labels came from a
single starter carrot loop: 5,360 `PASS`, 80 `PLANT`, 240 `WATER`, and 72
`HARVEST`.  It contained no farm-hand actions and did not model the market.

V4 replaces that table with self-generated official-simulator trajectories and
a decision-slot schema.  A row represents the main farmer, an active hand, or
one ordered market decision.  Submitted orders are followed by exactly one
`NO_ORDER` terminator; ten padded independent slots are not trained, preventing
the observed failure where the policy bought the same seed ten times.

The 25 targets cover all 18 unit operations plus `NO_ORDER`, `BUY_SEED`,
`BUY_ANIMAL`, `BUY_PRODUCT`, `SELL`, `HIRE`, and `BUY_LAND`.  Market generation
reserves cash in queue order and rejects duplicate orders.  The data artifact
retains episode, simulator seed, acting seat and decision slot.  Four-fold OOF
uses the connected episode/seed grouping in `farmer_ensemble.splitting`, so
mirrored seats and repeated states from a game cannot cross folds.

Reproduce locally:

```powershell
.venv\Scripts\python.exe scripts\build_joint_router_dataset.py `
  --output-dir artifacts\ensemble-retrain-v2\data `
  --episodes 20 --episode-steps 720 --seed-start 20260900
```

Upload only `joint_router_train.npz` plus `dataset-metadata.json` to Kaggle, then
push `kaggle/ensemble_gpu_v4`.  The kernel trains exactly LightGBM, XGBoost,
CatBoost, HGBC, ExtraTrees, Logistic Regression, FT-Transformer, and RealMLP.
It reports calibrated single-model and blended/stacked/hybrid OOF log loss,
balanced accuracy, macro-F1 and per-class recall.

The first full v4 run produced a 789.18 MiB pickle and therefore is a research
artifact, not a submission artifact.  Size attribution showed ExtraTrees alone
used 636.02 MiB; FT-Transformer used 44.61 MiB and the four boosting models used
21.81--29.75 MiB each.  `configs/ensemble/v4_compact.json` is the next retrain
profile: 64 shallower ExtraTrees, smaller boosting ensembles, and a 2-layer,
64-token FT-Transformer.  It preserves all eight model families while targeting
the 95 MiB budget.  Do not claim the compact profile fits until it is retrained
and its serialized artifact is measured.
