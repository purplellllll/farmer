# Farmer

Research and training foundations for the Kaggle **Kaggriculture** simulation
competition. The repository deliberately keeps two complementary routes:

1. a leakage-safe Transformer Actor-Critic pipeline with behaviour cloning,
   legal-action masking and population self-play; and
2. an eight-model tabular ensemble for expert routing, legal-candidate ranking
   and auxiliary value/risk prediction.

The tabular models are helpers, not a flat classifier over the full joint game
action. Farmer, variable farm-hand actions and ordered market operations are
compiled through legal candidates and deterministic safety checks.

## Repository layout

```text
configs/rl/             RL model, population and data-manifest configs
configs/ensemble/       eight-model and smoke ensemble configs
docs/rl/                licensed sources, strategies and RL architecture
docs/ensemble/          model selection, validation and training guide
src/farmer_rl/          environment, trajectories, tokens, actions, BC/PPO
src/farmer_ensemble/    adapters, grouped OOF, calibration and stacking
kaggle/ensemble_gpu/    reproducible Kaggle GPU training script
tests/                  dependency-light unit and smoke tests
```

## Selected tabular models

- LightGBM
- XGBoost
- CatBoost
- scikit-learn HistGradientBoostingClassifier (HGBC)
- ExtraTrees
- Logistic Regression
- FT-Transformer through PyTabKit
- RealMLP through PyTabKit

Training uses episode/seed-connected grouped out-of-fold predictions, fold-local
class balancing, cross-fitted temperature calibration, non-negative blending,
logistic stacking and a hybrid predictor. Missing optional dependencies are
reported and skipped; the framework never substitutes a different estimator
under an advertised model name.

See [the model decision record](docs/ensemble/model_selection.md) and
[ensemble usage](docs/ensemble/README.md).

## RL route

The RL path treats each player as one policy controlling the entire farm. Farm
hands are variable joint-action slots rather than independent cooperative
agents. The intended progression is:

```text
licensed rule experts and self-play trajectories
    -> behaviour cloning
    -> PPO against frozen opponents
    -> population/league self-play
    -> distillation and CPU deployment checks
```

Acting-seat private observations are kept separate at collection time. Public
replays or competition data are not committed to this repository. Sources with
unverified licences remain quarantined in the source registry.

See [the RL guide](docs/rl/README.md), [architecture](docs/rl/architecture.md),
[strategy catalogue](docs/rl/strategy_catalog.md), and
[source/licence registry](docs/rl/sources.md).

## Installation

Use an isolated virtual environment; the official simulation package has a
large, tightly pinned dependency graph.

```bash
python -m venv .venv
```

Core ensemble and data-contract code:

```bash
python -m pip install -e .
```

Optional model families:

```bash
python -m pip install -e ".[boosting]"
python -m pip install -e ".[tabular-neural]"
python -m pip install -e ".[rl]"
```

Install only the extras required for the current experiment. Kaggle deployment
must still be audited against the competition's one-second action timeout,
100 MiB submission limit and CPU-only resource budget.

## Quick checks

Validate the RL configuration without Ray or Kaggle dependencies:

```bash
farmer-rl validate --config configs/rl/ppo.json
```

On an NVIDIA Windows workstation, warm-start the eight-layer Transformer from
a licensed behaviour-cloning checkpoint and use the checkpointed native PPO
runner (this avoids Ray worker bootstrap issues seen with CUDA on Windows):

```bash
farmer-rl native-self-play \
  --config configs/rl/local_4060.json \
  --iterations 2100 \
  --output artifacts/local-4060-gated \
  --bc-checkpoint checkpoints/starter_bc_v5.pt
```

The runner alternates the learner seat, applies bounded potential shaping plus
the terminal win/loss result, samples frozen checkpoints with PFSP-style
weights, logs one JSON object per iteration, and keeps only the newest local
checkpoints. `--resume` accepts a native checkpoint after interruption.

Train the ensemble from an NPZ file containing at least `X`, `y`, and `groups`:

```bash
farmer-ensemble \
  --data artifacts/training/features.npz \
  --config configs/ensemble/default.json \
  --output artifacts/ensemble
```

Kaggriculture does not provide a conventional tabular training set. Build
router/candidate/value labels from licensed or self-generated trajectories
first; `farmer_ensemble.router_dataset.build_router_npz` creates the bounded
farmer-action router table used by the Kaggle GPU job. The reproducible Kaggle
script and metadata live under `kaggle/ensemble_gpu/`.

Run dependency-light tests:

```bash
python -m unittest discover -s tests -v
```

## Data and publication policy

- Pin the official environment version and record source hashes in manifests.
- Store only the acting player's private observation in an actor sample.
- Split related seats, episodes, seeds and derived agents as atomic groups.
- Record URL, licence, revision and behavioural fingerprint for every expert.
- Do not commit downloaded competition data, replays, checkpoints or trained
  artifacts; the `.gitignore` excludes their standard locations.
- Select agents by paired-seat, held-out-seed win rate, not classifier accuracy
  or shaped training reward alone.
