# Kaggle GPU router run v3

- Kernel: `llllllc666/farmer-eight-model-ensemble`, version 3
- Hardware: Tesla P100-PCIE-16GB
- Input: 5,752 self-generated official-starter states, 55 public/acting-seat-safe features
- Target: bounded farmer route (`PASS`, `PLANT`, `WATER`, `HARVEST`)
- Validation: four-fold episode/seed-connected grouped OOF
- Wall time: 304.74 seconds
- Artifact: 21.67 MiB, below the 95 MiB research budget

All eight selected models trained successfully: LightGBM, XGBoost, CatBoost,
HGBC, ExtraTrees, Logistic Regression, FT-Transformer and RealMLP. The final
non-negative blend retained a non-zero weight for every model.

The near-perfect OOF result is a teacher-reproduction check, not evidence of a
medal-strength game agent. The labels come from one deterministic carrot-loop
teacher and cover only four farmer actions. Before deployment, add stronger and
more diverse licensed/self-generated teachers, market/candidate/value targets,
held-out strategy families, and paired-seat league evaluation.
