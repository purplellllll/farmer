# S6E7 方法迁移到八模型的实验清单

这是一份训练/评测门禁，不是已生效的训练配置。它不要求修改 RL，且所有实验应使用独立输出目录，避免覆盖 v4 产物。

## A. 冻结输入与切分

- [ ] 记录训练 NPZ、curriculum JSONL 和专家策略源码的 SHA256。
- [ ] 保存 `episode_id`、simulator seed、seat、decision slot、action family。
- [ ] 至少收集 40 个独立 seed component；80 个为建议目标。
- [ ] 保证每个稀有动作至少跨 5 个 component；不满足时并入动作族做共享融合权重。
- [ ] 冻结外层 `StratifiedGroupKFold` fold assignment，所有模型完全共用。
- [ ] 双座位镜像和相同 seed 的轨迹不得跨 fold。
- [ ] 元学习器评估使用严格 nested group CV：外层 held-out component 不得参与生成 inner OOF 特征的任何基础模型训练。

## B. 每个基础模型必须保存

- [ ] 原始 OOF 概率、温度校准 OOF 概率、fold/test 或 full-fit 预测接口。
- [ ] NLL、Brier、ECE、macro-F1、Balanced Accuracy 和每类 recall。
- [ ] 每动作类的样本数、独立 component 数、NLL 和 top-2 confusion。
- [ ] 每 fold 训练时间、峰值内存/GPU 显存。
- [ ] 序列化后的模型级 MiB、冷启动、warm P50/P95。
- [ ] `classes_` 与 25 类 manifest 的严格对齐断言。

## C. 融合消融

按顺序运行；后一项只有在外层 held-out 上改善才保留。

| ID | 方案 | 自由度 | 必须报告 |
|---|---|---:|---|
| E0 | 等权校准概率 | 0 | 基线指标、对局 |
| E1 | 当前全局非负 simplex | 7 | cross-fitted 权重、权重方差 |
| E2 | LogisticRegression stack | 约 8×25×25 | 与 E1 的 held-out 差及过拟合差 |
| E3 | 当前 hybrid | 1 + E1/E2 | alpha 的外层稳定性 |
| E4 | 动作族共享 class-specific simplex | 8×动作族 | 收缩前后、每族覆盖 component |
| E5 | 25 类 partial-pooling simplex | 最多 200 | `λ_c`、正则、effective DoF |
| E6 | 类别/slot 决策乘数 | 仅高支持类 | OOF 代理指标和完整对局同时报告 |

- [ ] SLSQP 返回 `success=True`，记录终止原因、迭代数、初值敏感性。
- [ ] 至少使用均匀、当前权重和三个随机 simplex 初值。
- [ ] 记录权重 bootstrap/group-jackknife 区间。
- [ ] 任何只在全量 OOF 上拟合并评分的结果标记为 `fit score`，不得标记为 OOF。

## D. 八模型有效性

- [ ] 对每个成员做 leave-one-model-out。
- [ ] 计算模型间 OOF 真类概率相关性、预测分歧率和错误 Jaccard。
- [ ] 一个成员只有在至少两个外层 fold 或一个明确动作族上稳定改善才算“生效”。
- [ ] 零权重且 leave-one-out 无改善的成员必须替换或移除，不能设置人为权重下限。
- [ ] 对 CatBoost、FT-Transformer、Logistic Regression 先完成失败原因诊断。
- [ ] 对 ExtraTrees 生成树数/深度—NLL—MiB Pareto 曲线。

## E. 完整对局门禁

- [ ] 固定未参与训练的 simulator seeds。
- [ ] 每个对手做双座位换边，报告胜率、平均/中位分差和 bootstrap 95% CI。
- [ ] 对手至少包括 starter、三种不同风格启发式、三个公开强策略、一个历史 ensemble、一个历史 RL checkpoint。
- [ ] 至少 30 个独立 seed；候选进入提交前建议 100 个。
- [ ] 报告非法动作、重复订单、预算冲突、PASS 连击、动作族分布。
- [ ] 记录 720 步完成率、异常和超时。

## F. 提交约束门禁

- [ ] 整包压缩后和解压后体积均记录；模型预算目标 `< 95 MiB`。
- [ ] 在 Kaggle 目标 Python/NumPy 环境做真实 import 和 pickle/load smoke。
- [ ] 双座位 720 步运行成功，无 traceback/error。
- [ ] 记录首步 cold latency、warm P50/P95/max 和总 episode 时长。
- [ ] 验证八个期望模型的加载 sentinel；同时记录最终实际非零权重/有效动作族。
- [ ] 只有 OOF、对局、体积、时延四类门禁全部通过才生成 submission artifact。

## G. 最小实验命名与产物

建议使用不覆盖 v4 的目录，例如：

```text
artifacts/ensemble-s6e7-transfer/<run-id>/
  dataset_manifest.json
  fold_assignments.npz
  base_oof/<model>.npz
  blend_report.json
  model_size_audit.json
  tournament.json
  decision_gate.json
```

`decision_gate.json` 至少应包含 `oof_pass`、`tournament_pass`、`size_pass`、`latency_pass` 和最终 `deployable` 布尔值。
