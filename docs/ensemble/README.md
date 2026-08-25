# Kaggriculture 表格模型集成

本模块训练的不是“完整联合动作多分类器”。它只服务于三个边界明确、能生成合法候选的任务：

1. **专家路由**：在作物、畜牧、恢复、领先保守、落后冒险等策略之间选择；
2. **合法候选排序/分类**：规则执行器先生成满足资金、库存、位置和订单约束的候选，再由模型排序；
3. **辅助价值预测**：预测胜率、金币差区间、断粮/溢仓/无法回本等风险，供 RL Actor、Critic 或安全门使用。

禁止把 `farmer + N hands + 0..10 market orders` 压成一个无约束类别。那会造成组合类别爆炸、相互冲突动作和“分类准确但输掉整局”的目标错配。

## 快速使用

输入是 `.npz`：

- `X`：二维数值特征；
- `y`：专家、候选或辅助价值类别；
- `groups`：episode/match 标识，必需；
- `seeds`：环境 seed，推荐；
- `seats`：座位，推荐；
- `feature_names`：可选但推荐。

```powershell
$env:PYTHONPATH = "F:\Kaggle\farm\farmer\src"
python -m farmer_ensemble `
  --data artifacts\router_train.npz `
  --config configs\ensemble\default.json `
  --output artifacts\ensemble_router
```

训练产物包括：

- `ensemble_bundle.pkl`：完整可加载模型；
- `manifest.json`：依赖版本、八模型训练/跳过状态、OOF 指标、融合权重、文件哈希、大小与本机延迟；
- `feature_manifest.json`：严格特征顺序、类型、缺失值策略和目标语义。

```python
from farmer_ensemble import load_bundle

model = load_bundle("artifacts/ensemble_router")
probabilities = model.predict_proba(one_or_more_states)
route = model.predict(one_or_more_states)
```

## 泄漏与评估约束

- OOF 使用 `StratifiedGroupKFold`；同一 episode 和复用 seed 通过联通分量同时锁在一个 fold。
- `seat` 不作为可拆分样本；镜像座位随 episode/seed 原子组一起移动。
- 类别权重只使用每个训练 fold 计算。原生不接收 `sample_weight` 的模型执行可审计的加权重采样。
- 每个基模型的温度校准、stacking 和 blending 都交叉拟合；报告指标不是在训练自身的 meta-model 上直接回测。
- 必看 `log_loss`、macro-F1、balanced accuracy、ECE 和最终双座位成对联赛胜率；不能用普通 accuracy 代替比赛目标。

## 可选依赖

基础 smoke 路径只需 `numpy` 与 `scikit-learn`。其余模型按需安装：

```powershell
pip install lightgbm xgboost catboost
pip install pytabkit torch
```

缺失依赖时对应模型会标为 `skipped_missing_dependency`，其他模型仍会训练。FT-Transformer 和 RealMLP 通过 PyTabKit 的真实 sklearn 接口调用；没有 PyTabKit 时不会用占位分类器冒充成功。

默认保存的是研究/训练集成。Kaggle 的约 100 MiB 包限制和每回合约 1 秒预算必须在目标镜像中再次验证。若 manifest 超预算，应按 OOF 权重与消融结果裁剪基模型、减少树数/深度，或蒸馏到较小学生，不能仅依赖开发机延迟。

详细选择依据见 [model_selection.md](model_selection.md)。
