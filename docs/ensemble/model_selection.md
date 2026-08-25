# 八种分类器的选择与定位

## 结论

选择八种互补模型进入训练池：

| 优先级 | 模型 | 入选理由 | 主要风险与控制 |
|---:|---|---|---|
| 1 | LightGBM | 高吞吐直方树，适合大量手工状态/市场交叉特征；支持多分类、样本权重和类别特征 | 类别权重会损害概率解释，必须用 OOF 校准；限制叶数、深度和树数 |
| 2 | CatBoost | 对 phase、route、shop type 等类别特征和非线性交互强；Ordered Boosting 有助于抑制目标统计泄漏 | 高基数类别会放大模型；CPU 线程固定为 1，并限制深度/迭代 |
| 3 | XGBoost (`hist`) | 稳健、成熟，与 LGBM/CatBoost 的树构造偏差不同，能增加融合多样性 | 深树的模型大小和推理成本；限制深度并使用 histogram tree method |
| 4 | HGBC | sklearn 原生、缺依赖时最可靠的强树基线；原生缺失值、类别权重和早停 | 能力通常低于充分调参的三大 GBDT，但部署简单且可做稳定回退 |
| 5 | ExtraTrees | 高随机性提供与 boosting 不同的误差相关性，适合非平滑阈值规则 | 树太多会超过包大小；限制树数、深度、叶样本并固定单线程 |
| 6 | Logistic Regression | 极小、极快、概率基线稳定；捕获金币差、剩余回合等近线性决策边界，也是融合的“低方差锚” | 单模上限有限；只作为多样性与保底，输入做 fold 内标准化 |
| 7 | FT-Transformer | 特征 token 自注意力能学习状态字段的条件交互，与树模型偏差互补 | CPU 延迟与数据需求较高；通过 PyTabKit 真实实现，限制 epoch/batch，部署前消融 |
| 8 | RealMLP-TD | 现代强表格 MLP，预处理和 tuned defaults 对中型表格数据有竞争力 | 训练/推理和包体高于树/线性；限制 CPU 模型规模，缺 PyTabKit 时明确跳过 |

这八种并不意味着线上一定同时携带八个模型。八模型用于 OOF 竞争与融合；最终提交必须依据交叉拟合权重、成对联赛消融、模型 MiB 和真实镜像 p95 延迟做裁剪或蒸馏。

## 对题目点名六类模型的判断

### LightGBM：高度适合

适合专家路由、合法候选打分和风险分类。官方 `LGBMClassifier` 支持 multiclass、`class_weight`、`sample_weight` 和类别特征；官方也明确提醒 class weighting 可能造成较差的类别概率，因此本框架在独立 OOF 预测上做温度校准。[LightGBM classifier API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)

### XGBoost：高度适合

`tree_method="hist"` 对数值化的农场状态特征有效，CPU 推理可控，并提供与 LightGBM 不同的归纳偏置。官方文档确认 sklearn classifier、categorical data 和 JSON/UBJSON 模型保存路径；本框架训练时固定 `n_jobs=1`，并对树深/树数设上限。[XGBoost categorical tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html) [XGBoost model IO](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html)

### CatBoost：高度适合

游戏状态包含大量低/中基数类别（天数阶段、商店、产品、路线、对手 archetype），CatBoost 能直接构造类别组合；官方建议不要预先 one-hot。它支持 multiclass 和 `class_weights`/`auto_class_weights`。高基数类别可能造成大模型，因此 manifest 必须检查大小。[CatBoost categorical features](https://catboost.ai/docs/en/features/categorical-features) [CatBoost class weights](https://catboost.ai/docs/en/references/training-parameters/common#class_weights)

### FT-Transformer：条件适合

它通过 feature tokenizer 和 Transformer 建模跨字段交互，能补充树模型。原论文报告其在多种表格任务上超过其他深度基线，但并没有证明它必然胜过 GBDT；在本题还受 CPU/包体限制。因此它是训练池中的多样性模型，而不是默认部署主模型。[NeurIPS 2021 paper](https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html) [官方推荐实现](https://github.com/yandex-research/rtdl-revisiting-models)

### RealMLP：条件适合

RealMLP 的目标正是给出更强的预调参 MLP；PyTabKit 提供 `RealMLP_TD_Classifier` sklearn 接口、CPU 设备、epoch/batch 和预测概率。本框架调用这个真实接口，不在依赖缺失时伪造实现。它适合提供神经网络多样性，但需用最终联赛验证收益是否覆盖延迟。[RealMLP paper](https://arxiv.org/abs/2407.04491) [PyTabKit model API](https://pytabkit.readthedocs.io/en/latest/models/01_sklearn_interfaces.html)

### sklearn HGBC：高度适合的保底

它没有第三方运行时负担，直方训练在大于约一万样本时比经典 GBC 快，支持 NaN、`class_weight`、`sample_weight` 和早停。它是最适合 CI/smoke 与稳定提交的基模型。[HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

## 新增候选的取舍

最终新增 **ExtraTrees** 和 **Logistic Regression**，原因不是期待它们单模第一，而是融合需要低相关误差，同时它们能在 CPU 上快速给出 `predict_proba`，并支持类别/样本权重。ExtraTrees 官方实现还支持 `class_weight` 和并行控制。[ExtraTreesClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

未入选：

- RandomForest：与 ExtraTrees/HGBC 高度重复，通常需要更多/更深树才能贡献差异，包体不利；
- kNN：保存全部训练样本，内存和单回合推理随数据量增长；
- RBF-SVM：大样本训练和概率校准昂贵，部署延迟难控；
- TabPFN/TabR：可能很强，但预训练权重、检索存储或运行依赖对 100 MiB/1 秒约束不友好；
- TabNet/SAINT：相较已选择的 FT-Transformer/RealMLP 增量多样性有限，却继续增加深度运行栈；
- BalancedRandomForest：对类别不平衡有效，但引入 imbalanced-learn，且本框架已有 fold 内权重/重采样与校准。

## 为什么不能做扁平联合动作分类

Kaggriculture 一步是由 farmer、可变数量 hands 和最多十条有序 market order 构成的组合动作。平铺会产生三个问题：

1. 类别数随 hand 数量和订单参数组合爆炸，稀有关键动作几乎没有标签；
2. 独立多头又会产生资金、库存、位置、顺序冲突；
3. macro-F1/accuracy 不具备 720 回合信用分配，无法判断某次扩地是否最终提高胜率。

因此模型只能接收规则生成的合法候选，预测 `P(expert | state)`、`P(good candidate | state, candidate)` 或辅助价值。最终动作仍由动作掩码、确定性编译器和 RL 长期价值共同决定。

## 数据与验证协议

普通随机 K-fold 会把同一对局相邻时刻、同一 seed 或两个座位泄漏到训练和验证。框架用 episode 与 seed 的**联通分量**作为原子 group，再执行 stratified grouped split；这比简单拼接 `episode + seed` 更严格。`StratifiedGroupKFold` 的目标正是在 group 不重叠时尽量维持类分布。[scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html#stratifiedgroupkfold)

概率校准必须使用基模型没见过的预测。每个基模型先生成 OOF 概率，再交叉拟合温度；stacker 和 blend 权重也按外层 fold 交叉拟合，最后才用完整 OOF 拟合部署 meta-model。官方校准指南同样强调 calibrator 应使用与基模型拟合数据独立的数据。[Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)

类别不平衡默认使用平方根 balanced 权重，避免极稀有动作对损失产生过度放大。应同时报告 log loss、macro-F1、balanced accuracy、ECE；最终选择必须使用交换座位、成对 seed 的 agent 胜率和置信区间。

## 与强化学习的关系

推荐职责分工：

```text
state/history
    -> 表格集成：专家路由、候选质量、风险/价值辅助
    -> 规则系统：合法候选、硬约束、安全否决
    -> Transformer Actor-Critic：长期信用分配与自博弈适应
    -> deterministic compiler：输出合法 action dict
```

表格集成提供稳定下限、监督数据和可解释特征；Actor-Critic 负责从 720 回合胜负中学习长期策略。两者应通过离线联赛消融比较，不能用分类器离线 accuracy 与 RL training reward 直接比较。
