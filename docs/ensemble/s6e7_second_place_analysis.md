# S6E7 第二名方案对 Kaggriculture 八模型的迁移分析

更新日期：2026-08-26

## 结论先行

S6E7 第二名方案最值得迁移的不是“模型越多越好”，而是三件事：所有模型使用完全对齐且无泄漏的 OOF 预测；以概率质量为目标做受约束的直接线性融合；把最终决策规则的调整与基础概率融合分开验证。它不能直接证明 Kaggriculture 的 25 类动作路由器会因此提高对局得分，因为 S6E7 是独立同分布的静态三分类，而本项目是带合法性约束、资源耦合和长期回报的序贯决策。

对当前 v4/v4_compact，优先级应是：

1. 保住现有 episode/seed/seat 分组 CV，增加严格的二层交叉拟合和跨 seed 稳定性报告。
2. 在温度校准后的 OOF 概率上，比较当前“每模型一个权重”与带强收缩的“每动作类一个权重”。
3. 用 leave-one-model-out、误差相关性和本地对局增益决定八个模型是否真的有效，不能为“八模型”强行设置正权重。
4. 先解决 ExtraTrees 的 636 MiB 体积和三个零权重成员，再谈更复杂的 stacker。
5. 所有 OOF 改善都必须通过本地双座位对局、合法动作率、重复订单率、时延和模型包体积这五道门。

## 阅读覆盖与可见性

### 已完整读取

- [2nd Place Solution: Trusting CV & Mathematical Precision](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/2nd-place-solution)：通过 Kaggle 公开 `GetWriteUpBySlug` 接口取得完整原始 Markdown（7,244 字符）、作者、链接和 topic 元数据，并与搜索索引交叉核验。
- [Trust your CV: the CV-LB relation so far](https://www.kaggle.com/competitions/playground-series-s6e7/discussion/718258)：通过公开 `GetForumTopicById` 和 `BatchGetForumMessages` 读取首帖原始 Markdown。该帖比较了 13 个使用同一 7 折划分的模型；作者报告 CV 与 Public LB 的差值均在约 ±0.0011 内，并把 Public LB 当作只用于 sanity check 的额外样本，而不是调参集。
- 下列八个关键外链 notebook 已用 Kaggle CLI 拉取公开源代码并检查训练、CV、特征、校准和输出逻辑：
  - [Tuned LightGBM Stacking](https://www.kaggle.com/code/kava1/s6e7-overcoming-0-95-tuned-lightgbm-stacking)
  - [DCNv2](https://www.kaggle.com/code/kava1/s6e7-dcnv2-lb-0-94962)
  - [FT-Transformer v2](https://www.kaggle.com/code/masayakawamata/s6e7-ft-transformer-v2-cv-0-95063)
  - [XGBoost OvR](https://www.kaggle.com/code/masayakawamata/s6e7-xgb-ovr-cv-0-95036)
  - [Unweighted LightGBM + Prior Correction](https://www.kaggle.com/code/robschieber/s06e07-unweighted-lightgbm-prior-correction)
  - [RealMLP PyTorch](https://www.kaggle.com/code/yekenot/ps-s6-e7-realmlp-pytorch)
  - [HGBC baseline](https://www.kaggle.com/code/redamountassir/ps-s6e7-hgbc-baseline-lb-0-95034-cv-0-95026)
  - [Per-value target encoding HGBC](https://www.kaggle.com/code/yaoguang516/s6e7-per-value-te-hgbc-0-9505-single-model)
- 技术机制以 [SciPy SLSQP 文档](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)、[SciPy Nelder-Mead 文档](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html)、[StratifiedGroupKFold 文档](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) 和 [scikit-learn 概率校准文档](https://scikit-learn.org/stable/modules/calibration.html) 交叉核验。

### 评论区覆盖限制

Kaggle 公共 discussion 列表在抓取时显示该 writeup 有 4 条评论，最近评论者为 Hamza Nazim Siddiqui；实时 topic 元数据则报告 `totalMessages=6`，它可能把主帖、回复或已删除消息一并计数。评论正文由 `GetForumMessagesInTopic` 返回 `forumTopics.get` 权限拒绝，搜索索引也没有暴露这些评论正文。因此本报告没有把评论内容编造或转述为事实，只记录了可核验的评论数量/最近评论者元数据。正文所链接的“Trust your CV”讨论首帖已完整读取；它不是本 writeup 的评论区。

这个限制很重要：下文所有方法结论来自正文、公开 notebook 源码和可读取的关联讨论，不声称覆盖了不可见评论正文。

## 第二名方案到底做了什么

### 1. 18 个基础预测器，不等于 18 种完全不同的算法

基础预测器覆盖 LightGBM、XGBoost、CatBoost、FT-Transformer、RealMLP 和 HGBC，也包含同一家族的不同实现和特征处理。多样性来源同时包括：模型族、OvR 与原生多分类目标、特征编码、随机种子、训练权重和概率校准，而不只是换一个 estimator 名字。

写作者报告等权 OOF log loss 为 0.105495，受约束融合后为 0.086310。最终不是再训练一个非线性元模型，而是直接线性组合基础概率。

### 2. 每个类别拥有一套模型权重

设基础模型数为 `M`、类别数为 `C`，方案优化 `W ∈ R^(M×C)`。对类别 `c`，模型权重非负并归一化：

```text
s[n,c] = Σ_m W[m,c] * p[m,n,c]
q[n,c] = s[n,c] / Σ_k s[n,k]
```

以全量 OOF 的多分类 log loss 为目标，用 SLSQP 搜索权重。FT-Transformer 在三个类别上的报告权重分别为 71.81%、62.45%、50.28%；XGB OvR 在类别 1、2 上提供了 28.19%、45.59% 的互补贡献。这正是当前单一全局权重表达不了的信息：一个总体稍弱的模型可能只在某类动作上很强。

### 3. 融合与决策规则分两步

得到融合概率后，方案再用 Nelder-Mead 调三个类别的乘数，以 Balanced Accuracy 为目标；报告乘数为 `[0.124044, 1.499826, 1.250181]`，OOF raw argmax 的 0.889132 提升到 0.950737。这里的提升主要是三分类极端先验和 Balanced Accuracy 的特殊结果，不是基础概率突然更准确。

### 4. 信任 OOF，而不是追 Public LB

关联讨论给出的依据是：690,088 行训练集的统一 7 折 OOF 比约 59,151 行 Public LB 切片更稳定；Balanced Accuracy 又主要受两个少数类左右，因此 Public LB 的 ±0.001–0.002 波动可以由抽样噪声解释。13 个模型使用完全相同的 `StratifiedKFold(7, shuffle=True, random_state=42)`，CV/LB 差值基本围绕零。

## 关键外链补充了哪些细节

### 可迁移的部分

- FT-Transformer：7 折；每折只在训练分区拟合全列 multiclass target encoding；原始列与 39 个 TE 数值一起输入；2 个 Transformer block、8 heads、`d_block=128`、内置 4-member ensemble。它说明神经模型有可能成为“某些类别的专家”，而非必须是最佳总体单模。
- XGBoost OvR：同一份无泄漏 target-encoded 特征上训练三个 binary XGBoost，并把三个 sigmoid 归一化。OvR 是有意义的结构多样性，尤其适合检查稀有动作是否受原生 25 类 softmax 压制。
- HGBC：原始特征保留 native categorical/NaN 路径，同时叠加训练折内的 per-value multiclass target encoding；一个版本还对三个 fold seed 做平均。对应到本项目，就是“相同模型族 + 不同视图”也可以制造有用的误差差异。
- DCNv2：3 seeds × 7 folds；折内 TargetEncoder 与 QuantileTransformer；4 层 cross network 加两层 MLP。它可以作为结构化交互模型候选，但当前提交包预算和单步时延下不能未经审计直接加入。
- LightGBM prior correction：模型本身不使用 class/sample weight，决策时再做 `argmax(p / prior)`。这证明训练时重采样/重加权与推理时先验调整是两种不同实验，不能混成一个开关。
- RealMLP：模型内 8-member ensemble、周期数值嵌入、EMA、balanced class loss 和折内 target encoding，说明“RealMLP”不是一个单一默认配置；当前 v4 的短训练版本与该公开实现不可视为等价。

### 不能混为最终第二名方案的部分

作者公开的 Tuned LightGBM Stacking notebook 是较早的 7 模型非线性 LightGBM 元学习器，并对每个基础模型和最终 stack 输出做类别乘数优化。最终 writeup 明确改成 18 模型直接线性融合。不能用这个 notebook 证明最终提交使用了 LightGBM stacker。

公开 notebook 中常见的全 OOF `beta`/类别乘数搜索会在同一 OOF 上选择参数并报告成绩。数据量 690k 时乐观偏差可能较小，但这仍不是严格的二层 OOF。迁移到本项目只有 20 个独立 episode/seed 组时，必须把乘数/融合权重也放进外层 group CV 的训练分区。

## 比赛特定技巧与通用方法

| 方法 | S6E7 中的角色 | 对 Kaggriculture 的判断 |
|---|---|---|
| 统一 fold 的 OOF 概率 | 所有模型可对齐融合 | 直接迁移，而且必须按 episode/seed/seat 分组 |
| 每类模型权重 | 三类专家互补 | 可迁移，但 8×25=200 参数，必须强收缩和嵌套 CV |
| 概率校准后再融合 | 让权重有可比意义 | 直接迁移；当前 temperature cross-fit 是正确起点 |
| 线性 blend 替代复杂 stack | 降低元层过拟合 | 应作为 v4 的默认对照；当前 stack/hybrid 必须用外层结果证明价值 |
| `p / prior^β` 或类别乘数 | 针对极不平衡的 Balanced Accuracy | 不能直接搬到对局策略；动作频率不是部署目标先验 |
| per-value target encoding | 利用合成表格重复值 | 游戏状态连续、强关联且按轨迹生成，直接照搬风险高；只可做折内消融 |
| Public LB 只做 sanity check | 避免榜单过拟合 | 对本项目对应为：Kaggle episode/单次公开分数不可作为调参反馈 |
| 多 seed/多特征视图 | 增加概率和误差多样性 | 可迁移，优先于盲目增加同质 estimator |
| 伪标签 | 第二名正文没有使用证据 | 不应归因于该方案；序贯专家轨迹可做 self-training，但需另立实验 |

## 当前 v4 的诊断

当前结果的有效样本量不是 101,570 行，而更接近 20 个独立 episode/seed component。行级 Balanced Accuracy 可能很高，却无法证明对未见策略、未见地图或完整 720 步决策的泛化。

现有优点：

- `StratifiedGroupKFold` 的四折审计显示 group/seed overlap 均为 0，双座位被放在同一 component。
- 八模型概率先做跨折 temperature calibration，再用于 blend/stack。
- meta blend 和 stack 至少在固定 base-OOF 矩阵上按 fold 拟合/验证，没有直接拿最终 fitted meta 的训练分数冒充验证分数。
- 保存了 per-class recall、log loss、Brier、ECE、时延和模型包体积。

主要缺口：

- 只有 20 个独立 component；25 类中 `BUY_LAND=40`、`BUILD_COOP=56`、`BUILD_PASTURE=60`、`PLACE=72` 等稀有类无法支撑 200 个自由的 class-specific blend 参数。
- 现有 meta cross-fit 还不是严格 nested stacking：对某个 meta held-out fold，其他 fold 行的 base-OOF 预测可能由看过该 held-out fold 的 base 模型产生。下一版需在每个外层 group fold 内重建 inner OOF，或把当前结果明确标为 semi-cross-fitted estimate。
- 当前全局 blend 权重为 LightGBM 0.2295、XGBoost 0.0801、HGBC 0.3176、ExtraTrees 0.3037、RealMLP 0.0691；CatBoost、LogReg、FT-Transformer 为 0。模型“训练成功”不等于八模型“真正生效”。
- ExtraTrees 单体约 636 MiB，是 789.18 MiB 包体积失控的主因；v4_compact 只是配置，尚未训练和测量。
- Hybrid OOF log loss 0.07273 优于 stack 0.07564 和 blend 0.08162，但 Balanced Accuracy 0.96705 略低于 stack 0.96715；差异远小于仅 20 个 component 带来的不确定性。
- 分类代理指标不等于对局收益；错误的 `PASS`、移动或市场单会产生完全不同的长期代价。

## 八模型逐项改造优先级

| 模型 | 当前状态 | 优先级 | 建议实验 | 预期收益 | 主要风险 |
|---|---|---:|---|---|---|
| ExtraTrees | BA 0.9685、全局权重 0.3037、约 636 MiB | P0 | 64/96/128 trees × depth 10/12/14；保存体积曲线；用 OOF 蒸馏到 HGBC 作为备选 | 最大体积收益，可能保留其非参数边界 | 压缩后精度和多样性骤降 |
| HGBC | BA 0.9579、权重 0.3176 | P0 | 保留为主干；尝试 raw view 与折内派生 view 两个变体，但只能选一个进包 | 高概率保留稳定增益 | 增加特征视图可能泄漏轨迹身份 |
| LightGBM | BA 0.9536、权重 0.2295 | P0 | 比较 `none/balanced/sqrt_balanced`；校准后再 blend；加叶数/深度压缩曲线 | 强、成熟、可压缩 | 与 HGBC/XGB 误差高度相关 |
| XGBoost | BA 0.9471、权重 0.0801 | P1 | 原生 25 类 vs OvR/分头 OvR；按动作族报告 class-wise NLL | 可能补稀有动作 | 25 个 OvR 体积和时延不可接受 |
| RealMLP | BA 0.9532、权重 0.0691 | P1 | 保留；增加 2–3 seeds 只用于训练期 OOF，最终选择或蒸馏一个紧凑成员 | 提供树模型外的误差多样性 | seed bag 增大包体积 |
| FT-Transformer | BA 0.9379、权重 0、约 44.6 MiB | P1 | 先做 class-wise leave-one-in/leave-one-out；若只在少数动作有增益，再用收缩的每类权重；否则不进部署包 | 可能成为少数动作专家 | 当前独立组少、训练慢、体积高 |
| CatBoost | BA 0.9071、权重 0 | P2 | 检查特征/NaN/权重适配；用更浅模型只保留 native ordered-boosting 差异；无增益则替换 | 可能补与 LGBM 不同的边界 | 当前质量差，强行正权重会降分 |
| Logistic Regression | BA 0.7731、权重 0 | P2 | 不再把它当必保基础预测器；保留为可解释 meta baseline，或替换成 compact DCNv2/OvR 变体 | 降体积或提升真实多样性 | “固定八种名称”与性能目标冲突 |

若产品要求恰好八个已加载模型，仍不能设定权重下限来伪造“全部生效”。正确门槛是：每个成员至少在一个外层 fold/动作族上带来稳定的 held-out log loss 或本地对局收益，并且总体体积、P95 时延达标。否则应替换该成员，而不是强制非零权重。

## 推荐的融合实现

### 第一步：全局线性融合基线

保留当前非负 simplex 全局权重，在每个外层 group fold 内拟合，在 held-out fold 评估。目标优先用未加权 NLL/Brier；同时报告 macro recall，但不以行级 Balanced Accuracy 单独选最终模型。

### 第二步：带收缩的每类权重

不要直接优化 8×25 个独立参数。使用部分池化：

```text
W[m,c] = (1 - λ_c) * w_global[m] + λ_c * w_class[m,c]
```

- 对样本数和独立组都足够的动作类，才允许 `λ_c > 0`。
- 对极稀有动作，固定 `λ_c = 0` 或与所属动作族共享权重。
- 给 `w_class` 加熵/L2 正则，并限制单模型最大权重，防止 20 个 group 下塌到单模型。
- 所有权重、正则系数和 `λ` 都只在外层训练 component 内拟合。

### 第三步：决策规则单独做

不建议直接做 `p/action_prior`。可比较以下三种推理规则：

1. 校准概率 + 合法性 mask + argmax。
2. 校准概率 + 按 decision slot/action family 的固定乘数 + mask。
3. 候选动作效用 `log p + value_bonus - legality/risk_penalty`。

乘数或 bonus 的选择依据必须包括完整对局得分，不能只最大化行级 macro recall。

## 验收标准

详细实验矩阵见 [S6E7 迁移实验清单](s6e7_transfer_experiment_checklist.md)。任何新 ensemble 只有同时满足下列条件才可替换 v4：

- 外层 episode/seed 分组 OOF 的 NLL/Brier 不恶化，且提升跨 fold/seed 稳定。
- 稀有动作收益不是由一个 component 独占；报告每类覆盖的独立 component 数。
- 对 starter、三个公开强策略和历史 checkpoint 做双座位对局，得分差置信区间不劣。
- 非法动作、重复订单、预算冲突和长 PASS 连击不增加。
- 序列化模型包低于 95 MiB，冷启动和 P95 单步时间满足提交约束。

## 不应做的事

- 不用全量 OOF 同时拟合 200 个权重并在同一 OOF 上宣称提升。
- 不把动作类不平衡等同于需要 `1/prior`；稀有动作往往本来就只在极少状态合法。
- 不把 Public LB/Kaggle 单次 episode 的小幅提升用作权重或乘数反馈。
- 不把相同树模型换几个 seed 就称为充分多样化。
- 不因为要求“八模型”而给劣质成员设置最低权重。
- 不在未完成体积、依赖、冷启动和本地对局审计前提交 v4_compact。
