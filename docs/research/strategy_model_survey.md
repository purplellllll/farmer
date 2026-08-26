# Kaggriculture 强策略与可迁移模型调研

核验时间：2026-08-26 10:59（Asia/Shanghai）
目标：为本地对弈、行为克隆（BC）、离线 RL、自博弈对手池和 ensemble teacher 筛选互补来源；尤其解决当前策略网络把 27 个动作槽独立采样后，在同一回合生成大量重复 `BUY_SEED`，继而长期 `PASS` 的结构性问题。

## 结论先行

最值得投入的不是把现有独立动作头继续加宽，而是把动作表示改为“**合法候选生成器 + 条件式联合解码器**”：

1. 规则层根据当前公开状态生成少量完整、合法的动作 bundle，负责现金、库存、单位位置、市场 10 单上限和互斥约束。
2. 第一版先用 masked categorical 在 32–128 个完整 bundle 中选一个；它最容易稳定训练，也是 ensemble router、BC、PPO 和离线 RL 都能共用的接口。
3. 第二版把 farmer、hands 和 market block 改成 autoregressive decoder；每生成一个动作就更新剩余预算、库存、已用商品和合法 mask，以 `EOS` 结束。PPO 的联合 log-prob 是各条件分布 log-prob 之和。
4. Transformer 负责时序和实体关系，残差连接保留；但市场订单不能继续使用彼此独立的 10 个头。大参数量本身不会修复错误的概率分解。

外部来源方面，当前真正可用的高价值资产是 Apache-2.0 Kaggle 公共策略和 MIT/Apache-2.0 GitHub 策略，而不是通用 Hugging Face 权重。本次公开 HF API 查询没有发现 Kaggriculture 模型仓库；仅发现两个含比赛 replay 的数据集，均存在明确的访问/再分发限制，应隔离，不进入训练。

推荐的首批多样教师池是：官方 `starter/random`、Kaito v27 固定路线、Flexonafft 多路线策略、Andrey V16 市场队列策略、lonespear 算法式闭环策略、Seyamalam 恢复/市场策略、我方历史 checkpoint，以及饥饿/溢出/现金死锁等针对性 exploiter。不要用多个共享同一 replay 路线的 notebook 伪装成“多样性”；先按动作流指纹聚类，再按 lineage 分组采样和切分。

## 核验状态定义

| 状态 | 含义 |
|---|---|
| `RUN-720` | 已由本次调研实际下载输出，并在本机官方 `kaggle-environments` 里从两个席位各完成一局 720 回合；只证明协议和运行，不证明对强手胜率 |
| `PULLED` | 已用已认证 Kaggle CLI 拉取 notebook 源码或输出，记录了摘要；尚未纳入仓库或完整联赛 |
| `API-OK` | 官方仓库/模型 API 和技术文档可访问，许可证元数据已核对；未在本仓库安装或训练 |
| `PAGE-ONLY` | 搜索或网页可见，但本次未能用 CLI/API 获得可复现工件 |
| `QUARANTINE` | 即使能看见，也因许可证、比赛数据、来源或再分发风险不得进入训练 |

所有第三方代码进入训练前还需要固定 revision、保存具体文件 SHA-256 和 notice。`RUN-720` 也不等于允许直接合并源码。

## 为什么 27 槽独立头会失败

现结构近似假设：

```text
P(a_farmer, a_hand_1, ..., a_market_10 | s)
  ≈ Π_i P(a_i | s)
```

每个市场槽只看到同一个状态，因此它们可以同时认定 `BUY_SEED WHEAT` 是局部最优。单槽合法 mask 只能阻止“这个订单在初始状态非法”，不能表达“前一个槽已经买过”“现金已经被前几个订单花掉”“同商品订单应合并”“先卖哪一单会改变后续价格”等条件约束。重复订单耗尽现金后，网络又很容易退化到低风险但无收益的 `PASS`。

建议的联合分解是：

```text
P(A | s) = P(a_farmer | s)
           Π_h P(a_hand_h | s, a_farmer, a_hand_<h)
           Π_m P(a_market_m | s, unit_actions, a_market_<m)
```

每个条件步都重新计算：

- 剩余现金、种子、产品和订单数；
- 已占用单位、目标格、物品和市场槽；
- 商品数量上限、同类订单合并规则与 `EOS`；
- 冲突图（例如同一单位不能同时移动和工作、同一预算不能重复花费）；
- 最终由确定性 `compile_and_validate()` 再做一次 fail-closed 校验。

最稳妥的工程顺序是先选择完整合法 bundle，再升级成 token 级 autoregressive decoder。仅把 MaskablePPO 套到当前 `MultiDiscrete` 独立头上不够，因为交叉槽约束会随已经采样的前缀变化。

## Kaggle Code、排行榜与公开研究

### 当前可见范围

比赛仍在进行，排行榜和 meta 都会变化，因此不存在可当作最终结论的“获奖者 writeup”。本次 Kaggle CLI 在 2026-08-26 10:58 抓到的榜首片段为 Crop Dusta 3073.7、Ryo Hasegawa 3064.9，后续队伍分数较低；这只是实时 skill-rating 快照，不是最终名次。官方说明明确：胜负/平局影响 rating，金币差不直接决定 rating，截止后还会继续比赛并运行最终 Bradley–Terry tournament。[官方比赛说明](https://www.kaggle.com/competitions/kaggriculture/overview/citation)

搜索没有找到当前榜首两队公开、可归因的完整 writeup。下表记录的是公开 Notebook 的自身页面分数、公开回放 holdout 或工程证据，不能把它们冒充榜首方案。

### 策略候选

| 候选 | 核验 | 许可证 | 策略类别与互补价值 | 输入/输出适配 | 资源画像 | 最适合用途 | 主要风险 |
|---|---|---|---|---|---|---|---|
| [Kaito v27 Midgame Meta Reset](https://www.kaggle.com/code/kaitofukami/25-27-strict-future-v27-midgame-meta-reset) | `RUN-720`；页面曾显示 Public 3090.1；输出可下载 | 页面 Apache-2.0 | 固定完整路线 + 稀疏 SELL 排序；适合高质量 BC 和路线型对手 | 已是 `agent(obs)`；低 | 纯 Python/CPU，提交体积小 | BC teacher、对手池、路线先验 | replay 路线会随 meta 过时；页面的 25/27 是固定 action trace holdout，不是官方 LB |
| [Flexonafft Multi-Route Farming Agent](https://www.kaggle.com/code/flexonafft/kaggriculture-multi-route-farming-agent) | `RUN-720`；页面 Best 2767.3、抓取时 Public 1902.5 | Apache-2.0 | 根据早期 shop 次序在多条完整路线间选择，并做杂草/市场恢复；比单路线更适合训练 route selector | 已是 `agent(obs)`；低 | 148 KB 源码，CPU | 对手池、route label、BC teacher | 路线之间可能与其他公开策略同 lineage；rating 波动说明不能只按历史峰值排序 |
| [Andrey V16 Same-Turn Slot Sniper](https://www.kaggle.com/code/andrewsokolovsky/kaggriculture) | `RUN-720`；页面 Best 2671.3 | Apache-2.0 | 完整生产路线不变，只调整同回合 premium SELL 队列；提供“市场排序而非产量变化”的互补专家 | 已是 `agent(obs)`；低 | 29 KB 源码，CPU | market-order teacher、对手池、反镜像测试 | 窄策略；依赖近克隆假设，不能充当通用生产教师 |
| [Kaito v21.1 Conditional Memory](https://www.kaggle.com/code/kaitofukami/177-180-fresh-top-30-v21-1-conditional-memory) | `PULLED`；页面 Best/Public 2665.8 | Apache-2.0 | 30 条 public-route 原型的 1-NN 记忆，仅重排既有 SELL 交集；是低延迟 opponent model 的直接参考 | agent 接口现成；中等，需保留状态签名定义 | CPU，低延迟 | opponent embedding、route memory、market overlay | replay 固定对手不能反应；原型按时间迅速陈旧；训练/holdout 必须按 lineage 分组 |
| [Boatlee V21-R1 Public-State Route Portfolio](https://www.kaggle.com/code/boatlee/v21-r1-public-state-route-portfolio) | `PULLED`，输出含 `main.py` | Kaggle 页面许可证需在真正采集时再次截图/固化 | 多完整生产计划 + 公共状态路由 + 市场时序 overlay；结构上适合 mixture-of-experts | agent 接口现成；中等 | CPU | route ensemble teacher、对手池 | Notebook metadata JSON 不含许可证字段；必须在 ingest 前重新固化页面许可证与具体输出 hash |
| [lonespear/kaggriculture](https://github.com/lonespear/kaggriculture) | `API-OK`；仓库已有固定 commit 记录 | MIT | 算法式闭环：畜牧/草莓、任务分配、双席位 league、并行 sweep；比 replay tape 更能应对随机扰动 | `main.py` agent；低 | README 报告亚毫秒级 agent 逻辑，CPU | 多样化对手、规则标签、局部恢复 teacher | 仓库历史结果会过时；仍需当前环境 720 回合验证和 per-file notice |
| [Seyamalam/Kaggriculture](https://github.com/Seyamalam/Kaggriculture) | `API-OK`；仓库已有固定 commit 记录 | MIT；第三方文件另审 | 闭环路线、市场恢复、公开 replay regression 工具；研究过程和失败门槛完整 | 低到中 | CPU | 评测方法、市场恢复对手、可解释 teacher | 仓库明确包含外部策略 lineage，根 MIT 不自动覆盖每个导入文件 |
| [Roman Barnyard Economist](https://www.kaggle.com/code/romanrozen/strong-barnyard-economist) | `PULLED` | Kaggle 页面 Apache-2.0 | livestock-first、完整路线、杂草恢复和市场保护 | 低 | CPU | 畜牧专家、生产组合多样性 | 与其他公开强路线可能高度相似，先做动作流聚类 |
| [Adaptive Farming Strategy](https://www.kaggle.com/code/tetsutani/adaptive-farming-strategy-for-kaggriculture) | `PULLED`；页面 Best 1433.6 | Apache-2.0 | 五条由 shop demand 选择的路线，包含 storage/terminal guard | 低 | CPU | 较弱但机制不同的 curriculum 对手 | 历史表现明显弱于前几项；只作为多样性与故障恢复样本 |
| [Salem “3094 score”](https://www.kaggle.com/code/salemali7/3094-score-kaggriculture) | `PAGE-ONLY`；网页 Best 2684.1，CLI pull 返回 404 | 页面 Apache-2.0 | 页面标题声称 3094，但可见 Best 与标题不一致 | 未验证 | 未验证 | 暂无 | 标题不能当证据；拿不到固定源码前不得进入训练 |
| [Pure Architecture “2600+ Elo”](https://www.kaggle.com/code/saitejabandaruin/kaggriculture-pure-architecture-2600-elo-v3) | `PULLED` | Apache-2.0 | 轻量 heuristic/fallback 架构说明 | 低 | CPU | 仅阅读架构 | Notebook 中“mathematically capable”等表述缺少与标题相称的可复现强度证据，暂不进入核心池 |

### 本次实际运行证据

固定 seed `260826/260827`，分别交换席位对官方 `starter`；每项只运行一对，不能推断对强手胜率。

| Agent | seat 0 终局（agent / starter） | seat 1 终局（starter / agent） | 状态 | 总局耗时 |
|---|---:|---:|---|---:|
| Kaito v27 | 143428 / 3512 | 3568 / 144703 | 两局均 `DONE/DONE` | 3.39 s / 3.50 s |
| Flex Multi-Route | 143997 / 3788 | 3669 / 133486 | 两局均 `DONE/DONE` | 3.68 s / 3.46 s |
| Andrey V16 | 133842 / 3706 | 3545 / 128152 | 两局均 `DONE/DONE` | 3.41 s / 3.74 s |

这些工件下载到临时目录执行，没有复制进仓库。若后续引入，必须重新下载、固定版本并纳入许可证 manifest。

### 评测与模拟加速来源

- [Kaggriculture Rank Your Agent](https://www.kaggle.com/code/raykkretzschmar/kaggriculture-rank-your-agent)：`PULLED`。适合参考多对手、双席位评测，但 Notebook 展示的 2817.8/2990.4 是其提交页面分数，不等于该 harness 对任意模型的保证。
- [4000x environment speedup](https://www.kaggle.com/code/nikital7/4000x-environment-speedup-kaggriculture)：`PULLED`。作者提供 C++17 环境端口，并声明以逐步 state diff 验证 15/15 traces；这是训练吞吐工具，不是策略。比赛环境刚发生过平衡调整，采用前必须针对当前 `kaggle-environments==1.32.7` 和我们自己的多类 action traces 重新 bit-exact 验证，任何差异都应 fail closed。
- [官方 kaggle-environments Kaggriculture](https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/kaggriculture)：`API-OK`、Apache-2.0，是规则真值、最终 gate 和训练数据自生成的首要来源。

## GitHub / 通用框架候选

以下仓库的存在、默认分支和 SPDX 许可证已在 2026-08-26 通过 GitHub API 核验；“API-OK”不表示已安装或可直接处理 Kaggriculture 联合动作。

| 来源 | 许可证 | 能力 | 对组合动作的适配判断 | 推荐用途 | 代价/风险 |
|---|---|---|---|---|---|
| [Ray RLlib](https://github.com/ray-project/ray) | Apache-2.0 | PPO、自定义 RLModule、action masking、autoregressive action distribution、多 runner | **高**：官方文档明确区分默认独立 `TorchMultiDistribution` 与条件式 `P(a2|a1,obs)`；可实现我们所需的前缀 mask | 保留现有 PPO 训练框架，重写 distribution/decoder | Ray 依赖重、Windows worker 复杂；自定义联合 log-prob 和 entropy 必须单测 |
| [RL4CO](https://github.com/ai4co/rl4co) | MIT | autoregressive pointer/attention decoder、动态 infeasible mask、constructive/improvement policy | **最高的架构参考**：与“逐个选单位/任务/订单并更新 mask”同构 | 借鉴 decoder、candidate embedding、beam/multistart；不要直接把 TSP 模型当 farm policy | 需要自定义环境、奖励和 candidate schema；不是开箱即用的博弈 agent |
| [TorchRL](https://github.com/pytorch/rl) | MIT | PPO/CQL/IQL/DT/BC、TensorDict、`MaskedCategorical`、ActionMask transform | **高**：适合实现自定义条件 decoder；单一静态 mask 本身不能解决跨槽依赖 | 轻量自研 PPO/BC、masked distribution、部署前蒸馏 | 训练环节需自己组织较多组件；不能误把 per-slot mask 当联合约束 |
| [d3rlpy](https://github.com/takuseno/d3rlpy) | MIT | BC、DiscreteCQL、DiscreteBCQ、Decision Transformer 等离线 RL | **中**：若先把合法完整 bundle 编译成一个离散 candidate ID，则很适合；原生算法不会自动理解嵌套字典动作 | replay/self-play 数据上的 BC、DiscreteCQL 对照 | 需要 candidate compiler；IQL 当前表中不支持离散控制，不能宣传成离散 IQL 方案 |
| [SB3-Contrib MaskablePPO](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) | MIT | 简洁的 MaskablePPO，支持 Discrete/MultiDiscrete/MultiBinary | **中低**：选完整 bundle 时很好；直接套 27 个 MultiDiscrete 槽仍无法表达前缀依赖 | 快速 masked-bundle baseline | 不支持循环依赖 mask 的现成 AR 解码；不要作为最终结构 |
| [LightZero](https://github.com/opendilab/LightZero) | Apache-2.0 | AlphaZero、MuZero、Sampled/Gumbel/Stochastic MuZero、C++/Python MCTS | **中**：必须先有少量候选 joint bundle，否则 720 回合、巨大分支会让搜索不可行 | world-model/MCTS teacher、短视窗 market/terminal planner | 训练和接入成本最高；4060 上全季搜索不现实，提交 CPU 时还需蒸馏 |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel) | Apache-2.0 | 博弈算法、同时动作、legal action list/mask、NFSP 等 | **中**：适合对手池、评测抽象和合法性契约，不直接提供 Kaggriculture 策略 | league/tournament 设计参考、博弈指标 | 自定义游戏接入工作大；环境并非 OpenSpiel 原生 game |

关键官方资料：

- [RLlib autoregressive actions](https://github.com/ray-project/ray/blob/master/doc/source/rllib/rl-modules.rst)：默认多分量动作独立采样，在动作分量相关的环境里无法学习；官方给出自定义 autoregressive RLModule 路径。
- [RL4CO policies](https://rl4co.ai4co.org/docs/content/intro/policies/) 与 [constructive policy API](https://rl4co.ai4co.org/docs/content/api/networks/base_policies/)：AR decoder 逐步构造解，NAR 方法也需要动态 mask 不可行赋值。
- [TorchRL MaskedCategorical](https://docs.pytorch.org/rl/main/reference/generated/torchrl.modules.MaskedCategorical.html)：被 mask 的概率归零后重新归一化。
- [SB3 MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)：可做离散/多离散 baseline，但需要我们自己把跨槽条件编译进动作或环境。
- [d3rlpy](https://github.com/takuseno/d3rlpy)：官方算法表包含离散 BC、DiscreteCQL、DiscreteBCQ 和 Decision Transformer。

## Hugging Face 核验

### 公开搜索结果

2026-08-26 通过 `https://huggingface.co/api/models?search=kaggriculture` 查询，返回 **0 个模型**。因此没有证据表明 Hugging Face 上存在可直接下载、已针对 Kaggriculture 训练且许可证清晰的通用 policy checkpoint。

[Hugging Face Decision Transformer](https://huggingface.co/docs/transformers/main/model_doc/decision_transformer) 可作为架构和训练 API 参考。`edbeeching/decision-transformer-gym-*` 两个抽查 checkpoint 的 API 可访问，约 19.8 MB，但模型卡 API 没有 license 字段，且权重针对 Hopper/HalfCheetah 连续控制，状态和动作空间完全不同。结论是：**可以复用 Apache-2.0 Transformers 中的 DT 实现思路，从我们的合法轨迹重新训练；不要迁移这些 Gym 权重。**

### 发现但必须隔离的数据

| 来源 | 可见信息 | 许可证/访问 | 判断 |
|---|---|---|---|
| [ThanThoai9x/kaggriculture-rsp-decisions](https://huggingface.co/datasets/ThanThoai9x/kaggriculture-rsp-decisions) | 743,899 条 processed rows，1,774 个 public episodes，约 55.4 GB；页面要求先接受条件 | 卡片未给标准 SPDX，明确要求仅限接受比赛规则的参与者并禁止未经审查再分发 | `QUARANTINE`。不能作为“公开自由数据”；若团队确有合法访问权，也要单独做规则/权利审查和 lineage-safe split |
| [KiroSamurai/kaggriculture-il](https://huggingface.co/datasets/KiroSamurai/kaggriculture-il) | 8,268 episodes、约 3.92 GB，含 4.9 MB `width=128/depth=4` BC checkpoint，声称 conditional units/market | `license: other / kaggle-competition-replays`；卡片明确“Private on purpose / do not redistribute”；其配套 GitHub URL 本次返回 404 | `QUARANTINE`。conditional market 架构有研究价值，但代码不可核验、数据许可受限、checkpoint 不能进入训练或提交 |

HF 搜索的价值在这里主要是发现“数据治理风险”，不是找到可直接使用的大模型。

## 模型优先级矩阵

评分：5 最佳；“联合合法性”越高，越能避免重复订单/独立槽冲突。显存与延迟为相对工程估计，必须以本机 profiler 和 Kaggle CPU gate 为准。

| 优先级 | 模型/策略 | 联合合法性 | 多用途性 | 4060 训练 | Kaggle CPU 推理 | 主要角色 | 判断 |
|---:|---|---:|---:|---|---|---|---|
| P0 | 候选 bundle 编译器 + masked selector（Transformer/MLP/八模型 scorer 都可） | 5 | 5 | 低 | 极低 | BC、PPO、CQL、ensemble teacher、部署 | **先做**。最快修复 27 槽结构错误，所有训练路线共享同一合法动作接口 |
| P0 | 残差 Transformer Actor-Critic + AR pointer/candidate decoder | 5 | 5 | 中 | 中，需蒸馏/量化 | 主 RL、BC、league self-play | **主线**。保留大模型和多层 Transformer，但动作必须条件式生成 |
| P1 | Set/GNN 单位-任务编码 + pointer/matching decoder | 5 | 4 | 中 | 低到中 | 可变 farm hands、路径/任务分配 | **强辅助头**。天然处理单位/任务集合和置换不变性；可与 Actor 共用 encoder |
| P1 | 规则路线池 + learned router / residual policy | 5（规则层保证） | 5 | 低 | 极低 | teacher、opponent pool、fallback、线上部署 | **奖牌优先的稳健路线**。让模型只学路线选择、局部残差和市场顺序，避免从零生成全季计划 |
| P1 | Decision Transformer（从本地合法轨迹训练） | 4 | 4 | 中 | 中 | offline sequence baseline、蒸馏 teacher | 适合长时序和 return conditioning；仍需 AR 合法 decoder，不能直接用 Gym 权重 |
| P1 | DiscreteCQL / DiscreteBCQ over bundle IDs | 5 | 4 | 低到中 | 低 | 离线 RL、比 BC 更保守的价值改进 | 适合已有大量合法轨迹；动作若仍是独立槽则不推荐 |
| P2 | MaskablePPO over full bundle IDs | 5 | 3 | 低 | 低 | 快速强 baseline、训练管线回归 | 能验证 reward/league 是否健康；不是最终高容量模型，但应作为结构对照 |
| P2 | Sampled MuZero / LightZero over bundles | 4 | 3 | 高 | 高，通常需蒸馏 | planning teacher、终局/市场短视窗搜索 | 仅在快速模拟器验证后做研究支线；不应阻塞 P0/P1 |
| P3 | 当前 27 槽独立头，无条件 mask | 1 | 2 | 已可运行 | 中 | 仅故障对照 | **停止扩容**。重复 `BUY_SEED` 与长期 `PASS` 是结构性故障，不是容量不足 |

## 推荐的多样化组合

### 训练与部署组合

1. **共同 encoder**：多层 Pre-LN 残差 Transformer；board、farmer/hands、market、shop、day/step 分成类型化 tokens。
2. **任务分配头**：set/pointer decoder 为每个单位逐步选合法 task candidate，选择后更新占用 mask。
3. **市场头**：以 `order_type → product → bounded quantity` 的条件 token 生成 0–10 个 order blocks；每个 block 后更新现金/库存/重复项 mask，以 `EOS` 结束。
4. **价值头**：terminal win probability 为主，coin margin、unbanked inventory、starvation/overflow risk 为辅助。
5. **teacher scorer**：八模型 ensemble 只在合法 candidate/bundle 上打分，提供 pseudo-label、risk estimate 和 PPO warm start，不直接输出未经校验的 27 槽动作。
6. **部署层**：蒸馏到较小 Actor 或 route+residual policy；最终 `compile_and_validate()` 无条件保留，异常时回退到明确安全规则。

### 对手/教师池的八个槽位

| 槽位 | 来源族 | 提供的差异 |
|---:|---|---|
| 1 | 官方 `starter/random` | 协议 curriculum、低强度探索 |
| 2 | Kaito v27 | 当前 meta 固定路线与稀疏市场排序 |
| 3 | Flex Multi-Route | shop-conditioned 多路线决策 |
| 4 | Andrey V16 | 同回合市场 slot preemption |
| 5 | lonespear | 算法式闭环生产/任务分配，不是单一 replay tape |
| 6 | Seyamalam（只取许可清晰文件） | 恢复、市场和 regression 工具族 |
| 7 | 我方历史 BC/PPO checkpoint | 当前模型分布与抗遗忘 |
| 8 | 定向 exploiters | starvation、shed overflow、cash deadlock、terminal liquidation、重复订单压力 |

采样前按以下 fingerprint 聚类，避免同源复制：前 120 步 action hash、每日作物/动物/雇工曲线、market order n-gram、最终生产组合和公开路线来源。训练/验证/测试以 `(lineage, episode, seed)` 为原子分组。

## 建议实施顺序

1. **立即**：为 market/actions 写 joint legality property tests，重现“10 个重复 BUY_SEED”并要求 compiler 合并或拒绝；将 `PASS` 改成单一 `EOS/fallback`，而不是每槽默认类别。
2. **第一版**：生成 32–128 个完整 bundle，训练 masked-bundle BC/PPO baseline；它应该先稳定击败 starter 和现有坍缩 checkpoint。
3. **数据**：仅用固定官方模拟器、许可明确的固定策略和我方 self-play 重新生成轨迹；先不碰 HF replay 数据。
4. **主模型**：把现有多层残差 Transformer 的输出替换为 AR candidate pointer decoder；保留 checkpoint encoder，动作头重新训练。
5. **多样训练**：BC 混合多个 lineage，之后对许可清晰的固定对手与历史 checkpoint 做 PFSP；评测必须交换席位。
6. **离线对照**：在同一 bundle 数据上训练 DiscreteCQL/BCQ 与 Decision Transformer，和 PPO 做 paired-seed 比较。
7. **研究支线**：只有在 C++ 快速环境对当前引擎与我方动作全量 bit-exact 后，才尝试 Sampled MuZero 或大规模搜索。

## 风险与合规边界

- 比赛仍活跃，Notebook 的 Public/Best score 会变化；抓取值只能作为时间戳证据，不能写成最终名次。
- 公共 replay 可观察不等于可自由再分发。原始 replay、HF gated/restricted 数据不得提交到本仓库，也不得在未审查时进入可发布权重。
- Kaggle Notebook 页面许可证与其中再次引入的第三方路线是两层问题；必须保留原作者、Notebook slug/version、输出 hash 和第三方 notice。
- 同一公开路线被多人包装会造成严重 lineage leakage。不能按用户名随机切分。
- 只用历史 leaderboard replay 训练会过拟合 meta；至少保留自生成 unseen seeds、算法式 closed-loop opponents 和时间后移 holdout。
- 公开 replay opponent 在 counterfactual 对局中不会响应，适合 regression screen，不等于 live win rate。
- 大模型训练可使用 GPU，但提交环境是严格 CPU/体积/时限约束；Actor 的 CPU P95、首回合冷启动、archive 大小必须作为晋级 gate。
- 最终动作校验器不得读取对手私有字段、未来 shop、episode/team identity 或任何部署时不可见信息。

## 本次工件证据

下列是临时目录中通过 Kaggle CLI 拉取的文件摘要，用于证明本次调研不是只看搜索摘要；它们没有复制进仓库：

| 工件 | SHA-256 |
|---|---|
| Kaito v27 notebook | `CC61BB10378C555E2CB3090BDE2DD8EE442C34D9A8EE98B7AB918DD3ACB3DB8D` |
| Kaito v27 output `main.py` | `F48C21166EAC68D1B05A401F04F94A2EB6154E65415AF64893672365FF33C7B8` |
| Flex Multi-Route notebook | `50CC0B06B60F885D36B588609D6E7B165E96273625F4C57F402B9A3E13A570E6` |
| Flex output `main.py` | `8AC34ABCE129CF5C9456776C90EDF7D2233B3A280BBDCF7622628825EF3669A0` |
| Andrey notebook | `C5ACAEB646801BBC06C35915345147B82C828DD47E01FA77D0CC9BA379ADE39B` |
| Andrey output `main.py` | `0827987DAB3F52E1F50661C355B42F289B8E305D90A263B16F3054AAECFE943C` |
| Kaito v21.1 notebook | `32471F25E3BC3C206A0539D8FD08AB3DB52DB8460C6AB90CAC75249C6819E2D5` |
| Rank Your Agent notebook | `F326EF39B52CA868B1AABC36A8787EFAD03740D00BF84CD717766E2983FC281E` |
| 4000x simulator notebook | `61B84DF81F0E5348217EB7A91A44390214FD56456D38CDCEA6B398E257272681` |

这些 hash 只对应 2026-08-26 拉取到的最新版本；后续使用必须重新固定 Kaggle `scriptVersionId`，不能只依赖可漂移的 slug。
