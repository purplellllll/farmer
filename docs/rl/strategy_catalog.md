# 策略目录与训练标签

策略目录的目标不是把整局压成一个“动作类别”，而是提供多样教师、对手类型、辅助任务和失败恢复标签。每个策略实现都应先经过合法动作率、720 回合完成率、双席位对称性和末期现金化四个门槛。

## 1. 生产物流

| 策略 ID | 决策内容 | 轨迹/辅助标签 | 主要风险 |
|---|---|---|---|
| `crop_fast_cycle` | 小麦/胡萝卜快速周转 | 作物阶段、预计收获日、单位动作成本、现金回收期 | 低价拥堵、重复浇水 |
| `premium_crop` | 草莓/甜瓜的高价值长周期路线 | 最晚种植日、肥水覆盖、价格敏感度 | 终局未回本、市场踩踏 |
| `livestock` | 鹅/牛/羊、建筑、喂养、CARE、采肥 | 未来 1/3/7 日小麦缺口、断粮风险、产品日均产出 | 维护动作不足、动物逃跑 |
| `labor_scheduler` | farmer/hands 的任务分配与最短路径 | task ID、deadline、travel cost、collision count | 路径占用压倒名义利润 |
| `capacity_guard` | shed 100 容量、随身库存与 DROP | 预计日末溢出、需要清仓数量 | 产品无声丢失 |

## 2. 市场

| 策略 ID | 决策内容 | 轨迹/辅助标签 | 主要风险 |
|---|---|---|---|
| `sell_execution` | 分批卖出、队列顺序、相对价格冲击 | 执行前后库存/价格、realized revenue、slippage | 同回合与对手交错成交 |
| `town_demand` | 根据已公开商店估计消耗速度 | 每产品公开 demand rate、库存恢复时间 | 禁止使用未来商店解锁 |
| `opponent_exposure` | 从公开农场估计未来供给 | 对手公开作物/动物数量、成熟窗口 | 不能使用对手私有库存 |
| `capital_timing` | 卖货为扩地、动物、种子、雇工融资 | 现金缺口、回本截止日 | 卖得过早或投资过晚 |

## 3. 专家路由

- 每日或公开商店解锁时，在 `crop_fast_cycle`、`premium_crop`、`livestock`、`mixed` 之间选择，而非每回合抖动。
- 路由输入只能包含当前 observation 和合法历史记忆；标签包含 route ID、切换原因和预计回报。
- XGBoost/LightGBM 一类表格模型适合先做路由教师，Transformer Actor 学习时间依赖；路由器不能绕过动作 mask。
- 加入保守/均衡/高波动三种风险模式，领先时以胜率为目标降低方差，落后时允许高上限路线。

## 4. 恢复

按优先级生成独立 recovery 轨迹：

1. `animal_starvation`：连续未喂风险，补麦、PICKUP、FEED。
2. `crop_death`：新种作物当天未浇与连续缺水风险。
3. `shed_overflow`：在日末前出售或减少 DROP 损失。
4. `weed_repair`：清理随机杂草并让局部路线重新对齐。
5. `cash_deadlock`：没有资金完成维护/生产时的最小变现。
6. `malformed_or_timeout`：返回与当前 hand 数一致的 PASS，不传播异常。

恢复动作由安全规则拥有否决权；训练时既保存专家修正，也保存被否决动作形成 preference pair。

## 5. 终局

- 建立每种资产的 `last_profitable_buy_day` 和 `last_harvestable_plant_day`。
- 最后两天停止无回报资本支出，单位回到 shed，先 DROP 再按可执行队列 SELL。
- 终局 reward 以 `win/draw/loss = +1/0/-1` 为主，金币差只作为 critic 辅助头。
- 创建 `terminal_inventory_value`、`unbanked_inventory`、`last_executable_sale_step` 标签，专门惩罚“资产很多但银行没钱”。

## 6. 自博弈

建议对手占比初值：

- 10% 官方弱 curriculum；
- 30% 明确许可的不同风格规则专家；
- 40% 历史强 checkpoint；
- 20% 当前模型、镜像与针对性 exploiter。

先使用 `population.json` 的 PFSP 风格权重；胜率接近 50% 的对手优先，同时保留最低采样概率避免遗忘。每个 seed 交换席位；晋级模型至少报告对阵矩阵、Wilson 区间、非法动作率、超时率和终局未变现库存。

## 训练课程

1. 官方 pass/random/starter 验证协议与动作合法性。
2. 明确许可的专家互战，生成 BC 数据和辅助价值标签。
3. BC 后对固定专家 PPO，先用中间 potential shaping，逐步退火。
4. 冻结 checkpoint 池自博弈，定期加入不同策略 exploiter。
5. 在永不参与训练的 seed/对手 holdout 上双席位选模。
6. 蒸馏、int8 量化，仅部署 Actor；Critic 和训练时特权信息不得进入提交。
