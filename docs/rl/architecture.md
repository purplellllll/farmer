# RL 架构与实现边界

## 数据流

```text
当前席位 observation
  -> acting-seat 校验
  -> ObservationTokenizer（自身农场始终排在对手前）
  -> Pre-LN Residual Transformer
  -> 27 个动作槽：1 farmer + 16 hands + 10 market
  -> 每槽至多 64 个动态候选及 mask
  -> JointActionCodec
  -> 官方 action dict
  -> 官方双玩家同步 step
  -> 两条各自 observation 的 Transition
```

模型默认 8 层、`d_model=256`、8 heads、FFN 1024。Actor 为每个动作槽输出候选 logits，Critic 从 masked mean pooled token 预测 value。PyTorch 的 `TransformerEncoderLayer(norm_first=True)` 自带层内残差；最终 LayerNorm 后分别进入 Actor/Critic head。

## 席位与信息安全

`KaggricultureEnv` 只从官方两个 AgentState 分别取 `state.observation`，若列表位置与 `observation.player` 冲突立即报错。collector 在调用策略、保存当前 observation 和 next observation 时都做 defensive copy。`Transition` 再次验证两端 `player`。

Actor 输入允许：双方公开 farm、双方银行、本人 private、公共 market、当前已解锁 town shops 和过去合法 observation 历史。禁止：对手 private、未来商店、episode seed、未来动作。若以后实现 centralized critic，特权张量必须使用独立编码器和独立序列缓存，不能复用或导出到 Actor。

## 动作候选和 mask

`CandidateGenerator` 根据单位位置、当前 tile、本人种子/库存和公开状态生成候选；固定 `PASS`/`NO_ORDER` 处理 padding。`JointActionCodec` 把固定形状的索引还原为官方的可变 hand 数和有序 market 队列。候选集是有意保守的，BC 遇到集合外专家动作会计入 `skipped`，不会偷偷改成 PASS。

目前 market 队列 mask 是“单槽在当前 observation 下可行”，同一物品被多个槽重复出售仍需要自回归资源预留。正式长跑前应把市场 decoder 升级为逐槽更新资金、shed 数量和订单效果的 autoregressive mask；在此之前用 deterministic safety compiler 对整队列二次过滤。

## BC 到自博弈 PPO

`farmer_rl.train bc` 读取 `Transition` JSONL、重新生成候选并监督每槽索引，输出含 format、模型配置和 state dict 的 checkpoint。`farmer_rl.train self-play` 延迟导入 Ray，建立两个 agent ID；每局根据 episode ID 交换 learner 席位，并在若干冻结 opponent policy 之间轮换。每到 promotion interval，把 learner 权重复制进一个 frozen slot。

这只是可运行的 population checkpoint 骨架，不是完整 league：

- 还未实现按对阵结果动态选择 RLlib policy slot；独立 `OpponentPool` 已提供 PFSP 权重逻辑。
- 还未实现专门 exploiter、TrueSkill/Bradley–Terry 选拔和分布式 rollout 容错。
- 默认 value 只用合法 actor observation，不存在信息泄漏；privileged critic 是后续显式功能。
- RLlib API 易变，入口固定在 Ray 2.x old ModelV2 seam。首次训练前必须创建依赖 lockfile 并做 2 回合、720 回合、断点恢复三个 smoke gate。

## 奖励

第一版保持官方终局奖励，避免先把金币差误当比赛目标。课程期可增加 potential-based shaping：现金、保守折价后的可变现库存、未来成熟产出、维护风险和终局未变现惩罚；shaping 系数必须退火，checkpoint 只按无 shaping 的双席位胜率晋级。

## 部署边界

Kaggle 提交最终只保留 Actor、tokenizer、候选生成器和 safety compiler。训练期 Critic、optimizer、Ray、数据清单和对手池都不打包。正式提交前验证 100 MiB、1.6 vCPU、6.5 GiB RAM 和逐回合延迟；默认 8 层模型是否满足限制尚未测量，不能把训练可运行等同于线上可部署。
