# 本地双座位策略测评

该工具在固定版本的官方 Kaggriculture 环境中，用同一批 seed 交换候选策略的 0/1 席位。它输出逐局 JSON/CSV，并汇总胜、平、负、平均 outcome、现金差、分席位结果、动作分布、策略调用 P50/P95、策略故障率与环境故障率。坍缩诊断还包括同一步重复市场订单率、跨订单库存/预算冲突、最长纯 PASS 空转、联合动作多样性和前 72 step 的最大现金消耗。

## 策略入口

- `builtin:starter/pass/random`：官方 starter、绝对下限和可复现随机下限。
- `builtin:crop_fast/premium/expansion/market/defense`：生产、长周期、资本扩张、动态市场和低风险五种不同风格的本地规则对手。这些是课程/回归基线，强度必须由本工具实测，不能直接视为排行榜强策略。
- `checkpoint:path`：加载 `farmer-rl-bc/v1` 或 `farmer-native-ppo/v1`。目录或 `latest.json` 会解析到最新 checkpoint；默认只用 CPU，以免抢占正在训练的 4060。只有显式写 `checkpoint:path@cuda` 才会用 GPU。
- `python:path/to/agent.py:agent`：加载经许可证审核的 Kaggle Code、write-up 或 GitHub/Hugging Face 策略。函数只接收该 acting seat 的 observation。

查看内置策略：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\local_strategy_benchmark.py --list-policies
```

## 快速验证与正式评测

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\local_strategy_benchmark.py --config configs\eval\smoke.json

.\.venv\Scripts\python.exe scripts\local_strategy_benchmark.py `
  --config configs\eval\default.json `
  --candidate "checkpoint:artifacts\local-4060-gated" `
  --output-dir artifacts\eval\rl-latest
```

正式配置是 8 个 seed × 7 个对手 × 双席位，共 112 局。先用 smoke 配置验证依赖和动作形状，再提高 seed 数。模型 checkpoint 默认串行运行，避免每个进程重复加载大模型；纯规则策略可用 `--workers 2` 或更高并行。

输出：

- `benchmark.json`：完整配置、总表、按对手、按候选席位和逐局诊断。
- `games.csv`：适合表格分析的一局一行结果。

## 故障与公平性边界

策略异常、动作 envelope/hand 数错误或超时统一降级为形状正确的 PASS。超时后该策略在本局禁用，避免线程持续堆积；未知第三方策略建议以 `workers>1` 做整局进程隔离。Python daemon 线程能隔离普通阻塞，但不能安全抢占长期持有 GIL 的原生扩展。

官方引擎会把部分状态非法动作静默转为 no-op，因此 `invalid_actions` 指结构/hand 数非法，不等价于精确的状态非法率。内置随机和 checkpoint 通过仓库的保守候选动作集约束；外部 agent 的精确合法动作率若要审计，需要补充官方引擎级 action trace。

`duplicate_market_order_rate` 把同一联合动作内重复的精确订单计为重复；`resource_conflict_orders` 检测累计超卖、重复扩满土地等跨槽冲突；`overbudget_orders` 采用“不预支同一步卖货收入”的保守现金预算。因此它们是快速回归告警，不等同于官方引擎最终判罚。若模型出现“每步十个相同 BUY_SEED、随后长期 PASS”，重复率、前期现金消耗、最长空转和联合动作多样性会同时报警。

所有模型比较必须固定：环境 wheel、配置、seed 集、对手版本/hash、双席位协议和 timeout。调参/训练 seed 不得混入最终 holdout 测评。
