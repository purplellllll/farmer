# CPU RL v2：约束解码、扩展课程与隔离验证

CPU v2 不会恢复或覆盖 `artifacts/local-4060-gated`。它从旧 BC 权重迁移，但使用独立目录 `artifacts/cpu-rl-v2`，并把 PyTorch、OpenMP 和 MKL 限制为 4 个线程；Windows 后台进程使用 `BelowNormal` 优先级，避免和仍在运行的 4060 训练争抢调度资源。

## 旧训练诊断

用机器可读审计脚本同时检查 `metrics.jsonl`、`latest.json` 指向的 checkpoint、参数有限性和本地 benchmark：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\audit_rl_training.py `
  --run-dir artifacts\local-4060-gated `
  --benchmark artifacts\eval\rl-latest-collapse-24\benchmark.json `
  --output artifacts\cpu-rl-v2\diagnostics\gpu_run_audit.json
```

旧训练的进程和参数本身是健康的，但近期胜率为零、快照没有持续晋级，并出现十槽重复购买和长 PASS streak；因此 CPU v2 只迁移 BC checkpoint，不从已经坍缩的 PPO optimizer/state 恢复。

## 数据和泄漏边界

输入为 `artifacts/ensemble-retrain-v2/data/curriculum_v2.jsonl`：20 个 episode/seed、28,760 个 observation/action 记录、25 类动作覆盖。BC 读取器同时支持原始完整 transition 和 curriculum-v2 observation/action schema。

数据先按 `(episode_id, seed)` 整组分配训练或验证，再在组内用 `(episode, seed, seat, step)` 的稳定哈希抽样。任何 seat 或 step 都不能跨越训练/验证边界；checkpoint 内记录 group 数和 overlap。短训示例：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m farmer_rl.train bc `
  --config configs\rl\cpu_v2.json `
  --input artifacts\ensemble-retrain-v2\data\curriculum_v2.jsonl `
  --output artifacts\cpu-rl-v2\bc\curriculum_v2_short.pt `
  --epochs 1 --batch-size 8 --device cpu `
  --initial-checkpoint checkpoints\starter_bc_v5.pt `
  --validation-fraction 0.2 --split-seed 20260826 `
  --record-sample-modulus 90 --max-examples-per-group 24 `
  --torch-threads 4
```

`.pt`、原始数据、逐步 metrics 和 benchmark 位于 gitignored 的 `artifacts/`；仓库只同步配置、实现、测试、文档及不含模型权重的小型结果摘要。

## 约束解码 v2

模型仍是 8 层、256 维残差 Transformer Actor-Critic。27 个固定输出槽仍用于保持旧 checkpoint 的形状兼容，但最终动作必须经过按序编译：

- 对市场订单按 `(operation, item)` 跨槽去重；
- 依次预留现金、棚库存、市场库存、shed capacity、雇工 Fibonacci 成本和土地成本；
- 多个单位从棚中 `PICKUP` 时共同预留库存；
- curriculum-v2 的三粒种子、1/4 个产品和最多 6 个 pickup 数量均可无损编码；
- 不预支同一步卖货收入。

这是一层确定性安全编译器，并非真正自回归 Transformer decoder；后续替换模型头时，本约束仍应作为提交前最后一道防线。

## CPU 自博弈与评测

先运行 24-step smoke，确认反向传播、checkpoint 和本地环境闭环。官方环境会在内存中保留 episode 历史；16 GiB 主机并行运行 GPU 训练时，实测 720-step CPU rollout 占用约 7.31 GiB 私有内存。144-step 恢复试跑也升至约 7.7 GiB，并使系统可用内存降到 0.49 GiB。因此安全默认值采用 24-step 课程 rollout、每轮至少 48 个目标 transition，并继续依赖 potential shaping；完整 720-step 只用于阶段 checkpoint 的低频 holdout 测评：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_cpu_rl_v2.ps1 -Iterations 50
```

若任务因 Codex 会话或机器重启而中断，从 `latest.json` 指向的 checkpoint
继续时，`-Iterations` 表示还要新增的轮数。例如第 20 轮后继续到第 50 轮：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_cpu_rl_v2.ps1 `
  -Iterations 30 `
  -Resume artifacts\cpu-rl-v2\self-play-24\checkpoints\iteration_000020.pt
```

恢复时不能再传入 BC warm-start；脚本会在 `process.json` 中分别记录
`bc_checkpoint` 与 `resume_checkpoint`，日志继续写入同一条单调迭代序列。

进程信息写到 `artifacts/cpu-rl-v2/self-play-24/process.json`。`latest.json` 只有在首轮迭代和 checkpoint 真正完成后才会出现；仅有 PID 不代表训练完成。若在更大内存的机器独占训练，可把 `environment.configuration.episodeSteps` 恢复为 720，并把 `native.train_batch_steps` 设为 719。

checkpoint 必须通过双座位本地评测后才能进入提交候选：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\local_strategy_benchmark.py `
  --candidate "checkpoint:artifacts\cpu-rl-v2\self-play-24" `
  --opponent "builtin:starter" `
  --seeds 20261001 `
  --episode-steps 720 --workers 1 --timeout 10 `
  --output-dir artifacts\eval\cpu-rl-v2
```

最低验收字段包括双座位胜负/分差、重复市场订单、预算或库存冲突、最长 PASS streak，以及策略调用 P50/P95。训练 seed `20260900..20260919` 不得进入最终 holdout。
