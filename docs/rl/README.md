# Kaggriculture 强化学习工作区

这里提供的是可审计的训练底座，不包含伪造数据、下载后的比赛 replay，也不声称已经完成线上训练。核心边界如下：

- Actor 每一步只接收官方给当前 `player` 的 observation；轨迹构造会拒绝席位不一致的数据。
- 数据优先由固定版本的官方模拟器与明确许可的本地策略闭环生成，并且对每个 seed 交换双方席位。
- 公开 episode/replay 可以用于本人有权访问时的本地分析，但在许可证和比赛条款逐项确认前不得进入训练 allowlist，也不得提交到仓库。
- BC 先学习保守合法候选集，PPO 再在冻结 checkpoint 槽中自博弈；真实选模指标是交换席位后的成对 seed 胜率。

## 目录

- `sources.md`：官方规则、模型论文、公开策略仓库、许可证与固定版本。
- `strategy_catalog.md`：生产物流、市场、专家路由、恢复、终局、自博弈六类策略及标签。
- `architecture.md`：模型、轨迹、动作掩码、BC→PPO 和已知工程缺口。
- `configs/rl/data_manifest.schema.json`：任何训练集进入流水线前必须满足的 manifest。
- `src/farmer_rl`：无外部依赖的核心接口，以及按需加载的 PyTorch/RLlib 训练入口。

## 最小命令

在仓库根目录执行：

```powershell
$env:PYTHONPATH = "src"
python -m farmer_rl.train validate --config configs/rl/ppo.json
python -m unittest discover -s tests/rl -v
```

BC 和 PPO 属于显式训练操作，不会在导入包或运行单元测试时下载数据：

```powershell
python -m farmer_rl.train bc `
  --config configs/rl/ppo.json `
  --input artifacts/rl/trajectories/train.jsonl `
  --output artifacts/rl/checkpoints/bc.pt `
  --epochs 3

python -m farmer_rl.train self-play `
  --config configs/rl/ppo.json `
  --bc-checkpoint artifacts/rl/checkpoints/bc.pt `
  --iterations 20
```

所示 `artifacts/` 路径只是本地运行约定，本仓库没有创建或写入原始比赛 replay。启用采集前，先复制 `data_manifest.example.json`，替换全部占位值并验证来源状态。
