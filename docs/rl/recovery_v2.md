# GPU recovery v2 与 CPU 弱表现诊断

## 结论

`artifacts/local-4060-gated` 不是数值溢出或 CUDA 故障，而是策略坍缩。第 1375 轮
checkpoint 没有非有限参数，但最近 50 轮胜率为 0、平均分差约 -3366；策略池中的
初始快照累计只被当前 learner 战胜 35/1731 次，晋升快照只被战胜 2/1040 次。
这说明继续从第 1375 轮恢复 optimizer/model 没有意义。

主要原因：

1. 十个市场槽原先独立采样，重复或超资源动作随后被安全编译器删除；PPO 却仍按
   删除前的索引计算 likelihood ratio，实际执行动作与训练动作不一致。
2. 720-step 配置每轮约 2876 transition，但只来自 4 局完整游戏；八层模型每轮更新
   4 epoch，胜负信号方差过大且极易过拟合单轮轨迹。
3. shaping 在终局仍保留 `potential(next_state)`，使优化目标包含终局现金差，错误奖励
   不投资和长期 PASS，而不是只优化官方胜负。
4. checkpoint 晋升只要求战胜历史快照；第一个晋升模型没有经过 scripted 基线门槛，
   因此策略池会接受只会利用旧弱模型、却不会经营农场的策略。
5. 原 BC checkpoint 只有 5752 个同质示例；扩展课程短训又只抽到 261 个训练样本，
   validation joint accuracy 为 0。更关键的是 `DiverseCurriculumPolicy` 的设计目标是覆盖
   25 类动作，而不是按胜率筛选专家轨迹；它适合训练动作语法，不能被当作强策略 teacher。

## recovery v2 修复

- Transformer 仍保持 8 层、`d_model=256`、8 heads 和残差连接。
- 固定 27 槽输出改为前缀条件选择：后续槽会预留共享种子、shed、现金、市场库存、
  雇工和土地成本；一个 `NO_ORDER` 后其余市场槽强制结束。
- PPO checkpoint 记录并重放当时真正使用的条件 mask，policy log-probability 与 entropy
  按仍有选择空间的槽归一化。
- 终止状态 potential 强制为 0；中间 shaping 使用本人可见资产的保守清算代理，但
  telescoping 后不改变终局胜负目标。
- 晋升同时要求历史快照窗口和 scripted 窗口过线。
- recovery 阶段使用 144 step（6 天，足以完成 WHEAT/CARROT 生产周期），每轮固定
  2288 transition，约 16 局；学习率 `1e-5`、2 update epochs、clip `0.1`。

启动命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_gpu_rl_recovery_v2.ps1 -Iterations 600
```

训练目录为 `artifacts/local-4060-recovery-v2`。不得从坍缩的
`iteration_001375.pt` 恢复；首次运行从 `checkpoints/starter_bc_v5.pt` 重新建立干净的
optimizer 和对手池。完成 144-step recovery 后，必须通过 720-step 双席位 holdout，
再逐阶段扩展到 288/720 step。

## CPU v2 为什么仍弱

CPU v2 的 24-step episode 只有一个游戏日，WHEAT/CARROT 最早也要第 2 天才有产出。
因此买种子、建造和雇工在截断终局都只表现为现金损失，PASS 反而是短局最优响应。
直接对照也证实了这一点：同一 seed、交换席位后，官方 starter 在 24-step 两局均输给
纯 PASS，平均分差 -40，恰好对应尚未回收的种子投入。
训练还只有每轮 3 局，BC 扩展短训的 261/56 条 train/validation 样本不足以改变原模型；
课程数据本身又以动作覆盖而非获胜质量为目标。所以第 15 轮仍对 scripted 为 0/31，最长
PASS streak 接近整局。

CPU 后续不应继续 24-step PPO。可选路径是：先扩大并修复 BC teacher，再在独占内存时
使用至少 144 step；或者只让 CPU 负责离线 BC/评测，把在线 rollout 留给 GPU。
