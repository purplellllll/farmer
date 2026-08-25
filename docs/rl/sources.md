# 数据、策略与算法来源审计

核对日期：2026-08-25。GitHub 来源必须按表中的 40 位 commit checkout，不能依赖会漂移的 `main`/`master`。复核远端版本的方法是 `git ls-remote <repo-url> HEAD`；下载后再用 `git rev-parse HEAD` 和 `Get-FileHash -Algorithm SHA256 <artifact>` 固定实际工件。

## 可直接使用的官方来源

| ID | 内容 | 固定版本 | 许可证 | 当前用途 |
|---|---|---|---|---|
| `kaggle-environments-official` | [官方环境、规则与内置 pass/random/starter](https://github.com/Kaggle/kaggle-environments/tree/28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c/kaggle_environments/envs/kaggriculture) | 规则源码 commit `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`；训练 wheel 固定 `kaggle-environments==1.32.7`，PyPI SHA-256 `2a1bb862ad2d6463080f80f6a766f46d94b53fd57168cfeddb9857fc3dbc4c8f` | [Apache-2.0](https://github.com/Kaggle/kaggle-environments/blob/28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c/LICENSE) | 模拟器、规则、弱 curriculum、闭环自生成轨迹 |
| `competition-pages` | [官方 Overview / Evaluation / FAQ](https://www.kaggle.com/competitions/kaggriculture/overview) | 访问时快照；赛制页面会更新，不作为训练样本 | Kaggle 页面条款 | 确认 720 回合、胜负排名、100 MiB 与 1.6 vCPU / 6.5 GiB 约束 |
| `competition-download-local` | Kaggle CLI 下载的 `AGENTS.md`、`README.md` | 本机 `kaggriculture.zip` SHA-256 `91298b47c5dd34500177e6cf6c41a9c2ca254804f893d135a7c06c3d920ee739` | 以比赛规则和官方仓库 Apache-2.0 文件为准 | 规则核对；不能把 hash 当作未来下载仍相同的承诺 |

固定官方源码示例：

```powershell
git clone https://github.com/Kaggle/kaggle-environments.git vendor/kaggle-environments
git -C vendor/kaggle-environments checkout 28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c
git -C vendor/kaggle-environments rev-parse HEAD
```

## 明确许可的公开策略仓库

这些仓库可以作为策略研究对象或本地闭环对手，但“仓库有许可证”不自动覆盖其再次引入的第三方文件。采集前仍须把具体文件 SHA-256、作者、许可证及 notice 写进数据 manifest。

| ID | 仓库与策略价值 | 固定 commit | 许可证 | allowlist 判断 |
|---|---|---|---|---|
| `cok-kaggriculture` | [COK-ZhangZiliang/Kaggriculture](https://github.com/COK-ZhangZiliang/Kaggriculture/tree/c3e2e89a06c9f7874f3ecf73163b07e00e3517e8)：公开商店路由、恢复控制、市场排序、末期清仓 | `c3e2e89a06c9f7874f3ecf73163b07e00e3517e8` | [Apache-2.0](https://github.com/COK-ZhangZiliang/Kaggriculture/blob/c3e2e89a06c9f7874f3ecf73163b07e00e3517e8/LICENSE) | `allow`，同时保留 `THIRD_PARTY_NOTICES.md` 和 per-file notice |
| `lonespear-kaggriculture` | [lonespear/kaggriculture](https://github.com/lonespear/kaggriculture/tree/774b26093ccf4246525517d48420349b841b6e50)：生产/动物/市场路线、并行 sweep、双席位 league | `774b26093ccf4246525517d48420349b841b6e50` | [MIT](https://github.com/lonespear/kaggriculture/blob/774b26093ccf4246525517d48420349b841b6e50/LICENSE) | `allow`，保留版权声明；公开 ladder 结果只是历史证据 |
| `seyamalam-kaggriculture` | [Seyamalam/Kaggriculture](https://github.com/Seyamalam/Kaggriculture/tree/8b8c421eb10634c756583ce10c75189f50c83a72)：agent 历史、市场顺序、回放分析工具 | `8b8c421eb10634c756583ce10c75189f50c83a72` | [MIT](https://github.com/Seyamalam/Kaggriculture/blob/8b8c421eb10634c756583ce10c75189f50c83a72/LICENSE) | `conditional allow`：仅使用作者 MIT 文件；其 README 明确说含外部策略，必须按 `THIRD_PARTY_NOTICES.md` 逐文件处理 |
| `deepeshumrao-agent` | [deepeshumrao/kaggriculture-agent](https://github.com/deepeshumrao/kaggriculture-agent/tree/65724ce530af8e8ea0410d5b7f0e2a997ca676cb)：协议、local env、契约测试 | `65724ce530af8e8ea0410d5b7f0e2a997ca676cb` | [MIT](https://github.com/deepeshumrao/kaggriculture-agent/blob/65724ce530af8e8ea0410d5b7f0e2a997ca676cb/LICENSE) | `reference_only`：项目背景不同，必须先用当前官方环境做 720 回合兼容性验证，才可升为对手 |

建议获取方式：

```powershell
git clone <repo-url> <local-directory>
git -C <local-directory> checkout <40-char-commit>
git -C <local-directory> rev-parse HEAD
Get-FileHash <local-directory>/main.py -Algorithm SHA256
```

本仓库不自动运行这些命令，不自动复制第三方代码，也没有伪造任何已下载轨迹。

## 隔离来源

| 来源 | 为什么隔离 | 解封条件 |
|---|---|---|
| [GzmCR/Kaggriculture](https://github.com/GzmCR/Kaggriculture/tree/6a76335397d5cd2facffa91c938f629b119ea350) | 固定 commit `6a76335397d5cd2facffa91c938f629b119ea350` 的根 `LICENSE` 请求返回 404 | 作者增加明确许可证，并核对具体文件来源 |
| Kaggle 公共 notebooks，包括 Adaptive Replay、Night Harvest、Diversified Scheduler 等 | “公开可见”不等于自动授予代码再利用/再分发许可证，且 notebook 版本会变化 | 记录 owner、slug、version number、页面明确许可证和导出文件 SHA-256；遵守比赛公开分享规则 |
| `raykkretzschmar/kaggriculture-reference-agents` 数据集 | 搜索能确认 manifest 被 notebook 引用，但本次未获得足够的逐 agent 许可证证据 | 每个 agent 的源 URL、版本、许可证和 notice 可追溯；不能只依赖聚合数据集页面 |
| 公开 ladder replay / 其他参赛者日志 | 官方 CLI 支持本人有权访问的 episode，但内容可能包含第三方行为，比赛规则也限制重分发 | 法务/规则确认训练与保存范围；仅本地按 episode ID 拉取；永不提交原始 replay；manifest 标 `redistributable=false` |
| 无许可证 GitHub 代码、Discussion 片段、匿名粘贴 | 无法证明可训练、修改或再分发 | 获得作者许可或找到等价明确许可原始来源 |

隔离不是“不能阅读”；它表示不得进入 BC、自博弈对手池或生成可提交权重的流水线。

## 原始算法与框架资料

- [PPO 原始论文](https://arxiv.org/abs/1707.06347)：clipped on-policy Actor-Critic。
- [GTrXL 原始论文](https://arxiv.org/abs/1910.06764)：RL 场景下的门控残差 Transformer 设计依据。当前骨架先使用 Pre-LN residual Transformer，门控记忆是后续实验项。
- [Invalid Action Masking 原始论文](https://arxiv.org/abs/2006.14171)：大离散动作空间中对无效动作应用 mask。
- [Decision Transformer 原始论文](https://arxiv.org/abs/2106.01345)：可作为离线序列预训练对照，不代替 PPO 主线。
- [AlphaStar / league training 原始论文](https://doi.org/10.1038/s41586-019-1724-z)：历史 checkpoint 与 exploiter 对手池思路。
- [Ray RLlib Multi-Agent 官方文档](https://docs.ray.io/en/latest/rllib/multi-agent-envs.html) 和 [PPO 官方文档](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html)：框架接口。由于 API 会变化，运行环境需要额外记录 Ray lockfile 和实际版本。

## 数据可用性结论

第一批可合法、可复现的数据不需要下载别人的 replay：用固定官方模拟器，让 `starter`、明确许可的固定 commit 策略、我们的规则专家和历史 checkpoint 双席位闭环对战。每条 transition 记录环境版本、策略文件 hash、seed、acting seat 和 opponent ID；训练/验证/测试 seed 段永久分离。
