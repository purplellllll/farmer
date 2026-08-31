# Kaggriculture 金牌区回放（2026-08-31 快照）

本目录保存从 Kaggle 公开回放数据集中筛选的金牌区代表 episode。每个 ZIP
包含一个完整的 `episode_id.json` 回放文件；`manifest.json` 保存了排名字段、
每日数据集来源、episode 分数以及原始与压缩文件的 SHA-256。

## 筛选规则

- 覆盖索引中截至 2026-08-30 的 32 个每日数据集。
- 每日按 `avg_score` 降序选择第 1 名。
- 同分时按 `min_score` 降序，再按 `episode_id` 升序确定唯一文件。
- 原始每日数据集每个约 20 GiB，本快照只下载上述 32 个 JSON，不下载完整数据集。

## 来源与许可

来源是 [Kaggriculture Episodes Index](https://www.kaggle.com/datasets/kaggle/kaggriculture-episodes-index)
及其 `kaggle/kaggriculture-episodes-YYYY-MM-DD` 每日数据集。Kaggle 数据集标注为
CC0-1.0；本目录只保留公开回放和必要的可复现元数据。详细来源 URL、原始文件名、
分数和哈希见同目录的 `manifest.json`。
