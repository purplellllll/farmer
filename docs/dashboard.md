# 本地仪表盘

仪表盘是一个不依赖 Streamlit 或数据库的单页 HTML 快照。页面按“核心状态、趋势、验证、远程成绩”组织，并采用半透明材质、系统字体和响应式布局；在降低动态效果、降低透明度或提高对比度的系统设置下会自动降级。它读取本地的：

- `artifacts/**/metrics.jsonl`：训练/实验趋势；
- `artifacts/eval/**/benchmark.json`：双座位评测；
- 启发式路线挖掘和提交验收清单；
- 可选的 Kaggle 提交历史与排行榜快照。

生成并刷新 Kaggle 的只读数据：

```powershell
.\.venv\Scripts\python.exe scripts\build_dashboard.py --refresh-kaggle
```

输出位于 `artifacts/dashboard/index.html`。直接在浏览器打开即可；右上角可复制刷新命令。它不启动训练、不会提交新 agent，也不会改动策略。再次运行命令会刷新所有本地指标和远程只读快照。
