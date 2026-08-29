"""Build a dependency-free local dashboard for Kaggriculture artifacts.

The generated HTML embeds a read-only snapshot of local logs, paired
benchmarks, heuristic-route reports, submission validation, and (optionally)
the user's Kaggle submission history plus the public leaderboard.  Re-run this
script to refresh the snapshot; it never starts training or submits an agent.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, (dict, list)) else None


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _paths(root: Path, pattern: str) -> list[Path]:
    return sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)


def _collect_benchmarks(repo_root: Path, *, maximum: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in _paths(repo_root / "artifacts" / "eval", "**/benchmark.json"):
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            continue
        config = payload.get("config", {}) or {}
        summary = payload.get("summary", {}) or {}
        overall = summary.get("overall", {}) if isinstance(summary, Mapping) else {}
        if not isinstance(config, Mapping) or not isinstance(overall, Mapping):
            continue
        results.append(
            {
                "path": _relative(path, repo_root),
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
                "candidate": str(config.get("candidate", "unknown")),
                "opponents": [str(item) for item in (config.get("opponents", []) or [])],
                "games": overall.get("games"),
                "wins": overall.get("wins"),
                "draws": overall.get("draws"),
                "losses": overall.get("losses"),
                "win_rate": overall.get("win_rate"),
                "mean_score_difference": overall.get("mean_score_difference"),
                "latency_p95_ms": overall.get("policy_latency_p95_ms"),
                "crash_rate": overall.get("policy_crash_rate"),
                "engine_failure_rate": overall.get("engine_failure_rate"),
            }
        )
        if len(results) >= maximum:
            break
    return results


def _collect_metric_runs(repo_root: Path, *, maximum_runs: int, maximum_rows: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in _paths(repo_root / "artifacts", "**/metrics.jsonl")[:maximum_runs]:
        try:
            with path.open(encoding="utf-8") as handle:
                lines = list(deque(handle, maxlen=maximum_rows))
        except OSError:
            continue
        rows: list[dict[str, Any]] = []
        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
        if not rows:
            continue
        latest = rows[-1]
        result.append(
            {
                "path": _relative(path, repo_root),
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
                "rows": rows,
                "latest": latest,
            }
        )
    return result


def _collect_processes(repo_root: Path, *, maximum: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in _paths(repo_root / "artifacts", "**/process.json")[:maximum]:
        value = _read_json(path)
        if not isinstance(value, Mapping):
            continue
        result.append(
            {
                "path": _relative(path, repo_root),
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
                "status": value.get("status", "unknown"),
                "pid": value.get("pid", value.get("worker_pid")),
                "device": value.get("device"),
                "iteration": value.get("iteration", value.get("completed_iterations")),
            }
        )
    return result


def _collect_json_reports(repo_root: Path, pattern: str, *, maximum: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in _paths(repo_root / "artifacts", pattern)[:maximum]:
        value = _read_json(path)
        if not isinstance(value, Mapping):
            continue
        result.append(
            {
                "path": _relative(path, repo_root),
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
                "report": value,
            }
        )
    return result


def _decode_cli_json(output: str) -> list[dict[str, Any]]:
    """Kaggle may print a page token before its JSON payload."""

    start = output.find("[")
    if start < 0:
        raise ValueError("Kaggle CLI did not return a JSON list")
    value, _ = json.JSONDecoder().raw_decode(output[start:])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("Kaggle CLI JSON has an unexpected shape")
    return value


def _kaggle_command(arguments: Iterable[str]) -> list[dict[str, Any]]:
    # The globally installed Kaggle CLI is bound to its own Python runtime on
    # this Windows host.  Removing the active project virtualenv prevents that
    # launcher from looking for Kaggle inside the inference/training venv.
    environment = dict(os.environ)
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("PYTHONHOME", None)
    completed = subprocess.run(
        ["kaggle", *arguments, "--format", "json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
    )
    return _decode_cli_json(completed.stdout)


def _refresh_kaggle(competition: str) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"competition": competition, "refreshed_at": _now(), "errors": []}
    try:
        snapshot["submissions"] = _kaggle_command(["competitions", "submissions", competition])
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        snapshot["submissions"] = []
        snapshot["errors"].append(f"submissions: {exc}")
    try:
        snapshot["leaderboard"] = _kaggle_command(
            ["competitions", "leaderboard", competition, "--show", "--page-size", "20"]
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        snapshot["leaderboard"] = []
        snapshot["errors"].append(f"leaderboard: {exc}")
    return snapshot


def _load_kaggle(cache: Path, *, competition: str, refresh: bool) -> dict[str, Any]:
    if refresh:
        value = _refresh_kaggle(competition)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return value
    value = _read_json(cache)
    if isinstance(value, Mapping):
        return dict(value)
    return {
        "competition": competition,
        "refreshed_at": None,
        "submissions": [],
        "leaderboard": [],
        "errors": ["No Kaggle snapshot yet. Re-run with --refresh-kaggle."],
    }


def _dashboard_html(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Farmer · Kaggriculture 控制室</title>
  <style>
    :root { color-scheme:dark; --page:#07100c; --page-deep:#030705; --surface:rgba(18,31,24,.74); --surface-raised:rgba(29,45,35,.82); --surface-solid:#15251c; --line:rgba(207,238,216,.14); --line-strong:rgba(207,238,216,.25); --text:#f2f7f2; --muted:#a5b6a8; --quiet:#75877a; --mint:#78dfb1; --mint-strong:#a6f4ce; --sky:#98c9ff; --amber:#ffd187; --rose:#ff9eaa; --shadow:0 20px 54px rgba(0,0,0,.22); }
    * { box-sizing:border-box; }
    html { background:var(--page-deep); }
    body { min-width:320px; margin:0; color:var(--text); background:radial-gradient(900px 520px at -10% -10%,rgba(82,170,119,.22),transparent 68%),radial-gradient(780px 480px at 105% 4%,rgba(72,135,175,.16),transparent 68%),linear-gradient(155deg,#0b1811 0%,var(--page) 44%,#08110d 100%); font:14px/1.45 -apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",sans-serif; font-optical-sizing:auto; letter-spacing:-.01em; }
    button, input, select { font:inherit; }
    button, select, input { -webkit-tap-highlight-color:transparent; }
    main { width:min(1440px,100%); margin:0 auto; padding:14px 24px 42px; }
    .topbar { position:sticky; top:12px; z-index:5; display:flex; align-items:center; justify-content:space-between; gap:24px; margin:0 0 22px; padding:16px 18px 16px 20px; background:rgba(17,31,23,.74); border:1px solid var(--line); border-radius:24px; box-shadow:var(--shadow); backdrop-filter:blur(24px) saturate(145%); -webkit-backdrop-filter:blur(24px) saturate(145%); }
    .eyebrow { display:block; margin:0 0 4px; color:var(--mint); font-size:11px; font-weight:700; letter-spacing:.09em; text-transform:uppercase; }
    h1 { margin:0; font-size:clamp(24px,3.4vw,34px); font-weight:720; line-height:1.08; letter-spacing:-.045em; }
    h2 { margin:0; font-size:16px; font-weight:700; letter-spacing:-.025em; }
    .sub { max-width:720px; margin:7px 0 0; color:var(--muted); font-size:12px; }
    .top-actions { display:flex; flex:0 0 auto; align-items:center; justify-content:flex-end; gap:9px; flex-wrap:wrap; }
    .status-badge, .pill { display:inline-flex; align-items:center; min-height:29px; border:1px solid var(--line); border-radius:999px; padding:5px 10px; color:var(--muted); background:rgba(255,255,255,.035); font-size:11px; font-weight:650; letter-spacing:.01em; }
    .status-badge::before { width:7px; height:7px; margin-right:6px; border-radius:50%; background:var(--mint); box-shadow:0 0 0 3px rgba(120,223,177,.12); content:""; }
    .button { min-height:34px; border:1px solid var(--line-strong); border-radius:999px; padding:7px 12px; color:var(--text); background:rgba(255,255,255,.075); box-shadow:inset 0 1px rgba(255,255,255,.09); cursor:pointer; transition:transform 180ms ease,background 180ms ease,border-color 180ms ease; }
    .button:hover { border-color:rgba(166,244,206,.48); background:rgba(166,244,206,.12); }
    .button:active { transform:scale(.96); }
    .button:focus-visible, select:focus-visible, input:focus-visible { outline:2px solid var(--sky); outline-offset:2px; }
    .grid { display:grid; gap:16px; }
    .cards { grid-template-columns:repeat(5,minmax(0,1fr)); margin-bottom:16px; }
    .two { grid-template-columns:minmax(0,1.35fr) minmax(350px,.65fr); margin-bottom:16px; }
    .surface { background:var(--surface); border:1px solid var(--line); border-radius:22px; box-shadow:0 1px rgba(255,255,255,.035) inset,0 10px 30px rgba(0,0,0,.12); backdrop-filter:blur(20px) saturate(135%); -webkit-backdrop-filter:blur(20px) saturate(135%); }
    .panel { padding:19px; }
    .card { position:relative; min-height:126px; overflow:hidden; padding:17px; }
    .card::after { position:absolute; right:-26px; bottom:-36px; width:116px; height:116px; border-radius:50%; background:radial-gradient(circle,rgba(120,223,177,.16),transparent 67%); content:""; pointer-events:none; }
    .card:nth-child(2)::after { background:radial-gradient(circle,rgba(152,201,255,.16),transparent 67%); }
    .card:nth-child(3)::after { background:radial-gradient(circle,rgba(255,209,135,.17),transparent 67%); }
    .card:nth-child(4)::after { background:radial-gradient(circle,rgba(120,223,177,.1),transparent 67%); }
    .label { color:var(--muted); font-size:11px; font-weight:650; letter-spacing:.045em; text-transform:uppercase; }
    .value { margin:8px 0 5px; font-size:clamp(23px,2.4vw,30px); font-weight:730; letter-spacing:-.045em; line-height:1.05; font-variant-numeric:tabular-nums; }
    .detail { overflow:hidden; color:var(--muted); font-size:12px; text-overflow:ellipsis; white-space:nowrap; }
    .good { color:var(--mint-strong); } .warn { color:var(--amber); } .bad { color:var(--rose); }
    .section-head { display:flex; align-items:center; gap:12px; margin:0 0 15px; }
    .section-head .quiet { margin-left:auto; color:var(--quiet); font-size:11px; }
    .toolbar { display:flex; align-items:center; gap:8px; margin:0 0 12px; flex-wrap:wrap; }
    .toolbar .section-head { flex:1 1 180px; margin:0; }
    select, input { min-height:35px; max-width:100%; border:1px solid var(--line); border-radius:11px; padding:7px 10px; color:var(--text); background:rgba(5,13,9,.55); box-shadow:inset 0 1px rgba(255,255,255,.03); }
    select { max-width:250px; cursor:pointer; } input { width:100%; }
    .chart-shell { overflow:hidden; border:1px solid rgba(207,238,216,.1); border-radius:16px; padding:10px 10px 6px; background:linear-gradient(145deg,rgba(157,222,184,.075),rgba(7,16,12,.12)); }
    canvas { width:100%; height:258px; display:block; border-radius:11px; opacity:1; transition:opacity 160ms ease,transform 220ms cubic-bezier(.2,.8,.2,1); }
    canvas.is-updating { opacity:.66; transform:translateY(2px); }
    .note { margin:10px 1px 0; color:var(--muted); font-size:12px; }
    .metric-summary { margin:0 0 4px; color:var(--quiet); font-size:11px; font-variant-numeric:tabular-nums; }
    .table-wrap { max-height:380px; overflow:auto; border:1px solid rgba(207,238,216,.09); border-radius:15px; background:rgba(4,11,7,.22); scrollbar-color:rgba(166,244,206,.35) transparent; }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th, td { border-bottom:1px solid rgba(207,238,216,.08); padding:9px 8px; text-align:left; vertical-align:top; }
    tr:last-child td { border-bottom:0; }
    tbody tr { transition:background 150ms ease; }
    tbody tr:hover { background:rgba(166,244,206,.055); }
    th { position:sticky; top:0; z-index:1; color:var(--muted); background:rgba(20,35,26,.94); font-weight:650; backdrop-filter:blur(12px); }
    .mono { font-family:"SF Mono",ui-monospace,Consolas,monospace; font-size:11px; letter-spacing:-.025em; }
    .empty { color:var(--quiet); text-align:center; }
    .fact-list { display:grid; gap:0; margin:0; }
    .fact { display:flex; align-items:baseline; justify-content:space-between; gap:15px; padding:11px 0; border-bottom:1px solid rgba(207,238,216,.09); }
    .fact:last-child { border-bottom:0; }
    .fact dt { color:var(--muted); font-size:12px; } .fact dd { margin:0; text-align:right; font-weight:660; font-variant-numeric:tabular-nums; }
    .build { margin:0; padding:12px 0; border-bottom:1px solid rgba(207,238,216,.09); } .build:last-child { border-bottom:0; } .build-name { margin:0 0 5px; overflow-wrap:anywhere; color:var(--text); } .build-detail { margin:0; color:var(--muted); font-size:11px; }
    .live { position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }
    @media (max-width:1120px) { .cards { grid-template-columns:repeat(3,minmax(0,1fr)); } .two { grid-template-columns:minmax(0,1fr); } }
    @media (max-width:700px) { main { padding:10px 12px 28px; } .topbar { top:7px; align-items:flex-start; padding:15px; border-radius:20px; } .top-actions { justify-content:flex-start; } .cards { grid-template-columns:repeat(2,minmax(0,1fr)); } .card { min-height:112px; } .panel { padding:15px; border-radius:19px; } .surface { border-radius:19px; } .toolbar { align-items:stretch; } select { max-width:none; flex:1 1 150px; } .table-wrap { margin:0 -2px; } th, td { padding:8px 7px; } }
    @media (max-width:430px) { .topbar { display:block; } .top-actions { margin-top:13px; } .cards { grid-template-columns:1fr; } }
    @media (hover:none) { tbody tr:hover { background:transparent; } }
    @media (prefers-reduced-motion:reduce) { *,*::before,*::after { scroll-behavior:auto!important; animation-duration:.01ms!important; animation-iteration-count:1!important; transition-duration:.01ms!important; } }
    @media (prefers-reduced-transparency:reduce) { .topbar,.surface { background:var(--surface-solid); backdrop-filter:none; -webkit-backdrop-filter:none; } }
    @media (prefers-contrast:more) { :root { --line:rgba(235,255,240,.34); --muted:#d4dfd5; } .topbar,.surface { background:#102017; } }
    @supports not ((backdrop-filter:blur(1px)) or (-webkit-backdrop-filter:blur(1px))) { .topbar { background:#14251b; } .surface { background:#17291e; } }
  </style>
</head>
<body><main>
  <header class="topbar">
    <div><span class="eyebrow">Kaggriculture · Local command center</span><h1>Farmer 决策控制室</h1><p class="sub" id="stamp"></p></div>
    <div class="top-actions"><span class="status-badge" id="snapshot-state"></span><button class="button" type="button" id="copy-command">复制刷新命令</button></div>
  </header>
  <section class="grid cards" id="cards" aria-label="核心指标"></section>
  <section class="grid two"><article class="surface panel"><div class="toolbar"><div class="section-head"><h2>训练日志趋势</h2><span class="quiet">本地 metrics 快照</span></div><select id="run" aria-label="选择日志运行"></select><select id="metric" aria-label="选择趋势指标"></select></div><div class="chart-shell"><p class="metric-summary" id="chart-summary"></p><canvas id="chart" role="img" aria-label="训练指标折线图"></canvas></div><p class="note" id="log-note"></p></article><article class="surface panel"><div class="section-head"><h2>启发式路线挖掘</h2><span class="quiet">仅观测</span></div><div id="heuristic"></div></article></section>
  <section class="grid two"><article class="surface panel"><div class="section-head"><h2>本地对局评测</h2><span class="quiet" id="benchmark-count"></span></div><div class="toolbar"><input id="benchmark-filter" aria-label="筛选本地评测" placeholder="筛选策略或路径"></div><div class="table-wrap"><table><thead><tr><th>评测</th><th>候选</th><th>对局</th><th>胜率</th><th>平均分差</th><th>p95</th><th>故障</th></tr></thead><tbody id="benchmarks"></tbody></table></div></article><article class="surface panel"><div class="section-head"><h2>提交验收</h2><span class="quiet">构建产物</span></div><div id="submission-builds"></div><div class="section-head" style="margin-top:22px"><h2>本地进程快照</h2><span class="quiet">不启动进程</span></div><div class="table-wrap"><table><thead><tr><th>日志</th><th>状态</th><th>PID</th><th>迭代</th></tr></thead><tbody id="processes"></tbody></table></div></article></section>
  <section class="grid two"><article class="surface panel"><div class="section-head"><h2>Kaggle 提交记录</h2><span class="quiet">远程只读</span></div><p class="note" id="kaggle-note"></p><div class="table-wrap" style="margin-top:12px"><table><thead><tr><th>ID</th><th>文件</th><th>状态</th><th>Public</th><th>时间</th></tr></thead><tbody id="submissions"></tbody></table></div></article><article class="surface panel"><div class="section-head"><h2>公开排行榜 Top 20</h2><span class="quiet" id="leaderboard-count"></span></div><div class="table-wrap"><table><thead><tr><th>#</th><th>队伍</th><th>分数</th><th>提交时间</th></tr></thead><tbody id="leaderboard"></tbody></table></div></article></section>
  <div class="live" id="live" aria-live="polite"></div>
</main><script>const DATA=__DATA__;
const $=id=>document.getElementById(id);const text=value=>String(value??'—');const numeric=value=>Number.isFinite(Number(value));const n=value=>value===null||value===undefined||value===''?'—':numeric(value)?Number(value).toLocaleString(undefined,{maximumFractionDigits:2}):text(value);const short=value=>text(value||'—').replace(/^python:artifacts\/third-party-quarantine\//,'').replace(/\/output\/main\.py:agent$/,'').slice(0,72);const tone=value=>Number(value)>0?'good':Number(value)<0?'bad':'';
function element(tag,className='',value){const node=document.createElement(tag);if(className)node.className=className;if(value!==undefined)node.textContent=text(value);return node}function td(row,value,className=''){row.append(element('td',className,value))}function emptyRow(body,columns,message){const row=document.createElement('tr'),cell=element('td','empty',message);cell.colSpan=columns;row.append(cell);body.append(row)}function card(label,value,detail,className=''){const root=element('article','surface card'),name=element('div','label',label),number=element('div',`value ${className}`,value),description=element('div','detail',detail);root.append(name,number,description);$('cards').append(root)}
const kg=DATA.kaggle||{},subs=Array.isArray(kg.submissions)?kg.submissions:[],leaders=Array.isArray(kg.leaderboard)?kg.leaderboard:[],latest=subs[0]||{},best=subs.reduce((winner,item)=>Number(item.publicScore)>Number(winner.publicScore)?item:winner,{}),leader=leaders[0]||{},heur=(DATA.heuristic_reports||[])[0]?.report||{},local=(DATA.benchmarks||[]).find(item=>item.path.includes('kaito_v48_heuristic_control_holdout'))||{};
$('stamp').textContent=`生成于 ${DATA.generated_at} · 本地快照；Kaggle 数据刷新于 ${kg.refreshed_at||'尚未刷新'}`;$('snapshot-state').textContent=kg.refreshed_at?'远程快照已同步':'本地快照';card('最新 Kaggle Public',n(latest.publicScore),latest.fileName||'没有提交快照',Number(latest.publicScore)>0?'good':'');card('历史最佳 Public',n(best.publicScore),best.fileName||'—','good');card('排行榜第一',n(leader.score),leader.teamName||'没有远程快照','warn');card('V48 本地平均分差',n(local.mean_score_difference),`${n(local.games)} 局固定对手`,tone(local.mean_score_difference));card('路线模拟候选',n(heur.simulation_candidate_count),heur.split?.method||'无路线报告',heur.simulation_candidate_count?'warn':'');
const runs=Array.isArray(DATA.metric_runs)?DATA.metric_runs:[],runSel=$('run'),metricSel=$('metric'),preferredMetrics=['mean_score_difference','learner_win_rate','scripted_win_rate','entropy','kl','value_loss','steps_per_second'],seenMetrics=new Set(preferredMetrics);runs.forEach((run,index)=>{const option=element('option','',run.path);option.value=index;runSel.append(option);(run.rows||[]).forEach(row=>Object.entries(row||{}).forEach(([key,value])=>{if(numeric(value))seenMetrics.add(key)}))});[...seenMetrics].forEach(key=>{const option=element('option','',key);option.value=key;metricSel.append(option)});metricSel.value=[...seenMetrics].includes('mean_score_difference')?'mean_score_difference':[...seenMetrics][0]||'';
let resizeFrame;function drawChart(){const run=runs[Number(runSel.value)||0],metric=metricSel.value,canvas=$('chart');canvas.classList.add('is-updating');requestAnimationFrame(()=>canvas.classList.remove('is-updating'));if(!run){$('chart-summary').textContent='没有可用的 metrics.jsonl 运行。';$('log-note').textContent='生成新的本地日志后，重新构建仪表盘即可查看。';return}const points=(run.rows||[]).map((row,index)=>({x:Number(row.iteration??index),y:Number(row[metric])})).filter(point=>Number.isFinite(point.y));$('chart-summary').textContent=`${metric} · ${points.length} 个采样点`;$('log-note').textContent=`${run.path} · 最新 ${metric}: ${n(run.latest?.[metric])}`;canvas.setAttribute('aria-label',`${run.path} 的 ${metric} 趋势图，${points.length} 个采样点，最新值 ${n(run.latest?.[metric])}`);const box=canvas.getBoundingClientRect(),ratio=devicePixelRatio||1,width=Math.max(300,box.width),height=258;canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);const context=canvas.getContext('2d');context.setTransform(ratio,0,0,ratio,0,0);context.clearRect(0,0,width,height);context.strokeStyle='rgba(207,238,216,.14)';context.lineWidth=1;for(let index=0;index<5;index+=1){const y=22+index*(height-53)/4;context.beginPath();context.moveTo(46,y);context.lineTo(width-14,y);context.stroke()}if(!points.length){context.fillStyle='#a5b6a8';context.font='12px -apple-system,system-ui';context.fillText('所选指标尚无数值采样',46,height/2);return}const ys=points.map(point=>point.y),xs=points.map(point=>point.x);let low=Math.min(...ys),high=Math.max(...ys);if(low===high){low-=1;high+=1}const xLow=Math.min(...xs),xHigh=Math.max(...xs)||xLow+1,px=point=>46+(point.x-xLow)/(xHigh-xLow||1)*(width-60),py=point=>22+(high-point.y)/(high-low)*(height-53);const gradient=context.createLinearGradient(0,22,width,height);gradient.addColorStop(0,'#a6f4ce');gradient.addColorStop(1,'#78bfff');context.strokeStyle=gradient;context.lineWidth=2.5;context.lineCap='round';context.lineJoin='round';context.beginPath();points.forEach((point,index)=>index?context.lineTo(px(point),py(point)):context.moveTo(px(point),py(point)));context.stroke();const last=points[points.length-1];context.fillStyle='#a6f4ce';context.beginPath();context.arc(px(last),py(last),3.8,0,Math.PI*2);context.fill();context.fillStyle='#a5b6a8';context.font='11px -apple-system,system-ui';context.fillText(n(high),2,26);context.fillText(n(low),2,height-27);context.fillText(text(xLow),46,height-8);context.fillText(text(xHigh),Math.max(46,width-40),height-8)}function scheduleChart(){cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(drawChart)}runSel.onchange=drawChart;metricSel.onchange=drawChart;if('ResizeObserver'in window)new ResizeObserver(scheduleChart).observe($('chart'));else addEventListener('resize',scheduleChart);drawChart();
function fact(root,label,value,className=''){const row=element('div','fact'),name=element('dt','',label),result=element('dd',className,value);row.append(name,result);root.append(row)}function renderHeuristic(){const root=$('heuristic');root.textContent='';if(!Object.keys(heur).length){root.append(element('p','note','没有启发式训练报告。'));return}const list=element('dl','fact-list');fact(list,'待闭环模拟候选',n(heur.simulation_candidate_count),heur.simulation_candidate_count?'warn':'good');fact(list,'输入路线',`${n(heur.input_route_count)} 条`);fact(list,'历史段 / 留出段',`${n(heur.split?.train_routes)} / ${n(heur.split?.holdout_routes)}`);const statuses={};(heur.events||[]).forEach(event=>(event.candidates||[]).forEach(candidate=>statuses[candidate.status]=(statuses[candidate.status]||0)+1));fact(list,'候选状态',Object.entries(statuses).map(([key,value])=>`${key}: ${value}`).join(' · ')||'—');root.append(list)}renderHeuristic();
function renderBenchmarks(){const query=$('benchmark-filter').value.toLowerCase(),body=$('benchmarks'),items=(DATA.benchmarks||[]).filter(item=>`${item.path} ${item.candidate}`.toLowerCase().includes(query));body.textContent='';$('benchmark-count').textContent=`${items.length} 项`;if(!items.length){emptyRow(body,7,'没有匹配的本地评测。');return}items.forEach(item=>{const row=document.createElement('tr');td(row,item.path,'mono');td(row,short(item.candidate));td(row,n(item.games));td(row,`${n(item.wins)}-${n(item.draws)}-${n(item.losses)} (${n(Number(item.win_rate)*100)}%)`);td(row,n(item.mean_score_difference),tone(item.mean_score_difference));td(row,`${n(item.latency_p95_ms)} ms`);td(row,`${n(Number(item.crash_rate)*100)}% / ${n(Number(item.engine_failure_rate)*100)}%`);body.append(row)})}$('benchmark-filter').oninput=renderBenchmarks;renderBenchmarks();
function renderBuilds(){const root=$('submission-builds'),builds=DATA.submission_builds||[];root.textContent='';if(!builds.length){root.append(element('p','note','没有本地提交验收清单。'));return}builds.forEach(item=>{const report=item.report||{},build=element('article','build'),name=element('p','build-name mono',item.path),detail=element('p','build-detail',`${n(report.archive_bytes)} bytes · ${text(report.entrypoint||'—')}`),proof=element('p','build-detail',`SHA ${text(report.main_sha256||'—').slice(0,16)}… · smoke ${Array.isArray(report.smoke)?'DONE × '+report.smoke.length:'—'}`);build.append(name,detail,proof);root.append(build)})}renderBuilds();
function renderProcesses(){const body=$('processes'),items=DATA.processes||[];body.textContent='';if(!items.length){emptyRow(body,4,'没有记录到本地进程。');return}items.forEach(item=>{const row=document.createElement('tr');td(row,item.path,'mono');td(row,item.status);td(row,n(item.pid));td(row,n(item.iteration));body.append(row)})}renderProcesses();
$('kaggle-note').textContent=(kg.errors||[]).join(' · ')||`比赛：${kg.competition||'kaggriculture'}；只读快照，刷新不会提交新 agent。`;function remote(){const submissionBody=$('submissions'),leaderboardBody=$('leaderboard');submissionBody.textContent='';leaderboardBody.textContent='';if(!subs.length)emptyRow(submissionBody,5,'尚无 Kaggle 提交快照。');else subs.forEach(item=>{const row=document.createElement('tr');td(row,item.ref);td(row,item.fileName,'mono');td(row,item.status);td(row,n(item.publicScore),Number(item.publicScore)>0?'good':'');td(row,item.date);submissionBody.append(row)});if(!leaders.length)emptyRow(leaderboardBody,4,'尚无排行榜快照。');else leaders.forEach((item,index)=>{const row=document.createElement('tr');td(row,index+1);td(row,item.teamName);td(row,n(item.score),'warn');td(row,item.submissionDate);leaderboardBody.append(row)});$('leaderboard-count').textContent=leaders.length?`${leaders.length} 支队伍`:''}remote();
const refreshCommand='.\\.venv\\Scripts\\python.exe scripts\\build_dashboard.py --refresh-kaggle';$('copy-command').onclick=async()=>{const button=$('copy-command');try{if(navigator.clipboard?.writeText)await navigator.clipboard.writeText(refreshCommand);else{const area=document.createElement('textarea');area.value=refreshCommand;document.body.append(area);area.select();document.execCommand('copy');area.remove()}button.textContent='已复制';$('live').textContent='刷新命令已复制到剪贴板。'}catch{button.textContent='复制失败';$('live').textContent='无法访问剪贴板，请从文档复制刷新命令。'}setTimeout(()=>button.textContent='复制刷新命令',1500)};
</script></body></html>""".replace("__DATA__", data)


def build_payload(repo_root: Path, *, kaggle_cache: Path, competition: str, refresh_kaggle: bool) -> dict[str, Any]:
    return {
        "schema_version": "farmer-dashboard/v1",
        "generated_at": _now(),
        "benchmarks": _collect_benchmarks(repo_root, maximum=80),
        "metric_runs": _collect_metric_runs(repo_root, maximum_runs=24, maximum_rows=240),
        "processes": _collect_processes(repo_root, maximum=30),
        "heuristic_reports": _collect_json_reports(repo_root, "heuristic-training/*.json", maximum=10),
        "submission_builds": _collect_json_reports(repo_root, "submissions/*.manifest.json", maximum=10),
        "kaggle": _load_kaggle(kaggle_cache, competition=competition, refresh=refresh_kaggle),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/dashboard/index.html", type=Path)
    parser.add_argument("--data-output", type=Path)
    parser.add_argument("--kaggle-cache", default="artifacts/dashboard/kaggle_snapshot.json", type=Path)
    parser.add_argument("--competition", default="kaggriculture")
    parser.add_argument("--refresh-kaggle", action="store_true", help="read current scores with the configured Kaggle CLI")
    return parser


def main() -> int:
    args = _parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    data_output = (args.data_output or output.with_name("dashboard-data.json")).resolve()
    payload = build_payload(
        repo_root,
        kaggle_cache=args.kaggle_cache.resolve(),
        competition=args.competition,
        refresh_kaggle=args.refresh_kaggle,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dashboard_html(payload), encoding="utf-8")
    data_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(output), "data": str(data_output), "generated_at": payload["generated_at"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
