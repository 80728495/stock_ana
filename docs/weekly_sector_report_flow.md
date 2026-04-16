# Weekly Sector Report 流程说明（clawbot 集成）

本文档说明 `weekly_sector_report` 的执行流程、时间调度、输出文件与 clawbot 读取方式。

## 1. 调度时间

当前 cron 任务：

- 每周日 08:30 执行
- 命令：

```bash
cd /Users/wl/stock_ana && /Users/wl/.pyenv/shims/python3 -m stock_ana.workflows.weekly_sector_report >> /Users/wl/stock_ana/data/logs/cron_weekly_sector_report.log 2>&1
```

说明：
- 回看窗口统一按近一周（默认 `lookback=5` 个交易日）
- 如需临时改窗口，可手动加 `--lookback N`

## 2. 执行流程

入口脚本：
- `src/stock_ana/workflows/weekly_sector_report.py`

流程步骤：

1. 更新美股价格（可 `--skip-update` 跳过）
2. 扫描全市场异动股票（默认最近 5 个交易日）
3. 按 sector/sub-label 聚合板块异动
4. 发送聚合结果给 Gemini 生成周报正文
5. 落盘 Markdown 周报 + JSON 摘要

## 3. 输出目录与文件

输出目录：
- `data/output/weekly_sector/`

每周产生两类文件：

1. 周报正文（Markdown）
- 文件名：`weekly_sector_{YYYY}_W{WW}.md`
- 示例：`weekly_sector_2026_W15.md`

2. 周报摘要（JSON，供 clawbot 读取）
- 文件名：`weekly_sector_{YYYY}_W{WW}_summary.json`
- 示例：`weekly_sector_2026_W15_summary.json`

## 4. summary.json 字段说明

示例：

```json
{
  "workflow": "weekly_sector_report",
  "week_label": "2026年第15周",
  "year": 2026,
  "week": 15,
  "period_start": "2026-04-06",
  "period_end": "2026-04-12",
  "generated_at": "2026-04-12",
  "lookback_days": 5,
  "window_desc": "最近 5 个交易日",
  "min_breadth": 2,
  "momentum_count": 38,
  "report_path": "/Users/wl/stock_ana/data/output/weekly_sector/weekly_sector_2026_W15.md",
  "status": "ok",
  "error": "",
  "top_tickers": [
    {"ticker": "NVDA", "score": 7.2, "sector": "Technology"},
    {"ticker": "AVGO", "score": 6.8, "sector": "Technology"}
  ]
}
```

字段解释：
- `status`:
  - `ok`: 周报正常生成
  - `preview`: 仅预览模式（未调用 Gemini）
  - `empty`: 本周无异动
- `report_path`:
  - `ok` 时为 Markdown 文件路径
  - `preview/empty` 时可能为 `null`
- `top_tickers`: 按异动分数排序的 Top 标的（最多 20）

## 5. clawbot 集成建议

clawbot 每周读取最新一个 `*_summary.json`：

1. 若 `status=ok`：
- 推送周报摘要（week_label、momentum_count、top_tickers）
- 附带 `report_path` 作为完整阅读入口

2. 若 `status=empty`：
- 推送“本周无明显板块异动”

3. 若 `status=preview`：
- 仅用于人工调试，不建议自动推送

推荐消息模板：

```text
📈 周度板块异动报告 {week_label}
窗口：{window_desc}
异动股票数：{momentum_count}

Top 标的：
- {ticker1} score={score1}
- {ticker2} score={score2}

完整报告：{report_path}
```

## 6. 手动运行命令

```bash
cd /Users/wl/stock_ana

# 完整周报（默认近一周 = 5 个交易日）
python -m stock_ana.workflows.weekly_sector_report

# 仅预览板块聚合结果，不调用 Gemini
python -m stock_ana.workflows.weekly_sector_report --preview

# 跳过价格更新
python -m stock_ana.workflows.weekly_sector_report --skip-update

# 自定义窗口（例如最近 7 个交易日）
python -m stock_ana.workflows.weekly_sector_report --lookback 7
```
