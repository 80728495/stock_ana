# 每日自动化流水线 — clawbot 集成指南

> 本文档覆盖股票相关的所有每日自动化任务：数据更新状态监控 + Vegas Mid 扫描 + Gemini 基本面分析。
> clawbot 读取两个 JSON 入口即可构建每日推送消息。

---

## 1. 触发时间与流程

### Cron 任务表（每天）

| 时间  | 任务 | 说明 |
|-------|------|------|
| 07:00 | `cron_daily_update.sh` | 更新行情数据（OHLCV + 技术指标 + Wave 结构） |
| 08:00 | `vegas_mid_daily_scan.py` | 扫描信号 + Gemini 分析 + 生成 summary.json |

07:00 的数据更新需要约 30-60 分钟，08:00 扫描时数据已准备好。

### daily_scan.py 执行步骤

```
Step 1：构建 US 科技板块 watchlist（data/us_tech_universe.csv，约 339 只）
Step 2：运行 Vegas Mid touch 策略扫描（lookback=1，min_signal=HOLD）
         → 每只触发信号的股票自动生成 K 线图（PNG）
Step 3：将 HOLD 以上信号发送给 Gemini，一次请求批量分析
         → 基本面 / 估值 / 技术面 / 综合建议
Step 4：从 Gemini 报告末尾汇总表提取每只股票的综合建议 + 评分
Step 5：生成 summary.json（clawbot 消费入口）
```

---

## 2. 输出目录结构

每天运行后产生以下文件（按日期隔离）：

```
data/output/
  daily_update/{YYYY-MM-DD}/
    status.json               ← 数据更新结果（clawbot 入口 1）

  daily_scan/{YYYY-MM-DD}/
    summary.json              ← 扫描 + Gemini 结论（clawbot 入口 2）

  vegas_scan/{YYYY-MM-DD}/
    signals.json              ← 原始扫描信号（含所有字段）
    signals_full.json         ← 含 base64 编码图表
    STRONG_BUY_US_RNG_*.png  ← 各标的 K 线图（文件名含信号等级）
    BUY_US_AKAM_*.png
    HOLD_US_LRCX_*.png
    ...

  scan_analysis/{YYYY-MM-DD}/
    2026-04-12_RNG_AKAM_....md  ← Gemini 完整分析报告（Markdown）
```

---

## 3. status.json 格式说明（数据更新）

`data/output/daily_update/{YYYY-MM-DD}/status.json`

```json
{
  "update_date": "2026-04-12",
  "generated_at": "2026-04-12T07:52:11",
  "all_ok": true,
  "total_elapsed": 1830,
  "steps": [
    {"name": "美股OHLCV",   "ok": true, "elapsed": 45,  "updated": 120, "skipped": 1380, "failed": 3},
    {"name": "纳指100OHLCV","ok": true, "elapsed": 18,  "count": 100},
    {"name": "港股OHLCV",   "ok": true, "elapsed": 22,  "count": 330},
    {"name": "A股OHLCV",    "ok": true, "elapsed": 8,   "count": 12},
    {"name": "技术指标",    "ok": true, "elapsed": 310},
    {"name": "Wave结构",    "ok": true, "elapsed": 620,  "ok_count": 1490, "skip_count": 10, "fail_count": 2}
  ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `all_ok` | bool | 所有步骤均成功时为 true，任意一步失败为 false |
| `total_elapsed` | int | 全流程耗时（秒） |
| `steps[].name` | string | 步骤名称 |
| `steps[].ok` | bool | 该步是否成功 |
| `steps[].elapsed` | int | 该步耗时（秒） |
| `steps[].updated` | int | 美股：本次实际更新的股票数 |
| `steps[].skipped` | int | 美股：已是最新、跳过的股票数 |
| `steps[].failed` | int | 美股：拉取失败的股票数 |
| `steps[].count` | int | 港股/纳指/A股：更新的股票总数 |
| `steps[].ok_count` | int | Wave 结构：成功数 |
| `steps[].error` | string | 步骤异常时的错误信息 |

### clawbot 推荐消息格式（数据更新）

```
🗄 数据更新报告 {update_date}  ({'✅ 全部正常' if all_ok else '⚠️ 有步骤失败'})
━━━━━━━━━━━━━━━━━━━━
✅ 美股 OHLCV   更新120 跳过1380 失败3   45s
✅ 纳指100      100只                    18s
✅ 港股 OHLCV   330只                    22s
✅ A股 OHLCV    12只                      8s
✅ 技术指标                             310s
✅ Wave 结构    成功1490 跳过10 失败2    620s
总耗时: 1830s
```

如有步骤失败（`ok=false`），额外附上 `error` 字段内容。

---

## 4. summary.json 格式说明（扫描结果）

`data/output/daily_scan/{YYYY-MM-DD}/summary.json`

```json
{
  "scan_date": "2026-04-12",
  "generated_at": "2026-04-12T08:17:45",
  "lookback_days": 1,
  "total_scanned": 339,
  "signals_found": 6,
  "has_gemini_analysis": true,
  "gemini_report_path": "/Users/wl/stock_ana/data/output/scan_analysis/2026-04-12/2026-04-12_RNG_AKAM_NYT_LRCX_KLAC_plus1more.md",
  "gemini_summary_table": "| 代号 | 公司 | 基本面/10 | 估值/10 | 技术/10 | 综合建议 |\n|...",
  "signals": [
    {
      "symbol": "RNG",
      "name": "RingCentral Inc",
      "signal": "STRONG_BUY",
      "score": 4,
      "entry_date": "2026-04-09",
      "support_band": "vegas_mid",
      "chart_path": "/Users/wl/stock_ana/data/output/vegas_scan/2026-04-12/STRONG_BUY_US_RNG_RingCentralInc_2026-04-09.png",
      "gemini_conclusion": "逐渐买入",
      "gemini_fundamental_score": "7.0",
      "gemini_valuation_score": "9.0",
      "gemini_technical_score": "8.0"
    }
  ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `scan_date` | string | 扫描日期 YYYY-MM-DD |
| `generated_at` | string | 生成时间戳 ISO 8601 |
| `total_scanned` | int | 本次扫描总标的数 |
| `signals_found` | int | 触发信号总数（含所有等级） |
| `has_gemini_analysis` | bool | Gemini 分析是否成功完成 |
| `gemini_report_path` | string/null | Gemini 报告完整路径（null 表示未运行或失败） |
| `gemini_summary_table` | string | 报告末尾汇总表格（Markdown 原文），供直接转发 |
| `signals[].signal` | string | `STRONG_BUY` / `BUY` / `HOLD` / `AVOID` |
| `signals[].score` | int | Vegas Mid 评分（+4 最强，负数为 AVOID） |
| `signals[].chart_path` | string | 对应 K 线图绝对路径 |
| `signals[].gemini_conclusion` | string | Gemini 综合建议（如 "逐渐买入"），分析失败时为空字符串 |

### 信号等级颜色参考

| signal | 建议含义 | 推荐前缀 |
|--------|----------|----------|
| `STRONG_BUY` | 强买入，多项共振 | 🟢 |
| `BUY` | 买入，待确认 | 🔵 |
| `HOLD` | 观察，暂不操作 | 🟡 |
| `AVOID` | 弱势，回避 | 🔴 |

---

## 5. clawbot 消息推送规范

### 推荐消息格式（Telegram / 微信）

clawbot 每天 08:30 左右读取 `summary.json`，按以下格式生成每日摘要并推送：

```
📊 每日扫描报告 {scan_date}
━━━━━━━━━━━━━━━━━━━━

扫描 {total_scanned} 只美股科技股 → 发现 {signals_found} 个信号

🟢 STRONG_BUY
• RNG（RingCentral Inc）score=+4  入场:2026-04-09
  Gemini: 逐渐买入

🔵 BUY
• AKAM（Akamai Technologies Inc）score=+3  入场:2026-04-09
  Gemini: 暂时观望

🟡 HOLD
• LRCX（Lam Research Corp）score=+0  入场:2026-04-02
  Gemini: 暂时观望
...

📋 完整报告：{gemini_report_path}
```

如当天无信号，推送：
```
📊 每日扫描报告 {scan_date} — 今日无信号触发
扫描 {total_scanned} 只科技股，均未满足入场条件。
```

如 Gemini 分析失败（`has_gemini_analysis=false`），推送信号但省略 Gemini 结论行。

### 图表推送

- 图表路径在 `signals[].chart_path`，为本机绝对路径 PNG 文件
- 只推送 `STRONG_BUY` 和 `BUY` 等级的图表
- `HOLD` 等级图表可选推送（视渠道流量而定）

---

## 6. 手动运行方式

```bash
cd /Users/wl/stock_ana

# 完整流程（扫描 + Gemini）
python vegas_mid_daily_scan.py

# 仅扫描，不调 Gemini（快速测试）
python vegas_mid_daily_scan.py --scan-only

# 扩大回看窗口（补漏信号）
python vegas_mid_daily_scan.py --lookback 3

# 手动触发数据更新（完整）
python daily_update.py

# 仅更新美股数据
python daily_update.py --us
```

---

## 7. 异常处理说明

| 场景 | 行为 |
|------|------|
| 无信号触发 | `signals_found=0`，`signals=[]`，仍生成 summary.json |
| Gemini 调用失败 | `has_gemini_analysis=false`，`gemini_report_path=null`，所有 `gemini_*` 字段为空字符串 |
| 数据未更新（今日无新K线） | run_scan 仍正常运行，lookback 内若有信号仍输出 |
| 扫描脚本崩溃 | 日志写入 `data/logs/cron_daily_scan.log` |
| 数据更新某步失败 | `status.json` 中对应 step `ok=false`，`all_ok=false`，其余步骤继续执行 |
| 数据更新脚本崩溃 | 日志写入 `data/logs/cron_daily_update.log` |

---

## 8. 相关文件索引

| 文件 | 说明 |
|------|------|
| `docs/weekly_sector_report_flow.md` | 周报流程与输出说明（weekly_sector_report + clawbot） |
| `vegas_mid_daily_scan.py` | 扫描 + Gemini 流水线脚本 |
| `daily_update.py` | 数据更新脚本（07:00 运行） |
| `cron_daily_update.sh` | 数据更新 shell 包装 |
| `data/us_tech_universe.csv` | 美股科技板块标的列表（339只） |
| `data/scan_signal_prompt.md` | Gemini 分析 prompt 模板 |
| `src/stock_ana/scan/vegas_mid_scan.py` | Vegas Mid 扫描核心逻辑 |
| `src/stock_ana/utils/scan_analyst.py` | Gemini 批量分析模块 |
| `data/output/daily_update/` | 数据更新状态目录（clawbot 入口 1） |
| `data/output/daily_scan/` | 扫描结论目录（clawbot 入口 2） |
