# 逃顶信号系统（Escape Signal Tracker）

持仓「见顶逃顶」的每日信号系统：发现疑似见顶 → 观察 → 结构确认下跌 / 恢复上涨取消。

- 核心模块：[`escape_signal_tracker.py`](../src/stock_ana/research/top_reversal/escape_signal_tracker.py)
- 每日入口：[`daily_escape_scan.py`](../src/stock_ana/research/top_reversal/daily_escape_scan.py)

> **一句话**：每天对持仓做一次轻量扫描，产生「新报警 / 取消 / 下跌确认」三类事件，
> 状态持久化、逐日顺延。**每日任务不跑 build**——build 只是周期性地产训练集和模型。

---

## 1. 为什么分「早发现」和「进一步确认」两段

一个见顶判断有两个不同阶段，混在一起会互相拖累：

| | 段 A 早发现 | 段 B 进一步确认 |
|---|---|---|
| 目的 | 尽早发现「疑似见顶」，抢在确认前 | 判断疑似顶是否**真的**走出下跌结构 |
| 方法 | **ML 打分**（lgb，早期可得特征） | **确定性规则**（SMC 结构反转，非 ML） |
| 时点 | 顶后 1-2 天（召回即触发） | 顶后结构走出来才成立 |
| 类比 | `smc_early` | `smc_confirm` |

**关键纪律**：早发现的打分**不能依赖「顶后才发育」的特征**（如「确认日距顶跌幅」、
swing-confirmed 召回等）。这类特征要等价格跌出来才有值，会系统性压低新鲜点的分数
（= 变相「等确认」），且实测对样本外 AUC 零增益。因此段 A 用
`DISCOVERY_FEATURE_COLS`（= `REALTIME_FEATURE_COLS` 剔除 14 个顶后确认型特征，
见 `feature_registry.POST_CONFIRMATION_FEATURE_COLS`）。「确认」交给段 B 的结构规则。

> 打标签（训练样本）那侧仍用最严格的 swing CHoCH↓（保样本干净），与发现侧刻意不同。

---

## 2. 状态机

每个标的一条活跃 saga（信号从生到灭），terminal 后由**更新的顶**重新起 saga。

```
 (无 / 已了结)
     │  当日 strength ≥ entry
     ▼
  watching ──[strength ≥ alert 且未报过]──▶ (发 alert)
     │
     ├─[段B: 收盘越顶 = 恢复上涨]────────▶ cancelled   （发 cancel，仅当此前 alert 过）
     │
     └─[段B: swing 级看跌破位，且未收复]──▶ confirmed_down （发 confirm_down，无条件）
```

**四类事件**（交外围发通知）：

| 事件 | 触发 | 通知规则 |
|---|---|---|
| `new_signal` | strength ≥ `entry_threshold`（0.35）→ 进观察队列 | 入队，通常不单独打扰 |
| `alert` | strength ≥ `alert_threshold`（0.60） | **报警**，要提醒 |
| `confirm_down` | 观察中的顶走出 swing 级看跌破位、且从未收复顶部 | **无条件通知**（哪怕之前没报警过） |
| `cancel` | 观察中的顶被收盘越顶（恢复上涨、信号被吃回中继） | **仅当此前 alert 过**才发取消 |

**段 B 判定优先级**：**收盘创新高越过顶 = 顶被证伪 → `resumed`（优先）**；
只有从未重新越顶、又走出 swing 级看跌破位，才 `confirmed_down`。这样越顶永不误判成确认。

### confirm 判据（可切换）

`SignalConfig.confirm_include_bos`：

- `False` — 仅 swing **CHoCH↓**（= 打标签同一判据）。对慢反转干净，但**急跌会漏**：
  价格崩太快时 swing 级先出 BOS↓、等不到一个干净的 CHoCH↓（实测 LITE 跌 26% 仍卡「观察」）。
- `True`（**默认**）— swing **CHoCH↓ 或 BOS↓**（任一 swing 级看跌破位）。能抓急跌。
  对比实测：比 CHoCH-only 多确认真下跌（LITE −26%、NVDA 等），且 cancel 集合完全一致
  （因越顶优先，不会把恢复上涨误判成确认）。故默认采用。

命令行 `--confirm-mode {choch, choch_bos}` 切换。

---

## 3. 数据流：扫描（每日）vs build（周期性）

```
周期性(build)                          每日(scan)
──────────────                        ──────────────
build --stage all                     daily_escape_scan
  → labeled 训练集 ────────训练───▶  段A: score_discovery(lgb)
  → 模型/评估                              │  用最新价召回持仓候选(scan_candidates)
                                           ▼
                                     段B: structural_status(SMC 结构)
                                           ▼
                                     状态机 advance_state → 事件 + 状态
```

- **`scan_candidates(asof=None)`**：每日推断所需的「召回 + 特征」，复用 build 的
  `_build_symbol_research_rows` + `add_research_features`（与训练**零漂移**）。
  **只对持仓召回候选**，但加载全宇宙价格（仅读 parquet）以给横截面/行业特征提供上下文。
  **不打标签、不训模型、不评估、不扫全宇宙候选** → 比 build 轻得多。
- 训练集 `watchlist_unified_recall_candidates_labeled.csv` 由 **build 周期性**产出（模型的记忆），
  日常不需要重跑。价格数据由你的日常同步流程更新到最新交易日。

---

## 4. 每日脚本用法

```bash
# 首次：回填最近 N 天（默认 60）并建立状态
python -m stock_ana.research.top_reversal.daily_escape_scan --backfill

# 之后每天（外围定时调用）：增量顺延，只推进上次扫描日之后的新交易日
python -m stock_ana.research.top_reversal.daily_escape_scan

# 回测/补跑：把「今日」定到某历史日
python -m stock_ana.research.top_reversal.daily_escape_scan --asof 2026-06-30
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--state-path` | `data/output/top_candidate_research/escape_signal_state.json` | 状态文件 |
| `--events-out` | 状态目录下 `escape_events_<date>.json` | 本轮事件（供外围发通知） |
| `--backfill` | 关 | 强制从空态回填（重建状态） |
| `--backfill-days` | 60 | 回填天数 |
| `--confirm-mode` | `choch_bos` | 下跌确认判据（见 §2） |
| `--entry-threshold` / `--alert-threshold` | 0.35 / 0.60 | 入队 / 报警强度阈值 |
| `--asof` | 最新价格日 | 覆盖「今日」 |

**首跑 vs 增量**：无状态或 `--backfill` → 回填整段并存盘；否则加载状态、只推进
上次扫描日之后的新交易日，老信号在已存状态上顺延演化，**已通知过的不重发**。

---

## 5. 文件格式

### 状态文件 `escape_signal_state.json`

```jsonc
{
  "last_scan": "2026-07-02",
  "config": { "entry": 0.35, "alert": 0.60, "confirm_include_bos": true },
  "signals": {
    "US:GOOG": {
      "market": "US", "sym": "GOOG",
      "state": "watching",          // watching / confirmed_down / cancelled
      "top_date": "2026-05-18", "top_high": 210.5,
      "first_signal_date": "2026-05-19", "last_update": "2026-07-02",
      "peak_strength": 0.71, "alerted": true,
      "resolved_date": null, "drop_pct": NaN
    }
  }
}
```

### 事件文件 `escape_events_<date>.json`

一个事件数组，每项：

```jsonc
{
  "kind": "confirm_down",          // new_signal / alert / confirm_down / cancel
  "market": "US", "sym": "LITE", "date": "2026-07-01",
  "top_date": "2026-05-12",
  "strength": NaN, "drop_pct": -26.2,
  "mechanism": "swing_break_down",
  "message": "下跌结构确认（swing CHoCH↓ @ 2026-07-01），距顶 -26.2%"
}
```

---

## 6. 外围系统脚本怎么接

本系统**只负责扫描 + 状态**，不发消息。外围（定时任务）：

1. 定时调用 `daily_escape_scan`（每交易日收盘后）。
2. 读 `--events-out` 的事件数组：
   - `alert` → 发「⚠️ 见顶报警」；
   - `confirm_down` → 发「✅ 下跌结构确认」（无条件，含距顶跌幅）；
   - `cancel` → 发「↩︎ 信号取消（恢复上涨）」；
   - `new_signal` 一般静默（仅入队，按需汇总）。
3. 状态文件已由脚本自动持久化，无需外围维护。

---

## 7. 相关

- 特征分层：`feature_registry.DISCOVERY_FEATURE_COLS` / `POST_CONFIRMATION_FEATURE_COLS`
- 结构判据与打标签的关系：见 `_smc_structural_top_confirmed`（build，swing CHoCH↓）
- 训练/评估：`build_top_candidate_research.py`（`--stage`）、`eval_watchlist_oos.py`
