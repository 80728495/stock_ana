# SMC Order Block 打分系统 — 设计文档

> 供 Codex / AI 快速接手使用，记录截至 2026-06-01 的完整实现状态。

---

## 目录

1. [系统概览](#1-系统概览)
2. [核心文件地图](#2-核心文件地图)
3. [因果 OB 检测 `_ob_causal`](#3-因果-ob-检测-_ob_causal)
4. [质量特征提取 `ob_quality_score`](#4-质量特征提取-ob_quality_score)
5. [连续打分规则 `OB_SCORE_RULES`](#5-连续打分规则-ob_score_rules)
6. [Zone Score 叠加](#6-zone-score-叠加)
7. [每日追踪器 `smc_ob_tracker`](#7-每日追踪器-smc_ob_tracker)
8. [图表生成 `gen_ob_score_charts`](#8-图表生成-gen_ob_score_charts)
9. [运行入口](#9-运行入口)
10. [已知问题 / 待改进](#10-已知问题--待改进)

---

## 1. 系统概览

```
OHLCV 历史数据
    │
    ▼
_ob_causal()          ← 无未来信息的因果 OB 检测
    │
    ▼
ob_quality_score()    ← 提取 7 个量化特征
    │
    ▼
ob_quality_rating()   ← 按 OB_SCORE_RULES 加权打分 → 0~100 连续分
    │
    ▼
_compute_zone_scores()← 同方向重叠 OB 的分数叠加 → zone_score
    │
    ├─▶ smc_ob_tracker  每日增量扫描，生成 new_ob / touched / mitigated 事件
    │
    └─▶ gen_ob_score_charts  历史图表（含已突破 OB），供视觉验证
```

**设计理念：**
- 替换原始的二元 `quality_pass` 筛选，改为连续打分（0-100），分数代表 OB 的支撑/阻力强度
- 得分高的 OB 在图表中颜色更深，在通知中优先展示
- Zone score 解决"同价位多个 OB 叠加 = 更强支撑"的表达问题

---

## 2. 核心文件地图

| 文件 | 角色 |
|------|------|
| `src/stock_ana/strategies/impl/smc.py` | **核心**：OB 检测 + 特征提取 + 打分规则 |
| `src/stock_ana/scan/smc_ob_tracker.py` | **扫描器**：每日增量状态追踪，输出事件 |
| `scripts/gen_ob_score_charts.py` | **图表**：历史 OB 评分可视化 |
| `daily_update.py --smc` | **入口**：定时调度 tracker |
| `data/cache/smc_ob_state/{market}/{symbol}.json` | **状态文件**：持久化每只股票的 OB 状态 |
| `data/output/ob_score_charts/` | **图表输出**：`{MARKET_}symbol_ob_score.png` |

---

## 3. 因果 OB 检测 `_ob_causal`

**文件：** `src/stock_ana/strategies/impl/smc.py`

**问题背景：**
上游 `smartmoneyconcepts.smc.ob()` 的 `swing_highs_lows()` 使用前后各 `swing_length` 根 K 线来确认摆动点，导致实际运行时存在未来信息泄漏（look-ahead bias）。

**修复策略：**
自行实现 OB 检测。处理 bar `i` 时，只有满足 `k + swing_length ≤ i` 的 swing 点才可见：

```python
visible_mask = all_swing_high_indices + swing_length <= i
visible_highs = all_swing_high_indices[visible_mask]
```

**函数签名：**
```python
def _ob_causal(
    ohlc: DataFrame,
    swing_highs_lows: DataFrame,
    swing_length: int = 5,
    close_mitigation: bool = False,
) -> DataFrame
```

**返回列：**
`OB (1=bull, -1=bear)`, `Top`, `Bottom`, `OBVolume`, `MitigatedIndex`, `Percentage`

**OB 消除逻辑：**
- 看涨 OB：`low[i] < bottom` → breaker；`low[i] < bottom` 再次则删除
- 看跌 OB：`high[i] > top` → breaker；`high[i] > top` 再次则删除
- `close_mitigation=True` 时改用收盘价判断

---

## 4. 质量特征提取 `ob_quality_score`

**文件：** `src/stock_ana/strategies/impl/smc.py`

```python
def ob_quality_score(ohlcv, ob_df, bar_idx) -> dict
```

| 特征 | 计算方式 | 含义 |
|------|----------|------|
| `direction` | `int(row["OB"])` | 1=看涨, -1=看跌 |
| `ob_width_pct` | `(top-bottom)/mid*100` | OB 区间相对宽度 (%) |
| `body_ratio` | `abs(close-open)/(high-low)` | 实体占影线比 |
| `vol_ratio` | `vol / mean(vol[-20:])` | OB bar 相对成交量 |
| `trend_before_5d` | `(close[i] - close[i-5]) / close[i-5] * 100` | OB 前 5 日涨跌幅 (%) |
| `ob_atr_ratio` | `(top-bottom) / ATR14` | OB 宽度相对 ATR |
| `percentage` | `row["Percentage"]` (来自上游) | 成交量不对称度 (0~100) |

---

## 5. 连续打分规则 `OB_SCORE_RULES`

**文件：** `src/stock_ana/strategies/impl/smc.py`

### 打分机制

```python
def _linear_score(value, zero_val, full_val) -> float:
    # zero_val → 0 分, full_val → 满分, 线性插值, 截断到 [0, 1]
    return max(0.0, min(1.0, (value - zero_val) / (full_val - zero_val)))

def ob_quality_rating(ohlcv, ob_df, bar_idx, features=None) -> tuple[float, dict]:
    # 返回 (total_score 0~100, breakdown dict)
```

每条规则格式：`(特征名, zero_val, full_val, weight)`

### 看涨 OB 规则（权重合计 100）

| 特征 | 0 分值 | 满分值 | 权重 | 逻辑说明 |
|------|--------|--------|------|----------|
| `ob_width_pct` | 1.0% | 10.0% | 30 | 区间越宽 → 需求区越厚实 |
| `trend_before_5d` | 0% | -10% | 25 | OB 前跌得越深 → 买盘介入更有意义 |
| `vol_ratio` | 0.5x | 2.5x | 15 | 放量 → 机构参与 |
| `ob_atr_ratio` | 0.3x | 2.0x | 15 | 宽度相对 ATR 大 → 信号明显 |
| `body_ratio` | 0.2 | 0.8 | 10 | 实体大 → 方向明确 |
| `percentage` | 50% | 10% | 5 | 成交量不对称 → 单边力量（反向） |

### 看跌 OB 规则（权重合计 100）

| 特征 | 0 分值 | 满分值 | 权重 | 逻辑说明 |
|------|--------|--------|------|----------|
| `ob_atr_ratio` | 0.3x | 2.5x | 25 | 供应区宽度显著 |
| `vol_ratio` | 0.5x | 2.5x | 20 | 放量 → 机构卖出 |
| `percentage` | 50% | 15% | 20 | 成交量不对称 → 卖压集中（反向） |
| `ob_width_pct` | 1.0% | 8.0% | 15 | 供应区厚实 |
| `trend_before_5d` | 0% | +8% | 10 | OB 前涨得多 → 获利盘堆积 |
| `body_ratio` | 0.2 | 0.8 | 10 | 实体大 → 方向明确 |

### 评分分档（图表着色用）

| 分段 | 标签 | 看涨颜色 | 看跌颜色 |
|------|------|----------|----------|
| ≥ 70 | 极强 | 深绿 `#006400` | 深红 `#B71C1C` |
| ≥ 50 | 强 | 绿 `#009900` | 红 `#E53935` |
| ≥ 30 | 中 | 浅绿 `#66BB6A` | 浅红 `#EF9A9A` |
| < 30 | 弱 | 很浅绿 `#A5D6A7` | 很浅红 `#FFCDD2` |

---

## 6. Zone Score 叠加

**文件：** `smc.py` (`_apply_zone_scores`) 和 `smc_ob_tracker.py` (`_compute_zone_scores`)

**逻辑：**
同方向的活跃 OB，若价格区间 `[bottom, top]` 相互重叠，视为同一 zone。
该 zone 内所有 OB 的个体 `score` 相加 → `zone_score`。

```
zone_score = sum(ob.score for ob in zone_obs)
```

**意义：**
- 单个 OB score=60 → 中等强度支撑
- 同位置两个 OB score=60+55 → zone_score=115，表示该价位有双层机构支撑

**在追踪器中：**
```python
def _compute_zone_scores(ob_map: dict) -> None:
    # 修改 ob_map 中每个 OB 的 "zone_score" 字段（in-place）
```

**在图表中：**
活跃 OB 右边显示 `▲60` 或 `▼45`，若 zone_score > score 则额外显示 `z=115`。

---

## 7. 每日追踪器 `smc_ob_tracker`

**文件：** `src/stock_ana/scan/smc_ob_tracker.py`

### 状态文件格式

路径：`data/cache/smc_ob_state/{market}/{symbol}.json`

```json
{
  "last_updated": "2026-06-01",
  "swing_length": 5,
  "obs": {
    "2026-05-20_bull": {
      "ob_id": "2026-05-20_bull",
      "direction": 1,
      "top": 185.5,
      "bottom": 181.2,
      "formed_date": "2026-05-20",
      "score": 72.5,
      "zone_score": 72.5,
      "score_detail": {"ob_width_pct": 18.5, "trend_before_5d": 20.0, ...},
      "status": "active",
      "mitigated_date": null
    }
  }
}
```

### 事件类型

| 事件 | 触发条件 | 携带字段 |
|------|----------|----------|
| `new_ob` | 今日首次检测到的新 OB | `score`, `zone_score`, `top`, `bottom`, `direction`, `formed_date` |
| `touched` | 活跃 OB 被今日 high/low 刺入 | `score`, `zone_score`, `top`, `bottom` |
| `mitigated` | 活跃 OB 被消除（breaker） | `score`, `mitigated_date` |

### 主要函数

```python
# 处理单只股票
process_symbol(symbol, market, df_ohlcv, swing_length=5) -> list[dict]

# 批量每日扫描（读 futu_watchlist.md）
run_daily(watchlist=None, swing_length=5) -> dict[str, list[dict]]

# 查询单只股票活跃 OB（不重算，读状态文件）
get_active_obs(symbol, market) -> list[dict]
```

### 二元筛选（已废弃）

原有的 `quality_filter` / `quality_thresholds` 参数保留为向后兼容，但实际不再使用。
所有 OB 均计算 `score`，由调用方根据 score 决定是否展示/通知。

---

## 8. 图表生成 `gen_ob_score_charts`

**文件：** `scripts/gen_ob_score_charts.py`

### 特点
- 展示全部 OB，包括历史已突破的（以虚线灰色矩形标注）
- 活跃 OB 按评分四档着色
- 矩形横跨：从 OB 形成 bar 延伸到突破 bar（或图表末尾）
- 每个 OB 右边标注 `▲score`（看涨）或 `▼score`（看跌）
- 若 zone_score > score，额外标注 `z=zone_score`

### 文件名规则
- 美股：`{symbol}_ob_score.png`（如 `NVDA_ob_score.png`）
- 港股：`HK_{symbol}_ob_score.png`（如 `HK_00700_ob_score.png`）
- A 股：`CN_{symbol}_ob_score.png`（如 `CN_688981_ob_score.png`）

### 用法
```bash
# 全量自选股
python scripts/gen_ob_score_charts.py

# 仅美股
python scripts/gen_ob_score_charts.py --market us

# 仅港股
python scripts/gen_ob_score_charts.py --market hk

# 仅 A 股
python scripts/gen_ob_score_charts.py --market cn

# 指定个股
python scripts/gen_ob_score_charts.py NVDA AAPL 00700

# 自定义回看 K 线数
python scripts/gen_ob_score_charts.py --lookback 300 --swing_length 5
```

---

## 9. 运行入口

```bash
# 每日完整 SMC 更新（读 futu_watchlist.md，增量计算）
.venv\Scripts\python.exe daily_update.py --smc

# 全量重建 OB 状态（删除状态文件后重跑）
Remove-Item -Recurse data/cache/smc_ob_state
.venv\Scripts\python.exe daily_update.py --smc

# 生成评分图表（全量）
.venv\Scripts\python.exe scripts/gen_ob_score_charts.py

# 生成指定市场图表
.venv\Scripts\python.exe scripts/gen_ob_score_charts.py --market hk
.venv\Scripts\python.exe scripts/gen_ob_score_charts.py --market cn
```

---

## 10. 已知问题 / 待改进

### 待实验
- [ ] **权重调参**：`OB_SCORE_RULES` 中的 `zero_val` / `full_val` 均基于经验设定，未经系统优化。可用 `scripts/backtest_smc_ob.py`（已有）对历史 OB 做回归分析，找出与胜率相关性最强的阈值。
- [ ] **look-back 窗口**：`trend_before_5d` 固定用 5 日，`ATR` 固定用 14 日。不同市场波动率差异大，是否应自适应？
- [ ] **看跌 OB 验证**：回测数据主要为美股科技股（多头市场），看跌 OB 的规则权重较少回测支撑。

### 已知限制
- `ob_quality_score` 的 `trend_before_5d` 对 `bar_idx < 5` 的 OB 返回 `0.0`（早期数据不足）
- 状态文件以 `formed_date_direction` 为 key，同一天同方向最多记录一个 OB（极少发生，但理论上可能冲突）
- `_compute_zone_scores` 仅处理活跃 OB（已突破的不参与叠加计算）

### 扩展方向
- 加入时间衰减因子（老 OB 权重递减）
- FVG 与 OB 重叠检测（FVG 填满 + OB 活跃 = 双重确认）
- 多时间框架叠加（日线 OB + 周线 OB 同区间 = 更强信号）
