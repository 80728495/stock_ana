# Vegas Long 大浪回踩策略（vegas_long）

## 策略思想

强势上涨股的走势通常由连续几个大波浪构成，每个大浪的回踩深度一般止于
Long Vegas 通道（EMA144/169/200），而不会更深。本策略在**上涨周期**中捕捉
"从大涨幅浪顶回踩到 Long Vegas 通道并止跌回弹"的机会，并且：

- 偏好大浪序列中的**前两次回踩**（越靠后，深跌概率越高）；
- 只做**历史上显著以 Long Vegas 为回踩节点**的标的（统计硬门槛）。

与 vegas_mid（升浪内 EMA34/55 中继回踩）互补：mid 做浪内呼吸，long 做浪间交接。

## 四层过滤

### 1. 触发 — 触碰 + 站稳回弹（零前瞻）

`impl/vegas_long.detect_long_touch_and_hold`，复用
`primitives/vegas_pullback.detect_vegas_pullback` 状态机（`spans=LONG_EMAS`）：

- 单日触碰收回（low ≤ LV×1.02 且 close ≥ LV）→ 当日确认；
- 短暂刺破（≤2 日 close < LV）→ 两日站稳确认；
- `above_lookback=20`：触发前 20 日 ≥60% 收盘在 LV 上方（上方回踩，非下跌反抽）；
- `cooldown=15`（大浪节奏慢于 mid）。

### 2. 上涨周期门控 — `check_long_wave_structure`

Hard gate（全过才 passed）：

| 条件 | 含义 |
|---|---|
| long_rising | LV 上沿 20 日斜率 > 0 |
| long_order | EMA144 > EMA169（长期多头排列） |
| above_ratio ≥ 0.6 | 近 60 日 ≥60% 收盘在 LV 上方（上涨周期，非震荡） |
| rise_from_1y_low ≥ 30% | 较年低点涨幅（强势筛选） |
| peak_gap ≥ 10% | 近 60 日收盘高点高于当前 LV 上沿 ≥10%（确系"从大涨幅高点回踩"） |

辅助（不过滤）：`long_slope_strong`（LV 斜率 ≥ 2%）。

### 3. 大浪回踩判定 + 浪序 — `locate_wave_pullback`（硬门槛）

把触发点映射到它「终结」的大浪。大浪结构里每个浪的 `end_pivot` 就是一次
LV 回踩（终结本浪、启动下一浪），`wave_number` 即连续升浪链中的位置：

- **就近匹配已完成浪的 end_pivot**（|Δiloc| ≤ 15）→ 该触碰终结此浪，
  `pullback_seq = 该浪 wave_number`。
  注意**不用 `find_wave_context`**：共享边界处它返回「下一浪」，会把
  第 1 次回踩误报成第 2 次（off-by-one）——修复浪结构后此偏差才显现。
- **进行中末浪**（end_pivot=None）峰之后的触碰 → 实时回踩尚未被 zigzag
  确认，正在终结该浪，`pullback_seq = 末浪 wave_number`。
- 其余（浪内小回踩、建底期触碰）→ `is_wave_end=False`，**门槛否决**。

`is_wave_end` 是硬门槛：只做真正的大浪回踩，把"上涨途中贴一下 LV 又走"
和"上市初期在 LV 附近磨底"的噪声触碰全部挡掉。`pullback_seq` 第 1/2 次
加分、第 3 次起减分（"前两次回踩最好，越靠后深跌概率越高"）。

浪编号语义依赖 `analyze_wave_structure`，2026-07 大修后：连续性按
**浪间是否深破 LV** 判定（未深破 = 连续，`connected_prev=True`，编号递增；
真实断裂重置 W1），LV 触碰用真实 low 判定（EMA8 平滑值漏检下影触碰）。
详见 `docs/wave_structure_fix_2026-07.md`。

### 4. 统计硬门槛 — `compute_lv_respect_stats`

对历史所有已完成大浪的终点（即历史 LV 回踩）统计：

- **held（尊重）**：回踩点后 40 根 K 线内，收盘未连续 3 日 < LV×0.97
  （止跌于通道；浪间横盘不算失败）；
- **breach（深破）**：发生上述深破（synthetic 截断浪天然属于此类）；
- 最近 20 根 K 线内结束的最后一浪视为"进行中"，不计入（避免当下事件污染历史口径）。

`qualified = 事件数 ≥ 2 且尊重率 ≥ 0.6`。不合格 → 信号一律 AVOID。

## 打分（v1 结构先验，待回测校准）

`score_long_pullback`：

| 因子 | 规则 |
|---|---|
| seq | 第1次 +2 / 第2次 +1 / 第3次 -1 / ≥4次 -2 / 未知 0 |
| history | 尊重率 ≥0.75 且 ≥3 事件 +2；≥0.6 且 ≥2 事件 +1；其余 0 |
| slope | LV 斜率 ≥2% +1 |
| wave_rise | 本浪涨幅 30–150% +1；>250% -2（透支）；其余 0 |

`classify_long_signal`：≥4 STRONG_BUY / ≥2 BUY / ≥0 HOLD / <0 AVOID。

只有 `struct.passed AND stats.qualified AND is_wave_end` 三个硬门槛全过，
才用 score 分级；任一不过一律 AVOID。`best` 信号选择优先取非 AVOID、
再比分数（避免被否触碰盖过真实买点）。

## 模块接线

- 算法核心：`src/stock_ana/strategies/impl/vegas_long.py`
  （与原 `detect_long_touch_immediate` OBSERVE 观察层共存）
- 标准入口：`strategies/api.py` — `screen_vegas_long_pullback` /
  `scan_vegas_long_pullbacks` / `explain_vegas_long_pullback`
- 注册：`strategies/registry.py` — `scan_strategy("vegas_long", ...)`，
  kind = `stateful_signal`
- 测试：`tests/test_vegas_long_pullback.py`

## 实测口径

**连续多浪持仓（lookback=800，验证浪序打分）**：心动 seq 1→2→3→4 对应
STRONG_BUY(+6)→STRONG_BUY(+5)→BUY(+3)→HOLD(+1)，"前两次最好、越靠后越
降级"完全生效；华虹 seq1 STRONG_BUY→seq2 HOLD；中芯 15 个触碰仅 3 个真·
大浪回踩通过 `is_wave_end`，建底/浪内噪声全被挡。

**美股宇宙（前 600 只，lookback=60）**：306 触发 → 结构门槛否 154、统计
门槛否 137、`is_wave_end` 额外否少量 → 合格信号按 seq 1/2/3 分布，
STRONG_BUY 集中于前两次回踩（CFG、AIR、ARR、EIX 等）。

## 已知限制 / 后续

- 打分为结构先验，尚未做历史回测校准（vegas_mid 的 v2 权重是回测校准过的）；
  可仿照 `backtest/backtest_vegas_mid.py` 建 `backtest_vegas_long.py`。
- 统计门槛依赖 `analyze_wave_structure` 的浪识别质量；上市 <2 年或
  首轮主升的标的天然事件数不足（n_events < 2），会被排除——这是保守取舍。
- 每日扫描如需图表/邮件流，可仿照 `scan/vegas_mid_scan.py` 加 `scan/vegas_long_scan.py`；
  目前 `vegas_mid_scan` 的 `long_touch` OBSERVE 通道保持原样未动。
