# 浪结构连续性修复（2026-07-07）

## 背景

`analyze_wave_structure` 的浪起点/终点 refinement（把边界精确到贴近 LV
的 bar）会把相邻两浪原本共享的 LV 触点向相反方向推开，导致两处下游
判定失效（300 只美股实测）：

1. `backward_consecutive_count` 要求 `end.iloc == next.start.iloc` 严格
   相等 —— **0% 命中**，consec 恒为 1，vegas_mid 的 `three_wave` 因子
   从未生效；
2. 重编号要求浪间 gap ≤ 15 bar —— 只有 23% 满足（实际浪间横盘中位
   62 bar），488/554 个浪编号为 1，164/268 只标的出现重复 W1。

衍生 bug：`wave_number` 被当作身份键使用（`wave_map`、
`wave_touch_counter`），重复编号导致查错浪 / touch_seq 跨浪段累加。

## 修复内容

### 1. boundary id 同源判定（wave.py）

相邻两浪在 refinement **之前**共享同一个 merged LV 触点。现在每个浪
记录：

- `start_boundary_id`：起点原始触点 iloc（refinement 前）
- `end_boundary_id`：终点原始触点 iloc；synthetic 深破截断时置 None

连续 = `prev.end_boundary_id == curr.start_boundary_id` 且本浪起点价
≥ 前浪起点价。不再使用 iloc 距离。

### 2. 编号在资格过滤前、对完整浪链进行

弱浪（涨幅 <15% 被过滤）也是一次真实的 LV 回踩事件，"第 N 次回踩"
序次将其计入——幸存浪保留完整链中的序号（可能出现 W1、W3 跳号）。

### 3. 身份键修正

- `backward_consecutive_count(waves, wave)` —— 改传 wave dict（原来传
  wave_number int，会撞重复编号）；内部按列表位置回溯 boundary 链。
- `wave_touch_counter` 改按 `start_pivot.iloc`（浪身份）分桶
  （vegas_mid_scan、api 两处）。

### 4. EMA200 暖机期过滤

`analyze_wave_structure(ema_warmup_bars=200)`：前 200 bar 的低点不参与
LV 触碰判定——ewm(adjust=False) 早期 EMA 贴价格走，"触碰"是伪事件
（修复前 32% 的浪起点落在暖机期内）。

### 5. 杂项清理

移除 `detect_ema8_swings` 未使用的 `window` 参数、未使用的
`max_gap_bars`/`end_bar`；修正多处 "EMA34/55/60" 过时文档（实为 34/55）。

## 修复后实测（300 只美股）

| 指标 | 修复前 | 修复后 |
|---|---|---|
| wave_number 分布 | W1×487 W2×60 W3×6 | W1×314 W2×53 W3×4（且暖机伪浪已剔除） |
| consec 计数 | 恒为 1 | 1×219 / 2×16 / 3×3 |
| 断点构成 | — | 62% 深破重置（正确语义）、17% 弱浪断链（保序跳号）、21% 连续 |
| vegas_long seq 分布 | 恒为 1 | 1×208 / 2×47 / 3×5 |

vegas_long 实际效果：AGI/BBIO/CBOE 等第 3 次回踩信号从 STRONG_BUY
正确降级为 BUY（seq 减分生效）。

## 增补（2026-07-08）：LV 触碰改用真实 low 判定

持仓实盘图复查发现三例浪被错误合并（华虹 2025-06/2026-04、心动
2024-09/2025-04 的 LV 触碰未被识别为浪边界；小米 2024-03 起点因
tail 截断落入暖机期）。根因：

1. **EMA8 枢轴值漏检真实触碰**：`_touches_long_vegas` 原来只比较
   EMA8 平滑值 ≤ LV×1.03。V 型底处 EMA8 比真实低点高 4~10%，深下影
   触碰系统性漏检（华虹 2025-06-16：EMA8/LV=1.042 判负，真实
   low/LV=0.988）。
   **修复**：判据改为并集——EMA8 判据 **或** 枢轴附近 [i-7, i+2]
   窗口内任一 bar 的真实 low ≤ 当日 LV×1.03。
2. **调用方不得截断历史**：浪结构必须用全量缓存计算（EMA 暖机吃掉
   前 200 bar），只在绘图/展示层截取时间窗。

修复后全市场（300 只美股）：浪总数 371→400（+8%，无碎浪爆炸），
W2×87 / W3×11 / W4×2，短浪（峰距起点<50bar）占比 12%。
持仓实测：华虹 W2→W3→W4、小米 W1(2024-03~08)→W2→W3、
心动 W1→W2→W3→W4，均与人工判读一致。

注意：`_build_major_wave_v2` 内的 `_touches_mid`（子浪计数）仍是
EMA8 值判据，存在同类漏检——影响 sub_number/sub_pos 因子输入，
留待 vegas_mid 重校准时一并处理。

## 增补（2026-07-08 之二）：编号改为"过滤后 survivor + 浪间深破"判定

持仓图复查发现 AAOI/DDOG/LUNR 等被标成连续 W1→W2，但两段其实是相隔
6 个月、中间深破 LV 的两次独立行情。根因：上一版把编号放在**资格
过滤之前**对完整浪链做，中间被过滤掉的弱浪用 boundary id 把编号链跨过
深破缺口续上了，survivor 保留了误导性的链号。

**改法**：编号移回**过滤之后**，只对 survivor 排号；连续性判据从
"boundary id 相等"改为"**相邻两个合规浪之间价格是否深破 Long Vegas**"
（复用 `_has_lv_breach`，≥3 日收盘 < LV×0.97 即断裂）+ 起点价不降。
每个浪写 `connected_prev` 布尔，`backward_consecutive_count` 改走该标志
（不再需要 boundary id，也不再访问 EMA 序列）。

优点：中间被过滤的弱回踩不再打断编号（不跳号，W1→W2→W3 干净递增），
真实断裂处（大缺口+深破）正确重置为 W1。实测：华虹/心动/小米 =
连续多浪链；AAOI/DDOG/LUNR/GOOG/MRVL = 各自独立 W1。

全市场（300 只美股）：W1×346 / W2×48 / W3×5 / W4×1，
connected_prev 与 wave_number 一致性 0 违规。

`start_boundary_id` / `end_boundary_id` 字段保留（信息性，标记 refinement
前的原始触点），但已不是编号判据。`compute_lv_respect_stats` 本就用
自己的深破判定，不受影响。

## 输出目录注意

持仓浪结构图**不要**输出到 `data/output/` —— `tests/test_screener.py`
的 fixture 会 `shutil.rmtree(data/output)`，跑测试会连图一起删。改输出到
项目根 `charts/`（已加入 .gitignore）。生成脚本：
`scratchpad/plot_holding_waves.py`（临时研究脚本，不入库）。

## 后续注意

- **数据窗口限制**：缓存仅约 3 年，扣除暖机期后大多数标的只装得下
  1-2 个大浪，浪链天然偏短。若要充分利用"第 N 浪"统计（以及
  vegas_long 的历史尊重率门槛），应加深历史行情缓存（建议 ≥8 年）。
- **vegas_mid 打分需重校准**：`touch_seq` / `consec_waves` 行为已改变
  （touch_seq 不再跨浪段累加、consec 可 >1），v2 权重是在旧行为上
  校准的，建议重跑 `backtest_vegas_mid` 校准。
- wave_structure JSON 缓存（data/cache/wave_structure/）中的旧数据缺
  boundary 字段，下次每日更新会自动覆盖；消费方均用 `.get()` 读取，
  无兼容问题。
