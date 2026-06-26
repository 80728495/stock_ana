# 顶部识别当前系统说明

> 截至 2026-06-21。本文档描述当前代码里的顶部识别研究系统，重点说明候选如何产生、样本如何标注、模型如何训练，以及双顶逻辑在当前体系中的位置。
>
> **下方「§0 2026-06-21 修订」是当前最新状态的权威描述；其后 §1–§6 为 2026-06-20 基线，部分细节已被 §0 覆盖（尤其召回源、特征体系、分类器）。**

---

## 0. 2026-06-21 修订（当前最新状态）

本轮对召回源、特征体系、分类器做了较大调整，关键结论：**顶部判断的瓶颈在「有效特征」而非「分类器复杂度」**；regime（板块/股性）特征是当前最重要的新增维度。

### 0.1 召回源变化

| 召回源 | 状态 | 说明 |
|---|---|---|
| `shadow` 长上影 | 保留 | 蜡烛召回 |
| `doji` 跳空十字星 | 保留 | 蜡烛召回 |
| `gap_fail` 跳空大阴线 | **移出主召回** | 前向统计证明它是「即刻回撤/逃顶择时」信号而非持续顶（20日后收涨占多数）。参数已收紧为高精度（`gap_fail_reversal.py`），仅作持仓逃顶提醒/确认特征 |
| `smc_appear` | **移出主召回** | 只借 OB 几何外壳、无位移无结构破坏，本质「近高位即刻回落」探测器，在上涨途中每个中继高点都误报 |
| `smc_supply_held` | **新增** | appear 的正确替代：近高位高分拒绝 bar，未来 ≤8 天内「高点不被收复 且 收盘跌破锚低」才确认（供给真正守住） |
| `smc_top_confirmed` | **新增（统一源）** | 对每个近高位锚点，取 `supply_held / early / confirmed` 三机制中**最早确认**者（min lag）。三者互补，并集真顶覆盖 ~74% > 任一单源 ~70% |

代码：`candidate_sources.py::collect_smc_supply_held_candidates` / `collect_smc_top_confirmed_candidates`。主 build 统一召回现为 `[pattern(shadow/doji), smc_top_confirmed]`。

### 0.2 Mid Vegas 短历史豁免

`vegas_context.py::_bar_context`：原 `history_ready = pos >= max(LONG_EMAS)=200` 会把总长不足 200 根的短历史标的（如 02788 仅 ~130 根）**整只剔除**，即便它在自身尺度上完全满足 Vegas 多头结构。现改为：总长 < 200 的标的只要 `pos >= max(MID_EMAS)=55` 即视为 ready（真实多头结构仍由 `check_mid_vegas_structure` 把关）。修复后 02788 的 04-08 大顶被模型判为强顶（top9%）。

### 0.3 新增特征组

| 组 | 文件 | 特征 | 验证 |
|---|---|---|---|
| `prior_high_structure` | `prior_high_context.py` | `top_close_above_prior_high`、`top_close_vs_prior_high_pct`、`top_high_vs_prior_high_pct`、`top_m_shape`（严格双顶，间隔≥15根）等 | 「收盘站不上前高」单变量真顶率 9.5% vs 5.5%；但**股性依赖**（US 有效、CN 反转） |
| `macro_micro` | `macro_micro_context.py` | 宏观：`is_semiconductor`、`sector_peers_ret60_mean`、`sector_overheat_pct`、`xsec_ret20/60_pct`；微观：`micro_accel_20_60`（抛物线加速）、`micro_verticality_atr`、`micro_dist_ema20_atr`、`micro_up_streak` | AUC +0.028；这些 regime 特征成为两个模型的核心（LR 系数前列、lgb gain 前6占5） |

板块映射：US 用 **SIC 3 位主组**（如 `US_SIC367`=半导体）做二级分类，比粗 GICS sector 更细（半导体独立 peer 组）；HK 用行业映射；CN 暂为市场级。`is_semiconductor` 在模型里是强「continuation+」折扣——**半导体异常 melt-up 中顶信号更不可信，正是这一 regime 被学进模型**。

### 0.4 分类器：LR + lightgbm 并行

`modeling.py::fit_lightgbm`（强正则 GBT：d3/leaf7/min40/类权重/5seed 平均）与 `fit_logistic` 并行训练，每次 build 同时输出两套打分（`top_candidate_lgbm_scored.csv` 等）。LR 保留为可解释基线。

**为什么需要树**：regime 信号是「特征×板块」交互（前高在不同股性下方向翻转、半导体顶信号打折），LR 线性结构无法表达，树能原生学习。

### 0.5 当前结果（holding 样本外，train=watchlist−holding / test=holding）

| 指标 | 值 |
|---|---|
| 统一召回真顶覆盖 | 75.9%（旧 73.7%） |
| LR：top8% / top30% / AUC | ~25% / 13.8% / **0.762** |
| lgb：top8% / top30% / AUC | ~24% / 13.3% / **0.783** |

- **AUC 上 lgb 清晰更优、细化板块逐步抬升 AUC**；
- **top8%（最高置信档）仍是 ~24-25% 的噪音平局**，没拉开 → 突破要靠更厚的有效特征，不是模型。

### 0.6 待办

- 更细 regime：跟随过热指数的 beta、板块宽度。
- 评估口径：holding 样本外为准；**不在宇宙池验证**（传统行业+科技股性不同会互相对冲）。
- **A+H 双重上市单独策略**（HK 中有一些；CN牛/HK熊，同公司两地走势可对照）。
- **逐时点历史前向 PE**：当前 PE 是快照（见 §0.8 caveat），需要 point-in-time 历史前向 PE 才能干净进历史训练。

---

## 0.7 市场分离训练（2026-06-21 夜）

**训练宇宙改为三市场每日 Mid Vegas 科技池**（`load_tech_pools_data`）：US `us_tech_list`(461) / HK `hk_techman`(256) / CN `cn_hightech_list`(294) + 持仓，共 ~975 只（含缓存 US461/CN289/HK225）。比 watchlist(256) 大 ~4x，**各市场样本充足**（真顶 US 530 / CN 256 / HK 216）。开关：`--train-universe tech`（默认）/`--per-market-model`（默认开）。

**按市场分组训练**（`_research_bundle` per_market）：US/HK/CN **各训一套 LR+lgb，永不合并**（CN牛/HK熊股性相反）。市场专属权重自然涌现（M顶在 CN 自动趋零）。输出 scored 带 `market_model` 列、coef 带 `market_model`、score_perf 分市场。

**验证结论**：
- **任务1（holding 分市场 OOS）**：US 分市场胜（LR top8 35.7% vs 全局 26.2%）；HK/CN holding 测试集太小（14/3 顶）噪音大。
- **任务2（watchlist 作验证集，样本大）**：**CN 必须分离**——分市场 AUC 0.751 vs 全局 0.621，铁证（CN 股性独立，混进 US/HK 严重拖累）。US/HK 上分市场 vs 全局在噪音内（全局偶尔借 US 样本略优）。
- 绝对 AUC：watchlist-test US 仅 0.63-0.68（比 holding-test 0.77 低）→ 对更广人群泛化一般，holding OOS 数字偏乐观（持仓与训练相似）。

## 0.8 个股基本面特征：估值多乘数 + 增长（2026-06-23 扩充）

估值绝对值无意义，**只有「相对」才有意义**：相对市场（市场内分位）、相对增长（PEG）。
不同股性适用不同乘数——盈利成长股看 PE、重资产/代工（中芯/华虹）看 PB、SaaS 看 PS——
故同时提供三类乘数，由 SIC 子赛道 + 树自行选用。

**`valuation_context.py`（`VALUATION_FEATURES`，6 个）**：`valuation_{pe,pb,ps}` + 各自 `_pct_mkt`（**市场内分位**，三市场估值中枢不同、绝不跨市场比）。
- US PE(前向)/PS(前向)/PB — stockanalysis.com `/api/symbol/s/{t}/statistics`
- HK/CN PE(ttm)/PB — Futu OpenD（PS 暂无源）

**`growth_context.py`（`GROWTH_FEATURES`，4 个）**：`earnings_growth`、`revenue_growth`、`sector_earnings_growth_mean`（子赛道盈利增长均值＝赛道基本面热度，**`is_semiconductor` 静态标签的通用替代**）、`valuation_peg`（=前向PE/盈利增长，仅 growth>5；PE 相对增长才有意义）。增长为**因果**（取候选年之前已披露财年，见 `_causal_year`），US 优先；HK/CN 增长源待接（Futu 财报）。
数据缓存 `data/cache/fundamentals/{us_forward_pe,futu_pe,us_growth}.csv`，刷新 `src/stock_ana/research/top_reversal/refresh_valuation_pe.py`。

**效果（2026-06-23，US holding-OOS，213 因果特征干净 BASE）**：
| 特征集 | AUC | Δ |
|---|---|---|
| BASE（无基本面） | 0.696 | — |
| +PE（pe + pct） | 0.726 | +0.030 |
| +PE/PB/PS（全估值乘数） | 0.778 | +0.082 |
| +growth（eg/rg/sector/peg） | 0.763 | +0.067 |
| **+全基本面** | **0.843** | **+0.147** |

- **PB 是最强单项估值乘数（+0.092）**，PS +0.064——**直接实证「代工看 PB、SaaS 看 PS」**：华虹(01347) PE=517 失真，PB=5.54 才有意义。
- **`sector_earnings_growth_mean` 是最重要的基本面特征**（从全集移除 → 0.843 跌到 0.754），确认「赛道热度」就是 `is_semiconductor` 的通用动态替代。
- PE/PEG 作为组贡献，PEG 单独略负、但在全集中 +0.007；raw `revenue_growth` 边际可负（与 earnings_growth 共线，候删）。

**HK/CN 增长接入（2026-06-23）**：Futu `get_stock_filter` 最新年报 → `futu_growth.csv`（HK 258 / CN 293）。earnings←归母净利增长、revenue←营业总收入增长。华虹(01347) eg=−5.4%（盈利负增长）正解释其 PE=517 失真、PB=5.54 才有意义。

**CN 板块映射 + watchlist 验证（2026-06-23）**：`src/stock_ana/research/top_reversal/build_cn_industry_map.py` 用 Futu `get_plate_list(SH, INDUSTRY)`→`get_plate_stock` 建 `cn_industry_map.csv`（5607 只 / 131 行业，如 688981=半导体、600519=白酒Ⅱ），`_build_sector_map` 加 CN 分支 → CN 也有 `sector_earnings_growth_mean`（非空 2207/2565）。CN 在 holding 只有 4 个正例噪音大，改用 **watchlist 作验证集**（训练=科技池非 watchlist 2121 只/正175，验证=watchlist 444 只/正19）：

| 市场 | 验证方式 | BASE | +全基本面 | Δ |
|---|---|---|---|---|
| US | holding-OOS（16 正） | 0.696 | 0.843 | +0.147 |
| HK | holding-OOS（10 正） | 0.756 | 0.796 | +0.040 |
| **CN** | **watchlist 验证（19 正）** | 0.712 | **0.815** | **+0.103** |

CN：+估值(pe/pb) 0.764、+增长(含sector) 0.772、**仅 `sector_earnings_growth_mean` +0.048**。三市场基本面均验证有效，sector 热度都是强特征。早先 CN holding-OOS −0.068 系 4 正例噪音，已被推翻（印证 §0.7：CN 须用 watchlist 验证）。

> ⚠️ **重大 caveat（必读）**：估值乘数是**今天的快照**，用在历史候选上 = **未来信息泄漏**，且静态个股值会让树**记忆个股身份**。**所以估值侧 AUC 偏乐观，不是干净的实时信号；增长侧是因果的（年度序列、取候选年前），干净。** 估值乘数对**实时打分（今天的候选用今天的乘数）有效**。已 wire 进 pipeline（REALTIME 240 特征），解读历史回测时对估值乘数务必扣除这层乐观。

---

## 1. 三个蜡烛图的识别策略以及对应代码

当前顶部蜡烛图召回源有三类：

| 策略 | 代码 | 候选含义 | 当前定位 |
|---|---|---|---|
| 长上影顶部 | `src/stock_ana/strategies/impl/top_reversal.py` | 阶段高位出现长上影，次日走弱确认 | 低频、高信号质量的蜡烛图召回 |
| 高位跳空十字星 | `src/stock_ana/strategies/impl/evening_star_gap.py` | 大阳线后跳空放量十字星，随后转弱确认 | 旧有高位派发形态 |
| 跳空高开大阴线 | `src/stock_ana/strategies/impl/gap_fail_reversal.py` | 跳空高开后当天大阴线回补大部分缺口 | 低频、异常强情绪反转形态 |

三类策略统一在研究脚本中合并：

```python
# src/stock_ana/research/top_reversal/build_top_candidate_research.py
def _scan_pattern_candidates(df, args):
    shadow = scan_high_shadow(...)
    doji = scan_evening_star(...)
    gap_fail = scan_gap_fail(...)
    signals = _signal_rows(shadow, "shadow", df) \
        + _signal_rows(doji, "doji", df) \
        + _signal_rows(gap_fail, "gap_fail", df)
    return _cluster_signals(signals, df, merge_bars=args.merge_bars)
```

### 1.1 长上影顶部

代码入口：

- `detect_high_shadow_reversal(df)`
- `scan_history(df)`

文件：

- `src/stock_ana/strategies/impl/top_reversal.py`

核心条件：

| 条件 | 当前默认 |
|---|---:|
| 阶段新高窗口 | `20` 日 |
| 近几日出现新高 | `3` 日 |
| 高点距离近期峰值 | `3%` 以内 |
| 上影线 / 实体 | `>= 1.5` |
| 强上影线绝对幅度 | 上影线 / 收盘价 `>= 3%` |
| 放量配合上影线 | 上影线 / 收盘价 `>= 2%` 且量能 `>= 50日均量 * 1.5` |
| 上影线相对 ATR | `>= 0.5 ATR` |
| 默认历史扫描冷却 | `10` 个交易日 |

确认日条件为以下之一：

- 收盘低于信号日开盘，`engulf_open`。
- 收盘低于信号日实体中点，`below_midpoint`。
- 阴线且收盘低于信号日收盘，`bearish_close`。

输出特征会带入模型：

- `shadow_shadow_ratio`
- `shadow_shadow_pct`
- `shadow_shadow_atr`
- `shadow_prior_rise_pct`
- `shadow_vol_ratio`
- `shadow_confirm_body_ratio`
- `shadow_d2_break_d1_low`

### 1.2 高位跳空十字星

代码入口：

- `detect_evening_star_gap(df)`
- `scan_history(df)`

文件：

- `src/stock_ana/strategies/impl/evening_star_gap.py`

三根 K 线结构：

| K 线 | 要求 |
|---|---|
| Day-2 | 阳线，大阳线作为加分项 |
| Day-1 | 向上跳空，实体很小，放量，处于阶段新高区域 |
| Day-0 | 向下跳空、跌破 Day-2 中点，或阴线收低 |

核心参数：

| 条件 | 当前默认 |
|---|---:|
| 阶段新高窗口 | `20` 日 |
| 高点距离近期峰值 | `3%` 以内 |
| Day-1 跳空容差 | `1%` |
| 十字星实体 / 振幅 | `<= 0.30` |
| 严格十字星 | `<= 0.10` |
| Day-1 放量 | `>= 50日均量 * 1.5` |
| Day-1 巨量加分 | `>= 50日均量 * 2.0` |
| Day-0 放量加分 | `>= 50日均量 * 1.3` |

输出特征：

- `doji_d1_body_ratio`
- `doji_d1_vol_ratio`
- `doji_d0_vol_ratio`
- `doji_is_strict_doji`
- `doji_is_d2_big_bull`

### 1.3 跳空高开大阴线

代码入口：

- `detect_gap_fail_reversal(df)`
- `scan_history(df)`

文件：

- `src/stock_ana/strategies/impl/gap_fail_reversal.py`

策略含义：

高开代表强情绪或资金诱多，但当日直接收大阴线，并且几乎吃掉跳空高开的缺口。这类形态在 A 股更常见，美股也偶尔出现。

核心条件：

| 条件 | 当前默认 |
|---|---:|
| 高开幅度，相对前收 | `>= 5%` |
| 真跳空，相对前高 | `>= 0.2%` |
| 缺口回补比例 | `>= 80%` |
| 阴线实体幅度 | `>= 2%` |
| 实体 / 全日振幅 | `>= 0.55` |
| 实体 / 20日平均实体 | `>= 1.20` |
| 收盘在全日振幅位置 | `<= 40%` |
| 高点靠近 20 日高点 | `3%` 以内 |
| 前 60 日涨幅 | `>= 5%` |
| 默认冷却 | `5` 个交易日 |

输出特征：

- `gap_fail_gap_open_pct`
- `gap_fail_true_gap_pct`
- `gap_fail_gap_fill_ratio`
- `gap_fail_true_gap_fill_ratio`
- `gap_fail_effective_gap_fill_ratio`
- `gap_fail_open_to_close_drop_pct`
- `gap_fail_body_pct`
- `gap_fail_body_ratio`
- `gap_fail_body_vs_avg20`
- `gap_fail_close_position_pct`
- `gap_fail_vol_ratio`
- `gap_fail_close_below_prev_high`
- `gap_fail_close_below_prev_close`
- `gap_fail_close_below_prev_low`

---

## 2. SMC 的实现来源，以及对应代码

### 2.1 实现来源

SMC 底层来自 Python 包 `smartmoneyconcepts`，项目注释中记录的上游为：

- `https://github.com/joshyattridge/smart-money-concepts`

本项目没有直接无脑使用上游 `smc.ob()` 的实时结果，而是在策略层做了因果修正。

原因是：上游 `swing_highs_lows()` 使用前后各 `swing_length` 根 K 线确认 swing 点，如果直接拿来做实时 OB，会引入 look-ahead bias。

### 2.2 核心代码地图

| 文件 | 作用 |
|---|---|
| `src/stock_ana/strategies/impl/smc.py` | SMC 底层封装、因果 OB 检测、OB 质量打分 |
| `src/stock_ana/research/top_reversal/smc_context.py` | 顶部研究专用 SMC event 构建、raw/appear/early/confirmed 特征 |
| `src/stock_ana/research/top_reversal/candidate_sources.py` | SMC 召回源：`smc_appear`、`smc_early`，可选 `smc_raw`、`smc_confirmed` |
| `src/stock_ana/research/top_reversal/feature_registry.py` | SMC 特征注册 |
| `docs/smc_ob_scoring_design.md` | OB 检测和打分设计说明 |

### 2.3 因果 OB 检测

代码：

- `src/stock_ana/strategies/impl/smc.py::_ob_causal`

核心修正：

```python
visible_mask = all_swing_high_indices + swing_length <= i
visible_highs = all_swing_high_indices[visible_mask]
```

含义：

处理第 `i` 根 K 线时，只允许使用已经被确认的 swing 点。也就是说，只有 `k + swing_length <= i` 的 swing 点才是可见信息。

返回列与上游 OB 格式保持一致：

- `OB`: `1` 看涨，`-1` 看跌。
- `Top`
- `Bottom`
- `OBVolume`
- `MitigatedIndex`
- `Percentage`

### 2.4 SMC 信号层级

当前研究中把 SMC 拆成几个层级：

| 名称 | 含义 | 是否默认召回 |
|---|---|---|
| `smc_raw` | 潜在 OB zone 原始候选，信号出现 | 默认不加入主召回 |
| `smc_appear` | OB 原始 zone 出现后，3 天内价格离开区间 | 默认加入主召回 |
| `smc_early` | 3 天内出现内部结构转弱、FVG、流动性扫高、bull OB 失效等早期证据 | 默认加入主召回 |
| `smc_confirmed` | 最终结构确认后的 OB | 默认不加入主召回，确认偏晚 |

当前主流程默认加入：

- `smc_appear`
- `smc_early`

默认不加入：

- `smc_raw`
- `smc_confirmed`

### 2.5 SMC 特征

SMC 特征分组在 `feature_registry.py` 中登记。

严格实时可用特征：

| 组 | 示例 | 含义 |
|---|---|---|
| `SMC_LIVE_FEATURES` | `smc_live_bull_ob_count_60d`, `smc_live_bear_ob_count_20d` | 截止评分日已经确认的 OB 状态 |
| `SMC_RAW_FEATURES` | `smc_raw_bear_present_3d`, `smc_raw_bear_score_max_3d` | 顶部后 3 天内潜在 bearish OB 证据 |
| `SMC_EARLY_FEATURES` | `smc_early_score_3d`, `smc_early_micro_low_break_3d` | 顶部后 3 天内结构转弱 |

研究解释或延迟确认特征：

| 组 | 示例 | 注意 |
|---|---|---|
| `SMC_DELAYED_FEATURES` | `smc_d10_bear_ob_confirmed`, `smc_d10_bull_ob_mitigated_count` | 顶后 5-10 天信息，不能作为 2-3 天早期模型证据 |
| `SMC_DIAGNOSTIC_FEATURES` | `smc_diag_top_inside_bear_ob_confirmed_zone` | 解释性强，但含后验确认 |

主模型当前使用 `REALTIME_FEATURE_COLS`，默认只使用实时可见或评分日可见特征，不把全局 ZigZag 或后验 SMC diagnostic 当作实时特征。

---

## 3. 当前顶部识别的候选集合如何获得，以及如何根据已有信息评估候选集合的覆盖度，以及对于候选集合里面的候选进行自动标注

### 3.1 两类集合

当前系统明确区分两类集合：

| 集合 | 代码 | 含义 |
|---|---|---|
| 候选召回集合 | `merge_recall_candidates(...)` | 实际策略能在实时中召回出来的点 |
| 顶部全集 | `collect_zigzag_peak_candidates(...)` | 用全局 ZigZag 得到的所有宏观高点，用于评估召回天花板 |

顶部全集不是实时策略，它是研究标签集合。

### 3.2 候选召回集合

代码入口：

- `src/stock_ana/research/top_reversal/build_top_candidate_research.py::_build_symbol_research_rows`
- `src/stock_ana/research/top_reversal/candidate_sources.py`

默认候选来源：

```text
蜡烛图候选:
  shadow
  doji
  gap_fail

SMC 候选:
  smc_appear
  smc_early
```

当前不默认纳入主召回：

```text
smc_raw
smc_confirmed
double_top
```

候选合并：

- 通过 `merge_bars` 合并临近信号，默认 `3` bars。
- 合并后保留来源标记，例如：
  - `recalled_by_shadow`
  - `recalled_by_doji`
  - `recalled_by_gap_fail`
  - `recalled_by_smc_appear`
  - `recalled_by_smc_early`
  - `recall_source_count`

### 3.3 顶部全集

代码：

- `src/stock_ana/research/top_reversal/candidate_sources.py::collect_zigzag_peak_candidates`

逻辑：

- 对每只股票运行 `analyze_wave_structure(df)`。
- 从 `wave_result["all_pivots"]` 中取所有 ZigZag 高点。
- 用高点附近真实最高价修正 `top_pos`。
- 形成 `candidate_source = "zigzag_peak"` 的全集样本。

顶部全集用于回答：

> 在全局历史结构中，真正的顶部总共有多少？当前策略能召回多少？

### 3.4 覆盖度评估

代码：

- `src/stock_ana/research/top_reversal/coverage.py::strategy_coverage_report`
- `src/stock_ana/research/top_reversal/candidate_sources.py::attach_strategy_matches`
- `src/stock_ana/research/top_reversal/build_top_candidate_research.py::_score_performance_vs_universe`

覆盖度计算方式：

1. 先用全局 ZigZag 顶部全集构建 `universe`。
2. 对 universe 中 `label == true_top` 的点，检查其附近 `near_bars=merge_bars` 是否有实际召回候选。
3. 统计：
   - `covered_by_recall`
   - `covered_by_patterns`
   - `covered_by_smc_appear`
   - `covered_by_smc_early`
   - `missed_by_recall`
   - `recall_coverage_pct`

### 3.5 自动标注

代码：

- `src/stock_ana/research/top_reversal/build_top_candidate_research.py::_macro_zigzag_label`
- `src/stock_ana/research/top_reversal/build_top_candidate_research.py::_label_candidate`

标注分两层：

1. `macro_zigzag`：当前默认标签，使用全局 ZigZag 和大结构，允许用未来信息，只用于训练标签。
2. `legacy_forward`：旧的未来窗口涨跌规则，目前作为辅助字段保留，不作为默认标签。

当前默认标签：

| label | 含义 |
|---|---|
| `true_top` | 宏观结构顶部，或双顶后进入下跌结构 |
| `continuation` | 之后出现明显更高高点，属于上涨中继 |
| `ambiguous` | 大结构不足，不能判断是顶部还是中继 |
| `downtrend_continuation` | 宏观顶部已过，候选点是更低高点 |

关键规则顺序：

1. 如果候选点之后出现明显更高的实际高点，直接标注为 `continuation`。
2. 如果之后有更高 ZigZag 高点，标注为 `continuation`。
3. 如果候选点在宏观波段峰值附近，且没有后续明显更高高点，标注为 `true_top`。
4. 如果候选点和另一个 ZigZag 高点形成双顶，并且后续破颈线、反弹失败，标注为 `true_top`。
5. 如果结构不足，标注为 `ambiguous`。

这一步修复了此前大量把上涨中继误标为真顶的问题。当前输出中：

- `future_higher_actual_high` 在统一候选中有 `3376` 条，全部为 `continuation`。
- `future_higher_actual_high` 在顶部全集中有 `1226` 条，全部为 `continuation`。

### 3.6 全局标签和实时特征的边界

必须严格区分：

| 类型 | 是否允许未来信息 | 用途 |
|---|---|---|
| 全局 ZigZag 标签 | 允许 | 构建训练标签、研究全集 |
| `oracle_*` 字段 | 允许 | 诊断、复盘、解释 |
| 模型特征 | 不允许 | 实时评分 |
| causal ZigZag 特征 | 只用 `score_asof_pos` 之前数据 | 模型训练和实盘评分 |

代码中对应：

- `_oracle_zigzag_context(...)`：全局信息，仅诊断。
- `_causal_zigzag_context(...)`：截断到 `score_asof_pos` 后重新计算 wave structure，进入模型。

---

## 4. 如何构建模型进行学习

### 4.1 数据生成主流程

入口：

```bash
python3 src/stock_ana/research/top_reversal/build_top_candidate_research.py
```

主要流程：

```text
load_watchlist_data
  -> 对每只股票扫描蜡烛图候选
  -> 构建 SMC bundle
  -> 收集 smc_appear / smc_early 候选
  -> merge_recall_candidates 得到统一候选
  -> collect_zigzag_peak_candidates 得到顶部全集
  -> enrich:
       price context
       oracle zigzag context
       causal zigzag context
       label
  -> add_research_features:
       Mid Vegas
       指数轧空
       SMC OB
       candle interaction
  -> Mid Vegas 严格上涨趋势过滤
  -> fit_logistic
  -> 输出 CSV / Markdown 报告
```

### 4.2 特征体系

代码：

- `src/stock_ana/research/top_reversal/feature_registry.py`
- `src/stock_ana/research/top_reversal/feature_pipeline.py`

主要特征组：

| 特征组 | 代表字段 | 含义 |
|---|---|---|
| `candidate_recall` | `recalled_by_smc_early`, `score_lag_bars` | 哪个召回源发现了候选 |
| `candle_pattern` | `shadow_*`, `doji_*`, `gap_fail_*` | 蜡烛图形态强度 |
| `candle_interaction` | `smc_early_with_any_candle` | SMC 与蜡烛图是否共振 |
| `mid_vegas_trend` | `mid_vegas_passed` 等 | 是否处于严格上涨趋势 |
| `price_context` | `prior_ret_*`, `top_dist_ema144_pct` | 涨幅、均线乖离、价格上下文 |
| `technical_exhaustion` | `rsi14_bear_div_60d`, `high_volume_stall_score` | 顶背离、超买、缩量上涨、滞涨 |
| `zigzag_anchor` | `rise_from_anchor_low_pct` | 因果 ZigZag 锚点涨幅 |
| `index_squeeze` | `china_hk_index_squeeze_spike` | 中概/HK 指数轧空背景 |
| `wave_structure` | `major_wave_rise_pct` | 因果大波段结构 |
| `smc_causal` | `smc_raw_*`, `smc_early_*`, `smc_live_*` | 实时可见 SMC 结构 |

### 4.3 模型

代码：

- `src/stock_ana/research/top_reversal/modeling.py::fit_logistic`
- `src/stock_ana/research/top_reversal/build_top_candidate_research.py::_research_bundle`

当前模型是轻量 logistic regression：

- 只训练 `label in ["true_top", "continuation"]` 的样本。
- `ambiguous` 不参与训练，但会保留在输出中。
- 特征缺失用中位数填充。
- 特征做标准化。
- L2 正则。
- 输出：
  - `top_prob`
  - logistic 系数表
  - score band 表现

当前有两套任务：

| 任务 | 数据 | 输出 |
|---|---|---|
| 主下跌顶模型 | `watchlist_unified_recall_candidates_labeled.csv` | `top_candidate_logistic_*` |
| 真正逃顶模型 | `watchlist_escape_top_candidates_labeled.csv` | `escape_top_candidate_logistic_*` |

### 4.4 真正逃顶模型

真正逃顶模型不是所有下跌顶的模型，而是只在高位大涨候选池里训练。

候选池默认条件：

```text
max(anchor rise, major wave rise) >= 80%
and top is within 8% of 252-day high
and (top distance to EMA144 >= 25% or top distance to EMA200 >= 30%)
```

训练标签映射：

| `escape_top_label` | 模型 label |
|---|---|
| `escape_top` | `true_top` |
| `high_continuation` | `continuation` |
| 其他 | `ambiguous` |

### 4.5 SMC 分层模型对比

代码：

- `src/stock_ana/research/top_reversal/build_top_candidate_research.py::_smc_model_comparison`

对比模型：

- `base_no_smc`
- `base_plus_smc_live`
- `base_plus_smc_raw`
- `base_plus_smc_early`
- `base_plus_smc_raw_early`
- `base_plus_smc_causal`
- `base_plus_smc_live_delayed`
- `base_plus_smc_all`

注意：

- `base_plus_smc_causal` 才是严格实时候选。
- `base_plus_smc_live_delayed` 和 `base_plus_smc_all` 有解释价值，但包含顶后 5-10 天或诊断信息，不应直接作为 2-3 天早期判断效果。

---

## 5. 当前结果如何

当前结果来自最近一次运行：

```bash
python3 src/stock_ana/research/top_reversal/build_top_candidate_research.py
```

### 5.1 当前集合规模

Mid Vegas 严格上涨趋势过滤后：

| 集合 | 数量 |
|---|---:|
| 蜡烛图形态候选 | `641` |
| SMC appear 候选 | `3182` |
| SMC early 候选 | `3062` |
| 统一候选 | `3708` |
| 顶部全集 | `1496` |
| 失败股票 | `0` |

### 5.2 统一候选标签分布

文件：

- `data/output/top_candidate_research/watchlist_unified_recall_candidates_labeled.csv`

| label | n | pct |
|---|---:|---:|
| `continuation` | `3376` | `91.0%` |
| `true_top` | `264` | `7.1%` |
| `ambiguous` | `65` | `1.8%` |
| `downtrend_continuation` | `3` | `0.1%` |

### 5.3 顶部全集标签分布

文件：

- `data/output/top_candidate_research/watchlist_universe_candidates_labeled.csv`

| label | n | pct |
|---|---:|---:|
| `continuation` | `1226` | `82.0%` |
| `true_top` | `270` | `18.0%` |

### 5.4 召回覆盖率

文件：

- `data/output/top_candidate_research/recall_coverage_by_true_top.csv`

全部市场：

| 指标 | 数值 |
|---|---:|
| 顶部全集样本 | `1496` |
| 全集真顶 | `270` |
| 当前召回覆盖真顶 | `199` |
| 漏召回真顶 | `71` |
| 召回覆盖率 | `73.7%` |
| 蜡烛图覆盖率 | `11.9%` |
| SMC appear 覆盖率 | `69.3%` |
| SMC early 覆盖率 | `70.4%` |

分市场：

| 市场 | 真顶 | 当前召回覆盖率 | SMC appear | SMC early | 蜡烛图 |
|---|---:|---:|---:|---:|---:|
| CN | `36` | `86.1%` | `83.3%` | `80.6%` | `22.2%` |
| HK | `43` | `69.8%` | `62.8%` | `65.1%` | `11.6%` |
| US | `191` | `72.3%` | `68.1%` | `69.6%` | `9.9%` |

结论：

- 当前召回主力是 SMC appear / early。
- 三类蜡烛图单独召回很少，但保留为高置信、结构共振特征。
- 当前真正值得继续扩展的是结构召回源，例如双顶、前高阻力、头肩顶，而不是继续无约束增加普通蜡烛图形态。

### 5.5 主模型表现

文件：

- `data/output/top_candidate_research/top_candidate_score_performance.csv`
- `data/output/top_candidate_research/top_candidate_universe_recall_score_performance.csv`

模型内部分层：

| 分数段 | 信号数 | true_top | precision |
|---|---:|---:|---:|
| top 9% | `371` | `107` | `28.8%` |
| top 19% | `744` | `155` | `20.8%` |
| top 30% | `1115` | `187` | `16.8%` |
| top 40% | `1486` | `210` | `14.1%` |
| top 50% | `1854` | `227` | `12.2%` |

以“当前召回能触达的全集真顶”为分母：

| 分数段 | 信号数 | 命中全集真顶 | precision | recall |
|---|---:|---:|---:|---:|
| top 9% | `371` | `98` | `26.4%` | `49.2%` |
| top 30% | `1115` | `153` | `13.7%` | `76.9%` |
| top 50% | `1854` | `179` | `9.7%` | `89.9%` |

### 5.6 真正逃顶模型表现

文件：

- `data/output/top_candidate_research/watchlist_escape_top_candidates_labeled.csv`
- `data/output/top_candidate_research/escape_top_candidate_score_performance.csv`
- `data/output/top_candidate_research/escape_top_universe_recall_score_performance.csv`

逃顶候选标签：

| label | n |
|---|---:|
| `continuation` | `1944` |
| `true_top` | `158` |
| `ambiguous` | `51` |

模型内部分层：

| 分数段 | 信号数 | true_top | precision |
|---|---:|---:|---:|
| top 9% | `216` | `73` | `33.8%` |
| top 19% | `431` | `103` | `23.9%` |
| top 30% | `646` | `122` | `18.9%` |
| top 50% | `1077` | `140` | `13.0%` |

以“当前召回能触达的逃顶真顶”为分母：

| 分数段 | 信号数 | 命中全集逃顶 | precision | recall |
|---|---:|---:|---:|---:|
| top 9% | `216` | `69` | `31.9%` | `53.9%` |
| top 30% | `646` | `105` | `16.3%` | `82.0%` |
| top 50% | `1077` | `115` | `10.7%` | `89.8%` |

### 5.7 当前最大改进点

当前最重要的改进不是模型精度小幅变化，而是标签边界修正：

- 只要后续出现明显更高高点，优先标注为 `continuation`。
- 统一候选中 `future_higher_actual_high = 3376` 条，全部是 `continuation`，没有再进入 `true_top`。
- 顶部全集中 `future_higher_actual_high = 1226` 条，全部是 `continuation`。
- 这修复了之前大量把上涨中继误判为真顶的问题。

---

## 6. 双顶识别的专门说明

### 6.1 当前定位

双顶当前有两个用途：

1. 训练标签修正：在全局宏观结构中，把典型双顶后进入下跌结构的点从 `ambiguous` 升级为 `true_top`。
2. 通用 detector：已经实现独立函数，未来可以作为召回源或模型特征，但当前主召回默认还没有启用。

### 6.2 代码

文件：

- `src/stock_ana/research/top_reversal/double_top.py`

核心函数：

| 函数 | 作用 |
|---|---|
| `evaluate_double_top_pair(...)` | 判断两个高点是否构成已确认双顶 |
| `best_double_top_for_candidate(...)` | 对单个候选点寻找最佳双顶配对 |
| `find_double_top_patterns(...)` | 扫描全局 ZigZag 高点对，返回已确认双顶 |

候选源预留：

- `src/stock_ana/research/top_reversal/candidate_sources.py::collect_double_top_candidates`

标签接入：

- `src/stock_ana/research/top_reversal/build_top_candidate_research.py::_macro_zigzag_label`

### 6.3 判断规则

默认参数：

| 参数 | 默认 |
|---|---:|
| 两头高度容差 | `2.5%` |
| 两头最小间隔 | `5` bars |
| 两头最大间隔 | `80` bars |
| 跌破颈线幅度 | `5%` |
| 反弹高点允许高出颈线 | `2%` |

确认流程：

```text
候选点与另一个 ZigZag 高点
  -> 两头高度接近
  -> 两头间隔合理
  -> 两头之间存在颈线低点
  -> 第二头后出现 ZigZag 低点，且跌破颈线至少 5%
  -> 后续出现反弹高点，但不能有效收复颈线
  -> double_top_confirmed = 1
```

候选点可以是第一头，也可以是第二头。

### 6.4 为什么需要“破颈线 + 反弹失败”

只看两个高点接近，会把很多正常上涨中的整理误判为顶部。

当前规则要求后续大结构已经进入下跌：

- 单纯形成两个类似高度高点，不够。
- 单纯跌破颈线，也不够。
- 跌破颈线后反弹失败，才说明结构开始转弱。

因此右侧结构不足的样本仍保留为 `ambiguous`。

### 6.5 与上涨中继的关系

双顶逻辑不会覆盖“未来明显更高高点”的规则。

在 `_macro_zigzag_label` 中，优先级是：

1. 如果后面出现明显更高实际高点，直接 `continuation`。
2. 如果后面出现更高 ZigZag 高点，直接 `continuation`。
3. 之后才检查双顶确认。

这保证双顶不会重新引入“把上涨中继标成真顶”的老问题。

### 6.6 当前双顶对标签的影响

当前输出：

| 集合 | `macro_double_top_downtrend_confirmed` 数量 |
|---|---:|
| 统一候选 | `72` |
| 顶部全集 | `43` |

典型效果：

- `JD 2025-03-06`：双顶后破颈线并反弹失败，升级为 `true_top`。
- `DBX 2025-01-31`：双顶后破颈线并反弹失败，升级为 `true_top`。
- `ET 2026-05-05`：没有有效跌破颈线，保留 `ambiguous`。
- `ENTG 2026-05-06`：有破位但缺少反弹失败确认，保留 `ambiguous`。

### 6.7 后续使用建议

下一步可以做两件事：

1. 把 `collect_double_top_candidates` 作为独立召回源做 A/B，观察能否提升召回。
2. 先不直接加入主模型，先批量出图核验，尤其看：
   - 第一头信号是否过早。
   - 第二头确认是否过晚。
   - 破颈线后反弹失败是否比 SMC early 更晚。
   - 在港股和 A 股中是否更有效。

如果图形质量稳定，再把双顶作为：

- 召回源：`recalled_by_double_top`
- 特征：`double_top_recall_confirm_lag_min`、`double_top_recall_neckline_break_pct`、`double_top_recall_failed_rebound_vs_neckline_pct`

---

## 关键输出文件

| 文件 | 内容 |
|---|---|
| `data/output/top_candidate_research/watchlist_unified_recall_candidates_labeled.csv` | 当前主候选训练集 |
| `data/output/top_candidate_research/watchlist_universe_candidates_labeled.csv` | ZigZag 顶部全集 |
| `data/output/top_candidate_research/recall_coverage_by_true_top.csv` | 当前召回对全集真顶覆盖率 |
| `data/output/top_candidate_research/top_candidate_logistic_scored.csv` | 主模型打分 |
| `data/output/top_candidate_research/top_candidate_logistic_coefficients.csv` | 主模型系数 |
| `data/output/top_candidate_research/watchlist_escape_top_candidates_labeled.csv` | 真正逃顶候选训练集 |
| `data/output/top_candidate_research/escape_top_candidate_logistic_scored.csv` | 真正逃顶模型打分 |
| `data/output/top_candidate_research/escape_top_recall_coverage_by_true_top.csv` | 逃顶任务召回覆盖率 |
