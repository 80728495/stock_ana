# 顶部识别当前系统说明

> 截至 2026-06-27。本文档只描述当前仍在主流程中使用的顶部识别体系：样本入口、候选召回、自动标注、特征、模型、评估与输出。

## 1. 当前唯一任务

当前顶部识别只有一套主任务：

```text
召回候选
  -> Mid Vegas 严格上涨趋势过滤
  -> 统一样本集
  -> 自动标注 true_top / continuation / ambiguous / unconfirmed
  -> 因果特征构建
  -> 分市场模型打分
  -> 覆盖率、精度、图表复盘
```

`Mid Vegas` 是样本入口过滤。候选点不满足严格上涨趋势时，不进入后续训练与评估。通过过滤后的点统一进入同一套标签、特征和模型流程。

## 2. 代码入口

| 模块 | 作用 |
|---|---|
| `src/stock_ana/research/top_reversal/build_top_candidate_research.py` | 主流程：扫描候选、打标签、构建特征、训练、输出 |
| `src/stock_ana/research/top_reversal/candidate_sources.py` | 候选召回源与覆盖标记 |
| `src/stock_ana/research/top_reversal/feature_pipeline.py` | 串联所有特征构建器 |
| `src/stock_ana/research/top_reversal/feature_registry.py` | 模型特征注册表 |
| `src/stock_ana/research/top_reversal/modeling.py` | Logistic 与 LightGBM 训练、打分、报表 |
| `src/stock_ana/research/top_reversal/coverage.py` | 候选召回对顶部全集的覆盖率 |
| `src/stock_ana/research/top_reversal/double_top.py` | 双顶结构检测，用于标签修正和未来召回实验 |
| `src/stock_ana/research/top_reversal/gen_holding_top_pdf.py` | 持仓顶部信号图表输出 |

主命令：

```bash
python3 src/stock_ana/research/top_reversal/build_top_candidate_research.py
```

常用阶段化运行：

```bash
python3 src/stock_ana/research/top_reversal/build_top_candidate_research.py --stage sample
python3 src/stock_ana/research/top_reversal/build_top_candidate_research.py --stage model
```

只跑单市场：

```bash
python3 src/stock_ana/research/top_reversal/build_top_candidate_research.py --markets US
```

## 3. 数据与样本

默认训练宇宙：

- `--train-universe tech`
- 三市场科技池：`us_tech_list`、`hk_techman`、`cn_hightech_list`
- 自动合并 holding
- 默认 `--per-market-model`，US / HK / CN 分市场独立训练

可选旧研究口径：

- `--train-universe watchlist`

阶段化缓存目录：

- `data/output/top_candidate_research/_samples/`

## 4. 候选召回

### 4.1 主召回源

主召回集合由两类来源合并：

| 来源 | 代码 | 当前定位 |
|---|---|---|
| `shadow` | `src/stock_ana/strategies/impl/top_reversal.py` | 长上影顶部候选 |
| `doji` | `src/stock_ana/strategies/impl/evening_star_gap.py` | 高位跳空十字星候选 |
| `smc_top_confirmed` | `candidate_sources.py::collect_smc_top_confirmed_candidates` | 当前主力 SMC 顶部候选 |

`smc_top_confirmed` 是统一 SMC 顶部源。它对近高位 bearish OB 锚点取三种机制里最早成立的确认：

| 机制 | 含义 |
|---|---|
| `smc_supply_held` | 高分供给锚点出现后，高点没有被收复，且收盘跌破锚点低位 |
| `smc_early` | OB raw 证据叠加 FVG、sweep、CHoCH/BOS、微观破位等早期结构证据 |
| `smc_confirmed` | bearish OB 最终结构确认，通常更晚但更可靠 |

统一候选由以下逻辑合并：

```python
unified_candidates = merge_recall_candidates(
    [*pattern_candidates, *smc_raw_candidates, *smc_top_confirmed_candidates],
    df,
    merge_bars=args.merge_bars,
)
```

默认 `smc_raw_candidates` 为空，只有显式传入 `--include-smc-raw-recall` 才会进入统一召回。

### 4.2 诊断召回源

以下数据集会输出用于诊断、覆盖率对照和特征研究，但不作为默认主召回源：

| 数据集 | 输出文件 |
|---|---|
| SMC raw | `watchlist_smc_raw_recall_candidates_labeled.csv` |
| SMC appear | `watchlist_smc_appear_recall_candidates_labeled.csv` |
| SMC early | `watchlist_smc_early_recall_candidates_labeled.csv` |
| SMC confirmed | `watchlist_smc_confirmed_recall_candidates_labeled.csv` |
| 蜡烛图候选 | `watchlist_pattern_candidates_labeled.csv` |

`gap_fail_reversal.py` 当前不进入主召回。它保留为低频异常反转形态和持仓提醒/特征研究来源。

## 5. 顶部全集

顶部全集来自全局 ZigZag 高点：

```python
universe_candidates = collect_zigzag_peak_candidates(df, wave_result)
```

用途：

- 作为“全量可解释顶部样本”。
- 评估当前召回源能覆盖多少真实顶部。
- 统计漏召回真顶。

注意边界：

- 顶部全集和训练标签允许使用全局 ZigZag，这是后验样本构建。
- 模型特征不能使用全局 ZigZag 未来信息。
- 进入模型的 ZigZag / wave 特征都由 `_causal_zigzag_context` 按 `score_asof_pos` 截断历史数据重新计算。

## 6. 自动标注

当前标签来源：

- 全局 ZigZag / 大波浪结构。
- 未来是否出现显著更高高点。
- 候选是否处在宏观波段高点附近。
- 双顶结构是否成立。
- 近端样本是否已经出现 SMC 摆动级 CHoCH 向下确认。

标签含义：

| 标签 | 含义 | 是否参与训练 |
|---|---|---|
| `true_top` | 宏观结构上成立的顶部，或确认后的双顶顶部 | 是 |
| `continuation` | 后续出现显著更高高点，属于上涨中继 | 是 |
| `ambiguous` | 结构证据不足 | 否 |
| `unconfirmed` | 近端趋势尚未走完，未出现摆动级结构反转确认 | 否 |
| `downtrend_continuation` | 大顶后下跌途中的 lower high | 否 |

近端处理：

- 距离数据末尾小于 `--label-recent-smc-window` 的候选，即使全局结构看似顶部，也要求顶后出现摆动级 CHoCH 向下。
- 没有结构反转确认的近端候选降为 `unconfirmed`。

## 7. 特征体系

所有模型特征由 `feature_registry.py::REALTIME_FEATURE_GROUPS` 管理。主模型使用：

```python
PRIMARY_MODEL_FEATURE_COLS = list(REALTIME_FEATURE_COLS)
```

当前特征组：

| 特征组 | 含义 |
|---|---|
| `candidate_recall` | 候选由哪些召回源发现、确认延迟、SMC/双顶召回质量 |
| `candle_pattern` | 长上影、跳空十字星、gap-fail 原始形态强度 |
| `candle_interaction` | SMC 与蜡烛图是否共振 |
| `mid_vegas_trend` | Mid Vegas 趋势结构和短历史豁免后的趋势状态 |
| `price_context` | 前涨幅、均线乖离、ATR、量能、确认日位置 |
| `technical_exhaustion` | RSI/MACD 顶背离、严重超买、缩量上涨、高位放量滞涨 |
| `zigzag_anchor` | 因果 ZigZag 锚点、前低、M 型中间低点处理 |
| `index_squeeze` | 中概/HK 指数轧空背景 |
| `wave_structure` | 因果大波段结构和高点簇 |
| `smc_causal` | SMC live/raw/early 中当前可见的结构特征 |
| `prior_high_structure` | 前高、严格双顶、阻力位相关结构 |
| `macro_micro` | 板块 regime、横截面强弱、抛物线过热 |
| `valuation` | 市场内估值分位；历史训练解释时需注意估值快照泄漏 |
| `growth` | 因果增长、赛道盈利增长热度、PEG |

因果边界：

- `oracle_*` 字段只用于标签解释和诊断，不进主模型。
- `SMC_DELAYED_FEATURES` 与 `SMC_DIAGNOSTIC_FEATURES` 只用于 SMC 分层对照，不进主模型。
- 历史估值乘数是当前快照，用于历史回测会偏乐观；实时打分可使用当日快照。

## 8. 模型训练

模型代码：

- `modeling.py::fit_logistic`
- `modeling.py::fit_lightgbm`

训练规则：

- 只训练 `true_top` 和 `continuation`。
- `ambiguous`、`unconfirmed`、`downtrend_continuation` 保留在输出，但不参与二分类训练。
- 默认按市场分开训练 US / HK / CN。
- Logistic 保留可解释性。
- LightGBM 用强正则，主要处理非线性和特征交互。

输出：

- `top_prob`
- Logistic 系数表
- LightGBM gain 表
- 分数段 precision
- 以当前召回覆盖到的全集真顶为分母的 recall

## 9. 当前结果快照

当前输出目录：

- `data/output/top_candidate_research/`

最近主输出时间：

- `2026-06-26 21:11`

样本规模：

| 集合 | 数量 |
|---|---:|
| 统一候选 | `12194` |
| 顶部全集 | `5361` |
| 蜡烛图候选 | `2453` |
| SMC early 诊断候选 | `9920` |
| SMC appear 诊断候选 | `10176` |

统一候选标签：

| label | n |
|---|---:|
| `continuation` | `10995` |
| `true_top` | `852` |
| `ambiguous` | `184` |
| `unconfirmed` | `150` |
| `downtrend_continuation` | `13` |

顶部全集标签：

| label | n |
|---|---:|
| `continuation` | `4368` |
| `true_top` | `820` |
| `unconfirmed` | `173` |

顶部全集真顶覆盖率：

| 市场 | 全集真顶 | 召回覆盖 | 覆盖率 | 蜡烛图覆盖 | SMC supply-held 覆盖 |
|---|---:|---:|---:|---:|---:|
| US | `374` | `282` | `75.4%` | `9.4%` | `66.6%` |
| HK | `215` | `155` | `72.1%` | `19.1%` | `63.3%` |
| CN | `231` | `163` | `70.6%` | `24.2%` | `59.7%` |
| 合计 | `820` | `600` | `73.2%` | `16.1%` | `63.8%` |

三市场样本外 AUC：

| 市场 | LR AUC | LightGBM AUC | test true_top |
|---|---:|---:|---:|
| US | `0.622` | `0.731` | `144` |
| HK | `0.809` | `0.771` | `19` |
| CN | `0.690` | `0.670` | `22` |

## 10. 关键输出文件

| 文件 | 内容 |
|---|---|
| `watchlist_unified_recall_candidates_labeled.csv` | 当前主候选训练集 |
| `watchlist_universe_candidates_labeled.csv` | ZigZag 顶部全集 |
| `watchlist_pattern_candidates_labeled.csv` | 蜡烛图候选诊断集 |
| `watchlist_smc_raw_recall_candidates_labeled.csv` | SMC raw 诊断集 |
| `watchlist_smc_appear_recall_candidates_labeled.csv` | SMC appear 诊断集 |
| `watchlist_smc_early_recall_candidates_labeled.csv` | SMC early 诊断集 |
| `watchlist_smc_confirmed_recall_candidates_labeled.csv` | SMC confirmed 诊断集 |
| `recall_coverage_by_true_top.csv` | 当前召回对顶部全集真顶的覆盖率 |
| `universe_true_tops_missed_by_recall.csv` | 当前召回漏掉的真顶 |
| `top_candidate_logistic_scored.csv` | Logistic 主模型打分 |
| `top_candidate_lgbm_scored.csv` | LightGBM 主模型打分 |
| `top_candidate_logistic_coefficients.csv` | Logistic 权重 |
| `top_candidate_lgbm_coefficients.csv` | LightGBM 特征重要性 |
| `top_candidate_score_performance.csv` | Logistic 分数段表现 |
| `top_candidate_lgbm_score_performance.csv` | LightGBM 分数段表现 |
| `top_candidate_universe_recall_score_performance.csv` | 主候选打分相对召回可触达真顶的 recall |

## 11. 双顶说明

双顶当前不是默认主召回源。它有两个用途：

1. 标签修正：当候选点与另一个高点构成确认双顶，并且后续跌破颈线、反弹不能有效收复颈线时，可把结构不清的点标为 `true_top`。
2. 未来实验：`collect_double_top_candidates` 可作为独立召回源接入，但默认未启用。

双顶检测边界：

- 使用全局 ZigZag 时，只用于后验标签和研究。
- 如果未来作为实时召回源，必须用候选日以前的数据截断后重新计算。

主要参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--label-double-top-tolerance-pct` | `2.5` | 两个头部价格容差 |
| `--label-double-top-min-separation-bars` | `5` | 两头最小间隔 |
| `--label-double-top-max-separation-bars` | `80` | 两头最大间隔 |
| `--label-double-top-neckline-break-pct` | `5.0` | 跌破颈线最小幅度 |
| `--label-double-top-failed-rebound-neckline-pct` | `2.0` | 反弹允许高出颈线的幅度 |
