# 每日「持仓见顶」流程与新增功能说明

> 用顶部识别模型，每日检查 holding 列表是否出现见顶形态，并把见顶股票的 K 线图 +
> 见顶核心要素打印到一个 PDF。本文档面向部署（含 Windows）。

---

## 1. 流程总览（两层）

| 频率 | 步骤 | 命令 | 耗时 |
|---|---|---|---|
| 每日盘后 | ① 刷新行情数据（你的既有更新流程） | — | — |
| 每日盘后 | ② 同步持仓/自选 | `python sync_holding.py` | 秒级 |
| 周期(可每日夜里) | ③ 刷新估值 PE | `python src/stock_ana/research/top_reversal/refresh_valuation_pe.py` | US ~8min / Futu ~2min |
| 周期(可每日夜里) | ④ 重训模型 + 全量打分 | `python src/stock_ana/research/top_reversal/build_top_candidate_research.py` | ~1h(975只) |
| 每日早上 | ⑤ 生成「持仓见顶」PDF | `python src/stock_ana/research/top_reversal/gen_holding_top_pdf.py --days 5` | 秒级 |

**建议**：③④放夜里跑（build 含候选生成+特征+按市场训练，~1h），⑤早上看 PDF。
build 已把 holdings 折入训练宇宙并打分，⑤只做筛选+出图，秒级。

PDF 输出：`data/output/top_candidate_research/holding_signal_eval/holding_top_<日期>.pdf`，
每页 = 一只见顶持仓的 K 线（标信号）+ 该信号高分的**核心要素**（lgb SHAP 正贡献 Top7）。

---

## 2. 顶部研究入口（src/stock_ana/research/top_reversal/）

| 入口 | 作用 |
|---|---|
| `src/stock_ana/research/top_reversal/build_top_candidate_research.py` | 顶部识别主流程。`--train-universe tech`(默认，三市场科技池+持仓)、`--per-market-model`(默认，US/HK/CN 各训一套 LR+lgb，永不合并)。输出 labeled / per-market scored / coef。 |
| `src/stock_ana/research/top_reversal/refresh_valuation_pe.py` | 刷新估值 PE：US 前向(stockanalysis.com)、HK/CN trailing(Futu OpenD)。→ `data/cache/fundamentals/{us_forward_pe,futu_pe}.csv` |
| `src/stock_ana/research/top_reversal/gen_holding_top_pdf.py` | 把持仓近 N 日、模型判见顶的信号出成 PDF（图 + 核心要素）。`--days/--band` |

## 3. 新增模型/特征模块（src/stock_ana/research/top_reversal/）

| 模块 | 内容 |
|---|---|
| `candidate_sources.py` | 召回源：`collect_smc_supply_held_candidates`、`collect_smc_top_confirmed_candidates`(supply_held/early/confirmed 取最早确认) 等 |
| `prior_high_context.py` | 前高结构特征：收盘是否站上前 ZigZag 高、严格 M 双顶 |
| `macro_micro_context.py` | 宏观(板块 SIC 二级分类/过热度/截面分位) + 微观(抛物线加速/垂直度/延展) |
| `valuation_context.py` | 估值 PE 特征（市场内分位归一）。见 caveat（§5） |
| `modeling.py` | `fit_logistic` + `fit_lightgbm`(并行)；`_build_scored_frame` 共享 |
| `vegas_context.py` | Mid Vegas 趋势特征 + **短历史豁免**（总长<200根用 mid EMA 成熟即可） |
| `feature_registry.py` / `feature_pipeline.py` | 特征注册与组装(REALTIME 233 特征) |

## 4. 模型设计要点

- **市场分离**：CN牛/HK熊股性相反，CN/HK/US **永不合并**，各训各打分（scored 带 `market_model`）。数据证实 CN 必须分离（分市场 AUC 0.751 vs 全局 0.621）。
- **双分类器**：LR(可解释基线) + lightgbm(更优 AUC、能学 regime×结构交互)并行，每次 build 同时输出。
- **召回**：蜡烛(shadow/doji) + `smc_top_confirmed`(supply_held/early/confirmed 最早确认)。gap_fail/appear 已移出主召回。

## 5. 估值 PE 的重要 caveat（务必知晓）

当前 PE 是**当前快照**，用在历史候选上有 **look-ahead**（且静态值会让树记忆个股身份）。
所以历史回测里 PE 的增益偏乐观；**对实时打分（今天的候选用今天的 PE）有效**。
真正干净需要**逐时点历史前向 PE**（待数据源）。详见 `top_reversal_current_system.md` §0.8。

## 6. Windows 部署注意

- 依赖：`pip install lightgbm mplfinance pillow pandas numpy futu-api`；lightgbm 在 Win 通常自带 OpenMP，无需额外装 libomp（macOS 才需要 `brew install libomp`）。
- Futu OpenD 需在本机运行（127.0.0.1:11111），用于 ③ 的 HK/CN PE 与 `sync_holding.py`。
- 中文出图字体：`utils/plot_renderers.py` 已按 OS 选字体（Win 用 Microsoft YaHei/SimHei）。
- 路径全部基于项目根，跨平台安全。
- `data/cache`、`data/output` 已被 `.gitignore` 忽略（不入库，在目标机本地重建）。
