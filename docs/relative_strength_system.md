# 多市场相对强度（RS）系统说明

> 当前实现版本：2026-07-12  
> 核心代码：`src/stock_ana/data/relative_strength_store.py`、`src/stock_ana/data/benchmark_store.py`

## 1. 目标与范围

RS 系统用于衡量股票相对于其适配市场或行业指数的强弱，而不是只比较股票自身涨跌幅。系统覆盖 US、CN、HK 三个市场，主要产出：

1. 每只股票逐日的相对收益、RS 曲线和 RS 动量；
2. 每个交易日市场内部的 63 日 RS 百分位；
3. 股票当前采用的 benchmark、回归 beta 和 R²；
4. 可直接用于扫描、研究和模型训练的历史 parquet 与每日 CSV。

当前股票池是“今天配置的股票池”的历史回放：US 使用科技股列表，CN 使用高科技列表与 watchlist 并集，HK 使用港股 universe。它不是历史时点成分股快照，因此用于训练时需要注意幸存者偏差。

## 2. 行情与 Benchmark 数据

### 2.1 股票行情来源

| 市场 | 股票池 | 日线来源 | 本地目录 | 常规历史范围 |
|---|---|---|---|---|
| US | `us_tech_list.md` | Futu OpenD | `data/cache/us/` | 取决于上市日期及已有缓存 |
| CN | `cn_hightech_list.md` 与 CN watchlist 并集 | Futu OpenD | `data/cache/cn/` | 主要从 2020 年开始 |
| HK | `hk_universe_list.md` | Futu OpenD，失败时保留兼容回落 | `data/cache/hk/` | 近 3 年，约从 2023 年开始 |

读入 RS 前会进行以下清洗：

- 日期归一化为日频 `DatetimeIndex`；
- 日期重复时保留最后一条；
- 收盘价转为数值；
- 删除空值、零和负收盘价。

### 2.2 Benchmark 注册表

所有 benchmark 均通过 OpenD 获取，使用带市场前缀的完整代码，并独立存入 `data/cache/benchmarks/`，避免指数代码与股票代码冲突。

| ID | Futu 代码 | 名称 | 用途 |
|---|---|---|---|
| `US_QQQ` | `US.QQQ` | NASDAQ-100 ETF | US 科技股 |
| `CN_CHINEXT` | `SZ.399006` | 创业板指 | 创业板及一般 CN 科技候选 |
| `CN_STAR_COMPOSITE` | `SH.000680` | 科创综指 | 科创板、半导体等候选 |
| `CN_STAR50` | `SH.000688` | 科创50 | 科创板大盘科技候选 |
| `HK_HSI` | `HK.800000` | 恒生指数 | 一般港股 |
| `HK_HSTECH` | `HK.800700` | 恒生科技指数 | 港股互联网、软件、游戏等 |

截至 2026-07-12 的最近一次运行，六个 benchmark 均更新到 **2026-07-10**，历史从 2020 年开始；恒生科技指数自身从 2020-07-27 开始。

## 3. 股票与 Benchmark 映射

映射分为两层：先验候选集合，以及按历史相关性进行的因果动态选择。

### 3.1 US

US 科技池固定使用 `US_QQQ`。当前没有在 SPY、SOXX、行业 ETF 之间动态选择。

### 3.2 CN

- `300/301` 开头：先验为创业板指；候选为创业板指、科创综指、科创50。
- `688/689` 开头：先验为科创综指；候选为科创综指、科创50、创业板指。
- 其他 CN 科技股：先验为创业板指，使用同一组三个候选。

CN 当前主要按上市板块建立先验，尚未接入完整行业分类。

### 3.3 HK

HK 先读取 `data/hk_industry_map.csv`：

- 互联网、应用软件、支付、游戏、数码服务等：先验恒生科技，候选恒生科技/恒指；
- 半导体、半导体设备、电子零件、消费电子、储能及部分制造业：先验科创综指，同时允许科创50、创业板、恒生科技和恒指参与选择；
- 其他或行业缺失：先验恒指，候选恒指/恒生科技。

这使中芯国际、华虹半导体等股票可以使用内地科技指数作为 beta，而不是被强制映射到恒指。

### 3.4 因果动态选择

系统每月第一个股票交易日重新评估 benchmark：

1. 只使用该日之前最多 120 个股票交易日；
2. 股票和各候选 benchmark 使用对数日收益；
3. 至少需要 40 对有效收益；
4. 对每个候选计算回归 beta 与相关系数平方 R²；
5. 候选 R² 至少达到 0.10；
6. 切换到非先验 benchmark 时，R² 通常需要比先验高至少 0.05；
7. 当月选定后向前填充到该月各交易日。

所有筛选窗口均使用 `index < rebalance_date`，因此不会使用当日之后的数据。测试会修改未来股价并确认历史映射不发生变化。

## 4. RS 计算方法

设股票收盘价为 `P_t`，当日选定 benchmark 收盘价为 `B_t`：

```text
stock_log_return_t = log(P_t) - log(P_{t-1})
benchmark_log_return_t = log(B_t) - log(B_{t-1})
excess_log_return_t = stock_log_return_t - benchmark_log_return_t
```

跨 benchmark 切换时，将每日超额对数收益连续链接：

```text
rs_line_t = 100 * exp(sum(excess_log_return_1 ... excess_log_return_t))
```

窗口相对收益：

```text
rs_return_21d = (exp(sum(last 21 excess_log_return)) - 1) * 100
rs_return_63d = (exp(sum(last 63 excess_log_return)) - 1) * 100
```

RS 动量先计算 `rs_line` 相对其 21/63 日滚动均值的 z-score，再使用 span=10 的 EWM 平滑。

### 4.1 Beta 的准确含义

`benchmark_beta` 和 `benchmark_r2` 来自股票收益对 benchmark 收益的一元回归，用于：

- 判断哪个 benchmark 更能解释股票波动；
- 展示股票对该 benchmark 的敏感度和拟合质量。

当前 `excess_log_return` **没有**计算成 `stock_return - beta * benchmark_return`，而是标准相对强度口径 `stock_return - benchmark_return`。因此当前 RS 是 benchmark 相对收益，不是 beta-neutral alpha 或回归残差。如果后续模型需要市场中性特征，应额外增加独立字段，不能直接改变现有 RS 的含义。

## 5. 市场内百分位

系统按市场分别构建当日所有股票的 `rs_return_63d` 横截面：

```text
rs_rank_63d = percentile_rank(rs_return_63d) * 100
```

- 90 表示该股票的 63 日相对收益超过当日同市场约 90% 的有效股票；
- US、CN、HK 分别排名，不能直接把不同市场的百分位当成同一个横截面；
- 新股或不足 63 个有效交易日的股票没有该字段；
- 每个历史日期只使用该日已经存在的收益，不使用未来行情。

## 6. 输出字段与文件

### 6.1 单股历史

目录：`data/cache/relative_strength/{us|cn|hk}/{symbol}.parquet`

| 字段 | 含义 |
|---|---|
| `benchmark_id` | 当日采用的 benchmark |
| `benchmark_beta` | 最近一次月度回归 beta |
| `benchmark_r2` | 最近一次月度回归 R² |
| `excess_log_return` | 股票减 benchmark 的日对数收益 |
| `rs_line` | 以 100 为基准连续链接的 RS 曲线 |
| `rs_return_21d` | 21 日累计相对收益百分比 |
| `rs_return_63d` | 63 日累计相对收益百分比 |
| `rs_momentum_21d` | 21 日 RS 动量 |
| `rs_momentum_63d` | 63 日 RS 动量 |
| `rs_rank_63d` | 当日市场内 63 日 RS 百分位 |

### 6.2 每日汇总

目录：`data/output/relative_strength/YYYY-MM-DD/`

- `{market}_rs_latest.csv`：各市场每只股票最新有效 RS；
- `{market}_benchmark_mapping.csv`：当前 benchmark、候选、beta、R²；
- `rs_latest.csv`：三市场合并结果；
- `benchmark_mapping.csv`：三市场合并映射；
- `summary.json`：数量及 benchmark 最新日期；
- `holding_rs_latest.csv`：按需生成的 holding/focus/watchlist 查看表，不是 RS 主流程必需产物。

## 7. 当前数据状态

截至 2026-07-12 重算结果：

| 市场 | 映射/历史股票数 | 最新有效 RS 到 2026-07-10 | 例外情况 |
|---|---:|---:|---|
| US | 461 | 456 | 5 只失效代码或无新行情 |
| CN | 305 | 303 | 1 只到 7月8日，1 只不足有效窗口 |
| HK | 575 | 571 | 4 只无有效63日窗口或代码异常 |
| 合计 | 1341 | 1336 | 见各市场输出明细 |

历史训练可用范围不能只看 parquet 的第一行，应以目标字段第一个非空日期为准：

- US 63 日 RS 最早约为 2020-04-02；
- CN 63 日 RS 最早约为 2020-04-09；
- HK 按当前近三年策略，63 日 RS 最早约为 2023-05-22；
- 三市场统一训练建议从 2023-05-22 或更晚日期开始，并按股票过滤无效窗口。

部分股票上市较晚，其有效起点会晚于市场统一起点。

## 8. 每日更新流程

`daily_update.py` 已接入两个步骤：

1. `RS benchmark`：在股票 RS 之前通过 OpenD 更新六个 benchmark；
2. `多市场RS`：股票行情更新后，重新计算三市场历史和最新横截面排名。

常用命令：

```bash
# 完整每日流程，包含行情、benchmark 和 RS
.venv/bin/python daily_update.py

# 仅更新 benchmark
.venv/bin/python daily_update.py --benchmarks

# 仅基于现有行情重算 RS
.venv/bin/python daily_update.py --rs
```

RS 和 benchmark 在每日状态中标记为非阻塞步骤：失败会记录，但不会阻止 Vegas 等核心扫描读取已更新行情。

## 9. 用于模型训练的建议

1. 使用单股 parquet 构建逐日特征，不要只使用 `rs_latest.csv`；
2. 先按字段非空过滤，63 日特征至少需要 63 个有效收益；
3. 按时间切分训练、验证、测试集，禁止随机打散同一股票的相邻日期；
4. 标签日期之后的数据只能用于标签，不能进入特征；
5. `benchmark_id` 可以作为类别特征，`benchmark_beta/r2` 可以作为连续特征；
6. 对跨市场模型增加 market 特征，或分别做市场内标准化；
7. `rs_rank_63d` 已是市场内横截面值，但仍存在当前股票池带来的幸存者偏差；
8. 对退市、失效代码和历史时点成分股进行补全，才能构建严格的无幸存者偏差训练集；
9. 如果使用 beta-neutral alpha，应新增 `beta_adjusted_excess_return`，不要覆盖当前 RS 字段。

## 10. 已知限制与后续改进

- US 当前统一用 QQQ，行业 ETF（如 SOXX）尚未进入候选；
- CN 缺少完整行业元数据，主要依赖代码板块；
- HK 部分股票行业缺失，会回落到恒指/恒生科技候选；
- 杠杆、反向及跨市场 ETF 不应沿用普通股票映射，需要单独维护底层指数和杠杆倍数；
- 当前股票池是当前列表，历史排名存在幸存者偏差；
- 当前 beta 只用于映射与解释，尚未形成 beta-neutral RS；
- HK 仅维护近三年历史，限制了更长周期的跨市场统一训练。

