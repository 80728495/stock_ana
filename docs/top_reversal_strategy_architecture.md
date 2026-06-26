# 顶部识别策略架构设计

> 临时设计文档。目标是在继续实现 SMC 前，先把顶部识别的策略边界、特征体系、评分方式和迭代路径理顺。

---

## 1. 当前目标

当前我们不是在写一个固定死的顶部规则，而是在构建一个可迭代的顶部研究框架：

1. 从 watchlist / 后续全量市场中提取顶部候选。
2. 用后验标签区分 `true_top`、`continuation`、`ambiguous`。
3. 反复查看正例、假阳性、漏召回图表。
4. 新增特征，重新训练打分模型。
5. 保留可解释性，直到样本量足够后再切换到更复杂模型。

### 1.1 双任务拆分

顶部研究现在拆成两个不同业务目标：

| 任务 | 正例 | 负例 | 特征侧重点 |
|---|---|---|---|
| 大跌规避 `drawdown` | 未来出现持续、明显下跌的阶段顶部 | 很快创新高或回撤较浅的中继 | 支撑阻力、均线、SMC 结构、下跌确认、关键位跌破 |
| 真正逃顶 `escape_top` | 高位大幅上涨后的主升浪顶部，且后续深跌 | 上涨中继、反弹顶、下跌中继顶、磨底波动顶 | 大波段涨幅、远离长均线、接近 252 日高点、顶部扩张耗尽 |

这两个任务不能混在同一个 `true_top` 中训练。反弹后的波段顶如果后续大跌，是大跌规避任务的正例；但它不是逃顶任务的正例，因为它的支撑、均线和阻力结构通常已经离当前价格很近，和主升浪高位顶部的市场含义不同。

当前代码保留原来的 `label=true_top/continuation/ambiguous` 作为大跌规避任务，同时新增：

| 字段 | 含义 |
|---|---|
| `drawdown_label` | 原始大跌规避标签，和旧 `label` 保持一致 |
| `escape_top_candidate_pool` | 第一层高位逃顶候选池，只使用当日可见的涨幅/高位/长均线乖离条件 |
| `escape_top_label` | 逃顶任务标签：`escape_top/high_continuation/ambiguous/out_of_escape_pool` |
| `escape_top_rise_ok` | 是否满足大波段涨幅 |
| `escape_top_near_high_ok` | 是否足够接近 252 日高点 |
| `escape_top_long_ema_ok` | 是否明显远离 EMA144/EMA200 |

第一层高位候选池的默认条件：

```text
max(anchor rise, major wave rise) >= 80%
and top is within 8% of 252-day high
and (top distance to EMA144 >= 25% or top distance to EMA200 >= 30%)
```

第二层逃顶模型只在 `escape_top_candidate_pool == 1` 的样本里训练：

```text
positive:
  escape_top_candidate_pool == 1
  and drawdown_label == true_top

negative:
  escape_top_candidate_pool == 1
  and drawdown_label == continuation
```

这样负例是同样处在高位、大涨幅、远离长均线背景下的上涨中继，而不是低位反弹或普通震荡点。第一层阈值是研究初始值，后续需要通过误判图和漏召回图继续调。

### 1.2 技术衰竭特征

逃顶二阶段模型新增 `technical_exhaustion` 特征组，全部只使用候选顶部或确认日之前的数据：

| 类型 | 特征示例 | 含义 |
|---|---|---|
| 顶背离 | `rsi14_bear_div_60d`, `macd_hist_bear_div_60d`, `macd_line_divergence_pct_60d` | 价格相对前 60 日前高继续走高，但 RSI/MACD 动能没有同步走高 |
| 严重超买 | `rsi14_top`, `stoch14_top`, `bb20_zscore_top`, `overbought_score` | RSI、随机指标、布林带偏离共同刻画高位过热 |
| 缩量上涨 | `vol5_vs_vol20_top`, `vol5_ret20_pct`, `volume_dryup_rise20`, `price_up_volume_down_20d` | 顶前价格上涨时成交活跃度是否跟不上 |
| 高位放量/滞涨 | `vol_ratio_top_20`, `vol_ratio_top_50`, `top_close_position_pct`, `top_upper_shadow_pct`, `high_volume_stall_score` | 顶部附近是否出现放量但收盘位置偏弱或上影明显 |

目前主流程是：

```text
两类 K 线顶部形态
  ├─ 长上影顶部
  └─ 高位十字星
        ↓
候选合并与去重
        ↓
ZigZag / anchor / 涨幅上下文
        ↓
指数轧空上下文
        ↓
后验标签 true_top / continuation / ambiguous
        ↓
逻辑回归打分 top_prob
        ↓
按 predicted_true_top / false_positive / missed_true_top 打图复盘
```

---

## 2. PDD 案例复盘

### 2.1 现象

`US:PDD 2024-10-02` 是一个典型的中概指数轧空行情下的短期尖顶。

关键特征：

| 维度 | 数值 |
|---|---:|
| 顶部日期 | `2024-10-02` |
| 标签 | `true_top` |
| 原始模型分数 | `0.0762` |
| 加入指数轧空后分数 | `0.9781` |
| anchor | `2024-09-13` |
| anchor 后天数 | `13` |
| anchor 涨幅 | `69.97%` |
| 5 日涨幅 | `36.04%` |
| 10 日涨幅 | `57.79%` |
| 20 日涨幅 | `67.12%` |
| 确认日从顶部跌幅 | `-1.54%` |
| 后续最大跌幅 | `-39.26%` |
| 恒生科技 10 日涨幅 | `47.45%` |
| 恒生科技 20 日涨幅 | `44.84%` |

### 2.2 原模型为什么漏掉

原模型偏向识别成熟趋势顶部：

- 顶部后 2-3 天内要有明显确认跌幅。
- 上涨周期通常更长，anchor 到顶部的天数更接近几十天。
- 长周期涨幅、均线趋势、确认下杀是主要加分项。

PDD 的形态恰好相反：

- 13 个交易日快速暴涨。
- 顶部确认日跌幅很小。
- 真正下跌周期在之后展开。
- 个股暴涨和指数轧空高度共振。

因此，PDD 应被归类为一个独立策略类型：指数轧空型疯牛见顶，而不是成熟趋势顶部。

---

## 3. 策略类型拆分

顶部候选不是一种形态，而是多个策略子类型的合集。

### 3.1 K 线顶部形态

这是候选来源，不是最终判定。

当前已有：

- 长上影线顶部。
- 高位十字星顶部。

作用：

- 提供顶部候选点。
- 给模型输入基础 candle score。
- 不单独决定 `true_top`。

### 3.2 成熟趋势顶部

典型特征：

- 前期经历较长时间上涨。
- 距 252 日高点很近。
- anchor 涨幅较大。
- 均线多头但开始远离或钝化。
- 顶部后 2-3 天出现较强确认下跌。

当前逻辑回归对这类识别较好。

### 3.3 指数轧空型疯牛见顶

这是从 PDD 案例中新增的独立策略模块。

核心假设：

> 当指数或板块出现极端短期轧空上涨时，指数成分股或强相关股票会集体出现短期疯牛走势。个股顶部可能在 2-3 天内并不充分确认，但之后很容易进入明显下跌周期。

这个策略尤其适用于：

- 港股恒生科技 / 恒生指数短期暴涨。
- 中概 ADR 集体轧空。
- ETF 或指数代理标的短期剧烈上涨。
- 港股流动性较弱时的逼空行情。

### 3.4 SMC 结构切换顶部

这是待实现模块。

核心假设：

> 稳定上涨过程中通常连续产生看涨 OB，形成支撑阶梯；顶部或下跌初期会开始出现看跌 OB，或者看涨 OB 阶梯被破坏。

SMC 不应作为首发顶部信号，而应作为结构确认和中继过滤器。

---

## 4. 指数轧空型疯牛策略

### 4.1 策略定位

该策略不替代 K 线顶部形态，而是在 K 线候选出现后，提供一组强上下文特征。

```text
个股顶部候选
  + 个股短期暴涨
  + 指数短期暴涨
  + 个股属于指数/板块共振范围
  = 指数轧空型疯牛顶部候选
```

### 4.2 当前已验证特征

| 特征 | 含义 |
|---|---|
| `china_hk_focus` | 是否是港股或中概相关美股 |
| `max_ret_5_10_20` | 个股 5/10/20 日最大涨幅 |
| `short_spike_like` | 是否短期尖峰暴涨 |
| `hstech_ret_10d` | 恒生科技 10 日涨幅 |
| `hstech_ret_20d` | 恒生科技 20 日涨幅 |
| `china_hk_short_spike` | 中概/HK 个股短期尖峰 |
| `china_hk_index_squeeze_spike` | 中概/HK + 恒科轧空 + 个股短期尖峰 |
| `china_hk_index_squeeze_weak_confirm` | 指数轧空尖峰且顶部确认日跌幅不充分 |

当前临时阈值：

```text
short_spike_like:
  bars_from_anchor_low <= 25
  rise_from_anchor_low_pct >= 45
  max(prior_ret_5d, prior_ret_10d, prior_ret_20d) >= 35

index_squeeze:
  hstech_ret_10d >= 15
  or hstech_ret_20d >= 20
```

### 4.3 PDD 后的模型变化

加入指数轧空特征后：

| 指标 | 加入前 | 加入后 |
|---|---:|---:|
| PDD top_prob | `0.0762` | `0.9781` |
| PDD 是否召回 | 否 | 是 |
| 前 20% 真顶率 | `70.7%` | `73.9%` |
| 未召回真顶 | `213` | `207` |
| 严格 PDD-like 未召回 | `1` | `0` |

### 4.4 风险

指数轧空特征会提高对中概/HK 集体行情的敏感度，但也会带来假阳性。

已观察到的风险样本：

| 样本 | 问题 |
|---|---|
| `HK:01772 2024-10-02` | 被指数轧空推高分数，但实际更像上涨中继 |

后续需要用以下特征过滤：

- 是否快速收复顶部。
- 下跌是否真正进入阶段性周期。
- 是否仍有看涨 OB 阶梯支撑。
- 是否出现看跌 OB 接管结构。
- 顶部后 5-10 天是否跌破关键低点。

---

## 5. SMC 顶部策略设计

### 5.1 原则

SMC 不能简单作为 `has_bearish_ob` 硬过滤。

原因：

1. 看跌 OB 的确认通常偏晚，等确认时可能已经跌了很多。
2. OB 的 `origin` 和 `confirmation` 必须拆开看。
3. 稳定上涨阶段往往是看涨 OB 主导，看跌 OB 更多出现在顶部或下跌过程中。
4. SMC 更适合判断结构切换，而不是替代顶部 K 线形态。

### 5.2 OB 时间字段

需要把 OB 拆成两个时间点：

| 字段 | 含义 | 是否可用于实时判断 |
|---|---|---|
| `ob_origin_pos` | OB 所在 K 线 | 只能用于研究解释，不能直接用于实时判断 |
| `ob_confirmed_pos` | 结构突破后 OB 被确认的 K 线 | 可用于实时判断 |
| `confirm_delay_bars` | `confirmed_pos - origin_pos` | 衡量 SMC 延迟 |

当前 SMC 代码把 OB 记录在 origin bar 上，但没有显式输出 confirmed bar。接入顶部模型前，需要先补这个字段，避免未来函数。

### 5.3 SMC 特征分层

#### A. Live 特征

只使用 `ob_confirmed_pos <= 当前判断日` 的信息。

用途：2-3 天内早期判断。

候选字段：

| 特征 | 含义 |
|---|---|
| `smc_live_bull_ob_count_60d` | 过去 60 天已确认看涨 OB 数量 |
| `smc_live_bull_ob_score_sum_60d` | 已确认看涨 OB 质量总分 |
| `smc_live_last_bull_ob_age` | 最近看涨 OB 距今多少天 |
| `smc_live_nearest_bull_ob_dist_pct` | 当前价到最近看涨 OB 的距离 |
| `smc_live_bull_ob_mitigated_10d` | 近 10 天看涨 OB 失效数 |
| `smc_live_bear_ob_count_20d` | 近 20 天已确认看跌 OB 数量 |
| `smc_live_bear_ob_score_max_20d` | 近 20 天最高看跌 OB 分数 |
| `smc_live_ob_regime_score` | 近 60 天看涨 OB 分数 - 近 20 天看跌 OB 分数，越高代表看涨 OB 阶梯仍占优 |

#### B. Delayed 特征

使用顶部后 5-10 天内出现的 SMC 信息。

用途：二次确认和模型研究，不作为顶部当天的首发条件。

候选字段：

| 特征 | 含义 |
|---|---|
| `smc_d5_bear_ob_confirmed` | 顶部后 5 天内是否确认看跌 OB |
| `smc_d10_bear_ob_confirmed` | 顶部后 10 天内是否确认看跌 OB |
| `smc_d10_bear_ob_score_max` | 顶部后 10 天最高看跌 OB 质量 |
| `smc_d10_bull_ob_mitigated_count` | 顶部后 10 天看涨 OB 失效数量 |
| `smc_d10_ob_regime_flip` | 是否从 bull-dominant 切到 bear-dominant |

#### C. Diagnostic 特征

用于解释，不直接喂给 2-3 天实时模型。

| 特征 | 含义 |
|---|---|
| `smc_diag_bear_ob_confirmed_near_top` | 结构最终确认的看跌 OB 形成区是否落在顶部附近 |
| `smc_diag_top_inside_bear_ob_confirmed_zone` | 顶部价格是否在该看跌 OB 区间 |
| `smc_diag_bear_ob_confirm_delay` | 看跌 OB 从形成到结构确认用了几天 |

### 5.4 当前 OB 特征实现

已在研究层实现 OB 特征，定位为“顶部候选被发现后的特征源”，不作为召回源。

实现边界：

| 模块 | 说明 |
|---|---|
| `src/stock_ana/research/top_reversal/smc_context.py` | 生成带 `origin_pos` / `confirmed_pos` 的 OB event，并为候选点构建 SMC 特征 |
| `src/stock_ana/research/top_reversal/feature_pipeline.py` | 在指数轧空特征之后追加 SMC OB 特征 |
| `src/stock_ana/research/top_reversal/feature_registry.py` | 将 `smc_live_*` / `smc_d5_*` / `smc_d10_*` / `smc_diag_*` 登记到 `smc_structure` |

当前实现刻意不直接改底层 `src/stock_ana/strategies/impl/smc.py`，而是在 research wrapper 中补出确认时间字段。
这样可以保留完整 OB 事件史，包括后续已经失效或清除的 OB，用于统计 `smc_d10_bull_ob_mitigated_count`。

当前在两形态顶部候选上的分层效果：

| 模型 | top 19% precision | 结论 |
|---|---:|---|
| base_no_smc | 73.9% | 非 SMC 基线 |
| base_plus_smc_live | 72.3% | 2-3 天内可见的 live OB 特征暂未提升 |
| base_plus_smc_live_delayed | 78.8% | 顶后 5-10 天确认信息有提升，但需警惕后验跌幅代理 |
| base_plus_smc_all | 84.8% | diagnostic 特征解释力最强，但不能直接当实时特征 |

进一步做 delayed SMC 反证测试：

| 模型 | top 19% precision | 观察 |
|---|---:|---|
| base_plus_smc_delayed | 78.8% | delayed SMC 原始特征 |
| base_plus_smc_event_drop_proxy | 80.4% | 将 delayed SMC 替换为“SMC 确认/失效发生时已产生的跌幅”后，效果相近且更强 |
| base_plus_generic_d10_price_proxy | 88.6% | 直接使用顶后 10 天价格回撤，效果更强，说明后验价格信息本身很强 |
| base_plus_smc_delayed_plus_event_proxy | 80.4% | 加入事件跌幅代理后，delayed SMC 没有明显增量 |

这说明 delayed SMC 的提升不能直接理解为“SMC 结构有领先预测力”。更合理的解释是：

```text
delayed SMC confirmation
  ≈ 一个带 SMC 触发条件的后验下跌确认
  ≈ 间接编码了顶部后已经发生的跌幅和结构破坏
```

当前最强 SMC 证据是：

| 特征 | 观察 |
|---|---|
| `smc_diag_top_inside_bear_ob_confirmed_zone` | true_top 中非零约 28%，continuation 中约 3%，区分度强 |
| `smc_d10_bull_ob_mitigated_count` | 顶后 10 天看涨 OB 失效更偏向 true_top |
| `smc_d10_bear_ob_score_max` | true_top 出现率高，但在线性模型中方向受相关特征影响，需要继续复盘 |

因此，SMC 的当前结论是：

```text
live OB 特征适合保留，但暂不能单独提高早期识别；
delayed OB 特征只能作为二次确认/研究解释，不应作为 2-3 天早期模型的有效性证据；
diagnostic OB 特征适合解释和样本复盘，不能混入严格实时模型。
```

当前主模型输出 `top_candidate_logistic_scored.csv` 默认不使用任何 `smc_*` 特征。
SMC 相关效果单独输出到：

| 文件 | 说明 |
|---|---|
| `top_candidate_smc_model_comparison.csv` | base / live / delayed / all SMC 分层对比 |
| `top_candidate_smc_delay_proxy_comparison.csv` | delayed SMC 与已发生跌幅代理的反证对比 |

### 5.5 SMC 对不同策略的作用

| 策略类型 | SMC 作用 |
|---|---|
| 成熟趋势顶部 | 看跌 OB 确认、看涨 OB 失效可增强分数 |
| 指数轧空型疯牛顶部 | SMC 常常滞后，不应作为首发条件 |
| 上涨中继 | 如果看涨 OB 阶梯完整，应降低顶部概率 |
| 假突破尖峰 | 如果无指数共振且无 OB regime flip，应降低分数 |

### 5.6 SMC raw / early / confirmed 重新设计

外部 SMC 指标里常见的“隔天出现 OB/SMC 信号”，通常不是严格意义上的 confirmed bearish OB。
更常见的机制是：

1. 使用 internal structure，而不是只使用 swing structure。
2. 使用更短的 swing 长度或微结构低点，所以 CHoCH / MSS 更快出现。
3. 先画 potential / historical OB zone，后续结构确认后再把 zone 回填到形成 K 线。
4. 使用 displacement / FVG / liquidity sweep 作为早期结构切换证据。
5. 图表展示层可以显示 formed zone，但模型层必须用 `detected_pos` 控制信息可见时间。

因此，SMC 在顶部模型里应拆成三层生命周期：

```text
smc_raw
  = 潜在供给区/需求区被形成
  = 不要求完整 BOS/CHoCH
  = 要记录 detected_pos，防止把后续确认回填成顶部当天可见

smc_early
  = 内部结构已经开始转弱
  = 允许 0-3 天内使用 internal CHoCH/MSS、bearish FVG、bull OB mitigation
  = 目标是比 confirmed OB 早，但比纯 K 线形态更有结构含义

smc_confirmed
  = 当前 `_build_ob_events` 里的正式 confirmed OB
  = 适合复盘、二次确认和 delayed 对照
  = 不作为 2-3 天早期模型的主要增量证据
```

#### A. `smc_raw`：原始候选，信号出现

`smc_raw` 不是“已确认看跌 OB”，而是顶部附近出现了可能成为后续看跌 OB 的 supply zone。

建议检测条件：

| 维度 | 规则 |
|---|---|
| formed 位置 | 取候选顶部附近 `[-1, +1]` 的最高成交量 K 线、最高 high K 线或最后一根强阳 K 线 |
| zone 范围 | 默认使用 formed candle 的 `[low, high]`，可增加 body-only / wick-refined 两种版本 |
| 前置趋势 | 候选前需要已有可观涨幅，避免横盘里的普通 supply zone |
| displacement | formed bar 后 1-3 天出现向下实体扩张，`body / ATR` 或 `true_range / ATR` 达阈值 |
| bearish FVG | formed bar 后 1-3 天出现 bearish FVG，并与 formed zone 有重叠或相邻 |
| liquidity sweep | 顶部 high 扫过近端 swing/equal high 后收回，增强 raw 可信度 |
| volume | formed bar 或 displacement bar 放量，增强供给区可信度 |

候选字段：

| 特征 | 含义 |
|---|---|
| `smc_raw_bear_present_3d` | 顶部后 3 天内是否出现 bearish raw setup |
| `smc_raw_bear_score_max_3d` | 最高 raw setup 质量分 |
| `smc_raw_bear_detect_lag` | `detected_pos - formed_pos`，衡量是否足够早 |
| `smc_raw_bear_zone_overlap_top` | 顶部价格/顶部 K 线是否落在 formed zone 内 |
| `smc_raw_bear_displacement_atr` | formed bar 后向下 displacement 强度 |
| `smc_raw_bear_has_fvg` | formed bar 后是否有 bearish FVG |
| `smc_raw_bear_has_sweep` | formed bar 是否伴随流动性扫高后回落 |
| `smc_raw_bear_zone_width_atr` | zone 宽度相对 ATR，避免过宽无效区间 |
| `smc_raw_bear_volume_ratio` | formed bar 成交量相对 20 日均量 |

`smc_raw` 的核心价值：

```text
把后验看起来落在顶部附近的 bearish OB，
改造成一个有 detected_pos 的早期 potential OB 信号。
```

#### B. `smc_early`：内部结构开始转弱

`smc_early` 是 2-3 天内真正可用的结构特征。
它不要求正式看跌 OB 确认，但要求至少出现一个内部结构破坏。

建议检测条件：

| 维度 | 规则 |
|---|---|
| internal CHoCH/MSS | 使用 `swing_length=1/2/3` 分层计算，候选后 0-3 天跌破内部 higher low |
| micro low break | 收盘跌破候选前 2-5 天最近微低点，或跌破顶部 K 线 low |
| bearish FVG | 顶后 1-3 天出现 bearish imbalance，代表下跌速度开始占优 |
| bull OB mitigation | 稳定上涨中的最近看涨 OB 被跌破/mitigated，代表支撑阶梯断裂 |
| failed retest | 下跌后回抽 formed zone / 顶部 K 线下沿失败 |
| bull ladder intact | 如果看涨 OB 阶梯仍完整且价格远在最近 bull OB 上方，应作为负向特征 |

候选字段：

| 特征 | 含义 |
|---|---|
| `smc_early_internal_choch_down_3d` | 3 天内 internal CHoCH/MSS 向下 |
| `smc_early_micro_low_break_3d` | 3 天内跌破近端微低点 |
| `smc_early_top_low_break_3d` | 3 天内跌破顶部 K 线 low |
| `smc_early_bear_fvg_3d` | 3 天内出现 bearish FVG |
| `smc_early_bull_ob_mitigated_3d` | 3 天内最近 bull OB 被 mitigation |
| `smc_early_bull_ladder_intact` | 看涨 OB 阶梯是否仍完整，完整则降低顶部概率 |
| `smc_early_retest_reject_5d` | 5 天内回抽 supply zone 失败 |
| `smc_early_score_3d` | early 结构切换综合分 |

建议 `smc_early_score_3d` 初始打分：

| 分组 | 权重 | 例子 |
|---|---:|---|
| 结构破坏 | 35 | internal CHoCH/MSS、micro low break、top low break |
| 供给区质量 | 30 | origin overlap、displacement、FVG、volume |
| 支撑阶梯破坏 | 20 | bull OB mitigation、bull ladder 从完整转弱 |
| 反弹失败 | 15 | retest rejection、无法收复 formed zone |

#### C. 实现边界

新增 SMC 事件表时，必须显式区分以下字段：

| 字段 | 含义 |
|---|---|
| `event_type` | `raw` / `early` / `confirmed` / `mitigated` / `cleared` |
| `direction` | `1` 看涨，`-1` 看跌 |
| `origin_pos` | zone 所在 K 线 |
| `detected_pos` | 算法首次能知道该事件的 K 线 |
| `confirmed_pos` | 正式 OB/BOS/CHoCH 确认 K 线，可为空 |
| `zone_top` / `zone_bottom` | 结构区间 |
| `structure_scale` | `internal_1` / `internal_2` / `internal_3` / `swing_5` |
| `score` | 当前生命周期阶段的质量分 |

图表可以把 zone 画回形成 K 线，但模型特征只能在 `detected_pos <= asof_pos` 时使用。
这是和 TradingView 展示效果最容易混淆的地方。

#### D. 验证方式

新增后必须做四组对照：

| 模型 | 目的 |
|---|---|
| `base_no_smc` | 非 SMC 基线 |
| `base_plus_smc_raw` | 判断 potential OB zone 是否提供早期增量 |
| `base_plus_smc_early` | 判断内部结构破坏是否提供 2-3 天增量 |
| `base_plus_smc_raw_early` | 判断两者组合是否稳定提升 |

同时继续做反证测试：

1. 加入 `d1/d2/d3` 已发生跌幅代理，确认 `smc_early` 不是简单复制短期跌幅。
2. 按市场、策略类型、顶部子类型分桶看 precision / recall。
3. 对 `predicted_true_top`、`false_positive`、`missed_true_top` 分别出图，图上同时标注 raw zone、early event、confirmed OB。

如果 `base_plus_smc_raw_early` 在加入短期跌幅代理后仍然有稳定增量，才说明 SMC 捕捉到的不是单纯“已经跌了”，而是结构切换。

#### E. 外部参考

- [TradingView / LuxAlgo Smart Money Concepts](https://www.tradingview.com/script/CnB3fSph-Smart-Money-Concepts-LUX/)：同时包含 internal structure、swing structure、internal/swing OB、FVG、EQH/EQL，并支持 Historical / Present 展示模式。
- [`smartmoneyconcepts` Python 包](https://pypi.org/project/smartmoneyconcepts/0.0.26/)：提供 FVG、swing high/low、BOS/CHoCH、OB、liquidity 等函数；其中 swing high/low 明确依赖前后 `swing_length` 根 K 线，这也是 confirmed OB 延迟的重要来源。

---

## 6. 多维特征体系

最终顶部模型应包含以下特征组。

### 6.1 K 线形态特征

- `has_shadow`
- `has_doji`
- `score_max`
- `score_sum`
- 上影线长度、ATR 比例、成交量倍数
- 十字星实体比例、确认成交量

### 6.2 个股量价位置

- `prior_ret_5d / 10d / 20d / 40d / 60d / 120d`
- `rise_from_anchor_low_pct`
- `bars_from_anchor_low`
- `top_vs_252high_pct`
- `top_vs_252low_pct`
- `dist_ema55_pct / dist_ema144_pct / dist_ema200_pct`
- `ema55_slope_20d_pct`
- `confirm_drop_from_top_pct`

### 6.3 ZigZag / Anchor 结构

- `recent_zigzag_low`
- `anchor_low`
- `anchor_source`
- `middle_low`
- `pre_head_low`
- `middle_vs_pre_head_pct`
- `m_top_like`
- `top_cluster_high_count`

### 6.4 指数轧空特征

- `china_hk_focus`
- `hstech_ret_5d / 10d / 20d / 40d`
- `hsi_ret_10d / 20d`
- `short_spike_like`
- `china_hk_index_squeeze_spike`
- `china_hk_index_squeeze_weak_confirm`

### 6.5 SMC 结构特征

- `smc_live_*`
- `smc_raw_*`
- `smc_early_*`
- `smc_d5_*`
- `smc_d10_*`
- `smc_diag_*`

### 6.6 标签特征

标签仍然保持三类：

| 标签 | 含义 |
|---|---|
| `true_top` | 后续出现明显下跌周期和明显跌幅 |
| `continuation` | 后续继续创新高或快速收复 |
| `ambiguous` | 中间态，不参与二分类训练 |

---

## 7. 样本构建 v2：从形态召回转向 ZigZag 顶部全集

当前样本构建方式是：

```text
长上影 / 高位十字星
  → 合并候选点
  → ZigZag / anchor / 未来走势打标签
  → true_top / continuation / ambiguous
```

这个方式的问题是存在样本选择偏差：

> 模型只学习了“已经被两个顶部形态召回的候选里，哪些更像真顶”，而不是学习“所有阶段顶部长什么样”。

如果两个顶部形态没有 100% 召回，那么后续逻辑回归、SMC、双顶、前高阻力位等特征，都会基于一个不完整的训练集合。

### 7.1 新原则

候选来源和特征来源要分开。

```text
ZigZag / swing high
  → 提供更完整的顶部候选全集

长上影 / 高位十字星
  → 不再决定样本是否进入训练集
  → 改为候选来源之一 + K 线形态特征
```

更准确地说，不是“用 ZigZag 直接找所有真顶”，而是：

```text
用 ZigZag 找所有有意义的 swing high 候选
再用未来走势标签判断哪些是真顶
```

### 7.2 训练候选全集

建议新增候选生成层：

| candidate source | 作用 |
|---|---|
| `pattern_shadow` | 长上影顶部候选 |
| `pattern_doji` | 高位十字星顶部候选 |
| `zigzag_peak` | 所有 ZigZag 高点候选 |
| `future_double_top` | 后续双顶/多头部候选 |
| `future_resistance` | 后续前高阻力候选 |

训练集应使用：

```text
combined_candidates = merge(
  zigzag_peak_candidates,
  pattern_shadow_candidates,
  pattern_doji_candidates,
  future_extra_candidate_sources...
)
```

其中 `zigzag_peak_candidates` 是主召回来源，其他来源提供补充和特征。

### 7.3 ZigZag 候选过滤

不能把所有微小 ZigZag 高点都放进训练，否则负例会被噪音淹没。

建议先使用保守过滤：

| 条件 | 建议 |
|---|---|
| 前置涨幅 | `rise_from_anchor_low_pct >= 20%` 或训练时更宽松 |
| swing 幅度 | 使用现有 ZigZag / wave 的最小波动过滤 |
| 位置 | 可保留所有位置，但记录 `top_vs_252high_pct` |
| 样本末尾 | 必须有足够 `lookahead_bars` 用于标签 |
| 近邻合并 | 相邻顶部候选在 `merge_bars` 内合并 |

标签仍然用未来走势决定：

```text
true_top:
  anchor rise 足够
  未来出现明显下跌
  下跌周期足够长
  下跌前没有快速收复顶部

continuation:
  后续继续创新高或快速收复

ambiguous:
  介于两者之间，暂不参与二分类训练
```

### 7.4 避免未来函数

ZigZag 高点本身常常需要未来 K 线确认。

因此要区分两套时间：

| 字段 | 含义 |
|---|---|
| `top_pos` | 顶部实际发生的位置 |
| `candidate_confirm_pos` | 该 ZigZag 高点在实时中可确认的位置 |
| `candidate_confirm_lag` | `candidate_confirm_pos - top_pos` |
| `score_asof_pos` | 模型允许使用信息的时间点 |

研究数据可以用 ZigZag 高点扩充样本，但实时模型不能把“未来确认的 ZigZag 顶”当成顶部当天已知。

因此应分两类模型/评估：

| 模型 | 用途 |
|---|---|
| `oracle_universe_model` | 研究所有真顶和特征差异，允许用 ZigZag 扩大样本全集 |
| `live_candidate_model` | 实盘候选打分，只能使用当时已触发的形态/结构信号 |

### 7.5 输出文件建议

新增一组 universe 数据输出：

```text
top_candidate_research/
  watchlist_pattern_candidates_labeled.csv      # 当前两形态候选
  watchlist_universe_candidates_labeled.csv     # ZigZag 扩充后的候选全集
  universe_true_tops_missed_by_patterns.csv     # 真顶但未被两形态召回
  universe_top_candidate_score_performance.csv
  universe_prediction_band_charts/
```

关键复盘表：

| 文件 | 目的 |
|---|---|
| `universe_true_tops_missed_by_patterns.csv` | 找出两个顶部形态漏掉的真顶 |
| `pattern_coverage_by_true_top.csv` | 评估长上影/十字星对真顶的覆盖率 |
| `universe_top_candidate_feature_diff.csv` | 在完整样本上重新看特征差异 |

### 7.6 对现有架构的影响

候选生成层要重构为：

```text
candidate_sources.py
  collect_pattern_candidates()
  collect_zigzag_peak_candidates()
  merge_candidate_sources()
```

特征层保持不变：

```text
feature_pipeline.py
  add_research_features()
```

模型层保持不变：

```text
modeling.py
  fit_logistic()
  feature_diff()
  bucket_stats()
```

这样后续加入 SMC、双顶、前高阻力时，只需要：

1. 决定它是 candidate source 还是 feature source。
2. 如果是 candidate source，接入 `candidate_sources.py`。
3. 如果是 feature source，接入 `feature_pipeline.py`。
4. 如果入模，登记到 `feature_registry.py`。

---

## 8. 打分架构

### 8.1 当前阶段

继续使用逻辑回归。

理由：

- 样本量仍不大。
- 特征含义明显。
- 权重可解释，便于人肉复盘。
- 每轮改特征后能快速知道谁在加分、谁在减分。

### 8.2 模块化分数

后续可以把总分拆成几个可解释子分：

```text
final_top_prob
  = model(
      candle_top_features,
      price_context_features,
      zigzag_anchor_features,
      index_squeeze_features,
      smc_structure_features
    )
```

展示上可输出：

| 分数 | 含义 |
|---|---|
| `candle_top_score` | K 线顶部形态强度 |
| `trend_exhaustion_score` | 个股涨幅/位置/均线耗尽程度 |
| `index_squeeze_score` | 指数轧空共振强度 |
| `smc_regime_score` | SMC 结构切换程度 |
| `final_top_prob` | 最终顶部概率 |

### 8.3 后续模型升级

当样本扩展到全量 US/HK/CN 后，可以考虑：

1. Gradient Boosting / XGBoost / LightGBM
2. 小型 MLP
3. 分策略专家模型

建议升级顺序：

```text
Logistic baseline
  ↓
Gradient Boosting
  ↓
Strategy-specific models
  ↓
Neural network
```

不建议太早上神经网络。当前更重要的是把标签、候选和特征做干净。

---

## 9. 输出与复盘

每次训练后至少输出：

```text
top_candidate_research/
  watchlist_top_candidates_labeled.csv
  top_candidate_logistic_scored.csv
  top_candidate_logistic_coefficients.csv
  top_candidate_score_performance.csv
  prediction_band_charts/
    predicted_true_top_gallery.md
    predicted_false_positive_gallery.md
    missed_true_top_gallery.md
```

图表右侧面板应包含：

- 标签与 top_prob。
- K 线策略和 score。
- anchor 来源与涨幅。
- 指数轧空特征。
- SMC live / delayed 核心字段。
- 未来结果统计。

复盘重点：

1. 高分真顶是否形态一致。
2. 高分假阳性为什么被误判。
3. 漏召回是否属于新策略类型。
4. 新特征是否解决一个问题但引入新假阳性。

---

## 10. 实现阶段

### Phase 0：设计沉淀

- 写入本文档。
- 不改 SMC 实现。
- 保持当前指数轧空特征已接入的状态。

### Phase 1：顶部研究脚本整理

当前已完成第一版代码骨架整理：

| 模块 | 角色 |
|---|---|
| `src/stock_ana/research/top_reversal/build_top_candidate_research.py` | CLI 编排入口：加载 watchlist、扫描候选、调用特征模块、训练、写输出 |
| `src/stock_ana/research/top_reversal/candidate_sources.py` | 候选来源层：ZigZag 顶部全集、后续策略召回匹配 |
| `src/stock_ana/research/top_reversal/coverage.py` | 策略覆盖率：统计两形态策略对顶部全集真顶的覆盖 |
| `src/stock_ana/research/top_reversal/feature_registry.py` | 特征注册表：按策略模块登记模型特征 |
| `src/stock_ana/research/top_reversal/feature_pipeline.py` | 特征流水线门面：统一调用指数、SMC、双顶、阻力位等特征 builder |
| `src/stock_ana/research/top_reversal/market_context.py` | 指数轧空特征：恒生/恒科、中概/HK、短期 spike |
| `src/stock_ana/research/top_reversal/modeling.py` | 研究模型：逻辑回归、分桶、特征差异、分数表现 |

拆分后的原则：

- 主脚本只做流程编排，不继续堆新增策略逻辑。
- 候选来源和模型特征分离：ZigZag 负责训练全集，两形态策略负责实盘召回和形态特征。
- 新特征先挂到 `feature_pipeline.py`，再在 `feature_registry.py` 登记入模字段。
- 每个策略特征源独立成模块。
- 每个新增特征必须先进入 `feature_registry.py` 的某个 feature group。
- 统计输出带 `feature_group` 字段，方便判断模型到底依赖哪类特征。

Phase 1 已落地的输出分成两条线：

| 输出线 | 文件 | 说明 |
|---|---|---|
| 旧兼容 / live candidate | `watchlist_top_candidates_labeled.csv` | 保持旧文件名，供已有图表脚本继续使用 |
| 旧兼容 / live candidate | `watchlist_pattern_candidates_labeled.csv` | 两形态候选的显式命名副本 |
| 旧兼容 / live candidate | `top_candidate_logistic_scored.csv` | 旧策略候选打分，作为重构回归基准 |
| universe / oracle research | `watchlist_universe_candidates_labeled.csv` | ZigZag 顶部全集 |
| universe / oracle research | `universe_true_tops_missed_by_patterns.csv` | 顶部全集中未被两形态召回的真顶 |
| universe / oracle research | `pattern_coverage_by_true_top.csv` | 两形态策略对全集真顶的覆盖率 |
| universe / oracle research | `universe_top_candidate_logistic_scored.csv` | 顶部全集上的研究模型打分 |

当前注册的 feature group：

| group | 状态 |
|---|---|
| `candle_pattern` | 已接入，两类 K 线顶部形态 |
| `price_context` | 已接入，量价/均线/确认跌幅 |
| `zigzag_anchor` | 已接入，anchor/M 山头/涨幅结构 |
| `index_squeeze` | 已接入，PDD 类指数轧空疯牛策略 |
| `wave_structure` | 已接入，波段结构和高位头部簇 |
| `smc_structure` | 已预留，等待 SMC confirmed_pos 改造 |
| `double_top` | 已预留，等待双顶特征 |
| `resistance` | 已预留，等待前高阻力位特征 |

- 将特征提取拆成模块：
  - candle features
  - price context
  - zigzag anchor
  - market index squeeze
  - smc context
- 保持当前输出兼容。

### Phase 2：SMC confirmed_pos 改造

- 在 `_ob_causal` 或包装层中输出：
  - `origin_pos`
  - `confirmed_pos`
  - `confirmed_date`
  - `confirm_delay_bars`
- 检查是否存在未来函数。

### Phase 3：SMC 特征接入

- 先接 `smc_live_*`。
- 再接 `smc_d5_* / smc_d10_*`。
- 最后接 `smc_diag_*` 用于图表解释。

### Phase 4：模型对比

对比四组：

| 模型 | 目的 |
|---|---|
| base | 当前非 SMC 基线 |
| base + index squeeze | 已验证 PDD 类 |
| base + smc live | 看 SMC 是否帮助早期判断 |
| base + smc live + delayed | 看 SMC 是否帮助确认和过滤 |

关注指标：

- 前 10% / 20% 真顶率。
- 未召回真顶数量。
- PDD-like 召回。
- 2024-10 中概簇表现。
- `HK:01772 2024-10-02` 这类假阳性是否下降。

---

## 11. 当前判断

目前顶部识别不应被理解为单一策略，而应理解为多策略候选 + 统一打分：

```text
K 线顶部形态负责发现候选。
ZigZag/anchor 负责解释上涨幅度和结构。
指数轧空负责识别群体性疯牛尖顶。
SMC 负责判断支撑阶梯是否失效、供给区是否接管。
模型负责把这些证据统一成概率分数。
```

PDD 的价值不是单个样本，而是暴露了一个重要策略类型：

> 在指数疯长轧空行情中，个股顶部可能很早出现，但传统 2-3 天确认特征不足。此时应利用指数级别的异常上涨作为核心上下文，提高对共振股票的顶部敏感度。

SMC 的价值也不是替代顶部形态，而是：

> 在顶部候选出现后，判断原有看涨 OB 阶梯是否开始失效，以及看跌 OB 是否开始接管结构。
