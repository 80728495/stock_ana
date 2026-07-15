# LazyBear Squeeze Momentum（SQZMOM_LB）指标说明

> 当前实现日期：2026-07-12  
> 实现代码：`src/stock_ana/data/indicators.py`  
> 持久化代码：`src/stock_ana/data/indicators_store.py`

## 1. 指标定位

SQZMOM_LB 用于同时描述：

1. 波动率是否处于压缩或释放状态；
2. 价格动量位于多头还是空头方向；
3. 当前方向的动量正在增强还是减弱。

它由 Bollinger Bands、Keltner Channel 和线性回归动量组成。虽然有时被归入“量能/动能指标”，原始算法并不使用成交量 `volume`，本质上是波动率和价格动量指标。

参考实现：

- LazyBear TradingView 原版：<https://www.tradingview.com/script/nqQ1DT5a-Squeeze-Momentum-Indicator-LazyBear/>
- pandas-ta-classic：<https://github.com/xgboosted/pandas-ta-classic/blob/main/pandas_ta_classic/momentum/squeeze.py>

项目采用本地轻量实现，没有引入完整的 pandas-ta-classic 依赖。

## 2. 默认参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `bb_length` | 20 | Bollinger Bands 窗口 |
| `bb_mult` | 2.0 | Bollinger 标准差倍数 |
| `kc_length` | 20 | Keltner Channel 窗口 |
| `kc_mult` | 1.5 | Keltner 波动范围倍数 |
| `use_true_range` | `True` | KC 使用 True Range |

这些参数与常见的 LazyBear 原版设置一致。

## 3. 计算方法

### 3.1 Bollinger Bands

```text
bb_basis = SMA(close, 20)
bb_dev = population_std(close, 20) * 2.0
upper_bb = bb_basis + bb_dev
lower_bb = bb_basis - bb_dev
```

标准差使用 `ddof=0`，对应总体标准差口径。

### 3.2 Keltner Channel

默认使用 True Range：

```text
true_range = max(
    high - low,
    abs(high - previous_close),
    abs(low - previous_close)
)

kc_basis = SMA(close, 20)
range_mean = SMA(true_range, 20)
upper_kc = kc_basis + range_mean * 1.5
lower_kc = kc_basis - range_mean * 1.5
```

### 3.3 Squeeze 状态

```text
squeeze_on  = lower_bb > lower_kc and upper_bb < upper_kc
squeeze_off = lower_bb < lower_kc and upper_bb > upper_kc
no_squeeze  = not squeeze_on and not squeeze_off
```

- `squeeze_on`：BB 完全进入 KC，波动率压缩；
- `squeeze_off`：BB 完全扩展到 KC 之外，波动率释放；
- `no_squeeze`：过渡或两者均不满足。

Squeeze 状态只描述波动率，不预测释放方向。

### 3.4 线性回归动量

先对收盘价去趋势：

```text
center = 0.25 * (highest(high, 20) + lowest(low, 20))
       + 0.50 * SMA(close, 20)

detrended = close - center
```

然后对最近 20 日 `detrended` 做最小二乘线性回归，取当前窗口末端的回归值，得到 `sqzmom_value`。

整个计算只使用当日及此前数据，不使用未来行情。

## 4. 输出字段

指标保存在：

```text
data/cache/indicators/{market}/{symbol}.parquet
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `sqzmom_value` | float | LazyBear 线性回归动量值 |
| `sqzmom_squeeze_on` | bool | 是否处于波动率压缩 |
| `sqzmom_squeeze_off` | bool | 是否处于波动率释放 |
| `sqzmom_no_squeeze` | bool | 是否处于过渡状态 |
| `sqzmom_squeeze_state` | Int8 | `1=on`、`0=neutral`、`-1=off` |
| `sqzmom_bar_state` | Int8 | 动量方向和加速度四态编码 |

`sqzmom_bar_state`：

| 值 | 状态 | TradingView 常见颜色语义 |
|---:|---|---|
| `2` | 正动量增强 | 浅绿 |
| `1` | 正动量减弱 | 深绿 |
| `-2` | 负动量增强 | 红色 |
| `-1` | 负动量减弱 | 暗红 |

颜色只用于图表展示，策略和模型应使用数值编码。

## 5. 正确用法

### 5.1 先看 Squeeze，再看方向

| 组合 | 常见解释 |
|---|---|
| squeeze-off + 正动量增强 | 向上释放，偏多 |
| squeeze-off + 负动量增强 | 向下释放，偏空 |
| squeeze-off + 正动量减弱 | 上涨仍在，但开始降速 |
| squeeze-off + 负动量减弱 | 下跌仍在，但跌势开始衰竭 |

`squeeze_on` 只是准备阶段，不能作为独立买入信号。

### 5.2 零轴确认

- 从负值上穿零轴：动量由空转多；
- 从正值跌破零轴：动量由多转空；
- 零轴确认通常晚于柱体开始减弱，确认性更强但延迟更大。

### 5.3 柱体变化作为预警

- 正动量增强转为正动量减弱：上涨加速度下降；
- 负动量增强转为负动量减弱：下跌加速度下降；
- 只代表动量变化，不等于价格立即反转。

## 6. 不建议的用法

不要采用以下简单规则：

- squeeze-on 后立即买入；
- 第一根正柱出现就追涨；
- 第一根负柱出现就清仓；
- 不考虑大趋势、位置和市场状态；
- 比较不同股票的 `sqzmom_value` 绝对大小；
- 把负动量减弱直接理解为买点；
- 用未收盘的日K柱状态作为稳定信号。

`sqzmom_value` 没有按价格或 ATR 归一化。例如价格较高的股票可能天然拥有更大的绝对动量值。跨股票建模时应使用方向、变化率、持续天数、零轴距离的归一值，或新增 ATR/价格归一化版本。

## 7. 在当前项目中的建议用途

SQZMOM_LB 更适合作为特征和状态过滤器，而不是独立召回策略。

### 7.1 顶部识别

可考虑以下特征：

- 高位正动量连续减弱；
- 股价创新高，但 SQZMOM 峰值降低；
- 正动量减弱持续天数；
- 动量由正转负；
- 高位 squeeze 后向下释放；
- squeeze-off 后负动量扩张速度；
- 与 RS 转弱、EMA 高乖离、看跌K线和 SMC 信号的共振。

单独的正动量减弱不能定义顶部。在上涨初期或强趋势中，它经常只是普通整理。

### 7.2 趋势突破

可作为以下组合的确认特征：

- Vegas 上涨趋势成立；
- RS 处于市场高分位；
- squeeze 从 on 转 off；
- SQZMOM 位于零轴上方并增强；
- 成交量或价格结构进一步确认突破。

### 7.3 下跌风险

风险较高的组合包括：

- 动量跌破零轴；
- 负动量持续增强；
- squeeze 向下释放；
- 同时跌破关键均线、支撑或出现看跌结构。

## 8. 是否属于高胜率指标

不存在脱离市场、周期和交易规则的统一胜率。SQZMOM_LB 是状态描述指标，不是完整交易系统。

它通常在方向明确的波动释放阶段更有效，在震荡环境中容易产生反复切换。胜率取决于：

- 股票池和市场；
- 日线或周线周期；
- squeeze-on、squeeze-off 或零轴突破的入场定义；
- 趋势与位置过滤；
- 持有周期、止损和退出方式；
- 对正确结果的标签定义。

在项目中使用前，应分别统计 squeeze-off 后 5、10、20、40 日收益和最大回撤，并按市场、Vegas趋势、RS分位及动量方向分组评估。

## 9. 每日更新流程

SQZMOM_LB 已加入 `add_daily_indicators()`，随每日指标步骤自动计算。

覆盖市场：

- US；
- NDX100；
- HK；
- CN。

完整日更：

```bash
.venv/bin/python daily_update.py
```

仅更新技术指标：

```bash
.venv/bin/python daily_update.py --indicators
```

执行顺序为：读取各市场 OHLCV 缓存，计算 EMA、成交量均线、前高和 SQZMOM_LB，然后覆盖保存对应指标 parquet。不会修改原始 OHLCV 文件。

2026-07-12 已完成一次历史全量计算：

| 市场 | 成功 | 跳过 | 失败 |
|---|---:|---:|---:|
| US | 1604 | 0 | 0 |
| NDX100 | 101 | 0 | 0 |
| HK | 852 | 0 | 0 |
| CN | 307 | 0 | 0 |

## 10. 测试与因果性

测试文件：`tests/test_squeeze_momentum_lazybear.py`

当前覆盖：

- 线性回归动量与独立参考公式一致；
- 状态编码只能取合法值；
- 修改未来价格不会改变过去指标；
- 缺少 OHLC 必要列时明确报错；
- 每日指标持久化入口包含全部 SQZMOM 字段。

