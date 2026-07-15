"""
技术指标计算模块
使用 ta 库（纯 Python），确保 Windows/macOS 跨平台无障碍运行
"""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

# Vegas 体系及扩展 EMA 窗口（用于 add_ema_extended）
_EMA_EXTENDED_WINDOWS = [8, 21, 34, 55, 60, 144, 169, 200, 250]

# 成交量均值窗口
_VOL_MA_WINDOWS = [5, 10, 20, 50]

# 前高回望窗口（交易日）
PREV_HIGH_WINDOW = 252  # ≈1 年


def add_ma(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """添加移动平均线（SMA + EMA）"""
    if windows is None:
        windows = [5, 10, 20, 60]

    for w in windows:
        df[f"sma_{w}"] = SMAIndicator(close=df["close"], window=w).sma_indicator()
        df[f"ema_{w}"] = EMAIndicator(close=df["close"], window=w).ema_indicator()
    return df


def add_ema_extended(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """
    添加 Vegas 体系扩展 EMA：EMA8, 21, 34, 55, 60, 144, 169, 200, 250。

    使用 pandas ewm（span=N, adjust=False）计算，与策略层保持一致。
    输出列名：ema_8, ema_21, ema_34, ema_55, ema_60, ema_144, ema_169, ema_200, ema_250
    """
    if windows is None:
        windows = _EMA_EXTENDED_WINDOWS
    close = df["close"].astype(float)
    for w in windows:
        df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()
    return df


def add_volume_ma(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """
    添加成交量移动平均：vol_ma_5, vol_ma_10, vol_ma_20, vol_ma_50。

    使用简单移动平均（SMA），min_periods=1 防止头部 NaN。
    """
    if windows is None:
        windows = _VOL_MA_WINDOWS
    vol = df["volume"].astype(float)
    for w in windows:
        df[f"vol_ma_{w}"] = vol.rolling(window=w, min_periods=1).mean()
    return df


def add_prev_high(df: pd.DataFrame, window: int = PREV_HIGH_WINDOW) -> pd.DataFrame:
    """
    添加前高价格：过去 N 个交易日（含当日）的收盘价滚动最高值。

    默认 window=252（≈1 年）。
    输出列名：prev_high_252d（或 prev_high_{window}d）
    """
    col_name = f"prev_high_{window}d"
    df[col_name] = df["close"].astype(float).rolling(window=window, min_periods=1).max()
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """添加 MACD 指标"""
    macd = MACD(close=df["close"], window_fast=fast, window_slow=slow, window_sign=signal)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """添加 RSI 指标"""
    df["rsi"] = RSIIndicator(close=df["close"], window=window).rsi()
    return df


def add_bollinger(df: pd.DataFrame, window: int = 20, std: int = 2) -> pd.DataFrame:
    """添加布林带"""
    bb = BollingerBands(close=df["close"], window=window, window_dev=std)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """添加 OBV 量能指标"""
    df["obv"] = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
    return df


def _rolling_linreg_endpoint(series: pd.Series, window: int) -> pd.Series:
    """Return the offset-zero endpoint of each rolling least-squares line."""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denominator = float(np.square(x - x_mean).sum())

    def endpoint(values: np.ndarray) -> float:
        y_mean = float(values.mean())
        slope = float(np.dot(x - x_mean, values - y_mean) / denominator)
        return y_mean + slope * (window - 1 - x_mean)

    return series.rolling(window=window, min_periods=window).apply(endpoint, raw=True)


def add_squeeze_momentum_lazybear(
    df: pd.DataFrame,
    *,
    bb_length: int = 20,
    bb_mult: float = 2.0,
    kc_length: int = 20,
    kc_mult: float = 1.5,
    use_true_range: bool = True,
) -> pd.DataFrame:
    """Add the causal Squeeze Momentum Indicator popularized by LazyBear.

    The implementation follows SQZMOM_LB: Bollinger Bands determine whether
    volatility is inside/outside the Keltner Channel, while momentum is the
    rolling linear-regression endpoint of detrended closing price.

    ``sqzmom_bar_state`` preserves the original four histogram states:
      2 = positive and rising, 1 = positive and falling,
     -2 = negative and falling, -1 = negative and rising.

    ``sqzmom_squeeze_state`` is 1 for squeeze-on, -1 for squeeze-off, and 0
    for the transition/neutral state.
    """
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"SQZMOM_LB 缺少必要列: {sorted(missing)}")
    if min(bb_length, kc_length) <= 1:
        raise ValueError("SQZMOM_LB 窗口必须大于 1")
    if bb_mult <= 0 or kc_mult <= 0:
        raise ValueError("SQZMOM_LB 倍数必须大于 0")

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    bb_basis = close.rolling(bb_length, min_periods=bb_length).mean()
    bb_dev = close.rolling(bb_length, min_periods=bb_length).std(ddof=0) * bb_mult
    upper_bb = bb_basis + bb_dev
    lower_bb = bb_basis - bb_dev

    kc_basis = close.rolling(kc_length, min_periods=kc_length).mean()
    if use_true_range:
        previous_close = close.shift(1)
        price_range = pd.concat(
            [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
            axis=1,
        ).max(axis=1)
    else:
        price_range = high - low
    range_mean = price_range.rolling(kc_length, min_periods=kc_length).mean()
    upper_kc = kc_basis + range_mean * kc_mult
    lower_kc = kc_basis - range_mean * kc_mult

    valid_bands = upper_bb.notna() & lower_bb.notna() & upper_kc.notna() & lower_kc.notna()
    squeeze_on = valid_bands & (lower_bb > lower_kc) & (upper_bb < upper_kc)
    squeeze_off = valid_bands & (lower_bb < lower_kc) & (upper_bb > upper_kc)
    no_squeeze = valid_bands & ~squeeze_on & ~squeeze_off

    highest_high = high.rolling(kc_length, min_periods=kc_length).max()
    lowest_low = low.rolling(kc_length, min_periods=kc_length).min()
    detrended = close - (0.25 * (highest_high + lowest_low) + 0.5 * kc_basis)
    momentum = _rolling_linreg_endpoint(detrended, kc_length)
    previous_momentum = momentum.shift(1).fillna(0.0)

    bar_state = pd.Series(pd.NA, index=df.index, dtype="Int8")
    valid_momentum = momentum.notna()
    bar_state.loc[valid_momentum & (momentum > 0) & (momentum > previous_momentum)] = 2
    bar_state.loc[valid_momentum & (momentum > 0) & (momentum <= previous_momentum)] = 1
    bar_state.loc[valid_momentum & (momentum <= 0) & (momentum < previous_momentum)] = -2
    bar_state.loc[valid_momentum & (momentum <= 0) & (momentum >= previous_momentum)] = -1

    squeeze_state = pd.Series(pd.NA, index=df.index, dtype="Int8")
    squeeze_state.loc[squeeze_on] = 1
    squeeze_state.loc[squeeze_off] = -1
    squeeze_state.loc[no_squeeze] = 0

    df["sqzmom_value"] = momentum
    df["sqzmom_squeeze_on"] = squeeze_on
    df["sqzmom_squeeze_off"] = squeeze_off
    df["sqzmom_no_squeeze"] = no_squeeze
    df["sqzmom_squeeze_state"] = squeeze_state
    df["sqzmom_bar_state"] = bar_state
    return df


def add_vegas_channel(df: pd.DataFrame) -> pd.DataFrame:
    """添加 Vegas 长期通道（EMA144 + EMA169）"""
    df["ema_144"] = EMAIndicator(close=df["close"], window=144).ema_indicator()
    df["ema_169"] = EMAIndicator(close=df["close"], window=169).ema_indicator()
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加全部常用技术指标（基础版）"""
    df = add_ma(df)
    df = add_macd(df)
    df = add_rsi(df)
    df = add_bollinger(df)
    df = add_obv(df)
    df = add_squeeze_momentum_lazybear(df)
    return df


def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加每日更新所需的全部指标：
      - 扩展 EMA（8/21/34/55/60/144/169/200/250）
      - 成交量均线（vol_ma_5/10/20/50）
      - 前高价格（prev_high_252d）
      - LazyBear Squeeze Momentum（SQZMOM_LB）

    这是 indicators_store 每日批量计算时调用的主入口。
    """
    df = add_ema_extended(df)
    df = add_volume_ma(df)
    df = add_prev_high(df)
    df = add_squeeze_momentum_lazybear(df)
    return df


# ─────────────────────── 周线层 ───────────────────────

# 周线前高回望窗口（交易周）
PREV_HIGH_WINDOW_WEEKLY = 52  # ≈1 年


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    将日线 OHLCV DataFrame 重采样为周线（按周五收盘聚合）。

    输入：含 open/high/low/close/volume 列，index 为 DatetimeIndex
    输出：同结构，每行代表一个完整的交易周
    规则：
      - open  = 周内第一根日线开盘
      - high  = 周内最高
      - low   = 周内最低
      - close = 周内最后一根日线收盘
      - volume = 周内成交量之和
      - 删除当前未收盘的不完整周（dropna on close）
    """
    agg_dict: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    # 只聚合存在的列（兼容不含 volume 的 DataFrame）
    exist = {k: v for k, v in agg_dict.items() if k in df.columns}
    weekly = df.resample("W-FRI").agg(exist)
    weekly = weekly.dropna(subset=["close"])
    return weekly


def add_weekly_indicators(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    在周线 OHLCV DataFrame 上计算与日线体系对应的全套指标。

    列名统一加 ``w_`` 前缀，避免与日线指标列冲突：
      - w_ema_8 / 21 / 34 / 55 / 60 / 144 / 169 / 200 / 250
      - w_vol_ma_5 / 10 / 20 / 50
      - w_prev_high_52w（近52周收盘最高价，≈1年）

    与日线 add_daily_indicators() 使用相同的 ewm(span, adjust=False) 计算逻辑，
    保证策略层在日线/周线之间切换时行为一致。
    """
    df = df_weekly.copy()
    close = df["close"].astype(float)

    # 扩展 EMA（与日线窗口一致）
    for w in _EMA_EXTENDED_WINDOWS:
        df[f"w_ema_{w}"] = close.ewm(span=w, adjust=False).mean()

    # 成交量均线（仅在含 volume 列时计算）
    if "volume" in df.columns:
        vol = df["volume"].astype(float)
        for w in _VOL_MA_WINDOWS:
            df[f"w_vol_ma_{w}"] = vol.rolling(window=w, min_periods=1).mean()

    # 周线前高
    df["w_prev_high_52w"] = close.rolling(window=PREV_HIGH_WINDOW_WEEKLY, min_periods=1).max()

    return df
