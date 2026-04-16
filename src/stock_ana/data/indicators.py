"""
技术指标计算模块
使用 ta 库（纯 Python），确保 Windows/macOS 跨平台无障碍运行
"""

import pandas as pd
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

# Vegas 体系及扩展 EMA 窗口（用于 add_ema_extended）
_EMA_EXTENDED_WINDOWS = [8, 21, 34, 55, 60, 144, 169, 200, 250]

# 成交量均值窗口
_VOL_MA_WINDOWS = [5, 10, 50]

# 前高回望窗口（交易日）
PREV_HIGH_WINDOW = 252   # ≈1 年


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
    添加成交量移动平均：vol_ma_5, vol_ma_10, vol_ma_50。

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
    return df


def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加每日更新所需的全部指标：
      - 扩展 EMA（8/21/34/55/60/144/169/200/250）
      - 成交量均线（vol_ma_5/10/50）
      - 前高价格（prev_high_252d）

    这是 indicators_store 每日批量计算时调用的主入口。
    """
    df = add_ema_extended(df)
    df = add_volume_ma(df)
    df = add_prev_high(df)
    return df
