"""
技术指标计算模块
使用 ta 库（纯 Python），确保 Windows/macOS 跨平台无障碍运行
"""

import pandas as pd
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator


def add_ma(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """添加移动平均线（SMA + EMA）"""
    if windows is None:
        windows = [5, 10, 20, 60]

    for w in windows:
        df[f"sma_{w}"] = SMAIndicator(close=df["close"], window=w).sma_indicator()
        df[f"ema_{w}"] = EMAIndicator(close=df["close"], window=w).ema_indicator()
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
    """添加全部常用技术指标"""
    df = add_ma(df)
    df = add_macd(df)
    df = add_rsi(df)
    df = add_bollinger(df)
    df = add_obv(df)
    return df
