"""Trend-template filters shared by pattern strategies."""

from __future__ import annotations

import pandas as pd


def check_trend_template(df: pd.DataFrame) -> bool:
    """Minervini stage-2 style trend filter used by pattern strategies."""
    if len(df) < 200:
        return False

    curr = df.iloc[-1]
    close = curr["close"]
    ma_50 = df["close"].rolling(50).mean().iloc[-1]
    ma_150 = df["close"].rolling(150).mean().iloc[-1]
    ma_200 = df["close"].rolling(200).mean().iloc[-1]
    ma_200_prev = df["close"].rolling(200).mean().iloc[-20]

    lookback_52w = min(260, len(df))
    high_52w = df["high"].iloc[-lookback_52w:].max()
    low_52w = df["low"].iloc[-lookback_52w:].min()

    if not (close > ma_150 and ma_150 > ma_200):
        return False
    if ma_200 <= ma_200_prev:
        return False
    if ma_50 <= ma_150:
        return False
    if close < ma_200:
        return False
    if close < low_52w * 1.25:
        return False
    if close < high_52w * 0.75:
        return False
    return True
