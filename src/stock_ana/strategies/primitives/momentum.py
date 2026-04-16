"""Reusable momentum-signal scoring primitives for strategy composition."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_volume_surge(
    recent: pd.DataFrame,
    ref: pd.DataFrame,
    *,
    price_up: bool,
    mild_ratio: float,
    strong_ratio: float,
) -> dict:
    """Score recent volume expansion versus a reference window."""
    avg_vol_recent = recent["volume"].mean()
    avg_vol_ref = ref["volume"].mean()
    vol_ratio = avg_vol_recent / avg_vol_ref if avg_vol_ref > 0 else 0.0

    score = 0.0
    if price_up:
        if vol_ratio >= strong_ratio:
            score = 2.0
        elif vol_ratio >= 2.0:
            score = 1.5
        elif vol_ratio >= mild_ratio:
            score = 1.0

    return {"ratio": round(vol_ratio, 2), "score": score}


def score_abnormal_return(
    df: pd.DataFrame,
    ref: pd.DataFrame,
    *,
    lookback: int,
    mild_z: float,
    strong_z: float,
) -> dict:
    """Score abnormal return strength using a volatility-adjusted Z-score."""
    ret = (df["close"].iloc[-1] / df["close"].iloc[-lookback - 1]) - 1
    daily_std = ref["close"].pct_change().dropna().std()
    expected_std = daily_std * np.sqrt(lookback) if daily_std > 0 else 1e-9
    z_score = ret / expected_std

    score = 0.0
    if z_score >= strong_z:
        score = 2.0
    elif z_score >= 2.0:
        score = 1.5
    elif z_score >= mild_z:
        score = 1.0

    return {
        "z_score": round(z_score, 2),
        "pct": round(ret * 100, 2),
        "score": score,
    }


def score_breakout(df: pd.DataFrame, *, lookback: int) -> dict:
    """Score fresh 20-day or 60-day closing breakouts."""
    n = len(df)
    cur_close = df["close"].iloc[-1]
    end = n - lookback
    start_20d = max(0, end - 20)
    start_60d = max(0, end - 60)
    high_20d = df["high"].iloc[start_20d:end].max()
    high_60d = df["high"].iloc[start_60d:end].max()

    level = ""
    score = 0.0
    if cur_close > high_60d:
        level, score = "60d_high", 2.0
    elif cur_close > high_20d:
        level, score = "20d_high", 1.0
    return {"level": level, "score": score}


def score_gap_up(
    df: pd.DataFrame,
    *,
    lookback: int,
    gap_thresh: float,
) -> dict:
    """Score the largest positive opening gap in the recent window."""
    n = len(df)
    max_gap = 0.0
    for i in range(n - lookback, n):
        prev_close = df["close"].iloc[i - 1]
        if prev_close > 0:
            gap = (df["open"].iloc[i] - prev_close) / prev_close
            if gap > max_gap:
                max_gap = gap

    score = min(1.0, max_gap / 0.04) if max_gap >= gap_thresh else 0.0
    return {"max_gap_pct": round(max_gap * 100, 2), "score": round(score, 2)}


def score_ma_breakout(df: pd.DataFrame, *, lookback: int) -> dict:
    """Score fresh breakouts above 50-day and 200-day moving averages."""
    n = len(df)
    cur_close = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-lookback - 1]

    score = 0.0
    above_50ma = False
    above_200ma = False

    if n >= 55:
        ma50 = df["close"].iloc[-50:].mean()
        above_50ma = cur_close > ma50
        if above_50ma and prev_close < ma50:
            score += 0.5

    if n >= 205:
        ma200 = df["close"].iloc[-200:].mean()
        above_200ma = cur_close > ma200
        if above_200ma and prev_close < ma200:
            score += 0.5

    return {
        "above_50ma": above_50ma,
        "above_200ma": above_200ma,
        "score": score,
    }


def score_accumulation(
    df: pd.DataFrame,
    ref: pd.DataFrame,
    *,
    lookback: int,
    price_up: bool,
    accum_vol_factor: float,
) -> dict:
    """Score repeated volume-backed up days inside the recent window."""
    n = len(df)
    avg_vol_base = ref["volume"].mean()
    accum_days = 0

    for i in range(n - lookback, n):
        row = df.iloc[i]
        if row["close"] > row["open"] and row["volume"] > avg_vol_base * accum_vol_factor:
            accum_days += 1

    score = 0.0
    if price_up:
        if accum_days >= lookback:
            score = 2.0
        elif accum_days >= 3:
            score = 1.0
        elif accum_days >= 2:
            score = 0.5

    return {"days": accum_days, "score": score}
