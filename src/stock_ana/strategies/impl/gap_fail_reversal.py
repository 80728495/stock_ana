"""Gap-up failure bearish reversal candle.

The pattern targets a single-bar failure after an optimistic gap-up open:

* open gaps above the prior high and prior close;
* the same candle closes as a large bearish body;
* the close gives back most of the open-vs-prior-close gap;
* the bar is near a recent high area.

It is intentionally narrower than generic bearish reversal candles.
"""

from __future__ import annotations

import pandas as pd

NEW_HIGH_WINDOW: int = 20
NEAR_HIGH_PCT: float = 3.0
MIN_GAP_OPEN_PCT: float = 5.0
MIN_TRUE_GAP_PCT: float = 0.2
# 低覆盖、高精度定位：只抓「大幅高开 + 一根大实体阴线砸到当日最低」。
# 缺口回补率与前期涨幅是错误代理（MSFT 2025-07-31 只回补 55%、TEM 2026-01-12 前期跌 17%），
# 改由「实体相对 20 日均实体」与「收盘贴近当日最低」直接刻画大阴线的决绝程度。
MIN_GAP_FILL_RATIO: float = 0.50      # 至少回补一半缺口（原 0.80 误伤 MSFT 这类未回补但大幅下砸的）
MIN_BODY_PCT: float = 2.0
MIN_BODY_RATIO: float = 0.55
MIN_BODY_VS_AVG20: float = 3.50       # 核心精度杠杆：实体须远大于近期常态（MSFT 8.4x / TEM 4.1x）
MAX_CLOSE_POSITION_PCT: float = 20.0  # 收盘落在当日振幅最低 20% 内（MSFT 6.8% / TEM 11.7%）
VOL_SPIKE_RATIO: float = 1.3
PRIOR_RISE_LOOKBACK: int = 60
MIN_PRIOR_RISE_PCT: float = 0.0       # 停用前期涨幅门槛：gap-fail 衰竭本身即信号，TEM 为超跌反抽失败
COOLDOWN_DAYS: int = 5


def _reset_index_with_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_index().reset_index()
    first_col = out.columns[0]
    if first_col != "date":
        if "date" in out.columns:
            out = out.drop(columns=[first_col])
        else:
            out = out.rename(columns={first_col: "date"})
    return out


def _apply_cooldown(rows: list[dict], cooldown_days: int) -> list[dict]:
    if cooldown_days <= 0 or len(rows) <= 1:
        return rows

    out: list[dict] = []
    cluster: list[dict] = []
    cluster_end = -1

    def flush() -> None:
        if not cluster:
            return
        out.append(max(cluster, key=lambda row: (int(row.get("score", 0)), float(row.get("gap_fill_ratio", 0.0)))))

    for row in sorted(rows, key=lambda item: int(item.get("_signal_pos", 0))):
        pos = int(row.get("_signal_pos", 0))
        if not cluster or pos <= cluster_end:
            cluster.append(row)
            cluster_end = max(cluster_end, pos + cooldown_days)
            continue
        flush()
        cluster = [row]
        cluster_end = pos + cooldown_days

    flush()
    return sorted(out, key=lambda item: int(item.get("_signal_pos", 0)))


def _ensure_volume_ma(df: pd.DataFrame) -> pd.DataFrame:
    if "vol_ma_50" not in df.columns:
        df["vol_ma_50"] = df["volume"].astype(float).rolling(50, min_periods=1).mean()
    return df


def detect_gap_fail_reversal(
    df: pd.DataFrame,
    *,
    new_high_window: int = NEW_HIGH_WINDOW,
    near_high_pct: float = NEAR_HIGH_PCT,
    min_gap_open_pct: float = MIN_GAP_OPEN_PCT,
    min_true_gap_pct: float = MIN_TRUE_GAP_PCT,
    min_gap_fill_ratio: float = MIN_GAP_FILL_RATIO,
    min_body_pct: float = MIN_BODY_PCT,
    min_body_ratio: float = MIN_BODY_RATIO,
    min_body_vs_avg20: float = MIN_BODY_VS_AVG20,
    max_close_position_pct: float = MAX_CLOSE_POSITION_PCT,
    prior_rise_lookback: int = PRIOR_RISE_LOOKBACK,
    min_prior_rise_pct: float = MIN_PRIOR_RISE_PCT,
) -> dict:
    """Detect whether the latest candle is a gap-up failure reversal."""

    required_len = max(new_high_window + 1, 25, prior_rise_lookback + 1 if min_prior_rise_pct > 0 else 0)
    if len(df) < required_len:
        return {"triggered": False, "score": 0, "reason": "insufficient_history"}

    x = df.copy()
    x.columns = [str(col).lower() for col in x.columns]
    x.index = pd.to_datetime(x.index)
    x = x.sort_index()
    x = _ensure_volume_ma(x)

    cur = x.iloc[-1]
    prev = x.iloc[-2]
    signal_date = str(x.index[-1].date())

    open_ = float(cur["open"])
    high = float(cur["high"])
    low = float(cur["low"])
    close = float(cur["close"])
    prev_close = float(prev["close"])
    prev_high = float(prev["high"])
    prev_low = float(prev["low"])

    if min(open_, high, low, close, prev_close, prev_high) <= 0:
        return {"triggered": False, "score": 0, "signal_date": signal_date, "reason": "invalid_price"}

    gap_open_pct = (open_ / prev_close - 1) * 100
    true_gap_pct = (open_ / prev_high - 1) * 100
    if gap_open_pct < min_gap_open_pct or true_gap_pct < min_true_gap_pct:
        return {
            "triggered": False,
            "score": 0,
            "signal_date": signal_date,
            "gap_open_pct": round(gap_open_pct, 2),
            "true_gap_pct": round(true_gap_pct, 2),
            "reason": "not_gap_up",
        }

    if close >= open_:
        return {"triggered": False, "score": 0, "signal_date": signal_date, "reason": "not_bearish"}

    candle_range = max(high - low, 1e-9)
    body = open_ - close
    body_pct = body / open_ * 100
    body_ratio = body / candle_range
    close_position_pct = (close - low) / candle_range * 100
    if body_pct < min_body_pct or body_ratio < min_body_ratio or close_position_pct > max_close_position_pct:
        return {
            "triggered": False,
            "score": 0,
            "signal_date": signal_date,
            "body_pct": round(body_pct, 2),
            "body_ratio": round(body_ratio, 3),
            "close_position_pct": round(close_position_pct, 2),
            "reason": "bear_body_not_strong",
        }

    gap_denom = max(open_ - prev_close, 1e-9)
    gap_fill_ratio = (open_ - close) / gap_denom
    true_gap_fill_ratio = (open_ - close) / max(open_ - prev_high, 1e-9)
    effective_gap_fill_ratio = max(gap_fill_ratio, true_gap_fill_ratio)
    if effective_gap_fill_ratio < min_gap_fill_ratio:
        return {
            "triggered": False,
            "score": 0,
            "signal_date": signal_date,
            "gap_fill_ratio": round(gap_fill_ratio, 3),
            "true_gap_fill_ratio": round(true_gap_fill_ratio, 3),
            "reason": "gap_not_filled_enough",
        }

    avg_body20 = (x["close"].astype(float) - x["open"].astype(float)).abs().iloc[-21:-1].mean()
    body_vs_avg20 = body / avg_body20 if avg_body20 and avg_body20 > 0 else 0.0
    if body_vs_avg20 < min_body_vs_avg20:
        return {
            "triggered": False,
            "score": 0,
            "signal_date": signal_date,
            "body_vs_avg20": round(body_vs_avg20, 2),
            "reason": "body_not_large_vs_recent",
        }

    recent_high = float(x["high"].astype(float).iloc[-new_high_window - 1:-1].max())
    near_high = high >= recent_high * (1 - near_high_pct / 100)
    if not near_high:
        return {"triggered": False, "score": 0, "signal_date": signal_date, "reason": "not_near_recent_high"}

    prior_rise_pct = 0.0
    if min_prior_rise_pct > 0:
        base = float(x["close"].iloc[-1 - prior_rise_lookback])
        prior_rise_pct = (high / base - 1) * 100 if base > 0 else 0.0
        if prior_rise_pct < min_prior_rise_pct:
            return {
                "triggered": False,
                "score": 0,
                "signal_date": signal_date,
                "prior_rise_pct": round(prior_rise_pct, 2),
                "reason": "insufficient_prior_rise",
            }

    vol_ma50 = float(cur["vol_ma_50"]) if float(cur["vol_ma_50"]) > 0 else 1.0
    vol_ratio = float(cur["volume"]) / vol_ma50

    score = 1
    score += int(effective_gap_fill_ratio >= 1.0)
    score += int(close <= prev_high)
    score += int(close <= prev_close)
    score += int(close <= prev_low)
    score += int(body_vs_avg20 >= 1.8)
    score += int(vol_ratio >= VOL_SPIKE_RATIO)

    return {
        "triggered": True,
        "signal_date": signal_date,
        "confirm_date": signal_date,
        "confirm_mode": "same_day_gap_fail",
        "gap_open_pct": round(gap_open_pct, 2),
        "true_gap_pct": round(true_gap_pct, 2),
        "gap_fill_ratio": round(gap_fill_ratio, 3),
        "true_gap_fill_ratio": round(true_gap_fill_ratio, 3),
        "effective_gap_fill_ratio": round(effective_gap_fill_ratio, 3),
        "open_to_close_drop_pct": round((close / open_ - 1) * 100, 2),
        "body_pct": round(body_pct, 2),
        "body_ratio": round(body_ratio, 3),
        "body_vs_avg20": round(body_vs_avg20, 2),
        "close_position_pct": round(close_position_pct, 2),
        "vol_ratio": round(vol_ratio, 2),
        "prior_rise_pct": round(prior_rise_pct, 2),
        "top_is_20d_high": bool(high >= recent_high),
        "close_below_prev_high": bool(close <= prev_high),
        "close_below_prev_close": bool(close <= prev_close),
        "close_below_prev_low": bool(close <= prev_low),
        "day1_high": round(high, 4),
        "day1_close": round(close, 4),
        "score": score,
        "reason": "gap_up_failed_large_bear_candle",
    }


def scan_history(
    df: pd.DataFrame,
    *,
    new_high_window: int = NEW_HIGH_WINDOW,
    near_high_pct: float = NEAR_HIGH_PCT,
    min_gap_open_pct: float = MIN_GAP_OPEN_PCT,
    min_true_gap_pct: float = MIN_TRUE_GAP_PCT,
    min_gap_fill_ratio: float = MIN_GAP_FILL_RATIO,
    min_body_pct: float = MIN_BODY_PCT,
    min_body_ratio: float = MIN_BODY_RATIO,
    min_body_vs_avg20: float = MIN_BODY_VS_AVG20,
    max_close_position_pct: float = MAX_CLOSE_POSITION_PCT,
    prior_rise_lookback: int = PRIOR_RISE_LOOKBACK,
    min_prior_rise_pct: float = MIN_PRIOR_RISE_PCT,
    cooldown_days: int = COOLDOWN_DAYS,
    forward_days: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """Scan historical OHLCV for gap-up failure reversal candles."""

    min_len = max(new_high_window + 1, 25, prior_rise_lookback + 1 if min_prior_rise_pct > 0 else 0)
    if len(df) < min_len:
        return pd.DataFrame()

    x = df.copy()
    x.columns = [str(col).lower() for col in x.columns]
    x.index = pd.to_datetime(x.index)
    x = _reset_index_with_date(x)
    x = _ensure_volume_ma(x)

    o = x["open"].astype(float)
    h = x["high"].astype(float)
    low = x["low"].astype(float)
    c = x["close"].astype(float)
    v = x["volume"].astype(float)
    vm50 = x["vol_ma_50"].astype(float)
    prev_close = c.shift(1)
    prev_high = h.shift(1)
    prev_low = low.shift(1)

    gap_open_pct = (o / prev_close.replace(0, 1e-9) - 1) * 100
    true_gap_pct = (o / prev_high.replace(0, 1e-9) - 1) * 100
    gap_up = (gap_open_pct >= min_gap_open_pct) & (true_gap_pct >= min_true_gap_pct)

    candle_range = (h - low).replace(0, 1e-9)
    body = o - c
    body_pct = body / o.replace(0, 1e-9) * 100
    body_ratio = body / candle_range
    close_position_pct = (c - low) / candle_range * 100
    bearish_body = (
        (body > 0)
        & (body_pct >= min_body_pct)
        & (body_ratio >= min_body_ratio)
        & (close_position_pct <= max_close_position_pct)
    )

    avg_body20 = (c - o).abs().rolling(20, min_periods=5).mean().shift(1)
    body_vs_avg20 = body / avg_body20.replace(0, 1e-9)
    large_body = body_vs_avg20 >= min_body_vs_avg20

    gap_fill_ratio = (o - c) / (o - prev_close).replace(0, 1e-9)
    true_gap_fill_ratio = (o - c) / (o - prev_high).replace(0, 1e-9)
    effective_gap_fill_ratio = pd.concat([gap_fill_ratio, true_gap_fill_ratio], axis=1).max(axis=1)
    gap_filled = effective_gap_fill_ratio >= min_gap_fill_ratio

    recent_high = h.shift(1).rolling(new_high_window, min_periods=5).max()
    near_high = h >= recent_high * (1 - near_high_pct / 100)

    if min_prior_rise_pct > 0:
        prior_base = c.shift(prior_rise_lookback)
        prior_rise_pct = (h / prior_base.replace(0, 1e-9) - 1) * 100
        prior_ok = prior_rise_pct >= min_prior_rise_pct
    else:
        prior_rise_pct = pd.Series(0.0, index=x.index)
        prior_ok = pd.Series(True, index=x.index)

    triggered = (gap_up & bearish_body & large_body & gap_filled & near_high & prior_ok).fillna(False)
    vol_ratio = v / vm50.replace(0, 1e-9)
    rows: list[dict] = []
    n = len(x)
    for pos in x.index[triggered]:
        signal_pos = int(pos)
        base = float(c.iloc[signal_pos])
        score = 1
        score += int(float(effective_gap_fill_ratio.iloc[signal_pos]) >= 1.0)
        score += int(float(c.iloc[signal_pos]) <= float(prev_high.iloc[signal_pos]))
        score += int(float(c.iloc[signal_pos]) <= float(prev_close.iloc[signal_pos]))
        score += int(float(c.iloc[signal_pos]) <= float(prev_low.iloc[signal_pos]))
        score += int(float(body_vs_avg20.iloc[signal_pos]) >= 1.8)
        score += int(float(vol_ratio.iloc[signal_pos]) >= VOL_SPIKE_RATIO)

        row: dict[str, object] = {
            "signal_date": str(x["date"].iloc[signal_pos])[:10],
            "confirm_date": str(x["date"].iloc[signal_pos])[:10],
            "confirm_mode": "same_day_gap_fail",
            "gap_open_pct": round(float(gap_open_pct.iloc[signal_pos]), 2),
            "true_gap_pct": round(float(true_gap_pct.iloc[signal_pos]), 2),
            "gap_fill_ratio": round(float(gap_fill_ratio.iloc[signal_pos]), 3),
            "true_gap_fill_ratio": round(float(true_gap_fill_ratio.iloc[signal_pos]), 3),
            "effective_gap_fill_ratio": round(float(effective_gap_fill_ratio.iloc[signal_pos]), 3),
            "open_to_close_drop_pct": round((float(c.iloc[signal_pos]) / float(o.iloc[signal_pos]) - 1) * 100, 2),
            "body_pct": round(float(body_pct.iloc[signal_pos]), 2),
            "body_ratio": round(float(body_ratio.iloc[signal_pos]), 3),
            "body_vs_avg20": round(float(body_vs_avg20.iloc[signal_pos]), 2),
            "close_position_pct": round(float(close_position_pct.iloc[signal_pos]), 2),
            "vol_ratio": round(float(vol_ratio.iloc[signal_pos]), 2),
            "prior_rise_pct": round(float(prior_rise_pct.iloc[signal_pos]), 2),
            "top_is_20d_high": bool(float(h.iloc[signal_pos]) >= float(recent_high.iloc[signal_pos])),
            "close_below_prev_high": bool(float(c.iloc[signal_pos]) <= float(prev_high.iloc[signal_pos])),
            "close_below_prev_close": bool(float(c.iloc[signal_pos]) <= float(prev_close.iloc[signal_pos])),
            "close_below_prev_low": bool(float(c.iloc[signal_pos]) <= float(prev_low.iloc[signal_pos])),
            "day1_high": round(float(h.iloc[signal_pos]), 4),
            "day1_close": round(float(c.iloc[signal_pos]), 4),
            "score": score,
            "_signal_pos": signal_pos,
        }

        for fd in forward_days:
            end_pos = min(signal_pos + fd, n - 1)
            fwd_c = float(c.iloc[end_pos])
            fwd_min = float(low.iloc[signal_pos + 1:end_pos + 1].min()) if signal_pos + 1 <= end_pos else base
            row[f"fwd_ret_{fd}d"] = round((fwd_c - base) / base * 100, 2)
            row[f"fwd_min_{fd}d"] = round((fwd_min - base) / base * 100, 2)

        rows.append(row)

    rows = _apply_cooldown(rows, cooldown_days)
    for row in rows:
        row.pop("_signal_pos", None)

    return pd.DataFrame(rows)
