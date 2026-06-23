"""SMC structure features for top-reversal candidates.

The research layer keeps an explicit event timeline. Some SMC structures are
drawn back to their formation candle on charts, but model features must only
use events whose ``detected_pos`` is already visible at the candidate's as-of
bar.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from stock_ana.research.top_reversal.feature_registry import SMC_STRUCTURE_FEATURES
from stock_ana.strategies.impl.smc import ob_quality_rating, ob_quality_score

INTERNAL_SWING_LENGTHS = (1, 2, 3)
SWING_STRUCTURE_LENGTHS = (5,)


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out.sort_index()


def _empty_features() -> dict[str, float]:
    values = {col: 0.0 for col in SMC_STRUCTURE_FEATURES}
    for col in (
        "smc_live_last_bull_ob_age",
        "smc_live_nearest_bull_ob_dist_pct",
        "smc_raw_bear_detect_lag",
        "smc_raw_bear_displacement_atr",
        "smc_raw_bear_zone_width_atr",
        "smc_raw_bear_volume_ratio",
        "smc_diag_bear_ob_confirm_delay",
    ):
        values[col] = np.nan
    return values


def _atr14(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=5).mean().bfill().fillna(high - low).replace(0, np.nan)


def _causal_swing_points(df: pd.DataFrame, swing_length: int) -> pd.DataFrame:
    """Return raw confirmed swing points with the bar when each point is visible."""

    if len(df) < swing_length * 2 + 1:
        return pd.DataFrame(columns=["pos", "visible_pos", "direction", "level"])

    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    points: list[dict[str, float | int]] = []
    for pos in range(swing_length, len(df) - swing_length):
        start = pos - swing_length
        end = pos + swing_length + 1
        is_high = high[pos] >= float(np.nanmax(high[start:end]))
        is_low = low[pos] <= float(np.nanmin(low[start:end]))
        if is_high and not is_low:
            points.append({
                "pos": int(pos),
                "visible_pos": int(pos + swing_length),
                "direction": 1,
                "level": float(high[pos]),
            })
        elif is_low and not is_high:
            points.append({
                "pos": int(pos),
                "visible_pos": int(pos + swing_length),
                "direction": -1,
                "level": float(low[pos]),
            })

    return pd.DataFrame(points, columns=["pos", "visible_pos", "direction", "level"])


def _empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_type",
            "direction",
            "scale",
            "swing_length",
            "origin_pos",
            "detected_pos",
            "broken_pos",
            "mitigated_pos",
            "swept_pos",
            "zone_top",
            "zone_bottom",
            "level",
            "score",
        ]
    )


def _empty_ob_event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "direction",
            "origin_pos",
            "origin_date",
            "confirmed_pos",
            "confirmed_date",
            "confirm_delay_bars",
            "top",
            "bottom",
            "ob_volume",
            "percentage",
            "mitigated_pos",
            "cleared_pos",
            "score",
            "struct_confirm_score",
            "struct_score",
            "confirm_has_structure",
        ]
    )


def _empty_raw_ob_event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "raw_id",
            "direction",
            "origin_pos",
            "formed_pos",
            "detected_pos",
            "origin_date",
            "formed_date",
            "detected_date",
            "top",
            "bottom",
            "zone_mid",
            "score",
            "raw_score",
            "score_detail",
            "raw_reason",
            "near_high_pct",
            "near_low_pct",
            "near_extreme_pct",
            "prior_ret_20d",
            "prior_ret_60d",
            "zone_width_atr",
            "volume_ratio",
            "body_atr",
            "upper_wick_pct",
            "lower_wick_pct",
            "close_location",
            "early_pos",
            "early_date",
            "early_first_pos",
            "early_first_date",
            "early_score",
            "early_struct_score",
            "early_raw_score",
            "early_total_score",
            "early_confirm_lag_bars",
            "early_has_fvg",
            "early_has_sweep",
            "early_choch_down",
            "early_bos_down",
            "early_micro_low_break",
            "early_top_low_break",
            "early_bull_ob_mitigated",
            "early_low_break_pct",
            "confirmed_pos",
            "confirmed_date",
            "confirm_delay_bars",
            "confirmed_score",
            "confirmed_struct_score",
            "confirmed_has_structure",
            "confirmed_zone_top",
            "confirmed_zone_bottom",
            "confirmed_ob_volume",
            "confirmed_percentage",
            "mitigated_pos",
            "cleared_pos",
        ]
    )


def _build_fvg_events(df: pd.DataFrame) -> pd.DataFrame:
    """Build causal FVG events.

    A three-candle FVG at middle bar ``i`` is only visible after bar ``i + 1``.
    """

    if len(df) < 3:
        return _empty_event_frame()

    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    open_arr = df["open"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    atr = _atr14(df).to_numpy()
    rows: list[dict[str, float | int | str]] = []
    for pos in range(1, len(df) - 1):
        direction = 0
        top = np.nan
        bottom = np.nan
        if high[pos - 1] < low[pos + 1] and close[pos] > open_arr[pos]:
            direction = 1
            top = low[pos + 1]
            bottom = high[pos - 1]
        elif low[pos - 1] > high[pos + 1] and close[pos] < open_arr[pos]:
            direction = -1
            top = low[pos - 1]
            bottom = high[pos + 1]
        if direction == 0:
            continue

        mitigated_pos = np.nan
        for j in range(pos + 2, len(df)):
            if (direction == 1 and low[j] <= top) or (direction == -1 and high[j] >= bottom):
                mitigated_pos = int(j)
                break
        width = abs(float(top) - float(bottom))
        atr_val = float(atr[pos]) if np.isfinite(atr[pos]) and atr[pos] > 0 else np.nan
        body_atr = abs(float(close[pos] - open_arr[pos])) / atr_val if np.isfinite(atr_val) else 0.0
        width_atr = width / atr_val if np.isfinite(atr_val) else 0.0
        score = min(100.0, 45.0 * min(body_atr, 2.0) / 2.0 + 55.0 * min(width_atr, 2.5) / 2.5)
        rows.append({
            "event_type": "fvg",
            "direction": int(direction),
            "scale": "candle",
            "swing_length": 0,
            "origin_pos": int(pos),
            "detected_pos": int(pos + 1),
            "broken_pos": np.nan,
            "mitigated_pos": mitigated_pos,
            "swept_pos": np.nan,
            "zone_top": float(max(top, bottom)),
            "zone_bottom": float(min(top, bottom)),
            "level": float((top + bottom) / 2),
            "score": round(float(score), 2),
        })
    return pd.DataFrame(rows) if rows else _empty_event_frame()


def _build_structure_break_events(
    df: pd.DataFrame,
    swings: pd.DataFrame,
    *,
    swing_length: int,
    scale: str,
    close_break: bool = True,
) -> pd.DataFrame:
    """Build BOS/CHoCH events with detected_pos set to first causal visibility."""

    if swings.empty or len(swings) < 4:
        return _empty_event_frame()

    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    rows: list[dict[str, float | int | str]] = []
    levels: list[float] = []
    directions: list[int] = []
    positions: list[int] = []
    visible_positions: list[int] = []

    def add_event(event_type: str, direction: int, signal_pos: int, current_visible: int, level: float) -> None:
        broken_pos = np.nan
        start = min(len(df), signal_pos + 2)
        if direction == 1:
            arr = close if close_break else high
            for j in range(start, len(df)):
                if arr[j] > level:
                    broken_pos = int(j)
                    break
        else:
            arr = close if close_break else low
            for j in range(start, len(df)):
                if arr[j] < level:
                    broken_pos = int(j)
                    break
        if pd.isna(broken_pos):
            return
        detected_pos = int(max(int(broken_pos), current_visible))
        rows.append({
            "event_type": event_type,
            "direction": int(direction),
            "scale": scale,
            "swing_length": int(swing_length),
            "origin_pos": int(signal_pos),
            "detected_pos": detected_pos,
            "broken_pos": int(broken_pos),
            "mitigated_pos": np.nan,
            "swept_pos": np.nan,
            "zone_top": np.nan,
            "zone_bottom": np.nan,
            "level": float(level),
            "score": 70.0 if event_type == "choch" else 55.0,
        })

    for _, swing in swings.iterrows():
        direction = int(swing["direction"])
        level = float(swing["level"])
        pos = int(swing["pos"])
        visible_pos = int(swing["visible_pos"])
        if directions and directions[-1] == direction:
            replace = (direction == 1 and level >= levels[-1]) or (direction == -1 and level <= levels[-1])
            if not replace:
                continue
            positions[-1] = pos
            visible_positions[-1] = visible_pos
            levels[-1] = level
        else:
            positions.append(pos)
            visible_positions.append(visible_pos)
            directions.append(direction)
            levels.append(level)
        if len(levels) < 4:
            continue

        last_dirs = np.array(directions[-4:], dtype=int)
        last_levels = np.array(levels[-4:], dtype=float)
        signal_pos = positions[-2]
        current_visible = visible_positions[-1]

        if np.array_equal(last_dirs, np.array([-1, 1, -1, 1])):
            if last_levels[0] < last_levels[2] < last_levels[1] < last_levels[3]:
                add_event("bos", 1, signal_pos, current_visible, float(last_levels[1]))
            if last_levels[3] > last_levels[1] > last_levels[0] > last_levels[2]:
                add_event("choch", 1, signal_pos, current_visible, float(last_levels[1]))
        elif np.array_equal(last_dirs, np.array([1, -1, 1, -1])):
            if last_levels[0] > last_levels[2] > last_levels[1] > last_levels[3]:
                add_event("bos", -1, signal_pos, current_visible, float(last_levels[1]))
            if last_levels[3] < last_levels[1] < last_levels[0] < last_levels[2]:
                add_event("choch", -1, signal_pos, current_visible, float(last_levels[1]))

    if not rows:
        return _empty_event_frame()
    out = pd.DataFrame(rows)
    out = out.drop_duplicates(["event_type", "direction", "scale", "origin_pos", "detected_pos", "level"])
    return out.reset_index(drop=True)


def _build_liquidity_events(
    df: pd.DataFrame,
    swings: pd.DataFrame,
    *,
    swing_length: int,
    scale: str,
    range_percent: float = 0.01,
) -> pd.DataFrame:
    """Build equal-high/equal-low liquidity events and optional sweep timing."""

    if swings.empty:
        return _empty_event_frame()

    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    pip_range = (float(np.nanmax(high)) - float(np.nanmin(low))) * range_percent
    rows: list[dict[str, float | int | str]] = []

    for direction in (1, -1):
        same = swings[swings["direction"] == direction].copy().reset_index(drop=True)
        used: set[int] = set()
        for i, base in same.iterrows():
            if i in used:
                continue
            base_level = float(base["level"])
            range_low = base_level - pip_range
            range_high = base_level + pip_range
            group = [base]
            sweep_pos = np.nan
            start = int(base["pos"]) + 1
            if direction == 1:
                for j in range(start, len(df)):
                    if high[j] >= range_high:
                        sweep_pos = int(j)
                        break
            else:
                for j in range(start, len(df)):
                    if low[j] <= range_low:
                        sweep_pos = int(j)
                        break

            for j, other in same.iloc[i + 1:].iterrows():
                other_pos = int(other["pos"])
                if pd.notna(sweep_pos) and other_pos >= int(sweep_pos):
                    break
                other_level = float(other["level"])
                if range_low <= other_level <= range_high:
                    group.append(other)
                    used.add(j)

            if len(group) < 2:
                continue
            detected_pos = int(max(int(g["visible_pos"]) for g in group))
            if pd.notna(sweep_pos):
                detected_pos = int(max(detected_pos, int(sweep_pos)))
            level = float(np.mean([float(g["level"]) for g in group]))
            rows.append({
                "event_type": "liquidity",
                "direction": int(direction),
                "scale": scale,
                "swing_length": int(swing_length),
                "origin_pos": int(group[0]["pos"]),
                "detected_pos": detected_pos,
                "broken_pos": np.nan,
                "mitigated_pos": np.nan,
                "swept_pos": sweep_pos,
                "zone_top": float(range_high),
                "zone_bottom": float(range_low),
                "level": level,
                "score": 65.0 if pd.notna(sweep_pos) else 40.0,
            })

    return pd.DataFrame(rows) if rows else _empty_event_frame()


def _build_structure_events(df: pd.DataFrame) -> pd.DataFrame:
    parts = [_build_fvg_events(df)]
    for swing_length in (*INTERNAL_SWING_LENGTHS, *SWING_STRUCTURE_LENGTHS):
        scale = f"internal_{swing_length}" if swing_length in INTERNAL_SWING_LENGTHS else f"swing_{swing_length}"
        swings = _causal_swing_points(df, swing_length)
        parts.append(
            _build_structure_break_events(
                df,
                swings,
                swing_length=swing_length,
                scale=scale,
            )
        )
        parts.append(
            _build_liquidity_events(
                df,
                swings,
                swing_length=swing_length,
                scale=scale,
            )
        )
    non_empty_parts = [p for p in parts if not p.empty]
    out = pd.concat(non_empty_parts, ignore_index=True) if non_empty_parts else _empty_event_frame()
    if out.empty:
        return _empty_event_frame()
    numeric_cols = [
        "direction", "swing_length", "origin_pos", "detected_pos", "broken_pos",
        "mitigated_pos", "swept_pos", "zone_top", "zone_bottom", "level", "score",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(["detected_pos", "origin_pos", "event_type", "scale"]).reset_index(drop=True)


def _set_mitigated(events: dict[tuple[int, int], dict], idx: int, direction: int, pos: int) -> None:
    event = events.get((idx, direction))
    if event is not None and pd.isna(event.get("mitigated_pos", np.nan)):
        event["mitigated_pos"] = int(pos)


def _set_cleared(events: dict[tuple[int, int], dict], idx: int, direction: int, pos: int) -> None:
    event = events.get((idx, direction))
    if event is not None and pd.isna(event.get("cleared_pos", np.nan)):
        event["cleared_pos"] = int(pos)


def _build_ob_events(
    df: pd.DataFrame,
    *,
    swing_length: int = 5,
    close_mitigation: bool = False,
) -> pd.DataFrame:
    """Build historical OB events with both origin and confirmation positions."""

    if len(df) < swing_length * 2 + 20:
        return _empty_ob_event_frame()

    ohlc_len = len(df)
    open_arr = df["open"].astype(float).values
    high_arr = df["high"].astype(float).values
    low_arr = df["low"].astype(float).values
    close_arr = df["close"].astype(float).values
    volume_arr = df["volume"].astype(float).values

    crossed = np.full(ohlc_len, False, dtype=bool)
    ob = np.zeros(ohlc_len, dtype=np.int32)
    top_arr = np.zeros(ohlc_len, dtype=np.float64)
    bottom_arr = np.zeros(ohlc_len, dtype=np.float64)
    ob_volume = np.zeros(ohlc_len, dtype=np.float64)
    low_volume = np.zeros(ohlc_len, dtype=np.float64)
    high_volume = np.zeros(ohlc_len, dtype=np.float64)
    percentage = np.zeros(ohlc_len, dtype=np.float64)
    mitigated_index = np.zeros(ohlc_len, dtype=np.int32)
    breaker = np.full(ohlc_len, False, dtype=bool)
    events: dict[tuple[int, int], dict] = {}

    swings = _causal_swing_points(df, swing_length)
    if swings.empty:
        return _empty_ob_event_frame()
    all_swing_highs = swings[swings["direction"] == 1][["pos", "visible_pos"]].to_numpy(dtype=int)
    all_swing_lows = swings[swings["direction"] == -1][["pos", "visible_pos"]].to_numpy(dtype=int)

    def add_event(direction: int, origin_pos: int, confirm_pos: int, raw_top: float, raw_bottom: float) -> None:
        top = max(float(raw_top), float(raw_bottom))
        bottom = min(float(raw_top), float(raw_bottom))
        v0 = volume_arr[confirm_pos]
        v1 = volume_arr[confirm_pos - 1] if confirm_pos >= 1 else 0.0
        v2 = volume_arr[confirm_pos - 2] if confirm_pos >= 2 else 0.0
        hv = v0 + v1 if direction == 1 else v2
        lv = v2 if direction == 1 else v0 + v1
        mx = max(hv, lv)
        pct = min(hv, lv) / mx * 100.0 if mx != 0 else 100.0
        ob[origin_pos] = direction
        top_arr[origin_pos] = top
        bottom_arr[origin_pos] = bottom
        ob_volume[origin_pos] = v0 + v1 + v2
        low_volume[origin_pos] = lv
        high_volume[origin_pos] = hv
        percentage[origin_pos] = pct
        mitigated_index[origin_pos] = 0
        events[(origin_pos, direction)] = {
            "direction": direction,
            "origin_pos": int(origin_pos),
            "origin_date": pd.Timestamp(df.index[origin_pos]),
            "confirmed_pos": int(confirm_pos),
            "confirmed_date": pd.Timestamp(df.index[confirm_pos]),
            "confirm_delay_bars": int(confirm_pos - origin_pos),
            "top": top,
            "bottom": bottom,
            "ob_volume": float(v0 + v1 + v2),
            "percentage": float(pct),
            "mitigated_pos": np.nan,
            "cleared_pos": np.nan,
        }

    active_bullish: list[int] = []
    for i in range(ohlc_len):
        for idx in active_bullish.copy():
            if breaker[idx]:
                if high_arr[i] > top_arr[idx]:
                    _set_cleared(events, idx, 1, i)
                    ob[idx] = 0
                    top_arr[idx] = 0.0
                    bottom_arr[idx] = 0.0
                    ob_volume[idx] = 0.0
                    low_volume[idx] = 0.0
                    high_volume[idx] = 0.0
                    mitigated_index[idx] = 0
                    percentage[idx] = 0.0
                    active_bullish.remove(idx)
            elif (
                (not close_mitigation and low_arr[i] < bottom_arr[idx])
                or (close_mitigation and min(open_arr[i], close_arr[i]) < bottom_arr[idx])
            ):
                breaker[idx] = True
                mitigated_index[idx] = i - 1
                _set_mitigated(events, idx, 1, i - 1)

        visible_highs = all_swing_highs[all_swing_highs[:, 1] <= i, 0] if len(all_swing_highs) else np.array([], dtype=int)
        pos = np.searchsorted(visible_highs, i)
        last_top_index = visible_highs[pos - 1] if pos > 0 else None

        if last_top_index is not None and close_arr[i] > high_arr[last_top_index] and not crossed[last_top_index]:
            crossed[last_top_index] = True
            ob_btm = high_arr[i - 1]
            ob_top = low_arr[i - 1]
            ob_index = i - 1
            if i - last_top_index > 1:
                start = last_top_index + 1
                end = i
                if end > start:
                    segment = low_arr[start:end]
                    min_val = segment.min()
                    candidates = np.nonzero(segment == min_val)[0]
                    if candidates.size:
                        ci = start + candidates[-1]
                        ob_btm = low_arr[ci]
                        ob_top = high_arr[ci]
                        ob_index = ci
            add_event(1, int(ob_index), i, ob_top, ob_btm)
            active_bullish.append(int(ob_index))

    active_bearish: list[int] = []
    for i in range(ohlc_len):
        for idx in active_bearish.copy():
            if breaker[idx]:
                if low_arr[i] < bottom_arr[idx]:
                    _set_cleared(events, idx, -1, i)
                    ob[idx] = 0
                    top_arr[idx] = 0.0
                    bottom_arr[idx] = 0.0
                    ob_volume[idx] = 0.0
                    low_volume[idx] = 0.0
                    high_volume[idx] = 0.0
                    mitigated_index[idx] = 0
                    percentage[idx] = 0.0
                    active_bearish.remove(idx)
            elif (
                (not close_mitigation and high_arr[i] > top_arr[idx])
                or (close_mitigation and max(open_arr[i], close_arr[i]) > top_arr[idx])
            ):
                breaker[idx] = True
                mitigated_index[idx] = i
                _set_mitigated(events, idx, -1, i)

        visible_lows = all_swing_lows[all_swing_lows[:, 1] <= i, 0] if len(all_swing_lows) else np.array([], dtype=int)
        pos = np.searchsorted(visible_lows, i)
        last_btm_index = visible_lows[pos - 1] if pos > 0 else None

        if last_btm_index is not None and close_arr[i] < low_arr[last_btm_index] and not crossed[last_btm_index]:
            crossed[last_btm_index] = True
            ob_top = high_arr[i - 1]
            ob_btm = low_arr[i - 1]
            ob_index = i - 1
            if i - last_btm_index > 1:
                start = last_btm_index + 1
                end = i
                if end > start:
                    segment = high_arr[start:end]
                    max_val = segment.max()
                    candidates = np.nonzero(segment == max_val)[0]
                    if candidates.size:
                        ci = start + candidates[-1]
                        ob_top = high_arr[ci]
                        ob_btm = low_arr[ci]
                        ob_index = ci
            add_event(-1, int(ob_index), i, ob_top, ob_btm)
            active_bearish.append(int(ob_index))

    if not events:
        return _empty_ob_event_frame()

    ob_df = pd.DataFrame(
        {
            "OB": np.full(ohlc_len, np.nan),
            "Top": np.full(ohlc_len, np.nan),
            "Bottom": np.full(ohlc_len, np.nan),
            "OBVolume": np.full(ohlc_len, np.nan),
            "MitigatedIndex": np.full(ohlc_len, np.nan),
            "Percentage": np.full(ohlc_len, np.nan),
        }
    )
    for event in events.values():
        origin_pos = int(event["origin_pos"])
        ob_df.at[origin_pos, "OB"] = int(event["direction"])
        ob_df.at[origin_pos, "Top"] = float(event["top"])
        ob_df.at[origin_pos, "Bottom"] = float(event["bottom"])
        ob_df.at[origin_pos, "OBVolume"] = float(event["ob_volume"])
        ob_df.at[origin_pos, "MitigatedIndex"] = 0 if pd.isna(event.get("mitigated_pos", np.nan)) else int(event["mitigated_pos"])
        ob_df.at[origin_pos, "Percentage"] = float(event["percentage"])

    rows = []
    for event in events.values():
        origin_pos = int(event["origin_pos"])
        try:
            features = ob_quality_score(df, ob_df, origin_pos)
            score, detail = ob_quality_rating(df, ob_df, origin_pos, features=features)
        except Exception:
            features = {}
            score, detail = 0.0, {}
        rows.append({
            **event,
            "score": float(score),
            "score_detail": detail,
            **{f"feat_{key}": round(float(value), 4) for key, value in features.items() if key != "direction"},
        })
    return pd.DataFrame(rows).sort_values(["confirmed_pos", "origin_pos", "direction"]).reset_index(drop=True)


def _attach_ob_structure_scores(ob_events: pd.DataFrame, structure_events: pd.DataFrame) -> pd.DataFrame:
    """Add a structure-confirmed score to OB events without changing raw detection."""

    if ob_events.empty:
        return ob_events

    out = ob_events.copy()
    out["struct_confirm_score"] = 0.0
    out["struct_score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0)
    out["confirm_has_structure"] = 0.0
    if structure_events.empty:
        return out

    detected = pd.to_numeric(structure_events["detected_pos"], errors="coerce")
    origin = pd.to_numeric(structure_events["origin_pos"], errors="coerce")
    swept = pd.to_numeric(structure_events["swept_pos"], errors="coerce")
    for idx, event in out.iterrows():
        direction = int(event["direction"])
        origin_pos = int(event["origin_pos"])
        confirmed_pos = int(event["confirmed_pos"])
        start = max(0, origin_pos - 3)
        end = confirmed_pos + 1

        same = structure_events[
            (structure_events["direction"] == direction)
            & detected.notna()
            & (detected >= start)
            & (detected <= end)
        ]
        score = 0.0
        if not same[same["event_type"] == "choch"].empty:
            score += 25.0
        if not same[same["event_type"] == "bos"].empty:
            score += 15.0
        if not same[same["event_type"] == "fvg"].empty:
            score += 20.0

        if direction == -1:
            sweep = structure_events[
                (structure_events["event_type"] == "liquidity")
                & (structure_events["direction"] == 1)
                & swept.notna()
                & (swept >= start)
                & (swept <= end)
                & (detected <= end)
            ]
        else:
            sweep = structure_events[
                (structure_events["event_type"] == "liquidity")
                & (structure_events["direction"] == -1)
                & swept.notna()
                & (swept >= start)
                & (swept <= end)
                & (detected <= end)
            ]
        if not sweep.empty:
            score += 20.0

        same_origin = pd.to_numeric(same["origin_pos"], errors="coerce")
        local_origin = same[(same_origin >= origin_pos - 2) & (same_origin <= origin_pos + 2)]
        if not local_origin.empty:
            score += 10.0

        score = min(100.0, score)
        out.at[idx, "struct_confirm_score"] = round(float(score), 2)
        out.at[idx, "struct_score"] = round(float(event.get("score", 0.0)) + score, 2)
        out.at[idx, "confirm_has_structure"] = float(int(score > 0))

    return out


def _bounded_score(value: float, full_value: float, weight: float) -> float:
    if not np.isfinite(value) or full_value <= 0:
        return 0.0
    return float(min(weight, max(0.0, value) / full_value * weight))


def _raw_zone_row(
    df: pd.DataFrame,
    *,
    direction: int,
    pos: int,
    atr: pd.Series,
    vol20: pd.Series,
    rolling_high: pd.Series,
    rolling_low: pd.Series,
    ret20: pd.Series,
    ret60: pd.Series,
) -> dict[str, object]:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    zone_top = float(high.iloc[pos])
    zone_bottom = float(low.iloc[pos])
    zone_mid = (zone_top + zone_bottom) / 2.0
    atr_val = float(atr.iloc[pos]) if pd.notna(atr.iloc[pos]) and atr.iloc[pos] > 0 else np.nan
    width = max(0.0, zone_top - zone_bottom)
    zone_width_atr = width / atr_val if np.isfinite(atr_val) else np.nan
    vol_base = float(vol20.iloc[pos]) if pd.notna(vol20.iloc[pos]) and vol20.iloc[pos] > 0 else np.nan
    volume_ratio = float(volume.iloc[pos]) / vol_base if np.isfinite(vol_base) else np.nan

    body = abs(float(close.iloc[pos]) - float(open_.iloc[pos]))
    body_atr = body / atr_val if np.isfinite(atr_val) else np.nan
    upper_wick = max(0.0, zone_top - max(float(open_.iloc[pos]), float(close.iloc[pos])))
    lower_wick = max(0.0, min(float(open_.iloc[pos]), float(close.iloc[pos])) - zone_bottom)
    upper_wick_pct = upper_wick / width * 100.0 if width > 0 else 0.0
    lower_wick_pct = lower_wick / width * 100.0 if width > 0 else 0.0
    close_location = (float(close.iloc[pos]) - zone_bottom) / width if width > 0 else 0.5

    rh = float(rolling_high.iloc[pos]) if pd.notna(rolling_high.iloc[pos]) and rolling_high.iloc[pos] > 0 else np.nan
    rl = float(rolling_low.iloc[pos]) if pd.notna(rolling_low.iloc[pos]) and rolling_low.iloc[pos] > 0 else np.nan
    near_high_pct = (rh / zone_top - 1.0) * 100.0 if np.isfinite(rh) and zone_top > 0 else np.nan
    near_low_pct = (zone_bottom / rl - 1.0) * 100.0 if np.isfinite(rl) and rl > 0 else np.nan
    prior_ret_20d = float(ret20.iloc[pos]) if pd.notna(ret20.iloc[pos]) else np.nan
    prior_ret_60d = float(ret60.iloc[pos]) if pd.notna(ret60.iloc[pos]) else np.nan

    reasons: list[str] = []
    detail: dict[str, float] = {}
    if direction == -1:
        near_extreme_pct = near_high_pct
        near_score = _bounded_score(10.0 - float(near_high_pct), 10.0, 25.0) if pd.notna(near_high_pct) else 0.0
        trend_score = _bounded_score(float(prior_ret_20d), 20.0, 25.0) if pd.notna(prior_ret_20d) else 0.0
        trend60_score = _bounded_score(float(prior_ret_60d), 35.0, 10.0) if pd.notna(prior_ret_60d) else 0.0
        wick_score = min(15.0, upper_wick_pct / 45.0 * 15.0)
        close_score = max(0.0, (0.65 - close_location) / 0.65 * 10.0)
        if pd.notna(near_high_pct) and near_high_pct <= 3.0:
            reasons.append("near_high")
        if pd.notna(prior_ret_20d) and prior_ret_20d >= 5.0:
            reasons.append("prior_rise")
        if upper_wick_pct >= 35.0:
            reasons.append("upper_wick")
        if close_location <= 0.35:
            reasons.append("weak_close")
    else:
        near_extreme_pct = near_low_pct
        near_score = _bounded_score(10.0 - float(near_low_pct), 10.0, 25.0) if pd.notna(near_low_pct) else 0.0
        trend_score = _bounded_score(-float(prior_ret_20d), 20.0, 25.0) if pd.notna(prior_ret_20d) else 0.0
        trend60_score = _bounded_score(-float(prior_ret_60d), 35.0, 10.0) if pd.notna(prior_ret_60d) else 0.0
        wick_score = min(15.0, lower_wick_pct / 45.0 * 15.0)
        close_score = max(0.0, (close_location - 0.35) / 0.65 * 10.0)
        if pd.notna(near_low_pct) and near_low_pct <= 3.0:
            reasons.append("near_low")
        if pd.notna(prior_ret_20d) and prior_ret_20d <= -5.0:
            reasons.append("prior_drop")
        if lower_wick_pct >= 35.0:
            reasons.append("lower_wick")
        if close_location >= 0.65:
            reasons.append("strong_close")

    volume_score = _bounded_score(float(volume_ratio) - 0.8, 1.2, 15.0) if pd.notna(volume_ratio) else 0.0
    if pd.notna(zone_width_atr):
        width_score = max(0.0, 10.0 - abs(float(zone_width_atr) - 1.0) / 2.0 * 10.0)
    else:
        width_score = 0.0
    if pd.notna(volume_ratio) and volume_ratio >= 1.2:
        reasons.append("volume")

    score = min(100.0, near_score + trend_score + trend60_score + wick_score + close_score + volume_score + width_score)
    detail.update({
        "near_score": round(float(near_score), 2),
        "trend_score": round(float(trend_score), 2),
        "trend60_score": round(float(trend60_score), 2),
        "wick_score": round(float(wick_score), 2),
        "close_score": round(float(close_score), 2),
        "volume_score": round(float(volume_score), 2),
        "width_score": round(float(width_score), 2),
    })

    return {
        "raw_id": f"{'bear' if direction == -1 else 'bull'}_{pos}",
        "direction": int(direction),
        "origin_pos": int(pos),
        "formed_pos": int(pos),
        "detected_pos": int(pos),
        "origin_date": pd.Timestamp(df.index[pos]),
        "formed_date": pd.Timestamp(df.index[pos]),
        "detected_date": pd.Timestamp(df.index[pos]),
        "top": zone_top,
        "bottom": zone_bottom,
        "zone_mid": zone_mid,
        "score": round(float(score), 2),
        "raw_score": round(float(score), 2),
        "score_detail": detail,
        "raw_reason": "+".join(reasons) if reasons else "candle_zone",
        "near_high_pct": round(float(near_high_pct), 2) if pd.notna(near_high_pct) else np.nan,
        "near_low_pct": round(float(near_low_pct), 2) if pd.notna(near_low_pct) else np.nan,
        "near_extreme_pct": round(float(near_extreme_pct), 2) if pd.notna(near_extreme_pct) else np.nan,
        "prior_ret_20d": round(float(prior_ret_20d), 2) if pd.notna(prior_ret_20d) else np.nan,
        "prior_ret_60d": round(float(prior_ret_60d), 2) if pd.notna(prior_ret_60d) else np.nan,
        "zone_width_atr": round(float(zone_width_atr), 2) if pd.notna(zone_width_atr) else np.nan,
        "volume_ratio": round(float(volume_ratio), 2) if pd.notna(volume_ratio) else np.nan,
        "body_atr": round(float(body_atr), 2) if pd.notna(body_atr) else np.nan,
        "upper_wick_pct": round(float(upper_wick_pct), 2),
        "lower_wick_pct": round(float(lower_wick_pct), 2),
        "close_location": round(float(close_location), 4),
        "early_pos": np.nan,
        "early_date": pd.NaT,
        "early_first_pos": np.nan,
        "early_first_date": pd.NaT,
        "early_score": 0.0,
        "early_struct_score": 0.0,
        "early_raw_score": round(float(score), 2),
        "early_total_score": 0.0,
        "early_confirm_lag_bars": np.nan,
        "early_has_fvg": 0,
        "early_has_sweep": 0,
        "early_choch_down": 0,
        "early_bos_down": 0,
        "early_micro_low_break": 0,
        "early_top_low_break": 0,
        "early_bull_ob_mitigated": 0.0,
        "early_low_break_pct": np.nan,
        "confirmed_pos": np.nan,
        "confirmed_date": pd.NaT,
        "confirm_delay_bars": np.nan,
        "confirmed_score": 0.0,
        "confirmed_struct_score": 0.0,
        "confirmed_has_structure": 0.0,
        "confirmed_zone_top": np.nan,
        "confirmed_zone_bottom": np.nan,
        "confirmed_ob_volume": np.nan,
        "confirmed_percentage": np.nan,
        "mitigated_pos": np.nan,
        "cleared_pos": np.nan,
    }


def _early_state_for_raw_zone(
    df: pd.DataFrame,
    ob_events: pd.DataFrame,
    structure_events: pd.DataFrame,
    raw_event: pd.Series,
    *,
    max_early_lag: int,
) -> dict[str, object]:
    if int(raw_event["direction"]) != -1:
        return {}

    origin_pos = int(raw_event["origin_pos"])
    if origin_pos >= len(df) - 1:
        return {}

    early_end = min(len(df) - 1, origin_pos + max_early_lag)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    atr = _atr14(df)
    raw_score = float(raw_event.get("raw_score", 0.0) or 0.0)
    best: dict[str, object] | None = None
    first_evidence_pos: int | None = None

    detected = pd.to_numeric(structure_events["detected_pos"], errors="coerce") if not structure_events.empty else pd.Series(dtype=float)
    mitigated_pos = pd.to_numeric(ob_events["mitigated_pos"], errors="coerce") if not ob_events.empty else pd.Series(dtype=float)
    confirmed_pos = pd.to_numeric(ob_events["confirmed_pos"], errors="coerce") if not ob_events.empty else pd.Series(dtype=float)

    for asof_pos in range(origin_pos + 1, early_end + 1):
        visible = structure_events[
            detected.notna()
            & (detected >= origin_pos)
            & (detected <= asof_pos)
        ].copy() if not structure_events.empty else pd.DataFrame()
        internal = visible[visible["scale"].astype(str).str.startswith("internal_")] if not visible.empty else visible
        internal_bear = internal[internal["direction"] == -1] if not internal.empty else internal
        choch_down = internal_bear[internal_bear["event_type"] == "choch"] if not internal_bear.empty else internal_bear
        bos_down = internal_bear[internal_bear["event_type"] == "bos"] if not internal_bear.empty else internal_bear
        bear_fvg = visible[(visible["event_type"] == "fvg") & (visible["direction"] == -1)] if not visible.empty else visible
        high_sweep = visible[
            (visible["event_type"] == "liquidity")
            & (visible["direction"] == 1)
            & pd.to_numeric(visible["swept_pos"], errors="coerce").notna()
        ] if not visible.empty else visible

        micro_start = max(0, origin_pos - 5)
        micro_ref = float(low.iloc[micro_start:origin_pos].min()) if micro_start < origin_pos else float(low.iloc[origin_pos])
        after = close.iloc[origin_pos + 1:asof_pos + 1]
        micro_low_break = int(not after.empty and float(after.min()) < micro_ref)
        top_low = float(low.iloc[origin_pos])
        top_low_break = int(not after.empty and float(after.min()) < top_low)

        bull_mitigated = ob_events[
            (ob_events["direction"] == 1)
            & confirmed_pos.notna()
            & (confirmed_pos <= asof_pos)
            & mitigated_pos.notna()
            & (mitigated_pos >= origin_pos)
            & (mitigated_pos <= asof_pos)
        ] if not ob_events.empty else pd.DataFrame()

        post_low = float(low.iloc[origin_pos + 1:asof_pos + 1].min()) if origin_pos + 1 <= asof_pos else np.nan
        atr_val = float(atr.iloc[origin_pos]) if pd.notna(atr.iloc[origin_pos]) and atr.iloc[origin_pos] > 0 else np.nan
        displacement_atr = (float(close.iloc[origin_pos]) - post_low) / atr_val if np.isfinite(atr_val) and pd.notna(post_low) else np.nan

        structure_score = min(
            35.0,
            20.0 * int(not choch_down.empty)
            + 15.0 * int(not bos_down.empty)
            + 10.0 * micro_low_break
            + 10.0 * top_low_break,
        )
        supply_score = min(30.0, 15.0 * int(not bear_fvg.empty) + 15.0 * int(not high_sweep.empty))
        ladder_score = min(20.0, 20.0 * min(1.0, len(bull_mitigated)))
        displacement_score = min(15.0, max(0.0, float(displacement_atr)) / 1.5 * 15.0) if pd.notna(displacement_atr) else 0.0
        struct_score = float(structure_score + supply_score + ladder_score + displacement_score)
        total_score = min(100.0, raw_score * 0.55 + struct_score * 0.75)
        has_evidence = bool(
            struct_score > 0.0
            or (pd.notna(displacement_atr) and displacement_atr >= 0.8)
            or micro_low_break
            or top_low_break
        )
        if has_evidence and first_evidence_pos is None:
            first_evidence_pos = asof_pos
        if not has_evidence:
            continue

        candidate = {
            "early_pos": int(asof_pos),
            "early_date": pd.Timestamp(df.index[asof_pos]),
            "early_first_pos": int(first_evidence_pos if first_evidence_pos is not None else asof_pos),
            "early_first_date": pd.Timestamp(df.index[first_evidence_pos if first_evidence_pos is not None else asof_pos]),
            "early_score": round(float(total_score), 2),
            "early_struct_score": round(float(struct_score), 2),
            "early_raw_score": round(float(raw_score), 2),
            "early_total_score": round(float(total_score), 2),
            "early_confirm_lag_bars": int(asof_pos - origin_pos),
            "early_has_fvg": int(not bear_fvg.empty),
            "early_has_sweep": int(not high_sweep.empty),
            "early_choch_down": int(not choch_down.empty),
            "early_bos_down": int(not bos_down.empty),
            "early_micro_low_break": int(micro_low_break),
            "early_top_low_break": int(top_low_break),
            "early_bull_ob_mitigated": float(len(bull_mitigated)),
            "early_low_break_pct": round((float(low.iloc[asof_pos]) / float(high.iloc[origin_pos]) - 1.0) * 100.0, 2),
        }
        if best is None or float(candidate["early_score"]) > float(best["early_score"]):
            best = candidate

    return best or {}


def _attach_raw_early_states(
    raw_events: pd.DataFrame,
    df: pd.DataFrame,
    ob_events: pd.DataFrame,
    structure_events: pd.DataFrame,
    *,
    max_early_lag: int,
) -> pd.DataFrame:
    if raw_events.empty:
        return raw_events
    out = raw_events.copy()
    for idx, raw_event in out.iterrows():
        state = _early_state_for_raw_zone(
            df,
            ob_events,
            structure_events,
            raw_event,
            max_early_lag=max_early_lag,
        )
        for col, value in state.items():
            out.at[idx, col] = value
    return out


def _attach_confirmed_to_raw(
    raw_events: pd.DataFrame,
    ob_events: pd.DataFrame,
    *,
    raw_match_bars: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw_events.empty or ob_events.empty:
        return raw_events, _empty_ob_event_frame()

    out = raw_events.copy()
    unlinked: list[dict] = []
    raw_origin = pd.to_numeric(out["origin_pos"], errors="coerce")
    for _, event in ob_events.iterrows():
        direction = int(event.get("direction", 0))
        origin_pos = int(event.get("origin_pos", -1))
        same_direction = out["direction"].astype(int) == direction
        exact = out[same_direction & (raw_origin == origin_pos)]
        match_idx = None
        if not exact.empty:
            match_idx = exact.index[0]
        else:
            nearby = out[same_direction & raw_origin.notna() & ((raw_origin - origin_pos).abs() <= raw_match_bars)].copy()
            if not nearby.empty:
                confirmed_top = float(event.get("top", np.nan))
                confirmed_bottom = float(event.get("bottom", np.nan))
                if pd.notna(confirmed_top) and pd.notna(confirmed_bottom):
                    nearby["_overlaps"] = [
                        int(_intersects(float(bottom), float(top), float(confirmed_bottom), float(confirmed_top)))
                        for bottom, top in zip(nearby["bottom"], nearby["top"], strict=False)
                    ]
                else:
                    nearby["_overlaps"] = 0
                nearby["_distance"] = (pd.to_numeric(nearby["origin_pos"], errors="coerce") - origin_pos).abs()
                nearby = nearby.sort_values(["_overlaps", "_distance", "raw_score"], ascending=[False, True, False])
                match_idx = nearby.index[0]

        if match_idx is None:
            unlinked.append(dict(event))
            continue

        out.at[match_idx, "confirmed_pos"] = int(event["confirmed_pos"])
        out.at[match_idx, "confirmed_date"] = pd.Timestamp(event["confirmed_date"])
        out.at[match_idx, "confirm_delay_bars"] = int(event["confirm_delay_bars"])
        out.at[match_idx, "confirmed_score"] = float(event.get("score", 0.0) or 0.0)
        out.at[match_idx, "confirmed_struct_score"] = float(event.get("struct_score", event.get("score", 0.0)) or 0.0)
        out.at[match_idx, "confirmed_has_structure"] = float(event.get("confirm_has_structure", 0.0) or 0.0)
        out.at[match_idx, "confirmed_zone_top"] = float(event.get("top", np.nan))
        out.at[match_idx, "confirmed_zone_bottom"] = float(event.get("bottom", np.nan))
        out.at[match_idx, "confirmed_ob_volume"] = float(event.get("ob_volume", np.nan))
        out.at[match_idx, "confirmed_percentage"] = float(event.get("percentage", np.nan))
        out.at[match_idx, "mitigated_pos"] = event.get("mitigated_pos", np.nan)
        out.at[match_idx, "cleared_pos"] = event.get("cleared_pos", np.nan)

    unlinked_df = pd.DataFrame(unlinked) if unlinked else _empty_ob_event_frame()
    return out, unlinked_df


def _build_raw_ob_events(
    df: pd.DataFrame,
    ob_events: pd.DataFrame,
    structure_events: pd.DataFrame,
    *,
    max_early_lag: int = 3,
    raw_match_bars: int = 3,
    include_early_states: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the full potential OB-zone table.

    Raw zones are causal candle zones: once a candle closes, both its possible
    supply and demand zones are known. Later early/confirmed states are attached
    to those same rows instead of being allowed to introduce new top anchors.
    """

    if df.empty or not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
        return _empty_raw_ob_event_frame(), _empty_ob_event_frame()

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    atr = _atr14(df)
    vol20 = df["volume"].astype(float).rolling(20, min_periods=5).mean()
    rolling_high = high.rolling(60, min_periods=1).max()
    rolling_low = low.rolling(60, min_periods=1).min()
    ret20 = (close / close.shift(20) - 1.0) * 100.0
    ret60 = (close / close.shift(60) - 1.0) * 100.0

    rows: list[dict[str, object]] = []
    for pos in range(len(df)):
        if pd.isna(high.iloc[pos]) or pd.isna(low.iloc[pos]) or high.iloc[pos] <= low.iloc[pos]:
            continue
        rows.append(
            _raw_zone_row(
                df,
                direction=-1,
                pos=pos,
                atr=atr,
                vol20=vol20,
                rolling_high=rolling_high,
                rolling_low=rolling_low,
                ret20=ret20,
                ret60=ret60,
            )
        )
        rows.append(
            _raw_zone_row(
                df,
                direction=1,
                pos=pos,
                atr=atr,
                vol20=vol20,
                rolling_high=rolling_high,
                rolling_low=rolling_low,
                ret20=ret20,
                ret60=ret60,
            )
        )

    if not rows:
        return _empty_raw_ob_event_frame(), _empty_ob_event_frame()

    raw_events = pd.DataFrame(rows)
    if include_early_states:
        raw_events = _attach_raw_early_states(
            raw_events,
            df,
            ob_events,
            structure_events,
            max_early_lag=max_early_lag,
        )
    raw_events, unlinked = _attach_confirmed_to_raw(
        raw_events,
        ob_events,
        raw_match_bars=raw_match_bars,
    )
    return raw_events.sort_values(["detected_pos", "direction"]).reset_index(drop=True), unlinked


def attach_smc_early_states(
    df: pd.DataFrame,
    raw_ob_events: pd.DataFrame,
    ob_events: pd.DataFrame,
    structure_events: pd.DataFrame,
    *,
    max_early_lag: int = 3,
) -> pd.DataFrame:
    """Attach early SMC states to selected raw OB-zone rows."""

    return _attach_raw_early_states(
        raw_ob_events,
        df,
        ob_events,
        structure_events,
        max_early_lag=max_early_lag,
    )


def _build_smc_bundle(
    df: pd.DataFrame,
    *,
    swing_length: int = 5,
    close_mitigation: bool = False,
    include_raw: bool = True,
    include_early_states: bool = False,
) -> dict[str, pd.DataFrame]:
    structure_events = _build_structure_events(df)
    ob_events = _build_ob_events(
        df,
        swing_length=swing_length,
        close_mitigation=close_mitigation,
    )
    ob_events = _attach_ob_structure_scores(ob_events, structure_events)
    if include_raw:
        raw_ob_events, unlinked_confirmed = _build_raw_ob_events(
            df,
            ob_events,
            structure_events,
            max_early_lag=3,
            raw_match_bars=3,
            include_early_states=include_early_states,
        )
    else:
        raw_ob_events, unlinked_confirmed = _empty_raw_ob_event_frame(), _empty_ob_event_frame()
    return {
        "raw_ob_events": raw_ob_events,
        "ob_events": ob_events,
        "structure_events": structure_events,
        "unlinked_confirmed_ob_events": unlinked_confirmed,
    }


def build_smc_bundle(
    df: pd.DataFrame,
    *,
    swing_length: int = 5,
    close_mitigation: bool = False,
    include_raw: bool = True,
    include_early_states: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build SMC event tables for recall-source and feature builders."""

    return _build_smc_bundle(
        df,
        swing_length=swing_length,
        close_mitigation=close_mitigation,
        include_raw=include_raw,
        include_early_states=include_early_states,
    )


def smc_early_reversal_features_for_position(
    df: pd.DataFrame,
    ob_events: pd.DataFrame,
    structure_events: pd.DataFrame,
    *,
    top_pos: int,
    asof_pos: int,
) -> dict[str, float]:
    """Return causal SMC raw/early evidence for one possible top bar."""

    if df.empty:
        return {}
    top_pos = int(max(0, min(len(df) - 1, top_pos)))
    asof_pos = int(max(top_pos, min(len(df) - 1, asof_pos)))
    top_price = float(df["high"].iloc[top_pos])
    row = pd.Series({
        "top_pos": top_pos,
        "top_price": top_price,
        "score_asof_pos": asof_pos,
    })
    features: dict[str, float] = {}
    features.update(
        _raw_features_for_candidate(
            df,
            structure_events,
            row,
            top_pos=top_pos,
            asof_pos=asof_pos,
            top_price=top_price,
        )
    )
    features.update(
        _early_features_for_candidate(
            df,
            ob_events,
            structure_events,
            top_pos=top_pos,
            asof_pos=asof_pos,
        )
    )
    return features


def _zone_distance_pct(price: float, bottom: float, top: float) -> float:
    if bottom <= price <= top:
        return 0.0
    if price > top:
        return (price / top - 1) * 100 if top > 0 else np.nan
    return (price / bottom - 1) * 100 if bottom > 0 else np.nan


def _regime_score(events: pd.DataFrame, asof_pos: int) -> float:
    if events.empty:
        return 0.0
    confirmed = events[pd.to_numeric(events["confirmed_pos"], errors="coerce") <= asof_pos].copy()
    if confirmed.empty:
        return 0.0
    recent_bull = confirmed[
        (confirmed["direction"] == 1)
        & (confirmed["confirmed_pos"] >= asof_pos - 60)
    ]
    recent_bear = confirmed[
        (confirmed["direction"] == -1)
        & (confirmed["confirmed_pos"] >= asof_pos - 20)
    ]
    bull_sum = pd.to_numeric(recent_bull["score"], errors="coerce").fillna(0).sum()
    bear_sum = pd.to_numeric(recent_bear["score"], errors="coerce").fillna(0).sum()
    return round(float(bull_sum - bear_sum), 2)


def _intersects(low_a: float, high_a: float, low_b: float, high_b: float) -> bool:
    return low_a <= high_b and high_a >= low_b


def _visible_structure_events(structure_events: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    if structure_events.empty or end < start:
        return _empty_event_frame()
    detected = pd.to_numeric(structure_events["detected_pos"], errors="coerce")
    return structure_events[detected.notna() & (detected >= start) & (detected <= end)].copy()


def _raw_features_for_candidate(
    df: pd.DataFrame,
    structure_events: pd.DataFrame,
    row: pd.Series,
    *,
    top_pos: int,
    asof_pos: int,
    top_price: float,
) -> dict[str, float]:
    features: dict[str, float] = {}
    origin_start = max(0, top_pos - 1)
    origin_end = min(len(df) - 1, top_pos + 1, asof_pos)
    if origin_start > origin_end:
        return features

    origin_slice = df.iloc[origin_start:origin_end + 1]
    if origin_slice.empty:
        return features

    volume = df["volume"].astype(float)
    vol20 = volume.rolling(20, min_periods=5).mean()
    atr = _atr14(df)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    candidates = origin_slice.copy()
    candidates["_vol"] = volume.iloc[origin_start:origin_end + 1].values
    candidates["_high"] = high.iloc[origin_start:origin_end + 1].values
    origin_idx = candidates.sort_values(["_high", "_vol"], ascending=[False, False]).index[0]
    origin_pos = int(df.index.get_loc(origin_idx))
    zone_top = float(high.iloc[origin_pos])
    zone_bottom = float(low.iloc[origin_pos])
    candle_low = float(low.iloc[top_pos])
    candle_high = float(high.iloc[top_pos])
    zone_overlap = int((zone_bottom <= top_price <= zone_top) or _intersects(candle_low, candle_high, zone_bottom, zone_top))

    early_end = min(len(df) - 1, top_pos + 3, asof_pos)
    post_start = min(len(df) - 1, origin_pos + 1)
    if post_start <= early_end:
        post_low = float(low.iloc[post_start:early_end + 1].min())
        atr_val = float(atr.iloc[origin_pos]) if pd.notna(atr.iloc[origin_pos]) and atr.iloc[origin_pos] > 0 else np.nan
        displacement_atr = (float(close.iloc[origin_pos]) - post_low) / atr_val if np.isfinite(atr_val) else np.nan
    else:
        displacement_atr = np.nan

    visible = _visible_structure_events(structure_events, top_pos, early_end)
    bear_fvg = visible[(visible["event_type"] == "fvg") & (visible["direction"] == -1)]
    high_sweep = visible[
        (visible["event_type"] == "liquidity")
        & (visible["direction"] == 1)
        & pd.to_numeric(visible["swept_pos"], errors="coerce").notna()
    ]
    bear_structure = visible[
        (visible["direction"] == -1)
        & visible["event_type"].isin(["bos", "choch"])
    ]
    evidence_positions = []
    for part in (bear_fvg, high_sweep, bear_structure):
        if not part.empty:
            evidence_positions.extend(pd.to_numeric(part["detected_pos"], errors="coerce").dropna().astype(int).tolist())
    if pd.notna(displacement_atr) and displacement_atr >= 0.8:
        evidence_positions.append(early_end)

    atr_val = float(atr.iloc[origin_pos]) if pd.notna(atr.iloc[origin_pos]) and atr.iloc[origin_pos] > 0 else np.nan
    zone_width_atr = (zone_top - zone_bottom) / atr_val if np.isfinite(atr_val) else np.nan
    vol_base = float(vol20.iloc[origin_pos]) if pd.notna(vol20.iloc[origin_pos]) and vol20.iloc[origin_pos] > 0 else np.nan
    volume_ratio = float(volume.iloc[origin_pos]) / vol_base if np.isfinite(vol_base) else np.nan

    score = 0.0
    score += 20.0 * zone_overlap
    if pd.notna(displacement_atr):
        score += min(20.0, max(0.0, float(displacement_atr)) / 1.5 * 20.0)
    score += 20.0 * int(not bear_fvg.empty)
    score += 20.0 * int(not high_sweep.empty)
    if pd.notna(volume_ratio):
        score += min(20.0, max(0.0, float(volume_ratio) - 0.8) / 1.2 * 20.0)
    score = min(100.0, score)

    present = int(bool(evidence_positions) and score >= 20.0)
    features["smc_raw_bear_present_3d"] = float(present)
    features["smc_raw_bear_score_max_3d"] = round(float(score), 2) if present else 0.0
    features["smc_raw_bear_detect_lag"] = float(min(evidence_positions) - origin_pos) if evidence_positions else np.nan
    features["smc_raw_bear_zone_overlap_top"] = float(zone_overlap)
    features["smc_raw_bear_displacement_atr"] = round(float(displacement_atr), 2) if pd.notna(displacement_atr) else np.nan
    features["smc_raw_bear_has_fvg"] = float(int(not bear_fvg.empty))
    features["smc_raw_bear_has_sweep"] = float(int(not high_sweep.empty))
    features["smc_raw_bear_zone_width_atr"] = round(float(zone_width_atr), 2) if pd.notna(zone_width_atr) else np.nan
    features["smc_raw_bear_volume_ratio"] = round(float(volume_ratio), 2) if pd.notna(volume_ratio) else np.nan
    return features


def _early_features_for_candidate(
    df: pd.DataFrame,
    ob_events: pd.DataFrame,
    structure_events: pd.DataFrame,
    *,
    top_pos: int,
    asof_pos: int,
) -> dict[str, float]:
    features: dict[str, float] = {}
    early_end = min(len(df) - 1, top_pos + 3, asof_pos)
    if early_end < top_pos:
        return features

    visible = _visible_structure_events(structure_events, top_pos, early_end)
    internal = visible[visible["scale"].astype(str).str.startswith("internal_")] if not visible.empty else visible
    internal_bear = internal[internal["direction"] == -1] if not internal.empty else internal
    choch_down = internal_bear[internal_bear["event_type"] == "choch"]
    bos_down = internal_bear[internal_bear["event_type"] == "bos"]
    bear_fvg = visible[(visible["event_type"] == "fvg") & (visible["direction"] == -1)] if not visible.empty else visible
    high_sweep = visible[
        (visible["event_type"] == "liquidity")
        & (visible["direction"] == 1)
        & pd.to_numeric(visible["swept_pos"], errors="coerce").notna()
    ] if not visible.empty else visible

    close = df["close"].astype(float)
    low = df["low"].astype(float)
    high = df["high"].astype(float)
    micro_start = max(0, top_pos - 5)
    micro_ref = float(low.iloc[micro_start:top_pos].min()) if micro_start < top_pos else float(low.iloc[top_pos])
    after = close.iloc[top_pos + 1:early_end + 1] if top_pos + 1 <= early_end else pd.Series(dtype=float)
    micro_low_break = int(not after.empty and float(after.min()) < micro_ref)
    top_low = float(low.iloc[top_pos])
    top_low_break = int(not after.empty and float(after.min()) < top_low)

    mitigated_pos = pd.to_numeric(ob_events["mitigated_pos"], errors="coerce") if not ob_events.empty else pd.Series(dtype=float)
    confirmed_pos = pd.to_numeric(ob_events["confirmed_pos"], errors="coerce") if not ob_events.empty else pd.Series(dtype=float)
    bull_mitigated = ob_events[
        (ob_events["direction"] == 1)
        & confirmed_pos.notna()
        & (confirmed_pos <= early_end)
        & mitigated_pos.notna()
        & (mitigated_pos >= top_pos)
        & (mitigated_pos <= early_end)
    ] if not ob_events.empty else pd.DataFrame()

    active_bull = ob_events[
        (ob_events["direction"] == 1)
        & confirmed_pos.notna()
        & (confirmed_pos <= early_end)
        & (mitigated_pos.isna() | (mitigated_pos > early_end))
    ].copy() if not ob_events.empty else pd.DataFrame()
    bull_ladder_intact = 0
    if not active_bull.empty:
        asof_close = float(close.iloc[early_end])
        active_bull["_dist"] = [
            _zone_distance_pct(asof_close, float(bottom), float(top))
            for bottom, top in zip(active_bull["bottom"], active_bull["top"], strict=False)
        ]
        near = active_bull.sort_values(["_dist", "score"], ascending=[True, False]).iloc[0]
        bull_ladder_intact = int(asof_close >= float(near["bottom"]))

    retest_end = min(len(df) - 1, top_pos + 5, asof_pos)
    origin_low = float(low.iloc[top_pos])
    origin_high = float(high.iloc[top_pos])
    retest_reject = 0
    if top_pos + 1 <= retest_end:
        retest = df.iloc[top_pos + 1:retest_end + 1]
        retest_reject = int(((retest["high"].astype(float) >= origin_low) & (retest["close"].astype(float) < origin_low)).any())

    features["smc_early_internal_choch_down_3d"] = float(int(not choch_down.empty))
    features["smc_early_internal_bos_down_3d"] = float(int(not bos_down.empty))
    features["smc_early_micro_low_break_3d"] = float(micro_low_break)
    features["smc_early_top_low_break_3d"] = float(top_low_break)
    features["smc_early_bear_fvg_3d"] = float(int(not bear_fvg.empty))
    features["smc_early_bull_ob_mitigated_3d"] = float(len(bull_mitigated))
    features["smc_early_bull_ladder_intact"] = float(bull_ladder_intact)
    features["smc_early_liquidity_sweep_high_3d"] = float(int(not high_sweep.empty))
    features["smc_early_retest_reject_5d"] = float(retest_reject)

    structure_score = min(35.0, 20.0 * int(not choch_down.empty) + 15.0 * int(not bos_down.empty) + 10.0 * micro_low_break + 10.0 * top_low_break)
    supply_score = min(30.0, 15.0 * int(not bear_fvg.empty) + 15.0 * int(not high_sweep.empty))
    ladder_score = min(20.0, 20.0 * min(1.0, len(bull_mitigated)) + (0.0 if bull_ladder_intact else 5.0))
    retest_score = 15.0 * retest_reject
    features["smc_early_score_3d"] = round(float(structure_score + supply_score + ladder_score + retest_score), 2)
    return features


def _features_for_candidate(
    df: pd.DataFrame,
    events: pd.DataFrame,
    structure_events: pd.DataFrame,
    row: pd.Series,
) -> dict[str, float]:
    features = _empty_features()
    if events.empty and structure_events.empty:
        return features

    top_pos = int(row.get("top_pos", row.get("signal_pos", 0)))
    asof_pos = int(row.get("score_asof_pos", row.get("confirm_pos", top_pos)))
    top_pos = max(0, min(len(df) - 1, top_pos))
    asof_pos = max(0, min(len(df) - 1, asof_pos))
    top_price = float(row.get("top_price", df["high"].iloc[top_pos]))
    asof_price = float(df["close"].iloc[asof_pos])

    features.update(
        _raw_features_for_candidate(
            df,
            structure_events,
            row,
            top_pos=top_pos,
            asof_pos=asof_pos,
            top_price=top_price,
        )
    )
    features.update(
        _early_features_for_candidate(
            df,
            events,
            structure_events,
            top_pos=top_pos,
            asof_pos=asof_pos,
        )
    )

    confirmed = events[pd.to_numeric(events["confirmed_pos"], errors="coerce") <= asof_pos].copy()
    if not confirmed.empty:
        recent_bull_60 = confirmed[(confirmed["direction"] == 1) & (confirmed["confirmed_pos"] >= asof_pos - 60)]
        recent_bear_20 = confirmed[(confirmed["direction"] == -1) & (confirmed["confirmed_pos"] >= asof_pos - 20)]
        features["smc_live_bull_ob_count_60d"] = float(len(recent_bull_60))
        features["smc_live_bull_ob_score_sum_60d"] = round(float(pd.to_numeric(recent_bull_60["score"], errors="coerce").fillna(0).sum()), 2)
        features["smc_live_bear_ob_count_20d"] = float(len(recent_bear_20))
        features["smc_live_bear_ob_score_max_20d"] = round(float(pd.to_numeric(recent_bear_20["score"], errors="coerce").fillna(0).max()), 2) if not recent_bear_20.empty else 0.0
        features["smc_live_bear_ob_struct_score_max_20d"] = (
            round(float(pd.to_numeric(recent_bear_20["struct_score"], errors="coerce").fillna(0).max()), 2)
            if not recent_bear_20.empty and "struct_score" in recent_bear_20.columns
            else 0.0
        )
        features["smc_live_ob_regime_score"] = _regime_score(events, asof_pos)

        confirmed_bull = confirmed[confirmed["direction"] == 1]
        if not confirmed_bull.empty:
            last_bull_confirm = int(pd.to_numeric(confirmed_bull["confirmed_pos"], errors="coerce").max())
            features["smc_live_last_bull_ob_age"] = float(asof_pos - last_bull_confirm)

        mitigated = pd.to_numeric(events["mitigated_pos"], errors="coerce")
        bull_mitigated_10d = events[
            (events["direction"] == 1)
            & mitigated.notna()
            & (mitigated > asof_pos - 10)
            & (mitigated <= asof_pos)
        ]
        features["smc_live_bull_ob_mitigated_10d"] = float(len(bull_mitigated_10d))

        active_bull = confirmed[
            (confirmed["direction"] == 1)
            & (
                pd.to_numeric(confirmed["mitigated_pos"], errors="coerce").isna()
                | (pd.to_numeric(confirmed["mitigated_pos"], errors="coerce") > asof_pos)
            )
        ].copy()
        if not active_bull.empty:
            active_bull["_dist"] = [
                _zone_distance_pct(asof_price, float(bottom), float(top))
                for bottom, top in zip(active_bull["bottom"], active_bull["top"], strict=False)
            ]
            active_bull["_abs_dist"] = active_bull["_dist"].abs()
            best = active_bull.sort_values(["_abs_dist", "score"], ascending=[True, False]).iloc[0]
            features["smc_live_nearest_bull_ob_dist_pct"] = round(float(best["_dist"]), 2)

    d5_end = min(len(df) - 1, top_pos + 5)
    d10_end = min(len(df) - 1, top_pos + 10)
    confirmed_pos = pd.to_numeric(events["confirmed_pos"], errors="coerce")
    bear_d5 = events[(events["direction"] == -1) & (confirmed_pos > top_pos) & (confirmed_pos <= d5_end)]
    bear_d10 = events[(events["direction"] == -1) & (confirmed_pos > top_pos) & (confirmed_pos <= d10_end)]
    features["smc_d5_bear_ob_confirmed"] = float(int(not bear_d5.empty))
    features["smc_d10_bear_ob_confirmed"] = float(int(not bear_d10.empty))
    features["smc_d10_bear_ob_score_max"] = round(float(pd.to_numeric(bear_d10["score"], errors="coerce").fillna(0).max()), 2) if not bear_d10.empty else 0.0
    features["smc_d10_bear_ob_struct_score_max"] = (
        round(float(pd.to_numeric(bear_d10["struct_score"], errors="coerce").fillna(0).max()), 2)
        if not bear_d10.empty and "struct_score" in bear_d10.columns
        else 0.0
    )

    mitigated_pos = pd.to_numeric(events["mitigated_pos"], errors="coerce")
    bull_mitigated_d10 = events[
        (events["direction"] == 1)
        & mitigated_pos.notna()
        & (mitigated_pos > top_pos)
        & (mitigated_pos <= d10_end)
    ]
    features["smc_d10_bull_ob_mitigated_count"] = float(len(bull_mitigated_d10))
    start_regime = _regime_score(events, asof_pos)
    end_regime = _regime_score(events, d10_end)
    features["smc_d10_ob_regime_flip"] = float(int(start_regime > 0 and end_regime <= 0))

    near_bear = events[
        (events["direction"] == -1)
        & ((pd.to_numeric(events["origin_pos"], errors="coerce") - top_pos).abs() <= 5)
    ].copy()
    features["smc_diag_bear_ob_confirmed_near_top"] = float(int(not near_bear.empty))
    if not near_bear.empty:
        candle_low = float(df["low"].iloc[top_pos])
        candle_high = float(df["high"].iloc[top_pos])
        near_bear["top_intersects"] = [
            bool((bottom <= top_price <= top) or (candle_low <= top and candle_high >= bottom))
            for bottom, top in zip(near_bear["bottom"], near_bear["top"], strict=False)
        ]
        features["smc_diag_top_inside_bear_ob_confirmed_zone"] = float(int(near_bear["top_intersects"].any()))
        match = near_bear.sort_values(["top_intersects", "score"], ascending=[False, False]).iloc[0]
        features["smc_diag_bear_ob_confirm_delay"] = float(match["confirm_delay_bars"])
        features["smc_diag_bear_ob_confirm_has_structure"] = float(match.get("confirm_has_structure", 0.0))

    return features


def _symbol_data_map(symbol_data: Mapping[str, dict] | None) -> dict[tuple[str, str], pd.DataFrame]:
    out: dict[tuple[str, str], pd.DataFrame] = {}
    if not symbol_data:
        return out
    for item in symbol_data.values():
        if not isinstance(item, Mapping) or "df" not in item:
            continue
        market = str(item.get("market", ""))
        symbol = str(item.get("symbol", item.get("sym", "")))
        if market and symbol:
            out[(market, symbol)] = _normalize_df(item["df"])
    return out


def add_smc_ob_features(
    dataset: pd.DataFrame,
    *,
    symbol_data: Mapping[str, dict] | None = None,
    swing_length: int = 5,
    close_mitigation: bool = False,
) -> pd.DataFrame:
    """Attach causal SMC structure features to already discovered top candidates."""

    out = dataset.copy()
    for col, value in _empty_features().items():
        if col not in out.columns:
            out[col] = value

    data_map = _symbol_data_map(symbol_data)
    if out.empty or not data_map:
        return out

    event_cache: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    feature_rows: list[dict[str, float]] = []
    for _, row in out.iterrows():
        key = (str(row.get("market", "")), str(row.get("sym", "")))
        df = data_map.get(key)
        if df is None or not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
            feature_rows.append(_empty_features())
            continue
        if key not in event_cache:
            event_cache[key] = build_smc_bundle(
                df,
                swing_length=swing_length,
                close_mitigation=close_mitigation,
                include_raw=False,
            )
        bundle = event_cache[key]
        feature_rows.append(_features_for_candidate(df, bundle["ob_events"], bundle["structure_events"], row))

    feature_df = pd.DataFrame(feature_rows, index=out.index)
    for col in SMC_STRUCTURE_FEATURES:
        out[col] = feature_df[col]
    return out
