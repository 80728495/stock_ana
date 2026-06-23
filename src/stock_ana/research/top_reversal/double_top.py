"""Double-top structure helpers for top-reversal research.

The functions in this module use full-series ZigZag pivots.  They are meant
for oracle sample labels and structural recall experiments, not for realtime
model features unless the caller explicitly truncates the input first.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def date_text(ts) -> str:
    if pd.isna(ts):
        return ""
    return str(pd.Timestamp(ts).date())


def _empty_result(reason: str) -> dict[str, object]:
    return {
        "double_top_candidate": 0,
        "double_top_confirmed": 0,
        "double_top_reason": reason,
        "double_top_head_role": "",
        "double_top_pair_pos": np.nan,
        "double_top_pair_date": "",
        "double_top_pair_price": np.nan,
        "double_top_first_head_pos": np.nan,
        "double_top_first_head_date": "",
        "double_top_first_head_price": np.nan,
        "double_top_second_head_pos": np.nan,
        "double_top_second_head_date": "",
        "double_top_second_head_price": np.nan,
        "double_top_head_separation_bars": np.nan,
        "double_top_head_price_diff_pct": np.nan,
        "double_top_neckline_pos": np.nan,
        "double_top_neckline_date": "",
        "double_top_neckline_price": np.nan,
        "double_top_neckline_source": "",
        "double_top_neckline_drop_pct": np.nan,
        "double_top_break_low_pos": np.nan,
        "double_top_break_low_date": "",
        "double_top_break_low_price": np.nan,
        "double_top_break_neckline_pct": np.nan,
        "double_top_failed_rebound_pos": np.nan,
        "double_top_failed_rebound_date": "",
        "double_top_failed_rebound_price": np.nan,
        "double_top_failed_rebound_vs_neckline_pct": np.nan,
        "double_top_confirm_pos": np.nan,
        "double_top_confirm_date": "",
    }


def actual_high_near(df: pd.DataFrame, pos: int, *, radius: int = 2) -> tuple[int | None, float]:
    if df.empty or "high" not in df.columns:
        return None, float("nan")
    pos = int(max(0, min(len(df) - 1, pos)))
    start = max(0, pos - radius)
    end = min(len(df) - 1, pos + radius)
    window = df["high"].astype(float).iloc[start:end + 1]
    if window.empty:
        return None, float("nan")
    label = window.idxmax()
    return int(df.index.get_loc(label)), float(window.loc[label])


def actual_low_near(df: pd.DataFrame, pos: int, *, radius: int = 2) -> tuple[int | None, float]:
    if df.empty or "low" not in df.columns:
        return None, float("nan")
    pos = int(max(0, min(len(df) - 1, pos)))
    start = max(0, pos - radius)
    end = min(len(df) - 1, pos + radius)
    window = df["low"].astype(float).iloc[start:end + 1]
    if window.empty:
        return None, float("nan")
    label = window.idxmin()
    return int(df.index.get_loc(label)), float(window.loc[label])


def high_pivots_with_actual(df: pd.DataFrame, wave_result: dict | None) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    if not wave_result:
        return out
    for pivot in wave_result.get("all_pivots", []):
        if pivot.get("type") != "H":
            continue
        actual_pos, actual_price = actual_high_near(df, int(pivot["iloc"]))
        if actual_pos is None or not math.isfinite(actual_price):
            continue
        out.append({
            "pivot": pivot,
            "pivot_pos": int(pivot["iloc"]),
            "actual_pos": int(actual_pos),
            "actual_price": float(actual_price),
        })
    return out


def low_pivots_with_actual(df: pd.DataFrame, wave_result: dict | None) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    if not wave_result:
        return out
    for pivot in wave_result.get("all_pivots", []):
        if pivot.get("type") != "L":
            continue
        actual_pos, actual_price = actual_low_near(df, int(pivot["iloc"]))
        if actual_pos is None or not math.isfinite(actual_price):
            continue
        out.append({
            "pivot": pivot,
            "pivot_pos": int(pivot["iloc"]),
            "actual_pos": int(actual_pos),
            "actual_price": float(actual_price),
        })
    return out


def _head_price_diff_pct(price_a: float, price_b: float) -> float:
    base = max(float(price_a), float(price_b))
    return abs(float(price_a) - float(price_b)) / base * 100.0 if base > 0 else float("nan")


def _base_result(
    df: pd.DataFrame,
    *,
    candidate_pos: int,
    candidate_price: float,
    pair_pos: int,
    pair_price: float,
) -> dict[str, object]:
    first_pos, first_price = (candidate_pos, candidate_price)
    second_pos, second_price = (pair_pos, pair_price)
    if second_pos < first_pos:
        first_pos, second_pos = second_pos, first_pos
        first_price, second_price = second_price, first_price

    lower_head = min(first_price, second_price)
    role = "first_head" if candidate_pos == first_pos else "second_head"
    if candidate_pos == pair_pos:
        role = "same_head"

    out = _empty_result("")
    out.update({
        "double_top_candidate": 1,
        "double_top_head_role": role,
        "double_top_pair_pos": int(pair_pos),
        "double_top_pair_date": date_text(df.index[int(pair_pos)]),
        "double_top_pair_price": round(float(pair_price), 4),
        "double_top_first_head_pos": int(first_pos),
        "double_top_first_head_date": date_text(df.index[int(first_pos)]),
        "double_top_first_head_price": round(float(first_price), 4),
        "double_top_second_head_pos": int(second_pos),
        "double_top_second_head_date": date_text(df.index[int(second_pos)]),
        "double_top_second_head_price": round(float(second_price), 4),
        "double_top_head_separation_bars": int(abs(second_pos - first_pos)),
        "double_top_head_price_diff_pct": round(_head_price_diff_pct(first_price, second_price), 2),
    })
    if lower_head > 0:
        out["double_top_lower_head_price"] = round(float(lower_head), 4)
    return out


def evaluate_double_top_pair(
    df: pd.DataFrame,
    wave_result: dict | None,
    candidate_pos: int,
    pair_pos: int,
    *,
    price_tolerance_pct: float = 2.5,
    min_separation_bars: int = 5,
    max_separation_bars: int = 80,
    min_neckline_break_pct: float = 5.0,
    failed_rebound_neckline_pct: float = 2.0,
    actual_radius: int = 2,
) -> dict[str, object]:
    """Evaluate whether two highs form a confirmed macro double top.

    Confirmation intentionally requires structural evidence after the second
    head: a ZigZag low breaks the neckline, then a later ZigZag high fails to
    reclaim the neckline.  That keeps right-edge and still-unresolved cases as
    ambiguous instead of forcing them into top/continuation.
    """

    if df.empty or not wave_result:
        return _empty_result("no_wave_result")

    candidate_actual_pos, candidate_price = actual_high_near(df, int(candidate_pos), radius=0)
    pair_actual_pos, pair_price = actual_high_near(df, int(pair_pos), radius=actual_radius)
    if candidate_actual_pos is None or pair_actual_pos is None:
        return _empty_result("missing_head_price")
    if not math.isfinite(candidate_price) or not math.isfinite(pair_price):
        return _empty_result("missing_head_price")

    result = _base_result(
        df,
        candidate_pos=int(candidate_actual_pos),
        candidate_price=float(candidate_price),
        pair_pos=int(pair_actual_pos),
        pair_price=float(pair_price),
    )

    first_pos = int(result["double_top_first_head_pos"])
    second_pos = int(result["double_top_second_head_pos"])
    first_price = float(result["double_top_first_head_price"])
    second_price = float(result["double_top_second_head_price"])
    separation = int(result["double_top_head_separation_bars"])
    diff_pct = float(result["double_top_head_price_diff_pct"])

    if separation < min_separation_bars:
        result.update({"double_top_candidate": 0, "double_top_reason": "heads_too_close"})
        return result
    if separation > max_separation_bars:
        result.update({"double_top_candidate": 0, "double_top_reason": "heads_too_far"})
        return result
    if diff_pct > price_tolerance_pct:
        result.update({"double_top_candidate": 0, "double_top_reason": "head_price_diff_too_wide"})
        return result

    lows = [
        item for item in low_pivots_with_actual(df, wave_result)
        if first_pos < int(item["pivot_pos"]) < second_pos
    ]
    neckline_source = "zigzag_low"
    if lows:
        neckline = min(lows, key=lambda item: float(item["actual_price"]))
        neckline_pos = int(neckline["actual_pos"])
        neckline_price = float(neckline["actual_price"])
    else:
        middle_low = df["low"].astype(float).iloc[first_pos + 1:second_pos]
        if middle_low.empty:
            result.update({"double_top_reason": "no_neckline_low"})
            return result
        neckline_label = middle_low.idxmin()
        neckline_pos = int(df.index.get_loc(neckline_label))
        neckline_price = float(middle_low.loc[neckline_label])
        neckline_source = "actual_low_between_heads"
    lower_head = min(first_price, second_price)
    result.update({
        "double_top_neckline_pos": neckline_pos,
        "double_top_neckline_date": date_text(df.index[neckline_pos]),
        "double_top_neckline_price": round(neckline_price, 4),
        "double_top_neckline_source": neckline_source,
        "double_top_neckline_drop_pct": round((neckline_price / lower_head - 1.0) * 100.0, 2) if lower_head > 0 else np.nan,
    })

    if neckline_price <= 0:
        result.update({"double_top_reason": "invalid_neckline"})
        return result

    break_threshold = neckline_price * (1.0 - min_neckline_break_pct / 100.0)
    future_lows = [
        item for item in low_pivots_with_actual(df, wave_result)
        if int(item["pivot_pos"]) > second_pos and float(item["actual_price"]) <= break_threshold
    ]
    if not future_lows:
        result.update({"double_top_reason": "no_material_neckline_break"})
        return result

    break_low = min(future_lows, key=lambda item: int(item["pivot_pos"]))
    break_pos = int(break_low["actual_pos"])
    break_price = float(break_low["actual_price"])
    result.update({
        "double_top_break_low_pos": break_pos,
        "double_top_break_low_date": date_text(df.index[break_pos]),
        "double_top_break_low_price": round(break_price, 4),
        "double_top_break_neckline_pct": round((break_price / neckline_price - 1.0) * 100.0, 2),
    })

    reclaim_ceiling = neckline_price * (1.0 + failed_rebound_neckline_pct / 100.0)
    future_highs = [
        item for item in high_pivots_with_actual(df, wave_result)
        if int(item["pivot_pos"]) > int(break_low["pivot_pos"])
    ]
    if not future_highs:
        result.update({"double_top_reason": "no_rebound_after_break"})
        return result

    first_rebound = min(future_highs, key=lambda item: int(item["pivot_pos"]))
    rebound_pos = int(first_rebound["actual_pos"])
    rebound_price = float(first_rebound["actual_price"])
    result.update({
        "double_top_failed_rebound_pos": rebound_pos,
        "double_top_failed_rebound_date": date_text(df.index[rebound_pos]),
        "double_top_failed_rebound_price": round(rebound_price, 4),
        "double_top_failed_rebound_vs_neckline_pct": round((rebound_price / neckline_price - 1.0) * 100.0, 2),
    })
    if rebound_price > reclaim_ceiling:
        result.update({"double_top_reason": "rebound_reclaimed_neckline"})
        return result

    result.update({
        "double_top_confirmed": 1,
        "double_top_reason": "confirmed_neckline_break_failed_rebound",
        "double_top_confirm_pos": rebound_pos,
        "double_top_confirm_date": date_text(df.index[rebound_pos]),
    })
    return result


def best_double_top_for_candidate(
    df: pd.DataFrame,
    wave_result: dict | None,
    candidate_pos: int,
    *,
    price_tolerance_pct: float = 2.5,
    min_separation_bars: int = 5,
    max_separation_bars: int = 80,
    min_neckline_break_pct: float = 5.0,
    failed_rebound_neckline_pct: float = 2.0,
) -> dict[str, object]:
    """Find the best confirmed double-top pair for one candidate bar."""

    if not wave_result:
        return _empty_result("no_wave_result")

    candidate_pos = int(candidate_pos)
    high_pivots = high_pivots_with_actual(df, wave_result)
    candidates: list[dict[str, object]] = []
    fallbacks: list[dict[str, object]] = []
    for item in high_pivots:
        pair_pos = int(item["actual_pos"])
        if pair_pos == candidate_pos:
            continue
        if abs(pair_pos - candidate_pos) > max_separation_bars + 2:
            continue
        result = evaluate_double_top_pair(
            df,
            wave_result,
            candidate_pos,
            pair_pos,
            price_tolerance_pct=price_tolerance_pct,
            min_separation_bars=min_separation_bars,
            max_separation_bars=max_separation_bars,
            min_neckline_break_pct=min_neckline_break_pct,
            failed_rebound_neckline_pct=failed_rebound_neckline_pct,
        )
        if int(result.get("double_top_candidate", 0)) != 1:
            continue
        fallbacks.append(result)
        if int(result.get("double_top_confirmed", 0)) == 1:
            candidates.append(result)

    if candidates:
        return min(
            candidates,
            key=lambda item: (
                abs(float(item.get("double_top_head_price_diff_pct", 999.0))),
                int(item.get("double_top_head_separation_bars", 10**9)),
            ),
        )
    if fallbacks:
        return min(
            fallbacks,
            key=lambda item: (
                abs(float(item.get("double_top_head_price_diff_pct", 999.0))),
                int(item.get("double_top_head_separation_bars", 10**9)),
            ),
        )
    return _empty_result("no_double_top_pair")


def find_double_top_patterns(
    df: pd.DataFrame,
    wave_result: dict | None,
    *,
    min_top_pos: int = 0,
    price_tolerance_pct: float = 2.5,
    min_separation_bars: int = 5,
    max_separation_bars: int = 80,
    min_neckline_break_pct: float = 5.0,
    failed_rebound_neckline_pct: float = 2.0,
) -> list[dict[str, object]]:
    """Return confirmed structural double tops found from ZigZag high pairs."""

    if df.empty or not wave_result:
        return []
    highs = high_pivots_with_actual(df, wave_result)
    rows: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for i, left in enumerate(highs):
        left_pos = int(left["actual_pos"])
        if left_pos < min_top_pos:
            continue
        for right in highs[i + 1:]:
            right_pos = int(right["actual_pos"])
            if right_pos - left_pos < min_separation_bars:
                continue
            if right_pos - left_pos > max_separation_bars:
                break
            result = evaluate_double_top_pair(
                df,
                wave_result,
                left_pos,
                right_pos,
                price_tolerance_pct=price_tolerance_pct,
                min_separation_bars=min_separation_bars,
                max_separation_bars=max_separation_bars,
                min_neckline_break_pct=min_neckline_break_pct,
                failed_rebound_neckline_pct=failed_rebound_neckline_pct,
            )
            if int(result.get("double_top_confirmed", 0)) != 1:
                continue
            key = (int(result["double_top_first_head_pos"]), int(result["double_top_second_head_pos"]))
            if key in seen:
                continue
            seen.add(key)
            top_pos = (
                int(result["double_top_first_head_pos"])
                if float(result["double_top_first_head_price"]) >= float(result["double_top_second_head_price"])
                else int(result["double_top_second_head_pos"])
            )
            rows.append({
                "candidate_source": "double_top",
                "signal_date": pd.Timestamp(df.index[top_pos]),
                "confirm_date": pd.Timestamp(df.index[int(result["double_top_confirm_pos"])]),
                "signal_pos": top_pos,
                "confirm_pos": int(result["double_top_confirm_pos"]),
                "top_date": pd.Timestamp(df.index[top_pos]),
                "top_pos": top_pos,
                "top_price": float(df["high"].iloc[top_pos]),
                "strategies": "double_top",
                "has_shadow": 0,
                "has_doji": 0,
                "has_gap_fail": 0,
                "signal_count": 0,
                "score_max": 0,
                "score_sum": 0,
                "confirm_modes": "double_top_confirmed",
                "signal_dates": date_text(df.index[top_pos]),
                "score_asof_pos": int(result["double_top_confirm_pos"]),
                "score_asof_date": date_text(df.index[int(result["double_top_confirm_pos"])]),
                **result,
            })
    return sorted(rows, key=lambda row: (int(row["top_pos"]), int(row["score_asof_pos"])))
