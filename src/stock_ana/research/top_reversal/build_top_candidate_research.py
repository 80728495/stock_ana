#!/usr/bin/env python3
"""Build a watchlist top-candidate research dataset with ZigZag/wave labels."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def _find_project_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError("Cannot find project root containing pyproject.toml")


PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data" / "cache" / "matplotlib"))

from stock_ana.config import DATA_DIR, OUTPUT_DIR  # noqa: E402
from stock_ana.data.market_data import load_tech_pools_data, load_watchlist_data  # noqa: E402
from stock_ana.research.top_reversal.candidate_sources import (  # noqa: E402
    attach_strategy_matches,
    collect_smc_appear_candidates,
    collect_smc_early_candidates,
    collect_smc_confirmed_candidates,
    collect_smc_raw_candidates,
    collect_smc_top_confirmed_candidates,
    collect_zigzag_peak_candidates,
    merge_recall_candidates,
    recall_flags_for_sources,
)
from stock_ana.research.top_reversal.coverage import strategy_coverage_report  # noqa: E402
from stock_ana.research.top_reversal.double_top import best_double_top_for_candidate  # noqa: E402
from stock_ana.research.top_reversal.feature_registry import (  # noqa: E402
    BUCKET_COLS,
    FEATURE_COLS,
    REALTIME_FEATURE_COLS,
    SMC_CAUSAL_FEATURES,
    SMC_DELAYED_FEATURES,
    SMC_DIAGNOSTIC_FEATURES,
    SMC_EARLY_FEATURES,
    SMC_LIVE_FEATURES,
    SMC_RAW_FEATURES,
    feature_group_summary,
)
from stock_ana.research.top_reversal.feature_pipeline import add_research_features  # noqa: E402
from stock_ana.research.top_reversal.modeling import (  # noqa: E402
    bucket_stats,
    feature_diff,
    fit_lightgbm,
    fit_logistic,
    label_summary,
    score_performance,
    to_markdown_table,
)
from stock_ana.strategies.impl.evening_star_gap import scan_history as scan_evening_star  # noqa: E402
from stock_ana.strategies.impl.gap_fail_reversal import scan_history as scan_gap_fail  # noqa: E402
from stock_ana.strategies.impl.top_reversal import scan_history as scan_high_shadow  # noqa: E402
from stock_ana.strategies.primitives.wave import analyze_wave_structure  # noqa: E402
from stock_ana.research.top_reversal.smc_context import build_smc_bundle  # noqa: E402


OUT_DIR = OUTPUT_DIR / "top_candidate_research"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NON_SMC_FEATURE_COLS = [col for col in REALTIME_FEATURE_COLS if not col.startswith("smc_")]
PRIMARY_MODEL_FEATURE_COLS = list(REALTIME_FEATURE_COLS)


def _date_text(ts) -> str:
    if pd.isna(ts):
        return ""
    return str(pd.Timestamp(ts).date())


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).lower() for c in x.columns]
    x.index = pd.to_datetime(x.index)
    if x.index.tz is not None:
        x.index = x.index.tz_localize(None)
    x.index.name = "date"
    return x.sort_index()


def _position_for_date(df: pd.DataFrame, value) -> int:
    pos = df.index.get_indexer([pd.Timestamp(value)], method="nearest")[0]
    return int(max(0, min(len(df) - 1, pos)))


def _pivot_actual_price(df: pd.DataFrame, pivot: dict | None, *, max_pos: int | None = None) -> float:
    if not pivot:
        return float("nan")
    pos = int(pivot["iloc"])
    start = max(0, pos - 2)
    end = min(len(df) - 1, pos + 2)
    if max_pos is not None:
        end = min(end, int(max_pos))
    if end < start:
        return float("nan")
    if pivot.get("type") == "H":
        return float(df["high"].iloc[start:end + 1].max())
    return float(df["low"].iloc[start:end + 1].min())


def _pivot_actual_high(df: pd.DataFrame, pivot: dict | None, *, radius: int = 2) -> tuple[int | None, float]:
    if not pivot:
        return None, float("nan")
    pos = int(pivot["iloc"])
    start = max(0, pos - radius)
    end = min(len(df) - 1, pos + radius)
    if end < start:
        return None, float("nan")
    window = df["high"].iloc[start:end + 1]
    if window.empty:
        return None, float("nan")
    label = window.idxmax()
    return int(df.index.get_loc(label)), float(window.loc[label])


def _signal_rows(hits: pd.DataFrame, strategy: str, df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    if hits.empty:
        return rows
    for _, row in hits.iterrows():
        signal_date = pd.Timestamp(row["signal_date"])
        confirm_date = pd.Timestamp(row["confirm_date"])
        signal_pos = _position_for_date(df, signal_date)
        confirm_pos = _position_for_date(df, confirm_date)
        payload = {
            "strategy": strategy,
            "signal_date": signal_date,
            "confirm_date": confirm_date,
            "signal_pos": signal_pos,
            "confirm_pos": confirm_pos,
            "score": int(row.get("score", 0)),
            "confirm_mode": str(row.get("confirm_mode", "")),
            "day1_high": float(row.get("day1_high", np.nan)),
            "day1_close": float(row.get("day1_close", np.nan)),
        }
        for key, value in row.items():
            if key in payload or key.startswith("fwd_"):
                continue
            if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)) or pd.isna(value):
                payload[f"{strategy}_{key}"] = value
        rows.append(payload)
    return rows


def _cluster_signals(signals: list[dict], df: pd.DataFrame, merge_bars: int) -> list[dict]:
    if not signals:
        return []
    ordered = sorted(signals, key=lambda s: (int(s["signal_pos"]), int(s["confirm_pos"])))
    clusters: list[list[dict]] = []
    current: list[dict] = []
    end_pos = -1
    for sig in ordered:
        pos = int(sig["signal_pos"])
        if not current or pos <= end_pos:
            current.append(sig)
            end_pos = max(end_pos, pos + merge_bars)
            continue
        clusters.append(current)
        current = [sig]
        end_pos = pos + merge_bars
    if current:
        clusters.append(current)

    rows: list[dict] = []
    for cluster in clusters:
        start_signal_pos = min(int(s["signal_pos"]) for s in cluster)
        end_confirm_pos = max(int(s["confirm_pos"]) for s in cluster)
        top_slice = df.iloc[start_signal_pos:end_confirm_pos + 1]
        top_date = top_slice["high"].idxmax()
        top_pos = int(df.index.get_loc(top_date))
        strategies = sorted({s["strategy"] for s in cluster})
        row = {
            "signal_date": pd.Timestamp(df.index[start_signal_pos]),
            "confirm_date": pd.Timestamp(df.index[end_confirm_pos]),
            "signal_pos": start_signal_pos,
            "confirm_pos": end_confirm_pos,
            "top_date": pd.Timestamp(top_date),
            "top_pos": top_pos,
            "top_price": float(df["high"].iloc[top_pos]),
            "strategies": "+".join(strategies),
            "has_shadow": int("shadow" in strategies),
            "has_doji": int("doji" in strategies),
            "has_gap_fail": int("gap_fail" in strategies),
            "signal_count": len(cluster),
            "score_max": max(int(s.get("score", 0)) for s in cluster),
            "score_sum": sum(int(s.get("score", 0)) for s in cluster),
            "confirm_modes": ",".join(sorted({str(s.get("confirm_mode", "")) for s in cluster if s.get("confirm_mode")})),
            "signal_dates": ",".join(_date_text(s["signal_date"]) for s in cluster),
            "score_asof_pos": end_confirm_pos,
            "score_asof_date": _date_text(df.index[end_confirm_pos]),
        }
        row.update(recall_flags_for_sources(set(strategies), top_pos=top_pos, score_asof_pos=end_confirm_pos))
        for sig in cluster:
            prefix = sig["strategy"]
            for key, value in sig.items():
                if not key.startswith(f"{prefix}_"):
                    continue
                short = key.removeprefix(f"{prefix}_")
                out_key = f"{prefix}_{short}"
                existing = row.get(out_key)
                if existing is None or (isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value) and (pd.isna(existing) or float(value) > float(existing))):
                    row[out_key] = value
        rows.append(row)
    return rows


def _recent_return(close: pd.Series, pos: int, lookback: int, price: float) -> float:
    if pos - lookback < 0:
        return float("nan")
    base = float(close.iloc[pos - lookback])
    return (price / base - 1) * 100 if base > 0 else float("nan")


def _coerce_float(value, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _round_float(value, digits: int = 2):
    out = _coerce_float(value)
    return round(out, digits) if not pd.isna(out) else np.nan


def _major_wave_for_pos(wave_result: dict, pos: int) -> dict | None:
    best = None
    for wave in wave_result.get("major_waves", []):
        start = int(wave["start_pivot"]["iloc"])
        end = int(wave["end_pivot"]["iloc"]) if wave.get("end_pivot") else 10**9
        peak = int(wave["peak_pivot"]["iloc"])
        if start <= pos <= end or start <= pos <= peak + 120:
            best = wave
    return best


def _score_asof_pos(df: pd.DataFrame, row: dict) -> int:
    raw = row.get("score_asof_pos", row.get("confirm_pos", row.get("top_pos", len(df) - 1)))
    try:
        pos = int(raw)
    except (TypeError, ValueError):
        pos = int(row.get("confirm_pos", row.get("top_pos", len(df) - 1)))
    return int(max(0, min(len(df) - 1, pos)))


def _prefixed(prefix: str, values: dict) -> dict:
    if not prefix:
        return values
    return {f"{prefix}{key}": value for key, value in values.items()}


def _zigzag_context_from_wave(
    df: pd.DataFrame,
    row: dict,
    wave_result: dict,
    *,
    prefix: str = "",
    asof_pos: int | None = None,
) -> dict:
    pos = int(row["top_pos"])
    top_price = float(row["top_price"])
    visible_pos = len(df) - 1 if asof_pos is None else int(max(0, min(len(df) - 1, asof_pos)))
    # Realtime features may use confirmed historical pivots, but they must not
    # reinterpret the current candidate bar as a known high/low from hindsight.
    structure_max_pos = min(visible_pos, pos - 1)
    pivots = wave_result.get("all_pivots", [])
    prior_lows = [p for p in pivots if p.get("type") == "L" and int(p["iloc"]) <= structure_max_pos]
    prior_highs = [p for p in pivots if p.get("type") == "H" and int(p["iloc"]) <= structure_max_pos]

    recent_low = prior_lows[-1] if prior_lows else None
    high_cluster = [
        p for p in prior_highs
        if 0 <= pos - int(p["iloc"]) <= 120
        and _pivot_actual_price(df, p, max_pos=structure_max_pos) >= top_price * 0.92
    ]
    first_cluster_high = min(high_cluster, key=lambda p: int(p["iloc"])) if high_cluster else None
    low_before_first_head = None
    if first_cluster_high:
        lows_before_head = [p for p in prior_lows if int(p["iloc"]) < int(first_cluster_high["iloc"])]
        low_before_first_head = lows_before_head[-1] if lows_before_head else None
    middle_low = None
    if first_cluster_high:
        first_head_pos = int(first_cluster_high["iloc"])
        lows_after_head = [p for p in prior_lows if first_head_pos < int(p["iloc"]) < pos]
        middle_low = min(lows_after_head, key=lambda p: _pivot_actual_price(df, p, max_pos=structure_max_pos)) if lows_after_head else None

    major_wave = _major_wave_for_pos(wave_result, pos)
    wave_start = major_wave.get("start_pivot") if major_wave else None

    fallback_low = None
    look_start = max(0, pos - 252)
    if pos > look_start:
        local_end = min(pos, visible_pos + 1)
        local = df["low"].iloc[look_start:local_end]
        if not local.empty:
            fallback_iloc = int(df.index.get_loc(local.idxmin()))
            fallback_low = {"type": "L", "iloc": fallback_iloc, "value": float(local.min())}

    recent_low_price = _pivot_actual_price(df, recent_low, max_pos=structure_max_pos)
    pre_head_low_price = _pivot_actual_price(df, low_before_first_head, max_pos=structure_max_pos)
    middle_low_price = _pivot_actual_price(df, middle_low, max_pos=structure_max_pos)
    wave_start_price = _pivot_actual_price(df, wave_start, max_pos=structure_max_pos)
    fallback_low_price = _pivot_actual_price(df, fallback_low, max_pos=structure_max_pos)
    middle_vs_pre_head_pct = (
        (middle_low_price / pre_head_low_price - 1) * 100
        if middle_low_price > 0 and pre_head_low_price > 0
        else np.nan
    )

    anchor = None
    anchor_source = ""
    if low_before_first_head and middle_low:
        if middle_vs_pre_head_pct > 8:
            anchor = low_before_first_head
            anchor_source = "m_pre_head_low"
        else:
            anchor = middle_low
            anchor_source = "m_middle_reset_low"
    else:
        candidates = [
            ("wave_start", wave_start, wave_start_price),
            ("recent_zigzag_low", recent_low, recent_low_price),
            ("fallback_252d_low", fallback_low, fallback_low_price),
        ]
        valid = [(name, pivot, price) for name, pivot, price in candidates if pivot and price > 0]
        if valid:
            anchor_source, anchor, _ = min(valid, key=lambda item: item[2])

    anchor_low_price = _pivot_actual_price(df, anchor, max_pos=structure_max_pos)

    return _prefixed(prefix, {
        "recent_zigzag_low_date": _date_text(df.index[int(recent_low["iloc"])]) if recent_low else "",
        "recent_zigzag_low_price": round(recent_low_price, 4) if not math.isnan(recent_low_price) else np.nan,
        "rise_from_recent_zigzag_low_pct": round((top_price / recent_low_price - 1) * 100, 2) if recent_low_price > 0 else np.nan,
        "pre_head_low_date": _date_text(df.index[int(low_before_first_head["iloc"])]) if low_before_first_head else "",
        "pre_head_low_price": round(pre_head_low_price, 4) if not math.isnan(pre_head_low_price) else np.nan,
        "middle_low_date": _date_text(df.index[int(middle_low["iloc"])]) if middle_low else "",
        "middle_low_price": round(middle_low_price, 4) if not math.isnan(middle_low_price) else np.nan,
        "middle_vs_pre_head_pct": round(middle_vs_pre_head_pct, 2) if not math.isnan(middle_vs_pre_head_pct) else np.nan,
        "anchor_source": anchor_source,
        "anchor_is_middle_reset": int(anchor_source == "m_middle_reset_low"),
        "anchor_is_pre_head_low": int(anchor_source == "m_pre_head_low"),
        "anchor_low_date": _date_text(df.index[int(anchor["iloc"])]) if anchor else "",
        "anchor_low_price": round(anchor_low_price, 4) if not math.isnan(anchor_low_price) else np.nan,
        "rise_from_anchor_low_pct": round((top_price / anchor_low_price - 1) * 100, 2) if anchor_low_price > 0 else np.nan,
        "bars_from_anchor_low": int(pos - int(anchor["iloc"])) if anchor else np.nan,
        "wave_start_date": _date_text(df.index[int(wave_start["iloc"])]) if wave_start else "",
        "wave_start_price": round(wave_start_price, 4) if not math.isnan(wave_start_price) else np.nan,
        "major_wave_rise_pct": float(major_wave.get("rise_pct", np.nan)) if major_wave else np.nan,
        "major_wave_number": int(major_wave.get("wave_number", 0)) if major_wave else 0,
        "major_sub_wave_count": int(major_wave.get("sub_wave_count", 0)) if major_wave else 0,
        "top_cluster_high_count": len(high_cluster),
    })


def _oracle_zigzag_context(df: pd.DataFrame, row: dict, wave_result: dict) -> dict:
    values = _zigzag_context_from_wave(df, row, wave_result, prefix="oracle_")
    values["oracle_zigzag_feature_mode"] = "global_full_series"
    values["label_anchor_rise_pct"] = values.get("oracle_rise_from_anchor_low_pct", np.nan)
    return values


def _causal_zigzag_context(df: pd.DataFrame, row: dict) -> dict:
    asof_pos = _score_asof_pos(df, row)
    hist = df.iloc[:asof_pos + 1].copy()
    wave_result = analyze_wave_structure(hist)
    values = _zigzag_context_from_wave(df, row, wave_result, asof_pos=asof_pos)
    values["zigzag_feature_mode"] = "causal_asof"
    values["zigzag_feature_asof_pos"] = asof_pos
    values["zigzag_feature_asof_date"] = _date_text(df.index[asof_pos])
    return values


def _high_pivots_with_actual(df: pd.DataFrame, wave_result: dict) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for pivot in wave_result.get("all_pivots", []):
        if pivot.get("type") != "H":
            continue
        actual_pos, actual_price = _pivot_actual_high(df, pivot)
        if actual_pos is None or not math.isfinite(actual_price):
            continue
        out.append({
            "pivot": pivot,
            "pivot_pos": int(pivot["iloc"]),
            "actual_pos": int(actual_pos),
            "actual_price": float(actual_price),
        })
    return out


def _macro_zigzag_label(
    df: pd.DataFrame,
    row: dict,
    wave_result: dict | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Classify a candidate with full-series ZigZag/wave structure.

    This is intentionally an oracle label builder for training data.  It may
    use future pivots from the full series.  Realtime/model features remain
    causal and are produced separately by ``_causal_zigzag_context``.
    """

    if not wave_result:
        return {
            "structure_label": "ambiguous",
            "structure_label_reason": "no_global_wave_result",
        }

    top_pos = int(row["top_pos"])
    top_price = float(row["top_price"])
    tol_pct = float(getattr(args, "label_double_top_tolerance_pct", 2.5))
    match_bars = int(getattr(args, "label_pivot_match_bars", 6))
    tol_mult = 1 + tol_pct / 100
    lower_tol_mult = 1 - tol_pct / 100

    high_pivots = _high_pivots_with_actual(df, wave_result)
    matched = None
    if high_pivots:
        matched = sorted(
            high_pivots,
            key=lambda item: (abs(int(item["actual_pos"]) - top_pos), -float(item["actual_price"])),
        )[0]
        if abs(int(matched["actual_pos"]) - top_pos) > match_bars:
            matched = None

    major_wave = _major_wave_for_pos(wave_result, top_pos)
    if major_wave is None and matched is not None:
        major_wave = _major_wave_for_pos(wave_result, int(matched["actual_pos"]))

    structure_end_pos = len(df) - 1
    if major_wave and major_wave.get("end_pivot"):
        structure_end_pos = min(structure_end_pos, int(major_wave["end_pivot"]["iloc"]))

    next_actual_high_pos = None
    next_actual_high_price = float("nan")
    if top_pos + 1 < len(df):
        future_actual = df["high"].iloc[top_pos + 1:]
        if not future_actual.empty:
            next_actual_high_date = future_actual.idxmax()
            candidate_price = float(future_actual.loc[next_actual_high_date])
            if candidate_price > top_price * tol_mult:
                next_actual_high_pos = int(df.index.get_loc(next_actual_high_date))
                next_actual_high_price = candidate_price

    next_material_high = None
    future_highs = [
        item for item in high_pivots
        if top_pos < int(item["actual_pos"]) <= structure_end_pos
    ]
    if future_highs:
        higher = [item for item in future_highs if float(item["actual_price"]) > top_price * tol_mult]
        if higher:
            next_material_high = min(higher, key=lambda item: int(item["actual_pos"]))

    fields: dict[str, object] = {
        "structure_label_mode": "global_zigzag_macro",
        "structure_label_double_top_tolerance_pct": tol_pct,
        "structure_matched_pivot_pos": int(matched["pivot_pos"]) if matched else np.nan,
        "structure_matched_high_pos": int(matched["actual_pos"]) if matched else np.nan,
        "structure_matched_high_date": _date_text(df.index[int(matched["actual_pos"])]) if matched else "",
        "structure_matched_high_price": round(float(matched["actual_price"]), 4) if matched else np.nan,
        "structure_window_end_pos": int(structure_end_pos),
        "structure_window_end_date": _date_text(df.index[int(structure_end_pos)]),
        "structure_next_actual_high_pos": int(next_actual_high_pos) if next_actual_high_pos is not None else np.nan,
        "structure_next_actual_high_date": _date_text(df.index[int(next_actual_high_pos)]) if next_actual_high_pos is not None else "",
        "structure_next_actual_high_price": round(next_actual_high_price, 4) if math.isfinite(next_actual_high_price) else np.nan,
        "structure_next_actual_high_pct": (
            round((next_actual_high_price / top_price - 1) * 100, 2)
            if math.isfinite(next_actual_high_price) and top_price > 0
            else np.nan
        ),
        "structure_next_material_high_pos": int(next_material_high["actual_pos"]) if next_material_high else np.nan,
        "structure_next_material_high_date": _date_text(df.index[int(next_material_high["actual_pos"])]) if next_material_high else "",
        "structure_next_material_high_price": round(float(next_material_high["actual_price"]), 4) if next_material_high else np.nan,
        "structure_next_material_high_pct": (
            round((float(next_material_high["actual_price"]) / top_price - 1) * 100, 2)
            if next_material_high and top_price > 0
            else np.nan
        ),
    }

    if major_wave:
        peak_pos, peak_price = _pivot_actual_high(df, major_wave.get("peak_pivot"))
        peak_pct = (peak_price / top_price - 1) * 100 if peak_price > 0 and top_price > 0 else np.nan
        fields.update({
            "structure_major_wave_number": int(major_wave.get("wave_number", 0)),
            "structure_major_wave_peak_pos": int(peak_pos) if peak_pos is not None else np.nan,
            "structure_major_wave_peak_date": _date_text(df.index[int(peak_pos)]) if peak_pos is not None else "",
            "structure_major_wave_peak_price": round(peak_price, 4) if math.isfinite(peak_price) else np.nan,
            "structure_major_wave_peak_vs_top_pct": round(peak_pct, 2) if math.isfinite(peak_pct) else np.nan,
        })
        if next_actual_high_pos is not None:
            fields.update({
                "structure_label": "continuation",
                "structure_label_reason": "future_higher_actual_high",
            })
            return fields
        if next_material_high is not None:
            fields.update({
                "structure_label": "continuation",
                "structure_label_reason": "future_higher_zigzag_high",
            })
            return fields
        if peak_pos is not None and peak_price > top_price * tol_mult and peak_pos > top_pos:
            fields.update({
                "structure_label": "continuation",
                "structure_label_reason": "higher_macro_wave_peak_ahead",
            })
            return fields
        double_top = best_double_top_for_candidate(
            df,
            wave_result,
            top_pos,
            price_tolerance_pct=tol_pct,
            min_separation_bars=int(getattr(args, "label_double_top_min_separation_bars", 5)),
            max_separation_bars=int(getattr(args, "label_double_top_max_separation_bars", 80)),
            min_neckline_break_pct=float(getattr(args, "label_double_top_neckline_break_pct", 5.0)),
            failed_rebound_neckline_pct=float(getattr(args, "label_double_top_failed_rebound_neckline_pct", 2.0)),
        )
        fields.update({f"structure_{key}": value for key, value in double_top.items()})
        if int(double_top.get("double_top_confirmed", 0)) == 1:
            fields.update({
                "structure_label": "true_top",
                "structure_label_reason": "macro_double_top_downtrend_confirmed",
            })
            return fields
        near_macro_peak = (
            peak_pos is not None
            and math.isfinite(peak_price)
            and abs(int(peak_pos) - top_pos) <= match_bars
            and top_price >= peak_price * lower_tol_mult
        )
        near_matched_pivot = (
            matched is not None
            and top_price >= float(matched["actual_price"]) * lower_tol_mult
        )
        if near_macro_peak or near_matched_pivot:
            fields.update({
                "structure_label": "true_top",
                "structure_label_reason": "macro_wave_peak_or_double_top",
            })
            return fields
        if peak_pos is not None and peak_pos < top_pos and top_price < peak_price * lower_tol_mult:
            fields.update({
                "structure_label": "downtrend_continuation",
                "structure_label_reason": "lower_high_after_macro_wave_peak",
            })
            return fields
        fields.update({
            "structure_label": "ambiguous",
            "structure_label_reason": "inside_macro_wave_unclear",
        })
        return fields

    if next_actual_high_pos is not None:
        fields.update({
            "structure_label": "continuation",
            "structure_label_reason": "future_higher_actual_high",
        })
        return fields

    if next_material_high is not None:
        fields.update({
            "structure_label": "continuation",
            "structure_label_reason": "future_higher_zigzag_high",
        })
        return fields

    double_top = best_double_top_for_candidate(
        df,
        wave_result,
        top_pos,
        price_tolerance_pct=tol_pct,
        min_separation_bars=int(getattr(args, "label_double_top_min_separation_bars", 5)),
        max_separation_bars=int(getattr(args, "label_double_top_max_separation_bars", 80)),
        min_neckline_break_pct=float(getattr(args, "label_double_top_neckline_break_pct", 5.0)),
        failed_rebound_neckline_pct=float(getattr(args, "label_double_top_failed_rebound_neckline_pct", 2.0)),
    )
    fields.update({f"structure_{key}": value for key, value in double_top.items()})
    if int(double_top.get("double_top_confirmed", 0)) == 1:
        fields.update({
            "structure_label": "true_top",
            "structure_label_reason": "macro_double_top_downtrend_confirmed",
        })
        return fields

    if matched is not None:
        fields.update({
            "structure_label": "true_top",
            "structure_label_reason": "highest_visible_zigzag_high",
        })
        return fields

    fields.update({
        "structure_label": "ambiguous",
        "structure_label_reason": "no_matched_global_high_pivot",
    })
    return fields


def _price_context_features(df: pd.DataFrame, row: dict) -> dict:
    top_pos = int(row["top_pos"])
    confirm_pos = _score_asof_pos(df, row)
    top_price = float(row["top_price"])
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    ema = {span: close.ewm(span=span, adjust=False).mean() for span in (8, 20, 34, 55, 144, 200)}
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=5).mean()
    vol50 = volume.rolling(50, min_periods=5).mean()
    vol20 = volume.rolling(20, min_periods=5).mean()
    vol10 = volume.rolling(10, min_periods=5).mean()
    vol5 = volume.rolling(5, min_periods=3).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(14, min_periods=5).mean()
    avg_loss = loss.rolling(14, min_periods=5).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi14 = 100 - (100 / (1 + rs))
    rsi14 = rsi14.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi14 = rsi14.mask((avg_loss == 0) & (avg_gain == 0), 50)

    low14 = low.rolling(14, min_periods=5).min()
    high14 = high.rolling(14, min_periods=5).max()
    stoch14 = (close - low14) / (high14 - low14).replace(0, np.nan) * 100

    ma20 = close.rolling(20, min_periods=10).mean()
    std20 = close.rolling(20, min_periods=10).std(ddof=0)
    bb20_upper = ma20 + 2 * std20

    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    confirm_close = float(close.iloc[confirm_pos])
    signal_close = float(close.iloc[top_pos])
    recent_252_high = float(high.iloc[max(0, top_pos - 252):top_pos + 1].max())
    recent_252_low = float(low.iloc[max(0, top_pos - 252):top_pos + 1].min())
    high_20_before = float(high.iloc[max(0, top_pos - 20):top_pos + 1].max())
    low_20_before = float(low.iloc[max(0, top_pos - 20):top_pos + 1].min())

    features = {
        "confirm_close": round(confirm_close, 4),
        "signal_close": round(signal_close, 4),
        "confirm_drop_from_top_pct": round((confirm_close / top_price - 1) * 100, 2),
        "top_vs_252high_pct": round((top_price / recent_252_high - 1) * 100, 2) if recent_252_high > 0 else np.nan,
        "top_vs_252low_pct": round((top_price / recent_252_low - 1) * 100, 2) if recent_252_low > 0 else np.nan,
        "range20_before_pct": round((high_20_before / low_20_before - 1) * 100, 2) if low_20_before > 0 else np.nan,
        "vol_ratio_confirm_50": round(float(volume.iloc[confirm_pos] / vol50.iloc[confirm_pos]), 2) if vol50.iloc[confirm_pos] > 0 else np.nan,
        "atr14_pct": round(float(atr14.iloc[confirm_pos] / confirm_close * 100), 2) if confirm_close > 0 else np.nan,
    }
    for lb in (5, 10, 20, 40, 60, 120):
        features[f"prior_ret_{lb}d"] = round(_recent_return(close, top_pos, lb, top_price), 2)

    rsi_top = _coerce_float(rsi14.iloc[top_pos])
    rsi_confirm = _coerce_float(rsi14.iloc[confirm_pos])
    stoch_top = _coerce_float(stoch14.iloc[top_pos])
    ma20_top = _coerce_float(ma20.iloc[top_pos])
    std20_top = _coerce_float(std20.iloc[top_pos])
    bb20_upper_top = _coerce_float(bb20_upper.iloc[top_pos])
    macd_line_top_pct = _coerce_float(macd_line.iloc[top_pos] / top_price * 100) if top_price > 0 else np.nan
    macd_hist_top_pct = _coerce_float(macd_hist.iloc[top_pos] / top_price * 100) if top_price > 0 else np.nan
    macd_hist_confirm_pct = _coerce_float(macd_hist.iloc[confirm_pos] / confirm_close * 100) if confirm_close > 0 else np.nan

    prior_start = max(0, top_pos - 60)
    prior_end = top_pos
    prior_high_idx = None
    prior_high = np.nan
    if prior_end - prior_start >= 5:
        prior_high_window = high.iloc[prior_start:prior_end]
        if prior_high_window.notna().any():
            prior_high_label = prior_high_window.idxmax()
            prior_high_idx = int(df.index.get_loc(prior_high_label))
            prior_high = _coerce_float(high.iloc[prior_high_idx])
    made_price_high = bool(not pd.isna(prior_high) and top_price >= prior_high * 1.005)
    if prior_high_idx is not None:
        prior_rsi = _coerce_float(rsi14.iloc[prior_high_idx])
        prior_macd_line_pct = _coerce_float(macd_line.iloc[prior_high_idx] / top_price * 100) if top_price > 0 else np.nan
        prior_macd_hist_pct = _coerce_float(macd_hist.iloc[prior_high_idx] / top_price * 100) if top_price > 0 else np.nan
    else:
        prior_rsi = np.nan
        prior_macd_line_pct = np.nan
        prior_macd_hist_pct = np.nan
    rsi_divergence = prior_rsi - rsi_top if not pd.isna(prior_rsi) and not pd.isna(rsi_top) else np.nan
    macd_line_divergence = (
        prior_macd_line_pct - macd_line_top_pct
        if not pd.isna(prior_macd_line_pct) and not pd.isna(macd_line_top_pct)
        else np.nan
    )
    macd_hist_divergence = (
        prior_macd_hist_pct - macd_hist_top_pct
        if not pd.isna(prior_macd_hist_pct) and not pd.isna(macd_hist_top_pct)
        else np.nan
    )

    vol20_top = _coerce_float(vol20.iloc[top_pos])
    vol50_top = _coerce_float(vol50.iloc[top_pos])
    vol5_top = _coerce_float(vol5.iloc[top_pos])
    vol10_top = _coerce_float(vol10.iloc[top_pos])
    top_volume = _coerce_float(volume.iloc[top_pos])
    vol5_prev = _coerce_float(vol5.iloc[top_pos - 20]) if top_pos >= 20 else np.nan
    prior_ret_20d = _coerce_float(features.get("prior_ret_20d"))
    vol5_ret20_pct = (vol5_top / vol5_prev - 1) * 100 if vol5_prev > 0 else np.nan
    vol_ratio_top_20 = top_volume / vol20_top if vol20_top > 0 else np.nan
    vol_ratio_top_50 = top_volume / vol50_top if vol50_top > 0 else np.nan
    vol5_vs_vol20_top = vol5_top / vol20_top if vol20_top > 0 else np.nan
    vol10_vs_vol50_top = vol10_top / vol50_top if vol50_top > 0 else np.nan
    top_range = _coerce_float(high.iloc[top_pos] - low.iloc[top_pos])
    top_close_position_pct = ((signal_close - low.iloc[top_pos]) / top_range * 100) if top_range > 0 else np.nan
    top_upper_shadow_pct = ((high.iloc[top_pos] - max(signal_close, df["open"].astype(float).iloc[top_pos])) / top_range * 100) if top_range > 0 else np.nan
    high_volume_top = bool(not pd.isna(vol_ratio_top_50) and vol_ratio_top_50 >= 2.0)
    stall_close = bool(not pd.isna(top_close_position_pct) and top_close_position_pct <= 55)
    upper_shadow = bool(not pd.isna(top_upper_shadow_pct) and top_upper_shadow_pct >= 35)
    high_volume_stall_score = int(high_volume_top) + int(stall_close) + int(upper_shadow)

    features.update({
        "rsi14_top": _round_float(rsi_top),
        "rsi14_confirm": _round_float(rsi_confirm),
        "rsi14_overbought_70": int(rsi_top >= 70) if not pd.isna(rsi_top) else 0,
        "rsi14_overbought_80": int(rsi_top >= 80) if not pd.isna(rsi_top) else 0,
        "stoch14_top": _round_float(stoch_top),
        "stoch14_overbought_90": int(stoch_top >= 90) if not pd.isna(stoch_top) else 0,
        "bb20_zscore_top": _round_float((top_price - ma20_top) / std20_top) if std20_top > 0 else np.nan,
        "bb20_above_upper_pct": _round_float((top_price / bb20_upper_top - 1) * 100) if bb20_upper_top > 0 else np.nan,
        "macd_line_top_pct": _round_float(macd_line_top_pct, 4),
        "macd_hist_top_pct": _round_float(macd_hist_top_pct, 4),
        "macd_hist_confirm_pct": _round_float(macd_hist_confirm_pct, 4),
        "rsi14_divergence_pts_60d": _round_float(rsi_divergence),
        "rsi14_bear_div_60d": int(made_price_high and not pd.isna(rsi_divergence) and rsi_divergence >= 5),
        "macd_line_divergence_pct_60d": _round_float(macd_line_divergence, 4),
        "macd_line_bear_div_60d": int(made_price_high and not pd.isna(macd_line_divergence) and macd_line_divergence > 0),
        "macd_hist_divergence_pct_60d": _round_float(macd_hist_divergence, 4),
        "macd_hist_bear_div_60d": int(made_price_high and not pd.isna(macd_hist_divergence) and macd_hist_divergence > 0),
        "overbought_score": int(rsi_top >= 70 if not pd.isna(rsi_top) else False)
        + int(rsi_top >= 80 if not pd.isna(rsi_top) else False)
        + int(stoch_top >= 90 if not pd.isna(stoch_top) else False)
        + int(((top_price - ma20_top) / std20_top) >= 2.5 if std20_top > 0 else False),
        "vol_ratio_top_20": _round_float(vol_ratio_top_20, 2),
        "vol_ratio_top_50": _round_float(vol_ratio_top_50, 2),
        "high_volume_top_50": int(high_volume_top),
        "vol5_vs_vol20_top": _round_float(vol5_vs_vol20_top, 2),
        "vol10_vs_vol50_top": _round_float(vol10_vs_vol50_top, 2),
        "vol5_ret20_pct": _round_float(vol5_ret20_pct),
        "volume_dryup_rise20": int(prior_ret_20d >= 20 and not pd.isna(vol5_vs_vol20_top) and vol5_vs_vol20_top <= 0.8),
        "price_up_volume_down_20d": int(prior_ret_20d >= 20 and not pd.isna(vol5_ret20_pct) and vol5_ret20_pct <= -15),
        "top_close_position_pct": _round_float(top_close_position_pct),
        "top_upper_shadow_pct": _round_float(top_upper_shadow_pct),
        "high_volume_stall_score": high_volume_stall_score,
    })

    for span, series in ema.items():
        val = float(series.iloc[confirm_pos])
        features[f"dist_ema{span}_pct"] = round((confirm_close / val - 1) * 100, 2) if val > 0 else np.nan
        if span in (144, 200):
            top_val = float(series.iloc[top_pos])
            features[f"top_dist_ema{span}_pct"] = round((top_price / top_val - 1) * 100, 2) if top_val > 0 else np.nan
    if confirm_pos >= 20:
        e55_now = float(ema[55].iloc[confirm_pos])
        e55_prev = float(ema[55].iloc[confirm_pos - 20])
        features["ema55_slope_20d_pct"] = round((e55_now / e55_prev - 1) * 100, 2) if e55_prev > 0 else np.nan
    else:
        features["ema55_slope_20d_pct"] = np.nan
    return features


def _smc_structural_top_confirmed(
    top_pos: int,
    structure_events: pd.DataFrame | None,
) -> tuple[bool, int | None, str]:
    """候选顶之后，最早的「摆动级 CHoCH 向下」确认位（趋势结构真正反转）。

    见顶坐实只认 **摆动级（swing）CHoCH 向下**：价格收盘跌破已确认的摆动更高低点，
    标志上涨结构转为下跌——教科书式趋势反转。internal_1/2/3 的微观破位在每次中继
    回调都会出现、无区分力，故排除；BOS（结构已处下行的延续破位）也不算见顶首发，
    只认 CHoCH（性质由涨转跌）。这样训练标签的 true_top 尽量接近 100% 干净（样本验证：
    尺度是杠杆——any/internal 确认率 93%，swing CHoCH 仅 31%，把近端 Q2 真顶率从
    虚高的 21.9% 拉回 7.9%）。摆动 CHoCH 可能 ~10 根 K 线就成立，故快顶在近端也能
    确认；到数据末尾仍无摆动级反转的，交由调用方降级 unconfirmed。
    返回 (是否确认, 确认位, 机制名)。
    """

    if structure_events is None or structure_events.empty:
        return False, None, ""
    etype = structure_events.get("event_type")
    scale = structure_events.get("scale")
    if etype is None or scale is None:
        return False, None, ""
    direction = pd.to_numeric(structure_events.get("direction"), errors="coerce")
    broken = pd.to_numeric(structure_events.get("broken_pos"), errors="coerce")
    mask = (
        (direction == -1)
        & (etype == "choch")
        & scale.astype(str).str.startswith("swing")
        & broken.notna()
        & (broken > top_pos)
    )
    if not bool(mask.any()):
        return False, None, ""
    return True, int(broken[mask].min()), "swing_choch_down"


def _label_candidate(
    df: pd.DataFrame,
    row: dict,
    args: argparse.Namespace,
    wave_result: dict | None = None,
    smc_bundle: dict | None = None,
) -> dict:
    top_pos = int(row["top_pos"])
    confirm_pos = int(row["confirm_pos"])
    top_price = float(row["top_price"])
    end_pos = min(len(df) - 1, confirm_pos + args.lookahead_bars)

    if confirm_pos < end_pos:
        future = df.iloc[confirm_pos + 1:end_pos + 1]
        low_date = future["low"].idxmin()
        low_pos = int(df.index.get_loc(low_date))
        low_price = float(df["low"].iloc[low_pos])
        high_date = future["high"].idxmax()
        high_pos = int(df.index.get_loc(high_date))
        high_price = float(df["high"].iloc[high_pos])
        drawdown_pct = (low_price / top_price - 1) * 100
        future_high_pct = (high_price / top_price - 1) * 100
        decline_bars = low_pos - top_pos
        bars_to_new_high = high_pos - top_pos
        before_low_high = float(df["high"].iloc[top_pos + 1:low_pos + 1].max()) if top_pos + 1 <= low_pos else top_price
        pre_low_reclaim_pct = (before_low_high / top_price - 1) * 100
    else:
        low_date = pd.NaT
        low_pos = top_pos
        low_price = np.nan
        high_date = pd.NaT
        high_pos = top_pos
        high_price = np.nan
        drawdown_pct = np.nan
        future_high_pct = np.nan
        decline_bars = 0
        bars_to_new_high = 0
        pre_low_reclaim_pct = np.nan

    structure = _macro_zigzag_label(df, row, wave_result, args)
    label = str(structure.get("structure_label", "ambiguous"))
    reason = str(structure.get("structure_label_reason", "macro_unclassified"))

    # 标签确认的两段式（SMC 只替换「最近 N 根内趋势没走完、无法判断」这一段，不碰更早的点）：
    #   ‹远端› 距数据末尾 >= recent_window 根：趋势已走完，完全由全局 pivot/大浪结构判定，
    #          true_top / continuation 保持 _macro_zigzag_label 的结论，SMC 不插手。
    #   ‹近端› 最近 recent_window 根内：大结构「之后没创新高」还不可信（趋势没走完），改由
    #          SMC 结构反转确认——顶后已出现摆动级 CHoCH 向下、跌破已确认的摆动更高低点
    #          （可能 ~10 根就发生）才坐实 true_top；否则降级 unconfirmed（真正无法判断）。
    #          continuation 由「已出现更高高点」这一已发生证据确定，近端依旧可信，不进此门。
    structure_events = smc_bundle.get("structure_events") if smc_bundle else None
    struct_confirmed, struct_confirm_pos, struct_mechanism = _smc_structural_top_confirmed(
        top_pos, structure_events
    )
    structure["smc_top_structural_confirmed"] = int(struct_confirmed)
    structure["smc_top_structural_confirm_pos"] = int(struct_confirm_pos) if struct_confirm_pos is not None else np.nan
    structure["smc_top_structural_confirm_lag"] = (
        int(struct_confirm_pos - top_pos) if struct_confirm_pos is not None else np.nan
    )
    structure["smc_top_structural_confirm_mechanism"] = struct_mechanism
    recent_window = int(getattr(args, "label_recent_smc_window", 60))
    in_recent_zone = ((len(df) - 1) - top_pos) < recent_window
    structure["label_in_recent_zone"] = int(in_recent_zone)
    if label == "true_top" and in_recent_zone and not struct_confirmed:
        label = "unconfirmed"
        reason = "recent_zone_no_smc_structural_reversal"

    values = {
        "label": label,
        "label_reason": reason,
        "future_low_date": _date_text(low_date),
        "future_low_price": round(low_price, 4) if not pd.isna(low_price) else np.nan,
        "future_drawdown_pct": round(drawdown_pct, 2) if not pd.isna(drawdown_pct) else np.nan,
        "decline_bars": int(decline_bars),
        "future_high_date": _date_text(high_date),
        "future_high_price": round(high_price, 4) if not pd.isna(high_price) else np.nan,
        "future_high_pct": round(future_high_pct, 2) if not pd.isna(future_high_pct) else np.nan,
        "bars_to_future_high": int(bars_to_new_high),
        "pre_low_reclaim_pct": round(pre_low_reclaim_pct, 2) if not pd.isna(pre_low_reclaim_pct) else np.nan,
    }
    values.update(structure)
    return values


def _enrich_candidates(
    item: dict,
    df: pd.DataFrame,
    candidates: list[dict],
    wave_result: dict,
    args: argparse.Namespace,
    smc_bundle: dict | None = None,
) -> list[dict]:
    out: list[dict] = []
    for candidate in candidates:
        row = {
            "market": item["market"],
            "sym": item["symbol"],
            "name": item["name"],
            **candidate,
        }
        row.update(_price_context_features(df, row))
        row.update(_oracle_zigzag_context(df, row, wave_result))
        row.update(_causal_zigzag_context(df, row))
        row.update(_label_candidate(df, row, args, wave_result, smc_bundle=smc_bundle))
        out.append(row)
    return out


def _scan_pattern_candidates(df: pd.DataFrame, args: argparse.Namespace) -> list[dict]:
    shadow = scan_high_shadow(
        df,
        forward_days=(5, 10, 20, 60, 90),
        cooldown_days=args.strategy_cooldown,
        min_prior_rise_pct=args.strategy_min_prior_rise_pct,
    )
    doji = scan_evening_star(df, forward_days=(5, 10, 20, 60, 90))
    # gap_fail 已从主召回移除：其本质是「即刻回撤/逃顶择时」信号而非「持续顶」，
    # 前向统计显示 20 日后收涨占多数（任何切法收跌占比都 <50%），不适合作召回源。
    # 保留 strategies/impl/gap_fail_reversal.py（已收紧为高精度参数）供持仓逃顶提醒/
    # 模型确认特征单独使用。
    signals = (
        _signal_rows(shadow, "shadow", df)
        + _signal_rows(doji, "doji", df)
    )
    return _cluster_signals(signals, df, merge_bars=args.merge_bars)


def _build_symbol_research_rows(item: dict, args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    df = _normalize_df(item["df"])
    if len(df) < args.min_history or "volume" not in df.columns:
        return [], [], [], [], [], [], []

    pattern_candidates = _scan_pattern_candidates(df, args)
    smc_bundle = build_smc_bundle(df)
    raw_ob_events = smc_bundle["raw_ob_events"]
    include_smc_raw_recall = bool(getattr(args, "include_smc_raw_recall", False))
    include_smc_confirmed_recall = bool(getattr(args, "include_smc_confirmed_recall", False))
    smc_raw_candidates = (
        collect_smc_raw_candidates(
            df,
            raw_ob_events,
            min_raw_score=args.smc_raw_min_score,
            max_confirm_lag=args.smc_early_max_confirm_lag,
            high_lookback=args.smc_early_high_lookback,
            near_high_pct=args.smc_early_near_high_pct,
            min_prior_ret_20d=args.smc_early_min_prior_ret_20d,
            merge_bars=args.merge_bars,
        )
        if include_smc_raw_recall
        else []
    )
    smc_confirmed_candidates = (
        collect_smc_confirmed_candidates(
            df,
            raw_ob_events,
            min_score=args.smc_confirmed_min_score,
            min_raw_score=args.smc_raw_min_score,
            high_lookback=args.smc_early_high_lookback,
            near_high_pct=args.smc_early_near_high_pct,
            min_prior_ret_20d=args.smc_early_min_prior_ret_20d,
        )
        if include_smc_confirmed_recall
        else []
    )
    smc_appear_candidates = collect_smc_appear_candidates(
        df,
        raw_ob_events,
        max_confirm_lag=args.smc_appear_max_confirm_lag,
        high_lookback=args.smc_early_high_lookback,
        near_high_pct=args.smc_early_near_high_pct,
        min_prior_ret_20d=args.smc_early_min_prior_ret_20d,
        min_leave_pct=args.smc_appear_min_leave_pct,
        merge_bars=args.merge_bars,
    )
    smc_early_candidates = collect_smc_early_candidates(
        df,
        raw_ob_events,
        smc_bundle["ob_events"],
        smc_bundle["structure_events"],
        min_raw_presence_score=args.smc_raw_min_score,
        min_raw_score=args.smc_early_min_raw_score,
        min_struct_score=args.smc_early_min_struct_score,
        min_total_score=args.smc_early_min_total_score,
        max_confirm_lag=args.smc_early_max_confirm_lag,
        high_lookback=args.smc_early_high_lookback,
        near_high_pct=args.smc_early_near_high_pct,
        min_prior_ret_20d=args.smc_early_min_prior_ret_20d,
        merge_bars=args.merge_bars,
    )
    # smc_appear 已从主召回移除：它只借用 OB 的几何外壳（用 K 线高低当 zone），
    # 触发条件「3 天内跌破信号 K 线自己的低点」既无位移也无结构破坏，本质是
    # 「近高位即刻回落」探测器，在上涨途中每个中继高点都会误报（见 02788 的
    # 03-11/12/13）。保留独立 dataset 仅作诊断，不进入统一训练候选。
    #
    # 统一 SMC 顶部源 smc_top_confirmed：对每个近高位锚点，取 supply_held /
    # early / confirmed 三种确认机制中「最早确认」者（min lag）。三者互补，
    # 并集真顶覆盖 ~74% > 任一单源 ~70%。确认机制与 lag 作为特征交给模型，
    # confirmed 虽晚但最可靠、由模型给更高先验权重。
    smc_top_confirmed_candidates = collect_smc_top_confirmed_candidates(
        df,
        raw_ob_events,
        smc_bundle["ob_events"],
        smc_bundle["structure_events"],
        supply_held_min_anchor_score=args.smc_raw_min_score,
        high_lookback=args.smc_early_high_lookback,
        near_high_pct=args.smc_early_near_high_pct,
        min_prior_ret_20d=args.smc_early_min_prior_ret_20d,
        merge_bars=args.merge_bars,
        smc_early_kwargs=dict(
            min_raw_presence_score=args.smc_raw_min_score,
            min_raw_score=args.smc_early_min_raw_score,
            min_struct_score=args.smc_early_min_struct_score,
            min_total_score=args.smc_early_min_total_score,
            max_confirm_lag=args.smc_early_max_confirm_lag,
        ),
    )
    unified_candidates = merge_recall_candidates(
        [*pattern_candidates, *smc_raw_candidates, *smc_top_confirmed_candidates],
        df,
        merge_bars=args.merge_bars,
    )
    wave_result = analyze_wave_structure(df)
    universe_candidates = collect_zigzag_peak_candidates(df, wave_result)

    pattern_rows = _enrich_candidates(item, df, pattern_candidates, wave_result, args, smc_bundle=smc_bundle)
    smc_confirmed_rows = _enrich_candidates(item, df, smc_confirmed_candidates, wave_result, args, smc_bundle=smc_bundle)
    smc_raw_rows = _enrich_candidates(item, df, smc_raw_candidates, wave_result, args, smc_bundle=smc_bundle)
    smc_appear_rows = _enrich_candidates(item, df, smc_appear_candidates, wave_result, args, smc_bundle=smc_bundle)
    smc_early_rows = _enrich_candidates(item, df, smc_early_candidates, wave_result, args, smc_bundle=smc_bundle)
    unified_rows = _enrich_candidates(item, df, unified_candidates, wave_result, args, smc_bundle=smc_bundle)
    universe_rows = _enrich_candidates(item, df, universe_candidates, wave_result, args, smc_bundle=smc_bundle)
    return pattern_rows, smc_confirmed_rows, smc_raw_rows, smc_appear_rows, smc_early_rows, unified_rows, universe_rows


def _format_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date_cols = [
        "signal_date", "confirm_date", "top_date", "candidate_confirm_date",
        "score_asof_date", "zigzag_feature_asof_date", "zigzag_pivot_date", "future_low_date", "future_high_date",
        "recent_zigzag_low_date", "pre_head_low_date", "middle_low_date",
        "anchor_low_date", "wave_start_date",
        "oracle_recent_zigzag_low_date", "oracle_pre_head_low_date", "oracle_middle_low_date",
        "oracle_anchor_low_date", "oracle_wave_start_date",
        "smc_appear_recall_formed_date", "smc_appear_recall_appear_date",
        "smc_confirmed_recall_formed_date", "smc_confirmed_recall_confirmed_date",
        "structure_double_top_pair_date", "structure_double_top_first_head_date",
        "structure_double_top_second_head_date", "structure_double_top_neckline_date",
        "structure_double_top_break_low_date", "structure_double_top_failed_rebound_date",
        "structure_double_top_confirm_date",
    ]
    for col in date_cols:
        if col not in out.columns:
            continue
        values = pd.to_datetime(out[col], errors="coerce")
        out[col] = values.dt.strftime("%Y-%m-%d").fillna("")
    return out


def _fit_per_market(dataset: pd.DataFrame, cols: list[str], fit_fn) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按市场分别训练并打分（CN牛/HK熊/US 股性不同，永不合并）。

    每个市场只在自身样本上训练 + 打分，市场专属特征权重自然涌现（如 M顶在 CN
    自动趋零）。返回 (coef_with_market, scored_with_market_model)。
    """
    coef_parts, scored_parts = [], []
    for mk in sorted(dataset["market"].dropna().astype(str).unique()):
        grp = dataset[dataset["market"].astype(str) == mk]
        coef_m, scored_m = fit_fn(grp, cols)
        if not coef_m.empty:
            coef_m = coef_m.copy(); coef_m.insert(0, "market_model", mk); coef_parts.append(coef_m)
        if not scored_m.empty:
            scored_m = scored_m.copy(); scored_m["market_model"] = mk; scored_parts.append(scored_m)
    coef = pd.concat(coef_parts, ignore_index=True) if coef_parts else pd.DataFrame()
    scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    return coef, scored


def _score_perf_per_market(scored: pd.DataFrame) -> pd.DataFrame:
    """各市场分别算分数段表现（避免跨市场 top_prob 标度不一致的混淆）。"""
    if scored.empty or "market_model" not in scored.columns:
        return score_performance(scored)
    parts = []
    for mk, g in scored.groupby("market_model"):
        perf = score_performance(g)
        if not perf.empty:
            perf = perf.copy(); perf.insert(0, "market", mk); parts.append(perf)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _research_bundle(dataset: pd.DataFrame, model_feature_cols: list[str] | None = None, per_market: bool = False) -> dict[str, pd.DataFrame]:
    summary = label_summary(dataset)
    feature_diff_df = feature_diff(dataset, FEATURE_COLS)
    buckets = bucket_stats(dataset, BUCKET_COLS)
    cols = model_feature_cols or FEATURE_COLS
    use_pm = per_market and "market" in dataset.columns and dataset["market"].astype(str).nunique() > 1
    if use_pm:
        coef, scored = _fit_per_market(dataset, cols, fit_logistic)
        lgb_importance, lgb_scored = _fit_per_market(dataset, cols, fit_lightgbm)
        score_perf = _score_perf_per_market(scored)
        lgb_score_perf = _score_perf_per_market(lgb_scored) if not lgb_scored.empty else pd.DataFrame()
    else:
        coef, scored = fit_logistic(dataset, cols)
        lgb_importance, lgb_scored = fit_lightgbm(dataset, cols)
        score_perf = score_performance(scored)
        lgb_score_perf = score_performance(lgb_scored) if not lgb_scored.empty else pd.DataFrame()
    return {
        "summary": summary,
        "feature_diff": feature_diff_df,
        "buckets": buckets,
        "coef": coef,
        "scored": scored,
        "score_perf": score_perf,
        "lgb_coef": lgb_importance,
        "lgb_scored": lgb_scored,
        "lgb_score_perf": lgb_score_perf,
    }


def _filter_mid_vegas_dataset(dataset: pd.DataFrame, args: argparse.Namespace, label: str) -> pd.DataFrame:
    if dataset.empty or not getattr(args, "require_mid_vegas_uptrend", True):
        return dataset
    if "mid_vegas_passed" not in dataset.columns:
        logger.warning(f"{label} 缺少 mid_vegas_passed，Mid Vegas 过滤后为空")
        return dataset.iloc[0:0].copy()
    passed = pd.to_numeric(dataset["mid_vegas_passed"], errors="coerce").fillna(0).astype(int).eq(1)
    filtered = dataset[passed].copy()
    logger.info(f"{label} Mid Vegas 严格上涨趋势过滤: {len(dataset)} -> {len(filtered)}")
    return filtered


def _write_research_outputs(
    *,
    title: str,
    dataset: pd.DataFrame,
    bundle: dict[str, pd.DataFrame],
    dataset_csv: Path,
    summary_md: Path,
    feature_diff_csv: Path,
    feature_diff_md: Path,
    buckets_csv: Path,
    coef_csv: Path,
    scored_csv: Path,
    score_perf_csv: Path,
) -> None:
    dataset.to_csv(dataset_csv, index=False, encoding="utf-8-sig")
    bundle["feature_diff"].to_csv(feature_diff_csv, index=False, encoding="utf-8-sig")
    bundle["buckets"].to_csv(buckets_csv, index=False, encoding="utf-8-sig")
    if not bundle["coef"].empty:
        bundle["coef"].to_csv(coef_csv, index=False, encoding="utf-8-sig")
        bundle["scored"].to_csv(scored_csv, index=False, encoding="utf-8-sig")
        bundle["score_perf"].to_csv(score_perf_csv, index=False, encoding="utf-8-sig")
    # lightgbm 并行输出（与 LR 同名规则，避免撞名）
    if bundle.get("lgb_scored") is not None and not bundle["lgb_scored"].empty:
        def _lgbm_path(p: Path) -> Path:
            s = str(p)
            if "logistic" in s:
                return Path(s.replace("logistic", "lgbm"))
            if "score_performance" in s:
                return Path(s.replace("score_performance", "lgbm_score_performance"))
            return Path(s.replace(".csv", "_lgbm.csv"))
        bundle["lgb_coef"].to_csv(_lgbm_path(coef_csv), index=False, encoding="utf-8-sig")
        bundle["lgb_scored"].to_csv(_lgbm_path(scored_csv), index=False, encoding="utf-8-sig")
        bundle["lgb_score_perf"].to_csv(_lgbm_path(score_perf_csv), index=False, encoding="utf-8-sig")

    feature_groups = pd.DataFrame(feature_group_summary())
    summary_lines = [
        f"# {title}",
        "",
        "## Feature Groups",
        "",
        to_markdown_table(feature_groups),
        "## Label Counts",
        "",
        to_markdown_table(bundle["summary"]),
        "## Top Feature Differences",
        "",
        to_markdown_table(bundle["feature_diff"].head(30)),
    ]
    if not bundle["coef"].empty:
        summary_lines.extend([
            "## Lightweight Logistic Coefficients",
            "",
            to_markdown_table(bundle["coef"].head(30)),
            "## Score Band Performance",
            "",
            to_markdown_table(bundle["score_perf"]),
        ])
    summary_md.write_text("\n".join(summary_lines), encoding="utf-8")
    feature_diff_md.write_text(to_markdown_table(bundle["feature_diff"]), encoding="utf-8")


def _smc_model_comparison(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare baseline and staged SMC feature sets on pattern candidates."""

    base_cols = NON_SMC_FEATURE_COLS
    feature_sets = {
        "base_no_smc": base_cols,
        "base_plus_smc_live": base_cols + list(SMC_LIVE_FEATURES),
        "base_plus_smc_raw": base_cols + list(SMC_RAW_FEATURES),
        "base_plus_smc_early": base_cols + list(SMC_EARLY_FEATURES),
        "base_plus_smc_raw_early": base_cols + list(SMC_RAW_FEATURES) + list(SMC_EARLY_FEATURES),
        "base_plus_smc_causal": base_cols + list(SMC_CAUSAL_FEATURES),
        "base_plus_smc_live_delayed": base_cols + list(SMC_LIVE_FEATURES) + list(SMC_DELAYED_FEATURES),
        "base_plus_smc_all": base_cols + list(SMC_LIVE_FEATURES) + list(SMC_DELAYED_FEATURES) + list(SMC_DIAGNOSTIC_FEATURES),
    }

    perf_rows: list[dict[str, object]] = []
    coef_parts: list[pd.DataFrame] = []
    for name, cols in feature_sets.items():
        coef, scored = fit_logistic(dataset, cols)
        perf = score_performance(scored)
        for _, row in perf.iterrows():
            perf_rows.append({"model": name, **row.to_dict()})
        if not coef.empty:
            coef = coef.copy()
            coef.insert(0, "model", name)
            coef_parts.append(coef)

    perf_df = pd.DataFrame(perf_rows)
    coef_df = pd.concat(coef_parts, ignore_index=True) if coef_parts else pd.DataFrame()
    return perf_df, coef_df


def _score_performance_vs_universe(
    scored: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    near_bars: int,
) -> pd.DataFrame:
    """Evaluate selected model signals against unique recalled universe true tops."""

    if scored.empty or universe.empty or "top_prob" not in scored.columns:
        return pd.DataFrame()

    true_universe = universe[universe["label"] == "true_top"].copy()
    if "covered_by_recall" in true_universe.columns:
        covered = pd.to_numeric(true_universe["covered_by_recall"], errors="coerce").fillna(0) > 0
        true_universe = true_universe[covered].copy()
    if true_universe.empty:
        return pd.DataFrame()

    true_universe = true_universe.reset_index(drop=True)
    true_universe["_true_top_uid"] = np.arange(len(true_universe))
    ceiling = int(len(true_universe))

    scoreable = scored[pd.to_numeric(scored["top_prob"], errors="coerce").notna()].copy()
    if scoreable.empty:
        return pd.DataFrame()
    scoreable["top_prob"] = pd.to_numeric(scoreable["top_prob"], errors="coerce")

    grouped_universe = {
        (str(market), str(sym)): group.copy()
        for (market, sym), group in true_universe.groupby(["market", "sym"], observed=True)
    }

    def evaluate(selected: pd.DataFrame, score_band: str, threshold: float) -> dict[str, object]:
        matched: set[int] = set()
        hit_rows = 0
        duplicate_hits = 0
        for _, row in selected.iterrows():
            key = (str(row.get("market", "")), str(row.get("sym", "")))
            group = grouped_universe.get(key)
            if group is None or group.empty:
                continue
            top_pos = int(row.get("top_pos", -10**9))
            distances = (pd.to_numeric(group["top_pos"], errors="coerce") - top_pos).abs()
            matches = group[distances <= near_bars]
            if matches.empty:
                continue
            hit_rows += 1
            nearest = matches.loc[distances.loc[matches.index].sort_values().index[0]]
            uid = int(nearest["_true_top_uid"])
            duplicate_hits += int(uid in matched)
            matched.add(uid)

        y = int(len(selected))
        z = int(len(matched))
        return {
            "score_band": score_band,
            "threshold": round(float(threshold), 4),
            "signals_y": y,
            "matched_true_tops_z": z,
            "recall_ceiling_true_tops": ceiling,
            "precision_z_over_y": round(z / y * 100, 1) if y else np.nan,
            "recall_z_over_ceiling": round(z / ceiling * 100, 1) if ceiling else np.nan,
            "row_hit_rate": round(hit_rows / y * 100, 1) if y else np.nan,
            "duplicate_hits": int(duplicate_hits),
        }

    rows: list[dict[str, object]] = []
    for quantile in (0.90, 0.80, 0.70, 0.60, 0.50):
        threshold = float(scoreable["top_prob"].quantile(quantile))
        selected = scoreable[scoreable["top_prob"] >= threshold]
        rows.append(evaluate(selected, f"top_{int((1 - quantile) * 100)}pct", threshold))

    for top_n in (100, 200, 300, 450, 600, 900):
        selected = scoreable.sort_values("top_prob", ascending=False).head(top_n)
        if selected.empty:
            continue
        threshold = float(selected["top_prob"].min())
        rows.append(evaluate(selected, f"top_{len(selected)}", threshold))
    return pd.DataFrame(rows)


_SAMPLE_NAMES = ("pattern", "smc_confirmed", "smc_raw", "smc_appear", "smc_early", "unified", "universe")


def _stage1_select_samples(data: dict, args: argparse.Namespace) -> tuple[tuple[pd.DataFrame, ...], int]:
    """Stage 1：逐股扫描召回 + ZigZag 顶部全集 + 打标签 → 7 个样本集 DataFrame（不含模型特征）。"""
    pattern_rows: list[dict] = []
    smc_confirmed_rows: list[dict] = []
    smc_raw_rows: list[dict] = []
    smc_appear_rows: list[dict] = []
    smc_early_rows: list[dict] = []
    unified_rows: list[dict] = []
    universe_rows: list[dict] = []
    failed = 0
    for i, item in enumerate(data.values(), 1):
        try:
            (
                symbol_pattern_rows, symbol_smc_rows, symbol_smc_raw_rows, symbol_smc_appear_rows,
                symbol_smc_early_rows, symbol_unified_rows, symbol_universe_rows,
            ) = _build_symbol_research_rows(item, args)
            pattern_rows.extend(symbol_pattern_rows)
            smc_confirmed_rows.extend(symbol_smc_rows)
            smc_raw_rows.extend(symbol_smc_raw_rows)
            smc_appear_rows.extend(symbol_smc_appear_rows)
            smc_early_rows.extend(symbol_smc_early_rows)
            unified_rows.extend(symbol_unified_rows)
            universe_rows.extend(symbol_universe_rows)
            if symbol_unified_rows or symbol_universe_rows or symbol_pattern_rows:
                logger.info(
                    f"[{i}/{len(data)}] {item['market']}:{item['symbol']} "
                    f"形态 {len(symbol_pattern_rows)} / SMC appear {len(symbol_smc_appear_rows)} / "
                    f"SMC early {len(symbol_smc_early_rows)} / 统一 {len(symbol_unified_rows)} / "
                    f"顶部全集 {len(symbol_universe_rows)}"
                )
        except Exception as exc:
            failed += 1
            logger.warning(f"{item['market']}:{item['symbol']} 失败: {exc}")
        if i % 50 == 0 or i == len(data):
            logger.info(
                f"进度 [{i}/{len(data)}] 形态={len(pattern_rows)} 统一={len(unified_rows)} "
                f"顶部全集={len(universe_rows)} 失败={failed}"
            )
    return (
        pd.DataFrame(pattern_rows), pd.DataFrame(smc_confirmed_rows), pd.DataFrame(smc_raw_rows),
        pd.DataFrame(smc_appear_rows), pd.DataFrame(smc_early_rows), pd.DataFrame(unified_rows),
        pd.DataFrame(universe_rows),
    ), failed


def _save_samples(samples_dir: Path, frames: tuple[pd.DataFrame, ...]) -> None:
    """Stage 1 产物（召回+标签的样本集，不含模型特征）落盘，作为 Stage 1→2 的接口。"""
    samples_dir.mkdir(parents=True, exist_ok=True)
    for name, df in zip(_SAMPLE_NAMES, frames, strict=True):
        df.to_csv(samples_dir / f"{name}.csv", index=False, encoding="utf-8-sig")


def _load_samples(samples_dir: Path) -> tuple[pd.DataFrame, ...]:
    missing = [n for n in _SAMPLE_NAMES if not (samples_dir / f"{n}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"样本集缺失 {missing}，请先跑 `--stage sample`。目录: {samples_dir}")

    def _read(name: str) -> pd.DataFrame:
        try:  # sym 必须按字符串读——否则 HK/CN 代码("00700")丢前导零变 700，按(market,sym)匹配
            return pd.read_csv(samples_dir / f"{name}.csv", low_memory=False, dtype={"sym": str})
        except pd.errors.EmptyDataError:  # 被禁用的数据集(smc_confirmed/smc_raw)写出的是空文件
            return pd.DataFrame()

    return tuple(_read(n) for n in _SAMPLE_NAMES)


def _filter_sample_markets(frames: tuple[pd.DataFrame, ...], markets: set[str]) -> tuple[pd.DataFrame, ...]:
    if not markets or markets == {"US", "HK", "CN"}:
        return frames
    out = []
    for df in frames:
        if df.empty or "market" not in df.columns:
            out.append(df)
            continue
        mask = df["market"].astype(str).str.upper().isin(markets)
        out.append(df[mask].copy())
    return tuple(out)


def build_arg_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器（抽出以便扫描等脚本复用默认召回/特征参数：build_arg_parser().parse_args([])）。"""
    parser = argparse.ArgumentParser(description="构建 watchlist 顶部候选研究数据集并做标签/特征统计")
    parser.add_argument("--watchlist", type=Path, default=DATA_DIR / "lists" / "watchlist.md")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="研究输出目录，默认 data/output/top_candidate_research")
    parser.add_argument("--min-history", type=int, default=120)
    parser.add_argument("--merge-bars", type=int, default=3)
    parser.add_argument("--strategy-cooldown", type=int, default=0, help="候选研究默认不做冷却，保留所有形态")
    parser.add_argument("--strategy-min-prior-rise-pct", type=float, default=0.0, help="候选研究默认不使用长上影策略内部前涨幅过滤")
    parser.add_argument("--lookahead-bars", type=int, default=90)
    parser.add_argument("--label-double-top-tolerance-pct", type=float, default=2.5, help="宏观结构标签中双顶/同一顶部的价格容差")
    parser.add_argument("--label-double-top-min-separation-bars", type=int, default=5, help="宏观双顶两头之间的最小 bar 数")
    parser.add_argument("--label-double-top-max-separation-bars", type=int, default=80, help="宏观双顶两头之间的最大 bar 数")
    parser.add_argument("--label-double-top-neckline-break-pct", type=float, default=5.0, help="宏观双顶确认要求跌破颈线的最小幅度")
    parser.add_argument("--label-double-top-failed-rebound-neckline-pct", type=float, default=2.0, help="宏观双顶确认中反弹高点允许高出颈线的幅度")
    parser.add_argument("--label-pivot-match-bars", type=int, default=6, help="候选高点匹配全局 ZigZag 高点的最大 bar 距离")
    parser.add_argument(
        "--label-recent-smc-window",
        type=int,
        default=60,
        help="近端区边界（根）：距数据末尾不足该根数的 true_top 视为趋势未走完、大结构不可信，改由 SMC 结构反转确认（确认→true_top，否则→unconfirmed）；更早的点完全由全局 pivot 结构判定，不受 SMC 影响",
    )
    parser.add_argument(
        "--smc-confirmed-min-score",
        dest="smc_confirmed_min_score",
        type=float,
        default=20.0,
        help="SMC confirmed 召回使用的 bearish OB 最低质量分",
    )
    parser.add_argument("--smc-origin-min-score", dest="smc_confirmed_min_score", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--smc-raw-min-score", type=float, default=20.0, help="SMC raw 召回使用的最低 raw setup 分数")
    parser.add_argument("--smc-early-max-confirm-lag", type=int, default=3, help="SMC early 最多等待几根K线确认")
    parser.add_argument("--smc-appear-max-confirm-lag", type=int, default=3, help="SMC appear 最多等待几根K线离开OB区间")
    parser.add_argument("--smc-appear-min-leave-pct", type=float, default=0.0, help="SMC appear 离开OB区间的最小幅度")
    parser.add_argument(
        "--smc-early-min-raw-score",
        dest="smc_early_min_raw_score",
        type=float,
        default=45.0,
        help="SMC early raw 证据最低分",
    )
    parser.add_argument("--smc-early-min-origin-score", dest="smc_early_min_raw_score", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--smc-early-min-struct-score", type=float, default=35.0, help="SMC early 结构证据最低分")
    parser.add_argument("--smc-early-min-total-score", type=float, default=55.0, help="SMC early 综合证据最低分")
    parser.add_argument("--smc-early-high-lookback", type=int, default=60, help="SMC early 近高位过滤窗口")
    parser.add_argument("--smc-early-near-high-pct", type=float, default=3.0, help="SMC early 候选距离近高位最大百分比")
    parser.add_argument("--smc-early-min-prior-ret-20d", type=float, default=5.0, help="SMC early 候选前20日最低涨幅")
    parser.add_argument(
        "--require-mid-vegas-uptrend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="样本入口是否强制要求 Mid Vegas 严格上涨趋势；默认开启，可用 --no-require-mid-vegas-uptrend 做旧口径对照",
    )
    parser.add_argument(
        "--include-smc-raw-recall",
        action="store_true",
        help="将 smc_raw 原始候选也加入统一召回；默认关闭，仅用于对照实验",
    )
    parser.add_argument(
        "--include-smc-confirmed-recall",
        action="store_true",
        help="将 smc_confirmed 最终确认候选也加入统一召回；默认关闭，仅用于对照实验",
    )
    parser.add_argument(
        "--train-universe",
        choices=["tech", "watchlist"],
        default="tech",
        help="训练宇宙：tech=三市场每日 Mid Vegas 科技池(us_tech/hk_techman/cn_hightech)+持仓，按市场分离训练；watchlist=旧口径",
    )
    parser.add_argument(
        "--per-market-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="按市场分组各训一套 LR/lgb（CN/HK/US 股性不同，永不合并）；--no-per-market-model 退回全局单模型",
    )
    parser.add_argument(
        "--markets",
        default="US,HK,CN",
        help="只处理这些市场（逗号分隔，如 US 或 HK,CN）。US/HK/CN 本就分市场独立建模，"
             "分市场单独跑可把内存峰值压到单市场量级、并提供分阶段 checkpoint。默认三市场全跑",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "sample", "model"],
        default="all",
        help="两段式：sample=只跑 Stage1(扫描召回+打标签)→落盘样本集；model=跳过扫描、读样本集只跑 Stage2"
             "(特征+训练+评估)；all=两段连跑(默认)。改召回/标签才需重跑 sample；只调特征/模型用 model 即可，省去重扫",
    )
    return parser


def main() -> None:
    global OUT_DIR
    args = build_arg_parser().parse_args()
    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.train_universe == "tech":
        data = load_tech_pools_data(min_history=args.min_history, include_holding=True)
        logger.info(f"训练宇宙=三市场科技池+持仓，共 {len(data)} 只")
    else:
        data = load_watchlist_data(path=args.watchlist, min_history=args.min_history)
    sel_markets = {m.strip().upper() for m in str(args.markets).split(",") if m.strip()}
    if sel_markets and sel_markets != {"US", "HK", "CN"}:
        before = len(data)
        data = {k: v for k, v in data.items() if str(v.get("market", "")).upper() in sel_markets}
        logger.info(f"按市场过滤 {sorted(sel_markets)}: {before} -> {len(data)} 只（分市场跑以控内存）")
    samples_dir = OUT_DIR / "_samples"
    failed = 0
    if args.stage == "model":
        logger.info(f"Stage2(仅特征+训练+评估)：从 {samples_dir} 读入样本集，跳过扫描/标签")
        (pattern_dataset, smc_confirmed_dataset, smc_raw_dataset, smc_appear_dataset,
         smc_early_dataset, dataset, universe) = _load_samples(samples_dir)
    else:
        logger.info(f"Stage1(扫描召回+打标签)：{len(data)} 只，开始 ...")
        (pattern_dataset, smc_confirmed_dataset, smc_raw_dataset, smc_appear_dataset,
         smc_early_dataset, dataset, universe), failed = _stage1_select_samples(data, args)
        _save_samples(samples_dir, (pattern_dataset, smc_confirmed_dataset, smc_raw_dataset,
                                    smc_appear_dataset, smc_early_dataset, dataset, universe))
        logger.info(f"Stage1 完成：样本集落盘 {samples_dir}（统一候选 {len(dataset)} / 顶部全集 {len(universe)}）")
        if args.stage == "sample":
            return  # 只选样本，特征+训练交给 `--stage model`
    (pattern_dataset, smc_confirmed_dataset, smc_raw_dataset, smc_appear_dataset,
     smc_early_dataset, dataset, universe) = _filter_sample_markets(
        (pattern_dataset, smc_confirmed_dataset, smc_raw_dataset, smc_appear_dataset,
         smc_early_dataset, dataset, universe),
        sel_markets,
    )

    if pattern_dataset.empty and smc_confirmed_dataset.empty and smc_raw_dataset.empty and smc_appear_dataset.empty and smc_early_dataset.empty:
        logger.error("没有任何召回候选")
        return
    if dataset.empty:
        logger.error("没有任何统一召回候选")
        return
    if universe.empty:
        logger.error("没有任何 ZigZag 顶部全集候选")
        return

    dataset = add_research_features(dataset, symbol_data=data)
    dataset = _filter_mid_vegas_dataset(dataset, args, "统一召回候选")
    if dataset.empty:
        logger.error("Mid Vegas 过滤后没有任何统一召回候选")
        return
    dataset = _format_date_columns(dataset)
    dataset = dataset.sort_values(["market", "sym", "confirm_date", "strategies"]).reset_index(drop=True)
    unified_bundle = _research_bundle(dataset, model_feature_cols=PRIMARY_MODEL_FEATURE_COLS, per_market=args.per_market_model)
    smc_model_perf, smc_model_coef = _smc_model_comparison(dataset)

    pattern_dataset = add_research_features(pattern_dataset, symbol_data=data) if not pattern_dataset.empty else pattern_dataset
    pattern_dataset = _filter_mid_vegas_dataset(pattern_dataset, args, "形态候选")
    pattern_dataset = _format_date_columns(pattern_dataset)
    pattern_dataset = pattern_dataset.sort_values(["market", "sym", "confirm_date", "strategies"]).reset_index(drop=True) if not pattern_dataset.empty else pattern_dataset
    pattern_bundle = _research_bundle(pattern_dataset, model_feature_cols=PRIMARY_MODEL_FEATURE_COLS) if not pattern_dataset.empty else {}

    smc_confirmed_dataset = add_research_features(smc_confirmed_dataset, symbol_data=data) if not smc_confirmed_dataset.empty else smc_confirmed_dataset
    smc_confirmed_dataset = _filter_mid_vegas_dataset(smc_confirmed_dataset, args, "SMC confirmed候选")
    smc_confirmed_dataset = _format_date_columns(smc_confirmed_dataset)
    smc_confirmed_dataset = smc_confirmed_dataset.sort_values(["market", "sym", "confirm_date", "strategies"]).reset_index(drop=True) if not smc_confirmed_dataset.empty else smc_confirmed_dataset
    smc_raw_dataset = add_research_features(smc_raw_dataset, symbol_data=data) if not smc_raw_dataset.empty else smc_raw_dataset
    smc_raw_dataset = _filter_mid_vegas_dataset(smc_raw_dataset, args, "SMC raw候选")
    smc_raw_dataset = _format_date_columns(smc_raw_dataset)
    smc_raw_dataset = smc_raw_dataset.sort_values(["market", "sym", "confirm_date", "strategies"]).reset_index(drop=True) if not smc_raw_dataset.empty else smc_raw_dataset
    smc_appear_dataset = add_research_features(smc_appear_dataset, symbol_data=data) if not smc_appear_dataset.empty else smc_appear_dataset
    smc_appear_dataset = _filter_mid_vegas_dataset(smc_appear_dataset, args, "SMC appear候选")
    smc_appear_dataset = _format_date_columns(smc_appear_dataset)
    smc_appear_dataset = smc_appear_dataset.sort_values(["market", "sym", "confirm_date", "strategies"]).reset_index(drop=True) if not smc_appear_dataset.empty else smc_appear_dataset
    smc_early_dataset = add_research_features(smc_early_dataset, symbol_data=data) if not smc_early_dataset.empty else smc_early_dataset
    smc_early_dataset = _filter_mid_vegas_dataset(smc_early_dataset, args, "SMC early候选")
    smc_early_dataset = _format_date_columns(smc_early_dataset)
    smc_early_dataset = smc_early_dataset.sort_values(["market", "sym", "confirm_date", "strategies"]).reset_index(drop=True) if not smc_early_dataset.empty else smc_early_dataset

    universe = attach_strategy_matches(universe, dataset, near_bars=args.merge_bars)
    universe = add_research_features(universe, symbol_data=data)
    universe = _filter_mid_vegas_dataset(universe, args, "顶部全集")
    if universe.empty:
        logger.error("Mid Vegas 过滤后没有任何顶部全集候选")
        return
    universe = _format_date_columns(universe)
    universe = universe.sort_values(["market", "sym", "top_date", "strategies"]).reset_index(drop=True)
    universe_bundle = _research_bundle(universe, model_feature_cols=PRIMARY_MODEL_FEATURE_COLS, per_market=args.per_market_model)
    coverage = strategy_coverage_report(universe)
    unified_universe_score_perf = _score_performance_vs_universe(
        unified_bundle["scored"],
        universe,
        near_bars=args.merge_bars,
    )

    covered_flag = pd.to_numeric(universe["covered_by_recall"], errors="coerce").fillna(0)
    missed_universe = universe[(universe["label"] == "true_top") & (covered_flag == 0)].copy()
    missed_universe = missed_universe.sort_values(["market", "sym", "top_date"]).reset_index(drop=True)

    unified_dataset_csv = OUT_DIR / "watchlist_unified_recall_candidates_labeled.csv"
    pattern_dataset_csv = OUT_DIR / "watchlist_pattern_candidates_labeled.csv"
    smc_confirmed_dataset_csv = OUT_DIR / "watchlist_smc_confirmed_recall_candidates_labeled.csv"
    smc_raw_dataset_csv = OUT_DIR / "watchlist_smc_raw_recall_candidates_labeled.csv"
    smc_appear_dataset_csv = OUT_DIR / "watchlist_smc_appear_recall_candidates_labeled.csv"
    smc_early_dataset_csv = OUT_DIR / "watchlist_smc_early_recall_candidates_labeled.csv"
    summary_md = OUT_DIR / "top_candidate_label_summary.md"
    feature_diff_csv = OUT_DIR / "top_candidate_feature_diff.csv"
    feature_diff_md = OUT_DIR / "top_candidate_feature_diff.md"
    buckets_csv = OUT_DIR / "top_candidate_bucket_stats.csv"
    coef_csv = OUT_DIR / "top_candidate_logistic_coefficients.csv"
    scored_csv = OUT_DIR / "top_candidate_logistic_scored.csv"
    score_perf_csv = OUT_DIR / "top_candidate_score_performance.csv"
    universe_recall_score_perf_csv = OUT_DIR / "top_candidate_universe_recall_score_performance.csv"
    smc_model_perf_csv = OUT_DIR / "top_candidate_smc_model_comparison.csv"
    smc_model_coef_csv = OUT_DIR / "top_candidate_smc_model_comparison_coefficients.csv"

    universe_dataset_csv = OUT_DIR / "watchlist_universe_candidates_labeled.csv"
    universe_missed_csv = OUT_DIR / "universe_true_tops_missed_by_recall.csv"
    universe_missed_by_patterns_csv = OUT_DIR / "universe_true_tops_missed_by_patterns.csv"
    coverage_csv = OUT_DIR / "recall_coverage_by_true_top.csv"
    universe_summary_md = OUT_DIR / "universe_top_candidate_label_summary.md"
    universe_feature_diff_csv = OUT_DIR / "universe_top_candidate_feature_diff.csv"
    universe_feature_diff_md = OUT_DIR / "universe_top_candidate_feature_diff.md"
    universe_buckets_csv = OUT_DIR / "universe_top_candidate_bucket_stats.csv"
    universe_coef_csv = OUT_DIR / "universe_top_candidate_logistic_coefficients.csv"
    universe_scored_csv = OUT_DIR / "universe_top_candidate_logistic_scored.csv"
    universe_score_perf_csv = OUT_DIR / "universe_top_candidate_score_performance.csv"

    _write_research_outputs(
        title="Unified Recall Top Candidate Research Summary",
        dataset=dataset,
        bundle=unified_bundle,
        dataset_csv=unified_dataset_csv,
        summary_md=summary_md,
        feature_diff_csv=feature_diff_csv,
        feature_diff_md=feature_diff_md,
        buckets_csv=buckets_csv,
        coef_csv=coef_csv,
        scored_csv=scored_csv,
        score_perf_csv=score_perf_csv,
    )
    pattern_dataset.to_csv(pattern_dataset_csv, index=False, encoding="utf-8-sig")
    smc_confirmed_dataset.to_csv(smc_confirmed_dataset_csv, index=False, encoding="utf-8-sig")
    smc_raw_dataset.to_csv(smc_raw_dataset_csv, index=False, encoding="utf-8-sig")
    smc_appear_dataset.to_csv(smc_appear_dataset_csv, index=False, encoding="utf-8-sig")
    smc_early_dataset.to_csv(smc_early_dataset_csv, index=False, encoding="utf-8-sig")
    unified_universe_score_perf.to_csv(universe_recall_score_perf_csv, index=False, encoding="utf-8-sig")
    smc_model_perf.to_csv(smc_model_perf_csv, index=False, encoding="utf-8-sig")
    smc_model_coef.to_csv(smc_model_coef_csv, index=False, encoding="utf-8-sig")

    _write_research_outputs(
        title="Universe Top Candidate Research Summary",
        dataset=universe,
        bundle=universe_bundle,
        dataset_csv=universe_dataset_csv,
        summary_md=universe_summary_md,
        feature_diff_csv=universe_feature_diff_csv,
        feature_diff_md=universe_feature_diff_md,
        buckets_csv=universe_buckets_csv,
        coef_csv=universe_coef_csv,
        scored_csv=universe_scored_csv,
        score_perf_csv=universe_score_perf_csv,
    )
    missed_universe.to_csv(universe_missed_csv, index=False, encoding="utf-8-sig")
    pattern_covered_flag = pd.to_numeric(universe["covered_by_patterns"], errors="coerce").fillna(0)
    universe[(universe["label"] == "true_top") & (pattern_covered_flag == 0)].copy().sort_values(["market", "sym", "top_date"]).to_csv(
        universe_missed_by_patterns_csv,
        index=False,
        encoding="utf-8-sig",
    )
    coverage.to_csv(coverage_csv, index=False, encoding="utf-8-sig")
    if not coverage.empty:
        universe_summary = universe_summary_md.read_text(encoding="utf-8")
        universe_summary_md.write_text(
            universe_summary + "\n## Recall Coverage On Universe True Tops\n\n" + to_markdown_table(coverage),
            encoding="utf-8",
        )
    if not unified_universe_score_perf.empty:
        summary = summary_md.read_text(encoding="utf-8")
        summary_md.write_text(
            summary
            + "\n## Model Performance Against Recalled Universe True Tops\n\n"
            + to_markdown_table(unified_universe_score_perf),
            encoding="utf-8",
        )
    print("\n" + "═" * 78)
    print("  Watchlist 顶部候选研究数据集")
    print("═" * 78)
    print(f"  可用股票:   {len(data)}")
    print(f"  形态候选:   {len(pattern_dataset)}")
    print(f"  SMC confirmed候选:  {len(smc_confirmed_dataset)}")
    print(f"  SMC raw候选: {len(smc_raw_dataset)}")
    print(f"  SMC appear候选: {len(smc_appear_dataset)}")
    print(f"  SMC early候选: {len(smc_early_dataset)}")
    print(f"  统一候选:   {len(dataset)}")
    print(f"  顶部全集:   {len(universe)}")
    print(f"  失败:       {failed}")

    print("\n  统一召回候选标签分布:")
    print(unified_bundle["summary"].to_string(index=False))
    if not unified_bundle["feature_diff"].empty:
        print("\n  统一召回候选：真顶 vs 上涨中继差异最大的特征:")
        print(unified_bundle["feature_diff"].head(12).to_string(index=False))
    if not unified_bundle["coef"].empty:
        print("\n  统一召回候选：轻量 logistic 权重 Top:")
        print(unified_bundle["coef"].head(12).to_string(index=False))
        print("\n  统一召回候选：高分段真顶率:")
        print(unified_bundle["score_perf"].to_string(index=False))
    if not unified_universe_score_perf.empty:
        print("\n  主模型：以当前召回可触达真顶为分母的表现:")
        print(unified_universe_score_perf.to_string(index=False))
    if not smc_model_perf.empty:
        print("\n  统一召回候选：SMC 特征分层对比:")
        print(smc_model_perf.to_string(index=False))

    print("\n  顶部全集标签分布:")
    print(universe_bundle["summary"].to_string(index=False))
    if not coverage.empty:
        print("\n  召回策略对顶部全集真顶的覆盖:")
        print(coverage.to_string(index=False))
    print(f"\n  统一候选数据集:   {unified_dataset_csv}")
    print(f"  形态候选数据集:   {pattern_dataset_csv}")
    print(f"  SMC confirmed候选数据集:  {smc_confirmed_dataset_csv}")
    print(f"  SMC raw候选数据集: {smc_raw_dataset_csv}")
    print(f"  SMC appear候选数据集: {smc_appear_dataset_csv}")
    print(f"  SMC early候选数据集: {smc_early_dataset_csv}")
    print(f"  顶部全集数据集:   {universe_dataset_csv}")
    print(f"  漏召回真顶:       {universe_missed_csv}")
    print(f"  蜡烛图漏召回真顶: {universe_missed_by_patterns_csv}")
    print(f"  召回覆盖率:       {coverage_csv}")
    print(f"  主候选汇总报告:   {summary_md}")
    print(f"  顶部全集汇总报告: {universe_summary_md}")
    if not unified_bundle["coef"].empty:
        print(f"  主候选打分:       {scored_csv}")
        print(f"  主候选分数表现:   {score_perf_csv}")
        print(f"  主候选全集召回表现: {universe_recall_score_perf_csv}")
        print(f"  SMC 分层对比:     {smc_model_perf_csv}")
    if not universe_bundle["coef"].empty:
        print(f"  顶部全集候选打分: {universe_scored_csv}")
        print(f"  顶部全集分数表现: {universe_score_perf_csv}")


if __name__ == "__main__":
    main()
