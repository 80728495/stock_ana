"""Candidate-source builders for top-reversal research.

Candidate sources answer "which bars should be studied or scored?". They are
kept separate from feature builders so recall experiments can evolve without
entangling model inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ana.research.top_reversal.double_top import find_double_top_patterns
from stock_ana.research.top_reversal.feature_registry import apply_legacy_feature_aliases
from stock_ana.research.top_reversal.smc_context import attach_smc_early_states

RECALL_FLAG_DEFAULTS = {
    "recalled_by_candle": 0,
    "recalled_by_shadow": 0,
    "recalled_by_doji": 0,
    "recalled_by_gap_fail": 0,
    "recalled_by_double_top": 0,
    "recalled_by_smc_raw": 0,
    "recalled_by_smc_appear": 0,
    "recalled_by_smc_confirmed": 0,
    "recalled_by_smc_early": 0,
    "recalled_by_smc_supply_held": 0,
    "recall_source_count": 0,
    "score_lag_bars": 0,
    "smc_raw_recall_count": 0,
    "smc_raw_recall_score_max": 0.0,
    "smc_raw_recall_detect_lag_min": float("nan"),
    "smc_raw_recall_confirm_lag_min": float("nan"),
    "smc_appear_recall_count": 0,
    "smc_appear_recall_confirm_lag_min": float("nan"),
    "smc_appear_recall_leave_pct_max": 0.0,
    "smc_appear_recall_zone_width_pct": float("nan"),
    "smc_confirmed_recall_count": 0,
    "smc_confirmed_recall_score_max": 0.0,
    "smc_confirmed_recall_struct_score_max": 0.0,
    "smc_confirmed_recall_confirm_lag_min": float("nan"),
    "smc_confirmed_recall_zone_width_pct": float("nan"),
    "smc_confirmed_recall_volume_ratio": float("nan"),
    "smc_early_recall_count": 0,
    "smc_early_recall_score_max": 0.0,
    "smc_early_recall_raw_score_max": 0.0,
    "smc_early_recall_struct_score_max": 0.0,
    "smc_early_recall_confirm_lag_min": float("nan"),
    "smc_supply_held_recall_count": 0,
    "smc_supply_held_recall_anchor_score_max": 0.0,
    "smc_supply_held_recall_confirm_lag_min": float("nan"),
    "smc_supply_held_recall_break_low_pct": float("nan"),
    "double_top_recall_count": 0,
    "double_top_recall_confirm_lag_min": float("nan"),
    "double_top_recall_neckline_break_pct": float("nan"),
    "double_top_recall_failed_rebound_vs_neckline_pct": float("nan"),
}

SOURCE_ALIASES = {
    "smc_origin": "smc_confirmed",
}


def date_text(ts) -> str:
    if pd.isna(ts):
        return ""
    return str(pd.Timestamp(ts).date())


def pivot_actual_high_pos(df: pd.DataFrame, pivot: dict, radius: int = 2) -> int:
    """Return the actual high bar near an EMA/ZigZag high pivot."""

    pos = int(pivot["iloc"])
    start = max(0, pos - radius)
    end = min(len(df) - 1, pos + radius)
    local = df["high"].iloc[start:end + 1]
    return int(df.index.get_loc(local.idxmax()))


def collect_zigzag_peak_candidates(
    df: pd.DataFrame,
    wave_result: dict,
    *,
    min_top_pos: int = 0,
) -> list[dict]:
    """Collect all meaningful ZigZag high pivots as universe candidates."""

    pivots = wave_result.get("all_pivots", [])
    if not pivots:
        return []

    rows: list[dict] = []
    seen_top_pos: set[int] = set()
    for i, pivot in enumerate(pivots):
        if pivot.get("type") != "H":
            continue
        top_pos = pivot_actual_high_pos(df, pivot)
        if top_pos < min_top_pos or top_pos in seen_top_pos:
            continue
        seen_top_pos.add(top_pos)

        next_pivot = next((p for p in pivots[i + 1:] if int(p.get("iloc", -1)) > int(pivot["iloc"])), None)
        candidate_confirm_pos = int(next_pivot["iloc"]) if next_pivot else top_pos
        candidate_confirm_pos = max(top_pos, min(len(df) - 1, candidate_confirm_pos))

        rows.append({
            "candidate_source": "zigzag_peak",
            "signal_date": pd.Timestamp(df.index[top_pos]),
            "confirm_date": pd.Timestamp(df.index[top_pos]),
            "signal_pos": top_pos,
            "confirm_pos": top_pos,
            "top_date": pd.Timestamp(df.index[top_pos]),
            "top_pos": top_pos,
            "top_price": float(df["high"].iloc[top_pos]),
            "strategies": "zigzag_peak",
            "has_shadow": 0,
            "has_doji": 0,
            "has_gap_fail": 0,
            "signal_count": 0,
            "score_max": 0,
            "score_sum": 0,
            "confirm_modes": "",
            "signal_dates": "",
            "zigzag_pivot_pos": int(pivot["iloc"]),
            "zigzag_pivot_date": date_text(df.index[int(pivot["iloc"])]),
            "candidate_confirm_date": date_text(df.index[candidate_confirm_pos]),
            "candidate_confirm_pos": candidate_confirm_pos,
            "candidate_confirm_lag": int(candidate_confirm_pos - top_pos),
            "score_asof_pos": top_pos,
            "score_asof_date": date_text(df.index[top_pos]),
        })
    return sorted(rows, key=lambda row: int(row["top_pos"]))


def collect_double_top_candidates(
    df: pd.DataFrame,
    wave_result: dict,
    *,
    min_top_pos: int = 0,
    price_tolerance_pct: float = 2.5,
    min_separation_bars: int = 5,
    max_separation_bars: int = 80,
    min_neckline_break_pct: float = 5.0,
    failed_rebound_neckline_pct: float = 2.0,
) -> list[dict]:
    """Collect confirmed structural double-top candidates.

    This is a structural recall source, but the main research build does not
    include it by default.  Callers can opt in once the detector has been
    validated for the target market/style.
    """

    rows = find_double_top_patterns(
        df,
        wave_result,
        min_top_pos=min_top_pos,
        price_tolerance_pct=price_tolerance_pct,
        min_separation_bars=min_separation_bars,
        max_separation_bars=max_separation_bars,
        min_neckline_break_pct=min_neckline_break_pct,
        failed_rebound_neckline_pct=failed_rebound_neckline_pct,
    )
    out: list[dict] = []
    for row in rows:
        top_pos = int(row["top_pos"])
        score_asof_pos = int(row.get("score_asof_pos", row.get("confirm_pos", top_pos)))
        item = {
            **recall_flags_for_sources({"double_top"}, top_pos=top_pos, score_asof_pos=score_asof_pos),
            **row,
            "double_top_recall_count": 1,
            "double_top_recall_confirm_lag_min": int(score_asof_pos - top_pos),
            "double_top_recall_neckline_break_pct": row.get("double_top_break_neckline_pct", float("nan")),
            "double_top_recall_failed_rebound_vs_neckline_pct": row.get(
                "double_top_failed_rebound_vs_neckline_pct",
                float("nan"),
            ),
        }
        out.append(item)
    return sorted(out, key=lambda row: (int(row["top_pos"]), int(row["score_asof_pos"])))


def _split_sources(value) -> set[str]:
    if pd.isna(value):
        return set()
    sources = set()
    for part in str(value).replace(",", "+").split("+"):
        if not part or part == "nan":
            continue
        sources.add(SOURCE_ALIASES.get(part, part))
    return sources


def _as_int_flag(value) -> int:
    if pd.isna(value):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _row_sources(row: dict | pd.Series) -> set[str]:
    sources = _split_sources(row.get("strategies", ""))
    sources |= _split_sources(row.get("candidate_source", ""))
    if _as_int_flag(row.get("has_shadow", 0)):
        sources.add("shadow")
    if _as_int_flag(row.get("has_doji", 0)):
        sources.add("doji")
    if _as_int_flag(row.get("has_gap_fail", 0)):
        sources.add("gap_fail")
    if _as_int_flag(row.get("recalled_by_double_top", 0)):
        sources.add("double_top")
    if _as_int_flag(row.get("recalled_by_smc_raw", 0)):
        sources.add("smc_raw")
    if _as_int_flag(row.get("recalled_by_smc_appear", 0)):
        sources.add("smc_appear")
    if _as_int_flag(row.get("recalled_by_smc_confirmed", row.get("recalled_by_smc_origin", 0))):
        sources.add("smc_confirmed")
    if _as_int_flag(row.get("recalled_by_smc_early", 0)):
        sources.add("smc_early")
    if _as_int_flag(row.get("recalled_by_smc_supply_held", 0)):
        sources.add("smc_supply_held")
    return sources


def recall_flags_for_sources(sources: set[str], *, top_pos: int, score_asof_pos: int) -> dict[str, object]:
    sources = {SOURCE_ALIASES.get(source, source) for source in sources}
    flags = dict(RECALL_FLAG_DEFAULTS)
    flags["recalled_by_shadow"] = int("shadow" in sources)
    flags["recalled_by_doji"] = int("doji" in sources)
    flags["recalled_by_gap_fail"] = int("gap_fail" in sources)
    flags["recalled_by_candle"] = int(bool({"shadow", "doji", "gap_fail"} & sources))
    flags["recalled_by_double_top"] = int("double_top" in sources)
    flags["recalled_by_smc_raw"] = int("smc_raw" in sources)
    flags["recalled_by_smc_appear"] = int("smc_appear" in sources)
    flags["recalled_by_smc_confirmed"] = int("smc_confirmed" in sources)
    flags["recalled_by_smc_early"] = int("smc_early" in sources)
    flags["recalled_by_smc_supply_held"] = int("smc_supply_held" in sources)
    flags["recall_source_count"] = int(len(sources - {"zigzag_peak"}))
    flags["score_lag_bars"] = int(max(0, score_asof_pos - top_pos))
    return flags


def _numeric_values(rows: list[dict], col: str) -> pd.Series:
    values = [row.get(col) for row in rows if col in row]
    if not values:
        return pd.Series(dtype=float)
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _max_numeric(rows: list[dict], col: str, default=0.0):
    values = _numeric_values(rows, col).dropna()
    return default if values.empty else values.max()


def _min_numeric(rows: list[dict], col: str, default=float("nan")):
    values = _numeric_values(rows, col).dropna()
    return default if values.empty else values.min()


def _joined_values(rows: list[dict], col: str) -> str:
    vals = []
    for row in rows:
        value = row.get(col, "")
        if pd.isna(value) or str(value) in {"", "nan"}:
            continue
        vals.extend(str(value).split(","))
    return ",".join(dict.fromkeys(v for v in vals if v))


def _as_float(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(out) else out


def _as_pos(value, default: int = -1) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _passes_bearish_top_context(
    df: pd.DataFrame,
    *,
    top_pos: int,
    high_lookback: int,
    near_high_pct: float,
    min_prior_ret_20d: float,
) -> bool:
    if top_pos < high_lookback or top_pos >= len(df):
        return False
    high = df["high"].astype(float)
    close = df["close"].astype(float)
    top_price = float(high.iloc[top_pos])
    look_start = max(0, top_pos - high_lookback)
    recent_high = float(high.iloc[look_start:top_pos + 1].max())
    if recent_high <= 0 or top_price < recent_high * (1 - near_high_pct / 100):
        return False
    if top_pos >= 20:
        prior_base = float(close.iloc[top_pos - 20])
        prior_ret_20d = (top_price / prior_base - 1) * 100 if prior_base > 0 else np.nan
        if pd.notna(prior_ret_20d) and prior_ret_20d < min_prior_ret_20d:
            return False
    return True


def _cluster_best(rows: list[dict], *, merge_bars: int, score_col: str) -> list[dict]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: (int(row["top_pos"]), -float(row.get(score_col, 0.0))))
    out: list[dict] = []
    cluster: list[dict] = []
    cluster_end = -1
    for row in ordered:
        top_pos = int(row["top_pos"])
        if not cluster or top_pos <= cluster_end:
            cluster.append(row)
            cluster_end = max(cluster_end, top_pos + merge_bars)
            continue
        out.append(max(cluster, key=lambda r: (float(r.get(score_col, 0.0)), float(r.get("top_price", 0.0)))))
        cluster = [row]
        cluster_end = top_pos + merge_bars
    if cluster:
        out.append(max(cluster, key=lambda r: (float(r.get(score_col, 0.0)), float(r.get("top_price", 0.0)))))
    return sorted(out, key=lambda row: (int(row["top_pos"]), int(row["score_asof_pos"])))


def collect_smc_confirmed_candidates(
    df: pd.DataFrame,
    raw_ob_events: pd.DataFrame,
    *,
    min_top_pos: int = 0,
    min_score: float = 20.0,
    min_raw_score: float = 20.0,
    high_lookback: int = 60,
    near_high_pct: float = 3.0,
    min_prior_ret_20d: float = 5.0,
    formed_radius: int = 1,
    raw_candidates: list[dict] | None = None,
    raw_match_bars: int = 3,
) -> list[dict]:
    """Collect final-confirmed bearish OB states from the raw OB-zone table."""

    del formed_radius, raw_candidates, raw_match_bars
    if raw_ob_events.empty:
        return []

    rows: list[dict] = []
    seen: set[tuple[int, int]] = set()
    volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(dtype=float)
    vol20 = volume.rolling(20, min_periods=5).mean() if not volume.empty else pd.Series(dtype=float)
    confirmed_pos_values = pd.to_numeric(raw_ob_events.get("confirmed_pos", pd.Series(dtype=float)), errors="coerce")
    for idx, event in raw_ob_events.iterrows():
        if int(event.get("direction", 0)) != -1:
            continue
        if pd.isna(confirmed_pos_values.loc[idx]):
            continue
        score = _as_float(event.get("confirmed_score", event.get("score", 0.0)), 0.0)
        if score < min_score:
            continue
        origin_pos = int(event["origin_pos"])
        confirmed_pos = int(confirmed_pos_values.loc[idx])
        if origin_pos < min_top_pos or confirmed_pos < origin_pos or confirmed_pos >= len(df):
            continue
        raw_score = _as_float(event.get("raw_score", 0.0), 0.0)
        if raw_score < min_raw_score:
            continue
        if not _passes_bearish_top_context(
            df,
            top_pos=origin_pos,
            high_lookback=high_lookback,
            near_high_pct=near_high_pct,
            min_prior_ret_20d=min_prior_ret_20d,
        ):
            continue

        top_pos = origin_pos
        key = (top_pos, confirmed_pos)
        if key in seen:
            continue
        seen.add(key)

        top_price = float(df["high"].iloc[top_pos])
        zone_top = _as_float(event.get("confirmed_zone_top", event.get("top", float("nan"))))
        zone_bottom = _as_float(event.get("confirmed_zone_bottom", event.get("bottom", float("nan"))))
        zone_width_pct = (
            (zone_top / zone_bottom - 1) * 100
            if pd.notna(zone_top) and pd.notna(zone_bottom) and zone_bottom > 0
            else float("nan")
        )
        vol_base = float(vol20.iloc[confirmed_pos]) if not vol20.empty and pd.notna(vol20.iloc[confirmed_pos]) and vol20.iloc[confirmed_pos] > 0 else float("nan")
        volume_ratio = (
            _as_float(event.get("confirmed_ob_volume", float("nan"))) / (vol_base * 3)
            if pd.notna(vol_base) and vol_base > 0
            else float("nan")
        )
        struct_score = _as_float(event.get("confirmed_struct_score", score), score)
        confirm_lag = int(confirmed_pos - origin_pos)

        rows.append({
            **recall_flags_for_sources({"smc_confirmed"}, top_pos=top_pos, score_asof_pos=confirmed_pos),
            "candidate_source": "smc_confirmed",
            "signal_date": pd.Timestamp(df.index[origin_pos]),
            "confirm_date": pd.Timestamp(df.index[confirmed_pos]),
            "signal_pos": origin_pos,
            "confirm_pos": confirmed_pos,
            "top_date": pd.Timestamp(df.index[top_pos]),
            "top_pos": top_pos,
            "top_price": top_price,
            "strategies": "smc_confirmed",
            "has_shadow": 0,
            "has_doji": 0,
            "has_gap_fail": 0,
            "signal_count": 0,
            "score_max": 0,
            "score_sum": 0,
            "confirm_modes": "smc_confirmed",
            "signal_dates": date_text(df.index[origin_pos]),
            "score_asof_pos": confirmed_pos,
            "score_asof_date": date_text(df.index[confirmed_pos]),
            "smc_raw_recall_count": 1,
            "smc_raw_recall_score_max": round(raw_score, 2),
            "smc_raw_recall_detect_lag_min": int(_as_pos(event.get("detected_pos", origin_pos), origin_pos) - origin_pos),
            "smc_raw_recall_confirm_lag_min": 0,
            "smc_raw_recall_raw_id": str(event.get("raw_id", "")),
            "smc_raw_recall_zone_top": round(_as_float(event.get("top", float("nan"))), 4) if pd.notna(event.get("top", float("nan"))) else float("nan"),
            "smc_raw_recall_zone_bottom": round(_as_float(event.get("bottom", float("nan"))), 4) if pd.notna(event.get("bottom", float("nan"))) else float("nan"),
            "smc_raw_recall_raw_reason": str(event.get("raw_reason", "")),
            "smc_confirmed_recall_count": 1,
            "smc_confirmed_recall_score_max": round(score, 2),
            "smc_confirmed_recall_struct_score_max": round(struct_score, 2),
            "smc_confirmed_recall_confirm_lag_min": confirm_lag,
            "smc_confirmed_recall_zone_width_pct": round(float(zone_width_pct), 2) if pd.notna(zone_width_pct) else float("nan"),
            "smc_confirmed_recall_volume_ratio": round(float(volume_ratio), 2) if pd.notna(volume_ratio) else float("nan"),
            "smc_confirmed_recall_formed_date": date_text(df.index[origin_pos]),
            "smc_confirmed_recall_confirmed_date": date_text(df.index[confirmed_pos]),
            "smc_confirmed_recall_zone_top": round(zone_top, 4) if pd.notna(zone_top) else float("nan"),
            "smc_confirmed_recall_zone_bottom": round(zone_bottom, 4) if pd.notna(zone_bottom) else float("nan"),
            "smc_confirmed_recall_matched_raw": 1,
            "smc_confirmed_recall_raw_top_pos": top_pos,
            "smc_confirmed_recall_raw_id": str(event.get("raw_id", "")),
        })
    return sorted(rows, key=lambda row: (int(row["top_pos"]), int(row["score_asof_pos"])))


def collect_smc_raw_candidates(
    df: pd.DataFrame,
    raw_ob_events: pd.DataFrame,
    structure_events: pd.DataFrame | None = None,
    *,
    min_top_pos: int = 0,
    max_confirm_lag: int = 3,
    min_raw_score: float = 20.0,
    high_lookback: int = 60,
    near_high_pct: float = 3.0,
    min_prior_ret_20d: float = 5.0,
    merge_bars: int = 3,
) -> list[dict]:
    """Collect bearish raw potential OB zones near recent highs."""

    del structure_events, max_confirm_lag
    if df.empty or raw_ob_events.empty or len(df) < max(high_lookback, 20) + 2:
        return []

    high = df["high"].astype(float)
    raw_rows: list[dict] = []
    for _, event in raw_ob_events.iterrows():
        if int(event.get("direction", 0)) != -1:
            continue
        top_pos = _as_pos(event.get("origin_pos", event.get("formed_pos", -1)))
        if top_pos < min_top_pos or top_pos >= len(df):
            continue
        raw_score = _as_float(event.get("raw_score", event.get("score", 0.0)), 0.0)
        if raw_score < min_raw_score:
            continue
        if not _passes_bearish_top_context(
            df,
            top_pos=top_pos,
            high_lookback=high_lookback,
            near_high_pct=near_high_pct,
            min_prior_ret_20d=min_prior_ret_20d,
        ):
            continue

        detected_pos = _as_pos(event.get("detected_pos", top_pos), top_pos)
        detected_pos = max(top_pos, min(len(df) - 1, detected_pos))
        top_price = float(high.iloc[top_pos])
        raw_rows.append({
            **recall_flags_for_sources({"smc_raw"}, top_pos=top_pos, score_asof_pos=detected_pos),
            "candidate_source": "smc_raw",
            "signal_date": pd.Timestamp(df.index[detected_pos]),
            "confirm_date": pd.Timestamp(df.index[detected_pos]),
            "signal_pos": detected_pos,
            "confirm_pos": detected_pos,
            "top_date": pd.Timestamp(df.index[top_pos]),
            "top_pos": top_pos,
            "top_price": top_price,
            "strategies": "smc_raw",
            "has_shadow": 0,
            "has_doji": 0,
            "has_gap_fail": 0,
            "signal_count": 0,
            "score_max": 0,
            "score_sum": 0,
            "confirm_modes": "smc_raw",
            "signal_dates": date_text(df.index[detected_pos]),
            "score_asof_pos": detected_pos,
            "score_asof_date": date_text(df.index[detected_pos]),
            "smc_raw_recall_count": 1,
            "smc_raw_recall_score_max": round(raw_score, 2),
            "smc_raw_recall_detect_lag_min": int(detected_pos - top_pos),
            "smc_raw_recall_confirm_lag_min": int(detected_pos - top_pos),
            "smc_raw_recall_has_fvg": int(_as_float(event.get("early_has_fvg", 0), 0.0) > 0),
            "smc_raw_recall_has_sweep": int(_as_float(event.get("early_has_sweep", 0), 0.0) > 0),
            "smc_raw_recall_zone_overlap_top": 1.0,
            "smc_raw_recall_displacement_atr": float("nan"),
            "smc_raw_recall_zone_width_atr": _as_float(event.get("zone_width_atr", float("nan"))),
            "smc_raw_recall_volume_ratio": _as_float(event.get("volume_ratio", float("nan"))),
            "smc_raw_recall_zone_top": round(_as_float(event.get("top", float("nan"))), 4),
            "smc_raw_recall_zone_bottom": round(_as_float(event.get("bottom", float("nan"))), 4),
            "smc_raw_recall_raw_id": str(event.get("raw_id", "")),
            "smc_raw_recall_raw_reason": str(event.get("raw_reason", "")),
            "smc_raw_recall_prior_ret_20d": _as_float(event.get("prior_ret_20d", float("nan"))),
            "smc_raw_recall_near_high_pct": _as_float(event.get("near_high_pct", float("nan"))),
        })

    del merge_bars
    return sorted(raw_rows, key=lambda row: (int(row["top_pos"]), int(row["score_asof_pos"])))


def collect_smc_appear_candidates(
    df: pd.DataFrame,
    raw_ob_events: pd.DataFrame,
    *,
    min_top_pos: int = 0,
    max_confirm_lag: int = 3,
    high_lookback: int = 60,
    near_high_pct: float = 3.0,
    min_prior_ret_20d: float = 5.0,
    min_leave_pct: float = 0.0,
    merge_bars: int = 3,
) -> list[dict]:
    """Collect OB zones that quickly appear by leaving their price range.

    This source is deliberately not score-gated.  For bearish top research,
    a potential supply zone "appears" once a later low moves below the
    zone bottom within the causal confirmation window.
    """

    if df.empty or raw_ob_events.empty or len(df) < max(high_lookback, 20) + max_confirm_lag + 2:
        return []

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    raw_rows: list[dict] = []
    for _, event in raw_ob_events.iterrows():
        if int(event.get("direction", 0)) != -1:
            continue
        top_pos = _as_pos(event.get("origin_pos", event.get("formed_pos", -1)))
        if top_pos < min_top_pos or top_pos >= len(df) - 1:
            continue
        if not _passes_bearish_top_context(
            df,
            top_pos=top_pos,
            high_lookback=high_lookback,
            near_high_pct=near_high_pct,
            min_prior_ret_20d=min_prior_ret_20d,
        ):
            continue

        zone_top = _as_float(event.get("top", float("nan")))
        zone_bottom = _as_float(event.get("bottom", float("nan")))
        if pd.isna(zone_top) or pd.isna(zone_bottom) or zone_bottom <= 0 or zone_top <= zone_bottom:
            continue

        max_pos = min(len(df) - 1, top_pos + max_confirm_lag)
        appear_pos: int | None = None
        leave_pct = float("nan")
        for asof_pos in range(top_pos + 1, max_pos + 1):
            leave_price = float(low.iloc[asof_pos])
            if leave_price <= 0 or leave_price >= zone_bottom:
                continue
            current_leave_pct = (zone_bottom / leave_price - 1.0) * 100.0
            if current_leave_pct < min_leave_pct:
                continue
            appear_pos = asof_pos
            leave_pct = current_leave_pct
            break
        if appear_pos is None:
            continue

        top_price = float(high.iloc[top_pos])
        zone_width_pct = (zone_top / zone_bottom - 1.0) * 100.0
        raw_score = _as_float(event.get("raw_score", event.get("score", 0.0)), 0.0)
        raw_rows.append({
            **recall_flags_for_sources({"smc_appear"}, top_pos=top_pos, score_asof_pos=appear_pos),
            "candidate_source": "smc_appear",
            "signal_date": pd.Timestamp(df.index[top_pos]),
            "confirm_date": pd.Timestamp(df.index[appear_pos]),
            "signal_pos": top_pos,
            "confirm_pos": appear_pos,
            "top_date": pd.Timestamp(df.index[top_pos]),
            "top_pos": top_pos,
            "top_price": top_price,
            "strategies": "smc_appear",
            "has_shadow": 0,
            "has_doji": 0,
            "has_gap_fail": 0,
            "signal_count": 0,
            "score_max": 0,
            "score_sum": 0,
            "confirm_modes": "smc_appear",
            "signal_dates": date_text(df.index[top_pos]),
            "score_asof_pos": appear_pos,
            "score_asof_date": date_text(df.index[appear_pos]),
            "smc_raw_recall_count": 1,
            "smc_raw_recall_score_max": round(raw_score, 2),
            "smc_raw_recall_detect_lag_min": int(_as_pos(event.get("detected_pos", top_pos), top_pos) - top_pos),
            "smc_raw_recall_confirm_lag_min": int(appear_pos - top_pos),
            "smc_raw_recall_zone_top": round(zone_top, 4),
            "smc_raw_recall_zone_bottom": round(zone_bottom, 4),
            "smc_raw_recall_raw_id": str(event.get("raw_id", "")),
            "smc_raw_recall_raw_reason": str(event.get("raw_reason", "")),
            "smc_raw_recall_prior_ret_20d": _as_float(event.get("prior_ret_20d", float("nan"))),
            "smc_raw_recall_near_high_pct": _as_float(event.get("near_high_pct", float("nan"))),
            "smc_appear_recall_count": 1,
            "smc_appear_recall_confirm_lag_min": int(appear_pos - top_pos),
            "smc_appear_recall_leave_pct_max": round(float(leave_pct), 2),
            "smc_appear_recall_zone_width_pct": round(float(zone_width_pct), 2),
            "smc_appear_recall_formed_date": date_text(df.index[top_pos]),
            "smc_appear_recall_appear_date": date_text(df.index[appear_pos]),
            "smc_appear_recall_raw_id": str(event.get("raw_id", "")),
        })

    return _cluster_best(raw_rows, merge_bars=merge_bars, score_col="smc_appear_recall_leave_pct_max")


def collect_smc_early_candidates(
    df: pd.DataFrame,
    raw_ob_events: pd.DataFrame,
    ob_events: pd.DataFrame | None = None,
    structure_events: pd.DataFrame | None = None,
    *,
    min_top_pos: int = 0,
    max_confirm_lag: int = 3,
    min_raw_presence_score: float = 20.0,
    min_raw_score: float = 45.0,
    min_struct_score: float = 35.0,
    min_total_score: float = 55.0,
    high_lookback: int = 60,
    near_high_pct: float = 3.0,
    min_prior_ret_20d: float = 5.0,
    merge_bars: int = 3,
) -> list[dict]:
    """Collect early-confirmed bearish OB states from the raw OB-zone table."""

    if df.empty or raw_ob_events.empty or len(df) < max(high_lookback, 20) + max_confirm_lag + 2:
        return []

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    selected_events: list[pd.Series] = []
    for _, event in raw_ob_events.iterrows():
        if int(event.get("direction", 0)) != -1:
            continue
        top_pos = _as_pos(event.get("origin_pos", event.get("formed_pos", -1)))
        if top_pos < min_top_pos or top_pos >= len(df):
            continue
        raw_score = _as_float(event.get("raw_score", 0.0), 0.0)
        if raw_score < min_raw_presence_score:
            continue
        if not _passes_bearish_top_context(
            df,
            top_pos=top_pos,
            high_lookback=high_lookback,
            near_high_pct=near_high_pct,
            min_prior_ret_20d=min_prior_ret_20d,
        ):
            continue
        selected_events.append(event)

    if not selected_events:
        return []

    selected = pd.DataFrame(selected_events)
    if ob_events is not None and structure_events is not None:
        selected = attach_smc_early_states(
            df,
            selected,
            ob_events,
            structure_events,
            max_early_lag=max_confirm_lag,
        )

    raw_rows: list[dict] = []
    early_pos_values = pd.to_numeric(selected.get("early_pos", pd.Series(dtype=float)), errors="coerce")
    for idx, event in selected.iterrows():
        if pd.isna(early_pos_values.loc[idx]):
            continue
        top_pos = _as_pos(event.get("origin_pos", event.get("formed_pos", -1)))
        asof_pos = int(early_pos_values.loc[idx])
        if asof_pos < top_pos or asof_pos > min(len(df) - 1, top_pos + max_confirm_lag):
            continue
        raw_score = _as_float(event.get("raw_score", 0.0), 0.0)
        struct_score = _as_float(event.get("early_struct_score", 0.0), 0.0)
        total_score = _as_float(event.get("early_score", event.get("early_total_score", 0.0)), 0.0)
        triggered = (
            total_score >= min_total_score
            or (raw_score >= min_raw_score and struct_score >= 15.0)
            or (struct_score >= min_struct_score and raw_score >= 20.0)
        )
        if not triggered:
            continue

        top_price = float(high.iloc[top_pos])
        raw_rows.append({
            **recall_flags_for_sources({"smc_early"}, top_pos=top_pos, score_asof_pos=asof_pos),
            "candidate_source": "smc_early",
            "signal_date": pd.Timestamp(df.index[top_pos]),
            "confirm_date": pd.Timestamp(df.index[asof_pos]),
            "signal_pos": top_pos,
            "confirm_pos": asof_pos,
            "top_date": pd.Timestamp(df.index[top_pos]),
            "top_pos": top_pos,
            "top_price": top_price,
            "strategies": "smc_early",
            "has_shadow": 0,
            "has_doji": 0,
            "has_gap_fail": 0,
            "signal_count": 0,
            "score_max": 0,
            "score_sum": 0,
            "confirm_modes": "smc_early",
            "signal_dates": date_text(df.index[top_pos]),
            "score_asof_pos": asof_pos,
            "score_asof_date": date_text(df.index[asof_pos]),
            "smc_raw_recall_count": 1,
            "smc_raw_recall_score_max": round(raw_score, 2),
            "smc_raw_recall_detect_lag_min": int(_as_pos(event.get("detected_pos", top_pos), top_pos) - top_pos),
            "smc_raw_recall_confirm_lag_min": int(asof_pos - top_pos),
            "smc_raw_recall_raw_id": str(event.get("raw_id", "")),
            "smc_raw_recall_zone_top": round(_as_float(event.get("top", float("nan"))), 4),
            "smc_raw_recall_zone_bottom": round(_as_float(event.get("bottom", float("nan"))), 4),
            "smc_raw_recall_raw_reason": str(event.get("raw_reason", "")),
            "smc_early_recall_count": 1,
            "smc_early_recall_score_max": round(total_score, 2),
            "smc_early_recall_raw_score_max": round(raw_score, 2),
            "smc_early_recall_struct_score_max": round(struct_score, 2),
            "smc_early_recall_confirm_lag_min": int(asof_pos - top_pos),
            "smc_early_recall_has_fvg": int(_as_float(event.get("early_has_fvg", 0), 0.0) > 0),
            "smc_early_recall_has_sweep": int(_as_float(event.get("early_has_sweep", 0), 0.0) > 0),
            "smc_early_recall_choch_down": int(_as_float(event.get("early_choch_down", 0), 0.0) > 0),
            "smc_early_recall_bos_down": int(_as_float(event.get("early_bos_down", 0), 0.0) > 0),
            "smc_early_recall_micro_low_break": int(_as_float(event.get("early_micro_low_break", 0), 0.0) > 0),
            "smc_early_recall_top_low_break": int(_as_float(event.get("early_top_low_break", 0), 0.0) > 0),
            "smc_early_recall_bull_ob_mitigated": _as_float(event.get("early_bull_ob_mitigated", 0.0), 0.0),
            "smc_early_recall_low_break_pct": _as_float(
                event.get("early_low_break_pct", round((float(low.iloc[asof_pos]) / top_price - 1) * 100, 2))
            ),
            "smc_early_recall_raw_id": str(event.get("raw_id", "")),
        })

    return _cluster_best(raw_rows, merge_bars=merge_bars, score_col="smc_early_recall_score_max")


def collect_smc_supply_held_candidates(
    df: pd.DataFrame,
    raw_ob_events: pd.DataFrame,
    *,
    min_top_pos: int = 0,
    min_anchor_score: float = 60.0,
    max_confirm_lag: int = 8,
    high_lookback: int = 60,
    near_high_pct: float = 3.0,
    min_prior_ret_20d: float = 5.0,
    merge_bars: int = 3,
) -> list[dict]:
    """Collect 'supply-held' bearish tops — the principled replacement for appear.

    A high-score bearish rejection candle near a recent high is confirmed as a
    potential top once, within ``max_confirm_lag`` bars, price both:

    * never reclaims the anchor candle's high (the supply zone holds as
      resistance — this filters interim pullbacks that resume higher); and
    * closes below the anchor candle's low (price has actually rolled over).

    Unlike ``smc_appear`` (which only required a later low to dip below the candle
    bottom and thus fired on every near-high pullback), the "high not reclaimed"
    constraint gives the signal real meaning and works for slow rollover tops that
    structure-break confirmation (CHoCH/BOS/OB-confirmed) detects far too late.
    """

    if df.empty or raw_ob_events.empty or len(df) < max(high_lookback, 20) + 2:
        return []

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    n = len(df)
    raw_rows: list[dict] = []
    for _, event in raw_ob_events.iterrows():
        if int(event.get("direction", 0)) != -1:
            continue
        top_pos = _as_pos(event.get("origin_pos", event.get("formed_pos", -1)))
        if top_pos < min_top_pos or top_pos >= n - 1:
            continue
        raw_score = _as_float(event.get("raw_score", event.get("score", 0.0)), 0.0)
        if raw_score < min_anchor_score:
            continue
        if not _passes_bearish_top_context(
            df,
            top_pos=top_pos,
            high_lookback=high_lookback,
            near_high_pct=near_high_pct,
            min_prior_ret_20d=min_prior_ret_20d,
        ):
            continue

        anchor_high = float(high.iloc[top_pos])
        anchor_low = float(low.iloc[top_pos])
        max_pos = min(n - 1, top_pos + max_confirm_lag)
        confirm_pos: int | None = None
        invalidated = False
        for asof_pos in range(top_pos + 1, max_pos + 1):
            if float(high.iloc[asof_pos]) > anchor_high:
                invalidated = True       # 高点被收复 -> 中继，作废
                break
            if float(close.iloc[asof_pos]) < anchor_low:
                confirm_pos = asof_pos    # 供给守住高点 且 收盘跌破锚低 -> 确认
                break
        if invalidated or confirm_pos is None:
            continue

        break_low_pct = (float(close.iloc[confirm_pos]) / anchor_low - 1.0) * 100.0 if anchor_low > 0 else float("nan")
        raw_rows.append({
            **recall_flags_for_sources({"smc_supply_held"}, top_pos=top_pos, score_asof_pos=confirm_pos),
            "candidate_source": "smc_supply_held",
            "signal_date": pd.Timestamp(df.index[top_pos]),
            "confirm_date": pd.Timestamp(df.index[confirm_pos]),
            "signal_pos": top_pos,
            "confirm_pos": confirm_pos,
            "top_date": pd.Timestamp(df.index[top_pos]),
            "top_pos": top_pos,
            "top_price": anchor_high,
            "strategies": "smc_supply_held",
            "has_shadow": 0,
            "has_doji": 0,
            "has_gap_fail": 0,
            "signal_count": 0,
            "score_max": 0,
            "score_sum": 0,
            "confirm_modes": "smc_supply_held",
            "signal_dates": date_text(df.index[top_pos]),
            "score_asof_pos": confirm_pos,
            "score_asof_date": date_text(df.index[confirm_pos]),
            "smc_raw_recall_count": 1,
            "smc_raw_recall_score_max": round(raw_score, 2),
            "smc_raw_recall_detect_lag_min": int(_as_pos(event.get("detected_pos", top_pos), top_pos) - top_pos),
            "smc_raw_recall_confirm_lag_min": int(confirm_pos - top_pos),
            "smc_raw_recall_zone_top": round(_as_float(event.get("top", float("nan"))), 4),
            "smc_raw_recall_zone_bottom": round(_as_float(event.get("bottom", float("nan"))), 4),
            "smc_raw_recall_raw_id": str(event.get("raw_id", "")),
            "smc_raw_recall_raw_reason": str(event.get("raw_reason", "")),
            "smc_raw_recall_prior_ret_20d": _as_float(event.get("prior_ret_20d", float("nan"))),
            "smc_raw_recall_near_high_pct": _as_float(event.get("near_high_pct", float("nan"))),
            "smc_supply_held_recall_count": 1,
            "smc_supply_held_recall_anchor_score_max": round(raw_score, 2),
            "smc_supply_held_recall_confirm_lag_min": int(confirm_pos - top_pos),
            "smc_supply_held_recall_break_low_pct": round(float(break_low_pct), 2) if pd.notna(break_low_pct) else float("nan"),
        })

    return _cluster_best(raw_rows, merge_bars=merge_bars, score_col="smc_supply_held_recall_anchor_score_max")


def collect_smc_top_confirmed_candidates(
    df: pd.DataFrame,
    raw_ob_events: pd.DataFrame,
    ob_events: pd.DataFrame | None = None,
    structure_events: pd.DataFrame | None = None,
    *,
    min_top_pos: int = 0,
    supply_held_min_anchor_score: float = 60.0,
    supply_held_max_lag: int = 8,
    include_confirmed: bool = True,
    high_lookback: int = 60,
    near_high_pct: float = 3.0,
    min_prior_ret_20d: float = 5.0,
    merge_bars: int = 3,
    smc_early_kwargs: dict | None = None,
    smc_confirmed_kwargs: dict | None = None,
) -> list[dict]:
    """Unified SMC top source — earliest confirmation across mechanisms.

    A near-high bearish anchor can be confirmed as a potential top by three
    complementary mechanisms, each timely for a different top morphology:

    * ``smc_supply_held`` — high not reclaimed + close below anchor low (slow
      rollover tops);
    * ``smc_early`` — sharp internal structure break / FVG / sweep (sharp
      reversals);
    * ``smc_confirmed`` — full swing-OB confirmation (slow but most reliable).

    For each anchor we keep the **earliest** confirming mechanism (min confirm
    time).  This is causally honest: at the earliest alert only that mechanism
    has fired, so only its evidence is used; ``score_lag_bars`` records the min
    lag and ``recalled_by_smc_*`` records which mechanism fired.  Later
    mechanisms simply have not happened yet at score time.  The union of the
    three lifts true-top coverage above any single source (~74% vs ~70%).
    """

    sh = collect_smc_supply_held_candidates(
        df,
        raw_ob_events,
        min_top_pos=min_top_pos,
        min_anchor_score=supply_held_min_anchor_score,
        max_confirm_lag=supply_held_max_lag,
        high_lookback=high_lookback,
        near_high_pct=near_high_pct,
        min_prior_ret_20d=min_prior_ret_20d,
        merge_bars=merge_bars,
    )
    ea = collect_smc_early_candidates(
        df,
        raw_ob_events,
        ob_events,
        structure_events,
        min_top_pos=min_top_pos,
        high_lookback=high_lookback,
        near_high_pct=near_high_pct,
        min_prior_ret_20d=min_prior_ret_20d,
        merge_bars=merge_bars,
        **(smc_early_kwargs or {}),
    )
    cf: list[dict] = []
    if include_confirmed:
        cf = collect_smc_confirmed_candidates(
            df,
            raw_ob_events,
            min_top_pos=min_top_pos,
            high_lookback=high_lookback,
            near_high_pct=near_high_pct,
            min_prior_ret_20d=min_prior_ret_20d,
            **(smc_confirmed_kwargs or {}),
        )

    members = [*sh, *ea, *cf]
    if not members:
        return []

    ordered = sorted(
        members,
        key=lambda r: (int(r["top_pos"]), int(r.get("score_asof_pos", r["top_pos"]))),
    )
    clusters: list[list[dict]] = []
    current: list[dict] = []
    end_pos = -1
    for row in ordered:
        top_pos = int(row["top_pos"])
        if not current or top_pos <= end_pos:
            current.append(row)
            end_pos = max(end_pos, top_pos + merge_bars)
            continue
        clusters.append(current)
        current = [row]
        end_pos = top_pos + merge_bars
    if current:
        clusters.append(current)

    out: list[dict] = []
    for cluster in clusters:
        # 取最早确认者：因果上该时点只有它已发生，特征可用
        earliest = min(cluster, key=lambda r: int(r.get("score_asof_pos", r["top_pos"])))
        all_mechanisms = sorted(set().union(*(_row_sources(r) for r in cluster)))
        out.append({
            **earliest,
            "smc_top_confirm_all_mechanisms": "+".join(all_mechanisms),
            "smc_top_confirm_mechanism_count": len(all_mechanisms),
        })
    return sorted(out, key=lambda row: (int(row["top_pos"]), int(row["score_asof_pos"])))


def merge_recall_candidates(
    candidates: list[dict],
    df: pd.DataFrame,
    *,
    merge_bars: int = 3,
) -> list[dict]:
    """Merge nearby recall-source rows into one unified candidate row."""

    if not candidates:
        return []

    normalized: list[dict] = []
    for row in candidates:
        item = dict(row)
        top_pos = int(item["top_pos"])
        score_asof_pos = int(item.get("score_asof_pos", item.get("confirm_pos", top_pos)))
        sources = _row_sources(item)
        item = {
            **recall_flags_for_sources(sources, top_pos=top_pos, score_asof_pos=score_asof_pos),
            **item,
        }
        normalized.append(item)

    ordered = sorted(normalized, key=lambda row: (int(row["top_pos"]), int(row.get("score_asof_pos", row["top_pos"]))))
    clusters: list[list[dict]] = []
    current: list[dict] = []
    end_pos = -1
    for row in ordered:
        top_pos = int(row["top_pos"])
        if not current or top_pos <= end_pos:
            current.append(row)
            end_pos = max(end_pos, top_pos + merge_bars)
            continue
        clusters.append(current)
        current = [row]
        end_pos = top_pos + merge_bars
    if current:
        clusters.append(current)

    merged: list[dict] = []
    pattern_prefixes = ("shadow_", "doji_", "gap_fail_", "double_top_")
    for cluster in clusters:
        sources = set().union(*(_row_sources(row) for row in cluster))
        top_choice = max(cluster, key=lambda row: float(row.get("top_price", 0.0)))
        top_pos = int(top_choice["top_pos"])
        signal_pos = min(int(row.get("signal_pos", top_pos)) for row in cluster)
        score_asof_pos = max(int(row.get("score_asof_pos", row.get("confirm_pos", top_pos))) for row in cluster)
        score_asof_pos = max(top_pos, min(len(df) - 1, score_asof_pos))
        flags = recall_flags_for_sources(sources, top_pos=top_pos, score_asof_pos=score_asof_pos)

        row = {
            **flags,
            "candidate_source": "+".join(sorted(sources)),
            "signal_date": pd.Timestamp(df.index[signal_pos]),
            "confirm_date": pd.Timestamp(df.index[score_asof_pos]),
            "signal_pos": signal_pos,
            "confirm_pos": score_asof_pos,
            "top_date": pd.Timestamp(df.index[top_pos]),
            "top_pos": top_pos,
            "top_price": float(df["high"].iloc[top_pos]),
            "strategies": "+".join(sorted(sources)),
            "has_shadow": int("shadow" in sources),
            "has_doji": int("doji" in sources),
            "has_gap_fail": int("gap_fail" in sources),
            "signal_count": int(_max_numeric(cluster, "signal_count", 0)),
            "score_max": int(_max_numeric(cluster, "score_max", 0)),
            "score_sum": int(_numeric_values(cluster, "score_sum").fillna(0).sum()),
            "confirm_modes": _joined_values(cluster, "confirm_modes"),
            "signal_dates": _joined_values(cluster, "signal_dates"),
            "source_top_dates": ",".join(dict.fromkeys(date_text(df.index[int(r["top_pos"])]) for r in cluster)),
            "source_score_asof_dates": ",".join(dict.fromkeys(date_text(df.index[int(r.get("score_asof_pos", r.get("confirm_pos", r["top_pos"])))]) for r in cluster)),
            "score_asof_pos": score_asof_pos,
            "score_asof_date": date_text(df.index[score_asof_pos]),
            "smc_raw_recall_count": int(_numeric_values(cluster, "smc_raw_recall_count").fillna(0).sum()),
            "smc_raw_recall_score_max": round(float(_max_numeric(cluster, "smc_raw_recall_score_max", 0.0)), 2),
            "smc_raw_recall_detect_lag_min": _min_numeric(cluster, "smc_raw_recall_detect_lag_min"),
            "smc_raw_recall_confirm_lag_min": _min_numeric(cluster, "smc_raw_recall_confirm_lag_min"),
            "smc_appear_recall_count": int(_numeric_values(cluster, "smc_appear_recall_count").fillna(0).sum()),
            "smc_appear_recall_confirm_lag_min": _min_numeric(cluster, "smc_appear_recall_confirm_lag_min"),
            "smc_appear_recall_leave_pct_max": round(float(_max_numeric(cluster, "smc_appear_recall_leave_pct_max", 0.0)), 2),
            "smc_appear_recall_zone_width_pct": _min_numeric(cluster, "smc_appear_recall_zone_width_pct"),
            "smc_appear_recall_formed_date": _joined_values(cluster, "smc_appear_recall_formed_date"),
            "smc_appear_recall_appear_date": _joined_values(cluster, "smc_appear_recall_appear_date"),
            "smc_confirmed_recall_count": int(_numeric_values(cluster, "smc_confirmed_recall_count").fillna(0).sum()),
            "smc_confirmed_recall_score_max": round(float(_max_numeric(cluster, "smc_confirmed_recall_score_max", 0.0)), 2),
            "smc_confirmed_recall_struct_score_max": round(float(_max_numeric(cluster, "smc_confirmed_recall_struct_score_max", 0.0)), 2),
            "smc_confirmed_recall_confirm_lag_min": _min_numeric(cluster, "smc_confirmed_recall_confirm_lag_min"),
            "smc_confirmed_recall_zone_width_pct": _min_numeric(cluster, "smc_confirmed_recall_zone_width_pct"),
            "smc_confirmed_recall_volume_ratio": round(float(_max_numeric(cluster, "smc_confirmed_recall_volume_ratio", float("nan"))), 2),
            "smc_confirmed_recall_formed_date": _joined_values(cluster, "smc_confirmed_recall_formed_date"),
            "smc_confirmed_recall_confirmed_date": _joined_values(cluster, "smc_confirmed_recall_confirmed_date"),
            "smc_early_recall_count": int(_numeric_values(cluster, "smc_early_recall_count").fillna(0).sum()),
            "smc_early_recall_score_max": round(float(_max_numeric(cluster, "smc_early_recall_score_max", 0.0)), 2),
            "smc_early_recall_raw_score_max": round(float(_max_numeric(cluster, "smc_early_recall_raw_score_max", 0.0)), 2),
            "smc_early_recall_struct_score_max": round(float(_max_numeric(cluster, "smc_early_recall_struct_score_max", 0.0)), 2),
            "smc_early_recall_confirm_lag_min": _min_numeric(cluster, "smc_early_recall_confirm_lag_min"),
            "smc_supply_held_recall_count": int(_numeric_values(cluster, "smc_supply_held_recall_count").fillna(0).sum()),
            "smc_supply_held_recall_anchor_score_max": round(float(_max_numeric(cluster, "smc_supply_held_recall_anchor_score_max", 0.0)), 2),
            "smc_supply_held_recall_confirm_lag_min": _min_numeric(cluster, "smc_supply_held_recall_confirm_lag_min"),
            "smc_supply_held_recall_break_low_pct": _min_numeric(cluster, "smc_supply_held_recall_break_low_pct"),
            "double_top_recall_count": int(_numeric_values(cluster, "double_top_recall_count").fillna(0).sum()),
            "double_top_recall_confirm_lag_min": _min_numeric(cluster, "double_top_recall_confirm_lag_min"),
            "double_top_recall_neckline_break_pct": _min_numeric(cluster, "double_top_recall_neckline_break_pct"),
            "double_top_recall_failed_rebound_vs_neckline_pct": _min_numeric(cluster, "double_top_recall_failed_rebound_vs_neckline_pct"),
        }

        passthrough_cols = sorted({
            col
            for item in cluster
            for col in item
            if col.startswith(pattern_prefixes)
        })
        for col in passthrough_cols:
            numeric = _numeric_values(cluster, col).dropna()
            row[col] = numeric.max() if not numeric.empty else _joined_values(cluster, col)

        merged.append(row)
    return merged


def attach_strategy_matches(
    universe: pd.DataFrame,
    strategy_candidates: pd.DataFrame,
    *,
    near_bars: int = 3,
) -> pd.DataFrame:
    """Annotate universe candidates with strategy recall matches."""

    out = universe.copy()
    if out.empty:
        return out

    defaults = {
        "covered_by_recall": 0,
        "covered_by_patterns": 0,
        "covered_by_shadow": 0,
        "covered_by_doji": 0,
        "covered_by_gap_fail": 0,
        "covered_by_double_top": 0,
        "covered_by_smc_raw": 0,
        "covered_by_smc_appear": 0,
        "covered_by_smc_confirmed": 0,
        "covered_by_smc_early": 0,
        "covered_by_smc_supply_held": 0,
        **RECALL_FLAG_DEFAULTS,
        "matched_recall_count": 0,
        "matched_recall_strategies": "",
        "matched_recall_top_dates": "",
        "matched_pattern_count": 0,
        "matched_pattern_strategies": "",
        "matched_pattern_top_dates": "",
    }
    for col, value in defaults.items():
        out[col] = value

    if strategy_candidates.empty:
        return out
    strategy_candidates = apply_legacy_feature_aliases(strategy_candidates)

    pattern_cols = [
        c for c in strategy_candidates.columns
        if c.startswith("shadow_") or c.startswith("doji_") or c.startswith("gap_fail_") or c.startswith("double_top_")
    ]
    for col in pattern_cols:
        if col not in out.columns:
            out[col] = pd.Series(pd.NA, index=out.index, dtype="object")
        else:
            out[col] = out[col].astype("object")

    grouped = {
        (str(market), str(sym)): group.copy()
        for (market, sym), group in strategy_candidates.groupby(["market", "sym"], observed=True)
    }

    for idx, row in out.iterrows():
        key = (str(row["market"]), str(row["sym"]))
        group = grouped.get(key)
        if group is None or group.empty:
            continue

        distances = (pd.to_numeric(group["top_pos"], errors="coerce") - int(row["top_pos"])).abs()
        matches = group[distances <= near_bars].copy()
        if matches.empty:
            continue

        nearest = matches.loc[distances.loc[matches.index].sort_values().index[0]]
        strategies = sorted(set().union(*(_row_sources(row) for _, row in matches.iterrows())))
        pattern_strategies = [s for s in strategies if s in {"shadow", "doji", "gap_fail"}]
        candle_strategies = [s for s in strategies if s in {"shadow", "doji", "gap_fail"}]
        out.at[idx, "covered_by_recall"] = 1
        out.at[idx, "covered_by_patterns"] = int(bool(pattern_strategies))
        out.at[idx, "covered_by_shadow"] = int("shadow" in strategies)
        out.at[idx, "covered_by_doji"] = int("doji" in strategies)
        out.at[idx, "covered_by_gap_fail"] = int("gap_fail" in strategies)
        out.at[idx, "covered_by_double_top"] = int("double_top" in strategies)
        out.at[idx, "covered_by_smc_raw"] = int("smc_raw" in strategies)
        out.at[idx, "covered_by_smc_appear"] = int("smc_appear" in strategies)
        out.at[idx, "covered_by_smc_confirmed"] = int("smc_confirmed" in strategies)
        out.at[idx, "covered_by_smc_early"] = int("smc_early" in strategies)
        out.at[idx, "covered_by_smc_supply_held"] = int("smc_supply_held" in strategies)
        flags = recall_flags_for_sources(set(strategies), top_pos=int(row["top_pos"]), score_asof_pos=int(row.get("score_asof_pos", row["top_pos"])))
        for flag, value in flags.items():
            out.at[idx, flag] = value
        out.at[idx, "matched_recall_count"] = int(len(matches))
        out.at[idx, "matched_recall_strategies"] = "+".join(strategies)
        out.at[idx, "matched_recall_top_dates"] = ",".join(date_text(x) for x in matches["top_date"])
        if "recalled_by_candle" in matches.columns:
            pattern_mask = pd.to_numeric(matches.get("recalled_by_candle", 0), errors="coerce").fillna(0).astype(int).eq(1)
            out.at[idx, "matched_pattern_count"] = int(pattern_mask.sum())
            out.at[idx, "matched_pattern_top_dates"] = ",".join(date_text(x) for x in matches[pattern_mask]["top_date"])
        else:
            out.at[idx, "matched_pattern_count"] = int(bool(pattern_strategies))
            out.at[idx, "matched_pattern_top_dates"] = ""
        out.at[idx, "matched_pattern_strategies"] = "+".join(pattern_strategies)
        out.at[idx, "strategies"] = "zigzag_peak+" + "+".join(strategies)
        out.at[idx, "has_shadow"] = int("shadow" in strategies)
        out.at[idx, "has_doji"] = int("doji" in strategies)
        out.at[idx, "has_gap_fail"] = int("gap_fail" in strategies)
        out.at[idx, "signal_count"] = int(pd.to_numeric(matches.get("signal_count", 0), errors="coerce").fillna(0).sum())
        out.at[idx, "score_max"] = int(pd.to_numeric(matches.get("score_max", 0), errors="coerce").fillna(0).max())
        out.at[idx, "score_sum"] = int(pd.to_numeric(matches.get("score_sum", 0), errors="coerce").fillna(0).sum())
        out.at[idx, "double_top_recall_count"] = int(pd.to_numeric(matches.get("double_top_recall_count", 0), errors="coerce").fillna(0).sum())
        double_top_lag = pd.to_numeric(matches.get("double_top_recall_confirm_lag_min", pd.Series(dtype=float)), errors="coerce").dropna()
        if not double_top_lag.empty:
            out.at[idx, "double_top_recall_confirm_lag_min"] = float(double_top_lag.min())
        double_top_break = pd.to_numeric(matches.get("double_top_recall_neckline_break_pct", pd.Series(dtype=float)), errors="coerce").dropna()
        if not double_top_break.empty:
            out.at[idx, "double_top_recall_neckline_break_pct"] = float(double_top_break.min())
        double_top_rebound = pd.to_numeric(matches.get("double_top_recall_failed_rebound_vs_neckline_pct", pd.Series(dtype=float)), errors="coerce").dropna()
        if not double_top_rebound.empty:
            out.at[idx, "double_top_recall_failed_rebound_vs_neckline_pct"] = float(double_top_rebound.min())
        out.at[idx, "smc_raw_recall_count"] = int(pd.to_numeric(matches.get("smc_raw_recall_count", 0), errors="coerce").fillna(0).sum())
        out.at[idx, "smc_raw_recall_score_max"] = float(pd.to_numeric(matches.get("smc_raw_recall_score_max", 0), errors="coerce").fillna(0).max())
        raw_detect_lag = pd.to_numeric(matches.get("smc_raw_recall_detect_lag_min", pd.Series(dtype=float)), errors="coerce").dropna()
        if not raw_detect_lag.empty:
            out.at[idx, "smc_raw_recall_detect_lag_min"] = float(raw_detect_lag.min())
        raw_confirm_lag = pd.to_numeric(matches.get("smc_raw_recall_confirm_lag_min", pd.Series(dtype=float)), errors="coerce").dropna()
        if not raw_confirm_lag.empty:
            out.at[idx, "smc_raw_recall_confirm_lag_min"] = float(raw_confirm_lag.min())
        out.at[idx, "smc_appear_recall_count"] = int(pd.to_numeric(matches.get("smc_appear_recall_count", 0), errors="coerce").fillna(0).sum())
        appear_lag_values = pd.to_numeric(matches.get("smc_appear_recall_confirm_lag_min", pd.Series(dtype=float)), errors="coerce").dropna()
        if not appear_lag_values.empty:
            out.at[idx, "smc_appear_recall_confirm_lag_min"] = float(appear_lag_values.min())
        out.at[idx, "smc_appear_recall_leave_pct_max"] = float(pd.to_numeric(matches.get("smc_appear_recall_leave_pct_max", 0), errors="coerce").fillna(0).max())
        appear_width_values = pd.to_numeric(matches.get("smc_appear_recall_zone_width_pct", pd.Series(dtype=float)), errors="coerce").dropna()
        if not appear_width_values.empty:
            out.at[idx, "smc_appear_recall_zone_width_pct"] = float(appear_width_values.min())
        out.at[idx, "smc_confirmed_recall_count"] = int(pd.to_numeric(matches.get("smc_confirmed_recall_count", 0), errors="coerce").fillna(0).sum())
        out.at[idx, "smc_confirmed_recall_score_max"] = float(pd.to_numeric(matches.get("smc_confirmed_recall_score_max", 0), errors="coerce").fillna(0).max())
        out.at[idx, "smc_confirmed_recall_struct_score_max"] = float(pd.to_numeric(matches.get("smc_confirmed_recall_struct_score_max", 0), errors="coerce").fillna(0).max())
        lag_values = pd.to_numeric(matches.get("smc_confirmed_recall_confirm_lag_min", pd.Series(dtype=float)), errors="coerce").dropna()
        if not lag_values.empty:
            out.at[idx, "smc_confirmed_recall_confirm_lag_min"] = float(lag_values.min())
        out.at[idx, "smc_early_recall_count"] = int(pd.to_numeric(matches.get("smc_early_recall_count", 0), errors="coerce").fillna(0).sum())
        out.at[idx, "smc_early_recall_score_max"] = float(pd.to_numeric(matches.get("smc_early_recall_score_max", 0), errors="coerce").fillna(0).max())
        early_raw_scores = matches.get("smc_early_recall_raw_score_max", matches.get("smc_early_recall_origin_score_max", 0))
        out.at[idx, "smc_early_recall_raw_score_max"] = float(pd.to_numeric(early_raw_scores, errors="coerce").fillna(0).max())
        out.at[idx, "smc_early_recall_struct_score_max"] = float(pd.to_numeric(matches.get("smc_early_recall_struct_score_max", 0), errors="coerce").fillna(0).max())
        early_lag_values = pd.to_numeric(matches.get("smc_early_recall_confirm_lag_min", pd.Series(dtype=float)), errors="coerce").dropna()
        if not early_lag_values.empty:
            out.at[idx, "smc_early_recall_confirm_lag_min"] = float(early_lag_values.min())
        out.at[idx, "confirm_modes"] = str(nearest.get("confirm_modes", ""))
        out.at[idx, "signal_dates"] = str(nearest.get("signal_dates", ""))

        for col in pattern_cols:
            if col in nearest.index:
                out.at[idx, col] = nearest[col]

    return out
