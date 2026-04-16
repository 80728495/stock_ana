"""Reusable VCP micro-structure primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ana.strategies.primitives.pivots import argrel_pivots


def detect_vcp_micro_structure(
    df: pd.DataFrame,
    pivot_order: int = 5,
    max_contraction_ratio: float = 0.75,
    min_contractions: int = 3,
) -> tuple[bool, dict]:
    """Detect VCP contraction by checking consecutive swing-range shrinkage."""
    if len(df) < 50:
        return False, {"error": "数据量不足"}

    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)

    sh_idx, _ = argrel_pivots(highs, lows, order=pivot_order)
    if len(sh_idx) < 2:
        return False, {"error": "swing high 不足", "n_swing_highs": len(sh_idx)}

    swing_ranges: list[float] = []
    swing_details: list[dict] = []
    for k in range(len(sh_idx) - 1):
        seg_start = sh_idx[k]
        seg_end = sh_idx[k + 1]
        seg_high = float(np.max(highs[seg_start : seg_end + 1]))
        seg_low = float(np.min(lows[seg_start : seg_end + 1]))
        rng = seg_high - seg_low
        swing_ranges.append(rng)
        swing_details.append(
            {
                "start_idx": int(seg_start),
                "end_idx": int(seg_end),
                "high": seg_high,
                "low": seg_low,
                "range": round(rng, 4),
            }
        )

    last_sh = sh_idx[-1]
    if last_sh < len(highs) - 1:
        seg_high = float(np.max(highs[last_sh:]))
        seg_low = float(np.min(lows[last_sh:]))
        rng = seg_high - seg_low
        swing_ranges.append(rng)
        swing_details.append(
            {
                "start_idx": int(last_sh),
                "end_idx": len(highs) - 1,
                "high": seg_high,
                "low": seg_low,
                "range": round(rng, 4),
            }
        )

    if len(swing_ranges) < 2:
        return False, {"error": "摆幅段数不足", "n_segments": len(swing_ranges)}

    contraction_ratios: list[float] = []
    for k in range(1, len(swing_ranges)):
        if swing_ranges[k - 1] > 1e-9:
            contraction_ratios.append(swing_ranges[k] / swing_ranges[k - 1])
        else:
            contraction_ratios.append(999.0)

    consecutive_contractions = 0
    for ratio in reversed(contraction_ratios):
        if ratio <= max_contraction_ratio:
            consecutive_contractions += 1
        else:
            break

    is_contracting = consecutive_contractions >= min_contractions
    chain_start_seg = len(swing_ranges) - consecutive_contractions - 1 if consecutive_contractions > 0 else -1
    chain_end_seg = len(swing_ranges) - 1

    if consecutive_contractions > 0 and chain_start_seg >= 0:
        chain_start_iloc = swing_details[chain_start_seg]["start_idx"]
        chain_end_iloc = swing_details[chain_end_seg]["end_idx"]
    else:
        chain_start_iloc = -1
        chain_end_iloc = -1

    sma50 = pd.Series(closes).rolling(50).mean().values
    latest_close = float(closes[-1])
    latest_sma50 = float(sma50[-1]) if not np.isnan(sma50[-1]) else 0.0
    is_above_floor = latest_close >= (latest_sma50 * 0.98) if latest_sma50 > 0 else False
    is_vcp_valid = is_contracting and is_above_floor

    stats = {
        "n_swing_highs": len(sh_idx),
        "n_segments": len(swing_ranges),
        "consecutive_contractions": consecutive_contractions,
        "swing_ranges": [round(r, 4) for r in swing_ranges],
        "contraction_ratios": [round(r, 4) for r in contraction_ratios],
        "chain_start_seg": chain_start_seg,
        "chain_end_seg": chain_end_seg,
        "chain_start_iloc": chain_start_iloc,
        "chain_end_iloc": chain_end_iloc,
        "chain_ranges": [
            round(swing_ranges[k], 4)
            for k in range(max(0, chain_start_seg), chain_end_seg + 1)
        ]
        if consecutive_contractions > 0
        else [],
        "chain_ratios": [
            round(contraction_ratios[k], 4)
            for k in range(len(contraction_ratios) - consecutive_contractions, len(contraction_ratios))
        ]
        if consecutive_contractions > 0
        else [],
        "latest_close_vs_sma50_ratio": round(latest_close / latest_sma50, 4) if latest_sma50 > 0 else 0.0,
        "is_contracting": is_contracting,
        "swing_high_indices": sh_idx.tolist(),
    }
    return is_vcp_valid, stats
