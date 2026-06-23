"""Coverage reports for top-reversal candidate sources."""

from __future__ import annotations

import pandas as pd


def _sum_flag(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def _sum_flag_any(df: pd.DataFrame, *cols: str) -> int:
    for col in cols:
        if col in df.columns:
            return _sum_flag(df, col)
    return 0


def _coverage_row(scope: str, df: pd.DataFrame) -> dict[str, object]:
    true_tops = df[df["label"] == "true_top"].copy()
    n_true = len(true_tops)
    any_recall = _sum_flag(true_tops, "covered_by_recall")
    covered = _sum_flag(true_tops, "covered_by_patterns")
    shadow = _sum_flag(true_tops, "covered_by_shadow")
    doji = _sum_flag(true_tops, "covered_by_doji")
    gap_fail = _sum_flag(true_tops, "covered_by_gap_fail")
    smc_raw = _sum_flag(true_tops, "covered_by_smc_raw")
    smc_appear = _sum_flag(true_tops, "covered_by_smc_appear")
    smc_confirmed = _sum_flag_any(true_tops, "covered_by_smc_confirmed", "covered_by_smc_origin")
    smc_early = _sum_flag(true_tops, "covered_by_smc_early")
    smc_supply_held = _sum_flag(true_tops, "covered_by_smc_supply_held")
    return {
        "scope": scope,
        "universe_candidates": len(df),
        "universe_true_top": n_true,
        "covered_by_recall": any_recall,
        "missed_by_recall": int(n_true - any_recall),
        "recall_coverage_pct": round(any_recall / n_true * 100, 1) if n_true else 0.0,
        "covered_by_patterns": covered,
        "missed_by_patterns": int(n_true - covered),
        "pattern_coverage_pct": round(covered / n_true * 100, 1) if n_true else 0.0,
        "covered_by_shadow": shadow,
        "shadow_coverage_pct": round(shadow / n_true * 100, 1) if n_true else 0.0,
        "covered_by_doji": doji,
        "doji_coverage_pct": round(doji / n_true * 100, 1) if n_true else 0.0,
        "covered_by_gap_fail": gap_fail,
        "gap_fail_coverage_pct": round(gap_fail / n_true * 100, 1) if n_true else 0.0,
        "covered_by_smc_raw": smc_raw,
        "smc_raw_coverage_pct": round(smc_raw / n_true * 100, 1) if n_true else 0.0,
        "covered_by_smc_appear": smc_appear,
        "smc_appear_coverage_pct": round(smc_appear / n_true * 100, 1) if n_true else 0.0,
        "covered_by_smc_confirmed": smc_confirmed,
        "smc_confirmed_coverage_pct": round(smc_confirmed / n_true * 100, 1) if n_true else 0.0,
        "covered_by_smc_early": smc_early,
        "smc_early_coverage_pct": round(smc_early / n_true * 100, 1) if n_true else 0.0,
        "covered_by_smc_supply_held": smc_supply_held,
        "smc_supply_held_coverage_pct": round(smc_supply_held / n_true * 100, 1) if n_true else 0.0,
    }


def strategy_coverage_report(universe: pd.DataFrame) -> pd.DataFrame:
    """Summarize how well live candidate sources cover the true-top universe."""

    if universe.empty:
        return pd.DataFrame()

    rows = [_coverage_row("all", universe)]
    if "market" in universe.columns:
        for market, group in universe.groupby("market", observed=True):
            rows.append(_coverage_row(f"market:{market}", group))
    return pd.DataFrame(rows)
