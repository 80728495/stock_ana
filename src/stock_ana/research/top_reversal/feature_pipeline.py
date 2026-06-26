"""Feature pipeline for top-reversal research candidates."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from stock_ana.research.top_reversal.market_context import add_index_squeeze_features
from stock_ana.research.top_reversal.smc_context import add_smc_ob_features
from stock_ana.research.top_reversal.macro_micro_context import add_macro_micro_features
from stock_ana.research.top_reversal.prior_high_context import add_prior_high_features
from stock_ana.research.top_reversal.growth_context import add_growth_features
from stock_ana.research.top_reversal.valuation_context import add_valuation_features
from stock_ana.research.top_reversal.vegas_context import add_mid_vegas_features


def _flag(out: pd.DataFrame, col: str) -> pd.Series:
    """Return a numeric 0/1 flag series for a possibly missing column."""

    if col not in out.columns:
        return pd.Series(0, index=out.index, dtype=int)
    return pd.to_numeric(out[col], errors="coerce").fillna(0).ne(0).astype(int)


def add_candle_interaction_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add explicit candle-pattern overlap features for scoring candidates.

    Existing candle fields are mostly source-specific.  These interaction
    fields make the useful case explicit: an SMC early candidate that also has
    a high-confidence candle-top signal in the same merged candidate window.
    """

    out = dataset.copy()
    if out.empty:
        return out

    shadow = _flag(out, "has_shadow") | _flag(out, "recalled_by_shadow")
    doji = _flag(out, "has_doji") | _flag(out, "recalled_by_doji")
    gap_fail = _flag(out, "has_gap_fail") | _flag(out, "recalled_by_gap_fail")
    smc_early = _flag(out, "recalled_by_smc_early")

    shooting_star = _flag(out, "shadow_is_shooting_star")
    strict_doji = _flag(out, "doji_is_strict_doji")

    old_candle = (shadow | doji).astype(int)
    any_candle = (old_candle | gap_fail).astype(int)
    strict_old_candle = (shooting_star | strict_doji).astype(int)
    old_count = shadow + doji
    candle_count = old_count + gap_fail
    score_max = pd.to_numeric(out.get("score_max", pd.Series(0, index=out.index)), errors="coerce").fillna(0)

    out["candle_top_pattern"] = any_candle
    out["candle_top_pattern_count"] = candle_count.astype(int)
    out["candle_old_top_pattern"] = old_candle
    out["candle_old_top_pattern_count"] = old_count.astype(int)
    out["candle_strict_old_top_pattern"] = strict_old_candle
    out["smc_early_with_shadow"] = (smc_early & shadow).astype(int)
    out["smc_early_with_doji"] = (smc_early & doji).astype(int)
    out["smc_early_with_gap_fail"] = (smc_early & gap_fail).astype(int)
    out["smc_early_with_any_candle"] = (smc_early & any_candle).astype(int)
    out["smc_early_with_old_candle"] = (smc_early & old_candle).astype(int)
    out["smc_early_with_strict_old_candle"] = (smc_early & strict_old_candle).astype(int)
    out["smc_early_candle_score_max"] = smc_early * score_max
    return out


def add_research_features(
    dataset: pd.DataFrame,
    symbol_data: Mapping[str, dict] | None = None,
) -> pd.DataFrame:
    """Apply dataset-level feature builders.

    `symbol_data` is used by feature builders that need the original OHLCV
    data, such as SMC or resistance-zone matching.
    """

    out = add_mid_vegas_features(dataset, symbol_data=symbol_data)
    out = add_index_squeeze_features(out)
    out = add_smc_ob_features(out, symbol_data=symbol_data)
    out = add_prior_high_features(out, symbol_data=symbol_data)
    out = add_macro_micro_features(out, symbol_data=symbol_data)
    out = add_valuation_features(out, symbol_data=symbol_data)
    out = add_growth_features(out, symbol_data=symbol_data)
    out = add_candle_interaction_features(out)
    return out
