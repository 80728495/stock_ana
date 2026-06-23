"""Valuation (PE) features — market-separated, market-relative.

US uses forward PE (stockanalysis.com); HK/CN use trailing PE_TTM (Futu OpenD),
because forward estimates are not readily available there.  The three markets'
valuation centers differ completely, so PE is **normalized within each market**
(percentile rank) — never compared across markets.

Caveat: the PE values are a *current snapshot*, not the PE at each historical
candidate's date.  So this behaves as a static per-symbol attribute ("is this a
richly-valued name within its market"), most accurate for live scoring and an
approximation for historical training.  Documented intentionally.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from stock_ana.config import DATA_DIR

VALUATION_FEATURES: tuple[str, ...] = (
    "valuation_pe",
    "valuation_pe_pct_mkt",
)

_DEFAULTS = {"valuation_pe": np.nan, "valuation_pe_pct_mkt": np.nan}


def _load_pe_map() -> dict[tuple[str, str], float]:
    """(market, symbol) -> PE（US 前向，HK/CN trailing TTM）。"""
    pe: dict[tuple[str, str], float] = {}
    fdir = DATA_DIR / "cache" / "fundamentals"

    us_path = fdir / "us_forward_pe.csv"
    if us_path.exists():
        us = pd.read_csv(us_path)
        for _, r in us.iterrows():
            t = str(r.get("ticker", "")).strip().upper()
            v = r.get("forward_pe")
            if not t:
                continue
            if pd.isna(v):
                v = r.get("pe")  # 退化到 trailing
            if pd.notna(v) and 0 < float(v) < 1000:
                pe[("US", t)] = float(v)

    futu_path = fdir / "futu_pe.csv"
    if futu_path.exists():
        ft = pd.read_csv(futu_path)
        for _, r in ft.iterrows():
            mk = str(r.get("market", "")).strip()
            sym = str(r.get("symbol", "")).strip()
            if mk == "US":
                continue  # US 用 forward
            sym = sym.zfill(5) if mk == "HK" else (sym.zfill(6) if mk == "CN" else sym)
            v = r.get("pe_ttm")
            if pd.isna(v):
                v = r.get("pe")
            if mk and sym and pd.notna(v) and 0 < float(v) < 1000:
                pe[(mk, sym)] = float(v)
    return pe


def add_valuation_features(
    dataset: pd.DataFrame,
    symbol_data: Mapping[str, dict] | None = None,
) -> pd.DataFrame:
    """Attach market-relative PE features (static per-symbol attribute)."""

    out = dataset.copy()
    for col in VALUATION_FEATURES:
        if col not in out.columns:
            out[col] = _DEFAULTS[col]
    if out.empty:
        return out

    pe_map = _load_pe_map()
    if not pe_map:
        return out

    markets = out["market"].astype(str)
    syms = out["sym"].astype(str)
    out["valuation_pe"] = [pe_map.get((mk, s), np.nan) for mk, s in zip(markets, syms, strict=False)]

    # 市场内分位：用各市场「唯一标的」的 PE 排名，再映射回候选（避免按候选频次加权）
    pe_col = pd.to_numeric(out["valuation_pe"], errors="coerce")
    pct = pd.Series(np.nan, index=out.index)
    for mk in markets.unique():
        mask = markets == mk
        uniq = (
            pd.DataFrame({"sym": syms[mask], "pe": pe_col[mask]})
            .dropna()
            .drop_duplicates("sym")
        )
        if len(uniq) >= 5:
            uniq["rank"] = uniq["pe"].rank(pct=True) * 100
            rank_map = dict(zip(uniq["sym"], uniq["rank"], strict=False))
            pct.loc[mask] = syms[mask].map(rank_map)
    out["valuation_pe_pct_mkt"] = pct.round(1)
    return out
