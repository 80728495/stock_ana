"""Fundamental growth features — point-in-time, the general replacement for static industry tags.

行业差异/股价走势的本质是核心基本面（增长）。与其打 `is_semiconductor` 这种静态身份标签
（不通用、会随 regime 轮动失效），不如用「当前增长率」这种通用、动态特征：

  earnings_growth / revenue_growth — 个股盈利/营收 **TTM YoY**（最近4季滚动同比）
  sector_earnings_growth_mean      — 子赛道成分股盈利增长均值 = 该赛道当前「基本面热度」
  valuation_peg                    — PE/增长（绝对 PE 无意义，PEG 才有；仅 growth>5）

**point-in-time**：每个候选用 ``score_asof_date`` 当时已披露的最近一期算（见 pit_fundamentals）：
US stockanalysis 季度→TTM YoY、CN akshare 累计宽表→TTM YoY、HK 仅年报→年度 YoY（年末等价 TTM）。
口径统一 YoY；三市场永不合并，per-market 模型对跨市场口径差异不敏感。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

GROWTH_FEATURES: tuple[str, ...] = (
    "earnings_growth",
    "revenue_growth",
    "sector_earnings_growth_mean",
    "valuation_peg",  # PE/growth：估值相对增长（绝对 PE 无意义，PEG 才有）
)

_DEFAULTS = {c: np.nan for c in GROWTH_FEATURES}


def _asof_series(out: pd.DataFrame) -> pd.Series:
    for col in ("score_asof_date", "confirm_date", "top_date"):
        if col in out.columns:
            return pd.to_datetime(out[col], errors="coerce")
    return pd.Series(pd.NaT, index=out.index)


def add_growth_features(dataset: pd.DataFrame, symbol_data: Mapping[str, dict] | None = None) -> pd.DataFrame:
    from stock_ana.research.top_reversal.macro_micro_context import _build_sector_map
    from stock_ana.research.top_reversal.pit_fundamentals import _norm_sym, pit_growth

    out = dataset.copy()
    for c in GROWTH_FEATURES:
        if c not in out.columns:
            out[c] = _DEFAULTS[c]
    if out.empty:
        return out

    asof = _asof_series(out)
    markets = out["market"].astype(str)
    syms = out["sym"].astype(str)
    eg, rg = [], []
    for mk, sym, a in zip(markets, syms, asof, strict=False):
        g = pit_growth(mk, sym, a)
        eg.append(g[0]); rg.append(g[1])
    out["earnings_growth"] = np.round(pd.to_numeric(pd.Series(eg, index=out.index), errors="coerce"), 2)
    out["revenue_growth"] = np.round(pd.to_numeric(pd.Series(rg, index=out.index), errors="coerce"), 2)

    # 子赛道盈利增长均值（赛道基本面热度）：按 (market, 子赛道) 对 PIT earnings_growth 求均值。
    # US 用 SIC 子组 / HK industry / CN cn_industry_map；不跨市场混合。
    sector_map = _build_sector_map()
    sec = [(sector_map.get((mk, _norm_sym(mk, s)), {}) or {}).get("sector") for mk, s in zip(markets, syms, strict=False)]
    tmp = out.assign(_sec=sec)
    means = tmp[out["earnings_growth"].notna()].groupby(["market", "_sec"], observed=True)["earnings_growth"].mean()
    out["sector_earnings_growth_mean"] = np.round(pd.Series(
        [means.get((mk, sc), np.nan) if sc else np.nan for mk, sc in zip(markets, sec, strict=False)],
        index=out.index), 2)

    # PEG = PE / 盈利增长（只对有意义的正增长 growth>5；PE 取 as-of valuation_pe，估值上下文已先算）
    pe_vals = (pd.to_numeric(out["valuation_pe"], errors="coerce")
               if "valuation_pe" in out.columns else pd.Series(np.nan, index=out.index))
    out["valuation_peg"] = [round(p / g, 2) if (pd.notna(p) and pd.notna(g) and g > 5) else np.nan
                            for p, g in zip(pe_vals, out["earnings_growth"], strict=False)]
    return out
