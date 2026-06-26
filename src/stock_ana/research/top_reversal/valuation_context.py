"""Valuation features — point-in-time, market-separated, market-relative, multi-multiple.

估值绝对值无意义，只有「相对」才有意义：相对市场（市场内分位）、相对增长（PEG，见 growth_context）。
不同股性适用不同乘数——盈利成长股看 PE，重资产/代工(中芯/华虹)看 PB，SaaS 看 PS——
故同时提供 PE/PB/PS，由 SIC 子赛道 + 树自行选用。每个乘数都做 **市场内分位** 归一
(三市场估值中枢不同，绝不跨市场比)。

**point-in-time**：每个候选用 ``score_asof_date`` 当时已披露的最近一期基本面 + as-of 价重构，
不再用今日快照（杜绝 look-ahead；实现见 pit_fundamentals）。今天的实时候选 asof=今天 →
取最新一期 = 与快照等价，实时打分不受影响。详见 docs/top_reversal_current_system.md §0.8。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

VALUATION_FEATURES: tuple[str, ...] = (
    "valuation_pe", "valuation_pe_pct_mkt", "valuation_pe_pct_sector",
    "valuation_pb", "valuation_pb_pct_mkt", "valuation_pb_pct_sector",
    "valuation_ps", "valuation_ps_pct_mkt", "valuation_ps_pct_sector",
)

# 子赛道内分位的最小同业样本数（不足则该分位置 NaN，仍由市场内分位兜底）
_MIN_SECTOR_PEERS = 8

_DEFAULTS = {c: np.nan for c in VALUATION_FEATURES}


def _asof_series(out: pd.DataFrame) -> pd.Series:
    for col in ("score_asof_date", "confirm_date", "top_date"):
        if col in out.columns:
            return pd.to_datetime(out[col], errors="coerce")
    return pd.Series(pd.NaT, index=out.index)


def _coarse_sector(sec: object) -> str | None:
    """US SIC 3 位主组 → 2 位大类（如 US_SIC367 → US_SIC36），作为分层回退的中间层。

    HK/CN 行业映射只有一层 → 返回 None（直接回退市场）。回退到 SIC2 比回退到全市场更优：
    仍保留行业归属（电子大类内排名 > 全市场混排），是严格更优的同侪集。
    """
    if isinstance(sec, str) and sec.startswith("US_SIC"):
        digits = sec[len("US_SIC"):]
        if digits.isdigit() and len(digits) >= 3:  # 仅 3 位→2 位；已是 2 位的直接回退市场
            return "US_SIC" + digits[:-1]
    return None


def add_valuation_features(dataset: pd.DataFrame, symbol_data: Mapping[str, dict] | None = None) -> pd.DataFrame:
    """Attach point-in-time PE/PB/PS (as-of score date) + market-relative percentile."""

    from stock_ana.research.top_reversal.pit_fundamentals import pit_valuation

    out = dataset.copy()
    for c in VALUATION_FEATURES:
        if c not in out.columns:
            out[c] = _DEFAULTS[c]
    if out.empty:
        return out

    asof = _asof_series(out)
    markets = out["market"].astype(str)
    syms = out["sym"].astype(str)
    pe, pb, ps = [], [], []
    for mk, sym, a in zip(markets, syms, asof, strict=False):
        v = pit_valuation(mk, sym, a)
        pe.append(v[0]); pb.append(v[1]); ps.append(v[2])
    out["valuation_pe"] = np.round(pd.to_numeric(pd.Series(pe, index=out.index), errors="coerce"), 2)
    out["valuation_pb"] = np.round(pd.to_numeric(pd.Series(pb, index=out.index), errors="coerce"), 2)
    out["valuation_ps"] = np.round(pd.to_numeric(pd.Series(ps, index=out.index), errors="coerce"), 2)

    # 市场内分位（绝不跨市场比；每行按 as-of 自身值排名）
    for metric in ("pe", "pb", "ps"):
        col = f"valuation_{metric}"
        pct = pd.Series(np.nan, index=out.index)
        for mk in markets.unique():
            mask = markets == mk
            v = pd.to_numeric(out.loc[mask, col], errors="coerce")
            if v.notna().sum() >= 5:
                pct.loc[mask] = v.rank(pct=True) * 100
        out[f"{col}_pct_mkt"] = pct.round(1)

    # 行业内 as-of 分位（核心归一化）：不同行业 PE/PB/PS 基准不同（银行 PE~5 vs 科技 PE~40），
    # 绝对值跨行业不可比。把候选估值排进「全行业宇宙成员在 ≤当月最近一个≥8 同侪的月」的分布
    # （sector_valuation.sector_pct，SIC3→US-SIC2）。
    # 纪律（实测验证）：① 跨全行业(市场级)排名 = 绝对值单调变换、无意义，**绝不回退市场**，行业内
    # 凑不齐→NaN 由绝对值兜底；② 候选集内「跨期池化」排名会用未来候选=未来函数泄漏（CN 0.83→0.71），
    # 故改用全行业成员 as-of 分布。候选用 panel-一致估值排名。
    from stock_ana.research.top_reversal.macro_micro_context import _build_sector_map
    from stock_ana.research.top_reversal.pit_fundamentals import _norm_sym
    from stock_ana.research.top_reversal.sector_valuation import candidate_value, sector_pct
    sector_map = _build_sector_map()
    sectors = pd.Series(
        [(sector_map.get((mk, _norm_sym(mk, s)), {}) or {}).get("sector")
         for mk, s in zip(markets, syms, strict=False)],
        index=out.index, dtype=object)
    sectors_coarse = sectors.map(_coarse_sector)  # US SIC3→SIC2；HK/CN 为 None
    cand_vals = [candidate_value(mk, s, a) for mk, s, a in zip(markets, syms, asof, strict=False)]
    for metric in ("pe", "pb", "ps"):
        pcts = [sector_pct(mk, sec, sec2, a, metric, cv.get(metric))
                for mk, sec, sec2, a, cv in zip(markets, sectors, sectors_coarse, asof, cand_vals, strict=False)]
        out[f"valuation_{metric}_pct_sector"] = np.round(
            pd.to_numeric(pd.Series(pcts, index=out.index), errors="coerce"), 1)
    return out
