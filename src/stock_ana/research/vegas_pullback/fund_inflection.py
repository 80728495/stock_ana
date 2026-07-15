"""基本面拐点特征（PIT，as-of 因果）：增长的一阶差分。

背景：静态增长水平区分不了「困境反转的 W1」——华虹 2024-11 大二浪起点时
盈利增速仍是 -66%（静态看是烂票），但相比两个季度前的 -83% 在**改善**。
决定「W1 能否孕育大二浪」的常常是基本面拐点（增速的方向），而非水平。

特征（复用 top_reversal.pit_fundamentals.pit_growth 的 as-of 查询）：
  fund_earn_accel : earnings YoY 增速 asof 当前 − asof 200 天前（pp）
  fund_rev_accel  : revenue  YoY 增速 asof 当前 − asof 200 天前（pp）

200 天回看兼容三市场披露节奏（HK 半年报、US/CN 季报），保证 prev 至少隔
一个披露期。两次查询均以各自时点的已披露数据计算——因果。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FUND_INFLECTION_FEATURES: tuple[str, ...] = (
    "fund_earn_accel", "fund_rev_accel",
)

_PREV_LAG_DAYS = 200


def add_fund_inflection_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """按 (market, sym, score_asof_date) 附加增长加速度特征。"""
    from stock_ana.research.top_reversal.pit_fundamentals import pit_growth

    out = dataset.copy()
    for c in FUND_INFLECTION_FEATURES:
        if c not in out.columns:
            out[c] = np.nan
    if out.empty:
        return out

    asof = pd.to_datetime(out.get("score_asof_date", out.get("signal_date")), errors="coerce")
    markets = out["market"].astype(str)
    syms = out["sym"].astype(str)

    cache: dict[tuple, tuple] = {}

    def growth_cached(mk: str, sym: str, d: pd.Timestamp) -> tuple:
        # 按月缓存：同一披露期内增长值不变，避免逐行重复查询
        key = (mk, sym, d.year, d.month)
        if key not in cache:
            cache[key] = pit_growth(mk, sym, d)
        return cache[key]

    ea, ra = [], []
    for mk, sym, d in zip(markets, syms, asof, strict=False):
        if pd.isna(d):
            ea.append(np.nan); ra.append(np.nan)
            continue
        g_now = growth_cached(mk, sym, d)
        g_prev = growth_cached(mk, sym, d - pd.Timedelta(days=_PREV_LAG_DAYS))
        e = g_now[0] - g_prev[0] if pd.notna(g_now[0]) and pd.notna(g_prev[0]) else np.nan
        r = g_now[1] - g_prev[1] if pd.notna(g_now[1]) and pd.notna(g_prev[1]) else np.nan
        ea.append(e); ra.append(r)

    out["fund_earn_accel"] = np.round(pd.to_numeric(pd.Series(ea, index=out.index), errors="coerce"), 2)
    out["fund_rev_accel"] = np.round(pd.to_numeric(pd.Series(ra, index=out.index), errors="coerce"), 2)
    return out
