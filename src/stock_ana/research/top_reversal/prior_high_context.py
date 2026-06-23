"""Causal prior-high / M-top structure features for top-reversal research.

An up-trend continuation typically makes a clean higher high over the prior
ZigZag highs.  A true top often fails to *close* above the prior swing high — a
bull-trap / M-top — even when its intraday high marginally pokes through.  These
causal features encode (1) whether the candidate held its close above the prior
swing high, and (2) whether it forms a strict (not a few-day) double-top shape
with a prior peak of similar height.

All features are strictly causal: only swing highs whose confirmation bar
(``visible_pos = pos + swing_length``) is visible by the candidate's
``score_asof_pos`` and that occur before the candidate top are used.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

PRIOR_HIGH_FEATURES: tuple[str, ...] = (
    "top_high_vs_prior_high_pct",
    "top_close_vs_prior_high_pct",
    "top_close_above_prior_high",
    "top_m_shape",
    "top_m_separation_bars",
    "top_m_price_diff_pct",
)

SWING_LENGTH = 5
# 严格双顶：两个头之间至少间隔 15 根（约 3 周），杜绝上涨中继回调里几日的弱 M 结构。
M_MIN_SEPARATION = 15
M_MAX_SEPARATION = 120
M_PRICE_TOL_PCT = 3.0

_DEFAULTS: dict[str, object] = {
    "top_high_vs_prior_high_pct": np.nan,
    "top_close_vs_prior_high_pct": np.nan,
    "top_close_above_prior_high": 0,
    "top_m_shape": 0,
    "top_m_separation_bars": np.nan,
    "top_m_price_diff_pct": np.nan,
}


def _as_pos(value: object, default: int = -1) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _causal_swing_highs(df: pd.DataFrame) -> np.ndarray:
    from stock_ana.research.top_reversal.smc_context import _causal_swing_points  # 延迟导入避免循环

    sw = _causal_swing_points(df, SWING_LENGTH)
    if sw.empty:
        return np.empty((0, 3))
    return sw[sw["direction"] == 1][["pos", "visible_pos", "level"]].to_numpy(dtype=float)


def prior_high_features_for_candidate(
    df: pd.DataFrame,
    row: Mapping[str, object],
    swing_highs: np.ndarray | None = None,
) -> dict[str, object]:
    out = dict(_DEFAULTS)
    top_pos = _as_pos(row.get("top_pos", -1))
    asof_pos = _as_pos(row.get("score_asof_pos", top_pos), top_pos)
    if top_pos < 0 or top_pos >= len(df):
        return out

    high = df["high"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    swh = swing_highs if swing_highs is not None else _causal_swing_highs(df)
    if len(swh) == 0:
        return out

    # 仅用「发生在候选顶之前」且「确认可见日 <= 评分日」的因果前高
    prior = swh[(swh[:, 0] < top_pos) & (swh[:, 1] <= asof_pos)]
    if len(prior) == 0:
        return out

    top_high = float(high[top_pos])
    top_close = float(close[top_pos])
    prior_high = float(prior[-1, 2])  # 最近一个前高
    if prior_high > 0:
        out["top_high_vs_prior_high_pct"] = round((top_high / prior_high - 1) * 100, 2)
        out["top_close_vs_prior_high_pct"] = round((top_close / prior_high - 1) * 100, 2)
        out["top_close_above_prior_high"] = int(top_close > prior_high)

    # 严格 M / 双顶形态：某个前高与当前高同高度(±tol)，且间隔在 [MIN, MAX]
    best: tuple[int, float] | None = None
    for p, _vis, lvl in prior:
        if lvl <= 0:
            continue
        sep = top_pos - int(p)
        if sep < M_MIN_SEPARATION or sep > M_MAX_SEPARATION:
            continue
        diff = abs(top_high / lvl - 1) * 100
        if diff <= M_PRICE_TOL_PCT and (best is None or diff < best[1]):
            best = (sep, diff)
    if best is not None:
        out["top_m_shape"] = 1
        out["top_m_separation_bars"] = best[0]
        out["top_m_price_diff_pct"] = round(best[1], 2)
    return out


def add_prior_high_features(
    dataset: pd.DataFrame,
    symbol_data: Mapping[str, dict] | None = None,
) -> pd.DataFrame:
    """Attach causal prior-high / M-top features to candidate rows."""

    out = dataset.copy()
    for col in PRIOR_HIGH_FEATURES:
        if col not in out.columns:
            out[col] = _DEFAULTS[col]

    data_map: dict[tuple[str, str], pd.DataFrame] = {}
    if symbol_data:
        for item in symbol_data.values():
            mk = str(item.get("market", ""))
            sym = str(item.get("symbol", item.get("sym", "")))
            d = item.get("df")
            if mk and sym and isinstance(d, pd.DataFrame):
                data_map[(mk, sym)] = d
    if out.empty or not data_map:
        return out

    df_cache: dict[tuple[str, str], pd.DataFrame] = {}
    swing_cache: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for _, row in out.iterrows():
        key = (str(row.get("market", "")), str(row.get("sym", "")))
        df = df_cache.get(key)
        if df is None and key not in df_cache:
            d = data_map.get(key)
            if d is not None:
                d = d.copy()
                d.columns = [str(c).lower() for c in d.columns]
                d.index = pd.to_datetime(d.index)
                d = d.sort_index()
                df_cache[key] = d
                swing_cache[key] = _causal_swing_highs(d)
                df = d
            else:
                df_cache[key] = None  # type: ignore[assignment]
        if df is None:
            rows.append({c: out.at[row.name, c] for c in PRIOR_HIGH_FEATURES})
            continue
        rows.append(prior_high_features_for_candidate(df, row, swing_highs=swing_cache.get(key)))

    feature_df = pd.DataFrame(rows, index=out.index)
    for col in PRIOR_HIGH_FEATURES:
        out[col] = feature_df[col]
    return out
