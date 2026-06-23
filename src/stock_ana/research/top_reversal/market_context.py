"""Market-index context features for top-reversal candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.config import DATA_DIR


CHINA_HK_US_SYMBOLS = {
    "PDD", "BABA", "MPNGY", "HSAI", "FUTU", "TME", "NIO", "XPEV", "BIDU", "GCT",
    "NTES", "JD", "LI", "BILI", "MNSO", "TCOM", "KC",
    "CWEB", "KWEB", "FXI", "YINN", "YANG", "CHAU",
}


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index.name = "date"
    return out.sort_index()


def _load_index_cache(symbol: str) -> pd.DataFrame | None:
    path = DATA_DIR / "cache" / "hk" / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        return _normalize_df(pd.read_parquet(path))
    except Exception as exc:
        logger.warning(f"指数缓存读取失败 {path}: {exc}")
        return None


def _index_return_to_date(index_df: pd.DataFrame | None, date_value, lookback: int) -> float:
    if index_df is None or pd.isna(date_value):
        return float("nan")
    loc = index_df.index.searchsorted(pd.Timestamp(date_value), side="right") - 1
    if loc - lookback < 0:
        return float("nan")
    close = index_df["close"].astype(float)
    prev = float(close.iloc[loc - lookback])
    if prev <= 0:
        return float("nan")
    return (float(close.iloc[loc]) / prev - 1) * 100


def add_index_squeeze_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add China/HK index-squeeze context to candidate rows.

    These features intentionally separate raw index returns from scoped returns
    that only apply to HK and China ADR names. The scoped columns are what the
    model uses, preventing HSTECH/HIS moves from becoming a broad date factor
    for unrelated US/CN names.
    """

    out = dataset.copy()
    top_dates = pd.to_datetime(out["top_date"], errors="coerce")
    china_hk_focus = (
        out["market"].eq("HK")
        | (out["market"].eq("US") & out["sym"].astype(str).isin(CHINA_HK_US_SYMBOLS))
    ).astype(int)
    out["china_hk_focus"] = china_hk_focus
    out["max_ret_5_10_20"] = out[["prior_ret_5d", "prior_ret_10d", "prior_ret_20d"]].max(axis=1)
    out["short_spike_like"] = (
        (pd.to_numeric(out["bars_from_anchor_low"], errors="coerce") <= 25)
        & (pd.to_numeric(out["rise_from_anchor_low_pct"], errors="coerce") >= 45)
        & (pd.to_numeric(out["max_ret_5_10_20"], errors="coerce") >= 35)
    ).astype(int)
    out["weak_confirm_short_spike"] = (
        (out["short_spike_like"] == 1)
        & (pd.to_numeric(out["confirm_drop_from_top_pct"], errors="coerce") > -4)
    ).astype(int)
    out["china_hk_short_spike"] = out["china_hk_focus"] * out["short_spike_like"]

    hsi = _load_index_cache("800000")
    hstech = _load_index_cache("800700")
    for lookback in (5, 10, 20, 40):
        hsi_ret = top_dates.apply(lambda x, lb=lookback: _index_return_to_date(hsi, x, lb))
        hstech_ret = top_dates.apply(lambda x, lb=lookback: _index_return_to_date(hstech, x, lb))
        out[f"hsi_ret_{lookback}d"] = hsi_ret.round(2)
        out[f"hstech_ret_{lookback}d"] = hstech_ret.round(2)
        out[f"china_hk_hsi_ret_{lookback}d"] = (hsi_ret * china_hk_focus).round(2)
        out[f"china_hk_hstech_ret_{lookback}d"] = (hstech_ret * china_hk_focus).round(2)

    hstech_squeeze = (
        (out["china_hk_focus"] == 1)
        & (
            (pd.to_numeric(out["hstech_ret_10d"], errors="coerce") >= 15)
            | (pd.to_numeric(out["hstech_ret_20d"], errors="coerce") >= 20)
        )
    )
    out["hstech_squeeze_10d"] = (
        (out["china_hk_focus"] == 1)
        & (pd.to_numeric(out["hstech_ret_10d"], errors="coerce") >= 15)
    ).astype(int)
    out["hstech_squeeze_20d"] = (
        (out["china_hk_focus"] == 1)
        & (pd.to_numeric(out["hstech_ret_20d"], errors="coerce") >= 20)
    ).astype(int)
    out["china_hk_index_squeeze_spike"] = (hstech_squeeze & (out["short_spike_like"] == 1)).astype(int)
    out["china_hk_index_squeeze_weak_confirm"] = (
        hstech_squeeze & (out["weak_confirm_short_spike"] == 1)
    ).astype(int)
    return out

