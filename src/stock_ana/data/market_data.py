"""
统一市场数据门面（local-cache facade）。

目标：
  - 给策略/回测提供统一的数据读取入口
  - 避免在业务层重复拼接 us/ndx100/hk 目录和回退逻辑
  - 保留底层 fetcher_* 的市场实现差异
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from stock_ana.data.fetcher import load_all_ndx100_data, load_all_us_data
from stock_ana.data.fetcher_cn import load_all_cn_data
from stock_ana.data.fetcher_hk import load_all_hk_data
from stock_ana.data.list_manager import parse_watchlist  # re-export

Market = Literal["us", "ndx100", "hk", "cn"]
Universe = Literal["us", "ndx100", "hk", "cn", "us+ndx100", "all"]

# ── Market registry ─────────────────────────────────────────────────────────
# Maps market name → (subdirectory under CACHE_DIR, display label)
# Adding a new data source only requires a new entry here.
_MARKET_REGISTRY: dict[str, tuple[str, str]] = {
    "us":     ("us",     "US"),
    "ndx100": ("ndx100", "US"),
    "hk":     ("hk",     "HK"),
    "cn":     ("cn",     "CN"),
}

_MARKET_CACHE: dict[tuple[str, int], dict[str, pd.DataFrame]] = {}


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名/索引格式，确保上层逻辑一致。"""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    return out


def _filter_min_history(data: dict[str, pd.DataFrame], min_history: int) -> dict[str, pd.DataFrame]:
    """Drop symbols whose cached history is shorter than the requested minimum."""
    if min_history <= 0:
        return data
    return {k: v for k, v in data.items() if len(v) >= min_history}


def load_market_data(market: Market, min_history: int = 0) -> dict[str, pd.DataFrame]:
    """
    读取单市场本地缓存数据（不触发网络请求）。

    Args:
        market: "us" | "ndx100" | "hk"
        min_history: 最少历史行数过滤
    """
    cache_key = (market, int(min_history))
    cached = _MARKET_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if market == "us":
        data = load_all_us_data()
    elif market == "ndx100":
        data = load_all_ndx100_data()
    elif market == "hk":
        data = load_all_hk_data()
    elif market == "cn":
        data = load_all_cn_data()
    else:
        raise ValueError(f"Unsupported market: {market}")

    normalized = {k: _normalize_df(v) for k, v in data.items()}
    filtered = _filter_min_history(normalized, min_history)
    _MARKET_CACHE[cache_key] = filtered
    return filtered


def load_universe_data(universe: Universe = "us+ndx100", min_history: int = 0) -> dict[str, pd.DataFrame]:
    """
    读取组合市场数据，并按 ticker 去重。

    去重优先级：
      - universe="us+ndx100" 时，US 优先，NDX100 仅补缺
      - universe="all" 时，先 US，再 NDX100 补缺，再 HK 补缺
    """
    if universe == "us":
        return load_market_data("us", min_history=min_history)
    if universe == "ndx100":
        return load_market_data("ndx100", min_history=min_history)
    if universe == "hk":
        return load_market_data("hk", min_history=min_history)

    merged: dict[str, pd.DataFrame] = {}

    if universe in ("us+ndx100", "all"):
        for t, df in load_market_data("us", min_history=min_history).items():
            merged[t] = df
        for t, df in load_market_data("ndx100", min_history=min_history).items():
            if t not in merged:
                merged[t] = df

    if universe == "all":
        for t, df in load_market_data("hk", min_history=min_history).items():
            if t not in merged:
                merged[t] = df
        for t, df in load_market_data("cn", min_history=min_history).items():
            if t not in merged:
                merged[t] = df

    return merged


def build_watchlist(markets: list[str] | None = None) -> dict:
    """Build a scan watchlist from one or more cached market directories.

    Each entry in the returned dict:
        symbol → (market_label, display_name, parquet_path)

    ``markets`` is an ordered list of keys from ``_MARKET_REGISTRY``.
    When a symbol exists in multiple markets the first occurrence wins
    (preserving dedup priority).

    If ``markets`` is None, the Shawn watch-list is used:
      - US symbols are looked up in ["us", "ndx100"] in that order
      - HK / CN symbols are looked up in their respective directories

    Adding a new data source only requires:
      1. A new entry in ``_MARKET_REGISTRY`` above.
      2. Optionally, a new section in ``watchlist.md`` (for watchlist mode).
    """
    from stock_ana.config import CACHE_DIR

    if markets is not None:
        # Generic mode: scan all parquet files in the requested cache dirs
        watchlist: dict = {}
        for mkt in markets:
            if mkt not in _MARKET_REGISTRY:
                raise ValueError(f"Unknown market '{mkt}'. Known: {list(_MARKET_REGISTRY)}")
            subdir, label = _MARKET_REGISTRY[mkt]
            cache_dir = CACHE_DIR / subdir
            if not cache_dir.exists():
                continue
            for p in sorted(cache_dir.glob("*.parquet")):
                sym = p.stem.upper() if label in ("US",) else p.stem
                if sym not in watchlist:
                    watchlist[sym] = (label, sym, p)
        return watchlist

    # Shawn-list mode: honour per-symbol cache-dir preference
    parsed = parse_watchlist()
    watchlist = {}

    for entry in parsed.get("us", []):
        sym = entry["symbol"]
        for subdir in ["us", "ndx100"]:
            p = CACHE_DIR / subdir / f"{sym}.parquet"
            if p.exists():
                watchlist[sym] = ("US", entry["name"], p)
                break

    for entry in parsed.get("hk", []):
        sym = entry["symbol"]
        p = CACHE_DIR / "hk" / f"{sym}.parquet"
        if p.exists():
            watchlist[sym] = ("HK", entry["name"], p)

    for mkt_key, label in [(k, v[1]) for k, v in _MARKET_REGISTRY.items()
                            if k not in ("us", "ndx100", "hk")]:
        for entry in parsed.get(mkt_key, []):
            sym = entry["symbol"]
            p = CACHE_DIR / _MARKET_REGISTRY[mkt_key][0] / f"{sym}.parquet"
            if p.exists():
                watchlist[sym] = (label, entry["name"], p)

    return watchlist


def load_symbol_data(symbol: str, universe: Universe = "us+ndx100") -> pd.DataFrame | None:
    """按统一 universe 规则读取单只标的；不存在时返回 None。"""
    data = load_universe_data(universe=universe, min_history=0)
    return data.get(symbol)


def clear_market_data_cache() -> None:
    """清空进程内市场数据缓存。"""
    _MARKET_CACHE.clear()


# ── Shawn watchlist data loader ───────────────────────────────────────────────

def load_watchlist_data(
    path: Path | None = None,
    min_history: int = 0,
) -> dict[str, dict]:
    """Load local-cache data for all tickers in the shawn watchlist.

    No auto-fetch is performed; symbols without a local parquet file are silently
    skipped.

    Returns:
        dict keyed by ticker/code:
            {'market': 'US'|'HK', 'symbol': ..., 'name': ..., 'df': DataFrame}
    """
    from stock_ana.config import CACHE_DIR  # avoid circular import at module level

    parsed = parse_watchlist(path)
    result: dict[str, dict] = {}

    for entry in parsed["us"]:
        sym = entry["symbol"]
        df = None
        for cache_dir in [CACHE_DIR / "us", CACHE_DIR / "ndx100"]:
            p = cache_dir / f"{sym}.parquet"
            if p.exists():
                try:
                    raw = pd.read_parquet(p)
                    df = _normalize_df(raw)
                except Exception:
                    pass
                break
        if df is not None and len(df) >= min_history:
            result[sym] = {"market": "US", "symbol": sym, "name": entry["name"], "df": df}

    for entry in parsed["hk"]:
        sym = entry["symbol"]
        p = CACHE_DIR / "hk" / f"{sym}.parquet"
        if p.exists():
            try:
                raw = pd.read_parquet(p)
                df = _normalize_df(raw)
                if len(df) >= min_history:
                    result[sym] = {"market": "HK", "symbol": sym, "name": entry["name"], "df": df}
            except Exception:
                pass

    for entry in parsed.get("cn", []):
        sym = entry["symbol"]
        p = CACHE_DIR / "cn" / f"{sym}.parquet"
        if p.exists():
            try:
                raw = pd.read_parquet(p)
                df = _normalize_df(raw)
                if len(df) >= min_history:
                    result[sym] = {"market": "CN", "symbol": sym, "name": entry["name"], "df": df}
            except Exception:
                pass

    return result
