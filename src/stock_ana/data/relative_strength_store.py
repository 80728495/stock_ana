"""Causal multi-market relative-strength calculation and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.config import CACHE_DIR, DATA_DIR, OUTPUT_DIR
from stock_ana.data.benchmark_store import BENCHMARKS, load_all_benchmarks

RS_CACHE_DIR = CACHE_DIR / "relative_strength"
RS_OUTPUT_DIR = OUTPUT_DIR / "relative_strength"
RS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_HK_HSTECH_INDUSTRIES = {
    "互动媒体及服务",
    "互联网服务及基础设施",
    "应用软件",
    "支付服务",
    "数码解决方案服务",
    "游戏软件",
}

_HK_MAINLAND_TECH_INDUSTRIES = {
    "半导体",
    "半导体设备与材料",
    "工业零件及器材",
    "新能源物料",
    "消费性电讯设备",
    "消费电子产品",
    "电子零件",
    "电脑及周边器材",
    "能源储存装置",
    "非传统/可再生能源",
    "重型机械",
}

_CN_BENCHMARK_ONLY_CODES = {"000680", "000688", "399006"}


@dataclass(frozen=True)
class StockMeta:
    market: str
    symbol: str
    name: str = ""
    industry: str = ""


@dataclass(frozen=True)
class BenchmarkChoice:
    prior: str
    candidates: tuple[str, ...]


def benchmark_choice(meta: StockMeta) -> BenchmarkChoice:
    """Return the causal benchmark candidate set implied by board and industry."""
    if meta.market == "US":
        return BenchmarkChoice("US_QQQ", ("US_QQQ",))

    if meta.market == "CN":
        if meta.symbol.startswith(("300", "301")):
            return BenchmarkChoice(
                "CN_CHINEXT",
                ("CN_CHINEXT", "CN_STAR_COMPOSITE", "CN_STAR50"),
            )
        if meta.symbol.startswith(("688", "689")):
            return BenchmarkChoice(
                "CN_STAR_COMPOSITE",
                ("CN_STAR_COMPOSITE", "CN_STAR50", "CN_CHINEXT"),
            )
        return BenchmarkChoice(
            "CN_CHINEXT",
            ("CN_CHINEXT", "CN_STAR_COMPOSITE", "CN_STAR50"),
        )

    if meta.market == "HK":
        if meta.industry in _HK_HSTECH_INDUSTRIES:
            return BenchmarkChoice("HK_HSTECH", ("HK_HSTECH", "HK_HSI"))
        if meta.industry in _HK_MAINLAND_TECH_INDUSTRIES:
            return BenchmarkChoice(
                "CN_STAR_COMPOSITE",
                ("CN_STAR_COMPOSITE", "CN_STAR50", "CN_CHINEXT", "HK_HSTECH", "HK_HSI"),
            )
        return BenchmarkChoice("HK_HSI", ("HK_HSI", "HK_HSTECH"))

    raise ValueError(f"不支持的市场: {meta.market}")


def _normalise_price(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty or "close" not in df.columns:
        return None
    clean = df[["close"]].copy()
    clean.index = pd.to_datetime(clean.index).normalize()
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()
    clean["close"] = pd.to_numeric(clean["close"], errors="coerce")
    clean = clean.dropna(subset=["close"])
    return clean.loc[clean["close"] > 0]


def _load_hk_industry_map() -> dict[str, str]:
    path = DATA_DIR / "hk_industry_map.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    return {
        str(row.futu_code).split(".", 1)[-1].zfill(5): str(row.industry)
        for row in df.itertuples(index=False)
    }


def load_market_stock_data(market: str) -> tuple[dict[str, pd.DataFrame], dict[str, StockMeta]]:
    """Load the configured daily RS universe for one market."""
    market = market.upper()
    data: dict[str, pd.DataFrame] = {}
    metadata: dict[str, StockMeta] = {}

    if market == "US":
        from stock_ana.data.fetcher import load_us_local_data
        from stock_ana.data.list_manager import load_us_tech_list

        for entry in load_us_tech_list():
            symbol = str(entry.get("ticker", "")).strip().upper()
            df = _normalise_price(load_us_local_data(symbol))
            if symbol and df is not None:
                data[symbol] = df
                metadata[symbol] = StockMeta(
                    market="US",
                    symbol=symbol,
                    name=str(entry.get("company", "")),
                    industry=str(entry.get("sector", "")),
                )

    elif market == "CN":
        from stock_ana.data.fetcher_cn import load_cn_local
        from stock_ana.data.list_manager import load_cn_hightech_list, load_cn_list

        symbols = list(dict.fromkeys(load_cn_hightech_list() + load_cn_list()))
        for symbol in symbols:
            symbol = str(symbol).strip()
            if symbol in _CN_BENCHMARK_ONLY_CODES:
                continue
            df = _normalise_price(load_cn_local(symbol))
            if df is not None:
                data[symbol] = df
                metadata[symbol] = StockMeta(market="CN", symbol=symbol)

    elif market == "HK":
        from stock_ana.data.fetcher_hk import load_hk_local
        from stock_ana.data.list_manager import load_hk_universe_list

        industries = _load_hk_industry_map()
        for symbol in load_hk_universe_list():
            symbol = str(symbol).strip().zfill(5)
            df = _normalise_price(load_hk_local(symbol))
            if df is not None:
                data[symbol] = df
                metadata[symbol] = StockMeta(
                    market="HK",
                    symbol=symbol,
                    industry=industries.get(symbol, ""),
                )
    else:
        raise ValueError(f"不支持的市场: {market}")

    logger.info(f"RS {market}: 加载 {len(data)} 只股票")
    return data, metadata


def _benchmark_return_on_stock_dates(
    benchmark: pd.DataFrame,
    stock_dates: pd.DatetimeIndex,
) -> pd.Series:
    close = benchmark["close"].copy()
    close.index = pd.to_datetime(close.index).normalize()
    union = close.index.union(stock_dates).sort_values()
    aligned = close.reindex(union).ffill().reindex(stock_dates)
    return np.log(aligned).diff()


def _fit_beta_r2(stock_return: pd.Series, benchmark_return: pd.Series) -> tuple[float, float]:
    paired = pd.concat([stock_return, benchmark_return], axis=1).dropna()
    if len(paired) < 40:
        return np.nan, np.nan
    x = paired.iloc[:, 1].to_numpy(dtype=float)
    y = paired.iloc[:, 0].to_numpy(dtype=float)
    variance = float(np.var(x))
    if variance <= 0:
        return np.nan, np.nan
    beta = float(np.cov(x, y, ddof=0)[0, 1] / variance)
    corr = float(np.corrcoef(x, y)[0, 1])
    r2 = corr * corr if np.isfinite(corr) and beta > 0 else 0.0
    return beta, r2


def build_causal_benchmark_history(
    stock: pd.DataFrame,
    choice: BenchmarkChoice,
    benchmarks: dict[str, pd.DataFrame],
    *,
    lookback: int = 120,
    min_r2: float = 0.10,
    switch_margin: float = 0.05,
) -> pd.DataFrame:
    """Choose beta monthly using only returns available before each rebalance date."""
    dates = pd.DatetimeIndex(stock.index).sort_values()
    stock_return = np.log(stock["close"]).diff()
    candidate_returns = {
        benchmark_id: _benchmark_return_on_stock_dates(benchmarks[benchmark_id], dates)
        for benchmark_id in choice.candidates
        if benchmark_id in benchmarks
    }
    if choice.prior not in candidate_returns:
        raise RuntimeError(f"缺少先验 benchmark 数据: {choice.prior}")

    month_keys = pd.Series(dates.to_period("M"), index=dates)
    rebalance_dates = month_keys[~month_keys.duplicated()].index
    rows: list[dict[str, object]] = []
    current = choice.prior

    for rebalance_date in rebalance_dates:
        history = stock_return.loc[stock_return.index < rebalance_date].tail(lookback)
        scores: dict[str, tuple[float, float]] = {}
        for benchmark_id, benchmark_return in candidate_returns.items():
            bench_history = benchmark_return.loc[benchmark_return.index < rebalance_date].tail(lookback)
            scores[benchmark_id] = _fit_beta_r2(history, bench_history)

        prior_r2 = scores.get(choice.prior, (np.nan, 0.0))[1]
        valid = {
            benchmark_id: values
            for benchmark_id, values in scores.items()
            if np.isfinite(values[1])
        }
        if valid:
            best_id, (best_beta, best_r2) = max(valid.items(), key=lambda item: item[1][1])
            if best_r2 >= min_r2 and (
                best_id == current
                or best_id == choice.prior
                or best_r2 >= (prior_r2 if np.isfinite(prior_r2) else 0.0) + switch_margin
            ):
                current = best_id
        beta, r2 = scores.get(current, (np.nan, np.nan))
        rows.append({
            "effective_date": rebalance_date,
            "benchmark_id": current,
            "beta": beta,
            "r2": r2,
        })

    mapping = pd.DataFrame(rows).set_index("effective_date")
    mapping = mapping.reindex(dates).ffill()
    mapping["benchmark_id"] = mapping["benchmark_id"].fillna(choice.prior)
    return mapping


def compute_stock_rs_history(
    stock: pd.DataFrame,
    mapping: pd.DataFrame,
    benchmarks: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build a chain-linked causal RS series against a time-varying benchmark."""
    dates = pd.DatetimeIndex(stock.index)
    stock_log_return = np.log(stock["close"]).diff()
    selected_benchmark_return = pd.Series(np.nan, index=dates, dtype=float)

    for benchmark_id in mapping["benchmark_id"].dropna().unique():
        benchmark_id = str(benchmark_id)
        benchmark_return = _benchmark_return_on_stock_dates(benchmarks[benchmark_id], dates)
        mask = mapping["benchmark_id"] == benchmark_id
        selected_benchmark_return.loc[mask] = benchmark_return.loc[mask]

    excess_log_return = stock_log_return - selected_benchmark_return
    cumulative = excess_log_return.fillna(0.0).cumsum()
    rs_line = 100.0 * np.exp(cumulative)

    result = pd.DataFrame(index=dates)
    result.index.name = "date"
    result["benchmark_id"] = mapping["benchmark_id"]
    result["benchmark_beta"] = mapping["beta"]
    result["benchmark_r2"] = mapping["r2"]
    result["excess_log_return"] = excess_log_return
    result["rs_line"] = rs_line
    result["rs_return_21d"] = (np.exp(excess_log_return.rolling(21).sum()) - 1.0) * 100.0
    result["rs_return_63d"] = (np.exp(excess_log_return.rolling(63).sum()) - 1.0) * 100.0

    for window in (21, 63):
        mean = rs_line.rolling(window).mean()
        std = rs_line.rolling(window).std().replace(0, np.nan)
        zscore = (rs_line - mean) / std
        result[f"rs_momentum_{window}d"] = zscore.ewm(span=10, adjust=False).mean()

    return result


def _save_market_histories(
    market: str,
    histories: dict[str, pd.DataFrame],
) -> None:
    out_dir = RS_CACHE_DIR / market.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol, df in histories.items():
        df.to_parquet(out_dir / f"{symbol}.parquet")


def compute_market_rs(
    market: str,
    *,
    benchmarks: dict[str, pd.DataFrame] | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute causal RS history and latest benchmark mapping for one market."""
    market = market.upper()
    benchmark_data = benchmarks or load_all_benchmarks()
    stock_data, metadata = load_market_stock_data(market)
    histories: dict[str, pd.DataFrame] = {}
    mapping_rows: list[dict[str, object]] = []

    for i, (symbol, stock) in enumerate(stock_data.items(), 1):
        meta = metadata[symbol]
        choice = benchmark_choice(meta)
        try:
            mapping = build_causal_benchmark_history(stock, choice, benchmark_data)
            history = compute_stock_rs_history(stock, mapping, benchmark_data)
            histories[symbol] = history
            latest = mapping.iloc[-1]
            mapping_rows.append({
                "market": market,
                "symbol": symbol,
                "name": meta.name,
                "industry": meta.industry,
                "prior_benchmark": choice.prior,
                "benchmark_id": latest["benchmark_id"],
                "benchmark_name": BENCHMARKS[str(latest["benchmark_id"])].name,
                "benchmark_beta": latest["beta"],
                "benchmark_r2": latest["r2"],
                "candidates": ",".join(choice.candidates),
                "data_end_date": history.index.max().date().isoformat(),
            })
        except Exception as exc:
            logger.warning(f"RS {market}/{symbol} 计算失败: {exc}")
        if i % 100 == 0:
            logger.info(f"RS {market}: {i}/{len(stock_data)}")

    if histories:
        rank_input = pd.concat(
            {symbol: history["rs_return_63d"] for symbol, history in histories.items()},
            axis=1,
        )
        rank = rank_input.rank(axis=1, pct=True, method="average") * 100.0
        for symbol, history in histories.items():
            history["rs_rank_63d"] = rank[symbol].reindex(history.index)

    if save:
        _save_market_histories(market, histories)

    latest_rows: list[dict[str, object]] = []
    mapping_by_symbol = {str(row["symbol"]): row for row in mapping_rows}
    for symbol, history in histories.items():
        valid = history.dropna(subset=["rs_return_63d", "rs_rank_63d"])
        if valid.empty:
            continue
        row = valid.iloc[-1]
        meta_row = mapping_by_symbol[symbol]
        latest_rows.append({
            **meta_row,
            "date": valid.index[-1].date().isoformat(),
            "rs_line": row["rs_line"],
            "rs_return_21d": row["rs_return_21d"],
            "rs_return_63d": row["rs_return_63d"],
            "rs_momentum_21d": row["rs_momentum_21d"],
            "rs_momentum_63d": row["rs_momentum_63d"],
            "rs_rank_63d": row["rs_rank_63d"],
        })

    latest_df = pd.DataFrame(latest_rows)
    if not latest_df.empty:
        latest_df = latest_df.sort_values("rs_rank_63d", ascending=False).reset_index(drop=True)
    mapping_df = pd.DataFrame(mapping_rows)
    logger.info(f"RS {market}: 完成 {len(histories)} 只，最新有效 {len(latest_df)} 只")
    return latest_df, mapping_df


def update_all_relative_strength(
    markets: tuple[str, ...] = ("US", "CN", "HK"),
) -> dict[str, object]:
    """Calculate and persist all daily RS histories and current beta mappings."""
    benchmarks = load_all_benchmarks()
    missing = sorted(set(BENCHMARKS) - set(benchmarks))
    if missing:
        raise RuntimeError(f"benchmark 数据缺失: {missing}")

    run_dir = RS_OUTPUT_DIR / date.today().isoformat()
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_frames: list[pd.DataFrame] = []
    mapping_frames: list[pd.DataFrame] = []
    market_counts: dict[str, int] = {}

    for market in markets:
        latest, mapping = compute_market_rs(market, benchmarks=benchmarks, save=True)
        latest.to_csv(run_dir / f"{market.lower()}_rs_latest.csv", index=False, encoding="utf-8-sig")
        mapping.to_csv(run_dir / f"{market.lower()}_benchmark_mapping.csv", index=False, encoding="utf-8-sig")
        latest_frames.append(latest)
        mapping_frames.append(mapping)
        market_counts[market] = len(latest)

    all_latest = pd.concat(latest_frames, ignore_index=True) if latest_frames else pd.DataFrame()
    all_mapping = pd.concat(mapping_frames, ignore_index=True) if mapping_frames else pd.DataFrame()
    all_latest.to_csv(run_dir / "rs_latest.csv", index=False, encoding="utf-8-sig")
    all_mapping.to_csv(run_dir / "benchmark_mapping.csv", index=False, encoding="utf-8-sig")

    summary = {
        "date": date.today().isoformat(),
        "markets": market_counts,
        "latest_count": len(all_latest),
        "mapping_count": len(all_mapping),
        "benchmark_last_dates": {
            benchmark_id: df.index.max().date().isoformat()
            for benchmark_id, df in benchmarks.items()
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.success(f"RS 全市场更新完成: {summary}")
    return summary
