from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ana.data.benchmark_store import BENCHMARKS
from stock_ana.data.fetcher_futu import to_futu_code
from stock_ana.data.relative_strength_store import (
    BenchmarkChoice,
    StockMeta,
    benchmark_choice,
    build_causal_benchmark_history,
    compute_stock_rs_history,
)


def _prices(returns: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"close": close}, index=dates)


def test_benchmark_registry_preserves_full_futu_index_codes() -> None:
    assert BENCHMARKS["CN_CHINEXT"].futu_code == "SZ.399006"
    assert BENCHMARKS["CN_STAR_COMPOSITE"].futu_code == "SH.000680"
    assert BENCHMARKS["CN_STAR50"].futu_code == "SH.000688"
    assert BENCHMARKS["HK_HSTECH"].futu_code == "HK.800700"


def test_cn_etf_exchange_inference() -> None:
    assert to_futu_code("515880") == "SH.515880"
    assert to_futu_code("588000") == "SH.588000"
    assert to_futu_code("300750") == "SZ.300750"


def test_benchmark_choice_uses_market_board_and_hk_industry() -> None:
    assert benchmark_choice(StockMeta("US", "NVDA")).prior == "US_QQQ"
    assert benchmark_choice(StockMeta("CN", "300750")).prior == "CN_CHINEXT"
    assert benchmark_choice(StockMeta("CN", "688981")).prior == "CN_STAR_COMPOSITE"
    assert benchmark_choice(StockMeta("HK", "00700", industry="互动媒体及服务")).prior == "HK_HSTECH"
    assert benchmark_choice(StockMeta("HK", "00981", industry="半导体")).prior == "CN_STAR_COMPOSITE"


def test_monthly_benchmark_mapping_does_not_use_future_returns() -> None:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=320)
    a_returns = rng.normal(0.0005, 0.012, len(dates))
    b_returns = rng.normal(0.0002, 0.012, len(dates))
    stock_returns = a_returns + rng.normal(0, 0.002, len(dates))
    stock = _prices(stock_returns, dates)
    benchmarks = {
        "A": _prices(a_returns, dates),
        "B": _prices(b_returns, dates),
    }
    choice = BenchmarkChoice("A", ("A", "B"))

    original = build_causal_benchmark_history(stock, choice, benchmarks)
    cutoff = dates[230]
    changed_stock = stock.copy()
    changed_stock.loc[changed_stock.index > cutoff, "close"] *= np.linspace(
        1.0,
        8.0,
        (changed_stock.index > cutoff).sum(),
    )
    changed = build_causal_benchmark_history(changed_stock, choice, benchmarks)

    pd.testing.assert_frame_equal(original.loc[:cutoff], changed.loc[:cutoff])


def test_rs_history_is_excess_return_against_selected_benchmark() -> None:
    dates = pd.bdate_range("2024-01-02", periods=100)
    benchmark_returns = np.full(len(dates), 0.001)
    stock_returns = np.full(len(dates), 0.002)
    benchmark = _prices(benchmark_returns, dates)
    stock = _prices(stock_returns, dates)
    mapping = pd.DataFrame(
        {"benchmark_id": "A", "beta": 2.0, "r2": 1.0},
        index=dates,
    )

    result = compute_stock_rs_history(stock, mapping, {"A": benchmark})

    assert result["rs_return_63d"].iloc[-1] > 0
    assert result["rs_line"].iloc[-1] > result["rs_line"].iloc[0]
    assert (result["benchmark_id"] == "A").all()
