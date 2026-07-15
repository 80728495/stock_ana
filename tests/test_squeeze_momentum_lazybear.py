from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_ana.data.indicators import add_squeeze_momentum_lazybear
from stock_ana.data.indicators_store import compute_indicators


def _ohlc(rows: int = 100) -> pd.DataFrame:
    x = np.arange(rows, dtype=float)
    close = 100.0 + 0.15 * x + 2.0 * np.sin(x / 6.0)
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 1.0 + 0.1 * np.cos(x),
            "low": close - 1.0 - 0.1 * np.sin(x),
            "close": close,
            "volume": 1_000_000 + x * 100,
        },
        index=pd.bdate_range("2025-01-02", periods=rows),
    )


def _manual_linreg_endpoint(values: pd.Series, window: int) -> float:
    y = values.iloc[-window:].to_numpy(dtype=float)
    x = np.arange(window, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept + slope * (window - 1))


def test_lazybear_matches_reference_formula_on_last_row() -> None:
    source = _ohlc()
    result = add_squeeze_momentum_lazybear(source.copy())

    close = source["close"]
    high = source["high"]
    low = source["low"]
    basis = close.rolling(20).mean()
    highest = high.rolling(20).max()
    lowest = low.rolling(20).min()
    detrended = close - (0.25 * (highest + lowest) + 0.5 * basis)
    expected = _manual_linreg_endpoint(detrended.dropna(), 20)

    assert result["sqzmom_value"].iloc[-1] == pytest.approx(expected)
    assert result["sqzmom_bar_state"].dropna().isin([-2, -1, 1, 2]).all()
    assert result["sqzmom_squeeze_state"].dropna().isin([-1, 0, 1]).all()


def test_lazybear_is_causal_when_future_prices_change() -> None:
    source = _ohlc()
    cutoff = source.index[69]
    original = add_squeeze_momentum_lazybear(source.copy())
    changed = source.copy()
    changed.loc[changed.index > cutoff, ["open", "high", "low", "close"]] *= 3.0
    changed_result = add_squeeze_momentum_lazybear(changed)

    columns = [
        "sqzmom_value",
        "sqzmom_squeeze_on",
        "sqzmom_squeeze_off",
        "sqzmom_no_squeeze",
        "sqzmom_squeeze_state",
        "sqzmom_bar_state",
    ]
    pd.testing.assert_frame_equal(original.loc[:cutoff, columns], changed_result.loc[:cutoff, columns])


def test_lazybear_requires_ohlc_columns() -> None:
    with pytest.raises(ValueError, match="缺少必要列"):
        add_squeeze_momentum_lazybear(pd.DataFrame({"close": [1.0, 2.0]}))


def test_daily_indicator_store_includes_lazybear_columns() -> None:
    result = compute_indicators(_ohlc())

    assert {
        "sqzmom_value",
        "sqzmom_squeeze_on",
        "sqzmom_squeeze_off",
        "sqzmom_no_squeeze",
        "sqzmom_squeeze_state",
        "sqzmom_bar_state",
    } <= set(result.columns)
