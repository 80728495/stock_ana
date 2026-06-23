import numpy as np
import pandas as pd

from stock_ana.strategies.impl.gap_fail_reversal import (
    detect_gap_fail_reversal,
    scan_history,
)


def _make_gap_fail_df(*, fill_gap: bool = True, index_name: str | None = None) -> pd.DataFrame:
    n = 90
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.linspace(10.0, 20.0, n)
    open_ = close - 0.1
    high = close + 0.2
    low = close - 0.2
    volume = np.ones(n) * 1000

    prev_close = float(close[-2])
    prev_high = float(high[-2])
    open_[-1] = prev_close * 1.075
    if fill_gap:
        high[-1] = open_[-1] + 0.4
        close[-1] = prev_close - 0.4
        low[-1] = close[-1] - 0.25
    else:
        high[-1] = open_[-1] + 0.1
        close[-1] = open_[-1] - 0.45
        low[-1] = close[-1] - 0.2
    volume[-1] = 2600

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
    df.index.name = index_name
    return df


def test_detect_gap_fail_reversal_with_same_day_failed_gap():
    df = _make_gap_fail_df(fill_gap=True)

    result = detect_gap_fail_reversal(df)

    assert result["triggered"] is True
    assert result["confirm_mode"] == "same_day_gap_fail"
    assert result["gap_fill_ratio"] >= 0.8
    assert result["close_below_prev_close"] is True
    assert result["body_vs_avg20"] >= 1.2


def test_detect_gap_fail_reversal_rejects_unfilled_gap():
    df = _make_gap_fail_df(fill_gap=False)

    result = detect_gap_fail_reversal(df)

    assert result["triggered"] is False
    assert result["reason"] == "gap_not_filled_enough"


def test_scan_history_accepts_unnamed_datetime_index():
    df = _make_gap_fail_df(fill_gap=True, index_name=None)

    hits = scan_history(df)

    assert not hits.empty
    assert hits.iloc[-1]["signal_date"] == str(df.index[-1].date())
    assert hits.iloc[-1]["confirm_date"] == str(df.index[-1].date())
    assert "gap_fill_ratio" in hits.columns
