import numpy as np
import pandas as pd

from stock_ana.strategies.impl.top_reversal import (
    detect_high_shadow_reversal,
    scan_history,
)


def _make_top_reversal_df(*, index_name: str | None = None, rising: bool = True) -> pd.DataFrame:
    n = 90
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.linspace(10.0, 20.0, n) if rising else np.linspace(10.0, 10.4, n)
    open_ = close - 0.1
    high = close + 0.2
    low = close - 0.2
    volume = np.ones(n) * 1000

    open_[-2] = close[-2] - 0.1
    close[-2] = close[-2]
    high[-2] = close[-2] + 2.0
    low[-2] = close[-2] - 0.2
    volume[-2] = 2200

    open_[-1] = close[-2] - 0.1
    close[-1] = close[-2] - 0.5
    high[-1] = close[-2]
    low[-1] = close[-2] - 0.7
    volume[-1] = 1800

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
    df.index.name = index_name
    return df


def test_detect_high_shadow_reversal_with_top_context():
    df = _make_top_reversal_df(rising=True)

    result = detect_high_shadow_reversal(df)

    assert result["triggered"] is True
    assert result["confirm_mode"] == "engulf_open"
    assert result["d2_break_d1_low"] is True
    assert result["prior_rise_pct"] >= 10.0
    assert result["shadow_atr"] >= 0.5


def test_detect_high_shadow_reversal_rejects_flat_context():
    df = _make_top_reversal_df(rising=False)

    result = detect_high_shadow_reversal(df)

    assert result["triggered"] is False
    assert "顶部背景阈值" in result["reason"]


def test_scan_history_accepts_unnamed_datetime_index():
    df = _make_top_reversal_df(index_name=None, rising=True)

    hits = scan_history(df)

    assert not hits.empty
    assert hits.iloc[-1]["signal_date"] == str(df.index[-2].date())
    assert "shadow_atr" in hits.columns
    assert "prior_rise_pct" in hits.columns
