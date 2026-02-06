"""screener 模块基础测试"""

import pandas as pd

from stock_ana.screener import screen_golden_cross, screen_rsi_oversold


def _make_df(data: dict) -> pd.DataFrame:
    return pd.DataFrame(data)


def test_golden_cross_true():
    df = _make_df({
        "close": [10, 11, 12, 13],
        "sma_5":  [9, 10, 11, 13],
        "sma_20": [10, 10.5, 11.5, 12],
    })
    assert screen_golden_cross(df) is True


def test_golden_cross_false():
    df = _make_df({
        "close": [10, 11, 12, 13],
        "sma_5":  [12, 13, 14, 15],
        "sma_20": [10, 10.5, 11, 11.5],
    })
    assert screen_golden_cross(df) is False


def test_rsi_oversold():
    df = _make_df({"close": [10], "rsi": [25.0]})
    assert screen_rsi_oversold(df) is True

    df2 = _make_df({"close": [10], "rsi": [55.0]})
    assert screen_rsi_oversold(df2) is False
