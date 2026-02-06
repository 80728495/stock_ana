"""
股票筛选模块 - 基于技术指标的筛选策略
"""

import pandas as pd


def screen_golden_cross(df: pd.DataFrame, short_ma: str = "sma_5", long_ma: str = "sma_20") -> bool:
    """
    金叉筛选：短期均线上穿长期均线

    Args:
        df: 带有均线指标的 DataFrame
        short_ma: 短期均线列名
        long_ma: 长期均线列名

    Returns:
        True 如果最近发生金叉
    """
    if len(df) < 2:
        return False

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    return prev[short_ma] <= prev[long_ma] and curr[short_ma] > curr[long_ma]


def screen_rsi_oversold(df: pd.DataFrame, threshold: float = 30.0) -> bool:
    """RSI 超卖筛选：RSI 低于阈值"""
    if "rsi" not in df.columns or df["rsi"].isna().all():
        return False
    return df["rsi"].iloc[-1] < threshold


def screen_macd_bullish(df: pd.DataFrame) -> bool:
    """MACD 看涨筛选：MACD 柱状图由负转正"""
    if "macd_hist" not in df.columns or len(df) < 2:
        return False
    return df["macd_hist"].iloc[-2] < 0 and df["macd_hist"].iloc[-1] > 0


def screen_bollinger_squeeze(df: pd.DataFrame, threshold: float = 0.05) -> bool:
    """布林带收窄筛选：带宽占比低于阈值"""
    if "bb_upper" not in df.columns:
        return False
    curr = df.iloc[-1]
    bandwidth = (curr["bb_upper"] - curr["bb_lower"]) / curr["bb_mid"]
    return bandwidth < threshold


def run_screen(df: pd.DataFrame, strategies: list[str] | None = None) -> dict[str, bool]:
    """
    运行指定的筛选策略

    Args:
        df: 带有技术指标的 DataFrame
        strategies: 要运行的策略列表，默认全部运行

    Returns:
        策略名称 -> 是否通过筛选
    """
    all_strategies = {
        "golden_cross": screen_golden_cross,
        "rsi_oversold": screen_rsi_oversold,
        "macd_bullish": screen_macd_bullish,
        "bollinger_squeeze": screen_bollinger_squeeze,
    }

    if strategies is None:
        strategies = list(all_strategies.keys())

    results = {}
    for name in strategies:
        if name in all_strategies:
            results[name] = all_strategies[name](df)

    return results
