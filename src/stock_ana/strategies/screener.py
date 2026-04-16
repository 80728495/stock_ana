"""
股票筛选模块 - 基础技术指标筛选 + 统一策略扫描门面。
"""

import pandas as pd
from loguru import logger

from stock_ana.data.indicators import add_macd
from stock_ana.data.market_data import load_market_data


# ──────── 基础策略 ────────


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
    """MACD 看涨筛选：MACD 柱状图由负转正（仅检测最后一天）"""
    if "macd_hist" not in df.columns or len(df) < 2:
        return False
    return df["macd_hist"].iloc[-2] < 0 and df["macd_hist"].iloc[-1] > 0


def screen_macd_cross_in_period(df: pd.DataFrame, lookback_days: int = 5) -> bool:
    """
    检测最近 N 个交易日内是否发生了 MACD 金叉
    （MACD 线上穿信号线，即 macd_hist 由负转正）

    过滤噪声的条件：
    1. 正面阈值：转正后 macd_hist > 收盘价 × 0.1%（避免微弱穿越）
    2. 负面深度：交叉前的负值期内，最低 hist < -收盘价 × 0.1%
    3. 金叉后若又发生死叉，则该金叉无效

    Args:
        df: 带有 macd_hist 和 close 列的 DataFrame
        lookback_days: 回看交易日数（1周 ≈ 5个交易日）

    Returns:
        True 如果在回看期内最后一次有意义的交叉是 MACD 金叉
    """
    if "macd_hist" not in df.columns or "close" not in df.columns or len(df) < 2:
        return False

    hist_all = df["macd_hist"]
    close_all = df["close"]

    n = lookback_days + 1
    start_idx = len(df) - n

    last_cross_is_golden = None

    for i in range(max(start_idx + 1, 1), len(df)):
        prev_hist = hist_all.iloc[i - 1]
        curr_hist = hist_all.iloc[i]
        curr_close = close_all.iloc[i]
        threshold = curr_close * 0.001

        # 金叉：负 → 正
        if prev_hist < 0 and curr_hist > 0:
            if curr_hist <= threshold:
                continue

            min_neg_hist = prev_hist
            for j in range(i - 2, -1, -1):
                h = hist_all.iloc[j]
                if h >= 0:
                    break
                if h < min_neg_hist:
                    min_neg_hist = h

            if abs(min_neg_hist) < threshold:
                continue

            last_cross_is_golden = True

        # 死叉：正 → 负
        elif prev_hist > 0 and curr_hist < 0:
            if abs(curr_hist) > threshold:
                last_cross_is_golden = False

    return last_cross_is_golden is True


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


def scan_macd_cross(
    lookback_days: int = 5,
    universe: str = "ndx100",
    stock_data: dict[str, pd.DataFrame] | None = None,
) -> list[dict]:
    """
    扫描指定股票池中最近 lookback_days 个交易日内发生 MACD 金叉的股票。

    Args:
        lookback_days: 回看交易日数（默认 5 天，约 1 周）
        universe: 股票池名称（当 stock_data 为 None 时生效）
        stock_data: 可选，外部注入股票数据 {ticker: df}

    Returns:
        包含金叉股票信息的列表
    """
    if stock_data is None:
        stock_data = load_market_data(universe)

    if not stock_data:
        logger.error(f"本地无数据！请先更新股票池数据: {universe}")
        return []

    hits: list[dict] = []
    processed = 0

    for ticker, df in stock_data.items():
        try:
            if len(df) < 35:
                logger.debug(f"{ticker}: 数据不足（{len(df)} 行），跳过")
                continue

            processed += 1
            df = add_macd(df.copy())
            if screen_macd_cross_in_period(df, lookback_days=lookback_days):
                logger.success(f"✅ {ticker} 在最近 {lookback_days} 个交易日内发生 MACD 金叉")
                hits.append({"ticker": ticker, "df": df})
        except Exception as e:
            logger.error(f"{ticker}: 处理失败 - {e}")
            continue

    logger.info(f"MACD 金叉扫描完成：股票池 {universe} 共 {len(stock_data)} 只，"
                f"有效处理 {processed} 只，{len(hits)} 只发生 MACD 金叉")
    return hits


# ──────── Vegas 长期通道策略 ────────


