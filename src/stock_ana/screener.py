"""
股票筛选模块 - 基于技术指标的基础筛选策略 + Vegas 通道策略

VCP 和三角形策略已拆分至独立模块，此处保留向后兼容的 re-export。
"""

import pandas as pd
from loguru import logger

from stock_ana.data_fetcher import load_all_ndx100_data
from stock_ana.indicators import add_macd, add_vegas_channel

# ──────── 向后兼容 re-export ────────
from stock_ana.strategy_base import check_trend_template  # noqa: F401
from stock_ana.strategy_vcp import (  # noqa: F401
    screen_vcp,
    scan_ndx100_vcp,
)
from stock_ana.strategy_triangle import (  # noqa: F401
    screen_ascending_triangle,
    scan_ndx100_ascending_triangle,
)


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


def scan_ndx100_macd_cross(lookback_days: int = 5) -> list[dict]:
    """
    扫描纳指100中最近 lookback_days 个交易日内发生 MACD 金叉的股票。

    Args:
        lookback_days: 回看交易日数（默认 5 天，约 1 周）

    Returns:
        包含金叉股票信息的列表
    """
    stock_data = load_all_ndx100_data()

    if not stock_data:
        logger.error("本地无数据！请先运行 update_ndx100_data() 下载数据")
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

    logger.info(f"扫描完成：本地共 {len(stock_data)} 只股票，"
                f"有效处理 {processed} 只，{len(hits)} 只发生 MACD 金叉")
    return hits


# ──────── Vegas 长期通道策略 ────────


def screen_vegas_channel_touch(df: pd.DataFrame, lookback_days: int = 5,
                                half_year_days: int = 120) -> dict | None:
    """
    Vegas 长期通道回踩策略（从上方测试）：

    条件：
    1. 股价在最近半年内达到最高点后开始波动下行
    2. 高点到触及通道之间，股价必须大部分时间在 Vegas 通道上方运行
    3. 最近 lookback_days 个交易日内，价格触及 Vegas 通道
    4. 没有有效跌破 EMA169

    Args:
        df: 带有 close, low, high, ema_144, ema_169 列的 DataFrame
        lookback_days: 回看交易日数
        half_year_days: 半年的交易日数

    Returns:
        形态信息 dict 或 None
    """
    required = {"close", "low", "high", "ema_144", "ema_169"}
    if not required.issubset(df.columns) or len(df) < half_year_days:
        return None

    half_year = df.iloc[-half_year_days:]
    peak_idx = half_year["high"].idxmax()
    peak_price = half_year.loc[peak_idx, "high"]

    recent_start = df.index[-lookback_days]
    if peak_idx >= recent_start:
        return None

    curr_close = df["close"].iloc[-1]
    if curr_close >= peak_price * 0.95:
        return None

    peak_iloc = df.index.get_loc(peak_idx)
    touch_start = len(df) - lookback_days

    between_section = df.iloc[peak_iloc:touch_start]
    if len(between_section) < 3:
        return None

    above_count = 0
    for _, row in between_section.iterrows():
        ema_upper = max(row["ema_144"], row["ema_169"])
        if row["close"] > ema_upper:
            above_count += 1

    above_ratio = above_count / len(between_section) if len(between_section) > 0 else 0
    if above_ratio < 0.70:
        return None

    recent = df.iloc[-lookback_days:]
    touched = False
    touch_date = None
    for idx_label, row in recent.iterrows():
        ema_upper = max(row["ema_144"], row["ema_169"])
        if row["low"] <= ema_upper:
            touched = True
            touch_date = idx_label
            break

    if not touched:
        return None

    for i in range(len(df) - lookback_days, len(df)):
        row = df.iloc[i]
        ema_lower = min(row["ema_144"], row["ema_169"])
        close_i = row["close"]

        if close_i < ema_lower:
            if i + 1 < len(df):
                next_row = df.iloc[i + 1]
                next_ema_lower = min(next_row["ema_144"], next_row["ema_169"])
                if next_row["close"] < next_ema_lower:
                    return None
            else:
                return None

    return {
        "peak_price": float(peak_price),
        "peak_date": str(peak_idx),
        "current_price": float(curr_close),
        "above_ratio": round(above_ratio, 2),
        "touch_date": str(touch_date) if touch_date else None,
    }


def scan_ndx100_vegas_touch(lookback_days: int = 5) -> list[dict]:
    """
    扫描纳指100中满足 Vegas 通道回踩条件的股票。

    Args:
        lookback_days: 回看交易日数（默认 5 天）

    Returns:
        [{"ticker": str, "df": DataFrame}, ...]
    """
    stock_data = load_all_ndx100_data()

    if not stock_data:
        logger.error("本地无数据！请先运行 update_ndx100_data() 下载数据")
        return []

    hits: list[dict] = []
    processed = 0

    for ticker, df in stock_data.items():
        try:
            if len(df) < 170:
                logger.debug(f"{ticker}: 数据不足（{len(df)} 行），需至少 170 行计算 EMA169")
                continue

            processed += 1
            df = add_vegas_channel(df.copy())
            result = screen_vegas_channel_touch(df, lookback_days=lookback_days)
            if result is not None:
                logger.success(f"✅ {ticker} 在最近 {lookback_days} 个交易日触及 Vegas 通道回踩（从上方）")
                hits.append({"ticker": ticker, "df": df})
        except Exception as e:
            logger.error(f"{ticker}: 处理失败 - {e}")
            continue

    logger.info(f"Vegas 通道扫描完成：本地共 {len(stock_data)} 只股票，"
                f"有效处理 {processed} 只，{len(hits)} 只满足条件")
    return hits
