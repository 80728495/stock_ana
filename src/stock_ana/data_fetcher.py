"""
数据获取模块 - 统一接口获取 A股/美股 数据
"""

from pathlib import Path

import pandas as pd

# 数据缓存目录（跨平台兼容）
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_cn_stock(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取 A 股历史行情数据

    Args:
        symbol: 股票代码，如 "600519"
        start_date: 起始日期，如 "20240101"
        end_date: 结束日期，如 "20241231"

    Returns:
        DataFrame，包含 date, open, high, low, close, volume 列
    """
    import akshare as ak

    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")

    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df[["open", "high", "low", "close", "volume"]]


def fetch_us_stock(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取美股历史行情数据

    Args:
        symbol: 股票代码，如 "AAPL"
        start_date: 起始日期，如 "2024-01-01"
        end_date: 结束日期，如 "2024-12-31"

    Returns:
        DataFrame，包含 date, open, high, low, close, volume 列
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]]
