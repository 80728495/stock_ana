"""
富途 OpenD 数据获取模块

通过本地 OpenD 网关获取港股/美股/A股历史 K 线。
OpenD 必须在本机已启动（默认 127.0.0.1:11111）。

代码格式约定（内部 → 富途）：
  美股: "AAPL"   → "US.AAPL"
  港股: "00700"  → "HK.00700"（5位，补前导零）
  沪股: "600519" → "SH.600519"
  深股: "000001" → "SZ.000001"
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Generator

import pandas as pd
from loguru import logger

# ─────────────────────── OpenD 连接配置 ───────────────────────

OPEND_HOST: str = os.environ.get("FUTU_OPEND_HOST", "127.0.0.1")
OPEND_PORT: int = int(os.environ.get("FUTU_OPEND_PORT", "11111"))


# ─────────────────────── 连接管理 ───────────────────────

@contextmanager
def quote_context(
    host: str = OPEND_HOST,
    port: int = OPEND_PORT,
) -> Generator:
    """
    富途行情上下文管理器，自动关闭连接。

    Usage:
        with quote_context() as ctx:
            ret, data, _ = ctx.request_history_kline('US.AAPL', ...)
    """
    from futu import OpenQuoteContext  # 延迟导入，避免未安装时崩溃

    ctx = OpenQuoteContext(host=host, port=port)
    try:
        yield ctx
    finally:
        ctx.close()


# ─────────────────────── 代码格式转换 ───────────────────────

def to_futu_code(symbol: str, market: str | None = None) -> str:
    """
    将本地代码格式转换为富途代码格式。

    Args:
        symbol: 股票代码，如 "AAPL"、"00700"、"600519"
        market: 可选市场标识 "US"/"HK"/"SH"/"SZ"，不传则自动推断

    Returns:
        富途格式代码，如 "US.AAPL"、"HK.00700"
    """
    if "." in symbol:
        return symbol  # 已是富途格式，直接返回

    if market:
        prefix = market.upper()
        if prefix == "HK":
            return f"HK.{symbol.zfill(5)}"
        return f"{prefix}.{symbol}"

    # 自动推断：纯数字 → 港股或 A 股
    if symbol.isdigit():
        if len(symbol) <= 5:
            return f"HK.{symbol.zfill(5)}"
        if symbol.startswith(("6", "9")):
            return f"SH.{symbol}"
        return f"SZ.{symbol}"

    # 纯字母 → 美股
    return f"US.{symbol.upper()}"


# ─────────────────────── 内部工具 ───────────────────────

def _request_all_kline(
    ctx,
    futu_code: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    处理分页，完整拉取指定股票的历史日 K 线（前复权）。

    Returns:
        富途原始 DataFrame，含 time_key/open/high/low/close/volume 等字段
    """
    from futu import RET_OK, AuType, KLType

    all_dfs: list[pd.DataFrame] = []
    page_req_key = None
    page = 0

    while True:
        ret, data, page_req_key = ctx.request_history_kline(
            futu_code,
            start=start,
            end=end,
            ktype=KLType.K_DAY,
            autype=AuType.QFQ,
            max_count=1000,
            page_req_key=page_req_key,
        )
        if ret != RET_OK:
            raise RuntimeError(
                f"request_history_kline failed [{futu_code}]: {data}"
            )

        all_dfs.append(data)
        page += 1
        logger.debug(f"{futu_code}: 第 {page} 页，{len(data)} 条")

        if page_req_key is None:
            break

    return pd.concat(all_dfs, ignore_index=True)


def _normalize_kline(raw: pd.DataFrame) -> pd.DataFrame:
    """
    将富途 K 线 DataFrame 转换为项目标准格式。

    Returns:
        DataFrame，index=date（datetime），columns=[open, high, low, close, volume]
    """
    df = raw[["time_key", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["time_key"]).dt.normalize()
    df = df.set_index("date").drop(columns=["time_key"])
    df = df.sort_index()
    return df


def fetch_hk_stock_with_ctx(
    ctx,
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    使用已有的 quote_context 获取港股日 K 线（供批量操作复用连接）。

    Args:
        ctx:        OpenQuoteContext 实例（调用方负责生命周期）
        symbol:     股票代码，如 "00700"（不含市场前缀）
        start_date: "YYYY-MM-DD"
        end_date:   "YYYY-MM-DD"

    Returns:
        DataFrame，index=date，columns=[open, high, low, close, volume]

    Raises:
        RuntimeError: 请求失败
    """
    futu_code = to_futu_code(symbol, market="HK")
    raw = _request_all_kline(ctx, futu_code, start_date, end_date)
    if raw.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return _normalize_kline(raw)

    df = df.sort_index()
    return df


# ─────────────────────── 公共接口 ───────────────────────

def fetch_us_stock_futu(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    通过富途 OpenD 获取美股历史日 K 线（前复权）。

    Args:
        symbol:     股票代码，如 "AAPL"（不含市场前缀）
        start_date: 起始日期 "YYYY-MM-DD"，默认 1 年前
        end_date:   结束日期 "YYYY-MM-DD"，默认今日

    Returns:
        DataFrame，index=date，columns=[open, high, low, close, volume]

    Raises:
        RuntimeError: OpenD 未启动或请求失败
    """
    today = datetime.now()
    if end_date is None:
        end_date = today.strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (today - timedelta(days=365)).strftime("%Y-%m-%d")

    futu_code = to_futu_code(symbol, market="US")
    logger.info(f"Futu 获取美股 {futu_code} [{start_date} → {end_date}]")

    with quote_context() as ctx:
        raw = _request_all_kline(ctx, futu_code, start_date, end_date)

    if raw.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return _normalize_kline(raw)


def fetch_hk_stock_futu(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    通过富途 OpenD 获取港股历史日 K 线（前复权）。

    Args:
        symbol:     股票代码，如 "00700" 或 "700"
        start_date: 起始日期 "YYYY-MM-DD"，默认 3 年前
        end_date:   结束日期 "YYYY-MM-DD"，默认今日

    Returns:
        DataFrame，index=date，columns=[open, high, low, close, volume]

    Raises:
        RuntimeError: OpenD 未启动或请求失败
    """
    today = datetime.now()
    if end_date is None:
        end_date = today.strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (today - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    futu_code = to_futu_code(symbol, market="HK")
    logger.info(f"Futu 获取港股 {futu_code} [{start_date} → {end_date}]")

    with quote_context() as ctx:
        raw = _request_all_kline(ctx, futu_code, start_date, end_date)

    if raw.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return _normalize_kline(raw)


def fetch_cn_stock_futu(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    通过富途 OpenD 获取 A 股历史日 K 线（前复权）。

    Args:
        symbol:     股票代码，如 "600519"（6 位数字）
        start_date: 起始日期 "YYYY-MM-DD"，默认 3 年前
        end_date:   结束日期 "YYYY-MM-DD"，默认今日

    Returns:
        DataFrame，index=date，columns=[open, high, low, close, volume]

    Raises:
        RuntimeError: OpenD 未启动或请求失败
    """
    today = datetime.now()
    if end_date is None:
        end_date = today.strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (today - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    futu_code = to_futu_code(symbol)  # 自动推断沪/深
    logger.info(f"Futu 获取 A 股 {futu_code} [{start_date} → {end_date}]")

    with quote_context() as ctx:
        raw = _request_all_kline(ctx, futu_code, start_date, end_date)

    if raw.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return _normalize_kline(raw)
