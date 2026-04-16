"""
大A股票数据获取模块

使用 akshare（东方财富源 stock_zh_a_hist）获取前复权日线数据，
支持本地 parquet 持久化存储与增量更新。

架构与 fetcher_hk.py 保持一致：
  - A 股缓存目录：data/cache/cn/
  - 每只个股一个 parquet 文件，文件名 = 6位代码（如 600519.parquet）
  - 增量更新：仅拉取上次缓存日期之后的新数据并追加
  - 复用 fetcher.py 中已有的 fetch_cn_stock 和 _bypass_proxy 工具
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from stock_ana.config import CACHE_DIR, DATA_DIR
from stock_ana.data.fetcher_hk import _bypass_proxy  # 复用代理绕过工具

# ── 大A缓存目录 ──────────────────────────────────────────────────────────────
CN_DIR = CACHE_DIR / "cn"
CN_DIR.mkdir(parents=True, exist_ok=True)

# ── 历史回看起点（全量拉取时） ────────────────────────────────────────────────
_DEFAULT_START = "20200101"


# ─────────────────────── 单只股票获取 ───────────────────────────────────────


def fetch_cn_stock_hist(
    symbol: str,
    start_date: str = _DEFAULT_START,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    获取单只 A 股历史日线行情（前复权）。

    方案 A（163 网易财经源）：ak.stock_zh_a_daily — 返回全量数据，
        本地按 start_date 截断。代码需要交易所前缀：sh/sz/bj。
    方案 B（东方财富源）：ak.stock_zh_a_hist — 支持日期范围参数，
        作为备选（部分网络环境下不可达）。

    Args:
        symbol:     6 位股票代码，如 "600519"（无需前缀）
        start_date: 起始日期，格式 "YYYYMMDD"，默认 20200101
        end_date:   结束日期，格式 "YYYYMMDD"，默认今天

    Returns:
        DataFrame，index=date，columns=[open, high, low, close, volume]
    """
    import akshare as ak

    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    sd = pd.to_datetime(start_date, format="%Y%m%d")
    ed = pd.to_datetime(end_date, format="%Y%m%d")

    # ── 方案 A：163 网易财经源（stock_zh_a_daily） ─────────────────────────
    # 代码需要市场前缀：sh=上交所(600/601/603/605/688xx)，sz=深交所(000/001/002/003/300)
    # bj=北交所(8xxxxxx / 4xxxxxx) — 暂不处理
    def _exchange_prefix(code: str) -> str:
        if code.startswith(("6",)):
            return f"sh{code}"
        return f"sz{code}"

    ak_symbol = _exchange_prefix(symbol)

    try:
        with _bypass_proxy():
            df = ak.stock_zh_a_daily(symbol=ak_symbol, adjust="qfq")

        df = df.rename(columns={
            "date": "date", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume",
        })
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[cols]
        df = df[(df.index >= sd) & (df.index <= ed)]
        return df

    except Exception as daily_err:
        logger.debug(f"CN {symbol}: 163源失败 ({daily_err})，尝试东方财富源 ...")

    # ── 方案 B：东方财富源（stock_zh_a_hist） ─────────────────────────────
    with _bypass_proxy():
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )

    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "turnover",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    cols = [c for c in ["open", "high", "low", "close", "volume", "turnover"] if c in df.columns]
    return df[cols]


# ─────────────────────── 本地存储 ───────────────────────────────────────────


def _cn_path(code: str) -> Path:
    """单只 A 股的本地 parquet 文件路径（code 为 6 位数字字符串）。"""
    return CN_DIR / f"{code}.parquet"


def load_cn_local(code: str) -> pd.DataFrame | None:
    """从本地加载已存储的 A 股数据，如无则返回 None。"""
    path = _cn_path(code)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def save_cn_local(code: str, df: pd.DataFrame) -> None:
    """将 A 股数据保存到本地 parquet。"""
    path = _cn_path(code)
    df.to_parquet(path)
    logger.debug(f"CN {code}: 已保存 {len(df)} 行 → {path}")


def load_all_cn_data() -> dict[str, pd.DataFrame]:
    """读取 data/cache/cn/ 下所有已缓存的 A 股数据。

    Returns:
        {code: DataFrame}，code 为 6 位字符串，如 "600519"
    """
    result: dict[str, pd.DataFrame] = {}
    for p in sorted(CN_DIR.glob("*.parquet")):
        code = p.stem
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        result[code] = df
    logger.debug(f"已从本地加载 {len(result)} 只 A 股数据")
    return result


# ─────────────────────── 增量更新 ───────────────────────────────────────────


def update_cn_data(
    codes: list[str] | None = None,
    max_stale_days: int = 0,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    增量更新 A 股本地缓存数据。

    Args:
        codes:          要更新的股票代码列表；None 则读 shawn_list 中的 cn 部分。
        max_stale_days: 允许的最大陈旧天数，超过才触发更新（0 = 每次强制更新）。
        force:          True 时忽略本地缓存，重新全量拉取。

    Returns:
        {code: DataFrame}，包含本次成功更新的标的。
    """
    if codes is None:
        from stock_ana.data.list_manager import parse_shawn_list
        parsed = parse_shawn_list()
        codes = [e["symbol"] for e in parsed.get("cn", [])]

    if not codes:
        logger.info("CN: 无需更新（列表为空）")
        return {}

    today = pd.Timestamp.now().normalize()
    updated: dict[str, pd.DataFrame] = {}

    for i, code in enumerate(codes, 1):
        local = load_cn_local(code)

        if not force and local is not None and not local.empty:
            last_date = pd.Timestamp(local.index.max()).normalize()
            stale_days = (today - last_date).days
            if stale_days <= max_stale_days:
                logger.debug(f"CN {code}: 数据已是最新（最新 {last_date.date()}，跳过）")
                continue
            start_date = (last_date + timedelta(days=1)).strftime("%Y%m%d")
        else:
            start_date = _DEFAULT_START

        try:
            df_new = fetch_cn_stock_hist(code, start_date=start_date)
            if df_new.empty:
                logger.debug(f"CN {code}: 无新数据")
                if local is not None:
                    updated[code] = local
                continue

            if local is not None and not local.empty and not force:
                df_merged = pd.concat([local, df_new])
                df_merged = df_merged[~df_merged.index.duplicated(keep="last")]
                df_merged = df_merged.sort_index()
            else:
                df_merged = df_new.sort_index()

            save_cn_local(code, df_merged)
            updated[code] = df_merged
            logger.info(
                f"[{i}/{len(codes)}] CN {code}: "
                f"更新至 {df_merged.index.max().date()}（共 {len(df_merged)} 行）"
            )

        except Exception as e:
            logger.warning(f"CN {code}: 获取失败 — {e}")

        # 轻微延时，避免对东方财富 API 施压
        if i % 10 == 0:
            time.sleep(0.5)

    logger.info(f"CN 更新完成：{len(updated)}/{len(codes)} 只成功")
    return updated
