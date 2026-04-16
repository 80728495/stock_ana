"""
数据获取模块 - 统一接口获取 A股/美股 数据，支持本地持久化存储与增量更新
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from stock_ana.config import CACHE_DIR
from stock_ana.data.list_manager import load_ndx100_list

# 纳指100本地存储目录
NDX100_DIR = CACHE_DIR / "ndx100"
NDX100_DIR.mkdir(parents=True, exist_ok=True)

# 美股全市场本地存储目录
US_DIR = CACHE_DIR / "us"
US_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────── 基础获取函数 ───────────────────────────

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


def fetch_us_stock_direct(symbol: str, start_date: str | None = None,
                          end_date: str | None = None, period: str = "3mo") -> pd.DataFrame:
    """
    通过 Yahoo Finance v8 API 直接获取美股数据（绕过 yfinance 限流）

    Args:
        symbol: 股票代码
        start_date: 起始日期 "YYYY-MM-DD"（与 period 二选一）
        end_date: 结束日期 "YYYY-MM-DD"
        period: 数据周期，如 "3mo", "6mo", "1y"（start_date 为空时使用）

    Returns:
        DataFrame，包含 open, high, low, close, volume 列
    """
    import requests as req

    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    if start_date:
        period1 = int(pd.Timestamp(start_date).timestamp())
        period2 = int(pd.Timestamp(end_date or datetime.now().strftime("%Y-%m-%d")).timestamp()) + 86400
        params = {"period1": period1, "period2": period2, "interval": "1d"}
    else:
        params = {"range": period, "interval": "1d"}

    # 重试逻辑：遇到 429 限流时等待后重试
    max_retries = 3
    for attempt in range(max_retries):
        r = req.get(url, params=params, headers=headers, timeout=15)
        if r.status_code == 429:
            wait = 5 * (attempt + 1)
            logger.warning(f"{symbol}: 被限流(429)，等待 {wait}s 后重试 ({attempt+1}/{max_retries})")
            time.sleep(wait)
            continue
        r.raise_for_status()
        break
    else:
        raise Exception(f"{symbol}: 多次重试后仍被限流")

    data = r.json()["chart"]["result"][0]
    ts = data["timestamp"]
    q = data["indicators"]["quote"][0]

    df = pd.DataFrame({
        "open": q["open"], "high": q["high"], "low": q["low"],
        "close": q["close"], "volume": q["volume"],
    }, index=pd.to_datetime(ts, unit="s"))
    df.index = df.index.normalize()  # 去掉时分秒
    df.index.name = "date"
    df = df.dropna()
    return df


# ─────────────────────── 纳指100成分股列表 ───────────────────────

def fetch_ndx100_tickers() -> list[str]:
    """
    获取纳斯达克100指数成分股代码列表。

    优先使用统一列表系统（data/lists/ndx100_list.md），
    避免在 fetcher 中维护单独的数据源抓取逻辑。

    Returns:
        成分股代码列表，如 ["AAPL", "MSFT", ...]
    """
    tickers = load_ndx100_list()
    logger.info(f"从列表系统加载到 {len(tickers)} 只纳指100成分股")
    return tickers


# ─────────────────── 本地存储 & 增量更新 ───────────────────

def _ticker_path(ticker: str) -> Path:
    """单只股票的本地 parquet 文件路径"""
    return NDX100_DIR / f"{ticker}.parquet"


def _us_ticker_path(ticker: str) -> Path:
    """单只美股（US universe）的本地 parquet 文件路径"""
    return US_DIR / f"{ticker}.parquet"


def load_local_data(ticker: str) -> pd.DataFrame | None:
    """从本地加载已存储的行情数据，如无则返回 None"""
    path = _ticker_path(ticker)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def save_local_data(ticker: str, df: pd.DataFrame) -> None:
    """将行情数据保存到本地 parquet"""
    path = _ticker_path(ticker)
    df.to_parquet(path)
    logger.debug(f"{ticker}: 已保存 {len(df)} 行 → {path}")


def load_us_local_data(ticker: str) -> pd.DataFrame | None:
    """从本地加载 US universe 已存储行情，如无则返回 None。"""
    path = _us_ticker_path(ticker)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def save_us_local_data(ticker: str, df: pd.DataFrame) -> None:
    """将 US universe 行情数据保存到本地 parquet。"""
    path = _us_ticker_path(ticker)
    df.to_parquet(path)
    logger.debug(f"US {ticker}: 已保存 {len(df)} 行 → {path}")


def _fetch_us_stock_akshare(symbol: str, start_date: str | None = None) -> pd.DataFrame:
    """
    使用 akshare（新浪财经源）获取美股历史数据，无限流风险

    Args:
        symbol: 股票代码，如 "AAPL"
        start_date: 起始日期 "YYYY-MM-DD"，为 None 则取最近 1 年

    Returns:
        DataFrame，包含 open, high, low, close, volume 列
    """
    import akshare as ak

    df = ak.stock_us_daily(symbol=symbol, adjust="qfq")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df[["open", "high", "low", "close", "volume"]]

    if start_date:
        df = df[df.index >= start_date]

    return df


def _batch_download_akshare(tickers: list[str],
                            start_date: str | None = None) -> dict[str, pd.DataFrame]:
    """
    使用 akshare 逐只下载美股数据（新浪源，无限流）

    Args:
        tickers: 股票代码列表
        start_date: 起始日期 "YYYY-MM-DD"

    Returns:
        {ticker: DataFrame} 字典
    """
    result: dict[str, pd.DataFrame] = {}
    total = len(tickers)
    logger.info(f"akshare 开始下载 {total} 只股票 ...")

    for i, ticker in enumerate(tickers, 1):
        try:
            df = _fetch_us_stock_akshare(ticker, start_date=start_date)
            if not df.empty:
                result[ticker] = df
                if i % 10 == 0 or i == total:
                    logger.info(f"[{i}/{total}] 已下载 {len(result)} 只")
        except Exception as e:
            logger.warning(f"[{i}/{total}] {ticker}: 下载失败 - {e}")
        # 轻微延时避免对新浪服务器施压
        if i % 10 == 0:
            time.sleep(0.5)

    logger.info(f"下载完成：成功 {len(result)}/{total} 只")
    return result


def _update_bucket_data(
    tickers: list[str],
    *,
    load_fn,
    save_fn,
    market_label: str,
    max_stale_days: int,
    force: bool = False,
    skip_if_in_ndx_cache: bool = False,
) -> dict[str, int]:
    """Generic incremental update routine for a local parquet bucket."""
    today = pd.Timestamp.now().normalize()
    three_years_ago = (today - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    need_full: list[str] = []
    need_incr: dict[str, pd.Timestamp] = {}
    skipped = 0

    for ticker in tickers:
        if skip_if_in_ndx_cache and not force and _ticker_path(ticker).exists():
            skipped += 1
            continue

        local = load_fn(ticker)
        if force:
            need_full.append(ticker)
            continue

        if local is None or local.empty:
            need_full.append(ticker)
            continue

        last_date = pd.Timestamp(local.index.max()).normalize()
        if (today - last_date).days > max_stale_days:
            need_incr[ticker] = last_date
        else:
            skipped += 1

    logger.info(
        f"{market_label} 数据状态：全量下载 {len(need_full)} 只 | "
        f"增量更新 {len(need_incr)} 只 | 已最新 {skipped} 只"
    )

    updated = 0
    failed = 0

    if need_full:
        batch_data = _batch_download_akshare(need_full, start_date=three_years_ago)
        for ticker, df in batch_data.items():
            save_fn(ticker, df)
            updated += 1
        failed += len(set(need_full) - set(batch_data.keys()))

    if need_incr:
        logger.info(f"开始增量更新 {len(need_incr)} 只 {market_label} 股票 ...")
        for i, (ticker, last_date) in enumerate(need_incr.items(), 1):
            try:
                start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                new_df = _fetch_us_stock_akshare(ticker, start_date=start)
                if not new_df.empty:
                    old_df = load_fn(ticker)
                    combined = pd.concat([old_df, new_df])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined = combined.sort_index()
                    save_fn(ticker, combined)
                    updated += 1
                    logger.info(
                        f"[{i}/{len(need_incr)}] {ticker}: 新增 {len(new_df)} 行，总计 {len(combined)} 行"
                    )
                else:
                    logger.debug(f"[{i}/{len(need_incr)}] {ticker}: 无新数据")
            except Exception as e:
                failed += 1
                logger.error(f"[{i}/{len(need_incr)}] {ticker}: 更新失败 - {e}")
            if i % 10 == 0:
                time.sleep(0.5)

    logger.info(f"{market_label} 更新完成: 成功 {updated}, 跳过 {skipped}, 失败 {failed}")
    return {"updated": updated, "skipped": skipped, "failed": failed}


def update_ndx100_data(max_stale_days: int = 0) -> dict[str, pd.DataFrame]:
    """
    增量更新纳指100全部成分股的本地数据。

    使用 akshare（新浪财经源）获取数据，无 Yahoo 限流风险。

    逻辑：
    1. 对于尚无本地数据的股票 → 下载全量数据（取最近 3 年）
    2. 对于已有本地数据的股票 → 下载缺失日期的数据并合并
    3. 如果数据最后日期距今 ≤ max_stale_days 天，跳过

    Args:
        max_stale_days: 数据过期阈值（天）。默认 0 = 每次都增量更新

    Returns:
        {ticker: DataFrame} 包含全部成分股的最新完整数据
    """
    tickers = fetch_ndx100_tickers()
    _update_bucket_data(
        tickers,
        load_fn=load_local_data,
        save_fn=save_local_data,
        market_label="纳指100",
        max_stale_days=max_stale_days,
        force=False,
    )

    # ── 加载全部本地数据返回 ──
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = load_local_data(ticker)
        if df is not None and not df.empty:
            result[ticker] = df

    logger.info(f"本地数据加载完毕：{len(result)}/{len(tickers)} 只股票可用")
    return result


def update_us_data(
    tickers: list[str] | None = None,
    force: bool = False,
    max_stale_days: int = 1,
) -> dict[str, int]:
    """
    增量更新 US universe 本地数据（data/cache/us）。

    逻辑与 NDX/HK 一致：
    1. 本地无数据 -> 全量下载（最近 3 年）
    2. 本地有数据且过期 -> 增量补缺
    3. 未过期 -> 跳过

    Args:
        tickers: 目标 ticker 列表；None 时读取 data/us_sec_profiles.csv
        force: 是否强制刷新
        max_stale_days: 允许最大陈旧天数

    Returns:
        {"updated": int, "skipped": int, "failed": int}
    """
    if tickers is None:
        from stock_ana.config import DATA_DIR
        profiles_path = DATA_DIR / "us_sec_profiles.csv"
        profiles = pd.read_csv(profiles_path, encoding="utf-8-sig")
        tickers = profiles["ticker"].dropna().astype(str).tolist()

    return _update_bucket_data(
        tickers,
        load_fn=load_us_local_data,
        save_fn=save_us_local_data,
        market_label="US",
        max_stale_days=max_stale_days,
        force=force,
        skip_if_in_ndx_cache=False,
    )


def load_all_ndx100_data() -> dict[str, pd.DataFrame]:
    """
    读取 NDX100 成分股数据。

    数据统一存储在 cache/us/，ndx100 只是一个筛选列表。
    优先从 cache/us/ 读取；若某只股票尚未出现在 us 目录
    （如 QQQ、ANSS、BKNG 等），则回退到 cache/ndx100/。

    Returns:
        {ticker: DataFrame} 字典
    """
    tickers = fetch_ndx100_tickers()
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        # 优先 cache/us/
        us_path = US_DIR / f"{ticker}.parquet"
        if us_path.exists():
            df = load_us_local_data(ticker)
        else:
            df = load_local_data(ticker)  # 回退到 cache/ndx100/
        if df is not None and not df.empty:
            result[ticker] = df
    logger.info(f"从本地加载 NDX100 {len(result)}/{len(tickers)} 只股票数据")
    return result


def load_all_us_data() -> dict[str, pd.DataFrame]:
    """
    读取本地 data/cache/us/ 下所有 parquet 文件的美股数据，不做任何网络请求。

    Returns:
        {ticker: DataFrame} 字典
    """
    result: dict[str, pd.DataFrame] = {}
    parquet_files = sorted(US_DIR.glob("*.parquet"))
    for path in parquet_files:
        ticker = path.stem
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            if not df.empty:
                result[ticker] = df
        except Exception as e:
            logger.warning(f"{ticker}: 加载失败 - {e}")
    logger.info(f"从本地 us/ 加载 {len(result)} 只股票数据")
    return result


# ─────────────────── QQQ benchmark 更新 ──────────────────────────────────────


def update_qqq_data() -> "pd.DataFrame | None":
    """Download or incrementally refresh local QQQ benchmark data."""
    from datetime import timedelta

    today = pd.Timestamp.now().normalize()
    three_years_ago = (today - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    local = load_local_data("QQQ")
    if local is not None and not local.empty:
        last_date = local.index.max()
        if (today - last_date).days <= 3:
            logger.info("QQQ 数据已是最新")
            return local

        start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            new_df = _fetch_us_stock_akshare("QQQ", start_date=start)
            if not new_df.empty:
                combined = pd.concat([local, new_df])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                save_local_data("QQQ", combined)
                logger.info(f"QQQ 增量更新完成：新增 {len(new_df)} 行，总计 {len(combined)} 行")
                return combined
            return local
        except Exception as exc:
            logger.error(f"QQQ 增量更新失败: {exc}")
            return local

    try:
        df = _fetch_us_stock_akshare("QQQ", start_date=three_years_ago)
        if not df.empty:
            save_local_data("QQQ", df)
            logger.info(f"QQQ 全量下载完成：{len(df)} 行")
            return df
    except Exception as exc:
        logger.error(f"QQQ 全量下载失败: {exc}")
    return None


# Compatibility alias – callers that previously imported update_us_price_data
# from us_market_updates can now import it directly from fetcher.
update_us_price_data = update_us_data

