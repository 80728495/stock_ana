"""
指标持久化存储模块

每日更新后，对每只股票计算并保存：
  - 扩展 EMA（8/21/34/55/60/144/169/200/250）
  - 成交量均线（vol_ma_5, vol_ma_10, vol_ma_50）
  - 前高价格（prev_high_252d）

存储路径：data/cache/indicators/{market}/{symbol}.parquet
  market: "us" | "hk" | "ndx100"

每次计算从对应市场的 OHLCV parquet 读取原始数据，
结果存储为独立 parquet，不修改原始 OHLCV 文件。
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from loguru import logger

from stock_ana.config import CACHE_DIR
from stock_ana.data.indicators import add_daily_indicators

# 指标缓存根目录
IND_DIR = CACHE_DIR / "indicators"
IND_DIR.mkdir(parents=True, exist_ok=True)

# 各市场 OHLCV parquet 目录
_OHLCV_DIRS: dict[str, Path] = {
    "us":     CACHE_DIR / "us",
    "ndx100": CACHE_DIR / "ndx100",
    "hk":     CACHE_DIR / "hk",
}


# ─────────────────────── 路径工具 ───────────────────────


def _ind_path(symbol: str, market: str) -> Path:
    """Return the parquet path used to persist derived indicators for one symbol."""
    d = IND_DIR / market
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{symbol}.parquet"


def _ohlcv_path(symbol: str, market: str) -> Path:
    """Return the source OHLCV parquet path for one symbol and market."""
    return _OHLCV_DIRS[market] / f"{symbol}.parquet"


# ─────────────────────── 核心计算 ───────────────────────


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 OHLCV DataFrame 计算全部每日指标。

    输入：含 open/high/low/close/volume 列，index 为日期
    输出：含 EMA×9 + vol_ma×3 + prev_high 列的 DataFrame
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_index()
    df = add_daily_indicators(df)
    # 只保留指标列（节省存储，OHLCV 保留在原文件）
    keep_cols = [c for c in df.columns if c not in ("open", "high", "low", "volume", "turnover", "turnover_rate", "pct_change")]
    return df[keep_cols]


def save_indicators(symbol: str, market: str, df: pd.DataFrame) -> None:
    """保存指标数据到 parquet。"""
    path = _ind_path(symbol, market)
    df.to_parquet(path, engine="pyarrow", compression="snappy")


def load_indicators(symbol: str, market: str) -> pd.DataFrame | None:
    """加载指标 parquet；不存在则返回 None。"""
    path = _ind_path(symbol, market)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


# ─────────────────────── 批量更新 ───────────────────────


def update_indicators_for_symbols(
    symbols: list[str],
    market: str,
    delay: float = 0.0,
) -> dict[str, str]:
    """
    批量计算并保存指标。

    Args:
        symbols: 股票代码列表
        market:  市场标识，"us" | "hk" | "ndx100"
        delay:   每只之间的延迟秒数（可选）

    Returns:
        {"ok": [...], "skip": [...], "fail": [...]} 统计
    """
    if market not in _OHLCV_DIRS:
        raise ValueError(f"未知市场: {market}，可选: {list(_OHLCV_DIRS)}")

    ok, skip, fail = [], [], []

    for i, symbol in enumerate(symbols, 1):
        ohlcv_path = _ohlcv_path(symbol, market)
        if not ohlcv_path.exists():
            skip.append(symbol)
            continue
        try:
            df_raw = pd.read_parquet(ohlcv_path)
            df_raw.index = pd.to_datetime(df_raw.index)
            if len(df_raw) < 10:
                skip.append(symbol)
                continue

            df_ind = compute_indicators(df_raw)
            save_indicators(symbol, market, df_ind)
            ok.append(symbol)

            if i % 50 == 0 or i == len(symbols):
                logger.info(f"[indicators {market}] {i}/{len(symbols)} | ok={len(ok)} skip={len(skip)} fail={len(fail)}")

        except Exception as e:
            logger.error(f"[indicators {market}] {symbol}: {e}")
            fail.append(symbol)

        if delay > 0:
            time.sleep(delay)

    logger.success(
        f"指标更新完成 [{market}]: 成功 {len(ok)}, 跳过 {len(skip)}, 失败 {len(fail)}"
    )
    return {"ok": ok, "skip": skip, "fail": fail}


def update_all_indicators(delay: float = 0.0) -> None:
    """
    更新全部市场的指标（US + NDX100 + HK）。
    从各自 OHLCV parquet 目录扫描现有文件，自动确定 symbol 列表。
    """
    for market, ohlcv_dir in _OHLCV_DIRS.items():
        if not ohlcv_dir.exists():
            continue
        symbols = [p.stem for p in ohlcv_dir.glob("*.parquet")]
        if not symbols:
            continue
        logger.info(f"更新指标 [{market}]：{len(symbols)} 只 ...")
        update_indicators_for_symbols(symbols, market, delay=delay)
