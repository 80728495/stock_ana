"""
Wave 结构持久化存储模块

对每只股票：
    1. 计算 analyze_wave_structure() 结果
    2. 序列化为 JSON，附带 iloc → date 的映射
    3. 保存到 data/cache/wave_structure/{market}/{symbol}.json

每日数据更新默认覆盖全量 US+HK；也支持仅更新 Shawn List。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.config import CACHE_DIR
from stock_ana.strategies.primitives.wave import analyze_wave_structure

# Wave 结构缓存目录
WAVE_DIR = CACHE_DIR / "wave_structure"
WAVE_DIR.mkdir(parents=True, exist_ok=True)

# 各市场 OHLCV parquet 目录
_OHLCV_DIRS: dict[str, Path] = {
    "us":     CACHE_DIR / "us",
    "ndx100": CACHE_DIR / "ndx100",
    "hk":     CACHE_DIR / "hk",
}


# ─────────────────────── 序列化工具 ───────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """JSON 序列化时将 numpy 数值类型转换为 Python 原生类型。"""

    def default(self, obj: Any) -> Any:
        """Normalize numpy/pandas scalar types before JSON serialization."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)


def _pivot_to_serializable(pivot: dict | None, date_index: pd.DatetimeIndex) -> dict | None:
    """将 pivot dict 转换为 JSON 可序列化格式，附加 date 字段。"""
    if pivot is None:
        return None
    out = {
        "iloc": int(pivot["iloc"]),
        "value": float(pivot["value"]),
        "type": str(pivot.get("type", "")),
    }
    iloc = int(pivot["iloc"])
    if 0 <= iloc < len(date_index):
        out["date"] = date_index[iloc].strftime("%Y-%m-%d")
    return out


def _serialize_wave_result(result: dict, date_index: pd.DatetimeIndex) -> dict:
    """将 analyze_wave_structure 返回值转换为 JSON 可序列化格式。"""

    def serialize_sub_wave(sw: dict) -> dict:
        """Serialize one sub-wave payload into a JSON-friendly dict."""
        return {
            "sub_number": int(sw.get("sub_number", 0)),
            "start_pivot": _pivot_to_serializable(sw.get("start_pivot"), date_index),
            "peak_pivot": _pivot_to_serializable(sw.get("peak_pivot"), date_index),
            "end_pivot": _pivot_to_serializable(sw.get("end_pivot"), date_index),
            "pullback_type": str(sw.get("pullback_type", "")),
            "pullback_band": str(sw.get("pullback_band", "")),
            "rise_pct": float(sw.get("rise_pct", 0)),
        }

    def serialize_major_wave(w: dict) -> dict:
        """Serialize one major-wave payload into a JSON-friendly dict."""
        return {
            "wave_number": int(w.get("wave_number", 0)),
            "start_pivot": _pivot_to_serializable(w.get("start_pivot"), date_index),
            "end_pivot": _pivot_to_serializable(w.get("end_pivot"), date_index),
            "peak_pivot": _pivot_to_serializable(w.get("peak_pivot"), date_index),
            "sub_waves": [serialize_sub_wave(s) for s in w.get("sub_waves", [])],
            "sub_wave_count": int(w.get("sub_wave_count", 0)),
            "mid_pullback_count": int(w.get("mid_pullback_count", 0)),
            "rise_pct": float(w.get("rise_pct", 0)),
            "duration_days": int(w.get("duration_days", 0)),
        }

    all_pivots_ser = [
        {
            "iloc": int(p["iloc"]),
            "value": float(p["value"]),
            "type": str(p.get("type", "")),
            "date": date_index[int(p["iloc"])].strftime("%Y-%m-%d")
                    if 0 <= int(p["iloc"]) < len(date_index) else "",
        }
        for p in result.get("all_pivots", [])
    ]

    return {
        "major_waves": [serialize_major_wave(w) for w in result.get("major_waves", [])],
        "current_wave_number": int(result.get("current_wave_number", 0)),
        "current_sub_wave": int(result.get("current_sub_wave", 0)),
        "current_status": str(result.get("current_status", "")),
        "all_pivots": all_pivots_ser,
    }


# ─────────────────────── 路径工具 ───────────────────────

def _wave_path(symbol: str, market: str) -> Path:
    """Return the JSON cache path for one symbol's persisted wave structure."""
    d = WAVE_DIR / market
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{symbol}.json"


# ─────────────────────── 核心 IO ───────────────────────

def save_wave_structure(symbol: str, market: str, wave_data: dict) -> None:
    """将序列化好的 wave 数据写入 JSON 文件。"""
    path = _wave_path(symbol, market)
    path.write_text(
        json.dumps(wave_data, ensure_ascii=False, indent=2, cls=_NumpyEncoder),
        encoding="utf-8",
    )


def load_wave_structure(symbol: str, market: str) -> dict | None:
    """加载已保存的 wave 结构 JSON；不存在则返回 None。"""
    path = _wave_path(symbol, market)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"加载 wave 结构失败 [{market}/{symbol}]: {e}")
        return None


# ─────────────────────── 单只股票计算 ───────────────────────

def compute_and_save_wave(
    symbol: str,
    market: str,
    df: pd.DataFrame | None = None,
) -> dict | None:
    """
    计算并保存单只股票的 wave 结构。

    Args:
        symbol: 股票代码
        market: "us" | "hk" | "ndx100"
        df:     已加载的 OHLCV DataFrame；None 则自动从本地 parquet 加载

    Returns:
        序列化后的 wave_data dict；失败返回 None
    """
    if df is None:
        ohlcv_path = _OHLCV_DIRS.get(market, CACHE_DIR / market) / f"{symbol}.parquet"
        if not ohlcv_path.exists():
            logger.warning(f"[wave] {market}/{symbol}: OHLCV 文件不存在")
            return None
        df = pd.read_parquet(ohlcv_path)
        df.index = pd.to_datetime(df.index)

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_index()

    if len(df) < 100:
        logger.debug(f"[wave] {market}/{symbol}: 数据太少 ({len(df)} 行)，跳过")
        return None

    try:
        result = analyze_wave_structure(df)
        date_index = df.index
        wave_data = _serialize_wave_result(result, date_index)
        wave_data["symbol"] = symbol
        wave_data["market"] = market
        wave_data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        wave_data["data_end_date"] = df.index[-1].strftime("%Y-%m-%d")
        save_wave_structure(symbol, market, wave_data)
        return wave_data
    except Exception as e:
        logger.error(f"[wave] {market}/{symbol}: 计算失败 - {e}")
        return None


# ─────────────────────── 批量更新 ───────────────────────

def update_wave_structures(
    symbols_market: list[tuple[str, str]],
    delay: float = 0.05,
) -> dict[str, list[str]]:
    """
    批量更新 wave 结构。

    Args:
        symbols_market: [(symbol, market), ...] 例如 [("00700", "hk"), ("NVDA", "us")]
        delay:          每只之间的延迟秒数

    Returns:
        {"ok": [...], "skip": [...], "fail": [...]}
    """
    ok, skip, fail = [], [], []
    total = len(symbols_market)

    for i, (symbol, market) in enumerate(symbols_market, 1):
        result = compute_and_save_wave(symbol, market)
        if result is None:
            skip.append(f"{market}/{symbol}")
        else:
            ok.append(f"{market}/{symbol}")

        if i % 10 == 0 or i == total:
            logger.info(
                f"[wave] 进度 {i}/{total} | ok={len(ok)} skip={len(skip)} fail={len(fail)}"
            )

        if delay > 0:
            time.sleep(delay)

    logger.success(
        f"Wave 结构更新完成：成功 {len(ok)}, 跳过 {len(skip)}, 失败 {len(fail)}"
    )
    return {"ok": ok, "skip": skip, "fail": fail}


def update_wave_structures_for_shawn_list() -> dict[str, list[str]]:
    """更新 Shawn List 中所有股票的 wave 结构。"""
    from stock_ana.data.list_manager import load_shawn_list
    shawn = load_shawn_list()

    pairs: list[tuple[str, str]] = []
    for code in shawn.get("hk", []):
        pairs.append((code, "hk"))
    for ticker in shawn.get("us", []):
        pairs.append((ticker, "us"))

    logger.info(f"更新 Shawn List wave 结构：{len(pairs)} 只 ...")
    return update_wave_structures(pairs)


def _list_symbols_in_market(market: str) -> list[str]:
    """从对应市场的 OHLCV 缓存目录列出全部 symbol。"""
    ohlcv_dir = _OHLCV_DIRS.get(market)
    if ohlcv_dir is None or not ohlcv_dir.exists():
        return []
    return sorted(p.stem for p in ohlcv_dir.glob("*.parquet"))


def update_wave_structures_for_all_us_hk() -> dict[str, list[str]]:
    """
    更新全量 US + HK 股票的 wave 结构。

    数据来源为本地缓存目录：
      - data/cache/us/*.parquet
      - data/cache/hk/*.parquet
    """
    pairs: list[tuple[str, str]] = []
    us_symbols = _list_symbols_in_market("us")
    hk_symbols = _list_symbols_in_market("hk")

    for sym in us_symbols:
        pairs.append((sym, "us"))
    for sym in hk_symbols:
        pairs.append((sym, "hk"))

    logger.info(
        f"更新全量 US+HK wave 结构：US {len(us_symbols)} 只 + HK {len(hk_symbols)} 只 = {len(pairs)} 只 ..."
    )
    return update_wave_structures(pairs)


def get_wave_summary(symbol: str, market: str) -> str:
    """返回单只股票 wave 结构的简短文字摘要。"""
    data = load_wave_structure(symbol, market)
    if data is None:
        return f"{symbol}: No wave data"

    status = data.get("current_status", "?")
    wave_n = data.get("current_wave_number", 0)
    sub_n = data.get("current_sub_wave", 0)
    n_major = len(data.get("major_waves", []))
    updated = data.get("updated_at", "?")

    return (
        f"{symbol} [{market}] 状态={status} 大浪={wave_n}/{n_major} "
        f"子浪={sub_n} 更新={updated}"
    )
