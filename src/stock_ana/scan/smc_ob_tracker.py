"""
SMC Order Block 每日追踪器

以因果 OB 检测结果为基础，维护每只股票的 OB 状态文件，
并在每次日常运行时输出三种增量事件：

  new_ob    — 今日首次触发（被今日收盘价突破 swing 而识别）的新 OB
  mitigated — 此前活跃的 OB 今日被价格穿越而失效
  touched   — 今日价格区间（high/low）刺入了某个仍活跃的 OB 区块

状态文件路径:
    data/cache/smc_ob_state/{market}/{symbol}.json

外部调用示例:
    from stock_ana.scan.smc_ob_tracker import run_daily, get_active_obs, process_symbol

    # 批量每日扫描
    events = run_daily(watchlist=my_watchlist)

    # 查询单只股票当前活跃 OB
    obs = get_active_obs("NVDA", "us")

    # 处理单只股票
    events = process_symbol("NVDA", "us", df_ohlcv)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.config import CACHE_DIR
from stock_ana.strategies.impl.smc import _ob_causal

# ── 状态存储根目录 ────────────────────────────────────────────────────────────
OB_STATE_DIR = CACHE_DIR / "smc_ob_state"

# 事件类型字面量
EventType = Literal["new_ob", "mitigated", "touched"]


# ═════════════════════════════════════════════════════════════════════════════
# 状态文件 I/O
# ═════════════════════════════════════════════════════════════════════════════

def _state_path(symbol: str, market: str) -> Path:
    return OB_STATE_DIR / market.lower() / f"{symbol}.json"


def load_ob_state(symbol: str, market: str) -> dict:
    """加载持久化的 OB 状态；文件不存在时返回空状态。

    返回结构:
        {
          "last_updated": "2026-05-22" | None,
          "swing_length": 5 | None,
          "obs": {
            "<ob_id>": {
              "ob_id": str,
              "direction": 1 | -1,
              "top": float,
              "bottom": float,
              "formed_date": str,
              "ob_volume": float,
              "percentage": float,
              "status": "active" | "touched" | "mitigated",
              "mitigated_date": str | None,
            },
            ...
          }
        }
    """
    p = _state_path(symbol, market)
    if not p.exists():
        return {"last_updated": None, "swing_length": None, "obs": {}}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_ob_state(symbol: str, market: str, state: dict) -> None:
    """持久化 OB 状态文件。"""
    p = _state_path(symbol, market)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def get_active_obs(symbol: str, market: str) -> list[dict]:
    """返回上次扫描后仍活跃（active 或 touched）的 OB 列表。

    不重新计算，直接读取状态文件。供外部快速查询使用。
    """
    state = load_ob_state(symbol, market)
    return [
        v for v in state.get("obs", {}).values()
        if v["status"] in ("active", "touched")
    ]


# ═════════════════════════════════════════════════════════════════════════════
# 内部工具
# ═════════════════════════════════════════════════════════════════════════════

def _ob_key(formed_date: str, direction: int) -> str:
    """生成 OB 唯一标识：日期 + 方向。"""
    tag = "bull" if direction == 1 else "bear"
    return f"{formed_date}_{tag}"


def _extract_ob_map(df: pd.DataFrame, ob_df: pd.DataFrame) -> dict[str, dict]:
    """将 _ob_causal 结果转换为以 ob_key 为索引的字典。

    每个 value 包含 OB 的完整信息及消除状态，供后续差分使用。
    """
    ob_map: dict[str, dict] = {}
    for bar_idx in range(len(df)):
        row = ob_df.iloc[bar_idx]
        if pd.isna(row.get("OB")) or row["OB"] == 0:
            continue

        direction = int(row["OB"])
        ts = df.index[bar_idx]
        formed_date = str(ts.date()) if hasattr(ts, "date") else str(ts)
        key = _ob_key(formed_date, direction)

        mit_idx = row["MitigatedIndex"]
        is_mitigated = pd.notna(mit_idx) and int(mit_idx) != 0
        mitigated_date: str | None = None
        if is_mitigated:
            mi = int(mit_idx)
            if 0 <= mi < len(df):
                mts = df.index[mi]
                mitigated_date = str(mts.date()) if hasattr(mts, "date") else str(mts)

        ob_map[key] = {
            "ob_id":         key,
            "direction":     direction,
            "top":           float(row["Top"]),
            "bottom":        float(row["Bottom"]),
            "formed_date":   formed_date,
            "ob_volume":     float(row["OBVolume"]) if pd.notna(row.get("OBVolume")) else 0.0,
            "percentage":    float(row["Percentage"]) if pd.notna(row.get("Percentage")) else 0.0,
            "is_mitigated":  is_mitigated,
            "mitigated_date": mitigated_date,
        }
    return ob_map


def _touches(cur_low: float, cur_high: float, top: float, bot: float) -> bool:
    """判断今日 [low, high] 是否与 OB [bottom, top] 区间重叠。"""
    return cur_low <= top and cur_high >= bot


# ═════════════════════════════════════════════════════════════════════════════
# 核心：单只股票每日更新
# ═════════════════════════════════════════════════════════════════════════════

def process_symbol(
    symbol: str,
    market: str,
    df: pd.DataFrame,
    swing_length: int = 5,
    close_mitigation: bool = False,
) -> list[dict]:
    """对单只股票执行每日 OB 状态更新，返回当日事件列表。

    算法流程:
        1. 用因果 OB 检测算法重新计算全量 OB（每次都从全历史运行，保证一致性）
        2. 加载上次保存的 OB 状态文件（存储前一天的 OB 集合）
        3. 差分：
           - new_ob    = 今日结果中有但上次状态中没有的未消除 OB
           - mitigated = 上次为 active/touched，今日已被消除 (MitigatedIndex != 0)
           - touched   = 今日 [low, high] 刺入仍活跃的 OB 区块，
                         且上次状态不是 "touched"（避免持续在区块内重复报告）
        4. 更新状态文件并返回事件列表

    参数:
        symbol:           股票代码
        market:           市场（"us" / "hk" / "cn"，不区分大小写）
        df:               完整日线 OHLCV DataFrame（列名任意大小写，含 open/high/low/close/volume）
        swing_length:     摆动点确认窗口（默认 5）
        close_mitigation: 是否以收盘价判断 OB 消除（默认以 high/low）

    返回:
        list of event dicts，每个事件包含:
          event        : "new_ob" | "mitigated" | "touched"
          symbol       : 代码
          market       : 市场（大写）
          ob_id        : OB 唯一标识 "<formed_date>_<bull|bear>"
          direction    : 1=看涨  -1=看跌
          top          : OB 顶部价格
          bottom       : OB 底部价格
          formed_date  : OB 所在 K 线日期（历史 bar，不是触发日）
          as_of        : 本次检测日期（今日最新 bar 日期）
          percentage   : OB 强度 0~100（仅 new_ob）
          mitigated_date: 消除日期（仅 mitigated）
          current_close : 今日收盘（仅 touched）
          current_high  : 今日最高（仅 touched）
          current_low   : 今日最低（仅 touched）
    """
    if df is None or len(df) < swing_length * 2 + 10:
        return []

    # 统一列名 & 排序
    df = df.rename(columns={c: c.lower() for c in df.columns}).sort_index()

    # 今日（最新 bar）信息
    as_of_ts  = df.index[-1]
    as_of     = str(as_of_ts.date()) if hasattr(as_of_ts, "date") else str(as_of_ts)
    cur_close = float(df["close"].iloc[-1])
    cur_high  = float(df["high"].iloc[-1])
    cur_low   = float(df["low"].iloc[-1])

    # ── 步骤 1：计算全量因果 OB ──────────────────────────────────────────────
    try:
        from smartmoneyconcepts import smc as _upstream  # noqa: PLC0415
        swing_hl = _upstream.swing_highs_lows(df, swing_length=swing_length)
        ob_df    = _ob_causal(df, swing_hl, swing_length=swing_length,
                              close_mitigation=close_mitigation)
    except Exception as exc:
        logger.debug(f"{symbol} OB 计算失败: {exc}")
        return []

    current_ob_map = _extract_ob_map(df, ob_df)

    # ── 步骤 2：加载历史状态 ─────────────────────────────────────────────────
    state = load_ob_state(symbol, market)
    prev_obs: dict[str, dict] = state.get("obs", {})

    # 首次运行 or swing_length 参数变化 → 只建立基线，不产出事件
    is_first_run = state["last_updated"] is None
    if not is_first_run and state.get("swing_length") != swing_length:
        logger.warning(
            f"{symbol}: swing_length 从 {state.get('swing_length')} → {swing_length}，"
            "重建 OB 基线（本次无事件）"
        )
        is_first_run = True

    # ── 步骤 3：差分生成事件 ─────────────────────────────────────────────────
    events: list[dict] = []
    new_ob_keys: set[str] = set()  # 今日新增的 OB，避免同一个 OB 又报 touched

    if is_first_run:
        active_cnt = sum(1 for v in current_ob_map.values() if not v["is_mitigated"])
        logger.info(f"{market.upper()}:{symbol} 首次建立 OB 基线，{active_cnt} 个 active OB")

    else:
        # ── 3a. new_ob ────────────────────────────────────────────────────────
        for key, ob in current_ob_map.items():
            if ob["is_mitigated"]:
                continue
            if key not in prev_obs:
                new_ob_keys.add(key)
                events.append({
                    "event":       "new_ob",
                    "symbol":      symbol,
                    "market":      market.upper(),
                    "ob_id":       key,
                    "direction":   ob["direction"],
                    "top":         ob["top"],
                    "bottom":      ob["bottom"],
                    "formed_date": ob["formed_date"],
                    "percentage":  ob["percentage"],
                    "as_of":       as_of,
                })

        # ── 3b. mitigated ─────────────────────────────────────────────────────
        for key, prev_ob in prev_obs.items():
            if prev_ob.get("status") not in ("active", "touched"):
                continue  # 已经是 mitigated，无需重复
            cur = current_ob_map.get(key)
            if cur is None or not cur["is_mitigated"]:
                continue
            events.append({
                "event":          "mitigated",
                "symbol":         symbol,
                "market":         market.upper(),
                "ob_id":          key,
                "direction":      cur["direction"],
                "top":            cur["top"],
                "bottom":         cur["bottom"],
                "formed_date":    cur["formed_date"],
                "mitigated_date": cur["mitigated_date"] or as_of,
                "as_of":          as_of,
            })

        # ── 3c. touched ───────────────────────────────────────────────────────
        for key, ob in current_ob_map.items():
            if ob["is_mitigated"]:
                continue
            if key in new_ob_keys:
                continue  # 今日新 OB，用 new_ob 事件表达，不重复报 touched
            if not _touches(cur_low, cur_high, ob["top"], ob["bottom"]):
                continue
            # 若上次已是 touched（说明价格持续停留区间内），不重复报告
            if prev_obs.get(key, {}).get("status") == "touched":
                continue
            events.append({
                "event":         "touched",
                "symbol":        symbol,
                "market":        market.upper(),
                "ob_id":         key,
                "direction":     ob["direction"],
                "top":           ob["top"],
                "bottom":        ob["bottom"],
                "formed_date":   ob["formed_date"],
                "current_close": cur_close,
                "current_high":  cur_high,
                "current_low":   cur_low,
                "as_of":         as_of,
            })

    # ── 步骤 4：更新状态文件 ─────────────────────────────────────────────────
    new_obs_state: dict[str, dict] = {}
    for key, ob in current_ob_map.items():
        if ob["is_mitigated"]:
            status = "mitigated"
        elif _touches(cur_low, cur_high, ob["top"], ob["bottom"]):
            status = "touched"
        else:
            status = "active"

        new_obs_state[key] = {
            "ob_id":          key,
            "direction":      ob["direction"],
            "top":            ob["top"],
            "bottom":         ob["bottom"],
            "formed_date":    ob["formed_date"],
            "ob_volume":      ob["ob_volume"],
            "percentage":     ob["percentage"],
            "status":         status,
            "mitigated_date": ob["mitigated_date"],
        }

    save_ob_state(symbol, market, {
        "last_updated": as_of,
        "swing_length": swing_length,
        "obs":          new_obs_state,
    })

    # ── 日志摘要 ─────────────────────────────────────────────────────────────
    if events:
        n_new = sum(1 for e in events if e["event"] == "new_ob")
        n_mit = sum(1 for e in events if e["event"] == "mitigated")
        n_tch = sum(1 for e in events if e["event"] == "touched")
        logger.info(
            f"{market.upper()}:{symbol} @ {as_of}  "
            f"新OB={n_new}  消除={n_mit}  触碰={n_tch}"
        )

    return events


# ═════════════════════════════════════════════════════════════════════════════
# 批量入口
# ═════════════════════════════════════════════════════════════════════════════

def run_daily(
    watchlist: dict | None = None,
    swing_length: int = 5,
    close_mitigation: bool = False,
) -> dict[str, list[dict]]:
    """批量对 watchlist 执行每日 OB 状态更新。

    参数:
        watchlist:        {symbol: (market, name, ...)} 格式，None 则使用默认自选
        swing_length:     摆动点确认窗口
        close_mitigation: 是否以收盘价判断 OB 消除

    返回:
        {symbol: [events]} — 只包含有事件的股票，方便调用方过滤
    """
    if watchlist is None:
        from stock_ana.data.market_data import build_watchlist  # noqa: PLC0415
        watchlist = build_watchlist()

    results: dict[str, list[dict]] = {}
    total = len(watchlist)

    for i, (symbol, meta) in enumerate(watchlist.items(), 1):
        market = meta[0].lower()
        ohlcv_path = CACHE_DIR / market / f"{symbol}.parquet"
        if not ohlcv_path.exists():
            logger.debug(f"缓存不存在: {symbol} ({market})")
            continue

        try:
            df = pd.read_parquet(ohlcv_path)
            df.index = pd.to_datetime(df.index)
        except Exception as exc:
            logger.warning(f"加载失败 {symbol}: {exc}")
            continue

        evts = process_symbol(
            symbol, market, df,
            swing_length=swing_length,
            close_mitigation=close_mitigation,
        )
        if evts:
            results[symbol] = evts

        if i % 100 == 0 or i == total:
            logger.info(f"进度: [{i}/{total}]  有事件: {len(results)} 只")

    # 汇总
    all_evts = [e for evts in results.values() for e in evts]
    logger.info(
        f"SMC OB 每日扫描完成: {total} 只  "
        f"新OB={sum(1 for e in all_evts if e['event']=='new_ob')}  "
        f"消除={sum(1 for e in all_evts if e['event']=='mitigated')}  "
        f"触碰={sum(1 for e in all_evts if e['event']=='touched')}"
    )
    return results
