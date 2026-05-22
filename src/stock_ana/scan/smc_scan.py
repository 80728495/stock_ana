#!/usr/bin/env python3
"""
Smart Money Concepts (SMC) 扫描器

基于当前价格与 SMC 指标，扫描以下信号：
  1. 未消除的看涨 FVG（公允价值缺口）——价格回踩到 FVG 内
  2. 未消除的看涨订单块（OB）——价格进入 OB 区间
  3. CHoCH（角色转换）——最近出现看涨 CHoCH
  4. BOS（结构突破）——最近出现看涨 BOS

输出包含信号类型、价格水平、来源日期，可直接用于通知或进一步筛选。

用法:
    python -m stock_ana.scan.smc_scan                      # 扫描默认自选列表
    python -m stock_ana.scan.smc_scan --us                 # 扫描美股全量
    python -m stock_ana.scan.smc_scan --hk                 # 扫描港股宇宙池
    python -m stock_ana.scan.smc_scan --lookback 5         # 放宽到最近 5 根 K 线
    python -m stock_ana.scan.smc_scan --swing_length 3     # 摆动点灵敏度
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.config import CACHE_DIR, OUTPUT_DIR
from stock_ana.strategies.impl.smc import (
    compute_swing_hl,
    compute_fvg,
    compute_bos_choch,
    compute_ob,
    compute_liquidity,
)
from stock_ana.data.market_data import build_watchlist

SCAN_OUT_DIR = OUTPUT_DIR / "smc_scan"


# ─────────────────────── 核心扫描逻辑 ───────────────────────


def _scan_symbol(
    symbol: str,
    df: pd.DataFrame,
    lookback: int,
    swing_length: int,
) -> list[dict]:
    """对单只股票运行 SMC 扫描，返回信号列表（可能为空）。"""
    if df is None or len(df) < swing_length * 2 + 20:
        return []

    # 确保列名小写
    df = df.rename(columns={c: c.lower() for c in df.columns})
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return []

    try:
        swing_hl = compute_swing_hl(df, swing_length=swing_length)
        fvg_df    = compute_fvg(df)
        bos_df    = compute_bos_choch(df, swing_hl)
        ob_df     = compute_ob(df, swing_hl)
    except Exception as e:
        logger.debug(f"SMC 计算失败 {symbol}: {e}")
        return []

    n = len(df)
    current_close = float(df["close"].iloc[-1])
    current_date  = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])

    signals: list[dict] = []

    # ─── 1. 看涨 FVG 回踩 ───
    # FVG 未消除（MitigatedIndex==0），且方向=1（看涨），且当前价格在 FVG 区间内
    fvg_active = fvg_df[
        (fvg_df["FVG"] == 1)
        & (fvg_df["MitigatedIndex"] == 0)
    ]
    # 只看最近 lookback 根之前形成的（避免最新形成还没验证）
    fvg_old = fvg_active[fvg_active.index < n - lookback]
    for idx, row in fvg_old.iterrows():
        top    = float(row["Top"])
        bottom = float(row["Bottom"])
        if bottom <= current_close <= top:
            signals.append({
                "type":      "FVG_BULLISH",
                "level_top": top,
                "level_bot": bottom,
                "formed_bar": int(idx),
                "formed_date": str(df.index[idx].date()) if hasattr(df.index[idx], "date") else str(df.index[idx]),
                "current_close": current_close,
                "current_date":  current_date,
            })
            break  # 只报告最近一个

    # ─── 2. 看涨 OB 回踩 ───
    ob_active = ob_df[
        (ob_df["OB"] == 1)
        & (ob_df["MitigatedIndex"] == 0)
    ]
    ob_old = ob_active[ob_active.index < n - lookback]
    for idx, row in ob_old.iterrows():
        top    = float(row["Top"])
        bottom = float(row["Bottom"])
        if bottom <= current_close <= top:
            signals.append({
                "type":        "OB_BULLISH",
                "level_top":   top,
                "level_bot":   bottom,
                "ob_pct":      float(row["Percentage"]) if not np.isnan(row["Percentage"]) else 0.0,
                "formed_bar":  int(idx),
                "formed_date": str(df.index[idx].date()) if hasattr(df.index[idx], "date") else str(df.index[idx]),
                "current_close": current_close,
                "current_date":  current_date,
            })
            break

    # ─── 3. CHoCH（角色转换：看涨）──
    choch_bull = bos_df[
        (bos_df["CHOCH"] == 1)             # 看涨 CHoCH
        & bos_df["BrokenIndex"].notna()
        & (bos_df["BrokenIndex"] > 0)
    ]
    if not choch_bull.empty:
        last_choch = choch_bull.iloc[-1]
        broken_bar  = int(last_choch["BrokenIndex"])
        if broken_bar >= n - lookback:
            signals.append({
                "type":         "CHOCH_BULLISH",
                "level":        float(last_choch["Level"]),
                "broken_bar":   broken_bar,
                "broken_date":  str(df.index[min(broken_bar, n - 1)].date()) if hasattr(df.index[0], "date") else str(df.index[min(broken_bar, n - 1)]),
                "current_close":  current_close,
                "current_date":   current_date,
            })

    # ─── 4. BOS（结构突破：看涨）──
    bos_bull = bos_df[
        (bos_df["BOS"] == 1)
        & bos_df["BrokenIndex"].notna()
        & (bos_df["BrokenIndex"] > 0)
    ]
    if not bos_bull.empty:
        last_bos = bos_bull.iloc[-1]
        broken_bar = int(last_bos["BrokenIndex"])
        if broken_bar >= n - lookback:
            signals.append({
                "type":        "BOS_BULLISH",
                "level":       float(last_bos["Level"]),
                "broken_bar":  broken_bar,
                "broken_date": str(df.index[min(broken_bar, n - 1)].date()) if hasattr(df.index[0], "date") else str(df.index[min(broken_bar, n - 1)]),
                "current_close": current_close,
                "current_date":  current_date,
            })

    # ─── 5. 看跌 FVG 反弹到区间内 ───
    fvg_bear = fvg_df[
        (fvg_df["FVG"] == -1)
        & (fvg_df["MitigatedIndex"] == 0)
    ]
    fvg_bear_old = fvg_bear[fvg_bear.index < n - lookback]
    for idx, row in fvg_bear_old.iterrows():
        top    = float(row["Top"])
        bottom = float(row["Bottom"])
        if bottom <= current_close <= top:
            signals.append({
                "type":        "FVG_BEARISH",
                "level_top":   top,
                "level_bot":   bottom,
                "formed_bar":  int(idx),
                "formed_date": str(df.index[idx].date()) if hasattr(df.index[idx], "date") else str(df.index[idx]),
                "current_close": current_close,
                "current_date":  current_date,
            })
            break

    # ─── 6. 看跌 OB 反弹到区间内 ───
    ob_bear = ob_df[
        (ob_df["OB"] == -1)
        & (ob_df["MitigatedIndex"] == 0)
    ]
    ob_bear_old = ob_bear[ob_bear.index < n - lookback]
    for idx, row in ob_bear_old.iterrows():
        top    = float(row["Top"])
        bottom = float(row["Bottom"])
        if bottom <= current_close <= top:
            signals.append({
                "type":        "OB_BEARISH",
                "level_top":   top,
                "level_bot":   bottom,
                "ob_pct":      float(row["Percentage"]) if not np.isnan(row["Percentage"]) else 0.0,
                "formed_bar":  int(idx),
                "formed_date": str(df.index[idx].date()) if hasattr(df.index[idx], "date") else str(df.index[idx]),
                "current_close": current_close,
                "current_date":  current_date,
            })
            break

    # ─── 7. CHoCH（角色转换：看跌）──
    choch_bear = bos_df[
        (bos_df["CHOCH"] == -1)
        & bos_df["BrokenIndex"].notna()
        & (bos_df["BrokenIndex"] > 0)
    ]
    if not choch_bear.empty:
        last_choch_b = choch_bear.iloc[-1]
        broken_bar_b = int(last_choch_b["BrokenIndex"])
        if broken_bar_b >= n - lookback:
            signals.append({
                "type":        "CHOCH_BEARISH",
                "level":       float(last_choch_b["Level"]),
                "broken_bar":  broken_bar_b,
                "broken_date": str(df.index[min(broken_bar_b, n - 1)].date()) if hasattr(df.index[0], "date") else str(df.index[min(broken_bar_b, n - 1)]),
                "current_close": current_close,
                "current_date":  current_date,
            })

    # ─── 8. BOS（结构突破：看跌）──
    bos_bear = bos_df[
        (bos_df["BOS"] == -1)
        & bos_df["BrokenIndex"].notna()
        & (bos_df["BrokenIndex"] > 0)
    ]
    if not bos_bear.empty:
        last_bos_b = bos_bear.iloc[-1]
        broken_bar_b = int(last_bos_b["BrokenIndex"])
        if broken_bar_b >= n - lookback:
            signals.append({
                "type":        "BOS_BEARISH",
                "level":       float(last_bos_b["Level"]),
                "broken_bar":  broken_bar_b,
                "broken_date": str(df.index[min(broken_bar_b, n - 1)].date()) if hasattr(df.index[0], "date") else str(df.index[min(broken_bar_b, n - 1)]),
                "current_close": current_close,
                "current_date":  current_date,
            })

    return signals


def run_scan(
    watchlist: dict | None = None,
    lookback: int = 3,
    swing_length: int = 5,
) -> list[dict]:
    """运行 SMC 扫描，返回所有命中信号列表。

    每个元素格式:
        symbol, name, market, signal_type, level_top, level_bot, formed_date, current_close ...

    parameters:
        watchlist:    {symbol: (market, name, path, sector)} 字典，None 则使用默认自选列表
        lookback:     信号有效的最近 K 线根数（CHoCH/BOS/FVG/OB 突破后多少根内有效）
        swing_length: 摆动高低点的左右确认K线数
    """
    if watchlist is None:
        watchlist = build_watchlist()

    all_signals: list[dict] = []

    for symbol, meta in watchlist.items():
        market, name = meta[0], meta[1]
        try:
            df = _load_daily_ohlcv(symbol, market)
        except Exception as e:
            logger.debug(f"加载失败 {symbol}: {e}")
            continue

        hits = _scan_symbol(symbol, df, lookback=lookback, swing_length=swing_length)
        for h in hits:
            h["symbol"] = symbol
            h["name"]   = name
            h["market"] = market
            all_signals.append(h)

    logger.info(f"SMC 扫描完成: {len(watchlist)} 只，命中 {len(all_signals)} 个信号")
    return all_signals


# ─────────────────────── 数据加载 ───────────────────────


def _load_daily_ohlcv(symbol: str, market: str) -> pd.DataFrame:
    """从原始 OHLCV 缓存加载数据（SMC 需要 open/high/low/close/volume）。"""
    market_lower = market.lower()
    # SMC 需要完整 OHLCV，优先从原始 OHLCV 目录读取
    for sub in (market_lower, f"{market_lower}"):
        ohlcv_path = CACHE_DIR / sub / f"{symbol}.parquet"
        if ohlcv_path.exists():
            df = pd.read_parquet(ohlcv_path)
            df.index = pd.to_datetime(df.index)
            return df

    raise FileNotFoundError(f"无 OHLCV 缓存: {symbol} ({market})")


# ─────────────────────── 列表构建 ───────────────────────


def _build_us_universe_watchlist() -> dict:
    from stock_ana.config import DATA_DIR
    csv_path = DATA_DIR / "us_universe.csv"
    if not csv_path.exists():
        logger.warning("us_universe.csv 不存在，请先运行 daily_update.py")
        return {}
    df = pd.read_csv(csv_path)
    watchlist = {}
    for _, row in df.iterrows():
        symbol = str(row.get("ticker", row.get("symbol", ""))).strip().upper()
        name   = str(row.get("name", symbol))
        if not symbol:
            continue
        watchlist[symbol] = ("US", name, None, "")
    return watchlist


def _build_hk_universe_watchlist() -> dict:
    from stock_ana.data.list_manager import _read_md_table
    from stock_ana.config import DATA_DIR
    path = DATA_DIR / "lists" / "hk_universe_list.md"
    if not path.exists():
        return {}
    rows = _read_md_table(path)
    watchlist = {}
    for r in rows:
        if len(r) < 3:
            continue
        code = r[1].strip().zfill(5)
        name_zh = r[2].strip() or code
        if not (CACHE_DIR / "hk" / f"{code}.parquet").exists():
            continue
        watchlist[code] = ("HK", name_zh, None, "")
    return watchlist


# ─────────────────────── CLI 入口 ───────────────────────


def _print_signals(signals: list[dict]) -> None:
    if not signals:
        print("  无信号")
        return
    for s in signals:
        sig_type = s["type"]
        sym  = s["symbol"]
        name = s["name"]
        close = s["current_close"]
        date  = s["current_date"]
        if "level_top" in s:
            print(f"  [{sig_type}] {sym} ({name})  close={close:.2f}  OB/FVG=[{s['level_bot']:.2f},{s['level_top']:.2f}]  formed={s['formed_date']}")
        else:
            print(f"  [{sig_type}] {sym} ({name})  close={close:.2f}  level={s['level']:.2f}  broken={s['broken_date']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMC 信号扫描器")
    parser.add_argument("--us",           action="store_true", help="扫描美股全量")
    parser.add_argument("--hk",           action="store_true", help="扫描港股宇宙池")
    parser.add_argument("--lookback",     type=int, default=3, help="信号有效 K 线根数（默认 3）")
    parser.add_argument("--swing_length", type=int, default=5, help="摆动点灵敏度（默认 5）")
    parser.add_argument("--output",       type=str, default="", help="输出 JSON 路径（可选）")
    args = parser.parse_args()

    if args.us:
        wl = _build_us_universe_watchlist()
        label = "美股全量"
    elif args.hk:
        wl = _build_hk_universe_watchlist()
        label = "港股宇宙池"
    else:
        wl = build_watchlist()
        label = "默认自选列表"

    print(f"\n扫描 {label}：{len(wl)} 只股票，lookback={args.lookback}，swing_length={args.swing_length}")
    signals = run_scan(watchlist=wl, lookback=args.lookback, swing_length=args.swing_length)

    # 按信号类型分组打印
    for sig_type in (
        "CHOCH_BULLISH", "BOS_BULLISH", "FVG_BULLISH", "OB_BULLISH",
        "CHOCH_BEARISH", "BOS_BEARISH", "FVG_BEARISH", "OB_BEARISH",
    ):
        grp = [s for s in signals if s["type"] == sig_type]
        print(f"\n=== {sig_type}: {len(grp)} 个 ===")
        _print_signals(grp)

    # 可选 JSON 输出
    if args.output:
        out_path = Path(args.output)
    else:
        SCAN_OUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = SCAN_OUT_DIR / f"smc_scan_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(signals, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")
