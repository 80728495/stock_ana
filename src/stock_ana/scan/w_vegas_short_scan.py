#!/usr/bin/env python3
"""
Weekly Vegas Short (w_ema_8/21) 回踩周线扫描器

扫描逻辑与 Vegas Mid 日线扫描（vegas_mid_scan.py）对称，
但基于预计算的周线 OHLCV（data/cache/indicators/{market}/{symbol}_w.parquet）。

通道映射:
  Short Vegas  (回踩目标) : w_ema_8 / w_ema_21
  Mid Vegas    (趋势守卫) : w_ema_34 / w_ema_55
  Long Vegas   (不使用)   : —

结构条件: price > short_upper > mid_upper，mid 斜率 > 0，价格持续在 mid 以上

用法:
    python -m stock_ana.scan.w_vegas_short_scan               # 扫描默认自选列表
    python -m stock_ana.scan.w_vegas_short_scan --us          # 扫描美股全量
    python -m stock_ana.scan.w_vegas_short_scan --hk          # 扫描港股宇宙池
    python -m stock_ana.scan.w_vegas_short_scan --lookback 4  # 放宽到最近 4 周
"""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.config import CACHE_DIR, OUTPUT_DIR
from stock_ana.data.indicators_store import load_weekly_indicators
from stock_ana.strategies.impl.w_vegas_short import (
    SHORT_EMAS,
    MID_EMAS_W,
    compute_w_short_emas,
    detect_w_short_touch_and_hold,
    detect_w_short_touch_immediate,
    check_w_short_structure,
    score_w_pullback,
    classify_signal,
)
from stock_ana.data.market_data import build_watchlist

SCAN_OUT_DIR = OUTPUT_DIR / "w_vegas_scan"

# 指标市场目录（与 indicators_store 保持一致，小写）
_MARKET_MAP = {"US": "us", "HK": "hk", "CN": "cn"}


# ─────────────────────── 方向过滤 ───────────────────────

def _came_from_above(touch_bar: int, close: np.ndarray, recent_bars: int = 12) -> bool:
    """判断价格在 touch_bar 时是从 short Vegas 上方回落（真正回踩）。

    周线版：无波浪结构点可用，直接用近期最高收盘判断方向。
    12 周 ≈ 3 个月，足以确认方向。
    """
    start = max(0, touch_bar - recent_bars)
    if start >= touch_bar:
        return True  # 数据不足，默认放行
    recent_high = float(np.max(close[start:touch_bar]))
    return recent_high > float(close[touch_bar]) * 1.01


# ─────────────────────── watchlist 构建 ───────────────────────

def _build_hk_universe_watchlist() -> dict:
    """港股宇宙池（hk_universe_list.md，市值≥100亿）。"""
    from stock_ana.data.list_manager import _read_md_table
    from stock_ana.config import DATA_DIR
    path = DATA_DIR / "lists" / "hk_universe_list.md"
    if not path.exists():
        logger.warning(f"未找到港股宇宙池列表: {path}")
        return {}
    rows = _read_md_table(path)
    watchlist = {}
    for r in rows:
        if len(r) < 3:
            continue
        code = r[1].strip().zfill(5)
        name_zh = r[2].strip() or code
        ohlcv_path = CACHE_DIR / "hk" / f"{code}.parquet"
        if not ohlcv_path.exists():
            continue
        watchlist[code] = ("HK", name_zh, ohlcv_path, "")
    logger.info(f"港股宇宙池：{len(watchlist)} 只有数据")
    return watchlist


def _build_us_full_watchlist() -> dict:
    """美股全量列表（us_full_list.md）。"""
    from stock_ana.data.list_manager import load_us_full_list
    entries = load_us_full_list()
    watchlist = {}
    for entry in entries:
        ticker = entry["ticker"]
        if not ticker:
            continue
        path = CACHE_DIR / "us" / f"{ticker}.parquet"
        if not path.exists():
            continue
        name = entry.get("company") or ticker
        watchlist[ticker] = ("US", name, path, "")
    logger.info(f"美股全量：{len(watchlist)} 只有数据")
    return watchlist


def _build_us_universe_watchlist() -> dict:
    """美股科技/通信板块列表（us_tech_list.md）。"""
    from stock_ana.data.list_manager import load_us_tech_list
    entries = load_us_tech_list()
    watchlist = {}
    for entry in entries:
        ticker = entry["ticker"]
        if not ticker:
            continue
        path = CACHE_DIR / "us" / f"{ticker}.parquet"
        if not path.exists():
            continue
        name = entry.get("company") or ticker
        watchlist[ticker] = ("US", name, path, "")
    logger.info(f"美股科技列表：{len(watchlist)} 只有数据")
    return watchlist


# ─────────────────────── 单只股票扫描 ───────────────────────

def scan_one_weekly(
    sym: str,
    market: str,
    name: str,
    df_weekly: pd.DataFrame,
    lookback: int = 1,
) -> list[dict]:
    """对一只股票的周线 DataFrame 扫描最近 lookback 周内的信号。

    返回同时包含两种策略信号：
      touch_strategy="hold"  : 触碰 + 站稳确认（T+1 周入场）
      touch_strategy="touch" : 触碰即出（当周即为信号）
    """
    x = df_weekly.copy()
    x.columns = [c.lower() for c in x.columns]
    x.index = pd.to_datetime(x.index)
    x = x.sort_index()
    if len(x) < 30:
        return []

    close = x["close"].astype(float).values
    low_arr = x["low"].astype(float).values
    close_s = x["close"].astype(float)
    n = len(x)
    dates = x.index

    emas = compute_w_short_emas(close_s)
    cutoff_bar = n - lookback

    hold_raws = detect_w_short_touch_and_hold(close, low_arr, emas)
    touch_raws = detect_w_short_touch_immediate(close, low_arr, emas)

    results: list[dict] = []

    for touch_strategy, raw_signals in [("hold", hold_raws), ("touch", touch_raws)]:
        for sig in raw_signals:
            curr_touch_bar = sig["touch_bar"]
            entry_bar = sig["entry_bar"]

            # 方向过滤
            if not _came_from_above(curr_touch_bar, close):
                continue

            if entry_bar < cutoff_bar:
                continue

            # 入场价 / 日期
            if entry_bar >= n:
                confirm_bar = sig["confirm_bar"]
                entry_price = float(close[confirm_bar])
                entry_date = str(dates[confirm_bar].date()) + "(T+1w)"
                check_bar = confirm_bar
            else:
                entry_price = float(close[entry_bar])
                entry_date = str(dates[entry_bar].date())
                check_bar = entry_bar

            struct = check_w_short_structure(check_bar, close, emas)
            structure_passed = struct["passed"]

            score, score_details = score_w_pullback(
                market=market,
                mid_slope_pct=struct["mid_slope_pct"],
                short_mid_gap_pct=struct["short_mid_gap_pct"],
            )
            signal = classify_signal(score) if structure_passed else "AVOID"

            results.append({
                "symbol": sym,
                "market": market,
                "name": name,
                "entry_date": entry_date,
                "entry_price": round(entry_price, 3),
                "support_band": sig["support_band"],
                "signal": signal,
                "score": score,
                "touch_strategy": touch_strategy,
                # 结构条件
                "structure_passed": structure_passed,
                "short_above_mid": struct["short_above_mid"],
                "price_above_mid": struct["price_above_mid"],
                "mid_rising": struct["mid_rising"],
                "price_above_mid_nw": struct["price_above_mid_nw"],
                # 辅助参考
                "gap_enough": struct["gap_enough"],
                "mid_slope_strong": struct["mid_slope_strong"],
                # 量化指标
                "short_mid_gap_pct": struct["short_mid_gap_pct"],
                "mid_slope_pct": struct["mid_slope_pct"],
                # 因子明细
                **{f"factor_{k}": v for k, v in score_details.items()},
            })

    return results


def generate_signal_chart(
    sym: str,
    market: str,
    name: str,
    df_weekly: pd.DataFrame,
    sig: dict,
    out_dir: Path,
    name_en: str = "",
) -> Path | None:
    """渲染周线 Short Vegas 扫描信号图表。"""
    from stock_ana.utils.plot_renderers import plot_w_vegas_short_chart
    return plot_w_vegas_short_chart(
        sym=sym,
        market=market,
        name=name,
        df_weekly=df_weekly,
        signal_info=sig,
        out_dir=out_dir,
        name_en=name_en,
        context_bars=104,  # 约 2 年周线
    )


# ─────────────────────── 主扫描流程 ───────────────────────

def run_scan(
    watchlist: dict | None = None,
    lookback: int = 1,
    min_signal: str = "BUY",
    touch_only: bool = False,
) -> list[dict]:
    """扫描所有标的，输出最近 lookback 周内触发的信号。

    Args:
        watchlist : 股票列表 dict；None 则用 Shawn 自选列表。
        lookback  : 检查最近几周（默认 1 = 仅当周）。
        min_signal: 最低信号等级 "STRONG_BUY" / "BUY" / "HOLD"。
        touch_only: True 时只输出 touch 策略信号（触碰即出），与日线扫描行为一致。

    Returns:
        信号列表，每个元素带完整参数（含 chart_path）。
    """
    if watchlist is None:
        watchlist = build_watchlist()

    signal_rank = {"STRONG_BUY": 4, "BUY": 3, "HOLD": 2, "AVOID": 1}
    min_rank = signal_rank.get(min_signal, 3)

    today = datetime.now().strftime("%Y-%m-%d")
    scan_out = SCAN_OUT_DIR / today
    scan_out.mkdir(parents=True, exist_ok=True)

    all_signals: list[dict] = []

    for sym, tup in watchlist.items():
        mkt, name = tup[0], tup[1]
        ohlcv_path: Path = tup[2]
        name_en = tup[3] if len(tup) > 3 else ""

        # 推导指标市场目录（从 OHLCV 路径取父目录名）
        ind_market = ohlcv_path.parent.name  # "us", "hk", "ndx100", "cn"

        df_weekly = load_weekly_indicators(sym, ind_market)
        if df_weekly is None or len(df_weekly) < 30:
            continue

        signals = scan_one_weekly(sym, mkt, name, df_weekly, lookback=lookback)

        for sig in signals:
            # touch_only 模式：只保留 touch 策略信号
            if touch_only and sig.get("touch_strategy") != "touch":
                continue

            # touch 策略不过滤信号等级；hold 策略按 min_rank 过滤
            if sig.get("touch_strategy") != "touch":
                rank = signal_rank.get(sig["signal"], 0)
                if rank < min_rank:
                    continue

            # AVOID：只保留结构通过但评分不足的
            if sig.get("signal") == "AVOID" and not sig.get("structure_passed", False):
                continue

            chart_path = generate_signal_chart(
                sym, mkt, name, df_weekly, sig, out_dir=scan_out, name_en=name_en,
            )
            sig["chart_path"] = str(chart_path) if chart_path else None

            if chart_path and chart_path.exists():
                with open(chart_path, "rb") as f:
                    sig["chart_base64"] = base64.b64encode(f.read()).decode("ascii")
            else:
                sig["chart_base64"] = None

            all_signals.append(sig)
            logger.success(
                f"{mkt}:{sym} {name} — {sig['signal']} "
                f"(score={sig['score']:+d}) @ {sig['entry_date']} [{sig['support_band']}]"
            )

    all_signals.sort(key=lambda s: s["score"], reverse=True)

    # ── 输出 JSON ────────────────────────────────────────
    def _dump(path: Path, signals: list[dict], include_b64: bool) -> None:
        out = []
        for sig in signals:
            sc = dict(sig)
            if not include_b64:
                sc.pop("chart_base64", None)
            for k in ["structure_passed", "short_above_mid", "price_above_mid",
                      "mid_rising", "price_above_mid_nw",
                      "gap_enough", "mid_slope_strong"]:
                if k in sc:
                    sc[k] = bool(sc[k])
            out.append(sc)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "scan_date": today,
                "lookback_weeks": lookback,
                "min_signal": min_signal,
                "total_scanned": len(watchlist),
                "signals_found": len(signals),
                "signals": out,
            }, f, ensure_ascii=False, indent=2)

    _dump(scan_out / "signals.json", all_signals, include_b64=False)
    _dump(scan_out / "signals_full.json", all_signals, include_b64=True)

    if all_signals:
        logger.info(f"\n{'='*60}")
        logger.info(f"周线扫描完成 — {today}")
        logger.info(f"扫描标的: {len(watchlist)} 只 | 信号: {len(all_signals)} 个")
        for sig in all_signals:
            logger.info(
                f"  {sig['market']}:{sig['symbol']:>6s} {sig['name']:<8s} "
                f"{sig['signal']:<12s} score={sig['score']:+d}  "
                f"{sig['support_band']}  slope={sig['mid_slope_pct']:.1f}%"
            )
        logger.info(f"输出: {scan_out / 'signals.json'}")
        logger.info(f"{'='*60}")
    else:
        logger.info(f"周线扫描完成 — {today} — 无信号触发")

    return all_signals


def main():
    """Weekly Vegas Short 回踩扫描 CLI。"""
    parser = argparse.ArgumentParser(description="Weekly Vegas Short 回踩周线扫描")
    parser.add_argument("--lookback", type=int, default=1,
                        help="检查最近几周 (默认 1)")
    parser.add_argument("--min-signal", default="BUY",
                        choices=["STRONG_BUY", "BUY", "HOLD"],
                        help="最低信号等级 (默认 BUY)")
    parser.add_argument("--shawn", action="store_true",
                        help="扫描关注列表 (watchlist.md，默认)")
    parser.add_argument("--hk", action="store_true",
                        help="扫描港股宇宙池 (hk_universe_list.md)")
    parser.add_argument("--us", action="store_true",
                        help="扫描美股科技列表 (us_tech_list.md)")
    parser.add_argument("--us-full", action="store_true",
                        help="扫描美股全量列表 (us_full_list.md)")
    parser.add_argument("--touch-only", action="store_true",
                        help="只输出 touch 策略信号（触碰即出，不等站稳确认）")
    args = parser.parse_args()

    combined: dict = {}
    any_explicit = args.shawn or args.hk or args.us or args.us_full

    if args.shawn or not any_explicit:
        combined.update(build_watchlist())
    if args.hk:
        combined.update(_build_hk_universe_watchlist())
    if args.us:
        combined.update(_build_us_universe_watchlist())
    if args.us_full:
        combined.update(_build_us_full_watchlist())

    signals = run_scan(
        watchlist=combined if combined else None,
        lookback=args.lookback,
        min_signal=args.min_signal,
        touch_only=args.touch_only,
    )
    return signals


if __name__ == "__main__":
    main()
