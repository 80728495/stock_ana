#!/usr/bin/env python3
"""
Vegas Mid-Vegas 回踩每日扫描器

扫描给定股票列表，判断最新交易日是否触发 Mid Vegas 回踩买入信号。
输出 JSON 文件（含信号参数 + 图表路径），供 OpenClaw 机器人邮件发送。

用法:
    python -m stock_ana.scan.vegas_mid_scan                # 扫描默认列表
    python -m stock_ana.scan.vegas_mid_scan --update       # 先更新数据再扫描
    python -m stock_ana.scan.vegas_mid_scan --lookback 3   # 放宽到最近 3 个交易日
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
from stock_ana.strategies.primitives.wave import analyze_wave_structure
from stock_ana.strategies.impl.vegas_mid import (
    MID_EMAS,
    LONG_EMAS,
    compute_vegas_emas as _compute_all_emas,
    check_mid_vegas_structure as _check_structure,
    find_wave_context as _find_wave_context,
    backward_consecutive_count,
    detect_mid_touch_and_hold,
    detect_mid_touch_immediate,
    check_orderly_pullback,
    score_pullback,
    classify_signal,
)

from stock_ana.utils.plot_renderers import plot_vegas_mid_scan_chart
from stock_ana.data.market_data import build_watchlist

SCAN_OUT_DIR = OUTPUT_DIR / "vegas_scan"


# ─────────────────── 港股大市值列表构建 ───────────────────
# NOTE: This is the only market-specific builder retained because it reads from
# a curated CSV (hk_main_largecap_list.csv) with name metadata — the generic
# build_watchlist() covers all cache-dir–based sources automatically.

def _build_hk_largecap_watchlist() -> dict:
    """港股大市值列表（from hk_main_largecap_list.csv）。"""
    csv_path = OUTPUT_DIR.parent / "hk_main_largecap_list.csv"
    if not csv_path.exists():
        logger.warning(f"未找到港股列表: {csv_path}，回退到 Shawn HK 列表")
        return {k: v for k, v in build_watchlist().items() if v[0] == "HK"}

    df = pd.read_csv(csv_path, dtype={"code": str})
    df["code"] = df["code"].str.zfill(5)
    watchlist = {}
    for _, row in df.iterrows():
        code = row["code"]
        try:
            if int(code) >= 10000:
                continue
        except ValueError:
            continue
        path = CACHE_DIR / "hk" / f"{code}.parquet"
        if not path.exists():
            continue
        name_zh = str(row.get("name_zh") or "").strip()
        name_en = str(row.get("name_en") or "").strip()
        name = name_zh or name_en or code
        watchlist[code] = ("HK", name, path, name_en)
    logger.info(f"港股大市值列表：共 {len(watchlist)} 只有缓存数据的标的")
    return watchlist


# ─────────────────── 美股宇宙池列表构建 ───────────────────
# Kept because it reads from us_universe.csv (company names, filters).

def _build_us_universe_watchlist() -> dict:
    """美股科技/通信板块列表（from us_tech_list.md），附带行业和主营业务信息。

    列表来源：data/lists/us_tech_list.md（Technology + Communication Services，可手动增删）。
    """
    from stock_ana.data.list_manager import load_us_tech_list
    entries = load_us_tech_list()  # [{ticker, company, sector, market_cap_b}]

    # 加载 SEC profiles（行业 + 主营业务摘要）
    _sec_profiles: dict[str, dict] = {}
    _sec_profile_path = Path("data/us_sec_profiles.csv")
    if _sec_profile_path.exists():
        try:
            _sp = pd.read_csv(_sec_profile_path, dtype=str)
            _sp.columns = [c.lstrip("\ufeff").strip() for c in _sp.columns]
            for _, r in _sp.iterrows():
                t = str(r.get("ticker", "")).strip().upper()
                if t:
                    _sec_profiles[t] = {
                        "sector": str(r.get("sector", "")).strip(),
                        "industry": str(r.get("sic_description", "")).strip(),
                        "business_summary": str(r.get("business_summary", "")).strip(),
                    }
        except Exception as e:
            logger.warning(f"加载 SEC profiles 失败: {e}")

    watchlist = {}
    for entry in entries:
        ticker = entry["ticker"]
        if not ticker:
            continue

        # 数据统一存储在 cache/us/，ndx100 只是筛选列表，不再是独立数据源
        path = CACHE_DIR / "us" / f"{ticker}.parquet"
        if not path.exists():
            continue

        name = entry.get("company") or ticker
        sec = _sec_profiles.get(ticker, {})
        sector   = sec.get("sector") or entry.get("sector", "")
        industry = sec.get("industry", "")
        biz_summary = sec.get("business_summary", "")
        watchlist[ticker] = ("US", name, path, "", sector, industry, biz_summary)
    logger.info(f"美股列表：共 {len(watchlist)} 只有缓存数据的标的")
    return watchlist


def _build_us_full_watchlist() -> dict:
    """美股全量列表（from us_full_list.md），附带行业和主营业务信息。

    列表来源：data/lists/us_full_list.md（全量 ~1550 只，含所有行业）。
    """
    from stock_ana.data.list_manager import load_us_full_list
    entries = load_us_full_list()  # [{ticker, company, sector, market_cap_b}]

    # 加载 SEC profiles（行业 + 主营业务摘要）
    _sec_profiles: dict[str, dict] = {}
    _sec_profile_path = Path("data/us_sec_profiles.csv")
    if _sec_profile_path.exists():
        try:
            _sp = pd.read_csv(_sec_profile_path, dtype=str)
            _sp.columns = [c.lstrip("\ufeff").strip() for c in _sp.columns]
            for _, r in _sp.iterrows():
                t = str(r.get("ticker", "")).strip().upper()
                if t:
                    _sec_profiles[t] = {
                        "sector": str(r.get("sector", "")).strip(),
                        "industry": str(r.get("sic_description", "")).strip(),
                        "business_summary": str(r.get("business_summary", "")).strip(),
                    }
        except Exception as e:
            logger.warning(f"加载 SEC profiles 失败: {e}")

    watchlist = {}
    for entry in entries:
        ticker = entry["ticker"]
        if not ticker:
            continue

        path = CACHE_DIR / "us" / f"{ticker}.parquet"
        if not path.exists():
            continue

        name = entry.get("company") or ticker
        sec = _sec_profiles.get(ticker, {})
        sector   = sec.get("sector") or entry.get("sector", "")
        industry = sec.get("industry", "")
        biz_summary = sec.get("business_summary", "")
        watchlist[ticker] = ("US", name, path, "", sector, industry, biz_summary)
    logger.info(f"美股全量列表：共 {len(watchlist)} 只有缓存数据的标的")
    return watchlist


# ═══════════════════════════════════════════════════════
#  内部工具：信号注释处理（局灯hold和touch两种策略共用）
# ═══════════════════════════════════════════════════════

def _came_from_above(touch_bar: int, close: np.ndarray, all_pivots: list[dict], recent_bars: int = 25) -> bool:
    """判断股价在 touch_bar 时是从 vegas mid 上方回落而来（真正的回踩）。

    **策略**：
    1. 优先在 `recent_bars` 根K线内查找最近的 ZigZag 结构点，用其高低来判断方向。
    2. 若附近没有结构点（ZigZag 把一段连续上涨视为单波浪，没有中间节点），
       则退回到检查近期最高收盘价：
       - 若最近 recent_bars 根收盘最高值 > close[touch_bar] * 1.01
         → 价格从更高水平回落，属于真实回踩 → True
       - 否则 → 大概率从下方穿越 → False
    3. 历史数据不足时默认放行（True）。

    避免旧逻辑的缺陷：以距离极远的结构低点代入判断，导致整段升浪中的
    每次 Mid Vegas 回踩都被误判为"从下方穿越"而被过滤。
    """
    c_now = float(close[touch_bar])

    # 1. 优先查找附近的结构点
    if all_pivots:
        nearby = [p for p in all_pivots
                  if p["iloc"] < touch_bar and touch_bar - p["iloc"] <= recent_bars]
        if nearby:
            last_pivot = max(nearby, key=lambda p: p["iloc"])
            return last_pivot["value"] > c_now

    # 2. 无附近结构点 → 用最近 recent_bars 根收盘最高价判断
    start = max(0, touch_bar - recent_bars)
    if start < touch_bar:
        return float(np.max(close[start:touch_bar])) > c_now * 1.01

    return True  # 数据不足，默认放行


def _process_touch_signals(
    raw_signals: list[dict],
    close: np.ndarray,
    emas: dict,
    waves: list,
    all_pivots: list[dict],
    dates: pd.DatetimeIndex,
    market: str,
    sym: str,
    name: str,
    cutoff_bar: int,
    n: int,
    touch_strategy: str,
) -> list[dict]:
    """将原始信号列表经过波浪上下文注释、结构检查、打分，返回完整信号字典列表。

    hold策略：entry_bar 在確认日（T+1/T+2）。
    touch策略：entry_bar == touch_bar，当日即出。
    """
    wave_touch_counter: dict[int, int] = {}
    prev_touch_bar: int = -1
    results: list[dict] = []

    for sig in raw_signals:
        entry_bar = sig["entry_bar"]
        curr_touch_bar = sig["touch_bar"]

        # 方向过滤：排除从 vegas mid 下方上穿的假回踩
        if not _came_from_above(curr_touch_bar, close, all_pivots):
            continue

        wave_ctx = _find_wave_context(waves, curr_touch_bar) if waves else None
        wave_number = wave_ctx["wave_number"] if wave_ctx else 0

        wave_touch_counter[wave_number] = wave_touch_counter.get(wave_number, 0) + 1
        touch_seq = wave_touch_counter[wave_number]

        orderly = check_orderly_pullback(close, emas, prev_touch_bar, curr_touch_bar)
        prev_touch_bar = curr_touch_bar

        if entry_bar < cutoff_bar:
            continue

        if entry_bar >= n:
            confirm_bar = sig["confirm_bar"]
            entry_price = float(close[confirm_bar])
            struct = _check_structure(confirm_bar, close, emas)
            entry_date = str(dates[confirm_bar].date()) + "(T+1)"
        else:
            entry_price = float(close[entry_bar])
            struct = _check_structure(entry_bar, close, emas)
            entry_date = str(dates[entry_bar].date())
        structure_passed = struct["passed"]

        wave_rise_so_far = 0.0
        consec_count = 0
        sub_number = 0

        if wave_ctx:
            sub_count = sum(
                1 for sw in wave_ctx.get("sub_waves", [])
                if sw.get("end_pivot") and sw["end_pivot"]["iloc"] <= curr_touch_bar
            )
            sub_number = sub_count + 1

            sp_val = wave_ctx["start_pivot"]["value"]
            if sp_val > 0:
                look_start = wave_ctx["start_pivot"]["iloc"]
                look_end = min(curr_touch_bar + 1, n)
                if look_start < look_end:
                    peak_val = float(max(close[look_start:look_end]))
                    wave_rise_so_far = (peak_val / sp_val - 1) * 100

            consec_count = backward_consecutive_count(waves, wave_number)

        touch_seq_ok = touch_seq <= 3

        score, score_details = score_pullback(
            sub_number=sub_number,
            wave_rise_pct=wave_rise_so_far,
            wave_number=wave_number,
            market=market,
            consecutive_wave_count=consec_count,
            mid_long_gap_pct=struct["mid_long_gap_pct"],
            orderly_pullback=orderly,
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
            "mid_above_long": struct["mid_above_long"],
            "price_above_long": struct["price_above_long"],
            "price_above_long_3m": struct["price_above_long_3m"],
            "long_rising": struct["long_rising"],
            # 辅助参考条件
            "gap_enough": struct["gap_enough"],
            "long_slope_strong": struct["long_slope_strong"],
            "touch_seq": touch_seq if waves and wave_ctx else 0,
            "touch_seq_ok": touch_seq_ok if waves and wave_ctx else True,
            # 波浪上下文
            "wave_number": wave_number,
            "sub_number": sub_number,
            "consec_waves": consec_count,
            "wave_rise_pct": round(wave_rise_so_far, 2),
            "mid_long_gap_pct": struct["mid_long_gap_pct"],
            "long_slope_pct": struct["long_slope_pct"],
            "orderly_pullback": orderly,
            # 因子明细
            **{f"factor_{k}": v for k, v in score_details.items()},
        })

    return results


# ═══════════════════════════════════════════════════════
#  单只股票扫描：获取最新信号
# ═══════════════════════════════════════════════════════

def scan_one(
    sym: str,
    market: str,
    name: str,
    df: pd.DataFrame,
    lookback: int = 1,
) -> list[dict]:
    """
    对一只股票做实时信号检测，返回最近 lookback 个交易日内的信号。

    返回同时包含两种策略的信号：
      touch_strategy="hold" : 触碰 + 站稳确认，在 T+1/T+2 入场（原策略）
      touch_strategy="touch": 触碰即出，当日即为入场信号，无需站稳确认
    """
    x = df.copy()
    x.columns = [c.lower() for c in x.columns]
    x.index = pd.to_datetime(x.index)
    x = x.sort_index()
    if len(x) < 200:
        return []

    close = x["close"].astype(float).values
    low_arr = x["low"].astype(float).values
    close_s = x["close"].astype(float)
    n = len(x)

    emas = _compute_all_emas(close_s)

    result = analyze_wave_structure(df)
    waves = result["major_waves"]
    all_pivots = result.get("all_pivots", [])

    cutoff_bar = n - lookback
    common = dict(
        close=close, emas=emas, waves=waves,
        all_pivots=all_pivots,
        dates=x.index, market=market, sym=sym, name=name,
        cutoff_bar=cutoff_bar, n=n,
    )

    hold_signals = _process_touch_signals(
        detect_mid_touch_and_hold(close, low_arr, emas),
        touch_strategy="hold",
        **common,
    )
    touch_signals = _process_touch_signals(
        detect_mid_touch_immediate(close, low_arr, emas),
        touch_strategy="touch",
        **common,
    )

    return hold_signals + touch_signals


def generate_signal_chart(
    sym: str,
    market: str,
    name: str,
    df: pd.DataFrame,
    sig: dict,
    out_dir: Path,
    name_en: str = "",
    sector: str = "",
    industry: str = "",
    biz_summary: str = "",
) -> Path | None:
    """Render a focused Vegas Mid-Vegas scan signal chart.

    Thin wrapper around plot_vegas_mid_scan_chart so that run_scan can call it
    without importing plot_renderers directly.
    """
    return plot_vegas_mid_scan_chart(
        sym=sym,
        market=market,
        name=name,
        df_price=df,
        signal_info=sig,
        out_dir=out_dir,
        name_en=name_en,
        context_bars=252,
        sector=sector,
        industry=industry,
        biz_summary=biz_summary,
    )


# ═══════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════

def run_scan(
    watchlist: dict | None = None,
    lookback: int = 1,
    min_signal: str = "BUY",
    update_data: bool = False,
    touch_only: bool = False,
) -> list[dict]:
    """
    扫描所有标的，输出最近 lookback 天内触发的 BUY / STRONG_BUY 信号。

    Args:
        watchlist: 股票列表 dict，格式同 build_shawn_watchlist() 返回值。None 则用 Shawn 自选列表。
        lookback: 检查最近几个交易日（默认 1 = 仅当天）。
        min_signal: 最低信号等级  "STRONG_BUY" / "BUY" / "HOLD"
        update_data: 是否先更新行情数据。

    Returns:
        信号列表，每个元素是带完整参数的 dict（含 chart_path）。
    """
    if watchlist is None:
        watchlist = build_watchlist()  # Shawn list (all markets)

    signal_rank = {"STRONG_BUY": 4, "BUY": 3, "HOLD": 2, "AVOID": 1}
    min_rank = signal_rank.get(min_signal, 3)

    if update_data:
        logger.info("更新行情数据 ...")
        from stock_ana.data.fetcher_hk import update_hk_data
        from stock_ana.data.fetcher import update_us_price_data
        from stock_ana.data.fetcher_cn import update_cn_data
        hk_syms = [s for s, tup in watchlist.items() if tup[0] == "HK"]
        us_syms = [s for s, tup in watchlist.items() if tup[0] == "US"]
        cn_syms = [s for s, tup in watchlist.items() if tup[0] == "CN"]
        if us_syms:
            update_us_price_data(tickers=us_syms)
        if hk_syms:
            update_hk_data()
        if cn_syms:
            update_cn_data(codes=cn_syms)
        logger.info("数据更新完成")

    today = datetime.now().strftime("%Y-%m-%d")
    scan_out = SCAN_OUT_DIR / today
    scan_out.mkdir(parents=True, exist_ok=True)

    all_signals: list[dict] = []

    for sym, tup in watchlist.items():
        mkt, name, path = tup[0], tup[1], tup[2]
        name_en     = tup[3] if len(tup) > 3 else ""
        sector      = tup[4] if len(tup) > 4 else ""
        industry    = tup[5] if len(tup) > 5 else ""
        biz_summary = tup[6] if len(tup) > 6 else ""
        if not path.exists():
            logger.warning(f"{mkt}:{sym} — 缺少数据: {path}")
            continue

        df = pd.read_parquet(path)
        signals = scan_one(sym, mkt, name, df, lookback=lookback)

        for sig in signals:
            # touch_only 模式：只保留 touch 策略信号
            if touch_only and sig.get("touch_strategy") != "touch":
                continue
            # touch 策略不过滤信号等级，hold 策略按 min_rank 过滤
            if sig.get("touch_strategy") != "touch":
                rank = signal_rank.get(sig["signal"], 0)
                if rank < min_rank:
                    continue

            # AVOID 信号：只保留结构通过但评分不足的（structure_passed=True, score<0）
            # 结构本身不过关的直接丢弃，不出图表报告
            if sig.get("signal") == "AVOID" and not sig.get("structure_passed", False):
                continue

            # 生成聚焦图表
            chart_path = generate_signal_chart(
                sym, mkt, name, df, sig, out_dir=scan_out, name_en=name_en,
                sector=sector, industry=industry, biz_summary=biz_summary,
            )
            sig["chart_path"] = str(chart_path) if chart_path else None

            # base64 编码图表（方便 JSON 传输）
            if chart_path and chart_path.exists():
                with open(chart_path, "rb") as f:
                    sig["chart_base64"] = base64.b64encode(f.read()).decode("ascii")
            else:
                sig["chart_base64"] = None

            all_signals.append(sig)
            logger.success(
                f"{mkt}:{sym} {name} — {sig['signal']} "
                f"(score={sig['score']:+d}) @ {sig['entry_date']}"
            )

    # 按分数降序排列
    all_signals.sort(key=lambda s: s["score"], reverse=True)

    # 输出 JSON
    json_path = scan_out / "signals.json"
    # JSON 中不包含 base64 数据（太大），单独输出
    json_signals = []
    for sig in all_signals:
        sig_copy = {k: v for k, v in sig.items() if k != "chart_base64"}
        # 确保 bool 类型正确序列化
        for k in ["structure_passed", "mid_above_long", "price_above_long",
                   "price_above_long_3m",
                   "long_rising", "gap_enough", "long_slope_strong", "touch_seq_ok"]:
            if k in sig_copy:
                sig_copy[k] = bool(sig_copy[k])
        json_signals.append(sig_copy)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "scan_date": today,
            "lookback_days": lookback,
            "min_signal": min_signal,
            "total_scanned": len(watchlist),
            "signals_found": len(all_signals),
            "signals": json_signals,
        }, f, ensure_ascii=False, indent=2)

    # 同时输出含 base64 的完整版（供 OpenClaw 邮件直接读取）
    json_full_path = scan_out / "signals_full.json"
    json_full_signals = []
    for sig in all_signals:
        sig_copy = dict(sig)
        for k in ["structure_passed", "mid_above_long", "price_above_long",
                   "price_above_long_3m",
                   "long_rising", "gap_enough", "long_slope_strong", "touch_seq_ok"]:
            if k in sig_copy:
                sig_copy[k] = bool(sig_copy[k])
        json_full_signals.append(sig_copy)

    with open(json_full_path, "w", encoding="utf-8") as f:
        json.dump({
            "scan_date": today,
            "lookback_days": lookback,
            "min_signal": min_signal,
            "total_scanned": len(watchlist),
            "signals_found": len(all_signals),
            "signals": json_full_signals,
        }, f, ensure_ascii=False, indent=2)

    # 摘要
    if all_signals:
        logger.info(f"\n{'='*60}")
        logger.info(f"扫描完成 — {today}")
        logger.info(f"扫描标的: {len(watchlist)} 只")
        logger.info(f"触发信号: {len(all_signals)} 个")
        for sig in all_signals:
            helpers = (
                f"Gap:{'Y' if sig['gap_enough'] else 'N'} "
                f"Slp:{'Y' if sig['long_slope_strong'] else 'N'} "
                f"Seq:{'Y' if sig['touch_seq_ok'] else 'N'}"
            )
            logger.info(
                f"  {sig['market']}:{sig['symbol']:>6s} {sig['name']:<8s} "
                f"{sig['signal']:<12s} score={sig['score']:+d}  "
                f"{sig['support_band']}  {helpers}"
            )
        logger.info(f"输出: {json_path}")
        logger.info(f"{'='*60}")
    else:
        logger.info(f"扫描完成 — {today} — 无信号触发")

    return all_signals


def main():
    """Run the daily Vegas touch scan CLI for the selected markets and signal floor."""
    parser = argparse.ArgumentParser(description="Vegas Mid-Vegas 回踩每日扫描")
    parser.add_argument("--update", action="store_true", help="先更新行情数据")
    parser.add_argument("--lookback", type=int, default=1,
                        help="检查最近几个交易日 (默认 1)")
    parser.add_argument("--min-signal", default="BUY",
                        choices=["STRONG_BUY", "BUY", "HOLD"],
                        help="最低信号等级 (默认 BUY)")
    parser.add_argument("--shawn", action="store_true",
                        help="扫描 Shawn 自选列表（data/lists/shawn_list.md，默认）")
    parser.add_argument("--hk", action="store_true",
                        help="扫描港股大市值列表（data/hk_main_largecap_list.csv）")
    parser.add_argument("--us", action="store_true",
                        help="扫描美股全量列表（data/us_universe.csv）")
    parser.add_argument("--ndx100", action="store_true",
                        help="扫描纳指 100（data/cache/ndx100/）")
    parser.add_argument("--cn", action="store_true",
                        help="扫描大A自选列表（data/cache/cn/）")
    parser.add_argument("--touch-only", action="store_true",
                        help="只输出 touch 策略信号（直接触及 mid vegas 线，不等待确认）")
    parser.add_argument("--update-universe", action="store_true",
                        help="强制从 Finviz 刷新美股列表")
    args = parser.parse_args()

    if args.update_universe:
        from stock_ana.data.us_universe_builder import build_us_stock_universe
        logger.info("正在从 Finviz 刷新美股列表 ...")
        build_us_stock_universe(force=True)

    # 构建合并的 watchlist（允许多个 flag 同时生效）
    combined: dict = {}
    any_explicit = args.shawn or args.hk or args.ndx100 or args.cn or args.us

    if args.shawn or not any_explicit:
        combined.update(build_watchlist())  # Shawn list (all markets)
    if args.ndx100:
        combined.update(build_watchlist(["ndx100"]))
    if args.hk:
        combined.update(_build_hk_largecap_watchlist())
    if args.us:
        combined.update(_build_us_universe_watchlist())
    if args.cn:
        combined.update(build_watchlist(["cn"]))

    watchlist = combined if combined else None

    signals = run_scan(
        watchlist=watchlist,
        lookback=args.lookback,
        min_signal=args.min_signal,
        update_data=args.update,
        touch_only=args.touch_only,
    )
    return signals


if __name__ == "__main__":
    main()
