#!/usr/bin/env python3
"""
上升收敛形态 专项回测脚本
========================

遍历每只股票的完整历史数据，在每个时间点检测上升收敛形态
（上升三角形 / 上升楔形 / 高位旗型），记录形态信息并计算后续收益。

用法：
    python -m stock_ana.backtest.backtest_triangle
    python -m stock_ana.backtest.backtest_triangle --step 3
    python -m stock_ana.backtest.backtest_triangle --step 10
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.data.market_data import load_market_data, load_universe_data, load_watchlist_data
from stock_ana.strategies.api import screen_triangle_ascending
from stock_ana.utils.plot_renderers import plot_triangle_backtest_signals
from stock_ana.scan.triangle_scan import scan_historical_triangles

# ──────── 常量 ────────
FORWARD_DAYS = [5, 10, 21, 63]  # 1周 / 2周 / 1月 / 3月
OUTPUT_DIR = Path("data") / "backtest_triangle"

_CN = {
    "ascending_triangle": "上升三角形",
    "rising_wedge": "上升楔形",
}


# ═══════════════════════════════════════════════════════════
# 历史扫描
# ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════
# 统计汇总
# ═══════════════════════════════════════════════════════════
def summarize_signals(signals: list[dict]) -> dict:
    """汇总回测统计。"""
    if not signals:
        return {"total_signals": 0}

    n = len(signals)
    unique = set(s["ticker"] for s in signals)

    summary: dict = {
        "total_signals": n,
        "unique_tickers": len(unique),
        "tickers": sorted(unique),
    }

    # ── 各持有期统计 ──
    for period_key in [f"{d}d" for d in FORWARD_DAYS]:
        rets, dds, gains = [], [], []
        for s in signals:
            info = s.get(period_key)
            if info and "return_pct" in info and "note" not in info:
                rets.append(info["return_pct"])
                if "max_drawdown_pct" in info:
                    dds.append(info["max_drawdown_pct"])
                if "max_gain_pct" in info:
                    gains.append(info["max_gain_pct"])

        if not rets:
            continue

        arr = np.array(rets)
        wins = arr > 0
        stats: dict = {
            "count": len(rets),
            "win_rate": round(float(np.mean(wins)) * 100, 1),
            "avg_return": round(float(np.mean(arr)), 2),
            "median_return": round(float(np.median(arr)), 2),
            "max_gain": round(float(np.max(arr)), 2),
            "max_loss": round(float(np.min(arr)), 2),
            "std": round(float(np.std(arr)), 2),
        }
        if dds:
            stats["avg_max_drawdown"] = round(float(np.mean(dds)), 2)
        if gains:
            stats["avg_max_gain"] = round(float(np.mean(gains)), 2)
        summary[period_key] = stats

    # ── 按形态类型分组 ──
    pat_groups: dict[str, list] = {}
    for s in signals:
        pat_groups.setdefault(s["pattern"], []).append(s)

    pat_summary: dict = {}
    for p, group in sorted(pat_groups.items()):
        rets_21d = []
        for s in group:
            info = s.get("21d")
            if info and "return_pct" in info and "note" not in info:
                rets_21d.append(info["return_pct"])
        if rets_21d:
            arr = np.array(rets_21d)
            pat_summary[p] = {
                "count": len(group),
                "avg_21d_return": round(float(np.mean(arr)), 2),
                "win_rate_21d": round(float(np.mean(arr > 0)) * 100, 1),
            }
    summary["by_pattern"] = pat_summary
    return summary


# ═══════════════════════════════════════════════════════════
# 报告打印
# ═══════════════════════════════════════════════════════════
PERIOD_LABELS = {
    "5d": "1 周",
    "10d": "2 周",
    "21d": "1 个月",
    "63d": "3 个月",
}


def print_report(summary: dict, signals: list[dict]):
    """打印回测报告。"""
    print()
    print("=" * 100)
    print("  上升收敛形态 专项回测报告")
    print("=" * 100)

    n = summary["total_signals"]
    if n == 0:
        print("  ⚠️  未发现任何形态信号")
        return

    print(f"  信号总数:     {n} 次")
    print(f"  涉及股票:     {summary['unique_tickers']} 只")
    print(f"  股票列表:     {', '.join(summary['tickers'])}")
    print()

    # ── 收益统计表 ──
    header = (
        f"  {'持有期':<10} {'样本':>6} {'胜率':>8} {'平均收益':>10} "
        f"{'中位收益':>10} {'最大盈':>10} {'最大亏':>10} {'波动率':>8} "
        f"{'均最大回撤':>12} {'均最大浮盈':>12}"
    )
    print(header)
    print("  " + "-" * 108)

    for pk in [f"{d}d" for d in FORWARD_DAYS]:
        stats = summary.get(pk)
        if not stats:
            continue

        label = PERIOD_LABELS.get(pk, pk)
        print(
            f"  {label:<10} {stats['count']:>6} "
            f"{stats['win_rate']:.0f}%{'':<3} "
            f"{stats['avg_return']:+.2f}%{'':<4} "
            f"{stats['median_return']:+.2f}%{'':<4} "
            f"{stats['max_gain']:+.2f}%{'':<4} "
            f"{stats['max_loss']:+.2f}%{'':<4} "
            f"{stats['std']:.2f}%{'':<2} "
            f"{stats.get('avg_max_drawdown', 0):+.2f}%{'':<6} "
            f"{stats.get('avg_max_gain', 0):+.2f}%"
        )

    # ── 按形态分组 ──
    pat_stats = summary.get("by_pattern", {})
    if pat_stats:
        print()
        print("  ── 按形态类型分组（21日收益）──")
        print(f"  {'形态':<14} {'信号数':>8} {'21日胜率':>10} {'21日均收益':>12}")
        print("  " + "-" * 48)
        for p, ps in sorted(pat_stats.items()):
            cn = _CN.get(p, p)
            print(
                f"  {cn:<12} {ps['count']:>8} "
                f"{ps['win_rate_21d']:.0f}%{'':<5} "
                f"{ps['avg_21d_return']:+.2f}%"
            )

    # ── 逐笔明细 ──
    print()
    print("  ── 逐笔明细 ──")
    print(
        f"  {'股票':<7} {'信号日':>12} {'周期':>5} {'形态':<10} "
        f"{'收敛比':>6} {'波幅缩':>6} {'量缩':>6} "
        f"{'入场价':>10} {'5日':>8} {'10日':>8} {'21日':>8} {'63日':>8}"
    )
    print("  " + "-" * 120)

    for s in sorted(signals, key=lambda x: x["signal_date"]):
        cn = _CN.get(s["pattern"], s["pattern"])
        cells = []
        for pk in [f"{d}d" for d in FORWARD_DAYS]:
            info = s.get(pk, {})
            r = info.get("return_pct")
            cells.append(f"{r:+.1f}%" if r is not None else "N/A")

        print(
            f"  {s['ticker']:<7} {s['signal_date']:>12} "
            f"{s['period']:>5} {cn:<10} "
            f"{s['convergence_ratio']:.0%}{'':<2} "
            f"{1 - s['spread_contraction']:.0%}{'':<2} "
            f"{1 - s['vol_contraction']:.0%}{'':<2} "
            f"${s['entry_price']:.2f}{'':<2} "
            f"{cells[0]:>8} {cells[1]:>8} {cells[2]:>8} {cells[3]:>8}"
        )

    print()


# ═══════════════════════════════════════════════════════════
# K 线图绘制
# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════
# 按形态分组完整统计
# ═══════════════════════════════════════════════════════════
def _print_per_pattern_report(signals: list[dict]):
    """按形态类型分组，打印所有持有期的完整统计。"""
    pat_groups: dict[str, list] = {}
    for s in signals:
        pat_groups.setdefault(s["pattern"], []).append(s)

    for pat in sorted(pat_groups):
        group = pat_groups[pat]
        cn = _CN.get(pat, pat)
        print()
        print("=" * 90)
        print(f"  {cn}  ({len(group)} 个信号)")
        print("=" * 90)
        header = (
            f"  {'持有期':<10} {'样本':>6} {'胜率':>8} {'平均收益':>10} "
            f"{'中位收益':>10} {'最大盈':>10} {'最大亏':>10}"
        )
        print(header)
        print("  " + "-" * 76)

        for d in FORWARD_DAYS:
            pk = f"{d}d"
            rets = []
            for s in group:
                info = s.get(pk, {})
                r = info.get("return_pct")
                if r is not None and "note" not in info:
                    rets.append(r)
            if not rets:
                continue
            arr = np.array(rets)
            label = PERIOD_LABELS.get(pk, pk)
            wr = float(np.mean(arr > 0)) * 100
            avg = float(np.mean(arr))
            med = float(np.median(arr))
            mx = float(np.max(arr))
            mn = float(np.min(arr))
            print(
                f"  {label:<10} {len(rets):>6} "
                f"{wr:.0f}%{'':>3} "
                f"{avg:+.2f}%{'':>4} "
                f"{med:+.2f}%{'':>4} "
                f"{mx:+.2f}%{'':>4} "
                f"{mn:+.2f}%"
            )


# ═══════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════
def run_backtest(step: int = 5, min_gap_days: int = 20, universe: str = "ndx100"):
    """执行上升收敛形态历史回测。

    Args:
        universe: 数据集，支持 ndx100 / us / us+ndx100 / hk / all / shawn
    """
    logger.info("=" * 60)
    logger.info(f"上升收敛形态 专项历史回测 [{universe}]")
    logger.info("=" * 60)

    if universe == "shawn":
        shawn_info = load_watchlist_data()
        stock_data = {sym: info["df"] for sym, info in shawn_info.items()}
    else:
        stock_data = load_universe_data(universe)  # type: ignore[arg-type]
    if not stock_data:
        logger.error(f"无数据！universe={universe}")
        return

    sample = next(iter(stock_data.values()))
    logger.info(
        f"数据范围: {sample.index[0].date()} ~ {sample.index[-1].date()} "
        f"({len(sample)} 交易日)"
    )
    logger.info(f"扫描步长: 每 {step} 个交易日 | 信号去重间隔: {min_gap_days} 天")
    logger.info("")

    all_signals: list[dict] = []
    processed = 0

    for ticker, df in stock_data.items():
        if len(df) < 265:
            logger.debug(f"{ticker}: 数据不足 ({len(df)} 行)，跳过")
            continue

        processed += 1
        sigs = scan_historical_triangles(
            ticker, df, step=step, min_gap_days=min_gap_days
        )
        if sigs:
            logger.success(f"  ✅ {ticker}: 发现 {len(sigs)} 个信号")
            all_signals.extend(sigs)

    logger.info(
        f"\n扫描完成: {processed} 只股票已处理，"
        f"共发现 {len(all_signals)} 个信号"
    )

    if not all_signals:
        logger.warning("未发现任何信号。可能是筛选条件过于严格。")
        return

    # ── 统计汇总 ──
    summary = summarize_signals(all_signals)
    print_report(summary, all_signals)

    # ── 按形态分组完整统计 ──
    _print_per_pattern_report(all_signals)

    # ── 绘图 ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_triangle_backtest_signals(all_signals, stock_data, OUTPUT_DIR, max_charts=30)

    logger.info("上升收敛形态 回测完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="上升收敛形态 专项回测")
    parser.add_argument(
        "--step", type=int, default=5, help="扫描步长（交易日数），默认 5"
    )
    parser.add_argument(
        "--gap", type=int, default=20, help="同股票信号最小间隔天数，默认 20"
    )
    parser.add_argument(
        "--universe", default="ndx100",
        choices=["ndx100", "us", "us+ndx100", "hk", "all", "shawn"],
        help="数据集 (默认 ndx100)",
    )
    args = parser.parse_args()

    run_backtest(step=args.step, min_gap_days=args.gap, universe=args.universe)
