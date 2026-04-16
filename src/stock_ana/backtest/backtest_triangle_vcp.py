#!/usr/bin/env python3
"""
VCP 三角形 专项回测脚本（Shawn List）
=====================================

对 Shawn List 中的美股，以宏观前高为起点，检测 VCP 三角形收敛，
计算前瞻收益并输出图表。

用法：
    python -m stock_ana.backtest.backtest_triangle_vcp
    python -m stock_ana.backtest.backtest_triangle_vcp --step 5 --gap 20
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.data.market_data import load_shawn_data
from stock_ana.strategies.primitives.peaks import find_macro_peaks
from stock_ana.strategies.impl.triangle_vcp import screen_triangle_vcp
from stock_ana.scan.triangle_vcp_scan import scan_historical_triangle_vcp
from stock_ana.utils.plot_renderers import plot_triangle_vcp_backtest_signals

# ──────── 常量 ────────
FORWARD_DAYS = [5, 10, 21, 63]
OUTPUT_DIR = Path("data") / "backtest_triangle_vcp"

_CN = {
    "ascending_triangle": "上升三角形(VCP)",
    "descending_triangle": "下降三角形(VCP)",
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

    # 按形态分组
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
    print("=" * 110)
    print("  VCP 三角形 专项回测报告 (Shawn List)")
    print("=" * 110)

    n = summary["total_signals"]
    if n == 0:
        print("  ⚠️  未发现任何形态信号")
        return

    print(f"  信号总数:     {n} 次")
    print(f"  涉及股票:     {summary['unique_tickers']} 只")
    print(f"  股票列表:     {', '.join(summary['tickers'])}")
    print()

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

    # 按形态分组
    pat_stats = summary.get("by_pattern", {})
    if pat_stats:
        print()
        print("  ── 按形态类型分组（21日收益）──")
        print(f"  {'形态':<20} {'信号数':>8} {'21日胜率':>10} {'21日均收益':>12}")
        print("  " + "-" * 54)
        for p, ps in sorted(pat_stats.items()):
            cn = _CN.get(p, p)
            print(
                f"  {cn:<18} {ps['count']:>8} "
                f"{ps['win_rate_21d']:.0f}%{'':<5} "
                f"{ps['avg_21d_return']:+.2f}%"
            )

    # 逐笔明细
    print()
    print("  ── 逐笔明细 ──")
    print(
        f"  {'股票':<7} {'信号日':>12} {'前高日':>12} {'间距':>5} {'形态':<18} "
        f"{'收敛比':>6} {'波幅缩':>6} {'量缩':>6} "
        f"{'入场价':>10} {'5日':>8} {'10日':>8} {'21日':>8} {'63日':>8}"
    )
    print("  " + "-" * 140)

    for s in sorted(signals, key=lambda x: x["signal_date"]):
        cn = _CN.get(s["pattern"], s["pattern"])
        cells = []
        for pk in [f"{d}d" for d in FORWARD_DAYS]:
            info = s.get(pk, {})
            r = info.get("return_pct")
            cells.append(f"{r:+.1f}%" if r is not None else "N/A")

        print(
            f"  {s['ticker']:<7} {s['signal_date']:>12} "
            f"{s['peak_date']:>12} {s['gap_days']:>5} {cn:<18} "
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
# 主函数
# ═══════════════════════════════════════════════════════════

def run_backtest(step: int = 5, min_gap_days: int = 20):
    """执行 VCP 三角形历史回测（Shawn List 美股）。"""
    logger.info("=" * 60)
    logger.info("VCP 三角形 专项历史回测 (Shawn List)")
    logger.info("=" * 60)

    stock_data_full = load_shawn_data()
    if not stock_data_full:
        logger.error("无数据！请确保 Shawn List 中的美股已下载到本地")
        return

    # 仅取美股（此策略针对 US 标的）
    stock_data = {sym: info["df"] for sym, info in stock_data_full.items()
                  if info["market"] == "US"}
    if not stock_data:
        logger.error("Shawn List 中无美股数据")
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
        sigs = scan_historical_triangle_vcp(
            ticker, df, step=step, min_gap_days=min_gap_days,
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

    # 统计汇总
    summary = summarize_signals(all_signals)
    print_report(summary, all_signals)

    # 绘图
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_triangle_vcp_backtest_signals(all_signals, stock_data, OUTPUT_DIR, max_charts=50)

    logger.info("VCP 三角形 回测完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VCP 三角形 专项回测 (Shawn List)")
    parser.add_argument(
        "--step", type=int, default=5, help="扫描步长（交易日数），默认 5",
    )
    parser.add_argument(
        "--gap", type=int, default=20, help="同股票信号最小间隔天数，默认 20",
    )
    args = parser.parse_args()

    run_backtest(step=args.step, min_gap_days=args.gap)
