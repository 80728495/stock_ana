#!/usr/bin/env python3
"""
VCP 专项回测脚本
================
遍历每只股票的完整历史数据，在每个时间点检测是否出现 VCP 形态，
一旦发现就记录 VCP 的起止点并计算后续持有期收益。

用法：
    python backtest_vcp.py
    python backtest_vcp.py --step 3        # 每 3 天扫描一次（更精细）
    python backtest_vcp.py --step 10       # 每 10 天扫描一次（更快）
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.data_fetcher import load_all_ndx100_data
from stock_ana.screener import screen_vcp

# ──────── 常量 ────────
FORWARD_DAYS = [5, 10, 21, 63]  # 1周 / 2周 / 1月 / 3月
OUTPUT_DIR = Path("data") / "backtest_vcp"


# ═══════════════════════════════════════════════════════════
# 历史 VCP 扫描器
# ═══════════════════════════════════════════════════════════
def scan_historical_vcps(
    ticker: str,
    df: pd.DataFrame,
    step: int = 5,
    min_data: int = 260,
    min_gap_days: int = 20,
) -> list[dict]:
    """
    在单只股票的完整历史中，滑动窗口检测所有 VCP 出现的位置。

    Args:
        ticker: 股票代码
        df: 完整历史数据 DataFrame
        step: 扫描步长（每隔多少交易日检测一次）
        min_data: 最少需要的历史数据天数（check_trend_template 需要 260）
        min_gap_days: 同一只股票两次 VCP 信号之间的最小间隔天数

    Returns:
        VCP 信号列表，每项包含形态信息和前瞻收益
    """
    signals = []
    total = len(df)

    if total < min_data + 5:
        return signals

    last_signal_idx = -999  # 上次信号的 iloc 位置

    for cutoff in range(min_data, total - 1, step):
        # 去重：距上次信号太近则跳过
        if cutoff - last_signal_idx < min_gap_days:
            continue

        df_slice = df.iloc[:cutoff + 1].copy()

        try:
            result = screen_vcp(df_slice)
        except Exception:
            continue

        if result is None:
            continue

        # ── 提取 VCP 起止信息 ──
        base_high_iloc = cutoff - result["base_days"]
        base_high_date = df.index[base_high_iloc]
        signal_date = df.index[cutoff]

        # ── 计算前瞻收益 ──
        entry_idx = cutoff + 1  # 信号日的下一个交易日入场
        if entry_idx >= total:
            continue

        entry_price = df.iloc[entry_idx]["close"]
        remaining = df.iloc[entry_idx:]

        fwd_returns = {}
        for fwd in FORWARD_DAYS:
            key = f"{fwd}d"
            if len(remaining) > fwd:
                exit_price = remaining.iloc[fwd]["close"]
                ret = (exit_price - entry_price) / entry_price * 100
                period_closes = remaining.iloc[:fwd + 1]["close"].values
                peak = np.maximum.accumulate(period_closes)
                drawdowns = (period_closes - peak) / peak * 100
                max_dd = float(np.min(drawdowns))
                max_gain = float((np.max(period_closes) - entry_price) / entry_price * 100)
                fwd_returns[key] = {
                    "return_pct": round(ret, 2),
                    "max_drawdown_pct": round(max_dd, 2),
                    "max_gain_pct": round(max_gain, 2),
                }
            else:
                actual_days = len(remaining) - 1
                if actual_days > 0:
                    exit_price = remaining.iloc[-1]["close"]
                    ret = (exit_price - entry_price) / entry_price * 100
                    fwd_returns[key] = {
                        "return_pct": round(ret, 2),
                        "actual_days": actual_days,
                        "note": "数据不足",
                    }

        signals.append({
            "ticker": ticker,
            "signal_date": str(signal_date.date()),
            "signal_iloc": cutoff,
            "entry_date": str(df.index[entry_idx].date()),
            "entry_price": round(float(entry_price), 2),
            # VCP 形态信息
            "base_high_date": str(base_high_date.date()),
            "base_high": result["base_high"],
            "base_days": result["base_days"],
            "wave_count": result["wave_count"],
            "depths": result["depths"],
            "tightness": result["tightness"],
            "vol_ratio": result["vol_ratio"],
            "distance_to_pivot_pct": result["distance_to_pivot_pct"],
            # 前瞻收益
            **fwd_returns,
        })

        last_signal_idx = cutoff

    return signals


# ═══════════════════════════════════════════════════════════
# 统计汇总
# ═══════════════════════════════════════════════════════════
def summarize_vcp_signals(signals: list[dict]) -> dict:
    """汇总 VCP 回测统计。"""
    if not signals:
        return {"total_signals": 0}

    n = len(signals)
    unique_tickers = set(s["ticker"] for s in signals)

    summary = {
        "total_signals": n,
        "unique_tickers": len(unique_tickers),
        "tickers": sorted(unique_tickers),
    }

    for period_key in [f"{d}d" for d in FORWARD_DAYS]:
        rets = []
        dds = []
        gains = []
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

        rets_arr = np.array(rets)
        wins = rets_arr > 0

        stats = {
            "count": len(rets),
            "win_rate": round(float(np.mean(wins)) * 100, 1),
            "avg_return": round(float(np.mean(rets_arr)), 2),
            "median_return": round(float(np.median(rets_arr)), 2),
            "max_gain": round(float(np.max(rets_arr)), 2),
            "max_loss": round(float(np.min(rets_arr)), 2),
            "std": round(float(np.std(rets_arr)), 2),
        }
        if dds:
            stats["avg_max_drawdown"] = round(float(np.mean(dds)), 2)
        if gains:
            stats["avg_max_gain"] = round(float(np.mean(gains)), 2)

        summary[period_key] = stats

    # 按波数分组统计
    wave_groups = {}
    for s in signals:
        wc = s["wave_count"]
        wave_groups.setdefault(wc, []).append(s)

    wave_summary = {}
    for wc, group in sorted(wave_groups.items()):
        rets_21d = []
        for s in group:
            info = s.get("21d")
            if info and "return_pct" in info and "note" not in info:
                rets_21d.append(info["return_pct"])
        if rets_21d:
            arr = np.array(rets_21d)
            wave_summary[wc] = {
                "count": len(group),
                "avg_21d_return": round(float(np.mean(arr)), 2),
                "win_rate_21d": round(float(np.mean(arr > 0)) * 100, 1),
            }

    summary["by_wave_count"] = wave_summary
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


def print_vcp_report(summary: dict, signals: list[dict]):
    """打印 VCP 专项回测报告。"""
    print()
    print("=" * 100)
    print("  VCP（波动率收缩形态）专项回测报告")
    print("=" * 100)

    n = summary["total_signals"]
    if n == 0:
        print("  ⚠️  历史数据中未发现任何 VCP 形态")
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

    for period_key in [f"{d}d" for d in FORWARD_DAYS]:
        stats = summary.get(period_key)
        if not stats:
            continue

        label = PERIOD_LABELS.get(period_key, period_key)
        cnt = f"{stats['count']}"
        wr = f"{stats['win_rate']:.0f}%"
        avg = f"{stats['avg_return']:+.2f}%"
        med = f"{stats['median_return']:+.2f}%"
        mx = f"{stats['max_gain']:+.2f}%"
        mn = f"{stats['max_loss']:+.2f}%"
        std = f"{stats['std']:.2f}%"
        dd = f"{stats.get('avg_max_drawdown', 0):+.2f}%" if "avg_max_drawdown" in stats else "N/A"
        mg = f"{stats.get('avg_max_gain', 0):+.2f}%" if "avg_max_gain" in stats else "N/A"

        print(
            f"  {label:<10} {cnt:>6} {wr:>8} {avg:>10} "
            f"{med:>10} {mx:>10} {mn:>10} {std:>8} "
            f"{dd:>12} {mg:>12}"
        )

    # ── 按波数分组 ──
    wave_stats = summary.get("by_wave_count", {})
    if wave_stats:
        print()
        print("  ── 按收缩波数分组（21日收益）──")
        print(f"  {'波数':>6} {'信号数':>8} {'21日胜率':>10} {'21日均收益':>12}")
        print("  " + "-" * 40)
        for wc, ws in sorted(wave_stats.items()):
            print(
                f"  {wc:>6} {ws['count']:>8} "
                f"{ws['win_rate_21d']:.0f}%{'':<5} "
                f"{ws['avg_21d_return']:+.2f}%"
            )

    # ── 逐笔明细 ──
    print()
    print("  ── 逐笔明细（含 VCP 起止点）──")
    print(
        f"  {'股票':<7} {'VCP起点':>12} {'信号日':>12} {'基底天数':>8} "
        f"{'波数':>4} {'收缩幅度':>20} {'量缩比':>8} "
        f"{'入场价':>10} {'5日':>8} {'10日':>8} {'21日':>8} {'63日':>8}"
    )
    print("  " + "-" * 138)

    for s in sorted(signals, key=lambda x: x["signal_date"]):
        ticker = s["ticker"]
        base_start = s["base_high_date"]
        sig_date = s["signal_date"]
        base_days = s["base_days"]
        waves = s["wave_count"]
        depths_str = "→".join(f"{d:.0f}%" for d in s["depths"])
        vol_r = f"{s['vol_ratio']:.0%}"
        entry = f"${s['entry_price']:.2f}"

        cells = []
        for pk in [f"{d}d" for d in FORWARD_DAYS]:
            info = s.get(pk, {})
            r = info.get("return_pct")
            if r is not None:
                cells.append(f"{r:+.1f}%")
            else:
                cells.append("N/A")

        print(
            f"  {ticker:<7} {base_start:>12} {sig_date:>12} {base_days:>8} "
            f"{waves:>4} {depths_str:>20} {vol_r:>8} "
            f"{entry:>10} {cells[0]:>8} {cells[1]:>8} {cells[2]:>8} {cells[3]:>8}"
        )

    print()


# ═══════════════════════════════════════════════════════════
# K 线图绘制
# ═══════════════════════════════════════════════════════════
def plot_vcp_signals(
    signals: list[dict],
    stock_data: dict[str, pd.DataFrame],
    output_dir: Path,
    pre_year_days: int = 200,
    post_days: int = 40,
    max_charts: int = 30,
):
    """
    为每个 VCP 信号绘制约 1 年的 K 线图。

    - 显示约 1 年数据，VCP 区域位于图表右侧，左侧为历史走势
    - 绿色圆圈标记 VCP 区段的起点（Base High）
    - 红色圆圈标记 VCP 区段的终点（信号日）
    - 每个信号单独一张图
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    to_plot = sorted(signals, key=lambda x: x["signal_date"])[:max_charts]
    chart_count = 0

    for s in to_plot:
        ticker = s["ticker"]
        if ticker not in stock_data:
            continue

        full_df = stock_data[ticker]
        sig_iloc = s["signal_iloc"]
        base_start_iloc = sig_iloc - s["base_days"]

        # 显示范围：VCP 起点前 pre_year_days 天 ~ 信号日后 post_days 天
        # 这样 VCP 区域自然落在图的右侧
        view_start = max(0, base_start_iloc - pre_year_days)
        view_end = min(len(full_df), sig_iloc + post_days)
        df_view = full_df.iloc[view_start:view_end].copy()

        if len(df_view) < 20:
            continue

        add_plots = []

        # SMA50 / SMA150 / SMA200（基于全量数据计算，再截取视图范围）
        closes_full = full_df["close"]
        for ma_len, color in [(50, "green"), (150, "blue"), (200, "purple")]:
            ma = closes_full.rolling(ma_len).mean().iloc[view_start:view_end]
            if not ma.isna().all():
                add_plots.append(mpf.make_addplot(ma, color=color, width=0.8, linestyle="-"))

        # 绿色圆圈：VCP 起点（Base High）
        start_marker = pd.Series(np.nan, index=df_view.index)
        if view_start <= base_start_iloc < view_end:
            start_date = full_df.index[base_start_iloc]
            start_marker.loc[start_date] = full_df.loc[start_date, "high"]

        # 红色圆圈：VCP 终点（信号日）
        end_marker = pd.Series(np.nan, index=df_view.index)
        if view_start <= sig_iloc < view_end:
            sig_date = full_df.index[sig_iloc]
            end_marker.loc[sig_date] = full_df.loc[sig_date, "close"]

        if start_marker.notna().any():
            add_plots.append(
                mpf.make_addplot(
                    start_marker, type="scatter", markersize=200,
                    marker="o", color="lime", edgecolors="green",
                    linewidths=2, alpha=0.9,
                )
            )
        if end_marker.notna().any():
            add_plots.append(
                mpf.make_addplot(
                    end_marker, type="scatter", markersize=200,
                    marker="o", color="red", edgecolors="darkred",
                    linewidths=2, alpha=0.9,
                )
            )

        # Base High 水平参考线
        base_high_line = pd.Series(s["base_high"], index=df_view.index)
        add_plots.append(
            mpf.make_addplot(base_high_line, color="magenta", width=0.7, linestyle=":")
        )

        # 收益信息
        ret_parts = []
        for pk, label in [("5d", "1W"), ("21d", "1M"), ("63d", "3M")]:
            info = s.get(pk, {})
            r = info.get("return_pct")
            if r is not None:
                ret_parts.append(f"{label}:{r:+.1f}%")
        ret_str = "  ".join(ret_parts)

        depths_str = "->".join(f"{d:.0f}%" for d in s["depths"])
        title = (
            f"{ticker} VCP ({s['wave_count']} waves: {depths_str})  "
            f"Base:{s['base_days']}d  |  {ret_str}"
        )

        style = mpf.make_mpf_style(base_mpf_style="charles", rc={"font.size": 9})
        save_path = output_dir / f"{ticker}_vcp_{s['signal_date']}.png"

        try:
            mpf.plot(
                df_view,
                type="candle",
                volume=True,
                title=title,
                style=style,
                figscale=1.5,
                figratio=(16, 9),
                addplot=add_plots,
                savefig=str(save_path),
                warn_too_much_data=len(df_view) + 1,
            )
            plt.close("all")
            chart_count += 1
        except Exception as e:
            logger.debug(f"{ticker}: 绘图失败 - {e}")
            continue

    logger.info(f"  已保存 {chart_count} 张 VCP 信号图 → {output_dir}")


# ═══════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════
def run_vcp_backtest(step: int = 5, min_gap_days: int = 20):
    """
    执行 VCP 专项历史回测。

    遍历所有纳指100成分股的完整历史数据，
    在每个时间点检测 VCP，记录起止点并计算后续收益。

    Args:
        step: 扫描步长（每隔多少交易日检测一次）
        min_gap_days: 同一股票两次 VCP 信号的最小间隔天数
    """
    logger.info("=" * 60)
    logger.info("VCP 专项历史回测")
    logger.info("=" * 60)

    logger.info("加载数据...")
    stock_data = load_all_ndx100_data()
    if not stock_data:
        logger.error("无数据！请先运行 update_ndx100_data()")
        return

    sample_df = next(iter(stock_data.values()))
    logger.info(
        f"数据范围: {sample_df.index[0].date()} ~ {sample_df.index[-1].date()} "
        f"({len(sample_df)} 交易日)"
    )
    logger.info(f"扫描步长: 每 {step} 个交易日 | 信号去重间隔: {min_gap_days} 天")
    logger.info("")

    # ── 遍历所有股票 ──
    all_signals = []
    processed = 0

    for ticker, df in stock_data.items():
        if len(df) < 265:
            logger.debug(f"{ticker}: 数据不足 ({len(df)} 行)，跳过")
            continue

        processed += 1
        signals = scan_historical_vcps(
            ticker, df, step=step, min_gap_days=min_gap_days,
        )
        if signals:
            logger.success(f"  ✅ {ticker}: 发现 {len(signals)} 个 VCP 信号")
            all_signals.extend(signals)

    logger.info(
        f"\n扫描完成: {processed} 只股票已处理，"
        f"共发现 {len(all_signals)} 个 VCP 信号"
    )

    if not all_signals:
        logger.warning("未发现任何 VCP 信号。可能是筛选条件过于严格或数据周期不够长。")
        return

    # ── 统计汇总 ──
    summary = summarize_vcp_signals(all_signals)
    print_vcp_report(summary, all_signals)

    # ── 绘制 K 线图 ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_vcp_signals(all_signals, stock_data, OUTPUT_DIR, max_charts=30)

    logger.info("VCP 专项回测完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VCP 专项历史回测")
    parser.add_argument("--step", type=int, default=5, help="扫描步长（交易日数），默认 5")
    parser.add_argument("--gap", type=int, default=20, help="同股票信号最小间隔天数，默认 20")
    args = parser.parse_args()

    run_vcp_backtest(step=args.step, min_gap_days=args.gap)
