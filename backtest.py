"""
策略回测验证脚本
===============
用前 N 个月的数据进行策略筛选，用后续数据验证股价表现。

支持滚动回测：在多个截止日分别筛选，累积样本量以获得更稳健的统计结论。
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from loguru import logger

# ──────── 项目导入 ────────
from stock_ana.data_fetcher import load_all_ndx100_data
from stock_ana.indicators import add_vegas_channel
from stock_ana.screener import (
    screen_ascending_triangle,
    screen_vegas_channel_touch,
    screen_vcp,
)

# ──────── 常量 ────────
FORWARD_DAYS = [5, 10, 21]  # 持有 1 周 / 2 周 / 1 个月
STRATEGY_NAMES = {
    "vegas": "Vegas 通道回弹",
    "triangle": "上升三角形",
    "vcp": "VCP 波动率收缩",
}

# 图表标题用英文避免字体问题
STRATEGY_NAMES_EN = {
    "vegas": "Vegas Channel Touchback",
    "triangle": "Ascending Triangle",
    "vcp": "VCP (Volatility Contraction)",
}


# ═══════════════════════════════════════════════════════════
# 核心回测引擎
# ═══════════════════════════════════════════════════════════
def _compute_forward_returns(
    full_df: pd.DataFrame,
    entry_idx: int,
    forward_days: list[int],
) -> dict:
    """
    从 entry_idx（筛选截止日的下一个交易日）开始，
    计算各持有期的收益率和最大回撤。
    """
    remaining = full_df.iloc[entry_idx:]
    if len(remaining) < 2:
        return {}

    entry_price = remaining.iloc[0]["close"]
    result = {"entry_date": str(remaining.index[0].date()), "entry_price": entry_price}

    for fwd in forward_days:
        key = f"{fwd}d"
        if len(remaining) > fwd:
            exit_price = remaining.iloc[fwd]["close"]
            ret = (exit_price - entry_price) / entry_price * 100
            # 区间最大回撤
            period_closes = remaining.iloc[: fwd + 1]["close"].values
            peak = np.maximum.accumulate(period_closes)
            drawdowns = (period_closes - peak) / peak * 100
            max_dd = float(np.min(drawdowns))
            # 区间最高收益
            max_gain = float(
                (np.max(period_closes) - entry_price) / entry_price * 100
            )
            result[key] = {
                "return_pct": round(ret, 2),
                "max_drawdown_pct": round(max_dd, 2),
                "max_gain_pct": round(max_gain, 2),
            }
        else:
            # 不够天数，用到最后一天
            exit_price = remaining.iloc[-1]["close"]
            actual_days = len(remaining) - 1
            ret = (exit_price - entry_price) / entry_price * 100
            result[key] = {
                "return_pct": round(ret, 2),
                "actual_days": actual_days,
                "note": "数据不足",
            }

    # 到数据末尾的总收益
    end_price = remaining.iloc[-1]["close"]
    total_ret = (end_price - entry_price) / entry_price * 100
    total_days = len(remaining) - 1
    period_closes = remaining["close"].values
    peak = np.maximum.accumulate(period_closes)
    drawdowns = (period_closes - peak) / peak * 100
    result["to_end"] = {
        "return_pct": round(total_ret, 2),
        "days": total_days,
        "max_drawdown_pct": round(float(np.min(drawdowns)), 2),
    }

    return result


def _screen_one(strategy: str, df: pd.DataFrame) -> dict | None:
    """统一调用筛选函数，所有策略统一返回 dict | None。"""
    if strategy == "vegas":
        df = add_vegas_channel(df.copy())
        return screen_vegas_channel_touch(df, lookback_days=5)
    elif strategy == "triangle":
        return screen_ascending_triangle(df)
    elif strategy == "vcp":
        return screen_vcp(df)
    return None


def backtest_single_cutoff(
    strategy: str,
    stock_data: dict[str, pd.DataFrame],
    cutoff_idx: int,
) -> list[dict]:
    """
    在单个截止点对所有股票运行策略筛选，计算前瞻收益。

    Args:
        strategy: "vegas" | "triangle" | "vcp"
        stock_data: {ticker: full_df}
        cutoff_idx: 数据截止位置（用 iloc[:cutoff_idx] 做筛选）

    Returns:
        命中股票的前瞻收益列表
    """
    trades = []

    for ticker, full_df in stock_data.items():
        # 截取筛选数据
        df_screen = full_df.iloc[:cutoff_idx].copy()

        # 各策略的最低数据要求
        if strategy == "vcp" and len(df_screen) < 200:
            continue
        if strategy == "triangle" and len(df_screen) < 60:
            continue
        if strategy == "vegas" and len(df_screen) < 170:
            continue

        try:
            result = _screen_one(strategy, df_screen)
        except Exception:
            continue

        if result is None:
            continue

        # 存储信号详细信息（用于绘图）
        signal_info = {}
        if strategy == "vcp":
            signal_info = {
                "pattern": result.get("pattern", "vcp"),
                "depths": result.get("depths", []),
                "score": result.get("score", 0),
                "contractions": result.get("contractions", []),
                "window_start": result.get("window_start"),
                "base_days": result.get("base_days"),
                "pivot_price": result.get("pivot_price"),
            }
        elif strategy == "triangle":
            signal_info = {
                "pattern": result.get("pattern", ""),
                "convergence": result.get("convergence_ratio", 0),
                "resistance": result.get("resistance"),
                "support": result.get("support"),
                "period": result.get("period"),
                "window_start": result.get("window_start"),
            }
        elif strategy == "vegas":
            signal_info = {
                "peak_price": result.get("peak_price"),
                "above_ratio": result.get("above_ratio"),
            }

        # 计算前瞻收益（从截止日的下一天开始算）
        fwd = _compute_forward_returns(full_df, cutoff_idx, FORWARD_DAYS)
        if not fwd:
            continue

        cutoff_date = str(full_df.index[cutoff_idx - 1].date())
        trades.append(
            {
                "ticker": ticker,
                "cutoff_date": cutoff_date,
                "cutoff_idx": cutoff_idx,
                "strategy": strategy,
                "signal_info": signal_info,
                **fwd,
            }
        )

    return trades


def compute_benchmark(
    stock_data: dict[str, pd.DataFrame],
    cutoff_idx: int,
) -> dict:
    """计算同期 NDX100 等权平均收益作为基准。"""
    returns = {f"{d}d": [] for d in FORWARD_DAYS}
    returns["to_end"] = []

    for ticker, full_df in stock_data.items():
        if len(full_df) <= cutoff_idx + 1:
            continue
        remaining = full_df.iloc[cutoff_idx:]
        entry_price = remaining.iloc[0]["close"]
        if entry_price <= 0:
            continue

        for fwd in FORWARD_DAYS:
            if len(remaining) > fwd:
                ret = (remaining.iloc[fwd]["close"] - entry_price) / entry_price * 100
                returns[f"{fwd}d"].append(ret)

        end_ret = (remaining.iloc[-1]["close"] - entry_price) / entry_price * 100
        returns["to_end"].append(end_ret)

    benchmark = {}
    for key, vals in returns.items():
        if vals:
            benchmark[key] = {
                "avg_return_pct": round(float(np.mean(vals)), 2),
                "median_return_pct": round(float(np.median(vals)), 2),
            }
    return benchmark


# ═══════════════════════════════════════════════════════════
# 回测信号 K 线图
# ═══════════════════════════════════════════════════════════
def _build_vegas_overlays(
    df_view: pd.DataFrame,
    full_df: pd.DataFrame,
    view_start: int,
    view_end: int,
) -> list:
    """为 Vegas 策略生成 EMA144/EMA169 辅助线。"""
    # 对完整数据计算 EMA，然后截取 view 范围
    df_full_ema = add_vegas_channel(full_df.copy())
    ema144 = df_full_ema["ema_144"].iloc[view_start:view_end]
    ema169 = df_full_ema["ema_169"].iloc[view_start:view_end]

    overlays = [
        mpf.make_addplot(ema144, color="blue", width=1.5, linestyle="-"),
        mpf.make_addplot(ema169, color="purple", width=1.5, linestyle="-"),
    ]
    return overlays


def _build_triangle_overlays(
    df_view: pd.DataFrame,
    signal_info: dict,
    full_df: pd.DataFrame,
    view_start: int,
    view_end: int,
) -> list:
    """为三角形策略生成上轨/下轨辅助线。"""
    resistance = signal_info.get("resistance")
    support = signal_info.get("support")
    window_start = signal_info.get("window_start")
    period = signal_info.get("period")

    if not resistance or not support or window_start is None or period is None:
        return []

    # 上下轨线是相对 window_start 的线性函数:
    #   y = slope * (x - window_start) + intercept
    # 其中 x 是 full_df 的 iloc 位置
    res_slope = resistance["slope"]
    res_intercept = resistance["intercept"]
    sup_slope = support["slope"]
    sup_intercept = support["intercept"]

    # 在 df_view 的每个位置计算趋势线值
    res_series = pd.Series(np.nan, index=df_view.index)
    sup_series = pd.Series(np.nan, index=df_view.index)

    for i, date in enumerate(df_view.index):
        abs_idx = view_start + i
        rel_x = abs_idx - window_start
        # 只在三角形窗口范围内（稍微延伸一些到未来）画线
        if rel_x >= -5 and rel_x <= period + 20:
            res_series.iloc[i] = res_slope * rel_x + res_intercept
            sup_series.iloc[i] = sup_slope * rel_x + sup_intercept

    overlays = [
        mpf.make_addplot(res_series, color="red", width=1.2, linestyle="--"),
        mpf.make_addplot(sup_series, color="green", width=1.2, linestyle="--"),
    ]
    return overlays


def _build_vcp_overlays(
    df_view: pd.DataFrame,
    signal_info: dict,
    full_df: pd.DataFrame,
    view_start: int,
    view_end: int,
) -> list:
    """为 VCP 策略生成收缩高低点连线、枢轴价线、SMA。"""
    contractions = signal_info.get("contractions", [])
    window_start = signal_info.get("window_start")
    pivot_price = signal_info.get("pivot_price")

    if not contractions or window_start is None:
        return []

    overlays = []

    # 收缩段的高点连线和低点连线
    high_series = pd.Series(np.nan, index=df_view.index)
    low_series = pd.Series(np.nan, index=df_view.index)

    for c in contractions:
        hi_abs = window_start + c["high_idx"]
        lo_abs = window_start + c["low_idx"]

        if view_start <= hi_abs < view_end:
            high_series.iloc[hi_abs - view_start] = c["high_val"]
        if view_start <= lo_abs < view_end:
            low_series.iloc[lo_abs - view_start] = c["low_val"]

    # 在高点和低点之间插值形成连线
    high_interp = high_series.interpolate(limit_area="inside")
    low_interp = low_series.interpolate(limit_area="inside")

    overlays.append(
        mpf.make_addplot(high_interp, color="red", width=1.0, linestyle="--")
    )
    overlays.append(
        mpf.make_addplot(low_interp, color="orange", width=1.0, linestyle="--")
    )

    # 枢轴价水平线
    if pivot_price:
        pivot_series = pd.Series(pivot_price, index=df_view.index)
        overlays.append(
            mpf.make_addplot(pivot_series, color="magenta", width=0.8, linestyle=":")
        )

    # SMA150 和 SMA200
    closes_full = full_df["close"]
    sma150 = closes_full.rolling(150).mean().iloc[view_start:view_end]
    sma200 = closes_full.rolling(200).mean().iloc[view_start:view_end]

    if not sma150.isna().all():
        overlays.append(
            mpf.make_addplot(sma150, color="blue", width=1.0, linestyle="-")
        )
    if not sma200.isna().all():
        overlays.append(
            mpf.make_addplot(sma200, color="purple", width=1.0, linestyle="-")
        )

    return overlays


def plot_backtest_signals(
    trades: list[dict],
    stock_data: dict[str, pd.DataFrame],
    strategy: str,
    output_dir: Path,
) -> None:
    """
    为回测信号绘制 K 线图，含红圈标注信号日 + 各策略辅助线。

    辅助线:
    - Vegas: EMA144 (蓝), EMA169 (紫)
    - Triangle: 上轨 (红虚线), 下轨 (绿虚线)
    - VCP: 收缩高点连线 (红虚线), 低点连线 (橙虚线),
           枢轴价 (洋红点线), SMA150 (蓝), SMA200 (紫)
    """
    strat_dir = output_dir / strategy
    strat_dir.mkdir(parents=True, exist_ok=True)

    # 按 ticker 分组
    ticker_trades: dict[str, list[dict]] = {}
    for t in trades:
        ticker_trades.setdefault(t["ticker"], []).append(t)

    name = STRATEGY_NAMES.get(strategy, strategy)

    for ticker, t_list in ticker_trades.items():
        if ticker not in stock_data:
            continue
        full_df = stock_data[ticker]

        # 找信号日期 & 确定显示范围（完整过去一年）
        signal_dates = []
        for t in t_list:
            cut_date = pd.Timestamp(t["cutoff_date"])
            idx_pos = full_df.index.searchsorted(cut_date)
            if idx_pos >= len(full_df):
                idx_pos = len(full_df) - 1
            signal_dates.append(idx_pos)

        earliest = min(signal_dates)
        latest = max(signal_dates)
        # 显示完整一年（251 交易日），确保信号日可见
        year_start = max(0, latest - 251)
        view_start = min(year_start, max(0, earliest - 10))
        view_end = min(len(full_df), latest + 40)
        df_view = full_df.iloc[view_start:view_end].copy()

        if len(df_view) < 5:
            continue

        # 信号日红圈标记
        marker_red = pd.Series(np.nan, index=df_view.index)
        for idx_pos in signal_dates:
            if view_start <= idx_pos < view_end:
                date = full_df.index[idx_pos]
                if date in marker_red.index:
                    marker_red.loc[date] = full_df.loc[date, "close"]

        last_trade = t_list[-1]
        signal_info = last_trade.get("signal_info", {})

        # VCP 起始日绿圈标记
        marker_green = pd.Series(np.nan, index=df_view.index)
        if strategy == "vcp":
            ws = signal_info.get("window_start")
            if ws is not None and view_start <= ws < view_end:
                date = full_df.index[ws]
                if date in marker_green.index:
                    marker_green.loc[date] = full_df.loc[date, "close"]

        # ── 构建辅助线 ──
        add_plots = []

        try:
            if strategy == "vegas":
                add_plots.extend(
                    _build_vegas_overlays(df_view, full_df, view_start, view_end)
                )
            elif strategy == "triangle":
                add_plots.extend(
                    _build_triangle_overlays(
                        df_view, signal_info,
                        full_df, view_start, view_end,
                    )
                )
            elif strategy == "vcp":
                add_plots.extend(
                    _build_vcp_overlays(
                        df_view, signal_info,
                        full_df, view_start, view_end,
                    )
                )
        except Exception as e:
            logger.debug(f"{ticker}: 辅助线绘制异常 {e}")

        # 绿圈（VCP 起始）
        if not marker_green.isna().all():
            add_plots.append(
                mpf.make_addplot(
                    marker_green,
                    type="scatter",
                    markersize=200,
                    marker="o",
                    color="green",
                    alpha=0.8,
                ),
            )

        # 红圈（信号日）
        add_plots.append(
            mpf.make_addplot(
                marker_red,
                type="scatter",
                markersize=200,
                marker="o",
                color="red",
                alpha=0.8,
            ),
        )

        # 标题（英文）
        ret_text = []
        for pk, label in [("5d", "1W"), ("10d", "2W"), ("21d", "1M"), ("to_end", "End")]:
            info = last_trade.get(pk, {})
            r = info.get("return_pct")
            if r is not None:
                ret_text.append(f"{label}:{r:+.1f}%")
        ret_str = "  ".join(ret_text)
        name_en = STRATEGY_NAMES_EN.get(strategy, strategy)
        title = f"{ticker} - {name_en}  |  {ret_str}"

        style = mpf.make_mpf_style(base_mpf_style="charles", rc={"font.size": 9})

        # 垂直红色虚线标注信号日
        vlines_dates = []
        for idx_pos in signal_dates:
            if view_start <= idx_pos < view_end:
                vlines_dates.append(full_df.index[idx_pos])

        vlines_kwargs = {}
        if vlines_dates:
            vlines_kwargs = {
                "vlines": dict(
                    vlines=vlines_dates,
                    colors="red",
                    linewidths=0.8,
                    linestyle="--",
                    alpha=0.5,
                ),
            }

        save_path = strat_dir / f"{ticker}_{strategy}.png"

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
            **vlines_kwargs,
        )
        plt.close("all")

    logger.info(f"  已保存 {len(ticker_trades)} 张 {name} 信号图 → {strat_dir}")


# ═══════════════════════════════════════════════════════════
# 汇总统计
# ═══════════════════════════════════════════════════════════
def summarize_trades(trades: list[dict], benchmark: dict) -> dict:
    """汇总交易结果统计。"""
    if not trades:
        return {"total_signals": 0}

    n = len(trades)
    unique_tickers = set(t["ticker"] for t in trades)
    summary = {
        "total_signals": n,
        "unique_tickers": len(unique_tickers),
        "tickers": sorted(unique_tickers),
    }

    for period_key in [f"{d}d" for d in FORWARD_DAYS] + ["to_end"]:
        rets = []
        for t in trades:
            info = t.get(period_key)
            if info and "return_pct" in info:
                rets.append(info["return_pct"])

        if not rets:
            continue

        rets_arr = np.array(rets)
        wins = rets_arr > 0
        losses = rets_arr < 0

        bm = benchmark.get(period_key, {})

        summary[period_key] = {
            "win_rate": round(float(np.mean(wins)) * 100, 1),
            "avg_return": round(float(np.mean(rets_arr)), 2),
            "median_return": round(float(np.median(rets_arr)), 2),
            "max_gain": round(float(np.max(rets_arr)), 2),
            "max_loss": round(float(np.min(rets_arr)), 2),
            "std": round(float(np.std(rets_arr)), 2),
            "benchmark_avg": bm.get("avg_return_pct", None),
            "alpha": (
                round(float(np.mean(rets_arr)) - bm["avg_return_pct"], 2)
                if "avg_return_pct" in bm
                else None
            ),
        }

        # 最大回撤汇总
        dds = []
        for t in trades:
            info = t.get(period_key)
            if info and "max_drawdown_pct" in info:
                dds.append(info["max_drawdown_pct"])
        if dds:
            summary[period_key]["avg_max_drawdown"] = round(float(np.mean(dds)), 2)

    return summary


# ═══════════════════════════════════════════════════════════
# 报告打印
# ═══════════════════════════════════════════════════════════
PERIOD_LABELS = {"5d": "1 周", "10d": "2 周", "21d": "1 个月", "to_end": "持有到期末"}


def _deduplicate_trades(trades: list[dict], min_gap_days: int = 15) -> list[dict]:
    """
    去除同一股票在相邻截止日产生的重复信号。

    同一只股票若在 min_gap_days 天之内被反复筛出，只保留最早那次。
    """
    trades_sorted = sorted(trades, key=lambda t: (t["ticker"], t["cutoff_date"]))
    deduped = []
    last_by_ticker: dict[str, str] = {}  # ticker -> last cutoff_date

    for t in trades_sorted:
        ticker = t["ticker"]
        cutoff = t["cutoff_date"]

        if ticker in last_by_ticker:
            last_date = pd.Timestamp(last_by_ticker[ticker])
            this_date = pd.Timestamp(cutoff)
            gap = (this_date - last_date).days
            if gap < min_gap_days:
                continue  # 太近，跳过

        last_by_ticker[ticker] = cutoff
        deduped.append(t)

    return deduped


def print_report(strategy: str, summary: dict, trades: list[dict]):
    """打印单策略回测报告。"""
    name = STRATEGY_NAMES.get(strategy, strategy)
    print()
    print("=" * 72)
    print(f"  策略回测报告：{name}")
    print("=" * 72)

    n = summary["total_signals"]
    if n == 0:
        print("  ⚠️  无信号，跳过")
        return

    print(f"  信号总数:     {n} 次")
    print(f"  涉及股票:     {summary['unique_tickers']} 只")
    print(f"  股票列表:     {', '.join(summary['tickers'])}")
    print()

    # 表格
    header = f"  {'持有期':<10} {'胜率':>8} {'平均收益':>10} {'中位收益':>10} {'最大盈':>10} {'最大亏':>10} {'波动率':>8} {'基准均':>10} {'Alpha':>8}"
    print(header)
    print("  " + "-" * 98)

    for period_key in ["5d", "10d", "21d", "to_end"]:
        stats = summary.get(period_key)
        if not stats:
            continue

        label = PERIOD_LABELS.get(period_key, period_key)
        wr = f"{stats['win_rate']:.0f}%"
        avg = f"{stats['avg_return']:+.2f}%"
        med = f"{stats['median_return']:+.2f}%"
        mx = f"{stats['max_gain']:+.2f}%"
        mn = f"{stats['max_loss']:+.2f}%"
        std = f"{stats['std']:.2f}%"
        bm = f"{stats['benchmark_avg']:+.2f}%" if stats["benchmark_avg"] is not None else "N/A"
        alpha = f"{stats['alpha']:+.2f}%" if stats["alpha"] is not None else "N/A"

        print(f"  {label:<10} {wr:>8} {avg:>10} {med:>10} {mx:>10} {mn:>10} {std:>8} {bm:>10} {alpha:>8}")

    # 平均最大回撤
    print()
    dd_line = "  平均最大回撤: "
    for period_key in ["5d", "10d", "21d", "to_end"]:
        stats = summary.get(period_key)
        if stats and "avg_max_drawdown" in stats:
            label = PERIOD_LABELS.get(period_key, period_key)
            dd_line += f"{label} {stats['avg_max_drawdown']:.2f}%  "
    print(dd_line)

    # 逐笔明细
    print()
    print("  ── 逐笔明细 ──")
    print(
        f"  {'股票':<8} {'筛选日':>12} {'入场价':>10}"
        f" {'5日收益':>10} {'10日收益':>10} {'21日收益':>10} {'到期末':>10}"
    )
    print("  " + "-" * 82)

    for t in sorted(trades, key=lambda x: x.get("to_end", {}).get("return_pct", 0), reverse=True):
        ticker = t["ticker"]
        cutoff = t["cutoff_date"]
        entry = f"${t['entry_price']:.2f}"

        cells = []
        for pk in ["5d", "10d", "21d", "to_end"]:
            info = t.get(pk, {})
            r = info.get("return_pct")
            cells.append(f"{r:+.2f}%" if r is not None else "N/A")

        print(f"  {ticker:<8} {cutoff:>12} {entry:>10} {cells[0]:>10} {cells[1]:>10} {cells[2]:>10} {cells[3]:>10}")


def print_cross_strategy_comparison(all_summaries: dict):
    """打印三策略对比总结。"""
    print()
    print("=" * 72)
    print("  三策略对比总结")
    print("=" * 72)
    print()

    header = f"  {'策略':<16} {'信号数':>6} {'5日胜率':>8} {'10日胜率':>9} {'21日胜率':>9} {'21日均收':>10} {'21日Alpha':>10}"
    print(header)
    print("  " + "-" * 78)

    for strategy, summary in all_summaries.items():
        name = STRATEGY_NAMES.get(strategy, strategy)
        n = summary["total_signals"]
        if n == 0:
            print(f"  {name:<16} {'0':>6} {'N/A':>8} {'N/A':>9} {'N/A':>9} {'N/A':>10} {'N/A':>10}")
            continue

        cells = []
        for pk in ["5d", "10d", "21d"]:
            stats = summary.get(pk, {})
            wr = f"{stats['win_rate']:.0f}%" if "win_rate" in stats else "N/A"
            cells.append(wr)

        s21 = summary.get("21d", {})
        avg21 = f"{s21['avg_return']:+.2f}%" if "avg_return" in s21 else "N/A"
        alpha21 = f"{s21['alpha']:+.2f}%" if s21.get("alpha") is not None else "N/A"

        print(f"  {name:<16} {n:>6} {cells[0]:>8} {cells[1]:>9} {cells[2]:>9} {avg21:>10} {alpha21:>10}")

    print()
    print("  注: Alpha = 策略平均收益 - NDX100 等权基准平均收益")
    print("  注: 采用滚动回测，同一股票可能在不同截止日出现多次")


# ═══════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════
def run_backtest(
    rolling_step: int = 5,
    min_validation_days: int = 21,
):
    """
    执行完整回测。

    使用滚动截止点：从足够靠后的位置开始（保证各策略有足够数据），
    每隔 rolling_step 个交易日取一个截止点，直到 total - min_validation_days。

    Args:
        rolling_step: 滚动步长（交易日），默认每周一次
        min_validation_days: 截止点后至少要保留的验证天数
    """
    logger.info("加载数据...")
    stock_data = load_all_ndx100_data()
    if not stock_data:
        logger.error("无数据！请先运行 update_ndx100_data()")
        return

    # 取第一个股票确定时间轴
    sample_df = next(iter(stock_data.values()))
    total_days = len(sample_df)

    # ── 策略最低数据要求 ──
    # VCP 需要 200 行（200 日 SMA），Vegas 需要 ~170 行（EMA169），Triangle 需要 ~60 行
    # 统一用 200 作为最早截止点，保证所有策略都能运行
    min_screen_days = 200
    first_cutoff = min_screen_days
    last_cutoff = total_days - min_validation_days

    if first_cutoff >= last_cutoff:
        logger.error(
            f"数据不足：总共 {total_days} 天，"
            f"策略需要 {min_screen_days} 天 + 验证需要 {min_validation_days} 天"
        )
        return

    cutoff_indices = list(range(first_cutoff, last_cutoff + 1, rolling_step))
    # 确保最后一个点也包含
    if cutoff_indices[-1] != last_cutoff:
        cutoff_indices.append(last_cutoff)

    logger.info(
        f"数据范围: {sample_df.index[0].date()} ~ {sample_df.index[-1].date()} "
        f"({total_days} 交易日)"
    )
    logger.info(
        f"滚动回测: {len(cutoff_indices)} 个截止点, "
        f"范围 第{first_cutoff}天 ~ 第{last_cutoff}天, "
        f"步长 {rolling_step} 天"
    )
    logger.info(
        f"  首个截止: {sample_df.index[first_cutoff].date()} "
        f"(验证期 {total_days - first_cutoff} 天)"
    )
    logger.info(
        f"  末个截止: {sample_df.index[last_cutoff].date()} "
        f"(验证期 {total_days - last_cutoff} 天)"
    )

    # ── 汇总基准 ──
    all_benchmarks = {}
    for idx in cutoff_indices:
        bm = compute_benchmark(stock_data, idx)
        for key, vals in bm.items():
            if key not in all_benchmarks:
                all_benchmarks[key] = []
            all_benchmarks[key].append(vals["avg_return_pct"])

    avg_benchmark = {}
    for key, vals in all_benchmarks.items():
        avg_benchmark[key] = {
            "avg_return_pct": round(float(np.mean(vals)), 2),
            "median_return_pct": round(float(np.median(vals)), 2),
        }

    logger.info("基准计算完成")
    for key in ["5d", "10d", "21d", "to_end"]:
        if key in avg_benchmark:
            label = PERIOD_LABELS.get(key, key)
            logger.info(
                f"  基准 {label}: 平均 {avg_benchmark[key]['avg_return_pct']:+.2f}%"
            )

    # ── 运行策略（triangle 暂时屏蔽） ──
    strategies = ["vegas", "vcp"]
    all_summaries = {}

    for strategy in strategies:
        name = STRATEGY_NAMES[strategy]
        logger.info(f"回测策略: {name} ...")
        all_trades = []
        signal_counts = []

        for idx in cutoff_indices:
            trades = backtest_single_cutoff(strategy, stock_data, idx)
            signal_counts.append(len(trades))
            all_trades.extend(trades)

        total_signals = sum(signal_counts)
        active_cutoffs = sum(1 for c in signal_counts if c > 0)
        logger.info(
            f"  汇总: {total_signals} 个信号, "
            f"在 {active_cutoffs}/{len(cutoff_indices)} 个截止点触发"
        )

        # ── 去重：同一股票在相邻截止日的重复信号 ──
        # 如果同一只股票在连续截止点都被选中，保留最早那次（避免重复计算收益）
        deduped_trades = _deduplicate_trades(all_trades, min_gap_days=15)
        if len(deduped_trades) < len(all_trades):
            logger.info(
                f"  去重: {len(all_trades)} -> {len(deduped_trades)} "
                f"(合并 {15} 天内同股票重复信号)"
            )

        summary = summarize_trades(deduped_trades, avg_benchmark)
        all_summaries[strategy] = summary
        print_report(strategy, summary, deduped_trades)

        # ── 绘制信号 K 线图 ──
        if deduped_trades:
            chart_dir = Path("data") / "backtest_charts"
            plot_backtest_signals(
                deduped_trades, stock_data, strategy, chart_dir,
            )

    # ── 对比总结 ──
    print_cross_strategy_comparison(all_summaries)

    # ── 未命中任何策略的股票 → others 文件夹（带 Vegas 通道线） ──
    all_signaled_tickers: set[str] = set()
    for strategy in strategies:
        s = all_summaries.get(strategy, {})
        all_signaled_tickers.update(s.get("tickers", []))

    others_tickers = sorted(set(stock_data.keys()) - all_signaled_tickers)
    if others_tickers:
        chart_dir = Path("data") / "backtest_charts"
        others_dir = chart_dir / "others"
        others_dir.mkdir(parents=True, exist_ok=True)

        for ticker in others_tickers:
            full_df = stock_data[ticker]
            if len(full_df) < 20:
                continue

            # 显示完整一年
            view_start = max(0, len(full_df) - 251)
            view_end = len(full_df)
            df_view = full_df.iloc[view_start:view_end].copy()

            if len(df_view) < 5 or df_view["close"].isna().all():
                continue

            try:
                add_plots = _build_vegas_overlays(
                    df_view, full_df, view_start, view_end,
                )
            except Exception:
                add_plots = []

            style = mpf.make_mpf_style(
                base_mpf_style="charles", rc={"font.size": 9},
            )
            title = f"{ticker} - No Strategy Match (Vegas Channel)"
            save_path = others_dir / f"{ticker}_others.png"

            try:
                mpf.plot(
                    df_view,
                    type="candle",
                    volume=True,
                    title=title,
                    style=style,
                    figscale=1.5,
                    figratio=(16, 9),
                    addplot=add_plots if add_plots else None,
                    savefig=str(save_path),
                )
                plt.close("all")
            except Exception as e:
                logger.debug(f"{ticker}: others 图绘制异常 {e}")

        logger.info(
            f"  已保存 {len(others_tickers)} 张 others 图 → {others_dir}"
        )

    print()


if __name__ == "__main__":
    run_backtest()
