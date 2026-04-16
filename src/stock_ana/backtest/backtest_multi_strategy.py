"""
策略回测验证脚本
===============
用前 N 个月的数据进行策略筛选，用后续数据验证股价表现。

支持滚动回测：在多个截止日分别筛选，累积样本量以获得更稳健的统计结论。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ──────── 项目导入 ────────
from stock_ana.data.market_data import load_market_data, load_universe_data, load_shawn_data
from stock_ana.data.indicators import add_vegas_channel
from stock_ana.strategies.api import (
    screen_ma_squeeze,
    screen_momentum,
    screen_vegas_touch,
    screen_rs_acceleration,
    screen_rs_trap_alert,
    screen_triangle_ascending,
    screen_triangle_kde_setup,
    screen_triangle_parallel_channel,
    screen_triangle_rising_wedge,
    screen_vcp_setup,
)
from stock_ana.strategies.impl.rs import (
    compute_rs_rank_at_cutoff,
    _load_qqq,
    update_qqq_data,
    compute_rs_line,
)
from stock_ana.utils.plot_renderers import plot_multi_strategy_backtest_signals

# ──────── 常量 ────────
FORWARD_DAYS = [5, 10, 21]  # 持有 1 周 / 2 周 / 1 个月
STRATEGY_NAMES = {
    "vegas": "Vegas 通道回弹",
    "triangle": "上升三角形(OLS)",
    "triangle_kde": "上升三角形(KDE)",
    "parallel": "上升平行通道",
    "wedge": "上升楔形",
    "vcp": "VCP 波动率收缩",
    "rs_strict": "RS加速(严格)",
    "rs_loose": "RS加速(宽松)",
    "rs_trap_strict": "RS陷阱预警(严格)",
    "rs_trap_loose": "RS陷阱预警(宽松)",
    "ma_squeeze": "MA Squeeze",
    "momentum": "Momentum 异动",
}

# 图表标题用英文避免字体问题
STRATEGY_NAMES_EN = {
    "vegas": "Vegas Channel Touchback",
    "triangle": "Ascending Triangle (OLS)",
    "triangle_kde": "Ascending Triangle (KDE)",
    "parallel": "Parallel Channel",
    "wedge": "Rising Wedge",
    "vcp": "VCP (Volatility Contraction)",
    "rs_strict": "RS Acceleration (Strict)",
    "rs_loose": "RS Acceleration (Loose)",
    "rs_trap_strict": "RS Trap Warning (Strict)",
    "rs_trap_loose": "RS Trap Warning (Loose)",
    "ma_squeeze": "MA Squeeze",
    "momentum": "Momentum Detection",
}

STRATEGY_KIND_MAP = {
    "vegas": "pattern",
    "triangle": "pattern",
    "triangle_kde": "pattern",
    "parallel": "pattern",
    "wedge": "pattern",
    "vcp": "pattern",
    "rs_strict": "stateful_signal",
    "rs_loose": "stateful_signal",
    "rs_trap_strict": "stateful_signal",
    "rs_trap_loose": "stateful_signal",
    "ma_squeeze": "stateful_signal",
    "momentum": "stateful_signal",
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


def _screen_one(strategy: str, df: pd.DataFrame, **kwargs) -> dict | None:
    """统一调用筛选函数，所有策略统一返回 dict | None。"""
    strategy_kind = STRATEGY_KIND_MAP.get(strategy)
    if strategy_kind == "pattern":
        return _screen_pattern_strategy(strategy, df)
    if strategy_kind == "stateful_signal":
        return _screen_stateful_signal_strategy(strategy, df, **kwargs)
    return None


def _screen_pattern_strategy(strategy: str, df: pd.DataFrame) -> dict | None:
    """形态发现类策略：事件稀疏，主要提取几何/结构特征。"""
    if strategy == "vegas":
        decision = screen_vegas_touch(df, lookback_days=5)
        return decision.features if decision.passed else None
    if strategy == "triangle":
        decision = screen_triangle_ascending(df)
        return decision.features if decision.passed else None
    if strategy == "triangle_kde":
        decision = screen_triangle_kde_setup(df)
        return decision.features if decision.passed else None
    if strategy == "parallel":
        decision = screen_triangle_parallel_channel(df)
        return decision.features if decision.passed else None
    if strategy == "wedge":
        decision = screen_triangle_rising_wedge(df)
        return decision.features if decision.passed else None
    if strategy == "vcp":
        decision = screen_vcp_setup(df)
        return decision.features if decision.passed else None
    return None


def _screen_stateful_signal_strategy(
    strategy: str,
    df: pd.DataFrame,
    **kwargs,
) -> dict | None:
    """日频状态类策略：每个截止日重新计算状态，再判断是否触发。"""
    if strategy == "ma_squeeze":
        decision = screen_ma_squeeze(df)
        return decision.features if decision.passed else None
    if strategy == "momentum":
        decision = screen_momentum(df)
        return decision.features if decision.passed else None
    if strategy in ("rs_strict", "rs_loose"):
        df_market = kwargs.get("df_market")
        rs_rank = kwargs.get("rs_rank")
        if df_market is None or rs_rank is None:
            return None
        decision = screen_rs_acceleration(df, df_market, rs_rank, variant=strategy)
        return decision.features if decision.passed else None
    if strategy in ("rs_trap_strict", "rs_trap_loose"):
        df_market = kwargs.get("df_market")
        rs_rank = kwargs.get("rs_rank")
        if df_market is None or rs_rank is None:
            return None
        decision = screen_rs_trap_alert(df, df_market, rs_rank, variant=strategy)
        return decision.features if decision.passed else None
    return None


def backtest_single_cutoff(
    strategy: str,
    stock_data: dict[str, pd.DataFrame],
    cutoff_idx: int,
    **kwargs,
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
    strategy_kind = STRATEGY_KIND_MAP.get(strategy)

    for ticker, full_df in stock_data.items():
        # 截取筛选数据
        df_screen = full_df.iloc[:cutoff_idx].copy()

        # 各策略的最低数据要求
        if strategy_kind == "pattern":
            if strategy == "vcp" and len(df_screen) < 200:
                continue
            if strategy in ("triangle", "triangle_kde", "parallel", "wedge") and len(df_screen) < 60:
                continue
            if strategy == "vegas" and len(df_screen) < 170:
                continue
        elif strategy_kind == "stateful_signal":
            if strategy in ("rs_strict", "rs_loose", "rs_trap_strict", "rs_trap_loose") and len(df_screen) < 252:
                continue
            if strategy == "ma_squeeze" and len(df_screen) < 210:
                continue
            if strategy == "momentum" and len(df_screen) < 60:
                continue
        else:
            continue

        try:
            # RS 策略需要额外参数
            if strategy in ("rs_strict", "rs_loose", "rs_trap_strict", "rs_trap_loose"):
                df_market = kwargs.get("df_market")
                rs_ranks = kwargs.get("rs_ranks", {})
                result = _screen_one(
                    strategy, df_screen,
                    df_market=df_market.iloc[:cutoff_idx] if df_market is not None else None,
                    rs_rank=rs_ranks.get(ticker, -1),
                )
            else:
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
                "green_idx_rel": result.get("green_idx_rel"),  # T1 高点(窗口内)
                "red_idx_rel": result.get("red_idx_rel"),      # 最终收缩高点(窗口内)
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
        elif strategy in ("parallel", "wedge"):
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
        elif strategy in ("rs_strict", "rs_loose"):
            signal_info = {
                "rs_rank": result.get("rs_rank"),
                "rs_chg_21d": result.get("rs_chg_21d"),
                "rs_chg_63d": result.get("rs_chg_63d"),
                "acceleration": result.get("acceleration"),
                "rs_below_days": result.get("rs_below_days"),
                "atr_ratio": result.get("atr_ratio"),
                "price_vs_52w_high": result.get("price_vs_52w_high"),
                "variant": result.get("variant"),
            }
        elif strategy in ("rs_trap_strict", "rs_trap_loose"):
            signal_info = {
                "rs_rank": result.get("rs_rank"),
                "mkt_ret": result.get("mkt_ret"),
                "stk_ret": result.get("stk_ret"),
                "outperform": result.get("outperform"),
                "rs_chg_63d": result.get("rs_chg_63d"),
                "rs_chg_short": result.get("rs_chg_short"),
                "rs_vs_ema": result.get("rs_vs_ema"),
                "price_below_sma200": result.get("price_below_sma200"),
                "death_cross": result.get("death_cross"),
                "variant": result.get("variant"),
            }
        elif strategy == "ma_squeeze":
            signal_info = {
                "stage1_triggered": result.get("stage1_triggered"),
                "stage2_triggered": result.get("stage2_triggered"),
                "stage2_score": result.get("stage2_score"),
            }
        elif strategy == "momentum":
            signal_info = {
                "abnormal_return": result.get("abnormal_return", {}),
                "breakout": result.get("breakout", {}),
                "vol_surge": result.get("vol_surge", {}),
                "accumulation": result.get("accumulation", {}),
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


def _run_strategy_backtest_rollup(
    strategy: str,
    stock_data: dict[str, pd.DataFrame],
    cutoff_indices: list[int],
    avg_benchmark: dict,
    extra_kwargs_builder=None,
) -> tuple[dict, list[dict]]:
    """执行单个策略在全部截止点上的回测汇总。"""
    name = STRATEGY_NAMES[strategy]
    logger.info(f"回测策略: {name} ...")
    all_trades = []
    signal_counts = []

    for idx in cutoff_indices:
        extra_kwargs = extra_kwargs_builder(idx) if extra_kwargs_builder else {}
        trades = backtest_single_cutoff(
            strategy,
            stock_data,
            idx,
            **extra_kwargs,
        )
        signal_counts.append(len(trades))
        all_trades.extend(trades)

    total_signals = sum(signal_counts)
    active_cutoffs = sum(1 for count in signal_counts if count > 0)
    logger.info(
        f"  汇总: {total_signals} 个信号, "
        f"在 {active_cutoffs}/{len(cutoff_indices)} 个截止点触发"
    )

    deduped_trades = _deduplicate_trades(all_trades, min_gap_days=15)
    if len(deduped_trades) < len(all_trades):
        logger.info(
            f"  去重: {len(all_trades)} -> {len(deduped_trades)} "
            f"(合并 {15} 天内同股票重复信号)"
        )

    summary = summarize_trades(deduped_trades, avg_benchmark)
    print_report(strategy, summary, deduped_trades)

    if deduped_trades:
        chart_dir = Path("data") / "backtest_charts"
        plot_multi_strategy_backtest_signals(deduped_trades, stock_data, strategy, chart_dir)

    return summary, deduped_trades


def _run_pattern_backtests(
    stock_data: dict[str, pd.DataFrame],
    cutoff_indices: list[int],
    avg_benchmark: dict,
) -> dict[str, dict]:
    """事件驱动回测路径：形态发现类策略。"""
    strategies = ["vegas", "triangle", "parallel", "wedge", "vcp"]
    all_summaries: dict[str, dict] = {}
    for strategy in strategies:
        summary, _ = _run_strategy_backtest_rollup(
            strategy,
            stock_data,
            cutoff_indices,
            avg_benchmark,
        )
        all_summaries[strategy] = summary
    return all_summaries


def _run_stateful_signal_backtests(
    stock_data: dict[str, pd.DataFrame],
    cutoff_indices: list[int],
    avg_benchmark: dict,
) -> dict[str, dict]:
    """日频信号回测路径：状态类策略。"""
    all_summaries: dict[str, dict] = {}

    for strategy in ["ma_squeeze", "momentum"]:
        summary, _ = _run_strategy_backtest_rollup(
            strategy,
            stock_data,
            cutoff_indices,
            avg_benchmark,
        )
        all_summaries[strategy] = summary

    df_market = _load_qqq()
    if df_market is None:
        logger.warning("QQQ 数据不存在，跳过 RS 策略回测。请运行 update_qqq_data() 下载数据")
        return all_summaries

    rs_ranks_by_cutoff = {
        idx: compute_rs_rank_at_cutoff(stock_data, df_market, idx)
        for idx in cutoff_indices
    }

    for strategy in ["rs_strict", "rs_loose", "rs_trap_strict", "rs_trap_loose"]:
        summary, _ = _run_strategy_backtest_rollup(
            strategy,
            stock_data,
            cutoff_indices,
            avg_benchmark,
            extra_kwargs_builder=lambda idx, rs_map=rs_ranks_by_cutoff: {
                "df_market": df_market,
                "rs_ranks": rs_map[idx],
            },
        )
        all_summaries[strategy] = summary

    return all_summaries


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
    universe: str = "ndx100",
):
    """
    执行完整回测。

    使用滚动截止点：从足够靠后的位置开始（保证各策略有足够数据），
    每隔 rolling_step 个交易日取一个截止点，直到 total - min_validation_days。

    Args:
        rolling_step: 滚动步长（交易日），默认每周一次
        min_validation_days: 截止点后至少要保留的验证天数
        universe: 数据集，支持 ndx100 / us / us+ndx100 / hk / all / shawn
    """
    logger.info("加载数据...")
    if universe == "shawn":
        shawn_info = load_shawn_data()
        stock_data = {sym: info["df"] for sym, info in shawn_info.items()}
    else:
        stock_data = load_universe_data(universe)  # type: ignore[arg-type]
    if not stock_data:
        logger.error(f"无数据！universe={universe}")
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

    # ── 运行两类回测路径 ──
    all_summaries = {}
    pattern_summaries = _run_pattern_backtests(stock_data, cutoff_indices, avg_benchmark)
    all_summaries.update(pattern_summaries)

    stateful_summaries = _run_stateful_signal_backtests(
        stock_data,
        cutoff_indices,
        avg_benchmark,
    )
    all_summaries.update(stateful_summaries)

    # ── 对比总结 ──
    print_cross_strategy_comparison(all_summaries)

    # ── 未命中任何策略的股票 → others 文件夹（带 Vegas 通道线） ──
    all_signaled_tickers: set[str] = set()
    for strategy in pattern_summaries:
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
    import argparse
    parser = argparse.ArgumentParser(description="多策略滚动回测")
    parser.add_argument("--step", type=int, default=5, help="滚动步长（交易日）")
    parser.add_argument("--validation-days", type=int, default=21, help="最少验证天数")
    parser.add_argument(
        "--universe", default="ndx100",
        choices=["ndx100", "us", "us+ndx100", "hk", "all", "shawn"],
        help="数据集 (默认 ndx100)",
    )
    _args = parser.parse_args()
    run_backtest(
        rolling_step=_args.step,
        min_validation_days=_args.validation_days,
        universe=_args.universe,
    )
