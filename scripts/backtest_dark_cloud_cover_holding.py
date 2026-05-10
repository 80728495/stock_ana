#!/usr/bin/env python3
"""
backtest_dark_cloud_cover_holding.py
对 holding.md 所有标的进行「跳空巨阴线」策略的完整历史回测。

用法：
    python scripts/backtest_dark_cloud_cover_holding.py
    python scripts/backtest_dark_cloud_cover_holding.py --min-score 3
    python scripts/backtest_dark_cloud_cover_holding.py --no-chart
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.config import CACHE_DIR, OUTPUT_DIR
from stock_ana.strategies.impl.dark_cloud_cover import scan_history
from _backtest_shared import (
    cjk_font_prop, make_ema_addplots, load_ohlcv, parse_holding_md,
    print_stats, print_per_sym_table,
    add_chart_legend, add_stat_panel, set_chart_title, safe_save,
    EMA_COLORS,
)

OUT_DIR = OUTPUT_DIR / "dark_cloud_cover" / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDING_PATH = PROJECT_ROOT / "data" / "lists" / "holding.md"

MARKET_CACHE = {
    "US": CACHE_DIR / "us",
    "HK": CACHE_DIR / "hk",
    "CN": CACHE_DIR / "cn",
}


# ═══════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════

def _load_ohlcv(sym: str, market: str) -> pd.DataFrame | None:
    return load_ohlcv(sym, market, MARKET_CACHE)


def _parse_holding_md() -> list[dict]:
    return parse_holding_md(HOLDING_PATH)


# ═══════════════════════════════════════════════════
#  K 线图（带全部历史信号标注）
# ═══════════════════════════════════════════════════

def _render_backtest_chart(
    sym: str, market: str, name: str,
    df: pd.DataFrame,
    hits: pd.DataFrame,
    min_score: int,
) -> Path:
    n_bars = min(len(df), 500)
    df_plot = df.iloc[-n_bars:].copy()[["open", "high", "low", "close", "volume"]]

    add_plots = make_ema_addplots(df, df_plot)

    # 信号标记（Day[0]）
    marker_high = pd.Series(np.nan, index=df_plot.index)
    for _, row in hits.iterrows():
        if row["score"] < min_score:
            continue
        try:
            d0 = pd.Timestamp(row["signal_date"])
            if d0 in df_plot.index:
                pos = df_plot.index.get_loc(d0)
                marker_high.iloc[pos] = float(df_plot.iloc[pos]["high"]) * 1.018
        except Exception:
            pass

    if marker_high.notna().any():
        add_plots.append(mpf.make_addplot(
            marker_high, type="scatter", marker="v",
            markersize=120, color="#FF4444", edgecolors="#990000",
        ))

    mc = mpf.make_marketcolors(
        up="#CC3333", down="#00AA00", edge="inherit", wick="inherit", volume="in",
    )
    mpf_style = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=":", gridcolor="#E0E0E0",
        rc={"axes.unicode_minus": False},
    )

    fig, axes = mpf.plot(
        df_plot, type="candle", style=mpf_style,
        addplot=add_plots if add_plots else None,
        volume=True, figsize=(24, 9), returnfig=True,
        tight_layout=False,
        scale_padding={"left": 0.03, "right": 0.30, "top": 0.6, "bottom": 0.5},
        warn_too_much_data=n_bars + 1,
    )
    ax = axes[0]

    # 标注每个信号
    for _, row in hits.iterrows():
        if row["score"] < min_score:
            continue
        try:
            d0 = pd.Timestamp(row["signal_date"])
            if d0 not in df_plot.index:
                continue
            pos = df_plot.index.get_loc(d0)
            high_val = float(df_plot.iloc[pos]["high"])

            ret_5 = row.get("fwd_min_5d", 0)
            ret_20 = row.get("fwd_ret_20d", 0)
            accurate = ret_5 <= -2.0 or ret_20 <= -5.0
            label_color = "#CC0000" if accurate else "#888888"

            date_str = str(row["signal_date"])[5:]  # MM-DD
            vol_str = f"V{row.get('d0_vol_vs_ma50', 0):.1f}x"
            reclaim_mark = "▼" if row.get("close_below_prev") else "▽"
            p60 = row.get('pre60_ret', 0)
            p60_str = f"P{p60:+.0f}%"

            label = (
                f"{date_str}\n"
                f"{reclaim_mark} s{row['score']} {p60_str}\n"
                f"{ret_5:+.0f}%/{ret_20:+.0f}%"
            )
            ax.annotate(
                label,
                xy=(pos, high_val * 1.025),
                fontsize=6.0, fontweight="bold", color=label_color,
                ha="center", va="bottom",
            )
        except Exception:
            pass

    # 右侧统计面板
    hits_filtered = hits[hits["score"] >= min_score]
    total = len(hits_filtered)
    if total > 0:
        acc_5d = (hits_filtered["fwd_min_5d"] <= -2.0).sum()
        acc_10d = (hits_filtered["fwd_ret_10d"] <= -3.0).sum()
        acc_20d = (hits_filtered["fwd_ret_20d"] <= -5.0).sum()
        avg_ret_5 = hits_filtered["fwd_ret_5d"].mean()
        avg_min_5 = hits_filtered["fwd_min_5d"].mean()
        avg_ret_20 = hits_filtered["fwd_ret_20d"].mean()

        near_high_cnt = hits_filtered["is_near_high"].sum()
        below_prev_cnt = hits_filtered["close_below_prev"].sum()
        avg_gap = hits_filtered["gap_pct"].mean()
        avg_pre60 = hits_filtered["pre60_ret"].mean() if "pre60_ret" in hits_filtered.columns else 0
        high_pre60 = (hits_filtered["pre60_ret"] >= 25).sum() if "pre60_ret" in hits_filtered.columns else 0

        panel = [
            f"── 回测统计（score≥{min_score}）──",
            f"  信号总数: {total}",
            f"  阶段新高: {near_high_cnt} / {total}",
            f"  收下昨收: {below_prev_cnt} / {total}",
            f"  平均跳空: {avg_gap:+.1f}%",
            f"  pre60均值: {avg_pre60:+.0f}%  (≥25%:{high_pre60})",
            "",
            "── 准确率 ──",
            f"  5d最低跌>2%:  {acc_5d}/{total} = {acc_5d/total*100:.0f}%",
            f"  10d收盘跌>3%: {acc_10d}/{total} = {acc_10d/total*100:.0f}%",
            f"  20d收盘跌>5%: {acc_20d}/{total} = {acc_20d/total*100:.0f}%",
            "",
            "── 平均收益 ──",
            f"  5d 收盘均值:   {avg_ret_5:+.1f}%",
            f"  5d 最低均值:   {avg_min_5:+.1f}%",
            f"  20d 收盘均值:  {avg_ret_20:+.1f}%",
        ]
    else:
        panel = [f"无信号（score≥{min_score}）"]

    add_stat_panel(ax, panel, bg_color="#FFEBEE", edge_color="#EF9A9A")
    set_chart_title(fig,
        f"{market}:{sym}  {name}  —  跳空巨阴线 回测（共{total}次，score≥{min_score}）")
    add_chart_legend(ax, strategy_label="Day[0] 信号")

    _UNSAFE = set('/\\:*?"<>|')
    safe_name = "".join(c for c in name if c not in _UNSAFE).strip()[:15] or sym
    out_path = OUT_DIR / f"{market}_{sym}_{safe_name}.png"
    safe_save(fig, out_path)
    return out_path


# ═══════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="跳空巨阴线策略历史回测（holding.md）")
    parser.add_argument("--min-score", type=int, default=1,
                        help="只统计 score >= N 的信号（默认 1=全部）")
    parser.add_argument("--no-chart", action="store_true",
                        help="跳过图表生成，只打印统计")
    args = parser.parse_args()

    items = _parse_holding_md()
    logger.info(f"holding.md 共 {len(items)} 只标的，开始跳空巨阴线回测 ...")

    all_hits: list[dict] = []
    chart_paths: list[Path] = []

    for item in items:
        sym, market, name = item["sym"], item["market"], item["name"]
        df = _load_ohlcv(sym, market)
        if df is None or len(df) < 30:
            logger.debug(f"{market}:{sym} 跳过")
            continue

        hits = scan_history(df, forward_days=(5, 10, 20))
        if hits.empty:
            logger.debug(f"{market}:{sym} 无信号")
            continue

        for _, row in hits.iterrows():
            all_hits.append({
                "sym": sym, "market": market, "name": name,
                "section": item["section"],
                **row.to_dict(),
            })

        hits_filtered = hits[hits["score"] >= args.min_score]
        logger.info(f"  {market}:{sym:8s} {name[:10]:10s}  信号 {len(hits)} 次  (score≥{args.min_score}: {len(hits_filtered)} 次)")

        # 只有当前 min-score 下有信号才生成图表
        if not args.no_chart and not hits_filtered.empty:
            try:
                path = _render_backtest_chart(sym, market, name, df, hits, args.min_score)
                chart_paths.append(path)
            except Exception as e:
                logger.error(f"{market}:{sym} 图表失败: {e}")

    if not all_hits:
        print("没有找到任何跳空巨阴线历史信号。")
        return

    df_all = pd.DataFrame(all_hits)
    filt = df_all[df_all["score"] >= args.min_score].copy()

    # ── 汇总统计 ──
    print("\n" + "═" * 70)
    print(f"  跳空巨阴线策略 历史回测汇总  (score ≥ {args.min_score}，共 {len(filt)} 次信号)")
    print("═" * 70)

    print_stats(filt, "全部")

    for mkt in sorted(filt["market"].unique()):
        print_stats(filt[filt["market"] == mkt], f"市场={mkt}")

    for sc in sorted(filt["score"].unique(), reverse=True):
        print_stats(filt[filt["score"] == sc], f"score={sc}")

    # 按新高/非新高分类
    for label, sub in [("新高区", filt[filt["is_near_high"]]), ("非新高", filt[~filt["is_near_high"]])]:
        print_stats(sub, f"位置={label}")

    # 按股票列出
    print_per_sym_table(filt)

    print(f"\n  图表已保存至：{OUT_DIR}  （共 {len(chart_paths)} 张）")


if __name__ == "__main__":
    main()
