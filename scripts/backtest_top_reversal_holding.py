#!/usr/bin/env python3
"""
backtest_top_reversal_holding.py
对 holding.md 所有标的（持仓+关注+观察）进行「阶段新高长上影+次日确认」策略的
完整历史回测，统计准确率，并为每只有信号的标的生成 K 线标注图。

准确定义（三个维度）：
  - 5日内最低跌幅 < -2%   → 短期有效
  - 10日内收盘跌幅 < -3%  → 中期有效
  - 20日内收盘跌幅 < -5%  → 中长期有效

输出：
  - 终端打印汇总准确率（按市场/score分层）
  - data/output/top_reversal/backtest/ 下每只标的一张图（带全部信号标注）

用法：
    python scripts/backtest_top_reversal_holding.py
    python scripts/backtest_top_reversal_holding.py --min-score 2  # 只看 score>=2 的信号
"""
from __future__ import annotations

import argparse
import re
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
from stock_ana.strategies.impl.top_reversal import scan_history
from _backtest_shared import (
    cjk_font_prop, make_ema_addplots, load_ohlcv, parse_holding_md,
    print_stats, print_per_sym_table,
    build_base_chart, add_chart_legend, add_stat_panel, set_chart_title, safe_save,
    EMA_COLORS,
)

OUT_DIR = OUTPUT_DIR / "top_reversal" / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDING_PATH = PROJECT_ROOT / "data" / "lists" / "holding.md"

MARKET_CACHE = {
    "US": CACHE_DIR / "us",
    "HK": CACHE_DIR / "hk",
    "CN": CACHE_DIR / "cn",
}

# 准确率判定阈值
ACC_THRESHOLDS = {
    "short_5d":  ("fwd_min_5d",  -2.0),
    "mid_10d":   ("fwd_ret_10d", -3.0),
    "long_20d":  ("fwd_ret_20d", -5.0),
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

    # 信号标记（Day-1）
    marker_high  = pd.Series(np.nan, index=df_plot.index)
    for _, row in hits.iterrows():
        if row["score"] < min_score:
            continue
        try:
            d1 = pd.Timestamp(row["signal_date"])
            if d1 in df_plot.index:
                pos = df_plot.index.get_loc(d1)
                marker_high.iloc[pos] = float(df_plot.iloc[pos]["high"]) * 1.018
        except Exception:
            pass

    mc = mpf.make_marketcolors(
        up="#CC3333", down="#00AA00", edge="inherit", wick="inherit", volume="in",
    )
    mpf_style = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=":", gridcolor="#E0E0E0",
        rc={"axes.unicode_minus": False},
    )

    if marker_high.notna().any():
        add_plots.append(mpf.make_addplot(
            marker_high, type="scatter", markersize=160,
            marker="v", color="#FF4444", edgecolors="#990000",
        ))

    fig, axes = mpf.plot(
        df_plot, type="candle", volume=True, style=mpf_style,
        addplot=add_plots, figsize=(24, 9), returnfig=True,
        tight_layout=False,
        scale_padding={"left": 0.03, "right": 0.30, "top": 0.6, "bottom": 0.5},
        warn_too_much_data=n_bars + 1,
    )
    ax = axes[0]

    # 标注每个信号的日期、上影线%、score 和后续表现
    for _, row in hits.iterrows():
        if row["score"] < min_score:
            continue
        try:
            d1 = pd.Timestamp(row["signal_date"])
            if d1 not in df_plot.index:
                continue
            pos = df_plot.index.get_loc(d1)
            high_val = float(df_plot.iloc[pos]["high"])

            ret_5  = row.get("fwd_min_5d", 0)
            ret_20 = row.get("fwd_ret_20d", 0)
            accurate = ret_5 <= -2.0 or ret_20 <= -5.0
            label_color = "#CC0000" if accurate else "#888888"

            star_mark  = "★" if row.get("is_shooting_star") else "▲"
            vol_mark   = "V" if row.get("vol_spike") else ""
            shadow_pct = row.get("shadow_pct", 0)   # 上影线/收盘价 %
            date_str   = str(row["signal_date"])[5:]  # MM-DD 格式节省空间

            label = (
                f"{date_str}\n"
                f"{star_mark}{vol_mark} s{row['score']} {shadow_pct:.1f}%\n"
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
        acc_5d  = (hits_filtered["fwd_min_5d"]  <= -2.0).sum()
        acc_10d = (hits_filtered["fwd_ret_10d"] <= -3.0).sum()
        acc_20d = (hits_filtered["fwd_ret_20d"] <= -5.0).sum()
        avg_ret_5  = hits_filtered["fwd_ret_5d"].mean()
        avg_min_5  = hits_filtered["fwd_min_5d"].mean()
        avg_ret_20 = hits_filtered["fwd_ret_20d"].mean()

        mode_cnts = hits_filtered["confirm_mode"].value_counts()
        star_cnt  = hits_filtered["is_shooting_star"].sum()
        vol_cnt   = hits_filtered["vol_spike"].sum()

        panel = [
            f"── 回测统计（score≥{min_score}）──",
            f"  信号总数: {total}",
            f"  射击之星: {star_cnt} / {total}",
            f"  放量:    {vol_cnt} / {total}",
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
            "",
            "── 确认模式 ──",
        ] + [f"  {k}: {v}" for k, v in mode_cnts.items()]
    else:
        panel = [f"无信号（score≥{min_score}）"]

    add_stat_panel(ax, panel, bg_color="#FFF3E0", edge_color="#FFCC80")

    set_chart_title(fig,
        f"{market}:{sym}  {name}  —  顶部信号回测（共{total}次，score≥{min_score}）")
    add_chart_legend(ax, strategy_label="Day-1 信号")

    _UNSAFE = set('/\\:*?"<>|')
    safe_name = "".join(c for c in name if c not in _UNSAFE).strip()[:15] or sym
    out_path = OUT_DIR / f"{market}_{sym}_{safe_name}.png"
    safe_save(fig, out_path)
    return out_path


# ═══════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="顶部反转策略历史回测（holding.md）")
    parser.add_argument("--min-score", type=int, default=1,
                        help="只统计 score >= N 的信号（默认 1=全部）")
    parser.add_argument("--no-chart", action="store_true",
                        help="跳过图表生成，只打印统计")
    args = parser.parse_args()

    items = _parse_holding_md()
    logger.info(f"holding.md 共 {len(items)} 只标的，开始回测 ...")

    all_hits: list[dict] = []
    chart_paths: list[Path] = []

    for item in items:
        sym, market, name = item["sym"], item["market"], item["name"]
        df = _load_ohlcv(sym, market)
        if df is None or len(df) < 30:
            logger.debug(f"{market}:{sym} 跳过（无缓存数据）")
            continue

        hits = scan_history(df, forward_days=(5, 10, 20))
        if hits.empty:
            logger.debug(f"{market}:{sym} 无历史信号")
            continue

        for _, row in hits.iterrows():
            all_hits.append({
                "sym": sym, "market": market, "name": name,
                "section": item["section"],
                **row.to_dict(),
            })

        logger.info(f"  {market}:{sym:8s} {name[:10]:10s}  信号 {len(hits)} 次")

        if not args.no_chart:
            try:
                path = _render_backtest_chart(sym, market, name, df, hits, args.min_score)
                chart_paths.append(path)
            except Exception as e:
                logger.error(f"{market}:{sym} 图表失败: {e}")

    if not all_hits:
        print("没有找到任何历史信号。")
        return

    df_all = pd.DataFrame(all_hits)
    filt = df_all[df_all["score"] >= args.min_score].copy()

    # ── 汇总统计 ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"  顶部反转策略 历史回测汇总  (score ≥ {args.min_score}，共 {len(filt)} 次信号)")
    print("═" * 70)

    def _stats(sub: pd.DataFrame, label: str) -> None:
        if sub.empty:
            return
        n = len(sub)
        a5  = (sub["fwd_min_5d"]  <= -2.0).sum()
        a10 = (sub["fwd_ret_10d"] <= -3.0).sum()
        a20 = (sub["fwd_ret_20d"] <= -5.0).sum()
        r5  = sub["fwd_ret_5d"].mean()
        r20 = sub["fwd_ret_20d"].mean()
        m5  = sub["fwd_min_5d"].mean()
        print(f"\n  [{label}]  n={n}")
        print(f"    准确率  5d>2%跌: {a5/n*100:5.1f}%  10d>3%跌: {a10/n*100:5.1f}%  20d>5%跌: {a20/n*100:5.1f}%")
        print(f"    均值    5d收盘: {r5:+.1f}%  5d最低: {m5:+.1f}%  20d收盘: {r20:+.1f}%")

    _stats(filt, "全部")

    for mkt in sorted(filt["market"].unique()):
        _stats(filt[filt["market"] == mkt], f"市场={mkt}")

    for sc in sorted(filt["score"].unique(), reverse=True):
        _stats(filt[filt["score"] == sc], f"score={sc}")

    # ── 按股票列出信号数 ──────────────────────────────────────────────────────
    print("\n\n  各标的信号明细：")
    print(f"  {'市场':<4} {'代码':<8} {'名称':<12} {'信号数':>4} {'5d准确%':>7} {'20d准确%':>8}  {'5d均值':>7} {'20d均值':>8}")
    print("  " + "-" * 68)
    per_sym = (
        filt.groupby(["market", "sym", "name"])
        .apply(lambda g: pd.Series({
            "n":    len(g),
            "acc5": (g["fwd_min_5d"] <= -2.0).mean() * 100,
            "acc20": (g["fwd_ret_20d"] <= -5.0).mean() * 100,
            "ret5": g["fwd_ret_5d"].mean(),
            "ret20": g["fwd_ret_20d"].mean(),
        }), include_groups=False)
        .reset_index()
        .sort_values("acc5", ascending=False)
    )
    for _, r in per_sym.iterrows():
        print(
            f"  {r['market']:<4} {r['sym']:<8} {str(r['name'])[:12]:<12}"
            f"  {int(r['n']):>4}  {r['acc5']:>6.0f}%  {r['acc20']:>7.0f}%"
            f"  {r['ret5']:>+6.1f}%  {r['ret20']:>+7.1f}%"
        )

    print(f"\n  图表已保存至：{OUT_DIR}  （共 {len(chart_paths)} 张）")


if __name__ == "__main__":
    main()
