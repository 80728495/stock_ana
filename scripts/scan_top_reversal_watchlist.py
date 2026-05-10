#!/usr/bin/env python3
"""
scan_top_reversal_watchlist.py
扫描 watchlist.md 所有标的，检测「阶段新高 + 长上影 + 次日确认」顶部反转形态，
对有信号的标的生成 K 线图，保存至 data/output/top_reversal/。

用法：
    python scripts/scan_top_reversal_watchlist.py              # 扫描全部
    python scripts/scan_top_reversal_watchlist.py --lookback 5 # 最近5日内有信号均显示
    python scripts/scan_top_reversal_watchlist.py --only-triggered  # 只生成有信号的图
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
from stock_ana.data.list_manager import parse_watchlist
from stock_ana.strategies.impl.top_reversal import detect_high_shadow_reversal

# ── 输出目录 ──────────────────────────────────────────────────────────────────
OUT_DIR = OUTPUT_DIR / "top_reversal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── CJK 字体 ──────────────────────────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = [
    "Heiti TC", "PingFang HK", "STHeiti", "Songti SC",
    "Arial Unicode MS", "Arial",
]
plt.rcParams["axes.unicode_minus"] = False

_EMA_COLORS = {
    34:  "#4FC3F7",
    55:  "#29B6F6",
    144: "#FF8A65",
    200: "#E64A19",
}

MARKET_CACHE = {
    "us": CACHE_DIR / "us",
    "hk": CACHE_DIR / "hk",
    "cn": CACHE_DIR / "cn",
}


# ═══════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════

def _load_ohlcv(sym: str, market: str) -> pd.DataFrame | None:
    path = MARKET_CACHE[market.lower()] / f"{sym}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        return df
    except Exception as e:
        logger.warning(f"{sym} 数据加载失败: {e}")
        return None


def _load_watchlist() -> list[dict]:
    wl = parse_watchlist()
    items = []
    for mkt, entries in wl.items():
        for e in entries:
            sym = e["symbol"]
            # 过滤掉指数和期货（无价格缓存）
            if any(x in sym for x in ["800000", "800700", "HTI"]):
                continue
            items.append({
                "sym": sym,
                "market": mkt.upper(),
                "name": e.get("name") or e.get("name_cn") or sym,
            })
    return items


# ═══════════════════════════════════════════════════════
#  图表渲染
# ═══════════════════════════════════════════════════════

def _render_chart(
    sym: str,
    market: str,
    name: str,
    df: pd.DataFrame,
    result: dict,
    lookback_bars: int = 60,
) -> Path:
    """渲染顶部反转信号 K 线图，标注 Day-1 / Day-2 信号位置。"""
    df_plot = df.iloc[-lookback_bars:].copy()[["open", "high", "low", "close", "volume"]]

    # EMA 叠加（在完整数据上计算后截取）
    add_plots = []
    for span, color in _EMA_COLORS.items():
        ema_full = df["close"].ewm(span=span, adjust=False).mean()
        ema_view = ema_full.reindex(df_plot.index)
        add_plots.append(mpf.make_addplot(
            ema_view, color=color, width=1.2,
            linestyle="-" if span >= 144 else "--",
        ))

    # 标记 Day-1（信号日）：箭头在高价之上
    marker_d1 = pd.Series(np.nan, index=df_plot.index)
    d1_date = pd.Timestamp(result["signal_date"])
    if d1_date in df_plot.index:
        pos = df_plot.index.get_loc(d1_date)
        marker_d1.iloc[pos] = float(df_plot.iloc[pos]["high"]) * 1.015
    if marker_d1.notna().any():
        add_plots.append(mpf.make_addplot(
            marker_d1, type="scatter", markersize=180,
            marker="v", color="#FF4444", edgecolors="#990000",
        ))

    # 标记 Day-2（确认日）：小圆点在收盘之上
    marker_d2 = pd.Series(np.nan, index=df_plot.index)
    d2_date = pd.Timestamp(result["confirm_date"])
    if d2_date in df_plot.index:
        pos2 = df_plot.index.get_loc(d2_date)
        marker_d2.iloc[pos2] = float(df_plot.iloc[pos2]["high"]) * 1.015
    if marker_d2.notna().any():
        add_plots.append(mpf.make_addplot(
            marker_d2, type="scatter", markersize=130,
            marker="v", color="#FF8800", edgecolors="#994400",
        ))

    # vol_ma_20 叠加在成交量面板
    if "vol_ma_20" in df.columns:
        vol_ma = df["vol_ma_20"].reindex(df_plot.index)
        add_plots.append(mpf.make_addplot(
            vol_ma, panel=1, color="#AAAAAA", width=1.0, linestyle="--",
            secondary_y=False,
        ))

    mc = mpf.make_marketcolors(
        up="#CC3333", down="#00AA00",
        edge="inherit", wick="inherit", volume="in",
    )
    mpf_style = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=":", gridcolor="#E0E0E0",
        rc={
            "font.sans-serif": plt.rcParams["font.sans-serif"],
            "axes.unicode_minus": False,
        },
    )

    score = result["score"]
    score_color = "#CC0000" if score >= 3 else "#FF8800" if score >= 2 else "#888888"
    triggered_label = "⚠ 顶部反转信号" if result["triggered"] else "无信号（历史回溯）"

    fig, axes = mpf.plot(
        df_plot, type="candle", volume=True, style=mpf_style,
        addplot=add_plots, figsize=(20, 9), returnfig=True,
        tight_layout=False,
        scale_padding={"left": 0.05, "right": 0.32, "top": 0.6, "bottom": 0.5},
        warn_too_much_data=lookback_bars + 1,
    )
    ax = axes[0]

    # 标题
    fig.suptitle(
        f"{market}:{sym}  {name}  [{triggered_label}  score={score}]",
        fontsize=16, fontweight="bold", y=0.98, color=score_color,
    )
    ax.set_title(result.get("reason", ""), fontsize=10, color="#555555", pad=4)

    # 注释 Day-1
    if d1_date in df_plot.index:
        pos = df_plot.index.get_loc(d1_date)
        high_val = float(df_plot.iloc[pos]["high"])
        star_label = "射击之星" if result.get("is_shooting_star") else "长上影"
        vol_label = "📈放量" if result.get("day1_vol_spike") else ""
        ax.annotate(
            f"Day-1\n{star_label}\n×{result.get('day1_upper_shadow_ratio', '')} {vol_label}",
            xy=(pos, high_val * 1.02),
            fontsize=8, fontweight="bold", color="#CC0000",
            ha="center", va="bottom",
        )

    # 注释 Day-2
    if d2_date in df_plot.index:
        pos2 = df_plot.index.get_loc(d2_date)
        high_val2 = float(df_plot.iloc[pos2]["high"])
        mode_short = {
            "engulf_open":    "完全吞没",
            "below_midpoint": "跌破中点",
            "bearish_close":  "阴线收低",
        }.get(result.get("confirm_mode", ""), "确认")
        ax.annotate(
            f"Day-2\n{mode_short}",
            xy=(pos2, high_val2 * 1.02),
            fontsize=8, fontweight="bold", color="#FF6600",
            ha="center", va="bottom",
        )

    # 右侧信息面板
    panel_lines = [
        "── 顶部反转信号 ──",
        f"  信号日: {result.get('signal_date', '-')}",
        f"  确认日: {result.get('confirm_date', '-')}",
        "",
        "── Day-1 条件 ──",
        f"  阶段新高: {result.get('day1_new_high_n', '-')} 日",
        f"  上影/实体: {result.get('day1_upper_shadow_ratio', '-')}x",
        f"  射击之星: {'是' if result.get('is_shooting_star') else '否'}",
        f"  放量: {'是' if result.get('day1_vol_spike') else '否'}",
        "",
        "── Day-2 确认 ──",
        f"  模式: {result.get('confirm_mode', '未触发')}",
        "",
        f"  综合评分: {score} / 4",
    ]
    ax.text(
        1.01, 0.99, "\n".join(panel_lines),
        transform=ax.transAxes,
        fontsize=8.5, ha="left", va="top",
        fontfamily=plt.rcParams["font.sans-serif"][0],
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="#FFF3E0",
            alpha=0.93, edgecolor="#FFCC80", linewidth=0.8,
        ),
    )

    # 图例
    from matplotlib.lines import Line2D
    legend_handles = []
    for span, color in _EMA_COLORS.items():
        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=1.5, label=f"EMA{span}")
        )
    legend_handles.append(
        Line2D([0], [0], marker="v", color="w",
               markerfacecolor="#FF4444", markeredgecolor="#990000",
               markersize=9, label="Day-1 信号")
    )
    legend_handles.append(
        Line2D([0], [0], marker="v", color="w",
               markerfacecolor="#FF8800", markeredgecolor="#994400",
               markersize=9, label="Day-2 确认")
    )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.8)

    # 保存
    _UNSAFE = set('/\\:*?"<>|')
    safe_name = "".join(c for c in name if c not in _UNSAFE).strip()[:20] or sym
    triggered_str = f"score{score}" if result["triggered"] else "no_signal"
    out_path = OUT_DIR / f"{market}_{sym}_{safe_name}_{triggered_str}.png"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Glyph.*missing from font")
        fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ═══════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="顶部反转形态扫描（watchlist）")
    parser.add_argument("--lookback", type=int, default=1,
                        help="向前回溯天数：检测最近 N 个交易日内是否出现信号（默认 1=仅最新）")
    parser.add_argument("--only-triggered", action="store_true",
                        help="只为有信号的标的生成图表（默认：有无信号都生成）")
    parser.add_argument("--chart-bars", type=int, default=60,
                        help="图表显示最近 N 根 K 线（默认 60）")
    args = parser.parse_args()

    items = _load_watchlist()
    logger.info(f"Watchlist 共 {len(items)} 只标的，开始扫描 ...")

    triggered_list: list[dict] = []
    chart_paths: list[Path] = []

    for item in items:
        sym, market, name = item["sym"], item["market"], item["name"]
        df = _load_ohlcv(sym, market)
        if df is None or len(df) < 30:
            logger.debug(f"{market}:{sym} 跳过（无数据或数据不足）")
            continue

        # 确保 vol_ma_20 存在
        if "vol_ma_20" not in df.columns:
            df["vol_ma_20"] = df["volume"].astype(float).rolling(20, min_periods=1).mean()

        # 支持 lookback > 1（回溯多日）
        result = None
        for offset in range(args.lookback):
            sub_df = df.iloc[:len(df) - offset] if offset > 0 else df
            if len(sub_df) < 22:
                continue
            r = detect_high_shadow_reversal(sub_df)
            if r["triggered"]:
                result = r
                break

        if result is None:
            result = detect_high_shadow_reversal(df)  # 取最后一次（可能 triggered=False）

        if result["triggered"]:
            triggered_list.append({
                "symbol": sym, "market": market, "name": name,
                **{k: v for k, v in result.items() if k != "triggered"},
            })
            logger.info(
                f"[触发] {market}:{sym} {name}  score={result['score']}  {result['reason']}"
            )

        # 生成图表
        if result["triggered"] or not args.only_triggered:
            try:
                path = _render_chart(sym, market, name, df, result,
                                     lookback_bars=args.chart_bars)
                chart_paths.append(path)
            except Exception as e:
                logger.error(f"{market}:{sym} 图表生成失败: {e}")

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"扫描完成：{len(items)} 只标的  |  触发信号：{len(triggered_list)} 只")
    print("═" * 60)
    if triggered_list:
        print("\n触发标的汇总：")
        for t in sorted(triggered_list, key=lambda x: -x["score"]):
            print(
                f"  [{t['market']}] {t['symbol']:8s} {t['name'][:10]:10s}"
                f"  score={t['score']}  {t['confirm_mode']:16s}  {t['reason'][:60]}"
            )
    print(f"\n图表已保存至：{OUT_DIR}  （共 {len(chart_paths)} 张）")


if __name__ == "__main__":
    main()
