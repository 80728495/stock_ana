#!/usr/bin/env python3
"""
Shawn List 三浪结构回归分析 + 图表绘制

对每只持仓标的：
1. 调用 analyze_wave_structure 识别大浪与子浪
2. 输出结构化统计 CSV
3. 绘制带浪标注的 K 线图
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mplfinance as mpf
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from stock_ana.strategies.primitives.wave import (
    detect_ema8_swings,
    analyze_wave_structure,
)
from stock_ana.config import CACHE_DIR, OUTPUT_DIR

# ── 样本列表 ──
SAMPLES = {
    # symbol: (market, name, parquet_path)
    "APP":   ("US", "APP",       CACHE_DIR / "us" / "APP.parquet"),
    "NVDA":  ("US", "NVDA",      CACHE_DIR / "us" / "NVDA.parquet"),
    "META":  ("US", "Meta",      CACHE_DIR / "us" / "META.parquet"),
    "TSLA":  ("US", "Tesla",     CACHE_DIR / "us" / "TSLA.parquet"),
    "AMD":   ("US", "AMD",       CACHE_DIR / "us" / "AMD.parquet"),
    "RBLX":  ("US", "ROBLOX",    CACHE_DIR / "us" / "RBLX.parquet"),
    "MRNA":  ("US", "MRNA",      CACHE_DIR / "us" / "MRNA.parquet"),
    "MU":    ("US", "MU",        CACHE_DIR / "us" / "MU.parquet"),
    "TEM":   ("US", "TempusAI",  CACHE_DIR / "us" / "TEM.parquet"),
    "ALAB":  ("US", "ALAB",      CACHE_DIR / "us" / "ALAB.parquet"),
    "PDD":   ("US", "PDD",       CACHE_DIR / "us" / "PDD.parquet"),
    "MSFT":  ("US", "MSFT",      CACHE_DIR / "us" / "MSFT.parquet"),
    "GOOG":  ("US", "GOOG",      CACHE_DIR / "us" / "GOOG.parquet"),
    "09992": ("HK", "POP MART",  CACHE_DIR / "hk" / "09992.parquet"),
    "00700": ("HK", "Tencent",   CACHE_DIR / "hk" / "00700.parquet"),
    "01810": ("HK", "Xiaomi",    CACHE_DIR / "hk" / "01810.parquet"),
    "09988": ("HK", "Alibaba",   CACHE_DIR / "hk" / "09988.parquet"),
    "00981": ("HK", "SMIC",      CACHE_DIR / "hk" / "00981.parquet"),
    "03690": ("HK", "Meituan",   CACHE_DIR / "hk" / "03690.parquet"),
    "01347": ("HK", "Hua Hong",  CACHE_DIR / "hk" / "01347.parquet"),
    "02400": ("HK", "XD Inc",    CACHE_DIR / "hk" / "02400.parquet"),
    "01024": ("HK", "Kuaishou",  CACHE_DIR / "hk" / "01024.parquet"),
    "09626": ("HK", "Bilibili",  CACHE_DIR / "hk" / "09626.parquet"),
    "00189": ("HK", "Dongyue",   CACHE_DIR / "hk" / "00189.parquet"),
    "06869": ("HK", "CF Fiber",  CACHE_DIR / "hk" / "06869.parquet"),
    "02228": ("HK", "Jingta",    CACHE_DIR / "hk" / "02228.parquet"),
    "02788": ("HK", "ChuangXin", CACHE_DIR / "hk" / "02788.parquet"),
}

CHART_DIR = OUTPUT_DIR / "main_rally_pullback" / "wave_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# 大浪颜色（循环使用）
WAVE_COLORS = ["#4FC3F7", "#81C784", "#FFB74D", "#E57373", "#BA68C8", "#4DB6AC"]

style = mpf.make_mpf_style(base_mpf_style="charles", rc={"font.size": 8})


def _load(path: Path) -> pd.DataFrame | None:
    """Load one parquet OHLCV file into a normalized, date-indexed DataFrame."""
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def plot_wave_chart(
    sym: str,
    market: str,
    name: str,
    df: pd.DataFrame,
    result: dict,
    save_path: Path,
    view_bars: int = 500,
):
    """绘制带大浪/子浪标注的 K 线图。"""
    vb = min(view_bars, len(df))
    df_view = df.iloc[-vb:].copy()
    view_offset = len(df) - vb

    close_full = df["close"].astype(float)
    ema8_f  = close_full.ewm(span=8,   adjust=False).mean()
    ema34_f = close_full.ewm(span=34,  adjust=False).mean()
    ema55_f = close_full.ewm(span=55,  adjust=False).mean()
    ema144_f= close_full.ewm(span=144, adjust=False).mean()
    ema169_f= close_full.ewm(span=169, adjust=False).mean()

    add_plots = [
        mpf.make_addplot(ema8_f.iloc[-vb:],   panel=0, color="#B0B0B0", width=0.7, secondary_y=False),
        mpf.make_addplot(ema34_f.iloc[-vb:],  panel=0, color="#2E8B57", width=1.2, secondary_y=False),
        mpf.make_addplot(ema55_f.iloc[-vb:],  panel=0, color="#FF6600", width=1.5, secondary_y=False),
        mpf.make_addplot(ema144_f.iloc[-vb:], panel=0, color="#0066CC", width=1.0, linestyle="dashed", secondary_y=False),
        mpf.make_addplot(ema169_f.iloc[-vb:], panel=0, color="#9900CC", width=1.0, linestyle="dashed", secondary_y=False),
    ]

    # Zigzag line
    all_pivots = result["all_pivots"]
    view_pivots = [p for p in all_pivots if view_offset <= p["iloc"] < view_offset + vb]

    zz_line = pd.Series(np.nan, index=df_view.index)
    for p in view_pivots:
        vi = p["iloc"] - view_offset
        zz_line.iloc[vi] = p["value"]
    zz_line = zz_line.interpolate(method="index")
    if view_pivots:
        first_vi = view_pivots[0]["iloc"] - view_offset
        last_vi = view_pivots[-1]["iloc"] - view_offset
        zz_line.iloc[:first_vi] = np.nan
        if last_vi + 1 < len(zz_line):
            zz_line.iloc[last_vi + 1:] = np.nan
    if zz_line.notna().any():
        add_plots.append(mpf.make_addplot(
            zz_line, panel=0, color="#FFD700", width=1.2, secondary_y=False))

    # H/L markers
    h_marker = pd.Series(np.nan, index=df_view.index)
    l_marker = pd.Series(np.nan, index=df_view.index)
    for p in view_pivots:
        vi = p["iloc"] - view_offset
        if p["type"] == "H":
            h_marker.iloc[vi] = p["value"] * 1.012
        else:
            l_marker.iloc[vi] = p["value"] * 0.988
    if h_marker.notna().any():
        add_plots.append(mpf.make_addplot(
            h_marker, type="scatter", markersize=40,
            marker="v", color="#FF3333", edgecolors="#CC0000", linewidths=0.6))
    if l_marker.notna().any():
        add_plots.append(mpf.make_addplot(
            l_marker, type="scatter", markersize=40,
            marker="^", color="#33CC33", edgecolors="#009900", linewidths=0.6))

    # Major wave start markers (big blue diamonds)
    wave_start_marker = pd.Series(np.nan, index=df_view.index)
    for mw in result["major_waves"]:
        sp = mw["start_pivot"]
        if view_offset <= sp["iloc"] < view_offset + vb:
            vi = sp["iloc"] - view_offset
            wave_start_marker.iloc[vi] = sp["value"] * 0.96
    if wave_start_marker.notna().any():
        add_plots.append(mpf.make_addplot(
            wave_start_marker, type="scatter", markersize=200,
            marker="D", color="#2196F3", edgecolors="#0D47A1", linewidths=1.5))

    # Build title
    waves = result["major_waves"]
    wave_desc_parts = []
    for mw in waves:
        sub_cnt = mw["sub_wave_count"]
        mid_cnt = mw["mid_pullback_count"]
        wave_desc_parts.append(
            f"W{mw['wave_number']}: +{mw['rise_pct']:.0f}% {mw['duration_days']}d "
            f"({sub_cnt}sub/{mid_cnt}mid)"
        )
    wave_desc = "  |  ".join(wave_desc_parts) if wave_desc_parts else "no waves"

    title = (
        f"{market}:{sym} {name}  "
        f"[{result['current_status']}  wave={result['current_wave_number']}  "
        f"sub={result['current_sub_wave']}]\n"
        f"{wave_desc}"
    )

    fig, axes = mpf.plot(
        df_view, type="candle", volume=True,
        title=title, style=style,
        figscale=1.6, figratio=(22, 9),
        addplot=add_plots,
        savefig=str(save_path),
        warn_too_much_data=vb + 1,
        returnfig=True,
    )

    # Annotate wave numbers on chart
    ax = axes[0]
    for mw in waves:
        sp = mw["start_pivot"]
        if view_offset <= sp["iloc"] < view_offset + vb:
            vi = sp["iloc"] - view_offset
            color = WAVE_COLORS[(mw["wave_number"] - 1) % len(WAVE_COLORS)]
            ax.annotate(
                f"W{mw['wave_number']}",
                xy=(vi, sp["value"] * 0.94),
                fontsize=9, fontweight="bold", color=color,
                ha="center", va="top",
            )
        # Sub-wave labels
        for sw in mw["sub_waves"]:
            sp2 = sw["start_pivot"]
            if view_offset <= sp2["iloc"] < view_offset + vb:
                vi2 = sp2["iloc"] - view_offset
                label = f"s{sw['sub_number']}"
                if sw["pullback_type"] == "mid_vegas":
                    label += f"({sw['pullback_band'][-2:]})"
                ax.annotate(
                    label,
                    xy=(vi2, sp2["value"] * 0.97),
                    fontsize=7, color="#666666",
                    ha="center", va="top",
                )

    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close("all")


def main():
    """Run wave-structure analysis for the configured samples and export charts and CSVs."""
    all_rows: list[dict] = []

    for sym, (market, name, path) in SAMPLES.items():
        df = _load(path)
        if df is None or len(df) < 100:
            logger.warning(f"{market}:{sym} skip (no data or too short)")
            continue

        result = analyze_wave_structure(df)
        waves = result["major_waves"]

        logger.info(
            f"{market}:{sym}  waves={len(waves)}  "
            f"current: W{result['current_wave_number']} sub={result['current_sub_wave']}  "
            f"status={result['current_status']}"
        )

        # Per-wave detail
        for mw in waves:
            row = {
                "market": market,
                "symbol": sym,
                "name": name,
                "wave_number": mw["wave_number"],
                "start_date": mw["start_pivot"]["date"],
                "start_price": mw["start_pivot"]["value"],
                "peak_date": mw["peak_pivot"]["date"],
                "peak_price": mw["peak_pivot"]["value"],
                "end_date": mw["end_pivot"]["date"] if mw["end_pivot"] else "",
                "rise_pct": mw["rise_pct"],
                "duration_days": mw["duration_days"],
                "sub_wave_count": mw["sub_wave_count"],
                "mid_pullback_count": mw["mid_pullback_count"],
                "current_status": result["current_status"] if mw == waves[-1] else "completed",
            }
            all_rows.append(row)

            for sw in mw["sub_waves"]:
                logger.debug(
                    f"  W{mw['wave_number']}.s{sw['sub_number']}: "
                    f"+{sw['rise_pct']:.1f}%  pullback={sw['pullback_type']}/{sw['pullback_band']}"
                )

        # Chart
        chart_path = CHART_DIR / f"{market}_{sym}_waves.png"
        plot_wave_chart(sym, market, name, df, result, chart_path)
        logger.info(f"  chart -> {chart_path}")

    # Summary CSV
    if all_rows:
        out_df = pd.DataFrame(all_rows)
        csv_path = OUTPUT_DIR / "main_rally_pullback" / "wave_structure_summary.csv"
        out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"Summary saved: {csv_path}")

        print("\n" + "=" * 90)
        print("Wave Structure Summary")
        print("=" * 90)
        print(out_df.to_string(index=False))

        # Aggregate stats
        print("\n\nPattern Statistics:")
        print(f"  Total major waves: {len(out_df)}")
        print(f"  Avg sub-waves per major wave: {out_df['sub_wave_count'].mean():.1f}")
        print(f"  Avg mid-pullbacks per major wave: {out_df['mid_pullback_count'].mean():.1f}")
        print(f"  Avg rise per major wave: {out_df['rise_pct'].mean():.1f}%")
        print(f"  Avg duration per major wave: {out_df['duration_days'].mean():.0f} bars")
        print(f"\n  Sub-wave distribution:")
        for n in sorted(out_df["sub_wave_count"].unique()):
            cnt = (out_df["sub_wave_count"] == n).sum()
            print(f"    {n} sub-waves: {cnt} waves ({cnt/len(out_df)*100:.0f}%)")
        print(f"\n  Mid-pullback distribution:")
        for n in sorted(out_df["mid_pullback_count"].unique()):
            cnt = (out_df["mid_pullback_count"] == n).sum()
            print(f"    {n} mid-pullbacks: {cnt} waves ({cnt/len(out_df)*100:.0f}%)")


if __name__ == "__main__":
    main()
