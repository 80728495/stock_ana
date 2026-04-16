"""
VCP 摆幅收缩回测扫描器（MAG7 诊断工具）

方法：
  对每只股票，从 ~1 年前开始逐日扫描：
    1. 用 get_or_compute_peaks 获取预计算的宏观前高
    2. 取前高到当前日的区段，检查最近区域在 200 日线之上
    3. 对该区段调用 detect_vcp_micro_structure 检测摆幅逐次收窄
    4. 如果满足 → 输出一个 VCP 信号

测试目标：MAG7 (AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from loguru import logger

from stock_ana.data.peak_store import get_or_compute_peaks
from stock_ana.strategies.primitives.vcp import detect_vcp_micro_structure

MAG7 = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]
OUTPUT_DIR = Path("data/output/vcp_backtest")


def scan_vcp_backtest(
    ticker: str,
    df: pd.DataFrame,
    lookback_days: int = 252,
    min_gap: int = 40,
) -> list[dict]:
    """
    对单只股票做逐日 VCP 回测扫描。

    取「前高 → 当前日」的完整区段交给 detect_vcp_micro_structure，
    让它在该区段内自行找 swing highs 并检测摆幅收缩。
    """
    n = len(df)
    closes = df["close"].values.astype(float)

    sma200 = pd.Series(closes).rolling(200).mean().values

    bt_start = max(n - lookback_days, 250)
    if bt_start >= n:
        return []

    all_peaks = get_or_compute_peaks(ticker, df, force=True)
    if all_peaks.empty:
        return []
    peak_ilocs = [df.index.get_loc(d) for d in all_peaks.index]
    peak_prices = all_peaks["high"].values.astype(float)

    hits: list[dict] = []
    prev_signal_iloc = -999

    for cur_iloc in range(bt_start, n):
        if cur_iloc - prev_signal_iloc < 10:
            continue

        if np.isnan(sma200[cur_iloc]) or closes[cur_iloc] < sma200[cur_iloc]:
            continue

        # 找当前日之前最近的前高
        latest_peak_pos = -1
        for j in range(len(peak_ilocs) - 1, -1, -1):
            if peak_ilocs[j] <= cur_iloc:
                latest_peak_pos = j
                break
        if latest_peak_pos < 0:
            continue

        peak_iloc = peak_ilocs[latest_peak_pos]
        peak_price = float(peak_prices[latest_peak_pos])
        gap = cur_iloc - peak_iloc
        if gap < min_gap:
            continue

        # SMA200 检查：最近 20 天
        check_start = max(peak_iloc, cur_iloc - 20)
        region_closes = closes[check_start: cur_iloc + 1]
        region_sma200 = sma200[check_start: cur_iloc + 1]
        if np.any(np.isnan(region_sma200)):
            continue
        if np.any(region_closes < region_sma200 * 0.95):
            continue

        # 取前高→当前日的完整区段做 VCP 检测
        segment_df = df.iloc[peak_iloc: cur_iloc + 1]
        is_vcp, stats = detect_vcp_micro_structure(segment_df)

        if is_vcp:
            cur_date = df.index[cur_iloc]
            peak_date = df.index[peak_iloc]
            global_sh = [peak_iloc + si for si in stats.get("swing_high_indices", [])]
            # 收缩链起止在全局 df 中的 iloc
            chain_start_global = peak_iloc + stats["chain_start_iloc"] if stats["chain_start_iloc"] >= 0 else -1
            chain_end_global = peak_iloc + stats["chain_end_iloc"] if stats["chain_end_iloc"] >= 0 else -1
            hits.append({
                "ticker": ticker,
                "date": cur_date,
                "cur_iloc": cur_iloc,
                "peak_date": peak_date,
                "peak_iloc": peak_iloc,
                "peak_price": peak_price,
                "close": float(closes[cur_iloc]),
                "sma200": float(sma200[cur_iloc]),
                "gap_days": gap,
                "n_contractions": stats["consecutive_contractions"],
                "chain_ranges": stats["chain_ranges"],
                "chain_ratios": stats["chain_ratios"],
                "chain_start_global": chain_start_global,
                "chain_end_global": chain_end_global,
                "latest_close_vs_sma50_ratio": stats["latest_close_vs_sma50_ratio"],
                "global_swing_high_ilocs": global_sh,
            })
            prev_signal_iloc = cur_iloc
            chain_ranges_str = " → ".join(f"{r:.1f}" for r in stats["chain_ranges"])
            chain_ratios_str = " → ".join(f"{r:.0%}" for r in stats["chain_ratios"])
            logger.debug(
                f"  {ticker} VCP @ {cur_date.strftime('%Y-%m-%d')} "
                f"| peak={peak_date.strftime('%Y-%m-%d')}({peak_price:.1f}) "
                f"gap={gap}d  chain=[{chain_ranges_str}]  ratios=[{chain_ratios_str}]"
            )

    return hits


def plot_vcp_signals(
    ticker: str,
    df: pd.DataFrame,
    hits: list[dict],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """为一只股票的 VCP 信号画总览图 + 每个信号的局部放大图（含摆幅柱状图）。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hits:
        return

    n = len(df)
    sma200_full = df["close"].rolling(200).mean()
    sma50_full = df["close"].rolling(50).mean()
    style = mpf.make_mpf_style(base_mpf_style="charles")

    all_peaks = get_or_compute_peaks(ticker, df)

    # ── 总览图 ──
    earliest_peak = min(h["peak_iloc"] for h in hits)
    view_start = max(0, earliest_peak - 60)
    df_view = df.iloc[view_start:].copy()

    add_plots = []
    sma200_view = sma200_full.iloc[view_start:]
    sma50_view = sma50_full.iloc[view_start:]
    if not sma200_view.isna().all():
        add_plots.append(mpf.make_addplot(
            sma200_view, panel=0, color="purple", width=1.0,
            linestyle="dashed", secondary_y=False,
        ))
    if not sma50_view.isna().all():
        add_plots.append(mpf.make_addplot(
            sma50_view, panel=0, color="orange", width=1.0,
            linestyle="dashed", secondary_y=False,
        ))

    signal_marker = pd.Series(np.nan, index=df_view.index)
    for h in hits:
        pos = h["cur_iloc"] - view_start
        if 0 <= pos < len(df_view):
            signal_marker.iloc[pos] = df.iloc[h["cur_iloc"]]["low"] * 0.97
    if signal_marker.notna().any():
        add_plots.append(mpf.make_addplot(
            signal_marker, type="scatter", markersize=120,
            marker="^", color="lime", edgecolors="green", linewidths=1.5,
        ))

    peak_marker = pd.Series(np.nan, index=df_view.index)
    for d in all_peaks.index:
        if d in df_view.index:
            peak_marker.loc[d] = df.loc[d, "high"] * 1.03
    if peak_marker.notna().any():
        add_plots.append(mpf.make_addplot(
            peak_marker, type="scatter", markersize=120,
            marker="v", color="red", edgecolors="darkred", linewidths=1.5,
        ))

    save_path = output_dir / f"{ticker}_vcp_overview.png"
    mpf.plot(df_view, **{
        "type": "candle",
        "volume": True,
        "title": f"{ticker} VCP Signals ({len(hits)} hits)  ▲=VCP  ▼=peak  --SMA50/200",
        "style": style,
        "figscale": 1.5,
        "figratio": (20, 9),
        "addplot": add_plots,
        "savefig": str(save_path),
        "warn_too_much_data": len(df_view) + 1,
    })
    logger.info(f"总览图 → {save_path}")

    # ── 每个信号的局部放大图 + 摆幅柱状图 ──
    for i, h in enumerate(hits):
        local_start = max(0, h["peak_iloc"] - 10)
        local_end = min(n, h["cur_iloc"] + 10)
        df_local = df.iloc[local_start:local_end].copy()
        if len(df_local) < 10:
            continue

        sig_date = h["date"].strftime("%Y-%m-%d")
        ranges = h["chain_ranges"]
        ratios = h["chain_ratios"]
        ranges_str = " → ".join(f"{r:.1f}" for r in ranges)
        ratios_str = " → ".join(f"{r:.0%}" for r in ratios)

        # 先创建 axes
        fig = mpf.figure(figsize=(16, 10), style=style)
        ax_candle = fig.add_axes([0.06, 0.35, 0.88, 0.58])
        ax_vol = fig.add_axes([0.06, 0.18, 0.88, 0.15], sharex=ax_candle)
        ax_bar = fig.add_axes([0.06, 0.03, 0.88, 0.13])

        # 然后构建 addplots（需要 ax_candle 已存在）
        local_adds = []
        sma200_loc = sma200_full.iloc[local_start:local_end]
        sma50_loc = sma50_full.iloc[local_start:local_end]
        if not sma200_loc.isna().all():
            local_adds.append(mpf.make_addplot(
                sma200_loc, ax=ax_candle, color="purple", width=1.0,
                linestyle="dashed", secondary_y=False,
            ))
        if not sma50_loc.isna().all():
            local_adds.append(mpf.make_addplot(
                sma50_loc, ax=ax_candle, color="orange", width=1.0,
                linestyle="dashed", secondary_y=False,
            ))

        sig_mk = pd.Series(np.nan, index=df_local.index)
        pos = h["cur_iloc"] - local_start
        if 0 <= pos < len(df_local):
            sig_mk.iloc[pos] = df.iloc[h["cur_iloc"]]["low"] * 0.97
        if sig_mk.notna().any():
            local_adds.append(mpf.make_addplot(
                sig_mk, ax=ax_candle, type="scatter", markersize=200,
                marker="^", color="lime", edgecolors="green", linewidths=2,
            ))

        pk_mk = pd.Series(np.nan, index=df_local.index)
        pk_pos = h["peak_iloc"] - local_start
        if 0 <= pk_pos < len(df_local):
            pk_mk.iloc[pk_pos] = df.iloc[h["peak_iloc"]]["high"] * 1.03
        if pk_mk.notna().any():
            local_adds.append(mpf.make_addplot(
                pk_mk, ax=ax_candle, type="scatter", markersize=200,
                marker="v", color="red", edgecolors="darkred", linewidths=2,
            ))

        sh_mk = pd.Series(np.nan, index=df_local.index)
        for si in h.get("global_swing_high_ilocs", []):
            sh_pos = si - local_start
            if 0 <= sh_pos < len(df_local):
                sh_mk.iloc[sh_pos] = df.iloc[si]["high"] * 1.015
        if sh_mk.notna().any():
            local_adds.append(mpf.make_addplot(
                sh_mk, ax=ax_candle, type="scatter", markersize=80,
                marker="D", color="dodgerblue", edgecolors="navy", linewidths=1,
            ))

        # chain start / end 标记
        chain_start_mk = pd.Series(np.nan, index=df_local.index)
        cs_pos = h.get("chain_start_global", -1) - local_start
        if 0 <= cs_pos < len(df_local):
            chain_start_mk.iloc[cs_pos] = df.iloc[h["chain_start_global"]]["low"] * 0.97
        if chain_start_mk.notna().any():
            local_adds.append(mpf.make_addplot(
                chain_start_mk, ax=ax_candle, type="scatter", markersize=120,
                marker="|", color="magenta", linewidths=3,
            ))

        chain_end_mk = pd.Series(np.nan, index=df_local.index)
        ce_pos = h.get("chain_end_global", -1) - local_start
        if 0 <= ce_pos < len(df_local):
            chain_end_mk.iloc[ce_pos] = df.iloc[h["chain_end_global"]]["low"] * 0.97
        if chain_end_mk.notna().any():
            local_adds.append(mpf.make_addplot(
                chain_end_mk, ax=ax_candle, type="scatter", markersize=120,
                marker="|", color="magenta", linewidths=3,
            ))

        mpf.plot(df_local, type="candle", ax=ax_candle, volume=ax_vol,
                 addplot=local_adds, style=style,
                 warn_too_much_data=len(df_local) + 1)

        ax_candle.set_title(
            f"{ticker} VCP #{i+1} @ {sig_date}  "
            f"peak={h['peak_date'].strftime('%Y-%m-%d')}({h['peak_price']:.1f})  "
            f"gap={h['gap_days']}d\n"
            f"chain ranges=[{ranges_str}]  ratios=[{ratios_str}]",
            fontsize=10,
        )

        # 摆幅柱状图 — 只显示收缩链
        bar_x = list(range(len(ranges)))
        colors = ["green"] * len(ranges)
        ax_bar.bar(bar_x, ranges, color=colors, edgecolor="black", width=0.6)
        for k, rng in enumerate(ranges):
            ax_bar.text(k, rng, f"{rng:.1f}", ha="center", va="bottom", fontsize=8)
            if k > 0:
                ax_bar.text(k - 0.5, max(ranges) * 0.9,
                            f"{ratios[k-1]:.0%}", ha="center", va="top",
                            fontsize=8, color="navy", fontweight="bold")
        ax_bar.set_ylabel("Swing Range", fontsize=8)
        ax_bar.set_xticks(bar_x)
        ax_bar.set_xticklabels([f"T{k+1}" for k in bar_x], fontsize=8)

        local_path = output_dir / f"{ticker}_vcp_{i+1:02d}_{sig_date}.png"
        fig.savefig(str(local_path), dpi=100, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"{ticker}: {len(hits)} 个信号局部图已保存")


def main():
    """Backtest VCP detections on the configured MAG7 sample set and save local plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_hits: list[dict] = []
    for ticker in MAG7:
        path = Path(f"data/cache/us/{ticker}.parquet")
        if not path.exists():
            logger.warning(f"{ticker} 数据不存在，跳过")
            continue

        df = pd.read_parquet(path)
        logger.info(f"扫描 {ticker}: {len(df)} bars, {df.index[0].date()} ~ {df.index[-1].date()}")

        hits = scan_vcp_backtest(ticker, df, lookback_days=252)
        logger.info(f"  {ticker}: {len(hits)} VCP 信号")

        if hits:
            plot_vcp_signals(ticker, df, hits)
            all_hits.extend(hits)

    if all_hits:
        summary = pd.DataFrame(all_hits)
        summary["date"] = summary["date"].dt.strftime("%Y-%m-%d")
        summary["peak_date"] = summary["peak_date"].dt.strftime("%Y-%m-%d")
        summary["chain_ranges"] = summary["chain_ranges"].apply(
            lambda x: " → ".join(f"{r:.1f}" for r in x)
        )
        summary["chain_ratios"] = summary["chain_ratios"].apply(
            lambda x: " → ".join(f"{r:.0%}" for r in x)
        )
        cols = [
            "ticker", "date", "close", "peak_date", "peak_price",
            "gap_days", "n_contractions", "chain_ranges", "chain_ratios",
            "latest_close_vs_sma50_ratio",
        ]
        summary = summary[cols].sort_values(["ticker", "date"])
        csv_path = OUTPUT_DIR / "mag7_vcp_signals.csv"
        summary.to_csv(csv_path, index=False)
        logger.success(f"汇总 CSV → {csv_path}")
        print(f"\n=== MAG7 VCP 信号汇总 ({len(all_hits)} 个) ===")
        print(summary.to_string(index=False))
    else:
        logger.warning("MAG7 中未发现 VCP 信号")


if __name__ == "__main__":
    main()
