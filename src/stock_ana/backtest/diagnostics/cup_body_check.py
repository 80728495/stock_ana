"""扫描全部美股，只找杯身结构（跳过手柄/质量过滤），画图验证 P1/P2/P3。"""

from pathlib import Path

import numpy as np
import pandas as pd
import mplfinance as mpf
from loguru import logger

from stock_ana.data.market_data import load_market_data
from stock_ana.strategies.primitives import check_cup_ma_trend, find_cup_structure

OUTPUT_DIR = Path("data/output/cup_body_check")


def scan_cup_body_only(
    min_base_days: int = 60,
    max_base_days: int = 300,
) -> list[dict]:
    """只做均线过滤 + 杯身结构识别，跳过手柄/质量/收缩等后续校验。"""

    stock_data = load_market_data("us")
    if not stock_data:
        logger.error("无数据")
        return []

    hits: list[dict] = []
    processed = 0

    for ticker, df in stock_data.items():
        try:
            if len(df) < 300:
                continue
            processed += 1

            n = len(df)
            search_start = max(n - max_base_days, 0)
            search_end = n - 15
            if search_end - search_start < min_base_days:
                continue

            lookback = df.iloc[search_start:search_end]
            left_idx = lookback["high"].idxmax()
            left_val = float(lookback.loc[left_idx, "high"])
            left_iloc = df.index.get_loc(left_idx)

            passed, reason = check_cup_ma_trend(df, left_val, left_iloc)
            if not passed:
                continue

            cup = find_cup_structure(
                df,
                search_start_iloc=search_start,
                search_end_iloc=search_end,
                data_end_iloc=n - 1,
                symmetry_tol=0.05,
                min_depth_pct=15.0,
                max_depth_pct=33.0,
                min_weeks=7,
                max_weeks=65,
            )
            if cup is None:
                continue

            # P1 MA 复核
            p1_iloc = cup["p1_iloc"]
            if p1_iloc != left_iloc:
                ok, _ = check_cup_ma_trend(df, cup["p1_val"], p1_iloc)
                if not ok:
                    continue

            logger.success(
                f"✅ {ticker} 杯身 | P1={cup['p1_val']:.2f} P2={cup['p2_val']:.2f} "
                f"P3={cup['p3_val']:.2f} | 深={cup['depth_pct']:.1f}% "
                f"sym={cup['symmetry_pct']:+.1f}% days={cup['cup_days']}"
            )
            hits.append({"ticker": ticker, "df": df, "cup": cup})

        except Exception as e:
            logger.error(f"{ticker}: {e}")

    logger.info(f"杯身扫描：{len(stock_data)} 只 → 有效 {processed} → 命中 {len(hits)}")
    return hits


def plot_cup_body(hits: list[dict]) -> None:
    """为每只命中股绘制 K 线 + SMA50/150/200 + P1/P2/P3 标注。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for item in hits:
        ticker = item["ticker"]
        cup = item["cup"]
        df_full = item["df"].copy()

        p1_iloc = cup["p1_iloc"]
        p2_iloc = cup["p2_iloc"]
        p3_iloc = cup["p3_iloc"]

        # 视窗：P1 前 60 根 ~ 数据末尾 + 5
        pre = 60
        view_start = max(0, p1_iloc - pre)
        view_end = min(len(df_full), len(df_full))
        df = df_full.iloc[view_start:view_end].copy()
        if len(df) < 20:
            continue

        def _vp(abs_iloc: int) -> int:
            """Translate an absolute iloc from the full frame into the plotted view window."""
            return abs_iloc - view_start

        # ── 均线 ──
        sma50 = df_full["close"].rolling(50).mean().iloc[view_start:view_end]
        sma150 = df_full["close"].rolling(150).mean().iloc[view_start:view_end]
        sma200 = df_full["close"].rolling(200).mean().iloc[view_start:view_end]

        add_plots = []
        for sma, color, label in [
            (sma50, "orange", "SMA50"),
            (sma150, "steelblue", "SMA150"),
            (sma200, "purple", "SMA200"),
        ]:
            if not sma.isna().all():
                add_plots.append(mpf.make_addplot(
                    sma, panel=0, color=color, width=1.0,
                    linestyle="dashed", secondary_y=False,
                ))

        # ── P1 绿色 ▲ ──
        p1_marker = pd.Series(np.nan, index=df.index)
        pos = _vp(p1_iloc)
        if 0 <= pos < len(df):
            p1_marker.iloc[pos] = df_full.iloc[p1_iloc]["high"] * 1.03
        if p1_marker.notna().any():
            add_plots.append(mpf.make_addplot(
                p1_marker, type="scatter", markersize=200,
                marker="^", color="lime", edgecolors="green", linewidths=2,
            ))

        # ── P2 蓝色 ▼ ──
        p2_marker = pd.Series(np.nan, index=df.index)
        pos = _vp(p2_iloc)
        if 0 <= pos < len(df):
            p2_marker.iloc[pos] = df_full.iloc[p2_iloc]["low"] * 0.97
        if p2_marker.notna().any():
            add_plots.append(mpf.make_addplot(
                p2_marker, type="scatter", markersize=200,
                marker="v", color="deepskyblue", edgecolors="navy", linewidths=2,
            ))

        # ── P3 橙色 ▲ ──
        p3_marker = pd.Series(np.nan, index=df.index)
        pos = _vp(p3_iloc)
        if 0 <= pos < len(df):
            p3_marker.iloc[pos] = df_full.iloc[p3_iloc]["high"] * 1.03
        if p3_marker.notna().any():
            add_plots.append(mpf.make_addplot(
                p3_marker, type="scatter", markersize=200,
                marker="^", color="orange", edgecolors="darkorange", linewidths=2,
            ))

        # ── 枢轴线（P1 价格水平线）──
        pivot_line = pd.Series(cup["p1_val"], index=df.index)
        add_plots.append(mpf.make_addplot(
            pivot_line, panel=0, color="magenta", width=1.2,
            linestyle="--", secondary_y=False,
        ))

        style = mpf.make_mpf_style(base_mpf_style="charles")
        save_path = OUTPUT_DIR / f"{ticker}_cup_body.png"

        mpf.plot(df, **{
            "type": "candle",
            "volume": True,
            "title": (
                f"{ticker}  Cup Body: P1={cup['p1_val']:.1f} "
                f"P2={cup['p2_val']:.1f} P3={cup['p3_val']:.1f}  "
                f"depth={cup['depth_pct']:.1f}%  sym={cup['symmetry_pct']:+.1f}%  "
                f"days={cup['cup_days']}  "
                f"▲绿=P1  ▼蓝=P2  ▲橙=P3  --SMA50/150/200"
            ),
            "style": style,
            "figscale": 1.5,
            "figratio": (18, 9),
            "addplot": add_plots,
            "savefig": str(save_path),
            "warn_too_much_data": len(df) + 1,
        })
        logger.info(f"已保存 → {save_path}")


if __name__ == "__main__":
    hits = scan_cup_body_only()
    if hits:
        plot_cup_body(hits)
    else:
        logger.warning("未找到杯身结构")
