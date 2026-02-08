"""
可视化模块 - K线图与技术指标绘图
"""

from pathlib import Path

import numpy as np
import mplfinance as mpf
import pandas as pd
from loguru import logger


def plot_candlestick(
    df: pd.DataFrame,
    title: str = "K线图",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    绘制带成交量的K线图

    Args:
        df: 包含 open, high, low, close, volume 的 DataFrame（index 为日期）
        title: 图表标题
        save_path: 保存路径（可选）
        show: 是否显示图表
    """
    style = mpf.make_mpf_style(base_mpf_style="charles", rc={"font.size": 10})

    kwargs = {
        "type": "candle",
        "volume": True,
        "title": title,
        "style": style,
        "figscale": 1.2,
    }

    if save_path:
        kwargs["savefig"] = str(save_path)
    if show:
        kwargs["show_nontrading"] = False

    mpf.plot(df, **kwargs)


def plot_with_indicators(
    df: pd.DataFrame,
    title: str = "技术分析",
    indicators: list[str] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """
    绘制带技术指标的K线图

    Args:
        df: 带有技术指标列的 DataFrame
        title: 图表标题
        indicators: 要绘制的指标列名列表
        save_path: 保存路径
    """
    if indicators is None:
        indicators = ["sma_5", "sma_20", "sma_60"]

    add_plots = []

    # 主图叠加指标（均线、布林带）
    main_indicators = [col for col in indicators if col in df.columns and col.startswith(("sma_", "ema_", "bb_"))]
    for col in main_indicators:
        add_plots.append(mpf.make_addplot(df[col], panel=0))

    # 副图指标
    if "rsi" in indicators and "rsi" in df.columns:
        add_plots.append(mpf.make_addplot(df["rsi"], panel=2, ylabel="RSI"))

    if "macd" in indicators and "macd" in df.columns:
        add_plots.append(mpf.make_addplot(df["macd"], panel=3, color="blue", ylabel="MACD"))
        if "macd_signal" in df.columns:
            add_plots.append(mpf.make_addplot(df["macd_signal"], panel=3, color="orange"))
        if "macd_hist" in df.columns:
            add_plots.append(mpf.make_addplot(df["macd_hist"], panel=3, type="bar", color="gray"))

    style = mpf.make_mpf_style(base_mpf_style="charles")

    kwargs = {
        "type": "candle",
        "volume": True,
        "title": title,
        "style": style,
        "addplot": add_plots if add_plots else None,
        "figscale": 1.5,
        "figratio": (16, 9),
    }

    if save_path:
        kwargs["savefig"] = str(save_path)

    mpf.plot(df, **kwargs)


def plot_macd_cross_results(
    hits: list[dict],
    output_dir: str | Path | None = None,
    last_n_days: int = 60,
) -> None:
    """
    为 MACD 金叉扫描结果批量绘制 K 线图（含 MACD 副图）

    Args:
        hits: scan_ndx100_macd_cross 返回的结果列表
        output_dir: 图片保存目录，默认为 data/output
        last_n_days: 图中显示最近多少个交易日
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hits:
        logger.warning("没有找到符合条件的股票，无图可绘")
        return

    for item in hits:
        ticker = item["ticker"]
        df = item["df"].copy()

        # 截取最近 N 个交易日
        df = df.iloc[-last_n_days:]

        # 构建 MACD 副图
        add_plots = []
        if "macd" in df.columns:
            add_plots.append(mpf.make_addplot(df["macd"], panel=2, color="blue", ylabel="MACD"))
        if "macd_signal" in df.columns:
            add_plots.append(mpf.make_addplot(df["macd_signal"], panel=2, color="orange"))
        if "macd_hist" in df.columns:
            colors = ["green" if v >= 0 else "red" for v in df["macd_hist"]]
            add_plots.append(mpf.make_addplot(df["macd_hist"], panel=2, type="bar", color=colors))

        style = mpf.make_mpf_style(base_mpf_style="charles")
        save_path = output_dir / f"{ticker}_macd_cross.png"

        kwargs = {
            "type": "candle",
            "volume": True,
            "title": f"{ticker} - MACD Golden Cross",
            "style": style,
            "figscale": 1.5,
            "figratio": (16, 9),
            "savefig": str(save_path),
        }
        if add_plots:
            kwargs["addplot"] = add_plots

        mpf.plot(df, **kwargs)
        logger.info(f"已保存 {ticker} K线图 → {save_path}")


def plot_vegas_touch_results(
    hits: list[dict],
    output_dir: str | Path | None = None,
    last_n_days: int = 120,
) -> None:
    """
    为 Vegas 通道回踩扫描结果批量绘制 K 线图（含 EMA144/169 通道）

    Args:
        hits: scan_ndx100_vegas_touch 返回的结果列表
        output_dir: 图片保存目录，默认为 data/output
        last_n_days: 图中显示最近多少个交易日（默认 120 天看半年趋势）
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hits:
        logger.warning("没有找到符合条件的股票，无图可绘")
        return

    for item in hits:
        ticker = item["ticker"]
        df = item["df"].copy()

        # 截取最近 N 个交易日
        df = df.iloc[-last_n_days:]

        # 构建 Vegas 通道叠加到主图
        add_plots = []
        if "ema_144" in df.columns:
            add_plots.append(mpf.make_addplot(
                df["ema_144"], panel=0, color="blue", width=1.5,
                linestyle="dashed", secondary_y=False,
            ))
        if "ema_169" in df.columns:
            add_plots.append(mpf.make_addplot(
                df["ema_169"], panel=0, color="purple", width=1.5,
                linestyle="dashed", secondary_y=False,
            ))

        style = mpf.make_mpf_style(base_mpf_style="charles")
        save_path = output_dir / f"{ticker}_vegas_touch.png"

        kwargs = {
            "type": "candle",
            "volume": True,
            "title": f"{ticker} - Vegas Channel (EMA144/169) Touch",
            "style": style,
            "figscale": 1.5,
            "figratio": (16, 9),
            "savefig": str(save_path),
        }
        if add_plots:
            kwargs["addplot"] = add_plots

        mpf.plot(df, **kwargs)
        logger.info(f"已保存 {ticker} Vegas 通道图 → {save_path}")


def plot_ascending_triangle_results(
    hits: list[dict],
    output_dir: str | Path | None = None,
) -> None:
    """
    为上升三角形/楔形扫描结果绘制 K 线图（含趋势线）

    Args:
        hits: scan_ndx100_ascending_triangle 返回的结果列表
        output_dir: 图片保存目录
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hits:
        logger.warning("没有找到符合条件的股票，无图可绘")
        return

    for item in hits:
        ticker = item["ticker"]
        info = item["pattern_info"]
        df_full = item["df"].copy()

        period = info["period"]
        window_start = info["window_start"]

        # 显示区间：形态开始前留 10 天上下文
        chart_start = max(0, window_start - 10)
        df = df_full.iloc[chart_start:]

        # 在 df 坐标系下重新计算趋势线
        res = info["resistance"]
        sup = info["support"]

        # 趋势线 y = slope * x + intercept，x 是相对于 window 起始的位置
        # 对整个显示区间生成趋势线值
        offset = window_start - chart_start  # df 坐标中形态起点偏移
        n_display = len(df)

        resistance_line = pd.Series(np.nan, index=df.index)
        support_line = pd.Series(np.nan, index=df.index)

        for i in range(n_display):
            x = i - offset  # 相对于 window 起始的 x 坐标
            if 0 <= x < period:
                resistance_line.iloc[i] = res["slope"] * x + res["intercept"]
                support_line.iloc[i] = sup["slope"] * x + sup["intercept"]

        add_plots = [
            mpf.make_addplot(resistance_line, panel=0, color="red", width=2.0,
                             linestyle="dashed", secondary_y=False),
            mpf.make_addplot(support_line, panel=0, color="green", width=2.0,
                             linestyle="dashed", secondary_y=False),
        ]

        _PATTERN_EN = {
            "ascending_triangle": "Ascending Triangle",
            "rising_wedge": "Rising Wedge",
            "symmetrical_triangle": "Symmetrical Triangle",
            "descending_wedge": "Descending Wedge",
        }
        pattern_name = _PATTERN_EN.get(info["pattern"], info["pattern"])

        angle = info.get("convergence_angle_deg", 0)
        res_touches = res.get("touches", 0)
        sup_touches = sup.get("touches", 0)
        status = info.get("convergence_status", "")
        dtc = info.get("days_to_convergence", 0)
        status_label = "CONVERGED" if status == "converged" else f"~{dtc:.0f}d"

        style = mpf.make_mpf_style(base_mpf_style="charles")
        save_path = output_dir / f"{ticker}_triangle.png"

        kwargs = {
            "type": "candle",
            "volume": True,
            "title": (f"{ticker} - {pattern_name} [{status_label}] "
                      f"(angle={angle:.1f}° "
                      f"res_touch={res_touches} sup_touch={sup_touches})"),
            "style": style,
            "figscale": 1.5,
            "figratio": (16, 9),
            "addplot": add_plots,
            "savefig": str(save_path),
        }

        mpf.plot(df, **kwargs)
        logger.info(f"已保存 {ticker} {pattern_name} 图 → {save_path}")
