"""
可视化模块 - K线图与技术指标绘图
"""

from pathlib import Path

import mplfinance as mpf
import pandas as pd


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
