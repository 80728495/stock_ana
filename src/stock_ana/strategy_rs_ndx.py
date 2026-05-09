"""
纳指100 RS（相对强度）每日计算与报告输出

计算每只纳指100成分股相对于 QQQ 的 RS 值：
- 基准统一为 QQQ（纳斯达克100 ETF）

RS 定义：
  RS_raw  = 股票收盘价 / QQQ 收盘价（按日对齐后相除）
  RS_daily = RS_raw 归一化为起始日 = 100 的序列
  RS_rank  = 63 日收益率在全体中的百分位排名 (0~100)
"""

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.config import OUTPUT_DIR
from stock_ana.data_fetcher import (
    load_all_ndx100_data,
    load_local_data,
)

# 输出目录
NDX_RS_DIR = OUTPUT_DIR / "ndx_rs"
NDX_RS_DIR.mkdir(parents=True, exist_ok=True)

NDX_RS_SERIES_DIR = OUTPUT_DIR / "ndx_rs_series"
NDX_RS_SERIES_DIR.mkdir(parents=True, exist_ok=True)

NDX_RS_CHARTS_DIR = OUTPUT_DIR / "ndx_rs_charts"
NDX_RS_CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────── RS 计算 ───────────────────


def _compute_rs_line(df_stock: pd.DataFrame,
                     df_market: pd.DataFrame) -> pd.Series | None:
    """
    计算 RS Line = Stock Close / Market Close，归一化首日 = 100。
    """
    common_idx = df_stock.index.intersection(df_market.index)
    if len(common_idx) < 60:
        return None

    stock_close = df_stock.loc[common_idx, "close"]
    market_close = df_market.loc[common_idx, "close"]

    rs_raw = stock_close / market_close
    rs_line = rs_raw / rs_raw.iloc[0] * 100
    return rs_line


def _compute_rs_momentum(
    df_stock: pd.DataFrame,
    df_market: pd.DataFrame,
    lookback: int = 63,
    smooth: int = 10,
) -> pd.Series | None:
    """
    计算 RS Z-Score = 价格比的滚动标准化偏差。

    ratio  = Stock Close / Bench Close
    zscore = (ratio − SMA(ratio, N)) / StdDev(ratio, N)
    RS     = EMA(zscore, smooth)

    以 0 为中轴：
      > 0   当前相对表现优于近 N 日平均
      = 0   与近 N 日平均一致
      < 0   当前相对表现弱于近 N 日平均

    单位：标准差 (σ)，±1 = 较明显偏离，±2 = 显著偏离。
    """
    common_idx = df_stock.index.intersection(df_market.index)
    if len(common_idx) < lookback + smooth:
        return None

    stock_close = df_stock.loc[common_idx, "close"]
    market_close = df_market.loc[common_idx, "close"]

    ratio = stock_close / market_close
    ratio_sma = ratio.rolling(lookback).mean()
    ratio_std = ratio.rolling(lookback).std()

    zscore = (ratio - ratio_sma) / ratio_std.replace(0, np.nan)
    rs_mom = zscore.ewm(span=smooth).mean()
    return rs_mom


def _compute_rrg_momentum(
        rs_series: pd.Series,
        period: int = 10,
        smooth: int = 5,
) -> pd.Series | None:
        """
        计算 RRG 风格 RS 动量（100 中轴）。

        在不改变原有 RS Z-Score 计算的前提下，对 RS 序列做动量化：
            rs_ratio = 100 + 10 * rs_series
            rrg_mom = 100 + EMA( (rs_ratio / rs_ratio.shift(period) - 1) * 100, smooth )

        解释：
            >100: RS 动量增强
            <100: RS 动量减弱
        """
        if rs_series is None or len(rs_series.dropna()) < period + smooth + 5:
                return None

        rs_ratio = 100 + rs_series * 10
        base = rs_ratio.shift(period).replace(0, np.nan)
        roc = (rs_ratio / base - 1) * 100
        rrg_mom = 100 + roc.ewm(span=smooth, adjust=False).mean()
        return rrg_mom


def _compute_rs_rank(stock_data: dict[str, pd.DataFrame],
                     df_market: pd.DataFrame,
                     lookback: int = 63) -> dict[str, float]:
    """
    计算每只股票 N 日收益率在全体中的百分位排名 (0~100)。
    """
    returns: dict[str, float] = {}

    for ticker, df in stock_data.items():
        if ticker == "QQQ":
            continue  # 基准自身不参与排名
        common_idx = df.index.intersection(df_market.index)
        if len(common_idx) < lookback:
            continue
        aligned_close = df.loc[common_idx, "close"]
        if len(aligned_close) < lookback:
            continue
        ret = (aligned_close.iloc[-1] / aligned_close.iloc[-lookback] - 1) * 100
        returns[ticker] = ret

    if not returns:
        return {}

    all_rets = np.array(list(returns.values()))
    ranks = {}
    for ticker, ret in returns.items():
        pct = float(np.mean(all_rets <= ret)) * 100
        ranks[ticker] = round(pct, 1)
    return ranks


def _classify_strength(rank: float) -> str:
    """根据 63 日 RS 排名给出强弱分类"""
    if np.isnan(rank):
        return "N/A"
    if rank >= 90:
        return "极强"
    elif rank >= 70:
        return "强势"
    elif rank >= 50:
        return "中等偏强"
    elif rank >= 30:
        return "中等偏弱"
    elif rank >= 10:
        return "弱势"
    else:
        return "极弱"


# ─────────────────── 核心计算 ───────────────────


def compute_ndx_rs_daily() -> pd.DataFrame:
    """
    计算全部纳指100成分股的每日 RS 指标。

    Returns:
        DataFrame: columns = [
            ticker, benchmark, rs_rank_63d, rs_rank_21d,
            rs_momentum, rs_momentum_21d,
            rs_latest, rs_ema21, rs_chg_5d, rs_chg_21d, rs_chg_63d,
            close, pct_5d, pct_21d, strength
        ]
    """
    stock_data = load_all_ndx100_data()
    if not stock_data:
        raise RuntimeError("本地无数据，请先运行 update_ndx100_data()")

    df_qqq = load_local_data("QQQ")
    if df_qqq is None or df_qqq.empty:
        raise RuntimeError("QQQ 数据缺失，请先运行 update_ndx100_data()")

    # 排除 QQQ 自身
    stock_data_ex = {k: v for k, v in stock_data.items() if k != "QQQ"}

    # 计算 RS 排名
    rs_ranks_63d = _compute_rs_rank(stock_data_ex, df_qqq, lookback=63)
    rs_ranks_21d = _compute_rs_rank(stock_data_ex, df_qqq, lookback=21)

    rows = []

    for ticker, df_stock in stock_data_ex.items():
        rs_line = _compute_rs_line(df_stock, df_qqq)
        if rs_line is None or len(rs_line) < 21:
            continue

        # RS 动量 (Z-Score)
        rs_mom = _compute_rs_momentum(df_stock, df_qqq, lookback=63, smooth=10)
        rs_mom_val = round(float(rs_mom.iloc[-1]), 2) if rs_mom is not None and len(rs_mom.dropna()) > 0 else np.nan

        rs_mom_21 = _compute_rs_momentum(df_stock, df_qqq, lookback=21, smooth=10)
        rs_mom_21_val = round(float(rs_mom_21.iloc[-1]), 2) if rs_mom_21 is not None and len(rs_mom_21.dropna()) > 0 else np.nan

        # RS 指标
        rs_latest = rs_line.iloc[-1]
        rs_ema21 = rs_line.ewm(span=21).mean().iloc[-1]
        rs_chg_5d = (rs_line.iloc[-1] / rs_line.iloc[-6] - 1) * 100 if len(rs_line) >= 6 else np.nan
        rs_chg_21d = (rs_line.iloc[-1] / rs_line.iloc[-22] - 1) * 100 if len(rs_line) >= 22 else np.nan
        rs_chg_63d = (rs_line.iloc[-1] / rs_line.iloc[-64] - 1) * 100 if len(rs_line) >= 64 else np.nan

        # 价格指标
        close = df_stock["close"].iloc[-1]
        close_5 = df_stock["close"].iloc[-6] if len(df_stock) >= 6 else np.nan
        close_21 = df_stock["close"].iloc[-22] if len(df_stock) >= 22 else np.nan
        pct_5d = (close / close_5 - 1) * 100 if not np.isnan(close_5) else np.nan
        pct_21d = (close / close_21 - 1) * 100 if not np.isnan(close_21) else np.nan

        # 强弱判定
        rank_63 = rs_ranks_63d.get(ticker, np.nan)
        strength = _classify_strength(rank_63)

        rows.append({
            "ticker": ticker,
            "benchmark": "QQQ",
            "rs_rank_63d": rank_63,
            "rs_rank_21d": rs_ranks_21d.get(ticker, np.nan),
            "rs_momentum": rs_mom_val,
            "rs_momentum_21d": rs_mom_21_val,
            "rs_latest": round(rs_latest, 2),
            "rs_ema21": round(rs_ema21, 2),
            "rs_chg_5d": round(rs_chg_5d, 2) if not np.isnan(rs_chg_5d) else np.nan,
            "rs_chg_21d": round(rs_chg_21d, 2) if not np.isnan(rs_chg_21d) else np.nan,
            "rs_chg_63d": round(rs_chg_63d, 2) if not np.isnan(rs_chg_63d) else np.nan,
            "close": round(close, 2),
            "pct_5d": round(pct_5d, 2) if not np.isnan(pct_5d) else np.nan,
            "pct_21d": round(pct_21d, 2) if not np.isnan(pct_21d) else np.nan,
            "strength": strength,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("rs_rank_63d", ascending=False).reset_index(drop=True)

    logger.info(f"纳指100 RS 计算完成：{len(result)} 只标的")
    return result


# ─────────────────── 报告输出 ───────────────────


_REPORT_HEADER = """\
================================================================================
                   纳指100成分股 RS（相对强度）每日报告
================================================================================
生成时间: {timestamp}
数据截止: {data_date}

┌─────────────────────────────────────────────────────────────────────────────┐
│  RS 排名 (rs_rank_63d) 含义 — 63日收益率在纳指100全体中的百分位排名          │
│                                                                             │
│  90~100  极强     该股近63日涨幅超过组内90%的股票，处于领涨地位               │
│  70~89   强势     明显跑赢大多数纳指100成分股，趋势向上                      │
│  50~69   中等偏强  略优于中位数，表现尚可                                     │
│  30~49   中等偏弱  略弱于中位数，缺乏动能                                    │
│  10~29   弱势     明显跑输大多数纳指100成分股                                │
│   0~9    极弱     近63日涨幅垫底，处于最弱梯队                               │
│                                                                             │
│  基准: QQQ (纳斯达克100 ETF)                                                │
│                                                                             │
│  RS 动量: Z-Score of (股价/QQQ价) vs N日均值和标准差                        │
│    63d: 中期趋势视角 | 21d: 短期灵敏视角                                    │
│    > 0: 相对表现优于近N日平均 | = 0: 与平均一致 | < 0: 弱于平均              │
│    ±1σ: 较明显偏离 | ±2σ: 显著偏离                                          │
│  rs_chg_Nd: RS Line 近N日变化率(%)，正值=相对走强                            │
│  rs_ema21: RS Line 的21日指数均线，RS > EMA21 = 短期趋势向上                 │
└─────────────────────────────────────────────────────────────────────────────┘

标的总数: {total_count} 只  |  基准: QQQ
================================================================================

"""


def _fmt_pct(val) -> str:
    """格式化百分比值"""
    if pd.isna(val):
        return "N/A"
    return f"{val:+.2f}%"


def _fmt_pp(val) -> str:
    """格式化 Z-Score 值"""
    if pd.isna(val):
        return "N/A"
    return f"{val:+.2f}σ"


def generate_ndx_rs_report() -> Path:
    """
    计算全部纳指100成分股 RS 并输出 txt 报告到 data/output/ndx_rs/。

    Returns:
        输出文件路径
    """
    df = compute_ndx_rs_daily()
    if df.empty:
        logger.warning("无有效 RS 数据")
        return NDX_RS_DIR

    # 数据截止日
    sample_ticker = df["ticker"].iloc[0]
    sample_df = load_local_data(sample_ticker)
    data_date = sample_df.index.max().strftime("%Y-%m-%d") if sample_df is not None else "unknown"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = _REPORT_HEADER.format(
        timestamp=timestamp,
        data_date=data_date,
        total_count=len(df),
    )

    lines = [header]

    # ── 全部标的排名表 ──
    lines.append("【一、RS 排名总览（按 rs_rank_63d 降序）】\n")
    lines.append(
        f"{'排名':>4}  {'代码':<6}  {'基准':<4}  "
        f"{'RS排名63d':>9}  {'RS排名21d':>9}  {'RS动量63d':>9}  {'RS动量21d':>9}  "
        f"{'RS变化5d':>8}  {'RS变化21d':>9}  {'RS变化63d':>9}  "
        f"{'收盘价':>9}  {'涨跌5d':>8}  {'涨跌21d':>8}  {'强弱':>6}"
    )
    lines.append("-" * 150)

    for i, row in df.iterrows():
        lines.append(
            f"{i+1:>4}  {row['ticker']:<6}  {row['benchmark']:<4}  "
            f"{row['rs_rank_63d']:>9.1f}  {row['rs_rank_21d']:>9.1f}  "
            f"{_fmt_pp(row['rs_momentum']):>9}  "
            f"{_fmt_pp(row['rs_momentum_21d']):>9}  "
            f"{_fmt_pct(row['rs_chg_5d']):>8}  {_fmt_pct(row['rs_chg_21d']):>9}  "
            f"{_fmt_pct(row['rs_chg_63d']):>9}  "
            f"{row['close']:>9.2f}  {_fmt_pct(row['pct_5d']):>8}  "
            f"{_fmt_pct(row['pct_21d']):>8}  {row['strength']:>6}"
        )

    # ── 分组展示 ──
    lines.append("\n")
    lines.append("=" * 80)
    lines.append("【二、按强弱分组】\n")

    for level in ["极强", "强势", "中等偏强", "中等偏弱", "弱势", "极弱"]:
        group = df[df["strength"] == level]
        if group.empty:
            continue
        lines.append(f"── {level} ({len(group)} 只) ──")
        for _, row in group.iterrows():
            lines.append(
                f"  {row['ticker']:<6}  "
                f"RS排名={row['rs_rank_63d']:.1f}%  "
                f"RS动量63d={_fmt_pp(row['rs_momentum'])}  "
                f"RS动量21d={_fmt_pp(row['rs_momentum_21d'])}  "
                f"收盘={row['close']:.2f}  [QQQ]"
            )
        lines.append("")

    # ── 写入文件 ──
    today_str = datetime.now().strftime("%Y%m%d")
    filename = f"ndx_rs_{today_str}.txt"
    filepath = NDX_RS_DIR / filename
    filepath.write_text("\n".join(lines), encoding="utf-8")

    logger.info(f"纳指100 RS 报告已生成: {filepath}")
    return filepath


# ─────────────────── RS 每日序列 & 图表 ───────────────────


def compute_all_rs_series() -> dict[str, pd.DataFrame]:
    """
    计算每只纳指100成分股的完整每日 RS 序列。

    RS = Z-Score of (Stock/QQQ ratio) vs rolling mean & std。
    以 0 为中轴，单位为标准差，±1 = 较明显偏离，±2 = 显著偏离。

    在不改动现有 RS Z-Score 计算的前提下，额外加入 RRG 风格动量：
    - rrg_mom_63d: 基于 rs_zscore 的 RRG 动量（100 中轴）
    - rrg_mom_21d: 基于 rs_zscore_21d 的 RRG 动量（100 中轴）

    两个时间窗口：
    - 63d：中期趋势（rs_zscore / rs_zscore_fast）
    - 21d：短期灵敏（rs_zscore_21d / rs_zscore_21d_fast）

    Returns:
        {ticker: DataFrame} 其中 DataFrame columns =
            [rs_zscore, rs_zscore_fast, rs_zscore_21d, rs_zscore_21d_fast,
             rrg_mom_63d, rrg_mom_21d,
             close, bench_close]
        index = date
    """
    stock_data = load_all_ndx100_data()
    df_qqq = load_local_data("QQQ")
    if df_qqq is None or df_qqq.empty:
        raise RuntimeError("QQQ 数据缺失，请先运行 update_ndx100_data()")

    result: dict[str, pd.DataFrame] = {}

    for ticker, df_stock in stock_data.items():
        if ticker == "QQQ":
            continue

        # ── 63d 窗口：慢线 EMA10 和快线 EMA5 ──
        rs_slow = _compute_rs_momentum(df_stock, df_qqq, lookback=63, smooth=10)
        if rs_slow is None:
            continue
        rs_fast = _compute_rs_momentum(df_stock, df_qqq, lookback=63, smooth=5)

        # ── 21d 窗口：慢线 EMA10 和快线 EMA5 ──
        rs_slow_21 = _compute_rs_momentum(df_stock, df_qqq, lookback=21, smooth=10)
        rs_fast_21 = _compute_rs_momentum(df_stock, df_qqq, lookback=21, smooth=5)

        # ── 新增：RRG 风格 RS 动量（100 中轴）──
        rrg_mom_63d = _compute_rrg_momentum(rs_slow, period=10, smooth=5)
        rrg_mom_21d = _compute_rrg_momentum(rs_slow_21, period=10, smooth=5) if rs_slow_21 is not None else None

        common_idx = rs_slow.dropna().index
        if len(common_idx) < 30:
            continue

        stock_close = df_stock.loc[common_idx, "close"]
        bench_close = df_qqq.loc[common_idx, "close"]

        rs_df = pd.DataFrame({
            "rs_zscore": rs_slow.loc[common_idx],
            "rs_zscore_fast": rs_fast.loc[common_idx] if rs_fast is not None else np.nan,
            "rs_zscore_21d": rs_slow_21.reindex(common_idx) if rs_slow_21 is not None else np.nan,
            "rs_zscore_21d_fast": rs_fast_21.reindex(common_idx) if rs_fast_21 is not None else np.nan,
            "rrg_mom_63d": rrg_mom_63d.reindex(common_idx) if rrg_mom_63d is not None else np.nan,
            "rrg_mom_21d": rrg_mom_21d.reindex(common_idx) if rrg_mom_21d is not None else np.nan,
            "close": stock_close,
            "bench_close": bench_close,
        })
        rs_df.index.name = "date"
        result[ticker] = rs_df

    logger.info(f"纳指100 RS 序列计算完成：{len(result)} 只标的")
    return result


def save_rs_series(rs_data: dict[str, pd.DataFrame] | None = None) -> Path:
    """
    将每只股票的每日 RS 序列保存为 CSV 文件。

    Returns:
        输出目录路径
    """
    if rs_data is None:
        rs_data = compute_all_rs_series()

    for ticker, rs_df in rs_data.items():
        filename = f"{ticker}.csv"
        filepath = NDX_RS_SERIES_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {ticker} | 基准: QQQ\n")
            f.write(f"# rs_zscore: Z-Score of (股价/QQQ价) vs 63日均值和标准差, EMA10平滑\n")
            f.write(f"# rs_zscore_fast: 同上但用EMA5平滑(快线)\n")
            f.write(f"# rs_zscore_21d: Z-Score of (股价/QQQ价) vs 21日均值和标准差, EMA10平滑(更灵敏)\n")
            f.write(f"# rs_zscore_21d_fast: 同上但用EMA5平滑(快线)\n")
            f.write(f"# rrg_mom_63d: RRG风格RS动量(基于rs_zscore), 100中轴\n")
            f.write(f"# rrg_mom_21d: RRG风格RS动量(基于rs_zscore_21d), 100中轴\n")
            f.write(f"#   > 0 相对表现优于近N日平均 | = 0 与平均一致 | < 0 相对表现弱于平均\n")
            f.write(f"# 单位: 标准差(σ), ±1=较明显偏离, ±2=显著偏离\n")
            rs_df.round(4).to_csv(f)

    logger.info(f"纳指100 RS 序列已保存：{len(rs_data)} 个文件 → {NDX_RS_SERIES_DIR}")
    return NDX_RS_SERIES_DIR


def plot_rs_charts(rs_data: dict[str, pd.DataFrame] | None = None,
                   days: int = 0) -> Path:
    """
    为每只股票绘制 RS Line 历史趋势图并保存为 PNG。

    图表包含：
    - 第一行：股价走势
    - 第二行：RS 值（同图展示 63d 与 21d 的 Z-Score）
    - 第三行：RS 动量（RRG 风格，100 中轴）

    Args:
        rs_data: 预计算的 RS 数据，为 None 则自动计算
        days: 显示最近 N 天的数据，0 = 全部

    Returns:
        图表输出目录路径
    """
    if rs_data is None:
        rs_data = compute_all_rs_series()

    import sys
    if sys.platform == "win32":
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "Arial"]
    elif sys.platform == "darwin":
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "Arial"]
    else:
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "Arial Unicode MS", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    total = len(rs_data)
    for i, (ticker, rs_df) in enumerate(rs_data.items(), 1):
        plot_df = rs_df.iloc[-days:] if days > 0 else rs_df

        try:
            _draw_single_rs_chart(ticker, plot_df)
            if i % 10 == 0 or i == total:
                logger.info(f"[{i}/{total}] RS 图表生成中 ...")
        except Exception as e:
            logger.error(f"{ticker} 图表生成失败: {e}")

    plt.close("all")
    logger.info(f"纳指100 RS 图表已保存：{total} 张 → {NDX_RS_CHARTS_DIR}")
    return NDX_RS_CHARTS_DIR


def _draw_single_rs_chart(ticker: str, rs_df: pd.DataFrame) -> None:
    """
    绘制单只股票 RS 图（三面板：价格 + RS值(Z-Score) + RRG动量）。
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11),
                                         gridspec_kw={"height_ratios": [1, 1, 1]},
                                         sharex=True)
    fig.subplots_adjust(hspace=0.08)

    dates = rs_df.index
    close = rs_df["close"]
    rs_z = rs_df["rs_zscore"]
    rs_fast = rs_df["rs_zscore_fast"]
    rs_z_21 = rs_df["rs_zscore_21d"]
    rs_fast_21 = rs_df["rs_zscore_21d_fast"]
    rrg_mom_63d = rs_df["rrg_mom_63d"] if "rrg_mom_63d" in rs_df.columns else pd.Series(index=dates, dtype=float)
    rrg_mom_21d = rs_df["rrg_mom_21d"] if "rrg_mom_21d" in rs_df.columns else pd.Series(index=dates, dtype=float)

    # ── 上面板：股价 ──
    ax1.plot(dates, close, color="#2196F3", linewidth=1.2, label="Close")
    if len(close) >= 50:
        ma50 = close.rolling(50).mean()
        ax1.plot(dates, ma50, color="#FF9800", linewidth=0.8, alpha=0.7, label="MA50")
    if len(close) >= 200:
        ma200 = close.rolling(200).mean()
        ax1.plot(dates, ma200, color="#9C27B0", linewidth=0.8, alpha=0.7, label="MA200")

    ax1.set_ylabel("Price", fontsize=11)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{ticker}  |  Benchmark: QQQ", fontsize=14, fontweight="bold")

    # ── 中面板：同图展示 RS Z-Score 63d + 21d ──
    _draw_rs_value_panel(ax2, dates, rs_z, rs_fast, rs_z_21, rs_fast_21)

    # ── 下面板：RRG 风格 RS 动量 ──
    _draw_rrg_momentum_panel(ax3, dates, rrg_mom_63d, rrg_mom_21d)

    ax3.set_xlabel("Date", fontsize=11)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30)

    filepath = NDX_RS_CHARTS_DIR / f"{ticker}.png"
    fig.savefig(filepath, dpi=120, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


def _draw_rs_value_panel(ax, dates,
                         rs_z_63, rs_fast_63,
                         rs_z_21, rs_fast_21) -> None:
    """
    在同一面板绘制 63d 与 21d 的 RS Z-Score。
    """
    ax.axhline(y=0, color="gray", linewidth=1.0, linestyle="-", alpha=0.6)
    ax.axhline(y=1, color="#4CAF50", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.axhline(y=-1, color="#F44336", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.axhline(y=2, color="#4CAF50", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.axhline(y=-2, color="#F44336", linewidth=0.6, linestyle="--", alpha=0.4)

    valid_63 = rs_z_63.notna()

    ax.fill_between(dates, 0, rs_z_63, where=(rs_z_63 >= 0) & valid_63,
                    alpha=0.15, color="#4CAF50", interpolate=True)
    ax.fill_between(dates, 0, rs_z_63, where=(rs_z_63 < 0) & valid_63,
                    alpha=0.15, color="#F44336", interpolate=True)

    ax.plot(dates, rs_z_63, color="#E91E63", linewidth=1.5, label="RS 63d (EMA10)")
    if rs_fast_63 is not None and rs_fast_63.notna().any():
        ax.plot(dates, rs_fast_63, color="#FF9800", linewidth=0.9, alpha=0.6,
                label="RS 63d Fast (EMA5)")

    ax.plot(dates, rs_z_21, color="#1565C0", linewidth=1.5, label="RS 21d (EMA10)")
    if rs_fast_21 is not None and rs_fast_21.notna().any():
        ax.plot(dates, rs_fast_21, color="#26A69A", linewidth=0.9, alpha=0.6,
                label="RS 21d Fast (EMA5)")

    last_63 = rs_z_63.dropna().iloc[-1] if len(rs_z_63.dropna()) > 0 else np.nan
    last_21 = rs_z_21.dropna().iloc[-1] if len(rs_z_21.dropna()) > 0 else np.nan
    if not np.isnan(last_63) or not np.isnan(last_21):
        ax.text(0.02, 0.95,
                f"63d={last_63:+.2f}σ  |  21d={last_21:+.2f}σ",
                transform=ax.transAxes, fontsize=10,
                color="#424242", fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_ylabel("RS Z-Score (σ)", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    valid_vals = pd.concat([rs_z_63.dropna(), rs_z_21.dropna()])
    if len(valid_vals) > 0:
        y_abs_max = max(abs(valid_vals.min()), abs(valid_vals.max()), 3) * 1.05
    else:
        y_abs_max = 3
    ax.set_ylim(-y_abs_max, y_abs_max)


def _draw_rrg_momentum_panel(ax, dates,
                             mom_63: pd.Series,
                             mom_21: pd.Series) -> None:
    """
    绘制 RRG 风格 RS 动量面板（100 中轴）。
    """
    ax.axhline(y=100, color="gray", linewidth=1.0, linestyle="-", alpha=0.7)
    ax.axhline(y=102, color="#4CAF50", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.axhline(y=98, color="#F44336", linewidth=0.6, linestyle=":", alpha=0.5)

    ax.plot(dates, mom_63, color="#8E24AA", linewidth=1.5, label="RRG Momentum 63d")
    ax.plot(dates, mom_21, color="#FB8C00", linewidth=1.5, label="RRG Momentum 21d")

    last_63 = mom_63.dropna().iloc[-1] if len(mom_63.dropna()) > 0 else np.nan
    last_21 = mom_21.dropna().iloc[-1] if len(mom_21.dropna()) > 0 else np.nan

    if not np.isnan(last_63) or not np.isnan(last_21):
        status = "Momentum Up" if (not np.isnan(last_21) and last_21 >= 100) else "Momentum Down"
        status_color = "#2E7D32" if status == "Momentum Up" else "#C62828"
        ax.text(0.02, 0.95,
                f"63d={last_63:.2f}  |  21d={last_21:.2f}  ({status})",
                transform=ax.transAxes, fontsize=10,
                color=status_color, fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_ylabel("RRG Momentum", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    valid_vals = pd.concat([mom_63.dropna(), mom_21.dropna()])
    if len(valid_vals) > 0:
        dev = max(abs(valid_vals.max() - 100), abs(valid_vals.min() - 100), 2.0) * 1.1
        ax.set_ylim(100 - dev, 100 + dev)
    else:
        ax.set_ylim(96, 104)


# ─────────────────── 一键生成 ───────────────────


def generate_ndx_rs_all() -> dict[str, Path]:
    """
    一键生成全部纳指100 RS 产出：
    1. 每日 RS 排名报告 (txt)
    2. 每只股票每日 RS 序列 (csv)
    3. 每只股票 RS 趋势图 (png)

    Returns:
        {"report": path, "series": path, "charts": path}
    """
    # 先计算 RS 序列（复用）
    rs_data = compute_all_rs_series()

    # 1. 报告
    report_path = generate_ndx_rs_report()

    # 2. CSV 序列
    series_dir = save_rs_series(rs_data)

    # 3. 图表
    charts_dir = plot_rs_charts(rs_data)

    logger.info(f"纳指100 RS 全部产出完成:\n"
                f"  报告: {report_path}\n"
                f"  序列: {series_dir}\n"
                f"  图表: {charts_dir}")

    return {"report": report_path, "series": series_dir, "charts": charts_dir}
