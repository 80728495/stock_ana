#!/usr/bin/env python3
"""
NDX100 + 港股重点列表 主升浪回调（Vegas）回测脚本

覆盖市场：
- 美股：NASDAQ-100 成分股（data/cache/ndx100/）
- 港股：恒生恒科重点成分股（data/lists/hk_focus_list.md → data/cache/hk/）

全部使用本地缓存，不触发网络请求。
请先运行 daily_update.py 确保本地数据已更新，再执行此脚本。

用法:
    python -m stock_ana.backtest.backtest_main_rally_pullback
    python -m stock_ana.backtest.backtest_main_rally_pullback --step 3 --gap 12
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.config import OUTPUT_DIR
from stock_ana.data.fetcher import load_all_ndx100_data
from stock_ana.data.fetcher_hk import load_hk_list, load_hk_local
from stock_ana.data.list_manager import load_hk_focus_list
from stock_ana.utils.plot_renderers import plot_main_rally_pullback_signals
from stock_ana.scan.main_rally_scan import scan_one_symbol
from stock_ana.strategies.primitives.wave import analyze_wave_structure


FORWARD_DAYS = [5, 10, 21, 63]
OUT_DIR = OUTPUT_DIR / "main_rally_pullback"
ANALYSIS_YEARS = 2


def _analysis_start() -> pd.Timestamp:
    """Return the first date whose signals should be included in the backtest output."""
    return pd.Timestamp((datetime.now() - timedelta(days=365 * ANALYSIS_YEARS)).date())


def _prepare_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize one raw OHLCV frame to lower-case columns and a sorted datetime index."""
    x = df.copy()
    x.columns = [c.lower() for c in x.columns]
    x.index = pd.to_datetime(x.index)
    x.index.name = "date"
    return x.sort_index()


def load_universe_data() -> dict[str, dict]:
    """
    从本地缓存加载回测标的：NDX100 + 港股重点列表（hk_focus_list）。

    不触发任何网络请求，所有数据均从 data/cache/ 读取。
    若本地缓存缺失，请先运行 daily_update.py 更新数据。
    """
    result: dict[str, dict] = {}

    # ── NDX100 ──
    ndx_data = load_all_ndx100_data()
    for symbol, df in ndx_data.items():
        x = _prepare_price_frame(df)
        if x.empty:
            continue
        result[f"US:{symbol}"] = {
            "market": "US",
            "symbol": symbol,
            "name": symbol,
            "df": x,
        }
    logger.info(f"NDX100: {len(ndx_data)} 只本地数据已加载")

    # ── 港股重点列表 ──
    try:
        hk_codes = load_hk_focus_list()
    except FileNotFoundError as e:
        logger.warning(f"港股重点列表不存在，跳过港股: {e}")
        hk_codes = []

    try:
        hk_list_df = load_hk_list()
        code_name_map: dict[str, str] = dict(zip(
            hk_list_df["code"].astype(str), hk_list_df["name"]
        ))
    except Exception:
        code_name_map = {}

    hk_loaded = 0
    for raw_code in hk_codes:
        code = str(raw_code).zfill(5)
        df = load_hk_local(code)
        if df is None or df.empty:
            logger.debug(f"HK {code}: 本地无缓存，跳过（先运行 daily_update.py）")
            continue
        x = _prepare_price_frame(df)
        result[f"HK:{code}"] = {
            "market": "HK",
            "symbol": code,
            "name": code_name_map.get(code, code),
            "df": x,
        }
        hk_loaded += 1

    logger.info(f"港股重点列表: {hk_loaded}/{len(hk_codes)} 只本地数据已加载")
    logger.info(f"回测标的总计: {len(result)} 只（US {len(ndx_data)} + HK {hk_loaded}）")
    return result


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """按市场与持有期汇总统计。"""
    if df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for market in sorted(df["market"].unique()):
        sub = df[df["market"] == market]
        for d in FORWARD_DAYS:
            col = f"ret_{d}d"
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if vals.empty:
                continue
            rows.append({
                "market": market,
                "period": f"{d}d",
                "count": int(len(vals)),
                "win_rate_pct": round(float((vals > 0).mean() * 100), 2),
                "avg_return_pct": round(float(vals.mean()), 2),
                "median_return_pct": round(float(vals.median()), 2),
                "max_return_pct": round(float(vals.max()), 2),
                "min_return_pct": round(float(vals.min()), 2),
            })

    # 全市场汇总
    for d in FORWARD_DAYS:
        col = f"ret_{d}d"
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append({
            "market": "ALL",
            "period": f"{d}d",
            "count": int(len(vals)),
            "win_rate_pct": round(float((vals > 0).mean() * 100), 2),
            "avg_return_pct": round(float(vals.mean()), 2),
            "median_return_pct": round(float(vals.median()), 2),
            "max_return_pct": round(float(vals.max()), 2),
            "min_return_pct": round(float(vals.min()), 2),
        })

    return pd.DataFrame(rows)


def summarize_by_support_type(df: pd.DataFrame) -> pd.DataFrame:
    """按市场、回撤类型、持有期汇总统计。"""
    if df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for market in sorted(df["market"].unique()):
        market_df = df[df["market"] == market]
        for support_type in sorted(market_df["support_type"].unique()):
            sub = market_df[market_df["support_type"] == support_type]
            for d in FORWARD_DAYS:
                col = f"ret_{d}d"
                vals = pd.to_numeric(sub[col], errors="coerce").dropna()
                if vals.empty:
                    continue
                rows.append({
                    "market": market,
                    "support_type": support_type,
                    "period": f"{d}d",
                    "count": int(len(vals)),
                    "win_rate_pct": round(float((vals > 0).mean() * 100), 2),
                    "avg_return_pct": round(float(vals.mean()), 2),
                    "median_return_pct": round(float(vals.median()), 2),
                })

    for support_type in sorted(df["support_type"].unique()):
        sub = df[df["support_type"] == support_type]
        for d in FORWARD_DAYS:
            col = f"ret_{d}d"
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if vals.empty:
                continue
            rows.append({
                "market": "ALL",
                "support_type": support_type,
                "period": f"{d}d",
                "count": int(len(vals)),
                "win_rate_pct": round(float((vals > 0).mean() * 100), 2),
                "avg_return_pct": round(float(vals.mean()), 2),
                "median_return_pct": round(float(vals.median()), 2),
            })

    return pd.DataFrame(rows)


def summarize_symbol_support_styles(df: pd.DataFrame) -> pd.DataFrame:
    """按股票统计历史回撤风格，区分中期 Vegas / 长期 Vegas。"""
    if df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    grouped = df.groupby(["market", "symbol", "name"], dropna=False)
    for (market, symbol, name), sub in grouped:
        signal_count = int(len(sub))
        mid_count = int((sub["support_type"] == "mid_vegas").sum())
        long_count = int((sub["support_type"] == "long_vegas").sum())
        dominant_support_type = "mixed"
        if mid_count > long_count and mid_count / signal_count >= 0.6:
            dominant_support_type = "mid_vegas"
        elif long_count > mid_count and long_count / signal_count >= 0.6:
            dominant_support_type = "long_vegas"

        band_mode = sub["support_band"].mode(dropna=True)
        rows.append({
            "market": market,
            "symbol": symbol,
            "name": name,
            "signal_count": signal_count,
            "mid_vegas_count": mid_count,
            "long_vegas_count": long_count,
            "mid_vegas_ratio": round(mid_count / signal_count, 3),
            "long_vegas_ratio": round(long_count / signal_count, 3),
            "dominant_support_type": dominant_support_type,
            "dominant_support_band": band_mode.iloc[0] if not band_mode.empty else "",
            "avg_pullback_pct": round(float(pd.to_numeric(sub["pullback_pct_from_high"], errors="coerce").mean()), 2),
            "avg_ret_21d_pct": round(float(pd.to_numeric(sub["ret_21d"], errors="coerce").mean()), 2),
            "avg_ret_63d_pct": round(float(pd.to_numeric(sub["ret_63d"], errors="coerce").mean()), 2),
            "first_signal_date": str(sub["signal_date"].min()),
            "last_signal_date": str(sub["signal_date"].max()),
        })

    return pd.DataFrame(rows).sort_values(
        ["dominant_support_type", "signal_count", "market", "symbol"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)



def _wave1_starts_near_long_vegas(df: pd.DataFrame, start_iloc: int, max_below_pct: float = 15.0) -> bool:
    """
    验证第一浪起点在 Long Vegas 附近，而非远低于均线的大底。

    判断依据：start_pivot 价格 >= Long Vegas（EMA144/169/200 最大值） × (1 - max_below_pct%)
    - 底部围绕 Long Vegas 震荡时：起点价格通常在 Long Vegas 的 0~15% 以内
    - 大底直接从底部起算时：起点价格可能低于 Long Vegas 30~50% 甚至更多

    注意：不要求起点之前一定有"从上方回踩"的历史，只要求起点本身在合理范围内。
    """
    x = df.copy()
    x.columns = [c.lower() for c in x.columns]
    close = x["close"].astype(float)

    ema144 = close.ewm(span=144, adjust=False).mean()
    ema169 = close.ewm(span=169, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    long_upper = max(
        float(ema144.iloc[start_iloc]),
        float(ema169.iloc[start_iloc]),
        float(ema200.iloc[start_iloc]),
    )
    if long_upper <= 0:
        return False

    start_price = float(x["close"].iloc[start_iloc])
    return start_price >= long_upper * (1.0 - max_below_pct / 100.0)


def analyze_three_wave_amplitudes(market_data: dict[str, dict] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    统计拥有完整三浪结构的股票中，第 1、2、3 浪各自的涨幅分布。

    口径说明：
    - 只纳入前三浪**均已完结**（end_pivot is not None）的股票，避免末尾未完成浪带来偏差
    - 第一浪起点须在 Long Vegas 附近：起点价格不低于 Long Vegas 15% 以上。
      这排除了"第一浪从大底起算（价格远低于长期均线）"的情况，与第二、三浪口径一致。
      底部正常震荡的起点通常在 Long Vegas ±15% 以内；真正大底可能低 30~50% 甚至更多。
    - rise_pct = (浪峰值 / 浪起点 - 1) × 100，是浪本身的幅度，与回踩信号无关
    - 完整三浪 = 在历史 K 线中能清晰识别出至少连续 3 个完结大浪

    Returns:
        (detail_df, summary_df)
        detail_df: 每股每浪一行，含 rise_pct / duration_days / sub_wave_count
        summary_df: 按浪序号汇总的统计表
    """
    if market_data is None:
        market_data = load_universe_data()

    wave_rows: list[dict] = []
    three_wave_stocks = 0
    skipped_no_prior_above = 0

    for key, info in market_data.items():
        df = info["df"]
        result = analyze_wave_structure(df)
        major_waves = result.get("major_waves", [])

        # 取前三浪，且三浪均已完结（有 end_pivot，即不是正在进行的末尾浪）
        first_three = [w for w in major_waves if w["wave_number"] <= 3]
        completed_three = [w for w in first_three if w["end_pivot"] is not None]
        if len(completed_three) < 3:
            continue

        # 验证第一浪起点在 Long Vegas 附近（不是从大底远低于均线处起算）
        w1_start_iloc = completed_three[0]["start_pivot"]["iloc"]
        if not _wave1_starts_near_long_vegas(df, w1_start_iloc):
            skipped_no_prior_above += 1
            continue

        three_wave_stocks += 1
        for w in completed_three:
            wave_rows.append({
                "market": info["market"],
                "symbol": info["symbol"],
                "name": info["name"],
                "wave_number": w["wave_number"],
                "rise_pct": w["rise_pct"],
                "duration_days": w["duration_days"],
                "sub_wave_count": w["sub_wave_count"],
            })

    logger.info(
        f"完整三浪结构股票: {three_wave_stocks} 只 / {len(market_data)} 只"
        f"（跳过 {skipped_no_prior_above} 只：第一浪起点远低于 Long Vegas，视为大底起算）"
    )

    detail_df = pd.DataFrame(wave_rows)
    if detail_df.empty:
        return detail_df, pd.DataFrame()

    rows: list[dict] = []
    for wave_num in [1, 2, 3]:
        sub = detail_df[detail_df["wave_number"] == wave_num]["rise_pct"]
        if sub.empty:
            continue
        rows.append({
            "wave_number": wave_num,
            "stock_count": int(len(sub)),
            "avg_rise_pct": round(float(sub.mean()), 2),
            "median_rise_pct": round(float(sub.median()), 2),
            "p75_rise_pct": round(float(sub.quantile(0.75)), 2),
            "p25_rise_pct": round(float(sub.quantile(0.25)), 2),
            "max_rise_pct": round(float(sub.max()), 2),
            "min_rise_pct": round(float(sub.min()), 2),
            "avg_duration_days": round(float(detail_df[detail_df["wave_number"] == wave_num]["duration_days"].mean()), 1),
        })

    return detail_df, pd.DataFrame(rows)


def summarize_by_wave_number(df: pd.DataFrame) -> pd.DataFrame:
    """
    按浪序号统计信号的平均涨幅（第 1、2、3 浪等）。

    只统计 wave_number >= 1 的信号（wave_number == 0 表示识别失败）。
    对每个浪序号，按持有期分别计算胜率与平均/中位收益。
    """
    if df.empty or "wave_number" not in df.columns:
        return pd.DataFrame()

    valid = df[df["wave_number"] >= 1].copy()
    if valid.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for wave_num in sorted(valid["wave_number"].unique()):
        sub = valid[valid["wave_number"] == wave_num]
        for d in FORWARD_DAYS:
            col = f"ret_{d}d"
            if col not in sub.columns:
                continue
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if vals.empty:
                continue
            rows.append({
                "wave_number": int(wave_num),
                "period": f"{d}d",
                "count": int(len(vals)),
                "win_rate_pct": round(float((vals > 0).mean() * 100), 2),
                "avg_return_pct": round(float(vals.mean()), 2),
                "median_return_pct": round(float(vals.median()), 2),
                "max_return_pct": round(float(vals.max()), 2),
                "min_return_pct": round(float(vals.min()), 2),
            })

    return pd.DataFrame(rows)


def run_backtest(step: int = 3, gap: int = 12) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the NDX100 + HK-focus main-rally pullback backtest and return all result tables."""
    logger.info("=" * 70)
    logger.info("NDX100 + 港股重点列表 主升浪回调（Vegas）回测")
    logger.info("=" * 70)
    logger.info(f"仅统计近 {ANALYSIS_YEARS} 年信号 | 步长={step}d | 最小间隔={gap}d")

    market_data = load_universe_data()
    if not market_data:
        logger.error("无可用标的数据")
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty

    analysis_start = _analysis_start()

    all_hits: list[dict] = []
    for key, info in market_data.items():
        hits = scan_one_symbol(
            market=info["market"],
            symbol=info["symbol"],
            name=info["name"],
            df=info["df"],
            analysis_start=analysis_start,
            step=step,
            min_gap_days=gap,
        )
        if hits:
            logger.success(f"{key}: {len(hits)} 个信号")
            all_hits.extend(hits)

    detail_df = pd.DataFrame(all_hits)
    if detail_df.empty:
        logger.warning("未扫描到任何信号")
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty

    detail_df = detail_df.sort_values(["signal_date", "market", "symbol"]).reset_index(drop=True)
    summary_df = summarize(detail_df)
    support_summary_df = summarize_by_support_type(detail_df)
    symbol_style_df = summarize_symbol_support_styles(detail_df)
    wave_summary_df = summarize_by_wave_number(detail_df)
    _, three_wave_amp_df = analyze_three_wave_amplitudes(market_data)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = detail_df["signal_date"].min().replace("-", "")
    end = detail_df["signal_date"].max().replace("-", "")

    detail_file = OUT_DIR / f"main_rally_pullback_signals_{start}_{end}.csv"
    summary_file = OUT_DIR / f"main_rally_pullback_summary_{start}_{end}.csv"
    support_summary_file = OUT_DIR / f"main_rally_pullback_support_summary_{start}_{end}.csv"
    symbol_style_file = OUT_DIR / f"main_rally_pullback_symbol_styles_{start}_{end}.csv"
    wave_summary_file = OUT_DIR / f"main_rally_pullback_wave_summary_{start}_{end}.csv"
    three_wave_amp_file = OUT_DIR / "three_wave_amplitude_stats.csv"

    # CSV 先于图表保存，避免图表生成中断导致结果丢失
    detail_df.to_csv(detail_file, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
    support_summary_df.to_csv(support_summary_file, index=False, encoding="utf-8-sig")
    symbol_style_df.to_csv(symbol_style_file, index=False, encoding="utf-8-sig")
    wave_summary_df.to_csv(wave_summary_file, index=False, encoding="utf-8-sig")
    if not three_wave_amp_df.empty:
        three_wave_amp_df.to_csv(three_wave_amp_file, index=False, encoding="utf-8-sig")

    logger.info(f"信号明细已保存: {detail_file}")
    logger.info(f"统计汇总已保存: {summary_file}")
    logger.info(f"回撤类型汇总已保存: {support_summary_file}")
    logger.info(f"个股回撤风格已保存: {symbol_style_file}")
    logger.info(f"浪序号汇总已保存: {wave_summary_file}")
    if not three_wave_amp_df.empty:
        logger.info(f"三浪幅度统计已保存: {three_wave_amp_file}")

    # 图表生成（可选，数量多时耗时较长）
    if not detail_df.empty:
        chart_dir = OUT_DIR / "charts"
        plot_main_rally_pullback_signals(all_hits, market_data, chart_dir)

    print("\n" + "=" * 80)
    print("主升浪 Vegas 回调策略回测结果（NDX100 + 港股重点列表）")
    print("=" * 80)
    print(f"信号总数: {len(detail_df)}")
    print(f"涉及标的: {detail_df['symbol'].nunique()} 只")
    print(f"市场分布: {detail_df['market'].value_counts().to_dict()}")
    print(f"回撤类型: {detail_df['support_type'].value_counts().to_dict()}")
    if not summary_df.empty:
        print("\n按持有期统计:")
        print(summary_df.to_string(index=False))
    if not wave_summary_df.empty:
        print("\n按浪序号统计（各浪平均涨幅）:")
        print(wave_summary_df.to_string(index=False))
    if not three_wave_amp_df.empty:
        print("\n完整三浪结构——各浪实际涨幅统计:")
        print("（口径：前三浪均已完结的股票，rise_pct = 浪峰 / 浪起点 - 1）")
        print(three_wave_amp_df.to_string(index=False))
    if not symbol_style_df.empty:
        print("\n个股回撤风格（Top 20）:")
        print(symbol_style_df.head(20).to_string(index=False))

    return detail_df, summary_df, support_summary_df, symbol_style_df, wave_summary_df


def main():
    """CLI entrypoint for the NDX100 + HK-focus main-rally pullback backtest."""
    parser = argparse.ArgumentParser(description="NDX100 + 港股重点列表 主升浪回调（Vegas）回测")
    parser.add_argument("--step", type=int, default=3, help="滚动扫描步长（交易日），默认 3")
    parser.add_argument("--gap", type=int, default=12, help="同一标的信号最小间隔天数，默认 12")
    args = parser.parse_args()

    detail_df, summary_df, support_summary_df, symbol_style_df, wave_summary_df = run_backtest(
        step=args.step, gap=args.gap
    )


if __name__ == "__main__":
    main()
