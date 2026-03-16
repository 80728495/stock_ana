"""
半年期每日强势股票回测

对全部美股回测 6 个月的历史数据，逐日滚动运行异动检测，
按天记录触发强势信号的股票集合。

用法:
    python -m stock_ana.backtest_momentum
    python -m stock_ana.backtest_momentum --months 3 --min-score 4.0

输出:
    data/output/daily_momentum_{start}_{end}.csv        — 每条异动明细
    data/output/daily_momentum_summary_{start}_{end}.csv — 每日强势股票集合
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.config import DATA_DIR, CACHE_DIR, OUTPUT_DIR
from stock_ana.momentum_detector import (
    detect_momentum,
    _LOOKBACK,
    _TRIGGER_SCORE,
    _MIN_HISTORY,
    PROFILES_FILE,
    US_CACHE_DIR,
)


# ═══════════════════ 数据加载 ═══════════════════


def _load_all_prices() -> dict[str, pd.DataFrame]:
    """加载全部有缓存的股票价格数据（US 缓存 + NDX100 回退）。"""
    all_data = {}

    # US cache
    for f in US_CACHE_DIR.glob("*.parquet"):
        ticker = f.stem
        try:
            df = pd.read_parquet(f)
            df.columns = [c.lower() for c in df.columns]
            if len(df) >= _MIN_HISTORY:
                all_data[ticker] = df
        except Exception:
            pass

    # NDX100 cache (不覆盖已有的)
    ndx_dir = CACHE_DIR / "ndx100"
    if ndx_dir.exists():
        for f in ndx_dir.glob("*.parquet"):
            ticker = f.stem
            if ticker not in all_data:
                try:
                    df = pd.read_parquet(f)
                    df.columns = [c.lower() for c in df.columns]
                    if len(df) >= _MIN_HISTORY:
                        all_data[ticker] = df
                except Exception:
                    pass

    return all_data


# ═══════════════════ 核心回测 ═══════════════════


def backtest_daily_momentum(
    months: int = 6,
    lookback: int = _LOOKBACK,
    min_score: float = _TRIGGER_SCORE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    对全部美股回测 N 个月：逐日滚动检测异动信号。

    Args:
        months: 回测月数
        lookback: 异动检测窗口天数
        min_score: 最低触发分数

    Returns:
        (detail_df, summary_df)
        detail_df:  每条异动记录 [date, ticker, score, ...]
        summary_df: 每日汇总 [date, count, tickers]
    """
    logger.info("加载全部价格数据...")
    all_data = _load_all_prices()
    logger.info(f"已加载 {len(all_data)} 只股票")

    # 加载 profiles 用于板块信息
    profiles = pd.read_csv(PROFILES_FILE, encoding="utf-8-sig")
    profile_map = {}
    for _, row in profiles.iterrows():
        profile_map[row["ticker"]] = {
            "sector": row.get("sector", ""),
            "sic_code": row.get("sic_code", ""),
            "sic_description": row.get("sic_description", ""),
            "sub_label": row.get("sub_label", ""),
        }

    # ── 确定回测日期范围 ──
    trading_days = months * 21  # 每月约 21 个交易日

    # 构建主日历（所有股票的日期并集）
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)

    if len(all_dates) < trading_days + _MIN_HISTORY:
        logger.error("数据不足以进行回测")
        return pd.DataFrame(), pd.DataFrame()

    backtest_dates = all_dates[-trading_days:]
    start_date = backtest_dates[0]
    end_date = backtest_dates[-1]

    logger.info(
        f"回测区间: {start_date.strftime('%Y-%m-%d')} ~ "
        f"{end_date.strftime('%Y-%m-%d')} ({len(backtest_dates)} 交易日)"
    )

    # ── 逐股票滚动检测 ──
    detail_rows = []
    total_checks = 0
    t0 = time.time()

    tickers = list(all_data.keys())
    for idx, ticker in enumerate(tickers):
        df = all_data[ticker]
        dates_in_range = df.index[(df.index >= start_date) & (df.index <= end_date)]

        for eval_date in dates_in_range:
            # 截取到 eval_date 的数据
            sub_df = df.loc[:eval_date]
            if len(sub_df) < _MIN_HISTORY:
                continue

            total_checks += 1
            result = detect_momentum(sub_df, lookback=lookback)

            if result["score"] >= min_score:
                sig = result["signals"]
                row = {
                    "date": eval_date,
                    "ticker": ticker,
                    "score": result["score"],
                    "vol_ratio": sig.get("vol_surge", {}).get("ratio", 0),
                    "return_pct": sig.get("abnormal_return", {}).get("pct", 0),
                    "z_score": sig.get("abnormal_return", {}).get("z_score", 0),
                    "breakout": sig.get("breakout", {}).get("level", ""),
                    "gap_pct": sig.get("gap_up", {}).get("max_gap_pct", 0),
                    "accum_days": sig.get("accumulation", {}).get("days", 0),
                }
                # 附加板块信息
                p = profile_map.get(ticker, {})
                row["sector"] = p.get("sector", "")
                row["sic_description"] = p.get("sic_description", "")
                row["sub_label"] = p.get("sub_label", "")
                detail_rows.append(row)

        # 进度日志
        done = idx + 1
        if done % 200 == 0 or done == len(tickers):
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed > 0 else 0
            eta = (len(tickers) - done) / speed if speed > 0 else 0
            logger.info(
                f"进度: {done}/{len(tickers)} 股票 "
                f"({done*100/len(tickers):.0f}%), "
                f"已检测 {total_checks} 次, "
                f"异动 {len(detail_rows)} 条, "
                f"ETA {eta:.0f}s"
            )

    elapsed = time.time() - t0
    logger.info(
        f"回测完成: {total_checks} 次检测, "
        f"{len(detail_rows)} 条异动, 耗时 {elapsed:.1f}s"
    )

    # ── 构建结果 ──
    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        return detail_df, pd.DataFrame()

    detail_df["date"] = pd.to_datetime(detail_df["date"])
    detail_df = detail_df.sort_values(["date", "score"], ascending=[True, False])

    # 每日汇总
    summary_rows = []
    for d in backtest_dates:
        day_hits = detail_df[detail_df["date"] == d]
        tickers_str = ",".join(day_hits["ticker"].tolist())
        summary_rows.append({
            "date": d,
            "count": len(day_hits),
            "tickers": tickers_str,
        })
    summary_df = pd.DataFrame(summary_rows)

    return detail_df, summary_df


# ═══════════════════ CLI ═══════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(description="半年期每日强势股票回测")
    parser.add_argument(
        "--months", type=int, default=6, help="回测月数 (默认 6)"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=_TRIGGER_SCORE,
        help=f"最低触发分数 (默认 {_TRIGGER_SCORE})",
    )
    args = parser.parse_args()

    detail_df, summary_df = backtest_daily_momentum(
        months=args.months,
        min_score=args.min_score,
    )

    if detail_df.empty:
        logger.warning("没有检测到异动记录")
        return

    # ── 保存文件 ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start = detail_df["date"].min().strftime("%Y%m%d")
    end = detail_df["date"].max().strftime("%Y%m%d")

    detail_file = OUTPUT_DIR / f"daily_momentum_{start}_{end}.csv"
    summary_file = OUTPUT_DIR / f"daily_momentum_summary_{start}_{end}.csv"

    detail_df.to_csv(detail_file, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")

    logger.info(f"详细记录已保存: {detail_file}")
    logger.info(f"每日汇总已保存: {summary_file}")

    # ── 统计摘要 ──
    print(f"\n{'='*60}")
    print(f"  回测期间: {start} ~ {end}")
    print(f"  交易日数: {len(summary_df)}")
    print(f"  异动总数: {len(detail_df)} 条")
    print(f"  日均异动: {len(detail_df) / max(len(summary_df), 1):.1f} 只")
    print(f"  最多异动: {summary_df['count'].max()} 只 "
          f"({summary_df.loc[summary_df['count'].idxmax(), 'date'].strftime('%Y-%m-%d')})")
    print(f"  涉及股票: {detail_df['ticker'].nunique()} 只")
    print(f"{'='*60}")

    # 板块分布
    if "sector" in detail_df.columns:
        sector_counts = (
            detail_df.groupby("sector")
            .size()
            .sort_values(ascending=False)
        )
        print("\n板块异动次数分布:")
        for sector, count in sector_counts.items():
            pct = count / len(detail_df) * 100
            print(f"  {sector:30s}  {count:5d} ({pct:5.1f}%)")
    print()


if __name__ == "__main__":
    main()
