"""
多线压缩策略 — 全量回测

对全部美股回测：在每只股票的全部可用数据范围内，逐日扫描均线压缩形态（第一阶段）
与量价突破（第二阶段），并计算前瞻收益率。

用法:
    python -m stock_ana.backtest.backtest_ma_squeeze

输出:
    data/output/ma_squeeze_signals_{start}_{end}.csv   — 每条信号明细
    data/output/ma_squeeze_stats_{start}_{end}.csv     — 统计汇总
"""""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.config import DATA_DIR, OUTPUT_DIR
from stock_ana.data.market_data import load_universe_data
from stock_ana.strategies.impl.ma_squeeze import (
    detect_stage1,
    detect_stage2,
    _MIN_HISTORY,
)


PROFILES_FILE = DATA_DIR / "us_sec_profiles.csv"

# 前瞻收益计算周期
_FORWARD_DAYS = [5, 10, 20]


# ═══════════════════ 数据加载 ═══════════════════

def _load_all_prices() -> dict[str, pd.DataFrame]:
    """加载全部有缓存的股票价格数据（统一门面：US 优先 + NDX100 补缺）。"""
    return load_universe_data("us+ndx100", min_history=_MIN_HISTORY)


# ═══════════════════ 核心回测 ═══════════════════

def backtest_ma_squeeze() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    全量回测多线压缩策略。

    对每只股票，从有足够数据（>= _MIN_HISTORY）的第一天开始逐日检测：
      - 当日是否触发第一阶段（发现点）
      - 当日是否触发第二阶段（确认点）
    并计算前瞻收益。

    Returns:
        (signals_df, stats_df)
    """
    logger.info("加载全部价格数据...")
    all_data = _load_all_prices()
    logger.info(f"已加载 {len(all_data)} 只股票")

    profiles = pd.read_csv(PROFILES_FILE, encoding="utf-8-sig")
    profile_map = {}
    for _, row in profiles.iterrows():
        profile_map[row["ticker"]] = {
            "company_name": row.get("company_name", ""),
            "sector": row.get("sector", ""),
            "sub_label": row.get("sub_label", ""),
        }

    # 确定回测日期范围：每只股票从第 _MIN_HISTORY 天开始扫描
    # 先统计全局最早/最晚日期
    global_start = None
    global_end = None
    for df in all_data.values():
        if len(df) > _MIN_HISTORY:
            s = df.index[_MIN_HISTORY]
            e = df.index[-1]
            if global_start is None or s < global_start:
                global_start = s
            if global_end is None or e > global_end:
                global_end = e

    logger.info(
        f"回测范围: {global_start.strftime('%Y-%m-%d')} ~ "
        f"{global_end.strftime('%Y-%m-%d')} (全量数据)"
    )

    # ── 逐日滚动检测 ──
    signal_rows = []
    stage1_count = 0
    stage2_count = 0
    total_checks = 0
    t0 = time.time()

    tickers = list(all_data.keys())

    # 追踪每只股票近期是否有第一阶段信号（用于跳过第二阶段检测）
    # key=ticker, value=最近一次 stage1 触发的日期
    stage1_last = {}
    # 追踪每只股票前一天是否处于 S1 状态（用于只记录首次发现日）
    stage1_active = {}

    for idx, ticker in enumerate(tickers):
        df = all_data[ticker]
        # 从第 _MIN_HISTORY 天开始扫描到最后一天
        scan_dates = df.index[_MIN_HISTORY:]

        for eval_date in scan_dates:
            sub_df = df.loc[:eval_date]
            if len(sub_df) < _MIN_HISTORY:
                continue

            total_checks += 1

            # 第一阶段检测
            s1 = detect_stage1(sub_df)
            if s1["triggered"]:
                was_active = stage1_active.get(ticker, False)
                stage1_active[ticker] = True
                stage1_last[ticker] = eval_date

                # 只在首次发现日（从不满足→满足的第一天）记录信号
                if not was_active:
                    stage1_count += 1
                    signal_rows.append({
                        "date": eval_date,
                        "ticker": ticker,
                        "stage": 1,
                        "ma_squeeze_ratio": s1["details"].get("ma_squeeze_ratio"),
                        "ma_bullish_order": s1["details"].get("ma_bullish_order"),
                        "prange_10d": s1["details"].get("prange_10d"),
                        "prange_20d": s1["details"].get("prange_20d"),
                        "prange_60d": s1["details"].get("prange_60d"),
                        "prange_90d": s1["details"].get("prange_90d"),
                        "vol_ratio_5d_20d": s1["details"].get("vol_ratio_5d_20d"),
                        "daily_gain_pct": None,
                        "vol_ratio": None,
                        "golden_cross_count": None,
                        "price_ratio": None,
                        # 前瞻收益（第一阶段也计算）
                        **_calc_forward_returns(df, eval_date),
                    })
            else:
                stage1_active[ticker] = False

            # 第二阶段检测（仅在近10日内有stage1的股票上运行）
            last_s1 = stage1_last.get(ticker)
            if last_s1 is not None:
                day_diff = len(df.loc[last_s1:eval_date]) - 1
                if 0 < day_diff <= 15:
                    s2 = detect_stage2(sub_df)
                    if s2["triggered"]:
                        stage2_count += 1
                        # 触发后清除，避免重复触发
                        stage1_last.pop(ticker, None)

                        signal_rows.append({
                            "date": eval_date,
                            "ticker": ticker,
                            "stage": 2,
                            "s2_score": s2.get("score", 0),
                            "confirm_signals": ", ".join(
                                s2["details"].get("confirm_signals", [])
                            ),
                            "ma_squeeze_ratio": None,
                            "ma_bullish_order": None,
                            "daily_gain_pct": s2["details"].get("daily_gain_pct"),
                            "vol_ratio": s2["details"].get("vol_ratio"),
                            "price_ratio": s2["details"].get("price_ratio"),
                            **_calc_forward_returns(df, eval_date),
                        })

        # 进度日志
        done = idx + 1
        if done % 200 == 0 or done == len(tickers):
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed > 0 else 0
            eta = (len(tickers) - done) / speed if speed > 0 else 0
            logger.info(
                f"进度: {done}/{len(tickers)} 股票 "
                f"({done*100//len(tickers)}%), "
                f"检测 {total_checks} 次, "
                f"S1={stage1_count} S2={stage2_count}, "
                f"ETA {eta:.0f}s"
            )

    elapsed = time.time() - t0
    logger.info(
        f"回测完成: {total_checks} 次检测, "
        f"S1 信号 {stage1_count}, S2 信号 {stage2_count}, "
        f"耗时 {elapsed:.1f}s"
    )

    # ── 构建结果 ──
    signals_df = pd.DataFrame(signal_rows)
    if signals_df.empty:
        logger.warning("未检测到任何信号")
        return signals_df, pd.DataFrame()

    signals_df["date"] = pd.to_datetime(signals_df["date"])

    # 补充公司信息
    for col in ["company_name", "sector", "sub_label"]:
        signals_df[col] = signals_df["ticker"].map(
            lambda t, c=col: profile_map.get(t, {}).get(c, "")
        )

    signals_df = signals_df.sort_values(["date", "stage", "ticker"])

    # ── 统计汇总 ──
    stats_df = _compute_stats(signals_df)

    return signals_df, stats_df


def _calc_forward_returns(df: pd.DataFrame, eval_date) -> dict:
    """计算信号触发后 N 日的前瞻收益率。"""
    result = {}
    try:
        loc = df.index.get_loc(eval_date)
        entry_price = df["close"].iloc[loc]
    except (KeyError, IndexError):
        for d in _FORWARD_DAYS:
            result[f"fwd_{d}d_pct"] = None
        return result

    for d in _FORWARD_DAYS:
        target_loc = loc + d
        if target_loc < len(df):
            fwd_price = df["close"].iloc[target_loc]
            result[f"fwd_{d}d_pct"] = round((fwd_price / entry_price - 1) * 100, 2)
        else:
            result[f"fwd_{d}d_pct"] = None

    return result


def _compute_stats(signals_df: pd.DataFrame) -> pd.DataFrame:
    """计算各阶段信号的收益统计。"""
    rows = []
    for stage in [1, 2]:
        sub = signals_df[signals_df["stage"] == stage]
        if sub.empty:
            continue
        row = {
            "stage": stage,
            "total_signals": len(sub),
            "unique_tickers": sub["ticker"].nunique(),
        }
        for d in _FORWARD_DAYS:
            col = f"fwd_{d}d_pct"
            valid = sub[col].dropna()
            if len(valid) > 0:
                row[f"fwd_{d}d_avg"] = round(valid.mean(), 2)
                row[f"fwd_{d}d_median"] = round(valid.median(), 2)
                row[f"fwd_{d}d_win_rate"] = round((valid > 0).mean() * 100, 1)
                row[f"fwd_{d}d_count"] = len(valid)
            else:
                row[f"fwd_{d}d_avg"] = None
                row[f"fwd_{d}d_median"] = None
                row[f"fwd_{d}d_win_rate"] = None
                row[f"fwd_{d}d_count"] = 0
        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════ CLI ═══════════════════

def main():
    """Run the MA squeeze backtest from the command line and persist reports."""
    import argparse

    parser = argparse.ArgumentParser(description="多线压缩策略 - 全量回测")
    args = parser.parse_args()

    signals_df, stats_df = backtest_ma_squeeze()

    if signals_df.empty:
        print("回测未产生任何信号")
        return

    # 保存结果
    start_str = signals_df["date"].min().strftime("%Y%m%d")
    end_str = signals_df["date"].max().strftime("%Y%m%d")

    signals_path = OUTPUT_DIR / f"ma_squeeze_signals_{start_str}_{end_str}.csv"
    stats_path = OUTPUT_DIR / f"ma_squeeze_stats_{start_str}_{end_str}.csv"

    signals_df.to_csv(signals_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    logger.info(f"信号明细: {signals_path} ({len(signals_df)} 条)")
    logger.info(f"统计汇总: {stats_path}")

    # 打印统计
    print(f"\n{'='*70}")
    print(f"  多线压缩策略回测结果")
    print(f"  {start_str} ~ {end_str}")
    print(f"{'='*70}\n")

    for _, row in stats_df.iterrows():
        stage = int(row["stage"])
        stage_name = "第一阶段（发现点）" if stage == 1 else "第二阶段（确认点）"
        print(f"  {stage_name}")
        print(f"    信号数: {int(row['total_signals'])}, "
              f"涉及 {int(row['unique_tickers'])} 只股票")

        for d in _FORWARD_DAYS:
            avg = row.get(f"fwd_{d}d_avg")
            med = row.get(f"fwd_{d}d_median")
            wr = row.get(f"fwd_{d}d_win_rate")
            cnt = row.get(f"fwd_{d}d_count", 0)
            if avg is not None:
                print(f"    {d}日前瞻: 均值 {avg:+.2f}%, "
                      f"中位 {med:+.2f}%, "
                      f"胜率 {wr:.1f}% "
                      f"(N={int(cnt)})")
        print()

    # 按质量指标分组打印 S1 统计
    s1_df = signals_df[signals_df["stage"] == 1]
    if not s1_df.empty and "prange_20d" in s1_df.columns:
        print(f"  ── S1 按价格区间宽度(20d)分组 ──")
        for label, mask in [
            ("pr20d<6%", s1_df["prange_20d"] < 6),
            ("6%<=pr20d<10%", (s1_df["prange_20d"] >= 6) & (s1_df["prange_20d"] < 10)),
            ("pr20d>=10%", s1_df["prange_20d"] >= 10),
        ]:
            sub = s1_df[mask]
            if sub.empty:
                continue
            print(f"    {label}: {len(sub)} 信号, {sub['ticker'].nunique()} 只")
            for d in _FORWARD_DAYS:
                col = f"fwd_{d}d_pct"
                valid = sub[col].dropna()
                if len(valid) > 0:
                    print(f"      {d}日: 均值 {valid.mean():+.2f}%, "
                          f"中位 {valid.median():+.2f}%, "
                          f"胜率 {(valid > 0).mean()*100:.1f}% "
                          f"(N={len(valid)})")
            print()

    # 打印最近的 Stage2 信号
    s2 = signals_df[signals_df["stage"] == 2].tail(20)
    if not s2.empty:
        print(f"  ── 最近 Stage2 信号 (最多 20 条) ──")
        for _, row in s2.iterrows():
            print(f"    {row['date'].strftime('%Y-%m-%d')} "
                  f"{row['ticker']:6s} {row.get('company_name',''):25s} "
                  f"涨幅 {row.get('daily_gain_pct',''):>6}% "
                  f"量比 {row.get('vol_ratio',''):>5}x "
                  f"[{row.get('sector','')}]")
        print()


if __name__ == "__main__":
    main()
