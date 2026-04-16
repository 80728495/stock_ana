"""
多线压缩策略 (MA Squeeze Strategy)

基于多条均线收敛后量价突破的两阶段选股策略：

第一阶段 — 发现点（均线压缩形态）:
  1. 均线收敛：MA30/MA60/MA200 极差比 < 5%
  2. 三线向上：MA30/MA60/MA200 均呈上升趋势
  3. 多头排列：MA30 > MA200 且 MA60 > MA200

第二阶段 — 确认点（量价突破）:
  1. 大阳线突破：日涨幅 > 5%
  2. 均线拐头：MA5/MA10/MA30 全部向上
  3. 三线金叉：3日内 MA5/MA10/MA30 完成金叉
  4. 量能放大：当日量 > 5日均量 × 2 且 5日均量 > 10日均量
  5. 价格区间：收盘价在均线均价的 1.04~1.15 倍区间

用法:
    python -m stock_ana.strategies.impl.ma_squeeze --scan            # 扫描当前符合条件的股票
    python -m stock_ana.strategies.impl.ma_squeeze --scan --stage 1  # 仅扫描第一阶段（发现点）
    python -m stock_ana.strategies.impl.ma_squeeze --ticker NVDA     # 检测单只股票
"""

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.data.market_data import load_symbol_data, load_universe_data
from stock_ana.strategies.primitives.squeeze import (
    compute_ma_squeeze_ratio,
    compute_volume_trend_ratio,
    is_recent_crossover,
    normalized_price_range,
)


# ═══════════════════ 参数 ═══════════════════

# 第一阶段参数
_MA_SQUEEZE_RATIO = 1.05      # 均线收敛度阈值（MA max/min < 1.05）
_PRANGE_10D_MAX = 0.08        # 10日价格区间宽度上限 8%
_PRANGE_20D_MAX = 0.10        # 20日价格区间宽度上限 10%
_PRANGE_60D_MAX = 0.15        # 60日价格区间宽度上限 15%
_PRANGE_90D_MAX = 0.18        # 90日价格区间宽度上限 18%
_VOL_RATIO_MAX = 1.15         # 量能趋势上限: 5日均量/20日均量 < 1.15

# 第二阶段参数（评分制，满足 ≥2 个确认信号即触发）
_BREAKOUT_GAIN_PCT = 0.02     # 单日涨幅 ≥ 2% 视为突破性阳线
_VOLUME_CONFIRM_MULT = 1.5    # 当日量 ≥ 1.5 倍 5 日均量
_GOLDEN_CROSS_WINDOW = 5      # 金叉判定窗口
_STAGE2_MIN_SCORE = 2         # 最少满足 2 个确认信号

# 最低数据需求
_MIN_HISTORY = 210            # 至少需要 210 日数据（覆盖 MA200 + 趋势回看）


# ═══════════════════ 第一阶段：发现点 ═══════════════════

def detect_stage1(df: pd.DataFrame) -> dict:
    """
    检测第一阶段：均线压缩形态（发现点）。

    Args:
        df: OHLCV DataFrame，index 为日期，列包含 open/high/low/close/volume

    Returns:
        {
            "triggered": bool,
            "details": {
                "ma_converged": bool,         # 均线收敛
                "ma_squeeze_ratio": float,    # 均线极差比
                "ma_bullish_order": bool,     # 多头排列 MA30>MA200 & MA60>MA200
                "all_ma_rising": bool,        # 三线同时向上
                "quality_ok": bool,           # 质量过滤（窄价格区间+量能平稳）
                "prange_10d": float,          # 10日价格区间宽度%
                "prange_20d": float,          # 20日价格区间宽度%
                "prange_60d": float,          # 60日价格区间宽度%
                "prange_90d": float,          # 90日价格区间宽度%
                "vol_ratio_5d_20d": float,    # 量能趋势
            }
        }
    """
    empty = {"triggered": False, "details": {}}

    if df is None or len(df) < _MIN_HISTORY:
        return {**empty, "reason": "insufficient_data"}

    close = df["close"]
    volume = df["volume"]
    n = len(df)

    # ── 1. 均线收敛：MA30/MA60/MA200 极差比 < 5% ──
    ma30_series = close.rolling(30).mean()
    ma60_series = close.rolling(60).mean()
    ma200_series = close.rolling(200).mean()

    ma30 = ma30_series.iloc[-1]
    ma60 = ma60_series.iloc[-1]
    ma200 = ma200_series.iloc[-1]

    ma_squeeze_ratio = compute_ma_squeeze_ratio([ma30, ma60, ma200])
    ma_converged = ma_squeeze_ratio < _MA_SQUEEZE_RATIO

    # ── 2. 三线向上：MA30/MA60 上升，MA200 过去一个月至少不下降（走平或上升）──
    ma30_rising = ma30_series.iloc[-1] > ma30_series.iloc[-6]   # 5日前
    ma60_rising = ma60_series.iloc[-1] > ma60_series.iloc[-6]
    ma200_not_falling = ma200_series.iloc[-1] >= ma200_series.iloc[-21]  # 20日前，不下降即可
    all_ma_rising = ma30_rising and ma60_rising and ma200_not_falling

    # ── 3. 多头排列：MA30 > MA200 且 MA60 > MA200 ──
    ma_bullish_order = (ma30 > ma200) and (ma60 > ma200)

    # ── 4. 质量过滤：窄价格区间 + 量能平稳 ──
    # 价格区间宽度 = (max_close - min_close) / mean_close
    prange_10d = normalized_price_range(close.iloc[-10:])
    prange_20d = normalized_price_range(close.iloc[-20:])
    prange_60d = normalized_price_range(close.iloc[-60:]) if n >= 60 else 1.0
    prange_90d = normalized_price_range(close.iloc[-90:]) if n >= 90 else 1.0

    vol_ratio = compute_volume_trend_ratio(volume, short_window=5, long_window=20)

    quality_ok = (
        prange_10d < _PRANGE_10D_MAX
        and prange_20d < _PRANGE_20D_MAX
        and prange_60d < _PRANGE_60D_MAX
        and prange_90d < _PRANGE_90D_MAX
        and vol_ratio < _VOL_RATIO_MAX
    )

    triggered = ma_converged and all_ma_rising and ma_bullish_order and quality_ok

    return {
        "triggered": triggered,
        "details": {
            "ma_converged": ma_converged,
            "ma_squeeze_ratio": round(ma_squeeze_ratio, 4),
            "ma30": round(ma30, 2),
            "ma60": round(ma60, 2),
            "ma200": round(ma200, 2),
            "ma_bullish_order": ma_bullish_order,
            "ma30_rising": ma30_rising,
            "ma60_rising": ma60_rising,
            "ma200_not_falling": ma200_not_falling,
            "all_ma_rising": all_ma_rising,
            "quality_ok": quality_ok,
            "prange_10d": round(prange_10d * 100, 1),
            "prange_20d": round(prange_20d * 100, 1),
            "prange_60d": round(prange_60d * 100, 1),
            "prange_90d": round(prange_90d * 100, 1),
            "vol_ratio_5d_20d": round(vol_ratio, 2),
        },
    }


# ═══════════════════ 第二阶段：确认点 ═══════════════════

def detect_stage2(df: pd.DataFrame) -> dict:
    """
    检测第二阶段：确认压缩形态向上突破（确认点）。

    设计思想：这是机会发现系统，不是自动交易。第二阶段用于验证
    第一阶段的压缩形态已开始向上解除，最终由人工确认。
    因此采用评分制（OR 逻辑）——5 个确认信号满足 ≥2 个即触发。

    5 个确认信号：
      1. 价格突破：日涨幅 ≥ 2% 或 收盘创 10 日新高
      2. 量能确认：当日量 ≥ 1.5 倍 5 日均量
      3. 均线拐头：MA5 向上 且 价格站上 MA5
      4. 金叉出现：5 日内任一组均线金叉（MA5/MA10/MA30）
      5. 脱离压缩区：收盘价 > 均线均价 × 1.02

    Returns:
        {
            "triggered": bool,
            "score": int,          # 满足的确认信号数
            "details": {...},
        }
    """
    empty = {"triggered": False, "score": 0, "details": {}}

    if df is None or len(df) < _MIN_HISTORY:
        return {**empty, "reason": "insufficient_data"}

    close = df["close"]
    volume = df["volume"]
    n = len(df)

    # ── 前置条件：近 10 个交易日内曾出现过第一阶段信号 ──
    had_stage1 = False
    for lookback in range(1, 11):
        sub = df.iloc[:n - lookback]
        if len(sub) >= _MIN_HISTORY:
            s1 = detect_stage1(sub)
            if s1["triggered"]:
                had_stage1 = True
                break

    if not had_stage1:
        return {**empty, "details": {"had_stage1_recently": False}}

    c_today = close.iloc[-1]
    c_yesterday = close.iloc[-2]

    confirm_signals = []  # 收集触发的信号名称
    score = 0

    # ── 信号 1: 价格突破（日涨 ≥ 2% 或 创 10 日新高）──
    daily_gain = (c_today / c_yesterday - 1) if c_yesterday > 0 else 0
    high_10d = close.iloc[-11:-1].max()  # 不含当日
    price_breakout = (daily_gain >= _BREAKOUT_GAIN_PCT) or (c_today > high_10d)
    if price_breakout:
        score += 1
        confirm_signals.append("价格突破")

    # ── 信号 2: 量能确认（当日量 ≥ 1.5 倍 5 日均量）──
    vol_today = volume.iloc[-1]
    vol_ma5 = volume.iloc[-6:-1].mean()
    vol_ratio = vol_today / vol_ma5 if vol_ma5 > 0 else 0
    volume_confirm = vol_ratio >= _VOLUME_CONFIRM_MULT
    if volume_confirm:
        score += 1
        confirm_signals.append("量能放大")

    # ── 信号 3: 均线拐头（MA5 向上 且 价格站上 MA5）──
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma30 = close.rolling(30).mean()

    ma5_up = ma5.iloc[-1] > ma5.iloc[-2]
    ma10_up = ma10.iloc[-1] > ma10.iloc[-2]
    above_ma5 = c_today > ma5.iloc[-1]
    ma_turning = ma5_up and above_ma5
    if ma_turning:
        score += 1
        confirm_signals.append("均线拐头")

    # ── 信号 4: 金叉出现（5 日内任一组金叉）──
    cross_5_10 = is_recent_crossover(ma5, ma10, window=_GOLDEN_CROSS_WINDOW)
    cross_5_30 = is_recent_crossover(ma5, ma30, window=_GOLDEN_CROSS_WINDOW)
    cross_10_30 = is_recent_crossover(ma10, ma30, window=_GOLDEN_CROSS_WINDOW)
    golden_cross = cross_5_10 or cross_5_30 or cross_10_30
    if golden_cross:
        score += 1
        confirm_signals.append("金叉")

    # ── 信号 5: 脱离压缩区（收盘价 > 均线均价 × 1.02）──
    ma60_val = close.rolling(60).mean().iloc[-1]
    ma200_val = close.rolling(200).mean().iloc[-1]
    avg_e = (ma30.iloc[-1] + ma60_val + ma200_val) / 3
    price_ratio = c_today / avg_e if avg_e > 0 else 0
    breakout_zone = price_ratio >= 1.02
    if breakout_zone:
        score += 1
        confirm_signals.append("脱离压缩区")

    triggered = score >= _STAGE2_MIN_SCORE

    return {
        "triggered": triggered,
        "score": score,
        "details": {
            "had_stage1_recently": had_stage1,
            "confirm_signals": confirm_signals,
            "price_breakout": price_breakout,
            "daily_gain_pct": round(daily_gain * 100, 2),
            "new_10d_high": c_today > high_10d,
            "volume_confirm": volume_confirm,
            "vol_ratio": round(vol_ratio, 2),
            "ma_turning": ma_turning,
            "ma5_up": ma5_up,
            "ma10_up": ma10_up,
            "above_ma5": above_ma5,
            "golden_cross": golden_cross,
            "cross_5_10": cross_5_10,
            "cross_5_30": cross_5_30,
            "cross_10_30": cross_10_30,
            "breakout_zone": breakout_zone,
            "price_ratio": round(price_ratio, 4),
        },
    }


# ═══════════════════ 合并检测 ═══════════════════

def detect_squeeze(df: pd.DataFrame) -> dict:
    """
    完整两阶段检测。

    Returns:
        {
            "stage1": {"triggered": bool, "details": {...}},
            "stage2": {"triggered": bool, "details": {...}},
        }
    """
    s1 = detect_stage1(df)
    s2 = detect_stage2(df)
    return {"stage1": s1, "stage2": s2}


# ═══════════════════ 批量扫描 ═══════════════════

def scan_universe(
    stage: int = 0,
) -> pd.DataFrame:
    """
    扫描全部美股的均线压缩信号。

    Args:
        stage: 0=两阶段都扫，1=仅第一阶段，2=仅第二阶段

    Returns:
        DataFrame 包含所有触发信号的股票
    """
    from stock_ana.config import DATA_DIR

    profiles = pd.read_csv(DATA_DIR / "us_sec_profiles.csv", encoding="utf-8-sig")
    tickers = profiles["ticker"].tolist()

    name_map = profiles.set_index("ticker")["company_name"].to_dict()
    sector_map = profiles.set_index("ticker")["sector"].to_dict()
    sub_map = profiles.set_index("ticker")["sub_label"].to_dict()

    all_prices = load_universe_data("us+ndx100", min_history=_MIN_HISTORY)

    rows = []
    checked = 0
    for t in tickers:
        df = all_prices.get(t)
        if df is None or len(df) < _MIN_HISTORY:
            continue
        checked += 1

        s1_triggered = False
        s2_triggered = False
        s1_details = {}
        s2_details = {}

        if stage in (0, 1):
            s1 = detect_stage1(df)
            s1_triggered = s1["triggered"]
            s1_details = s1.get("details", {})

        if stage in (0, 2):
            s2 = detect_stage2(df)
            s2_triggered = s2["triggered"]
            s2_details = s2.get("details", {})

        if not s1_triggered and not s2_triggered:
            continue

        row = {
            "ticker": t,
            "company_name": name_map.get(t, ""),
            "sector": sector_map.get(t, ""),
            "sub_label": sub_map.get(t, ""),
            "stage1": s1_triggered,
            "stage2": s2_triggered,
        }
        # 第一阶段指标
        if s1_details:
            row["ma_squeeze_ratio"] = s1_details.get("ma_squeeze_ratio", None)
            row["ma_bullish_order"] = s1_details.get("ma_bullish_order", None)
            row["prange_10d"] = s1_details.get("prange_10d", None)
            row["prange_20d"] = s1_details.get("prange_20d", None)
            row["prange_60d"] = s1_details.get("prange_60d", None)
            row["prange_90d"] = s1_details.get("prange_90d", None)
            row["vol_ratio_5d_20d"] = s1_details.get("vol_ratio_5d_20d", None)
        # 第二阶段指标
        if s2_details:
            row["s2_score"] = s2.get("score", 0)
            row["confirm_signals"] = ", ".join(s2_details.get("confirm_signals", []))
            row["daily_gain_pct"] = s2_details.get("daily_gain_pct", None)
            row["vol_ratio"] = s2_details.get("vol_ratio", None)
            row["price_ratio"] = s2_details.get("price_ratio", None)

        rows.append(row)

    logger.info(f"扫描完成: 检测 {checked} 只, "
                f"第一阶段 {sum(1 for r in rows if r.get('stage1'))} 只, "
                f"第二阶段 {sum(1 for r in rows if r.get('stage2'))} 只")

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["stage2", "stage1"], ascending=[False, False]
        )
    return out


# ═══════════════════ CLI ═══════════════════

def main():
    """CLI entrypoint for manual MA squeeze scans and single-symbol inspection."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="多线压缩策略 - 均线收敛突破选股")
    parser.add_argument("--scan", action="store_true", help="扫描全市场")
    parser.add_argument("--stage", type=int, default=0,
                        help="1=仅第一阶段, 2=仅第二阶段, 0=全部 (默认 0)")
    parser.add_argument("--ticker", type=str, help="检测单只股票")
    args = parser.parse_args()

    if args.ticker:
        ticker = args.ticker.upper()
        df = load_symbol_data(ticker, universe="us+ndx100")
        if df is None:
            logger.error(f"未找到 {ticker} 的缓存数据")
            sys.exit(1)

        result = detect_squeeze(df)
        print(f"\n{'='*60}")
        print(f"  {ticker} 多线压缩策略检测")
        print(f"{'='*60}")

        for stage_name, stage_key in [("第一阶段（发现点）", "stage1"),
                                       ("第二阶段（确认点）", "stage2")]:
            stage = result[stage_key]
            flag = "✓ 触发" if stage["triggered"] else "✗ 未触发"
            score_info = f" (确认 {stage.get('score', '-')}/5)" if stage_key == "stage2" else ""
            print(f"\n  {stage_name}: {flag}{score_info}")
            details = stage.get("details", {})
            # 先显示确认信号列表
            if stage_key == "stage2" and details.get("confirm_signals"):
                print(f"    → 触发信号: {', '.join(details['confirm_signals'])}")
            for k, v in details.items():
                if k == "confirm_signals":
                    continue
                if isinstance(v, bool):
                    marker = "●" if v else "○"
                    print(f"    {marker} {k}: {v}")
                else:
                    print(f"      {k}: {v}")
        print()
        return

    if args.scan:
        result = scan_universe(stage=args.stage)
        if result.empty:
            print("未检测到符合条件的股票")
            return

        stage_label = {0: "全部", 1: "第一阶段", 2: "第二阶段"}.get(args.stage, "")
        print(f"\n{'='*60}")
        print(f"  多线压缩策略扫描结果 ({stage_label})")
        print(f"{'='*60}\n")

        if args.stage in (0, 2):
            s2 = result[result["stage2"] == True]
            if not s2.empty:
                print(f"  ── 第二阶段（确认点）: {len(s2)} 只 ──")
                for _, row in s2.iterrows():
                    signals = row.get('confirm_signals', '')
                    s2_score = row.get('s2_score', 0)
                    print(f"    {row['ticker']:6s} {row.get('company_name',''):30s} "
                          f"确认{int(s2_score)}/5 [{signals}] "
                          f"[{row.get('sector','')}]")
                print()

        if args.stage in (0, 1):
            s1 = result[result["stage1"] == True]
            if args.stage == 0:
                s1 = s1[s1["stage2"] != True]  # 避免重复显示
            if not s1.empty:
                print(f"  ── 第一阶段（发现点）: {len(s1)} 只 ──")
                for _, row in s1.iterrows():
                    print(f"    {row['ticker']:6s} {row.get('company_name',''):30s} "
                          f"压缩比 {row.get('ma_squeeze_ratio','N/A'):>6} "
                          f"pr10d {row.get('prange_10d','N/A'):>5}% "
                          f"pr60d {row.get('prange_60d','N/A'):>5}% "
                          f"pr90d {row.get('prange_90d','N/A'):>5}% "
                          f"[{row.get('sector','')}]")
                print()

        print(f"  共 {len(result)} 只股票\n")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
