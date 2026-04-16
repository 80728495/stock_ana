#!/usr/bin/env python3
"""
Mid Vegas 回踩三种结局分析（后验统计）

每次回踩 Mid Vegas 后，有三种结局：
  A. 继续上涨 (CONTINUE)  — 回踩后价格上涨，mid持续在long之上
  B. 直接跌穿 (BREAK_DIRECT) — 回踩后很快跌破 mid Vegas 且不再收回，向 long Vegas 甚至更低走
  C. 反复试探后跌穿 (BREAK_MULTI_TEST) — 多次回踩 mid Vegas 后最终跌破

判定规则（基于事后 63 个交易日的走势）：
  - 计算价格跌破 mid Vegas 下沿的天数占比
  - 如果最终破位（价格跌至 long Vegas 附近或以下），区分"直接破位"和"反复试探后破位"
  - 如果始终在 mid Vegas 上方运行，归类为"继续上涨"

目的：统计在什么结构条件下，"继续上涨"的概率更高。

用法:
    python analyze_three_wave_precision.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.config import CACHE_DIR, OUTPUT_DIR

# --- Chinese fonts ---
plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang HK", "STHeiti", "Songti SC", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# --- Stock list ---
SAMPLES = {
    "APP":   ("US", "APP",      CACHE_DIR / "us" / "APP.parquet"),
    "NVDA":  ("US", "NVDA",     CACHE_DIR / "us" / "NVDA.parquet"),
    "META":  ("US", "Meta",     CACHE_DIR / "us" / "META.parquet"),
    "TSLA":  ("US", "TSLA",     CACHE_DIR / "us" / "TSLA.parquet"),
    "AMD":   ("US", "AMD",      CACHE_DIR / "us" / "AMD.parquet"),
    "RBLX":  ("US", "ROBLOX",   CACHE_DIR / "us" / "RBLX.parquet"),
    "MRNA":  ("US", "MRNA",     CACHE_DIR / "us" / "MRNA.parquet"),
    "MU":    ("US", "MU",       CACHE_DIR / "us" / "MU.parquet"),
    "TEM":   ("US", "TempusAI", CACHE_DIR / "us" / "TEM.parquet"),
    "ALAB":  ("US", "ALAB",     CACHE_DIR / "us" / "ALAB.parquet"),
    "PDD":   ("US", "PDD",      CACHE_DIR / "us" / "PDD.parquet"),
    "MSFT":  ("US", "MSFT",     CACHE_DIR / "us" / "MSFT.parquet"),
    "GOOG":  ("US", "GOOG",     CACHE_DIR / "us" / "GOOG.parquet"),
    "09992": ("HK", "PopMart",  CACHE_DIR / "hk" / "09992.parquet"),
    "00700": ("HK", "Tencent",  CACHE_DIR / "hk" / "00700.parquet"),
    "01810": ("HK", "Xiaomi",   CACHE_DIR / "hk" / "01810.parquet"),
    "09988": ("HK", "Alibaba",  CACHE_DIR / "hk" / "09988.parquet"),
    "00981": ("HK", "SMIC",     CACHE_DIR / "hk" / "00981.parquet"),
    "03690": ("HK", "Meituan",  CACHE_DIR / "hk" / "03690.parquet"),
    "01347": ("HK", "HuaHong",  CACHE_DIR / "hk" / "01347.parquet"),
    "02400": ("HK", "XD_Inc",   CACHE_DIR / "hk" / "02400.parquet"),
    "01024": ("HK", "Kuaishou", CACHE_DIR / "hk" / "01024.parquet"),
    "09626": ("HK", "Bilibili", CACHE_DIR / "hk" / "09626.parquet"),
    "00189": ("HK", "Dongyue",  CACHE_DIR / "hk" / "00189.parquet"),
    "06869": ("HK", "CF_Fiber", CACHE_DIR / "hk" / "06869.parquet"),
    "02228": ("HK", "Jingta",   CACHE_DIR / "hk" / "02228.parquet"),
    "02788": ("HK", "ChuangXin",CACHE_DIR / "hk" / "02788.parquet"),
}

SIGNAL_CSV = OUTPUT_DIR / "vegas_wave_strategy" / "vegas_wave_strategy_signals.csv"
OUT_DIR = OUTPUT_DIR / "vegas_wave_strategy" / "three_outcome"

MID_EMAS = [34, 55, 60]
LONG_EMAS = [144, 169, 200]

FORWARD_WINDOW = 63


def _compute_emas(close_s: pd.Series) -> dict[int, pd.Series]:
    """Compute the mid- and long-Vegas EMA series used by the outcome analysis."""
    return {s: close_s.ewm(span=s, adjust=False).mean()
            for s in MID_EMAS + LONG_EMAS}


def classify_outcome(
    close: np.ndarray,
    low: np.ndarray,
    emas: dict[int, np.ndarray],
    entry_bar: int,
    window: int = FORWARD_WINDOW,
) -> dict | None:
    """
    Classify the post-entry trajectory into one of three outcomes.

    Uses the next `window` trading days after entry to determine:
    - CONTINUE: price stays above mid Vegas, uptrend continues
    - BREAK_DIRECT: price quickly falls below mid and reaches long Vegas
    - BREAK_MULTI_TEST: price tests mid multiple times before breaking down
    """
    n = len(close)
    end_bar = min(entry_bar + window, n - 1)
    actual_window = end_bar - entry_bar
    if actual_window < 10:
        return None

    entry_price = close[entry_bar]

    # daily mid lower = min(ema34, ema55, ema60)
    # daily long upper = max(ema144, ema169, ema200)
    mid_lower = np.minimum(np.minimum(emas[34], emas[55]), emas[60])
    long_upper = np.maximum(np.maximum(emas[144], emas[169]), emas[200])

    days_below_mid = 0
    days_below_long = 0
    consec_below_mid = 0
    first_break_day = None
    mid_touch_count = 0
    in_touch_episode = False
    max_price = entry_price
    entered_long_zone = False

    for offset in range(1, actual_window + 1):
        bar = entry_bar + offset
        c = close[bar]

        mid_val = mid_lower[bar]
        long_val = long_upper[bar]

        max_price = max(max_price, c)

        if c < mid_val:
            days_below_mid += 1
            consec_below_mid += 1
            if first_break_day is None and consec_below_mid >= 3:
                first_break_day = offset - 2
            if not in_touch_episode:
                in_touch_episode = True
                mid_touch_count += 1
        else:
            consec_below_mid = 0
            in_touch_episode = False

        if c < long_val:
            days_below_long += 1
            entered_long_zone = True

    final_ret = (close[end_bar] / entry_price - 1) * 100
    max_rally = (max_price / entry_price - 1) * 100
    below_mid_ratio = days_below_mid / actual_window

    # Outcome classification
    broke_down = entered_long_zone or below_mid_ratio > 0.40

    if not broke_down:
        outcome = "CONTINUE"
    elif first_break_day is not None and first_break_day <= 10:
        outcome = "BREAK_DIRECT"
    elif mid_touch_count >= 2:
        outcome = "BREAK_MULTI_TEST"
    elif first_break_day is not None:
        outcome = "BREAK_MULTI_TEST"
    else:
        outcome = "BREAK_MULTI_TEST"

    return {
        "outcome": outcome,
        "days_below_mid": days_below_mid,
        "days_below_long": days_below_long,
        "below_mid_ratio": round(below_mid_ratio, 3),
        "mid_touch_count": mid_touch_count,
        "first_break_day": first_break_day,
        "max_rally_pct": round(max_rally, 2),
        "final_ret_pct": round(final_ret, 2),
        "entered_long_zone": entered_long_zone,
    }


def load_price_data(sym: str) -> pd.DataFrame | None:
    """Load one configured sample symbol from parquet into a normalized DataFrame."""
    if sym not in SAMPLES:
        return None
    _, _, path = SAMPLES[sym]
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def analyze_all():
    """Classify all recorded signals into post-entry outcome buckets with summary features."""
    signals = pd.read_csv(SIGNAL_CSV)
    logger.info(f"Loaded {len(signals)} signals")

    results = []

    for sym, grp in signals.groupby("symbol"):
        price_df = load_price_data(sym)
        if price_df is None:
            logger.warning(f"{sym}: no price data")
            continue

        close = price_df["close"].astype(float)
        low_arr = price_df["low"].astype(float).values
        close_arr = close.values

        emas_series = _compute_emas(close)
        emas_arr = {s: v.values for s, v in emas_series.items()}

        for _, row in grp.iterrows():
            entry_date = pd.Timestamp(row["entry_date"])
            idx_pos = price_df.index.searchsorted(entry_date, side="left")
            if idx_pos >= len(price_df) or idx_pos < 200:
                continue

            outcome_info = classify_outcome(close_arr, low_arr, emas_arr, idx_pos)
            if outcome_info is None:
                continue

            result = {
                "symbol": row["symbol"],
                "market": row["market"],
                "name": row["name"],
                "entry_date": row["entry_date"],
                "entry_price": row["entry_price"],
                "support_band": row["support_band"],
                "wave_number": row["wave_number"],
                "sub_number": row["sub_number"],
                "consec_waves": row["consec_waves"],
                "structure_passed": row["structure_passed"],
                "mid_above_long": row["mid_above_long"],
                "price_above_long": row["price_above_long"],
                "long_rising": row["long_rising"],
                "long_slope_pct": row["long_slope_pct"],
                "mid_long_gap_pct": row["mid_long_gap_pct"],
                "wave_rise_so_far_pct": row["wave_rise_so_far_pct"],
                "score": row["score"],
                "signal": row["signal"],
                "ret_5d": row["ret_5d"],
                "ret_21d": row["ret_21d"],
                "ret_63d": row["ret_63d"],
                "max_dd_21d": row["max_dd_21d"],
                **outcome_info,
            }
            results.append(result)

    df = pd.DataFrame(results)
    logger.info(f"Classified {len(df)} signals")
    return df


def _ret_stats(series: pd.Series) -> dict:
    """Return compact return statistics for one numeric series after dropping missing data."""
    vals = series.dropna()
    if len(vals) == 0:
        return {"n": 0}
    return {
        "n": len(vals),
        "win_rate": (vals > 0).mean() * 100,
        "avg": vals.mean(),
        "median": vals.median(),
    }


def print_report(df: pd.DataFrame):
    """Print the full textual outcome report for the classified mid-Vegas samples."""
    print("=" * 100)
    print("Mid Vegas 回踩三种结局统计分析（后验）")
    print("=" * 100)

    outcome_order = ["CONTINUE", "BREAK_DIRECT", "BREAK_MULTI_TEST"]
    outcome_labels = {
        "CONTINUE": "A. 继续上涨（主升浪延续）",
        "BREAK_DIRECT": "B. 直接跌穿（快速破位进入回调）",
        "BREAK_MULTI_TEST": "C. 反复试探后跌穿（多次测试后破位）",
    }

    # --- Section 1: Overall Distribution ---
    print(f"\n{'=' * 100}")
    print("1. 三种结局的总体分布")
    print(f"{'=' * 100}")

    for oc in outcome_order:
        sub = df[df["outcome"] == oc]
        pct = len(sub) / len(df) * 100
        r21 = _ret_stats(pd.to_numeric(sub["ret_21d"], errors="coerce"))
        r63 = _ret_stats(pd.to_numeric(sub["ret_63d"], errors="coerce"))
        dd = pd.to_numeric(sub["max_dd_21d"], errors="coerce").dropna()
        print(f"\n  {outcome_labels[oc]}")
        print(f"    数量: {len(sub):>3d} ({pct:.1f}%)")
        if r21["n"]:
            print(f"    21d: WR={r21['win_rate']:.0f}%  avg={r21['avg']:+.1f}%  med={r21['median']:+.1f}%")
        if r63["n"]:
            print(f"    63d: WR={r63['win_rate']:.0f}%  avg={r63['avg']:+.1f}%  med={r63['median']:+.1f}%")
        if len(dd):
            print(f"    DD : avg={dd.mean():.1f}%  worst={dd.min():.1f}%")

    # --- Section 2: Post-entry behavior ---
    print(f"\n{'=' * 100}")
    print("2. 后验行为特征")
    print(f"{'=' * 100}")
    for oc in outcome_order:
        sub = df[df["outcome"] == oc]
        if sub.empty:
            continue
        print(f"\n  {outcome_labels[oc]} (n={len(sub)})")
        print(f"    Mid下方天数占比: avg={sub['below_mid_ratio'].mean():.2f}  med={sub['below_mid_ratio'].median():.2f}")
        print(f"    Mid再触次数:     avg={sub['mid_touch_count'].mean():.1f}  med={sub['mid_touch_count'].median():.0f}")
        fb = sub["first_break_day"].dropna()
        if len(fb):
            print(f"    首次破位天数:    avg={fb.mean():.0f}  med={fb.median():.0f}")
        print(f"    最大涨幅:        avg={sub['max_rally_pct'].mean():.1f}%  med={sub['max_rally_pct'].median():.1f}%")
        print(f"    进入Long区间:    {sub['entered_long_zone'].sum()} ({sub['entered_long_zone'].mean()*100:.0f}%)")

    # --- Section 3: Structure condition x outcome ---
    print(f"\n{'=' * 100}")
    print("3. 结构条件 x 结局分布（入场时结构是否通过 -> 最终结局）")
    print(f"{'=' * 100}")
    for passed in [True, False]:
        label = "结构通过" if passed else "结构未通过"
        sub = df[df["structure_passed"] == passed]
        total = len(sub)
        if total == 0:
            continue
        print(f"\n  【{label}】 (n={total})")
        for oc in outcome_order:
            oc_sub = sub[sub["outcome"] == oc]
            if oc_sub.empty:
                continue
            pct = len(oc_sub) / total * 100
            r21 = _ret_stats(pd.to_numeric(oc_sub["ret_21d"], errors="coerce"))
            r63 = _ret_stats(pd.to_numeric(oc_sub["ret_63d"], errors="coerce"))
            line = f"    {outcome_labels[oc]}: n={len(oc_sub):>3d} ({pct:.0f}%)"
            if r21["n"]:
                line += f"  21d wr={r21['win_rate']:.0f}% avg={r21['avg']:+.1f}%"
            if r63["n"]:
                line += f"  63d wr={r63['win_rate']:.0f}% avg={r63['avg']:+.1f}%"
            print(line)

    # --- Section 4: Factor x outcome ---
    print(f"\n{'=' * 100}")
    print("4. 入场因子 x 结局 -- 找出 CONTINUE 占比最高的结构")
    print(f"{'=' * 100}")

    factors = [
        ("wave_number", "大浪序号"),
        ("sub_number", "子浪编号"),
        ("consec_waves", "连续浪数"),
        ("support_band", "支撑EMA"),
        ("wave_rise_bins", "浪涨幅段"),
        ("mid_long_gap_bins", "Mid/Long间距段"),
        ("long_slope_bins", "Long斜率段"),
        ("signal", "信号等级"),
        ("score_bins", "总分段"),
        ("market", "市场"),
    ]

    df_work = df.copy()
    df_work["wave_rise_bins"] = pd.cut(
        df_work["wave_rise_so_far_pct"],
        bins=[-999, 30, 60, 100, 200, 9999],
        labels=["<30%", "30-60%", "60-100%", "100-200%", ">200%"])

    df_work["mid_long_gap_bins"] = pd.cut(
        df_work["mid_long_gap_pct"],
        bins=[-999, 5, 15, 25, 40, 999],
        labels=["<5%", "5-15%", "15-25%", "25-40%", ">40%"])

    df_work["long_slope_bins"] = pd.cut(
        df_work["long_slope_pct"],
        bins=[-999, 0, 2, 5, 10, 999],
        labels=["<0%", "0-2%", "2-5%", "5-10%", ">10%"])

    df_work["score_bins"] = pd.cut(
        df_work["score"],
        bins=[-999, 0, 2, 4, 999],
        labels=["<0", "0-1", "2-3", ">=4"])

    for col, col_name in factors:
        if col not in df_work.columns:
            continue
        print(f"\n  【{col_name}】")
        print(f"    {'val':>12s}  {'n':>4s}  {'CONTINUE':>10s}  {'BRK_DIRECT':>10s}  {'BRK_MULTI':>10s}  "
              f"{'CONT%':>6s}  {'CONT 21d avg':>12s}")

        for val, g in df_work.groupby(col, observed=True):
            n_grp = len(g)
            if n_grp < 5:
                continue
            cont = len(g[g["outcome"] == "CONTINUE"])
            brk_d = len(g[g["outcome"] == "BREAK_DIRECT"])
            brk_m = len(g[g["outcome"] == "BREAK_MULTI_TEST"])
            cont_rate = cont / n_grp * 100

            cont_r21 = pd.to_numeric(g[g["outcome"] == "CONTINUE"]["ret_21d"], errors="coerce").dropna()
            cont_avg = f"{cont_r21.mean():+.1f}%" if len(cont_r21) else "N/A"

            print(f"    {str(val):>12s}  {n_grp:>4d}  "
                  f"{cont:>4d} ({cont/n_grp*100:>4.0f}%)  "
                  f"{brk_d:>4d} ({brk_d/n_grp*100:>4.0f}%)  "
                  f"{brk_m:>4d} ({brk_m/n_grp*100:>4.0f}%)  "
                  f"{cont_rate:>5.0f}%  "
                  f"{cont_avg:>12s}")

    # --- Section 5: Cross-factor combinations ---
    print(f"\n{'=' * 100}")
    print("5. 最佳 CONTINUE 组合 -- 结构通过 + 各因子交叉")
    print(f"{'=' * 100}")

    passed_df = df_work[df_work["structure_passed"] == True].copy()
    if len(passed_df) == 0:
        print("  No data with structure_passed=True")
        return

    cross_combos = [
        (["wave_rise_bins", "score_bins"], "浪涨幅 x 总分"),
        (["mid_long_gap_bins", "score_bins"], "Mid/Long间距 x 总分"),
        (["wave_rise_bins", "mid_long_gap_bins"], "浪涨幅 x Mid/Long间距"),
        (["long_slope_bins", "wave_rise_bins"], "Long斜率 x 浪涨幅"),
    ]

    for cols, label in cross_combos:
        print(f"\n  【{label}】")
        print(f"    {'combo':>25s}  {'n':>4s}  {'CONT%':>6s}  {'BRK%':>6s}  {'CONT 21d avg':>12s}  {'CONT 63d avg':>12s}")

        for keys, g in passed_df.groupby(cols, observed=True):
            n_grp = len(g)
            if n_grp < 5:
                continue
            cont_g = g[g["outcome"] == "CONTINUE"]
            cont_pct = len(cont_g) / n_grp * 100
            brk_pct = 100 - cont_pct

            r21 = pd.to_numeric(cont_g["ret_21d"], errors="coerce").dropna()
            r63 = pd.to_numeric(cont_g["ret_63d"], errors="coerce").dropna()
            r21_s = f"{r21.mean():+.1f}%" if len(r21) else "N/A"
            r63_s = f"{r63.mean():+.1f}%" if len(r63) else "N/A"

            key_str = " x ".join(str(k) for k in keys) if isinstance(keys, tuple) else str(keys)
            print(f"    {key_str:>25s}  {n_grp:>4d}  {cont_pct:>5.0f}%  {brk_pct:>5.0f}%  {r21_s:>12s}  {r63_s:>12s}")

    # --- Section 6: Feature comparison CONTINUE vs BREAK ---
    print(f"\n{'=' * 100}")
    print("6. CONTINUE vs BREAK 的入场特征差异")
    print(f"{'=' * 100}")

    cont = df_work[df_work["outcome"] == "CONTINUE"]
    brk = df_work[df_work["outcome"].isin(["BREAK_DIRECT", "BREAK_MULTI_TEST"])]

    features = [
        ("wave_rise_so_far_pct", "浪涨幅%"),
        ("mid_long_gap_pct", "Mid/Long间距%"),
        ("long_slope_pct", "Long斜率%"),
        ("score", "总分"),
        ("sub_number", "子浪编号"),
        ("consec_waves", "连续浪数"),
    ]

    print(f"\n  {'feature':>16s}  {'CONT avg':>10s}  {'CONT med':>10s}  "
          f"{'BRK avg':>10s}  {'BRK med':>10s}  {'diff':>8s}")

    for col, col_name in features:
        c_vals = pd.to_numeric(cont[col], errors="coerce").dropna()
        b_vals = pd.to_numeric(brk[col], errors="coerce").dropna()
        if len(c_vals) == 0 or len(b_vals) == 0:
            continue
        diff = c_vals.mean() - b_vals.mean()
        print(f"  {col_name:>16s}  "
              f"{c_vals.mean():>+10.1f}  {c_vals.median():>+10.1f}  "
              f"{b_vals.mean():>+10.1f}  {b_vals.median():>+10.1f}  "
              f"{diff:>+8.1f}")

    # --- Section 7: Touch sequence within a wave ---
    print(f"\n{'=' * 100}")
    print("7. 浪内回踩位置分析 -- 同一大浪的第几次回踩 mid Vegas 更安全")
    print(f"{'=' * 100}")

    df_work = df_work.sort_values(["symbol", "entry_date"])
    df_work["touch_seq"] = df_work.groupby(["symbol", "wave_number"]).cumcount() + 1

    print(f"\n  {'seq':>10s}  {'n':>4s}  {'CONTINUE':>10s}  {'BRK%':>6s}  {'21d avg':>10s}  {'63d avg':>10s}")
    for seq_val, g in df_work.groupby("touch_seq"):
        if len(g) < 5:
            continue
        cont_g = g[g["outcome"] == "CONTINUE"]
        cont_pct = len(cont_g) / len(g) * 100
        brk_pct = 100 - cont_pct
        r21 = pd.to_numeric(g["ret_21d"], errors="coerce").dropna()
        r63 = pd.to_numeric(g["ret_63d"], errors="coerce").dropna()
        r21_avg = f"{r21.mean():+.1f}%" if len(r21) else "N/A"
        r63_avg = f"{r63.mean():+.1f}%" if len(r63) else "N/A"
        print(f"  touch#{int(seq_val):>2d}     {len(g):>4d}  "
              f"{len(cont_g):>4d} ({cont_pct:>4.0f}%)  {brk_pct:>5.0f}%  "
              f"{r21_avg:>10s}  {r63_avg:>10s}")


def save_results(df: pd.DataFrame):
    """Persist CSV summaries and generate pie and heatmap charts for the outcome study."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_DIR / "three_outcome_analysis.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Results saved: {csv_path}")

    # --- Pie charts ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    color_map = {"CONTINUE": "#228B22", "BREAK_DIRECT": "#CC0000", "BREAK_MULTI_TEST": "#FF8C00"}
    label_map = {"CONTINUE": "继续上涨", "BREAK_DIRECT": "直接跌穿", "BREAK_MULTI_TEST": "反复试探后跌穿"}

    counts = df["outcome"].value_counts()
    ax = axes[0]
    ax.pie([counts.get(k, 0) for k in color_map],
           labels=[f"{label_map[k]}\n({counts.get(k, 0)})" for k in color_map],
           colors=[color_map[k] for k in color_map],
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
    ax.set_title("全部信号 - 三种结局分布", fontsize=13)

    passed = df[df["structure_passed"] == True]
    counts_p = passed["outcome"].value_counts()
    ax = axes[1]
    ax.pie([counts_p.get(k, 0) for k in color_map],
           labels=[f"{label_map[k]}\n({counts_p.get(k, 0)})" for k in color_map],
           colors=[color_map[k] for k in color_map],
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
    ax.set_title("结构通过 - 三种结局分布", fontsize=13)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "outcome_pie.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- Heatmaps ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    _plot_heatmap(axes[0, 0], df[df["structure_passed"] == True],
                  "wave_rise_so_far_pct", "score", "浪涨幅 x 总分 - CONTINUE率",
                  bins_x=[-999, 30, 60, 100, 200, 9999],
                  labels_x=["<30%", "30-60%", "60-100%", "100-200%", ">200%"],
                  bins_y=[-999, 0, 2, 4, 999],
                  labels_y=["<0", "0-1", "2-3", ">=4"])

    _plot_heatmap(axes[0, 1], df[df["structure_passed"] == True],
                  "mid_long_gap_pct", "score", "Mid/Long间距 x 总分 - CONTINUE率",
                  bins_x=[-999, 5, 15, 25, 40, 999],
                  labels_x=["<5%", "5-15%", "15-25%", "25-40%", ">40%"],
                  bins_y=[-999, 0, 2, 4, 999],
                  labels_y=["<0", "0-1", "2-3", ">=4"])

    _plot_heatmap(axes[1, 0], df[df["structure_passed"] == True],
                  "long_slope_pct", "wave_rise_so_far_pct", "Long斜率 x 浪涨幅 - CONTINUE率",
                  bins_x=[-999, 0, 2, 5, 10, 999],
                  labels_x=["<0%", "0-2%", "2-5%", "5-10%", ">10%"],
                  bins_y=[-999, 30, 60, 100, 200, 9999],
                  labels_y=["<30%", "30-60%", "60-100%", "100-200%", ">200%"])

    _plot_heatmap(axes[1, 1], df[df["structure_passed"] == True],
                  "sub_number", "wave_rise_so_far_pct", "子浪编号 x 浪涨幅 - CONTINUE率",
                  bins_x=[0, 2, 4, 6, 99],
                  labels_x=["1-2", "3-4", "5-6", "7+"],
                  bins_y=[-999, 30, 60, 100, 200, 9999],
                  labels_y=["<30%", "30-60%", "60-100%", "100-200%", ">200%"])

    plt.tight_layout()
    fig.savefig(OUT_DIR / "outcome_heatmaps.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Charts saved to: {OUT_DIR}")


def _plot_heatmap(ax, df, col_x, col_y, title,
                  bins_x, labels_x, bins_y, labels_y):
    """Draw one continue-rate heatmap for a pair of binned feature dimensions."""
    df = df.copy()
    df["_x"] = pd.cut(df[col_x], bins=bins_x, labels=labels_x)
    df["_y"] = pd.cut(df[col_y], bins=bins_y, labels=labels_y)

    pivot_n = df.groupby(["_y", "_x"], observed=True).size().unstack(fill_value=0)
    pivot_cont = df[df["outcome"] == "CONTINUE"].groupby(["_y", "_x"], observed=True).size().unstack(fill_value=0)

    rate = (pivot_cont / pivot_n.replace(0, np.nan) * 100).fillna(0)

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=100)

    n_rows, n_cols = rate.shape
    im = ax.imshow(rate.values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(rate.columns, fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(rate.index, fontsize=8)

    for i in range(n_rows):
        for j in range(n_cols):
            n_val = pivot_n.values[i, j] if i < pivot_n.shape[0] and j < pivot_n.shape[1] else 0
            r_val = rate.values[i, j]
            if n_val >= 3:
                text_color = "white" if r_val < 30 or r_val > 70 else "black"
                ax.text(j, i, f"{r_val:.0f}%\n(n={int(n_val)})",
                        ha="center", va="center", fontsize=7, color=text_color)

    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8, label="CONTINUE%")


def main():
    """Run the three-outcome precision study end to end and export its artifacts."""
    df = analyze_all()
    if df.empty:
        logger.error("No valid data")
        return

    print_report(df)
    save_results(df)


if __name__ == "__main__":
    main()
