#!/usr/bin/env python3
"""
Vegas 升浪回踩 Mid Vegas 策略回测（v3 — 零前瞻偏差）

核心思路：
  1. 识别当前处于上涨浪中 —— 通过结构条件确认：
     • Mid Vegas (EMA34/55/60) 在 Long Vegas (EMA144/169/200) 之上
     • 股价在 Long Vegas 之上
     • Long Vegas 上升（斜率 > 0）
  2. 在升浪中，关注价格回调到 Mid Vegas 时的买入机会
  3. 过去的波段低点 + Long Vegas → 用于构建浪结构上下文（浪序号、连续浪等）

入场信号（实时，零前瞻）：
  价格触碰 Mid Vegas → 站稳（连续2天收在线上）→ 次日入场

打分因子（仅 mid vegas 相关）：
  • 市场: HK +1
  • 子浪位置: early/late +1, mid -1
  • 浪涨幅: <30% +2, 30-100% +1, >200% -2
  • 连续浪 ≥3: +1
  • 浪序号=3: +1
  • Mid/Long 间距合理（5-25%）: +1

信号分类:
  score >= 4: STRONG_BUY
  score >= 2: BUY
  score >= 0: HOLD
  score <  0: AVOID

用法:
    python -m stock_ana.backtest.backtest_vegas_mid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from stock_ana.utils.plot_renderers import plot_vegas_wave_summary

from stock_ana.config import OUTPUT_DIR
from stock_ana.scan.vegas_mid_scan import scan_one
from stock_ana.data.market_data import load_shawn_data

OUT_DIR = OUTPUT_DIR / "vegas_wave_strategy"
FORWARD_DAYS = [5, 10, 21, 63]

# ═══════════════════════════════════════════════════════
#  回测主体
# ═══════════════════════════════════════════════════════

def _compute_forward_returns(signals: list[dict], df: "pd.DataFrame") -> list[dict]:
    """Append ret_5d/10d/21d/63d and max_dd_21d to each signal using raw price data.

    Skips T+1 edge-of-data signals that have no tradeable future.
    """
    x = df.copy()
    x.columns = [c.lower() for c in x.columns]
    x.index = pd.to_datetime(x.index).normalize()
    x = x.sort_index()
    close = x["close"].astype(float).values

    rows: list[dict] = []
    for sig in signals:
        if "(T+1)" in sig["entry_date"]:
            continue  # live edge signal — no future data to measure
        entry_dt = pd.Timestamp(sig["entry_date"])
        try:
            entry_bar = x.index.get_loc(entry_dt)
        except KeyError:
            continue
        entry_price = close[entry_bar]
        fwd: dict = {}
        for d in FORWARD_DAYS:
            target = entry_bar + d
            fwd[f"ret_{d}d"] = (
                round((close[target] / entry_price - 1) * 100, 2) if target < len(close) else np.nan
            )
        if entry_bar + 21 < len(close):
            fwd["max_dd_21d"] = round(
                (min(close[entry_bar : entry_bar + 22]) / entry_price - 1) * 100, 2
            )
        else:
            fwd["max_dd_21d"] = np.nan
        rows.append({**sig, **fwd})
    return rows


def run_backtest() -> pd.DataFrame:
    """对 Shawn 自选列表（data/lists/shawn_list.md）所有股票执行回测。"""
    all_rows: list[dict] = []

    market_data = load_shawn_data()
    if not market_data:
        logger.error("Shawn 列表无可用数据，请先同步本地缓存")
        return pd.DataFrame()

    for sym, info in market_data.items():
        df = info["df"]
        signals = scan_one(sym, info["market"], info["name"], df, lookback=len(df))
        rows = _compute_forward_returns(signals, df)
        if rows:
            logger.success(f"{info['market']}:{sym} {info['name']}: {len(rows)} 个回踩事件")
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)


# ═══════════════════════════════════════════════════════
#  分析与报告
# ═══════════════════════════════════════════════════════

def _ret_stats(vals: pd.Series) -> dict:
    """Summarize win rate and return distribution statistics for one return series."""
    v = vals.dropna()
    if len(v) == 0:
        return {"n": 0}
    return {
        "n": len(v),
        "win_rate": round((v > 0).mean() * 100, 1),
        "avg": round(v.mean(), 2),
        "median": round(v.median(), 2),
        "max": round(v.max(), 2),
        "min": round(v.min(), 2),
    }


def print_report(df: pd.DataFrame):
    """打印完整的回测分析报告。"""
    if df.empty:
        print("No data.")
        return

    print("=" * 90)
    print("Vegas 升浪 Mid Vegas 回踩策略回测报告 (v4 — CONTINUE辅助指标)")
    print("=" * 90)
    print(f"回踩事件总数: {len(df)}")
    print(f"涉及标的: {df['symbol'].nunique()} 只")
    print(f"信号分布: {df['signal'].value_counts().to_dict()}")
    print(f"支撑线: {df['support_band'].value_counts().to_dict()}")
    print(f"市场: {df['market'].value_counts().to_dict()}")

    # ─── 结构条件统计 ───
    if "structure_passed" in df.columns:
        n_pass = df["structure_passed"].sum()
        n_fail = len(df) - n_pass
        print(f"\n结构条件过滤:")
        print(f"   通过（升浪确认）: {n_pass} ({n_pass/len(df)*100:.1f}%)")
        print(f"   未通过 → AVOID:  {n_fail} ({n_fail/len(df)*100:.1f}%)")
        # 各子条件
        for col, label in [("mid_above_long", "Mid>Long"),
                           ("price_above_long", "Price>Long"),
                           ("long_rising", "Long上升"),
                           ("gap_enough", "Mid/Long间距>=5%"),
                           ("long_slope_strong", "Long斜率>=2%"),
                           ("touch_seq_ok", "浪内前3次回踩")]:
            if col in df.columns:
                n = df[col].sum()
                print(f"     {label}: {n} ({n/len(df)*100:.1f}%)")

    # ─────────── Section 1: 按信号等级 ───────────
    print(f"\n{'─' * 90}")
    print("1. 按信号等级 — 各持有期收益统计")
    print(f"{'─' * 90}")
    for sig in ["STRONG_BUY", "BUY", "HOLD", "AVOID"]:
        sub = df[df["signal"] == sig]
        if sub.empty:
            continue
        print(f"\n  【{sig}】 ({len(sub)} 个)")
        for d in FORWARD_DAYS:
            s = _ret_stats(pd.to_numeric(sub[f"ret_{d}d"], errors="coerce"))
            if s["n"] == 0:
                continue
            print(f"    {d:>2d}d: n={s['n']:>3d}  wr={s['win_rate']:>5.1f}%  "
                  f"avg={s['avg']:>+7.2f}%  med={s['median']:>+7.2f}%  "
                  f"max={s['max']:>+7.2f}%  min={s['min']:>+7.2f}%")
        dd = pd.to_numeric(sub["max_dd_21d"], errors="coerce").dropna()
        if len(dd):
            print(f"    DD: avg={dd.mean():+.2f}%  med={dd.median():+.2f}%  worst={dd.min():+.2f}%")

    # ─────────── Section 2: BUY/STRONG_BUY vs HOLD/AVOID ───────────
    print(f"\n{'─' * 90}")
    print("2. 买入 vs 不买入 — 对比")
    print(f"{'─' * 90}")
    buy_mask = df["signal"].isin(["STRONG_BUY", "BUY"])
    for label, mask in [("BUY+STRONG_BUY", buy_mask), ("HOLD+AVOID", ~buy_mask)]:
        sub = df[mask]
        if sub.empty:
            continue
        print(f"\n  【{label}】 ({len(sub)} 个)")
        for d in FORWARD_DAYS:
            s = _ret_stats(pd.to_numeric(sub[f"ret_{d}d"], errors="coerce"))
            if s["n"] == 0:
                continue
            print(f"    {d:>2d}d: n={s['n']:>3d}  wr={s['win_rate']:>5.1f}%  "
                  f"avg={s['avg']:>+7.2f}%  med={s['median']:>+7.2f}%")

    # ─────────── Section 3: 按市场×信号 ───────────
    print(f"\n{'─' * 90}")
    print("3. 按市场 × 信号等级 — 21d/63d")
    print(f"{'─' * 90}")
    for mkt in ["US", "HK"]:
        for sig in ["STRONG_BUY", "BUY", "HOLD", "AVOID"]:
            sub = df[(df["market"] == mkt) & (df["signal"] == sig)]
            if sub.empty:
                continue
            parts = [f"  {mkt:>2s} {sig:>12s}: n={len(sub):>3d}"]
            for d in [21, 63]:
                s = _ret_stats(pd.to_numeric(sub[f"ret_{d}d"], errors="coerce"))
                if s["n"]:
                    parts.append(f"{d}d: wr={s['win_rate']:.0f}% avg={s['avg']:+.1f}% med={s['median']:+.1f}%")
            print(" | ".join(parts))

    # ─────────── Section 4: 结构通过 vs 未通过 ───────────
    print(f"\n{'─' * 90}")
    print("4. 结构通过 vs 未通过 — 21d/63d")
    print(f"{'─' * 90}")
    if "structure_passed" in df.columns:
        for passed, label in [(True, "结构通过（升浪中）"), (False, "结构未通过")]:
            sub = df[df["structure_passed"] == passed]
            if sub.empty:
                continue
            parts = [f"  {label}: n={len(sub):>3d}"]
            for d in [21, 63]:
                s = _ret_stats(pd.to_numeric(sub[f"ret_{d}d"], errors="coerce"))
                if s["n"]:
                    parts.append(f"{d}d: wr={s['win_rate']:.0f}% avg={s['avg']:+.1f}% med={s['median']:+.1f}%")
            print(" | ".join(parts))

    # ─────────── Section 4b: CONTINUE 辅助条件（仅供参考）───────────
    print(f"\n{'─' * 90}")
    print("4b. CONTINUE 辅助条件 × BUY+SB 收益（条件不参与过滤，供人工判断）")
    print(f"{'─' * 90}")
    buy_df_s4 = df[df["signal"].isin(["STRONG_BUY", "BUY"])]
    helper_conds = [
        ("gap_enough", "Mid/Long间距>=5%"),
        ("long_slope_strong", "Long斜率>=2%"),
        ("touch_seq_ok", "浪内前3次回踩"),
    ]
    for col, label in helper_conds:
        if col not in buy_df_s4.columns:
            continue
        for val, vlabel in [(True, "Y"), (False, "N")]:
            sub = buy_df_s4[buy_df_s4[col] == val]
            if sub.empty:
                continue
            r21 = _ret_stats(pd.to_numeric(sub["ret_21d"], errors="coerce"))
            r63 = _ret_stats(pd.to_numeric(sub["ret_63d"], errors="coerce"))
            dd = pd.to_numeric(sub["max_dd_21d"], errors="coerce").dropna()
            line = f"  {label} = {vlabel}: n={len(sub):>3d}"
            if r21["n"]:
                line += f"  21d wr={r21['win_rate']:.0f}% avg={r21['avg']:+.1f}%"
            if r63["n"]:
                line += f"  63d wr={r63['win_rate']:.0f}% avg={r63['avg']:+.1f}%"
            if len(dd):
                line += f"  DD={dd.mean():.1f}%"
            print(line)

    # 全部3条都通过 vs 不全通过
    if all(c in buy_df_s4.columns for c, _ in helper_conds):
        all_pass = buy_df_s4[
            buy_df_s4["gap_enough"] & buy_df_s4["long_slope_strong"] & buy_df_s4["touch_seq_ok"]
        ]
        not_all = buy_df_s4[
            ~(buy_df_s4["gap_enough"] & buy_df_s4["long_slope_strong"] & buy_df_s4["touch_seq_ok"])
        ]
        for sub, slabel in [(all_pass, "3条全通过"), (not_all, "未全通过")]:
            if sub.empty:
                continue
            r21 = _ret_stats(pd.to_numeric(sub["ret_21d"], errors="coerce"))
            r63 = _ret_stats(pd.to_numeric(sub["ret_63d"], errors="coerce"))
            dd = pd.to_numeric(sub["max_dd_21d"], errors="coerce").dropna()
            line = f"  ** {slabel}: n={len(sub):>3d}"
            if r21["n"]:
                line += f"  21d wr={r21['win_rate']:.0f}% avg={r21['avg']:+.1f}%"
            if r63["n"]:
                line += f"  63d wr={r63['win_rate']:.0f}% avg={r63['avg']:+.1f}%"
            if len(dd):
                line += f"  DD={dd.mean():.1f}%"
            print(line)

    # ─────────── Section 5: 各因子得分分布 ───────────
    print(f"\n{'─' * 90}")
    print("5. 各因子得分与 21d 收益的关系")
    print(f"{'─' * 90}")
    factor_cols = ["factor_mkt", "factor_sub_pos", "factor_wave_rise", "factor_three_wave", "factor_wave_num", "factor_ml_gap", "factor_orderly"]
    factor_names = ["市场", "子浪位置", "浪涨幅", "连续浪", "浪序号", "Mid/Long间距", "有序回踩"]
    for col, fname in zip(factor_cols, factor_names):
        if col not in df.columns:
            continue
        print(f"\n  {fname} ({col}):")
        for val in sorted(df[col].unique()):
            sub = df[df[col] == val]
            s = _ret_stats(pd.to_numeric(sub["ret_21d"], errors="coerce"))
            if s["n"] == 0:
                continue
            s63 = _ret_stats(pd.to_numeric(sub["ret_63d"], errors="coerce"))
            print(f"    score={val:>+2d}: n={s['n']:>3d}  "
                  f"21d wr={s['win_rate']:>5.1f}% avg={s['avg']:>+7.2f}%  |  "
                  f"63d wr={s63.get('win_rate',0):>5.1f}% avg={s63.get('avg',0):>+7.2f}%")

    # ─────────── Section 6: 按 score 分段 ───────────
    print(f"\n{'─' * 90}")
    print("6. 按总分段 — 收益分布")
    print(f"{'─' * 90}")
    for lo, hi, label in [(-99, -1, "score<0 (AVOID)"),
                           (0, 1, "score 0~1 (HOLD)"),
                           (2, 3, "score 2~3 (BUY)"),
                           (4, 99, "score≥4 (STRONG_BUY)")]:
        sub = df[(df["score"] >= lo) & (df["score"] <= hi)]
        if sub.empty:
            continue
        print(f"\n  {label} ({len(sub)} 个)")
        for d in FORWARD_DAYS:
            s = _ret_stats(pd.to_numeric(sub[f"ret_{d}d"], errors="coerce"))
            if s["n"] == 0:
                continue
            print(f"    {d:>2d}d: n={s['n']:>3d}  wr={s['win_rate']:>5.1f}%  "
                  f"avg={s['avg']:>+7.2f}%  med={s['median']:>+7.2f}%")

    # ─────────── Section 7: Score threshold sweep ───────────
    print(f"\n{'─' * 90}")
    print("7. Score 阈值扫描 — 找最佳买入门槛")
    print(f"{'─' * 90}")
    print(f"  {'threshold':>9s}  {'n':>4s}  {'wr_21d':>7s} {'avg_21d':>8s} {'med_21d':>8s}  "
          f"{'wr_63d':>7s} {'avg_63d':>8s} {'med_63d':>8s}  {'avg_dd':>7s}")
    for thr in range(-3, 10):
        sub = df[df["score"] >= thr]
        if sub.empty:
            continue
        r21 = pd.to_numeric(sub["ret_21d"], errors="coerce").dropna()
        r63 = pd.to_numeric(sub["ret_63d"], errors="coerce").dropna()
        dd = pd.to_numeric(sub["max_dd_21d"], errors="coerce").dropna()
        print(f"  score>={thr:>+3d}:  n={len(sub):>4d}  "
              f"wr={100*(r21>0).mean():>5.1f}% avg={r21.mean():>+7.2f}% med={r21.median():>+7.2f}%  "
              f"wr={100*(r63>0).mean():>5.1f}% avg={r63.mean():>+7.2f}% med={r63.median():>+7.2f}%  "
              f"dd={dd.mean():>+6.2f}%")

    # ─────────── Section 8: 个股表现 ───────────
    print(f"\n{'─' * 90}")
    print("8. 个股 BUY+STRONG_BUY 信号表现 (21d/63d)")
    print(f"{'─' * 90}")
    buy_df_s8 = df[buy_mask]
    stock_rows = []
    for (mkt, sym, name_), g in buy_df_s8.groupby(["market", "symbol", "name"]):
        r21 = pd.to_numeric(g["ret_21d"], errors="coerce").dropna()
        r63 = pd.to_numeric(g["ret_63d"], errors="coerce").dropna()
        if len(r21) == 0:
            continue
        stock_rows.append({
            "stock": f"{mkt}:{sym}",
            "name": name_,
            "n": len(g),
            "wr_21d": f"{(r21 > 0).mean()*100:.0f}%",
            "avg_21d": f"{r21.mean():+.1f}%",
            "wr_63d": f"{(r63 > 0).mean()*100:.0f}%" if len(r63) else "N/A",
            "avg_63d": f"{r63.mean():+.1f}%" if len(r63) else "N/A",
        })
    if stock_rows:
        sdf = pd.DataFrame(stock_rows)
        print(sdf.to_string(index=False))

    # ─────────── Section 9: 最高分信号列表 ───────────
    print(f"\n{'─' * 90}")
    print("9. 最高分信号 TOP 20")
    print(f"{'─' * 90}")
    top = df.nlargest(20, "score")[
        ["market", "symbol", "entry_date", "support_band",
         "score", "signal", "wave_number", "sub_number",
         "structure_passed", "ret_5d", "ret_21d", "ret_63d"]
    ]
    print(top.to_string(index=False))

    # ─────────── Section 10: 最低分信号列表 ───────────
    print(f"\n{'─' * 90}")
    print("10. 最低分信号 BOTTOM 20")
    print(f"{'─' * 90}")
    bot = df.nsmallest(20, "score")[
        ["market", "symbol", "entry_date", "support_band",
         "score", "signal", "wave_number", "sub_number",
         "structure_passed", "ret_5d", "ret_21d", "ret_63d"]
    ]
    print(bot.to_string(index=False))

    # ─────────── Section 11: BUY/STRONG_BUY 信号明细（含因子权重）───────────
    print(f"\n{'─' * 90}")
    print("11. 所有 BUY / STRONG_BUY 信号明细（按时间排序，含各因子得分）")
    print(f"{'─' * 90}")
    buy_signals = df[df["signal"].isin(["STRONG_BUY", "BUY"])].sort_values("entry_date")
    factor_cols_display = ["factor_mkt", "factor_sub_pos", "factor_wave_rise", "factor_three_wave", "factor_wave_num", "factor_ml_gap", "factor_orderly"]
    factor_labels = ["市场", "子浪位", "浪涨幅", "连续浪", "浪序号", "ML距", "有序"]
    header = (f"{'日期':>12s} {'市场':>3s} {'股票':>6s} {'名称':>8s} "
              f"{'支撑线':>7s} {'结构':>4s} {'Gap':>3s} {'Slp':>3s} {'Seq':>3s} W# Sub# "
              f"{'涨幅%':>6s} " +
              " ".join(f"{l:>4s}" for l in factor_labels) +
              f" {'总分':>4s} {'信号':>12s}" +
              f" {'5d%':>7s} {'21d%':>7s} {'63d%':>7s}")
    print(header)
    print("─" * len(header))
    for _, row in buy_signals.iterrows():
        factor_scores = " ".join(f"{int(row.get(c, 0)):>+4d}" for c in factor_cols_display)
        r5 = row.get("ret_5d", np.nan); r21 = row.get("ret_21d", np.nan); r63 = row.get("ret_63d", np.nan)
        r5s = f"{r5:>+7.1f}" if pd.notna(r5) else "    N/A"
        r21s = f"{r21:>+7.1f}" if pd.notna(r21) else "    N/A"
        r63s = f"{r63:>+7.1f}" if pd.notna(r63) else "    N/A"
        struct_flag = "Y" if row.get("structure_passed", False) else "N"
        gap_flag = "Y" if row.get("gap_enough", False) else "N"
        slp_flag = "Y" if row.get("long_slope_strong", False) else "N"
        seq_flag = "Y" if row.get("touch_seq_ok", False) else "N"
        print(f"{row['entry_date']:>12s} {row['market']:>3s} {row['symbol']:>6s} {str(row['name'])[:8]:>8s} "
              f"{row['support_band']:>7s} {struct_flag:>4s} {gap_flag:>3s} {slp_flag:>3s} {seq_flag:>3s} "
              f"{int(row['wave_number']):>2d} {int(row['sub_number']):>4d} "
              f"{row.get('wave_rise_pct', 0):>+6.0f} "
              f"{factor_scores} "
              f"{int(row['score']):>+4d} {row['signal']:>12s}"
              f" {r5s} {r21s} {r63s}")

    # ─────────── Section 12: touch_strategy 对比 ───────────
    _sep = "─" * 90
    print(f"\n{_sep}")
    print("12. touch策略 vs hold策略 — '触及即出'与'确认入场'的收益对比")
    print(_sep)
    if "touch_strategy" in df.columns:
        for strat in ["touch", "hold"]:
            sub = df[df["touch_strategy"] == strat]
            if sub.empty:
                continue
            buy_sub = sub[sub["signal"].isin(["STRONG_BUY", "BUY"])]
            print(f"\n  [策略: {strat.upper()}] 全部信号: {len(sub)} 个  其中 BUY+SB: {len(buy_sub)} 个")
            print(f"  信号分布: {sub['signal'].value_counts().to_dict()}")
            print(f"  结构通过: {sub['structure_passed'].sum()} / {len(sub)}")
            for d in FORWARD_DAYS:
                col = f"ret_{d}d"
                s_all = _ret_stats(pd.to_numeric(sub[col], errors="coerce"))
                s_buy = _ret_stats(pd.to_numeric(buy_sub[col], errors="coerce"))
                if s_all["n"] == 0:
                    continue
                print(f"    {d:>2d}d [全]: n={s_all['n']:>3d} wr={s_all['win_rate']:>5.1f}% avg={s_all['avg']:>+7.2f}%"
                      f"  | BUY+SB: n={s_buy.get('n',0):>3d} "
                      + (f"wr={s_buy['win_rate']:>5.1f}% avg={s_buy['avg']:>+7.2f}%" if s_buy.get('n',0) else "(none)"))
    else:
        print("  （没有 touch_strategy 字段，请先重跑 run_backtest）")


def print_per_stock(df: pd.DataFrame):
    """按标的分组，打印每只股票的全部信号明细表（含回测收益）。"""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    COLS = [
        "entry_date", "touch_strategy", "support_band",
        "signal", "score",
        "structure_passed", "mid_above_long", "price_above_long", "long_rising",
        "wave_number", "sub_number", "wave_rise_pct",
        "orderly_pullback", "touch_seq",
        "ret_5d", "ret_10d", "ret_21d", "ret_63d", "max_dd_21d",
    ]
    present = [c for c in COLS if c in df.columns]

    for (mkt, sym), g in df.groupby(["market", "symbol"]):
        name = g["name"].iloc[0]
        g_sorted = g.sort_values("entry_date")
        _eq = "=" * 100
        print(f"\n{_eq}")
        print(f"  {mkt}:{sym}  {name}  ({len(g_sorted)} 个信号)")
        print(_eq)
        sub = g_sorted[present].copy()
        # 简化布尔列
        for col in ["structure_passed", "mid_above_long", "price_above_long",
                    "long_rising", "orderly_pullback"]:
            if col in sub.columns:
                sub[col] = sub[col].map({True: "Y", False: "N", 1: "Y", 0: "N"})
        # 小数格式
        for col in ["wave_rise_pct", "ret_5d", "ret_10d", "ret_21d", "ret_63d", "max_dd_21d"]:
            if col in sub.columns:
                sub[col] = pd.to_numeric(sub[col], errors="coerce").map(
                    lambda x: f"{x:+.1f}" if pd.notna(x) else "N/A"
                )
        print(sub.to_string(index=False))
        # 小结
        buy_g = g_sorted[g_sorted["signal"].isin(["STRONG_BUY", "BUY"])]
        if len(buy_g):
            touch_g = buy_g[buy_g.get("touch_strategy", pd.Series(dtype=str)) == "touch"] if "touch_strategy" in buy_g.columns else pd.DataFrame()
            hold_g  = buy_g[buy_g.get("touch_strategy", pd.Series(dtype=str)) == "hold"]  if "touch_strategy" in buy_g.columns else buy_g
            for strat_label, sg in [("hold", hold_g), ("touch", touch_g)]:
                if sg.empty:
                    continue
                r21 = pd.to_numeric(sg["ret_21d"], errors="coerce").dropna() if "ret_21d" in sg.columns else pd.Series()
                r63 = pd.to_numeric(sg["ret_63d"], errors="coerce").dropna() if "ret_63d" in sg.columns else pd.Series()
                summary = f"  小结 [{strat_label.upper()}] BUY+SB x{len(sg)}"
                if len(r21):
                    summary += f"  21d wr={(r21>0).mean()*100:.0f}% avg={r21.mean():+.1f}%"
                if len(r63):
                    summary += f"  63d wr={(r63>0).mean()*100:.0f}% avg={r63.mean():+.1f}%"
                print(summary)


def save_results(df: pd.DataFrame):
    """保存结果到 CSV 和生成简要可视化。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = OUT_DIR / "vegas_wave_strategy_signals.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"信号明细已保存: {csv_path}")

    # 可视化：委托 chart.py 系一次性渲染
    plot_vegas_wave_summary(df, OUT_DIR)


def save_backtest_charts(df: pd.DataFrame, buy_only: bool = False):
    """为回测 DataFrame 中的每只标的生成一张汇总K线图，所有信号画在同一张图上。

    Args:
        df: run_backtest() 返回的 DataFrame。
        buy_only: 若为 True，只保留 signal 为 BUY / STRONG_BUY 的信号行。
    """
    from stock_ana.utils.plot_renderers import plot_stock_all_signals_chart

    chart_dir = OUT_DIR / "backtest_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    market_data = load_shawn_data()
    if not market_data:
        logger.error("无法加载市场数据，图表生成跳过")
        return

    sub = df[df["signal"].isin(["STRONG_BUY", "BUY"])] if buy_only else df
    stocks = sub.groupby("symbol")
    logger.info(f"开始生成图表：{len(stocks)} 只标的，共 {len(sub)} 个信号 → {chart_dir}")

    count = 0
    skipped = 0
    for sym, group in stocks:
        info = market_data.get(sym)
        if info is None:
            skipped += 1
            continue

        signals = group.to_dict(orient="records")
        mkt = info["market"]
        path = plot_stock_all_signals_chart(
            sym=sym,
            market=mkt,
            name=info["name"],
            df_price=info["df"],
            signals=signals,
            out_dir=chart_dir,
        )
        if path:
            count += 1
        else:
            skipped += 1

    logger.success(f"图表生成完毕: {count} 张成功，{skipped} 张跳过 → {chart_dir}")


def main():
    """Run the Vegas wave backtest CLI, print the report, and save CSV and charts."""
    parser = argparse.ArgumentParser(description="Vegas 浪结构买卖策略回测")
    parser.add_argument("--min-score", type=int, default=None,
                        help="仅打印 score>=N 的信号统计")
    parser.add_argument("--per-stock", action="store_true",
                        help="按标的分组打印每只股票的全部信号明细（含回测收益）")
    parser.add_argument("--charts", action="store_true",
                        help="为所有回测信号生成个股K线图表")
    parser.add_argument("--charts-buy-only", action="store_true",
                        help="仅为 BUY / STRONG_BUY 信号生成图表")
    args = parser.parse_args()

    df = run_backtest()
    if df.empty:
        print("No events found.")
        return

    print_report(df)
    save_results(df)

    if args.per_stock:
        print_per_stock(df)

    if args.charts or args.charts_buy_only:
        save_backtest_charts(df, buy_only=args.charts_buy_only)

    if args.min_score is not None:
        thr = args.min_score
        sub = df[df["score"] >= thr]
        print(f"\n{'=' * 90}")
        print(f"仅 score >= {thr} 的信号 ({len(sub)} 个):")
        for d in FORWARD_DAYS:
            s = _ret_stats(pd.to_numeric(sub[f"ret_{d}d"], errors="coerce"))
            if s["n"]:
                print(f"  {d}d: n={s['n']} wr={s['win_rate']:.1f}% "
                      f"avg={s['avg']:+.2f}% med={s['median']:+.2f}%")


if __name__ == "__main__":
    main()
