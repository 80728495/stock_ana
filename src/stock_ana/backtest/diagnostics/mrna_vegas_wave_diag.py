#!/usr/bin/env python3
"""
MRNA Vegas 波段结构诊断脚本。

单股 MRNA 中期 Vegas 通道附近回调条件的一次性诊断。
Ad-hoc diagnostic script for inspecting recent MRNA Vegas-wave structure conditions.
"""

import numpy as np
import pandas as pd
from stock_ana.backtest.backtest_vegas_mid import (
    _compute_all_emas, _check_structure, detect_mid_touch_and_hold,
    MID_EMAS, LONG_EMAS,
)

df = pd.read_parquet("data/cache/us/MRNA.parquet")
df.columns = [c.lower() for c in df.columns]
df.index = pd.to_datetime(df.index)
df = df.sort_index()

close_s = df["close"].astype(float)
low_arr = df["low"].astype(float).values
close   = close_s.values
n = len(df)

emas = _compute_all_emas(close_s)

# ── 最近 10 根 K 线 + EMA ──
print("=== 最近 10 根 K 线 + EMA ===")
tail_df = df[["close"]].tail(10).copy()
for s in [34, 55, 60, 144, 169, 200]:
    tail_df[f"ema{s}"] = emas[s][-10:]
print(tail_df.round(2).to_string())

# ── 最新一根结构检查 ──
entry_bar = n - 1
struct = _check_structure(entry_bar, close, emas)
mid_upper = max(emas[s][entry_bar] for s in MID_EMAS)
long_upper = max(emas[s][entry_bar] for s in LONG_EMAS)
price = close[entry_bar]

print("\n=== 最新 bar 结构检查 ===")
print(f"  price:            {price:.2f}")
print(f"  mid_upper (EMA60):{mid_upper:.2f}")
print(f"  long_upper(EMA200):{long_upper:.2f}")
print(f"  mid_above_long:   {struct['mid_above_long']}  (需 True)")
print(f"  price_above_long: {struct['price_above_long']}  (需 True)")
print(f"  long_rising:      {struct['long_rising']}  (需 True)")
print(f"  long_slope_pct:   {struct['long_slope_pct']:.2f}%")
print(f"  mid_long_gap%:    {struct['mid_long_gap_pct']:.2f}%")
print(f"  structure_passed: {struct['passed']}")

# ── Touch 信号检测 ──
print("\n=== Touch 信号检测 ===")
touch_sigs = detect_mid_touch_and_hold(close, low_arr, emas)
print(f"  全量历史 touch 信号: {len(touch_sigs)} 个")

cutoff_bar = n - 5
recent = [s for s in touch_sigs if s["entry_bar"] >= cutoff_bar]
print(f"  最近 5 根内信号:     {len(recent)} 个  (cutoff bar={cutoff_bar}, 日期={df.index[cutoff_bar].date()})")

if touch_sigs and not recent:
    print(f"  最近的 5 个历史信号:")
    for s in touch_sigs[-5:]:
        eb = s["entry_bar"]
        print(f"    bar={eb} ({df.index[eb].date()})  band={s['support_band']}  "
              f"entry_price={close[eb]:.2f}")
    print(f"\n  补充: touch_bar 在最近 5 根的情况:")
    recent_touch = [s for s in touch_sigs if s.get("touch_bar", s["entry_bar"]) >= cutoff_bar]
    print(f"    touch_bar 在最近 5 根: {len(recent_touch)} 个")
