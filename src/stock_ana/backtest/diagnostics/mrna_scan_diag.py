#!/usr/bin/env python3
"""直接调用 scan_one 诊断 MRNA"""

import pandas as pd
from stock_ana.scan.vegas_mid_scan import scan_one

df = pd.read_parquet("data/cache/us/MRNA.parquet")
signals = scan_one("MRNA", "US", "Moderna", df, lookback=5)

if signals:
    for s in signals:
        print(f"Signal: {s['signal']}  score={s['score']}  entry={s['entry_date']}  band={s['support_band']}")
        print(f"  structure_passed={s['structure_passed']}  gap={s['gap_enough']}  slp={s['long_slope_strong']}  seq={s['touch_seq_ok']}")
else:
    print("No signals from scan_one")
    # Debug: check raw touch signals
    from stock_ana.strategies.impl.vegas_mid import (
        compute_vegas_emas as _compute_all_emas,
        detect_mid_touch_and_hold,
    )
    x = df.copy()
    x.columns = [c.lower() for c in x.columns]
    x.index = pd.to_datetime(x.index); x = x.sort_index()
    close = x["close"].astype(float).values
    low_arr = x["low"].astype(float).values
    emas = _compute_all_emas(x["close"].astype(float))
    n = len(x)
    touch_sigs = detect_mid_touch_and_hold(close, low_arr, emas)
    cutoff = n - 5
    recent = [s for s in touch_sigs if s["entry_bar"] >= cutoff]
    print(f"\nRaw touch signals: {len(touch_sigs)}, recent (>=bar {cutoff}): {len(recent)}")
    for s in recent:
        print(f"  entry_bar={s['entry_bar']}  n={n}  band={s['support_band']}  confirm={s['confirm_bar']}")
