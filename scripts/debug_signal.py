"""Diagnose a specific signal date for any stock."""
import pandas as pd
import numpy as np
from stock_ana.data.market_data import load_watchlist_data
from stock_ana.scan.vegas_mid_scan import scan_one, _came_from_above, _compute_all_emas
from stock_ana.strategies.impl.vegas_mid import (
    detect_mid_touch_immediate, detect_mid_touch_and_hold,
    score_pullback, classify_signal,
    find_wave_context as _find_wave_context,
    check_mid_vegas_structure as _check_structure,
    backward_consecutive_count,
)
from stock_ana.strategies.primitives.wave import analyze_wave_structure

SYM    = "000537"
TARGET = "2026-03-30"

data = load_watchlist_data()
info = data.get(SYM)
df   = info["df"]

close_s    = df["close"].astype(float)
close      = close_s.values
low_arr    = df["low"].astype(float).values
emas       = _compute_all_emas(close_s)
result     = analyze_wave_structure(df)
waves      = result["major_waves"]
all_pivots = result.get("all_pivots", [])

# ── 1. EMA values & structure at target date ──────────────────────────────────
print("=" * 90)
print(f"HK:{SYM}  —  {TARGET} 信号诊断")
print("=" * 90)

target_dt = pd.Timestamp(TARGET)
i = df.index.searchsorted(target_dt, side="left")
if i >= len(df):
    i = len(df) - 1
print(f"\n  bar index = {i},  date = {df.index[i].date()}")
print(f"  close={close[i]:.2f}  low={low_arr[i]:.2f}")
for span in [34, 55, 60, 144, 169, 200]:
    print(f"  EMA{span:3d} = {emas[span][i]:.3f}")

struct = _check_structure(i, close, emas)
print(f"\n  structure_passed    = {struct['passed']}")
print(f"  mid_above_long      = {struct['mid_above_long']}")
print(f"  price_above_long    = {struct['price_above_long']}")
print(f"  price_above_long_3m = {struct['price_above_long_3m']}")
print(f"  long_rising         = {struct['long_rising']}")
print(f"  gap_enough (>=5%)   = {struct['gap_enough']}  ({struct['mid_long_gap_pct']:.1f}%)")
print(f"  long_slope_strong   = {struct['long_slope_strong']}  ({struct['long_slope_pct']:.2f}%)")

# ── 2. Wave context & scoring ─────────────────────────────────────────────────
wave_ctx = _find_wave_context(waves, i)
print(f"\n  wave_ctx present?   = {wave_ctx is not None}")
if wave_ctx:
    wn         = wave_ctx["wave_number"]
    sp_val     = wave_ctx["start_pivot"]["value"]
    look_start = wave_ctx["start_pivot"]["iloc"]
    look_end   = min(i + 1, len(df))
    peak_val   = float(max(close[look_start:look_end]))
    wave_rise  = (peak_val / sp_val - 1) * 100 if sp_val > 0 else 0
    sub_count  = sum(
        1 for sw in wave_ctx.get("sub_waves", [])
        if sw.get("end_pivot") and sw["end_pivot"]["iloc"] <= i
    )
    sub_number = sub_count + 1
    consec     = backward_consecutive_count(waves, wn)

    print(f"  wave_number         = {wn}")
    print(f"  sub_number          = {sub_number}")
    print(f"  wave_rise_pct       = {wave_rise:.1f}%")
    print(f"  consec_waves        = {consec}")

    score, details = score_pullback(
        sub_number=sub_number,
        wave_rise_pct=wave_rise,
        wave_number=wn,
        market="HK",
        consecutive_wave_count=consec,
        mid_long_gap_pct=struct["mid_long_gap_pct"],
        orderly_pullback=False,
    )
    print(f"\n  score               = {score}")
    for k, v in details.items():
        print(f"    factor_{k} = {v:+d}")
    sig = classify_signal(score) if struct["passed"] else "AVOID"
    print(f"  => signal           = {sig}")

# ── 3. Nearby ZigZag pivots ───────────────────────────────────────────────────
print(f"\n  ZigZag pivots within 50 bars of {TARGET}:")
for p in all_pivots:
    if abs(p["iloc"] - i) <= 50:
        d = str(df.index[p["iloc"]].date())
        print(f"    iloc={p['iloc']:4d}  date={d}  value={p['value']:.2f}  type={p.get('type','?')}")

# ── 4. came_from_above ────────────────────────────────────────────────────────
cfa = _came_from_above(i, close, all_pivots)
print(f"\n  came_from_above     = {cfa}")

# ── 5. Raw detector output around target ─────────────────────────────────────
print()
print("=" * 90)
print("RAW touch/hold detector output (no post-filter)")
print("=" * 90)
for label, raw in [("TOUCH", detect_mid_touch_immediate(close, low_arr, emas)),
                   ("HOLD",  detect_mid_touch_and_hold(close, low_arr, emas))]:
    for sig in raw:
        eb = sig["entry_bar"]
        if eb >= len(df):
            eb = len(df) - 1
        d = str(df.index[eb].date())
        if "2025-02" <= d[:7] <= "2025-04":
            cfa2 = _came_from_above(sig["touch_bar"], close, all_pivots)
            print(f"  {label}  {d}  band={sig['support_band']:<8} came_from_above={cfa2}")

# ── 6. All scan_one signals for 2025 ─────────────────────────────────────────
print()
print("=" * 90)
print(f"scan_one final signals HK:{SYM}  (2025)")
print("=" * 90)
sigs = scan_one(SYM, "HK", info["name"], df, lookback=len(df))
for s in sigs:
    ed = s.get("entry_date", "?")
    if ed[:4] == "2025":
        print(
            f"  {ed:<14} signal={s.get('signal'):<12} score={s.get('score'):+d}  "
            f"band={s.get('support_band'):<8} strategy={s.get('touch_strategy'):<6} "
            f"structure={s.get('structure_passed')}  "
            f"mid>long={s.get('mid_above_long')}  px>long={s.get('price_above_long')}  "
            f"gap={s.get('mid_long_gap_pct'):.1f}%  slope={s.get('long_slope_pct'):.2f}%"
        )
