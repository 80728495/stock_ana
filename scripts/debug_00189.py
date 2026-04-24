"""Diagnose a specific signal date for any stock."""
import pandas as pd
import numpy as np
from stock_ana.data.market_data import load_watchlist_data
from stock_ana.scan.vegas_mid_scan import scan_one, _came_from_above, _compute_all_emas
from stock_ana.strategies.impl.vegas_mid import (
    detect_mid_touch_immediate, detect_mid_touch_and_hold,
    score_pullback, classify_signal, find_wave_context as _find_wave_context,
    check_mid_vegas_structure as _check_structure,
    backward_consecutive_count,
)
from stock_ana.strategies.primitives.wave import analyze_wave_structure

SYM    = "00981"
TARGET = "2025-03-14"

data = load_watchlist_data()
info = data.get(SYM)
df   = info["df"]

close_s  = df["close"].astype(float)
close    = close_s.values
low_arr  = df["low"].astype(float).values
emas     = _compute_all_emas(close_s)
result   = analyze_wave_structure(df)
waves    = result["major_waves"]
all_pivots = result.get("all_pivots", [])

# ── 1. EMA / structure at target date ────────────────────────────────────────
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
    print(f"  EMA{span:3d} = {emas[span][i]:.2f}")

# Structure check at that bar
struct = _check_structure(i, close, emas)
print(f"\n  structure_passed    = {struct['passed']}")
print(f"  mid_above_long      = {struct['mid_above_long']}")
print(f"  price_above_long    = {struct['price_above_long']}")
print(f"  price_above_long_3m = {struct['price_above_long_3m']}")
print(f"  long_rising         = {struct['long_rising']}")
print(f"  gap_enough (>=5%)   = {struct['gap_enough']}  ({struct['mid_long_gap_pct']:.1f}%)")
print(f"  long_slope_strong   = {struct['long_slope_strong']}  ({struct['long_slope_pct']:.2f}%)")

# ── 2. Wave context ───────────────────────────────────────────────────────────
from stock_ana.strategies.impl.vegas_mid import backward_consecutive_count

wave_ctx = _find_wave_context(waves, i)
print(f"\n  wave_ctx            = {wave_ctx}")
    if wave_ctx:
        wn = wave_ctx["wave_number"]
    sp_val = wave_ctx["start_pivot"]["value"]
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
    print(f"  score details       = {details}")
    sig = classify_signal(score) if struct["passed"] else "AVOID"
    print(f"  signal              = {sig}")

# ── 3. Nearby ZigZag pivots ───────────────────────────────────────────────────
print(f"\n  ZigZag pivots within 60 bars of {TARGET}:")
for p in all_pivots:
    if abs(p["iloc"] - i) <= 60:
        d = str(df.index[p["iloc"]].date())
        print(f"    iloc={p['iloc']}  date={d}  value={p['value']:.2f}  type={p.get('type','?')}")

# ── 4. came_from_above ────────────────────────────────────────────────────────
cfa = _came_from_above(i, close, all_pivots)
print(f"\n  came_from_above     = {cfa}")

# ── 5. Full scan_one output around that date ──────────────────────────────────
print()
print("=" * 90)
print(f"scan_one signals for HK:{SYM} (full history, 2025 only)")
print("=" * 90)
sigs = scan_one(SYM, "HK", info["name"], df, lookback=len(df))
for s in sigs:
    ed = s.get("entry_date", "?")
    if ed[:4] == "2025":
        print(f"  {ed:<14} signal={s.get('signal'):<12} score={s.get('score'):+d}  "
              f"band={s.get('support_band'):<8} strategy={s.get('touch_strategy'):<6} "
              f"structure={s.get('structure_passed')}  "
              f"mid>long={s.get('mid_above_long')}  px>long={s.get('price_above_long')}  "
              f"gap={s.get('mid_long_gap_pct'):.1f}%  slope={s.get('long_slope_pct'):.2f}%")


data = load_watchlist_data()
info = data.get("00189")
df = info["df"]

close = df["close"]
ema34  = close.ewm(span=34,  adjust=False).mean()
ema55  = close.ewm(span=55,  adjust=False).mean()
ema60  = close.ewm(span=60,  adjust=False).mean()
ema144 = close.ewm(span=144, adjust=False).mean()

# ── 1. Print EMA values + structure checks for every bar Nov 2024 – Feb 2025 ──
print("=" * 110)
print("HK:00189  EMA & structure check (every bar that touches EMA34/55/60)")
print("=" * 110)
print(f"{'date':<12} {'close':>6} {'ema34':>6} {'ema55':>6} {'ema60':>6} {'ema144':>7}  "
      f"{'mid>long':^9} {'px>long':^8} {'rising':^7} {'gap%':>6} {'slope%':>7}  touch")

mask = (df.index >= "2024-11-01") & (df.index <= "2025-03-01")
for date, row in df[mask].iterrows():
    i   = df.index.get_loc(date)
    c   = close.iloc[i]
    m34 = ema34.iloc[i]
    m55 = ema55.iloc[i]
    m60 = ema60.iloc[i]
    m144= ema144.iloc[i]

    mid_above_long   = m34 > m144
    price_above_long = c > m144
    long_rising      = bool(m144 > ema144.iloc[i-5]) if i >= 5 else False
    gap              = (m34 - m144) / m144 * 100
    slope            = (m144 - ema144.iloc[i-10]) / ema144.iloc[i-10] * 100 if i >= 10 else 0

    t34 = row["low"] <= m34 <= row["high"]
    t55 = row["low"] <= m55 <= row["high"]
    t60 = row["low"] <= m60 <= row["high"]
    touches = []
    if t34: touches.append("ema34")
    if t55: touches.append("ema55")
    if t60: touches.append("ema60")

    if touches:
        print(f"{str(date.date()):<12} {c:>6.2f} {m34:>6.2f} {m55:>6.2f} {m60:>6.2f} {m144:>7.2f}  "
              f"{'Y' if mid_above_long else 'N':^9} {'Y' if price_above_long else 'N':^8} "
              f"{'Y' if long_rising else 'N':^7} {gap:>6.1f} {slope:>7.2f}  {','.join(touches)}")

# ── 2. Raw signals from detect_mid_touch_immediate (before any filtering) ────
print()
print("=" * 80)
print("RAW signals from detect_mid_touch_immediate (no structure/direction filter)")
print("=" * 80)
close_arr = df["close"].astype(float).values
low_arr   = df["low"].astype(float).values
close_s   = df["close"].astype(float)
emas      = _compute_all_emas(close_s)

result     = analyze_wave_structure(df)
all_pivots = result.get("all_pivots", [])

raw_touch = detect_mid_touch_immediate(close_arr, low_arr, emas)
raw_hold  = detect_mid_touch_and_hold(close_arr, low_arr, emas)

for sig in raw_touch:
    entry_bar = sig["entry_bar"]
    date = str(df.index[entry_bar].date())
    if "2024-10" <= date[:7] <= "2025-03":
        cfa = _came_from_above(sig["touch_bar"], close_arr, all_pivots)
        # also print above_lookback detail
        i = sig["touch_bar"]
        span = sig["ema_span"]
        ema_arr = emas[span]
        lb_start = max(0, i - 10)
        above_cnt = int(np.sum(close_arr[lb_start:i] >= ema_arr[lb_start:i]))
        total = i - lb_start
        print(f"  TOUCH {date}  band={sig['support_band']}  "
              f"above={above_cnt}/{total} ({above_cnt/max(total,1)*100:.0f}%)  "
              f"came_from_above={cfa}")

print()
print("=" * 80)
print("RAW signals from detect_mid_touch_and_hold")
print("=" * 80)
for sig in raw_hold:
    entry_bar = sig["entry_bar"]
    date = str(df.index[min(entry_bar, len(df)-1)].date())
    if "2024-10" <= date[:7] <= "2025-03":
        cfa = _came_from_above(sig["touch_bar"], close_arr, all_pivots)
        print(f"  HOLD  {date}  band={sig['support_band']}  came_from_above={cfa}")

# ── 3. ZigZag pivots around Jan 2025 ─────────────────────────────────────────
print()
print("=" * 80)
print("ZigZag all_pivots around Jan 2025 (last pivot before Jan 23)")
print("=" * 80)
for p in all_pivots:
    d = str(df.index[p["iloc"]].date())
    if "2024-10" <= d[:7] <= "2025-03":
        print(f"  iloc={p['iloc']}  date={d}  value={p['value']:.2f}  type={p.get('type','?')}")

# ── 4. Final scan_one signals ─────────────────────────────────────────────────
print()
print("=" * 80)
print("scan_one final signals for 00189 (full history)")
print("=" * 80)
sigs = scan_one("00189", "HK", info["name"], df, lookback=len(df))
for s in sigs:
    ed = s.get("entry_date", "?")
    if "2024-10" <= ed[:7] <= "2025-04":
        print(f"  {ed}  signal={s.get('signal')}  score={s.get('score')}  "
              f"band={s.get('support_band')}  strategy={s.get('touch_strategy')}  "
              f"structure={s.get('structure_passed')}")
