"""诊断 BWXT 的 OB 算法差异：因果版 vs 上游库 vs 近期 swing 点"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartmoneyconcepts import smc as _smc
from stock_ana.strategies.impl.smc import _ob_causal

df = pd.read_parquet(ROOT / "data/cache/us/BWXT.parquet")
df.columns = [c.lower() for c in df.columns]
df = df.sort_index()

# 只看 2026-01-01 以后（方便定位）
df2 = df[df.index >= "2026-01-01"].copy()
dates = df2.index.astype(str).str[:10].tolist()

swing_hl      = _smc.swing_highs_lows(df2, swing_length=5)
ob_causal     = _ob_causal(df2, swing_hl, swing_length=5)
ob_upstream   = _smc.ob(df2, swing_hl)

# ─── 打印近期 swing 高低点 ───────────────────────────────────────────────────
print("=== 近期 swing 高低点 (2026 全年) ===")
for i, row in swing_hl.iterrows():
    if pd.notna(row["HighLow"]) and row["HighLow"] != 0:
        tag = "HIGH" if row["HighLow"] == 1 else "LOW "
        print(f"  {tag}  bar={i:3d}  {dates[i]}  level={row['Level']:.2f}")

print()

# ─── 打印两版本 OB ──────────────────────────────────────────────────────────
def print_obs(ob_df, label):
    print(f"=== {label} ===")
    for i in range(len(ob_df)):
        v = ob_df["OB"].iloc[i]
        if pd.notna(v):
            bot  = ob_df["Bottom"].iloc[i]
            top  = ob_df["Top"].iloc[i]
            mit  = ob_df["MitigatedIndex"].iloc[i]
            tag  = "BULL" if v == 1 else "BEAR"
            mit_date = dates[int(mit)] if pd.notna(mit) and mit != 0 else "-"
            print(f"  {tag}  bar={i:3d}  {dates[i]}  [{bot:.2f}~{top:.2f}]  mit={mit_date}")
    print()

print_obs(ob_causal,   "因果版 OB（我们的 _ob_causal）")
print_obs(ob_upstream, "上游库 OB（smc.ob 含前瞻）")

# ─── 追踪 2026-05-14 bear OB 触发逻辑 ────────────────────────────────────────
print("=== 追踪 2026-05-14 bear OB 触发过程 ===")
# 找 2026-05-11 附近的 swing LOW
swing_low_indices = np.flatnonzero(swing_hl["HighLow"].values == -1)
print(f"所有 swing LOW bars: {[(b, dates[b], df2['low'].iloc[b]) for b in swing_low_indices]}")

# 找 2026-05-14 的 bar 序号
idx_514 = dates.index("2026-05-14")
print(f"\n2026-05-14 = bar {idx_514}  O={df2['open'].iloc[idx_514]:.2f}  H={df2['high'].iloc[idx_514]:.2f}  L={df2['low'].iloc[idx_514]:.2f}  C={df2['close'].iloc[idx_514]:.2f}")

# 看看 2026-05-14 附近的 OB 触发：scan bar 2026-05-18 onwards
print("\n逐 bar 检查看跌 OB 触发 (swing_length=5):")
for i, d in enumerate(dates):
    if d < "2026-05-15":
        continue
    # 可见的 swing LOW
    visible_mask = swing_low_indices + 5 <= i
    visible_lows = swing_low_indices[visible_mask]
    if len(visible_lows) == 0:
        continue
    last_btm = visible_lows[-1]
    last_btm_date = dates[last_btm]
    last_btm_low  = df2["low"].iloc[last_btm]
    close_i = df2["close"].iloc[i]
    triggered = close_i < last_btm_low
    if triggered or d <= "2026-05-22":
        print(f"  bar={i} {d}  close={close_i:.2f}  last_vis_swing_low={last_btm_date}(low={last_btm_low:.2f})  triggered={triggered}")
        if triggered:
            # find OB bar
            start = last_btm + 1
            end   = i
            seg   = df2["high"].iloc[start:end].values
            max_h = seg.max()
            ci    = start + int(np.nonzero(seg == max_h)[0][-1])
            print(f"    → OB bar={ci} {dates[ci]}  H={df2['high'].iloc[ci]:.2f}  L={df2['low'].iloc[ci]:.2f}  O={df2['open'].iloc[ci]:.2f}  C={df2['close'].iloc[ci]:.2f}")
            break

# ─── 查 2026-03-30 bull OB 的消除判断 ────────────────────────────────────────
print()
print("=== 检查 2026-03-30 bull OB [189.00~204.37] 消除情况 ===")
ob_top = 204.37
ob_bot = 189.00
for i, d in enumerate(dates):
    if d < "2026-05-10":
        continue
    lo = df2["low"].iloc[i]
    cl = df2["close"].iloc[i]
    entered   = lo < ob_top   # 刺入 OB 顶部
    cl_below  = cl < ob_top   # 收盘在 OB 内
    mitigated = lo < ob_bot   # 跌破底部
    print(f"  {d}  low={lo:.2f}  close={cl:.2f}  entered_zone={entered}  close_in_zone={cl_below}  full_mitigated={mitigated}")
