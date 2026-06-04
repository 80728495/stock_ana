"""诊断 DT 的看涨OB 消除情况"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartmoneyconcepts import smc as _smc
from stock_ana.strategies.impl.smc import _ob_causal

df = pd.read_parquet(ROOT / "data/cache/us/DT.parquet")
df.columns = [c.lower() for c in df.columns]
df = df.sort_index()
dates = df.index.astype(str).str[:10].tolist()

swing_hl  = _smc.swing_highs_lows(df, swing_length=5)
ob_causal = _ob_causal(df, swing_hl, swing_length=5)

print("=== DT 全量因果OB（所有看涨）===")
for i in range(len(ob_causal)):
    v = ob_causal["OB"].iloc[i]
    if pd.notna(v) and v == 1:
        bot = ob_causal["Bottom"].iloc[i]
        top = ob_causal["Top"].iloc[i]
        mit = ob_causal["MitigatedIndex"].iloc[i]
        mit_date = dates[int(mit)] if (pd.notna(mit) and mit != 0) else "-"
        lo_at_mit = df["low"].iloc[int(mit)] if mit_date != "-" else float("nan")
        cl_at_mit = df["close"].iloc[int(mit)] if mit_date != "-" else float("nan")
        print(f"  BULL  bar={i:3d}  {dates[i]}  [{bot:.2f}~{top:.2f}]  mitigated_at={mit_date}  (L={lo_at_mit:.2f}  C={cl_at_mit:.2f})")

print()
# 找最低价，看是否有OB底部被穿透
print("=== 近期 min LOW（2025-11起）===")
recent = df.loc["2025-11-01":]
for dt, row in recent.iterrows():
    print(f"  {str(dt)[:10]}  L={row['low']:.2f}  C={row['close']:.2f}")
