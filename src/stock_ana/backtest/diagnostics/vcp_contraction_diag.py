"""
MAG7 每个峰值的 VCP 收缩统计诊断工具。

检查 MAG7 等察察大市値股在每个寜观峰值附近的波幅收缩情况，
辅助验证 VCP 拉需收缩幅度、峰数阈值参数设定是否合理。
"""
import pandas as pd

from stock_ana.data.peak_store import get_or_compute_peaks
from stock_ana.strategies.primitives.vcp import detect_vcp_micro_structure

MAG7 = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]

for ticker in MAG7:
    df = pd.read_parquet(f"data/cache/us/{ticker}.parquet")
    peaks_df = get_or_compute_peaks(ticker, df)
    print(f"\n{'='*60}")
    print(f"{ticker}: {len(peaks_df)} peaks")

    for row_idx, (peak_dt, row) in enumerate(peaks_df.iterrows()):
        # peak_dt is the DatetimeIndex value
        peak_iloc = df.index.get_loc(peak_dt)
        peak_price = row["high"]
        segment_df = df.iloc[peak_iloc:]
        _, stats = detect_vcp_micro_structure(segment_df, min_contractions=1)
        cc = stats.get("consecutive_contractions", 0)
        n_seg = stats.get("n_segments", 0)
        n_sh = stats.get("n_swing_highs", 0)
        ranges = stats.get("swing_ranges", [])
        ratios = stats.get("contraction_ratios", [])
        print(f"  Peak {peak_dt.date()} ({peak_price:.1f}): "
              f"{n_sh} SH, {n_seg} seg, {cc} consec contractions")
        if ranges:
            r_str = " → ".join(f"{r:.1f}" for r in ranges)
            print(f"    ranges: {r_str}")
        if ratios:
            rat_str = " → ".join(f"{r:.0%}" for r in ratios)
            print(f"    ratios: {rat_str}")
