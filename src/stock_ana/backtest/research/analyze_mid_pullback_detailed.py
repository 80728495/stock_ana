#!/usr/bin/env python3
"""
Mid pullback 精细化统计：按股票、时间段、浪位置、涨幅强度等维度拆分，
看哪些条件下 EMA34/55 支撑效应更显著。
"""
import pandas as pd
import numpy as np
from stock_ana.strategies.primitives.wave import analyze_wave_structure
from stock_ana.config import CACHE_DIR

SAMPLES = {
    "APP": ("US", "APP", CACHE_DIR / "us" / "APP.parquet"),
    "NVDA": ("US", "NVDA", CACHE_DIR / "us" / "NVDA.parquet"),
    "META": ("US", "Meta", CACHE_DIR / "us" / "META.parquet"),
    "TSLA": ("US", "Tesla", CACHE_DIR / "us" / "TSLA.parquet"),
    "AMD": ("US", "AMD", CACHE_DIR / "us" / "AMD.parquet"),
    "RBLX": ("US", "ROBLOX", CACHE_DIR / "us" / "RBLX.parquet"),
    "MRNA": ("US", "MRNA", CACHE_DIR / "us" / "MRNA.parquet"),
    "MU": ("US", "MU", CACHE_DIR / "us" / "MU.parquet"),
    "TEM": ("US", "TempusAI", CACHE_DIR / "us" / "TEM.parquet"),
    "ALAB": ("US", "ALAB", CACHE_DIR / "us" / "ALAB.parquet"),
    "PDD": ("US", "PDD", CACHE_DIR / "us" / "PDD.parquet"),
    "MSFT": ("US", "MSFT", CACHE_DIR / "us" / "MSFT.parquet"),
    "GOOG": ("US", "GOOG", CACHE_DIR / "us" / "GOOG.parquet"),
    "09992": ("HK", "POP MART", CACHE_DIR / "hk" / "09992.parquet"),
    "00700": ("HK", "Tencent", CACHE_DIR / "hk" / "00700.parquet"),
    "01810": ("HK", "Xiaomi", CACHE_DIR / "hk" / "01810.parquet"),
    "09988": ("HK", "Alibaba", CACHE_DIR / "hk" / "09988.parquet"),
    "00981": ("HK", "SMIC", CACHE_DIR / "hk" / "00981.parquet"),
    "03690": ("HK", "Meituan", CACHE_DIR / "hk" / "03690.parquet"),
    "01347": ("HK", "Hua Hong", CACHE_DIR / "hk" / "01347.parquet"),
    "02400": ("HK", "XD Inc", CACHE_DIR / "hk" / "02400.parquet"),
    "01024": ("HK", "Kuaishou", CACHE_DIR / "hk" / "01024.parquet"),
    "09626": ("HK", "Bilibili", CACHE_DIR / "hk" / "09626.parquet"),
    "00189": ("HK", "Dongyue", CACHE_DIR / "hk" / "00189.parquet"),
    "06869": ("HK", "CF Fiber", CACHE_DIR / "hk" / "06869.parquet"),
    "02228": ("HK", "Jingta", CACHE_DIR / "hk" / "02228.parquet"),
    "02788": ("HK", "ChuangXin", CACHE_DIR / "hk" / "02788.parquet"),
}


def _precision_stats(sub: pd.DataFrame, label: str, total_n: int | None = None):
    """打印一组 pullback 的精度统计。"""
    n = len(sub)
    if n == 0:
        print(f"  {label}: (no data)")
        return
    if total_n is None:
        total_n = n

    med = sub["closer_dist_pct"].median()
    mean = sub["closer_dist_pct"].mean()
    mid_cols = ["dist_ema34_pct", "dist_ema55_pct", "dist_ema60_pct"]
    long_cols = ["dist_ema144_pct", "dist_ema169_pct", "dist_ema200_pct"]
    all_cols = mid_cols + long_cols
    # any mid EMA within tolerance
    def _any_within(df, cols, tol):
        """Count rows where any distance column stays within the given tolerance."""
        mask = pd.Series(False, index=df.index)
        for c in cols:
            if c in df.columns:
                mask = mask | (df[c].abs() <= tol)
        return mask.sum()
    within1 = _any_within(sub, mid_cols, 1)
    within2 = _any_within(sub, mid_cols, 2)
    within3 = _any_within(sub, mid_cols, 3)
    within5 = _any_within(sub, mid_cols, 5)

    print(f"  {label:40s}  n={n:3d}  median={med:+6.2f}%  mean={mean:+6.2f}%  "
          f"±1%={within1/n*100:5.1f}%  ±2%={within2/n*100:5.1f}%  "
          f"±3%={within3/n*100:5.1f}%  ±5%={within5/n*100:5.1f}%")


def main():
    """Analyze detailed mid-pullback precision splits across symbols, waves, and periods."""
    rows = []
    for sym, (mkt, name, path) in SAMPLES.items():
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        if len(df) < 100:
            continue

        close = df["close"].astype(float)
        ema34 = close.ewm(span=34, adjust=False).mean()
        ema55 = close.ewm(span=55, adjust=False).mean()
        ema60 = close.ewm(span=60, adjust=False).mean()
        ema144 = close.ewm(span=144, adjust=False).mean()
        ema169 = close.ewm(span=169, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()

        result = analyze_wave_structure(df)
        waves = result["major_waves"]

        for mw in waves:
            for sw in mw["sub_waves"]:
                if sw["pullback_type"] != "mid_vegas":
                    continue
                ep = sw["end_pivot"]
                v = ep["value"]
                i = ep["iloc"]
                e34 = float(ema34.iloc[i])
                e55 = float(ema55.iloc[i])
                e60 = float(ema60.iloc[i])
                e144 = float(ema144.iloc[i])
                e169 = float(ema169.iloc[i])
                e200 = float(ema200.iloc[i])
                dist34 = (v - e34) / e34 * 100 if e34 > 0 else np.nan
                dist55 = (v - e55) / e55 * 100 if e55 > 0 else np.nan
                dist60 = (v - e60) / e60 * 100 if e60 > 0 else np.nan
                dist144 = (v - e144) / e144 * 100 if e144 > 0 else np.nan
                dist169 = (v - e169) / e169 * 100 if e169 > 0 else np.nan
                dist200 = (v - e200) / e200 * 100 if e200 > 0 else np.nan

                # Find closest mid EMA
                mid_dists = {"ema34": dist34, "ema55": dist55, "ema60": dist60}
                closer = min(mid_dists, key=lambda k: abs(mid_dists[k]) if not np.isnan(mid_dists[k]) else 999)
                closer_dist = mid_dists[closer]

                # Distance to long vegas
                long_upper = max(e144, e169, e200)
                dist_long = (v - long_upper) / long_upper * 100 if long_upper > 0 else np.nan
                wave_rise = mw["rise_pct"]
                sub_pos_in_wave = sw["sub_number"]
                total_subs = mw["sub_wave_count"]
                sub_pos_ratio = sub_pos_in_wave / total_subs if total_subs > 0 else 0

                date_str = ep["date"]
                try:
                    dt = pd.Timestamp(date_str)
                    year = dt.year
                    quarter = f"{dt.year}Q{(dt.month-1)//3+1}"
                    half = f"{dt.year}H{'1' if dt.month <= 6 else '2'}"
                except Exception:
                    year, quarter, half = 0, "?", "?"

                rows.append({
                    "symbol": sym, "market": mkt, "name": name,
                    "wave_num": mw["wave_number"],
                    "sub_num": sw["sub_number"],
                    "total_subs": total_subs,
                    "sub_pos_ratio": round(sub_pos_ratio, 2),
                    "date": date_str, "year": year, "quarter": quarter, "half": half,
                    "low_val": round(v, 2),
                    "ema34": round(e34, 2), "ema55": round(e55, 2), "ema60": round(e60, 2),
                    "ema144": round(e144, 2), "ema169": round(e169, 2), "ema200": round(e200, 2),
                    "dist_ema34_pct": round(dist34, 2),
                    "dist_ema55_pct": round(dist55, 2),
                    "dist_ema60_pct": round(dist60, 2),
                    "dist_ema144_pct": round(dist144, 2),
                    "dist_ema169_pct": round(dist169, 2),
                    "dist_ema200_pct": round(dist200, 2),
                    "closer_band": closer,
                    "closer_dist_pct": round(closer_dist, 2),
                    "dist_long_pct": round(dist_long, 2),
                    "wave_rise_pct": wave_rise,
                    "wave_duration": mw["duration_days"],
                    "sub_rise_pct": sw["rise_pct"],
                })

    rdf = pd.DataFrame(rows)
    n = len(rdf)
    print(f"Total mid pullbacks: {n}\n")

    # ══════════════════════════════════════════════════════════
    # 1. 按股票拆分
    # ══════════════════════════════════════════════════════════
    print("=" * 120)
    print("1. 按股票 — 哪些股票的 mid Vegas (EMA34/55/60) 支撑精度更高？")
    print("=" * 120)
    for (mkt, sym, name), g in rdf.groupby(["market", "symbol", "name"]):
        _precision_stats(g, f"{mkt}:{sym:>5s} {name}")

    # ══════════════════════════════════════════════════════════
    # 2. 按市场 (US vs HK)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("2. 按市场 — US vs HK")
    print("=" * 120)
    for mkt, g in rdf.groupby("market"):
        _precision_stats(g, mkt)

    # ══════════════════════════════════════════════════════════
    # 3. 按时间段 (半年)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("3. 按时间段 (半年) — 哪个时间段支撑效应更强？")
    print("=" * 120)
    for half, g in rdf.groupby("half"):
        _precision_stats(g, half)

    # ══════════════════════════════════════════════════════════
    # 4. 按大浪涨幅强度分级
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("4. 按大浪涨幅强度 — 强势浪 vs 一般浪")
    print("=" * 120)
    rdf["wave_strength"] = pd.cut(
        rdf["wave_rise_pct"],
        bins=[0, 30, 60, 100, 200, 9999],
        labels=["weak(<30%)", "moderate(30-60%)", "strong(60-100%)", "very_strong(100-200%)", "extreme(>200%)"],
    )
    for strength, g in rdf.groupby("wave_strength", observed=True):
        _precision_stats(g, str(strength))

    # ══════════════════════════════════════════════════════════
    # 5. 按子浪在大浪中的位置 (前半段 vs 后半段)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("5. 按子浪位置 — 大浪初期 vs 中期 vs 末期")
    print("=" * 120)
    rdf["wave_phase"] = pd.cut(
        rdf["sub_pos_ratio"],
        bins=[-0.01, 0.33, 0.67, 1.01],
        labels=["early(0-33%)", "mid(33-67%)", "late(67-100%)"],
    )
    for phase, g in rdf.groupby("wave_phase", observed=True):
        _precision_stats(g, str(phase))

    # ══════════════════════════════════════════════════════════
    # 6. 按距 long vegas 远近 — pullback 时价格离 EMA169 多远
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("6. 按距 Long Vegas 远近 — pullback 时离 EMA169 多远")
    print("=" * 120)
    rdf["long_dist_band"] = pd.cut(
        rdf["dist_long_pct"],
        bins=[-999, 5, 15, 30, 50, 9999],
        labels=["near_long(<5%)", "moderate(5-15%)", "far(15-30%)", "very_far(30-50%)", "extreme(>50%)"],
    )
    for band, g in rdf.groupby("long_dist_band", observed=True):
        _precision_stats(g, str(band))

    # ══════════════════════════════════════════════════════════
    # 7. 按大浪持续时间 — 短浪 vs 长浪
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("7. 按大浪持续时间 — 短浪 vs 长浪")
    print("=" * 120)
    rdf["wave_dur_band"] = pd.cut(
        rdf["wave_duration"],
        bins=[0, 20, 50, 100, 200, 9999],
        labels=["very_short(<20d)", "short(20-50d)", "medium(50-100d)", "long(100-200d)", "very_long(>200d)"],
    )
    for band, g in rdf.groupby("wave_dur_band", observed=True):
        _precision_stats(g, str(band))

    # ══════════════════════════════════════════════════════════
    # 8. 综合排名 — 按股票精度排序
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("8. 股票精度排名 (按 ±3% 触及率降序)")
    print("=" * 120)
    rank_rows = []
    for (mkt, sym, name), g in rdf.groupby(["market", "symbol", "name"]):
        nn = len(g)
        if nn < 5:
            continue
        mid_cols = ["dist_ema34_pct", "dist_ema55_pct", "dist_ema60_pct"]
        w3_mask = pd.Series(False, index=g.index)
        w5_mask = pd.Series(False, index=g.index)
        for c in mid_cols:
            w3_mask = w3_mask | (g[c].abs() <= 3)
            w5_mask = w5_mask | (g[c].abs() <= 5)
        w3 = w3_mask.sum()
        w5 = w5_mask.sum()
        # Find best EMA per stock
        best_ema = {}
        for c in mid_cols:
            ema_name = c.replace("dist_", "").replace("_pct", "")
            best_ema[ema_name] = round((g[c].abs() <= 3).sum() / nn * 100, 1)
        best = max(best_ema, key=best_ema.get)
        rank_rows.append({
            "stock": f"{mkt}:{sym}",
            "name": name,
            "n": nn,
            "median_dist": g["closer_dist_pct"].median(),
            "within_3pct": round(w3 / nn * 100, 1),
            "within_5pct": round(w5 / nn * 100, 1),
            "best_ema": best,
            "best_3pct": best_ema[best],
            "ema34_3pct": best_ema.get("ema34", 0),
            "ema55_3pct": best_ema.get("ema55", 0),
            "ema60_3pct": best_ema.get("ema60", 0),
        })
    rank_df = pd.DataFrame(rank_rows).sort_values("within_3pct", ascending=False)
    print(rank_df.to_string(index=False))

    # ══════════════════════════════════════════════════════════
    # 9. 每只股票按时间段拆分的精度变化
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("9. 每只股票 × 时间段 (half) — 支撑精度是否随时间变化？")
    print("=" * 120)
    for (mkt, sym, name), g_stock in rdf.groupby(["market", "symbol", "name"]):
        if len(g_stock) < 8:
            continue
        print(f"\n  {mkt}:{sym} {name}:")
        for half, g in g_stock.groupby("half"):
            if len(g) < 2:
                continue
            nn = len(g)
            mid_cols = ["dist_ema34_pct", "dist_ema55_pct", "dist_ema60_pct"]
            w3_mask = pd.Series(False, index=g.index)
            for c in mid_cols:
                w3_mask = w3_mask | (g[c].abs() <= 3)
            w3 = w3_mask.sum()
            med = g["closer_dist_pct"].median()
            print(f"    {half}: n={nn:2d}  median={med:+6.2f}%  ±3%={w3/nn*100:5.1f}%")

    # ══════════════════════════════════════════════════════
    # 10. 每条 EMA 线单独统计 — 哪条线的支撑效果最好
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("10. 每条 EMA 线单独统计 — 哪条线的支撑效果最好？")
    print("=" * 120)
    ema_names = [("EMA34", "dist_ema34_pct"), ("EMA55", "dist_ema55_pct"), ("EMA60", "dist_ema60_pct")]
    print(f"  {'EMA':>6s}  {'median':>8s}  {'mean':>8s}  {'±1%':>6s}  {'±2%':>6s}  {'±3%':>6s}  {'±5%':>6s}")
    for ema_label, col in ema_names:
        med = rdf[col].median()
        mean = rdf[col].mean()
        w1 = (rdf[col].abs() <= 1).sum()
        w2 = (rdf[col].abs() <= 2).sum()
        w3 = (rdf[col].abs() <= 3).sum()
        w5 = (rdf[col].abs() <= 5).sum()
        print(f"  {ema_label:>6s}  {med:+7.2f}%  {mean:+7.2f}%  {w1/n*100:5.1f}%  {w2/n*100:5.1f}%  {w3/n*100:5.1f}%  {w5/n*100:5.1f}%")
    # Any mid EMA
    any1 = any2 = any3 = any5 = pd.Series(False, index=rdf.index)
    for _, col in ema_names:
        any1 = any1 | (rdf[col].abs() <= 1)
        any2 = any2 | (rdf[col].abs() <= 2)
        any3 = any3 | (rdf[col].abs() <= 3)
        any5 = any5 | (rdf[col].abs() <= 5)
    print(f"  {'ANY':>6s}  {rdf['closer_dist_pct'].median():+7.2f}%  {rdf['closer_dist_pct'].mean():+7.2f}%  "
          f"{any1.sum()/n*100:5.1f}%  {any2.sum()/n*100:5.1f}%  {any3.sum()/n*100:5.1f}%  {any5.sum()/n*100:5.1f}%")

    # ══════════════════════════════════════════════════════
    # 11. 每只股票的“最佳支撑线”—— 个性化线别
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("11. 每只股票的最佳支撑线 (±3% 触及率)")
    print("=" * 120)
    print(f"  {'Stock':>12s} {'Name':>10s} {'n':>4s}  {'EMA34':>6s}  {'EMA55':>6s}  {'EMA60':>6s}  {'Best':>6s}  {'Best%':>6s}")
    for (mkt, sym, name), g in rdf.groupby(["market", "symbol", "name"]):
        nn = len(g)
        if nn < 5:
            continue
        e34_pct = round((g["dist_ema34_pct"].abs() <= 3).sum() / nn * 100, 1)
        e55_pct = round((g["dist_ema55_pct"].abs() <= 3).sum() / nn * 100, 1)
        e60_pct = round((g["dist_ema60_pct"].abs() <= 3).sum() / nn * 100, 1)
        best_val = max(e34_pct, e55_pct, e60_pct)
        best_name = "EMA34" if best_val == e34_pct else ("EMA55" if best_val == e55_pct else "EMA60")
        print(f"  {mkt}:{sym:>5s} {name:>10s} {nn:4d}  {e34_pct:5.1f}%  {e55_pct:5.1f}%  {e60_pct:5.1f}%  {best_name:>6s}  {best_val:5.1f}%")


if __name__ == "__main__":
    main()
