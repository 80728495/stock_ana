#!/usr/bin/env python3
"""统计 mid pullback 落在 EMA34/55 上的精确度。"""
import pandas as pd
import numpy as np
from stock_ana.strategies.primitives.wave import analyze_wave_structure
from stock_ana.config import CACHE_DIR

SAMPLES = {
    "APP": ("US", CACHE_DIR / "us" / "APP.parquet"),
    "NVDA": ("US", CACHE_DIR / "us" / "NVDA.parquet"),
    "META": ("US", CACHE_DIR / "us" / "META.parquet"),
    "TSLA": ("US", CACHE_DIR / "us" / "TSLA.parquet"),
    "AMD": ("US", CACHE_DIR / "us" / "AMD.parquet"),
    "RBLX": ("US", CACHE_DIR / "us" / "RBLX.parquet"),
    "MRNA": ("US", CACHE_DIR / "us" / "MRNA.parquet"),
    "MU": ("US", CACHE_DIR / "us" / "MU.parquet"),
    "TEM": ("US", CACHE_DIR / "us" / "TEM.parquet"),
    "ALAB": ("US", CACHE_DIR / "us" / "ALAB.parquet"),
    "PDD": ("US", CACHE_DIR / "us" / "PDD.parquet"),
    "MSFT": ("US", CACHE_DIR / "us" / "MSFT.parquet"),
    "GOOG": ("US", CACHE_DIR / "us" / "GOOG.parquet"),
    "09992": ("HK", CACHE_DIR / "hk" / "09992.parquet"),
    "00700": ("HK", CACHE_DIR / "hk" / "00700.parquet"),
    "01810": ("HK", CACHE_DIR / "hk" / "01810.parquet"),
    "09988": ("HK", CACHE_DIR / "hk" / "09988.parquet"),
    "00981": ("HK", CACHE_DIR / "hk" / "00981.parquet"),
    "03690": ("HK", CACHE_DIR / "hk" / "03690.parquet"),
    "01347": ("HK", CACHE_DIR / "hk" / "01347.parquet"),
    "02400": ("HK", CACHE_DIR / "hk" / "02400.parquet"),
    "01024": ("HK", CACHE_DIR / "hk" / "01024.parquet"),
    "09626": ("HK", CACHE_DIR / "hk" / "09626.parquet"),
    "00189": ("HK", CACHE_DIR / "hk" / "00189.parquet"),
    "06869": ("HK", CACHE_DIR / "hk" / "06869.parquet"),
    "02228": ("HK", CACHE_DIR / "hk" / "02228.parquet"),
    "02788": ("HK", CACHE_DIR / "hk" / "02788.parquet"),
}


def main():
    """Summarize how closely sampled mid-pullbacks align with EMA34 versus EMA55."""
    rows = []
    for sym, (mkt, path) in SAMPLES.items():
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

        result = analyze_wave_structure(df)
        for mw in result["major_waves"]:
            for sw in mw["sub_waves"]:
                if sw["pullback_type"] != "mid_vegas":
                    continue
                ep = sw["end_pivot"]
                v = ep["value"]
                i = ep["iloc"]
                e34 = float(ema34.iloc[i])
                e55 = float(ema55.iloc[i])
                dist34 = (v - e34) / e34 * 100 if e34 > 0 else np.nan
                dist55 = (v - e55) / e55 * 100 if e55 > 0 else np.nan
                closer = "ema34" if abs(dist34) <= abs(dist55) else "ema55"
                closer_dist = dist34 if closer == "ema34" else dist55
                rows.append({
                    "symbol": sym,
                    "market": mkt,
                    "wave": mw["wave_number"],
                    "sub": sw["sub_number"],
                    "date": ep["date"],
                    "low_val": round(v, 2),
                    "ema34": round(e34, 2),
                    "ema55": round(e55, 2),
                    "dist_ema34_pct": round(dist34, 2),
                    "dist_ema55_pct": round(dist55, 2),
                    "closer_band": closer,
                    "closer_dist_pct": round(closer_dist, 2),
                })

    rdf = pd.DataFrame(rows)
    n = len(rdf)
    print(f"Total mid pullbacks: {n}")
    print()

    # 1) 更靠近哪条均线
    print("=" * 60)
    print("1. Closer Band Distribution")
    print("=" * 60)
    for band, cnt in rdf["closer_band"].value_counts().items():
        print(f"  {band}: {cnt} ({cnt/n*100:.1f}%)")
    print()

    # 2) 距离最近 EMA 的偏差统计
    print("=" * 60)
    print("2. Distance to Closer EMA (%)")
    print("=" * 60)
    desc = rdf["closer_dist_pct"].describe()
    for k, v in desc.items():
        print(f"  {k:>6s}: {v:.2f}")
    print()

    # 3) 精度分层
    print("=" * 60)
    print("3. Precision Distribution (vs closer EMA)")
    print("=" * 60)
    bins = [-999, -5, -3, -1, 0, 1, 2, 3, 5, 999]
    labels = ["<-5%", "-5~-3%", "-3~-1%", "-1~0%", "0~1%", "1~2%", "2~3%", "3~5%", ">5%"]
    rdf["precision_band"] = pd.cut(rdf["closer_dist_pct"], bins=bins, labels=labels)
    cnt = rdf["precision_band"].value_counts().sort_index()
    for lb, c in cnt.items():
        bar = "█" * int(c / n * 100)
        print(f"  {lb:>8s}: {c:4d}  ({c/n*100:5.1f}%)  {bar}")
    print()

    # 4) 分别统计 vs EMA34 和 vs EMA55
    print("=" * 60)
    print("4. Distance to EMA34 (%)")
    print("=" * 60)
    desc34 = rdf["dist_ema34_pct"].describe()
    for k, v in desc34.items():
        print(f"  {k:>6s}: {v:.2f}")
    print()

    print("=" * 60)
    print("5. Distance to EMA55 (%)")
    print("=" * 60)
    desc55 = rdf["dist_ema55_pct"].describe()
    for k, v in desc55.items():
        print(f"  {k:>6s}: {v:.2f}")
    print()

    # 5) 精确触及比例
    print("=" * 60)
    print("6. Precise Touch Rate")
    print("=" * 60)
    for tol_name, tol in [("±1%", 1), ("±2%", 2), ("±3%", 3)]:
        t34 = (rdf["dist_ema34_pct"].abs() <= tol).sum()
        t55 = (rdf["dist_ema55_pct"].abs() <= tol).sum()
        either = ((rdf["dist_ema34_pct"].abs() <= tol) | (rdf["dist_ema55_pct"].abs() <= tol)).sum()
        print(f"  {tol_name} of EMA34:  {t34:4d}/{n} = {t34/n*100:5.1f}%")
        print(f"  {tol_name} of EMA55:  {t55:4d}/{n} = {t55/n*100:5.1f}%")
        print(f"  {tol_name} of either: {either:4d}/{n} = {either/n*100:5.1f}%")
        print()

    # 6) 上穿 vs 下穿：swing low 在 EMA 上方 vs 下方
    print("=" * 60)
    print("7. Above/Below EMA at touch point")
    print("=" * 60)
    above34 = (rdf["dist_ema34_pct"] > 0).sum()
    below34 = (rdf["dist_ema34_pct"] <= 0).sum()
    above55 = (rdf["dist_ema55_pct"] > 0).sum()
    below55 = (rdf["dist_ema55_pct"] <= 0).sum()
    print(f"  vs EMA34:  above={above34} ({above34/n*100:.1f}%)  below={below34} ({below34/n*100:.1f}%)")
    print(f"  vs EMA55:  above={above55} ({above55/n*100:.1f}%)  below={below55} ({below55/n*100:.1f}%)")
    print()

    # 7) 每只股票的平均精度
    print("=" * 60)
    print("8. Per-stock avg distance to closer EMA (%)")
    print("=" * 60)
    g = rdf.groupby(["market", "symbol"]).agg(
        count=("closer_dist_pct", "count"),
        mean_dist=("closer_dist_pct", "mean"),
        median_dist=("closer_dist_pct", "median"),
        min_dist=("closer_dist_pct", "min"),
        max_dist=("closer_dist_pct", "max"),
    ).round(2)
    print(g.to_string())


if __name__ == "__main__":
    main()
