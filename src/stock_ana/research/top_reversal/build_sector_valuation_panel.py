#!/usr/bin/env python3
"""build_sector_valuation_panel.py — 构建静态「行业估值面板」（一次性算好、长期可用的研究数据）。

两个产物：
  1) 研究数据（用户点4）：每 (市场, 行业, 月) 的成员 PE/PB/PS 均值/中位数/数量。
  2) 行业内分位特征底座：as-of 把候选估值排进「该行业全宇宙成员在该月的分布」（成员级长表）。

口径（候选与成员同源、保证可比）：
  CN = stock_value_em 日频 PE(TTM)/市净率/市销率（pit/CN/{sym}__val.parquet）
  US = stockanalysis 逐季 pe/ps/pb（pit/US/{sym}__sa_q.parquet，季度→月 ffill）
  HK = eniu 日频 pe/pb（pit/HK/{sym}__val.parquet，无 ps）
行业：US=SIC3 + SIC2(sector2 供分层回退)；HK=行业；CN=cn_industry_map 行业。
清洗：PE/PB/PS ≤0 或 >1000 视为无意义 → NaN，不参与分布。

输出：
  data/cache/fundamentals/sector_valuation_members.parquet  (market,sector,sector2,month,sym,pe,pb,ps)
  data/cache/fundamentals/sector_valuation_summary.csv      (market,sector,month,n_*,*_mean,*_median)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # src/ 上 path（支持直接执行）

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stock_ana.config import DATA_DIR  # noqa: E402
from stock_ana.research.top_reversal.macro_micro_context import _build_sector_map  # noqa: E402
from stock_ana.research.top_reversal.valuation_context import _coarse_sector  # noqa: E402

PIT = DATA_DIR / "cache" / "fundamentals" / "pit"
# 永久研究数据，落 data/ 根（入 git，见 .gitignore 白名单），非 cache
OUT_MEMBERS = DATA_DIR / "sector_valuation_members.parquet"
OUT_SUMMARY = DATA_DIR / "sector_valuation_summary.csv"
MONTH_START = "2017-01"


def _clean(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    return v.where((v > 0) & (v < 1000))


def _cn_monthly(sym: str) -> pd.DataFrame | None:
    p = PIT / "CN" / f"{sym}__val.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    if "数据日期" not in d.columns:
        return None
    d = d.assign(_dt=pd.to_datetime(d["数据日期"], errors="coerce")).dropna(subset=["_dt"]).set_index("_dt").sort_index()
    out = pd.DataFrame({
        "pe": _clean(d.get("PE(TTM)")), "pb": _clean(d.get("市净率")), "ps": _clean(d.get("市销率")),
    }).resample("ME").last()
    return out


def _us_monthly(sym: str) -> pd.DataFrame | None:
    p = PIT / "US" / f"{sym}__sa_q.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    if "datekey" not in d.columns:
        return None
    d = d.assign(_dt=pd.to_datetime(d["datekey"], errors="coerce")).dropna(subset=["_dt"]).set_index("_dt").sort_index()
    q = pd.DataFrame({"pe": _clean(d.get("pe")), "pb": _clean(d.get("pb")), "ps": _clean(d.get("ps"))})
    # 季度值按月 ffill（最近一期已报比率持续到下期）
    return q.resample("ME").last().ffill()


def _hk_monthly(sym: str) -> pd.DataFrame | None:
    """HK：Futu 月末价 × 年报 EPS/BPS 重构 PE/PB（与候选 _hk_valuation 同口径；无 PS）。"""
    pp = ROOT / "data" / "cache" / "hk" / f"{sym}.parquet"
    fp = PIT / "HK" / f"{sym}__fin.parquet"
    if not (pp.exists() and fp.exists()):
        return None
    from stock_ana.research.top_reversal.smc_context import _normalize_df
    px = pd.to_numeric(_normalize_df(pd.read_parquet(pp))["close"], errors="coerce")
    px.index = pd.to_datetime(px.index)
    pxm = px.resample("ME").last().dropna()
    if pxm.empty:
        return None
    fin = pd.read_parquet(fp)
    if "REPORT_DATE" not in fin.columns:
        return None
    rd = pd.to_datetime(fin["REPORT_DATE"], errors="coerce")
    f = pd.DataFrame({
        "_avail": rd + pd.Timedelta(days=120),
        "eps": pd.to_numeric(fin.get("BASIC_EPS"), errors="coerce"),
        "bps": pd.to_numeric(fin.get("BPS"), errors="coerce"),
    }).dropna(subset=["_avail"]).sort_values("_avail")
    if f.empty:
        return None
    left = pd.DataFrame({"month": pxm.index, "price": pxm.to_numpy()})
    m = pd.merge_asof(left, f, left_on="month", right_on="_avail", direction="backward")
    pe = m["price"] / m["eps"].where(m["eps"] > 0)
    pb = m["price"] / m["bps"].where(m["bps"] > 0)
    out = pd.DataFrame({"pe": pe.to_numpy(), "pb": pb.to_numpy(), "ps": np.nan},
                       index=pd.DatetimeIndex(m["month"]))
    return out.apply(_clean)


_FN = {"CN": _cn_monthly, "US": _us_monthly, "HK": _hk_monthly}


def _members_by_sector(market: str) -> dict[str, list[str]]:
    """sector -> 全体成员 sym（来自行业映射 / SIC profiles）。复用 fetch_sector_universe 的口径。"""
    from stock_ana.research.top_reversal.fetch_sector_universe import _members_by_sector as mbs
    return mbs(market)


def main() -> None:
    sm = _build_sector_map()  # (market,sym) -> {sector,...}
    # 反查：sym -> sector，用映射成员表更直接
    rows: list[pd.DataFrame] = []
    for market in ["CN", "HK", "US"]:
        fn = _FN[market]
        members = _members_by_sector(market)
        n_sym = 0
        for sector, syms in members.items():
            sector2 = _coarse_sector(sector)
            for sym in syms:
                m = fn(sym)
                if m is None or m.dropna(how="all").empty:
                    continue
                m = m[m.index >= pd.Timestamp(MONTH_START)].copy()
                if m.empty:
                    continue
                m["month"] = m.index.strftime("%Y-%m")
                m["market"] = market
                m["sector"] = sector
                m["sector2"] = sector2
                m["sym"] = sym
                rows.append(m.reset_index(drop=True))
                n_sym += 1
        print(f"[{market}] 纳入成员 {n_sym} 只", flush=True)
    if not rows:
        print("无数据，退出"); return
    panel = pd.concat(rows, ignore_index=True)
    panel = panel[["market", "sector", "sector2", "month", "sym", "pe", "pb", "ps"]]
    OUT_MEMBERS.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_MEMBERS)
    print(f"成员级面板 {len(panel)} 行 → {OUT_MEMBERS}")

    # 聚合（研究数据）：每 (market,sector,month) 的均值/中位数/数量
    def agg(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
        g = df.groupby(by, observed=True)
        out = g.agg(
            pe_n=("pe", "count"), pe_mean=("pe", "mean"), pe_median=("pe", "median"),
            pb_n=("pb", "count"), pb_mean=("pb", "mean"), pb_median=("pb", "median"),
            ps_n=("ps", "count"), ps_mean=("ps", "mean"), ps_median=("ps", "median"),
        ).reset_index()
        for c in out.columns:
            if c.endswith(("_mean", "_median")):
                out[c] = out[c].round(2)
        return out

    summary = agg(panel, ["market", "sector", "month"])
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"行业估值聚合 {len(summary)} 行 → {OUT_SUMMARY}")
    # 摘要：每市场行业数、月份跨度、典型行业月成员数
    for market in ["CN", "HK", "US"]:
        s = summary[summary.market == market]
        if len(s):
            print(f"  {market}: 行业 {s['sector'].nunique()} 个，月份 {s['month'].min()}~{s['month'].max()}，"
                  f"行业-月 PE 成员数中位 {s['pe_n'].median():.0f}")


if __name__ == "__main__":
    main()
