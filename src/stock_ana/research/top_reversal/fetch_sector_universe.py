#!/usr/bin/env python3
"""fetch_sector_universe.py — 为「行业内估值分位」补抓各候选行业的更多成员历史估值。

目标：候选里出现的每个行业，尽量凑够 ≥MIN_PEERS(默认20) 个有历史估值的标的，做行业估值分布。
来源复用 fetch_pit_fundamentals（CN=stock_value_em 日频，与候选同源最干净；US=stockanalysis；
HK=akshare 年报，注意非候选缺价格、估值重构受限）。断点续抓（已存在跳过）。

用法: python scripts/fetch_sector_universe.py --market CN [--per-sector 25]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # src/ 上 path（支持直接执行）

import pandas as pd  # noqa: E402

from stock_ana.config import DATA_DIR  # noqa: E402
from stock_ana.research.top_reversal import fetch_pit_fundamentals as fpf  # noqa: E402
from stock_ana.research.top_reversal.macro_micro_context import _build_sector_map  # noqa: E402
from stock_ana.research.top_reversal.pit_fundamentals import _norm_sym  # noqa: E402

PITDIR = DATA_DIR / "cache" / "fundamentals" / "pit"


def _cand_sectors(market: str) -> set[str]:
    """候选集里出现的行业（sector）。"""
    sm = _build_sector_map()
    if market == "US":
        f = DATA_DIR / "output" / "top_candidate_research" / "_pm" / "US" / "watchlist_unified_recall_candidates_labeled.csv"
        if not f.exists():
            f = DATA_DIR / "output" / "top_candidate_research" / "watchlist_unified_recall_candidates_labeled.csv"
    else:
        f = DATA_DIR / "output" / "top_candidate_research" / "_pm" / market / "watchlist_unified_recall_candidates_labeled.csv"
    df = pd.read_csv(f, usecols=["market", "sym"], low_memory=False)
    df = df[df["market"] == market]
    df["sym"] = df["sym"].astype(str)
    secs = {(sm.get((market, _norm_sym(market, s)), {}) or {}).get("sector") for s in df["sym"].unique()}
    return {x for x in secs if x}


def _members_by_sector(market: str) -> dict[str, list[str]]:
    """该市场 sector -> 全体成员 sym（来自行业映射 / SIC profiles）。"""
    out: dict[str, list[str]] = {}
    if market == "CN":
        cn = pd.read_csv(DATA_DIR / "cn_industry_map.csv")
        for _, r in cn.iterrows():
            c = str(r.iloc[0])
            if c.startswith(("SZ.", "SH.")):
                out.setdefault(f"CN_{str(r['industry']).strip()}", []).append(c.split(".")[1].zfill(6))
    elif market == "HK":
        hk = pd.read_csv(DATA_DIR / "hk_industry_map.csv")
        for _, r in hk.iterrows():
            c = str(r.iloc[0])
            if c.startswith("HK."):
                out.setdefault(f"HK_{str(r['industry']).strip()}", []).append(c.split(".")[1].zfill(5))
    else:  # US: SIC 3 位主组
        us = pd.read_csv(DATA_DIR / "us_sec_profiles.csv")
        for _, r in us.iterrows():
            t = str(r.get("ticker", "")).strip().upper()
            sic = r.get("sic_code")
            if t and pd.notna(sic) and float(sic) > 0:
                out.setdefault(f"US_SIC{int(float(sic)) // 10}", []).append(t)
    return out


def _cached(market: str, sym: str) -> bool:
    if market == "US":
        return (PITDIR / "US" / f"{sym}__sa_q.parquet").exists()
    if market == "HK":
        return (PITDIR / "HK" / f"{sym}__fin.parquet").exists()
    return (PITDIR / "CN" / f"{sym}__val.parquet").exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True, choices=["CN", "HK", "US"])
    ap.add_argument("--per-sector", type=int, default=12, help="每个行业最多抓多少成员（目标 ≥8，留余量防失败）")
    args = ap.parse_args()
    mk = args.market

    secs = _cand_sectors(mk)
    members = _members_by_sector(mk)
    targets: list[str] = []
    seen: set[str] = set()
    for s in sorted(secs):
        for sym in members.get(s, [])[: args.per_sector]:
            if sym not in seen:
                seen.add(sym)
                if not _cached(mk, sym):
                    targets.append(sym)
    print(f"[{mk}] 候选行业 {len(secs)} 个，目标每行业≤{args.per_sector} 成员 → 待抓 {len(targets)} 只", flush=True)

    fn = {"CN": fpf.fetch_cn, "HK": fpf.fetch_hk, "US": fpf.fetch_us}[mk]
    for i, sym in enumerate(targets, 1):
        try:
            fn(sym)
        except Exception as e:  # noqa: BLE001
            print(f"  {mk} {sym} 失败: {repr(e)[:80]}", flush=True)
        if i % 25 == 0:
            print(f"  [{mk} {i}/{len(targets)}]", flush=True)
        time.sleep(0.3)
    print(f"[{mk}] done", flush=True)


if __name__ == "__main__":
    main()
