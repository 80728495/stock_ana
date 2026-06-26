#!/usr/bin/env python3
"""fetch_pit_fundamentals.py — 抓三市场历史基本面(估值日频 + 财报)，落盘缓存供 PIT 特征用。

源(已验证, akshare 1.18.22)：
  CN 日频估值  ak.stock_value_em(symbol=code)              -> 数据日期/PE(TTM)/市净率/市销率/PEG值
  CN 历史财报  ak.stock_financial_abstract(symbol=code)     -> 宽表 指标×报告期(季度)
  HK 日频PE/PB ak.stock_hk_indicator_eniu(symbol="hk"+code, indicator="市盈率"/"市净率")
  HK 历史财报  ak.stock_financial_hk_analysis_indicator_em(symbol=code, indicator="年度")
  US 历史财报  ak.stock_financial_us_analysis_indicator_em(symbol=ticker, indicator="年报"/"单季报")
               (含 NOTICE_DATE 公告日, BASIC_EPS, OPERATE_INCOME, ROE_AVG, NET_PROFIT_RATIO)
  US 日频估值  无现成(baidu挂)→特征构建时用 parquet 日价 × 历史每股重构。

落盘: data/cache/fundamentals/pit/{market}/{sym}__{kind}.parquet  (kind: val / fin)
可续抓: 已存在则跳过。逐只 try/except, 进度打到 stdout。
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # src/ 上 path（支持直接执行）

import json  # noqa: E402
import urllib.request  # noqa: E402

import akshare as ak  # noqa: E402
import pandas as pd  # noqa: E402

from stock_ana.config import DATA_DIR  # noqa: E402

_SA_UA = {"User-Agent": "Mozilla/5.0 (Macintosh)"}

# 用统一候选全集（escape-top 是其子集）作抓取 universe，覆盖所有训练/验证候选
LABELED = DATA_DIR / "output" / "top_candidate_research" / "watchlist_unified_recall_candidates_labeled.csv"
OUTDIR = DATA_DIR / "cache" / "fundamentals" / "pit"


def universe() -> dict[str, list[str]]:
    lab = pd.read_csv(LABELED, usecols=["market", "sym"], low_memory=False)
    lab["sym"] = lab["sym"].astype(str)
    out: dict[str, list[str]] = {}
    for mk, g in lab.groupby("market"):
        syms = sorted(g["sym"].unique())
        if mk == "HK":
            syms = [s.zfill(5) for s in syms]
        elif mk == "CN":
            syms = [s.zfill(6) for s in syms]
        out[str(mk)] = syms
    return out


def _save(df: pd.DataFrame | None, path: Path) -> bool:
    if df is None or len(df) == 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return True


def fetch_cn(sym: str) -> None:
    d = OUTDIR / "CN"
    vp, fp = d / f"{sym}__val.parquet", d / f"{sym}__fin.parquet"
    if not vp.exists():
        try:
            _save(ak.stock_value_em(symbol=sym), vp)
        except Exception as e:  # noqa: BLE001
            print(f"  CN {sym} val 失败: {repr(e)[:80]}")
    if not fp.exists():
        try:
            _save(ak.stock_financial_abstract(symbol=sym), fp)
        except Exception as e:  # noqa: BLE001
            print(f"  CN {sym} fin 失败: {repr(e)[:80]}")


def fetch_hk(sym: str) -> None:
    d = OUTDIR / "HK"
    vp, fp = d / f"{sym}__val.parquet", d / f"{sym}__fin.parquet"
    if not vp.exists():
        try:
            pe = ak.stock_hk_indicator_eniu(symbol="hk" + sym, indicator="市盈率")
            pb = ak.stock_hk_indicator_eniu(symbol="hk" + sym, indicator="市净率")
            m = pe.merge(pb[["date", "pb"]], on="date", how="outer") if {"date"} <= set(pe.columns) else pe
            _save(m, vp)
        except Exception as e:  # noqa: BLE001
            print(f"  HK {sym} val 失败: {repr(e)[:80]}")
    if not fp.exists():
        try:
            _save(ak.stock_financial_hk_analysis_indicator_em(symbol=sym, indicator="年度"), fp)
        except Exception as e:  # noqa: BLE001
            print(f"  HK {sym} fin 失败: {repr(e)[:80]}")


def _sa_get(url: str) -> dict:
    req = urllib.request.Request(url, headers=_SA_UA)
    return json.loads(urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "ignore"))


def _sa_struct(payload: dict) -> dict | None:
    """从 stockanalysis SvelteKit __data.json(devalue 索引格式) 取含 datekey 的对齐列结构。

    devalue: nodes[].data 是扁平数组，dict/list 的值是指向其他下标的索引；递归还原后
    找含 datekey 列表的最长结构即历史财务/比率表。
    """
    best = None
    for n in payload.get("nodes", []):
        if not (isinstance(n, dict) and isinstance(n.get("data"), list)):
            continue
        flat = n["data"]

        def resolve(i, seen):
            if not isinstance(i, int) or i < 0 or i >= len(flat) or i in seen:
                return None
            seen = seen | {i}
            v = flat[i]
            if isinstance(v, dict):
                return {k: resolve(x, seen) for k, x in v.items()}
            if isinstance(v, list):
                return [resolve(x, seen) if isinstance(x, int) else x for x in v]
            return v

        found: list[dict] = []

        def search(o):
            if isinstance(o, dict):
                if isinstance(o.get("datekey"), list):
                    found.append(o)
                for v in o.values():
                    search(v)
            elif isinstance(o, list):
                for v in o[:80]:
                    search(v)

        search(resolve(0, frozenset()))
        for f in found:
            if best is None or len(f["datekey"]) > len(best["datekey"]):
                best = f
    return best


def _sa_frame(sym: str, period: str) -> pd.DataFrame | None:
    """合并 stockanalysis financials + ratios → 历史 DataFrame（按 datekey 对齐）。

    列: datekey, epsBasic, epsDiluted, revenue, sharesDiluted, netIncome, pe, ps, pb, lastClosePrice
    （datekey=财报期末；revenue/eps 用于算 TTM YoY，pe/ps/pb 为逐期比率）。丢弃无法定日期的 TTM 行。
    """
    base = f"https://stockanalysis.com/stocks/{sym}/financials"
    try:
        fin = _sa_struct(_sa_get(f"{base}/__data.json?p={period}"))
        time.sleep(0.3)
        rat = _sa_struct(_sa_get(f"{base}/ratios/__data.json?p={period}"))
    except Exception as e:  # noqa: BLE001
        print(f"  US {sym} {period} 失败: {repr(e)[:80]}")
        return None
    if not fin or not fin.get("datekey"):
        return None

    def cols(struct: dict | None, names: list[str]) -> dict:
        if not struct:
            return {}
        n = len(struct["datekey"])
        out = {}
        for k in names:
            v = struct.get(k)
            out[k] = v if (isinstance(v, list) and len(v) == n) else [None] * n
        return out

    df = pd.DataFrame({"datekey": fin["datekey"], **cols(fin, ["epsBasic", "epsDiluted", "revenue", "sharesDiluted", "netIncome"])})
    if rat and rat.get("datekey"):
        r = pd.DataFrame({"datekey": rat["datekey"], **cols(rat, ["pe", "ps", "pb", "lastClosePrice"])})
        df = df.merge(r, on="datekey", how="left")
    df = df[df["datekey"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}")].copy()
    return df if len(df) else None


def fetch_us(sym: str) -> None:
    """US 历史基本面：stockanalysis.com 季度+年度（同源、日期可靠；不用 akshare 坏 NOTICE_DATE）。"""
    d = OUTDIR / "US"
    qp, yp = d / f"{sym}__sa_q.parquet", d / f"{sym}__sa_y.parquet"
    if not qp.exists():
        _save(_sa_frame(sym, "quarterly"), qp)
    if not yp.exists():
        _save(_sa_frame(sym, "annual"), yp)


def main() -> None:
    uni = universe()
    fns = {"CN": fetch_cn, "HK": fetch_hk, "US": fetch_us}
    for mk in ["CN", "HK", "US"]:
        syms = uni.get(mk, [])
        print(f"=== {mk}: {len(syms)} 只 ===", flush=True)
        for i, s in enumerate(syms, 1):
            fns[mk](s)
            if i % 20 == 0:
                print(f"  {mk} [{i}/{len(syms)}]", flush=True)
            time.sleep(0.3)  # 温和限速
        print(f"=== {mk} done ===", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
