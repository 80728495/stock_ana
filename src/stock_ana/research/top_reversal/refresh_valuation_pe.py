#!/usr/bin/env python3
"""refresh_valuation_pe.py — 刷新顶部模型用的估值 PE 数据（按市场分离）。

  US  前向 PE: stockanalysis.com  /api/symbol/s/{ticker}/statistics
  HK/CN trailing PE_TTM: Futu OpenD 行情快照（本地 127.0.0.1:11111）

输出：
  data/cache/fundamentals/us_forward_pe.csv   (ticker, forward_pe, pe, ps, ps_forward, pb)  ← valuation_context
  data/cache/fundamentals/futu_pe.csv         (market, symbol, code, pe, pe_ttm, pb, eps)    ← valuation_context
  data/cache/fundamentals/futu_growth.csv     (market, symbol, earnings_growth, revenue_growth) ← growth_context (HK/CN)

三市场估值中枢不同，特征侧在各自市场内做分位归一（见 valuation_context）。
US 前向 PE 是当前快照，用于历史回测有 look-ahead，仅适合实时打分；详见
docs/top_reversal_current_system.md §0.8。

用法：
    python src/stock_ana/research/top_reversal/refresh_valuation_pe.py            # US + HK/CN 全刷
    python src/stock_ana/research/top_reversal/refresh_valuation_pe.py --us       # 仅 US
    python src/stock_ana/research/top_reversal/refresh_valuation_pe.py --futu     # 仅 HK/CN
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
import time
import urllib.request
from pathlib import Path


def _find_project_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError("Cannot find project root containing pyproject.toml")


PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stock_ana.data.list_manager import _read_md_table, load_cn_hightech_list, load_us_tech_list  # noqa: E402
from stock_ana.data.market_data import _tech_pool_symbols  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "cache" / "fundamentals"
WANT = {"pe", "peForward", "ps", "psForward", "pb"}


def fetch_us_forward_pe() -> None:
    socket.setdefaulttimeout(20)
    syms = sorted({str(e.get("ticker", "")).strip().upper() for e in load_us_tech_list() if e.get("ticker")})
    print(f"[US] stockanalysis 前向PE: {len(syms)} 只", flush=True)

    def fetch(tk: str) -> dict:
        req = urllib.request.Request(
            f"https://stockanalysis.com/api/symbol/s/{tk}/statistics",
            headers={"User-Agent": "Mozilla/5.0 (Macintosh)"},
        )
        j = json.loads(urllib.request.urlopen(req).read().decode("utf-8", "ignore"))
        data = j.get("data", {})
        out: dict[str, float] = {}
        for _k, v in (data.items() if isinstance(data, dict) else []):
            items = v.get("data") if isinstance(v, dict) else (v if isinstance(v, list) else [])
            for it in items or []:
                if isinstance(it, dict) and it.get("id") in WANT:
                    try:
                        out[it["id"]] = float(str(it.get("value", "")).replace(",", "").replace("%", ""))
                    except ValueError:
                        pass
        return out

    rows, ok = [], 0
    for i, tk in enumerate(syms, 1):
        d: dict = {}
        for attempt in range(2):
            try:
                d = fetch(tk)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 1:
                    d = {"_err": repr(exc)[:60]}
                time.sleep(1.0)
        rows.append(dict(ticker=tk, forward_pe=d.get("peForward"), pe=d.get("pe"),
                         ps=d.get("ps"), ps_forward=d.get("psForward"), pb=d.get("pb"), err=d.get("_err", "")))
        ok += d.get("peForward") is not None
        if i % 50 == 0:
            print(f"  {i}/{len(syms)} ok={ok}", flush=True)
        time.sleep(0.35)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "us_forward_pe.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "forward_pe", "pe", "ps", "ps_forward", "pb", "err"])
        w.writeheader()
        w.writerows(rows)
    print(f"[US] 完成 {ok}/{len(syms)} 有前向PE -> {OUT_DIR/'us_forward_pe.csv'}", flush=True)


def fetch_futu_pe() -> None:
    from futu import RET_OK, OpenQuoteContext  # type: ignore[import]

    def fc(mk: str, s: str) -> str | None:
        if mk == "US":
            return f"US.{s}"
        if mk == "HK":
            return f"HK.{s.zfill(5)}"
        if mk == "CN":
            s = s.zfill(6)
            return f"SH.{s}" if s[0] == "6" else f"SZ.{s}"
        return None

    meta = {}
    for mk, s, _n in _tech_pool_symbols(include_holding=True):
        c = fc(mk, s)
        if c:
            meta[c] = (mk, s)
    codes = sorted(meta)
    print(f"[Futu] HK/CN trailing PE: {len(codes)} 只", flush=True)
    ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    rows: list[dict] = []

    def query(batch: list[str]) -> None:
        if not batch:
            return
        ret, data = ctx.get_market_snapshot(batch)
        if ret == RET_OK:
            for _, r in data.iterrows():
                mk, s = meta.get(r["code"], ("", ""))
                rows.append(dict(market=mk, symbol=s, code=r["code"], pe=r.get("pe_ratio"),
                                 pe_ttm=r.get("pe_ttm_ratio"), pb=r.get("pb_ratio"), eps=r.get("earning_per_share")))
            time.sleep(0.5)
            return
        if len(batch) == 1:  # 单个坏代码（指数/退市），跳过
            return
        mid = len(batch) // 2
        query(batch[:mid])
        query(batch[mid:])

    try:
        for i in range(0, len(codes), 100):
            query(codes[i:i + 100])
    finally:
        ctx.close()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "futu_pe.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["market", "symbol", "code", "pe", "pe_ttm", "pb", "eps"])
        w.writeheader()
        w.writerows(rows)
    import collections
    c = collections.Counter(x["market"] for x in rows if str(x["pe_ttm"]) not in ("None", "nan", ""))
    print(f"[Futu] 完成 {len(rows)} 行，有pe_ttm: {dict(c)} -> {OUT_DIR/'futu_pe.csv'}", flush=True)


def fetch_futu_growth() -> None:
    """HK/CN 个股增长率（Futu 财务筛选，最新年报快照）。

    earnings_growth ← PROFIT_TO_SHAREHOLDERS_GROWTH_RATE（归母净利增长，与 US earnings 对齐）
    revenue_growth  ← SUM_OF_BUSINESS_GROWTH（营业总收入增长）

    Futu 无历史逐年财务，只给最新年报 → 与估值乘数同档 look-ahead，仅适合实时打分。
    只保留科技池+持仓内的标的（filter 返回全市场）。
    """
    import re

    from futu import (  # type: ignore[import]
        FinancialFilter, FinancialQuarter, Market, OpenQuoteContext, RET_OK, SortDir, StockField,
    )

    want = {(mk, s.zfill(5) if mk == "HK" else s.zfill(6)) for mk, s, _n in _tech_pool_symbols(include_holding=True) if mk in ("HK", "CN")}

    def mk_filter(field: StockField) -> FinancialFilter:
        f = FinancialFilter()
        f.filter_min, f.filter_max = -1e9, 1e9
        f.stock_field, f.is_no_filter = field, False
        f.sort, f.quarter = SortDir.NONE, FinancialQuarter.ANNUAL
        return f

    flist = [mk_filter(StockField.PROFIT_TO_SHAREHOLDERS_GROWTH_RATE), mk_filter(StockField.SUM_OF_BUSINESS_GROWTH)]
    pat_eg = re.compile(r"profit_to_shareholders_growth_rate\(annual\):([-\d.]+)")
    pat_rg = re.compile(r"sum_of_business_growth\(annual\):([-\d.]+)")

    ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    rows: list[dict] = []
    try:
        for mk_name, fmk in (("HK", Market.HK), ("CN", Market.SH)):  # SH 返回整个 A 股
            begin, total = 0, None
            while True:
                ret, ls = ctx.get_stock_filter(market=fmk, filter_list=flist, begin=begin, num=200)
                if ret != RET_OK:
                    print(f"[Futu-growth] {mk_name} begin={begin} 失败: {ls}", flush=True)
                    break
                last_page, total, lst = ls
                for v in lst:
                    s = str(v)
                    code = re.search(r"stock_code:(\S+)", s)
                    if not code:
                        continue
                    raw = code.group(1)  # HK.06132 / SH.600519 / SZ.301669
                    sym = raw.split(".", 1)[1]
                    mk = "HK" if raw.startswith("HK") else "CN"
                    sym = sym.zfill(5) if mk == "HK" else sym.zfill(6)
                    if (mk, sym) not in want:
                        continue
                    eg = pat_eg.search(s)
                    rg = pat_rg.search(s)
                    rows.append(dict(market=mk, symbol=sym,
                                     earnings_growth=float(eg.group(1)) if eg else "",
                                     revenue_growth=float(rg.group(1)) if rg else ""))
                begin += len(lst)
                if last_page or begin >= (total or 0):
                    break
                time.sleep(3.2)  # 频率限制：每 30s ≤10 次
            time.sleep(3.2)
    finally:
        ctx.close()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "futu_growth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["market", "symbol", "earnings_growth", "revenue_growth"])
        w.writeheader()
        w.writerows(rows)
    import collections
    c = collections.Counter(r["market"] for r in rows if r["earnings_growth"] != "")
    print(f"[Futu-growth] 完成 {len(rows)} 行，有盈利增长: {dict(c)} -> {OUT_DIR/'futu_growth.csv'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="刷新顶部模型估值 PE（US 前向 / HK-CN trailing）+ HK/CN 增长")
    p.add_argument("--us", action="store_true", help="仅刷新 US 前向 PE")
    p.add_argument("--futu", action="store_true", help="仅刷新 HK/CN trailing PE")
    p.add_argument("--futu-growth", action="store_true", help="仅刷新 HK/CN 增长率")
    args = p.parse_args()
    do_all = not (args.us or args.futu or args.futu_growth)
    if args.us or do_all:
        fetch_us_forward_pe()
    if args.futu or do_all:
        fetch_futu_pe()
    if args.futu_growth or do_all:
        fetch_futu_growth()


if __name__ == "__main__":
    main()
