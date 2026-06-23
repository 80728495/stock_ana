#!/usr/bin/env python3
"""refresh_valuation_pe.py — 刷新顶部模型用的估值 PE 数据（按市场分离）。

  US  前向 PE: stockanalysis.com  /api/symbol/s/{ticker}/statistics
  HK/CN trailing PE_TTM: Futu OpenD 行情快照（本地 127.0.0.1:11111）

输出（被 valuation_context.py 读取）：
  data/cache/fundamentals/us_forward_pe.csv   (ticker, forward_pe, pe, peg, ...)
  data/cache/fundamentals/futu_pe.csv         (market, symbol, code, pe, pe_ttm, pb, eps)

三市场估值中枢不同，特征侧在各自市场内做分位归一（见 valuation_context）。
US 前向 PE 是当前快照，用于历史回测有 look-ahead，仅适合实时打分；详见
docs/top_reversal_current_system.md §0.8。

用法：
    python scripts/refresh_valuation_pe.py            # US + HK/CN 全刷
    python scripts/refresh_valuation_pe.py --us       # 仅 US
    python scripts/refresh_valuation_pe.py --futu     # 仅 HK/CN
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stock_ana.data.list_manager import _read_md_table, load_cn_hightech_list, load_us_tech_list  # noqa: E402
from stock_ana.data.market_data import _tech_pool_symbols  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "cache" / "fundamentals"
WANT = {"pe", "peForward", "peg", "ps", "psForward"}


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
        rows.append(dict(ticker=tk, forward_pe=d.get("peForward"), pe=d.get("pe"), peg=d.get("peg"),
                         ps=d.get("ps"), ps_forward=d.get("psForward"), err=d.get("_err", "")))
        ok += d.get("peForward") is not None
        if i % 50 == 0:
            print(f"  {i}/{len(syms)} ok={ok}", flush=True)
        time.sleep(0.35)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "us_forward_pe.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "forward_pe", "pe", "peg", "ps", "ps_forward", "err"])
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


def main() -> None:
    p = argparse.ArgumentParser(description="刷新顶部模型估值 PE（US 前向 / HK-CN trailing）")
    p.add_argument("--us", action="store_true", help="仅刷新 US 前向 PE")
    p.add_argument("--futu", action="store_true", help="仅刷新 HK/CN trailing PE")
    args = p.parse_args()
    do_all = not (args.us or args.futu)
    if args.us or do_all:
        fetch_us_forward_pe()
    if args.futu or do_all:
        fetch_futu_pe()


if __name__ == "__main__":
    main()
