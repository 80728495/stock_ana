#!/usr/bin/env python3
"""fetch_hk_member_prices.py — 用 Futu(项目自有源)补抓 HK 行业成员的日线价格。

背景：akshare 的 HK 历史源在本环境不可靠（eniu ~半数失败、stock_hk_hist 被墙、baidu 坏）。
项目自有 Futu 源可靠且与候选 HK 估值同口径（pit_fundamentals._hk_valuation = as-of 价 × 年报 EPS/BPS）。
本脚本对「有年报(__fin)但缺价格」的 HK 行业成员补抓价格 → data/cache/hk/{sym}.parquet，
之后行业估值面板即可用 价×年报 一致地重构 PE/PB。断点续抓（已存在跳过）。
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # src/ 上 path（支持直接执行）

import pandas as pd  # noqa: E402

from stock_ana.config import DATA_DIR  # noqa: E402
from stock_ana.data.fetcher_futu import fetch_hk_stock_with_ctx, quote_context  # noqa: E402
from stock_ana.research.top_reversal.fetch_sector_universe import _cand_sectors, _members_by_sector  # noqa: E402

HK_PRICE_DIR = DATA_DIR / "cache" / "hk"
PIT_HK = DATA_DIR / "cache" / "fundamentals" / "pit" / "HK"
START, END = "2017-01-01", "2026-06-30"
PER_SECTOR = 12


def main() -> None:
    secs = _cand_sectors("HK")
    mem = _members_by_sector("HK")
    targets: list[str] = []
    seen: set[str] = set()
    for s in sorted(secs):
        for sym in mem.get(s, [])[:PER_SECTOR]:
            if sym in seen:
                continue
            seen.add(sym)
            # 只为「有年报但缺价格」的成员补价（有价格的跳过；无年报的算不出 PE/PB，免抓）
            if (PIT_HK / f"{sym}__fin.parquet").exists() and not (HK_PRICE_DIR / f"{sym}.parquet").exists():
                targets.append(sym)
    print(f"[HK price] 待补抓 {len(targets)} 只（有年报、缺价格的行业成员）", flush=True)
    if not targets:
        print("无需补抓"); return

    HK_PRICE_DIR.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    with quote_context() as ctx:  # contextmanager：自动开关连接
        for i, sym in enumerate(targets, 1):
            try:
                df = fetch_hk_stock_with_ctx(ctx, sym, START, END)
                if df is not None and len(df):
                    df.to_parquet(HK_PRICE_DIR / f"{sym}.parquet")
                    ok += 1
                else:
                    fail += 1
            except Exception as e:  # noqa: BLE001
                fail += 1
                print(f"  HK {sym} 价格失败: {repr(e)[:80]}", flush=True)
            if i % 20 == 0:
                print(f"  [HK price {i}/{len(targets)}] ok={ok} fail={fail}", flush=True)
            time.sleep(0.5)  # Futu 频率限制
    print(f"[HK price] done ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
