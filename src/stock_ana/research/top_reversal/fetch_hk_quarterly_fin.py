#!/usr/bin/env python3
"""fetch_hk_quarterly_fin.py — 为已有年报(__fin)的 HK 股补抓 akshare「报告期」季度财务 → __fin_q。

HK 监管多数只半年报，但 akshare「报告期」能给近 ~2 年的季度 YoY（HOLDER_PROFIT_YOY/OPERATE_INCOME_YOY）；
与年报(__fin, 深历史)拼接：近段季度更新、远段退年度。HK 候选多在 2024–2026，正落在季度可得窗内。
断点续抓（已存在跳过）。
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # src/ 上 path（支持直接执行）

import akshare as ak  # noqa: E402

from stock_ana.config import DATA_DIR  # noqa: E402

PITHK = DATA_DIR / "cache" / "fundamentals" / "pit" / "HK"


def main() -> None:
    syms = sorted(p.name.split("__")[0] for p in PITHK.glob("*__fin.parquet"))
    todo = [s for s in syms if not (PITHK / f"{s}__fin_q.parquet").exists()]
    print(f"[HK fin_q] 已有年报 {len(syms)} 只，待补季度 {len(todo)} 只", flush=True)
    ok = fail = 0
    for i, s in enumerate(todo, 1):
        try:
            d = ak.stock_financial_hk_analysis_indicator_em(symbol=s, indicator="报告期")
            if d is not None and len(d):
                d.to_parquet(PITHK / f"{s}__fin_q.parquet")
                ok += 1
            else:
                fail += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  HK {s} 报告期失败: {repr(e)[:70]}", flush=True)
        if i % 25 == 0:
            print(f"  [HK fin_q {i}/{len(todo)}] ok={ok} fail={fail}", flush=True)
        time.sleep(0.3)
    print(f"[HK fin_q] done ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
