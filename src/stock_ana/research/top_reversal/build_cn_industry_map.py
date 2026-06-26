#!/usr/bin/env python3
"""build_cn_industry_map.py — 建立 A 股个股→行业板块映射（类比 hk_industry_map.csv）。

来源：Futu OpenD（本地 127.0.0.1:11111）
  get_plate_list(SH, INDUSTRY) → 全部 A 股行业板块（~131 个，code 形如 SH.LIST0002=半导体）
  逐板块 get_plate_stock(code) → 成分股，反向建 code→industry

输出：data/cn_industry_map.csv  (futu_code, industry)  —— 生成型 build 依赖，入 git（非 cache 快照）
被 macro_micro_context._build_sector_map 读取，为 CN 候选提供子赛道（→ sector_earnings_growth_mean）。

用法：python -m stock_ana.research.top_reversal.build_cn_industry_map
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # src/ 上 path（支持直接执行）

import pandas as pd  # noqa: E402

from stock_ana.config import DATA_DIR  # noqa: E402

OUT = DATA_DIR / "cn_industry_map.csv"
REQUEST_INTERVAL = 3.2  # Futu get_plate_stock 频率限制：每 30s ≤10 次


def main() -> None:
    from futu import Market, OpenQuoteContext, Plate, RET_OK  # type: ignore[import]

    ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    rows: list[dict] = []
    try:
        ret, plates = ctx.get_plate_list(Market.SH, Plate.INDUSTRY)
        if ret != RET_OK:
            print(f"[CN] get_plate_list 失败: {plates}", flush=True)
            return
        print(f"[CN] 行业板块 {len(plates)} 个，逐个拉成分股…", flush=True)
        for i, (_, p) in enumerate(plates.iterrows(), 1):
            code, name = p["code"], p["plate_name"]
            ret, data = ctx.get_plate_stock(code)
            if ret != RET_OK:
                print(f"  [warn] {name}({code}) 成分股失败: {data}", flush=True)
                time.sleep(REQUEST_INTERVAL)
                continue
            for _, r in data.iterrows():
                rows.append({"futu_code": r["code"], "industry": name})
            if i % 20 == 0:
                print(f"  {i}/{len(plates)} 累计 {len(rows)} 行", flush=True)
            time.sleep(REQUEST_INTERVAL)
    finally:
        ctx.close()

    df = pd.DataFrame(rows).drop_duplicates(subset="futu_code", keep="first")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"[CN] 完成 {len(df)} 只 -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
