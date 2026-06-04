#!/usr/bin/env python3
"""Export Futu OpenD account cash-flow records."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "performance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Futu cash-flow records.")
    parser.add_argument("--acc-id", type=int, required=True)
    parser.add_argument("--account-label", default="")
    parser.add_argument("--markets", nargs="+", default=["HK", "US"], choices=["HK", "US"])
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-06-03")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "cashflow_2026.csv")
    return parser.parse_args()


def dates(start: str, end: str):
    current = datetime.strptime(start, "%Y-%m-%d").date()
    end_day = datetime.strptime(end, "%Y-%m-%d").date()
    while current <= end_day:
        yield current.strftime("%Y-%m-%d")
        current = current + timedelta(days=1)


def main() -> None:
    from futu import RET_OK, OpenSecTradeContext, SecurityFirm, TrdEnv, TrdMarket

    args = parse_args()
    market_map = {"HK": TrdMarket.HK, "US": TrdMarket.US}
    frames: list[pd.DataFrame] = []

    for market_name in args.markets:
        market = market_map[market_name]
        ctx = OpenSecTradeContext(
            filter_trdmarket=market,
            host="127.0.0.1",
            port=11111,
            security_firm=SecurityFirm.FUTUSECURITIES,
        )
        try:
            request_count = 0
            for clearing_date in dates(args.start, args.end):
                if request_count > 0 and request_count % 18 == 0:
                    print("  reach OpenD cashflow rate limit guard, sleep 31s ...")
                    time.sleep(31)
                print(f"[{market_name}] cashflow {clearing_date}")
                for attempt in range(2):
                    ret, data = ctx.get_acc_cash_flow(
                        clearing_date=clearing_date,
                        trd_env=TrdEnv.REAL,
                        acc_id=args.acc_id,
                    )
                    request_count += 1
                    if ret == RET_OK:
                        break
                    msg = str(data)
                    if attempt == 0 and ("频率" in msg or "frequency" in msg.lower()):
                        print("  rate limited by OpenD, sleep 31s then retry ...")
                        time.sleep(31)
                        continue
                    break
                if ret != RET_OK:
                    raise RuntimeError(data)
                if data is not None and not data.empty:
                    frame = data.copy()
                    frame["query_market"] = market_name
                    frame["account_id"] = args.acc_id
                    frame["account_label"] = args.account_label or str(args.acc_id)
                    frames.append(frame)
                    print(frame.to_string())
        finally:
            ctx.close()

    result = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"cashflow csv: {args.out}")
    print(f"rows: {len(result)}")
    if not result.empty:
        print(result.columns.tolist())


if __name__ == "__main__":
    main()
