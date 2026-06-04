#!/usr/bin/env python3
"""Export current Futu OpenD positions for HK/US performance reporting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "performance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export current Futu positions.")
    parser.add_argument("--acc-id", type=int, required=True)
    parser.add_argument("--account-label", default="")
    parser.add_argument("--markets", nargs="+", default=["HK", "US"], choices=["HK", "US"])
    parser.add_argument("--out", type=Path, default=OUT_DIR / "current_positions.csv")
    return parser.parse_args()


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
            ret, data = ctx.position_list_query(
                position_market=market,
                trd_env=TrdEnv.REAL,
                acc_id=args.acc_id,
                refresh_cache=True,
            )
            print(f"[{market_name}] ret={ret}")
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

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    result = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"positions csv: {out}")
    print(f"rows: {len(result)}")


if __name__ == "__main__":
    main()
