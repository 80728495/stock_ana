#!/usr/bin/env python3
"""Combine per-account Futu deal exports into one audited history file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRADES_DIR = PROJECT_ROOT / "data" / "trades"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine Futu deal CSVs and recompute merged fills.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            TRADES_DIR / "futu_deals_2024_2025_acc3523_raw.csv",
            TRADES_DIR / "futu_deals_2024_2025_acc4752_hk_raw.csv",
            TRADES_DIR / "futu_deals_2024_2025_acc2412_us_raw.csv",
        ],
        help="Input raw CSV files.",
    )
    parser.add_argument("--prefix", default="futu_deals_2024_2025_all_accounts", help="Output file prefix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames: list[pd.DataFrame] = []
    for path in args.inputs:
        df = pd.read_csv(path, dtype={"deal_id": str, "order_id": str, "code": str})
        if "account_label" not in df.columns:
            df["account_label"] = path.stem
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw = raw.drop_duplicates(subset=["deal_id"]).copy()
    raw["create_time_sort"] = pd.to_datetime(raw["create_time"], errors="coerce")
    raw = raw.sort_values(["deal_market", "code", "create_time_sort", "deal_id"]).drop(columns=["create_time_sort"])

    from export_futu_deals import export_outputs, merge_same_order_price

    merged = merge_same_order_price(raw)
    export_args = SimpleNamespace(out_dir=TRADES_DIR, prefix=args.prefix, xlsx=True)

    export_outputs(raw, merged, export_args)

    print(f"combined raw rows: {len(raw)}")
    print(f"combined merged rows: {len(merged)}")
    print(raw.groupby(["account_label", "deal_market", "trd_side"]).size().to_string())


if __name__ == "__main__":
    main()
