#!/usr/bin/env python3
"""Summarize 2026 realized + unrealized performance by HK/US market."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEALS = PROJECT_ROOT / "data" / "trades" / "futu_deals_2024_2026_all_accounts_raw.csv"
DEFAULT_POSITIONS = PROJECT_ROOT / "data" / "performance" / "current_positions_3523.csv"
OUT_DIR = PROJECT_ROOT / "data" / "performance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate current-year market performance by HK/US.")
    parser.add_argument("--deals", type=Path, default=DEFAULT_DEALS)
    parser.add_argument("--positions", type=Path, default=DEFAULT_POSITIONS)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def profit_loss_split(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return float(numeric[numeric > 0].sum()), float(numeric[numeric < 0].sum())


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from calc_2025_capital_gains import allocate_long_sales, allocate_short_closures, load_deals, records

    deals = load_deals(args.deals)
    positions = pd.read_csv(args.positions)
    positions["market"] = positions["query_market"].fillna(positions.get("position_market", ""))
    positions["currency"] = positions["currency"].fillna(positions["market"].map({"HK": "HKD", "US": "USD"}))
    positions["unrealized_pl"] = pd.to_numeric(positions["unrealized_pl"], errors="coerce").fillna(0.0)

    realized = allocate_long_sales(
        deals,
        f"FIFO_PRIOR_HIFO_{args.year}",
        args.year,
        prior_year_method="FIFO",
        tax_year_method="HIFO",
    )
    realized = realized[(~realized["uncovered"]) & (pd.to_numeric(realized["realized_gain"], errors="coerce").notna())]

    short_closures = allocate_short_closures(deals, args.year)
    if not short_closures.empty:
        short_closures = short_closures[~short_closures["uncovered"]].copy()

    rows: list[dict] = []
    for market, currency in [("HK", "HKD"), ("US", "USD")]:
        realized_mkt = realized[(realized["market"] == market) & (realized["currency"] == currency)]
        pos_mkt = positions[(positions["market"] == market) & (positions["currency"] == currency)]
        realized_profit, realized_loss = profit_loss_split(realized_mkt["realized_gain"])
        floating_profit, floating_loss = profit_loss_split(pos_mkt["unrealized_pl"])

        short_profit = 0.0
        short_loss = 0.0
        if market == "US" and not short_closures.empty:
            short_profit, short_loss = profit_loss_split(short_closures["realized_gain"])

        total_profit_side = realized_profit + short_profit + floating_profit
        total_loss_side = realized_loss + short_loss + floating_loss
        rows.append(
            {
                "market": market,
                "currency": currency,
                "realized_profit": realized_profit,
                "realized_loss": realized_loss,
                "short_option_profit": short_profit,
                "short_option_loss": short_loss,
                "floating_profit": floating_profit,
                "floating_loss": floating_loss,
                "profit_plus_float_profit": total_profit_side,
                "loss_plus_float_loss": total_loss_side,
                "net": total_profit_side + total_loss_side,
                "position_count": int(len(pos_mkt)),
                "realized_allocation_rows": int(len(realized_mkt)),
            }
        )

    summary = pd.DataFrame(rows)
    realized_detail = realized.sort_values(["market", "code", "sell_time", "sell_deal_id"])
    positions_detail = positions.sort_values(["market", "code"])

    summary.to_csv(args.out_dir / "performance_2026_market_summary.csv", index=False, encoding="utf-8-sig")
    realized_detail.to_csv(args.out_dir / "performance_2026_realized_detail.csv", index=False, encoding="utf-8-sig")
    positions_detail.to_csv(args.out_dir / "performance_2026_positions_detail.csv", index=False, encoding="utf-8-sig")
    short_closures.to_csv(args.out_dir / "performance_2026_short_closures.csv", index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print(records(summary))


if __name__ == "__main__":
    main()
