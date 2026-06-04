#!/usr/bin/env python3
"""Calculate 2025 realized gains from exported Futu deal history.

This uses historical deals only. It does not read current holdings or watchlists.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "trades" / "futu_deals_2024_2025_raw.csv"
OUT_DIR = PROJECT_ROOT / "data" / "tax"


@dataclass
class Allocation:
    method: str
    asset_key: str
    market: str
    currency: str
    code: str
    stock_name: str
    sell_deal_id: str
    sell_order_id: str
    sell_time: str
    sell_price: float
    sell_qty_total: float
    allocated_qty: float
    proceeds: float
    buy_deal_id: str
    buy_order_id: str
    buy_time: str
    buy_price: float | None
    cost_basis: float | None
    realized_gain: float | None
    holding_days: int | None
    term: str
    uncovered: bool
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate 2025 realized gains from Futu deals.")
    parser.add_argument("--raw", type=Path, default=RAW_PATH, help="Raw Futu deals CSV.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory.")
    parser.add_argument("--tax-year", type=int, default=2025, help="Tax year to calculate. Default: 2025.")
    return parser.parse_args()


def load_deals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"deal_id": str, "order_id": str, "code": str})
    df["dt"] = pd.to_datetime(df["create_time"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["amount"] = df["qty"] * df["price"]
    df["market"] = df["deal_market"].fillna(df.get("query_market", ""))
    df["asset_key"] = df["market"].astype(str) + "." + df["code"].astype(str)
    df["currency"] = df["market"].map({"HK": "HKD", "US": "USD"}).fillna("")
    return df.sort_values(["asset_key", "dt", "deal_id"]).reset_index(drop=True)


def clean_float(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), 6)


def make_lot(row: pd.Series) -> dict:
    return {
        "deal_id": str(row["deal_id"]),
        "order_id": str(row["order_id"]),
        "time": row["dt"],
        "price": float(row["price"]),
        "qty_remaining": float(row["qty"]),
        "stock_name": str(row.get("stock_name", "")),
    }


def pop_lot(lots: list[dict] | deque[dict], method: str) -> dict | None:
    if not lots:
        return None
    if method == "FIFO":
        return lots.popleft() if isinstance(lots, deque) else lots.pop(0)

    best_idx = max(
        range(len(lots)),
        key=lambda idx: (float(lots[idx]["price"]), -pd.Timestamp(lots[idx]["time"]).timestamp()),
    )
    return lots.pop(best_idx)


def push_back_lot(lots: list[dict] | deque[dict], lot: dict, method: str) -> None:
    if method == "FIFO":
        lots.appendleft(lot) if isinstance(lots, deque) else lots.insert(0, lot)
    else:
        lots.append(lot)


def allocate_long_sales(
    df: pd.DataFrame,
    method: str,
    tax_year: int,
    prior_year_method: str | None = None,
    tax_year_method: str | None = None,
) -> pd.DataFrame:
    allocations: list[Allocation] = []
    prior_year_method = prior_year_method or method
    tax_year_method = tax_year_method or method
    for asset_key, group in df.groupby("asset_key", sort=True):
        lot_store: list[dict] = []

        for _, row in group.sort_values(["dt", "deal_id"]).iterrows():
            side = str(row["trd_side"])
            if side == "BUY":
                lot_store.append(make_lot(row))
                continue
            if side != "SELL":
                continue

            qty_to_allocate = float(row["qty"])
            active_method = tax_year_method if row["dt"].year == tax_year else prior_year_method
            while qty_to_allocate > 1e-9:
                lot = pop_lot(lot_store, active_method)
                if lot is None:
                    if row["dt"].year == tax_year:
                        proceeds = qty_to_allocate * float(row["price"])
                        allocations.append(
                            Allocation(
                                method=method,
                                asset_key=asset_key,
                                market=str(row["market"]),
                                currency=str(row["currency"]),
                                code=str(row["code"]),
                                stock_name=str(row.get("stock_name", "")),
                                sell_deal_id=str(row["deal_id"]),
                                sell_order_id=str(row["order_id"]),
                                sell_time=str(row["dt"]),
                                sell_price=float(row["price"]),
                                sell_qty_total=float(row["qty"]),
                                allocated_qty=qty_to_allocate,
                                proceeds=clean_float(proceeds) or 0.0,
                                buy_deal_id="",
                                buy_order_id="",
                                buy_time="",
                                buy_price=None,
                                cost_basis=None,
                                realized_gain=None,
                                holding_days=None,
                                term="Unknown",
                                uncovered=True,
                                note=(
                                    "No prior buy lot in exported history; "
                                    "basis must be supplied from pre-2024 records."
                                ),
                            )
                        )
                    break

                alloc_qty = min(qty_to_allocate, float(lot["qty_remaining"]))
                proceeds = alloc_qty * float(row["price"])
                cost_basis = alloc_qty * float(lot["price"])
                gain = proceeds - cost_basis
                holding_days = (row["dt"] - lot["time"]).days
                term = "Long-term" if holding_days > 365 else "Short-term"
                if row["dt"].year == tax_year:
                    allocations.append(
                        Allocation(
                            method=method,
                            asset_key=asset_key,
                            market=str(row["market"]),
                            currency=str(row["currency"]),
                            code=str(row["code"]),
                            stock_name=str(row.get("stock_name", "")),
                            sell_deal_id=str(row["deal_id"]),
                            sell_order_id=str(row["order_id"]),
                            sell_time=str(row["dt"]),
                            sell_price=float(row["price"]),
                            sell_qty_total=float(row["qty"]),
                            allocated_qty=clean_float(alloc_qty) or 0.0,
                            proceeds=clean_float(proceeds) or 0.0,
                            buy_deal_id=str(lot["deal_id"]),
                            buy_order_id=str(lot["order_id"]),
                            buy_time=str(lot["time"]),
                            buy_price=clean_float(float(lot["price"])),
                            cost_basis=clean_float(cost_basis),
                            realized_gain=clean_float(gain),
                            holding_days=int(holding_days),
                            term=term,
                            uncovered=False,
                            note="",
                        )
                    )

                lot["qty_remaining"] = float(lot["qty_remaining"]) - alloc_qty
                qty_to_allocate -= alloc_qty
                if lot["qty_remaining"] > 1e-9:
                    push_back_lot(lot_store, lot, active_method)

    return pd.DataFrame(asdict(row) for row in allocations)


def summarize_allocations(alloc: pd.DataFrame) -> pd.DataFrame:
    covered = alloc[~alloc["uncovered"]].copy()
    if covered.empty:
        return pd.DataFrame()
    return (
        covered.groupby(["method", "currency", "market", "code", "stock_name", "term"], dropna=False)
        .agg(
            sell_qty=("allocated_qty", "sum"),
            proceeds=("proceeds", "sum"),
            cost_basis=("cost_basis", "sum"),
            realized_gain=("realized_gain", "sum"),
            lots_used=("buy_deal_id", "count"),
        )
        .reset_index()
        .sort_values(["method", "currency", "market", "code", "term"])
    )


def summarize_methods(alloc: pd.DataFrame) -> pd.DataFrame:
    covered = alloc[~alloc["uncovered"]].copy()
    uncovered = alloc[alloc["uncovered"]].copy()
    rows: list[dict] = []
    for method in sorted(alloc["method"].unique()):
        for currency in sorted(alloc["currency"].dropna().unique()):
            method_covered = covered[(covered["method"] == method) & (covered["currency"] == currency)]
            method_uncovered = uncovered[(uncovered["method"] == method) & (uncovered["currency"] == currency)]
            for term in ["Short-term", "Long-term"]:
                term_df = method_covered[method_covered["term"] == term]
                rows.append(
                    {
                        "method": method,
                        "currency": currency,
                        "term": term,
                        "proceeds": term_df["proceeds"].sum(),
                        "cost_basis": term_df["cost_basis"].sum(),
                        "realized_gain": term_df["realized_gain"].sum(),
                        "uncovered_proceeds": 0.0,
                        "uncovered_qty": 0.0,
                    }
                )
            rows.append(
                {
                    "method": method,
                    "currency": currency,
                    "term": "Unknown basis",
                    "proceeds": 0.0,
                    "cost_basis": 0.0,
                    "realized_gain": 0.0,
                    "uncovered_proceeds": method_uncovered["proceeds"].sum() if not method_uncovered.empty else 0.0,
                    "uncovered_qty": method_uncovered["allocated_qty"].sum() if not method_uncovered.empty else 0.0,
                }
            )
            rows.append(
                {
                    "method": method,
                    "currency": currency,
                    "term": "Covered total",
                    "proceeds": method_covered["proceeds"].sum(),
                    "cost_basis": method_covered["cost_basis"].sum(),
                    "realized_gain": method_covered["realized_gain"].sum(),
                    "uncovered_proceeds": method_uncovered["proceeds"].sum() if not method_uncovered.empty else 0.0,
                    "uncovered_qty": method_uncovered["allocated_qty"].sum() if not method_uncovered.empty else 0.0,
                }
            )
    return pd.DataFrame(rows)


def allocate_short_closures(df: pd.DataFrame, tax_year: int) -> pd.DataFrame:
    rows: list[dict] = []
    for asset_key, group in df.groupby("asset_key", sort=True):
        shorts: deque[dict] = deque()
        for _, row in group.sort_values(["dt", "deal_id"]).iterrows():
            side = str(row["trd_side"])
            if side == "SELL_SHORT":
                shorts.append(make_lot(row))
                continue
            if side != "BUY_BACK":
                continue
            qty_to_cover = float(row["qty"])
            while qty_to_cover > 1e-9:
                if not shorts:
                    if row["dt"].year == tax_year:
                        rows.append(
                            {
                                "asset_key": asset_key,
                                "market": str(row["market"]),
                                "currency": str(row["currency"]),
                                "code": str(row["code"]),
                                "stock_name": str(row.get("stock_name", "")),
                                "cover_deal_id": str(row["deal_id"]),
                                "cover_time": str(row["dt"]),
                                "cover_qty": qty_to_cover,
                                "short_deal_id": "",
                                "short_time": "",
                                "short_price": None,
                                "cover_price": float(row["price"]),
                                "short_proceeds": None,
                                "cover_cost": qty_to_cover * float(row["price"]),
                                "realized_gain": None,
                                "term": "Short-term",
                                "uncovered": True,
                                "note": "No matching SELL_SHORT found in exported history.",
                            }
                        )
                    break
                short = shorts.popleft()
                alloc_qty = min(qty_to_cover, float(short["qty_remaining"]))
                gain = alloc_qty * (float(short["price"]) - float(row["price"]))
                if row["dt"].year == tax_year:
                    rows.append(
                        {
                            "asset_key": asset_key,
                            "market": str(row["market"]),
                            "currency": str(row["currency"]),
                            "code": str(row["code"]),
                            "stock_name": str(row.get("stock_name", "")),
                            "cover_deal_id": str(row["deal_id"]),
                            "cover_time": str(row["dt"]),
                            "cover_qty": alloc_qty,
                            "short_deal_id": str(short["deal_id"]),
                            "short_time": str(short["time"]),
                            "short_price": float(short["price"]),
                            "cover_price": float(row["price"]),
                            "short_proceeds": alloc_qty * float(short["price"]),
                            "cover_cost": alloc_qty * float(row["price"]),
                            "realized_gain": gain,
                            "term": "Short-term",
                            "uncovered": False,
                            "note": "Option/short closure; not part of long stock lot optimization.",
                        }
                    )
                short["qty_remaining"] = float(short["qty_remaining"]) - alloc_qty
                qty_to_cover -= alloc_qty
                if short["qty_remaining"] > 1e-9:
                    shorts.appendleft(short)
    return pd.DataFrame(rows)


def records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    clean = df.astype(object).where(pd.notna(df), None)
    return clean.to_dict(orient="records")


def detect_wash_sale_candidates(df: pd.DataFrame, hifo_alloc: pd.DataFrame, tax_year: int) -> pd.DataFrame:
    covered_losses = hifo_alloc[
        (~hifo_alloc["uncovered"])
        & (hifo_alloc["realized_gain"] < 0)
        & (pd.to_datetime(hifo_alloc["sell_time"], format="mixed").dt.year == tax_year)
    ].copy()
    buys = df[df["trd_side"] == "BUY"].copy()
    rows: list[dict] = []
    for _, loss in covered_losses.iterrows():
        sell_time = pd.to_datetime(loss["sell_time"], format="mixed")
        start = sell_time - pd.Timedelta(days=30)
        end = sell_time + pd.Timedelta(days=30)
        window_buys = buys[(buys["asset_key"] == loss["asset_key"]) & (buys["dt"] >= start) & (buys["dt"] <= end)]
        if window_buys.empty:
            continue
        rows.append(
            {
                "asset_key": loss["asset_key"],
                "sell_deal_id": loss["sell_deal_id"],
                "sell_time": loss["sell_time"],
                "loss_amount": loss["realized_gain"],
                "loss_qty": loss["allocated_qty"],
                "replacement_buy_qty_61d_window": window_buys["qty"].sum(),
                "first_replacement_buy": window_buys["dt"].min(),
                "last_replacement_buy": window_buys["dt"].max(),
                "note": (
                    "Candidate only; final wash-sale adjustment needs full account history "
                    "and substantially-identical security review."
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    deals = load_deals(args.raw)
    fifo_alloc = allocate_long_sales(deals, "FIFO", args.tax_year)
    hifo_alloc = allocate_long_sales(deals, "HIFO", args.tax_year)
    fifo_prior_hifo_alloc = allocate_long_sales(
        deals,
        "FIFO_PRIOR_HIFO_2025",
        args.tax_year,
        prior_year_method="FIFO",
        tax_year_method="HIFO",
    )
    allocations = pd.concat([fifo_alloc, hifo_alloc, fifo_prior_hifo_alloc], ignore_index=True)
    by_symbol = summarize_allocations(allocations)
    by_method = summarize_methods(allocations)
    short_closures = allocate_short_closures(deals, args.tax_year)
    wash = detect_wash_sale_candidates(deals, fifo_prior_hifo_alloc, args.tax_year)

    allocations.to_csv(args.out_dir / "tax_2025_lot_allocations.csv", index=False, encoding="utf-8-sig")
    by_symbol.to_csv(args.out_dir / "tax_2025_symbol_summary.csv", index=False, encoding="utf-8-sig")
    by_method.to_csv(args.out_dir / "tax_2025_method_summary.csv", index=False, encoding="utf-8-sig")
    short_closures.to_csv(args.out_dir / "tax_2025_short_closures.csv", index=False, encoding="utf-8-sig")
    wash.to_csv(args.out_dir / "tax_2025_wash_sale_candidates.csv", index=False, encoding="utf-8-sig")

    if short_closures.empty:
        short_summary: list[dict] = []
    else:
        short_summary = (
            short_closures[~short_closures["uncovered"]]
            .groupby(["currency", "term"], dropna=False)
            .agg(
                short_proceeds=("short_proceeds", "sum"),
                cover_cost=("cover_cost", "sum"),
                realized_gain=("realized_gain", "sum"),
                rows=("cover_deal_id", "count"),
            )
            .reset_index()
        )
        short_summary = records(short_summary)

    summary = {
        "source": str(args.raw),
        "tax_year": args.tax_year,
        "methods": records(by_method),
        "short_closures": short_summary,
        "wash_sale_candidate_count": int(len(wash)),
        "allocation_rows": int(len(allocations)),
    }
    (args.out_dir / "tax_2025_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    workbook_data = {
        "summary": summary,
        "method_summary": records(by_method),
        "symbol_summary": records(by_symbol),
        "lot_allocations": records(allocations),
        "uncovered_basis": records(allocations[allocations["uncovered"]]),
        "wash_sale_candidates": records(wash),
        "short_closures": records(short_closures),
    }
    (args.out_dir / "tax_2025_workbook_data.json").write_text(
        json.dumps(workbook_data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
