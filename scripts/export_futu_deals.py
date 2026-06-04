#!/usr/bin/env python3
"""Export and merge Futu OpenD historical stock deals.

The merged output keeps one row per order/price pair. If one order is filled
multiple times at the same price, those fills are combined by summing qty.
This script queries account deal history directly by market, without reading
current holdings or watchlists.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "trades"

OPEND_HOST = os.environ.get("FUTU_OPEND_HOST", "127.0.0.1")
OPEND_PORT = int(os.environ.get("FUTU_OPEND_PORT", "11111"))


@dataclass(frozen=True)
class DateWindow:
    start: datetime
    end: datetime

    @property
    def start_text(self) -> str:
        return self.start.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def end_text(self) -> str:
        return self.end.strftime("%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch 2024/2025 Futu OpenD historical deals and merge same-order same-price fills.",
    )
    parser.add_argument("--start", default="2024-01-01", help="Start date, inclusive. Default: 2024-01-01")
    parser.add_argument("--end", default="2025-12-31", help="End date, inclusive. Default: 2025-12-31")
    parser.add_argument(
        "--markets",
        nargs="+",
        default=["HK", "US"],
        choices=["HK", "US", "CN"],
        help="Markets to fetch. Default: HK US",
    )
    parser.add_argument("--host", default=OPEND_HOST, help=f"OpenD host. Default: {OPEND_HOST}")
    parser.add_argument("--port", type=int, default=OPEND_PORT, help=f"OpenD port. Default: {OPEND_PORT}")
    parser.add_argument(
        "--security-firm",
        default="FUTUSECURITIES",
        help="Futu SecurityFirm enum name. Default: FUTUSECURITIES",
    )
    parser.add_argument("--acc-id", type=int, default=0, help="Trading account ID. 0 means use acc-index.")
    parser.add_argument("--acc-index", type=int, default=0, help="Trading account index when acc-id is 0.")
    parser.add_argument(
        "--account-label",
        default="",
        help="Audit label written to output rows, for example 3523.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=89,
        help="Query window size in days. Keep <= 90 for OpenD history APIs. Default: 89",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument(
        "--prefix",
        default="futu_deals_2024_2025",
        help="Output file prefix. Default: futu_deals_2024_2025",
    )
    parser.add_argument(
        "--xlsx",
        action="store_true",
        help="Also export an XLSX workbook if an Excel writer is installed.",
    )
    return parser.parse_args()


def iter_windows(start_text: str, end_text: str, window_days: int) -> Iterable[DateWindow]:
    start_day = datetime.strptime(start_text, "%Y-%m-%d").date()
    end_day = datetime.strptime(end_text, "%Y-%m-%d").date()
    if end_day < start_day:
        raise ValueError("--end must be greater than or equal to --start")
    if window_days < 1 or window_days > 90:
        raise ValueError("--window-days must be between 1 and 90")

    current = start_day
    while current <= end_day:
        window_end = min(current + timedelta(days=window_days), end_day)
        yield DateWindow(
            start=datetime.combine(current, datetime.min.time()),
            end=datetime.combine(window_end, datetime.max.time()).replace(microsecond=999000),
        )
        current = window_end + timedelta(days=1)


def import_futu_enums():
    try:
        from futu import RET_OK, OpenSecTradeContext, SecurityFirm, TrdEnv, TrdMarket
    except ImportError as exc:
        raise SystemExit("futu-api is not installed. Run: pip install futu-api") from exc

    return OpenSecTradeContext, RET_OK, SecurityFirm, TrdEnv, TrdMarket


def market_enum(name: str, trd_market):
    mapping = {
        "HK": trd_market.HK,
        "US": trd_market.US,
        "CN": trd_market.CN,
    }
    return mapping[name]


def normalize_enum_value(value) -> str:
    text = str(value)
    if "." in text:
        return text.rsplit(".", 1)[-1]
    return text


def infer_currency(market: str) -> str:
    return {"HK": "HKD", "US": "USD", "CN": "CNY", "SH": "CNY", "SZ": "CNY"}.get(market.upper(), "")


def fetch_market_deals(
    market_name: str,
    args: argparse.Namespace,
    windows: list[DateWindow],
    rate_state: dict[str, int],
) -> pd.DataFrame:
    OpenSecTradeContext, RET_OK, SecurityFirm, TrdEnv, TrdMarket = import_futu_enums()
    try:
        security_firm = getattr(SecurityFirm, args.security_firm)
    except AttributeError as exc:
        valid = [name for name in dir(SecurityFirm) if name.isupper()]
        valid_examples = ", ".join(valid[:12])
        raise SystemExit(f"Unknown SecurityFirm: {args.security_firm}. Valid examples: {valid_examples}") from exc

    market = market_enum(market_name, TrdMarket)
    ctx = OpenSecTradeContext(
        filter_trdmarket=market,
        host=args.host,
        port=args.port,
        security_firm=security_firm,
    )
    frames: list[pd.DataFrame] = []
    try:
        for window in windows:
            if rate_state["requests"] > 0 and rate_state["requests"] % 9 == 0:
                print("  reach OpenD trade history rate limit guard, sleep 31s ...")
                time.sleep(31)

            print(f"[{market_name}] {window.start_text} -> {window.end_text}")
            for attempt in range(2):
                ret, data = ctx.history_deal_list_query(
                    deal_market=market,
                    start=window.start_text,
                    end=window.end_text,
                    trd_env=TrdEnv.REAL,
                    acc_id=args.acc_id,
                    acc_index=args.acc_index,
                )
                rate_state["requests"] += 1
                if ret == RET_OK:
                    break

                msg = str(data)
                if attempt == 0 and ("频率" in msg or "frequency" in msg.lower()):
                    print("  rate limited by OpenD, sleep 31s then retry ...")
                    time.sleep(31)
                    continue
                break

            if ret != RET_OK:
                raise RuntimeError(f"history_deal_list_query failed for {market_name}: {data}")
            if data is not None and not data.empty:
                frame = data.copy()
                frame["query_market"] = market_name
                frame["account_id"] = args.acc_id
                frame["account_index"] = args.acc_index
                frame["account_label"] = args.account_label or str(args.acc_id or args.acc_index)
                frames.append(frame)
                print(f"  rows: {len(frame)}")
    finally:
        ctx.close()

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_all_deals(args: argparse.Namespace) -> pd.DataFrame:
    windows = list(iter_windows(args.start, args.end, args.window_days))
    rate_state = {"requests": 0}
    frames = [fetch_market_deals(market, args, windows, rate_state) for market in args.markets]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    if "deal_id" in raw.columns:
        raw = raw.drop_duplicates(subset=["deal_id"])
    return raw


def prepare_raw(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    df = raw.copy()
    for column in ["code", "stock_name", "deal_id", "order_id"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].map(lambda value: "" if pd.isna(value) else str(value))

    for column in ["deal_market", "trd_side", "status", "query_market"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].map(lambda value: "" if pd.isna(value) else normalize_enum_value(value))

    df["qty"] = pd.to_numeric(df.get("qty"), errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce").fillna(0.0)
    df["amount"] = df["qty"] * df["price"]
    df["create_time"] = pd.to_datetime(df.get("create_time"), errors="coerce")
    df["market"] = df["deal_market"].replace("", pd.NA).fillna(df["query_market"])
    df["currency"] = df["market"].map(infer_currency)
    df["symbol"] = df["code"].str.split(".", n=1).str[-1]
    df = df.sort_values(["code", "create_time", "order_id", "price", "deal_id"], na_position="last")
    return df


def join_unique(values: pd.Series) -> str:
    seen: list[str] = []
    for value in values:
        text = "" if pd.isna(value) else str(value)
        if text and text not in seen:
            seen.append(text)
    return ";".join(seen)


def merge_same_order_price(raw: pd.DataFrame) -> pd.DataFrame:
    df = prepare_raw(raw)
    if df.empty:
        return df

    group_cols = ["code", "stock_name", "market", "symbol", "trd_side", "order_id", "price", "currency"]
    merged = (
        df.groupby(group_cols, dropna=False)
        .agg(
            qty=("qty", "sum"),
            amount=("amount", "sum"),
            first_time=("create_time", "min"),
            last_time=("create_time", "max"),
            deal_count=("deal_id", "count"),
            deal_ids=("deal_id", join_unique),
            status=("status", join_unique),
            source_markets=("query_market", join_unique),
        )
        .reset_index()
    )
    merged["avg_price"] = merged.apply(
        lambda row: row["amount"] / row["qty"] if row["qty"] else row["price"],
        axis=1,
    )
    merged = merged.sort_values(["code", "first_time", "trd_side", "order_id", "price"], na_position="last")

    columns = [
        "code",
        "stock_name",
        "market",
        "symbol",
        "trd_side",
        "order_id",
        "price",
        "qty",
        "amount",
        "avg_price",
        "currency",
        "first_time",
        "last_time",
        "deal_count",
        "deal_ids",
        "status",
        "source_markets",
    ]
    return merged[columns]


def export_outputs(raw: pd.DataFrame, merged: pd.DataFrame, args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / f"{args.prefix}_raw.csv"
    merged_path = args.out_dir / f"{args.prefix}_merged.csv"

    prepare_raw(raw).to_csv(raw_path, index=False, encoding="utf-8-sig")
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    print(f"raw csv:    {raw_path}")
    print(f"merged csv: {merged_path}")

    if args.xlsx:
        xlsx_path = args.out_dir / f"{args.prefix}.xlsx"
        try:
            with pd.ExcelWriter(xlsx_path) as writer:
                merged.to_excel(writer, sheet_name="merged", index=False)
                prepare_raw(raw).to_excel(writer, sheet_name="raw", index=False)
            print(f"xlsx:       {xlsx_path}")
        except Exception as exc:
            print(f"skip xlsx export: {exc}")


def main() -> None:
    args = parse_args()
    raw = fetch_all_deals(args)
    merged = merge_same_order_price(raw)
    export_outputs(raw, merged, args)
    print(f"raw rows: {len(raw)}")
    print(f"merged rows: {len(merged)}")


if __name__ == "__main__":
    main()
