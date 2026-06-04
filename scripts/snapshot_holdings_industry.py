#!/usr/bin/env python3
"""Snapshot current Futu positions and summarize by owner industry plate."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "holdings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snapshot current Futu positions and industry distribution.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--markets", nargs="+", default=["HK", "US"], choices=["HK", "US"])
    return parser.parse_args()


def enum_text(value) -> str:
    text = str(value)
    return text.rsplit(".", 1)[-1] if "." in text else text


def fetch_real_accounts(markets: list[str]) -> pd.DataFrame:
    from futu import RET_OK, OpenSecTradeContext, SecurityFirm, TrdMarket

    market_map = {"HK": TrdMarket.HK, "US": TrdMarket.US}
    frames: list[pd.DataFrame] = []
    for market_name in markets:
        ctx = OpenSecTradeContext(
            filter_trdmarket=market_map[market_name],
            host="127.0.0.1",
            port=11111,
            security_firm=SecurityFirm.FUTUSECURITIES,
        )
        try:
            ret, data = ctx.get_acc_list()
            if ret != RET_OK:
                raise RuntimeError(data)
            frame = data.copy()
            frame["query_market"] = market_name
            frames.append(frame)
        finally:
            ctx.close()

    accounts = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates(subset=["acc_id"])
    accounts["trd_env_text"] = accounts["trd_env"].map(enum_text)
    return accounts[accounts["trd_env_text"] == "REAL"].copy()


def fetch_positions(accounts: pd.DataFrame, markets: list[str]) -> pd.DataFrame:
    from futu import RET_OK, OpenSecTradeContext, SecurityFirm, TrdEnv, TrdMarket

    market_map = {"HK": TrdMarket.HK, "US": TrdMarket.US}
    frames: list[pd.DataFrame] = []
    for _, account in accounts.iterrows():
        acc_id = int(account["acc_id"])
        card_num = str(account.get("card_num", ""))
        account_label = card_num[-4:] if card_num and card_num != "nan" else str(acc_id)
        for market_name in markets:
            ctx = OpenSecTradeContext(
                filter_trdmarket=market_map[market_name],
                host="127.0.0.1",
                port=11111,
                security_firm=SecurityFirm.FUTUSECURITIES,
            )
            try:
                ret, data = ctx.position_list_query(
                    position_market=market_map[market_name],
                    trd_env=TrdEnv.REAL,
                    acc_id=acc_id,
                    refresh_cache=True,
                )
                if ret != RET_OK:
                    print(f"[warn] position query failed acc={account_label} market={market_name}: {data}")
                    continue
                if data is None or data.empty:
                    continue
                frame = data.copy()
                frame["query_market"] = market_name
                frame["account_id"] = acc_id
                frame["account_label"] = account_label
                frames.append(frame)
            finally:
                ctx.close()

    if not frames:
        return pd.DataFrame()
    positions = pd.concat(frames, ignore_index=True, sort=False)
    qty = pd.to_numeric(positions.get("qty"), errors="coerce").fillna(0)
    return positions[qty != 0].copy()


def fetch_owner_plates(codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from futu import RET_OK, OpenQuoteContext

    if not codes:
        empty = pd.DataFrame(columns=["code", "industry", "all_industry_plates", "all_plates"])
        return empty, empty

    frames: list[pd.DataFrame] = []
    request_count = 0

    def wait_before_request() -> None:
        nonlocal request_count
        if request_count > 0 and request_count % 9 == 0:
            print("[info] get_owner_plate rate guard: sleep 31s")
            time.sleep(31)
        request_count += 1

    ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        for i in range(0, len(codes), 100):
            batch = codes[i : i + 100]
            wait_before_request()
            ret, data = ctx.get_owner_plate(batch)
            if ret == RET_OK and data is not None and not data.empty:
                frames.append(data.copy())
                continue

            print(f"[warn] get_owner_plate batch failed; retry individually: {data}")
            for code in batch:
                wait_before_request()
                ret_one, data_one = ctx.get_owner_plate([code])
                if ret_one != RET_OK and "frequency" in str(data_one).lower():
                    print(f"[info] get_owner_plate rate limited for {code}; sleep 31s and retry")
                    time.sleep(31)
                    wait_before_request()
                    ret_one, data_one = ctx.get_owner_plate([code])
                if ret_one != RET_OK:
                    print(f"[warn] get_owner_plate skipped {code}: {data_one}")
                    continue
                if data_one is not None and not data_one.empty:
                    frames.append(data_one.copy())
    finally:
        ctx.close()

    if not frames:
        empty = pd.DataFrame(columns=["code", "industry", "all_industry_plates", "all_plates"])
        return empty, pd.DataFrame()

    plates = pd.concat(frames, ignore_index=True, sort=False)
    plates["plate_type_text"] = plates["plate_type"].map(enum_text)
    plates["plate_name"] = plates["plate_name"].astype(str)
    plates["is_industry"] = plates["plate_type_text"].str.contains("INDUSTRY|Plate.INDUSTRY", case=False, regex=True)

    rows: list[dict] = []
    for code, group in plates.groupby("code", dropna=False):
        industry_names = list(dict.fromkeys(group.loc[group["is_industry"], "plate_name"].tolist()))
        all_names = list(dict.fromkeys(group["plate_name"].tolist()))
        rows.append(
            {
                "code": code,
                "industry": " / ".join(industry_names) if industry_names else "Unclassified/ETF/Other",
                "all_industry_plates": " / ".join(industry_names),
                "all_plates": " / ".join(all_names),
            }
        )
    return pd.DataFrame(rows), plates


def build_summaries(positions: pd.DataFrame, industry_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = positions.copy()
    df["market"] = df["query_market"].fillna(df.get("position_market", ""))
    df["currency"] = df["currency"].fillna(df["market"].map({"HK": "HKD", "US": "USD"}))
    for col in ["qty", "market_val", "unrealized_pl", "pl_val", "nominal_price", "cost_price", "average_cost"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df.merge(industry_map, on="code", how="left")
    df["industry"] = df["industry"].fillna("Unclassified/ETF/Other")
    if "unrealized_pl" not in df.columns and "pl_val" in df.columns:
        df["unrealized_pl"] = df["pl_val"]
    df["unrealized_pl"] = pd.to_numeric(df["unrealized_pl"], errors="coerce").fillna(0.0)
    df["position_weight_in_currency"] = df["market_val"] / df.groupby(["currency"])["market_val"].transform("sum")
    df["position_weight_in_market"] = df["market_val"] / df.groupby(["market", "currency"])["market_val"].transform(
        "sum"
    )

    industry_summary = (
        df.groupby(["market", "currency", "industry"], dropna=False)
        .agg(
            market_value=("market_val", "sum"),
            unrealized_pl=("unrealized_pl", "sum"),
            symbols=("code", "count"),
            names=("stock_name", lambda s: " / ".join(s.astype(str).tolist())),
        )
        .reset_index()
    )
    industry_summary["weight_in_market"] = industry_summary["market_value"] / industry_summary.groupby(
        ["market", "currency"]
    )["market_value"].transform("sum")
    industry_summary = industry_summary.sort_values(["market", "market_value"], ascending=[True, False])

    detail = df.sort_values(["market", "industry", "market_val"], ascending=[True, True, False])
    return detail, industry_summary


def write_markdown(positions: pd.DataFrame, industry_summary: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Current Holdings Industry Snapshot", ""]
    market_totals = (
        positions.groupby(["market", "currency"], dropna=False)
        .agg(market_value=("market_val", "sum"), unrealized_pl=("unrealized_pl", "sum"), symbols=("code", "count"))
        .reset_index()
    )
    for _, total in market_totals.sort_values("market").iterrows():
        market = total["market"]
        currency = total["currency"]
        lines.extend(
            [
                f"## {market} ({currency})",
                "",
                f"- Market value: {total['market_value']:,.2f} {currency}",
                f"- Unrealized P/L: {total['unrealized_pl']:,.2f} {currency}",
                f"- Symbols: {int(total['symbols'])}",
                "",
                "| Industry | Market value | Weight | Unrealized P/L | Symbols | Holdings |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        subset = industry_summary[industry_summary["market"] == market]
        for _, row in subset.iterrows():
            lines.append(
                f"| {row['industry']} | {row['market_value']:,.2f} {currency} | "
                f"{row['weight_in_market']:.2%} | {row['unrealized_pl']:,.2f} {currency} | "
                f"{int(row['symbols'])} | {row['names']} |"
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    accounts = fetch_real_accounts(args.markets)
    positions = fetch_positions(accounts, args.markets)
    if positions.empty:
        print("No current positions found.")
        return

    codes = sorted(positions["code"].dropna().unique().tolist())
    industry_map, raw_plates = fetch_owner_plates(codes)
    detailed, industry_summary = build_summaries(positions, industry_map)

    accounts.to_csv(args.out_dir / "futu_accounts_snapshot.csv", index=False, encoding="utf-8-sig")
    positions.to_csv(args.out_dir / "current_positions_full_snapshot_raw.csv", index=False, encoding="utf-8-sig")
    raw_plates.to_csv(args.out_dir / "current_positions_owner_plates_raw.csv", index=False, encoding="utf-8-sig")
    detailed.to_csv(args.out_dir / "current_positions_by_symbol_industry.csv", index=False, encoding="utf-8-sig")
    industry_summary.to_csv(args.out_dir / "current_positions_industry_summary.csv", index=False, encoding="utf-8-sig")
    write_markdown(detailed, industry_summary, args.out_dir / "current_positions_industry_summary.md")

    print((args.out_dir / "current_positions_industry_summary.md").resolve())
    print(industry_summary.to_string(index=False))


if __name__ == "__main__":
    main()
