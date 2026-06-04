#!/usr/bin/env python3
"""Calculate 2026 YTD performance from year-start market value."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEALS_PATH = PROJECT_ROOT / "data" / "trades" / "futu_deals_2024_2026_all_accounts_raw.csv"
POSITIONS_PATH = PROJECT_ROOT / "data" / "performance" / "current_positions_3523.csv"
OUT_DIR = PROJECT_ROOT / "data" / "performance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate 2026 YTD P/L from year-start market value.")
    parser.add_argument("--deals", type=Path, default=DEALS_PATH)
    parser.add_argument("--positions", type=Path, default=POSITIONS_PATH)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--start-price-date", default="2025-12-31")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def normalize_code(market: str, code: str) -> str:
    code = str(code).strip()
    if "." in code:
        return code
    return f"{market}.{code}"


def fetch_start_prices(codes: list[str], start_price_date: str) -> dict[str, dict]:
    from futu import RET_OK, AuType, KLType, OpenQuoteContext

    prices: dict[str, dict] = {}
    ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        for code in codes:
            ret, data, _ = ctx.request_history_kline(
                code,
                start="2025-12-20",
                end="2026-01-05",
                ktype=KLType.K_DAY,
                autype=AuType.NONE,
                max_count=100,
            )
            if ret != RET_OK or data is None or data.empty:
                prices[code] = {"start_price": None, "price_date": "", "note": str(data)}
                continue
            data = data.copy()
            data["date"] = pd.to_datetime(data["time_key"]).dt.date
            target = pd.to_datetime(start_price_date).date()
            before = data[data["date"] <= target]
            row = before.iloc[-1] if not before.empty else data.iloc[0]
            prices[code] = {
                "start_price": float(row["close"]),
                "price_date": str(row["date"]),
                "note": "last close on/before start date" if not before.empty else "first available after start date",
            }
    finally:
        ctx.close()
    return prices


def fmt(value: float, currency: str = "") -> str:
    return f"{value:,.2f} {currency}".strip()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    deals = pd.read_csv(args.deals, dtype={"deal_id": str, "order_id": str, "code": str})
    deals["dt"] = pd.to_datetime(deals["create_time"], errors="coerce")
    deals["market"] = deals["deal_market"].fillna(deals.get("query_market", ""))
    deals["norm_code"] = deals.apply(lambda r: normalize_code(r["market"], r["code"]), axis=1)
    deals["currency"] = deals["market"].map({"HK": "HKD", "US": "USD"}).fillna("")
    deals["qty"] = pd.to_numeric(deals["qty"], errors="coerce").fillna(0.0)
    deals["amount"] = pd.to_numeric(deals["amount"], errors="coerce").fillna(deals["qty"] * deals["price"])

    ytd = deals[deals["dt"].dt.year == args.year].copy()
    long_flows = (
        ytd[ytd["trd_side"].isin(["BUY", "SELL"])]
        .groupby(["market", "currency", "norm_code"], dropna=False)
        .agg(
            stock_name=("stock_name", "last"),
            buy_qty=("qty", lambda s: s[ytd.loc[s.index, "trd_side"] == "BUY"].sum()),
            buy_cost=("amount", lambda s: s[ytd.loc[s.index, "trd_side"] == "BUY"].sum()),
            sell_qty=("qty", lambda s: s[ytd.loc[s.index, "trd_side"] == "SELL"].sum()),
            sell_proceeds=("amount", lambda s: s[ytd.loc[s.index, "trd_side"] == "SELL"].sum()),
        )
        .reset_index()
    )

    positions = pd.read_csv(args.positions, dtype={"code": str})
    positions["market"] = positions["query_market"].fillna(positions.get("position_market", ""))
    positions["norm_code"] = positions.apply(lambda r: normalize_code(r["market"], r["code"]), axis=1)
    positions["currency"] = positions["currency"].fillna(positions["market"].map({"HK": "HKD", "US": "USD"}))
    positions["qty"] = pd.to_numeric(positions["qty"], errors="coerce").fillna(0.0)
    positions["market_val"] = pd.to_numeric(positions["market_val"], errors="coerce").fillna(0.0)
    pos_summary = (
        positions.groupby(["market", "currency", "norm_code"], dropna=False)
        .agg(
            stock_name_pos=("stock_name", "last"),
            current_qty=("qty", "sum"),
            current_market_value=("market_val", "sum"),
        )
        .reset_index()
    )

    combined = pd.merge(long_flows, pos_summary, on=["market", "currency", "norm_code"], how="outer").fillna(0)
    combined["stock_name"] = combined.apply(
        lambda r: r["stock_name"] if r["stock_name"] else r["stock_name_pos"],
        axis=1,
    )
    combined["begin_qty"] = combined["current_qty"] - combined["buy_qty"] + combined["sell_qty"]

    price_codes = sorted(combined.loc[combined["begin_qty"].abs() > 1e-9, "norm_code"].unique())
    prices = fetch_start_prices(price_codes, args.start_price_date)
    combined["start_price"] = combined["norm_code"].map(lambda c: prices.get(c, {}).get("start_price"))
    combined["start_price_date"] = combined["norm_code"].map(lambda c: prices.get(c, {}).get("price_date", ""))
    combined["price_note"] = combined["norm_code"].map(lambda c: prices.get(c, {}).get("note", ""))
    combined["begin_market_value"] = combined["begin_qty"] * pd.to_numeric(
        combined["start_price"], errors="coerce"
    ).fillna(0)
    combined["total_net"] = (
        combined["current_market_value"]
        + combined["sell_proceeds"]
        - combined["begin_market_value"]
        - combined["buy_cost"]
    )

    total_available_qty = combined["begin_qty"] + combined["buy_qty"]
    combined["avg_ytd_cost"] = (combined["begin_market_value"] + combined["buy_cost"]) / total_available_qty.replace(
        0, pd.NA
    )
    combined["realized_net_year_start"] = combined["sell_proceeds"] - combined["sell_qty"] * combined[
        "avg_ytd_cost"
    ].fillna(0)
    combined["floating_net_year_start"] = combined["current_market_value"] - combined["current_qty"] * combined[
        "avg_ytd_cost"
    ].fillna(0)
    combined["profit_side"] = combined["total_net"].clip(lower=0)
    combined["loss_side"] = combined["total_net"].clip(upper=0)
    combined["symbol"] = combined["norm_code"].str.split(".", n=1).str[-1]
    combined = combined.sort_values(["market", "total_net"], ascending=[True, False])

    market_summary = (
        combined.groupby(["market", "currency"], dropna=False)
        .agg(
            profit_side=("profit_side", "sum"),
            loss_side=("loss_side", "sum"),
            total_net=("total_net", "sum"),
            begin_market_value=("begin_market_value", "sum"),
            buy_cost=("buy_cost", "sum"),
            sell_proceeds=("sell_proceeds", "sum"),
            current_market_value=("current_market_value", "sum"),
            symbols=("symbol", "count"),
        )
        .reset_index()
    )

    combined.to_csv(
        args.out_dir / "performance_2026_ytd_year_start_symbol_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    market_summary.to_csv(
        args.out_dir / "performance_2026_ytd_year_start_market_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.out_dir / "performance_2026_ytd_start_prices.json").write_text(
        json.dumps(prices, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = ["# 2026 YTD Performance From Year-Start Value", ""]
    for _, row in market_summary.sort_values("market").iterrows():
        market = row["market"]
        currency = row["currency"]
        lines.extend(
            [
                f"## {market} ({currency})",
                "",
                f"- 盈利标的合计：{fmt(row['profit_side'], currency)}",
                f"- 亏损标的合计：{fmt(row['loss_side'], currency)}",
                f"- 总净表现：{fmt(row['total_net'], currency)}",
                "",
                "| 排名 | 标的 | 名称 | 年初数量 | 年初价格 | 年初市值 | 今年买入 | 今年卖出 | 当前市值 | 合计盈亏 |",
                "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        subset = combined[combined["market"] == market].sort_values("total_net", ascending=False)
        for rank, (_, item) in enumerate(subset.iterrows(), start=1):
            lines.append(
                f"| {rank} | {item['symbol']} | {item['stock_name']} | {item['begin_qty']:,.0f} | "
                f"{item['start_price'] if pd.notna(item['start_price']) else 'N/A'} | "
                f"{fmt(item['begin_market_value'], currency)} | {fmt(item['buy_cost'], currency)} | "
                f"{fmt(item['sell_proceeds'], currency)} | {fmt(item['current_market_value'], currency)} | "
                f"{fmt(item['total_net'], currency)} |"
            )
        lines.append("")

    md_path = args.out_dir / "performance_2026_ytd_year_start_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)
    print(market_summary.to_string(index=False))


if __name__ == "__main__":
    main()
