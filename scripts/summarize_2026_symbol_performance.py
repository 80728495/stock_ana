#!/usr/bin/env python3
"""Create market/symbol performance summaries for 2026 YTD."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERF_DIR = PROJECT_ROOT / "data" / "performance"


def split_code(code: str) -> str:
    text = str(code)
    return text.split(".", 1)[1] if "." in text else text


def fmt(value: float, currency: str = "") -> str:
    return f"{value:,.2f} {currency}".strip()


def main() -> None:
    realized = pd.read_csv(PERF_DIR / "performance_2026_realized_detail.csv")
    positions = pd.read_csv(PERF_DIR / "performance_2026_positions_detail.csv")
    short_closures = pd.read_csv(PERF_DIR / "performance_2026_short_closures.csv")

    realized["symbol"] = realized["code"].map(split_code)
    realized["realized_gain"] = pd.to_numeric(realized["realized_gain"], errors="coerce").fillna(0.0)
    realized_summary = (
        realized.groupby(["market", "currency", "symbol", "stock_name"], dropna=False)
        .agg(
            realized_profit=("realized_gain", lambda s: s[s > 0].sum()),
            realized_loss=("realized_gain", lambda s: s[s < 0].sum()),
            realized_net=("realized_gain", "sum"),
            realized_rows=("sell_deal_id", "count"),
        )
        .reset_index()
    )

    positions["market"] = positions["query_market"].fillna(positions.get("position_market", ""))
    positions["symbol"] = positions["code"].map(split_code)
    positions["unrealized_pl"] = pd.to_numeric(positions["unrealized_pl"], errors="coerce").fillna(0.0)
    positions["market_val"] = pd.to_numeric(positions["market_val"], errors="coerce").fillna(0.0)
    positions["qty"] = pd.to_numeric(positions["qty"], errors="coerce").fillna(0.0)
    position_summary = (
        positions.groupby(["market", "currency", "symbol", "stock_name"], dropna=False)
        .agg(
            floating_profit=("unrealized_pl", lambda s: s[s > 0].sum()),
            floating_loss=("unrealized_pl", lambda s: s[s < 0].sum()),
            floating_net=("unrealized_pl", "sum"),
            position_qty=("qty", "sum"),
            market_value=("market_val", "sum"),
        )
        .reset_index()
    )

    combined = pd.merge(
        realized_summary,
        position_summary,
        on=["market", "currency", "symbol", "stock_name"],
        how="outer",
    ).fillna(0)

    if not short_closures.empty:
        short_closures["symbol"] = short_closures["code"].map(split_code)
        short_closures["realized_gain"] = pd.to_numeric(short_closures["realized_gain"], errors="coerce").fillna(0.0)
        short_summary = (
            short_closures.groupby(["market", "currency", "symbol", "stock_name"], dropna=False)
            .agg(
                short_option_profit=("realized_gain", lambda s: s[s > 0].sum()),
                short_option_loss=("realized_gain", lambda s: s[s < 0].sum()),
                short_option_net=("realized_gain", "sum"),
            )
            .reset_index()
        )
        combined = pd.merge(
            combined,
            short_summary,
            on=["market", "currency", "symbol", "stock_name"],
            how="outer",
        ).fillna(0)
    else:
        combined["short_option_profit"] = 0.0
        combined["short_option_loss"] = 0.0
        combined["short_option_net"] = 0.0

    numeric_cols = [
        "realized_profit",
        "realized_loss",
        "realized_net",
        "floating_profit",
        "floating_loss",
        "floating_net",
        "short_option_profit",
        "short_option_loss",
        "short_option_net",
        "position_qty",
        "market_value",
    ]
    for col in numeric_cols:
        if col not in combined.columns:
            combined[col] = 0.0
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0)

    combined["total_profit_side"] = (
        combined["realized_profit"] + combined["floating_profit"] + combined["short_option_profit"]
    )
    combined["total_loss_side"] = combined["realized_loss"] + combined["floating_loss"] + combined["short_option_loss"]
    combined["total_net"] = combined["realized_net"] + combined["floating_net"] + combined["short_option_net"]
    combined = combined.sort_values(["market", "total_net"], ascending=[True, False])

    market_summary = (
        combined.groupby(["market", "currency"], dropna=False)
        .agg(
            total_profit_side=("total_profit_side", "sum"),
            total_loss_side=("total_loss_side", "sum"),
            realized_net=("realized_net", "sum"),
            floating_net=("floating_net", "sum"),
            short_option_net=("short_option_net", "sum"),
            total_net=("total_net", "sum"),
            symbols=("symbol", "count"),
        )
        .reset_index()
    )

    combined.to_csv(PERF_DIR / "performance_2026_symbol_summary.csv", index=False, encoding="utf-8-sig")
    market_summary.to_csv(PERF_DIR / "performance_2026_market_symbol_summary.csv", index=False, encoding="utf-8-sig")

    lines = ["# 2026 YTD Market and Symbol Performance", ""]
    for _, market_row in market_summary.sort_values("market").iterrows():
        market = market_row["market"]
        currency = market_row["currency"]
        lines += [
            f"## {market} ({currency})",
            "",
            f"- 盈利 + 浮盈：{fmt(market_row['total_profit_side'], currency)}",
            f"- 亏损 + 浮亏：{fmt(market_row['total_loss_side'], currency)}",
            f"- 已实现净额：{fmt(market_row['realized_net'], currency)}",
            f"- 浮动净额：{fmt(market_row['floating_net'], currency)}",
            f"- 总净表现：{fmt(market_row['total_net'], currency)}",
            "",
            "### 标的净表现排序",
            "",
            "| 排名 | 标的 | 名称 | 已实现净额 | 浮动净额 | 合计净额 |",
            "|---:|---|---|---:|---:|---:|",
        ]
        subset = combined[combined["market"] == market].sort_values("total_net", ascending=False)
        for rank, (_, row) in enumerate(subset.iterrows(), start=1):
            lines.append(
                f"| {rank} | {row['symbol']} | {row['stock_name']} | "
                f"{fmt(row['realized_net'], currency)} | {fmt(row['floating_net'], currency)} | "
                f"{fmt(row['total_net'], currency)} |"
            )
        lines.append("")

    md_path = PERF_DIR / "performance_2026_symbol_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)
    print(market_summary.to_string(index=False))


if __name__ == "__main__":
    main()
