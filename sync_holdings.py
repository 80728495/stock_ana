#!/usr/bin/env python3
"""sync_holdings.py — 每日同步 Futu OpenD 账户持仓到 watchlist.md

运行方式：
    python sync_holdings.py
    python sync_holdings.py --sim        # 使用模拟账户
    python sync_holdings.py --dry-run    # 只打印，不写文件

持仓写入 watchlist.md 的 "## 持仓 (Holdings)" 区段。
已有区段时替换，不存在时追加到文件末尾。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WATCHLIST_PATH = PROJECT_ROOT / "data" / "lists" / "watchlist.md"

OPEND_HOST = "127.0.0.1"
OPEND_PORT = 11111

SECTION_HEADER = "## 持仓 (Holdings)"


# ─── Futu API ──────────────────────────────────────────────────────────────────

def _fetch_market_holdings(
    market,
    trd_env,
    security_firm,
) -> list[dict]:
    """从单一市场的 TradeContext 获取持仓列表。"""
    from futu import OpenSecTradeContext, RET_OK  # type: ignore[import]

    ctx = OpenSecTradeContext(
        filter_trdmarket=market,
        host=OPEND_HOST,
        port=OPEND_PORT,
        security_firm=security_firm,
    )
    try:
        ret, data = ctx.position_list_query(trd_env=trd_env, refresh_cache=True)
        if ret != RET_OK:
            print(f"  ⚠️  持仓查询失败（{market}）：{data}")
            return []
        if data is None or (hasattr(data, "empty") and data.empty):
            return []

        results = []
        for _, row in data.iterrows():
            code_raw = str(row.get("code", ""))         # e.g. "HK.01810" or "US.NVDA"
            parts = code_raw.split(".", 1)
            mkt_tag = parts[0].upper() if len(parts) == 2 else ""
            symbol  = parts[1] if len(parts) == 2 else code_raw

            # 决定市场标签
            if mkt_tag in ("HK", "US", "CN", "SH", "SZ"):
                mkt_label = "CN" if mkt_tag in ("SH", "SZ") else mkt_tag
            else:
                # 5位数字 → HK，否则 US
                mkt_label = "HK" if symbol.isdigit() and len(symbol) <= 5 else "US"

            pl_ratio      = row.get("pl_ratio", None)
            pl_ratio_valid = bool(row.get("pl_ratio_valid", False))

            results.append({
                "symbol":    symbol.zfill(5) if mkt_label in ("HK", "CN") else symbol.upper(),
                "market":    mkt_label,
                "name":      str(row.get("stock_name", "")),
                "qty":       float(row.get("qty", 0)),
                "cost_price": float(row.get("cost_price", 0)),
                "pl_ratio":  float(pl_ratio) if pl_ratio_valid and pl_ratio is not None else None,
            })
        return results
    finally:
        ctx.close()


def fetch_holdings(use_sim: bool = False) -> list[dict]:
    """获取所有市场持仓。返回 list of dict。"""
    try:
        from futu import TrdMarket, TrdEnv, SecurityFirm  # type: ignore[import]
    except ImportError:
        print("❌ futu-api 未安装，请运行：pip install futu-api")
        sys.exit(1)

    trd_env = TrdEnv.SIMULATE if use_sim else TrdEnv.REAL
    firm    = SecurityFirm.FUTUSECURITIES

    all_holdings: list[dict] = []
    seen: set[str] = set()

    for mkt in (TrdMarket.HK, TrdMarket.US):
        rows = _fetch_market_holdings(mkt, trd_env, firm)
        for r in rows:
            key = f"{r['market']}:{r['symbol']}"
            if key not in seen:
                seen.add(key)
                all_holdings.append(r)

    return all_holdings


# ─── watchlist.md 写入 ────────────────────────────────────────────────────────

def _build_section(holdings: list[dict]) -> str:
    """生成持仓 Markdown 区段文本（含标题行）。"""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    env_tag = "真实账户"
    lines = [
        SECTION_HEADER,
        "",
        f"> 自动更新：{now_str} | 来源：Futu OpenD（{env_tag}）",
        "",
        "| 代码 | 市场 | 中文名 | 数量 | 成本价 | 盈亏% |",
        "|------|------|--------|------|--------|-------|",
    ]
    for h in sorted(holdings, key=lambda x: (x["market"], x["symbol"])):
        pl_str = f"{h['pl_ratio']:+.2f}%" if h["pl_ratio"] is not None else "-"
        lines.append(
            f"| {h['symbol']} | {h['market']} | {h['name']} "
            f"| {h['qty']:.0f} | {h['cost_price']:.3f} | {pl_str} |"
        )
    return "\n".join(lines)


def update_watchlist(holdings: list[dict], dry_run: bool = False) -> None:
    """用新的持仓数据更新 watchlist.md 中的 `## 持仓 (Holdings)` 区段。"""
    text = WATCHLIST_PATH.read_text(encoding="utf-8")
    new_section = _build_section(holdings)

    if SECTION_HEADER in text:
        # 替换已有区段：从标题到下一个 ## 或文件末尾
        start = text.index(SECTION_HEADER)
        rest  = text[start + len(SECTION_HEADER):]
        next_sec = rest.find("\n## ")
        if next_sec >= 0:
            end = start + len(SECTION_HEADER) + next_sec
            new_text = text[:start] + new_section + text[end:]
        else:
            new_text = text[:start] + new_section + "\n"
    else:
        # 追加到文件末尾
        new_text = text.rstrip("\n") + "\n\n" + new_section + "\n"

    if dry_run:
        print("\n[dry-run] 写入内容预览：")
        print("─" * 60)
        print(new_section)
        print("─" * 60)
    else:
        WATCHLIST_PATH.write_text(new_text, encoding="utf-8")
        print(f"✅ watchlist.md 已更新 → {WATCHLIST_PATH}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="同步 Futu OpenD 持仓到 watchlist.md")
    p.add_argument("--sim",     action="store_true", help="使用模拟账户（TrdEnv.SIMULATE）")
    p.add_argument("--dry-run", action="store_true", help="只打印预览，不写文件")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    env_label = "模拟账户" if args.sim else "真实账户"
    print(f"[sync_holdings] 连接 Futu OpenD @ {OPEND_HOST}:{OPEND_PORT}（{env_label}）...")

    holdings = fetch_holdings(use_sim=args.sim)

    if not holdings:
        print("[sync_holdings] 没有持仓数据，跳过写入。")
        return 0

    print(f"[sync_holdings] 获取到 {len(holdings)} 只持仓：")
    for h in holdings:
        pl_str = f"{h['pl_ratio']:+.2f}%" if h["pl_ratio"] is not None else "N/A"
        print(f"  {h['market']:2}:{h['symbol']:<8}  {h['name']:<20}  "
              f"{h['qty']:>8.0f}股  成本 {h['cost_price']:.3f}  盈亏 {pl_str}")

    update_watchlist(holdings, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
