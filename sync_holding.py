#!/usr/bin/env python3
"""sync_holding.py — 从 Futu OpenD 同步持有、观察到 holding.md

生成 data/lists/holding.md，只含两个区段：
  ## 持有 (Holdings)  — 真实账户持仓 + Futu 自选分组「关注」合并去重
                        （关注视同持有；同一标的以真实持仓为准）
  ## 观察 (Watch)     — Futu 自选股分组「观察」

历史上「持仓」「关注」是分开的两块，现按需求硬编码合并为「持有」一块，
不再单独保留「关注」区段。

运行方式：
    python sync_holding.py
    python sync_holding.py --sim        # 持仓使用模拟账户
    python sync_holding.py --dry-run    # 只打印，不写文件
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
HOLDING_PATH = PROJECT_ROOT / "data" / "lists" / "holding.md"

OPEND_HOST = "127.0.0.1"
OPEND_PORT = 11111

# Futu 自选股分组名 → holding.md 区段
FOCUS_GROUP  = "关注"
WATCH_GROUP  = "观察"


# ═══════════════════════════════════════════════════════
#  持仓获取（Trading Context）
# ═══════════════════════════════════════════════════════

def _fetch_market_positions(market, trd_env, security_firm) -> list[dict]:
    from futu import OpenSecTradeContext, RET_OK  # type: ignore[import]
    ctx = OpenSecTradeContext(
        filter_trdmarket=market,
        host=OPEND_HOST, port=OPEND_PORT,
        security_firm=security_firm,
    )
    try:
        ret, data = ctx.position_list_query(trd_env=trd_env, refresh_cache=True)
        if ret != RET_OK or data is None or (hasattr(data, "empty") and data.empty):
            return []
        results = []
        for _, row in data.iterrows():
            code_raw = str(row.get("code", ""))
            parts = code_raw.split(".", 1)
            mkt_tag = parts[0].upper() if len(parts) == 2 else ""
            symbol  = parts[1] if len(parts) == 2 else code_raw
            if mkt_tag in ("SH", "SZ"):
                mkt_label = "CN"
            elif mkt_tag in ("HK", "US"):
                mkt_label = mkt_tag
            else:
                mkt_label = "HK" if (symbol.isdigit() and len(symbol) <= 5) else "US"
            pl_ratio = row.get("pl_ratio", None)
            pl_valid  = bool(row.get("pl_ratio_valid", False))
            results.append({
                "symbol":     symbol.zfill(5) if mkt_label in ("HK", "CN") else symbol.upper(),
                "market":     mkt_label,
                "name":       str(row.get("stock_name", "")),
                "qty":        float(row.get("qty", 0)),
                "cost_price": float(row.get("cost_price", 0)),
                "pl_ratio":   float(pl_ratio) if pl_valid and pl_ratio is not None else None,
            })
        return results
    finally:
        ctx.close()


def fetch_holdings(use_sim: bool = False) -> list[dict]:
    """获取 HK + US + CN 市场所有持仓。"""
    try:
        from futu import TrdMarket, TrdEnv, SecurityFirm  # type: ignore[import]
    except ImportError:
        print("❌ futu-api 未安装：pip install futu-api")
        sys.exit(1)

    trd_env = TrdEnv.SIMULATE if use_sim else TrdEnv.REAL
    firm    = SecurityFirm.FUTUSECURITIES

    seen: set[str] = set()
    holdings: list[dict] = []
    for mkt in (TrdMarket.HK, TrdMarket.US, TrdMarket.CN):
        for r in _fetch_market_positions(mkt, trd_env, firm):
            key = f"{r['market']}:{r['symbol']}"
            if key not in seen:
                seen.add(key)
                holdings.append(r)
    return holdings


# ═══════════════════════════════════════════════════════
#  自选股分组获取（Quote Context）
# ═══════════════════════════════════════════════════════

def open_quote_ctx():
    """打开一个 OpenQuoteContext（供所有分组复用，避免反复建连触发限流）。"""
    try:
        from futu import OpenQuoteContext  # type: ignore[import]
    except ImportError:
        print("❌ futu-api 未安装：pip install futu-api")
        sys.exit(1)
    return OpenQuoteContext(host=OPEND_HOST, port=OPEND_PORT)


def fetch_group_stocks(ctx, group_name: str, retries: int = 3) -> tuple[bool, list[dict]]:
    """获取指定 Futu 自选股分组的所有标的（复用传入的 ctx，失败重试）。

    Returns:
        (ok, stocks)
        ok=False  → 读取失败（RET 错误/异常），调用方应保留旧数据，切勿用空覆盖。
        ok=True   → 读取成功；stocks 为空表示该分组「确实为空」。
    """
    from futu import RET_OK  # type: ignore[import]

    last_err: object = None
    for attempt in range(1, retries + 1):
        try:
            ret, data = ctx.get_user_security(group_name)
        except Exception as exc:  # noqa: BLE001 — 连接/超时等
            last_err = exc
            time.sleep(1.0)
            continue
        if ret != RET_OK:
            last_err = data
            time.sleep(1.0)
            continue
        if data is None or (hasattr(data, "empty") and data.empty):
            return True, []  # RET_OK 但空 = 分组确实为空
        stocks = []
        for _, row in data.iterrows():
            code_raw = str(row.get("code", "")).strip()
            name     = str(row.get("name", "")).strip()
            parts = code_raw.split(".", 1)
            if len(parts) != 2:
                continue
            mkt_prefix, symbol = parts[0].upper(), parts[1]
            if mkt_prefix in ("SH", "SZ"):
                market, symbol = "CN", symbol.zfill(6)
            elif mkt_prefix == "HK":
                market, symbol = "HK", symbol.zfill(5)
            elif mkt_prefix == "US":
                market, symbol = "US", symbol.upper()
            else:
                continue
            stocks.append({"symbol": symbol, "market": market, "name": name})
        return True, stocks
    print(f"  ⚠️  读取分组 [{group_name}] 失败（重试 {retries} 次）：{last_err}")
    return False, []


def parse_existing_group_section(title_key: str) -> list[dict]:
    """从现有 holding.md 解析某分组区段已有标的（用于拉取失败时保留，防数据丢失）。"""
    if not HOLDING_PATH.exists():
        return []
    stocks: list[dict] = []
    in_section = False
    for line in HOLDING_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            in_section = title_key in line
            continue
        if in_section and line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 3 or cells[0] in ("代码", "") or set(cells[0]) <= {"-"}:
                continue
            stocks.append({"symbol": cells[0], "market": cells[1], "name": cells[2]})
    return stocks


def parse_existing_hold_section() -> list[dict]:
    """解析现有「持有」区段的行，带「来源」列（持仓/关注）用于失败回退。"""
    if not HOLDING_PATH.exists():
        return []
    stocks: list[dict] = []
    in_section = False
    for line in HOLDING_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            in_section = ("持有" in line) or ("持仓" in line)  # 兼容旧段名
            continue
        if in_section and line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 3 or cells[0] in ("代码", "") or set(cells[0]) <= {"-"}:
                continue
            src = cells[6] if len(cells) >= 7 else "持仓"
            stocks.append({"symbol": cells[0], "market": cells[1], "name": cells[2], "source": src})
    return stocks


def parse_existing_holdings_count() -> int:
    """现有「持有」区段中真实持仓（来源=持仓）的标的数（用于持仓拉空时的防呆校验）。"""
    return sum(1 for e in parse_existing_hold_section() if e.get("source") == "持仓")


# ═══════════════════════════════════════════════════════
#  Markdown 生成
# ═══════════════════════════════════════════════════════

def _merge_hold(holdings: list[dict], focus_stocks: list[dict]) -> list[dict]:
    """持仓 + 关注 合并去重为「持有」列表。

    同一标的（market:symbol）以真实持仓为准；只在关注里的标的补入，
    其数量/成本/盈亏为 None（渲染为「-」），来源标 focus。
    """
    merged: dict[str, dict] = {}
    for h in holdings:
        merged[f"{h['market']}:{h['symbol']}"] = {**h, "source": "hold"}
    for s in focus_stocks:
        key = f"{s['market']}:{s['symbol']}"
        if key not in merged:
            merged[key] = {
                "symbol": s["symbol"], "market": s["market"], "name": s.get("name", ""),
                "qty": None, "cost_price": None, "pl_ratio": None, "source": "focus",
            }
    return list(merged.values())


def _build_holdings_section(hold_entries: list[dict]) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "## 持有 (Holdings)",
        "",
        f"> 自动更新：{now_str} | 来源：Futu 真实账户持仓 + 自选分组「关注」合并",
        "",
        "| 代码 | 市场 | 名称 | 数量 | 成本价 | 盈亏% | 来源 |",
        "|------|------|------|------|--------|-------|------|",
    ]
    for h in sorted(hold_entries, key=lambda x: (x["market"], x["symbol"])):
        qty_str  = f"{h['qty']:.0f}" if h.get("qty") is not None else "-"
        cost_str = f"{h['cost_price']:.3f}" if h.get("cost_price") is not None else "-"
        pl_str   = f"{h['pl_ratio']:+.2f}%" if h.get("pl_ratio") is not None else "-"
        src      = "持仓" if h.get("source") == "hold" else "关注"
        lines.append(
            f"| {h['symbol']} | {h['market']} | {h['name']} "
            f"| {qty_str} | {cost_str} | {pl_str} | {src} |"
        )
    return "\n".join(lines)


def _build_watchgroup_section(title: str, subtitle: str, stocks: list[dict]) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"## {title}",
        "",
        f"> 自动更新：{now_str} | 来源：Futu 自选股分组「{subtitle}」",
        "",
        "| 代码 | 市场 | 名称 |",
        "|------|------|------|",
    ]
    for s in sorted(stocks, key=lambda x: (x["market"], x["symbol"])):
        lines.append(f"| {s['symbol']} | {s['market']} | {s['name']} |")
    return "\n".join(lines)


def build_holding_md(
    holdings: list[dict],
    focus_stocks: list[dict],
    watch_stocks: list[dict] | None = None,
    use_sim: bool = False,
    include_watch: bool = True,
) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    env_tag = "模拟账户" if use_sim else "真实账户"
    hold_entries = _merge_hold(holdings, focus_stocks)  # 持仓 + 关注 合并为「持有」
    parts = [
        f"# 核心跟踪列表",
        "",
        f"> 自动生成 {now_str} | Futu OpenD（{env_tag}）",
        "",
        _build_holdings_section(hold_entries),
    ]
    if include_watch:
        parts += [
            "",
            _build_watchgroup_section("观察 (Watch)", WATCH_GROUP, watch_stocks or []),
        ]
    parts.append("")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="同步 Futu OpenD 持有（持仓+关注）/观察到 holding.md")
    parser.add_argument("--sim",     action="store_true", help="持仓使用模拟账户")
    parser.add_argument("--dry-run", action="store_true", help="只打印预览，不写文件")
    parser.add_argument("--include-watch", action="store_true",
                        help="兼容旧参数：holding.md 现在默认包含「观察」")
    parser.add_argument("--force-empty", action="store_true",
                        help="即使持仓拉取为空也强制写入（默认会中止以防数据丢失）")
    args = parser.parse_args()

    print("📥 获取持仓 ...")
    holdings = fetch_holdings(use_sim=args.sim)
    print(f"   持仓 {len(holdings)} 只")

    # 单个 quote ctx 复用拉两个分组，避免反复建连触发限流
    ctx = open_quote_ctx()
    try:
        print(f"📥 获取自选股分组「{FOCUS_GROUP}」...")
        ok_focus, focus_stocks = fetch_group_stocks(ctx, FOCUS_GROUP)
        print(f"📥 获取自选股分组「{WATCH_GROUP}」...")
        ok_watch, watch_stocks = fetch_group_stocks(ctx, WATCH_GROUP)
    finally:
        ctx.close()

    # 拉取失败 → 保留旧数据，绝不用空段覆盖（防数据丢失）
    if not ok_focus:
        # 关注已并入「持有」段，从旧文件按来源=关注回捞
        focus_stocks = [e for e in parse_existing_hold_section() if e.get("source") == "关注"]
        print(f"   ⚠️ 关注拉取失败，保留 holding.md 旧数据 {len(focus_stocks)} 只")
    else:
        print(f"   关注 {len(focus_stocks)} 只（并入持有）")
    if not ok_watch:
        watch_stocks = parse_existing_group_section("观察")
        print(f"   ⚠️ 观察拉取失败，保留 holding.md 旧数据 {len(watch_stocks)} 只")
    else:
        print(f"   观察 {len(watch_stocks)} 只")

    # 防呆：持仓拉空但旧文件本有持仓 → 多半是 OpenD/账户/连接问题，中止写入以防整段丢失
    if not holdings:
        old_n = parse_existing_holdings_count()
        if old_n > 0 and not args.dry_run:
            print(f"\n❌ 持仓拉取为空，但旧 holding.md 有 {old_n} 只持仓——疑似 OpenD 未连/账户异常。")
            print("   已中止写入以防数据丢失。请检查 OpenD 后重试（或加 --force-empty 强制覆盖）。")
            if not args.force_empty:
                sys.exit(2)

    content = build_holding_md(
        holdings,
        focus_stocks,
        watch_stocks,
        use_sim=args.sim,
        include_watch=True,
    )

    if args.dry_run:
        print("\n[dry-run] 预览内容：")
        print("─" * 60)
        print(content[:2000])
        if len(content) > 2000:
            print(f"... （共 {len(content)} 字符）")
        print("─" * 60)
    else:
        HOLDING_PATH.parent.mkdir(parents=True, exist_ok=True)
        HOLDING_PATH.write_text(content, encoding="utf-8")
        n_hold = len(_merge_hold(holdings, focus_stocks))
        print(f"\n✅ holding.md 已写入 → {HOLDING_PATH}")
        print(f"   持有 {n_hold} 只（持仓 {len(holdings)} + 关注 {len(focus_stocks)} 合并去重）"
              f" | 观察 {len(watch_stocks)} 只")


if __name__ == "__main__":
    main()
