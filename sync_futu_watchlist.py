#!/usr/bin/env python3
"""sync_futu_watchlist.py — 每日从 Futu OpenD 同步自选股到 watchlist.md、big_a.md 及 universe 列表

功能：
  1. 读取 OpenD 所有 CUSTOM 分组标的
  2. 按市场分类（US / HK / CN(SH+SZ)）
  3. 更新 data/lists/watchlist.md 的三个区段（港股/美股/大A），保留现有条目
  4. 将 A 股标的写入 data/lists/big_a.md
  5. 港股个股（非 ETF/指数）不在 hk_universe_list.md 中的，追加进去
  6. 美股个股（非 ETF/指数）不在 us_universe_list.md 中的，追加进去

运行方式：
    python sync_futu_watchlist.py
    python sync_futu_watchlist.py --dry-run    # 只打印变更，不写文件
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LISTS_DIR = PROJECT_ROOT / "data" / "lists"
WATCHLIST_PATH = LISTS_DIR / "watchlist.md"
FUTU_WATCHLIST_PATH = LISTS_DIR / "futu_watchlist.md"
BIG_A_PATH = LISTS_DIR / "big_a.md"
HK_UNIVERSE_PATH = LISTS_DIR / "hk_universe_list.md"
US_UNIVERSE_PATH = LISTS_DIR / "us_universe_list.md"
CN_HIGHTECH_WATCHLIST_PATH = LISTS_DIR / "cn_hightech_watchlist.md"

OPEND_HOST = "127.0.0.1"
OPEND_PORT = 11111

# ─── Futu 读取 ────────────────────────────────────────────────────────────────


def _fetch_all_watchlist_stocks() -> list[dict]:
    """从 OpenD 读取所有 CUSTOM 分组的标的，返回去重后的列表。

    每条记录: {"code": "US.NVDA", "name": "英伟达", "market": "US", "symbol": "NVDA"}
    """
    try:
        from futu import OpenQuoteContext, RET_OK, UserSecurityGroupType  # type: ignore[import]
    except ImportError:
        print("❌ futu-api 未安装，请运行：pip install futu-api")
        sys.exit(1)

    ctx = OpenQuoteContext(host=OPEND_HOST, port=OPEND_PORT)
    try:
        # 获取所有自选股分组
        ret, group_data = ctx.get_user_security_group(UserSecurityGroupType.CUSTOM)
        if ret != RET_OK:
            print(f"❌ 获取自选股分组失败：{group_data}")
            sys.exit(1)

        groups = group_data["group_name"].tolist()
        print(f"  找到 {len(groups)} 个自选股分组：{groups}")

        all_stocks: dict[str, dict] = {}  # code → record，自动去重

        for group_name in groups:
            ret, sec_data = ctx.get_user_security(group_name)
            if ret != RET_OK:
                print(f"  ⚠️  读取分组 [{group_name}] 失败：{sec_data}")
                continue

            for _, row in sec_data.iterrows():
                code_raw = str(row.get("code", "")).strip()  # e.g. "US.NVDA" / "HK.00700"
                name = str(row.get("name", "")).strip()

                parts = code_raw.split(".", 1)
                if len(parts) != 2:
                    continue
                mkt_prefix, symbol = parts[0].upper(), parts[1]

                # 统一市场标识
                if mkt_prefix in ("SH", "SZ"):
                    market = "CN"
                    symbol = symbol.zfill(6)
                elif mkt_prefix == "HK":
                    market = "HK"
                    symbol = symbol.zfill(5)
                elif mkt_prefix == "US":
                    market = "US"
                    symbol = symbol.upper()
                else:
                    continue  # 忽略未知市场

                if code_raw not in all_stocks:
                    all_stocks[code_raw] = {
                        "code": code_raw,      # 原始富途格式 "US.NVDA"
                        "mkt_prefix": mkt_prefix,   # SH/SZ/HK/US
                        "market": market,      # CN/HK/US
                        "symbol": symbol,      # 纯代码（不含市场前缀）
                        "name": name,
                        "group": group_name,
                    }

        stocks = list(all_stocks.values())
        print(f"  共读取 {len(stocks)} 只标的（去重后）")
        return stocks

    finally:
        ctx.close()


def _get_stock_types(codes: list[str]) -> set[str]:
    """给定富途格式代码列表，返回其中属于「股票」类型的 code 集合（过滤 ETF/指数/期货）。

    注意：get_stock_basicinfo 的 stock_type 参数不做查询过滤，
    需对返回结果的 stock_type 列进行后过滤（仅保留 'STOCK'）。
    """
    if not codes:
        return set()

    # 预先排除明显非股票代码：指数（含 '.' 在符号中）和期货合约（字母+4位数字结尾）
    filtered_codes = [
        c for c in codes
        if not re.search(r'\.[A-Z]*\.[\.]', c)  # 排除 US..SPX 类指数
        and not re.search(r'[A-Za-z]+\d{4}$', c.split('.', 1)[-1])  # 排除 GC2604 类期货
    ]

    try:
        from futu import OpenQuoteContext, RET_OK, SecurityType, Market  # type: ignore[import]
    except ImportError:
        return set(filtered_codes)  # 无法判断时保守地全部保留

    ctx = OpenQuoteContext(host=OPEND_HOST, port=OPEND_PORT)
    try:
        stock_codes: set[str] = set()

        # 按市场前缀分组查询
        market_map: dict[str, list[str]] = {}
        for c in filtered_codes:
            mkt = c.split(".")[0].upper()
            market_map.setdefault(mkt, []).append(c)

        futu_market_map = {
            "HK": Market.HK,
            "US": Market.US,
            "SH": Market.SH,
            "SZ": Market.SZ,
        }

        for mkt_prefix, mkt_codes in market_map.items():
            futu_mkt = futu_market_map.get(mkt_prefix)
            if futu_mkt is None:
                continue
            ret, data = ctx.get_stock_basicinfo(futu_mkt, SecurityType.STOCK, mkt_codes)
            if ret == RET_OK and data is not None and not data.empty:
                # 对结果的 stock_type 列做后过滤，仅保留真正的股票
                if "stock_type" in data.columns:
                    data = data[data["stock_type"] == "STOCK"]
                for c in data["code"].tolist():
                    stock_codes.add(c)

        return stock_codes
    finally:
        ctx.close()


# ─── watchlist.md 解析/写入 ───────────────────────────────────────────────────


def _parse_watchlist_section(text: str, section_pattern: str) -> tuple[int, int, list[dict]]:
    """解析 watchlist.md 中指定区段的内容。

    Args:
        text: 完整文件文本
        section_pattern: 区段标题的正则（如 r"##.*港股"）

    Returns:
        (start_idx, end_idx, [{"symbol": "...", "name_cn": "...", "name_en": "..."}])
        start_idx: 区段标题在 text 中的起始位置
        end_idx:   区段结束位置（下一个 ## 的开始 或 文件末尾）
    """
    lines = text.splitlines(keepends=True)
    sec_start_line = -1
    sec_end_line = len(lines)

    for i, line in enumerate(lines):
        if re.search(section_pattern, line.strip()):
            sec_start_line = i
        elif sec_start_line >= 0 and i > sec_start_line and line.strip().startswith("##"):
            sec_end_line = i
            break

    if sec_start_line < 0:
        return -1, -1, []

    # 字符偏移量
    start_char = sum(len(l) for l in lines[:sec_start_line])
    end_char = sum(len(l) for l in lines[:sec_end_line])

    # 解析表格行
    entries: list[dict] = []
    for line in lines[sec_start_line + 1: sec_end_line]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        parts = [p.strip() for p in stripped.strip("|").split("|")]
        if len(parts) < 1:
            continue
        code = parts[0]
        if not code or re.fullmatch(r"[-: ]+", code) or code in ("代码", "Code", "#"):
            continue
        name_cn = parts[1].strip() if len(parts) > 1 else code
        name_en = parts[2].strip() if len(parts) > 2 else name_cn
        entries.append({"symbol": code, "name_cn": name_cn, "name_en": name_en})

    return start_char, end_char, entries


def _append_rows_to_section(text: str, section_pattern: str, new_rows: list[str]) -> str:
    """在 watchlist.md 的指定区段的最后一行表格之后插入新行，不改动标题和已有内容。"""
    if not new_rows:
        return text

    lines = text.splitlines(keepends=True)
    sec_start = -1
    last_table_line = -1

    for i, line in enumerate(lines):
        if re.search(section_pattern, line.strip()):
            sec_start = i
            last_table_line = -1
            continue
        if sec_start >= 0:
            stripped = line.strip()
            if stripped.startswith("##") and i > sec_start:
                break  # 到达下一个区段，停止
            if stripped.startswith("|"):
                last_table_line = i

    if last_table_line < 0:
        return text  # 区段未找到，不修改

    insert_pos = last_table_line + 1
    new_lines = [r + "\n" for r in new_rows]
    lines[insert_pos:insert_pos] = new_lines
    return "".join(lines)


def _merge_entries(
    existing: list[dict],
    new_stocks: list[dict],
    symbol_key: str = "symbol",
) -> tuple[list[dict], list[dict]]:
    """将新标的合并到已有列表（已有的保留原位，新增的追加到末尾）。

    Returns:
        (merged_list, added_list) — merged_list 是完整列表，added_list 是新增条目
    """
    existing_syms = {e["symbol"] for e in existing}
    added = []
    merged = list(existing)
    for s in new_stocks:
        sym = s[symbol_key]
        if sym not in existing_syms:
            entry = {
                "symbol": sym,
                "name_cn": s.get("name", sym),
                "name_en": s.get("name", sym),
            }
            merged.append(entry)
            added.append(entry)
            existing_syms.add(sym)
    return merged, added


def update_watchlist(
    hk_stocks: list[dict],
    us_stocks: list[dict],
    cn_stocks: list[dict],
    dry_run: bool = False,
) -> None:
    """更新 watchlist.md 的港股/美股/大A 三个区段（只增不减）。"""
    text = WATCHLIST_PATH.read_text(encoding="utf-8")

    # 解析三个区段（只需获取现有条目，用于去重）
    _, _, hk_existing = _parse_watchlist_section(text, r"##.*港股")
    _, _, us_existing = _parse_watchlist_section(text, r"##.*美股")
    _, _, cn_existing = _parse_watchlist_section(text, r"##.*大A")

    # 过滤明显非股票（指数：symbol 以 . 开头；期货：symbol 末尾有4位数字）
    def _is_individual(sym: str) -> bool:
        if sym.startswith("."):
            return False
        if re.search(r"[A-Za-z]+\d{4}$", sym):  # 期货，如 GC2604
            return False
        return True

    hk_stocks_f = [s for s in hk_stocks if _is_individual(s["symbol"])]
    us_stocks_f = [s for s in us_stocks if _is_individual(s["symbol"])]

    hk_merged, hk_added = _merge_entries(hk_existing, hk_stocks_f)
    us_merged, us_added = _merge_entries(us_existing, us_stocks_f)
    cn_merged, cn_added = _merge_entries(cn_existing, cn_stocks)

    total_added = len(hk_added) + len(us_added) + len(cn_added)

    if total_added == 0:
        print("  watchlist.md：无新增标的")
    else:
        if hk_added:
            print(f"  watchlist.md 港股新增 {len(hk_added)} 只：{[e['symbol'] for e in hk_added]}")
        if us_added:
            print(f"  watchlist.md 美股新增 {len(us_added)} 只：{[e['symbol'] for e in us_added]}")
        if cn_added:
            print(f"  watchlist.md 大A新增 {len(cn_added)} 只：{[e['symbol'] for e in cn_added]}")

    if dry_run or total_added == 0:
        return

    # 对每个区段，只在末尾表格后插入新行（不重建整个区段）
    new_text = text
    if cn_added:
        rows = [f"| {e['symbol']} | {e['name_cn']} | {e['name_en']} |" for e in cn_added]
        new_text = _append_rows_to_section(new_text, r"##.*大A", rows)
    if us_added:
        rows = [f"| {e['symbol']} | {e['name_cn']} | {e['name_en']} |" for e in us_added]
        new_text = _append_rows_to_section(new_text, r"##.*美股", rows)
    if hk_added:
        rows = [f"| {e['symbol']} | {e['name_cn']} | {e['name_en']} |" for e in hk_added]
        new_text = _append_rows_to_section(new_text, r"##.*港股", rows)

    # 同步更新文件头部的更新日期
    today_str = datetime.now().strftime("%Y-%m-%d")
    new_text = re.sub(
        r"(最后更新：)\d{4}-\d{2}-\d{2}",
        f"\\g<1>{today_str}",
        new_text,
        count=1,
    )

    WATCHLIST_PATH.write_text(new_text, encoding="utf-8")
    print(f"  ✅ watchlist.md 已更新（新增 {total_added} 只）")


# ─── big_a.md 写入 ────────────────────────────────────────────────────────────


def write_big_a(cn_stocks: list[dict], dry_run: bool = False) -> None:
    """将 watchlist 中的 A 股标的写入 big_a.md。"""
    now_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        "# 大A 关注列表",
        "",
        f"> 自动生成，最后更新：{now_str}",
        f"> 来源：Futu OpenD 自选股（所有 CUSTOM 分组中的 A 股标的）",
        f"> 共 {len(cn_stocks)} 只",
        "",
        "| 代码 | 中文名 | 市场 |",
        "|------|--------|------|",
    ]
    for s in sorted(cn_stocks, key=lambda x: x["symbol"]):
        mkt = "SH" if s["mkt_prefix"] == "SH" else "SZ"
        lines.append(f"| {s['symbol']} | {s['name']} | {mkt} |")
    lines.append("")
    content = "\n".join(lines)

    if dry_run:
        print(f"  [dry-run] big_a.md 将写入 {len(cn_stocks)} 只 A 股")
        return

    BIG_A_PATH.write_text(content, encoding="utf-8")
    print(f"  ✅ big_a.md 已写入 {len(cn_stocks)} 只 A 股")


# ─── Universe 列表追加 ────────────────────────────────────────────────────────


def _read_universe_symbols(path: Path) -> set[str]:
    """读取 universe MD 文件中已有的代码集合（第2列）。"""
    symbols: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        parts = [p.strip() for p in stripped.strip("|").split("|")]
        # 表格格式：| # | 代码 | ...
        if len(parts) < 2:
            continue
        code = parts[1]
        if not code or re.fullmatch(r"[-: #]+", code) or code in ("代码", "Code", "#"):
            continue
        symbols.add(code.upper())
    return symbols


def _append_to_universe(path: Path, new_entries: list[dict], market: str, dry_run: bool) -> None:
    """追加新标的到 universe MD 文件末尾。"""
    if not new_entries:
        print(f"  {path.name}：无新增标的")
        return

    symbols = [e["symbol"] for e in new_entries]
    print(f"  {path.name} 新增 {len(new_entries)} 只：{symbols}")

    if dry_run:
        return

    text = path.read_text(encoding="utf-8")

    # 找当前最大序号
    max_num = 0
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if parts and parts[0].isdigit():
            max_num = max(max_num, int(parts[0]))

    append_lines = []
    for i, e in enumerate(new_entries, start=max_num + 1):
        sym = e["symbol"]
        name = e["name"]
        if market == "HK":
            # HK universe 格式: | # | 代码 | 中文名 | 市值(亿) | 20日均成交(万) |
            append_lines.append(f"| {i} | {sym} | {name} | - | - |")
        else:
            # US universe 格式: | # | 代码 | 公司 | 行业 | 市值(B) |
            append_lines.append(f"| {i} | {sym} | {name} | - | - |")

    new_text = text.rstrip("\n") + "\n" + "\n".join(append_lines) + "\n"
    path.write_text(new_text, encoding="utf-8")
    print(f"  ✅ {path.name} 已追加 {len(new_entries)} 只")

    # 同步更新文件中的"共 N 只"和"最后更新"
    updated = path.read_text(encoding="utf-8")
    total_count = len(_read_universe_symbols(path))
    today_str = datetime.now().strftime("%Y-%m-%d")
    updated = re.sub(r"(最后更新：)\d{4}-\d{2}-\d{2}", f"\\g<1>{today_str}", updated, count=1)
    updated = re.sub(r"共 \d+ 只", f"共 {total_count} 只", updated, count=1)
    path.write_text(updated, encoding="utf-8")


def update_universes(
    hk_stocks: list[dict],
    us_stocks: list[dict],
    dry_run: bool = False,
) -> None:
    """将自选股中不在 universe 内的个股追加进去（先过滤 ETF/指数）。"""
    # 收集需要类型检查的代码
    hk_codes = [s["code"] for s in hk_stocks]
    us_codes = [s["code"] for s in us_stocks]

    print("  正在过滤 ETF/指数（查询 Futu stock_type）...")
    stock_type_codes = _get_stock_types(hk_codes + us_codes)

    hk_individual = [s for s in hk_stocks if s["code"] in stock_type_codes]
    us_individual = [s for s in us_stocks if s["code"] in stock_type_codes]

    etf_filtered = (len(hk_stocks) - len(hk_individual)) + (len(us_stocks) - len(us_individual))
    if etf_filtered:
        print(f"  已过滤 {etf_filtered} 只 ETF/指数")

    # 读取现有 universe 代码
    hk_existing = _read_universe_symbols(HK_UNIVERSE_PATH)
    us_existing = _read_universe_symbols(US_UNIVERSE_PATH)

    hk_new = [s for s in hk_individual if s["symbol"].upper() not in hk_existing]
    us_new = [s for s in us_individual if s["symbol"].upper() not in us_existing]

    _append_to_universe(HK_UNIVERSE_PATH, hk_new, "HK", dry_run)
    _append_to_universe(US_UNIVERSE_PATH, us_new, "US", dry_run)


# ─── cn_hightech_watchlist.md 维护 ───────────────────────────────────────────


def _read_cn_hightech_symbols() -> set[str]:
    """读取 cn_hightech_watchlist.md 中已有的代码集合。"""
    if not CN_HIGHTECH_WATCHLIST_PATH.exists():
        return set()
    symbols: set[str] = set()
    for line in CN_HIGHTECH_WATCHLIST_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        parts = [p.strip() for p in stripped.strip("|").split("|")]
        # 表格格式：| # | 代码 | 名称 | 来源 |，代码在第1列
        if len(parts) >= 2 and parts[1].isdigit() and len(parts[1]) == 6:
            symbols.add(parts[1])
    return symbols


def _init_cn_hightech_watchlist_from_csv() -> None:
    """若 cn_hightech_watchlist.md 不存在，从 cn_hightech.csv 初始化。"""
    csv_path = LISTS_DIR / "cn_hightech.csv"
    if not csv_path.exists():
        return

    import csv
    rows_out: list[tuple[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 只取 final_passed=True 的行
            if str(row.get("final_passed", "")).strip().lower() not in ("true", "1"):
                continue
            ticker = str(row.get("ticker", "")).strip().zfill(6)
            name = str(row.get("company_name", "")).strip() or ticker
            if ticker and ticker.isdigit():
                rows_out.append((ticker, name))

    if not rows_out:
        return

    today_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        "# 沪深高新技术关注列表",
        "",
        f"> 来源：高新技术筛选器 + Futu 自选股追加",
        f"> 最后更新：{today_str}",
        f"> 共 {len(rows_out)} 只",
        "",
        "| # | 代码 | 名称 | 来源 |",
        "|---|------|------|------|",
    ]
    for i, (ticker, name) in enumerate(rows_out, 1):
        lines.append(f"| {i} | {ticker} | {name} | 筛选器 |")
    lines.append("")
    CN_HIGHTECH_WATCHLIST_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✅ cn_hightech_watchlist.md 已初始化（{len(rows_out)} 只，来源：筛选器）")


def merge_cn_to_hightech_watchlist(
    cn_stocks: list[dict],
    dry_run: bool = False,
) -> list[str]:
    """将富途自选 A 股中 60/00 开头的个股追加到 cn_hightech_watchlist.md（不在其中才追加）。

    Args:
        cn_stocks: Futu 中的 A 股列表，每条含 {symbol, name, mkt_prefix}
        dry_run:   True 时只打印，不写文件

    Returns:
        新追加的股票代码列表
    """
    # 初始化（若文件不存在则从 CSV 建立）
    if not CN_HIGHTECH_WATCHLIST_PATH.exists():
        _init_cn_hightech_watchlist_from_csv()

    # 过滤：只要 60 / 00 开头的个股
    target = [
        s for s in cn_stocks
        if s["symbol"].startswith(("60", "00"))
    ]
    if not target:
        print("  cn_hightech_watchlist.md：富途自选中无 60/00 开头的 A 股")
        return []

    existing = _read_cn_hightech_symbols()
    new_stocks = [s for s in target if s["symbol"] not in existing]

    if not new_stocks:
        print("  cn_hightech_watchlist.md：无新增标的")
        return []

    print(f"  cn_hightech_watchlist.md 富途追加 {len(new_stocks)} 只：{[s['symbol'] for s in new_stocks]}")
    if dry_run:
        return [s["symbol"] for s in new_stocks]

    # 计算当前最大序号
    max_num = 0
    if CN_HIGHTECH_WATCHLIST_PATH.exists():
        for line in CN_HIGHTECH_WATCHLIST_PATH.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("|"):
                parts = [p.strip() for p in stripped.strip("|").split("|")]
                if parts and parts[0].isdigit():
                    max_num = max(max_num, int(parts[0]))

    append_lines = []
    for i, s in enumerate(new_stocks, start=max_num + 1):
        name = s.get("name", s["symbol"])
        append_lines.append(f"| {i} | {s['symbol']} | {name} | Futu |")

    text = CN_HIGHTECH_WATCHLIST_PATH.read_text(encoding="utf-8")
    new_text = text.rstrip("\n") + "\n" + "\n".join(append_lines) + "\n"

    # 更新文件头的"最后更新"和"共 N 只"
    today_str = datetime.now().strftime("%Y-%m-%d")
    new_text = re.sub(r"(最后更新：)\d{4}-\d{2}-\d{2}", f"\\g<1>{today_str}", new_text, count=1)
    total = max_num + len(new_stocks)
    new_text = re.sub(r"共 \d+ 只", f"共 {total} 只", new_text, count=1)

    CN_HIGHTECH_WATCHLIST_PATH.write_text(new_text, encoding="utf-8")
    print(f"  ✅ cn_hightech_watchlist.md 已追加 {len(new_stocks)} 只（总计 {total} 只）")
    return [s["symbol"] for s in new_stocks]


# ─── futu_watchlist.md 生成 ─────────────────────────────────────────────────


def write_futu_watchlist_md(
    hk_stocks: list[dict],
    us_stocks: list[dict],
    cn_stocks: list[dict],
    dry_run: bool = False,
) -> None:
    """将富途全量自选个股写入 futu_watchlist.md（同格式 watchlist.md，全量覆盖写）。"""

    def _is_individual(sym: str) -> bool:
        if sym.startswith("."):
            return False
        if re.search(r"[A-Za-z]+\d{4}$", sym):  # 期货，如 GC2604
            return False
        return True

    hk_f = [s for s in hk_stocks if _is_individual(s["symbol"])]
    us_f = [s for s in us_stocks if _is_individual(s["symbol"])]

    today_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        "# Futu 自选股完整列表",
        "",
        f"> 自动生成，最后更新：{today_str}",
        f"> 来源：Futu OpenD 所有 CUSTOM 分组（个股，已过滤指数/期货）",
        f"> HK {len(hk_f)} 只 / US {len(us_f)} 只 / CN {len(cn_stocks)} 只",
        "",
        "## 港股 (HK)",
        "",
        "| 代码 | 中文名 | 英文名 |",
        "|------|--------|--------|" ,
    ]
    for s in sorted(hk_f, key=lambda x: x["symbol"]):
        name = s.get("name", s["symbol"])
        lines.append(f"| {s['symbol']} | {name} | {name} |")

    lines += [
        "",
        "## 美股 (US)",
        "",
        "| 代码 | 中文名 | 英文名 |",
        "|------|--------|--------|" ,
    ]
    for s in sorted(us_f, key=lambda x: x["symbol"]):
        name = s.get("name", s["symbol"])
        lines.append(f"| {s['symbol']} | {name} | {name} |")

    lines += [
        "",
        "## 大A (CN)",
        "",
        "| 代码 | 中文名 | 英文名 |",
        "|------|--------|--------|" ,
    ]
    for s in sorted(cn_stocks, key=lambda x: x["symbol"]):
        name = s.get("name", s["symbol"])
        lines.append(f"| {s['symbol']} | {name} | {name} |")

    lines.append("")
    content = "\n".join(lines)

    total = len(hk_f) + len(us_f) + len(cn_stocks)
    if dry_run:
        print(f"  [dry-run] futu_watchlist.md 将写入 {total} 只")
        return

    FUTU_WATCHLIST_PATH.write_text(content, encoding="utf-8")
    print(f"  ✅ futu_watchlist.md 已写入 {total} 只（HK {len(hk_f)} / US {len(us_f)} / CN {len(cn_stocks)}）")


# ─── 主流程 ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="从 Futu OpenD 同步自选股到各 MD 列表")
    parser.add_argument("--dry-run", action="store_true", help="只打印变更，不写入文件")
    args = parser.parse_args()

    dry_run = args.dry_run
    if dry_run:
        print("🔍 [dry-run 模式] 不会实际写入文件\n")

    print("📡 正在从 Futu OpenD 读取自选股...")
    stocks = _fetch_all_watchlist_stocks()

    # 按市场分类
    hk_stocks = [s for s in stocks if s["market"] == "HK"]
    us_stocks = [s for s in stocks if s["market"] == "US"]
    cn_stocks = [s for s in stocks if s["market"] == "CN"]

    print(f"\n  HK: {len(hk_stocks)} 只，US: {len(us_stocks)} 只，CN: {len(cn_stocks)} 只\n")

    # 0. 保存富途全量自选股缓存（供 weekly scan 等过滤用）
    import json as _json
    futu_cache_path = PROJECT_ROOT / "data" / "lists" / "futu_watched_symbols.json"
    if not dry_run:
        _json.dump(
            {"symbols": sorted({s["symbol"] for s in stocks}),
             "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            open(futu_cache_path, "w", encoding="utf-8"),
            ensure_ascii=False, indent=2,
        )
        print(f"  ✅ futu_watched_symbols.json 已保存（{len(stocks)} 只）")

    # 1. 写入 futu_watchlist.md（富途完整自选个股，全量覆盖）
    print("📝 写入 futu_watchlist.md...")
    write_futu_watchlist_md(hk_stocks, us_stocks, cn_stocks, dry_run=dry_run)

    # 2. 写入 big_a.md
    print("\n📝 写入 big_a.md...")
    write_big_a(cn_stocks, dry_run=dry_run)

    # 3. 更新 cn_hightech_watchlist.md（追加 60/00 开头的个股）
    print("\n📝 更新 cn_hightech_watchlist.md...")
    merge_cn_to_hightech_watchlist(cn_stocks, dry_run=dry_run)

    # 4. 更新 universe 列表
    print("\n📝 更新 universe 列表...")
    update_universes(hk_stocks, us_stocks, dry_run=dry_run)

    print("\n✅ 同步完成！")


if __name__ == "__main__":
    main()
