"""
股票列表统一管理模块

维护六个核心列表（存储于 data/lists/），提供加载和同步功能：
  1. watchlist         — 关注列表，含手动维护 + 自动同步持仓（HK + US + CN）
  2. ndx100_list       — 纳斯达克 100 成分股（自动生成）
  3. us_universe_list  — 美股宇宙池 ~1500 只（自动生成）
  4. hk_focus_list     — 港股重点关注（恒生/恒科，市值≥200亿，自动生成）
  5. hk_full_list      — 港股全量标的（自动生成，旧版 HKEX 来源）
  6. hk_universe_list  — 港股投资标的池（富途 OpenD，市值≥100亿，自动生成）

列表文件路径：data/lists/{name}.md
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from stock_ana.config import CACHE_DIR, DATA_DIR

LISTS_DIR = DATA_DIR / "lists"
LISTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────── 内部工具 ───────────────────────


def _read_md_table(path: Path) -> list[list[str]]:
    """解析 MD 文件中第一个表格，返回数据行列表（跳过表头和分隔行）。"""
    rows: list[list[str]] = []
    in_table = False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            # 跳过分隔行（如 |---|---|）
            if all(re.fullmatch(r"-+:?", c) for c in cells):
                in_table = True
                continue
            if in_table:
                rows.append(cells)
        elif in_table and not line.startswith("|"):
            # 换块后再碰到另一个表格时继续
            in_table = False
    return rows


def _write_md_table(
    path: Path,
    title: str,
    subtitle: str,
    count: int,
    headers: list[str],
    rows: list[list[str]],
    today: str | None = None,
) -> None:
    """写入标准格式的 MD 文件（标题 + 元数据注释 + 单张表格）。"""
    if today is None:
        today = date.today().strftime("%Y-%m-%d")

    sep = "|" + "|".join("-" * (len(h) + 2) for h in headers) + "|"
    header_row = "| " + " | ".join(headers) + " |"

    lines = [
        f"# {title}",
        "",
        f"> 自动生成，最后更新：{today}",
        f"> {subtitle}",
        f"> 共 {count} 只",
        "",
        header_row,
        sep,
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────── 加载接口 ───────────────────────


def parse_watchlist(path: Path | None = None) -> dict[str, list[dict[str, str]]]:
    """解析关注列表（watchlist.md），含名称信息。

    支持四种区段：
      - ## .*港股      → hk
      - ## .*美股      → us
      - ## .*大A       → cn
      - ## 持仓.*      → holdings（按各行 market 列自动路由到 hk/us/cn）

    Args:
        path: 显式路径；默认读 data/lists/watchlist.md，不存在时尝试 data/stock_list.md。

    Returns:
        {"us": [{"symbol": "PDD", "name": "拼多多"}, ...], "hk": [...], "cn": [...]}
    """
    if path is None:
        default = LISTS_DIR / "watchlist.md"
        legacy = DATA_DIR / "stock_list.md"
        path = default if default.exists() else legacy

    text = path.read_text(encoding="utf-8")
    result: dict[str, list[dict[str, str]]] = {"us": [], "hk": [], "cn": []}
    section: str | None = None

    for line in text.splitlines():
        stripped = line.strip()

        if re.search(r"##.*港股", stripped):
            section = "hk"
            continue
        if re.search(r"##.*美股", stripped):
            section = "us"
            continue
        if re.search(r"##.*大A", stripped):
            section = "cn"
            continue
        if re.search(r"##.*持仓", stripped):
            section = "holdings"
            continue
        if stripped.startswith("##"):
            section = None
            continue

        if section is None:
            continue

        # Table format: | CODE | ... |
        if stripped.startswith("|"):
            parts = [p.strip() for p in stripped.strip("|").split("|")]
            if not parts:
                continue
            code = parts[0].strip()
            if not code or re.fullmatch(r"[-: ]+", code) or code in ("代码", "Code"):
                continue

            if section == "holdings":
                # Holdings row: | CODE | MARKET | NAME | QTY | COST | PL% |
                if len(parts) < 3:
                    continue
                mkt = parts[1].strip().upper()   # HK / US / CN
                name = parts[2].strip() or code
                if mkt == "HK":
                    result["hk"].append({"symbol": code.zfill(5), "name": name})
                elif mkt == "US":
                    result["us"].append({"symbol": code.upper(), "name": name})
                elif mkt == "CN":
                    result["cn"].append({"symbol": code.zfill(6), "name": name})
                continue

            if len(parts) < 2:
                continue
            name = parts[1].strip() or code
            if section == "hk":
                result["hk"].append({"symbol": code.zfill(5), "name": name})
            elif section == "us":
                result["us"].append({"symbol": code.upper(), "name": name})
            elif section == "cn":
                result["cn"].append({"symbol": code.zfill(6), "name": name})
            continue

        # Legacy dash format: - NAME CODE
        if stripped.startswith("- "):
            payload = stripped[2:].strip()
            if section == "us":
                parts_d = payload.split()
                if parts_d:
                    symbol = parts_d[-1].upper()
                    name = " ".join(parts_d[:-1]) if len(parts_d) > 1 else symbol
                    result["us"].append({"symbol": symbol, "name": name})
            elif section == "hk":
                m = re.search(r"\d{4,5}", payload)
                if m:
                    code = m.group().zfill(5)
                    name = payload.replace(m.group(), "").strip() or code
                    result["hk"].append({"symbol": code, "name": name})

    # Deduplicate per market (持仓可能与手动列表重叠，保留第一次出现)
    for mkt in ("hk", "us", "cn"):
        seen: set[str] = set()
        deduped = []
        for e in result[mkt]:
            if e["symbol"] not in seen:
                seen.add(e["symbol"])
                deduped.append(e)
        result[mkt] = deduped

    return result


def load_watchlist() -> dict[str, list[str]]:
    """加载 Shawn 关注股票列表（仅返回代码）。

    Returns:
        {"hk": ["02400", ...], "us": ["PDD", ...], "cn": ["600519", ...]}
    """
    detailed = parse_watchlist()
    hk = [e["symbol"] for e in detailed["hk"]]
    us = [e["symbol"] for e in detailed["us"]]
    cn = [e["symbol"] for e in detailed.get("cn", [])]
    logger.info(f"关注列表：HK {len(hk)} 只，US {len(us)} 只，CN {len(cn)} 只")
    return {"hk": hk, "us": us, "cn": cn}


def load_ndx100_list() -> list[str]:
    """加载纳指 100 成分股代码列表。"""
    path = LISTS_DIR / "ndx100_list.md"
    if not path.exists():
        raise FileNotFoundError(f"纳指100列表不存在: {path}")
    rows = _read_md_table(path)
    tickers = [r[1] for r in rows if len(r) >= 2]
    logger.info(f"纳指100列表：{len(tickers)} 只")
    return tickers


def load_us_universe_list() -> list[str]:
    """加载美股宇宙池代码列表（仅返回代码）。"""
    path = LISTS_DIR / "us_universe_list.md"
    if not path.exists():
        raise FileNotFoundError(f"美股宇宙池列表不存在: {path}")
    rows = _read_md_table(path)
    tickers = [r[1] for r in rows if len(r) >= 2]
    logger.info(f"美股宇宙池列表：{len(tickers)} 只")
    return tickers


def load_hk_focus_list() -> list[str]:
    """加载港股重点关注列表代码（5位格式，如 '00700'）。"""
    path = LISTS_DIR / "hk_focus_list.md"
    if not path.exists():
        raise FileNotFoundError(f"港股重点列表不存在: {path}")
    rows = _read_md_table(path)
    codes = [r[1] for r in rows if len(r) >= 2]
    logger.info(f"港股重点列表：{len(codes)} 只")
    return codes


def load_hk_full_list() -> list[str]:
    """加载港股全量代码列表（5位格式）。"""
    path = LISTS_DIR / "hk_full_list.md"
    if not path.exists():
        raise FileNotFoundError(f"港股全量列表不存在: {path}")
    rows = _read_md_table(path)
    codes = [r[1] for r in rows if len(r) >= 2]
    logger.info(f"港股全量列表：{len(codes)} 只")
    return codes


def load_hk_universe_list() -> list[str]:
    """加载港股投资标的池代码列表（5位格式，来源：富途 OpenD，市值≥100亿）。"""
    path = LISTS_DIR / "hk_universe_list.md"
    if not path.exists():
        raise FileNotFoundError(
            f"港股宇宙池列表不存在: {path}，请先运行 python -m stock_ana.data.hk_universe_builder_futu"
        )
    rows = _read_md_table(path)
    codes = [r[1] for r in rows if len(r) >= 2]
    logger.info(f"港股宇宙池列表：{len(codes)} 只")
    return codes


# ─────────────────────── 同步（自动生成）接口 ───────────────────────


def sync_us_full_list_md() -> Path:
    """
    生成美股全量合并列表 data/lists/us_full_list.md。

    内容：us_universe.csv 全量美股 + ndx100 成分股，按市値降序排序去重。
    作为策略扫描的标的总池，可直接在这里手动添加/删除标的。
    ndx100、科技股等只是筛选列表，不再单独维护数据源。
    """
    csv = DATA_DIR / "us_universe.csv"
    if not csv.exists():
        raise FileNotFoundError(f"请先运行 build_us_stock_universe() 生成: {csv}")
    df = pd.read_csv(csv, dtype={"ticker": str})

    # 合并 ndx100 中不在 us_universe 里的少数几只（如 ANSS、BKNG）
    ndx_dir = CACHE_DIR / "ndx100"
    ndx_tickers = {p.stem.upper() for p in ndx_dir.glob("*.parquet")}
    us_tickers  = set(df["ticker"].str.upper())
    extra = ndx_tickers - us_tickers - {"QQQ"}  # QQQ 是 ETF 排除
    if extra:
        extra_rows = pd.DataFrame([{"ticker": t, "company": t, "sector": "", "industry": "",
                                     "country": "USA", "market_cap": 0.0, "price": 0.0}
                                    for t in sorted(extra)])
        df = pd.concat([df, extra_rows], ignore_index=True)

    df = df.sort_values("market_cap", ascending=False).reset_index(drop=True)

    path = LISTS_DIR / "us_full_list.md"
    rows = []
    for i, row in enumerate(df.itertuples(), 1):
        cap_b = f"{row.market_cap / 1e9:.1f}" if hasattr(row, "market_cap") and pd.notna(row.market_cap) and row.market_cap > 0 else "-"
        sector = getattr(row, "sector", "-") or "-"
        rows.append([str(i), str(row.ticker), str(row.company), sector, cap_b])

    _write_md_table(
        path,
        title="美股全量列表（US Full Universe）",
        subtitle="来源：Finviz US Universe + NDX100 割入，去重下市値排序",
        count=len(df),
        headers=["#", "代码", "公司", "行业", "市値(B)"],
        rows=rows,
    )
    logger.info(f"已同步 us_full_list.md（{len(df)} 只）")
    return path


def load_us_full_list() -> list[dict[str, str]]:
    """读取 us_full_list.md，返回 [{ticker, company, sector, market_cap_b}, ...]。"""
    path = LISTS_DIR / "us_full_list.md"
    if not path.exists():
        raise FileNotFoundError(f"us_full_list.md 不存在，请先运行 sync_us_full_list_md()")
    rows = _read_md_table(path)
    result = []
    for row in rows:
        if len(row) < 2:
            continue
        result.append({
            "ticker":       row[1].strip().upper() if len(row) > 1 else "",
            "company":      row[2].strip()         if len(row) > 2 else "",
            "sector":       row[3].strip()         if len(row) > 3 else "",
            "market_cap_b": row[4].strip()         if len(row) > 4 else "-",
        })
    return result


_TECH_SECTORS = {
    "Technology",
    "Communication Services",
    "Consumer Cyclical",
    "Healthcare",
    "Industrials",
    "Financial",
}


def sync_us_tech_list_md() -> Path:
    """
    生成美股高弹性板块列表 data/lists/us_tech_list.md。

    覆盖行业：Technology, Communication Services, Consumer Cyclical,
    Healthcare, Industrials, Financial（共 6 个行业）。
    排除：Utilities, Energy, Basic Materials, Real Estate, Consumer Defensive。
    再补入 NDX100 里属于上述板块但不在 universe 中的股票，按市值降序。
    每日扫描 Vegas Mid 策略时读取此列表，而非全量 1500+ 只。
    """
    csv = DATA_DIR / "us_universe.csv"
    if not csv.exists():
        raise FileNotFoundError(f"请先运行 build_us_stock_universe() 生成: {csv}")
    df = pd.read_csv(csv, dtype={"ticker": str})
    df = df[df["sector"].isin(_TECH_SECTORS)].copy()

    # 补入 NDX100 里属于科技/通信但不在 universe 的（理论上极少）
    ndx_dir = CACHE_DIR / "ndx100"
    # 用 sec_profiles 补充 ndx100 的 sector 信息
    sec_profile_path = DATA_DIR / "us_sec_profiles.csv"
    sec_sectors: dict[str, str] = {}
    if sec_profile_path.exists():
        try:
            sp = pd.read_csv(sec_profile_path, dtype=str, encoding="utf-8-sig")
            sp.columns = [c.lstrip("\ufeff").strip() for c in sp.columns]
            for _, r in sp.iterrows():
                t = str(r.get("ticker", "")).strip().upper()
                s = str(r.get("sector", "")).strip()
                if t:
                    sec_sectors[t] = s
        except Exception:
            pass

    us_tickers = set(df["ticker"].str.upper())
    for p in ndx_dir.glob("*.parquet"):
        t = p.stem.upper()
        if t in us_tickers or t == "QQQ":
            continue
        if sec_sectors.get(t, "") in _TECH_SECTORS:
            df = pd.concat([df, pd.DataFrame([{
                "ticker": t, "company": t, "sector": sec_sectors[t],
                "industry": "", "country": "USA", "market_cap": 0.0, "price": 0.0,
            }])], ignore_index=True)

    df = df.sort_values("market_cap", ascending=False).reset_index(drop=True)

    path = LISTS_DIR / "us_tech_list.md"
    rows = []
    for i, row in enumerate(df.itertuples(), 1):
        cap_b = f"{row.market_cap / 1e9:.1f}" if hasattr(row, "market_cap") and pd.notna(row.market_cap) and row.market_cap > 0 else "-"
        sector = getattr(row, "sector", "-") or "-"
        rows.append([str(i), str(row.ticker), str(row.company), sector, cap_b])

    _write_md_table(
        path,
        title="美股高弹性板块列表（US Active Universe）",
        subtitle="来源：US Universe 过滤 Technology / Communication Services / Consumer Cyclical / Healthcare / Industrials / Financial",
        count=len(df),
        headers=["#", "代码", "公司", "行业", "市値(B)"],
        rows=rows,
    )
    logger.info(f"已同步 us_tech_list.md（{len(df)} 只）")
    return path


def load_us_tech_list() -> list[dict[str, str]]:
    """读取 us_tech_list.md，返回 [{ticker, company, sector, market_cap_b}, ...]。"""
    path = LISTS_DIR / "us_tech_list.md"
    if not path.exists():
        raise FileNotFoundError(f"us_tech_list.md 不存在，请先运行 sync_us_tech_list_md()")
    rows = _read_md_table(path)
    result = []
    for row in rows:
        if len(row) < 2:
            continue
        result.append({
            "ticker":       row[1].strip().upper() if len(row) > 1 else "",
            "company":      row[2].strip()         if len(row) > 2 else "",
            "sector":       row[3].strip()         if len(row) > 3 else "",
            "market_cap_b": row[4].strip()         if len(row) > 4 else "-",
        })
    return result





    """
    同步纳指100列表到 MD 文件。

    Args:
        tickers: 传入 ticker 列表；None 则从本地缓存目录读取
    """
    if tickers is None:
        ndx_dir = CACHE_DIR / "ndx100"
        tickers = sorted(p.stem for p in ndx_dir.glob("*.parquet"))

    path = LISTS_DIR / "ndx100_list.md"
    rows = [[str(i), t] for i, t in enumerate(tickers, 1)]
    _write_md_table(
        path,
        title="纳斯达克 100 成分股列表",
        subtitle="来源：Wikipedia Nasdaq-100 + 本地缓存",
        count=len(tickers),
        headers=["#", "代码"],
        rows=rows,
    )
    logger.info(f"已同步 ndx100_list.md ({len(tickers)} 只)")
    return path


def sync_us_universe_list_md(df: pd.DataFrame | None = None) -> Path:
    """
    同步美股宇宙池列表到 MD 文件。

    Args:
        df: 传入 DataFrame（含 ticker/company/sector/market_cap 列）；
            None 则从 data/us_universe.csv 读取
    """
    if df is None:
        csv = DATA_DIR / "us_universe.csv"
        if not csv.exists():
            raise FileNotFoundError(f"请先运行 build_us_stock_universe() 生成: {csv}")
        df = pd.read_csv(csv)

    path = LISTS_DIR / "us_universe_list.md"
    rows = []
    for i, row in enumerate(df.itertuples(), 1):
        cap_b = f"{row.market_cap / 1e9:.1f}" if hasattr(row, "market_cap") and pd.notna(row.market_cap) else "-"
        sector = getattr(row, "sector", "-") or "-"
        rows.append([str(i), str(row.ticker), str(row.company), sector, cap_b])

    _write_md_table(
        path,
        title="美股宇宙池（US Universe ~1500）",
        subtitle="来源：Finviz（市值≥$2B，均量≥50万，股价≥$5，日交易额≥$10M）",
        count=len(df),
        headers=["#", "代码", "公司", "行业", "市值(B)"],
        rows=rows,
    )
    logger.info(f"已同步 us_universe_list.md ({len(df)} 只)")
    return path


def sync_hk_focus_list_md(df: pd.DataFrame | None = None) -> Path:
    """
    同步港股重点列表到 MD 文件。

    Args:
        df: 含 code/name_zh/name_en/market_cap_yi 的 DataFrame；
            None 则从 data/hk_main_largecap_list.csv 读取
    """
    if df is None:
        csv = DATA_DIR / "hk_main_largecap_list.csv"
        if not csv.exists():
            raise FileNotFoundError(f"请先运行 python -m stock_ana.data.hk_universe_builder 生成: {csv}")
        df = pd.read_csv(csv)

    df = df.copy()
    df["code5"] = df["code"].apply(lambda x: str(x).zfill(5))

    path = LISTS_DIR / "hk_focus_list.md"
    rows = []
    for i, row in enumerate(df.itertuples(), 1):
        cap = f"{row.market_cap_yi:.0f}" if hasattr(row, "market_cap_yi") and pd.notna(row.market_cap_yi) else "-"
        rows.append([str(i), row.code5, str(row.name_zh), str(row.name_en), cap])

    _write_md_table(
        path,
        title="港股重点关注股票（恒生/恒生科技市场）",
        subtitle="来源：HKEX + Yahoo Finance，市值≥200亿港元",
        count=len(df),
        headers=["#", "代码", "中文名", "英文名", "市值(亿)"],
        rows=rows,
    )
    logger.info(f"已同步 hk_focus_list.md ({len(df)} 只)")
    return path


def sync_hk_full_list_md(df: pd.DataFrame | None = None) -> Path:
    """
    同步港股全量列表到 MD 文件。

    Args:
        df: 含 code/name_zh/name_en 的 DataFrame；
            None 则从 data/hk_full_list.csv 读取
    """
    if df is None:
        csv = DATA_DIR / "hk_full_list.csv"
        if not csv.exists():
            raise FileNotFoundError(f"请先运行 python -m stock_ana.data.hk_universe_builder 生成: {csv}")
        df = pd.read_csv(csv)

    df = df.copy()
    df["code5"] = df["code"].apply(lambda x: str(x).zfill(5))

    path = LISTS_DIR / "hk_full_list.md"
    rows = []
    for i, row in enumerate(df.itertuples(), 1):
        rows.append([str(i), row.code5, str(row.name_zh), str(row.name_en)])

    _write_md_table(
        path,
        title="港股全量列表",
        subtitle="来源：HKEX 日报价，主板股票（代号 00001–09999）",
        count=len(df),
        headers=["#", "代码", "中文名", "英文名"],
        rows=rows,
    )
    logger.info(f"已同步 hk_full_list.md ({len(df)} 只)")
    return path


def sync_hk_universe_list_md(df: pd.DataFrame | None = None) -> Path:
    """
    同步港股投资标的池列表到 data/lists/hk_universe_list.md。

    来源：data/hk_universe.csv（由 hk_universe_builder_futu 生成，市值≥100亿港元）。

    Args:
        df: 含 code/name_zh/market_cap_yi/avg_turnover_20d 的 DataFrame；
            None 则从 data/hk_universe.csv 读取
    """
    if df is None:
        csv = DATA_DIR / "hk_universe.csv"
        if not csv.exists():
            raise FileNotFoundError(
                f"请先运行 python -m stock_ana.data.hk_universe_builder_futu 生成: {csv}"
            )
        df = pd.read_csv(csv, dtype={"code": str})

    df = df.copy()
    df["code"] = df["code"].apply(lambda x: str(x).zfill(5))
    df = df.sort_values("market_cap_hkd", ascending=False).reset_index(drop=True)

    path = LISTS_DIR / "hk_universe_list.md"
    rows = []
    for i, row in enumerate(df.itertuples(), 1):
        cap = f"{row.market_cap_yi:.0f}" if hasattr(row, "market_cap_yi") and pd.notna(row.market_cap_yi) else "-"
        turn = f"{row.avg_turnover_20d / 1e4:.0f}" if hasattr(row, "avg_turnover_20d") and pd.notna(row.avg_turnover_20d) else "-"
        rows.append([str(i), row.code, str(row.name_zh), cap, turn])

    _write_md_table(
        path,
        title="港股投资标的池（HK Universe）",
        subtitle="来源：富途 OpenD，主板，市值≥100亿港元，20日均成交额≥3000万港元",
        count=len(df),
        headers=["#", "代码", "中文名", "市值(亿)", "20日均成交(万)"],
        rows=rows,
    )
    logger.info(f"已同步 hk_universe_list.md（{len(df)} 只）")
    return path


def sync_all_auto_lists(include_hk_full: bool = False) -> None:
    """
    同步自动生成列表。

    日常默认同步：NDX100、美股全量合并列表、美股宇宙池、港股重点、港股宇宙池。
    hk_full_list 作为旧版候选池，默认不参与日常同步；如需同步可显式开启。
    """
    logger.info("开始同步自动生成列表（日常）...")
    sync_ndx100_list_md()
    try:
        sync_us_full_list_md()
    except FileNotFoundError as e:
        logger.warning(f"跳过美股全量合并列表: {e}")
    try:
        sync_us_tech_list_md()
    except FileNotFoundError as e:
        logger.warning(f"跳过美股科技列表: {e}")
    try:
        sync_us_universe_list_md()
    except FileNotFoundError as e:
        logger.warning(f"跳过美股宇宙池: {e}")
    try:
        sync_hk_focus_list_md()
    except FileNotFoundError as e:
        logger.warning(f"跳过港股重点列表: {e}")
    try:
        sync_hk_universe_list_md()
    except FileNotFoundError as e:
        logger.warning(f"跳过港股宇宙池: {e}")

    if include_hk_full:
        try:
            sync_hk_full_list_md()
        except FileNotFoundError as e:
            logger.warning(f"跳过港股全量列表: {e}")

    logger.info("列表同步完成")
