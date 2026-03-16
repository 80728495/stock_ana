"""
美股筛选列表获取模块 — 通过 Finviz Screener 构建美股投资标的池

筛选规则：
1. Finviz 预筛：市值 ≥ $2B（Mid+Large+Mega），日均量 ≥ 50 万股，价格 ≥ $5
2. 流动性合格：估算日均成交额 ≥ 1000 万美元
3. 剔除壳公司（SPAC）、封闭式基金（CEF）、区域性小银行（市值 < 100 亿）
"""

import re
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests as req
from loguru import logger

from stock_ana.config import DATA_DIR

# ─────────────────── 输出路径 ───────────────────

US_UNIVERSE_FILE = DATA_DIR / "us_universe.csv"

# ─────────────────── Finviz 配置 ───────────────────

_FINVIZ_SCREENER = "https://finviz.com/screener.ashx"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finviz.com/screener.ashx",
}

# Finviz URL 筛选参数
#   cap_midover  = 市值 ≥ $2B（Mid + Large + Mega）
#   sh_avgvol_o500 = 日均成交量 ≥ 50 万股
#   sh_price_o5  = 股价 ≥ $5
_FILTERS = "cap_midover,sh_avgvol_o500,sh_price_o5"

# 每页结果数（Finviz 固定 20 行 / 页）
_PAGE_SIZE = 20

# 请求间隔（秒），过快会被封 IP
_REQUEST_DELAY = 1.5

# 最大页数安全上限（防止死循环，20 * 200 = 4000 只）
_MAX_PAGES = 200

# ─────────────────── 过滤规则 ───────────────────

# 估算日均成交额下限（美元） — 用 Price × Volume 近似
_MIN_DAILY_DOLLAR_VOL = 10e6  # $10M

# 需剔除的行业（完全匹配）
_EXCLUDE_INDUSTRIES: set[str] = {
    # ── SPAC / 壳公司 ──
    "Shell Companies",
    # ── 封闭式基金 (CEF) ──
    "Closed-End Fund - Debt",
    "Closed-End Fund - Equity",
    "Closed-End Fund - Foreign",
    # ── 交易所交易基金（非股票） ──
    "Exchange Traded Fund",
}

# SPAC 公司名称关键词正则（备用：防止行业标签缺失时遗漏）
_SPAC_NAME_RE = re.compile(
    r"Acquisition\s+Corp"
    r"|Acquisition\s+Holdings"
    r"|Acquisition\s+Company"
    r"|\bSPAC\b"
    r"|Blank\s+Check"
    r"|Merger\s+Corp"
    r"|Merger\s+Sub"
    r"|Capital\s+Acquisition",
    re.IGNORECASE,
)

# 区域性银行行业名
_REGIONAL_BANK_INDUSTRY = "Banks - Regional"

# 区域性银行：市值低于此值则视为"小银行"并剔除
_REGIONAL_BANK_MIN_CAP = 10e9  # $10B


# ═══════════════════ 内部工具函数 ═══════════════════


def _parse_market_cap(s) -> float | None:
    """
    解析 Finviz 市值字符串

    示例：
        "150.51B" → 150_510_000_000.0
        "5.23B"   → 5_230_000_000.0
        "890.12M" → 890_120_000.0
        "-"       → None
    """
    if pd.isna(s) or str(s).strip() in ("", "-"):
        return None
    s = str(s).strip()
    multipliers = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
    last = s[-1].upper()
    if last in multipliers:
        try:
            return float(s[:-1]) * multipliers[last]
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_number(s) -> float | None:
    """解析数字字符串，兼容千位逗号和缩写后缀"""
    if pd.isna(s) or str(s).strip() in ("", "-"):
        return None
    s = str(s).strip().replace(",", "")
    multipliers = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
    last = s[-1].upper()
    if last in multipliers:
        try:
            return float(s[:-1]) * multipliers[last]
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _is_spac_by_name(name: str) -> bool:
    """通过公司名称判断是否为 SPAC"""
    return bool(_SPAC_NAME_RE.search(str(name)))


def _find_total_count(html: str) -> int | None:
    """从 Finviz 页面 HTML 中提取结果总数"""
    # 匹配形如 "Total: <b>1845</b>" 或 "#1 / 1845 Total" 等
    m = re.search(r"Total\s*[:\-]?\s*<b>\s*(\d+)\s*</b>", html, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'(?:of|/)\s*(\d+)\s*Total', html, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _scrape_finviz_page(session: req.Session, offset: int) -> pd.DataFrame | None:
    """
    抓取 Finviz Screener 的一页结果（Overview 视图，v=111）

    Args:
        session: 复用的 requests Session
        offset:  起始行号（1-indexed），如 1, 21, 41 ...

    Returns:
        DataFrame（单页），或 None（无更多结果）
    """
    params = {
        "v": "111",     # Overview 视图
        "f": _FILTERS,  # 筛选条件
        "r": str(offset),
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            r = session.get(
                _FINVIZ_SCREENER, params=params,
                headers=_HEADERS, timeout=30,
            )
            if r.status_code == 429:
                wait = 5 * (attempt + 1)
                logger.warning(f"Finviz 限流 (429)，等待 {wait}s 后重试 ({attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            break
        except req.RequestException as e:
            if attempt < max_retries - 1:
                logger.warning(f"Finviz 请求失败: {e}，重试 ({attempt+1}/{max_retries})")
                time.sleep(3)
            else:
                raise
    else:
        raise RuntimeError("Finviz 多次请求失败")

    # 解析 HTML 中的表格
    try:
        tables = pd.read_html(StringIO(r.text))
    except ValueError:
        # read_html 未找到任何表格
        return None

    # Finviz 页面含多个带 "Ticker" 列的表格（导航/表头/数据），
    # 真正的数据表特征：列数 ≤ 12 且行数 > 1 且 No. 列含数字
    best: pd.DataFrame | None = None
    best_rows = 0

    for tbl in tables:
        cols = [str(c).strip() for c in tbl.columns]
        if "Ticker" not in cols:
            continue
        tbl.columns = cols
        # 排除列数异常的表（导航表通常有 20-30 列）
        if len(cols) > 12:
            continue
        # 排除全 NaN 行
        tbl = tbl.dropna(how="all")
        tbl = tbl[tbl["Ticker"].notna()]
        # 检验 No. 列含有数字（排除纯文本的导航行）
        if "No." in cols:
            tbl = tbl[pd.to_numeric(tbl["No."], errors="coerce").notna()]
        if len(tbl) > best_rows:
            best = tbl
            best_rows = len(tbl)

    if best is None or best.empty:
        return None
    return best.reset_index(drop=True)


def _scrape_all_finviz() -> pd.DataFrame:
    """
    分页抓取 Finviz Screener 全部结果

    Returns:
        合并后的原始 DataFrame
    """
    session = req.Session()
    all_pages: list[pd.DataFrame] = []
    offset = 1

    logger.info("开始从 Finviz 抓取美股列表 ...")

    for page_num in range(1, _MAX_PAGES + 1):
        page_df = _scrape_finviz_page(session, offset)
        if page_df is None or page_df.empty:
            logger.debug(f"第 {page_num} 页无数据，停止分页")
            break

        all_pages.append(page_df)
        fetched = sum(len(p) for p in all_pages)

        if page_num % 10 == 0:
            logger.info(f"已抓取 {page_num} 页，累计 {fetched} 只股票")

        # 如果本页不足 _PAGE_SIZE 行 → 已到最后一页
        if len(page_df) < _PAGE_SIZE:
            break

        offset += _PAGE_SIZE
        time.sleep(_REQUEST_DELAY)

    if not all_pages:
        raise RuntimeError("Finviz 未返回任何结果，请检查网络或筛选条件")

    df = pd.concat(all_pages, ignore_index=True)
    logger.info(f"Finviz 抓取完成：共 {len(df)} 只股票（{len(all_pages)} 页）")
    return df


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 Finviz 原始数据应用后处理过滤

    过滤项：
    1. 市值 ≥ $5B
    2. 估算日均成交额 ≥ $10M
    3. 剔除 SPAC（Shell Companies + 名称匹配）
    4. 剔除 CEF（封闭式基金）
    5. 剔除市值 < $10B 的区域性银行
    """
    initial = len(df)

    # ── 标准化列名 ──
    rename_map = {
        "Ticker": "ticker",
        "Company": "company",
        "Sector": "sector",
        "Industry": "industry",
        "Country": "country",
        "Market Cap": "market_cap_raw",
        "P/E": "pe",
        "Price": "price_raw",
        "Change": "change",
        "Volume": "volume_raw",
    }
    # 只重命名存在的列
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # ── 解析数值列 ──
    df["market_cap"] = df["market_cap_raw"].apply(_parse_market_cap)
    df["price"] = df.get("price_raw", pd.Series(dtype=float)).apply(_parse_number)
    df["volume"] = df.get("volume_raw", pd.Series(dtype=float)).apply(_parse_number)

    # 丢弃市值解析失败的行
    df = df[df["market_cap"].notna()].copy()
    logger.debug(f"市值解析后剩余 {len(df)} 只")

    # ── 1. 流动性过滤（估算日均成交额 = price × volume） ──
    if "price" in df.columns and "volume" in df.columns:
        df["est_daily_dollar_vol"] = df["price"] * df["volume"]
        before = len(df)
        df = df[
            df["est_daily_dollar_vol"].isna()  # 保留无法计算的（后续手动确认）
            | (df["est_daily_dollar_vol"] >= _MIN_DAILY_DOLLAR_VOL)
        ].copy()
        removed = before - len(df)
        if removed:
            logger.info(f"流动性不足剔除：{removed} 只")

    # ── 3. 剔除排除行业（SPAC 壳公司 + CEF + ETF） ──
    if "industry" in df.columns:
        industry_mask = df["industry"].isin(_EXCLUDE_INDUSTRIES)
        removed_ind = df[industry_mask]
        if not removed_ind.empty:
            logger.info(
                f"行业剔除 {len(removed_ind)} 只："
                + ", ".join(removed_ind["industry"].value_counts().to_dict().keys())
            )
        df = df[~industry_mask].copy()

    # ── 4. 名称匹配剔除 SPAC ──
    if "company" in df.columns:
        spac_mask = df["company"].apply(_is_spac_by_name)
        removed_spac = df[spac_mask]
        if not removed_spac.empty:
            logger.info(f"名称匹配剔除 SPAC {len(removed_spac)} 只")
            for _, row in removed_spac.iterrows():
                logger.debug(f"  SPAC: {row.get('ticker', '?')} - {row.get('company', '?')}")
        df = df[~spac_mask].copy()

    # ── 5. 剔除小型区域银行 ──
    if "industry" in df.columns:
        small_bank_mask = (
            (df["industry"] == _REGIONAL_BANK_INDUSTRY)
            & (df["market_cap"] < _REGIONAL_BANK_MIN_CAP)
        )
        removed_banks = df[small_bank_mask]
        if not removed_banks.empty:
            logger.info(
                f"区域小银行剔除 {len(removed_banks)} 只"
                f"（市值 < ${_REGIONAL_BANK_MIN_CAP/1e9:.0f}B）"
            )
        df = df[~small_bank_mask].copy()

    # ── 清理输出列 ──
    output_cols = ["ticker", "company", "sector", "industry", "country", "market_cap", "price"]
    output_cols = [c for c in output_cols if c in df.columns]
    df = df[output_cols].copy()
    if not df.empty and "market_cap" in df.columns:
        df = df.sort_values("market_cap", ascending=False).reset_index(drop=True)

    logger.info(f"过滤完成：{initial} → {len(df)} 只（剔除 {initial - len(df)} 只）")
    return df


# ═══════════════════ 公开接口 ═══════════════════


def fetch_us_stock_universe(force: bool = False) -> pd.DataFrame:
    """
    获取过滤后的美股投资标的池

    首次运行从 Finviz 在线抓取并缓存到本地 CSV；
    后续运行直接读取缓存，除非指定 force=True 强制刷新。

    Args:
        force: 是否强制重新抓取（忽略缓存）

    Returns:
        DataFrame，columns = [ticker, company, sector, industry, country, market_cap, price]
    """
    # ── 读缓存 ──
    if not force and US_UNIVERSE_FILE.exists():
        df = pd.read_csv(US_UNIVERSE_FILE)
        logger.info(f"从缓存加载美股列表：{len(df)} 只 ({US_UNIVERSE_FILE.name})")
        return df

    # ── 在线抓取 ──
    raw_df = _scrape_all_finviz()

    # ── 后处理过滤 ──
    filtered_df = _apply_filters(raw_df)

    # ── 保存缓存 ──
    filtered_df.to_csv(US_UNIVERSE_FILE, index=False, encoding="utf-8-sig")
    logger.info(f"美股列表已保存：{US_UNIVERSE_FILE}（{len(filtered_df)} 只）")

    return filtered_df


def get_us_tickers(force: bool = False) -> list[str]:
    """
    返回过滤后的美股代码列表

    Args:
        force: 是否强制刷新

    Returns:
        代码列表，如 ["AAPL", "MSFT", "NVDA", ...]
    """
    df = fetch_us_stock_universe(force=force)
    return df["ticker"].tolist()


def get_us_universe_summary(force: bool = False) -> str:
    """
    返回美股列表的统计摘要（板块分布、市值分布等）
    """
    df = fetch_us_stock_universe(force=force)

    lines = [
        f"美股投资标的池 — {len(df)} 只",
        f"数据来源：Finviz Screener",
        f"更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "── 板块分布 ──",
    ]
    sector_counts = df["sector"].value_counts()
    for sector, count in sector_counts.items():
        lines.append(f"  {sector}: {count}")

    lines.append("")
    lines.append("── 市值分布 ──")
    cap = df["market_cap"]
    bins = [5e9, 10e9, 50e9, 200e9, float("inf")]
    labels = ["$5B–$10B", "$10B–$50B", "$50B–$200B", ">$200B"]
    cap_dist = pd.cut(cap, bins=bins, labels=labels).value_counts().sort_index()
    for label, count in cap_dist.items():
        lines.append(f"  {label}: {count}")

    if "industry" in df.columns:
        lines.append("")
        lines.append("── 行业 TOP 15 ──")
        top_ind = df["industry"].value_counts().head(15)
        for ind, count in top_ind.items():
            lines.append(f"  {ind}: {count}")

    return "\n".join(lines)


# ═══════════════════ CLI 入口 ═══════════════════

if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv or "-f" in sys.argv

    df = fetch_us_stock_universe(force=force)
    print(get_us_universe_summary(force=False))
    print(f"\n前 20 只股票：")
    print(df.head(20).to_string(index=False))
