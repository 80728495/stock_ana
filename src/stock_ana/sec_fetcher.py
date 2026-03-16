"""
SEC EDGAR 公司业务描述抓取模块

从 SEC EDGAR 免费 API 获取每只美股标的的：
- CIK 编号
- SIC 行业代码 & 描述
- 最新 10-K 年报 Item 1 (Business) 段落原文

遵循 SEC EDGAR 速率限制：≤ 10 请求/秒（实际控制在 ~5 req/s）。
支持断点续传 — 通过进度文件记录已完成的 ticker，中断后可继续。

存储方式：每只股票一个 JSON 文件，另有汇总 CSV 索引。
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from stock_ana.config import CACHE_DIR, DATA_DIR

# ═══════════════════ 路径 & 常量 ═══════════════════

# 存储目录
SEC_DIR = CACHE_DIR / "sec_edgar"
SEC_DIR.mkdir(parents=True, exist_ok=True)

# 进度文件 — 记录已完成和出错的 ticker
PROGRESS_FILE = SEC_DIR / "_progress.json"

# 汇总索引 CSV
INDEX_FILE = DATA_DIR / "us_sec_profiles.csv"

# 美股列表
US_UNIVERSE_FILE = DATA_DIR / "us_universe.csv"

# SEC EDGAR API 限制：10 req/s，这里保守控制在 ~5 req/s
_REQUEST_INTERVAL = 0.22  # 秒/请求（≈4.5 req/s，留余量）

# SEC API 要求提供合规的 User-Agent
_SEC_HEADERS = {
    "User-Agent": "StockAna/1.0 (research tool; contact@example.com)",
    "Accept-Encoding": "gzip, deflate",
}

# 最大重试次数
_MAX_RETRIES = 3

# CIK 映射缓存文件
_CIK_MAP_FILE = SEC_DIR / "_cik_map.json"


# ═══════════════════ CIK 映射 ═══════════════════


def _load_cik_map(force: bool = False) -> dict[str, str]:
    """
    从 SEC EDGAR 获取 ticker → CIK(10位) 映射

    结果缓存到本地 JSON，避免重复请求。
    """
    if not force and _CIK_MAP_FILE.exists():
        data = json.loads(_CIK_MAP_FILE.read_text(encoding="utf-8"))
        logger.debug(f"从缓存加载 CIK 映射：{len(data)} 条")
        return data

    logger.info("从 SEC EDGAR 获取 ticker → CIK 映射 ...")
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=_SEC_HEADERS, timeout=30,
    )
    r.raise_for_status()
    raw = r.json()

    # 构建 ticker → CIK(10位零填充) 映射
    cik_map: dict[str, str] = {}
    for v in raw.values():
        ticker = v["ticker"].upper()
        cik = str(v["cik_str"]).zfill(10)
        cik_map[ticker] = cik

    _CIK_MAP_FILE.write_text(
        json.dumps(cik_map, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"CIK 映射已缓存：{len(cik_map)} 条 → {_CIK_MAP_FILE.name}")
    return cik_map


# ═══════════════════ 进度管理 ═══════════════════


def _load_progress() -> dict:
    """
    加载进度文件

    结构:
    {
        "completed": ["AAPL", "MSFT", ...],
        "failed": {"BADTK": "reason", ...},
        "last_update": "2026-03-14 10:30:00"
    }
    """
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {"completed": [], "failed": {}, "last_update": ""}


def _save_progress(progress: dict) -> None:
    """保存进度文件"""
    progress["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    PROGRESS_FILE.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_progress_summary() -> str:
    """返回当前抓取进度摘要"""
    prog = _load_progress()
    # 加载股票总数
    if US_UNIVERSE_FILE.exists():
        total = len(pd.read_csv(US_UNIVERSE_FILE))
    else:
        total = "?"
    completed = len(prog.get("completed", []))
    failed = len(prog.get("failed", {}))
    last = prog.get("last_update", "未开始")
    remaining = total - completed - failed if isinstance(total, int) else "?"

    return (
        f"SEC EDGAR 抓取进度\n"
        f"  总计: {total} 只\n"
        f"  已完成: {completed} 只\n"
        f"  失败: {failed} 只\n"
        f"  剩余: {remaining} 只\n"
        f"  上次更新: {last}"
    )


# ═══════════════════ 单只股票抓取 ═══════════════════


def _sec_get(url: str, session: requests.Session) -> requests.Response:
    """
    带重试和速率控制的 SEC EDGAR 请求

    遵循 SEC fair access 政策（≤ 10 req/s）。
    """
    for attempt in range(_MAX_RETRIES):
        try:
            r = session.get(url, headers=_SEC_HEADERS, timeout=30)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                logger.warning(f"SEC 限流 (429)，等待 {wait}s ({attempt+1}/{_MAX_RETRIES})")
                time.sleep(wait)
                continue
            if r.status_code == 403:
                logger.error("SEC 返回 403 Forbidden — 可能 IP 被临时封禁，请更换代理后重试")
                raise ConnectionError("SEC 403 Forbidden - IP blocked")
            r.raise_for_status()
            time.sleep(_REQUEST_INTERVAL)
            return r
        except requests.exceptions.ConnectionError as e:
            if attempt < _MAX_RETRIES - 1:
                wait = 5 * (attempt + 1)
                logger.warning(f"连接错误: {e}，等待 {wait}s")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"SEC 请求失败: {url}")


def _clean_html(html: str) -> str:
    """将 HTML 转换为纯文本，保留段落换行"""
    text = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
    # 实体还原
    text = re.sub(r'&#160;', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&#8217;', "'", text)
    text = re.sub(r'&#8220;|&#8221;', '"', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    # 块级标签 → 换行
    text = re.sub(r'<(?:br|/p|/div|/tr|/li|/h[1-6])[^>]*>', '\n', text, flags=re.IGNORECASE)
    # 移除所有标签
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\xa0', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text


def _extract_item1(html: str) -> str:
    """
    从 10-K HTML 中提取 Item 1 (Business) 段落文本

    策略：
    1. 先将整个 HTML 转换为纯文本（去掉所有标签和实体）
    2. 在纯文本中用正则定位 "ITEM 1. BUSINESS" 标题
    3. 截取到 "ITEM 1A" 之前的内容

    这种方式可以处理 "ITEM 1. B" / "USINESS" 跨 span 标签、
    Item 1. + Business 分属不同标签等各种复杂 HTML 结构。
    """
    plain = _clean_html(html)

    # 定位 Item 1 开头 — 找所有匹配位置，跳过目录中的短条目
    item1_patterns = [
        r'(?i)ITEM\s+1\.\s+B\s*U\s*S\s*I\s*N\s*E\s*S\s*S',
        r'(?i)Item\s+1[\.\s\u2014\u2013\-]+\s*Business',
        r'(?i)Item\s+1\.\s+Business',
    ]

    # 收集所有匹配位置
    candidates: list[int] = []
    for pat in item1_patterns:
        for m in re.finditer(pat, plain):
            candidates.append(m.start())
    candidates = sorted(set(candidates))

    if not candidates:
        return ""

    # 找 Item 1A 位置列表，用于判定每个候选区间的内容长度
    item1a_positions: list[int] = []
    for pat in [
        r'(?i)ITEM\s+1A[\.\s\u2014\u2013\-]+\s*R\s*I\s*S\s*K',
        r'(?i)Item\s+1A[\.\s\u2014\u2013\-]+\s*Risk',
        r'(?i)ITEM\s+1A\b',
    ]:
        for m in re.finditer(pat, plain):
            item1a_positions.append(m.start())
    item1a_positions = sorted(set(item1a_positions))

    # 选择内容最丰富的候选（跳过目录中的短条目）
    best_start = None
    best_end = None
    best_length = 0

    for start_pos in candidates:
        # 找紧随其后的 Item 1A 位置作为结束
        end_pos = start_pos + 80000
        for ia_pos in item1a_positions:
            if ia_pos > start_pos + 50:
                end_pos = ia_pos
                break
        content_len = end_pos - start_pos
        if content_len > best_length:
            best_length = content_len
            best_start = start_pos
            best_end = end_pos

    if best_start is None:
        return ""

    section = plain[best_start:best_end].strip()
    return section


def fetch_single_ticker(
    ticker: str,
    cik_map: dict[str, str],
    session: requests.Session,
) -> dict | None:
    """
    抓取单只股票的 SEC EDGAR 信息

    Returns:
        {
            "ticker": "MU",
            "cik": "0000723125",
            "company_name": "MICRON TECHNOLOGY INC",
            "sic_code": "3674",
            "sic_description": "Semiconductors & Related Devices",
            "category": "Large accelerated filer",
            "state": "DE",
            "filing_date": "2025-10-03",
            "item1_business": "... (full text) ...",
            "fetch_time": "2026-03-14 10:30:00"
        }
        或 None（失败时）
    """
    cik = cik_map.get(ticker.upper())
    if not cik:
        logger.warning(f"{ticker}: CIK 未找到")
        return None

    # ── 1. 获取公司提交记录 ──
    sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r_sub = _sec_get(sub_url, session)
    sub = r_sub.json()

    result = {
        "ticker": ticker,
        "cik": cik,
        "company_name": sub.get("name", ""),
        "sic_code": sub.get("sic", ""),
        "sic_description": sub.get("sicDescription", ""),
        "category": sub.get("category", ""),
        "state": sub.get("stateOfIncorporation", ""),
        "filing_date": "",
        "item1_business": "",
        "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── 2. 找最新 10-K ──
    recent = sub.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])

    tenk_idx = None
    for i, form in enumerate(forms):
        if form in ("10-K", "10-K/A"):
            tenk_idx = i
            break

    if tenk_idx is None:
        logger.debug(f"{ticker}: 无 10-K 提交记录")
        return result

    accession = accession_numbers[tenk_idx].replace("-", "")
    primary = primary_docs[tenk_idx]
    result["filing_date"] = filing_dates[tenk_idx]

    # ── 3. 抓取 10-K 文档，提取 Item 1 ──
    cik_num = cik.lstrip("0")
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{accession}/{primary}"

    try:
        r_doc = _sec_get(doc_url, session)
        item1_text = _extract_item1(r_doc.text)
        if item1_text:
            result["item1_business"] = item1_text
            logger.debug(f"{ticker}: Item 1 提取成功 ({len(item1_text)} chars)")
        else:
            logger.debug(f"{ticker}: Item 1 未能从文档中提取")
    except Exception as e:
        logger.warning(f"{ticker}: 10-K 文档获取失败 - {e}")

    return result


def _save_ticker_data(ticker: str, data: dict) -> None:
    """保存单只股票数据到 JSON 文件"""
    path = SEC_DIR / f"{ticker}.json"
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_ticker_data(ticker: str) -> dict | None:
    """加载单只股票的 JSON 数据"""
    path = SEC_DIR / f"{ticker}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


# ═══════════════════ 批量抓取 ═══════════════════


def fetch_all_sec_profiles(
    batch_size: int = 0,
    force: bool = False,
) -> None:
    """
    批量抓取所有 US Universe 股票的 SEC EDGAR 信息

    支持断点续传：
    - 已完成的 ticker 自动跳过（除非 force=True）
    - 中断后重新运行即从上次位置继续
    - 换代理后直接重新运行即可

    Args:
        batch_size: 单次最多抓取数量，0 = 全部
        force:      是否强制重新抓取已完成的
    """
    # 加载股票列表
    if not US_UNIVERSE_FILE.exists():
        raise FileNotFoundError(f"美股列表文件不存在: {US_UNIVERSE_FILE}")
    universe = pd.read_csv(US_UNIVERSE_FILE)
    all_tickers = universe["ticker"].tolist()
    total = len(all_tickers)

    # 加载 CIK 映射
    cik_map = _load_cik_map()

    # 加载进度
    progress = _load_progress()
    completed_set = set(progress.get("completed", []))
    failed_dict: dict[str, str] = progress.get("failed", {})

    if force:
        pending = all_tickers
        logger.info(f"强制模式：将重新抓取全部 {total} 只股票")
    else:
        pending = [t for t in all_tickers if t not in completed_set]
        logger.info(
            f"SEC EDGAR 抓取：总计 {total} 只，"
            f"已完成 {len(completed_set)}，待抓取 {len(pending)}"
        )

    if not pending:
        logger.info("所有股票已抓取完成！")
        return

    if batch_size > 0:
        pending = pending[:batch_size]
        logger.info(f"本次批量限制：{batch_size} 只")

    # 创建 Session 复用连接
    session = requests.Session()

    success_count = 0
    fail_count = 0

    for i, ticker in enumerate(pending, 1):
        try:
            data = fetch_single_ticker(ticker, cik_map, session)
            if data is not None:
                _save_ticker_data(ticker, data)
                completed_set.add(ticker)
                # 从失败列表中移除（如果之前失败过）
                failed_dict.pop(ticker, None)
                success_count += 1

                item1_len = len(data.get("item1_business", ""))
                sic = data.get("sic_code", "?")
                if i % 20 == 0 or i == len(pending):
                    logger.info(
                        f"[{i}/{len(pending)}] {ticker}: "
                        f"SIC={sic}, Item1={item1_len}字符 | "
                        f"总进度 {len(completed_set)}/{total}"
                    )
                else:
                    logger.debug(
                        f"[{i}/{len(pending)}] {ticker}: SIC={sic}, Item1={item1_len}字符"
                    )
            else:
                failed_dict[ticker] = "CIK not found"
                fail_count += 1
                logger.warning(f"[{i}/{len(pending)}] {ticker}: 抓取失败 (CIK 未找到)")

        except ConnectionError as e:
            # IP 被封，保存进度后退出
            logger.error(f"连接被拒 — 已保存进度，请更换代理后重新运行")
            failed_dict[ticker] = str(e)
            fail_count += 1
            break

        except Exception as e:
            failed_dict[ticker] = str(e)
            fail_count += 1
            logger.warning(f"[{i}/{len(pending)}] {ticker}: 错误 - {e}")

        # 每 50 只保存一次进度
        if i % 50 == 0:
            progress["completed"] = sorted(completed_set)
            progress["failed"] = failed_dict
            _save_progress(progress)
            logger.debug(f"进度已保存 ({len(completed_set)}/{total})")

    # 最终保存进度
    progress["completed"] = sorted(completed_set)
    progress["failed"] = failed_dict
    _save_progress(progress)

    logger.info(
        f"本次完成：成功 {success_count}，失败 {fail_count} | "
        f"总进度 {len(completed_set)}/{total}"
    )


# ═══════════════════ 摘要提取 ═══════════════════


def _extract_summary(item1_text: str, max_chars: int = 500) -> str:
    """
    从 Item 1 原文中提取业务摘要（前几句实质性描述）

    跳过标题行、目录（TOC）区域、页码等，
    保留第一段有实质内容的文本，截取到 max_chars。
    """
    if not item1_text:
        return ""

    # ── 第 1 步：跳过 TOC（目录）区域 ──
    # TOC 特征：连续出现多个 "Item X." 条目和页码数字
    # 找到 TOC 之后的实质内容起点
    text = item1_text

    # 检测是否以 TOC 开头：如果前 2000 字符中包含 ≥ 5 个 "Item" 条目
    head = text[:2000]
    item_count = len(re.findall(r'(?i)\bItem\s+\d', head))
    if item_count >= 5:
        # 这是一个 TOC 区域，跳到 TOC 结束后
        # 策略：找到最后一个 "Item" / "Signatures" / "Part IV" 条目之后的位置
        # 然后从那里开始提取
        toc_end_patterns = [
            r'(?i)(?:Signatures|Form\s+10-K\s+Summary)\s*\n\s*\d*\s*\n',
            r'(?i)Note\s+on\s+Incorporation',
        ]
        for pat in toc_end_patterns:
            m = re.search(pat, text[:5000])
            if m:
                text = text[m.end():]
                break
        else:
            # 备选：跳过前 N 个 "Item" 行之后到实质内容
            # 找 "Item 1." 在 TOC 之后第二次出现（正文中的）
            first_item1 = re.search(r'(?i)Item\s+1\.?\s*\n?\s*Business', text)
            if first_item1:
                # 从第一个 Item 1 之后找下一个更大的文本块
                remaining = text[first_item1.end():]
                # 跳过继续的 TOC 条目
                while remaining:
                    m = re.match(r'\s*(?:\d+\s*)?(?:Item\s+\d|Part\s+)', remaining, re.IGNORECASE)
                    if m:
                        next_newline = remaining.find('\n', m.end())
                        if next_newline > 0:
                            remaining = remaining[next_newline + 1:]
                        else:
                            break
                    else:
                        break
                text = remaining

    lines = text.split("\n")
    content_lines: list[str] = []

    # 需要跳过的非内容行模式
    skip_patterns = [
        r'^(?:ITEM|Item)\s+\d',           # Item 1, Item 1A, etc.
        r'^(?:PART|Part)\s+',             # Part I, Part II
        r'^\s*(?:Overview|General|Business)\s*$',  # 单独的标题行
        r'^\s*Table\s+of\s+Contents',     # 目录标题
        r'^\s*\d+\s*$',                   # 纯页码
        r'^\s*(?:Signatures?|Exhibits?)\s*$',  # 签名/附件标题
        r'^\s*$',                         # 空行
    ]
    skip_re = [re.compile(p, re.IGNORECASE) for p in skip_patterns]

    collected = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # 跳过标题行
        if any(p.match(stripped) for p in skip_re):
            continue
        # 跳过太短的行（可能是页码、标签等）
        if len(stripped) < 20:
            continue
        content_lines.append(stripped)
        collected += len(stripped)
        if collected >= max_chars:
            break

    summary = " ".join(content_lines)
    if len(summary) > max_chars:
        # 在最后一个完整句号处截断
        cut = summary[:max_chars]
        last_period = cut.rfind(".")
        if last_period > max_chars * 0.5:
            summary = cut[:last_period + 1]
        else:
            summary = cut + "..."

    return summary


# ═══════════════════ 汇总导出 ═══════════════════


def _load_taxonomy_sics() -> set:
    """读取 taxonomy_v2.yaml，返回需要 LLM sub-label 的 SIC 代码集合"""
    tax_file = DATA_DIR / "taxonomy_v2.yaml"
    if not tax_file.exists():
        return set()
    import yaml
    with open(tax_file) as f:
        tax = yaml.safe_load(f)
    return set(tax.get("sub_labels", {}).keys())


def build_index() -> pd.DataFrame:
    """
    将所有已抓取的 JSON 文件汇总为索引 CSV，并合并 Finviz sector/industry
    以及 taxonomy_v2 的 needs_sub_label 标记。

    Returns:
        DataFrame: [ticker, company_name, sic_code, sic_description,
                    sector, industry, needs_sub_label, sub_label,
                    category, state, filing_date, item1_length, business_summary]
    """
    rows = []
    for f in sorted(SEC_DIR.glob("*.json")):
        if f.name.startswith("_"):
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        item1 = data.get("item1_business", "")
        rows.append({
            "ticker": data.get("ticker", f.stem),
            "company_name": data.get("company_name", ""),
            "sic_code": data.get("sic_code", ""),
            "sic_description": data.get("sic_description", ""),
            "category": data.get("category", ""),
            "state": data.get("state", ""),
            "filing_date": data.get("filing_date", ""),
            "item1_length": len(item1),
            "business_summary": _extract_summary(item1),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 合并 Finviz sector
    universe_file = DATA_DIR / "us_universe.csv"
    if universe_file.exists():
        uv = pd.read_csv(universe_file)
        df = df.merge(uv[["ticker", "sector"]], on="ticker", how="left")
    else:
        df["sector"] = ""

    df["sub_label"] = ""

    # 保留已有 sub_label（如果旧 CSV 中已有人工/LLM 填写的结果）
    if INDEX_FILE.exists():
        old = pd.read_csv(INDEX_FILE, encoding="utf-8-sig")
        if "sub_label" in old.columns:
            old_labels = old[["ticker", "sub_label"]].dropna(subset=["sub_label"])
            old_labels = old_labels[old_labels["sub_label"] != ""]
            if not old_labels.empty:
                label_map = dict(zip(old_labels["ticker"], old_labels["sub_label"]))
                df["sub_label"] = df.apply(
                    lambda r: label_map.get(r["ticker"], r["sub_label"]), axis=1
                )

    # 列顺序
    col_order = [
        "ticker", "company_name", "sector", "sic_code", "sic_description",
        "sub_label", "category", "state", "filing_date", "item1_length",
        "business_summary",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    df.to_csv(INDEX_FILE, index=False, encoding="utf-8-sig")
    logger.info(f"汇总索引已保存：{INDEX_FILE}（{len(df)} 条）")
    return df


def load_ticker_item1(ticker: str) -> str:
    """读取某只股票的 Item 1 业务描述"""
    data = _load_ticker_data(ticker)
    if data:
        return data.get("item1_business", "")
    return ""


def reset_failed() -> None:
    """重置失败列表，使失败的 ticker 可以被重新抓取"""
    progress = _load_progress()
    failed_count = len(progress.get("failed", {}))
    progress["failed"] = {}
    _save_progress(progress)
    logger.info(f"已重置 {failed_count} 条失败记录")


# ═══════════════════ CLI 入口 ═══════════════════

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if "--status" in args or "-s" in args:
        print(get_progress_summary())
        sys.exit(0)

    if "--index" in args:
        df = build_index()
        print(f"索引已生成：{len(df)} 条")
        print(df.head(20).to_string(index=False))
        sys.exit(0)

    if "--reset-failed" in args:
        reset_failed()
        sys.exit(0)

    force = "--force" in args or "-f" in args

    # --batch N 限制单次数量
    batch_size = 0
    if "--batch" in args:
        try:
            bi = args.index("--batch")
            batch_size = int(args[bi + 1])
        except (IndexError, ValueError):
            print("用法: --batch N")
            sys.exit(1)

    fetch_all_sec_profiles(batch_size=batch_size, force=force)

    # 抓取结束后自动生成索引
    build_index()
    print("\n" + get_progress_summary())
