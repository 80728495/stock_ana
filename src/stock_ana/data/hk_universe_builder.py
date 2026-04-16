#!/usr/bin/env python3
"""
构建港股主板大市值股票列表并下载三年 K 线数据。

流程:
  1. 抓取 HKEX 每日报价页，解析股票代号 + 名称 + 收市价 + 日成交额
  2. 过滤至主板（代号 00001-09999）+ 排除停牌股
  3. 批量查询 Yahoo Finance 获取市值
  4. 保留市值 >= 200 亿港元的标的
  5. 保存名单到 data/hk_main_largecap_list.csv 及 data/hk_list.txt
  6. 用 akshare 下载三年 K 线数据存为 data/cache/hk/{code}.parquet

用法:
    python -m stock_ana.data.hk_universe_builder              # 用今天的日期拼 HKEX URL
    python -m stock_ana.data.hk_universe_builder --date 260330  # 指定 HKEX 页面日期 (YYMMDD)
    python -m stock_ana.data.hk_universe_builder --no-download  # 只更新名单，不下载 K 线
"""

from __future__ import annotations

import argparse
import csv
import re
import time
import urllib.request
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd
from loguru import logger

from stock_ana.config import CACHE_DIR, DATA_DIR

HK_CACHE_DIR = CACHE_DIR / "hk"
HK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MIN_MARKET_CAP_YI = 200        # 200 亿港元
PREFILTER_TURNOVER = 20_000_000  # 日成交额预过滤门槛（港元），减少 Yahoo 查询量
BAIDU_DELAY = 0.3              # 百度逐只查询间延迟（秒）
KLINE_YEARS = 3                # 下载多少年 K 线
KLINE_DELAY = 0.3              # 每只股票 K 线下载间延迟（秒）

# ═══════════════════════════════════════════════════════
#  Step 1: 解析 HKEX 每日报价页
# ═══════════════════════════════════════════════════════

def _hkex_url(date_str: str | None = None) -> str:
    """生成 HKEX 页面 URL。date_str 格式 YYMMDD，如 260330。"""
    if date_str is None:
        d = datetime.now()
        # 如果是周末则退回到周五
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        date_str = d.strftime("%y%m%d")
    return f"https://www.hkex.com.hk/chi/stat/smstat/dayquot/d{date_str}c.htm"


def fetch_hkex_quotation(date_str: str | None = None) -> str:
    """下载 HKEX 每日报价页，返回 Big5 解码后的字符串。"""
    url = _hkex_url(date_str)
    logger.info(f"抓取 HKEX 页面: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    return raw.decode("big5", errors="replace"), url


def parse_hkex_stocks(text: str) -> pd.DataFrame:
    """
    从 HKEX 报价页 <pre> 内容解析股票列表。

    返回 DataFrame，列：code, name_en, name_zh, close_price, turnover_hkd, suspended
    """
    # 报价区段
    q_start = text.find('"quotations">報價</a>')
    q_end   = text.find('"sales_all"', q_start)
    section = text[q_start:q_end]

    # 股票行正则（匹配含货币字符的有效行）
    head_pat = re.compile(
        r"^\s{0,8}(\d{1,5})\s+"
        r"([A-Z0-9&'()\-\.\/\+\* ]{2,24}?)"
        r"\s*([\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+"
        r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef ]*)?\s*"
        r"(HKD|USD|CNY|EUR|GBP|JPY|SGD|AUD|CAD)\b"
    )
    # 行尾两个逗号数（成交股数 + 成交金额）
    tail_nums = re.compile(r"([\d,]+)\s+([\d,]+)\s*\r?$")
    # 最后一个浮点数前紧接着两个逗号数（收市价在成交股数之前）
    price_pat = re.compile(
        r"(HKD|USD|CNY|EUR|GBP|JPY|SGD|AUD|CAD)\s+"
        r"[\d.]+\s+"          # prev
        r"(?:[\d.]+|-)\s+"   # bid
        r"(?:[\d.]+|-)\s+"   # ask
        r"(?:[\d.]+|-)\s+"   # high
        r"(?:[\d.]+|-)\s+"   # low
        r"([\d.]+)\s+"        # close  ← group(2)
    )

    # TRADING SUSPENDED 检测
    suspended_pat = re.compile(r"TRADING SUSPENDED|暫停買賣", re.IGNORECASE)

    rows = []
    seen_codes: set[str] = set()

    for line in section.split("\n"):
        m = head_pat.match(line)
        if not m:
            continue

        code = m.group(1).zfill(5)
        if code in seen_codes:
            continue
        seen_codes.add(code)

        eng = m.group(2).strip()
        chi_raw = m.group(3) or ""
        chi = chi_raw.replace("\u3000", "").replace("\xa0", "").strip()

        suspended = bool(suspended_pat.search(line))
        close_price = None
        turnover_hkd = None

        if not suspended:
            pm = price_pat.search(line)
            if pm:
                close_price = float(pm.group(2))
            tm = tail_nums.search(line)
            if tm:
                turnover_hkd = int(tm.group(2).replace(",", ""))

        rows.append({
            "code":         code,
            "name_en":      eng,
            "name_zh":      chi,
            "close_price":  close_price,
            "turnover_hkd": turnover_hkd,
            "suspended":    suspended,
        })

    df = pd.DataFrame(rows)
    logger.info(f"解析完成：共 {len(df)} 条证券")
    return df


# ═══════════════════════════════════════════════════════
#  Step 2: 过滤主板 + 去停牌
# ═══════════════════════════════════════════════════════

def filter_main_board(df: pd.DataFrame) -> pd.DataFrame:
    """保留主板（代号 00001–09999）且未停牌的股票。"""
    code_int = df["code"].astype(int)
    main = df[(code_int >= 1) & (code_int <= 9999) & (~df["suspended"])].copy()
    logger.info(f"主板未停牌：{len(main)} 只")
    return main.reset_index(drop=True)


# ═══════════════════════════════════════════════════════
#  Step 3: 百度股市通批量查询总市值（亿 HKD）
# ═══════════════════════════════════════════════════════

def fetch_market_caps(codes: list[str]) -> dict[str, float | None]:
    """
    用百度股市通逐只查询港股总市值（亿 HKD）。
    返回 {code: market_cap_yi_hkd}，查不到的为 None。
    """
    result: dict[str, float | None] = {c: None for c in codes}
    total = len(codes)
    logger.info(f"市值查询（百度）：{total} 只，预计 {total * BAIDU_DELAY:.0f}s")

    for i, code in enumerate(codes):
        code_str = str(code).zfill(5)  # 确保 5 位零填充字符串格式
        try:
            df = ak.stock_hk_valuation_baidu(
                symbol=code_str, indicator="总市值", period="近一年"
            )
            if df is not None and not df.empty:
                latest = float(df["value"].iloc[-1])
                result[code] = round(latest, 1)
        except Exception as e:
            pass  # 留 None，后续过滤时排除

        if i < total - 1:
            time.sleep(BAIDU_DELAY)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            found = sum(1 for v in result.values() if v is not None)
            logger.info(f"  进度：{i+1}/{total}，已获取 {found} 只市值")

    found = sum(1 for v in result.values() if v is not None)
    logger.info(f"市值查询完成：成功 {found}/{total} 只")
    return result


# ═══════════════════════════════════════════════════════
#  Step 4: 按市值过滤 + 保存名单
# ═══════════════════════════════════════════════════════

def build_largecap_list(
    df_main: pd.DataFrame,
    market_caps: dict[str, float | None],
    min_yi: float = MIN_MARKET_CAP_YI,
) -> pd.DataFrame:
    """将市值附加到 DataFrame，过滤 >= min_yi 亿的标的。"""
    df = df_main.copy()
    df["market_cap_yi"] = df["code"].map(market_caps)

    # 内存中无法查到市值的（OTC/停牌/代号问题），默认保留（以 NaN 标记）
    passed = df[df["market_cap_yi"] >= min_yi].copy()
    logger.info(
        f"市值 >= {min_yi}亿 HKD：{len(passed)} 只 "
        f"（查不到市值的排除 {df['market_cap_yi'].isna().sum()} 只）"
    )
    return passed.sort_values("market_cap_yi", ascending=False).reset_index(drop=True)


def save_lists(df: pd.DataFrame) -> tuple[Path, Path]:
    """
    保存两份文件：
    - data/hk_main_largecap_list.csv  完整字段
    - data/hk_list.txt               与现有系统兼容（code<TAB>name_zh）
    """
    csv_path = DATA_DIR / "hk_main_largecap_list.csv"
    txt_path = DATA_DIR / "hk_list.txt"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"保存 CSV → {csv_path}")

    # hk_list.txt 格式：code\tname_zh（原文件格式）
    with open(txt_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            name = row["name_zh"] if row["name_zh"] else row["name_en"]
            f.write(f"{row['code']}\t{name}\n")
    logger.info(f"保存 TXT → {txt_path}（{len(df)} 只）")
    return csv_path, txt_path


# ═══════════════════════════════════════════════════════
#  Step 5: 下载三年 K 线数据
# ═══════════════════════════════════════════════════════

def _yf_code(code: str) -> str:
    """HK 代号 → Yahoo Finance 格式: 00700 → '0700.HK'"""
    return f"{int(code):04d}.HK"


def _download_via_yfinance(code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """用 yfinance 下载 HK 股票历史数据，失败返回 None。"""
    import yfinance as yf
    sym = _yf_code(code)
    hist = yf.download(sym, start=start_date[:4]+"-"+start_date[4:6]+"-"+start_date[6:],
                       end=end_date[:4]+"-"+end_date[4:6]+"-"+end_date[6:],
                       auto_adjust=True, progress=False)
    if hist is None or hist.empty:
        return None
    # Flatten MultiIndex columns if present
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [c[0].lower() for c in hist.columns]
    else:
        hist.columns = [c.lower() for c in hist.columns]
    hist.index.name = "date"
    return hist


def download_kline(df: pd.DataFrame, years: int = KLINE_YEARS,
                   skip_existing: bool = True) -> None:
    """
    用 akshare stock_hk_hist() 下载每只股票的 K 线数据（最近 years 年）。
    当 EM 被封锁时，自动回退到 yfinance。
    保存到 data/cache/hk/{code}.parquet。
    """
    end_date   = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y%m%d")

    total = len(df)
    ok = skipped = failed = 0

    logger.info(f"开始下载 K 线：{total} 只，{start_date} ~ {end_date}"
                + ("（跳过已有）" if skip_existing else ""))

    for i, row in df.iterrows():
        code = str(row["code"]).zfill(5)
        name = str(row.get("name_zh") or row.get("name_en", code))
        out_path = HK_CACHE_DIR / f"{code}.parquet"

        if skip_existing and out_path.exists():
            skipped += 1
            continue

        hist = None
        source = "akshare"

        # 尝试 akshare (Eastmoney)
        try:
            hist = ak.stock_hk_hist(
                symbol=code, period="daily",
                start_date=start_date, end_date=end_date, adjust="qfq",
            )
            if hist is not None and not hist.empty:
                hist.columns = [c.lower() for c in hist.columns]
                hist = hist.rename(columns={
                    "日期": "date", "开盘": "open", "最高": "high",
                    "最低": "low",  "收盘": "close", "成交量": "volume",
                    "成交额": "turnover",
                })
                if "date" in hist.columns:
                    hist["date"] = pd.to_datetime(hist["date"])
                    hist = hist.set_index("date")
            else:
                hist = None
        except Exception:
            hist = None

        # 回退到 yfinance
        if hist is None or hist.empty:
            source = "yfinance"
            try:
                hist = _download_via_yfinance(code, start_date, end_date)
            except Exception:
                hist = None

        if hist is None or hist.empty:
            logger.warning(f"  [{i+1}/{total}] {code} {name}: 两个数据源均失败")
            failed += 1
        else:
            hist.to_parquet(out_path)
            logger.success(f"  [{i+1}/{total}] {code} {name}: {len(hist)} 行 [{source}] → {out_path.name}")
            ok += 1

        time.sleep(KLINE_DELAY)

    logger.info(f"K 线下载完成：成功 {ok}，跳过 {skipped}，失败 {failed}")


# ═══════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════

def build_hk_stock_universe(
    date_str: str | None = None,
    use_cache: bool = False,
    no_download: bool = False,
    min_cap: float = MIN_MARKET_CAP_YI,
) -> pd.DataFrame:
    """Build or refresh the HK main-board large-cap universe list.

    Equivalent to running ``python -m stock_ana.data.hk_universe_builder``
    programmatically.  Returns the filtered large-cap DataFrame.

    Args:
        date_str: HKEX quotation page date in YYMMDD format (e.g. ``"260330"``).
            Defaults to today (last weekday).
        use_cache: Skip HKEX page download; use existing data/hk_full_list.csv.
        no_download: Update the list only; skip K-line download.
        min_cap: Minimum market-cap threshold in 亿 HKD (default 200).
    """
    raw_csv = DATA_DIR / "hk_full_list.csv"
    if use_cache and raw_csv.exists():
        logger.info(f"使用缓存 CSV: {raw_csv}")
        df_all = pd.read_csv(raw_csv, dtype={"code": str})
        df_all["code"] = df_all["code"].str.zfill(5)
    else:
        text, url = fetch_hkex_quotation(date_str)
        df_all = parse_hkex_stocks(text)
        df_all.to_csv(raw_csv, index=False, encoding="utf-8-sig")
        logger.info(f"完整列表 → {raw_csv}（{len(df_all)} 条）")

    df_main = filter_main_board(df_all)

    df_pre = df_main[df_main["turnover_hkd"].fillna(0) >= PREFILTER_TURNOVER].copy()
    logger.info(
        f"日成交额 >= {PREFILTER_TURNOVER/1e6:.0f}M HKD 预过滤："
        f"{len(df_pre)} 只（共 {len(df_main)} 只主板）"
    )

    market_caps = fetch_market_caps(df_pre["code"].tolist())
    df_large = build_largecap_list(df_pre, market_caps, min_yi=min_cap)
    save_lists(df_large)

    if not no_download:
        download_kline(df_large, years=KLINE_YEARS, skip_existing=True)
    else:
        logger.info("no_download=True：跳过 K 线下载")

    return df_large


def main():
    """Build the HK main-board universe and optionally download its price history."""
    parser = argparse.ArgumentParser(description="构建港股主板大市值宇宙列表")
    parser.add_argument("--date", default=None,
                        help="HKEX 报价页日期，格式 YYMMDD，如 260330（默认今天）")
    parser.add_argument("--use-cache", action="store_true",
                        help="跳过 HKEX 页面下载，直接用已有的 data/hk_full_list.csv")
    parser.add_argument("--no-download", action="store_true",
                        help="只更新名单，不下载 K 线数据")
    parser.add_argument("--min-cap", type=float, default=MIN_MARKET_CAP_YI,
                        help=f"最低市值门槛（亿 HKD，默认 {MIN_MARKET_CAP_YI}）")
    args = parser.parse_args()

    df_large = build_hk_stock_universe(
        date_str=args.date,
        use_cache=args.use_cache,
        no_download=args.no_download,
        min_cap=args.min_cap,
    )

    csv_path = DATA_DIR / "hk_main_largecap_list.csv"
    txt_path = DATA_DIR / "hk_list.txt"
    print(f"\n{'='*62}")
    print(f"港股主板大市值宇宙（>= {args.min_cap}亿 HKD）共 {len(df_large)} 只")
    print(f"{'='*62}")
    print(f"{'代号':8s} {'中文名':12s} {'英文名':25s} {'市值(亿HKD)':>12s}")
    print("-" * 62)
    for _, r in df_large.head(30).iterrows():
        mc_str = f"{r['market_cap_yi']:,.0f}" if pd.notna(r['market_cap_yi']) else "N/A"
        print(f"{r['code']:8s} {str(r['name_zh']):<12s} {r['name_en']:<25s} {mc_str:>12s}")
    if len(df_large) > 30:
        print(f"  ... 共 {len(df_large)} 只（仅展示前 30）")
    print(f"{'='*62}")
    print(f"名单已保存: {csv_path}")
    print(f"           {txt_path}")


if __name__ == "__main__":
    main()
