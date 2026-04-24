"""
港股数据获取模块 - 宇宙池全量（市值≥100亿，~575只）
使用富途 OpenD 为主源（批量单连接），akshare 为备用回落
"""

import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from stock_ana.config import CACHE_DIR, DATA_DIR


# ─────────────────── 代理绕过工具 ───────────────────

_PROXY_ENV_KEYS = [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
]


@contextmanager
def _bypass_proxy():
    """
    临时禁用系统代理，确保对东方财富 API 的请求不经过代理。

    akshare 内部使用 requests.get()，而 requests 会自动读取
    macOS 系统代理设置（通过 urllib.request.getproxies()）。
    本上下文管理器通过设置 NO_PROXY=* 来绕过。
    """
    saved = {k: os.environ.get(k) for k in _PROXY_ENV_KEYS}
    saved["NO_PROXY"] = os.environ.get("NO_PROXY")
    saved["no_proxy"] = os.environ.get("no_proxy")

    # 清除所有代理环境变量
    for k in _PROXY_ENV_KEYS:
        os.environ.pop(k, None)
    # 设置 NO_PROXY=* 让 requests 跳过系统代理检测
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"

    # 同时 monkey-patch urllib.request.getproxies 返回空字典
    import urllib.request
    _orig_getproxies = urllib.request.getproxies
    urllib.request.getproxies = lambda: {}

    try:
        yield
    finally:
        # 恢复原始状态
        urllib.request.getproxies = _orig_getproxies
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

# 港股本地存储目录
HK_DIR = CACHE_DIR / "hk"
HK_DIR.mkdir(parents=True, exist_ok=True)

# 港股列表文件
HK_LIST_FILE = DATA_DIR / "hk_list.txt"

# 指数代码 → akshare 符号映射
_INDEX_MAP: dict[str, str] = {
    "800000": "HSI",       # 恒生指数
    "800700": "HSTECH",    # 恒生科技指数
}


# ─────────────────────── 港股成分股列表 ───────────────────────


def load_hk_list() -> pd.DataFrame:
    """
    读取 data/hk_list.txt，解析港股代码与名称。

    文件格式（由 stock_ana.data.hk_universe_builder 生成）：
        code<TAB>name_zh       — 每行一条，制表符分隔

    Returns:
        DataFrame: columns = [code, name]
            code:  股票代码，如 "03750"（5 位，含前导零）
            name:  中文名称
    """
    if not HK_LIST_FILE.exists():
        raise FileNotFoundError(f"港股列表文件不存在: {HK_LIST_FILE}")

    rows = []
    for line in HK_LIST_FILE.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        code = parts[0].strip()
        name = parts[1].strip()
        if not code:
            continue
        rows.append({"code": code, "name": name})

    df = pd.DataFrame(rows)
    logger.info(f"港股列表：共 {len(df)} 只标的")
    return df


def get_hk_stock_codes() -> list[str]:
    """返回所有港股标的代码（个股 + 指数）"""
    df = load_hk_list()
    return df["code"].tolist()


def get_hk_individual_codes() -> list[str]:
    """仅返回个股代码（排除指数）"""
    df = load_hk_list()
    return [c for c in df["code"] if c not in _INDEX_MAP]


def get_hk_index_codes() -> list[str]:
    """仅返回指数代码"""
    df = load_hk_list()
    return [c for c in df["code"] if c in _INDEX_MAP]


# ─────────────────────── 数据获取 ───────────────────────


def fetch_hk_stock(symbol: str,
                   start_date: str = "19700101",
                   end_date: str | None = None) -> pd.DataFrame:
    """
    获取单只港股历史行情数据（前复权）

    优先级：
      A. 富途 OpenD（本地运行，稳定无限流）
      B. akshare 东方财富源（EM，大陆访问需绕代理）
      C. akshare 新浪源（回落）

    Args:
        symbol:     股票代码，如 "00700"
        start_date: 起始日期，格式 "YYYYMMDD"
        end_date:   结束日期，格式 "YYYYMMDD"，默认今天

    Returns:
        DataFrame，index=date，columns=[open, high, low, close, volume]
        （如使用东方财富源，还包含 turnover 列）
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    # 日期转为 "YYYY-MM-DD" 格式，供 futu 使用
    sd_futu = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    ed_futu = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

    # ── 方案 A：富途 OpenD ──
    try:
        from stock_ana.data.fetcher_futu import fetch_hk_stock_futu
        df = fetch_hk_stock_futu(symbol, start_date=sd_futu, end_date=ed_futu)
        if df is not None and not df.empty:
            return df
        logger.debug(f"HK {symbol}: futu 返回空，回落 akshare ...")
    except Exception as futu_err:
        logger.debug(f"HK {symbol}: futu 失败 ({futu_err})，回落 akshare ...")

    # ── 方案 B：akshare 东方财富源 ──
    import akshare as ak

    try:
        with _bypass_proxy():
            df = ak.stock_hk_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )

        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "turnover",
            "换手率": "turnover_rate",
            "涨跌幅": "pct_change",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        cols = [c for c in ["open", "high", "low", "close", "volume", "turnover"] if c in df.columns]
        return df[cols]
    except Exception as em_err:
        logger.debug(f"HK {symbol}: 东方财富源失败 ({em_err})，尝试新浪源 ...")

    # ── 方案 C：akshare 新浪源 ──
    with _bypass_proxy():
        df = ak.stock_hk_daily(symbol=symbol, adjust="qfq")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    sd = pd.to_datetime(start_date, format="%Y%m%d")
    ed = pd.to_datetime(end_date, format="%Y%m%d")
    df = df[(df.index >= sd) & (df.index <= ed)]
    cols = [c for c in ["open", "high", "low", "close", "volume", "turnover"] if c in df.columns]
    return df[cols]


def fetch_hk_index(symbol: str) -> pd.DataFrame:
    """
    获取港股指数历史数据

    优先使用东方财富源，如不可用则回落到新浪源。

    Args:
        symbol: 指数代码 "800000"（恒生指数）或 "800700"（恒生科技）

    Returns:
        DataFrame，index=date，columns=[open, high, low, close]
    """
    import akshare as ak

    ak_symbol = _INDEX_MAP.get(symbol)
    if ak_symbol is None:
        raise ValueError(f"未知的港股指数代码: {symbol}，支持: {list(_INDEX_MAP)}")

    # ── 方案 A：东方财富源 ──
    try:
        with _bypass_proxy():
            df = ak.stock_hk_index_daily_em(symbol=ak_symbol)

        df = df.rename(columns={
            "latest": "close",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df[["open", "high", "low", "close"]]
    except Exception as em_err:
        logger.debug(f"HK Index {symbol}: 东方财富源失败 ({em_err})，尝试新浪源 ...")

    # ── 方案 B：新浪源（回落） ──
    with _bypass_proxy():
        df = ak.stock_hk_index_daily_sina(symbol=ak_symbol)

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    return df[cols]


# ─────────────────── 本地存储 & 增量更新 ───────────────────


def _hk_path(code: str) -> Path:
    """单只港股/指数的本地 parquet 文件路径"""
    return HK_DIR / f"{code}.parquet"


def load_hk_local(code: str) -> pd.DataFrame | None:
    """从本地加载已存储的港股数据，如无则返回 None"""
    path = _hk_path(code)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def save_hk_local(code: str, df: pd.DataFrame) -> None:
    """将港股数据保存到本地 parquet"""
    path = _hk_path(code)
    df.to_parquet(path)
    logger.debug(f"HK {code}: 已保存 {len(df)} 行 → {path}")


def _fetch_single(code: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
    """统一获取单只标的数据（自动区分个股/指数）"""
    if code in _INDEX_MAP:
        df = fetch_hk_index(code)
        # 指数接口不支持 start_date 参数，手动过滤
        if start_date:
            sd = pd.to_datetime(start_date, format="%Y%m%d")
            df = df[df.index >= sd]
        return df
    else:
        return fetch_hk_stock(code, start_date=start_date, end_date=end_date)


def update_hk_data(max_stale_days: int = 0) -> dict[str, pd.DataFrame]:
    """
    增量更新全部港股核心标的的本地数据（富途 OpenD 单连接批量更新）。

    标的来源优先级：
      1. data/lists/hk_universe_list.md（富途宇宙池，市值≥100亿）
      2. 回落到 data/hk_list.txt（旧版 HKEX 来源）

    数据获取：整个批次共用一个 futu 连接，逐只获取；单只失败时回落 akshare。

    逻辑：
    1. 尚无本地数据 → 下载近 3 年全量数据
    2. 已有但不是最新 → 下载缺失日期并合并
    3. 如果数据最后日期距今 ≤ max_stale_days 天，跳过

    Args:
        max_stale_days: 数据过期阈值（天）。默认 0 = 每次都增量更新

    Returns:
        {code: DataFrame} 包含全部标的的最新完整数据
    """
    # ── 读取标的列表（futu宇宙池优先，回落旧版） ──
    try:
        from stock_ana.data.list_manager import load_hk_universe_list
        codes = load_hk_universe_list()
        code_name_map: dict[str, str] = {}
        logger.info(f"使用富途港股宇宙池: {len(codes)} 只")
    except FileNotFoundError:
        logger.warning("hk_universe_list.md 不存在，回落到旧版 hk_list.txt")
        codes = get_hk_stock_codes()
        hk_list = load_hk_list()
        code_name_map = dict(zip(hk_list["code"], hk_list["name"]))

    today = pd.Timestamp.now().normalize()
    three_years_ago = (today - timedelta(days=365 * 3))
    three_years_ago_yyyymmdd = three_years_ago.strftime("%Y%m%d")
    three_years_ago_iso = three_years_ago.strftime("%Y-%m-%d")
    today_iso = today.strftime("%Y-%m-%d")

    # ── 分组 ──
    need_full: list[str] = []
    need_incr: dict[str, pd.Timestamp] = {}
    skipped: list[str] = []

    for code in codes:
        if code in _INDEX_MAP:
            # 指数走单独接口，不纳入 futu 批量
            local = load_hk_local(code)
            if local is None or local.empty or (today - local.index.max()).days > max_stale_days:
                need_full.append(code)
            else:
                skipped.append(code)
            continue

        local = load_hk_local(code)
        if local is None or local.empty:
            need_full.append(code)
        else:
            last_date = pd.Timestamp(local.index.max()).normalize()
            if (today - last_date).days > max_stale_days:
                need_incr[code] = last_date
            else:
                skipped.append(code)

    logger.info(
        f"港股数据状态：全量下载 {len(need_full)} 只 | "
        f"增量更新 {len(need_incr)} 只 | 已最新 {len(skipped)} 只"
    )

    updated = 0
    failed = 0

    # ── 主路径：futu 单连接批量 ──
    individual_codes_full = [c for c in need_full if c not in _INDEX_MAP]
    individual_codes_incr = {c: d for c, d in need_incr.items() if c not in _INDEX_MAP}
    index_codes_todo = [c for c in need_full if c in _INDEX_MAP]

    if individual_codes_full or individual_codes_incr:
        all_individual = individual_codes_full + list(individual_codes_incr.keys())
        logger.info(
            f"开始 futu 单连接批量获取 {len(all_individual)} 只个股 "
            f"(限频 0.5s/只，预计 {len(all_individual) * 0.5 / 60:.1f} 分钟) ..."
        )

        from stock_ana.data.fetcher_futu import fetch_hk_stock_with_ctx, quote_context
        futu_failed: list[str] = []

        with quote_context() as ctx:
            # 全量下载
            for i, code in enumerate(individual_codes_full, 1):
                name = code_name_map.get(code, "")
                try:
                    df = fetch_hk_stock_with_ctx(ctx, code, three_years_ago_iso, today_iso)
                    if not df.empty:
                        save_hk_local(code, df)
                        updated += 1
                        if i % 20 == 0 or i == len(individual_codes_full):
                            logger.info(
                                f"[全量 {i}/{len(individual_codes_full)}] {code} {name}: "
                                f"{len(df)} 行 ({df.index.min().date()} ~ {df.index.max().date()})"
                            )
                    else:
                        logger.warning(f"[全量 {i}/{len(individual_codes_full)}] {code} {name}: futu 返回空")
                        futu_failed.append(code)
                except Exception as e:
                    logger.debug(f"[全量 {i}/{len(individual_codes_full)}] {code} {name}: futu 失败({e})，加入回落列表")
                    futu_failed.append(code)
                time.sleep(0.5)  # 限频：60次/30秒

            # 增量更新
            for i, (code, last_date) in enumerate(individual_codes_incr.items(), 1):
                name = code_name_map.get(code, "")
                incr_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                try:
                    new_df = fetch_hk_stock_with_ctx(ctx, code, incr_start, today_iso)
                    if not new_df.empty:
                        old_df = load_hk_local(code)
                        combined = pd.concat([old_df, new_df])
                        combined = combined[~combined.index.duplicated(keep="last")]
                        combined = combined.sort_index()
                        save_hk_local(code, combined)
                        updated += 1
                        if i % 20 == 0 or i == len(individual_codes_incr):
                            logger.info(
                                f"[增量 {i}/{len(individual_codes_incr)}] {code} {name}: "
                                f"新增 {len(new_df)} 行，总计 {len(combined)} 行"
                            )
                    else:
                        logger.debug(f"[增量 {i}/{len(individual_codes_incr)}] {code} {name}: 无新数据")
                        updated += 1  # 无新数据不算失败
                except Exception as e:
                    logger.debug(f"[增量 {i}/{len(individual_codes_incr)}] {code} {name}: futu 失败({e})，加入回落列表")
                    futu_failed.append(code)
                time.sleep(0.5)  # 限频：60次/30秒

        # ── 回落路径：akshare（futu 失败的个股） ──
        if futu_failed:
            logger.warning(f"futu 失败 {len(futu_failed)} 只，回落 akshare ...")
            for i, code in enumerate(futu_failed, 1):
                name = code_name_map.get(code, "")
                start_yyyymmdd = (
                    three_years_ago_yyyymmdd
                    if code not in individual_codes_incr
                    else (individual_codes_incr[code] + timedelta(days=1)).strftime("%Y%m%d")
                )
                try:
                    df = _fetch_single(code, start_date=start_yyyymmdd)
                    if not df.empty:
                        if code in individual_codes_incr:
                            old_df = load_hk_local(code)
                            if old_df is not None:
                                df = pd.concat([old_df, df])
                                df = df[~df.index.duplicated(keep="last")].sort_index()
                        save_hk_local(code, df)
                        updated += 1
                        logger.info(f"[akshare回落 {i}/{len(futu_failed)}] {code} {name}: {len(df)} 行")
                    else:
                        failed += 1
                        logger.warning(f"[akshare回落 {i}/{len(futu_failed)}] {code} {name}: 返回空")
                except Exception as e:
                    failed += 1
                    logger.error(f"[akshare回落 {i}/{len(futu_failed)}] {code} {name}: 失败 - {e}")
                time.sleep(0.3)

    # ── 指数（走 akshare 专用接口） ──
    if index_codes_todo:
        logger.info(f"更新 {len(index_codes_todo)} 个港股指数 ...")
        for code in index_codes_todo:
            try:
                df = fetch_hk_index(code)
                if not df.empty:
                    save_hk_local(code, df)
                    updated += 1
            except Exception as e:
                failed += 1
                logger.error(f"指数 {code}: 更新失败 - {e}")

    logger.info(f"港股更新完成: 成功 {updated}, 跳过 {len(skipped)}, 失败 {failed}")

    # ── 加载全部本地数据返回 ──
    result: dict[str, pd.DataFrame] = {}
    for code in codes:
        df = load_hk_local(code)
        if df is not None and not df.empty:
            result[code] = df

    logger.info(f"港股本地数据加载完毕：{len(result)}/{len(codes)} 只标的可用")
    return result



def load_all_hk_data() -> dict[str, pd.DataFrame]:
    """
    仅读取本地已存储的港股数据，不做任何网络请求。

    Returns:
        {code: DataFrame} 字典
    """
    codes = get_hk_stock_codes()
    result: dict[str, pd.DataFrame] = {}
    for code in codes:
        df = load_hk_local(code)
        if df is not None and not df.empty:
            result[code] = df
    logger.info(f"从本地加载 {len(result)}/{len(codes)} 只港股数据")
    return result


def load_hk_index_data(index_code: str = "800000") -> pd.DataFrame | None:
    """
    加载单个港股指数数据

    Args:
        index_code: "800000"(恒生指数) 或 "800700"(恒生科技)
    """
    return load_hk_local(index_code)
