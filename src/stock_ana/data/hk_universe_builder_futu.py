"""
通过富途 OpenD 构建港股投资标的池

筛选规则（与美股 Finviz 方案对齐）：
  1. 仅保留主板（HK_MAINBOARD）与创业板（HK_GEM），剔除衍生品
  2. 市值 ≥ MIN_MARKET_CAP_HKD（默认 20 亿港元，约合 2.5 亿美元）
  3. 近 20 日均日成交额 ≥ MIN_AVG_TURNOVER_HKD（默认 3000 万港元）
  4. 现价 ≥ MIN_PRICE_HKD（默认 0.5 港元，排除仙股）
  5. 排除停牌或退市股票

输出：
  data/hk_universe.csv   — 完整机器可读表（供扫描/研究使用）
  data/hk_list.txt       — code<TAB>name_zh（向后兼容现有缓存系统）

用法：
    python -m stock_ana.data.hk_universe_builder_futu
    python -m stock_ana.data.hk_universe_builder_futu --min-cap 50   # 50 亿港元
    python -m stock_ana.data.hk_universe_builder_futu --min-turn 5000 # 5000 万港元
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from stock_ana.config import DATA_DIR
from stock_ana.data.fetcher_futu import quote_context

# ─────────────────────── 筛选阈值（单位：港元） ───────────────────────

MIN_MARKET_CAP_HKD: float = 100e8      # 100 亿港元
MIN_AVG_TURNOVER_HKD: float = 3000e4   # 3000 万港元（近20日均值）
MIN_PRICE_HKD: float = 0.5             # 0.5 港元

# 每次 get_stock_filter 最多拉取记录数
_FILTER_PAGE_SIZE: int = 200

# 防限频：每页请求之间等待（秒）
_PAGE_DELAY: float = 0.5

# 仅保留这些交易所类型的股票
_ALLOW_EXCHANGE_TYPES: set[str] = {"HK_MAINBOARD", "HK_GEM"}

# 输出文件路径
HK_UNIVERSE_FILE = DATA_DIR / "hk_universe.csv"
HK_LIST_TXT = DATA_DIR / "hk_list.txt"


# ═══════════════════════ 核心函数 ═══════════════════════

def _fetch_all_basicinfo() -> pd.DataFrame:
    """
    拉取港股全量静态信息（约 2600 只），返回 code / name / exchange_type / delisting 列。
    只保留未退市的个股。
    """
    from futu import Market, RET_OK, SecurityType

    logger.info("拉取港股全量静态信息 ...")
    with quote_context() as ctx:
        ret, data = ctx.get_stock_basicinfo(Market.HK, SecurityType.STOCK)

    if ret != RET_OK:
        raise RuntimeError(f"get_stock_basicinfo 失败: {data}")

    logger.info(f"  原始记录: {len(data)} 只")

    # 只保留未退市个股
    df = data[data["delisting"] == False].copy()
    df = df[["code", "name", "exchange_type"]].copy()
    df.columns = ["futu_code", "name_zh", "exchange_type"]

    # 只保留主板 / 创业板
    df = df[df["exchange_type"].isin(_ALLOW_EXCHANGE_TYPES)].copy()
    logger.info(f"  主板+创业板未退市: {len(df)} 只")
    return df


def _fetch_stocks_by_filter(
    min_market_cap: float,
    min_avg_turnover: float,
    min_price: float,
) -> pd.DataFrame:
    """
    通过 get_stock_filter 分页拉取满足市值+流动性+价格门槛的港股。

    Returns:
        DataFrame，列：futu_code, name_zh, cur_price, market_cap_hkd,
                       avg_turnover_20d, avg_volume_20d
    """
    from futu import (
        AccumulateFilter,
        Market,
        RET_OK,
        SimpleFilter,
        SortDir,
        StockField,
    )

    # ── 市值筛选（简单指标，降序排列） ──
    cap_filter = SimpleFilter()
    cap_filter.stock_field = StockField.MARKET_VAL
    cap_filter.filter_min = min_market_cap
    cap_filter.is_no_filter = False
    cap_filter.sort = SortDir.DESCEND

    # ── 20 日均成交额筛选 ──
    turn_filter = AccumulateFilter()
    turn_filter.stock_field = StockField.TURNOVER
    turn_filter.filter_min = min_avg_turnover
    turn_filter.is_no_filter = False
    turn_filter.days = 20

    # ── 价格下限（排除仙股） ──
    price_filter = SimpleFilter()
    price_filter.stock_field = StockField.CUR_PRICE
    price_filter.filter_min = min_price
    price_filter.is_no_filter = False

    filter_list = [cap_filter, turn_filter, price_filter]

    logger.info(
        f"条件选股 HK | 市值≥{min_market_cap/1e8:.0f}亿 | "
        f"20日均成交额≥{min_avg_turnover/1e4:.0f}万 | "
        f"股价≥{min_price}"
    )

    records: list[dict] = []
    begin = 0
    all_count: int | None = None

    with quote_context() as ctx:
        while True:
            ret, result = ctx.get_stock_filter(
                market=Market.HK,
                filter_list=filter_list,
                begin=begin,
                num=_FILTER_PAGE_SIZE,
            )
            if ret != RET_OK:
                raise RuntimeError(f"get_stock_filter 失败 (begin={begin}): {result}")

            last_page, all_count, stock_list = result

            if all_count is not None and begin == 0:
                logger.info(f"  符合条件总数: {all_count} 只")

            for item in stock_list:
                records.append(
                    {
                        "futu_code": item.stock_code,
                        "name_zh": item.stock_name,
                        "cur_price": item[price_filter],
                        "market_cap_hkd": item[cap_filter],
                        "avg_turnover_20d": item[turn_filter],
                    }
                )

            begin += len(stock_list)
            logger.debug(f"  已获取 {begin} / {all_count} 只")

            if last_page or not stock_list:
                break

            time.sleep(_PAGE_DELAY)

    logger.info(f"  条件选股共返回 {len(records)} 只")
    return pd.DataFrame(records)


def _fetch_snapshot_batch(codes: list[str]) -> pd.DataFrame:
    """
    批量快照：获取停牌状态（供二次过滤）。

    频率限制：每 30 秒最多 60 次请求，单次最多 400 只（普通权限）。
    策略：每批 400 只，间隔 0.6s ≈ 每 30s 最多 50 次，留有余量。
    BMP 用户单次上限 20 只，如遇失败可将 BATCH 改小。
    """
    from futu import RET_OK

    BATCH = 400          # 普通权限上限
    DELAY = 0.6          # 每批间隔（秒），60 次/30s 限制下留余量
    all_rows: list[pd.DataFrame] = []

    with quote_context() as ctx:
        for i in range(0, len(codes), BATCH):
            batch = codes[i : i + BATCH]
            ret, data = ctx.get_market_snapshot(batch)
            if ret != RET_OK:
                logger.warning(f"get_market_snapshot 批次 [{i}:{i+BATCH}] 失败: {data}")
                continue
            all_rows.append(data[["code", "suspension"]].copy())
            if i + BATCH < len(codes):   # 最后一批不需要等待
                time.sleep(DELAY)

    if not all_rows:
        return pd.DataFrame(columns=["code", "suspension"])
    return pd.concat(all_rows, ignore_index=True)


# ═══════════════════════ 主入口 ═══════════════════════

def build_hk_universe_futu(
    min_market_cap: float = MIN_MARKET_CAP_HKD,
    min_avg_turnover: float = MIN_AVG_TURNOVER_HKD,
    min_price: float = MIN_PRICE_HKD,
    save: bool = True,
) -> pd.DataFrame:
    """
    构建港股投资标的池。

    Args:
        min_market_cap:    市值下限（港元），默认 20 亿
        min_avg_turnover:  近 20 日均日成交额下限（港元），默认 3000 万
        min_price:         股价下限（港元），默认 0.5
        save:              是否写出 CSV / TXT 文件

    Returns:
        DataFrame，列：
            code          — 股票代码（5 位，如 "00700"）
            futu_code     — 富途格式代码（如 "HK.00700"）
            name_zh       — 中文名称
            exchange_type — 交易所类型
            cur_price     — 最新价（港元）
            market_cap_hkd   — 总市值（港元）
            market_cap_yi    — 总市值（亿港元，保留 1 位小数）
            avg_turnover_20d — 近 20 日均日成交额（港元）
            suspended     — 是否停牌
    """
    # ── Step 1: 获取全量基础信息（主板+创业板，未退市） ──
    basicinfo = _fetch_all_basicinfo()

    # ── Step 2: 条件选股（市值 + 流动性 + 价格） ──
    filtered = _fetch_stocks_by_filter(min_market_cap, min_avg_turnover, min_price)

    if filtered.empty:
        logger.warning("条件选股返回空列表，请检查 OpenD 行情权限或阈值设置")
        return pd.DataFrame()

    # ── Step 3: 与基础信息 inner join（确保只留主板/创业板，排除退市） ──
    merged = filtered.merge(basicinfo[["futu_code", "exchange_type"]], on="futu_code", how="inner")
    logger.info(f"与基础信息 join 后: {len(merged)} 只")

    # ── Step 4: 批量快照获取停牌状态 ──
    logger.info("获取停牌状态 ...")
    snap = _fetch_snapshot_batch(merged["futu_code"].tolist())
    snap = snap.rename(columns={"code": "futu_code", "suspension": "suspended"})
    merged = merged.merge(snap, on="futu_code", how="left")
    merged["suspended"] = merged["suspended"].fillna(False)

    # 排除停牌股
    before = len(merged)
    merged = merged[merged["suspended"] == False].copy()
    logger.info(f"排除停牌后: {len(merged)} 只（剔除 {before - len(merged)} 只）")

    # ── Step 5: 整理输出格式 ──
    # code: 5 位零填充，不含市场前缀（如 "HK.00700" → "00700"）
    merged["code"] = merged["futu_code"].str.removeprefix("HK.").str.zfill(5)
    merged["market_cap_yi"] = (merged["market_cap_hkd"] / 1e8).round(1)

    result = merged[
        [
            "code",
            "futu_code",
            "name_zh",
            "exchange_type",
            "cur_price",
            "market_cap_hkd",
            "market_cap_yi",
            "avg_turnover_20d",
            "suspended",
        ]
    ].sort_values("market_cap_hkd", ascending=False).reset_index(drop=True)

    logger.info(
        f"\n港股全量宇宙构建完成: {len(result)} 只\n"
        f"  市值中位数:        {result['market_cap_yi'].median():.1f} 亿港元\n"
        f"  市值前10:\n{result[['code','name_zh','market_cap_yi']].head(10).to_string(index=False)}"
    )

    # ── Step 6: 保存 ──
    if save:
        _save_outputs(result)

    return result


def _save_outputs(df: pd.DataFrame) -> None:
    """保存两个文件：hk_universe.csv 和 hk_list.txt"""
    # hk_universe.csv — 完整数据表
    df.to_csv(HK_UNIVERSE_FILE, index=False, encoding="utf-8-sig")
    logger.info(f"已保存: {HK_UNIVERSE_FILE}  ({len(df)} 只)")

    # hk_list.txt — code<TAB>name_zh（向后兼容缓存系统）
    lines = [f"{row.code}\t{row.name_zh}" for row in df.itertuples()]
    HK_LIST_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"已更新: {HK_LIST_TXT}  ({len(df)} 只)")


# ═══════════════════════ CLI ═══════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="通过富途 OpenD 构建港股投资标的池")
    p.add_argument(
        "--min-cap",
        type=float,
        default=MIN_MARKET_CAP_HKD / 1e8,
        metavar="亿HKD",
        help=f"市值下限（亿港元），默认 {MIN_MARKET_CAP_HKD/1e8:.0f}",
    )
    p.add_argument(
        "--min-turn",
        type=float,
        default=MIN_AVG_TURNOVER_HKD / 1e4,
        metavar="万HKD",
        help=f"20日均成交额下限（万港元），默认 {MIN_AVG_TURNOVER_HKD/1e4:.0f}",
    )
    p.add_argument(
        "--min-price",
        type=float,
        default=MIN_PRICE_HKD,
        metavar="HKD",
        help=f"股价下限（港元），默认 {MIN_PRICE_HKD}",
    )
    p.add_argument("--no-save", action="store_true", help="只打印统计，不写文件")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = build_hk_universe_futu(
        min_market_cap=args.min_cap * 1e8,
        min_avg_turnover=args.min_turn * 1e4,
        min_price=args.min_price,
        save=not args.no_save,
    )
    if not df.empty:
        print(f"\n总计 {len(df)} 只，市值分布：")
        buckets = [0, 20, 50, 100, 200, 500, 1000, float("inf")]
        labels = ["<20亿", "20-50亿", "50-100亿", "100-200亿", "200-500亿", "500-1000亿", ">1000亿"]
        for lo, hi, label in zip(buckets, buckets[1:], labels):
            n = ((df["market_cap_yi"] >= lo) & (df["market_cap_yi"] < hi)).sum()
            print(f"  {label:12s}: {n:4d} 只")
