#!/usr/bin/env python3
"""
全量刷新流程脚本（行情 + 二次计算，可选列表源刷新）

目标：一键执行“行情更新 + EMA/成交量/前高 + Wave 全量更新”，
并可按需执行列表源刷新。

默认流程：
  1) OHLCV 更新
     - US: data/cache/us/
     - NDX100: data/cache/ndx100/
     - HK: data/cache/hk/
  2) 列表 MD 同步
      - data/lists/*.md（日常列表；默认不含 hk_full_list）
  3) 二次数据全量更新
     - 指标: data/cache/indicators/{market}/
     - Wave: data/cache/wave_structure/{market}/

可选流程（默认关闭）：
  - 列表源更新
     - 美股宇宙池（Finviz）: data/us_universe.csv
     - 港股全量/重点列表（HKEX + 市值筛选）: stock_ana.data.hk_universe_builder

用法：
  python full_refresh_pipeline.py
    python full_refresh_pipeline.py --refresh-list-sources
  python full_refresh_pipeline.py --skip-hk-universe
  python full_refresh_pipeline.py --hk-no-download
  python full_refresh_pipeline.py --us-max-stale-days 1 --ndx-max-stale-days 0 --hk-max-stale-days 0
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loguru import logger


def _run_build_hk_universe(no_download: bool = False) -> None:
    """调用 hk_universe_builder 更新港股列表源（可选是否下载 K 线）。"""
    from stock_ana.data.hk_universe_builder import build_hk_stock_universe
    logger.info("刷新港股列表源（hk_universe_builder）...")
    build_hk_stock_universe(no_download=no_download)


def step_update_list_sources(skip_us_universe: bool, skip_hk_universe: bool, hk_no_download: bool) -> None:
    """Refresh optional list-source datasets before the broader market refresh pipeline."""
    logger.info("=" * 64)
    logger.info("[1/4] 更新列表源数据")
    logger.info("=" * 64)

    if not skip_us_universe:
        from stock_ana.data.us_universe_builder import build_us_stock_universe
        logger.info("刷新美股宇宙池（Finviz）...")
        df = build_us_stock_universe(force=True)
        logger.success(f"美股宇宙池刷新完成: {len(df)} 只")
    else:
        logger.info("跳过美股宇宙池刷新")

    if not skip_hk_universe:
        _run_build_hk_universe(no_download=hk_no_download)
        logger.success("港股全量/重点列表刷新完成")
    else:
        logger.info("跳过港股列表源刷新")


def step_update_ohlcv(us_max_stale_days: int, ndx_max_stale_days: int, hk_max_stale_days: int) -> None:
    """Refresh OHLCV caches for US, NDX100, and HK markets with stale-day controls."""
    logger.info("=" * 64)
    logger.info("[2/4] 更新 OHLCV 数据")
    logger.info("=" * 64)

    from stock_ana.data.fetcher import update_us_data, update_ndx100_data
    from stock_ana.data.fetcher_hk import update_hk_data

    logger.info("更新 US OHLCV ...")
    us_res = update_us_data(force=False, max_stale_days=us_max_stale_days)
    logger.success(
        f"US 完成: updated={us_res['updated']} skipped={us_res['skipped']} failed={us_res['failed']}"
    )

    logger.info("更新 NDX100 OHLCV ...")
    ndx_data = update_ndx100_data(max_stale_days=ndx_max_stale_days)
    logger.success(f"NDX100 完成: {len(ndx_data)} 只")

    logger.info("更新 HK OHLCV ...")
    hk_data = update_hk_data(max_stale_days=hk_max_stale_days)
    logger.success(f"HK 完成: {len(hk_data)} 只")


def step_sync_lists_md() -> None:
    """Regenerate Markdown list views derived from the refreshed source datasets."""
    logger.info("=" * 64)
    logger.info("[3/4] 同步 MD 列表")
    logger.info("=" * 64)

    from stock_ana.data.list_manager import sync_all_auto_lists
    sync_all_auto_lists()
    logger.success("MD 列表同步完成")


def step_secondary_compute() -> None:
    """Run secondary full-universe computations such as indicators and wave structures."""
    logger.info("=" * 64)
    logger.info("[4/4] 二次数据全量更新（Indicators + Wave）")
    logger.info("=" * 64)

    from stock_ana.data.indicators_store import update_all_indicators
    from stock_ana.data.wave_store import update_wave_structures_for_all_us_hk

    logger.info("更新 Indicators ...")
    update_all_indicators()
    logger.success("Indicators 更新完成")

    logger.info("更新 Wave（全量 US+HK）...")
    res = update_wave_structures_for_all_us_hk()
    logger.success(
        f"Wave 更新完成: ok={len(res.get('ok', []))} skip={len(res.get('skip', []))} fail={len(res.get('fail', []))}"
    )


def main() -> None:
    """Execute the full refresh pipeline CLI with optional list-source and compute stages."""
    parser = argparse.ArgumentParser(description="全量刷新流程脚本")
    parser.add_argument("--refresh-list-sources", action="store_true", help="执行列表源刷新（默认不执行）")
    parser.add_argument("--skip-us-universe", action="store_true", help="跳过美股宇宙池刷新")
    parser.add_argument("--skip-hk-universe", action="store_true", help="跳过港股列表源刷新")
    parser.add_argument("--hk-no-download", action="store_true", help="港股列表源刷新时不下载 K 线")
    parser.add_argument("--skip-ohlcv", action="store_true", help="跳过 OHLCV 更新")
    parser.add_argument("--skip-lists", action="store_true", help="跳过 MD 列表同步")
    parser.add_argument("--skip-secondary", action="store_true", help="跳过二次数据更新")
    parser.add_argument("--us-max-stale-days", type=int, default=1, help="US 允许最大陈旧天数")
    parser.add_argument("--ndx-max-stale-days", type=int, default=0, help="NDX100 允许最大陈旧天数")
    parser.add_argument("--hk-max-stale-days", type=int, default=0, help="HK 允许最大陈旧天数")
    args = parser.parse_args()

    logger.info("=" * 64)
    logger.info(f"全量刷新开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 64)

    t0 = time.time()

    if args.refresh_list_sources:
        step_update_list_sources(
            skip_us_universe=args.skip_us_universe,
            skip_hk_universe=args.skip_hk_universe,
            hk_no_download=args.hk_no_download,
        )
    else:
        logger.info("跳过列表源刷新（默认行为）")

    if not args.skip_ohlcv:
        step_update_ohlcv(
            us_max_stale_days=args.us_max_stale_days,
            ndx_max_stale_days=args.ndx_max_stale_days,
            hk_max_stale_days=args.hk_max_stale_days,
        )
    else:
        logger.info("跳过 OHLCV 更新")

    if not args.skip_lists:
        step_sync_lists_md()
    else:
        logger.info("跳过 MD 列表同步")

    if not args.skip_secondary:
        step_secondary_compute()
    else:
        logger.info("跳过二次数据更新")

    elapsed = time.time() - t0
    logger.info("=" * 64)
    logger.info(f"全量刷新完成，总耗时 {elapsed:.0f}s")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
