#!/usr/bin/env python3
"""
每日定时数据更新脚本

更新各市场股票列表的 OHLCV 数据，并刷新技术指标与 Wave 结构：

  1. 美股宇宙池 ~1500 只    → data/cache/us/
  2. 纳指 100               → data/cache/ndx100/
  3. 港股宇宙池（~575 只）  → data/cache/hk/
  3b. A 股（watchlist CN）  → data/cache/cn/
  4. 技术指标               → data/cache/indicators/{market}/
     (EMA8/21/34/55/60/144/169/200/250；成交量MA5/10/50；前高252日)
  5. Wave 结构（全量 US+HK） → data/cache/wave_structure/{market}/

用法：
    python daily_update.py              # 全部更新（1-5 含CN）
    python daily_update.py --us         # 仅更新美股 OHLCV
    python daily_update.py --ndx        # 仅更新纳指100 OHLCV
    python daily_update.py --hk         # 仅更新港股 OHLCV
    python daily_update.py --cn         # 仅更新A股 OHLCV
    python daily_update.py --indicators # 仅更新技术指标
    python daily_update.py --waves      # 仅更新 wave 结构（全量 US+HK）
    python daily_update.py --lists      # 仅同步 MD 列表文件
"""

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

# 确保项目路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loguru import logger

# 日志配置：写入文件 + 控制台
LOG_DIR = PROJECT_ROOT / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(
    LOG_DIR / "daily_update_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    encoding="utf-8",
)


# ─────────────────────── 各步更新函数 ───────────────────────


def update_us() -> dict:
    """更新 ~1500 只美股 OHLCV 数据（按最后交易日补缺）。"""
    logger.info("=" * 60)
    logger.info("【1/5】更新美股 ~1500 只 ...")
    logger.info("=" * 60)
    t0 = time.time()
    try:
        from stock_ana.data.fetcher import update_us_data
        # max_stale_days=1: 允许时区带来的 1 天日历差，避免无意义重复全量拉取。
        result = update_us_data(force=False, max_stale_days=1)
        elapsed = time.time() - t0
        logger.success(
            f"✅ 美股更新完成 ({elapsed:.0f}s): "
            f"更新 {result['updated']}, 跳过 {result['skipped']}, 失败 {result['failed']}"
        )
        return {"ok": True, "elapsed": round(elapsed), "updated": result["updated"],
                "skipped": result["skipped"], "failed": result["failed"]}
    except Exception as e:
        logger.error(f"❌ 美股更新失败: {e}")
        return {"ok": False, "elapsed": round(time.time() - t0), "error": str(e)}


def update_ndx() -> dict:
    """更新纳指 100 成分股 OHLCV 数据"""
    logger.info("=" * 60)
    logger.info("【2/5】更新纳指100 ...")
    logger.info("=" * 60)
    t0 = time.time()
    try:
        from stock_ana.data.fetcher import update_ndx100_data
        data = update_ndx100_data()
        elapsed = time.time() - t0
        logger.success(f"✅ 纳指100更新完成 ({elapsed:.0f}s): {len(data)} 只股票")
        return {"ok": True, "elapsed": round(elapsed), "count": len(data)}
    except Exception as e:
        logger.error(f"❌ 纳指100更新失败: {e}")
        return {"ok": False, "elapsed": round(time.time() - t0), "error": str(e)}


def update_hk() -> dict:
    """更新港股宇宙池 OHLCV 数据（市值≥100亿，~575 只，由富途 OpenD 提供）"""
    logger.info("=" * 60)
    logger.info("【3/5】更新港股 ...")
    logger.info("=" * 60)
    t0 = time.time()
    try:
        from stock_ana.data.fetcher_hk import update_hk_data
        data = update_hk_data()
        elapsed = time.time() - t0
        logger.success(f"✅ 港股更新完成 ({elapsed:.0f}s): {len(data)} 只股票")
        return {"ok": True, "elapsed": round(elapsed), "count": len(data)}
    except Exception as e:
        logger.error(f"❌ 港股更新失败: {e}")
        return {"ok": False, "elapsed": round(time.time() - t0), "error": str(e)}


def update_cn() -> dict:
    """更新 A 股（watchlist CN 部分）OHLCV 数据。"""
    logger.info("=" * 60)
    logger.info("【3b/5】更新A股 ...")
    logger.info("=" * 60)
    t0 = time.time()
    try:
        from stock_ana.data.fetcher_cn import update_cn_data
        data = update_cn_data(max_stale_days=1)
        elapsed = time.time() - t0
        logger.success(f"✅ A股更新完成 ({elapsed:.0f}s): {len(data)} 只股票")
        return {"ok": True, "elapsed": round(elapsed), "count": len(data)}
    except Exception as e:
        logger.error(f"❌ A股更新失败: {e}")
        return {"ok": False, "elapsed": round(time.time() - t0), "error": str(e)}


def update_indicators() -> dict:
    """
    更新全部市场的技术指标（US + NDX100 + HK）。

    计算项目：
      - 扩展 EMA：8, 21, 34, 55, 60, 144, 169, 200, 250
      - 成交量均线：vol_ma_5, vol_ma_10, vol_ma_50
      - 前高价格：prev_high_252d（252 日滚动最高收盘价）

    结果存储于：data/cache/indicators/{market}/{symbol}.parquet
    """
    logger.info("=" * 60)
    logger.info("【4/5】更新技术指标（EMA / 成交量均线 / 前高）...")
    logger.info("=" * 60)
    t0 = time.time()
    try:
        from stock_ana.data.indicators_store import update_all_indicators
        update_all_indicators()
        elapsed = time.time() - t0
        logger.success(f"✅ 技术指标更新完成 ({elapsed:.0f}s)")
        return {"ok": True, "elapsed": round(elapsed)}
    except Exception as e:
        logger.error(f"❌ 技术指标更新失败: {e}")
        return {"ok": False, "elapsed": round(time.time() - t0), "error": str(e)}


def update_waves() -> dict:
    """
    更新全量 US + HK 股票的 Wave 结构（大浪/子浪）。

    结果存储于：data/cache/wave_structure/{market}/{symbol}.json
    """
    logger.info("=" * 60)
    logger.info("【5/5】更新 Wave 结构（全量 US+HK）...")
    logger.info("=" * 60)
    t0 = time.time()
    try:
        from stock_ana.data.wave_store import update_wave_structures_for_all_us_hk
        result = update_wave_structures_for_all_us_hk()
        elapsed = time.time() - t0
        ok_n = len(result.get("ok", []))
        skip_n = len(result.get("skip", []))
        fail_n = len(result.get("fail", []))
        logger.success(
            f"✅ Wave 结构更新完成 ({elapsed:.0f}s): "
            f"成功 {ok_n}, 跳过 {skip_n}, 失败 {fail_n}"
        )
        return {"ok": True, "elapsed": round(elapsed), "ok_count": ok_n,
                "skip_count": skip_n, "fail_count": fail_n}
    except Exception as e:
        logger.error(f"❌ Wave 结构更新失败: {e}")
        return {"ok": False, "elapsed": round(time.time() - t0), "error": str(e)}


def validate_data() -> dict:
    """
    校验各市场缓存数据是否已更新到近期。

    规则：
    - 扫描 cache/us/ 和 cache/hk/ 所有 parquet
    - 若某只股票最新日期距今超过 5 个自然日，视为"落后"
    - 落后股票占比超过 5% → 整体标记 stale=True，并输出警告列表
    - 结果写入 status.json（validate 字段）
    """
    import pandas as pd
    from stock_ana.config import CACHE_DIR

    today = pd.Timestamp.now().normalize()
    threshold_days = 5   # 宽限5个自然日（含周末+节假日）
    stale_ratio_warn = 0.05  # 超过5%才报警

    markets = {
        "us": CACHE_DIR / "us",
        "hk": CACHE_DIR / "hk",
    }

    all_ok = True
    details = {}

    for market, cache_dir in markets.items():
        if not cache_dir.exists():
            continue
        files = list(cache_dir.glob("*.parquet"))
        if not files:
            continue
        stale = []
        for p in files:
            try:
                df = pd.read_parquet(p, columns=[])
                last = pd.Timestamp(df.index.max()).normalize()
                if (today - last).days > threshold_days:
                    stale.append({"ticker": p.stem, "last_date": last.date().isoformat(),
                                  "days_behind": int((today - last).days)})
            except Exception:
                pass
        ratio = len(stale) / len(files) if files else 0
        is_stale = ratio > stale_ratio_warn
        if is_stale:
            all_ok = False
            logger.warning(
                f"[校验] {market.upper()} 数据落后：{len(stale)}/{len(files)} 只 "
                f"({ratio*100:.1f}%) 超过 {threshold_days} 日未更新"
            )
            for s in stale[:10]:
                logger.warning(f"  {s['ticker']}: 最新 {s['last_date']} (落后 {s['days_behind']} 天)")
            if len(stale) > 10:
                logger.warning(f"  ... 共 {len(stale)} 只，仅显示前10")
        else:
            logger.info(f"[校验] {market.upper()} 数据正常：{len(files)} 只，落后 {len(stale)} 只 ({ratio*100:.1f}%)")
        details[market] = {
            "total": len(files),
            "stale_count": len(stale),
            "stale_ratio": round(ratio, 4),
            "is_stale": is_stale,
            "stale_tickers": [s["ticker"] for s in stale],
        }

    return {"ok": all_ok, "details": details}


def sync_lists() -> dict:
    """同步日常自动生成的 MD 列表文件（默认不含 hk_full_list）。"""
    logger.info("=" * 60)
    logger.info("【列表同步】同步 MD 列表文件 ...")
    logger.info("=" * 60)
    t0 = time.time()
    try:
        from stock_ana.data.list_manager import sync_all_auto_lists
        sync_all_auto_lists()
        elapsed = time.time() - t0
        logger.success(f"✅ 列表同步完成 ({elapsed:.0f}s)")
        return {"ok": True, "elapsed": round(elapsed)}
    except Exception as e:
        logger.error(f"❌ 列表同步失败: {e}")
        return {"ok": False, "elapsed": round(time.time() - t0), "error": str(e)}


# ─────────────────────── 主入口 ───────────────────────


def main():
    """Run the daily update CLI for OHLCV, indicators, waves, and list sync tasks."""
    parser = argparse.ArgumentParser(description="每日股票数据更新")
    parser.add_argument("--us",         action="store_true", help="仅更新美股 OHLCV")
    parser.add_argument("--ndx",        action="store_true", help="仅更新纳指100 OHLCV")
    parser.add_argument("--hk",         action="store_true", help="仅更新港股 OHLCV")
    parser.add_argument("--cn",         action="store_true", help="仅更新A股 OHLCV")
    parser.add_argument("--indicators", action="store_true", help="仅更新技术指标")
    parser.add_argument("--waves",      action="store_true", help="仅更新 Wave 结构（全量 US+HK）")
    parser.add_argument("--lists",      action="store_true", help="仅同步 MD 列表文件")
    args = parser.parse_args()

    # 若无任何参数，执行 1-5 全流程
    run_all = not any([args.us, args.ndx, args.hk, args.cn, args.indicators, args.waves, args.lists])

    logger.info(f"{'=' * 60}")
    logger.info(f"  每日数据更新 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'=' * 60}")

    t_total = time.time()
    results = {}

    # ── Step 1-3：OHLCV 数据更新 ──
    # 注：纳指100 股票已包含在美股全量更新中（cache/us/），不再单独更新。
    # --ndx 参数保留但在 run_all 时跳过，只在显式指定时执行（向后兼容）。
    if run_all or args.us:
        results["美股OHLCV"] = update_us()

    if args.ndx and not run_all:
        results["纳指100OHLCV"] = update_ndx()

    if run_all or args.hk:
        results["港股OHLCV"] = update_hk()

    if run_all or args.cn:
        results["A股OHLCV"] = update_cn()

    # ── Step 4：技术指标 ──
    if run_all or args.indicators:
        results["技术指标"] = update_indicators()

    # ── Step 5：Wave 结构 ──
    if run_all or args.waves:
        results["Wave结构"] = update_waves()

    # ── 列表同步（需显式指定 --lists）──
    if args.lists:
        results["列表同步"] = sync_lists()

    # ── 数据校验（run_all 或有 OHLCV 步骤时自动执行）──
    if run_all or args.us or args.hk:
        results["数据校验"] = validate_data()

    elapsed = time.time() - t_total
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  全部完成 — 总耗时 {elapsed:.0f}s")
    for name, res in results.items():
        ok = res.get("ok", False) if isinstance(res, dict) else bool(res)
        status = "✅" if ok else "❌"
        logger.info(f"    {status} {name}")
    logger.info(f"{'=' * 60}")

    # ── 保存 status.json（供 clawbot 读取）──
    _save_status(results, round(elapsed))


def _save_status(results: dict, total_elapsed: int) -> None:
    """将本次运行结果保存为 data/output/daily_update/{date}/status.json。"""
    today = date.today().isoformat()
    out_dir = PROJECT_ROOT / "data" / "output" / "daily_update" / today
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ok = all(
        (r.get("ok", False) if isinstance(r, dict) else bool(r))
        for r in results.values()
    )

    steps = []
    for name, res in results.items():
        if isinstance(res, dict):
            steps.append({"name": name, **res})
        else:
            steps.append({"name": name, "ok": bool(res)})

    status = {
        "update_date":    today,
        "generated_at":   datetime.now().isoformat(timespec="seconds"),
        "all_ok":         all_ok,
        "total_elapsed":  total_elapsed,
        "steps":          steps,
    }

    path = out_dir / "status.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    logger.info(f"状态文件已保存 → {path}")


if __name__ == "__main__":
    main()
