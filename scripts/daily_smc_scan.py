#!/usr/bin/env python3
"""
SMC OB 每日扫描脚本

用法:
    python scripts/daily_smc_scan.py                   # 默认自选列表
    python scripts/daily_smc_scan.py --holding         # 仅当前持仓
    python scripts/daily_smc_scan.py --us              # 美股科技宇宙
    python scripts/daily_smc_scan.py NVDA AMD MSFT     # 指定股票（自动判断市场）
    python scripts/daily_smc_scan.py --swing_length 3  # 调整摆动灵敏度
    python scripts/daily_smc_scan.py --init            # 首次初始化（重建基线，无事件输出）

输出示例:
    【新生成 OB】1 个
      US:NVDA     ↑看涨  [164.27 ~ 169.45]  形成=2026-05-20  强度=42%

    【OB 失效】1 个
      US:MSFT     ↑看涨  [350.76 ~ 361.97]  消除=2026-05-22

    【价格刺入 OB】2 个
      US:AMD      ↓看跌  [117.54 ~ 122.30]  收盘=119.80
      HK:00981    ↑看涨  [63.52 ~ 71.26]    收盘=68.40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loguru import logger

from stock_ana.config import CACHE_DIR, DATA_DIR
from stock_ana.scan.smc_ob_tracker import run_daily, process_symbol, load_ob_state
from stock_ana.data.market_data import build_watchlist


# ─────────────────────── Watchlist 构建 ──────────────────────────────────────

def _load_holding_watchlist() -> dict:
    """从 data/lists/holding.md 解析当前持仓为 watchlist 格式。"""
    holding_path = DATA_DIR / "lists" / "holding.md"
    watchlist: dict = {}
    if not holding_path.exists():
        logger.warning("holding.md 不存在")
        return watchlist
    with open(holding_path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 3 or parts[0] in ("代码", "---", ""):
                continue
            symbol = parts[0].strip()
            market = parts[1].strip()
            name   = parts[2].strip()
            if symbol and market and not symbol.startswith("-"):
                watchlist[symbol] = (market, name, None, "")
    return watchlist


def _build_us_universe_watchlist() -> dict:
    """读取美股宇宙池（us_universe.csv）。"""
    import pandas as pd
    csv_path = DATA_DIR / "us_universe.csv"
    if not csv_path.exists():
        logger.warning("us_universe.csv 不存在")
        return {}
    df = pd.read_csv(csv_path)
    watchlist = {}
    for _, row in df.iterrows():
        sym  = str(row.get("ticker", row.get("symbol", ""))).strip().upper()
        name = str(row.get("name", sym))
        if sym:
            watchlist[sym] = ("US", name, None, "")
    return watchlist


def _build_symbol_watchlist(symbols: list[str]) -> dict:
    """将用户指定的代码列表转换为 watchlist，自动推断市场。"""
    watchlist: dict = {}
    for sym in symbols:
        sym = sym.upper()
        # 港股：纯数字代码
        if sym.isdigit():
            market = "HK"
        # A股：6位数字开头
        elif sym[:2] in ("00", "30", "60", "68") and len(sym) == 6:
            market = "CN"
        else:
            market = "US"
        # 验证缓存存在
        path = CACHE_DIR / market.lower() / f"{sym}.parquet"
        if not path.exists():
            logger.warning(f"缓存不存在: {sym} ({market})，跳过")
            continue
        watchlist[sym] = (market, sym, None, "")
    return watchlist


# ─────────────────────── 输出格式化 ──────────────────────────────────────────

def _print_events(results: dict[str, list[dict]]) -> None:
    all_events = [e for evts in results.values() for e in evts]
    new_obs    = [e for e in all_events if e["event"] == "new_ob"]
    mitigated  = [e for e in all_events if e["event"] == "mitigated"]
    touched    = [e for e in all_events if e["event"] == "touched"]

    width = 62
    print(f"\n{'=' * width}")
    print(f"  SMC OB 每日扫描  |  共 {len(all_events)} 个事件")
    print(f"{'=' * width}")

    if new_obs:
        print(f"\n【新生成 OB】{len(new_obs)} 个")
        for e in sorted(new_obs, key=lambda x: (x["market"], x["symbol"])):
            tag = "↑看涨" if e["direction"] == 1 else "↓看跌"
            print(
                f"  {e['market']}:{e['symbol']:<8s} {tag}  "
                f"[{e['bottom']:.2f} ~ {e['top']:.2f}]  "
                f"形成={e['formed_date']}  强度={e.get('percentage', 0):.0f}%"
            )

    if mitigated:
        print(f"\n【OB 失效】{len(mitigated)} 个")
        for e in sorted(mitigated, key=lambda x: (x["market"], x["symbol"])):
            tag = "↑看涨" if e["direction"] == 1 else "↓看跌"
            print(
                f"  {e['market']}:{e['symbol']:<8s} {tag}  "
                f"[{e['bottom']:.2f} ~ {e['top']:.2f}]  "
                f"形成={e['formed_date']}  消除={e.get('mitigated_date', '?')}"
            )

    if touched:
        print(f"\n【价格刺入 OB】{len(touched)} 个")
        for e in sorted(touched, key=lambda x: (x["market"], x["symbol"])):
            tag = "↑看涨" if e["direction"] == 1 else "↓看跌"
            print(
                f"  {e['market']}:{e['symbol']:<8s} {tag}  "
                f"[{e['bottom']:.2f} ~ {e['top']:.2f}]  "
                f"收盘={e.get('current_close', 0):.2f}  "
                f"(H={e.get('current_high', 0):.2f} L={e.get('current_low', 0):.2f})  "
                f"形成={e['formed_date']}"
            )

    if not all_events:
        print("\n  今日无 OB 事件")

    print(f"\n  覆盖 {len(results)} 只股票有事件")
    print(f"{'=' * width}\n")


# ─────────────────────── 主入口 ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SMC OB 每日追踪扫描")
    parser.add_argument(
        "symbols", nargs="*",
        help="指定股票代码（不填则按 --holding / --us / 默认自选）",
    )
    parser.add_argument("--holding", action="store_true", help="仅扫描当前持仓")
    parser.add_argument("--us",      action="store_true", help="扫描美股宇宙池")
    parser.add_argument(
        "--init", action="store_true",
        help="强制重建所有 OB 基线（删除旧状态文件后重跑首次初始化）",
    )
    parser.add_argument("--swing_length", type=int, default=5, metavar="N",
                        help="摆动点确认窗口（默认 5）")
    parser.add_argument("--close_mitigation", action="store_true",
                        help="以收盘价判断 OB 消除（默认用 high/low）")
    args = parser.parse_args()

    # 构建 watchlist
    if args.symbols:
        watchlist = _build_symbol_watchlist(args.symbols)
    elif args.holding:
        watchlist = _load_holding_watchlist()
    elif args.us:
        watchlist = _build_us_universe_watchlist()
    else:
        watchlist = build_watchlist()

    if not watchlist:
        logger.error("watchlist 为空，退出")
        sys.exit(1)

    # --init：删除旧状态文件，强制首次初始化
    if args.init:
        logger.info("--init 模式：清除指定股票的 OB 状态文件，重建基线")
        for sym, meta in watchlist.items():
            mkt = meta[0].lower()
            p = OB_STATE_DIR / mkt / f"{sym}.json"
            if p.exists():
                p.unlink()
                logger.debug(f"已删除: {p.name}")

    logger.info(f"扫描列表: {len(watchlist)} 只股票  swing_length={args.swing_length}")

    results = run_daily(
        watchlist=watchlist,
        swing_length=args.swing_length,
        close_mitigation=args.close_mitigation,
    )

    _print_events(results)


if __name__ == "__main__":
    # 导入状态目录常量（需在 sys.path 设置之后）
    from stock_ana.config import CACHE_DIR
    OB_STATE_DIR = CACHE_DIR / "smc_ob_state"
    main()
