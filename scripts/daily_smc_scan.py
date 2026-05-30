#!/usr/bin/env python3
"""
SMC OB 每日扫描脚本

与 vegas_mid_daily_scan 使用完全相同的三个列表来源（us_tech_list.md /
hk_techman.md / cn_hightech_list.md）。

用法:
    # ── 首次历史初始化（全量，只运行一次）──
    python scripts/daily_smc_scan.py --list all --init

    # ── 每日增量更新（三市场合并）──
    python scripts/daily_smc_scan.py --list all

    # ── 单市场 ──
    python scripts/daily_smc_scan.py --list us
    python scripts/daily_smc_scan.py --list hk
    python scripts/daily_smc_scan.py --list cn

    # ── 其他便捷模式 ──
    python scripts/daily_smc_scan.py --list holding    # 仅当前持仓
    python scripts/daily_smc_scan.py --list watchlist  # 自选列表（默认）
    python scripts/daily_smc_scan.py NVDA AMD MSFT     # 指定股票

    # ── 参数调整 ──
    python scripts/daily_smc_scan.py --list all --swing_length 3
    python scripts/daily_smc_scan.py --list all --init --swing_length 5

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
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loguru import logger

LOG_DIR = ROOT / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(
    LOG_DIR / "smc_ob_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    encoding="utf-8",
    enqueue=True,
)

from stock_ana.config import CACHE_DIR, DATA_DIR
from stock_ana.scan.smc_ob_tracker import run_daily, load_ob_state
from stock_ana.data.market_data import build_watchlist

OB_STATE_DIR  = CACHE_DIR / "smc_ob_state"
SMC_OUT_DIR   = ROOT / "data" / "output" / "smc_ob_scan"
SMC_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────── Watchlist 构建 ──────────────────────────────────────

def _build_mid_vegas_watchlist(list_mode: str) -> dict:
    """与 vegas_mid_daily_scan 完全相同的列表来源。

    list_mode: "us"  → us_tech_list.md（科技/通信板块）
               "hk"  → hk_techman.md（港股高科技与制造业）
               "cn"  → cn_hightech_list.md（A股高新技术）
    """
    from stock_ana.scan.vegas_mid_scan import (
        _build_us_universe_watchlist,
        _build_hk_techman_watchlist,
        _build_cn_hightech_list,
    )
    builders = {
        "us": _build_us_universe_watchlist,
        "hk": _build_hk_techman_watchlist,
        "cn": _build_cn_hightech_list,
    }
    return builders[list_mode]()


def _load_futu_watchlist() -> dict:
    """从 data/lists/futu_watchlist.md 解析富途自选股，跳过无缓存的指数/ETF。"""
    futu_path = DATA_DIR / "lists" / "futu_watchlist.md"
    watchlist: dict = {}
    if not futu_path.exists():
        logger.warning("futu_watchlist.md 不存在")
        return watchlist

    market: str | None = None
    with open(futu_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("## 港股"):
                market = "HK"
                continue
            elif line.startswith("## 美股"):
                market = "US"
                continue
            elif line.startswith("## 大A"):
                market = "CN"
                continue
            if not line.startswith("|") or market is None:
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 2 or parts[0] in ("代码", "---", ""):
                continue
            symbol = parts[0].strip()
            name = parts[1].strip() if len(parts) > 1 else symbol
            if not symbol or symbol.startswith("-"):
                continue
            # 检查缓存存在（跳过无数据的指数/ETF/期货）
            p = CACHE_DIR / market.lower() / f"{symbol}.parquet"
            if not p.exists():
                if market == "US":
                    p2 = CACHE_DIR / "ndx100" / f"{symbol}.parquet"
                    if p2.exists():
                        watchlist[symbol] = (market, name, None, "")
                        continue
                logger.debug(f"跳过无缓存: {symbol} ({market})")
                continue
            watchlist[symbol] = (market, name, None, "")

    logger.info(f"futu_watchlist: {len(watchlist)} 只股票（已过滤无缓存标的）")
    return watchlist


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


def _build_symbol_watchlist(symbols: list[str]) -> dict:
    """将用户指定的代码列表转换为 watchlist，自动推断市场。"""
    watchlist: dict = {}
    for sym in symbols:
        sym = sym.upper()
        if sym.isdigit():
            market = "HK"
        elif sym[:2] in ("00", "30", "60", "68") and len(sym) == 6:
            market = "CN"
        else:
            market = "US"
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


def _save_events(results: dict[str, list[dict]], list_mode: str) -> Path:
    """把本次扫描的所有事件写入 JSON 文件，便于事后查询。

    输出路径: data/output/smc_ob_scan/{date}_{list_mode}_events.json
    """
    today = str(date.today())
    out_path = SMC_OUT_DIR / f"{today}_{list_mode}_events.json"
    all_events = [e for evts in results.values() for e in evts]
    payload = {
        "date":       today,
        "list_mode":  list_mode,
        "total":      len(all_events),
        "new_ob":     [e for e in all_events if e["event"] == "new_ob"],
        "mitigated":  [e for e in all_events if e["event"] == "mitigated"],
        "touched":    [e for e in all_events if e["event"] == "touched"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"扫描结果已写入: {out_path}")
    return out_path


def _show_active_obs(watchlist: dict, direction_filter: int | None = None) -> None:
    """直接读取状态文件，列出当前所有活跃 OB（不重新扫描）。

    direction_filter: 1=只看看涨, -1=只看看跌, None=全部
    """
    rows: list[dict] = []
    for sym, meta in watchlist.items():
        mkt = meta[0]
        state = load_ob_state(sym, mkt.lower())
        if state["last_updated"] is None:
            continue
        for ob in state.get("obs", {}).values():
            if ob.get("status") not in ("active", "touched"):
                continue
            if direction_filter is not None and ob["direction"] != direction_filter:
                continue
            rows.append({
                "market":      mkt,
                "symbol":      sym,
                "direction":   ob["direction"],
                "top":         ob["top"],
                "bottom":      ob["bottom"],
                "formed_date": ob["formed_date"],
                "status":      ob["status"],
                "percentage":  ob.get("percentage", 0),
                "last_updated": state["last_updated"],
            })

    dir_label = {1: "↑看涨", -1: "↓看跌", None: "全部"}[direction_filter]
    width = 90
    print(f"\n{'=' * width}")
    print(f"  SMC 活跃 OB 查询  |  {dir_label}  |  共 {len(rows)} 个")
    print(f"{'=' * width}")

    if not rows:
        print("  (无活跃 OB)")
    else:
        rows.sort(key=lambda x: (x["direction"], x["market"], x["symbol"]))
        bear = [r for r in rows if r["direction"] == -1]
        bull = [r for r in rows if r["direction"] == 1]
        for section_label, section in (("↓看跌 OB", bear), ("↑看涨 OB", bull)):
            if not section:
                continue
            print(f"\n【{section_label}】{len(section)} 个")
            print(f"  {'市场':4} {'代码':10} {'OB区间':25} {'形成日':12} {'状态':8} {'强度':>6} {'数据日期':12}")
            print(f"  {'-'*82}")
            for r in section:
                rng = f"[{r['bottom']:.2f} ~ {r['top']:.2f}]"
                print(
                    f"  {r['market']:4} {r['symbol']:10} {rng:25} "
                    f"{r['formed_date']:12} {r['status']:8} {r['percentage']:5.0f}%  "
                    f"{r['last_updated']:12}"
                )

    print(f"{'=' * width}\n")


# ─────────────────────── 内部工具 ───────────────────────────────────────────

def _clear_state(watchlist: dict) -> None:
    """删除 watchlist 中所有股票的 OB 状态文件（--init 前置步骤）。"""
    removed = 0
    for sym, meta in watchlist.items():
        mkt = meta[0].lower()
        p = OB_STATE_DIR / mkt / f"{sym}.json"
        if p.exists():
            p.unlink()
            removed += 1
    logger.info(f"--init：已清除 {removed} 个历史 OB 状态文件")


def _run_market(
    list_mode: str,
    swing_length: int,
    close_mitigation: bool,
    init: bool,
) -> dict[str, list[dict]]:
    """构建单市场 watchlist，（可选）清除旧状态后运行增量更新。"""
    label_map = {"us": "美股科技", "hk": "港股高科技与制造业", "cn": "A股高新技术"}
    logger.info(f"{'=' * 60}")
    logger.info(f"  [{list_mode.upper()}] {label_map.get(list_mode, list_mode)}")
    logger.info(f"{'=' * 60}")
    watchlist = _build_mid_vegas_watchlist(list_mode)
    if not watchlist:
        logger.warning(f"[{list_mode.upper()}] watchlist 为空，跳过")
        return {}
    if init:
        _clear_state(watchlist)
    return run_daily(
        watchlist=watchlist,
        swing_length=swing_length,
        close_mitigation=close_mitigation,
    )


# ─────────────────────── 主入口 ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SMC OB 每日追踪扫描")
    parser.add_argument(
        "symbols", nargs="*",
        help="指定股票代码（不填则按 --list 决定扫描列表）",
    )
    parser.add_argument(
        "--list",
        dest="list_mode",
        choices=["us", "hk", "cn", "all", "watchlist", "holding", "futu"],
        default="watchlist",
        help=(
            "扫描列表（默认 watchlist）：\n"
            "  us        → us_tech_list.md（科技/通信板块）\n"
            "  hk        → hk_techman.md（港股高科技与制造业）\n"
            "  cn        → cn_hightech_list.md（A股高新技术）\n"
            "  all       → us + hk + cn 三市场合并（每日推荐）\n"
            "  futu      → futu_watchlist.md（富途自选股，过滤无缓存ETF/指数）\n"
            "  watchlist → 自选列表（默认）\n"
            "  holding   → 当前持仓"
        ),
    )
    parser.add_argument(
        "--init", action="store_true",
        help="清除旧 OB 状态文件并重建基线（首次历史初始化，不产出增量事件）",
    )
    parser.add_argument("--swing_length", type=int, default=5, metavar="N",
                        help="摆动点确认窗口（默认 5）")
    parser.add_argument("--close_mitigation", action="store_true",
                        help="以收盘价判断 OB 消除（默认用 high/low）")
    parser.add_argument("--show-active", action="store_true",
                        help="查询当前状态文件中的活跃 OB（不重新扫描）")
    parser.add_argument("--bearish", action="store_true",
                        help="配合 --show-active，只显示看跌 OB")
    parser.add_argument("--bullish", action="store_true",
                        help="配合 --show-active，只显示看涨 OB")
    args = parser.parse_args()

    swing_length     = args.swing_length
    close_mitigation = args.close_mitigation
    init             = args.init
    show_active      = args.show_active
    dir_filter       = -1 if args.bearish else (1 if args.bullish else None)

    if init:
        logger.info("【--init 模式】将清除旧状态文件，重建全量 OB 基线（本次无增量事件）")

    # ── 构建 watchlist（所有路径都需要）────────────────────────────────────────
    if args.symbols:
        watchlist = _build_symbol_watchlist(args.symbols)
    elif args.list_mode == "holding":
        watchlist = _load_holding_watchlist()
    elif args.list_mode == "futu":
        watchlist = _load_futu_watchlist()
    elif args.list_mode in ("us", "hk", "cn", "all"):
        watchlist = None  # 多市场路径单独处理
    else:
        watchlist = build_watchlist()

    # ── --show-active：直接读状态文件，不重新扫描 ─────────────────────────────
    if show_active:
        if watchlist is None:
            # all/us/hk/cn 合并构建
            watchlist = {}
            for mkt in (["us", "hk", "cn"] if args.list_mode == "all" else [args.list_mode]):
                watchlist.update(_build_mid_vegas_watchlist(mkt))
        if not watchlist:
            logger.error("watchlist 为空，退出")
            sys.exit(1)
        _show_active_obs(watchlist, direction_filter=dir_filter)
        return

    # ── 正常扫描路径 ──────────────────────────────────────────────────────────
    if args.symbols:
        if not watchlist:
            logger.error("watchlist 为空，退出")
            sys.exit(1)
        if init:
            _clear_state(watchlist)
        results = run_daily(watchlist=watchlist,
                            swing_length=swing_length,
                            close_mitigation=close_mitigation)
        _print_events(results)
        _save_events(results, "symbols")
        return

    if args.list_mode == "all":
        all_results: dict[str, list[dict]] = {}
        for mkt in ("us", "hk", "cn"):
            r = _run_market(mkt, swing_length, close_mitigation, init)
            all_results.update(r)
        _print_events(all_results)
        _save_events(all_results, "all")
        return

    if args.list_mode in ("us", "hk", "cn"):
        results = _run_market(args.list_mode, swing_length, close_mitigation, init)
        _print_events(results)
        _save_events(results, args.list_mode)
        return

    # ── watchlist / holding / futu ────────────────────────────────────────────
    if not watchlist:
        logger.error("watchlist 为空，退出")
        sys.exit(1)

    logger.info(f"扫描列表: {len(watchlist)} 只股票  swing_length={swing_length}")
    if init:
        _clear_state(watchlist)
    results = run_daily(watchlist=watchlist,
                        swing_length=swing_length,
                        close_mitigation=close_mitigation)
    _print_events(results)
    _save_events(results, args.list_mode)


if __name__ == "__main__":
    main()
