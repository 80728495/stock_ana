"""
股票行情异动检测模块 (Momentum Detector)

检测单只股票是否进入异动状态（量价突变），为板块行情分析提供前置信号。
条件设计偏宽松，保证较高召回率。

检测维度（六类信号，总分 0–10）:
  1. 量能放大   — 近 N 日均量 vs 参考期均量          (0–2)
  2. 超预期涨幅 — 近 N 日收益率 Z-score              (0–2)
  3. 创新高突破 — 突破 20 日 / 60 日前高点            (0–2)
  4. 跳空高开   — 窗口内出现显著缺口                  (0–1)
  5. 均线突破   — 近期上穿 50MA / 200MA               (0–1)
  6. 量价共振   — 放量阳线天数                        (0–2)

用法:
    python -m stock_ana.strategies.impl.momentum_detector --update          # 更新价格数据
    python -m stock_ana.strategies.impl.momentum_detector --scan            # 扫描异动
    python -m stock_ana.strategies.impl.momentum_detector --ticker NVDA     # 单只检测
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.config import DATA_DIR
from stock_ana.data.fetcher import update_us_price_data
from stock_ana.data.market_data import load_symbol_data
from stock_ana.strategies.primitives.momentum import (
    score_abnormal_return,
    score_accumulation,
    score_breakout,
    score_gap_up,
    score_ma_breakout,
    score_volume_surge,
)


# ═══════════════════ 路径 ═══════════════════

PROFILES_FILE = DATA_DIR / "us_sec_profiles.csv"


# ═══════════════════ 数据下载参数 ═══════════════════

_DL_PERIOD = "3y"       # 下载周期（需覆盖 200MA + 形态检测最少 3 年）
_STALE_HOURS = 16       # 缓存过期时间（小时）
_HISTORY_BARS = 756     # 保留最近 K 线数（3 年 ≈ 756 交易日）


# ═══════════════════ 检测参数 ═══════════════════

_LOOKBACK = 5           # 异动检测窗口（交易日）
_REF_PERIOD = 50        # 参考基准期长度
_MIN_HISTORY = 60       # 最少历史天数

# 信号阈值
_VOL_RATIO_MILD = 1.8
_VOL_RATIO_STRONG = 3.0
_RETURN_Z_MILD = 1.5
_RETURN_Z_STRONG = 3.0
_GAP_THRESH = 0.02      # 2%
_ACCUM_VOL_FACTOR = 1.2 # 放量阳线的量比倍数

# 触发分数线
_TRIGGER_SCORE = 3.0


# ═══════════════════ 数据管理 ═══════════════════


def _load_price(ticker: str) -> pd.DataFrame | None:
    """加载单只股票的缓存价格数据（优先 US 缓存，回退到 NDX100 缓存）"""
    return load_symbol_data(ticker, universe="us+ndx100")


# ═══════════════════ 信号检测 ═══════════════════

def detect_momentum(df: pd.DataFrame, lookback: int = _LOOKBACK) -> dict:
    """
    检测单只股票的行情异动信号。

    Args:
        df: OHLCV DataFrame，列名 [open, high, low, close, volume]，按日期升序
        lookback: 检测窗口天数

    Returns:
        {
            "triggered": bool,
            "score": float,
            "signals": {
                "vol_surge":       {"ratio": float, "score": float},
                "abnormal_return": {"z_score": float, "pct": float, "score": float},
                "breakout":        {"level": str, "score": float},
                "gap_up":          {"max_gap_pct": float, "score": float},
                "ma_breakout":     {"above_50ma": bool, "above_200ma": bool, "score": float},
                "accumulation":    {"days": int, "score": float},
            },
        }
    """
    empty = {"triggered": False, "score": 0.0, "signals": {}}

    if df is None or len(df) < _MIN_HISTORY:
        return {**empty, "reason": "insufficient_data"}

    n = len(df)
    signals = {}
    total = 0.0

    # 参考期范围
    ref_start = max(0, n - _REF_PERIOD - lookback)
    ref_end = n - lookback
    recent = df.iloc[-lookback:]
    ref = df.iloc[ref_start:ref_end]

    if len(ref) < 20:
        return {**empty, "reason": "insufficient_ref"}

    # ── 1. 量能放大 (0–2) ──────────────────────────────
    #     仅在价格同期不跌时计分，避免下跌放量误判
    price_up = recent["close"].iloc[-1] >= df["close"].iloc[-lookback - 1]
    signals["vol_surge"] = score_volume_surge(
        recent,
        ref,
        price_up=price_up,
        mild_ratio=_VOL_RATIO_MILD,
        strong_ratio=_VOL_RATIO_STRONG,
    )
    total += signals["vol_surge"]["score"]

    # ── 2. 超预期涨幅 (0–2) ────────────────────────────
    signals["abnormal_return"] = score_abnormal_return(
        df,
        ref,
        lookback=lookback,
        mild_z=_RETURN_Z_MILD,
        strong_z=_RETURN_Z_STRONG,
    )
    total += signals["abnormal_return"]["score"]

    # ── 3. 创新高突破 (0–2) ────────────────────────────
    signals["breakout"] = score_breakout(df, lookback=lookback)
    total += signals["breakout"]["score"]

    # ── 4. 跳空高开 (0–1) ────────────────────────────
    signals["gap_up"] = score_gap_up(df, lookback=lookback, gap_thresh=_GAP_THRESH)
    total += signals["gap_up"]["score"]

    # ── 5. 均线突破 (0–1) ────────────────────────────
    signals["ma_breakout"] = score_ma_breakout(df, lookback=lookback)
    total += signals["ma_breakout"]["score"]

    # ── 6. 量价共振 (0–2) ────────────────────────────
    #     仅在窗口期整体不下跌时计分
    signals["accumulation"] = score_accumulation(
        df,
        ref,
        lookback=lookback,
        price_up=price_up,
        accum_vol_factor=_ACCUM_VOL_FACTOR,
    )
    total += signals["accumulation"]["score"]

    # ── 汇总 ──
    total = round(total, 2)
    return {
        "triggered": total >= _TRIGGER_SCORE,
        "score": total,
        "signals": signals,
    }


# ═══════════════════ 批量扫描 ═══════════════════

def scan_universe(
    tickers: list[str] | None = None,
    lookback: int = _LOOKBACK,
    min_score: float = _TRIGGER_SCORE,
    update: bool = False,
) -> pd.DataFrame:
    """
    扫描全部（或指定）美股的异动信号。

    Args:
        tickers: ticker 列表，None 则使用完整 universe
        lookback: 检测窗口天数
        min_score: 最低触发分数（0 则返回全部）
        update: 扫描前是否先更新价格数据

    Returns:
        DataFrame 按 score 降序，包含异动信号明细
    """
    profiles = pd.read_csv(PROFILES_FILE, encoding="utf-8-sig")
    if tickers is None:
        tickers = profiles["ticker"].tolist()

    if update:
        update_us_price_data(tickers)

    rows = []
    ok, skip = 0, 0
    for t in tickers:
        df = _load_price(t)
        if df is None:
            skip += 1
            continue

        result = detect_momentum(df, lookback=lookback)
        ok += 1

        if min_score > 0 and result["score"] < min_score:
            continue

        sig = result["signals"]
        rows.append({
            "ticker": t,
            "score": result["score"],
            "triggered": result["triggered"],
            "vol_ratio": sig.get("vol_surge", {}).get("ratio", 0),
            "return_pct": sig.get("abnormal_return", {}).get("pct", 0),
            "z_score": sig.get("abnormal_return", {}).get("z_score", 0),
            "breakout": sig.get("breakout", {}).get("level", ""),
            "gap_pct": sig.get("gap_up", {}).get("max_gap_pct", 0),
            "accum_days": sig.get("accumulation", {}).get("days", 0),
            "above_50ma": sig.get("ma_breakout", {}).get("above_50ma", False),
            "above_200ma": sig.get("ma_breakout", {}).get("above_200ma", False),
        })

    logger.info(f"扫描完成: 检测 {ok} 只，跳过 {skip} 只（无数据），"
                f"异动 {sum(1 for r in rows if r['triggered'])} 只")

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # 合并公司信息
    info_cols = ["ticker", "company_name", "sector", "sic_code",
                 "sic_description", "sub_label"]
    info = profiles[[c for c in info_cols if c in profiles.columns]]
    out = out.merge(info, on="ticker", how="left")

    # 重排列并按分数降序
    front = ["ticker", "company_name", "sector", "sic_description", "sub_label",
             "score", "triggered"]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest].sort_values("score", ascending=False)

    return out


# ═══════════════════ CLI ═══════════════════

def main():
    """CLI entrypoint for market-wide or single-symbol momentum analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="股票行情异动检测")
    parser.add_argument("--update", action="store_true", help="更新价格数据")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    parser.add_argument("--scan", action="store_true", help="扫描全部异动")
    parser.add_argument("--ticker", type=str, help="检测单只股票")
    parser.add_argument("--lookback", type=int, default=_LOOKBACK,
                        help=f"检测窗口天数 (默认 {_LOOKBACK})")
    parser.add_argument("--top", type=int, default=50, help="显示前 N 只 (默认 50)")
    parser.add_argument("--all", action="store_true", help="显示全部结果（不限 score）")
    args = parser.parse_args()

    if args.update:
        update_us_price_data(force=args.force)
        if not args.scan and not args.ticker:
            return

    if args.ticker:
        # 单只股票检测
        ticker = args.ticker.upper()
        df = _load_price(ticker)
        if df is None:
            logger.info(f"无缓存数据，正在下载 {ticker} ...")
            update_us_price_data([ticker], force=True)
            df = _load_price(ticker)

        if df is None:
            logger.error(f"无法获取 {ticker} 的价格数据")
            sys.exit(1)

        result = detect_momentum(df, lookback=args.lookback)
        print(f"\n{'='*60}")
        print(f"  {ticker}  异动检测  (lookback={args.lookback}d)")
        print(f"{'='*60}")
        print(f"  综合分: {result['score']:.1f} / 10")
        print(f"  触发:   {'✓ 是' if result['triggered'] else '✗ 否'}")
        print()
        for name, sig in result.get("signals", {}).items():
            score = sig.get("score", 0)
            flag = "●" if score > 0 else "○"
            detail_parts = [f"{k}={v}" for k, v in sig.items() if k != "score"]
            detail = ", ".join(detail_parts)
            print(f"  {flag} {name:20s}  {score:4.1f}  ({detail})")
        print()
        return

    if args.scan:
        min_score = 0.0 if args.all else _TRIGGER_SCORE
        result = scan_universe(lookback=args.lookback, min_score=min_score)

        if result.empty:
            print("未检测到异动股票")
            return

        # 截取 top N
        display = result.head(args.top)

        # 格式化输出
        show_cols = ["ticker", "sector", "sic_description", "sub_label",
                     "score", "vol_ratio", "return_pct", "z_score",
                     "breakout", "gap_pct", "accum_days"]
        show_cols = [c for c in show_cols if c in display.columns]
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 200)
        pd.set_option("display.max_colwidth", 30)

        triggered = result[result["triggered"]]
        print(f"\n异动股票: {len(triggered)} 只 (score ≥ {_TRIGGER_SCORE})")
        print(f"显示前 {min(args.top, len(display))} 只:\n")
        print(display[show_cols].to_string(index=False))

        # 按板块汇总
        if not triggered.empty:
            print(f"\n{'─'*60}")
            print("板块分布:")
            sector_counts = triggered.groupby("sector").size().sort_values(ascending=False)
            for sector, count in sector_counts.items():
                avg_score = triggered[triggered["sector"] == sector]["score"].mean()
                print(f"  {sector:30s}  {count:3d} 只  平均分 {avg_score:.1f}")
        print()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
