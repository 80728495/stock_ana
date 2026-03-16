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
    python -m stock_ana.momentum_detector --update          # 更新价格数据
    python -m stock_ana.momentum_detector --scan            # 扫描异动
    python -m stock_ana.momentum_detector --ticker NVDA     # 单只检测
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.config import DATA_DIR, CACHE_DIR


# ═══════════════════ 路径 ═══════════════════

US_CACHE_DIR = CACHE_DIR / "us"
PROFILES_FILE = DATA_DIR / "us_sec_profiles.csv"


# ═══════════════════ 数据下载参数 ═══════════════════

_DL_PERIOD = "1y"       # 下载周期（需覆盖 200MA）
_STALE_HOURS = 16       # 缓存过期时间（小时）


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


def _fetch_akshare(ticker: str) -> pd.DataFrame | None:
    """
    通过 akshare (新浪源) 下载单只美股的日线数据。
    返回 OHLCV DataFrame，index 为 date。
    """
    import akshare as ak

    try:
        df = ak.stock_us_daily(symbol=ticker, adjust="qfq")
        if df is None or df.empty or len(df) < 10:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.columns = [c.lower() for c in df.columns]
        keep = [c for c in ["open", "high", "low", "close", "volume"]
                if c in df.columns]
        if len(keep) < 5:
            return None
        # 只保留最近 1 年数据（约 252 个交易日，留 300 余量）
        df = df[keep].dropna().iloc[-300:]
        return df
    except Exception as e:
        logger.debug(f"  {ticker} akshare 下载失败: {e}")
        return None


def update_us_price_data(
    tickers: list[str] | None = None,
    force: bool = False,
) -> dict:
    """
    批量下载/更新美股价格数据（通过 akshare 新浪源），缓存为 parquet。
    支持断点续传：已有缓存的 ticker 自动跳过。

    Args:
        tickers: ticker 列表，None 则使用完整 universe
        force: 强制重新下载

    Returns:
        {"updated": int, "skipped": int, "failed": int}
    """
    US_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if tickers is None:
        profiles = pd.read_csv(PROFILES_FILE, encoding="utf-8-sig")
        tickers = profiles["ticker"].tolist()

    # 筛选需要下载的 ticker
    stale_cutoff = datetime.now() - timedelta(hours=_STALE_HOURS)
    ndx_dir = CACHE_DIR / "ndx100"
    to_download = []
    skipped = 0

    for t in tickers:
        us_file = US_CACHE_DIR / f"{t}.parquet"
        if not force and us_file.exists():
            mtime = datetime.fromtimestamp(us_file.stat().st_mtime)
            if mtime > stale_cutoff:
                skipped += 1
                continue
        ndx_file = ndx_dir / f"{t}.parquet"
        if not force and ndx_file.exists():
            skipped += 1
            continue
        to_download.append(t)

    if not to_download:
        logger.info(f"全部 {skipped} 只股票缓存有效，无需更新")
        return {"updated": 0, "skipped": skipped, "failed": 0}

    logger.info(f"需要下载: {len(to_download)} 只，已跳过: {skipped} 只")

    updated = 0
    failed = 0
    failed_tickers = []

    for idx, t in enumerate(to_download):
        if (idx + 1) % 50 == 0 or idx == 0:
            logger.info(f"进度: {idx + 1}/{len(to_download)} "
                        f"[成功 {updated}, 失败 {failed}]")

        df = _fetch_akshare(t)

        if df is not None and len(df) >= 10:
            df.to_parquet(US_CACHE_DIR / f"{t}.parquet")
            updated += 1
        else:
            failed += 1
            failed_tickers.append(t)

        # 间隔 0.3s 避免新浪限流
        time.sleep(0.3)

    logger.info(f"更新完成: 成功 {updated}, 跳过 {skipped}, 失败 {failed}")
    if failed_tickers:
        logger.info(f"失败列表（前30）: {failed_tickers[:30]}")
    return {"updated": updated, "skipped": skipped, "failed": failed}


def _load_price(ticker: str) -> pd.DataFrame | None:
    """加载单只股票的缓存价格数据（优先 US 缓存，回退到 NDX100 缓存）"""
    for d in [US_CACHE_DIR, CACHE_DIR / "ndx100"]:
        f = d / f"{ticker}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            df.columns = [c.lower() for c in df.columns]
            return df
    return None


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
    avg_vol_recent = recent["volume"].mean()
    avg_vol_ref = ref["volume"].mean()
    vol_ratio = avg_vol_recent / avg_vol_ref if avg_vol_ref > 0 else 0

    price_up = recent["close"].iloc[-1] >= df["close"].iloc[-lookback - 1]
    vs = 0.0
    if price_up:
        if vol_ratio >= _VOL_RATIO_STRONG:
            vs = 2.0
        elif vol_ratio >= 2.0:
            vs = 1.5
        elif vol_ratio >= _VOL_RATIO_MILD:
            vs = 1.0
    signals["vol_surge"] = {"ratio": round(vol_ratio, 2), "score": vs}
    total += vs

    # ── 2. 超预期涨幅 (0–2) ────────────────────────────
    ret = (recent["close"].iloc[-1] / df["close"].iloc[-lookback - 1]) - 1
    daily_std = ref["close"].pct_change().dropna().std()
    expected_std = daily_std * np.sqrt(lookback) if daily_std > 0 else 1e-9
    z = ret / expected_std

    rs = 0.0
    if z >= _RETURN_Z_STRONG:
        rs = 2.0
    elif z >= 2.0:
        rs = 1.5
    elif z >= _RETURN_Z_MILD:
        rs = 1.0
    signals["abnormal_return"] = {
        "z_score": round(z, 2),
        "pct": round(ret * 100, 2),
        "score": rs,
    }
    total += rs

    # ── 3. 创新高突破 (0–2) ────────────────────────────
    cur_close = df["close"].iloc[-1]

    # 检测窗口前的 20/60 日最高价
    h20_end = n - lookback
    h20_start = max(0, h20_end - 20)
    h60_start = max(0, h20_end - 60)
    high_20d = df["high"].iloc[h20_start:h20_end].max()
    high_60d = df["high"].iloc[h60_start:h20_end].max()

    bl, bs = "", 0.0
    if cur_close > high_60d:
        bl, bs = "60d_high", 2.0
    elif cur_close > high_20d:
        bl, bs = "20d_high", 1.0
    signals["breakout"] = {"level": bl, "score": bs}
    total += bs

    # ── 4. 跳空高开 (0–1) ────────────────────────────
    max_gap = 0.0
    for i in range(n - lookback, n):
        prev_c = df["close"].iloc[i - 1]
        if prev_c > 0:
            gap = (df["open"].iloc[i] - prev_c) / prev_c
            if gap > max_gap:
                max_gap = gap

    # 线性映射：2% → 0 分，4% → 1 分
    gs = min(1.0, max_gap / 0.04) if max_gap >= _GAP_THRESH else 0.0
    signals["gap_up"] = {"max_gap_pct": round(max_gap * 100, 2), "score": round(gs, 2)}
    total += gs

    # ── 5. 均线突破 (0–1) ────────────────────────────
    ms = 0.0
    a50, a200 = False, False
    prev_close = df["close"].iloc[-lookback - 1]

    if n >= 55:
        ma50 = df["close"].iloc[-50:].mean()
        a50 = cur_close > ma50
        if a50 and prev_close < ma50:
            ms += 0.5

    if n >= 205:
        ma200 = df["close"].iloc[-200:].mean()
        a200 = cur_close > ma200
        if a200 and prev_close < ma200:
            ms += 0.5

    signals["ma_breakout"] = {
        "above_50ma": a50,
        "above_200ma": a200,
        "score": ms,
    }
    total += ms

    # ── 6. 量价共振 (0–2) ────────────────────────────
    #     仅在窗口期整体不下跌时计分
    avg_vol_base = ref["volume"].mean()
    accum_days = 0
    for i in range(n - lookback, n):
        row = df.iloc[i]
        if row["close"] > row["open"] and row["volume"] > avg_vol_base * _ACCUM_VOL_FACTOR:
            accum_days += 1

    acs = 0.0
    if price_up:
        if accum_days >= lookback:
            acs = 2.0
        elif accum_days >= 3:
            acs = 1.0
        elif accum_days >= 2:
            acs = 0.5
    signals["accumulation"] = {"days": accum_days, "score": acs}
    total += acs

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
