#!/usr/bin/env python3
"""Vegas Long Touch 历史信号回顾 + 图表生成

对富途关注列表中的美股标的做历史 long_touch 信号回测：
  1. 扫描最近 N 个交易日的 long_touch 信号
  2. 对每个信号，用 ZigZag 结构找到入场后的第一个拐点，计算到拐点的涨跌幅
  3. 生成每条信号的图表

用法:
    python scripts/review_long_touch.py
    python scripts/review_long_touch.py --lookback 250   # 约1年
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.config import CACHE_DIR, OUTPUT_DIR
from stock_ana.data.list_manager import parse_watchlist
from stock_ana.scan.vegas_mid_scan import scan_one, generate_signal_chart
from stock_ana.strategies.primitives.wave import analyze_wave_structure


# ── ETF / 杠杆产品排除 ──
ETF_SKIP = {
    "QQQ", "SPY", "SOXX", "SOXL", "SOXS", "SQQQ", "PSQ", "TSLQ",
    "YINN", "YANG", "CWEB", "CHAU", "KWEB", "FXI", "GLD", "EWY",
    "MPNGY", "GCT", "SNDK",
}


def _rise_bucket(pct: float) -> str:
    if pct < 50:  return "①<50%"
    if pct < 100: return "②50-100%"
    if pct < 200: return "③100-200%"
    return "④>200%"


def _order_group(n: int) -> str:
    if n <= 3:
        return f"#{n}"
    return "#4+"


def _stat_table(df: "pd.DataFrame", col: str, title: str, baseline_wr: float) -> None:
    print(f"\n── {title} ──")
    hdr = f"  {'分组':18s} | {'条数':>4s} | {'胜率%':>6s} | {'Δ':>5s} | {'盈均%':>7s} | {'亏均%':>7s} | {'E[R]%':>7s}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for val in sorted(df[col].unique()):
        g = df[df[col] == val]
        w = g[g["win"]]
        lo = g[~g["win"]]
        wr = g["win"].mean() * 100
        avg_w = w["pivot_ret_pct"].mean() if len(w) else float("nan")
        avg_l = lo["pivot_ret_pct"].mean() if len(lo) else float("nan")
        er = g["pivot_ret_pct"].mean()
        delta = wr - baseline_wr
        avg_w_s = f"{avg_w:+7.1f}%" if not np.isnan(avg_w) else "    N/A"
        avg_l_s = f"{avg_l:+7.1f}%" if not np.isnan(avg_l) else "    N/A"
        print(f"  {str(val):18s} | {len(g):4d} | {wr:6.1f}% | {delta:+5.1f}% | {avg_w_s} | {avg_l_s} | {er:+6.1f}%")


def compute_forward_to_pivot(
    entry_bar: int,
    entry_price: float,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    pivots: list[dict],
) -> dict:
    """从 entry_bar 向前看，找到第一个 ZigZag 拐点，计算到拐点的涨跌幅。

    策略：
      1. 在 pivots 中找 iloc > entry_bar 的第一个拐点
      2. 如果是 H（高点）→ 价格先反弹 → 涨幅 = (pivot_value / entry_price - 1)
         如果是 L（低点）→ 价格继续跌 → 跌幅 = (pivot_value / entry_price - 1)
      3. 如果之后没有拐点（信号太新），用 entry_bar 之后的最高/最低收盘估算

    Returns:
        {
            "pivot_type": "H" | "L" | "none",
            "pivot_iloc": int,
            "pivot_value": float,
            "pivot_ret_pct": float,   # 正=涨, 负=跌
            "bars_to_pivot": int,     # entry_bar 到拐点的 bar 数
        }
    """
    n = len(close)
    if entry_price <= 0:
        return {"pivot_type": "none", "pivot_iloc": -1, "pivot_value": 0.0,
                "pivot_ret_pct": 0.0, "bars_to_pivot": 0}

    # 找第一个 iloc > entry_bar 的拐点
    future_pivots = [p for p in pivots if p["iloc"] > entry_bar]
    if future_pivots:
        first = future_pivots[0]
        ret = (first["value"] / entry_price - 1) * 100
        return {
            "pivot_type": first["type"],
            "pivot_iloc": first["iloc"],
            "pivot_value": round(first["value"], 3),
            "pivot_ret_pct": round(ret, 2),
            "bars_to_pivot": first["iloc"] - entry_bar,
        }

    # 没有后续拐点（信号太新）→ 用剩余数据的最高/最低估算
    if entry_bar + 1 < n:
        remain_high = float(np.max(high[entry_bar + 1:]))
        remain_low = float(np.min(low[entry_bar + 1:]))
        up_pct = (remain_high / entry_price - 1) * 100
        down_pct = (remain_low / entry_price - 1) * 100
        if abs(up_pct) >= abs(down_pct):
            idx = int(entry_bar + 1 + np.argmax(high[entry_bar + 1:]))
            return {
                "pivot_type": "H*",
                "pivot_iloc": idx,
                "pivot_value": round(remain_high, 3),
                "pivot_ret_pct": round(up_pct, 2),
                "bars_to_pivot": idx - entry_bar,
            }
        else:
            idx = int(entry_bar + 1 + np.argmin(low[entry_bar + 1:]))
            return {
                "pivot_type": "L*",
                "pivot_iloc": idx,
                "pivot_value": round(remain_low, 3),
                "pivot_ret_pct": round(down_pct, 2),
                "bars_to_pivot": idx - entry_bar,
            }

    return {"pivot_type": "none", "pivot_iloc": -1, "pivot_value": 0.0,
            "pivot_ret_pct": 0.0, "bars_to_pivot": 0}


def main():
    parser = argparse.ArgumentParser(description="Vegas Long Touch 历史回顾")
    parser.add_argument("--lookback", type=int, default=120, help="回看交易日数（默认120≈6个月）")
    parser.add_argument("--no-chart", action="store_true", help="不生成图表")
    args = parser.parse_args()

    wl = parse_watchlist()
    us_stocks = [
        (item["symbol"].strip().upper(), item.get("name", ""))
        for item in wl.get("us", [])
        if item["symbol"].strip().upper() not in ETF_SKIP
    ]
    logger.info(f"扫描标的数：{len(us_stocks)}")

    out_dir = OUTPUT_DIR / "vegas_scan" / "long_touch_review"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    df_cache: dict[str, pd.DataFrame] = {}   # ticker -> full df (for chart pass)
    no_data: list[str] = []

    # ── Pass 1: 收集信号 + 前向收益（不生成图表）──
    for ticker, name in us_stocks:
        path = CACHE_DIR / "us" / f"{ticker}.parquet"
        if not path.exists():
            no_data.append(ticker)
            continue

        try:
            df = pd.read_parquet(path)
            signals = scan_one(ticker, "US", name or ticker, df, lookback=args.lookback)
            long_sigs = [s for s in signals if s["touch_strategy"] == "long_touch"]
            if not long_sigs:
                continue

            df_cache[ticker] = (df, name or ticker)  # cache for chart pass

            # 准备 pivot 和价格数据用于 forward return 计算
            x = df.copy()
            x.columns = [c.lower() for c in x.columns]
            x.index = pd.to_datetime(x.index)
            x = x.sort_index()
            close_arr = x["close"].astype(float).values
            high_arr = x["high"].astype(float).values
            low_arr = x["low"].astype(float).values

            wave_result = analyze_wave_structure(df)
            pivots = wave_result.get("all_pivots", [])

            for sig_order, sig in enumerate(long_sigs, 1):
                sig["signal_order"] = sig_order
                # 计算 entry_bar 在全量数据中的位置
                entry_date_str = sig["entry_date"].split("(")[0]
                entry_dt = pd.Timestamp(entry_date_str)
                idx_pos = x.index.searchsorted(entry_dt, side="left")
                if idx_pos >= len(x):
                    idx_pos = len(x) - 1

                fwd = compute_forward_to_pivot(
                    entry_bar=idx_pos,
                    entry_price=sig["entry_price"],
                    close=close_arr,
                    high=high_arr,
                    low=low_arr,
                    pivots=pivots,
                )
                sig.update(fwd)
                all_results.append(sig)
                logger.info(
                    f"  {ticker:6s} | {sig['entry_date']:12s} | {sig['support_band']:6s} "
                    f"| wave={sig['wave_number']} | rise={sig['wave_rise_pct']:.0f}% "
                    f"| → {fwd['pivot_type']} {fwd['pivot_ret_pct']:+.1f}% ({fwd['bars_to_pivot']}bars)"
                )

        except Exception as e:
            logger.error(f"{ticker}: {e}")

    # ── 汇总 ──
    print("\n" + "=" * 70)
    n_stocks = len(set(s["symbol"] for s in all_results))
    print(f"有信号：{n_stocks} 只，共 {len(all_results)} 条")
    if no_data:
        print(f"无数据：{', '.join(no_data)}")

    if all_results:
        rets = [s["pivot_ret_pct"] for s in all_results if s["pivot_type"] != "none"]
        if rets:
            wins = [r for r in rets if r > 0]
            losses = [r for r in rets if r <= 0]
            print(f"\n胜率：{len(wins)}/{len(rets)} = {len(wins)/len(rets)*100:.1f}%")
            print(f"盈利信号平均涨幅：{np.mean(wins):.1f}%" if wins else "盈利信号：0")
            print(f"亏损信号平均跌幅：{np.mean(losses):.1f}%" if losses else "亏损信号：0")
            print(f"全体平均：{np.mean(rets):+.1f}%")

        # 按标的汇总
        print("\n── 按标的 ──")
        for sym in sorted(set(s["symbol"] for s in all_results)):
            sym_sigs = [s for s in all_results if s["symbol"] == sym]
            for s in sym_sigs:
                print(
                    f"  {sym:6s} | {s['entry_date']:12s} | {s['support_band']:6s} "
                    f"| → {s['pivot_type']:3s} {s['pivot_ret_pct']:+6.1f}% "
                    f"| {s['bars_to_pivot']:3d}bars"
                )

    # ── 统计分析 ──
    if all_results:
        df_a = pd.DataFrame(all_results)
        # 只用有完整拐点的信号（排除 "none"）
        dv = df_a[df_a["pivot_type"] != "none"].copy()
        dv["win"] = dv["pivot_ret_pct"] > 0
        dv["rise_bucket"] = dv["wave_rise_pct"].apply(_rise_bucket)
        dv["order_group"] = dv["signal_order"].apply(_order_group)
        dv["wave_group"] = dv["wave_number"].apply(
            lambda x: "波浪#1" if x == 1 else "波浪#2+"
        )

        baseline = dv["win"].mean() * 100

        print("\n\n" + "=" * 70)
        print(f"统计分析（基准胜率 {baseline:.1f}%，有效信号 {len(dv)} 条）")
        print("=" * 70)

        _stat_table(dv, "support_band", "按 EMA 线", baseline)
        _stat_table(dv, "wave_group",   "按波浪序号", baseline)
        _stat_table(dv, "rise_bucket",  "按波浪涨幅分桶", baseline)
        _stat_table(dv, "order_group",  "按标的内信号顺序", baseline)

        # EMA × 信号顺序 交叉表
        print("\n── EMA × 信号顺序 交叉胜率 ──")
        cross = dv.groupby(["support_band", "order_group"]).agg(
            wr=("win", "mean"), n=("win", "count")
        )
        cross["wr%"] = (cross["wr"] * 100).round(1)
        print(cross[["wr%", "n"]].to_string())

        # 评分公式：各桶边际胜率
        print("\n── 评分参考（各桶胜率 vs 基准） ──")
        print(f"基准胜率 = {baseline:.1f}%  |  正值 = 优于基准  |  负值 = 低于基准")
        dims = [
            ("support_band", "EMA 线"),
            ("wave_group",   "波浪序号"),
            ("rise_bucket",  "涨幅分桶"),
            ("order_group",  "信号顺序"),
        ]
        score_refs: dict[str, dict] = {}
        for col, label in dims:
            bucket_wr = dv.groupby(col)["win"].mean() * 100
            score_refs[col] = bucket_wr.to_dict()
            print(f"\n  [{label}]")
            for bucket, wr in sorted(bucket_wr.items()):
                edge = wr - baseline
                print(f"    {str(bucket):20s} → 胜率 {wr:5.1f}%  边际 {edge:+5.1f}pp")

        # 对现有信号应用评分并排序
        def _score_signal(row: pd.Series) -> float:
            """每个维度取 (桶胜率 - 基准)，累加得总分（单位：百分点）"""
            s = 0.0
            s += score_refs["support_band"].get(row["support_band"], baseline) - baseline
            s += score_refs["wave_group"].get(row["wave_group"], baseline) - baseline
            s += score_refs["rise_bucket"].get(row["rise_bucket"], baseline) - baseline
            s += score_refs["order_group"].get(row["order_group"], baseline) - baseline
            return round(s, 2)

        dv["score"] = dv.apply(_score_signal, axis=1)

        print("\n── Top 信号（得分最高） ──")
        top = dv.sort_values("score", ascending=False).head(20)
        for _, r in top.iterrows():
            outcome = f"→ {r['pivot_type']:3s} {r['pivot_ret_pct']:+6.1f}%"
            print(
                f"  {r['symbol']:6s} | {r['entry_date']:12s} | {r['support_band']:6s}"
                f" | {r['wave_group']:6s} | rise={r['wave_rise_pct']:4.0f}%"
                f" | #{r['signal_order']} | score={r['score']:+.1f}pp | {outcome}"
            )

        print("\n── Bottom 信号（得分最低） ──")
        bot = dv.sort_values("score", ascending=True).head(20)
        for _, r in bot.iterrows():
            outcome = f"→ {r['pivot_type']:3s} {r['pivot_ret_pct']:+6.1f}%"
            print(
                f"  {r['symbol']:6s} | {r['entry_date']:12s} | {r['support_band']:6s}"
                f" | {r['wave_group']:6s} | rise={r['wave_rise_pct']:4.0f}%"
                f" | #{r['signal_order']} | score={r['score']:+.1f}pp | {outcome}"
            )

        # 按评分分桶验证
        print("\n── 评分分桶验证 ──")
        bins = [-np.inf, -10, -5, 0, 5, 10, np.inf]
        labels = ["<-10", "-10~-5", "-5~0", "0~5", "5~10", ">10"]
        dv["score_bin"] = pd.cut(dv["score"], bins=bins, labels=labels)
        for sbin in labels:
            g = dv[dv["score_bin"] == sbin]
            if len(g) == 0:
                continue
            wr = g["win"].mean() * 100
            er = g["pivot_ret_pct"].mean()
            print(f"  score {sbin:8s} | n={len(g):3d} | 胜率 {wr:5.1f}% | E[R] {er:+6.1f}%")

        # ── 回填得分到 all_results ──
        score_map: dict[tuple, int] = {}
        for _, row in dv.iterrows():
            score_map[(row["symbol"], row["entry_date"])] = int(round(row["score"]))
        for sig in all_results:
            computed = score_map.get((sig["symbol"], sig["entry_date"]), 0)
            sig["score"] = computed
            # 在图表 info panel 里借用 factor_wave_rise 展示统计分
            sig["factor_wave_rise"] = computed

        # CSV 导出
        csv_path = out_dir / "signals.csv"
        export_cols = [
            "symbol", "entry_date", "support_band", "wave_number", "wave_group",
            "wave_rise_pct", "signal_order", "order_group", "rise_bucket",
            "pivot_type", "pivot_ret_pct", "bars_to_pivot", "score", "win",
        ]
        export_cols = [c for c in export_cols if c in dv.columns]
        dv[export_cols].sort_values(["symbol", "entry_date"]).to_csv(
            csv_path, index=False, encoding="utf-8-sig"
        )
        print(f"\nCSV 已保存至：{csv_path}")

    # ── Pass 2: 生成图表（带统计评分标注）──
    if not args.no_chart and all_results:
        logger.info(f"\n开始生成图表（共 {len(all_results)} 张）...")
        for ticker, (df, name) in df_cache.items():
            ticker_sigs = [s for s in all_results if s["symbol"] == ticker]
            for sig in ticker_sigs:
                try:
                    chart_path = generate_signal_chart(
                        sym=ticker, market="US", name=name,
                        df=df, sig=sig, out_dir=out_dir,
                    )
                    plt.close("all")
                    score_val = sig.get("score", 0)
                    fwd_type = sig.get("pivot_type", "?")
                    fwd_ret = sig.get("pivot_ret_pct", 0.0)
                    logger.info(
                        f"  {ticker:6s} | {sig['entry_date']:12s} "
                        f"| score={score_val:+d}pp | → {fwd_type} {fwd_ret:+.1f}%"
                    )
                except Exception as e:
                    logger.warning(f"{ticker} 图表生成失败: {e}")

    print(f"\n图表保存至：{out_dir}")


if __name__ == "__main__":
    main()
