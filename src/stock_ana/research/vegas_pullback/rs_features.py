"""相对强度（RS）特征：as-of 读取 RS 系统的单股逐日历史。

数据来源：data/cache/relative_strength/{us|cn|hk}/{symbol}.parquet
（见 docs/relative_strength_system.md）。RS 系统本身因果：窗口只向后看、
benchmark 每月按此前 120 日回归因果选择、市场内百分位只用当日横截面。
本模块按候选的 score_asof_date 取"当日或之前最近一行"（≤7 天容差），
零前瞻。

特征（7 列）：
  rs_return_21d / rs_return_63d : 21/63 日相对 benchmark 累计超额收益 %
  rs_momentum_21d / rs_momentum_63d : RS 曲线动量（z-score EWM 平滑）
  rs_rank_63d : 当日市场内 63 日 RS 百分位（0-100）
  rs_benchmark_beta / rs_benchmark_r2 : 月度回归 beta 与拟合质量

已知 caveat（文档 §9.7）：rs_rank_63d 的横截面是「今天的股票池」历史回放，
存在幸存者偏差；HK 的 63 日 RS 最早约 2023-05-22，此前候选该组特征为 NaN
（lgbm 原生处理缺失）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ana.config import CACHE_DIR

RS_FEATURES: tuple[str, ...] = (
    "rs_return_21d", "rs_return_63d",
    "rs_momentum_21d", "rs_momentum_63d",
    "rs_rank_63d", "rs_benchmark_beta", "rs_benchmark_r2",
    # ── 二阶化（2026-07-12，实证驱动）──
    # rs_line_dd_63 : RS 曲线离自身 63 日高点的回撤 %。实证：rank>80 且
    #                 RS 仍在峰顶(dd>-3%)的回踩胜率最差(0.400)——价格开始
    #                 踩线而 RS 还在峰顶 = 补跌起点（与 IBD 直觉相反）。
    # rs_rank_delta_20 : 63日RS百分位的 20 日变化（排名在爬升还是筑顶）。
    # bench_ret_21d/63d, bench_dist_ma50_pct : 该股当日 benchmark 自身的
    #                 涨跌与趋势位置。实证：大盘 regime 直接移动基础胜率
    #                 （跌市 0.33~0.50 vs 平市 0.45~0.53），且 r2 的效应
    #                 高度依赖 regime（跌市斜率 +0.164 vs 涨市 +0.045）——
    #                 补上方向维度，r2×regime 交互才能被树学到。
    "rs_line_dd_63", "rs_rank_delta_20",
    "bench_ret_21d", "bench_ret_63d", "bench_dist_ma50_pct",
)

_SRC_COLS = {
    "rs_return_21d": "rs_return_21d",
    "rs_return_63d": "rs_return_63d",
    "rs_momentum_21d": "rs_momentum_21d",
    "rs_momentum_63d": "rs_momentum_63d",
    "rs_rank_63d": "rs_rank_63d",
    "rs_benchmark_beta": "benchmark_beta",
    "rs_benchmark_r2": "benchmark_r2",
}

_MAX_STALE_DAYS = 7


def _rs_dir(market: str) -> str:
    return {"US": "us", "CN": "cn", "HK": "hk"}.get(str(market).upper(), "")


def _load_benchmarks() -> dict[str, dict[str, pd.Series]]:
    """载入全部 benchmark，预计算 21/63 日收益与 MA50 距离（因果滚动序列）。"""
    out: dict[str, dict[str, pd.Series]] = {}
    bdir = CACHE_DIR / "benchmarks"
    if not bdir.exists():
        return out
    for p in bdir.glob("*.parquet"):
        try:
            b = pd.read_parquet(p)
            b.index = pd.to_datetime(b.index)
            c = b["close"].astype(float).sort_index()
            out[p.stem] = {
                "ret21": (c / c.shift(21) - 1) * 100,
                "ret63": (c / c.shift(63) - 1) * 100,
                "dist_ma50": (c / c.rolling(50).mean() - 1) * 100,
            }
        except Exception:
            continue
    return out


def add_rs_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """按 (market, sym, score_asof_date) 附加 RS 特征（as-of，零前瞻）。

    含二阶化列：RS 曲线自身回撤、rank 动量、benchmark regime（经由该股
    当日 benchmark_id 映射到对应指数，方向维度补齐）。
    """
    out = dataset.copy()
    for c in RS_FEATURES:
        if c not in out.columns:
            out[c] = np.nan
    if out.empty:
        return out

    asof = pd.to_datetime(out.get("score_asof_date", out.get("signal_date")), errors="coerce")
    cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    bench = _load_benchmarks()

    def load(mk: str, sym: str) -> pd.DataFrame | None:
        key = (mk, sym)
        if key not in cache:
            d = _rs_dir(mk)
            # RS 文件名带前导零（HK 5 位 / CN 6 位）；top_reversal 候选 sym 是去零的，
            # 故补 zfill 变体重试。
            variants = [sym]
            if mk.upper() == "HK":
                variants.append(sym.zfill(5))
            elif mk.upper() == "CN":
                variants.append(sym.zfill(6))
            cache[key] = None
            for sv in dict.fromkeys(variants):
                p = CACHE_DIR / "relative_strength" / d / f"{sv}.parquet"
                if d and p.exists():
                    try:
                        df = pd.read_parquet(p)
                        df.index = pd.to_datetime(df.index)
                        cache[key] = df.sort_index()
                    except Exception:
                        cache[key] = None
                    break
        return cache[key]

    vals: dict[str, list] = {c: [] for c in RS_FEATURES}
    for mk, sym, d in zip(out["market"].astype(str), out["sym"].astype(str), asof, strict=False):
        rs = load(mk, sym)
        row = None
        pos = -1
        if rs is not None and pd.notna(d):
            pos = rs.index.searchsorted(d, side="right") - 1
            if pos >= 0 and (d - rs.index[pos]).days <= _MAX_STALE_DAYS:
                row = rs.iloc[pos]
            else:
                pos = -1
        # 一阶（原有）列
        for feat, src in _SRC_COLS.items():
            v = row.get(src) if row is not None else np.nan
            vals[feat].append(pd.to_numeric(v, errors="coerce") if v is not None else np.nan)

        # ── 二阶化列 ──
        rs_dd = rank_delta = np.nan
        b21 = b63 = bma = np.nan
        if row is not None and pos >= 0:
            # RS 曲线离自身 63 日高点的回撤（补跌风险度量）
            rl = pd.to_numeric(rs["rs_line"].iloc[max(0, pos - 63) : pos + 1], errors="coerce")
            if rl.notna().any() and float(rl.max()) > 0:
                rs_dd = (float(rl.iloc[-1]) / float(rl.max()) - 1) * 100
            # rank 20 日变化（排名爬升 vs 筑顶）
            if pos >= 20:
                r_now = pd.to_numeric(row.get("rs_rank_63d"), errors="coerce")
                r_prev = pd.to_numeric(rs["rs_rank_63d"].iloc[pos - 20], errors="coerce")
                if pd.notna(r_now) and pd.notna(r_prev):
                    rank_delta = float(r_now) - float(r_prev)
            # benchmark regime（按该股当日 benchmark_id 映射）
            bid = str(row.get("benchmark_id", ""))
            if bid in bench and pd.notna(d):
                for src_key, target in [("ret21", "b21"), ("ret63", "b63"), ("dist_ma50", "bma")]:
                    ser = bench[bid][src_key]
                    bpos = ser.index.searchsorted(d, side="right") - 1
                    if bpos >= 0 and (d - ser.index[bpos]).days <= _MAX_STALE_DAYS:
                        v = float(ser.iloc[bpos]) if pd.notna(ser.iloc[bpos]) else np.nan
                        if target == "b21":
                            b21 = v
                        elif target == "b63":
                            b63 = v
                        else:
                            bma = v
        vals["rs_line_dd_63"].append(rs_dd)
        vals["rs_rank_delta_20"].append(rank_delta)
        vals["bench_ret_21d"].append(b21)
        vals["bench_ret_63d"].append(b63)
        vals["bench_dist_ma50_pct"].append(bma)

    for c in RS_FEATURES:
        out[c] = np.round(pd.to_numeric(pd.Series(vals[c], index=out.index), errors="coerce"), 3)
    return out
