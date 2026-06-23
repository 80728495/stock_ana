#!/usr/bin/env python3
"""gen_holding_top_pdf.py — 每日：把 holding 列表中出现见顶形态的股票打印成 PDF。

对每只持仓近 N 个交易日内、被顶部模型(per-market lgb)判为见顶档(top9%/top30%)的
信号，输出一页：K线图(标信号/SMC供给区/前高线) + 该信号高分的核心要素(lgb SHAP 正贡献)。

依赖（由 build_top_candidate_research.py 周期性产出，holdings 已折入科技池训练宇宙、
天然被打分；本脚本只做筛选+出图，秒级）：
  data/output/top_candidate_research/watchlist_unified_recall_candidates_labeled.csv  (特征+标签)
  data/output/top_candidate_research/top_candidate_lgbm_scored.csv                    (per-market lgb 分)

用法：
    python scripts/gen_holding_top_pdf.py                 # 近10交易日, top30%档
    python scripts/gen_holding_top_pdf.py --days 5 --band top9
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import stock_ana.utils.plot_renderers  # noqa: E402,F401  配置中文字体
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import lightgbm as lgb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mplfinance as mpf  # noqa: E402
from PIL import Image  # noqa: E402

from stock_ana.research.top_reversal.feature_registry import REALTIME_FEATURE_COLS, apply_legacy_feature_aliases  # noqa: E402
from stock_ana.research.top_reversal.smc_context import _normalize_df  # noqa: E402

OUT = PROJECT_ROOT / "data" / "output" / "top_candidate_research"
CACHE = PROJECT_ROOT / "data" / "cache"

# 特征 -> 中文名（用于核心要素文字）
NAME = {
    "valuation_pe": "估值PE(US前向/HKCN TTM)", "valuation_pe_pct_mkt": "PE市场内分位",
    "micro_accel_20_60": "抛物线加速20/60", "micro_verticality_atr": "垂直度(ATR)",
    "micro_dist_ema20_atr": "距EMA20(ATR)", "micro_up_streak": "连阳天数",
    "sector_peers_ret60_mean": "子赛道60日涨幅", "sector_overheat_pct": "子赛道过热分位",
    "xsec_ret60_pct": "60日涨幅截面分位", "xsec_ret20_pct": "20日涨幅截面分位", "is_semiconductor": "半导体",
    "smc_raw_bear_displacement_atr": "SMC熊OB位移ATR", "confirm_drop_from_top_pct": "确认日距顶跌幅",
    "rise_from_anchor_low_pct": "自锚低涨幅", "top_close_above_prior_high": "收盘站上前高",
    "top_close_vs_prior_high_pct": "收盘vs前高%", "top_m_shape": "M顶形态",
    "prior_ret_120d": "前120日涨幅", "prior_ret_60d": "前60日涨幅", "prior_ret_20d": "前20日涨幅",
    "dist_ema144_pct": "距EMA144%", "top_dist_ema144_pct": "顶距EMA144%", "atr14_pct": "ATR14%",
    "rsi14_top": "RSI14", "overbought_score": "超买分", "high_volume_stall_score": "放量滞涨",
    "major_wave_number": "大浪序号", "smc_live_bear_ob_count_20d": "近20日熊OB数",
    "top_close_position_pct": "顶收盘位置%", "china_hk_hstech_ret_20d": "恒科20日(scoped)",
}


def _parse_holdings() -> tuple[set[str], dict[str, str]]:
    syms, names = set(), {}
    path = PROJECT_ROOT / "data" / "lists" / "holding.md"
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*([0-9A-Za-z\.]{1,7})\s*\|\s*(HK|US|CN|SH|SZ)\s*\|\s*([^|]*)", ln)
        if m:
            s = m.group(1).strip()
            syms.add(s)
            names[s] = m.group(3).strip()
    return syms, names


def _load_df(market: str, sym: str) -> pd.DataFrame | None:
    for dd in {"US": ["us", "ndx100"], "HK": ["hk"], "CN": ["cn"]}.get(market, []):
        p = CACHE / dd / f"{sym}.parquet"
        if p.exists():
            try:
                return _normalize_df(pd.read_parquet(p))
            except Exception:  # noqa: BLE001
                return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="持仓见顶信号 -> PDF（图表 + 核心要素）")
    ap.add_argument("--days", type=int, default=10, help="近 N 个交易日内的信号")
    ap.add_argument("--band", choices=["top9", "top30"], default="top30", help="模型见顶档阈值")
    args = ap.parse_args()
    q = 0.91 if args.band == "top9" else 0.70

    hold, holdname = _parse_holdings()
    d = apply_legacy_feature_aliases(pd.read_csv(OUT / "watchlist_unified_recall_candidates_labeled.csv"))
    lg = pd.read_csv(OUT / "top_candidate_lgbm_scored.csv")
    d["sym"] = d.sym.astype(str)
    lg["sym"] = lg.sym.astype(str)
    thr = {mk: g.top_prob.quantile(q) for mk, g in lg.groupby("market_model")}
    d = d.merge(lg[["market", "sym", "top_date", "top_prob"]], on=["market", "sym", "top_date"], how="left")

    # 最近交易日截止（用所有候选里的最大日期近似“今天”）
    last = pd.to_datetime(d.top_date, errors="coerce").max()
    cutoff = last - pd.Timedelta(days=int(args.days * 1.6) + 3)  # 交易日->自然日粗放
    d["is_hit"] = d.apply(
        lambda r: r.sym in hold and pd.notna(r.top_date) and pd.to_datetime(r.top_date) >= cutoff
        and pd.notna(r.top_prob) and r.top_prob >= thr.get(r.market, 1.0),
        axis=1,
    )
    hits = d[d.is_hit].sort_values("top_prob", ascending=False)
    print(f"截止 {last.date()} 近{args.days}日, {args.band} 档: 持仓见顶信号 {len(hits)} 个")
    if hits.empty:
        print("无见顶信号，未生成 PDF。")
        return

    # per-market lgb（用于 SHAP 核心要素）
    b = d[d.label.isin(["true_top", "continuation"])].copy()
    b["y"] = (b.label == "true_top").astype(int)
    feats = [c for c in REALTIME_FEATURE_COLS if c in b.columns]
    shap_cache: dict[str, tuple] = {}

    def shap_model(mk: str):
        if mk not in shap_cache:
            tr = b[b.market == mk]
            fs = [c for c in feats if pd.to_numeric(tr[c], errors="coerce").notna().sum() >= max(20, len(tr) * 0.4)
                  and pd.to_numeric(tr[c], errors="coerce").std() > 1e-9]
            x = tr[fs].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            y = tr.y.to_numpy(float)
            spw = float((y == 0).sum() / max(1, (y == 1).sum()))
            cfg = dict(objective="binary", verbose=-1, num_threads=4, max_depth=3, num_leaves=7,
                       min_data_in_leaf=40, learning_rate=0.03, feature_fraction=0.6, bagging_fraction=0.7,
                       bagging_freq=1, lambda_l2=5, lambda_l1=1, scale_pos_weight=spw, seed=0)
            shap_cache[mk] = (lgb.train(cfg, lgb.Dataset(x, label=y), num_boost_round=300), fs)
        return shap_cache[mk]

    def core_factors(row: pd.Series) -> list[str]:
        m, fs = shap_model(str(row["market"]))
        x = pd.DataFrame([row])[fs].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        contrib = m.predict(x, pred_contrib=True)[0]
        pos = sorted(((fs[i], contrib[i]) for i in range(len(fs)) if contrib[i] > 0), key=lambda t: -t[1])[:7]
        out = []
        for f, c in pos:
            v = row.get(f)
            try:
                vs = f"{float(v):.2f}"
            except (TypeError, ValueError):
                vs = str(v)
            out.append(f"  + {NAME.get(f, f)} = {vs}  (贡献 {c:+.2f})")
        return out

    mc = mpf.make_marketcolors(up="#d62728", down="#2ca02c", edge="inherit", wick="inherit")
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=":", gridcolor="#E8E8E8",
                               rc={"axes.unicode_minus": False, "font.sans-serif": plt.rcParams["font.sans-serif"]})
    tmp = OUT / "holding_signal_eval" / "_daily"
    tmp.mkdir(parents=True, exist_ok=True)
    pages = []
    for i, (_, r) in enumerate(hits.iterrows()):
        df = _load_df(str(r["market"]), str(r["sym"]))
        if df is None:
            continue
        try:
            tp = df.index.get_loc(pd.Timestamp(r["top_date"]))
        except KeyError:
            continue
        n = len(df)
        a, z = max(0, tp - 55), min(n, tp + 12)
        w = df.iloc[a:z][["open", "high", "low", "close", "volume"]].copy()
        w.columns = ["Open", "High", "Low", "Close", "Volume"]
        w.index = pd.to_datetime(w.index)
        nm = holdname.get(str(r["sym"]), str(r["sym"]))
        band = "top9%强顶" if r["top_prob"] >= thr.get(r["market"], 1) else "top30%判顶"
        title = f"[{r['market']}:{r['sym']} {nm}] 信号 {r['top_date']}  lgb={r['top_prob']:.2f} {band}  [{r['strategies']}]"
        try:
            fig, axes = mpf.plot(w, type="candle", style=style, figsize=(11, 9), returnfig=True, volume=False,
                                 xrotation=20, datetime_format="%m-%d", warn_too_much_data=10 ** 9)
            ax = axes[0]
            xi = tp - a
            ax.axvline(xi, color="#1f77b4", lw=1.2, alpha=0.6)
            ax.annotate("信号", xy=(xi, df["high"].iloc[tp]), xytext=(xi, df["high"].iloc[tp] * 1.01),
                        color="#1f77b4", fontsize=9, ha="center", fontweight="bold")
            ax.set_title(title, fontsize=10.5, fontweight="bold")
            fig.subplots_adjust(bottom=0.34, top=0.94)
            cap = ["【核心要素 (lgb SHAP 正贡献 Top7)】"] + core_factors(r)
            fig.text(0.06, 0.02, "\n".join(cap), fontsize=9, va="bottom", linespacing=1.55)
            pp = tmp / f"p{i:03d}.png"
            fig.savefig(pp, dpi=110)
            plt.close(fig)
            pages.append(pp)
        except Exception as e:  # noqa: BLE001
            print(f"跳过 {r['sym']}: {e}")
            plt.close("all")
    if not pages:
        print("无可渲染页。")
        return
    imgs = [Image.open(p).convert("RGB") for p in pages]
    pdf = OUT / "holding_signal_eval" / f"holding_top_{last.date()}.pdf"
    imgs[0].save(pdf, save_all=True, append_images=imgs[1:])
    print(f"PDF({len(imgs)}页) -> {pdf}")


if __name__ == "__main__":
    main()
