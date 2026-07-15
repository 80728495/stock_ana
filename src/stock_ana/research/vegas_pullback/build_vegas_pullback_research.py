#!/usr/bin/env python3
"""构建 Vegas 回踩「抄底 vs 跑路」带标签研究数据集（mid / long 各一份）。

流程（每个 support 独立跑一遍）:
  1. 载入三市场科技池 + 持仓（load_tech_pools_data）
  2. 逐标的：算 EMA / 浪结构 → 检测回踩触碰 → 每个触碰产出一条候选
       候选列: market, sym, name, signal_date, score_asof_date, iloc
       + 信号日特征（signal_features，因果 ≤t）
       + 浪结构标签（labels，用 t 之后走势）
  3. 批量补跨表特征：估值 / 增长 / 宏观 / 前高（复用 top_reversal context）
  4. 写 data/output/vegas_pullback_research/{support}_labeled.csv

用法:
    python -m stock_ana.research.vegas_pullback.build_vegas_pullback_research --support both
    python -m stock_ana.research.vegas_pullback.build_vegas_pullback_research --support long --limit 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def _find_project_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError("Cannot find project root containing pyproject.toml")


PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stock_ana.config import OUTPUT_DIR  # noqa: E402
from stock_ana.data.market_data import load_tech_pools_data  # noqa: E402
from stock_ana.strategies.primitives.wave import analyze_wave_structure  # noqa: E402
from stock_ana.strategies.impl import vegas_mid, vegas_long  # noqa: E402
from stock_ana.research.vegas_pullback.labels import label_pullback_outcome  # noqa: E402
from stock_ana.research.vegas_pullback.causal_wave import (  # noqa: E402
    causal_wave_view,
    containing_wave,
)
from stock_ana.research.vegas_pullback.signal_features import (  # noqa: E402
    compute_signal_features,
    compute_micro_features,
    compute_cluster_features,
    compute_fib_features,
    SIGNAL_FEATURE_COLS,
)

OUT_DIR = OUTPUT_DIR / "vegas_pullback_research"

# 回踩前浪顶的兜底窗口（无浪结构时用近端最高价）
_PRIOR_PEAK_WINDOW = {"mid": 40, "long": 60}


def _detect_touches(support: str, close, low, emas) -> list[dict]:
    if support == "mid":
        return vegas_mid.detect_mid_touch_and_hold(close, low, emas)
    return vegas_long.detect_long_touch_and_hold(close, low, emas)


def build_candidates_for_symbol(
    market: str,
    sym: str,
    name: str,
    df: pd.DataFrame,
    support: str,
) -> list[dict]:
    """对一只股票产出全部回踩候选（含信号特征 + 标签）。"""
    x = df.copy()
    x.columns = [str(c).lower() for c in x.columns]
    x.index = pd.to_datetime(x.index)
    x = x.sort_index()
    if len(x) < 260:
        return []

    close = x["close"].astype(float).values
    high = x["high"].astype(float).values
    low = x["low"].astype(float).values
    open_ = x["open"].astype(float).values if "open" in x.columns else close
    volume = x["volume"].astype(float).values if "volume" in x.columns else np.zeros(len(x))
    emas = vegas_mid.compute_vegas_emas(x["close"].astype(float))
    waves = analyze_wave_structure(df).get("major_waves", [])
    n = len(x)

    touches = _detect_touches(support, close, low, emas)
    peak_win = _PRIOR_PEAK_WINDOW[support]
    rows: list[dict] = []

    for sig in touches:
        # ── 锚点 = confirm_bar：信号在确认日才可知（~7% 的信号 confirm 滞后
        # touch 1-2 天）。特征 as-of、标签前瞻、入场衡量全部从锚点起算，
        # 消除「候选存在性用了未来 1-2 天」的选择泄漏与入场价高估。
        anchor = int(sig["confirm_bar"])
        tb = int(sig["touch_bar"])
        if anchor <= 0 or anchor >= n:
            continue
        # EMA200 暖机期内的候选：Long Vegas 无意义，丢弃
        if anchor < 200:
            continue

        # ── 特征侧：只用 anchor 时点可见的浪（因果视图，零前瞻）──
        view = causal_wave_view(waves, close, anchor)
        pb = vegas_long.locate_wave_pullback(view, tb)
        pb["consec_waves"] = (
            vegas_mid.backward_consecutive_count(view, pb["wave"]) if pb["wave"] else 0
        )
        wave_ctx = containing_wave(view, tb)
        if wave_ctx is not None:
            wave_ctx = dict(wave_ctx)
            wave_ctx["_consec_waves"] = vegas_mid.backward_consecutive_count(view, wave_ctx)

        # ── 标签侧：prior_peak 允许用全量浪（标签本就用未来），否则近端最高价 ──
        pb_full = vegas_long.locate_wave_pullback(waves, tb)
        if pb_full["wave"] is not None:
            prior_peak = float(pb_full["wave"]["peak_pivot"]["value"])
        else:
            prior_peak = float(np.max(high[max(0, anchor - peak_win) : anchor + 1]))

        # ── W1→W2 结构标签（用全量浪，标签可用未来）──
        # chained: 本次回踩终结的浪之后交接出连续的下一浪（wave_number+1，
        #          connected_prev=True）——用户目标形态「W1 回踩买入吃满 W2」
        # isolated: 浪结束但无连续下一浪（孤立 W1 / 深破断裂）
        # pending: 浪仍进行中；na: 触碰未对应浪终点
        label_w2, w2_rise = "na", np.nan
        w_full = pb_full["wave"]
        if w_full is not None:
            if w_full.get("end_pivot") is None:
                label_w2 = "pending"
            else:
                idx_w = next(
                    (k for k, ww in enumerate(waves)
                     if ww["start_pivot"]["iloc"] == w_full["start_pivot"]["iloc"]),
                    None,
                )
                nxt = waves[idx_w + 1] if idx_w is not None and idx_w + 1 < len(waves) else None
                if nxt is not None and nxt.get("connected_prev"):
                    label_w2 = "chained"
                    w2_rise = float(nxt.get("rise_pct", np.nan))
                else:
                    label_w2 = "isolated"

        feats = compute_signal_features(
            close, high, low, volume, emas, anchor, support, pb, wave_ctx=wave_ctx
        )
        feats.update(compute_micro_features(
            open_, high, low, close, volume, tb, anchor, emas=emas, support=support,
        ))
        # 簇特征按本次信号所踩的那条线计算（ema_span），绝不混线
        sig_span = int(sig.get("ema_span", 55 if support == "mid" else 144))
        feats.update(compute_cluster_features(
            high, low, close, volume, emas[sig_span], anchor,
        ))
        # 斐波那契回撤（swing 锚定：浪锚优先，滚动兜底）
        feats.update(compute_fib_features(high, low, close, anchor, wave_ctx))
        lab = label_pullback_outcome(
            close, high, low, emas, anchor, prior_peak, support=support
        )

        row = {
            "market": market,
            "sym": sym,
            "name": name,
            "signal_date": x.index[anchor],
            "score_asof_date": x.index[anchor],
            "iloc": anchor,
            "touch_iloc": tb,
            "support_band": sig.get("support_band", ""),
            "label_w2": label_w2,
            "w2_rise_pct": w2_rise,
            **feats,
            **lab,
        }
        rows.append(row)

    return rows


def _attach_cross_table_features(dataset: pd.DataFrame, symbol_data: dict) -> pd.DataFrame:
    """批量补跨表特征（估值/增长/宏观/前高），每个模块失败不影响其余。"""
    from stock_ana.research.top_reversal.valuation_context import add_valuation_features
    from stock_ana.research.top_reversal.growth_context import add_growth_features
    from stock_ana.research.top_reversal.macro_micro_context import add_macro_micro_features
    from stock_ana.research.vegas_pullback.fund_inflection import add_fund_inflection_features
    from stock_ana.research.vegas_pullback.rs_features import add_rs_features
    from stock_ana.research.vegas_pullback.sqz_features import add_sqz_features

    for name, fn, needs_data in [
        ("macro_micro", add_macro_micro_features, True),
        ("growth", add_growth_features, False),
        ("valuation", add_valuation_features, False),
        ("fund_inflection", add_fund_inflection_features, False),
        ("rs", add_rs_features, False),
        ("squeeze", add_sqz_features, False),
    ]:
        try:
            dataset = fn(dataset, symbol_data) if needs_data else fn(dataset)
            logger.info(f"  + {name} 特征已附加")
        except Exception as e:
            logger.warning(f"  ! {name} 特征附加失败（跳过）: {type(e).__name__}: {e}")
    return dataset


def build_support(support: str, symbol_data: dict, limit: int | None = None) -> pd.DataFrame:
    """构建一个 support（mid/long）的完整带标签数据集。"""
    items = list(symbol_data.values())
    if limit:
        items = items[:limit]

    logger.info(f"[{support}] 扫描 {len(items)} 只标的产出回踩候选 ...")
    all_rows: list[dict] = []
    for k, item in enumerate(items):
        if k % 100 == 0 and k:
            logger.info(f"  ... {k}/{len(items)}")
        try:
            rows = build_candidates_for_symbol(
                item["market"], item["symbol"], item.get("name", item["symbol"]),
                item["df"], support,
            )
            all_rows.extend(rows)
        except Exception as e:
            logger.warning(f"  {item.get('market')}:{item.get('symbol')} 失败: {type(e).__name__}: {e}")

    if not all_rows:
        logger.warning(f"[{support}] 无候选")
        return pd.DataFrame()

    dataset = pd.DataFrame(all_rows)
    logger.info(f"[{support}] 候选 {len(dataset)} 条，标签分布: "
                f"{dataset['label'].value_counts().to_dict()}")

    dataset = _attach_cross_table_features(dataset, symbol_data)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 Vegas 回踩抄底/跑路研究数据集")
    parser.add_argument("--support", choices=["mid", "long", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="每 support 最多扫描的标的数（调试用）")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("载入三市场科技池 + 持仓 ...")
    symbol_data = load_tech_pools_data(min_history=260)
    logger.info(f"共 {len(symbol_data)} 只标的有 ≥260 bar 数据")

    supports = ["mid", "long"] if args.support == "both" else [args.support]
    for support in supports:
        dataset = build_support(support, symbol_data, limit=args.limit)
        if dataset.empty:
            continue
        out_path = OUT_DIR / f"{support}_labeled.csv"
        dataset.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.success(f"[{support}] 写出 {len(dataset)} 条 → {out_path}")


if __name__ == "__main__":
    main()
