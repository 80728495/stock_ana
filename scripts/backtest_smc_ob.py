"""
backtest_smc_ob.py — SMC Order Block 回测与特征分析

回测 4 个方向:
  1. 看涨 OB 生成后 2 个月涨幅
  2. 看跌 OB 生成后 2 个月跌幅
  3. 股价刺入看涨 OB 区后，是否止跌并重回涨势
  4. 股价刺入看跌 OB 区后，是否止涨并重回跌势

输出:
  - 控制台打印汇总统计
  - data/output/backtest_smc_ob/ 目录下生成详细 CSV + 筛选器参数

用法:
  python scripts/backtest_smc_ob.py --market us
  python scripts/backtest_smc_ob.py --market us --list tech
  python scripts/backtest_smc_ob.py --market us --symbols NVDA,AAPL,MSFT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ── 路径 ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stock_ana.config import CACHE_DIR, OUTPUT_DIR
from stock_ana.strategies.impl.smc import _ob_causal

import os
os.environ.setdefault("SMC_CREDIT", "0")
from smartmoneyconcepts import smc  # noqa: E402

# ── 参数 ──────────────────────────────────────────────────────────────────────
HOLDING_DAYS = 42  # 约 2 个月交易日
SWING_LENGTH = 5
TOUCH_FORWARD_DAYS = 42  # touch 后观测窗口

OUTPUT_SUBDIR = OUTPUT_DIR / "backtest_smc_ob"
OUTPUT_SUBDIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# 数据加载
# ═════════════════════════════════════════════════════════════════════════════

def _cache_dir_for_market(market: str) -> Path:
    m = market.lower()
    if m in ("us", "ndx100"):
        return CACHE_DIR / "us"
    return CACHE_DIR / m


def load_symbol_list(market: str, list_name: str | None) -> list[str]:
    """从缓存目录或指定列表文件加载代码列表。"""
    if list_name == "tech":
        list_path = PROJECT_ROOT / "data" / "lists" / "us_tech_list.md"
        if list_path.exists():
            symbols = []
            for line in list_path.read_text(encoding="utf-8").splitlines():
                if not line.startswith("|"):
                    continue
                parts = [p.strip() for p in line.strip("|").split("|")]
                # 格式: | # | 代码 | 公司 | 行业 | 市值 |
                if len(parts) >= 2 and parts[0].isdigit():
                    sym = parts[1]
                    if sym and sym[0].isalpha():
                        symbols.append(sym)
            return symbols
    # 默认：从缓存目录获取全部
    cache_dir = _cache_dir_for_market(market)
    if not cache_dir.exists():
        return []
    return sorted([p.stem for p in cache_dir.glob("*.parquet")])


def load_ohlcv(symbol: str, market: str) -> pd.DataFrame | None:
    """加载单只股票的日线 OHLCV 数据。"""
    cache_dir = _cache_dir_for_market(market)
    path = cache_dir / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df.sort_index()
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# OB 特征提取
# ═════════════════════════════════════════════════════════════════════════════

def compute_ob_features(df: pd.DataFrame, ob_df: pd.DataFrame, bar_idx: int) -> dict:
    """为给定 bar 上的 OB 提取数据特征。"""
    row = ob_df.iloc[bar_idx]
    direction = int(row["OB"])
    top = float(row["Top"])
    bottom = float(row["Bottom"])
    ob_volume = float(row["OBVolume"]) if pd.notna(row.get("OBVolume")) else 0.0
    percentage = float(row["Percentage"]) if pd.notna(row.get("Percentage")) else 0.0

    # OB 宽度 (相对于价格的百分比)
    mid_price = (top + bottom) / 2
    ob_width_pct = (top - bottom) / mid_price * 100 if mid_price > 0 else 0

    # OB bar 的实体/影线比
    o, h, l, c = df["open"].iloc[bar_idx], df["high"].iloc[bar_idx], df["low"].iloc[bar_idx], df["close"].iloc[bar_idx]
    body = abs(c - o)
    full_range = h - l
    body_ratio = body / full_range if full_range > 0 else 0

    # 成交量相对于20日均量
    vol = df["volume"].iloc[bar_idx]
    start = max(0, bar_idx - 20)
    avg_vol_20 = df["volume"].iloc[start:bar_idx].mean() if bar_idx > 0 else vol
    vol_ratio = vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

    # OB 前 5 日趋势（涨跌幅）
    lookback = 5
    if bar_idx >= lookback:
        trend_before = (df["close"].iloc[bar_idx] - df["close"].iloc[bar_idx - lookback]) / df["close"].iloc[bar_idx - lookback] * 100
    else:
        trend_before = 0.0

    # ATR(14) 在 OB 生成时
    if bar_idx >= 15:
        hi = df["high"].values[bar_idx-14:bar_idx]
        lo = df["low"].values[bar_idx-14:bar_idx]
        prev_c = df["close"].values[bar_idx-15:bar_idx-1]
        tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_c), np.abs(lo - prev_c)))
        atr14 = float(tr.mean())
    else:
        atr14 = full_range

    # OB 宽度与 ATR 的比值
    ob_atr_ratio = (top - bottom) / atr14 if atr14 > 0 else 0

    return {
        "direction": direction,
        "top": top,
        "bottom": bottom,
        "ob_volume": ob_volume,
        "percentage": percentage,
        "ob_width_pct": ob_width_pct,
        "body_ratio": body_ratio,
        "vol_ratio": vol_ratio,
        "trend_before_5d": trend_before,
        "atr14": atr14,
        "ob_atr_ratio": ob_atr_ratio,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 回测核心
# ═════════════════════════════════════════════════════════════════════════════

def backtest_symbol(symbol: str, market: str, quality_filter: bool = False) -> tuple[list[dict], list[dict]]:
    """
    对单只股票执行回测。

    返回:
        (ob_records, touch_records)
        - ob_records:    方向 1&2 — OB 生成后 N 日涨跌
        - touch_records: 方向 3&4 — 价格刺入 OB 区后的表现
    """
    df = load_ohlcv(symbol, market)
    if df is None or len(df) < SWING_LENGTH * 2 + HOLDING_DAYS + 20:
        return [], []

    # 计算 OB
    swing_hl = smc.swing_highs_lows(df, swing_length=SWING_LENGTH)
    ob_df = _ob_causal(df, swing_hl, swing_length=SWING_LENGTH)

    # 质量筛选（如启用）
    if quality_filter:
        from stock_ana.strategies.impl.smc import ob_passes_quality  # noqa: PLC0415

    ob_records: list[dict] = []
    touch_records: list[dict] = []

    # 遍历所有 OB
    for bar_idx in range(len(df)):
        if pd.isna(ob_df.iloc[bar_idx]["OB"]):
            continue

        # 质量筛选：不符合条件的 OB 跳过
        if quality_filter and not ob_passes_quality(df, ob_df, bar_idx):
            continue

        direction = int(ob_df.iloc[bar_idx]["OB"])
        top = float(ob_df.iloc[bar_idx]["Top"])
        bottom = float(ob_df.iloc[bar_idx]["Bottom"])
        mit_idx = ob_df.iloc[bar_idx]["MitigatedIndex"]
        is_mitigated = pd.notna(mit_idx) and int(mit_idx) != 0
        mit_bar = int(mit_idx) if is_mitigated else None

        formed_date = str(df.index[bar_idx].date())

        # 需要至少 HOLDING_DAYS 天后续数据评估方向 1&2
        # OB 在结构突破时识别；形成 bar 就是 OB 所在位置
        # 评估区间：OB 生成后一个交易日起算
        eval_start = bar_idx + 1

        # ── 方向 1 & 2：OB 生成后 2 个月涨跌 ──────────────────────────────────
        if eval_start + HOLDING_DAYS <= len(df):
            close_at_form = df["close"].iloc[bar_idx]
            close_after = df["close"].iloc[eval_start:eval_start + HOLDING_DAYS]
            max_price = close_after.max()
            min_price = close_after.min()
            end_price = close_after.iloc[-1]

            ret_end = (end_price - close_at_form) / close_at_form * 100
            ret_max = (max_price - close_at_form) / close_at_form * 100
            ret_min = (min_price - close_at_form) / close_at_form * 100

            # 判断方向是否正确
            if direction == 1:
                # 看涨 OB：期望涨
                success = ret_end > 0
                max_favorable = ret_max
                max_adverse = ret_min
            else:
                # 看跌 OB：期望跌
                success = ret_end < 0
                max_favorable = -ret_min  # 最大跌幅（正数表示有利）
                max_adverse = ret_max     # 最大涨幅（不利）

            features = compute_ob_features(df, ob_df, bar_idx)
            ob_records.append({
                "symbol": symbol,
                "market": market.upper(),
                "formed_date": formed_date,
                "direction": direction,
                "top": top,
                "bottom": bottom,
                "is_mitigated": is_mitigated,
                "ret_end_pct": round(ret_end, 2),
                "ret_max_pct": round(ret_max, 2),
                "ret_min_pct": round(ret_min, 2),
                "max_favorable_pct": round(max_favorable, 2),
                "max_adverse_pct": round(max_adverse, 2),
                "success": success,
                **features,
            })

        # ── 方向 3 & 4：股价刺入 OB 区后是否反转 ──────────────────────────────
        # 寻找首次有效 touch（价格刺入 OB 区但未 mitigate）
        # 要求：(a) 价格先完全离开 OB 区域  (b) 距 OB 生成至少 5 个交易日
        # 对看涨 OB：low <= top (进入 OB 区)，且 low >= bottom (未穿透)
        # 对看跌 OB：high >= bottom (进入 OB 区)，且 high <= top (未穿透)
        MIN_GAP_DAYS = 5  # 至少间隔一周（5 个交易日）
        touch_bar = None
        search_end = mit_bar if mit_bar else len(df)
        departed = False  # 价格是否已经离开过 OB 区
        for j in range(eval_start, min(search_end, len(df))):
            if direction == 1:
                # 看涨 OB：价格在 OB 上方 = 离开（low > top）
                if not departed:
                    if df["low"].iloc[j] > top:
                        departed = True
                    continue
                # 已离开且间隔够久，检测回落刺入
                if j - bar_idx < MIN_GAP_DAYS:
                    continue
                if df["low"].iloc[j] <= top and df["low"].iloc[j] >= bottom:
                    touch_bar = j
                    break
            else:
                # 看跌 OB：价格在 OB 下方 = 离开（high < bottom）
                if not departed:
                    if df["high"].iloc[j] < bottom:
                        departed = True
                    continue
                if j - bar_idx < MIN_GAP_DAYS:
                    continue
                # 已离开后，检测反弹刺入
                if df["high"].iloc[j] >= bottom and df["high"].iloc[j] <= top:
                    touch_bar = j
                    break

        if touch_bar is not None and touch_bar + TOUCH_FORWARD_DAYS <= len(df):
            touch_date = str(df.index[touch_bar].date())
            close_at_touch = df["close"].iloc[touch_bar]
            forward_data = df["close"].iloc[touch_bar + 1:touch_bar + 1 + TOUCH_FORWARD_DAYS]

            if len(forward_data) >= TOUCH_FORWARD_DAYS:
                end_price_t = forward_data.iloc[-1]
                max_price_t = forward_data.max()
                min_price_t = forward_data.min()

                ret_end_t = (end_price_t - close_at_touch) / close_at_touch * 100
                ret_max_t = (max_price_t - close_at_touch) / close_at_touch * 100
                ret_min_t = (min_price_t - close_at_touch) / close_at_touch * 100

                if direction == 1:
                    # 刺入看涨 OB → 期望止跌回涨
                    touch_success = ret_end_t > 0
                    # 额外判断：是否先跌后涨（V 型反转）
                    # 前 10 日最低点 vs 后续恢复
                    first_half = forward_data.iloc[:TOUCH_FORWARD_DAYS // 2]
                    second_half = forward_data.iloc[TOUCH_FORWARD_DAYS // 2:]
                    reversal = (second_half.mean() > first_half.mean())
                else:
                    # 刺入看跌 OB → 期望止涨回跌
                    touch_success = ret_end_t < 0
                    first_half = forward_data.iloc[:TOUCH_FORWARD_DAYS // 2]
                    second_half = forward_data.iloc[TOUCH_FORWARD_DAYS // 2:]
                    reversal = (second_half.mean() < first_half.mean())

                # 是否最终被 mitigate（穿透）
                eventually_mitigated = is_mitigated and mit_bar is not None and mit_bar <= touch_bar + TOUCH_FORWARD_DAYS

                features = compute_ob_features(df, ob_df, bar_idx)
                touch_records.append({
                    "symbol": symbol,
                    "market": market.upper(),
                    "formed_date": formed_date,
                    "touch_date": touch_date,
                    "direction": direction,
                    "top": top,
                    "bottom": bottom,
                    "days_to_touch": touch_bar - bar_idx,
                    "ret_end_pct": round(ret_end_t, 2),
                    "ret_max_pct": round(ret_max_t, 2),
                    "ret_min_pct": round(ret_min_t, 2),
                    "touch_success": touch_success,
                    "reversal": reversal,
                    "eventually_mitigated": eventually_mitigated,
                    **features,
                })

    return ob_records, touch_records


# ═════════════════════════════════════════════════════════════════════════════
# 统计与输出
# ═════════════════════════════════════════════════════════════════════════════

def print_stats(df_ob: pd.DataFrame, df_touch: pd.DataFrame) -> None:
    """打印汇总统计。"""
    print("\n" + "=" * 72)
    print("  SMC Order Block 回测统计")
    print("=" * 72)

    # ── 方向 1：看涨 OB 生成后涨幅 ────────────────────────────────────────────
    bull_ob = df_ob[df_ob["direction"] == 1]
    print(f"\n▶ 方向 1：看涨 OB 生成后 {HOLDING_DAYS} 日表现（样本 {len(bull_ob)}）")
    if len(bull_ob) > 0:
        print(f"  胜率（终点收涨）:    {bull_ob['success'].mean()*100:.1f}%")
        print(f"  平均终点涨跌幅:     {bull_ob['ret_end_pct'].mean():.2f}%")
        print(f"  平均最大有利:       {bull_ob['max_favorable_pct'].mean():.2f}%")
        print(f"  平均最大不利:       {bull_ob['max_adverse_pct'].mean():.2f}%")
        print(f"  中位终点涨跌幅:     {bull_ob['ret_end_pct'].median():.2f}%")

    # ── 方向 2：看跌 OB 生成后跌幅 ────────────────────────────────────────────
    bear_ob = df_ob[df_ob["direction"] == -1]
    print(f"\n▶ 方向 2：看跌 OB 生成后 {HOLDING_DAYS} 日表现（样本 {len(bear_ob)}）")
    if len(bear_ob) > 0:
        print(f"  胜率（终点收跌）:    {bear_ob['success'].mean()*100:.1f}%")
        print(f"  平均终点涨跌幅:     {bear_ob['ret_end_pct'].mean():.2f}%")
        print(f"  平均最大有利:       {bear_ob['max_favorable_pct'].mean():.2f}%")
        print(f"  平均最大不利:       {bear_ob['max_adverse_pct'].mean():.2f}%")
        print(f"  中位终点涨跌幅:     {bear_ob['ret_end_pct'].median():.2f}%")

    # ── 方向 3：刺入看涨 OB 后止跌回涨 ────────────────────────────────────────
    bull_touch = df_touch[df_touch["direction"] == 1]
    print(f"\n▶ 方向 3：刺入看涨 OB 后 {TOUCH_FORWARD_DAYS} 日表现（样本 {len(bull_touch)}）")
    if len(bull_touch) > 0:
        print(f"  胜率（终点收涨）:    {bull_touch['touch_success'].mean()*100:.1f}%")
        print(f"  V型反转率:          {bull_touch['reversal'].mean()*100:.1f}%")
        print(f"  平均终点涨跌幅:     {bull_touch['ret_end_pct'].mean():.2f}%")
        print(f"  最终被穿透率:       {bull_touch['eventually_mitigated'].mean()*100:.1f}%")
        print(f"  平均 touch 天数:    {bull_touch['days_to_touch'].mean():.0f} 天")

    # ── 方向 4：刺入看跌 OB 后止涨回跌 ────────────────────────────────────────
    bear_touch = df_touch[df_touch["direction"] == -1]
    print(f"\n▶ 方向 4：刺入看跌 OB 后 {TOUCH_FORWARD_DAYS} 日表现（样本 {len(bear_touch)}）")
    if len(bear_touch) > 0:
        print(f"  胜率（终点收跌）:    {bear_touch['touch_success'].mean()*100:.1f}%")
        print(f"  V型反转率:          {bear_touch['reversal'].mean()*100:.1f}%")
        print(f"  平均终点涨跌幅:     {bear_touch['ret_end_pct'].mean():.2f}%")
        print(f"  最终被穿透率:       {bear_touch['eventually_mitigated'].mean()*100:.1f}%")
        print(f"  平均 touch 天数:    {bear_touch['days_to_touch'].mean():.0f} 天")


def analyze_features(df_ob: pd.DataFrame, df_touch: pd.DataFrame) -> dict:
    """
    分析成功/失败 OB 的特征差异，输出筛选器阈值建议。
    """
    results = {}

    feature_cols = [
        "ob_width_pct", "body_ratio", "vol_ratio",
        "trend_before_5d", "ob_atr_ratio", "percentage",
    ]

    for direction, label in [(1, "看涨OB"), (-1, "看跌OB")]:
        subset = df_ob[df_ob["direction"] == direction].copy()
        if len(subset) < 20:
            continue

        print(f"\n{'─' * 72}")
        print(f"  特征分析 — {label}（样本 {len(subset)}）")
        print(f"{'─' * 72}")

        winners = subset[subset["success"]]
        losers = subset[~subset["success"]]

        print(f"\n  {'特征':<20} {'成功组均值':>12} {'失败组均值':>12} {'差异':>10} {'建议方向':>10}")
        print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*10} {'─'*10}")

        feature_analysis = {}
        for col in feature_cols:
            if col not in subset.columns:
                continue
            w_mean = winners[col].mean() if len(winners) > 0 else 0
            l_mean = losers[col].mean() if len(losers) > 0 else 0
            diff = w_mean - l_mean
            # 判断应该取高值还是低值
            recommend = "取高" if diff > 0 else "取低"
            print(f"  {col:<20} {w_mean:>12.3f} {l_mean:>12.3f} {diff:>+10.3f} {recommend:>10}")
            feature_analysis[col] = {
                "winner_mean": w_mean,
                "loser_mean": l_mean,
                "diff": diff,
                "recommend": recommend,
            }

        results[label] = feature_analysis

    # ── touch 特征分析 ────────────────────────────────────────────────────────
    for direction, label in [(1, "看涨OB_touch"), (-1, "看跌OB_touch")]:
        subset = df_touch[df_touch["direction"] == direction].copy()
        if len(subset) < 20:
            continue

        print(f"\n{'─' * 72}")
        print(f"  Touch 特征分析 — {label}（样本 {len(subset)}）")
        print(f"{'─' * 72}")

        winners = subset[subset["touch_success"]]
        losers = subset[~subset["touch_success"]]

        extra_cols = feature_cols + ["days_to_touch"]
        print(f"\n  {'特征':<20} {'成功组均值':>12} {'失败组均值':>12} {'差异':>10} {'建议方向':>10}")
        print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*10} {'─'*10}")

        for col in extra_cols:
            if col not in subset.columns:
                continue
            w_mean = winners[col].mean() if len(winners) > 0 else 0
            l_mean = losers[col].mean() if len(losers) > 0 else 0
            diff = w_mean - l_mean
            recommend = "取高" if diff > 0 else "取低"
            print(f"  {col:<20} {w_mean:>12.3f} {l_mean:>12.3f} {diff:>+10.3f} {recommend:>10}")

    return results


def build_filter_thresholds(df_ob: pd.DataFrame, df_touch: pd.DataFrame) -> dict:
    """
    基于统计数据，为 OB 构建筛选器阈值。
    使用分桶法找最优单特征阈值，然后贪心组合，确保:
      - 每次加条件后，样本保留率 >= 30%
      - 胜率相对基线有提升
      - 最多 3 个条件
    """
    filters = {}

    feature_cols = [
        "ob_width_pct", "body_ratio", "vol_ratio",
        "trend_before_5d", "ob_atr_ratio", "percentage",
    ]

    def _find_best_conditions(subset: pd.DataFrame, target_col: str,
                              cols: list[str] | None = None, max_cond: int = 3):
        """贪心搜索最优条件组合。"""
        search_cols = cols or feature_cols
        base_wr = subset[target_col].mean()
        n_total = len(subset)
        min_samples = max(30, int(n_total * 0.3))  # 至少保留 30% 或 30 个

        # 收集所有候选条件
        candidates = []
        for col in search_cols:
            if col not in subset.columns:
                continue
            for q in np.arange(0.15, 0.85, 0.05):
                threshold = subset[col].quantile(q)
                # >= threshold
                mask_ge = subset[col] >= threshold
                cnt = mask_ge.sum()
                if cnt >= min_samples:
                    wr = subset.loc[mask_ge, target_col].mean()
                    if wr > base_wr:
                        candidates.append((wr - base_wr, f"{col}_min", threshold, mask_ge, cnt))
                # <= threshold
                mask_le = subset[col] <= threshold
                cnt = mask_le.sum()
                if cnt >= min_samples:
                    wr = subset.loc[mask_le, target_col].mean()
                    if wr > base_wr:
                        candidates.append((wr - base_wr, f"{col}_max", threshold, mask_le, cnt))

        # 按胜率提升排序
        candidates.sort(key=lambda x: x[0], reverse=True)

        # 贪心组合
        selected = []
        combined_mask = pd.Series(True, index=subset.index)
        for gain, key, threshold, mask, cnt in candidates:
            # 避免同一特征的重复条件
            col_name = key.rsplit("_", 1)[0]
            if any(col_name in s[0] for s in selected):
                continue

            new_mask = combined_mask & mask
            new_cnt = new_mask.sum()
            if new_cnt < min_samples:
                continue

            new_wr = subset.loc[new_mask, target_col].mean()
            if new_wr <= base_wr + 0.01:  # 必须比基线高 1% 以上
                continue

            selected.append((key, round(float(threshold), 3)))
            combined_mask = new_mask
            if len(selected) >= max_cond:
                break

        return dict(selected)

    # OB 方向筛选
    for direction, label in [(1, "bull"), (-1, "bear")]:
        subset = df_ob[df_ob["direction"] == direction].copy()
        if len(subset) < 50:
            continue
        f = _find_best_conditions(subset, "success")
        if f:
            filters[label] = f

    # Touch 方向筛选
    for direction, label in [(1, "bull_touch"), (-1, "bear_touch")]:
        subset = df_touch[df_touch["direction"] == direction].copy()
        if len(subset) < 50:
            continue
        touch_cols = feature_cols + ["days_to_touch"]
        f = _find_best_conditions(subset, "touch_success", cols=touch_cols)
        if f:
            filters[label] = f

    return filters


def _apply_filter_mask(subset: pd.DataFrame, f: dict) -> pd.Series:
    """根据筛选器字典生成布尔 mask。"""
    mask = pd.Series(True, index=subset.index)
    for key, threshold in f.items():
        if key.endswith("_min"):
            col = key[:-4]
            if col in subset.columns:
                mask &= subset[col] >= threshold
        elif key.endswith("_max"):
            col = key[:-4]
            if col in subset.columns:
                mask &= subset[col] <= threshold
    return mask


def apply_filter_and_report(df_ob: pd.DataFrame, df_touch: pd.DataFrame, filters: dict) -> None:
    """应用筛选器并报告筛选后的胜率改善。"""
    print(f"\n{'=' * 72}")
    print("  筛选器效果验证")
    print(f"{'=' * 72}")

    for direction, label, dir_name in [(1, "bull", "看涨OB"), (-1, "bear", "看跌OB")]:
        if label not in filters:
            continue
        f = filters[label]
        subset = df_ob[df_ob["direction"] == direction].copy()
        if len(subset) == 0:
            continue

        mask = _apply_filter_mask(subset, f)
        filtered = subset[mask]
        orig_wr = subset["success"].mean() * 100
        filt_wr = filtered["success"].mean() * 100 if len(filtered) > 0 else 0

        print(f"\n  {dir_name}:")
        print(f"    筛选前: 样本 {len(subset):>5}, 胜率 {orig_wr:.1f}%")
        print(f"    筛选后: 样本 {len(filtered):>5}, 胜率 {filt_wr:.1f}%")
        print(f"    保留率: {len(filtered)/len(subset)*100:.1f}%")
        print(f"    筛选器参数: {f}")

    for direction, label, dir_name in [(1, "bull_touch", "看涨OB_touch"), (-1, "bear_touch", "看跌OB_touch")]:
        if label not in filters or len(df_touch) == 0:
            continue
        f = filters[label]
        subset = df_touch[df_touch["direction"] == direction].copy()
        if len(subset) == 0:
            continue

        mask = _apply_filter_mask(subset, f)
        filtered = subset[mask]
        orig_wr = subset["touch_success"].mean() * 100
        filt_wr = filtered["touch_success"].mean() * 100 if len(filtered) > 0 else 0

        print(f"\n  {dir_name}:")
        print(f"    筛选前: 样本 {len(subset):>5}, 胜率 {orig_wr:.1f}%")
        print(f"    筛选后: 样本 {len(filtered):>5}, 胜率 {filt_wr:.1f}%")
        print(f"    保留率: {len(filtered)/len(subset)*100:.1f}%")
        print(f"    筛选器参数: {f}")


# ═════════════════════════════════════════════════════════════════════════════
# 主流程
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global HOLDING_DAYS, TOUCH_FORWARD_DAYS  # noqa: PLW0603

    parser = argparse.ArgumentParser(description="SMC OB 回测与特征分析")
    parser.add_argument("--market", default="us", help="市场 (us/hk/cn)")
    parser.add_argument("--list", default=None, help="列表名称 (tech / None=全部)")
    parser.add_argument("--symbols", default=None, help="指定代码（逗号分隔）")
    parser.add_argument("--max-symbols", type=int, default=0, help="最大处理数量（0=全部）")
    parser.add_argument("--holding-days", type=int, default=HOLDING_DAYS, help=f"持有天数（默认 {HOLDING_DAYS}）")
    parser.add_argument("--quality-filter", action="store_true", help="启用 OB 质量筛选")
    args = parser.parse_args()

    HOLDING_DAYS = args.holding_days
    TOUCH_FORWARD_DAYS = args.holding_days

    # 获取代码列表
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = load_symbol_list(args.market, args.list)

    if args.max_symbols > 0:
        symbols = symbols[:args.max_symbols]

    use_qf = args.quality_filter
    logger.info(f"开始回测: market={args.market}, 样本={len(symbols)} 只, 持有天数={HOLDING_DAYS}, 质量筛选={'ON' if use_qf else 'OFF'}")

    all_ob_records: list[dict] = []
    all_touch_records: list[dict] = []
    processed = 0

    for i, symbol in enumerate(symbols):
        if (i + 1) % 50 == 0:
            logger.info(f"  进度 {i+1}/{len(symbols)} ... OB={len(all_ob_records)}, Touch={len(all_touch_records)}")

        ob_recs, touch_recs = backtest_symbol(symbol, args.market, quality_filter=use_qf)
        all_ob_records.extend(ob_recs)
        all_touch_records.extend(touch_recs)
        processed += 1

    logger.info(f"回测完成: 处理 {processed} 只, OB 记录 {len(all_ob_records)}, Touch 记录 {len(all_touch_records)}")

    if not all_ob_records:
        print("无有效 OB 记录，退出。")
        return

    df_ob = pd.DataFrame(all_ob_records)
    df_touch = pd.DataFrame(all_touch_records) if all_touch_records else pd.DataFrame()

    # ── 打印统计 ──────────────────────────────────────────────────────────────
    print_stats(df_ob, df_touch)

    # ── 特征分析 ──────────────────────────────────────────────────────────────
    analyze_features(df_ob, df_touch)

    # ── 构建筛选器 ────────────────────────────────────────────────────────────
    filters = build_filter_thresholds(df_ob, df_touch)
    apply_filter_and_report(df_ob, df_touch, filters)

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    market_tag = args.market.upper()
    list_tag = f"_{args.list}" if args.list else ""

    ob_csv = OUTPUT_SUBDIR / f"ob_backtest_{market_tag}{list_tag}.csv"
    df_ob.to_csv(ob_csv, index=False, encoding="utf-8-sig")
    print(f"\n  OB 回测数据已保存: {ob_csv}")

    if len(df_touch) > 0:
        touch_csv = OUTPUT_SUBDIR / f"touch_backtest_{market_tag}{list_tag}.csv"
        df_touch.to_csv(touch_csv, index=False, encoding="utf-8-sig")
        print(f"  Touch 回测数据已保存: {touch_csv}")

    # 保存筛选器
    if filters:
        import json
        filter_path = OUTPUT_SUBDIR / f"ob_filter_{market_tag}{list_tag}.json"
        with open(filter_path, "w", encoding="utf-8") as f:
            json.dump(filters, f, indent=2, ensure_ascii=False)
        print(f"  筛选器参数已保存: {filter_path}")

    print(f"\n{'=' * 72}")
    print("  回测完成")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
