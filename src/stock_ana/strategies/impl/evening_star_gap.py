"""
顶部反转形态识别 — 策略二：高位跳空十字星（Evening Star / Abandoned Baby 变体）

三根 K 线形态：

  Day-2 (前天)：趋势大阳线
    - 阳线实体 ≥ 近期平均实体的 BODY_MULT 倍（默认 1.3）
    - 表示散户情绪高涨、追涨入场

  Day-1 (昨天)：跳空放量十字星 / 纺锤线（机构冰山卖单）
    - 向上跳空：Day-1 open > Day-2 close（允许 GAP_TOLERANCE 容差）
    - 实体极小：body / range ≤ DOJI_BODY_RATIO（默认 0.30）
    - 放量：成交量 ≥ vol_ma_50 × VOL_RATIO（默认 1.5）
    - 处于阶段新高区域

  Day-0 (今天)：看跌确认
    强确认（score+1）：
      a. 向下跳空：Day-0 open < Day-1 low（Abandoned Baby）
      b. 收盘 < Day-2 实体中点（阳线一半以上被吃掉）
    弱确认：
      c. 阴线且收盘 < Day-1 收盘

评分体系（0~5 分）：
  1  基础触发
  +1 严格十字星（body/range ≤ 0.10）
  +1 Day-1 巨量（vol ≥ vol_ma_50 × 2.0）
  +1 Day-0 向下跳空 或 收盘 < Day-2 中点
  +1 Day-0 放量 ≥ vol_ma_50 × 1.3

Public API
----------
detect_evening_star_gap(df) -> dict
    检测最新三根 K 线是否触发。

scan_history(df) -> pd.DataFrame
    向量化历史全量扫描。
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# ── 参数常量 ──────────────────────────────────────────────────────────────────
NEW_HIGH_WINDOW: int = 20            # 阶段新高回望窗口
NEW_HIGH_LOOKBACK: int = 3           # 近N日内出现新高的容忍窗口
NEAR_HIGH_PCT: float = 0.03          # Day-1 高点与近期新高最大偏差

# Day-2 大阳线
BODY_MULT: float = 1.3               # 阳线实体 ≥ 近20日平均实体 × 此倍数
BODY_RANGE_MIN: float = 0.50         # 阳线实体占全振幅下限

# Day-1 十字星/纺锤线
GAP_TOLERANCE: float = 0.01          # 跳空容差（Day-1 open >= Day-2 close × (1 - tol)，允许1%以内低开）
DOJI_BODY_RATIO: float = 0.30        # 实体/振幅 ≤ 此值算纺锤线
STRICT_DOJI_RATIO: float = 0.10      # 更严格的十字星（加分项）
VOL_RATIO_MIN: float = 1.5           # Day-1 放量门槛（vs vol_ma_50）
VOL_RATIO_STRONG: float = 2.0        # Day-1 巨量（加分项）

# Day-0 确认
CONFIRM_VOL_RATIO: float = 1.3       # Day-0 放量加分阈值


# ── 内部工具 ───────────────────────────────────────────────────────────────────

def _ensure_vol_ma(df: pd.DataFrame) -> pd.DataFrame:
    if "vol_ma_50" not in df.columns:
        df["vol_ma_50"] = df["volume"].astype(float).rolling(50, min_periods=1).mean()
    return df


# ── detect（最新三根 K 线） ─────────────────────────────────────────────────────

def detect_evening_star_gap(
    df: pd.DataFrame,
    new_high_window: int = NEW_HIGH_WINDOW,
    new_high_lookback: int = NEW_HIGH_LOOKBACK,
    near_high_pct: float = NEAR_HIGH_PCT,
) -> dict:
    """检测 df 最新三根 K 线是否构成 Evening Star Gap 形态。"""
    _empty = dict(triggered=False, score=0, reason="数据不足")
    if len(df) < max(new_high_window + 3, 30):
        return _empty

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df = _ensure_vol_ma(df)

    d2 = df.iloc[-3]  # Day-2（大阳线）
    d1 = df.iloc[-2]  # Day-1（十字星）
    d0 = df.iloc[-1]  # Day-0（确认）

    signal_date  = str(df.index[-2])[:10]
    confirm_date = str(df.index[-1])[:10]
    prev_date    = str(df.index[-3])[:10]

    o2, h2, l2, c2 = float(d2["open"]), float(d2["high"]), float(d2["low"]), float(d2["close"])
    o1, h1, l1, c1 = float(d1["open"]), float(d1["high"]), float(d1["low"]), float(d1["close"])
    o0, h0, l0, c0 = float(d0["open"]), float(d0["high"]), float(d0["low"]), float(d0["close"])

    # ── Day-2: 阳线（大阳线为加分项） ──
    body2 = c2 - o2
    range2 = h2 - l2
    if body2 <= 0 or range2 < 1e-9:
        return dict(triggered=False, score=0, reason="Day-2 不是阳线")

    body_ratio2 = body2 / range2
    recent_body = df["close"].iloc[-23:-3].values - df["open"].iloc[-23:-3].values
    avg_body = np.abs(recent_body).mean()
    is_d2_big_bull = (body_ratio2 >= BODY_RANGE_MIN) and (body2 >= avg_body * BODY_MULT)

    # ── Day-1: 跳空十字星 ──
    if o1 < c2 * (1 - GAP_TOLERANCE):
        return dict(triggered=False, score=0, reason=f"Day-1 未跳空（open={o1:.2f} < Day-2 close={c2:.2f}）")

    body1 = abs(c1 - o1)
    range1 = h1 - l1
    if range1 < 1e-9:
        return dict(triggered=False, score=0, reason="Day-1 无振幅")
    body_ratio1 = body1 / range1
    if body_ratio1 > DOJI_BODY_RATIO:
        return dict(triggered=False, score=0,
                    reason=f"Day-1 实体占幅 {body_ratio1:.2f} > {DOJI_BODY_RATIO}（非十字星）")

    vol_ma50 = float(d1["vol_ma_50"]) if float(d1["vol_ma_50"]) > 0 else 1.0
    vol_ratio1 = float(d1["volume"]) / vol_ma50
    if vol_ratio1 < VOL_RATIO_MIN:
        return dict(triggered=False, score=0,
                    reason=f"Day-1 量比 {vol_ratio1:.1f}x < {VOL_RATIO_MIN}x")

    # Day-1 处于阶段新高附近
    lookback_start = max(0, len(df) - 2 - new_high_window)
    lookback_end = len(df) - 2  # 不含 Day-1 本身
    recent_highs = df["high"].iloc[lookback_start:lookback_end].values
    rolling_max = recent_highs.max()
    if h1 < rolling_max * (1 - near_high_pct):
        return dict(triggered=False, score=0,
                    reason=f"Day-1 不在阶段新高附近（high={h1:.2f} vs max={rolling_max:.2f}）")

    # ── Day-0: 确认 ──
    d2_mid = (o2 + c2) / 2
    gap_down = o0 < l1                   # 向下跳空
    below_mid = c0 < d2_mid              # 收盘 < Day-2 中点
    bearish_close = c0 < o0 and c0 < c1  # 阴线且收低于 Day-1

    if not (gap_down or below_mid or bearish_close):
        return dict(triggered=False, score=0, reason="Day-0 无看跌确认")

    confirm_mode = (
        "gap_down"    if gap_down else
        "below_d2mid" if below_mid else
        "bearish_close"
    )

    # ── 评分（0~6） ──
    score = 1
    # Day-2 大阳线
    if is_d2_big_bull:
        score += 1
    # 严格十字星
    is_strict_doji = body_ratio1 <= STRICT_DOJI_RATIO
    if is_strict_doji:
        score += 1
    # Day-1 巨量
    is_strong_vol = vol_ratio1 >= VOL_RATIO_STRONG
    if is_strong_vol:
        score += 1
    # Day-0 强确认
    if gap_down or below_mid:
        score += 1
    # Day-0 放量
    vol_ma50_d0 = float(d0["vol_ma_50"]) if float(d0["vol_ma_50"]) > 0 else 1.0
    d0_vol_ratio = float(d0["volume"]) / vol_ma50_d0
    if d0_vol_ratio >= CONFIRM_VOL_RATIO:
        score += 1

    d2_label = "大阳线" if is_d2_big_bull else "阳线"
    pattern_label = "十字星" if is_strict_doji else "纺锤线"
    reason = (
        f"{prev_date} {d2_label} → {signal_date} 跳空{pattern_label}（量比{vol_ratio1:.1f}x）→ "
        f"{confirm_date} {confirm_mode}"
    )

    return dict(
        triggered=True,
        signal_date=signal_date,
        confirm_date=confirm_date,
        prev_date=prev_date,
        confirm_mode=confirm_mode,
        is_strict_doji=is_strict_doji,
        is_d2_big_bull=is_d2_big_bull,
        d2_body_pct=round(body2 / c2 * 100, 2),
        d1_body_ratio=round(body_ratio1, 3),
        d1_vol_ratio=round(vol_ratio1, 2),
        d0_vol_ratio=round(d0_vol_ratio, 2),
        day1_high=round(h1, 4),
        day1_close=round(c1, 4),
        score=score,
        reason=reason,
    )


# ── scan_history（向量化历史扫描） ──────────────────────────────────────────────

def scan_history(
    df: pd.DataFrame,
    new_high_window: int = NEW_HIGH_WINDOW,
    new_high_lookback: int = NEW_HIGH_LOOKBACK,
    near_high_pct: float = NEAR_HIGH_PCT,
    forward_days: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """
    向量化扫描历史中所有 Evening Star Gap 信号。

    Returns:
        DataFrame，每行一个信号，含 signal_date, confirm_date, score, fwd_ret_Nd 等。
    """
    if len(df) < max(new_high_window + 3, 30):
        return pd.DataFrame()

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().reset_index()  # iloc 与 position 对齐
    df = _ensure_vol_ma(df)

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float)
    vm50 = df["vol_ma_50"].astype(float)

    body = (c - o).abs().replace(0, 1e-9)
    signed_body = c - o  # 正=阳线，负=阴线
    rng = (h - l).replace(0, 1e-9)
    body_ratio = body / rng

    # Day-2（shift 2）: 大阳线
    body2 = signed_body.shift(2)
    rng2  = rng.shift(2)
    c2    = c.shift(2)
    o2    = o.shift(2)
    body_ratio2 = body.shift(2) / rng2

    avg_body_20 = body.rolling(20, min_periods=5).mean().shift(3)  # 取 Day-2 之前的均值
    is_d2_bullish = body2 > 0  # Day-2 只需是阳线
    is_d2_big_bull = is_d2_bullish & (body_ratio2 >= BODY_RANGE_MIN) & (body2 >= avg_body_20 * BODY_MULT)

    # Day-1（shift 1）: 跳空十字星
    o1 = o.shift(1)
    h1 = h.shift(1)
    l1 = l.shift(1)
    c1 = c.shift(1)
    v1 = v.shift(1)
    vm50_1 = vm50.shift(1)
    body_ratio1 = body.shift(1) / rng.shift(1)

    gap_up = o1 >= c2 * (1 - GAP_TOLERANCE)
    is_doji = body_ratio1 <= DOJI_BODY_RATIO
    vol_ratio1 = v1 / vm50_1.replace(0, 1e-9)
    is_vol = vol_ratio1 >= VOL_RATIO_MIN

    # Day-1 处于阶段新高
    rolling_high = h.rolling(new_high_window, min_periods=5).max()
    # 近N日内有新高
    is_at_new_high_raw = h >= rolling_high.shift(1)
    recent_had_new_high = is_at_new_high_raw.shift(1).rolling(new_high_lookback, min_periods=1).max().astype(bool)
    # Day-1 high 在近期峰值附近
    recent_peak = h.shift(1).rolling(new_high_lookback, min_periods=1).max()
    near_peak = h1 >= recent_peak.shift(1) * (1 - near_high_pct)
    is_new_high = recent_had_new_high.shift(1) & near_peak  # shift again for Day-1 position

    # Day-1 候选（Day-2 只需阳线，大阳线为加分项）
    day1_mask = is_d2_bullish & gap_up & is_doji & is_vol & is_new_high

    # Day-0 (当前行): 确认
    d2_mid = (o2 + c2) / 2
    cond_gap_down = o < l1                       # 向下跳空
    cond_below_mid = c < d2_mid                  # 收盘 < Day-2 中点
    cond_bearish = (c < o) & (c < c1)            # 阴线收低

    confirmed = cond_gap_down | cond_below_mid | cond_bearish

    triggered = day1_mask & confirmed
    triggered = triggered.fillna(False)

    # ── 评分 & 构建结果 ──
    is_strict = body_ratio1 <= STRICT_DOJI_RATIO
    is_strong_vol = vol_ratio1 >= VOL_RATIO_STRONG

    vol_ratio0 = v / vm50.replace(0, 1e-9)
    d0_vol_spike = vol_ratio0 >= CONFIRM_VOL_RATIO

    rows = []
    n = len(df)
    for i in df.index[triggered]:
        pos = df.index.get_loc(i)  # Day-0 位置
        pos_d1 = pos - 1
        pos_d2 = pos - 2

        score = 1
        _is_big = bool(is_d2_big_bull.iloc[pos])
        if _is_big:
            score += 1
        if bool(is_strict.iloc[pos_d1]):
            score += 1
        if bool(is_strong_vol.iloc[pos_d1]):
            score += 1
        if bool(cond_gap_down.iloc[pos]) or bool(cond_below_mid.iloc[pos]):
            score += 1
        if bool(d0_vol_spike.iloc[pos]):
            score += 1

        confirm_mode = (
            "gap_down"      if bool(cond_gap_down.iloc[pos]) else
            "below_d2mid"   if bool(cond_below_mid.iloc[pos]) else
            "bearish_close"
        )

        row: dict = {
            "signal_date":    str(df["date"].iloc[pos_d1])[:10],
            "confirm_date":   str(df["date"].iloc[pos])[:10],
            "prev_date":      str(df["date"].iloc[pos_d2])[:10],
            "is_strict_doji": bool(is_strict.iloc[pos_d1]),
            "is_d2_big_bull": _is_big,
            "d1_body_ratio":  round(float(body_ratio1.iloc[pos_d1]), 3) if pos_d1 < len(body_ratio1) else 0,
            "d1_vol_ratio":   round(float(vol_ratio1.iloc[pos_d1]), 2) if pos_d1 < len(vol_ratio1) else 0,
            "d0_vol_ratio":   round(float(vol_ratio0.iloc[pos]), 2),
            "confirm_mode":   confirm_mode,
            "score":          score,
            "day1_high":      round(float(h.iloc[pos_d1]), 4),
            "day1_close":     round(float(c.iloc[pos_d1]), 4),
        }

        # 后续收益（以 Day-1 收盘为基准）
        base = float(c.iloc[pos_d1])
        for fd in forward_days:
            end_pos = min(pos + fd, n - 1)
            fwd_c = float(c.iloc[end_pos])
            fwd_min = float(l.iloc[pos + 1: end_pos + 1].min()) if pos + 1 <= end_pos else base
            row[f"fwd_ret_{fd}d"] = round((fwd_c - base) / base * 100, 2)
            row[f"fwd_min_{fd}d"] = round((fwd_min - base) / base * 100, 2)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
