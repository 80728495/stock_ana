"""
顶部反转形态识别 — 策略一：阶段新高长上影 + 次日确认

逻辑：
  Day-1 (信号日，iloc[-2])：
    1. 近 NEW_HIGH_LOOKBACK 日（默认3日）内出现过 N 日新高，
       且 Day-1 高价在该近期高点的 NEAR_HIGH_PCT（默认3%）以内
    2. 长上影线：upper_shadow >= body * UPPER_SHADOW_RATIO_MIN (默认 1.5)
       且上影线绝对幅度 >= Day-1 收盘价 × MIN_SHADOW_PCT（默认 3%）
       极端情况（射击之星）：upper_shadow >= body * 2.0 且 body_ratio < 0.30
    3. 加分项：成交量 > vol_ma_50 * VOL_SPIKE_RATIO (默认 1.3)

  Day-2 (确认日，iloc[-1])：
    以下任一条件满足 → 看跌确认（未能维持涨势）：
      a. 收盘 < Day-1 开盘（空方彻底吃掉 Day-1 实体）
      b. 收盘 < Day-1 实体中点 (open+close)/2
      c. 阴线 + 收盘 < Day-1 收盘

Public API
----------
detect_high_shadow_reversal(df) -> dict
    检测最新两根 K 线是否触发顶部反转信号。
    返回字段：
      triggered     bool
      signal_date   str   Day-1 日期
      confirm_date  str   Day-2 日期
      confirm_mode  str   "engulf_open" / "below_midpoint" / "bearish_close"
      is_shooting_star  bool   Day-1 是否构成射击之星
      day1_new_high_n   int    阶段新高窗口
      day1_upper_shadow_ratio  float  上影线/实体 比
      day1_vol_spike    bool   是否放量
      day1_high         float
      day1_close        float
      score             int    0~4 综合评分
      reason            str    简要说明
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# ── 参数常量 ──────────────────────────────────────────────────────────────────
NEW_HIGH_WINDOW: int = 20          # 阶段新高回望窗口（交易日）
NEW_HIGH_LOOKBACK: int = 3         # 「近N日内出现新高」的容忍窗口（放宽条件，默认3日）
NEAR_HIGH_PCT: float = 0.03        # Day-1 高点与近期新高的最大偏差（默认3%）
UPPER_SHADOW_RATIO_MIN: float = 1.5  # 上影线/实体 最小倍数（宽松长上影）
SHOOTING_STAR_RATIO: float = 2.0     # 上影线/实体 倍数阈值（射击之星）
SHOOTING_STAR_BODY_MAX: float = 0.30 # 射击之星实体占总振幅上限
VOL_SPIKE_RATIO: float = 1.3         # 放量阈值（用于评分）

# 上影线长度双条件（OR 逻辑）：
MIN_SHADOW_PCT_STRONG: float = 0.030   # 条件A：上影线占收盘价 ≥ 3%，单独触发
MIN_SHADOW_PCT_WITH_VOL: float = 0.020 # 条件B：上影线占收盘价 ≥ 2%，需配合放量
BIG_VOL_RATIO: float = 1.5             # 条件B 中的「显著放量」阈值（vol / vol_ma_50 ≥ 1.5x）
# A OR B 表达的含义："上影≥ 3%"或"上影≥ 2%且放量≥ 50日均量×1.5"
# 门槛宜宽松以确保召回，靠 score 分层过滤信号质量

CONFIRM_BODY_RATIO_MIN: float = 0.8    # 加分项：Day-2 实体 / Day-1 上影线 ≥ 0.8，说明确认日卖压真实充分


# ── 内部工具 ──────────────────────────────────────────────────────────────────

def _upper_shadow(row: pd.Series) -> float:
    return float(row["high"]) - max(float(row["open"]), float(row["close"]))


def _body(row: pd.Series) -> float:
    return abs(float(row["close"]) - float(row["open"]))


def _body_ratio(row: pd.Series) -> float:
    rng = float(row["high"]) - float(row["low"])
    if rng < 1e-9:
        return 0.0
    return _body(row) / rng


# ── 核心检测函数 ──────────────────────────────────────────────────────────────

def detect_high_shadow_reversal(
    df: pd.DataFrame,
    new_high_window: int = NEW_HIGH_WINDOW,
    new_high_lookback: int = NEW_HIGH_LOOKBACK,
    near_high_pct: float = NEAR_HIGH_PCT,
    upper_shadow_ratio_min: float = UPPER_SHADOW_RATIO_MIN,
    vol_spike_ratio: float = VOL_SPIKE_RATIO,
    min_shadow_pct_strong: float = MIN_SHADOW_PCT_STRONG,
    min_shadow_pct_with_vol: float = MIN_SHADOW_PCT_WITH_VOL,
    big_vol_ratio: float = BIG_VOL_RATIO,
) -> dict:
    """
    检测最新两根 K 线是否触发顶部反转（近期新高 + 长上影 + 次日确认）。

    上影线长度采用 OR 逻辑：
      条件A：上影线 >= 收盘价 x min_shadow_pct_strong（4%）——单独足够强
      条件B：上影线 >= 收盘价 x min_shadow_pct_with_vol（2%）
              且 当日成交量 >= vol_ma_50 x big_vol_ratio（2.0x）

    Returns:
        dict，triggered=False 时表示未触发，其余字段填充检测中间结果。
    """
    _empty = dict(triggered=False, score=0, reason="数据不足")
    if len(df) < new_high_window + 2:
        return _empty

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # 确保 vol_ma_50 存在
    if "vol_ma_50" not in df.columns:
        df["vol_ma_50"] = df["volume"].astype(float).rolling(50, min_periods=1).mean()

    # Day-1 = iloc[-2]，Day-2 = iloc[-1]
    d1 = df.iloc[-2]
    d2 = df.iloc[-1]

    signal_date  = str(df.index[-2])[:10]
    confirm_date = str(df.index[-1])[:10]

    # ── Day-1 检测 ──────────────────────────────────────────────────────────

    # 条件1：近 new_high_lookback 日内出现过 new_high_window 日新高，
    #         且 Day-1 高价在该近期高点的 near_high_pct 以内
    # 「近 N 日」 = Day-1 及其前 (new_high_lookback-1) 根，共 new_high_lookback 根
    lookback_start = -(new_high_lookback + 1)   # 含 Day-1 的 lookback 窗口起点
    recent_window = df["high"].iloc[lookback_start:-1]   # 近 N 日（截至 Day-1，不含 Day-2）
    recent_peak = float(recent_window.max())             # 近 N 日最高价

    # 近 N 日内是否有任意一根创过 new_high_window 日新高
    # 用完整序列计算 rolling 新高，再取对应窗口检查
    rolling_20d = df["high"].shift(1).rolling(new_high_window, min_periods=new_high_window).max()
    recent_had_new_high = any(
        float(df["high"].iloc[i]) >= float(rolling_20d.iloc[i])
        for i in range(lookback_start, -1)   # Day-1 及之前 (lookback-1) 根
        if not pd.isna(rolling_20d.iloc[i])
    )

    # Day-1 高点须在近期峰值的 near_high_pct 以内（说明 Day-1 正在触碰该高点区域）
    d1_near_peak = float(d1["high"]) >= recent_peak * (1 - near_high_pct)

    if not recent_had_new_high or not d1_near_peak:
        return dict(
            triggered=False, score=0,
            signal_date=signal_date, confirm_date=confirm_date,
            reason=(
                f"近{new_high_lookback}日无 {new_high_window} 日新高"
                if not recent_had_new_high
                else f"Day-1 高点偏离近期峰值 >{near_high_pct*100:.0f}%"
            ),
        )

    # 条件2：长上影线（相对比例 + 绝对幅度 OR 逻辑）
    body_d1  = _body(d1)
    upper_d1 = _upper_shadow(d1)
    body_safe = max(body_d1, 1e-9)
    shadow_ratio = upper_d1 / body_safe

    if shadow_ratio < upper_shadow_ratio_min:
        return dict(
            triggered=False, score=0,
            signal_date=signal_date, confirm_date=confirm_date,
            day1_upper_shadow_ratio=round(shadow_ratio, 2),
            reason=f"Day-1 上影线/实体={shadow_ratio:.2f} < {upper_shadow_ratio_min}",
        )

    close_d1 = max(float(d1["close"]), 1e-9)
    abs_shadow_pct = upper_d1 / close_d1
    vol_ma50 = float(d1["vol_ma_50"]) if float(d1["vol_ma_50"]) > 0 else 1.0
    d1_vol_ratio = float(d1["volume"]) / vol_ma50

    # OR 逻辑：条件A（上影强） OR 条件B（上影较弱但大量配合）
    cond_a = abs_shadow_pct >= min_shadow_pct_strong
    cond_b = (abs_shadow_pct >= min_shadow_pct_with_vol) and (d1_vol_ratio >= big_vol_ratio)
    shadow_ok = cond_a or cond_b

    if not shadow_ok:
        return dict(
            triggered=False, score=0,
            signal_date=signal_date, confirm_date=confirm_date,
            day1_upper_shadow_ratio=round(shadow_ratio, 2),
            day1_shadow_pct=round(abs_shadow_pct * 100, 2),
            reason=(
                f"Day-1 上影线 {abs_shadow_pct*100:.2f}% < 强阈值 {min_shadow_pct_strong*100:.0f}%且"
                f"量比{d1_vol_ratio:.1f}x < 大放量阈值 {big_vol_ratio}x"
            ),
        )

    # 判断是否为射击之星（更严格的形态）
    is_shooting_star = (
        shadow_ratio >= SHOOTING_STAR_RATIO
        and _body_ratio(d1) < SHOOTING_STAR_BODY_MAX
    )

    # 加分项：放量
    day1_vol_spike = d1_vol_ratio > vol_spike_ratio

    # ── Day-2 确认 ──────────────────────────────────────────────────────────

    d1_open  = float(d1["open"])
    d1_close = float(d1["close"])
    d1_mid   = (d1_open + d1_close) / 2
    d2_close = float(d2["close"])
    d2_open  = float(d2["open"])
    d2_is_bearish = d2_close < d2_open

    confirm_mode: str | None = None
    if d2_close < d1_open:
        confirm_mode = "engulf_open"        # 空方吞掉 Day-1 实体
    elif d2_close < d1_mid:
        confirm_mode = "below_midpoint"     # 收于 Day-1 实体中点以下
    elif d2_is_bearish and d2_close < d1_close:
        confirm_mode = "bearish_close"      # 阴线 + 低于 Day-1 收盘

    if confirm_mode is None:
        return dict(
            triggered=False, score=0,
            signal_date=signal_date, confirm_date=confirm_date,
            day1_upper_shadow_ratio=round(shadow_ratio, 2),
            is_shooting_star=is_shooting_star,
            day1_vol_spike=day1_vol_spike,
            reason="Day-2 未确认（仍维持看涨）",
        )

    # ── 综合评分（0~5 分）───────────────────────────────────────────────────
    # 基础：触发即 1 分
    score = 1
    # 射击之星形态
    if is_shooting_star:
        score += 1
    # 放量
    if day1_vol_spike:
        score += 1
    # 强力确认（空方完全吞没）
    if confirm_mode == "engulf_open":
        score += 1
    # 确认日实体足够大（D2 实体 / D1 上影线 ≥ CONFIRM_BODY_RATIO_MIN）
    d2_body = _body(d2)
    confirm_body_ratio = d2_body / upper_d1 if upper_d1 > 0 else 0.0
    if confirm_body_ratio >= CONFIRM_BODY_RATIO_MIN:
        score += 1

    # ── 文字描述 ─────────────────────────────────────────────────────────────
    mode_label = {
        "engulf_open":    "Day-2 收于 Day-1 开盘以下（完全吞没）",
        "below_midpoint": "Day-2 收于 Day-1 实体中点以下",
        "bearish_close":  "Day-2 阴线收低于 Day-1 收盘",
    }[confirm_mode]
    pattern_label = "射击之星" if is_shooting_star else "长上影线"
    vol_label = "放量" if day1_vol_spike else "未放量"

    reason = (
        f"{signal_date} {pattern_label}（上影/实体={shadow_ratio:.1f}x）{vol_label}，"
        f"{confirm_date} {mode_label}"
    )

    return dict(
        triggered=True,
        signal_date=signal_date,
        confirm_date=confirm_date,
        confirm_mode=confirm_mode,
        is_shooting_star=is_shooting_star,
        day1_new_high_n=new_high_window,
        day1_upper_shadow_ratio=round(shadow_ratio, 2),
        day1_vol_spike=day1_vol_spike,
        day1_high=round(float(d1["high"]), 4),
        day1_close=round(d1_close, 4),
        score=score,
        reason=reason,
    )


# ── 历史全量扫描（向量化）────────────────────────────────────────────────────

def scan_history(
    df: pd.DataFrame,
    new_high_window: int = NEW_HIGH_WINDOW,
    new_high_lookback: int = NEW_HIGH_LOOKBACK,
    near_high_pct: float = NEAR_HIGH_PCT,
    upper_shadow_ratio_min: float = UPPER_SHADOW_RATIO_MIN,
    vol_spike_ratio: float = VOL_SPIKE_RATIO,
    min_shadow_pct_strong: float = MIN_SHADOW_PCT_STRONG,
    min_shadow_pct_with_vol: float = MIN_SHADOW_PCT_WITH_VOL,
    big_vol_ratio: float = BIG_VOL_RATIO,
    forward_days: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """
    对完整历史数据进行全量向量化扫描，返回所有触发点及后续收益。

    Args:
        df: 完整 OHLCV DataFrame（index 为日期）。
        new_high_window: 阶段新高回望窗口。
        upper_shadow_ratio_min: 上影线/实体 最小倍数。
        vol_spike_ratio: 放量阈值倍数。
        forward_days: 评估后续表现的天数列表（默认 5/10/20 日）。

    Returns:
        DataFrame，每行一个触发点，含信号特征与后续收益列：
          signal_date, confirm_date, is_shooting_star, shadow_ratio,
          vol_spike, confirm_mode, score,
          fwd_ret_5d, fwd_ret_10d, fwd_ret_20d（收盘价相对 Day-1 收盘的涨跌幅）
          fwd_min_5d, fwd_min_10d, fwd_min_20d（窗口内最低价跌幅）
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().reset_index()  # 让 iloc 与 position 一一对应

    if "vol_ma_50" not in df.columns:
        df["vol_ma_50"] = df["volume"].astype(float).rolling(50, min_periods=1).mean()

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float)
    vm20 = df["vol_ma_50"].astype(float)

    # — Day-1 候选条件（向量化）—
    body      = (c - o).abs().replace(0, 1e-9)
    upper_shd = h - pd.concat([o, c], axis=1).max(axis=1)
    rng       = (h - l).replace(0, 1e-9)
    body_ratio = body / rng
    shadow_ratio = upper_shd / body

    # 阶段新高：向量化计算每根 bar 是否为新高
    rolling_high = h.shift(1).rolling(new_high_window, min_periods=new_high_window).max()
    was_new_high = (h >= rolling_high).fillna(False)  # 每根 bar 是否创 N 日新高

    # 近 new_high_lookback 日内是否有新高
    recent_had_new_high = was_new_high.rolling(new_high_lookback, min_periods=1).max().astype(bool)

    # 近 new_high_lookback 日内的最高价（含当日）
    recent_peak = h.rolling(new_high_lookback, min_periods=1).max()

    # Day-1 高点在近期峰值 near_high_pct 以内
    near_peak = h >= recent_peak * (1 - near_high_pct)

    is_new_high = recent_had_new_high & near_peak

    # 长上影（相对比例 + 绝对幅度 OR 逻辑双重门槛）
    abs_shadow_pct = upper_shd / c.replace(0, 1e-9)  # 上影线占收盘价比例
    vol_ratio_series = v / vm20.replace(0, 1e-9)      # 成交量比例
    cond_shadow_a = abs_shadow_pct >= min_shadow_pct_strong                          # 条件A
    cond_shadow_b = (abs_shadow_pct >= min_shadow_pct_with_vol) & (vol_ratio_series >= big_vol_ratio)  # 条件B
    is_long_shadow = (shadow_ratio >= upper_shadow_ratio_min) & (cond_shadow_a | cond_shadow_b)

    # 射击之星（更严格）
    is_star = (shadow_ratio >= SHOOTING_STAR_RATIO) & (body_ratio < SHOOTING_STAR_BODY_MAX)

    # 放量
    vol_spike = v > vm20 * vol_spike_ratio

    # Day-1 候选
    day1_mask = is_new_high & is_long_shadow

    # — Day-2 确认（向量化）—
    d1_open  = o.shift(1)
    d1_close = c.shift(1)
    d1_mid   = (d1_open + d1_close) / 2
    d2_close = c
    d2_open  = o

    cond_engulf   = d2_close < d1_open
    cond_midpoint = (~cond_engulf) & (d2_close < d1_mid)
    cond_bearish  = (~cond_engulf) & (~cond_midpoint) & (d2_close < d2_open) & (d2_close < d1_close)

    confirmed = cond_engulf | cond_midpoint | cond_bearish

    # 触发 = Day-1 候选（前一根）+ Day-2 确认（当根）
    # 在位置 i，Day-1 = i-1，Day-2 = i
    triggered_at = day1_mask.shift(1) & confirmed   # Day-1 在 i-1 位置触发，Day-2 在 i 确认
    triggered_at = triggered_at.fillna(False)

    # — 构建结果 —
    rows = []
    n = len(df)
    for i in df.index[triggered_at]:
        pos = df.index.get_loc(i)   # confirm (Day-2) 位置
        pos_d1 = pos - 1            # Day-1 位置

        # 评分
        score = 1
        if bool(is_star.iloc[pos_d1]):
            score += 1
        if bool(vol_spike.iloc[pos_d1]):
            score += 1
        if bool(cond_engulf.iloc[pos]):
            score += 1
        # 加分项：确认日实体 / Day-1 上影线
        d1_upper_val = float(upper_shd.iloc[pos_d1])
        d2_body_val  = float(body.iloc[pos])
        cbr = d2_body_val / d1_upper_val if d1_upper_val > 0 else 0.0
        if cbr >= CONFIRM_BODY_RATIO_MIN:
            score += 1

        confirm_mode = (
            "engulf_open"    if bool(cond_engulf.iloc[pos]) else
            "below_midpoint" if bool(cond_midpoint.iloc[pos]) else
            "bearish_close"
        )

        row: dict = {
            "signal_date":   str(df["date"].iloc[pos_d1])[:10],
            "confirm_date":  str(df["date"].iloc[pos])[:10],
            "is_shooting_star": bool(is_star.iloc[pos_d1]),
            "shadow_ratio":  round(float(shadow_ratio.iloc[pos_d1]), 2),
            "shadow_pct":    round(float(abs_shadow_pct.iloc[pos_d1]) * 100, 2),  # 上影线/收盘价 %
            "vol_spike":     bool(vol_spike.iloc[pos_d1]),
            "vol_ratio":     round(float(vol_ratio_series.iloc[pos_d1]), 2),
            "confirm_body_ratio": round(cbr, 2),
            "confirm_mode":  confirm_mode,
            "score":         score,
            "day1_high":     round(float(h.iloc[pos_d1]), 4),
            "day1_close":    round(float(c.iloc[pos_d1]), 4),
        }

        # 后续收益（以 Day-1 收盘为基准）
        base = float(c.iloc[pos_d1])
        for fd in forward_days:
            end_pos = min(pos + fd, n - 1)
            fwd_c   = float(c.iloc[end_pos])
            fwd_min = float(l.iloc[pos + 1: end_pos + 1].min()) if pos + 1 <= end_pos else base
            row[f"fwd_ret_{fd}d"]  = round((fwd_c - base) / base * 100, 2)
            row[f"fwd_min_{fd}d"]  = round((fwd_min - base) / base * 100, 2)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    return result
