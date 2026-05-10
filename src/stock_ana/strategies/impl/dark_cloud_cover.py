"""
顶部反转形态识别 — 策略三：跳空巨阴线（Gap & Cram）

单根 K 线形态（Day[0] 自身即为完整信号，无需次日确认）：

  Day[0] (今天)：跳空高开后巨量大阴线
    硬门槛：
    - 跳空高开：open > Day[1] close × (1 + MIN_GAP_PCT)
    - 收阴：close < open
    - 大实体阴线：实体 ≥ 近20日平均实体 × BEAR_BODY_MULT
    - 放量：vol / Day[1] vol ≥ 1.3x 或 vol / vol_ma_50 ≥ 1.5x

  Day[1] 不限方向（阴线/阳线均可），不要求处于新高区域。
  这是事件驱动型形态（财报、消息），核心是"跳空诱多后单边杀跌"。

  跳空吃回比 gap_reclaim：
    = (open - close) / (open - Day[1] close)
    表示跳空涨幅被阴线吃回了多少，≥70% 为加分项，
    ≥100%（收盘 < 昨收）说明跳空完全作废。

评分体系（0~5 分）：
  1  基础触发（跳空 + 大阴线 + 放量）
  +1 吃回跳空 ≥ 70%（或 close ≤ Day[1] close）
  +1 巨量（vol ≥ vol_ma_50 × 2.0）
  +1 尾盘封死（(close - low) / (high - low) ≤ 0.15）
  +1 阶段新高区域（20日内新高附近）

Public API
----------
detect_dark_cloud_cover(df) -> dict
    检测最新两根 K 线是否触发。

scan_history(df) -> pd.DataFrame
    向量化历史全量扫描。
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# ── 参数常量 ──────────────────────────────────────────────────────────────────
NEW_HIGH_WINDOW: int = 20            # 阶段新高回望窗口（加分用）
NEW_HIGH_LOOKBACK: int = 3           # 近N日内出现新高的容忍窗口
NEAR_HIGH_PCT: float = 0.03          # 高点与近期新高最大偏差

# Day[0] 跳空巨阴线
MIN_GAP_PCT: float = 0.003           # 最小跳空幅度（open > d1_close × 1.003）
BEAR_BODY_MULT: float = 1.5          # 大实体阴线：实体 ≥ 近20日平均 × 此倍数

# 成交量
VOL_VS_D1_MIN: float = 1.3           # Day[0] vol / Day[1] vol ≥ 此值（相对前日）
VOL_VS_MA50_MIN: float = 1.5         # Day[0] vol / vol_ma_50 ≥ 此值（绝对水平）
VOL_STRONG: float = 2.0              # 放量加分阈值（vs vol_ma_50，+1分）
VOL_HUGE: float = 5.0                # 巨量加分阈值（vs vol_ma_50，+2分）

# 前驱涨幅分档（pre60_ret：Day[1] close vs 60日前 close，捕捉中长期趋势强度）
PRE60_L1: float = 10.0               # +1分门槛
PRE60_L2: float = 25.0               # +2分门槛
PRE60_L3: float = 50.0               # +3分门槛
PRE60_L4: float = 75.0               # +4分门槛

# 跳空吃回分档
GAP_RECLAIM_STRONG: float = 1.0      # 完全吃回（收盘≤昨收），+2分
GAP_RECLAIM_PARTIAL: float = 0.70    # 部分吃回≥70%，+1分

# 跳空幅度
GAP_BIG: float = 5.0                 # 跳空≥5% 加分


# ── 内部工具 ───────────────────────────────────────────────────────────────────

def _ensure_vol_ma(df: pd.DataFrame) -> pd.DataFrame:
    if "vol_ma_50" not in df.columns:
        df["vol_ma_50"] = df["volume"].astype(float).rolling(50, min_periods=1).mean()
    return df


# ── detect（最新两根 K 线） ─────────────────────────────────────────────────────

def detect_dark_cloud_cover(
    df: pd.DataFrame,
    new_high_window: int = NEW_HIGH_WINDOW,
    new_high_lookback: int = NEW_HIGH_LOOKBACK,
    near_high_pct: float = NEAR_HIGH_PCT,
) -> dict:
    """检测 df 最新两根 K 线是否构成跳空巨阴线形态。"""
    _empty = dict(triggered=False, score=0, reason="数据不足")
    if len(df) < max(new_high_window + 3, 30):
        return _empty

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df = _ensure_vol_ma(df)

    d1 = df.iloc[-2]  # Day[1]
    d0 = df.iloc[-1]  # Day[0]

    signal_date = str(df.index[-1])[:10]
    prev_date   = str(df.index[-2])[:10]

    o1, h1, l1, c1 = float(d1["open"]), float(d1["high"]), float(d1["low"]), float(d1["close"])
    o0, h0, l0, c0 = float(d0["open"]), float(d0["high"]), float(d0["low"]), float(d0["close"])

    # ── 跳空高开 ──
    if o0 <= c1 * (1 + MIN_GAP_PCT):
        return dict(triggered=False, score=0,
                    reason=f"Day[0] 未跳空高开（open={o0:.2f} vs D1 close={c1:.2f}）")

    # ── 收阴 ──
    if c0 >= o0:
        return dict(triggered=False, score=0, reason="Day[0] 不是阴线")

    # ── 大实体阴线 ──
    d0_body = o0 - c0
    recent_body = (df["close"].iloc[-22:-2] - df["open"].iloc[-22:-2]).abs()
    avg_body = recent_body.mean()
    d0_body_mult = d0_body / avg_body if avg_body > 0 else 0
    if d0_body_mult < BEAR_BODY_MULT:
        return dict(triggered=False, score=0,
                    reason=f"Day[0] 实体倍数 {d0_body_mult:.1f}x < {BEAR_BODY_MULT}x")

    # ── 放量（OR） ──
    v0, v1_vol = float(d0["volume"]), float(d1["volume"])
    vol_ma50 = float(d0["vol_ma_50"]) if float(d0["vol_ma_50"]) > 0 else 1.0

    vol_vs_d1 = v0 / max(v1_vol, 1.0)
    vol_vs_ma50 = v0 / vol_ma50

    if vol_vs_d1 < VOL_VS_D1_MIN and vol_vs_ma50 < VOL_VS_MA50_MIN:
        return dict(triggered=False, score=0,
                    reason=f"Day[0] 量不足（vs D1={vol_vs_d1:.1f}x, vs MA50={vol_vs_ma50:.1f}x）")

    # ── 衍生指标 ──
    gap_pct = (o0 / c1 - 1) * 100
    gap_reclaim = (o0 - c0) / (o0 - c1) if (o0 - c1) > 1e-9 else 0
    close_below_prev = c0 <= c1

    # 前60日涨幅（Day[1] close vs 60日前收盘）
    pre60_ref = float(df["close"].iloc[max(0, len(df) - 2 - 60)])
    pre60_ret = (c1 / pre60_ref - 1) * 100 if pre60_ref > 0 else 0.0

    # ── 评分（0~9） ──
    score = 0
    # 前驱涨幅（最大4分）
    if pre60_ret >= PRE60_L4:   score += 4
    elif pre60_ret >= PRE60_L3: score += 3
    elif pre60_ret >= PRE60_L2: score += 2
    elif pre60_ret >= PRE60_L1: score += 1
    # 跳空吃回（最大2分）
    if gap_reclaim >= GAP_RECLAIM_STRONG:  score += 2
    elif gap_reclaim >= GAP_RECLAIM_PARTIAL: score += 1
    # 量能（最大2分）
    if vol_vs_ma50 >= VOL_HUGE:    score += 2
    elif vol_vs_ma50 >= VOL_STRONG: score += 1
    # 跳空幅度（1分）
    if gap_pct >= GAP_BIG: score += 1

    reason = (
        f"{prev_date} → {signal_date} 跳空{gap_pct:+.1f}%后巨阴"
        f"（pre60={pre60_ret:+.0f}%，吃回{gap_reclaim:.0%}，"
        f"vol/MA50={vol_vs_ma50:.1f}x）"
    )

    return dict(
        triggered=True,
        signal_date=signal_date,
        prev_date=prev_date,
        gap_pct=round(gap_pct, 2),
        gap_reclaim=round(gap_reclaim, 3),
        close_below_prev=close_below_prev,
        pre60_ret=round(pre60_ret, 1),
        d0_body_mult=round(d0_body_mult, 2),
        d0_vol_vs_d1=round(vol_vs_d1, 2),
        d0_vol_vs_ma50=round(vol_vs_ma50, 2),
        is_near_high=h0 >= float(df["high"].iloc[max(0, len(df)-2-new_high_window):len(df)-2].max()) * (1 - near_high_pct),
        day0_high=round(h0, 4),
        day0_close=round(c0, 4),
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
    """向量化扫描历史中所有跳空巨阴线信号。"""
    if len(df) < max(new_high_window + 3, 30):
        return pd.DataFrame()

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().reset_index()
    df = _ensure_vol_ma(df)

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float)
    vm50 = df["vol_ma_50"].astype(float)

    body = (c - o).abs().replace(0, 1e-9)
    rng = (h - l).replace(0, 1e-9)

    c1 = c.shift(1)
    v1 = v.shift(1)

    # Day[0] 跳空高开
    gap_up = o > c1 * (1 + MIN_GAP_PCT)

    # Day[0] 收阴
    is_bearish = c < o

    # Day[0] 大实体阴线
    d0_body = o - c  # 阴线实体（正数）
    avg_body_20 = body.rolling(20, min_periods=5).mean().shift(1)
    d0_body_mult = d0_body / avg_body_20.replace(0, 1e-9)
    is_big_bear = d0_body_mult >= BEAR_BODY_MULT

    # 放量 OR
    vol_vs_d1 = v / v1.replace(0, 1e-9)
    vol_vs_ma50 = v / vm50.replace(0, 1e-9)
    vol_ok = (vol_vs_d1 >= VOL_VS_D1_MIN) | (vol_vs_ma50 >= VOL_VS_MA50_MIN)

    # 触发
    triggered = gap_up & is_bearish & is_big_bear & vol_ok
    triggered = triggered.fillna(False)

    # ── 评分向量 ──
    gap_reclaim = (o - c) / (o - c1).replace(0, 1e-9)
    close_below_prev = c <= c1

    # 前60日涨幅（向量化：c1 vs 60日前收盘）
    c60_ago = c.shift(61)
    pre60_ret = (c1 / c60_ago.replace(0, 1e-9) - 1) * 100

    # 阶段新高
    rolling_high = h.rolling(new_high_window, min_periods=5).max()
    is_near_high = h >= rolling_high.shift(1) * (1 - near_high_pct)

    # ── 构建结果 ──
    rows = []
    n = len(df)
    for i in df.index[triggered]:
        pos = df.index.get_loc(i)
        pos_d1 = pos - 1
        if pos_d1 < 0:
            continue

        _reclaim  = float(gap_reclaim.iloc[pos])
        _pre60    = float(pre60_ret.iloc[pos]) if not pd.isna(pre60_ret.iloc[pos]) else 0.0
        _vm50     = float(vol_vs_ma50.iloc[pos])
        _gap      = (float(o.iloc[pos]) / float(c.iloc[pos_d1]) - 1) * 100

        # 评分（0~9）
        score = 0
        # 前60日涨幅（最大4分）
        if _pre60 >= PRE60_L4:   score += 4
        elif _pre60 >= PRE60_L3: score += 3
        elif _pre60 >= PRE60_L2: score += 2
        elif _pre60 >= PRE60_L1: score += 1
        # 跳空吃回（最大2分）
        if _reclaim >= GAP_RECLAIM_STRONG:    score += 2
        elif _reclaim >= GAP_RECLAIM_PARTIAL: score += 1
        # 量能（最大2分）
        if _vm50 >= VOL_HUGE:    score += 2
        elif _vm50 >= VOL_STRONG: score += 1
        # 跳空幅度（1分）
        if _gap >= GAP_BIG: score += 1

        row: dict = {
            "signal_date":    str(df["date"].iloc[pos])[:10],
            "prev_date":      str(df["date"].iloc[pos_d1])[:10],
            "gap_pct":        round(_gap, 2),
            "gap_reclaim":    round(_reclaim, 3),
            "close_below_prev": bool(close_below_prev.iloc[pos]),
            "pre60_ret":      round(_pre60, 1),
            "d0_body_mult":   round(float(d0_body_mult.iloc[pos]), 2),
            "d0_vol_vs_d1":   round(float(vol_vs_d1.iloc[pos]), 2),
            "d0_vol_vs_ma50": round(_vm50, 2),
            "is_near_high":   bool(is_near_high.iloc[pos]),
            "day0_high":      round(float(h.iloc[pos]), 4),
            "day0_close":     round(float(c.iloc[pos]), 4),
            "score":          score,
        }

        # 后续收益（以 Day[0] 收盘为基准）
        base = float(c.iloc[pos])
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
