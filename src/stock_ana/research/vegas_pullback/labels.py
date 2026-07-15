"""浪结构确认标签：一次 Vegas 回踩后该「抄底」还是「跑路」。

标签只用信号日 t 之后的走势构造（训练目标可用未来；特征另在 ≤t 侧构造，
零泄漏）。判定完全对齐已修好的浪结构与策略假设：

  bounce（抄底成功） : 回踩止于 Vegas 通道并重拾升势——在深破通道之前
                       先创出高于回踩前浪顶的新高。
  breakdown（该跑路）: 回踩失败——先深破通道（≥breach_days 连续收盘 <
                       通道 × breach_margin）并随后创出低于回踩低点的更低低点
                       （结构破坏 / CHoCH）。
  ambiguous          : 前瞻窗口内两者都未明确触发——训练时丢弃。

「先到先得」：新高先于深破 → bounce；深破先于新高 → breakdown。

支持 mid / long 两种通道：
  mid  通道上沿 = max(EMA34, EMA55)
  long 通道上沿 = max(EMA144, EMA169, EMA200)
"""

from __future__ import annotations

import numpy as np

from stock_ana.strategies.primitives.vegas_zones import MID_EMAS, LONG_EMAS


def _channel_upper(emas: dict[int, np.ndarray], spans: list[int]) -> np.ndarray:
    """通道上沿逐 bar 序列 = 各 span EMA 的逐点最大值。"""
    arr = emas[spans[0]].astype(float).copy()
    for s in spans[1:]:
        arr = np.maximum(arr, emas[s].astype(float))
    return arr


def label_pullback_outcome(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    emas: dict[int, np.ndarray],
    touch_bar: int,
    prior_peak: float,
    support: str = "long",
    horizons: tuple[int, ...] = (21, 42, 63),
    max_horizon: int = 126,
    new_high_eps: float = 0.01,
    breach_days: int = 3,
    breach_margin: float = 0.97,
) -> dict:
    """对 touch_bar 处的一次回踩产出结构标签 + 多前瞻窗口辅助量。

    Args:
        close/high/low: 全序列 ndarray。
        emas: compute_vegas_emas 结果（含 34/55/144/169/200）。
        touch_bar: 回踩触碰 bar（信号锚点）。
        prior_peak: 回踩前的浪顶价（来自浪结构；无浪时用近端最高价兜底）。
        support: "mid" | "long" —— 决定用哪组通道判深破。
        horizons: 计算前瞻收益/回撤的窗口（辅助列，不决定 label）。
        max_horizon: 结构判定的最大前瞻 bar 数（新高/深破谁先到）。
        new_high_eps: 创新高需超过 prior_peak × (1+eps)。
        breach_days: 连续收盘 < 通道×margin 达到该天数即「深破」。
        breach_margin: 深破阈值（0.97 = 跌破通道 3%）。

    Returns dict with keys:
        label ("bounce" | "breakdown" | "ambiguous")
        label_reason
        breach_bar, new_high_bar (相对 touch_bar 的偏移；未触发 = -1)
        prior_peak
        fwd_ret_{h}     : t→t+h 收盘收益 %
        fwd_maxdd_{h}   : t→t+h 期间最深回撤 %（相对 t 收盘）
        fwd_maxup_{h}   : t→t+h 期间最大涨幅 %（相对 t 收盘）
        future_bars     : 实际可用的未来 bar 数（右侧删失诊断）
    """
    spans = MID_EMAS if support == "mid" else LONG_EMAS
    lv = _channel_upper(emas, spans)
    n = len(close)
    c0 = float(close[touch_bar])

    result: dict = {
        "label": "ambiguous",
        "label_reason": "",
        "breach_bar": -1,
        "new_high_bar": -1,
        "prior_peak": round(float(prior_peak), 4),
    }

    # ── 多前瞻窗口辅助量（收益/回撤/涨幅）──
    future_bars = n - 1 - touch_bar
    result["future_bars"] = int(future_bars)
    for h in horizons:
        end = min(n - 1, touch_bar + h)
        if end > touch_bar and c0 > 0:
            seg_close = close[touch_bar : end + 1]
            seg_low = low[touch_bar : end + 1]
            seg_high = high[touch_bar : end + 1]
            result[f"fwd_ret_{h}"] = round((float(close[end]) / c0 - 1) * 100, 2)
            result[f"fwd_maxdd_{h}"] = round((float(np.min(seg_low)) / c0 - 1) * 100, 2)
            result[f"fwd_maxup_{h}"] = round((float(np.max(seg_high)) / c0 - 1) * 100, 2)
        else:
            result[f"fwd_ret_{h}"] = np.nan
            result[f"fwd_maxdd_{h}"] = np.nan
            result[f"fwd_maxup_{h}"] = np.nan

    # ── 结构标签：新高 vs 深破，谁先到 ──
    end = min(n, touch_bar + max_horizon + 1)
    target = prior_peak * (1 + new_high_eps)
    breach_bar = -1
    new_high_bar = -1
    consec = 0
    for i in range(touch_bar + 1, end):
        # 深破检测
        lvv = float(lv[i])
        if lvv > 0 and float(close[i]) < lvv * breach_margin:
            consec += 1
            if consec >= breach_days and breach_bar < 0:
                breach_bar = i
        else:
            consec = 0
        # 创新高检测
        if new_high_bar < 0 and float(high[i]) >= target:
            new_high_bar = i
        # 谁先到即可判定
        if new_high_bar >= 0 and (breach_bar < 0 or new_high_bar <= breach_bar):
            break
        if breach_bar >= 0 and (new_high_bar < 0 or breach_bar < new_high_bar):
            break

    result["breach_bar"] = int(breach_bar - touch_bar) if breach_bar >= 0 else -1
    result["new_high_bar"] = int(new_high_bar - touch_bar) if new_high_bar >= 0 else -1

    # ── 买点标签（收益导向三重栅栏）──
    result.update(label_buy_outcome(close, high, low, touch_bar))
    # ── 大浪买点标签（大二浪猎手：先 +20% vs 先 -10%）──
    big = label_buy_outcome(close, high, low, touch_bar, profit_pct=20.0, stop_pct=10.0)
    result["label_buy_big"] = big["label_buy"]
    result["buy_big_outcome_bar"] = big["buy_outcome_bar"]

    has_new_high = new_high_bar >= 0
    has_breach = breach_bar >= 0

    _finalize_structure_label(result, has_new_high, has_breach,
                              new_high_bar, breach_bar,
                              low, touch_bar, end)
    return result


def _finalize_structure_label(result, has_new_high, has_breach,
                              new_high_bar, breach_bar, low, touch_bar, end) -> None:
    """结构标签收尾（从主函数拆出便于复用；就地写入 result）。"""

    if has_new_high and (not has_breach or new_high_bar <= breach_bar):
        # 深破前先创新高 → 回踩止于通道、升势重拾
        result["label"] = "bounce"
        result["label_reason"] = "new_high_before_breach"
    elif has_breach and (not has_new_high or breach_bar < new_high_bar):
        # 需再确认破位（深破后创更低低点 = 结构破坏），否则算 ambiguous
        after = low[breach_bar : end]
        pull_low = float(np.min(low[max(0, touch_bar - 2) : touch_bar + 3]))
        lower_low = bool(after.size and float(np.min(after)) < pull_low)
        if lower_low:
            result["label"] = "breakdown"
            result["label_reason"] = "breach_then_lower_low"
        else:
            result["label"] = "ambiguous"
            result["label_reason"] = "breach_no_lower_low"
    else:
        result["label"] = "ambiguous"
        result["label_reason"] = "unresolved_within_horizon"

    return result


def label_buy_outcome(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    anchor: int,
    profit_pct: float = 12.0,
    stop_pct: float = 8.0,
    horizon: int = 63,
) -> dict:
    """收益导向的「买点」标签：路径版三重栅栏（回答「该不该抄底」）。

    背景：结构标签（bounce/breakdown）衡量「通道是否守住」，与交易价值
    有 25~29% 的错位（先破位后暴涨 / 创新高后跌回）。实测按结构标签训练，
    模型分数与前瞻收益几乎无关（各分位 ret63 平坦），只与回撤相关——
    它学的是「拿得舒服」，不是「能赚钱」。买点标签直接以「先到哪条线」
    定义好坏交易：

      good_buy : horizon 内先触盈利线 close0×(1+profit_pct%)
      bad_buy  : 先触止损线 close0×(1-stop_pct%)；同日双触保守判 bad
                 （日线无法分辨盘中先后，宁可错杀）
      neutral  : 窗口内两线都未触 → 训练时丢弃

    Returns dict with keys:
        label_buy, buy_outcome_bar（相对 anchor 的偏移，未触发 = -1）
    """
    n = len(close)
    c0 = float(close[anchor])
    if c0 <= 0:
        return {"label_buy": "neutral", "buy_outcome_bar": -1}
    profit_lvl = c0 * (1 + profit_pct / 100)
    stop_lvl = c0 * (1 - stop_pct / 100)

    end = min(n, anchor + horizon + 1)
    for i in range(anchor + 1, end):
        hit_stop = float(low[i]) <= stop_lvl
        hit_profit = float(high[i]) >= profit_lvl
        if hit_stop:  # 同日双触也判 bad（保守）
            return {"label_buy": "bad_buy", "buy_outcome_bar": int(i - anchor)}
        if hit_profit:
            return {"label_buy": "good_buy", "buy_outcome_bar": int(i - anchor)}
    return {"label_buy": "neutral", "buy_outcome_bar": -1}
