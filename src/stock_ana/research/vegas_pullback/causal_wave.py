"""浪结构的 as-of 因果视图：只暴露「在 asof 当天已经可见」的浪信息。

背景：analyze_wave_structure 在全量历史上拟合，浪的资格（≥40 bar 峰龄、
≥15% 涨幅）、终点 refinement、synthetic 截断都可能用到 asof 之后的数据。
实测 48% 的 mid 触碰所在浪的峰在触碰之后——「这个浪存在」本身就是未来
信息（≈ 价格后来涨了），直接作特征即泄漏。

本模块把全量浪列表投影成 asof 时点的可见状态：

  1. 起点在 asof 之后的浪 → 不可见；
  2. 起点在过去、但截至 asof 尚未满足资格（asof 前的涨幅 <15% 或
     峰龄 <40 bar）的浪 → 不可见（实时跑浪识别时它还不是一个浪）；
  3. 可见浪的 peak / rise / duration / sub_waves 全部截断到 ≤ asof：
     peak = argmax(close[start..asof])，rise = 该峰相对起点涨幅；
  4. 终点在 asof 之后（含 synthetic 截断在未来）→ 视为进行中（end=None）。

wave_number / connected_prev 只由「更早的浪」决定（它们的数据全在
asof 之前），直接沿用全量编号。

仅供训练特征构造使用；生产实时打分天然只有 ≤今天 的数据，两侧口径一致。
"""

from __future__ import annotations

import numpy as np

_MIN_WAVE_BARS = 40        # 与 analyze_wave_structure 的 min_wave_bars 一致
_MIN_WAVE_RISE_PCT = 15.0  # 与 min_wave_rise_pct 一致


def causal_wave_view(waves: list[dict], close: np.ndarray, asof: int) -> list[dict]:
    """把全量浪列表投影为 asof 时点的可见浪列表（零前瞻）。"""
    view: list[dict] = []
    for w in waves:
        sp = w["start_pivot"]["iloc"]
        if sp > asof:
            break  # 后面的浪都在未来
        sp_val = float(w["start_pivot"]["value"])

        # 截至 asof 的峰（只用 start..asof 的收盘）
        seg = close[sp : asof + 1].astype(float)
        if seg.size == 0 or sp_val <= 0:
            continue
        peak_off = int(np.argmax(seg))
        peak_val = float(seg[peak_off])
        peak_iloc = sp + peak_off
        rise_by_asof = (peak_val / sp_val - 1) * 100
        duration_by_asof = peak_iloc - sp

        # 资格必须在 asof 前已达成，否则实时视角下它还不是一个浪
        if duration_by_asof < _MIN_WAVE_BARS or rise_by_asof < _MIN_WAVE_RISE_PCT:
            continue

        ep = w.get("end_pivot")
        ep_visible = ep if (ep is not None and ep["iloc"] <= asof) else None

        wv = dict(w)
        wv["peak_pivot"] = {
            "type": "H", "iloc": peak_iloc, "value": peak_val,
        }
        wv["rise_pct"] = round(rise_by_asof, 2)
        wv["end_pivot"] = ep_visible
        if ep_visible is None:
            wv["end_boundary_id"] = None
        wv["sub_waves"] = [
            s for s in w.get("sub_waves", [])
            if s.get("end_pivot") and s["end_pivot"]["iloc"] <= asof
        ]
        wv["sub_wave_count"] = len(wv["sub_waves"])
        view.append(wv)
    return view


def containing_wave(view: list[dict], bar: int) -> dict | None:
    """asof 视图中包含 bar 的浪（终点为 None 视为延伸到 asof）。"""
    best = None
    for w in view:
        sp = w["start_pivot"]["iloc"]
        ep = w["end_pivot"]["iloc"] if w["end_pivot"] else float("inf")
        if sp <= bar <= ep:
            best = w
    if best is None:
        for w in reversed(view):
            if w["start_pivot"]["iloc"] <= bar:
                best = w
                break
    return best
