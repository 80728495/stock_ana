import numpy as np
import pandas as pd

from stock_ana.strategies.impl.vegas_mid import backward_consecutive_count
from stock_ana.strategies.primitives.wave import analyze_wave_structure


# ─────────────────────────────────────────────────────────────
# backward_consecutive_count (connected_prev based)
# ─────────────────────────────────────────────────────────────

def _major(num, start_iloc, start_val, rise, connected_prev):
    """构造最小化的大浪 dict（只含连续性判定用到的字段）。"""
    return {
        "wave_number": num,
        "start_pivot": {"type": "L", "iloc": start_iloc, "value": start_val},
        "end_pivot": {"type": "L", "iloc": start_iloc + 100, "value": start_val * 1.1},
        "rise_pct": rise,
        "connected_prev": connected_prev,
    }


def test_consec_counts_connected_chain():
    # 三浪连续链：每浪 connected_prev=True
    waves = [
        _major(1, 100, 10.0, 50.0, connected_prev=False),
        _major(2, 200, 12.0, 40.0, connected_prev=True),
        _major(3, 300, 15.0, 30.0, connected_prev=True),
    ]

    assert backward_consecutive_count(waves, waves[-1]) == 3
    assert backward_consecutive_count(waves, waves[1]) == 2
    assert backward_consecutive_count(waves, waves[0]) == 1


def test_consec_stops_at_disconnect():
    # 浪2 与浪1 断开（浪间深破 / 新一轮浪段）→ 链在此重置
    waves = [
        _major(1, 100, 10.0, 50.0, connected_prev=False),
        _major(1, 400, 12.0, 40.0, connected_prev=False),
        _major(2, 500, 15.0, 30.0, connected_prev=True),
    ]

    assert backward_consecutive_count(waves, waves[-1]) == 2


def test_consec_stops_at_break():
    # 浪2 connected_prev=False（与浪1 之间深破）→ 不连续
    waves = [
        _major(1, 100, 10.0, 50.0, connected_prev=False),
        _major(1, 300, 12.0, 40.0, connected_prev=False),
    ]

    assert backward_consecutive_count(waves, waves[-1]) == 1


def test_consec_requires_predecessor_min_rise():
    # 前驱浪 rise 12% < 20% → 链在此断开（当前浪自身不要求）
    waves = [
        _major(1, 100, 10.0, 12.0, connected_prev=False),
        _major(2, 200, 11.0, 8.0, connected_prev=True),
    ]

    assert backward_consecutive_count(waves, waves[-1]) == 1


def test_consec_unknown_wave_returns_zero():
    waves = [_major(1, 100, 10.0, 50.0, connected_prev=False)]
    stranger = _major(9, 999, 1.0, 1.0, connected_prev=False)

    assert backward_consecutive_count(waves, stranger) == 0


# ─────────────────────────────────────────────────────────────
# analyze_wave_structure integration (合成多浪序列)
# ─────────────────────────────────────────────────────────────

def _make_multiwave_df(
    warmup_bars: int = 230,
    n_waves: int = 3,
    rally_bars: int = 75,
    rally_daily: float = 0.0042,
    pullback_bars: int = 18,
) -> pd.DataFrame:
    """暖机段 + 连续 n_waves 个「上涨→回踩到 LV」的大浪序列。

    每浪涨 ~37%，随后回踩到当时的 Long Vegas 上沿附近，构成
    LV 触点交接的连续浪链。
    """
    prices = list(10.0 * np.cumprod(np.full(warmup_bars, 1.0008)))

    for _ in range(n_waves):
        # 上涨段
        for _ in range(rally_bars):
            prices.append(prices[-1] * (1 + rally_daily))
        # 回踩段：线性回落到当时 LV 上沿 ×1.01
        s = pd.Series(prices)
        lv = max(
            s.ewm(span=144, adjust=False).mean().iloc[-1],
            s.ewm(span=169, adjust=False).mean().iloc[-1],
            s.ewm(span=200, adjust=False).mean().iloc[-1],
        )
        peak = prices[-1]
        # 回踩期间 LV 仍会缓慢上移，目标再上浮少许
        target = lv * 1.045
        for k in range(pullback_bars):
            frac = (k + 1) / pullback_bars
            prices.append(peak + (target - peak) * frac)
        # 触点停留几日（让 EMA8 zigzag 在低位形成 L pivot）
        for _ in range(4):
            prices.append(prices[-1] * 1.001)

    close = np.array(prices)
    dates = pd.bdate_range("2022-01-03", periods=len(close))
    return pd.DataFrame(
        {
            "open": close * 0.998,
            "high": close * 1.008,
            "low": close * 0.99,
            "close": close,
            "volume": np.ones(len(close)) * 1e6,
        },
        index=dates,
    )


def test_multiwave_numbering_increments_through_lv_touches():
    df = _make_multiwave_df()
    r = analyze_wave_structure(df)
    waves = r["major_waves"]

    assert len(waves) >= 2, f"应识别出至少 2 个大浪, got {len(waves)}"
    nums = [w["wave_number"] for w in waves]
    # 连续（浪间未深破 LV）的浪链：编号应递增而非全部重置为 1
    assert max(nums) >= 2, f"浪编号应递增, got {nums}"

    # 首浪 connected_prev 必为 False，递增浪必为 True
    assert waves[0]["connected_prev"] is False
    for w in waves[1:]:
        if w["wave_number"] > 1:
            assert w["connected_prev"] is True

    # 连续计数与编号一致
    assert backward_consecutive_count(waves, waves[-1]) == waves[-1]["wave_number"]


def test_no_waves_inside_ema_warmup():
    df = _make_multiwave_df()
    r = analyze_wave_structure(df)

    for w in r["major_waves"]:
        assert w["start_pivot"]["iloc"] >= 200, "暖机期内不应出现浪边界"


def test_downtrend_yields_no_waves():
    n = 500
    close = 100.0 * np.cumprod(np.full(n, 1 - 0.001))
    dates = pd.bdate_range("2022-01-03", periods=n)
    df = pd.DataFrame(
        {
            "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close,
            "volume": np.ones(n),
        },
        index=dates,
    )

    r = analyze_wave_structure(df)

    assert r["major_waves"] == []
