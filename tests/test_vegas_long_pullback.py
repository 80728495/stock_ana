import numpy as np
import pandas as pd

from stock_ana.strategies.impl.vegas_long import (
    check_long_wave_structure,
    classify_long_signal,
    compute_lv_respect_stats,
    detect_long_touch_and_hold,
    locate_wave_pullback,
    score_long_pullback,
)
from stock_ana.strategies.primitives.vegas_zones import compute_vegas_emas


# ─────────────────────────────────────────────────────────────
# Synthetic data helpers
# ─────────────────────────────────────────────────────────────

def _make_uptrend_pullback_df(
    n: int = 420,
    daily_growth: float = 0.002,
    pullback_bars: int = 8,
) -> pd.DataFrame:
    """指数上涨 + 末尾回踩到 Long Vegas 上沿的合成序列。"""
    close = 10.0 * np.cumprod(np.full(n, 1 + daily_growth))

    # 末尾 pullback_bars 根 K 线从高点线性回落到 LV 上沿附近
    emas = compute_vegas_emas(pd.Series(close))
    lv_upper_est = max(emas[144][-1], emas[169][-1], emas[200][-1])
    peak = close[n - pullback_bars - 1]
    target = lv_upper_est * 1.005
    for i in range(pullback_bars):
        frac = (i + 1) / pullback_bars
        close[n - pullback_bars + i] = peak + (target - peak) * frac

    low = close * 0.985
    high = close * 1.01
    open_ = close * 0.998
    dates = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": np.ones(n) * 1e6},
        index=dates,
    )


def _wave(num: int, end_iloc: int | None, synthetic: bool = False) -> dict:
    """构造最小化的大浪 dict（只含统计函数用到的字段）。"""
    end_pivot = None
    if end_iloc is not None:
        end_pivot = {"type": "L", "iloc": end_iloc, "value": 100.0}
        if synthetic:
            end_pivot["synthetic"] = True
    return {"wave_number": num, "end_pivot": end_pivot}


# ─────────────────────────────────────────────────────────────
# compute_lv_respect_stats
# ─────────────────────────────────────────────────────────────

def _flat_lv_env(n: int = 500, lv: float = 100.0, px: float = 110.0):
    """LV 恒为 lv、收盘恒在上方的最简环境（emas 三条长线同值）。"""
    close = np.full(n, px)
    emas = {s: np.full(n, lv) for s in (144, 169, 200)}
    return close, emas


def test_lv_respect_stats_held_when_no_deep_breach_after_touch():
    close, emas = _flat_lv_env()
    # 两次已完成回踩 + 一个进行中浪；回踩后价格始终在 LV 上方 → 全部 held
    waves = [_wave(1, 100), _wave(2, 200), _wave(3, None)]

    stats = compute_lv_respect_stats(waves, close, emas)

    assert stats["n_events"] == 2
    assert stats["n_held"] == 2
    assert stats["n_breach"] == 0
    assert stats["respect_rate"] == 1.0
    assert stats["qualified"] is True


def test_lv_respect_stats_counts_breach_as_failure():
    close, emas = _flat_lv_env()
    # 回踩点 200 之后连续 3 日收盘 < LV*0.97 → 深破，计为失败
    close[205:208] = 96.0
    # synthetic 截断浪本身即深破
    waves = [_wave(1, 100), _wave(2, 200), _wave(1, 350, synthetic=True)]

    stats = compute_lv_respect_stats(waves, close, emas)

    assert stats["n_events"] == 3
    assert stats["n_held"] == 1
    assert stats["n_breach"] == 2
    assert stats["qualified"] is False


def test_lv_respect_stats_excludes_pending_last_touch():
    close, emas = _flat_lv_env()
    # 最后一浪刚在 495 结束（n=500，pending_margin=20 内）→ 不计入
    waves = [_wave(1, 100), _wave(2, 495)]

    stats = compute_lv_respect_stats(waves, close, emas)

    assert stats["n_events"] == 1
    assert stats["n_held"] == 1


def test_lv_respect_stats_empty_waves():
    close, emas = _flat_lv_env()
    stats = compute_lv_respect_stats([], close, emas)

    assert stats["n_events"] == 0
    assert stats["respect_rate"] == 0.0
    assert stats["qualified"] is False


# ─────────────────────────────────────────────────────────────
# locate_wave_pullback
# ─────────────────────────────────────────────────────────────

def _wave_full(num, start_iloc, peak_iloc, end_iloc, rise):
    """构造含 peak/end/rise 的浪 dict（locate_wave_pullback 用）。"""
    end_pivot = None
    if end_iloc is not None:
        end_pivot = {"type": "L", "iloc": end_iloc, "value": 100.0}
    return {
        "wave_number": num,
        "start_pivot": {"type": "L", "iloc": start_iloc, "value": 90.0},
        "peak_pivot": {"type": "H", "iloc": peak_iloc, "value": 150.0},
        "end_pivot": end_pivot,
        "rise_pct": rise,
    }


def test_locate_maps_touch_to_terminated_wave_not_next():
    # W1 结束于 i200（= W2 起点）。触碰在 i200 应映射到 W1（第 1 次回踩），
    # 不能像 find_wave_context 那样返回 W2。
    waves = [
        _wave_full(1, 100, 160, 200, 55.0),
        _wave_full(2, 200, 280, 340, 60.0),
    ]
    pb = locate_wave_pullback(waves, touch_bar=200)

    assert pb["is_wave_end"] is True
    assert pb["seq"] == 1
    assert pb["wave_rise_pct"] == 55.0


def test_locate_second_pullback():
    waves = [
        _wave_full(1, 100, 160, 200, 55.0),
        _wave_full(2, 200, 280, 340, 60.0),
    ]
    pb = locate_wave_pullback(waves, touch_bar=338)

    assert pb["is_wave_end"] is True
    assert pb["seq"] == 2
    assert pb["wave_rise_pct"] == 60.0


def test_locate_intrawave_dip_is_not_wave_end():
    # 触碰在 W1 上涨途中（远离任何 end_pivot、且不在进行中末浪的峰后）
    waves = [
        _wave_full(1, 100, 160, 200, 55.0),
        _wave_full(2, 200, 280, 340, 60.0),
    ]
    pb = locate_wave_pullback(waves, touch_bar=130)

    assert pb["is_wave_end"] is False
    assert pb["seq"] == 0
    assert pb["wave"] is None


def test_locate_inprogress_last_wave_after_peak():
    # 末浪仍在进行（end=None），触碰在其峰之后 → 实时回踩，终结该浪
    waves = [
        _wave_full(1, 100, 160, 200, 55.0),
        _wave_full(2, 200, 280, None, 60.0),
    ]
    pb = locate_wave_pullback(waves, touch_bar=300)

    assert pb["is_wave_end"] is True
    assert pb["seq"] == 2


def test_locate_before_peak_of_inprogress_is_not_end():
    waves = [_wave_full(1, 100, 260, None, 55.0)]
    pb = locate_wave_pullback(waves, touch_bar=180)  # 峰(260)之前

    assert pb["is_wave_end"] is False
    assert pb["seq"] == 0


def test_locate_empty_waves():
    pb = locate_wave_pullback([], touch_bar=100)
    assert pb["is_wave_end"] is False
    assert pb["seq"] == 0


# ─────────────────────────────────────────────────────────────
# score_long_pullback / classify_long_signal
# ─────────────────────────────────────────────────────────────

def test_score_first_pullback_with_strong_history_is_strong_buy():
    score, details = score_long_pullback(
        pullback_seq=1,
        respect_rate=0.8,
        respect_n=4,
        long_slope_strong=True,
        wave_rise_pct=60.0,
    )

    assert details == {"seq": 2, "history": 2, "slope": 1, "wave_rise": 1}
    assert score == 6
    assert classify_long_signal(score) == "STRONG_BUY"


def test_score_late_pullback_after_blowoff_is_avoid():
    score, details = score_long_pullback(
        pullback_seq=4,
        respect_rate=0.5,
        respect_n=2,
        long_slope_strong=False,
        wave_rise_pct=300.0,
    )

    assert details["seq"] == -2
    assert details["wave_rise"] == -2
    assert score < 0
    assert classify_long_signal(score) == "AVOID"


def test_classify_thresholds():
    assert classify_long_signal(4) == "STRONG_BUY"
    assert classify_long_signal(2) == "BUY"
    assert classify_long_signal(0) == "HOLD"
    assert classify_long_signal(-1) == "AVOID"


# ─────────────────────────────────────────────────────────────
# check_long_wave_structure
# ─────────────────────────────────────────────────────────────

def test_structure_gate_passes_on_uptrend_pullback():
    df = _make_uptrend_pullback_df()
    close = df["close"].values
    low = df["low"].values
    emas = compute_vegas_emas(df["close"])

    struct = check_long_wave_structure(len(df) - 1, close, low, emas)

    assert struct["long_rising"] is True
    assert struct["long_order"] is True
    assert struct["above_ratio_ok"] is True
    assert struct["rise_1y_ok"] is True
    assert struct["peak_gap_ok"] is True
    assert struct["passed"] is True


def test_structure_gate_rejects_downtrend():
    n = 420
    close = 100.0 * np.cumprod(np.full(n, 1 - 0.001))
    low = close * 0.99
    emas = compute_vegas_emas(pd.Series(close))

    struct = check_long_wave_structure(n - 1, close, low, emas)

    assert struct["passed"] is False
    assert struct["long_rising"] is False


# ─────────────────────────────────────────────────────────────
# detect_long_touch_and_hold + screen integration
# ─────────────────────────────────────────────────────────────

def test_detect_long_touch_and_hold_finds_terminal_touch():
    df = _make_uptrend_pullback_df()
    close = df["close"].values
    low = df["low"].values
    emas = compute_vegas_emas(df["close"])

    signals = detect_long_touch_and_hold(close, low, emas)

    assert signals, "末尾回踩到 LV 上沿应产生触碰信号"
    last = signals[-1]
    assert last["support_type"] == "long_vegas"
    assert last["touch_bar"] >= len(df) - 10


def test_screen_vegas_long_pullback_smoke():
    from stock_ana.strategies.api import screen_vegas_long_pullback

    df = _make_uptrend_pullback_df()
    decision = screen_vegas_long_pullback(df, lookback=5)

    assert decision.setup_type == "vegas_long_pullback"
    assert "lv_stats" in decision.features
    if decision.passed:
        best = decision.features["signals"][0]
        # 合成序列无历史大浪 → 统计门槛不通过 → 只能是 AVOID
        assert best["stats_qualified"] is False
        assert best["signal"] == "AVOID"


def test_screen_insufficient_data():
    from stock_ana.strategies.api import screen_vegas_long_pullback

    df = _make_uptrend_pullback_df(n=100)
    decision = screen_vegas_long_pullback(df, lookback=1)

    assert decision.passed is False
    assert decision.reason == "insufficient_data"
