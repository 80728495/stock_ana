"""Vegas 回踩 ML 研究包的单元测试（标签 / 信号特征 / 构建）。"""

import numpy as np
import pandas as pd

from stock_ana.research.vegas_pullback.labels import label_pullback_outcome
from stock_ana.research.vegas_pullback.signal_features import (
    compute_signal_features,
    SIGNAL_FEATURE_COLS,
)
from stock_ana.research.vegas_pullback.feature_registry import (
    REALTIME_FEATURE_COLS,
    feature_group_for,
)
from stock_ana.strategies.primitives.vegas_zones import compute_vegas_emas


def _emas_from_close(close: np.ndarray) -> dict:
    return compute_vegas_emas(pd.Series(close.astype(float)))


# ─────────────────────────────────────────────────────────────
# labels
# ─────────────────────────────────────────────────────────────

def test_label_bounce_when_new_high_before_breach():
    # 触碰后一路创新高、从不深破 → bounce
    n = 400
    close = 10.0 * np.cumprod(np.full(n, 1.002))
    high = close * 1.01
    low = close * 0.99
    emas = _emas_from_close(close)
    tb = 300
    prior_peak = float(close[tb])  # 之后必然创新高
    out = label_pullback_outcome(close, high, low, emas, tb, prior_peak, support="long")

    assert out["label"] == "bounce"
    assert out["new_high_bar"] >= 0
    assert out["breach_bar"] == -1


def test_label_breakdown_when_deep_breach_and_lower_low():
    # 触碰后深破通道并创更低低点 → breakdown
    n = 400
    close = np.concatenate([
        10.0 * np.cumprod(np.full(300, 1.003)),      # 前段上涨建立通道
        np.linspace(float(10.0 * 1.003**300), 5.0, 100),  # 后段崩塌
    ])
    high = close * 1.01
    low = close * 0.99
    emas = _emas_from_close(close)
    tb = 300
    prior_peak = float(close[tb]) * 1.2  # 需要很高的新高才算 bounce，不会达到
    out = label_pullback_outcome(close, high, low, emas, tb, prior_peak, support="long")

    assert out["label"] == "breakdown"
    assert out["breach_bar"] >= 0


def test_label_buy_good_when_profit_first():
    from stock_ana.research.vegas_pullback.labels import label_buy_outcome
    n = 200
    close = np.full(n, 100.0)
    high = close.copy(); low = close.copy()
    # 锚点 100 之后第 5 天 high 触 +12%（止损从未触）
    high[105] = 113.0
    out = label_buy_outcome(close, high, low, anchor=100)
    assert out["label_buy"] == "good_buy"
    assert out["buy_outcome_bar"] == 5


def test_label_buy_bad_when_stop_first_and_same_day_conservative():
    from stock_ana.research.vegas_pullback.labels import label_buy_outcome
    n = 200
    close = np.full(n, 100.0)
    high = close.copy(); low = close.copy()
    low[103] = 91.0    # 第 3 天先触 -8% 止损
    high[110] = 115.0  # 之后才触盈利线
    out = label_buy_outcome(close, high, low, anchor=100)
    assert out["label_buy"] == "bad_buy"
    assert out["buy_outcome_bar"] == 3

    # 同日双触 → 保守判 bad
    high2 = close.copy(); low2 = close.copy()
    high2[104] = 113.0; low2[104] = 91.0
    out2 = label_buy_outcome(close, high2, low2, anchor=100)
    assert out2["label_buy"] == "bad_buy"


def test_label_buy_neutral_when_neither():
    from stock_ana.research.vegas_pullback.labels import label_buy_outcome
    n = 200
    close = np.full(n, 100.0)
    out = label_buy_outcome(close, close * 1.01, close * 0.99, anchor=100)
    assert out["label_buy"] == "neutral"
    assert out["buy_outcome_bar"] == -1


def test_label_horizon_columns_present():
    n = 350
    close = 10.0 * np.cumprod(np.full(n, 1.001))
    emas = _emas_from_close(close)
    out = label_pullback_outcome(close, close * 1.01, close * 0.99, emas, 300, float(close[300]))
    for h in (21, 42, 63):
        assert f"fwd_ret_{h}" in out
        assert f"fwd_maxdd_{h}" in out
        assert f"fwd_maxup_{h}" in out


# ─────────────────────────────────────────────────────────────
# signal_features
# ─────────────────────────────────────────────────────────────

def test_signal_features_complete_and_finite():
    from stock_ana.research.vegas_pullback.signal_features import compute_micro_features

    n = 400
    close = 10.0 * np.cumprod(np.full(n, 1.0015))
    high = close * 1.01
    low = close * 0.99
    open_ = close * 0.998
    volume = np.ones(n) * 1e6
    emas = _emas_from_close(close)
    wave_pb = {"seq": 2, "is_wave_end": True, "wave": None, "wave_rise_pct": 40.0, "consec_waves": 2}

    from stock_ana.research.vegas_pullback.signal_features import compute_cluster_features

    feats = compute_signal_features(close, high, low, volume, emas, 300, "long", wave_pb)
    feats.update(compute_micro_features(open_, high, low, close, volume, 300, 300,
                                        emas=emas, support="long"))
    feats.update(compute_cluster_features(high, low, close, volume, emas[144], 300))
    from stock_ana.research.vegas_pullback.signal_features import compute_fib_features
    feats.update(compute_fib_features(high, low, close, 300, None))

    # 所有声明的信号特征列都产出（signal + micro 两个函数合并覆盖）
    for col in SIGNAL_FEATURE_COLS:
        assert col in feats, f"missing feature {col}"
    # micro 特征合法域
    assert 0 <= feats["micro_touch_close_pos"] <= 1
    assert 0 <= feats["micro_shadow_ratio"] <= 1
    # 结构特征透传
    assert feats["structure_pullback_seq"] == 2
    assert feats["structure_is_wave_end"] == 1
    # 上升序列：EMA 多头排列、动量为正
    assert feats["chan_ema_order_ok"] == 1
    assert feats["mom_ret_60"] > 0


def test_signal_features_mid_vs_long_channel_differs():
    n = 400
    close = 10.0 * np.cumprod(np.full(n, 1.0015))
    high = close * 1.01
    low = close * 0.99
    volume = np.ones(n) * 1e6
    emas = _emas_from_close(close)
    wave_pb = {"seq": 1, "is_wave_end": True, "wave": None, "wave_rise_pct": 30.0, "consec_waves": 1}

    mid = compute_signal_features(close, high, low, volume, emas, 300, "mid", wave_pb)
    lng = compute_signal_features(close, high, low, volume, emas, 300, "long", wave_pb)

    # 触碰深度按不同通道计算 → 应不同（mid 通道更靠近价格）
    assert mid["chan_touch_depth_pct"] != lng["chan_touch_depth_pct"]


# ─────────────────────────────────────────────────────────────
# feature_registry
# ─────────────────────────────────────────────────────────────

def test_registry_covers_signal_features():
    for col in SIGNAL_FEATURE_COLS:
        assert col in REALTIME_FEATURE_COLS
        assert feature_group_for(col) != "other"


def test_registry_no_duplicate_columns():
    cols = list(REALTIME_FEATURE_COLS)
    assert len(cols) == len(set(cols)), "REALTIME_FEATURE_COLS 有重复列"


def test_registry_excludes_pooled_market_percentiles():
    # *_pct_mkt = 候选集跨期池化分位（一行的值取决于未来候选的分布）→ 必须剔除
    assert not any(c.endswith("_pct_mkt") for c in REALTIME_FEATURE_COLS)


# ─────────────────────────────────────────────────────────────
# 「回踩」语义门（pullback_precondition + mid>long regime）
# ─────────────────────────────────────────────────────────────

def test_mid_touch_requires_mid_above_long():
    """mid/long 缠绕（震荡市）时的 mid 触碰不算回踩。"""
    from stock_ana.strategies.impl.vegas_mid import detect_mid_touch_and_hold
    n = 500
    # 长期横盘 ±2% 震荡：所有 EMA 缠绕，mid 不在 long 之上
    rng = np.random.default_rng(3)
    close = 100.0 * (1 + 0.02 * np.sin(np.arange(n) / 7) + rng.normal(0, 0.004, n))
    low = close * 0.99
    emas = compute_vegas_emas(pd.Series(close))
    sigs = detect_mid_touch_and_hold(close, low, emas)

    from stock_ana.strategies.impl.vegas_mid import mid_above_long_mask
    regime = mid_above_long_mask(emas)
    for s in sigs:
        assert regime[s["touch_bar"]], "mid≤long 区间不应产生 mid 回踩信号"


def test_pullback_rejects_cross_up_touch():
    """跌穿后重新涨回的上穿触碰 ≠ 回踩（前一根收盘在均线之下）。"""
    from stock_ana.strategies.primitives.vegas_pullback import pullback_precondition
    n = 300
    close = np.full(n, 110.0)
    ema = np.full(n, 100.0)
    # 价格跌穿均线 8 根，然后涨回在 i=250 上穿触碰
    close[242:250] = 92.0
    close[250] = 101.0
    assert pullback_precondition(250, close, ema, above_lookback=10) is False

    # 对照：一直在上方、从 112 跌下来触碰 → 是回踩
    close2 = np.full(n, 112.0)
    close2[250] = 100.5
    assert pullback_precondition(250, close2, ema, above_lookback=10) is True


def test_pullback_rejects_crawling_along_line():
    """一直贴着均线横爬（窗口内最高收盘不足 EMA×1.02）≠ 从上方跌下来。"""
    from stock_ana.strategies.primitives.vegas_pullback import pullback_precondition
    n = 300
    close = np.full(n, 100.5)   # 贴线 +0.5%
    ema = np.full(n, 100.0)
    assert pullback_precondition(250, close, ema, above_lookback=10) is False


# ─────────────────────────────────────────────────────────────
# 因果性回归测试：特征绝不能依赖锚点之后的数据
# ─────────────────────────────────────────────────────────────

def _make_wavy_df(n: int = 700) -> pd.DataFrame:
    """带真实回踩节奏的合成序列：足够产生浪结构和 mid/long 触碰。"""
    rng = np.random.default_rng(7)
    steps = 1 + 0.0022 + rng.normal(0, 0.012, n)
    # 周期性回踩段压低漂移
    for k in range(260, n, 90):
        steps[k : k + 14] = 1 - 0.008 + rng.normal(0, 0.008, min(14, n - k))
    close = 10.0 * np.cumprod(steps)
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {
            "open": close * 0.998, "high": close * 1.012,
            "low": close * 0.988, "close": close,
            "volume": rng.uniform(5e5, 2e6, n),
        },
        index=dates,
    )


def test_features_identical_with_and_without_future_data():
    """同一候选：全量历史 vs 截断到锚点+1，特征值必须逐列一致（零前瞻铁证）。"""
    from stock_ana.research.vegas_pullback.build_vegas_pullback_research import (
        build_candidates_for_symbol,
    )

    df = _make_wavy_df()
    for support in ("mid", "long"):
        rows_full = build_candidates_for_symbol("US", "SYN", "syn", df, support)
        if not rows_full:
            continue
        # 取远离两端的候选：≥300 保证截断后仍满足 len≥260 的最短历史要求，
        # ≤n-150 保证全量侧标签/检测不受右边界影响
        mid_rows = [r for r in rows_full if 300 <= r["iloc"] <= len(df) - 150]
        for row in mid_rows[:3]:
            anchor = int(row["iloc"])
            df_trunc = df.iloc[: anchor + 1]
            rows_trunc = build_candidates_for_symbol("US", "SYN", "syn", df_trunc, support)
            match = [r for r in rows_trunc if r["iloc"] == anchor]
            assert match, f"{support}: 截断后锚点 {anchor} 的候选消失（检测依赖未来？）"
            r2 = match[0]
            for col in SIGNAL_FEATURE_COLS:
                a, b = row.get(col), r2.get(col)
                both_nan = pd.isna(a) and pd.isna(b)
                assert both_nan or a == b, (
                    f"{support}: 特征 {col} 依赖未来数据 (full={a} vs trunc={b}, anchor={anchor})"
                )
