"""Unified public facade for standardized strategy screen/scan/explain entrypoints."""

from __future__ import annotations

from typing import Callable
from typing import Literal

import pandas as pd

from stock_ana.data.indicators import add_vegas_channel
from stock_ana.data.market_data import load_market_data, load_universe_data
from stock_ana.strategies.impl import (
    ma_squeeze,
    main_rally_pullback,
    momentum_detector,
    rs,
    triangle,
    triangle_kde,
    triangle_vcp,
    vcp,
    vegas_mid,
)
from stock_ana.strategies.primitives.wave import analyze_wave_structure
from stock_ana.strategies.contracts import ScanHit, ScanResult, StrategyDecision, StrategyKind


PATTERN_KIND: StrategyKind = "pattern"
STATEFUL_SIGNAL_KIND: StrategyKind = "stateful_signal"


def _meta_with_kind(strategy_kind: StrategyKind, **meta) -> dict:
    """Attach a normalized strategy-kind marker to supplemental metadata."""
    merged = {"strategy_kind": strategy_kind}
    merged.update(meta)
    return merged


def _resolve_stock_data(
    market: str,
    stock_data: dict[str, pd.DataFrame] | None,
) -> tuple[dict[str, pd.DataFrame], str]:
    """Resolve scan stock data from explicit input or market loader."""
    if stock_data is not None:
        return stock_data, market

    loaded = load_market_data(market)
    return (loaded or {}), market


def _resolve_universe_data(
    universe: str,
    stock_data: dict[str, pd.DataFrame] | None,
) -> tuple[dict[str, pd.DataFrame], str]:
    """Resolve combined-universe scan data from explicit input or local cache facade."""
    if stock_data is not None:
        return stock_data, universe

    loaded = load_universe_data(universe=universe)
    return (loaded or {}), universe


def _scan_data_map(
    data_map: dict[str, pd.DataFrame],
    decision_builder: Callable[[str, pd.DataFrame], StrategyDecision],
    *,
    min_history: int = 0,
    eligibility: Callable[[str, pd.DataFrame], bool] | None = None,
    hit_selector: Callable[[StrategyDecision], bool] | None = None,
) -> tuple[list[ScanHit], int, int, int]:
    """Generic scan loop over {symbol: dataframe} with unified counters."""
    hits: list[ScanHit] = []
    processed = 0
    skipped = 0
    failed = 0
    pick_hit = hit_selector or (lambda d: d.passed)

    for symbol, df in data_map.items():
        if df is None or len(df) < min_history:
            skipped += 1
            continue
        if eligibility is not None and not eligibility(symbol, df):
            skipped += 1
            continue
        try:
            processed += 1
            decision = decision_builder(symbol, df)
            if pick_hit(decision):
                hits.append(ScanHit(symbol=symbol, decision=decision))
        except Exception:
            failed += 1

    return hits, processed, skipped, failed


# =============================================================================
# Vegas Channel Touch
# =============================================================================

def _screen_vegas_touch_raw(
    df: pd.DataFrame,
    lookback_days: int = 5,
    half_year_days: int = 120,
) -> dict | None:
    """Run the raw Vegas touch heuristic and return feature details when matched."""
    required = {"close", "low", "high", "ema_144", "ema_169"}
    if not required.issubset(df.columns) or len(df) < half_year_days:
        return None

    trend_window = min(90, len(df) - 1)
    ema_mid_now = (df["ema_144"].iloc[-1] + df["ema_169"].iloc[-1]) / 2
    ema_mid_ago = (df["ema_144"].iloc[-trend_window] + df["ema_169"].iloc[-trend_window]) / 2
    if ema_mid_now <= ema_mid_ago:
        return None

    lookback_90 = min(90, len(df))
    section_90 = df.iloc[-lookback_90:]
    cross_below_count = 0
    was_above = True
    for _, row in section_90.iterrows():
        ema_upper = max(row["ema_144"], row["ema_169"])
        is_above = row["close"] > ema_upper
        if was_above and not is_above:
            cross_below_count += 1
        was_above = is_above

    if cross_below_count > 1:
        return None

    touch_events = []
    in_touch = False
    for i in range(len(section_90)):
        row = section_90.iloc[i]
        ema_upper = max(row["ema_144"], row["ema_169"])
        if row["low"] <= ema_upper:
            if not in_touch:
                touch_events.append(i)
                in_touch = True
        elif in_touch:
            still_away = True
            for j in range(1, 4):
                if i + j < len(section_90):
                    r2 = section_90.iloc[i + j]
                    eu2 = max(r2["ema_144"], r2["ema_169"])
                    if r2["low"] <= eu2:
                        still_away = False
                        break
            if still_away:
                in_touch = False

    if len(touch_events) >= 3:
        return None

    half_year = df.iloc[-half_year_days:]
    peak_idx = half_year["high"].idxmax()
    peak_price = half_year.loc[peak_idx, "high"]

    recent_start = df.index[-lookback_days]
    if peak_idx >= recent_start:
        return None

    curr_close = df["close"].iloc[-1]
    if curr_close >= peak_price * 0.95:
        return None

    peak_iloc = df.index.get_loc(peak_idx)
    touch_start = len(df) - lookback_days
    between_section = df.iloc[peak_iloc:touch_start]
    if len(between_section) < 3:
        return None

    above_count = 0
    for _, row in between_section.iterrows():
        ema_upper = max(row["ema_144"], row["ema_169"])
        if row["close"] > ema_upper:
            above_count += 1

    above_ratio = above_count / len(between_section) if len(between_section) > 0 else 0
    if above_ratio < 0.70:
        return None

    recent = df.iloc[-lookback_days:]
    touched = False
    touch_date = None
    for idx_label, row in recent.iterrows():
        ema_upper = max(row["ema_144"], row["ema_169"])
        if row["low"] <= ema_upper:
            touched = True
            touch_date = idx_label
            break

    if not touched:
        return None

    for i in range(len(df) - lookback_days, len(df)):
        row = df.iloc[i]
        ema_lower = min(row["ema_144"], row["ema_169"])
        close_i = row["close"]
        if close_i < ema_lower:
            if i + 1 < len(df):
                next_row = df.iloc[i + 1]
                next_ema_lower = min(next_row["ema_144"], next_row["ema_169"])
                if next_row["close"] < next_ema_lower:
                    return None
            else:
                return None

    return {
        "peak_price": float(peak_price),
        "peak_date": str(peak_idx),
        "current_price": float(curr_close),
        "above_ratio": round(above_ratio, 2),
        "touch_date": str(touch_date) if touch_date else None,
        "channel_trend_pct": round((ema_mid_now / ema_mid_ago - 1) * 100, 2),
        "cross_below_count": cross_below_count,
        "touch_events_90d": len(touch_events),
    }


def screen_vegas_touch(
    df: pd.DataFrame,
    lookback_days: int = 5,
    half_year_days: int = 120,
) -> StrategyDecision:
    """Standardized Vegas channel touchback single-symbol decision."""
    if df is None or df.empty:
        return StrategyDecision(
            passed=False,
            strategy_kind=PATTERN_KIND,
            setup_type="vegas_channel_touch",
            reason="empty_data",
        )

    if {"ema_144", "ema_169"}.issubset(df.columns):
        df_input = df.copy()
    else:
        df_input = add_vegas_channel(df.copy())

    raw = _screen_vegas_touch_raw(
        df_input,
        lookback_days=lookback_days,
        half_year_days=half_year_days,
    )
    if raw is None:
        return StrategyDecision(
            passed=False,
            strategy_kind=PATTERN_KIND,
            setup_type="vegas_channel_touch",
            reason="not_triggered",
        )

    score = 70.0 + min(20.0, float(raw.get("above_ratio", 0.0)) * 20.0)
    score += min(10.0, max(0.0, float(raw.get("channel_trend_pct", 0.0)) * 0.5))
    return StrategyDecision(
        passed=True,
        strategy_kind=PATTERN_KIND,
        score=max(0.0, min(100.0, score)),
        setup_type="vegas_channel_touch",
        trigger_date=df_input.index[-1],
        features=raw,
    )


def scan_vegas_touches(
    market: str = "ndx100",
    lookback_days: int = 5,
    half_year_days: int = 120,
    stock_data: dict[str, pd.DataFrame] | None = None,
) -> ScanResult:
    """Standardized Vegas channel touchback scan result."""
    data_map, resolved_market = _resolve_stock_data(market, stock_data)
    if not data_map:
        return ScanResult(
            strategy="vegas",
            market=resolved_market,
            strategy_kind=PATTERN_KIND,
        )

    hits, processed, skipped, failed = _scan_data_map(
        data_map,
        lambda _symbol, df: screen_vegas_touch(
            df,
            lookback_days=lookback_days,
            half_year_days=half_year_days,
        ),
        min_history=170,
    )

    return ScanResult(
        strategy="vegas",
        market=resolved_market,
        strategy_kind=PATTERN_KIND,
        hits=hits,
        total=len(data_map),
        processed=processed,
        skipped=skipped,
        failed=failed,
        params_snapshot={
            "lookback_days": lookback_days,
            "half_year_days": half_year_days,
        },
    )


def explain_vegas_touch(decision: StrategyDecision) -> str:
    """Human-readable explanation for Vegas touchback decision."""
    if not decision.passed:
        return "Vegas 通道回踩未触发。"

    peak = decision.features.get("peak_price")
    touch_date = decision.features.get("touch_date")
    ratio = float(decision.features.get("above_ratio", 0.0))
    return f"Vegas 通道回踩触发：前高 {peak:.2f}，触碰日期 {touch_date}，高点至回踩期间通道上方占比 {ratio:.2f}。"


# =============================================================================
# MA Squeeze
# =============================================================================

def screen_ma_squeeze(df: pd.DataFrame, stage: int = 0) -> StrategyDecision:
    """Standardized single-symbol MA squeeze screening decision."""
    if df is None or df.empty:
        return StrategyDecision(
            passed=False,
            strategy_kind=STATEFUL_SIGNAL_KIND,
            setup_type="ma_squeeze",
            reason="empty_data",
            warnings=["input dataframe is empty"],
        )

    s1 = ma_squeeze.detect_stage1(df) if stage in (0, 1) else {"triggered": False, "details": {}}
    s2 = ma_squeeze.detect_stage2(df) if stage in (0, 2) else {"triggered": False, "details": {}, "score": 0}

    passed = bool(s1.get("triggered", False) or s2.get("triggered", False))
    setup_type = "stage2_confirmed" if s2.get("triggered", False) else "stage1_setup"
    score = float(s2.get("score", 0) * 20) if s2.get("triggered", False) else (55.0 if s1.get("triggered", False) else 0.0)

    return StrategyDecision(
        passed=passed,
        strategy_kind=STATEFUL_SIGNAL_KIND,
        score=score,
        setup_type=setup_type if passed else "ma_squeeze_none",
        trigger_date=df.index[-1] if passed else None,
        reason=None if passed else (s2.get("reason") or s1.get("reason") or "not_triggered"),
        features={
            "stage1_triggered": bool(s1.get("triggered", False)),
            "stage2_triggered": bool(s2.get("triggered", False)),
            "stage2_score": int(s2.get("score", 0)),
            "stage1": s1.get("details", {}),
            "stage2": s2.get("details", {}),
        },
    )


def scan_ma_squeeze(
    stage: int = 0,
    stock_data: dict[str, pd.DataFrame] | None = None,
    market: str = "us+ndx100",
) -> ScanResult:
    """Standardized batch MA squeeze scan result."""
    data_map, resolved_market = _resolve_universe_data(market, stock_data)
    if not data_map:
        return ScanResult(
            strategy="ma_squeeze",
            market=resolved_market,
            strategy_kind=STATEFUL_SIGNAL_KIND,
            params_snapshot={"stage": stage},
        )

    hits, processed, skipped, failed = _scan_data_map(
        data_map,
        lambda _symbol, df: screen_ma_squeeze(df, stage=stage),
        min_history=1,
    )

    return ScanResult(
        strategy="ma_squeeze",
        market=resolved_market,
        strategy_kind=STATEFUL_SIGNAL_KIND,
        hits=hits,
        total=len(data_map),
        processed=processed,
        skipped=skipped,
        failed=failed,
        params_snapshot={"stage": stage},
    )


def explain_ma_squeeze(decision: StrategyDecision) -> str:
    """Human-readable explanation for MA squeeze decision."""
    if not decision.passed:
        return f"MA Squeeze 未触发，原因: {decision.reason or '条件不足'}"

    s1 = decision.features.get("stage1_triggered")
    s2 = decision.features.get("stage2_triggered")
    s2_score = decision.features.get("stage2_score", 0)
    if s2:
        return f"MA Squeeze 第二阶段已触发，确认信号得分 {s2_score}/5，适合进入观察或择时。"
    if s1:
        return "MA Squeeze 第一阶段触发（压缩形态成立），尚待第二阶段量价确认。"
    return "MA Squeeze 命中。"


# =============================================================================
# RS Acceleration
# =============================================================================

def screen_rs_acceleration(
    df_stock: pd.DataFrame,
    df_market: pd.DataFrame,
    rs_rank: float,
    variant: str = "rs_strict",
) -> StrategyDecision:
    """Standardized RS acceleration single-symbol decision."""
    raw = rs.screen_relative_strength(df_stock, df_market, rs_rank=rs_rank, variant=variant)
    if raw is None:
        return StrategyDecision(
            passed=False,
            strategy_kind=STATEFUL_SIGNAL_KIND,
            setup_type="rs_acceleration",
            reason="not_triggered",
            meta=_meta_with_kind(STATEFUL_SIGNAL_KIND, variant=variant),
        )

    accel = float(raw.get("acceleration", 0.0))
    score = max(0.0, min(100.0, 60.0 + accel * 5.0))
    return StrategyDecision(
        passed=True,
        strategy_kind=STATEFUL_SIGNAL_KIND,
        score=score,
        setup_type="rs_acceleration",
        trigger_date=df_stock.index[-1],
        features=raw,
        meta=_meta_with_kind(STATEFUL_SIGNAL_KIND, variant=variant),
    )


def scan_rs_acceleration(
    variant: str = "rs_strict",
    market: str = "ndx100",
    stock_data: dict[str, pd.DataFrame] | None = None,
    df_market: pd.DataFrame | None = None,
) -> ScanResult:
    """Standardized RS acceleration scan over NDX100 universe."""
    data_map, resolved_market = _resolve_stock_data(market, stock_data)
    if not data_map:
        return ScanResult(
            strategy="rs_acceleration",
            market=resolved_market,
            strategy_kind=STATEFUL_SIGNAL_KIND,
            params_snapshot={"variant": variant},
        )

    market_df = df_market or rs._load_qqq()
    if market_df is None:
        return ScanResult(
            strategy="rs_acceleration",
            market=resolved_market,
            strategy_kind=STATEFUL_SIGNAL_KIND,
            total=len(data_map),
            skipped=len(data_map),
            params_snapshot={"variant": variant, "reason": "missing_market_data"},
        )

    rs_ranks = rs.compute_rs_rank_63d(data_map, market_df)
    hits, processed, skipped, failed = _scan_data_map(
        data_map,
        lambda symbol, df: screen_rs_acceleration(
            df,
            market_df,
            rs_rank=rs_ranks[symbol],
            variant=variant,
        ),
        min_history=252,
        eligibility=lambda symbol, _df: symbol in rs_ranks,
    )

    return ScanResult(
        strategy="rs_acceleration",
        market=resolved_market,
        strategy_kind=STATEFUL_SIGNAL_KIND,
        hits=hits,
        total=len(data_map),
        processed=processed,
        skipped=skipped,
        failed=failed,
        params_snapshot={"variant": variant},
    )


def explain_rs_acceleration(decision: StrategyDecision) -> str:
    """Human-readable explanation for RS acceleration decision."""
    if not decision.passed:
        return "RS Acceleration 未触发，当前相对强度加速不足或结构不满足。"

    rank = decision.features.get("rs_rank")
    chg21 = decision.features.get("rs_chg_21d")
    accel = decision.features.get("acceleration")
    return f"RS 加速触发：RS排名 {rank}%，21日变化 {chg21:+.2f}%，加速度 {accel:+.2f}%。"


# =============================================================================
# VCP
# =============================================================================

def screen_vcp_setup(
    df: pd.DataFrame,
    min_base_days: int = 60,
    max_base_days: int = 300,
    loose: bool = False,
) -> StrategyDecision:
    """Standardized VCP single-symbol decision."""
    raw = vcp.screen_vcp(
        df,
        min_base_days=min_base_days,
        max_base_days=max_base_days,
        loose=loose,
    )
    if raw is None:
        return StrategyDecision(
            passed=False,
            strategy_kind=PATTERN_KIND,
            setup_type="vcp",
            reason="not_triggered",
            meta=_meta_with_kind(PATTERN_KIND, loose=loose),
        )

    distance = float(raw.get("distance_to_pivot_pct", 100.0))
    score = max(0.0, min(100.0, 90.0 - abs(distance) * 2.0))
    stop_hint = raw.get("p2_val")

    return StrategyDecision(
        passed=True,
        strategy_kind=PATTERN_KIND,
        score=score,
        setup_type="cup_with_handle",
        trigger_date=df.index[-1],
        stop_hint=float(stop_hint) if stop_hint is not None else None,
        invalidation="price_close_below_handle_or_cup_low",
        features=raw,
        meta=_meta_with_kind(PATTERN_KIND, loose=loose),
    )


def scan_vcp_setups(
    universe: Literal["ndx100", "us"] = "ndx100",
    min_base_days: int = 60,
    max_base_days: int = 300,
    loose: bool = False,
    stock_data: dict[str, pd.DataFrame] | None = None,
) -> ScanResult:
    """Standardized VCP batch scan for NDX100 or US universe."""
    data_map, resolved_market = _resolve_stock_data(universe, stock_data)
    if not data_map:
        return ScanResult(
            strategy="vcp",
            market=resolved_market,
            strategy_kind=PATTERN_KIND,
            params_snapshot={
                "min_base_days": min_base_days,
                "max_base_days": max_base_days,
                "loose": loose,
            },
        )

    hits, processed, skipped, failed = _scan_data_map(
        data_map,
        lambda _symbol, df: screen_vcp_setup(
            df,
            min_base_days=min_base_days,
            max_base_days=max_base_days,
            loose=loose,
        ),
        min_history=max(260, min_base_days + 10),
    )

    return ScanResult(
        strategy="vcp",
        market=resolved_market,
        strategy_kind=PATTERN_KIND,
        hits=hits,
        total=len(data_map),
        processed=processed,
        skipped=skipped,
        failed=failed,
        params_snapshot={
            "min_base_days": min_base_days,
            "max_base_days": max_base_days,
            "loose": loose,
        },
    )


def explain_vcp_setup(decision: StrategyDecision) -> str:
    """Human-readable explanation for VCP decision."""
    if not decision.passed:
        return "VCP 未触发，杯柄形态质量或趋势过滤条件不足。"

    base_days = decision.features.get("base_days")
    depth = float(decision.features.get("base_depth_pct", 0.0))
    dist = float(decision.features.get("distance_to_pivot_pct", 0.0))
    return f"VCP 触发：基底 {base_days} 天，深度 {depth:.1f}%，距枢轴 {dist:.2f}%。"


# =============================================================================
# Triangle Family
# =============================================================================

def _screen_triangle_common(raw: dict | None, setup_type: str, df: pd.DataFrame) -> StrategyDecision:
    """Convert raw triangle-family detector output into a normalized decision."""
    if raw is None:
        return StrategyDecision(
            passed=False,
            strategy_kind=PATTERN_KIND,
            setup_type=setup_type,
            reason="not_triggered",
        )

    score = float(raw.get("score", 0.0))
    norm_score = max(0.0, min(100.0, score))
    return StrategyDecision(
        passed=True,
        strategy_kind=PATTERN_KIND,
        score=norm_score,
        setup_type=setup_type,
        trigger_date=df.index[-1],
        features=raw,
    )


def screen_triangle_ascending(df: pd.DataFrame) -> StrategyDecision:
    """Evaluate one symbol for an ascending-triangle setup."""
    return _screen_triangle_common(
        triangle.screen_ascending_triangle(df),
        setup_type="ascending_triangle",
        df=df,
    )


def scan_triangle_ascending(
    market: str = "ndx100",
    stock_data: dict[str, pd.DataFrame] | None = None,
) -> ScanResult:
    """Scan a market universe for ascending-triangle candidates."""
    data_map, resolved_market = _resolve_stock_data(market, stock_data)
    hits, processed, skipped, failed = _scan_data_map(
        data_map,
        lambda _symbol, df: screen_triangle_ascending(df),
        min_history=60,
    )

    return ScanResult(
        strategy="triangle_ascending",
        market=resolved_market,
        strategy_kind=PATTERN_KIND,
        hits=hits,
        total=len(data_map),
        processed=processed,
        skipped=skipped,
        failed=failed,
    )


def explain_triangle_ascending(decision: StrategyDecision) -> str:
    """Render a concise explanation for an ascending-triangle decision."""
    if not decision.passed:
        return "上升三角形未触发。"
    conv = float(decision.features.get("convergence_ratio", 0.0))
    period = decision.features.get("period")
    return f"上升三角形触发：周期 {period} 天，收敛度 {conv:.2f}。"


def screen_triangle_parallel_channel(df: pd.DataFrame) -> StrategyDecision:
    """Evaluate one symbol for a parallel-channel setup."""
    return _screen_triangle_common(
        triangle.screen_parallel_channel(df),
        setup_type="parallel_channel",
        df=df,
    )


def scan_triangle_parallel_channel(
    market: str = "ndx100",
    stock_data: dict[str, pd.DataFrame] | None = None,
) -> ScanResult:
    """Scan a market universe for parallel-channel candidates."""
    data_map, resolved_market = _resolve_stock_data(market, stock_data)
    hits, processed, skipped, failed = _scan_data_map(
        data_map,
        lambda _symbol, df: screen_triangle_parallel_channel(df),
        min_history=60,
    )

    return ScanResult(
        strategy="triangle_parallel_channel",
        market=resolved_market,
        strategy_kind=PATTERN_KIND,
        hits=hits,
        total=len(data_map),
        processed=processed,
        skipped=skipped,
        failed=failed,
    )


def explain_triangle_parallel_channel(decision: StrategyDecision) -> str:
    """Render a concise explanation for a parallel-channel decision."""
    if not decision.passed:
        return "上升平行通道未触发。"
    period = decision.features.get("period")
    spread = float(decision.features.get("spread_contraction", 0.0))
    return f"上升平行通道触发：周期 {period} 天，波幅收缩比 {spread:.2f}。"


def screen_triangle_rising_wedge(df: pd.DataFrame) -> StrategyDecision:
    """Evaluate one symbol for a rising-wedge setup."""
    return _screen_triangle_common(
        triangle.screen_rising_wedge(df),
        setup_type="rising_wedge",
        df=df,
    )


def scan_triangle_rising_wedge(
    market: str = "ndx100",
    stock_data: dict[str, pd.DataFrame] | None = None,
) -> ScanResult:
    """Scan a market universe for rising-wedge candidates."""
    data_map, resolved_market = _resolve_stock_data(market, stock_data)
    hits, processed, skipped, failed = _scan_data_map(
        data_map,
        lambda _symbol, df: screen_triangle_rising_wedge(df),
        min_history=60,
    )

    return ScanResult(
        strategy="triangle_rising_wedge",
        market=resolved_market,
        strategy_kind=PATTERN_KIND,
        hits=hits,
        total=len(data_map),
        processed=processed,
        skipped=skipped,
        failed=failed,
    )


def explain_triangle_rising_wedge(decision: StrategyDecision) -> str:
    """Render a concise explanation for a rising-wedge decision."""
    if not decision.passed:
        return "上升楔形未触发。"
    conv = float(decision.features.get("convergence_ratio", 0.0))
    return f"上升楔形触发：收敛度 {conv:.2f}，属于偏风险扩大的末端结构。"


def screen_triangle_kde_setup(df: pd.DataFrame) -> StrategyDecision:
    """Evaluate one symbol for a KDE-based ascending-triangle setup."""
    return _screen_triangle_common(
        triangle_kde.screen_ascending_triangle_kde(df),
        setup_type="ascending_triangle_kde",
        df=df,
    )


def scan_triangle_kde_setups(
    market: str = "ndx100",
    stock_data: dict[str, pd.DataFrame] | None = None,
) -> ScanResult:
    """Scan a market universe for KDE-based ascending-triangle setups."""
    data_map, resolved_market = _resolve_stock_data(market, stock_data)
    hits, processed, skipped, failed = _scan_data_map(
        data_map,
        lambda _symbol, df: screen_triangle_kde_setup(df),
        min_history=60,
    )

    return ScanResult(
        strategy="triangle_kde",
        market=resolved_market,
        strategy_kind=PATTERN_KIND,
        hits=hits,
        total=len(data_map),
        processed=processed,
        skipped=skipped,
        failed=failed,
    )


def explain_triangle_kde_setup(decision: StrategyDecision) -> str:
    """Render a concise explanation for a KDE triangle decision."""
    if not decision.passed:
        return "KDE 上升三角形未触发。"
    sharp = float(decision.features.get("kde_sharpness", 0.0))
    return f"KDE 上升三角形触发：阻力峰尖锐度 {sharp:.2f}。"


def screen_triangle_vcp_setup(
    df: pd.DataFrame,
    peak_iloc: int | None = None,
    min_period: int = 25,
    max_period: int = 250,
    require_trend: bool = True,
) -> StrategyDecision:
    """Evaluate one symbol for a VCP-style triangle consolidation."""
    raw = triangle_vcp.screen_triangle_vcp(
        df,
        peak_iloc=peak_iloc,
        min_period=min_period,
        max_period=max_period,
        require_trend=require_trend,
    )
    return _screen_triangle_common(raw, setup_type="triangle_vcp", df=df)


def scan_triangle_vcp_setups(
    market: str = "ndx100",
    min_period: int = 25,
    max_period: int = 250,
    require_trend: bool = True,
    stock_data: dict[str, pd.DataFrame] | None = None,
) -> ScanResult:
    """Scan a market universe for VCP-style triangle candidates."""
    data_map, resolved_market = _resolve_stock_data(market, stock_data)
    if not data_map:
        return ScanResult(
            strategy="triangle_vcp",
            market=resolved_market,
            strategy_kind=PATTERN_KIND,
        )

    hits, processed, skipped, failed = _scan_data_map(
        data_map,
        lambda _symbol, df: screen_triangle_vcp_setup(
            df,
            min_period=min_period,
            max_period=max_period,
            require_trend=require_trend,
        ),
        min_history=max(260, min_period + 30),
    )

    return ScanResult(
        strategy="triangle_vcp",
        market=resolved_market,
        strategy_kind=PATTERN_KIND,
        hits=hits,
        total=len(data_map),
        processed=processed,
        skipped=skipped,
        failed=failed,
        params_snapshot={
            "min_period": min_period,
            "max_period": max_period,
            "require_trend": require_trend,
        },
    )


def explain_triangle_vcp_setup(decision: StrategyDecision) -> str:
    """Render a concise explanation for a triangle-VCP decision."""
    if not decision.passed:
        return "VCP 三角形未触发。"
    pattern = decision.features.get("pattern")
    conv = float(decision.features.get("convergence_ratio", 0.0))
    return f"VCP 三角形触发：形态 {pattern}，收敛度 {conv:.2f}。"


# =============================================================================
# Main Rally Pullback
# =============================================================================

def screen_main_rally_pullback(
    df: pd.DataFrame,
    trend_days: int = 63,
    high_lookback: int = 126,
    prior_above_days: int = 55,
    retrace_days: int = 5,
) -> StrategyDecision:
    """Evaluate one symbol for a main-rally pullback setup."""
    raw = main_rally_pullback.screen_main_rally_pullback(
        df,
        trend_days=trend_days,
        high_lookback=high_lookback,
        prior_above_days=prior_above_days,
        retrace_days=retrace_days,
    )
    if raw is None:
        return StrategyDecision(
            passed=False,
            strategy_kind=PATTERN_KIND,
            setup_type="main_rally_pullback",
            reason="not_triggered",
        )

    score = 75.0
    pullback_pct = float(raw.get("pullback_pct_from_high", 0.0))
    rise_pct = float(raw.get("rise_from_1y_low_pct", 0.0))
    score += min(15.0, max(0.0, (rise_pct - 30.0) * 0.2))
    score -= min(10.0, abs(pullback_pct - 12.0))
    return StrategyDecision(
        passed=True,
        strategy_kind=PATTERN_KIND,
        score=max(0.0, min(100.0, score)),
        setup_type=str(raw.get("pattern", "main_rally_pullback")),
        trigger_date=df.index[-1],
        features=raw,
    )


def scan_main_rally_pullback_setups(
    market: str = "ndx100",
    trend_days: int = 63,
    high_lookback: int = 126,
    prior_above_days: int = 55,
    retrace_days: int = 5,
    stock_data: dict[str, pd.DataFrame] | None = None,
) -> ScanResult:
    """Scan a market universe for main-rally pullback setups."""
    data_map, resolved_market = _resolve_stock_data(market, stock_data)
    if not data_map:
        return ScanResult(
            strategy="main_rally_pullback",
            market=resolved_market,
            strategy_kind=PATTERN_KIND,
        )

    hits, processed, skipped, failed = _scan_data_map(
        data_map,
        lambda _symbol, df: screen_main_rally_pullback(
            df,
            trend_days=trend_days,
            high_lookback=high_lookback,
            prior_above_days=prior_above_days,
            retrace_days=retrace_days,
        ),
        min_history=90,
    )

    return ScanResult(
        strategy="main_rally_pullback",
        market=resolved_market,
        strategy_kind=PATTERN_KIND,
        hits=hits,
        total=len(data_map),
        processed=processed,
        skipped=skipped,
        failed=failed,
        params_snapshot={
            "trend_days": trend_days,
            "high_lookback": high_lookback,
            "prior_above_days": prior_above_days,
            "retrace_days": retrace_days,
        },
    )


def explain_main_rally_pullback(decision: StrategyDecision) -> str:
    """Render a concise explanation for a main-rally pullback decision."""
    if not decision.passed:
        return "主升浪回踩信号未触发。"
    support = decision.features.get("support_type")
    pullback = float(decision.features.get("pullback_pct_from_high", 0.0))
    return f"主升浪回踩触发：支撑类型 {support}，较前高回撤 {pullback:.2f}%。"


# =============================================================================
# Momentum Detector
# =============================================================================

def screen_momentum(df: pd.DataFrame, lookback: int = 5) -> StrategyDecision:
    """Evaluate one symbol for a momentum anomaly signal."""
    raw = momentum_detector.detect_momentum(df, lookback=lookback)
    if not raw.get("triggered", False):
        return StrategyDecision(
            passed=False,
            strategy_kind=STATEFUL_SIGNAL_KIND,
            score=float(raw.get("score", 0.0) * 10),
            setup_type="momentum",
            reason=str(raw.get("reason", "not_triggered")),
            features=raw.get("signals", {}),
        )

    return StrategyDecision(
        passed=True,
        strategy_kind=STATEFUL_SIGNAL_KIND,
        score=float(raw.get("score", 0.0) * 10),
        setup_type="momentum",
        trigger_date=df.index[-1],
        features=raw.get("signals", {}),
    )


def scan_momentum(
    lookback: int = 5,
    min_score: float = 3.0,
    update: bool = False,
    stock_data: dict[str, pd.DataFrame] | None = None,
    market: str = "us+ndx100",
) -> ScanResult:
    """Scan a combined universe for momentum anomaly signals."""
    data_map, resolved_market = _resolve_universe_data(market, stock_data)
    if not data_map:
        return ScanResult(
            strategy="momentum",
            market=resolved_market,
            strategy_kind=STATEFUL_SIGNAL_KIND,
            params_snapshot={
                "lookback": lookback,
                "min_score": min_score,
                "update": update,
            },
        )

    hits, processed, skipped, failed = _scan_data_map(
        data_map,
        lambda _symbol, df: screen_momentum(df, lookback=lookback),
        min_history=60,
        hit_selector=lambda decision: decision.passed and decision.score >= min_score * 10,
    )

    return ScanResult(
        strategy="momentum",
        market=resolved_market,
        strategy_kind=STATEFUL_SIGNAL_KIND,
        hits=hits,
        total=len(data_map),
        processed=processed,
        skipped=skipped,
        failed=failed,
        params_snapshot={
            "lookback": lookback,
            "min_score": min_score,
            "update": update,
        },
    )


def explain_momentum(decision: StrategyDecision) -> str:
    """Render a concise explanation for a momentum decision."""
    if not decision.passed:
        return "动量异动未触发。"
    z = decision.features.get("abnormal_return", {}).get("z_score")
    br = decision.features.get("breakout", {}).get("level")
    return f"动量异动触发：收益 Z-score={z}，突破类型={br or 'none'}。"


# =============================================================================
# RS Trap
# =============================================================================

def screen_rs_trap_alert(
    df_stock: pd.DataFrame,
    df_market: pd.DataFrame,
    rs_rank: float,
    variant: str = "rs_trap_strict",
) -> StrategyDecision:
    """Evaluate one symbol for an RS-trap warning signal."""
    raw = rs.screen_rs_trap(df_stock, df_market, rs_rank=rs_rank, variant=variant)
    if raw is None:
        return StrategyDecision(
            passed=False,
            strategy_kind=STATEFUL_SIGNAL_KIND,
            setup_type="rs_trap",
            reason="not_triggered",
            meta=_meta_with_kind(STATEFUL_SIGNAL_KIND, variant=variant),
        )

    outperf = float(raw.get("outperform", 0.0))
    weak = abs(float(raw.get("rs_chg_63d", 0.0)))
    score = max(0.0, min(100.0, 60.0 + outperf * 5.0 + weak * 1.5))
    return StrategyDecision(
        passed=True,
        strategy_kind=STATEFUL_SIGNAL_KIND,
        score=score,
        setup_type="rs_trap",
        trigger_date=df_stock.index[-1],
        features=raw,
        meta=_meta_with_kind(STATEFUL_SIGNAL_KIND, variant=variant),
    )


def scan_rs_trap_alert(
    variant: str = "rs_trap_strict",
    market: str = "ndx100",
    stock_data: dict[str, pd.DataFrame] | None = None,
    df_market: pd.DataFrame | None = None,
) -> ScanResult:
    """Scan a market universe for RS-trap warning candidates."""
    data_map, resolved_market = _resolve_stock_data(market, stock_data)
    if not data_map:
        return ScanResult(
            strategy="rs_trap",
            market=resolved_market,
            strategy_kind=STATEFUL_SIGNAL_KIND,
            params_snapshot={"variant": variant},
        )

    market_df = df_market or rs._load_qqq()
    if market_df is None:
        return ScanResult(
            strategy="rs_trap",
            market=resolved_market,
            strategy_kind=STATEFUL_SIGNAL_KIND,
            total=len(data_map),
            skipped=len(data_map),
            params_snapshot={"variant": variant, "reason": "missing_market_data"},
        )

    rs_ranks = rs.compute_rs_rank_63d(data_map, market_df)
    hits, processed, skipped, failed = _scan_data_map(
        data_map,
        lambda symbol, df: screen_rs_trap_alert(
            df,
            market_df,
            rs_rank=rs_ranks[symbol],
            variant=variant,
        ),
        min_history=252,
        eligibility=lambda symbol, _df: symbol in rs_ranks,
    )

    return ScanResult(
        strategy="rs_trap",
        market=resolved_market,
        strategy_kind=STATEFUL_SIGNAL_KIND,
        hits=hits,
        total=len(data_map),
        processed=processed,
        skipped=skipped,
        failed=failed,
        params_snapshot={"variant": variant},
    )


def explain_rs_trap_alert(decision: StrategyDecision) -> str:
    """Render a concise explanation for an RS-trap alert decision."""
    if not decision.passed:
        return "RS 陷阱预警未触发。"
    outperf = float(decision.features.get("outperform", 0.0))
    rank = float(decision.features.get("rs_rank", 0.0))
    rs63 = float(decision.features.get("rs_chg_63d", 0.0))
    return f"RS 陷阱预警：短期跑赢市场 {outperf:+.2f}pct，但RS排名 {rank}% 且RS63d {rs63:+.2f}%。"


# =============================================================================
# Vegas Mid Pullback  (EMA34/55/60 touchback within major upwave)
# =============================================================================

def screen_vegas_mid_pullback(
    df: pd.DataFrame,
    lookback: int = 1,
    market: str = "US",
    name: str = "",
) -> StrategyDecision:
    """Evaluate the most recent Mid Vegas (EMA34/55/60) pullback signal for one symbol.

    Args:
        df: OHLCV DataFrame (date index, lowercase columns).
        lookback: How many recent trading days to consider (1 = today only).
        market: "US" or "HK" — affects scoring.
        name: Human-readable display name (informational only).

    Returns:
        StrategyDecision with ``passed=True`` if at least one qualifying signal
        was found within the lookback window.  ``features["signals"]`` holds the
        full list of raw signal dicts from the detection engine.
    """
    import numpy as np

    if df is None or df.empty or len(df) < 200:
        return StrategyDecision(
            passed=False,
            strategy_kind=STATEFUL_SIGNAL_KIND,
            setup_type="vegas_mid_pullback",
            reason="insufficient_data",
        )

    x = df.copy()
    x.columns = [str(c).lower() for c in x.columns]
    x.index = pd.to_datetime(x.index)
    x = x.sort_index()

    close = x["close"].astype(float).values
    low_arr = x["low"].astype(float).values
    close_s = x["close"].astype(float)
    n = len(x)

    emas = vegas_mid.compute_vegas_emas(close_s)
    result = analyze_wave_structure(df)
    waves = result.get("major_waves", [])

    touch_signals = vegas_mid.detect_mid_touch_and_hold(close, low_arr, emas)

    wave_touch_counter: dict[int, int] = {}
    cutoff_bar = n - lookback
    recent_signals: list[dict] = []

    for sig in touch_signals:
        entry_bar = sig["entry_bar"]

        wave_ctx = vegas_mid.find_wave_context(waves, sig["touch_bar"]) if waves else None
        wave_number = wave_ctx["wave_number"] if wave_ctx else 0
        wave_touch_counter[wave_number] = wave_touch_counter.get(wave_number, 0) + 1
        touch_seq = wave_touch_counter[wave_number]

        if entry_bar < cutoff_bar:
            continue

        if entry_bar >= n:
            confirm_bar = sig["confirm_bar"]
            entry_price = float(close[confirm_bar])
            struct = vegas_mid.check_mid_vegas_structure(confirm_bar, close, emas)
        else:
            entry_price = float(close[entry_bar])
            struct = vegas_mid.check_mid_vegas_structure(entry_bar, close, emas)

        wave_rise_so_far = 0.0
        consec_count = 0
        sub_number = 0

        if wave_ctx:
            sub_count = sum(
                1 for sw in wave_ctx.get("sub_waves", [])
                if sw.get("end_pivot") and sw["end_pivot"]["iloc"] <= sig["touch_bar"]
            )
            sub_number = sub_count + 1
            sp_val = wave_ctx["start_pivot"]["value"]
            if sp_val > 0:
                look_start = wave_ctx["start_pivot"]["iloc"]
                look_end = min(sig["touch_bar"] + 1, n)
                if look_start < look_end:
                    peak_val = float(np.max(close[look_start:look_end]))
                    wave_rise_so_far = (peak_val / sp_val - 1) * 100
            consec_count = vegas_mid.backward_consecutive_count(waves, wave_number)

        score, score_details = vegas_mid.score_pullback(
            sub_number=sub_number,
            wave_rise_pct=wave_rise_so_far,
            wave_number=wave_number,
            market=market,
            consecutive_wave_count=consec_count,
            mid_long_gap_pct=struct["mid_long_gap_pct"],
        )
        signal_label = vegas_mid.classify_signal(score) if struct["passed"] else "AVOID"

        recent_signals.append({
            "entry_bar": entry_bar,
            "entry_price": round(entry_price, 3),
            "support_band": sig["support_band"],
            "signal": signal_label,
            "score": score,
            "structure_passed": struct["passed"],
            "wave_number": wave_number,
            "sub_number": sub_number,
            "consec_waves": consec_count,
            "wave_rise_pct": round(wave_rise_so_far, 2),
            "touch_seq": touch_seq,
            "mid_long_gap_pct": struct["mid_long_gap_pct"],
            **{f"factor_{k}": v for k, v in score_details.items()},
        })

    if not recent_signals:
        return StrategyDecision(
            passed=False,
            strategy_kind=STATEFUL_SIGNAL_KIND,
            setup_type="vegas_mid_pullback",
            reason="not_triggered",
        )

    best = max(recent_signals, key=lambda s: s["score"])
    norm_score = max(0.0, min(100.0, 50.0 + best["score"] * 10.0))

    return StrategyDecision(
        passed=True,
        strategy_kind=STATEFUL_SIGNAL_KIND,
        score=norm_score,
        setup_type="vegas_mid_pullback",
        trigger_date=x.index[-1],
        features={
            "signals": recent_signals,
            "best_signal": best["signal"],
            "best_score": best["score"],
            "support_band": best["support_band"],
        },
    )


def scan_vegas_mid_pullbacks(
    market: str = "us",
    lookback: int = 1,
    stock_data: dict[str, pd.DataFrame] | None = None,
) -> ScanResult:
    """Scan a market universe for recent Mid Vegas pullback signals.

    Args:
        market: "us", "ndx100", or "hk" — used both for data loading and scoring.
        lookback: Recent trading days window passed to detection engine.
        stock_data: Optional pre-loaded {symbol: DataFrame} map.
    """
    data_map, resolved_market = _resolve_stock_data(market, stock_data)
    if not data_map:
        return ScanResult(
            strategy="vegas_mid",
            market=resolved_market,
            strategy_kind=STATEFUL_SIGNAL_KIND,
            params_snapshot={"lookback": lookback},
        )

    mkt_tag = "HK" if resolved_market == "hk" else "US"

    hits, processed, skipped, failed = _scan_data_map(
        data_map,
        lambda _symbol, df: screen_vegas_mid_pullback(
            df,
            lookback=lookback,
            market=mkt_tag,
        ),
        min_history=200,
    )

    return ScanResult(
        strategy="vegas_mid",
        market=resolved_market,
        strategy_kind=STATEFUL_SIGNAL_KIND,
        hits=hits,
        total=len(data_map),
        processed=processed,
        skipped=skipped,
        failed=failed,
        params_snapshot={"lookback": lookback},
    )


def explain_vegas_mid_pullback(decision: StrategyDecision) -> str:
    """Render a concise explanation for a Mid Vegas pullback decision."""
    if not decision.passed:
        return "Mid Vegas 回踩未触发。"
    best = decision.features.get("best_signal", "—")
    score = decision.features.get("best_score", 0)
    band = decision.features.get("support_band", "—")
    n = len(decision.features.get("signals", []))
    return (
        f"Mid Vegas 回踩触发：{n} 个近期信号，最优 {best}（score={score:+d}），"
        f"支撑线 {band}。"
    )

