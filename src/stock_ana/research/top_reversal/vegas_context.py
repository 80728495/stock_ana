"""Mid Vegas trend-context features for top-reversal research."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from stock_ana.strategies.impl.vegas_mid import (
    LONG_EMAS,
    MID_EMAS,
    check_mid_vegas_structure,
    compute_vegas_emas,
)


VEGAS_TREND_FEATURES: tuple[str, ...] = (
    "mid_vegas_passed",
    "mid_vegas_live_passed",
    "mid_vegas_base_passed",
    "mid_vegas_live_base_passed",
    "mid_vegas_top_history_ready",
    "mid_vegas_top_mid_above_long",
    "mid_vegas_top_mid_lower_above_long",
    "mid_vegas_top_price_above_long",
    "mid_vegas_top_price_above_mid",
    "mid_vegas_top_price_above_long_30d",
    "mid_vegas_top_long_rising",
    "mid_vegas_top_gap_enough",
    "mid_vegas_top_long_slope_strong",
    "mid_vegas_top_days_above_long",
    "mid_vegas_top_days_above_mid",
    "mid_vegas_top_mid_long_gap_pct",
    "mid_vegas_top_long_slope_pct",
    "mid_vegas_top_close_dist_mid_pct",
    "mid_vegas_top_close_dist_long_pct",
    "mid_vegas_top_high_dist_mid_pct",
    "mid_vegas_asof_history_ready",
    "mid_vegas_asof_mid_above_long",
    "mid_vegas_asof_mid_lower_above_long",
    "mid_vegas_asof_price_above_long",
    "mid_vegas_asof_price_above_mid",
    "mid_vegas_asof_price_above_long_30d",
    "mid_vegas_asof_long_rising",
    "mid_vegas_asof_gap_enough",
    "mid_vegas_asof_long_slope_strong",
    "mid_vegas_asof_days_above_long",
    "mid_vegas_asof_days_above_mid",
    "mid_vegas_asof_mid_long_gap_pct",
    "mid_vegas_asof_long_slope_pct",
    "mid_vegas_asof_close_dist_mid_pct",
    "mid_vegas_asof_close_dist_long_pct",
)

_FLAG_DEFAULTS = {
    "history_ready": 0,
    "base_passed": 0,
    "strict_passed": 0,
    "mid_above_long": 0,
    "mid_lower_above_long": 0,
    "price_above_long": 0,
    "price_above_mid": 0,
    "price_above_long_30d": 0,
    "long_rising": 0,
    "gap_enough": 0,
    "long_slope_strong": 0,
}

_NUM_DEFAULTS = {
    "days_above_long": 0,
    "days_above_mid": 0,
    "mid_long_gap_pct": np.nan,
    "long_slope_pct": np.nan,
    "close_dist_mid_pct": np.nan,
    "close_dist_long_pct": np.nan,
    "high_dist_mid_pct": np.nan,
}


def _as_pos(value: object, default: int = -1) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _symbol_data_map(symbol_data: Mapping[str, dict] | None) -> dict[tuple[str, str], pd.DataFrame]:
    if not symbol_data:
        return {}
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for item in symbol_data.values():
        market = str(item.get("market", ""))
        symbol = str(item.get("symbol", item.get("sym", "")))
        df = item.get("df")
        if market and symbol and isinstance(df, pd.DataFrame):
            out[(market, symbol)] = df
    return out


def _consecutive_true(mask: np.ndarray, pos: int) -> int:
    count = 0
    for i in range(pos, -1, -1):
        if not bool(mask[i]):
            break
        count += 1
    return count


def _empty_prefixed(prefix: str) -> dict[str, object]:
    values: dict[str, object] = {}
    for key, value in _FLAG_DEFAULTS.items():
        values[f"{prefix}_{key}"] = value
    for key, value in _NUM_DEFAULTS.items():
        values[f"{prefix}_{key}"] = value
    return values


def _bar_context(df: pd.DataFrame, pos: int, emas: dict[int, np.ndarray], *, prefix: str) -> dict[str, object]:
    values = _empty_prefixed(prefix)
    if pos < 0 or pos >= len(df):
        return values

    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    if close[pos] <= 0:
        return values

    struct = check_mid_vegas_structure(pos, close, emas)
    mid_vals = np.array([emas[span][pos] for span in MID_EMAS], dtype=float)
    long_vals = np.array([emas[span][pos] for span in LONG_EMAS], dtype=float)
    if not np.isfinite(mid_vals).all() or not np.isfinite(long_vals).all():
        return values

    mid_upper = float(np.max(mid_vals))
    mid_lower = float(np.min(mid_vals))
    long_upper = float(np.max(long_vals))
    # 短历史豁免：长 EMA 用 ewm(adjust=False) 恒有值，pos>=max(LONG_EMAS) 这一硬性
    # bar 数门槛会把总长不足 200 根的标的（如 02788 仅 129 根）整只剔除，即便它在
    # 自身可用尺度上完全满足 Vegas 多头结构。对总长不足 max(LONG_EMAS) 的短历史标的，
    # 改为只要 mid EMA 成熟（pos>=max(MID_EMAS)）即视为 ready；真正的多头结构由
    # check_mid_vegas_structure 把关，不放松。长历史标的维持原 pos>=max(LONG_EMAS)。
    long_max = max(LONG_EMAS)
    history_ready = pos >= (long_max if len(df) > long_max else max(MID_EMAS))
    base_passed = bool(struct.get("passed", False)) and history_ready

    mid_lower_above_long = mid_lower > long_upper
    price_above_mid = close[pos] > mid_upper
    strict_passed = bool(base_passed and mid_lower_above_long and price_above_mid)

    long_upper_arr = np.maximum.reduce([emas[span] for span in LONG_EMAS])
    mid_upper_arr = np.maximum.reduce([emas[span] for span in MID_EMAS])
    days_above_long = _consecutive_true(close > long_upper_arr, pos)
    days_above_mid = _consecutive_true(close > mid_upper_arr, pos)

    values.update({
        f"{prefix}_history_ready": int(history_ready),
        f"{prefix}_base_passed": int(base_passed),
        f"{prefix}_strict_passed": int(strict_passed),
        f"{prefix}_mid_above_long": int(bool(struct.get("mid_above_long", False))),
        f"{prefix}_mid_lower_above_long": int(mid_lower_above_long),
        f"{prefix}_price_above_long": int(bool(struct.get("price_above_long", False))),
        f"{prefix}_price_above_mid": int(price_above_mid),
        f"{prefix}_price_above_long_30d": int(bool(struct.get("price_above_long_3m", False))),
        f"{prefix}_long_rising": int(bool(struct.get("long_rising", False))),
        f"{prefix}_gap_enough": int(bool(struct.get("gap_enough", False))),
        f"{prefix}_long_slope_strong": int(bool(struct.get("long_slope_strong", False))),
        f"{prefix}_days_above_long": int(days_above_long),
        f"{prefix}_days_above_mid": int(days_above_mid),
        f"{prefix}_mid_long_gap_pct": round(float((mid_lower / long_upper - 1) * 100), 2) if long_upper > 0 else np.nan,
        f"{prefix}_long_slope_pct": float(struct.get("long_slope_pct", np.nan)),
        f"{prefix}_close_dist_mid_pct": round(float((close[pos] / mid_upper - 1) * 100), 2) if mid_upper > 0 else np.nan,
        f"{prefix}_close_dist_long_pct": round(float((close[pos] / long_upper - 1) * 100), 2) if long_upper > 0 else np.nan,
        f"{prefix}_high_dist_mid_pct": round(float((high[pos] / mid_upper - 1) * 100), 2) if mid_upper > 0 else np.nan,
    })
    return values


def mid_vegas_features_for_candidate(df: pd.DataFrame, row: Mapping[str, object]) -> dict[str, object]:
    """Return causal Mid Vegas trend features for one candidate row.

    ``mid_vegas_passed`` is the top/origin-bar gate. ``mid_vegas_live_passed``
    repeats the same structural test at the candidate score-as-of bar.
    """

    defaults = {col: 0 if col.endswith(("passed", "ready")) else np.nan for col in VEGAS_TREND_FEATURES}
    if df.empty or not {"high", "close"}.issubset(df.columns):
        return defaults

    top_pos = _as_pos(row.get("top_pos", row.get("signal_pos")), -1)
    asof_pos = _as_pos(row.get("score_asof_pos", row.get("confirm_pos", top_pos)), top_pos)
    top_pos = max(0, min(len(df) - 1, top_pos)) if top_pos >= 0 else -1
    asof_pos = max(0, min(len(df) - 1, asof_pos)) if asof_pos >= 0 else top_pos

    emas = compute_vegas_emas(df["close"].astype(float))
    top = _bar_context(df, top_pos, emas, prefix="mid_vegas_top")
    asof = _bar_context(df, asof_pos, emas, prefix="mid_vegas_asof")

    out = {**defaults, **top, **asof}
    out["mid_vegas_base_passed"] = int(out.get("mid_vegas_top_base_passed", 0))
    out["mid_vegas_live_base_passed"] = int(out.get("mid_vegas_asof_base_passed", 0))
    out["mid_vegas_passed"] = int(out.get("mid_vegas_top_strict_passed", 0))
    out["mid_vegas_live_passed"] = int(out.get("mid_vegas_asof_strict_passed", 0))
    return {col: out.get(col, defaults.get(col, np.nan)) for col in VEGAS_TREND_FEATURES}


def add_mid_vegas_features(
    dataset: pd.DataFrame,
    symbol_data: Mapping[str, dict] | None = None,
) -> pd.DataFrame:
    """Attach Mid Vegas context features to candidate rows."""

    out = dataset.copy()
    for col in VEGAS_TREND_FEATURES:
        if col not in out.columns:
            out[col] = 0 if col.endswith(("passed", "ready")) else np.nan

    data_map = _symbol_data_map(symbol_data)
    if out.empty or not data_map:
        return out

    feature_rows: list[dict[str, object]] = []
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for _, row in out.iterrows():
        key = (str(row.get("market", "")), str(row.get("sym", "")))
        df = cache.get(key)
        if df is None:
            df = data_map.get(key)
            if df is not None:
                df = df.copy()
                df.columns = [str(c).lower() for c in df.columns]
                df.index = pd.to_datetime(df.index)
                cache[key] = df
        if df is None:
            feature_rows.append({col: out.at[row.name, col] for col in VEGAS_TREND_FEATURES})
            continue
        feature_rows.append(mid_vegas_features_for_candidate(df, row))

    feature_df = pd.DataFrame(feature_rows, index=out.index)
    for col in VEGAS_TREND_FEATURES:
        out[col] = feature_df[col]
    return out
