"""Modeling and report helpers for top-reversal research."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ana.research.top_reversal.feature_registry import apply_legacy_feature_aliases, feature_group_for


def to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.to_numpy().tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell.replace("|", "/") for cell in row) + " |")
    return "\n".join(lines) + "\n"


def label_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summary = (
        df.groupby(["label"], observed=True)
        .agg(
            n=("label", "size"),
            avg_anchor_rise=("rise_from_anchor_low_pct", "mean"),
            avg_future_drawdown=("future_drawdown_pct", "mean"),
            avg_future_high=("future_high_pct", "mean"),
            avg_score=("score_max", "mean"),
        )
        .reset_index()
    )
    total = len(df)
    summary["pct"] = (summary["n"] / total * 100).round(1)
    for col in ["avg_anchor_rise", "avg_future_drawdown", "avg_future_high", "avg_score"]:
        summary[col] = summary[col].round(2)
    return summary.sort_values("n", ascending=False)


def feature_diff(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = apply_legacy_feature_aliases(df)
    binary = df[df["label"].isin(["true_top", "continuation"])].copy()
    rows = []
    for col in feature_cols:
        if col not in binary.columns:
            continue
        a = pd.to_numeric(binary[binary["label"] == "true_top"][col], errors="coerce")
        b = pd.to_numeric(binary[binary["label"] == "continuation"][col], errors="coerce")
        if a.notna().sum() < 3 or b.notna().sum() < 3:
            continue
        rows.append({
            "feature_group": feature_group_for(col),
            "feature": col,
            "true_median": round(float(a.median()), 3),
            "cont_median": round(float(b.median()), 3),
            "delta": round(float(a.median() - b.median()), 3),
            "true_mean": round(float(a.mean()), 3),
            "cont_mean": round(float(b.mean()), 3),
        })
    return pd.DataFrame(rows).sort_values("delta", key=lambda s: s.abs(), ascending=False)


def bucket_stats(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = apply_legacy_feature_aliases(df)
    rows = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        valid = df[values.notna()].copy()
        if len(valid) < 20 or valid[col].nunique() < 4:
            continue
        try:
            valid["_bucket"] = pd.qcut(pd.to_numeric(valid[col]), q=5, duplicates="drop")
        except ValueError:
            continue
        for bucket, group in valid.groupby("_bucket", observed=True):
            decisive = group[group["label"].isin(["true_top", "continuation"])]
            rows.append({
                "feature_group": feature_group_for(col),
                "feature": col,
                "bucket": str(bucket),
                "n": len(group),
                "true_top": int((group["label"] == "true_top").sum()),
                "continuation": int((group["label"] == "continuation").sum()),
                "ambiguous": int((group["label"] == "ambiguous").sum()),
                "true_rate_all": round((group["label"] == "true_top").mean() * 100, 1),
                "true_rate_decisive": round((decisive["label"] == "true_top").mean() * 100, 1) if len(decisive) else np.nan,
            })
    return pd.DataFrame(rows)


def _build_scored_frame(df: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    """Build the standard scored output frame (shared by LR and lightgbm)."""
    scored_cols = [
        "market", "sym", "name", "signal_date", "confirm_date", "top_date",
        "strategies", "label",
    ]
    optional_cols = [
        "candidate_source", "top_pos", "score_asof_pos", "score_asof_date",
        "recalled_by_candle", "recalled_by_shadow", "recalled_by_doji", "recalled_by_gap_fail",
        "recalled_by_smc_confirmed", "recalled_by_smc_early",
        "candle_top_pattern", "candle_old_top_pattern",
        "smc_early_with_shadow", "smc_early_with_doji", "smc_early_with_gap_fail",
        "smc_early_with_any_candle", "smc_early_with_old_candle",
        "smc_early_candle_score_max",
        "recall_source_count", "china_hk_focus",
        "mid_vegas_passed", "mid_vegas_live_passed",
        "mid_vegas_top_days_above_long", "mid_vegas_top_days_above_mid",
        "mid_vegas_top_mid_long_gap_pct", "mid_vegas_top_close_dist_mid_pct",
        "is_semiconductor",
    ]
    cols = scored_cols + [c for c in optional_cols if c in df.columns]
    scored = df[cols].copy()
    scored["top_prob"] = np.round(prob, 4)
    return scored.sort_values("top_prob", ascending=False)


def _usable_features(binary: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    return [
        c for c in feature_cols
        if c in binary.columns and pd.to_numeric(binary[c], errors="coerce").notna().sum() >= max(20, len(binary) * 0.5)
    ]


def fit_lightgbm(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train a regularized gradient-boosted tree (lightgbm) alongside LR.

    Returns (importance_df, scored_df) mirroring ``fit_logistic``.  Trees handle
    collinearity and feature×regime interactions (e.g. semiconductor × structure)
    that a linear model cannot.  Empty frames if lightgbm is unavailable or data
    is insufficient — the build then simply skips the lightgbm outputs.
    """
    df = apply_legacy_feature_aliases(df)
    binary = df[df["label"].isin(["true_top", "continuation"])].copy()
    if binary["label"].nunique() < 2 or len(binary) < 30:
        return pd.DataFrame(), pd.DataFrame()
    try:
        import lightgbm as lgb
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    usable = _usable_features(binary, feature_cols)
    if len(usable) < 3:
        return pd.DataFrame(), pd.DataFrame()

    x_tr = binary[usable].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_tr = (binary["label"] == "true_top").astype(float).to_numpy()
    pos = max(1.0, float((y_tr == 1).sum()))
    params = dict(
        objective="binary", verbose=-1, num_threads=4,
        max_depth=3, num_leaves=7, min_data_in_leaf=40, learning_rate=0.03,
        feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1,
        lambda_l2=5.0, lambda_l1=1.0, scale_pos_weight=float((y_tr == 0).sum() / pos),
    )
    all_x = df[usable].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    probs, gains = [], []
    for seed in range(5):  # 多 seed 平均降噪
        p = dict(params, seed=seed)
        model = lgb.train(p, lgb.Dataset(x_tr, label=y_tr), num_boost_round=300)
        probs.append(model.predict(all_x))
        gains.append(model.feature_importance(importance_type="gain"))
    prob = np.mean(probs, axis=0)
    scored = _build_scored_frame(df, prob)

    importance = pd.DataFrame({"feature": usable, "gain": np.round(np.mean(gains, axis=0), 2)})
    importance["feature_group"] = importance["feature"].map(lambda x: feature_group_for(str(x)))
    importance = importance.sort_values("gain", ascending=False)
    return importance[["feature_group", "feature", "gain"]], scored


def fit_logistic(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = apply_legacy_feature_aliases(df)
    binary = df[df["label"].isin(["true_top", "continuation"])].copy()
    if binary["label"].nunique() < 2 or len(binary) < 30:
        return pd.DataFrame(), pd.DataFrame()

    usable = []
    for col in feature_cols:
        if col in binary.columns and pd.to_numeric(binary[col], errors="coerce").notna().sum() >= max(20, len(binary) * 0.5):
            usable.append(col)
    if len(usable) < 3:
        return pd.DataFrame(), pd.DataFrame()

    xdf = binary[usable].apply(pd.to_numeric, errors="coerce")
    med = xdf.median()
    xdf = xdf.fillna(med)
    mean = xdf.mean()
    std = xdf.std(ddof=0).replace(0, 1)
    x = ((xdf - mean) / std).to_numpy(dtype=float)
    y = (binary["label"] == "true_top").astype(float).to_numpy()

    xb = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(xb.shape[1])
    lr = 0.04
    l2 = 0.05
    for _ in range(2500):
        z = np.clip(xb @ beta, -30, 30)
        p = 1 / (1 + np.exp(-z))
        grad = (xb.T @ (p - y)) / len(y)
        grad[1:] += l2 * beta[1:] / len(y)
        beta -= lr * grad

    all_x = df[usable].apply(pd.to_numeric, errors="coerce").fillna(med)
    all_x = ((all_x - mean) / std).to_numpy(dtype=float)
    all_xb = np.column_stack([np.ones(len(all_x)), all_x])
    prob = 1 / (1 + np.exp(-np.clip(all_xb @ beta, -30, 30)))
    scored = _build_scored_frame(df, prob)

    coef = pd.DataFrame({
        "feature": ["intercept", *usable],
        "coef": np.round(beta, 4),
    })
    coef["feature_group"] = coef["feature"].map(lambda x: "model" if x == "intercept" else feature_group_for(str(x)))
    coef["direction"] = np.where(coef["coef"] > 0, "true_top+", np.where(coef["coef"] < 0, "continuation+", "flat"))
    cols = ["feature_group", "feature", "coef", "direction"]
    return coef[cols].sort_values("coef", key=lambda s: s.abs(), ascending=False), scored


def score_performance(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    rows = []
    for quantile in (0.90, 0.80, 0.70, 0.60, 0.50):
        threshold = float(scored["top_prob"].quantile(quantile))
        group = scored[scored["top_prob"] >= threshold]
        rows.append({
            "score_band": f"top_{int((1 - quantile) * 100)}pct",
            "threshold": round(threshold, 4),
            "n": len(group),
            "true_top": int((group["label"] == "true_top").sum()),
            "continuation": int((group["label"] == "continuation").sum()),
            "precision": round((group["label"] == "true_top").mean() * 100, 1) if len(group) else np.nan,
        })
    return pd.DataFrame(rows)
