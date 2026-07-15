"""Vegas 回踩模型：严格 OOS 训练 + 评估（按 support × market 分离）。

无泄漏保证（对齐 top_reversal.eval_watchlist_oos）:
  1) train/test 按 **symbol** 切分：test = watchlist 成员，train = 非成员 →
     同一只股候选不跨集，杜绝「静态个股特征记忆身份」。
  2) fit 只用 train，score 只用 test。
  3) 标准化/填补统计量只在 train 估计。
  4) 两 support（mid/long）× 三市场（CN/HK/US）各自独立训练，绝不合并。
  5) 特征用 REALTIME_FEATURE_COLS（全因果）。

默认目标 = label_buy（三重栅栏买点标签），正类 good_buy，分数高 = 越该抄底。
分数低 ≠ 该跑路——它只表示「不值得买」，单边决策不输出离场建议。

用法:
    python -m stock_ana.research.vegas_pullback.eval_oos --support both
    python -m stock_ana.research.vegas_pullback.eval_oos --target structure  # 结构标签参考口径
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stock_ana.config import OUTPUT_DIR  # noqa: E402
from stock_ana.research.vegas_pullback.feature_registry import (  # noqa: E402
    REALTIME_FEATURE_COLS,
    feature_group_for,
)
from stock_ana.research.top_reversal.eval_watchlist_oos import (  # noqa: E402
    auc,
    topk,
    usable_features,
    load_watchlist_members,
)

OUT_DIR = OUTPUT_DIR / "vegas_pullback_research"
MARKETS = ["US", "HK", "CN"]

# 训练目标（--target 切换；默认 buy —— 回答「该不该抄底」）
#   buy      : label_buy ∈ {good_buy, bad_buy}，正类 good_buy（三重栅栏收益标签）
#   structure: label ∈ {bounce, breakdown}，正类 bounce（通道结构标签，仅参考）
# 实测结构标签训练出的分数与前瞻收益几乎无关（只与回撤相关），买点标签
# 训练后 top20% 调用 ret63 11.5%→13.9%、ret21 4.5%→6.0%。
TARGETS = {
    "buy": {"col": "label_buy", "pos": "good_buy", "neg": "bad_buy"},
    # 大二浪猎手：63日内先 +20%（vs 先 -10%）。为「第1浪回踩吃整个第二浪」
    # 设计——早期(seq≤1) long 回踩上 top20% 调用的 maxup63/ret63 显著优于
    # 标准 buy 标签（CN 34.7→47.6% / 14.2→26.4%），小米 2024-08 类早期
    # 回踩可过线。华虹型 seq=0 深度混沌回踩仍不可分（信息集极限）。
    "buy_big": {"col": "label_buy_big", "pos": "good_buy", "neg": "bad_buy"},
    # W2-chain：本次回踩终结的浪是否交接出连续下一浪（W1→W2 结构标签）。
    # 合训 OOS AUC 0.774、top20% 链率 0.37(基础 0.17)；第一特征 fund_rev_accel
    # （营收增速加速度，反向——已披露增速低迷者更易接 W2，市场提前定价拐点）。
    # 注意正样本仅 372，分市场训练偏薄。
    # micro（触碰日单日形态）对 W2 结构问题无关且实测稀释召回(0.38→0.34)，排除
    "w2_chain": {"col": "label_w2", "pos": "chained", "neg": "isolated",
                 "exclude_groups": ("micro",)},
    "structure": {"col": "label", "pos": "bounce", "neg": "breakdown"},
}
TARGET_COL = "label_buy"
POS_LABEL = "good_buy"
EXCLUDE_GROUPS: tuple = ()


# ── LR / lgbm：fit=train / score=test，正类可配置 ────────────────────────────

def _fit_logistic(train: pd.DataFrame, feats: list[str]) -> dict:
    xdf = train[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = xdf.median()
    xdf = xdf.fillna(med).fillna(0.0)
    mean = xdf.mean()
    std = xdf.std(ddof=0).replace(0, 1)
    x = np.clip(((xdf - mean) / std).to_numpy(float), -8.0, 8.0)
    y = (train[TARGET_COL] == POS_LABEL).astype(float).to_numpy()
    xb = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(xb.shape[1])
    lr, l2 = 0.04, 0.05
    for _ in range(2500):
        p = 1 / (1 + np.exp(-np.clip(xb @ beta, -30, 30)))
        grad = (xb.T @ (p - y)) / len(y)
        grad[1:] += l2 * beta[1:] / len(y)
        beta -= lr * grad
    return {"feats": list(feats), "med": med, "mean": mean, "std": std, "beta": beta}


def _predict_logistic(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    feats, med, mean, std, beta = (
        bundle["feats"], bundle["med"], bundle["mean"], bundle["std"], bundle["beta"]
    )
    tx = df.reindex(columns=feats).apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    tx = np.clip(((tx - mean) / std).to_numpy(float), -8.0, 8.0)
    txb = np.column_stack([np.ones(len(tx)), tx])
    return 1 / (1 + np.exp(-np.clip(txb @ beta, -30, 30)))


def _fit_lightgbm(train: pd.DataFrame, feats: list[str]):
    import lightgbm as lgb
    x_tr = train[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y_tr = (train[TARGET_COL] == POS_LABEL).astype(float).to_numpy()
    # 不用 scale_pos_weight：bounce/breakdown 类别本就接近平衡，加权只会压偏概率
    # 校准（持仓回测实测平均 P 0.37→0.43、AUC 0.649→0.661），不改善排序。
    params = dict(objective="binary", verbose=-1, num_threads=4, max_depth=3, num_leaves=7,
                  min_data_in_leaf=40, learning_rate=0.03, feature_fraction=0.6,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, lambda_l1=1.0)
    boosters, gains = [], []
    for seed in range(5):
        m = lgb.train(dict(params, seed=seed), lgb.Dataset(x_tr, label=y_tr), num_boost_round=300)
        boosters.append(m)
        gains.append(m.feature_importance(importance_type="gain"))
    return boosters, np.mean(gains, axis=0)


def _predict_lightgbm(boosters, feats: list[str], df: pd.DataFrame) -> np.ndarray:
    x = df.reindex(columns=feats).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return np.mean([b.predict(x) for b in boosters], axis=0)


def _eval_market(g: pd.DataFrame, wl_syms: set[str], feats_all: list[str]) -> dict | None:
    """对单市场做 symbol-split OOS，返回指标行。"""
    g = g.copy()
    g["sym"] = g["sym"].astype(str)
    test = g[g["sym"].isin(wl_syms)]
    train = g[~g["sym"].isin(wl_syms)]
    if len(train) < 60 or len(test) < 20 or train[TARGET_COL].nunique() < 2 or test[TARGET_COL].nunique() < 2:
        return {"n_train": len(train), "n_test": len(test), "note": "样本不足/单类"}

    feats = usable_features(train, feats_all)
    if len(feats) < 3:
        return {"n_train": len(train), "n_test": len(test), "note": "可用特征不足"}

    y_test = (test[TARGET_COL] == POS_LABEL).astype(float).to_numpy()
    base = float(y_test.mean())

    lr_p = _predict_logistic(_fit_logistic(train, feats), test)
    row = {
        "n_train": len(train), "n_test": len(test),
        "base_pos_rate": round(base * 100, 1),
        "lr_auc": round(auc(y_test, lr_p), 3),
        "lr_prec_top30": round(topk(y_test, lr_p, 0.30)[1], 1),
    }
    try:
        boosters, _ = _fit_lightgbm(train, feats)
        lgb_p = _predict_lightgbm(boosters, feats, test)
        row["lgb_auc"] = round(auc(y_test, lgb_p), 3)
        row["lgb_prec_top30"] = round(topk(y_test, lgb_p, 0.30)[1], 1)
        # buy-side 收益指标：top20% 调用的实际前瞻收益/回撤（模型价值的最终口径）
        te = test.copy()
        te["_p"] = lgb_p
        top = te[te["_p"] >= te["_p"].quantile(0.8)]
        for col, name in [("fwd_ret_21", "top20_ret21"), ("fwd_ret_63", "top20_ret63"),
                          ("fwd_maxdd_21", "top20_dd21")]:
            if col in top.columns:
                row[name] = round(float(pd.to_numeric(top[col], errors="coerce").mean()), 1)
    except Exception as e:
        row["lgb_auc"] = np.nan
        row["note"] = f"lgb skip: {type(e).__name__}"
    return row


def _importance_report(binary: pd.DataFrame, feats_all: list[str]) -> pd.DataFrame:
    """全量（in-sample）lgbm gain 重要性，按特征组归类——仅供解读，不用于 OOS。"""
    feats = usable_features(binary, feats_all)
    try:
        _, gain = _fit_lightgbm(binary, feats)
    except Exception:
        return pd.DataFrame()
    imp = pd.DataFrame({"feature": feats, "gain": np.round(gain, 1)})
    imp["group"] = imp["feature"].map(feature_group_for)
    return imp.sort_values("gain", ascending=False)


def evaluate_support(support: str) -> None:
    path = OUT_DIR / f"{support}_labeled.csv"
    if not path.exists():
        print(f"[{support}] 缺少数据集：{path}（先跑 build）")
        return
    lab = pd.read_csv(path, low_memory=False)
    neg = [v["neg"] for v in TARGETS.values() if v["col"] == TARGET_COL][0]
    lab = lab[lab[TARGET_COL].isin([POS_LABEL, neg])].copy()
    wl = load_watchlist_members()

    print(f"\n{'='*70}\n[{support}] 目标={TARGET_COL} 总候选(二分类) {len(lab)}，"
          f"{POS_LABEL}={int((lab[TARGET_COL]==POS_LABEL).sum())} "
          f"{neg}={int((lab[TARGET_COL]==neg).sum())}")

    feats_all = [c for c in REALTIME_FEATURE_COLS
                 if feature_group_for(c) not in set(EXCLUDE_GROUPS)]
    results = []
    for mk in MARKETS:
        g = lab[lab["market"].astype(str) == mk]
        if g.empty:
            continue
        r = _eval_market(g, wl.get(mk, set()), feats_all)
        if r:
            r = {"market": mk, **r}
            results.append(r)
    res_df = pd.DataFrame(results)
    print(f"\n[{support}] 严格 OOS（test=watchlist，train=池内非 watchlist）：")
    print(res_df.to_string(index=False))

    imp = _importance_report(lab, feats_all)
    if not imp.empty:
        print(f"\n[{support}] 特征重要性 Top15（全量 lgbm gain，仅解读）：")
        print(imp.head(15).to_string(index=False))
        grp = imp.groupby("group")["gain"].sum().sort_values(ascending=False)
        print(f"\n[{support}] 特征组贡献（gain 求和）：")
        print(grp.to_string())

    # 落盘
    out_csv = OUT_DIR / f"{support}_oos_metrics.csv"
    res_df.to_csv(out_csv, index=False)
    if not imp.empty:
        imp.to_csv(OUT_DIR / f"{support}_feature_importance.csv", index=False)
    print(f"\n[{support}] 指标写出 → {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vegas 回踩模型 OOS 评估")
    parser.add_argument("--support", choices=["mid", "long", "both"], default="both")
    parser.add_argument("--target", choices=["buy", "buy_big", "w2_chain", "structure"], default="buy",
                        help="训练目标：buy=三重栅栏买点标签(默认)，structure=通道结构标签")
    args = parser.parse_args()
    global TARGET_COL, POS_LABEL, EXCLUDE_GROUPS
    TARGET_COL = TARGETS[args.target]["col"]
    POS_LABEL = TARGETS[args.target]["pos"]
    EXCLUDE_GROUPS = TARGETS[args.target].get("exclude_groups", ())
    supports = ["mid", "long"] if args.support == "both" else [args.support]
    for support in supports:
        evaluate_support(support)


if __name__ == "__main__":
    main()
