"""严格 OOS 评估：test=watchlist，train=全集−test，按市场分离，对比 LR vs lightgbm。

无作弊保证（逐条）：
  1) 训练/测试按 **symbol** 切分（test=watchlist 成员，train=非成员）→ 同一只股的候选要么全在
     train、要么全在 test，绝不跨集 → 杜绝「静态个股特征记忆身份」的泄漏。
  2) fit 只用 train、score 只用 test（生产里的 fit_logistic/fit_lightgbm 是 fit+score 同集，会
     给出样本内 0.99 AUC 的假象；这里彻底分开）。
  3) 标准化/缺失填补的统计量（中位数/均值/方差）**只在 train 上估计**，再套用到 test。
  4) 三市场（CN/HK/US）永不合并，各自独立 train+test。
  5) 特征用 REALTIME_FEATURE_COLS（因果、score 日可见集；已排除 oracle_*/SMC delayed 等前视列）。

已知 caveat（非 train/test 泄漏，是数据属性）：估值乘数 + HK/CN 增长是**当前快照**贴到历史候选
（look-ahead），故绝对值偏乐观；US 增长是因果逐年。需要纯因果口径可另跑（去掉 valuation_* 等）。

LR 与生产 fit_logistic 完全一致（含本次修的 inf 清洗 + ±8σ winsorize）；lgbm 参数与生产一致。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # src/ 上 path（支持直接执行）
from stock_ana.config import DATA_DIR  # noqa: E402
from stock_ana.research.top_reversal.feature_registry import (  # noqa: E402
    REALTIME_FEATURE_COLS,
    apply_legacy_feature_aliases,
)

LABELED = DATA_DIR / "output" / "top_candidate_research" / "watchlist_unified_recall_candidates_labeled.csv"
WATCHLIST = DATA_DIR / "lists" / "watchlist.md"
MARKETS = ["US", "HK", "CN"]


def load_watchlist_members() -> dict[str, set[str]]:
    """解析 watchlist.md → {market: {归一化代码}}。HK 5 位补零 / CN 6 位补零 / US 大写。"""
    out: dict[str, set[str]] = {m: set() for m in MARKETS}
    cur = None
    sec = {"港股": "HK", "(HK)": "HK", "美股": "US", "(US)": "US", "大A": "CN", "(CN)": "CN"}
    for line in WATCHLIST.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            cur = next((v for k, v in sec.items() if k in line), None)
            continue
        if cur is None or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not cells or cells[0] in ("代码", "") or set(cells[0]) <= set("-"):
            continue
        code = cells[0]
        if cur == "HK" and code.isdigit():
            out["HK"].add(code.zfill(5))
        elif cur == "CN" and code.isdigit():
            out["CN"].add(code.zfill(6))
        elif cur == "US" and re.fullmatch(r"[A-Za-z.]+", code):
            out["US"].add(code.upper())
    return out


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """rank-based AUC（Mann–Whitney），并列取平均秩。"""
    y = np.asarray(y, float)
    score = np.asarray(score, float)
    m = np.isfinite(score)
    y, score = y[m], score[m]
    npos, nneg = int(y.sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    df = pd.DataFrame({"s": score, "r": ranks})
    ranks = df.groupby("s")["r"].transform("mean").to_numpy()
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def topk(y: np.ndarray, score: np.ndarray, frac: float) -> tuple[int, float, float]:
    """取分数最高的 frac 比例为预测正类 → (n, precision, recall)。"""
    y = np.asarray(y, float)
    score = np.asarray(score, float)
    order = np.argsort(-score, kind="mergesort")
    k = max(1, int(round(len(score) * frac)))
    sel = order[:k]
    tp = float(y[sel].sum())
    total_pos = float(y.sum())
    prec = tp / k if k else float("nan")
    rec = tp / total_pos if total_pos else float("nan")
    return k, prec * 100, rec * 100


def usable_features(train: pd.DataFrame, cols: list[str]) -> list[str]:
    thr = max(20, len(train) * 0.5)
    return [c for c in cols if c in train.columns
            and pd.to_numeric(train[c], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().sum() >= thr]


def fit_logistic_oos(train, test, feats):
    """与生产 fit_logistic 同口径（含 inf 清洗 + ±8σ clip），但 fit=train / score=test。"""
    xdf = train[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = xdf.median()
    xdf = xdf.fillna(med).fillna(0.0)
    mean = xdf.mean()
    std = xdf.std(ddof=0).replace(0, 1)
    x = np.clip(((xdf - mean) / std).to_numpy(float), -8.0, 8.0)
    y = (train["label"] == "true_top").astype(float).to_numpy()
    xb = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(xb.shape[1])
    lr, l2 = 0.04, 0.05
    for _ in range(2500):
        p = 1 / (1 + np.exp(-np.clip(xb @ beta, -30, 30)))
        grad = (xb.T @ (p - y)) / len(y)
        grad[1:] += l2 * beta[1:] / len(y)
        beta -= lr * grad
    tx = test[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    tx = np.clip(((tx - mean) / std).to_numpy(float), -8.0, 8.0)
    txb = np.column_stack([np.ones(len(tx)), tx])
    return 1 / (1 + np.exp(-np.clip(txb @ beta, -30, 30)))


def fit_lightgbm_oos(train, test, feats):
    """与生产 fit_lightgbm 同参数，5 seed 平均；fit=train / score=test。"""
    import lightgbm as lgb
    x_tr = train[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y_tr = (train["label"] == "true_top").astype(float).to_numpy()
    x_te = test[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    pos = max(1.0, float((y_tr == 1).sum()))
    params = dict(objective="binary", verbose=-1, num_threads=4, max_depth=3, num_leaves=7,
                  min_data_in_leaf=40, learning_rate=0.03, feature_fraction=0.6,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, lambda_l1=1.0,
                  scale_pos_weight=float((y_tr == 0).sum() / pos))
    preds = []
    for seed in range(5):
        m = lgb.train(dict(params, seed=seed), lgb.Dataset(x_tr, label=y_tr), num_boost_round=300)
        preds.append(m.predict(x_te))
    return np.mean(preds, axis=0)


def main():
    lab = apply_legacy_feature_aliases(pd.read_csv(LABELED, low_memory=False))
    lab = lab[lab["label"].isin(["true_top", "continuation"])].copy()
    lab["sym"] = lab["sym"].astype(str)
    wl = load_watchlist_members()

    rows = []
    for mk in MARKETS:
        g = lab[lab["market"].astype(str) == mk].copy()
        in_wl = g["sym"].apply(lambda s, m=mk: s in wl[m] or s.zfill(5) in wl[m] or s.zfill(6) in wl[m])
        train, test = g[~in_wl], g[in_wl]
        feats = usable_features(train, list(REALTIME_FEATURE_COLS))
        yte = (test["label"] == "true_top").astype(float).to_numpy()
        info = dict(market=mk, train_n=len(train), train_pos=int((train["label"] == "true_top").sum()),
                    test_n=len(test), test_pos=int(yte.sum()), feats=len(feats))
        if info["test_pos"] == 0 or (yte == 0).sum() == 0 or info["train_pos"] < 5:
            info["note"] = "test 正/负例不足，跳过"
            rows.append(info)
            continue
        for name, fn in [("LR", fit_logistic_oos), ("lgbm", fit_lightgbm_oos)]:
            s = fn(train, test, feats)
            a = auc(yte, s)
            _, p10, r10 = topk(yte, s, 0.10)
            _, p20, r20 = topk(yte, s, 0.20)
            rows.append({**info, "model": name, "AUC": round(a, 3),
                         "base%": round(yte.mean() * 100, 1),
                         "top10_prec": round(p10, 1), "top10_recall": round(r10, 1),
                         "top20_prec": round(p20, 1), "top20_recall": round(r20, 1)})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n================ 严格 OOS（test=watchlist, train=全集−test, 按市场分离）================")
    show = [c for c in ["market", "model", "train_n", "train_pos", "test_n", "test_pos", "base%",
                        "feats", "AUC", "top10_prec", "top10_recall", "top20_prec", "top20_recall", "note"]
            if c in res.columns]
    print(res[show].to_string(index=False))
    out = DATA_DIR / "output" / "top_candidate_research" / "watchlist_oos_eval.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n→ 落盘 {out}")


if __name__ == "__main__":
    main()
