#!/usr/bin/env python3
"""Vegas 回踩模型的训练 / 持久化 / 加载 / 预测（对标 top_reversal.discovery_model）。

模型按 **support(mid/long) × market(US/HK/CN) × target** 分离（绝不合并市场）。每个
= LR + lgb(5 seed)，训练一次落盘为**可移植 JSON**（lgb=model_to_string，LR=med/mean/std/beta）
入 git；其他机器/每日扫描直接加载预测，不再运行时重训。训练口径复用 eval_oos 的
`_fit_logistic`/`_fit_lightgbm`（一份真源，零漂移），只是把返回的 bundle 序列化。

默认落盘的生产目标（可 --targets 覆盖）：
  mid  : buy                       （mid 回踩「该不该抄底」）
  long : buy, buy_big, w2_chain    （long 抄底 / 大二浪早期猎手 / W2 链结构）

用法：
    python -m stock_ana.research.vegas_pullback.vegas_model --train
    python -m stock_ana.research.vegas_pullback.vegas_model --train --supports long --targets w2_chain
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stock_ana.config import DATA_DIR  # noqa: E402
from stock_ana.research.top_reversal.eval_watchlist_oos import usable_features  # noqa: E402
from stock_ana.research.vegas_pullback.eval_oos import (  # noqa: E402
    MARKETS,
    TARGETS,
    _fit_lightgbm,
    _fit_logistic,
)
from stock_ana.research.vegas_pullback.feature_registry import (  # noqa: E402
    REALTIME_FEATURE_COLS,
    feature_group_for,
)

warnings.filterwarnings("ignore")

MODEL_DIR = DATA_DIR / "models" / "vegas_pullback"                 # 入 git（!data/models/ 白名单）
LABELED_DIR = DATA_DIR / "output" / "vegas_pullback_research"       # {support}_labeled.csv（build 产物）
DEFAULT_TARGETS = {"mid": ("buy",), "long": ("buy", "buy_big", "w2_chain")}
MIN_TRAIN, MIN_POS = 60, 5


def _feats_for(target: str, train: pd.DataFrame) -> list[str]:
    exclude = set(TARGETS[target].get("exclude_groups", ()))
    feats_all = [c for c in REALTIME_FEATURE_COLS if feature_group_for(c) not in exclude]
    return usable_features(train, feats_all)


def _serialize(support: str, target: str, mk: str, g: pd.DataFrame, feats: list[str],
               col: str, pos: str, n_pos: int) -> dict:
    lrb = _fit_logistic(g, feats, col, pos)          # {feats, med(Series), mean, std, beta}
    boosters, _ = _fit_lightgbm(g, feats, col, pos)  # [Booster]×5
    return {
        "support": support, "target": target, "market": mk, "target_col": col, "pos_label": pos,
        "n_train": int(len(g)), "n_pos": int(n_pos), "n_features": len(feats),
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lr": {"feats": list(feats), "med": {k: float(v) for k, v in lrb["med"].items()},
               "mean": {k: float(v) for k, v in lrb["mean"].items()},
               "std": {k: float(v) for k, v in lrb["std"].items()}, "beta": lrb["beta"].tolist()},
        "lgb": {"feats": list(feats), "boosters": [b.model_to_string() for b in boosters]},
    }


# ── 训练 + 落盘 ────────────────────────────────────────────────────────────────

def train_vegas_models(supports_targets: dict[str, tuple[str, ...]] = DEFAULT_TARGETS,
                       out_dir: Path = MODEL_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}
    for support, targets in supports_targets.items():
        path = LABELED_DIR / f"{support}_labeled.csv"
        if not path.exists():
            print(f"[{support}] 缺训练集 {path}（先跑 build）")
            continue
        lab = pd.read_csv(path, low_memory=False)
        lab["sym"] = lab["sym"].astype(str)
        lab["market"] = lab["market"].astype(str)
        for target in targets:
            spec = TARGETS[target]
            col, pos, neg = spec["col"], spec["pos"], spec["neg"]
            if col not in lab.columns:
                print(f"  [{support}/{target}] 无标签列 {col}，跳过")
                continue
            sub = lab[lab[col].isin([pos, neg])]
            for mk in MARKETS:
                g = sub[sub["market"] == mk]
                n_pos = int((g[col] == pos).sum())
                if len(g) < MIN_TRAIN or n_pos < MIN_POS or g[col].nunique() < 2:
                    print(f"  [{support}/{target}/{mk}] 样本不足（{len(g)}/{n_pos}），跳过")
                    continue
                feats = _feats_for(target, g)
                if len(feats) < 3:
                    print(f"  [{support}/{target}/{mk}] 可用特征不足，跳过")
                    continue
                bundle = _serialize(support, target, mk, g, feats, col, pos, n_pos)
                fn = f"{support}_{target}_{mk}.json"
                (out_dir / fn).write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
                manifest[fn] = {k: bundle[k] for k in ("n_train", "n_pos", "n_features", "trained_at")}
                print(f"  [{support}/{target}/{mk}] {len(g)} 条（{n_pos} 正例）/ {len(feats)} 特征 → {fn}")
    (out_dir / "manifest.json").write_text(
        json.dumps({"models": manifest, "default_targets": {k: list(v) for k, v in supports_targets.items()},
                    "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


# ── 加载 + 预测 ────────────────────────────────────────────────────────────────

def load_vegas_models(support: str, target: str, model_dir: Path = MODEL_DIR) -> dict[str, dict]:
    """加载某 (support,target) 的各市场模型 → {market: bundle}。缺失返回空 dict。"""
    models: dict[str, dict] = {}
    for mk in MARKETS:
        p = model_dir / f"{support}_{target}_{mk}.json"
        if p.exists():
            models[mk] = json.loads(p.read_text(encoding="utf-8"))
    return models


def _predict_lr(lrb: dict, df: pd.DataFrame) -> np.ndarray:
    med = pd.Series(lrb["med"])
    mean = pd.Series(lrb["mean"])
    std = pd.Series(lrb["std"])
    beta = np.asarray(lrb["beta"], dtype=float)
    tx = df.reindex(columns=lrb["feats"]).apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    tx = np.clip(((tx - mean) / std).to_numpy(float), -8.0, 8.0)
    txb = np.column_stack([np.ones(len(tx)), tx])
    return 1 / (1 + np.exp(-np.clip(txb @ beta, -30, 30)))


def _predict_lgb(lgbb: dict, df: pd.DataFrame) -> np.ndarray:
    import lightgbm as lgb
    x = df.reindex(columns=lgbb["feats"]).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return np.mean([lgb.Booster(model_str=s).predict(x) for s in lgbb["boosters"]], axis=0)


def predict_vegas(models: dict[str, dict], candidates: pd.DataFrame) -> pd.DataFrame:
    """用已加载模型给回踩候选打分：新增 vegas_lgb / vegas_lr / score（=lgb 概率，高=越该抄底）。"""
    out = candidates.copy()
    out["vegas_lgb"] = np.nan
    out["vegas_lr"] = np.nan
    if out.empty:
        out["score"] = np.nan
        return out
    mk_series = out["market"].astype(str)
    for mk in mk_series.unique():
        if mk not in models:
            continue
        idx = out.index[mk_series == mk]
        sub = out.loc[idx]
        out.loc[idx, "vegas_lgb"] = _predict_lgb(models[mk]["lgb"], sub)
        out.loc[idx, "vegas_lr"] = _predict_lr(models[mk]["lr"], sub)
    out["score"] = pd.to_numeric(out["vegas_lgb"], errors="coerce")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--supports", nargs="*", choices=["mid", "long"], default=None)
    ap.add_argument("--targets", nargs="*", choices=list(TARGETS), default=None)
    ap.add_argument("--out", type=Path, default=MODEL_DIR)
    args = ap.parse_args()
    if not args.train:
        ap.print_help()
        return
    st = {s: DEFAULT_TARGETS[s] for s in (args.supports or DEFAULT_TARGETS)}
    if args.targets:  # 指定则对所选 support 一律用这批 target
        st = {s: tuple(args.targets) for s in st}
    print(f"训练 vegas 回踩模型 → {args.out}")
    train_vegas_models(st, out_dir=args.out)
    print("完成。")


if __name__ == "__main__":
    main()
