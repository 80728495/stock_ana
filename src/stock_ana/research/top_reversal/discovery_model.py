#!/usr/bin/env python3
"""早发现（段A）模型的训练 / 持久化 / 加载 / 预测。

模型 = 每市场 lgb(5 seed) + logistic，特征用 DISCOVERY_FEATURE_COLS。训练一次落盘为
**可移植 JSON**（lgb=model_to_string、LR=med/mean/std/beta），入 git；其他机器/每日扫描
**直接加载已训模型预测**，不再运行时重训。训练口径与 eval_watchlist_oos 的 fit_* 同源（零漂移）。

用法：
    # 从周期性 build 的 labeled 训练集训练并落盘（需要时重训）
    python -m stock_ana.research.top_reversal.discovery_model --train
    # 指定训练集/输出目录
    python -m stock_ana.research.top_reversal.discovery_model --train --labeled <csv> --out <dir>
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
from stock_ana.research.top_reversal.eval_watchlist_oos import (  # noqa: E402
    fit_lightgbm,
    fit_logistic,
    predict_lightgbm,
    predict_logistic,
    usable_features,
)
from stock_ana.research.top_reversal.feature_registry import (  # noqa: E402
    DISCOVERY_FEATURE_COLS,
    apply_legacy_feature_aliases,
)

warnings.filterwarnings("ignore")

MODEL_DIR = DATA_DIR / "models" / "top_reversal"          # 入 git（见 .gitignore 白名单）
DEFAULT_LABELED = DATA_DIR / "output" / "top_candidate_research" / "watchlist_unified_recall_candidates_labeled.csv"
MARKETS = ("US", "HK", "CN")
DECIDED = ("true_top", "continuation")


# ── 训练 + 落盘 ────────────────────────────────────────────────────────────────

def train_discovery_models(train_labeled: pd.DataFrame, out_dir: Path = MODEL_DIR,
                           min_train: int = 60) -> dict[str, dict]:
    """每市场用「已定论」样本训练 lgb+LR（DISCOVERY_FEATURE_COLS），落盘可移植 JSON。返回各市场元信息。"""
    df = apply_legacy_feature_aliases(train_labeled.copy())
    df = df[df["label"].isin(DECIDED)]
    df["market"] = df["market"].astype(str)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, dict] = {}
    for mk in MARKETS:
        tr = df[df["market"] == mk]
        n_pos = int((tr["label"] == "true_top").sum())
        if len(tr) < min_train or n_pos < 5:
            print(f"  [{mk}] 样本不足（{len(tr)} 条 / {n_pos} 正例），跳过")
            continue
        feats = usable_features(tr, list(DISCOVERY_FEATURE_COLS))
        bundle = {
            "market": mk, "n_train": int(len(tr)), "n_pos": n_pos, "n_features": len(feats),
            "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "feature_set": "DISCOVERY_FEATURE_COLS",
            "lr": fit_logistic(tr, feats),
            "lgb": fit_lightgbm(tr, feats),
        }
        (out_dir / f"discovery_{mk}.json").write_text(
            json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
        meta[mk] = {k: bundle[k] for k in ("n_train", "n_pos", "n_features", "trained_at")}
        print(f"  [{mk}] 训练 {len(tr)} 条（{n_pos} 正例）/ {len(feats)} 特征 → discovery_{mk}.json")
    (out_dir / "manifest.json").write_text(
        json.dumps({"markets": meta, "feature_set": "DISCOVERY_FEATURE_COLS",
                    "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


# ── 加载 + 预测 ────────────────────────────────────────────────────────────────

def load_discovery_models(model_dir: Path = MODEL_DIR) -> dict[str, dict]:
    """加载各市场已训模型。返回 {market: {"lr":..., "lgb":..., meta}}。缺失则返回空 dict。"""
    models: dict[str, dict] = {}
    for mk in MARKETS:
        p = model_dir / f"discovery_{mk}.json"
        if p.exists():
            models[mk] = json.loads(p.read_text(encoding="utf-8"))
    return models


def predict_discovery(models: dict[str, dict], candidates: pd.DataFrame) -> pd.DataFrame:
    """用已加载模型给候选打分：新增 discovery_lgb / discovery_lr / strength（= lgb 概率）。"""
    out = apply_legacy_feature_aliases(candidates.copy())
    out["discovery_lgb"] = np.nan
    out["discovery_lr"] = np.nan
    if out.empty:
        out["strength"] = np.nan
        return out
    mk_series = out["market"].astype(str)
    for mk in mk_series.unique():
        if mk not in models:
            continue
        idx = out.index[mk_series == mk]
        sub = out.loc[idx]
        out.loc[idx, "discovery_lgb"] = predict_lightgbm(models[mk]["lgb"], sub)
        out.loc[idx, "discovery_lr"] = predict_logistic(models[mk]["lr"], sub)
    out["strength"] = pd.to_numeric(out["discovery_lgb"], errors="coerce")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true", help="从 labeled 训练集训练并落盘")
    ap.add_argument("--labeled", type=Path, default=DEFAULT_LABELED, help="训练集 CSV（周期性 build 产物）")
    ap.add_argument("--out", type=Path, default=MODEL_DIR)
    args = ap.parse_args()
    if not args.train:
        ap.print_help()
        return
    if not args.labeled.exists():
        print(f"训练集不存在：{args.labeled}（先跑 build 产出 labeled）")
        return
    print(f"训练早发现模型：{args.labeled.name} → {args.out}")
    train_discovery_models(pd.read_csv(args.labeled, low_memory=False), out_dir=args.out)
    print("完成。")


if __name__ == "__main__":
    main()
