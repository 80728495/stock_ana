#!/usr/bin/env python3
"""每日逃顶信号扫描（状态机驱动）—— 只实现扫描 + 信号生成/确认/取消，**不发消息**。

外围由系统脚本包裹（定时触发 + 读 events 发通知）。本脚本职责：
  1. 用最新价格对持仓做轻量扫描（scan_candidates：召回+特征，不 build/不训模型）。
  2. 段A 早发现打分（DISCOVERY_FEATURE_COLS，模型训练集来自周期性 build 的 labeled 数据）。
  3. 状态机推进：新报警(new_signal/alert) / 恢复上涨取消(cancel) / 下跌确认(confirm_down)。
  4. 状态持久化到 JSON；本轮事件写到 events-out 供外围发送。

首跑（无状态或 --backfill）：回填最近 `--backfill-days` 天（默认 60）的信号并存盘，
之后每天增量顺延（只推进上次扫描日之后的新交易日），信号在已存状态上继续演化。

状态文件格式：{"last_scan": "YYYY-MM-DD", "config": {...}, "signals": {"MKT:SYM": WatchState}}

用法：
    # 首次回填最近两个月并建状态
    python -m stock_ana.research.top_reversal.daily_escape_scan --backfill
    # 之后每日增量（外围定时调用）
    python -m stock_ana.research.top_reversal.daily_escape_scan
    # 指定“今日”（回测/补跑）
    python -m stock_ana.research.top_reversal.daily_escape_scan --asof 2026-06-30
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402

from stock_ana.config import DATA_DIR  # noqa: E402
from stock_ana.research.top_reversal.discovery_model import (  # noqa: E402
    MODEL_DIR,
    load_discovery_models,
    predict_discovery,
)
from stock_ana.research.top_reversal.escape_signal_tracker import (  # noqa: E402
    SignalConfig,
    WatchState,
    _load_prices_for,
    _norm_key,
    _prep_prices,
    advance_state,
    scan_candidates,
)

warnings.filterwarnings("ignore")

_OUT = DATA_DIR / "output" / "top_candidate_research"
DEFAULT_STATE = _OUT / "escape_signal_state.json"


# ── 状态文件（带 last_scan 元信息）──────────────────────────────────────────────

def load_state_file(path: Path) -> tuple[str | None, dict[str, WatchState]]:
    if not path.exists():
        return None, {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "signals" in raw:  # 新格式
        sig = {k: WatchState(**v) for k, v in raw["signals"].items()}
        return raw.get("last_scan"), sig
    return None, {k: WatchState(**v) for k, v in raw.items()}  # 兼容裸 dict


def save_state_file(path: Path, last_scan: str, state: dict[str, WatchState], cfg: SignalConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_scan": last_scan,
        "config": {"entry": cfg.entry_threshold, "alert": cfg.alert_threshold,
                   "confirm_include_bos": cfg.confirm_include_bos},
        "signals": {k: v.to_dict() for k, v in state.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _trading_days(prices: dict[str, pd.DataFrame], lo: pd.Timestamp, hi: pd.Timestamp) -> list[pd.Timestamp]:
    days = {d for df in prices.values() for d in df.index if lo <= d <= hi}
    return sorted(days)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-path", type=Path, default=DEFAULT_STATE)
    ap.add_argument("--events-out", type=Path, default=None, help="本轮事件 JSON（外围据此发通知）")
    ap.add_argument("--backfill", action="store_true", help="强制从空态回填 --backfill-days 天（重建状态）")
    ap.add_argument("--backfill-days", type=int, default=60)
    ap.add_argument("--confirm-mode", choices=["choch", "choch_bos"], default="choch_bos",
                    help="下跌确认判据：choch=仅 swing CHoCH↓；choch_bos=CHoCH↓ 或 BOS↓（默认，能抓急跌）")
    ap.add_argument("--entry-threshold", type=float, default=None)
    ap.add_argument("--alert-threshold", type=float, default=None)
    ap.add_argument("--asof", type=str, default=None, help="覆盖“今日”为该日（回测/补跑）；默认最新价格日")
    args = ap.parse_args()

    base = SignalConfig()
    cfg = SignalConfig(
        entry_threshold=args.entry_threshold if args.entry_threshold is not None else base.entry_threshold,
        alert_threshold=args.alert_threshold if args.alert_threshold is not None else base.alert_threshold,
        confirm_include_bos=(args.confirm_mode == "choch_bos"),
    )

    # 1) 加载已训模型（周期性 discovery_model --train 落盘，入 git）+ 2) 扫描 + 3) 段A 预测
    models = load_discovery_models()
    if not models:
        print(f"未找到早发现模型（{MODEL_DIR}）。先训练一次：\n"
              f"  python -m stock_ana.research.top_reversal.discovery_model --train")
        return
    print(f"[1/4] 扫描持仓最新候选（scan_candidates，asof={args.asof or '最新'}）...", flush=True)
    cand = scan_candidates(asof=args.asof)
    if cand.empty:
        print("无候选，退出。")
        return
    print(f"[2/4] 段A 加载模型打分 {len(cand)} 候选（市场 {sorted(models)}）...", flush=True)
    scored = predict_discovery(models, cand)
    scored["_asof"] = pd.to_datetime(scored["score_asof_date"], errors="coerce")
    scored["market"] = scored["market"].astype(str)
    scored["sym"] = scored["sym"].astype(str)

    keys = {_norm_key(m, s) for m, s in zip(scored["market"], scored["sym"], strict=False)}
    prices = {k: _prep_prices(df) for k, df in _load_prices_for(keys).items()}
    today = pd.Timestamp(args.asof) if args.asof else max(
        (df.index[-1] for df in prices.values()), default=scored["_asof"].max())

    # 4) 状态机推进
    last_scan, prior = load_state_file(args.state_path)
    if args.backfill or not prior or last_scan is None:
        lo = today - pd.Timedelta(days=args.backfill_days)
        days = _trading_days(prices, lo, today)
        prior = {}
        mode = f"回填 {lo.date()} ~ {today.date()}（{len(days)} 交易日）"
    else:
        last = pd.Timestamp(last_scan)
        days = _trading_days(prices, last + pd.Timedelta(days=1), today)
        mode = f"增量 {last.date()} → {today.date()}（{len(days)} 新交易日）"
    print(f"[3/4] 状态机推进：{mode}；确认判据={args.confirm_mode}", flush=True)

    state, events = advance_state(prior, scored, prices, days, cfg)
    save_state_file(args.state_path, today.strftime("%Y-%m-%d"), state, cfg)

    # 事件落盘（供外围发送）
    events_out = args.events_out or (args.state_path.parent / f"escape_events_{today.date()}.json")
    events_out.write_text(
        json.dumps([dataclasses.asdict(e) for e in events], ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总打印
    print(f"[4/4] 完成。本轮事件 {len(events)}；状态 {args.state_path}；事件 {events_out}\n")
    by_kind = {k: [e for e in events if e.kind == k] for k in ("confirm_down", "cancel", "alert", "new_signal")}
    label = {"confirm_down": "下跌确认", "cancel": "取消(恢复上涨)", "alert": "报警", "new_signal": "新入队"}
    for k in ("confirm_down", "cancel", "alert", "new_signal"):
        evs = by_kind[k]
        if not evs:
            continue
        print(f"  【{label[k]}】{len(evs)}")
        for e in sorted(evs, key=lambda x: (x.date, x.market, x.sym)):
            extra = f" 距顶{e.drop_pct:+.0f}%" if e.kind == "confirm_down" and e.drop_pct == e.drop_pct else (
                f" strength={e.strength:.2f}" if e.kind in ("alert", "new_signal") and e.strength == e.strength else "")
            print(f"    {e.date} {e.market}:{e.sym:8}{extra}  {e.message}")
    watching = sorted(k for k, v in state.items() if v.state == "watching")
    print(f"\n  当前观察队列 {len(watching)}：" + "、".join(watching))


if __name__ == "__main__":
    main()
