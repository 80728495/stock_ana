"""逃顶信号状态机 —— 信号「生成 / 确认 / 取消」核心逻辑（每日扫描+通知派发由外部实现）。

设计（对齐用户使用流程，见 memory top-reversal-discovery-no-confirm-features）：

  段 A  早发现（L0，ML 打分）：每市场用历史「已定论」候选训练 lgb+lr（DISCOVERY_FEATURE_COLS，
        已剔除 14 个顶后确认型特征），对今日召回候选打分 → strength。召回本身就是 L0（最早检测，
        smc_raw/early/supply_held 三机制取最早确认，不等 swing CHoCH），此处只做「早期判断」。

  段 B  进一步确认（规则，非 ML，as-of 最新）：对被观察的顶，用最新价重算 SMC 结构，判定
        · resumed（优先）  = 恢复上涨：顶后收盘创新高越过顶 → 顶被证伪、信号被吃回中继
        · confirmed_down   = 顶后出现「摆动级 CHoCH 向下」且从未收复顶部（= 打标签同一判据，下跌坐实）
        越顶优先于下跌确认（一旦重新越顶，该顶即不成立）。这是 score 之外的独立确认层，
        用来「升级 / 取消」早期信号。

状态机（每标的一条活跃 saga，terminal 后由更新的顶重新起 saga）：
    (无/terminal) --[今日 strength≥entry]--> watching(+new_signal, strength≥alert 再+alert)
    watching --[段B confirmed_down]--> confirmed_down (+confirm_down，无条件通知)
    watching --[段B resumed]--------> cancelled       (+cancel，仅当此前已 alert 过)
    watching --[今日 strength≥alert 且未 alert 过]--> (+alert)

事件类型：new_signal / alert / confirm_down / cancel —— 交外部去发通知。

用法（库）：
    state = load_state(path)
    new_state, events = run_daily(prices, candidates_today, train_labeled, state, cfg, today)
    save_state(new_state, path)
外部：把 prices / 今日扫描候选喂进来，按 events 发通知，持久化 new_state。

演示（CLI）：
    python -m stock_ana.research.top_reversal.escape_signal_tracker --demo
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stock_ana.config import DATA_DIR  # noqa: E402
from stock_ana.research.top_reversal.eval_watchlist_oos import (  # noqa: E402
    fit_lightgbm_oos,
    fit_logistic_oos,
    usable_features,
)
from stock_ana.research.top_reversal.feature_registry import (  # noqa: E402
    DISCOVERY_FEATURE_COLS,
    apply_legacy_feature_aliases,
)
from stock_ana.research.top_reversal.smc_context import _normalize_df, build_smc_bundle  # noqa: E402

warnings.filterwarnings("ignore")

DECIDED_LABELS = ("true_top", "continuation")


# ── 配置 ──────────────────────────────────────────────────────────────────────

@dataclass
class SignalConfig:
    entry_threshold: float = 0.35      # 进观察队列的最低早发现强度（lgb 概率 0-1）
    alert_threshold: float = 0.60      # 触发提醒的强度
    recent_days: int = 7               # score_asof 距 today 在此天数内 → 算「今日新信号」
    resume_break_pct: float = 0.0      # 收盘越过顶 high 此百分比 → 判定恢复上涨（0=任意越顶收盘）
    min_train_decided: int = 60        # 该市场训练已定论样本下限，不足则跳过打分
    confirm_include_bos: bool = False  # confirm 是否接受 swing 级 BOS↓（除 CHoCH↓外）：
    #   False=只认 swing CHoCH↓（= 打标签同一判据，对慢反转干净、但急跌先出 BOS 会漏）；
    #   True=swing CHoCH↓ 或 BOS↓（任一 swing 级看跌破位，能抓急跌，供对比）。


# ── 状态与事件 ────────────────────────────────────────────────────────────────

@dataclass
class WatchState:
    market: str
    sym: str
    state: str                          # watching / confirmed_down / cancelled
    top_date: str                       # 被观察的顶（YYYY-MM-DD）
    top_high: float
    first_signal_date: str
    last_update: str
    peak_strength: float
    alerted: bool = False
    resolved_date: str | None = None
    drop_pct: float = float("nan")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class SignalEvent:
    kind: str                           # new_signal / alert / confirm_down / cancel
    market: str
    sym: str
    date: str
    top_date: str
    strength: float = float("nan")
    drop_pct: float = float("nan")
    mechanism: str = ""
    message: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class TodaySignal:
    top_date: str
    top_high: float
    strength: float
    asof_date: str
    lgb: float = float("nan")
    lr: float = float("nan")


@dataclass
class StructStatus:
    resolution: str | None = None       # None / confirmed_down / resumed
    confirm_date: str | None = None
    resume_date: str | None = None
    top_high: float = float("nan")      # 顶当根的 high（权威口径）
    drop_pct: float = float("nan")      # 现价距顶 %（负=在顶下方）
    high_since_top: float = float("nan")
    mechanism: str = ""


# ── 段 B：结构确认 / 取消（规则，非 ML，as-of 最新）────────────────────────────

def _swing_break_pos(structure_events: pd.DataFrame | None, top_pos: int, direction: int,
                     etypes: tuple[str, ...] = ("choch",)) -> int | None:
    """顶后最早的「摆动级结构破位，指定方向」确认位（broken_pos）。

    etypes=("choch",)         → 只认 CHoCH（性质由涨转跌，慢反转干净，= 打标签判据）；
    etypes=("choch","bos")    → CHoCH 或 BOS（任一 swing 级看跌破位，能抓急跌）。
    direction: -1=向下坐实见顶，+1=向上恢复上涨。
    """
    if structure_events is None or structure_events.empty:
        return None
    etype = structure_events.get("event_type")
    scale = structure_events.get("scale")
    if etype is None or scale is None:
        return None
    d = pd.to_numeric(structure_events.get("direction"), errors="coerce")
    broken = pd.to_numeric(structure_events.get("broken_pos"), errors="coerce")
    mask = (
        (d == direction)
        & etype.astype(str).isin(etypes)
        & scale.astype(str).str.startswith("swing")
        & broken.notna()
        & (broken > top_pos)
    )
    if not bool(mask.any()):
        return None
    return int(broken[mask].min())


def _swing_choch_pos(structure_events: pd.DataFrame | None, top_pos: int, direction: int) -> int | None:
    """向后兼容包装：只认 swing CHoCH。"""
    return _swing_break_pos(structure_events, top_pos, direction, ("choch",))


def _prep_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_df(df)
    df.index = pd.to_datetime(df.index)
    return df[~df.index.duplicated(keep="last")].sort_index()


def _top_pos(idx: pd.DatetimeIndex, top_date: str) -> int | None:
    ts = pd.Timestamp(top_date)
    if ts in idx:
        return int(idx.get_loc(ts))
    prior = idx[idx <= ts]  # 顶日无 bar（停牌等）→ 取不晚于顶日的最近一根
    return int(idx.get_loc(prior[-1])) if len(prior) else None


def top_high_at(df: pd.DataFrame, top_date: str) -> float:
    """顶当根的 high（权威顶价口径，供打分/状态复用）。"""
    df = _prep_prices(df)
    tp = _top_pos(df.index, top_date)
    return float(df["high"].iloc[tp]) if tp is not None else float("nan")


def structural_status(df: pd.DataFrame, top_date: str, cfg: SignalConfig) -> StructStatus:
    """用最新价重算 SMC 结构，判定被观察顶是否 恢复上涨 / 下跌坐实。

    优先级：**收盘创新高越过顶 = 顶被证伪 → resumed（优先）**；只有从未重新越顶、
    且顶后出现「摆动级 CHoCH 向下」才是 confirmed_down（下跌结构坐实且未收复顶部）。
    """
    df = _prep_prices(df)
    idx = df.index
    top_pos = _top_pos(idx, top_date)
    if top_pos is None:
        return StructStatus()
    top_high = float(df["high"].iloc[top_pos])
    last_close = float(df["close"].iloc[-1])
    after_high = df["high"].iloc[top_pos + 1:]
    high_since = float(after_high.max()) if len(after_high) else float("nan")
    st = StructStatus(top_high=round(top_high, 4),
                      drop_pct=round((last_close / top_high - 1) * 100, 2),
                      high_since_top=round(high_since, 4))

    # ① 恢复上涨（优先）：顶后首个收盘越过顶 high*(1+resume_break_pct) → 顶被证伪、信号被吃
    thr = top_high * (1 + cfg.resume_break_pct / 100.0)
    after_close = df["close"].iloc[top_pos + 1:]
    broke = np.where(after_close.to_numpy() > thr)[0]
    if broke.size:
        rpos = top_pos + 1 + int(broke[0])
        st.resolution = "resumed"
        st.resume_date = idx[rpos].strftime("%Y-%m-%d")
        st.mechanism = "reclaim_new_high"
        return st

    # ② 下跌坐实：顶后摆动级看跌破位（CHoCH↓，或按 cfg 也含 BOS↓），且从未收复顶部
    etypes = ("choch", "bos") if cfg.confirm_include_bos else ("choch",)
    down_pos = _swing_break_pos(build_smc_bundle(df).get("structure_events"), top_pos, -1, etypes)
    if down_pos is not None:
        st.resolution = "confirmed_down"
        st.confirm_date = idx[down_pos].strftime("%Y-%m-%d") if down_pos < len(idx) else None
        st.mechanism = "swing_break_down" if cfg.confirm_include_bos else "swing_choch_down"
    return st


# ── 段 A：早发现打分（ML）────────────────────────────────────────────────────

def score_discovery(candidates: pd.DataFrame, train_labeled: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    """每市场用「已定论」样本训练 lgb+lr（DISCOVERY_FEATURE_COLS），给候选打分。

    返回 candidates 副本，新增列 discovery_lgb / discovery_lr / strength（= lgb 概率）。
    strength 为 lgb 概率（0-1），阈值即施于此。
    """
    out = apply_legacy_feature_aliases(candidates.copy())
    train = apply_legacy_feature_aliases(train_labeled.copy())
    out["discovery_lgb"] = np.nan
    out["discovery_lr"] = np.nan
    if out.empty:
        out["strength"] = np.nan
        return out
    for mk in out["market"].astype(str).unique():
        cand = out[out["market"].astype(str) == mk]
        tr = train[(train["market"].astype(str) == mk) & train["label"].isin(DECIDED_LABELS)]
        if len(tr) < cfg.min_train_decided or cand.empty:
            continue
        feats = usable_features(tr, list(DISCOVERY_FEATURE_COLS))
        out.loc[cand.index, "discovery_lgb"] = fit_lightgbm_oos(tr, cand, feats)
        out.loc[cand.index, "discovery_lr"] = fit_logistic_oos(tr, cand, feats)
    out["strength"] = pd.to_numeric(out["discovery_lgb"], errors="coerce")
    return out


def _pick_today_signal(sym_candidates: pd.DataFrame, today: pd.Timestamp, cfg: SignalConfig) -> TodaySignal | None:
    """某标的今日新信号 = score_asof 在 [today-recent_days, today] 内、strength 最高的候选。"""
    g = sym_candidates.copy()
    g["_asof"] = pd.to_datetime(g.get("score_asof_date"), errors="coerce")
    g = g[(g["_asof"] >= today - pd.Timedelta(days=cfg.recent_days)) & (g["_asof"] <= today)]
    g = g[pd.to_numeric(g["strength"], errors="coerce").notna()]
    if g.empty:
        return None
    r = g.loc[pd.to_numeric(g["strength"], errors="coerce").idxmax()]
    return TodaySignal(
        top_date=pd.Timestamp(r["top_date"]).strftime("%Y-%m-%d"),
        top_high=float(r.get("top_high", r.get("top_price", np.nan))),
        strength=float(r["strength"]),
        asof_date=pd.Timestamp(r["_asof"]).strftime("%Y-%m-%d"),
        lgb=float(r.get("discovery_lgb", np.nan)),
        lr=float(r.get("discovery_lr", np.nan)),
    )


# ── 状态机核心（纯函数，可单测）───────────────────────────────────────────────

def step(prior: WatchState | None, today_signal: TodaySignal | None, struct: StructStatus,
         market: str, sym: str, today: str, cfg: SignalConfig) -> tuple[WatchState | None, list[SignalEvent]]:
    """单标的一日状态推进。返回 (新状态, 事件列表)。prior 可为 None 或 terminal 状态。"""
    events: list[SignalEvent] = []
    watching = prior is not None and prior.state == "watching"

    if watching:
        assert prior is not None
        st = dataclasses.replace(prior, last_update=today)
        # 强度刷新 + 补发提醒（此前未提醒且今日达阈值）
        if today_signal is not None and not np.isnan(today_signal.strength):
            st.peak_strength = max(st.peak_strength, today_signal.strength)
            if not st.alerted and today_signal.strength >= cfg.alert_threshold:
                st.alerted = True
                events.append(SignalEvent("alert", market, sym, today, st.top_date, strength=today_signal.strength,
                                          message=f"逃顶信号增强至提醒阈值 strength={today_signal.strength:.2f}"))
        # 段 B 结构判定（先发生者为准）
        if struct.resolution == "confirmed_down":
            st.state = "confirmed_down"
            st.resolved_date = today
            st.drop_pct = struct.drop_pct
            msg = f"下跌结构确认（swing CHoCH↓ @ {struct.confirm_date}），距顶 {struct.drop_pct:.1f}%"
            events.append(SignalEvent("confirm_down", market, sym, today, st.top_date, drop_pct=struct.drop_pct,
                                      mechanism=struct.mechanism, message=msg))
        elif struct.resolution == "resumed":
            st.state = "cancelled"
            st.resolved_date = today
            st.drop_pct = struct.drop_pct
            if prior.alerted:  # 仅当此前通知过才发取消
                events.append(SignalEvent("cancel", market, sym, today, st.top_date,
                                          message=f"逃顶信号取消：恢复上涨中继（@ {struct.resume_date}）"))
        return st, events

    # 未在观察（prior 为 None 或 terminal）→ 视今日信号起新 saga
    fresh = today_signal is not None and not np.isnan(today_signal.strength)
    if fresh and today_signal.strength >= cfg.entry_threshold:
        # terminal 后需「更新的顶」才重开，避免同一已了结顶反复触发
        if prior is not None and prior.state in ("confirmed_down", "cancelled"):
            if not (pd.Timestamp(today_signal.top_date) > pd.Timestamp(prior.top_date)):
                return prior, events
        st = WatchState(
            market=market, sym=sym, state="watching", top_date=today_signal.top_date,
            top_high=today_signal.top_high, first_signal_date=today, last_update=today,
            peak_strength=today_signal.strength, alerted=False,
        )
        events.append(SignalEvent("new_signal", market, sym, today, st.top_date, strength=today_signal.strength,
                                  message=f"发现逃顶信号 strength={today_signal.strength:.2f} → 进入观察队列"))
        if today_signal.strength >= cfg.alert_threshold:
            st.alerted = True
            events.append(SignalEvent("alert", market, sym, today, st.top_date, strength=today_signal.strength,
                                      message=f"逃顶信号达提醒阈值 strength={today_signal.strength:.2f}"))
        return st, events

    return prior, events


# ── 编排：一日全量更新 ────────────────────────────────────────────────────────

def _norm_key(market: str, sym: str) -> str:
    return f"{market}:{sym}"


def run_daily(prices: dict[str, pd.DataFrame], candidates_today: pd.DataFrame, train_labeled: pd.DataFrame,
              prior_state: dict[str, WatchState], cfg: SignalConfig,
              today: str | pd.Timestamp | None = None) -> tuple[dict[str, WatchState], list[SignalEvent]]:
    """一日全量更新。

    prices: {"MARKET:SYM": OHLCV df}（段 B 结构判定用；键用 _norm_key）。
    candidates_today: 今日扫描召回候选（带特征、market/sym/top_date/score_asof_date），外部产出。
    train_labeled: 历史 labeled 训练集（段 A 早发现模型）。
    prior_state: {key: WatchState}。
    返回 (新状态, 事件)。
    """
    scored = score_discovery(candidates_today, train_labeled, cfg)
    scored["market"] = scored["market"].astype(str)
    scored["sym"] = scored["sym"].astype(str)
    today_ts = pd.Timestamp(today) if today is not None else pd.to_datetime(
        scored.get("score_asof_date"), errors="coerce").max()
    today_str = today_ts.strftime("%Y-%m-%d")

    # 需处理的标的 = 已在观察的 ∪ 今日有候选的
    watching_keys = {k for k, v in prior_state.items() if v.state == "watching"}
    cand_keys = {_norm_key(m, s) for m, s in zip(scored["market"], scored["sym"], strict=False)}
    new_state: dict[str, WatchState] = dict(prior_state)
    events: list[SignalEvent] = []

    for key in sorted(watching_keys | cand_keys):
        market, sym = key.split(":", 1)
        prior = prior_state.get(key)
        sym_cands = scored[(scored["market"] == market) & (scored["sym"] == sym)]
        today_signal = _pick_today_signal(sym_cands, today_ts, cfg) if not sym_cands.empty else None

        # 决定要结构判定的顶：观察中→用被观察顶；否则→今日信号顶
        watched_top = prior.top_date if (prior and prior.state == "watching") else (
            today_signal.top_date if today_signal else None)
        has_px = key in prices and prices[key] is not None and len(prices[key])
        struct = StructStatus()
        if watched_top is not None and has_px:
            struct = structural_status(prices[key], watched_top, cfg)
        # 顶价一律以价格序列为权威口径（不信候选列，避免列名/缺失问题）
        if today_signal is not None and has_px:
            today_signal.top_high = top_high_at(prices[key], today_signal.top_date)

        updated, evs = step(prior, today_signal, struct, market, sym, today_str, cfg)
        if updated is not None:
            new_state[key] = updated
        events.extend(evs)
    return new_state, events


# ── 因果多日推进驱动（回填/增量共用；结构判定截至当日，不看未来）────────────────

def _causal_resolution(events, df: pd.DataFrame, top_date: str, cfg: SignalConfig):
    """一次性求被观察顶的 (top_pos, top_high, confirm_pos, reclaim_pos)（判据由 cfg 决定）。"""
    tp = _top_pos(df.index, top_date)
    if tp is None:
        return None
    top_high = float(df["high"].iloc[tp])
    etypes = ("choch", "bos") if cfg.confirm_include_bos else ("choch",)
    confirm_pos = _swing_break_pos(events, tp, -1, etypes)
    thr = top_high * (1 + cfg.resume_break_pct / 100.0)
    after = df["close"].iloc[tp + 1:]
    broke = np.where(after.to_numpy() > thr)[0]
    reclaim_pos = tp + 1 + int(broke[0]) if broke.size else None
    return tp, top_high, confirm_pos, reclaim_pos


def _struct_asof(df: pd.DataFrame, d_pos: int, sched, cfg: SignalConfig) -> StructStatus:
    """按截至 d_pos 的数据合成被观察顶 StructStatus（confirm/reclaim 先到者胜）。"""
    _tp, top_high, confirm_pos, reclaim_pos = sched
    st = StructStatus(top_high=round(top_high, 4))
    res: list[tuple[int, str]] = []
    if reclaim_pos is not None and reclaim_pos <= d_pos:
        res.append((reclaim_pos, "resumed"))
    if confirm_pos is not None and confirm_pos <= d_pos:
        res.append((confirm_pos, "confirmed_down"))
    if not res:
        return st
    res.sort()
    pos, kind = res[0]
    st.resolution = kind
    st.drop_pct = round((float(df["close"].iloc[d_pos]) / top_high - 1) * 100, 2)
    if kind == "confirmed_down":
        st.confirm_date = df.index[pos].strftime("%Y-%m-%d")
        st.mechanism = "swing_break_down" if cfg.confirm_include_bos else "swing_choch_down"
    else:
        st.resume_date = df.index[pos].strftime("%Y-%m-%d")
    return st


def advance_state(prior_state: dict[str, WatchState], scored: pd.DataFrame, prices: dict[str, pd.DataFrame],
                  days, cfg: SignalConfig) -> tuple[dict[str, WatchState], list[SignalEvent]]:
    """按交易日序列 `days` 因果推进状态机。回填(从空态跑整段)与每日增量(接已存状态跑新增日)共用。

    scored：已打分候选（含 market/sym/top_date/score_asof_date/strength）。prices：{key: 已 _prep 的 OHLCV}。
    每股 SMC bundle 只算一次；结构判定用「≤当日」的位置比较，绝不看未来。返回 (新状态, 事件时间线)。
    """
    sc = scored.copy()
    sc["market"] = sc["market"].astype(str)
    sc["sym"] = sc["sym"].astype(str)
    if "_asof" not in sc.columns:
        sc["_asof"] = pd.to_datetime(sc["score_asof_date"], errors="coerce")
    state = dict(prior_state)
    events: list[SignalEvent] = []
    ev_cache: dict[str, object] = {}
    sched_cache: dict[str, tuple] = {}

    def events_of(key: str):
        if key not in ev_cache:
            ev_cache[key] = build_smc_bundle(prices[key]).get("structure_events") if key in prices else None
        return ev_cache[key]

    for D in days:
        dts = pd.Timestamp(D)
        ds = dts.strftime("%Y-%m-%d")
        fresh = sc[sc["_asof"] == dts]
        fresh_keys = {_norm_key(m, s) for m, s in zip(fresh["market"], fresh["sym"], strict=False)}
        watching = {k for k, v in state.items() if v.state == "watching"}
        for key in sorted(watching | fresh_keys):
            m, s = key.split(":", 1)
            prior = state.get(key)
            today_signal = None
            g = fresh[(fresh["market"] == m) & (fresh["sym"] == s)]
            if not g.empty and key in prices:
                r = g.loc[pd.to_numeric(g["strength"], errors="coerce").idxmax()]
                if not pd.isna(r["strength"]):
                    td = pd.Timestamp(r["top_date"]).strftime("%Y-%m-%d")
                    today_signal = TodaySignal(
                        td, top_high_at(prices[key], td), float(r["strength"]), ds,
                        float(r.get("discovery_lgb", np.nan)), float(r.get("discovery_lr", np.nan)))
            watched_top = prior.top_date if (prior and prior.state == "watching") else (
                today_signal.top_date if today_signal else None)
            struct = StructStatus()
            if watched_top is not None and key in prices:
                df = prices[key]
                idx = df.index
                if key not in sched_cache or sched_cache[key][0] != watched_top:
                    sched_cache[key] = (watched_top, _causal_resolution(events_of(key), df, watched_top, cfg))
                sc_tuple = sched_cache[key][1]
                if sc_tuple is not None:
                    if dts in idx:
                        d_pos = int(idx.get_loc(dts))
                    else:
                        le = idx[idx <= dts]
                        d_pos = int(idx.get_loc(le[-1])) if len(le) else None
                    if d_pos is not None:
                        struct = _struct_asof(df, d_pos, sc_tuple, cfg)
            new, evs = step(prior, today_signal, struct, m, s, ds, cfg)
            if new is not None:
                state[key] = new
            events.extend(evs)
    return state, events


# ── 轻量每日扫描：召回+特征（复用 build 的召回/特征函数，不 build/不训模型）──────

def scan_candidates(asof: str | pd.Timestamp | None = None, holdings_only: bool = True,
                    markets: set[str] | None = None) -> pd.DataFrame:
    """每日扫描候选：只对持仓(或指定市场)召回顶部候选并算特征，返回带特征的候选 DataFrame。

    与 build 的区别：**不打标签、不训模型、不评估、不扫全宇宙候选**——只做「召回+特征」这一份
    每日推断所需的工作，复用 build 的 `_build_symbol_research_rows` + `add_research_features`
    保证与训练零漂移。横截面/行业特征需全宇宙上下文，故仍加载全宇宙价格(仅读 parquet)喂给
    特征构建器，但**只对持仓召回候选**。asof 给定则各票价格截到该日(因果回放/补算用)。
    """
    from stock_ana.data.market_data import load_tech_pools_data
    from stock_ana.research.top_reversal.build_top_candidate_research import (
        _build_symbol_research_rows,
        build_arg_parser,
    )
    from stock_ana.research.top_reversal.feature_pipeline import add_research_features

    args = build_arg_parser().parse_args([])
    data = load_tech_pools_data(min_history=args.min_history, include_holding=True)
    if asof is not None:
        cut = pd.Timestamp(asof)
        trimmed = {}
        for k, v in data.items():
            df = _prep_prices(v["df"])
            df = df[df.index <= cut]
            if len(df):
                trimmed[k] = {**v, "df": df}
        data = trimmed
    hold = _parse_holdings()

    def keep(m: str, s: str) -> bool:
        if markets and m not in markets:
            return False
        if not holdings_only:
            return True
        return s in hold.get(m, set()) or s.zfill(5) in hold.get(m, set()) or s.zfill(6) in hold.get(m, set())

    rows: list[dict] = []
    for item in data.values():
        if not keep(str(item["market"]), str(item["symbol"])):
            continue
        try:
            rows.extend(_build_symbol_research_rows(item, args)[5])  # [5] = unified 召回
        except Exception:  # noqa: BLE001
            continue
    ds = pd.DataFrame(rows)
    if ds.empty:
        return ds
    return add_research_features(ds, symbol_data=data)  # 全宇宙 symbol_data 供横截面特征


# ── 状态持久化（JSON）─────────────────────────────────────────────────────────

def load_state(path: str | Path) -> dict[str, WatchState]:
    p = Path(path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {k: WatchState(**v) for k, v in raw.items()}


def save_state(state: dict[str, WatchState], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({k: v.to_dict() for k, v in state.items()}, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 演示 CLI ──────────────────────────────────────────────────────────────────

_CACHE = DATA_DIR / "cache"
_OUT = DATA_DIR / "output" / "top_candidate_research"


def _load_prices_for(keys: set[str]) -> dict[str, pd.DataFrame]:
    dirs = {"US": ["us", "ndx100"], "HK": ["hk"], "CN": ["cn"]}
    out: dict[str, pd.DataFrame] = {}
    for key in keys:
        market, sym = key.split(":", 1)
        variants = [sym]
        if market == "CN":
            variants.append(sym.zfill(6))
        elif market == "HK":
            variants.append(sym.zfill(5))
        for dd in dirs.get(market, []):
            found = None
            for sv in variants:
                p = _CACHE / dd / f"{sv}.parquet"
                if p.exists():
                    found = p
                    break
            if found is not None:
                try:
                    out[key] = _normalize_df(pd.read_parquet(found))
                except Exception:  # noqa: BLE001
                    pass
                break
    return out


def _parse_holdings() -> dict[str, set[str]]:
    hold = {"HK": set(), "US": set(), "CN": set()}
    for ln in (DATA_DIR / "lists" / "holding.md").read_text(encoding="utf-8").splitlines():
        if not ln.strip().startswith("|"):
            continue
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) < 3 or c[0] in ("代码", "") or set(c[0]) <= set("-"):
            continue
        if c[1] in hold:
            hold[c[1]].add(c[0])
    return hold


def _demo() -> None:
    """用现有 unified labeled + cache 价格，回放持仓逃顶信号的全生命周期。

    训练集 = score_asof 较早的历史已定论候选；「今日」= 数据末端；对 [today-45, today-8] 的
    旧信号回放，展示段 B 的 confirm_down / cancel 解析；对最新信号展示 new_signal / alert。
    """
    cfg = SignalConfig()
    lab = pd.read_csv(_OUT / "watchlist_unified_recall_candidates_labeled.csv", low_memory=False)
    lab["sym"] = lab["sym"].astype(str)
    lab["_asof"] = pd.to_datetime(lab["score_asof_date"], errors="coerce")
    hold = _parse_holdings()

    def inhold(m: str, s: str) -> bool:
        return s in hold.get(m, set()) or s.zfill(5) in hold.get(m, set()) or s.zfill(6) in hold.get(m, set())

    today_ts = lab["_asof"].max()
    print(f"逃顶信号状态机 · 演示（today={today_ts.date()}, "
          f"entry={cfg.entry_threshold}, alert={cfg.alert_threshold}）\n")

    # 训练集：today-10 天以前的已定论候选（避免用未来）
    train = lab[lab["_asof"] < today_ts - pd.Timedelta(days=10)]
    hold_mask = lab.apply(lambda r: inhold(str(r["market"]), str(r["sym"])), axis=1)
    train = train[~train.apply(lambda r: inhold(str(r["market"]), str(r["sym"])), axis=1)]  # 持仓不入训练

    # 今日候选：持仓、score_asof 在最近 45 天（既含最新的新信号，也含够老、可被结构解析的旧信号）
    cand = lab[hold_mask & (lab["_asof"] >= today_ts - pd.Timedelta(days=45))].copy()
    keys = {_norm_key(str(m), str(s)) for m, s in zip(cand["market"], cand["sym"], strict=False)}
    prices = _load_prices_for(keys)
    print(f"训练已定论候选 {len(train)}；持仓今日窗口候选 {len(cand)}；载入价格 {len(prices)} 只\n")

    # —— 阶段①：先以「旧信号刚发出」的状态初始化队列（回放：这些信号在过去 8-45 天发出）——
    scored_all = score_discovery(cand, train, cfg)
    scored_all["market"] = scored_all["market"].astype(str)
    scored_all["sym"] = scored_all["sym"].astype(str)
    old = scored_all[(scored_all["_asof"] <= today_ts - pd.Timedelta(days=8)) &
                     (pd.to_numeric(scored_all["strength"], errors="coerce") >= cfg.entry_threshold)]
    seed_state: dict[str, WatchState] = {}
    for (m, s), g in old.groupby(["market", "sym"]):
        r = g.loc[pd.to_numeric(g["strength"], errors="coerce").idxmax()]
        key = _norm_key(m, s)
        top_hi = top_high_at(prices[key], pd.Timestamp(r["top_date"]).strftime("%Y-%m-%d")) if key in prices else np.nan
        seed_state[key] = WatchState(
            market=m, sym=s, state="watching", top_date=pd.Timestamp(r["top_date"]).strftime("%Y-%m-%d"),
            top_high=top_hi, first_signal_date=pd.Timestamp(r["_asof"]).strftime("%Y-%m-%d"),
            last_update=pd.Timestamp(r["_asof"]).strftime("%Y-%m-%d"),
            peak_strength=float(r["strength"]), alerted=bool(r["strength"] >= cfg.alert_threshold),
        )
    print(f"阶段① 队列初始化（过去发出的旧信号）：{len(seed_state)} 条 watching")
    for k, v in sorted(seed_state.items()):
        print(f"    {k:12} 顶{v.top_date} strength={v.peak_strength:.2f} alerted={v.alerted}")

    # —— 阶段②：推进到 today，段 B 结构解析 + 最新新信号 ——
    new_state, events = run_daily(prices, cand, train, seed_state, cfg, today=today_ts)
    print(f"\n阶段② 推进到 today={today_ts.date()} —— 事件 {len(events)} 条：")
    order = {"confirm_down": 0, "cancel": 1, "alert": 2, "new_signal": 3}
    for e in sorted(events, key=lambda x: (order.get(x.kind, 9), x.market, x.sym)):
        print(f"  [{e.kind:12}] {e.market}:{e.sym:8} {e.message}")

    print("\n最终队列状态：")
    for k, v in sorted(new_state.items()):
        tail = f" 距顶{v.drop_pct:.1f}%" if not np.isnan(v.drop_pct) else ""
        print(f"  {k:12} {v.state:14} 顶{v.top_date} peak={v.peak_strength:.2f} alerted={int(v.alerted)}{tail}")

    demo_state_path = _OUT / "escape_signal_state_demo.json"
    save_state(new_state, demo_state_path)
    print(f"\n状态已存 {demo_state_path}（真实使用时由外部每日 load→run_daily→save）")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true", help="用现有 labeled+cache 回放持仓逃顶信号生命周期")
    args = ap.parse_args()
    if args.demo:
        _demo()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
