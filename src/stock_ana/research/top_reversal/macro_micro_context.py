"""Macro (sector/cross-sectional) and micro (parabolic/over-extension) features.

Pure per-bar quantitative features are weak for tops, which are macro and noisy.
These features add two missing dimensions, all strictly causal (only data up to
the candidate's ``score_asof`` bar):

Macro
  * sector regime — is the stock in a hot/melt-up sector (e.g. semiconductors)
  * cross-sectional rank — how extreme is the stock's run vs the whole universe

Micro
  * parabolic / acceleration — how vertical and unsustainable the recent rise is
  * over-extension — distance above short EMA / consecutive up streak
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from stock_ana.config import DATA_DIR

MACRO_MICRO_FEATURES: tuple[str, ...] = (
    # micro
    "micro_accel_20_60",
    "micro_verticality_atr",
    "micro_dist_ema20_atr",
    "micro_up_streak",
    # macro cross-sectional
    "xsec_ret20_pct",
    "xsec_ret60_pct",
    # macro sector
    "is_semiconductor",
    "sector_peers_ret60_mean",
    "sector_overheat_pct",
)

_DEFAULTS: dict[str, object] = {c: np.nan for c in MACRO_MICRO_FEATURES}
_DEFAULTS["is_semiconductor"] = 0
_DEFAULTS["micro_up_streak"] = 0


# ── 板块映射 ──────────────────────────────────────────────────────────────────

def _build_sector_map() -> dict[tuple[str, str], dict]:
    smap: dict[tuple[str, str], dict] = {}
    us_path = DATA_DIR / "us_sec_profiles.csv"
    if us_path.exists():
        us = pd.read_csv(us_path)
        for _, r in us.iterrows():
            t = str(r.get("ticker", "")).strip().upper()
            if not t:
                continue
            sic = str(r.get("sic_description", "")).lower()
            # 二级分类：用 SIC 3 位主组（如 367=电子元件/半导体）做细分板块，
            # 比粗 GICS sector（Technology 把半导体+软件混在一起）更能刻画子行业 regime。
            sic_code = r.get("sic_code")
            if pd.notna(sic_code) and float(sic_code) > 0:
                sub_sector = f"US_SIC{int(float(sic_code)) // 10}"
            else:
                sub_sector = "US_" + (str(r.get("sector", "")).strip() or "other")
            smap[("US", t)] = {"sector": sub_sector, "is_semi": int("semiconductor" in sic)}
    hk_path = DATA_DIR / "cache" / "hk_industry_map.csv"
    if hk_path.exists():
        hk = pd.read_csv(hk_path)
        code_col = "futu_code" if "futu_code" in hk.columns else hk.columns[0]
        for _, r in hk.iterrows():
            code = str(r.get(code_col, "")).strip()
            if not code.startswith("HK."):
                continue
            sym = code.split(".", 1)[1].zfill(5)
            ind = str(r.get("industry", "")).strip() or "HK_other"
            is_semi = int(("半导体" in ind) or ("芯片" in ind))
            smap[("HK", sym)] = {"sector": f"HK_{ind}", "is_semi": is_semi}
    return smap


# ── 微观结构（单股因果） ───────────────────────────────────────────────────────

def _atr14(df: pd.DataFrame) -> np.ndarray:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=5).mean().bfill().to_numpy()


def _micro_features(df: pd.DataFrame, pos: int, atr: np.ndarray, ema20: np.ndarray) -> dict:
    out = {}
    close = df["close"].astype(float).to_numpy()
    n = len(df)
    a = float(atr[pos]) if pos < len(atr) and np.isfinite(atr[pos]) and atr[pos] > 0 else np.nan

    def ret(k):
        return (close[pos] / close[pos - k] - 1) * 100 if pos - k >= 0 and close[pos - k] > 0 else np.nan

    r20, r60 = ret(20), ret(60)
    # 加速度：近20日涨幅 - 按60日均速折算的20日涨幅（>0 表示近期在加速）
    out["micro_accel_20_60"] = round(float(r20 - r60 * (20.0 / 60.0)), 2) if np.isfinite(r20) and np.isfinite(r60) else np.nan
    # 垂直度：收盘价相对自身30日线性趋势的偏离（ATR 单位）——越高越像 blow-off 顶
    if pos >= 30 and np.isfinite(a):
        y = close[pos - 29: pos + 1]
        x = np.arange(len(y))
        coef = np.polyfit(x, y, 1)
        fit = coef[0] * x[-1] + coef[1]
        out["micro_verticality_atr"] = round(float((close[pos] - fit) / a), 2)
    else:
        out["micro_verticality_atr"] = np.nan
    # 相对 EMA20 的过度延展（ATR 单位）
    e = float(ema20[pos]) if pos < len(ema20) and np.isfinite(ema20[pos]) else np.nan
    out["micro_dist_ema20_atr"] = round(float((close[pos] - e) / a), 2) if np.isfinite(e) and np.isfinite(a) else np.nan
    # 连阳 streak（上限 20）
    streak = 0
    for i in range(pos, max(0, pos - 20), -1):
        if i >= 1 and close[i] > close[i - 1]:
            streak += 1
        else:
            break
    out["micro_up_streak"] = int(streak)
    return out


# ── 跨截面 + 板块面板 ─────────────────────────────────────────────────────────

def _build_return_panel(symbol_data: Mapping[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """构建 date×key 的 ret20 / ret60 面板（key=f"{market}:{symbol}"）。"""
    ret20_series: dict[str, pd.Series] = {}
    ret60_series: dict[str, pd.Series] = {}
    key_market: dict[str, str] = {}
    for item in symbol_data.values():
        mk = str(item.get("market", ""))
        sym = str(item.get("symbol", item.get("sym", "")))
        df = item.get("df")
        if not mk or not sym or not isinstance(df, pd.DataFrame) or "close" not in df.columns:
            continue
        c = df["close"].astype(float)
        idx = pd.to_datetime(df.index)
        key = f"{mk}:{sym}"
        ret20_series[key] = pd.Series(((c / c.shift(20) - 1) * 100).to_numpy(), index=idx)
        ret60_series[key] = pd.Series(((c / c.shift(60) - 1) * 100).to_numpy(), index=idx)
        key_market[key] = mk
    if not ret60_series:
        return pd.DataFrame(), pd.DataFrame(), key_market
    p20 = pd.DataFrame(ret20_series).sort_index().ffill()
    p60 = pd.DataFrame(ret60_series).sort_index().ffill()
    return p20, p60, key_market


def add_macro_micro_features(
    dataset: pd.DataFrame,
    symbol_data: Mapping[str, dict] | None = None,
) -> pd.DataFrame:
    """Attach macro (sector/cross-sectional) and micro (parabolic) features."""

    out = dataset.copy()
    for col in MACRO_MICRO_FEATURES:
        if col not in out.columns:
            out[col] = _DEFAULTS[col]
    if out.empty or not symbol_data:
        return out

    data_map: dict[tuple[str, str], pd.DataFrame] = {}
    for item in symbol_data.values():
        mk = str(item.get("market", ""))
        sym = str(item.get("symbol", item.get("sym", "")))
        df = item.get("df")
        if mk and sym and isinstance(df, pd.DataFrame):
            d = df.copy()
            d.columns = [str(c).lower() for c in d.columns]
            d.index = pd.to_datetime(d.index)
            data_map[(mk, sym)] = d.sort_index()

    sector_map = _build_sector_map()
    p20, p60, _ = _build_return_panel(symbol_data)
    panel_idx = p60.index if not p60.empty else pd.DatetimeIndex([])
    # 列 -> 板块
    col_sector = {}
    for key in (p60.columns if not p60.empty else []):
        mk, sym = key.split(":", 1)
        col_sector[key] = sector_map.get((mk, sym), {}).get("sector", f"{mk}_other")

    # 每个评分日的截面缓存
    xsec_cache: dict[pd.Timestamp, dict] = {}

    def xsec_at(ts: pd.Timestamp) -> dict:
        if ts in xsec_cache:
            return xsec_cache[ts]
        res = {"r20": None, "r60": None, "sector_mean": {}, "sector_rank": {}}
        if not panel_idx.empty:
            loc = panel_idx.searchsorted(ts, side="right") - 1
            if loc >= 0:
                row60 = p60.iloc[loc]
                row20 = p20.iloc[loc] if not p20.empty else None
                res["r60"] = row60
                res["r20"] = row20
                # 板块均值
                sec_means = {}
                by_sec: dict[str, list] = {}
                for key, sec in col_sector.items():
                    by_sec.setdefault(sec, []).append(key)
                for sec, keys in by_sec.items():
                    vals = pd.to_numeric(row60[keys], errors="coerce").dropna()
                    if len(vals) >= 2:
                        sec_means[sec] = float(vals.mean())
                res["sector_mean"] = sec_means
                if sec_means:
                    sm = pd.Series(sec_means)
                    ranks = sm.rank(pct=True)
                    res["sector_rank"] = ranks.to_dict()
        xsec_cache[ts] = res
        return res

    atr_cache: dict[tuple[str, str], tuple] = {}
    rows: list[dict] = []
    for _, row in out.iterrows():
        mk = str(row.get("market", ""))
        sym = str(row.get("sym", ""))
        key = f"{mk}:{sym}"
        vals = dict(_DEFAULTS)
        df = data_map.get((mk, sym))
        # 板块标签
        is_semi = int(sector_map.get((mk, sym), {}).get("is_semi", 0))
        sec = sector_map.get((mk, sym), {}).get("sector", f"{mk}_other")
        vals["is_semiconductor"] = is_semi

        asof = row.get("score_asof_date", row.get("confirm_date"))
        ts = pd.to_datetime(asof, errors="coerce")
        # 微观
        if df is not None:
            pos = int(row.get("score_asof_pos", row.get("top_pos", -1)) or -1)
            pos = max(0, min(len(df) - 1, pos)) if pos >= 0 else -1
            if pos >= 0:
                if (mk, sym) not in atr_cache:
                    atr_cache[(mk, sym)] = (_atr14(df), df["close"].astype(float).ewm(span=20, adjust=False).mean().to_numpy())
                atr, ema20 = atr_cache[(mk, sym)]
                vals.update(_micro_features(df, pos, atr, ema20))
        # 宏观截面
        if pd.notna(ts):
            xs = xsec_at(ts)
            if xs["r60"] is not None and key in xs["r60"].index:
                r60_cross = pd.to_numeric(xs["r60"], errors="coerce").dropna()
                v60 = xs["r60"].get(key, np.nan)
                if np.isfinite(v60) and len(r60_cross) >= 5:
                    vals["xsec_ret60_pct"] = round(float((r60_cross < v60).mean() * 100), 1)
            if xs["r20"] is not None and key in xs["r20"].index:
                r20_cross = pd.to_numeric(xs["r20"], errors="coerce").dropna()
                v20 = xs["r20"].get(key, np.nan)
                if np.isfinite(v20) and len(r20_cross) >= 5:
                    vals["xsec_ret20_pct"] = round(float((r20_cross < v20).mean() * 100), 1)
            if sec in xs["sector_mean"]:
                vals["sector_peers_ret60_mean"] = round(float(xs["sector_mean"][sec]), 2)
            if sec in xs["sector_rank"]:
                vals["sector_overheat_pct"] = round(float(xs["sector_rank"][sec] * 100), 1)
        rows.append({c: vals[c] for c in MACRO_MICRO_FEATURES})

    fdf = pd.DataFrame(rows, index=out.index)
    for c in MACRO_MICRO_FEATURES:
        out[c] = fdf[c]
    return out
