"""行业内 as-of 估值分位（无市场回退）+ 行业月度估值面板取数。

设计纪律（用户要求）：
  * 估值乘数的归一化只在 **行业内** 排名才有意义；跨全行业(市场级)排名 = 绝对值的单调变换、无意义，
    **绝不回退市场**。行业内凑不齐同侪 → 置 NaN，由绝对值特征兜底。
  * 严格 **as-of**：把候选估值排进「该(市场,行业)在 ≤当月最近一个≥MIN_PEERS同侪的月」的成员分布。
    候选集内「跨期池化」排名会用到未来候选 = 未来函数泄漏（实测 CN 0.83→0.71），故弃用。
  * 行业层级：SIC3(sector)；US 再加 SIC2(sector2) 作更粗的行业层（仍在行业内，不是市场）。

口径（候选与成员同源）：CN=stock_value_em 日频 PE(TTM)/市净率/市销率；US=stockanalysis 逐季 pe/ps/pb；
  HK=Futu 月末价 × 年报 EPS/BPS 重构 PE/PB（无 PS；已用 Futu 快照验证 PB corr=1.00、PE 0.905）。
面板由 temp_scripts/build_sector_valuation_panel.py 落盘到 sector_valuation_members.parquet。
"""

from __future__ import annotations

import functools
from collections import defaultdict

import numpy as np
import pandas as pd

from stock_ana.config import DATA_DIR

PIT = DATA_DIR / "cache" / "fundamentals" / "pit"
PANEL_PATH = DATA_DIR / "sector_valuation_members.parquet"  # 永久研究数据，入 git（见 .gitignore 白名单）
MIN_PEERS = 8
MONTH_START = "2017-01"
_HK_LAG = pd.Timedelta(days=120)


def _clean(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    return v.where((v > 0) & (v < 1000))


# ── 成员/候选 月度估值（与候选 PIT 同源，保证可比）─────────────────────────────

@functools.lru_cache(maxsize=8192)
def cn_monthly(sym: str) -> pd.DataFrame | None:
    p = PIT / "CN" / f"{sym}__val.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    if "数据日期" not in d.columns:
        return None
    d = d.assign(_dt=pd.to_datetime(d["数据日期"], errors="coerce")).dropna(subset=["_dt"]).set_index("_dt").sort_index()
    return pd.DataFrame({"pe": _clean(d.get("PE(TTM)")), "pb": _clean(d.get("市净率")),
                         "ps": _clean(d.get("市销率"))}).resample("ME").last()


@functools.lru_cache(maxsize=8192)
def us_monthly(sym: str) -> pd.DataFrame | None:
    p = PIT / "US" / f"{sym}__sa_q.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    if "datekey" not in d.columns:
        return None
    d = d.assign(_dt=pd.to_datetime(d["datekey"], errors="coerce")).dropna(subset=["_dt"]).set_index("_dt").sort_index()
    q = pd.DataFrame({"pe": _clean(d.get("pe")), "pb": _clean(d.get("pb")), "ps": _clean(d.get("ps"))})
    return q.resample("ME").last().ffill()


@functools.lru_cache(maxsize=8192)
def hk_monthly(sym: str) -> pd.DataFrame | None:
    pp = DATA_DIR / "cache" / "hk" / f"{sym}.parquet"
    fp = PIT / "HK" / f"{sym}__fin.parquet"
    if not (pp.exists() and fp.exists()):
        return None
    from stock_ana.research.top_reversal.smc_context import _normalize_df
    px = pd.to_numeric(_normalize_df(pd.read_parquet(pp))["close"], errors="coerce")
    px.index = pd.to_datetime(px.index)
    pxm = px.resample("ME").last().dropna()
    if pxm.empty:
        return None
    fin = pd.read_parquet(fp)
    if "REPORT_DATE" not in fin.columns:
        return None
    rd = pd.to_datetime(fin["REPORT_DATE"], errors="coerce")
    f = pd.DataFrame({"_avail": rd + _HK_LAG, "eps": pd.to_numeric(fin.get("BASIC_EPS"), errors="coerce"),
                      "bps": pd.to_numeric(fin.get("BPS"), errors="coerce")}).dropna(subset=["_avail"]).sort_values("_avail")
    if f.empty:
        return None
    left = pd.DataFrame({"month": pxm.index, "price": pxm.to_numpy()})
    m = pd.merge_asof(left, f, left_on="month", right_on="_avail", direction="backward")
    out = pd.DataFrame({"pe": (m["price"] / m["eps"].where(m["eps"] > 0)).to_numpy(),
                        "pb": (m["price"] / m["bps"].where(m["bps"] > 0)).to_numpy(), "ps": np.nan},
                       index=pd.DatetimeIndex(m["month"]))
    return out.apply(_clean)


_MFN = {"CN": cn_monthly, "US": us_monthly, "HK": hk_monthly}


def candidate_value(market: str, sym: str, asof: pd.Timestamp) -> dict[str, float]:
    """候选在 asof 当月的 panel-一致 (pe,pb,ps)。"""
    fn = _MFN.get(market)
    if fn is None or pd.isna(asof):
        return {"pe": np.nan, "pb": np.nan, "ps": np.nan}
    mm = fn(str(sym))
    if mm is None or mm.empty:
        return {"pe": np.nan, "pb": np.nan, "ps": np.nan}
    row = mm[mm.index <= pd.Timestamp(asof) + pd.offsets.MonthEnd(0)]
    if row.empty:
        return {"pe": np.nan, "pb": np.nan, "ps": np.nan}
    last = row.iloc[-1]
    return {"pe": last.get("pe", np.nan), "pb": last.get("pb", np.nan), "ps": last.get("ps", np.nan)}


# ── 行业月度成员分布（只 SIC3 / US-SIC2，绝不建市场级）─────────────────────────

@functools.lru_cache(maxsize=1)
def _levels() -> dict | None:
    if not PANEL_PATH.exists():
        return None
    panel = pd.read_parquet(PANEL_PATH)
    panel["month"] = panel["month"].astype(str)
    levels: dict = {}
    for metric in ("pe", "pb", "ps"):
        sub = panel.dropna(subset=[metric])
        for lvl, scol in (("sec", "sector"), ("sec2", "sector2")):
            s = sub.dropna(subset=[scol])
            d: dict = defaultdict(list)
            for (mkt, key, mo), v in s.groupby(["market", scol, "month"])[metric]:
                d[(mkt, key)].append((mo, np.sort(v.to_numpy())))
            for kk in d:
                d[kk].sort(key=lambda x: x[0])
            levels[(metric, lvl)] = dict(d)
    return levels


def sector_pct(market: str, sector, sector2, asof: pd.Timestamp, metric: str, value: float) -> float:
    """value 在「(市场,行业) ≤asof 最近一个≥MIN_PEERS同侪的月」分布中的分位(0-100)。

    层级 SIC3→(US)SIC2，行业内凑不齐 → NaN（绝不回退市场）。
    """
    levels = _levels()
    if levels is None or pd.isna(value) or value <= 0 or pd.isna(asof):
        return np.nan
    month = pd.Timestamp(asof).strftime("%Y-%m")
    for lvl, key in (("sec", (market, sector)), ("sec2", (market, sector2))):
        if lvl == "sec2" and not isinstance(sector2, str):
            continue
        if not isinstance(sector, str) and lvl == "sec":
            continue
        seq = levels.get((metric, lvl), {}).get(key)
        if not seq:
            continue
        for mo, arr in reversed(seq):
            if mo <= month and len(arr) >= MIN_PEERS:
                return float(np.searchsorted(arr, value, side="right") / len(arr) * 100)
    return np.nan
