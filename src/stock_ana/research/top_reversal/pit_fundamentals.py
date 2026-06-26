"""Point-in-time (as-of) 基本面取数：估值 PE/PB/PS + 增长 YoY，杜绝快照。

因果纪律：以候选 ``score_asof_date`` 为基准，只用「报告期末 + 保守披露滞后 ≤ asof」的最近一期
（US/CN 季度优先、HK 仅年报），绝不用当季/当年（未披露=偷看未来）。今天的实时候选 asof=今天
→ 取最新已披露一期 = 与快照等价，实时打分不受影响。

数据源（temp_scripts/fetch_pit_fundamentals.py 落盘）：
  US  stockanalysis.com 季度+年度  pit/US/{tk}__sa_{q,y}.parquet
      （datekey/epsDiluted/revenue/sharesDiluted + 逐期 pe/ps/pb/lastClosePrice；同源、日期可靠）
      ⚠️ 不用 akshare 的 NOTICE_DATE：历史季度被系统性后移~1年（实测中位 401 天），不可靠。
  CN  akshare  pit/CN/{sym}__val.parquet（日频 PE(TTM)/市净率/市销率，现成 PIT）
               + pit/CN/{sym}__fin.parquet（季度累计宽表，算 TTM YoY）
  HK  akshare  pit/HK/{sym}__fin.parquet（年报 BASIC_EPS/BPS/同比）+ 价格 parquet 重构 PE/PB（无 PS）

增长口径统一 **TTM YoY**（最近 4 季滚动同比；HK 仅年报→年度 YoY，年末等价 TTM）。
估值取「当时价 × 当时已披露 TTM 基本面」重构（捕捉顶部时点的高估值），CN 直接用日频现成 PIT。
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from stock_ana.config import DATA_DIR

PIT_DIR = DATA_DIR / "cache" / "fundamentals" / "pit"
_CACHE = DATA_DIR / "cache"

# 报告期末 → 披露可得 的保守滞后（宁可偏晚，不偷看未来）
US_Q_LAG = pd.Timedelta(days=60)    # 10-Q ≤40-45 天，60 保守
US_Y_LAG = pd.Timedelta(days=90)    # 10-K ≤60-75 天
HK_Y_LAG = pd.Timedelta(days=120)   # 港股年报披露较晚
_CN_LAG_DAYS = {"0331": 45, "0630": 75, "0930": 45, "1231": 125}

_NAN3 = (np.nan, np.nan, np.nan)
_NAN2 = (np.nan, np.nan)


@functools.lru_cache(maxsize=4096)
def _read(path_str: str) -> pd.DataFrame | None:
    from pathlib import Path
    p = Path(path_str)
    return pd.read_parquet(p) if p.exists() else None


@functools.lru_cache(maxsize=4096)
def _close_series(market: str, sym: str) -> pd.Series | None:
    """规范化收盘价序列（用于 US/HK 估值重构的 as-of 价）。"""
    from stock_ana.research.top_reversal.smc_context import _normalize_df
    dirs = {"US": ["us", "ndx100"], "HK": ["hk"], "CN": ["cn"]}.get(market, [])
    for d in dirs:
        fp = _CACHE / d / f"{sym}.parquet"
        if fp.exists():
            try:
                df = _normalize_df(pd.read_parquet(fp))
                s = pd.to_numeric(df["close"], errors="coerce").dropna()
                s.index = pd.to_datetime(s.index)
                return s[~s.index.duplicated(keep="last")].sort_index()
            except Exception:  # noqa: BLE001
                return None
    return None


def _price_asof(market: str, sym: str, asof: pd.Timestamp) -> float | None:
    s = _close_series(market, sym)
    if s is None or s.empty:
        return None
    i = s.index.searchsorted(asof, side="right") - 1
    return float(s.iloc[i]) if i >= 0 and s.iloc[i] > 0 else None


def _yoy(now: float, prior: float) -> float:
    """同比 %：(now-prior)/|prior|。prior=0 或缺失 → nan。负基数用绝对值，方向仍正确。"""
    if pd.isna(now) or pd.isna(prior) or prior == 0:
        return np.nan
    return (now - prior) / abs(prior) * 100.0


# ── US: stockanalysis 季度/年度 ──────────────────────────────────────────────

def _us_quarters(sym: str, asof: pd.Timestamp) -> pd.DataFrame | None:
    """已披露（datekey+60 ≤ asof）的季度序列，按期升序。"""
    d = _read(str(PIT_DIR / "US" / f"{sym}__sa_q.parquet"))
    if d is None or "datekey" not in d.columns:
        return None
    out = d.copy()
    out["_dt"] = pd.to_datetime(out["datekey"], errors="coerce")
    out = out[out["_dt"].notna() & (out["_dt"] + US_Q_LAG <= asof)].sort_values("_dt")
    return out if len(out) else None


def _us_growth(sym: str, asof: pd.Timestamp) -> tuple[float, float]:
    """US TTM YoY 增长（eps/营收）：最近 4 季 vs 再前 4 季。季度不足→年报年度 YoY。"""
    q = _us_quarters(sym, asof)
    if q is not None and len(q) >= 8:
        eps = pd.to_numeric(q["epsDiluted"], errors="coerce").to_numpy()
        rev = pd.to_numeric(q["revenue"], errors="coerce").to_numpy()
        eg = _yoy(np.nansum(eps[-4:]), np.nansum(eps[-8:-4]))
        rg = _yoy(np.nansum(rev[-4:]), np.nansum(rev[-8:-4]))
        if not (pd.isna(eg) and pd.isna(rg)):
            return eg, rg
    # 年报回退（早于季度窗口的老候选）
    y = _read(str(PIT_DIR / "US" / f"{sym}__sa_y.parquet"))
    if y is not None and "datekey" in y.columns:
        yy = y.copy()
        yy["_dt"] = pd.to_datetime(yy["datekey"], errors="coerce")
        yy = yy[yy["_dt"].notna() & (yy["_dt"] + US_Y_LAG <= asof)].sort_values("_dt")
        if len(yy) >= 2:
            eps = pd.to_numeric(yy["epsDiluted"], errors="coerce").to_numpy()
            rev = pd.to_numeric(yy["revenue"], errors="coerce").to_numpy()
            return _yoy(eps[-1], eps[-2]), _yoy(rev[-1], rev[-2])
    return _NAN2


def _us_valuation(sym: str, asof: pd.Timestamp) -> tuple[float, float, float]:
    """US 估值：as-of 价 × 已披露 TTM 基本面重构（捕捉顶部高估值）。pe/ps 必出，pb 需账面价值。"""
    q = _us_quarters(sym, asof)
    px = _price_asof("US", sym, asof)
    if q is None or len(q) < 4 or px is None:
        return _NAN3
    eps = pd.to_numeric(q["epsDiluted"], errors="coerce").to_numpy()
    rev = pd.to_numeric(q["revenue"], errors="coerce").to_numpy()
    sh = pd.to_numeric(q["sharesDiluted"], errors="coerce").to_numpy()
    ttm_eps = float(np.nansum(eps[-4:]))
    ttm_rev = float(np.nansum(rev[-4:]))
    sh_last = next((float(x) for x in sh[::-1] if pd.notna(x) and x > 0), np.nan)
    pe = px / ttm_eps if ttm_eps > 0 else np.nan
    ps = (px * sh_last) / ttm_rev if (pd.notna(sh_last) and ttm_rev > 0) else np.nan
    # PB：用最近一期 pb 反推每股账面 BVPS = lastClose_q / pb_q，再用 as-of 价
    pb = np.nan
    if {"pb", "lastClosePrice"}.issubset(q.columns):
        last = q.iloc[-1]
        pb_q = pd.to_numeric(last.get("pb"), errors="coerce")
        lc_q = pd.to_numeric(last.get("lastClosePrice"), errors="coerce")
        if pd.notna(pb_q) and pb_q > 0 and pd.notna(lc_q) and lc_q > 0:
            bvps = lc_q / pb_q
            pb = px / bvps if bvps > 0 else np.nan
    return (round(pe, 2) if pd.notna(pe) else np.nan,
            round(pb, 2) if pd.notna(pb) else np.nan,
            round(ps, 2) if pd.notna(ps) else np.nan)


# ── HK: akshare 年报 + 价格重构 ──────────────────────────────────────────────

def _hk_annual(sym: str, asof: pd.Timestamp) -> pd.Series | None:
    d = _read(str(PIT_DIR / "HK" / f"{sym}__fin.parquet"))
    if d is None or "REPORT_DATE" not in d.columns:
        return None
    rd = pd.to_datetime(d["REPORT_DATE"], errors="coerce")
    a = d[rd.notna() & (rd + HK_Y_LAG <= asof)].copy()
    a["_rd"] = rd[rd.notna() & (rd + HK_Y_LAG <= asof)]
    a = a.sort_values("_rd")
    return a.iloc[-1] if len(a) else None


def _hk_report_asof(sym: str, asof: pd.Timestamp) -> pd.Series | None:
    """HK 年报(__fin, 深历史) + 季度报告期(__fin_q, 近~2年) 拼接，取公告日≤asof 的最近一期。

    公告滞后：年报(1231) 120 天、其余季度/中期 90 天。近段季度更新、远段退年度。
    """
    frames = []
    for kind in ("fin", "fin_q"):
        d = _read(str(PIT_DIR / "HK" / f"{sym}__{kind}.parquet"))
        if d is not None and "REPORT_DATE" in d.columns:
            frames.append(d)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    rd = pd.to_datetime(d["REPORT_DATE"], errors="coerce")
    d = d[rd.notna()].copy()
    d["_rd"] = rd[rd.notna()]
    lag = d["_rd"].dt.strftime("%m%d").map(lambda m: 120 if m == "1231" else 90)
    d = d[d["_rd"] + pd.to_timedelta(lag, unit="D") <= asof]
    if d.empty:
        return None
    d = d.sort_values("_rd").drop_duplicates("_rd", keep="last")  # 同期年报+季度去重
    return d.iloc[-1]


def _hk_growth(sym: str, asof: pd.Timestamp) -> tuple[float, float]:
    r = _hk_report_asof(sym, asof)  # 年报+季度拼接，YoY 用季度可得则季度
    if r is None:
        return _NAN2
    return (pd.to_numeric(r.get("HOLDER_PROFIT_YOY"), errors="coerce"),
            pd.to_numeric(r.get("OPERATE_INCOME_YOY"), errors="coerce"))


def _hk_valuation(sym: str, asof: pd.Timestamp) -> tuple[float, float, float]:
    r = _hk_annual(sym, asof)
    px = _price_asof("HK", sym, asof)
    if r is None or px is None:
        return _NAN3
    eps = pd.to_numeric(r.get("BASIC_EPS"), errors="coerce")
    bps = pd.to_numeric(r.get("BPS"), errors="coerce")
    pe = px / eps if (pd.notna(eps) and eps > 0) else np.nan
    pb = px / bps if (pd.notna(bps) and bps > 0) else np.nan
    return (round(pe, 2) if pd.notna(pe) else np.nan,
            round(pb, 2) if pd.notna(pb) else np.nan, np.nan)  # HK 无 PS


# ── CN: akshare 日频估值(PIT) + 季度累计宽表(TTM YoY) ─────────────────────────

def _cn_valuation(sym: str, asof: pd.Timestamp) -> tuple[float, float, float]:
    d = _read(str(PIT_DIR / "CN" / f"{sym}__val.parquet"))
    if d is None or "数据日期" not in d.columns:
        return _NAN3
    dt = pd.to_datetime(d["数据日期"], errors="coerce")
    m = dt.notna() & (dt <= asof)
    if not m.any():
        return _NAN3
    r = d[m].iloc[-1]
    return (pd.to_numeric(r.get("PE(TTM)"), errors="coerce"),
            pd.to_numeric(r.get("市净率"), errors="coerce"),
            pd.to_numeric(r.get("市销率"), errors="coerce"))


def _cn_growth(sym: str, asof: pd.Timestamp) -> tuple[float, float]:
    """CN TTM YoY：累计宽表用 TTM=FY(y-1)+cum(期)−cum(y-1同期) 还原，再同比。"""
    d = _read(str(PIT_DIR / "CN" / f"{sym}__fin.parquet"))
    if d is None or "指标" not in d.columns:
        return _NAN2
    periods = [c for c in d.columns if str(c).isdigit() and len(str(c)) == 8]

    def avail(p: str) -> bool:
        lag = _CN_LAG_DAYS.get(str(p)[4:], 100)
        return pd.Timestamp(str(p)) + pd.Timedelta(days=lag) <= asof

    ok = sorted([p for p in periods if avail(p)], reverse=True)
    if not ok:
        return _NAN2
    cur = ok[0]
    rows = {n: d[d["指标"].astype(str) == n].iloc[0] if (d["指标"].astype(str) == n).any() else None
            for n in ("归母净利润", "营业总收入")}

    def val(name: str, period: str) -> float:
        r = rows.get(name)
        return pd.to_numeric(r.get(period), errors="coerce") if r is not None and period in d.columns else np.nan

    def ttm(name: str, period: str) -> float:
        y, mmdd = int(str(period)[:4]), str(period)[4:]
        if mmdd == "1231":
            return val(name, period)
        fy_prev = val(name, f"{y-1}1231")
        same_prev = val(name, f"{y-1}{mmdd}")
        cum = val(name, period)
        if pd.isna(fy_prev) or pd.isna(same_prev) or pd.isna(cum):
            return np.nan
        return fy_prev + cum - same_prev

    def yoy_ttm(name: str) -> float:
        y, mmdd = int(str(cur)[:4]), str(cur)[4:]
        return _yoy(ttm(name, cur), ttm(name, f"{y-1}{mmdd}"))

    return yoy_ttm("归母净利润"), yoy_ttm("营业总收入")


# ── 对外统一入口 ─────────────────────────────────────────────────────────────

def _norm_sym(market: str, sym: str) -> str:
    if market == "HK":
        return str(sym).zfill(5)
    if market == "CN":
        return str(sym).zfill(6)
    return str(sym).upper()


def pit_valuation(market: str, sym: str, asof: pd.Timestamp) -> tuple[float, float, float]:
    """as-of (pe, pb, ps)。缺失→nan。"""
    if pd.isna(asof):
        return _NAN3
    s = _norm_sym(market, sym)
    if market == "US":
        return _us_valuation(s, asof)
    if market == "HK":
        return _hk_valuation(s, asof)
    if market == "CN":
        return _cn_valuation(s, asof)
    return _NAN3


def pit_growth(market: str, sym: str, asof: pd.Timestamp) -> tuple[float, float]:
    """as-of (earnings_growth, revenue_growth) %，TTM YoY 口径。缺失→nan。"""
    if pd.isna(asof):
        return _NAN2
    s = _norm_sym(market, sym)
    if market == "US":
        return _us_growth(s, asof)
    if market == "HK":
        return _hk_growth(s, asof)
    if market == "CN":
        return _cn_growth(s, asof)
    return _NAN2
