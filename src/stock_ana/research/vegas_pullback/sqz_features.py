"""LazyBear Squeeze Momentum（SQZMOM_LB）特征：as-of 读取指标缓存。

数据来源：data/cache/indicators/{us|hk|cn}/{symbol}.parquet
（见 docs/squeeze_momentum_lazybear.md；指标本身只用当日及此前数据，因果，
且有"修改未来价格不改变过去指标"的测试）。

按文档 §6 的纪律：不用未归一的 sqzmom_value 原值跨股票比较——特征全部
用方向 / 变化率 / 持续天数 / 自归一形态（120 日滚动 std 归一）：

  sqz_mom_z          : sqzmom_value / rolling_std(value, 120)（自归一动量幅度，
                       正=多头动量，跨股票可比）
  sqz_mom_slope5_z   : 近 5 日动量变化 / 同一 std（动量加速度——回踩中
                       动量在衰竭还是在加深，文档 §5.3 的柱体变化预警）
  sqz_bar_state      : 四态编码 2/1/-1/-2（正/负动量 × 增强/减弱）
  sqz_squeeze_state  : 1=压缩 0=过渡 -1=释放
  sqz_on_days        : 连续 squeeze-on 天数（蓄势时长；释放前的弹簧长度）
  sqz_since_zero_cross : 距上次动量过零轴的 bar 数（capped 120；动量 regime
                       年龄，配合符号由 bar_state 提供）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ana.config import CACHE_DIR

SQZ_FEATURES: tuple[str, ...] = (
    "sqz_mom_z", "sqz_mom_slope5_z", "sqz_bar_state",
    "sqz_squeeze_state", "sqz_on_days", "sqz_since_zero_cross",
)

_MAX_STALE_DAYS = 7
_Z_WIN = 120


def _dir_of(market: str) -> str:
    return {"US": "us", "CN": "cn", "HK": "hk"}.get(str(market).upper(), "")


def add_sqz_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """按 (market, sym, score_asof_date) 附加 SQZMOM 特征（as-of，零前瞻）。"""
    out = dataset.copy()
    for c in SQZ_FEATURES:
        if c not in out.columns:
            out[c] = np.nan
    if out.empty:
        return out

    asof = pd.to_datetime(out.get("score_asof_date", out.get("signal_date")), errors="coerce")
    cache: dict[tuple[str, str], pd.DataFrame | None] = {}

    def load(mk: str, sym: str) -> pd.DataFrame | None:
        key = (mk, sym)
        if key not in cache:
            d = _dir_of(mk)
            p = CACHE_DIR / "indicators" / d / f"{sym}.parquet"
            if d and p.exists():
                try:
                    df = pd.read_parquet(p)
                    df.index = pd.to_datetime(df.index)
                    cache[key] = df.sort_index()
                except Exception:
                    cache[key] = None
            else:
                cache[key] = None
        return cache[key]

    vals: dict[str, list] = {c: [] for c in SQZ_FEATURES}
    for mk, sym, d in zip(out["market"].astype(str), out["sym"].astype(str), asof, strict=False):
        ind = load(mk, sym)
        z = slope = bar = st = on_days = zcross = np.nan
        if ind is not None and pd.notna(d) and "sqzmom_value" in ind.columns:
            pos = ind.index.searchsorted(d, side="right") - 1
            if pos >= 0 and (d - ind.index[pos]).days <= _MAX_STALE_DAYS:
                v = pd.to_numeric(ind["sqzmom_value"].iloc[max(0, pos - _Z_WIN) : pos + 1], errors="coerce")
                sd = float(v.std())
                v_now = float(v.iloc[-1]) if pd.notna(v.iloc[-1]) else np.nan
                if sd > 0 and pd.notna(v_now):
                    z = v_now / sd
                    if len(v) > 5 and pd.notna(v.iloc[-6]):
                        slope = (v_now - float(v.iloc[-6])) / sd
                bar = pd.to_numeric(ind["sqzmom_bar_state"].iloc[pos], errors="coerce")
                st = pd.to_numeric(ind["sqzmom_squeeze_state"].iloc[pos], errors="coerce")
                # 连续 squeeze-on 天数
                on = ind["sqzmom_squeeze_on"].iloc[: pos + 1].astype(bool).values
                k = 0
                while k < len(on) and on[len(on) - 1 - k]:
                    k += 1
                on_days = k
                # 距上次动量过零轴 bar 数
                sgn = np.sign(pd.to_numeric(ind["sqzmom_value"].iloc[max(0, pos - _Z_WIN) : pos + 1], errors="coerce").values)
                zc = np.where(np.diff(sgn) != 0)[0]
                zcross = int(len(sgn) - 2 - zc[-1]) if zc.size else _Z_WIN
        vals["sqz_mom_z"].append(z)
        vals["sqz_mom_slope5_z"].append(slope)
        vals["sqz_bar_state"].append(bar)
        vals["sqz_squeeze_state"].append(st)
        vals["sqz_on_days"].append(on_days)
        vals["sqz_since_zero_cross"].append(zcross)

    for c in SQZ_FEATURES:
        out[c] = np.round(pd.to_numeric(pd.Series(vals[c], index=out.index), errors="coerce"), 3)
    return out
