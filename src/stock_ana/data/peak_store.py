"""宏观峰值检测结果缓存与持久化工具。

对每只股票在本地存储已计算的峰值列表（JSON），避免重复计算。
提供 get_or_compute_peaks() ：如果缓存有效则直接读取，否则
调用 strategies.primitives.peaks.find_macro_peaks() 重新计算并写回缓存。

Persistence helpers for cached macro peak detection results.
"""

import json
from pathlib import Path

import pandas as pd

from stock_ana.config import CACHE_DIR


PEAKS_DIR = CACHE_DIR / "macro_peaks"
PEAKS_DIR.mkdir(parents=True, exist_ok=True)


def _peaks_cache_path(ticker: str) -> Path:
    """Return the JSON cache path for one symbol's macro peak data."""
    return PEAKS_DIR / f"{ticker}_peaks.json"


def save_peaks(ticker: str, peaks_df: pd.DataFrame) -> None:
    """Serialize and cache macro peak rows for one symbol."""
    records = []
    for idx, row in peaks_df.iterrows():
        records.append(
            {
                "date": str(idx) if isinstance(idx, pd.Timestamp) else str(idx),
                "high": float(row["high"]),
                "drawdown_pct": float(row.get("drawdown_pct", 0.0)),
            }
        )
    _peaks_cache_path(ticker).write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_peaks(ticker: str) -> pd.DataFrame | None:
    """Load cached macro peaks for one symbol."""
    path = _peaks_cache_path(ticker)
    if not path.exists():
        return None
    records = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        return None
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def get_or_compute_peaks(
    ticker: str,
    df: pd.DataFrame,
    *,
    min_gap_days: int = 65,
    min_drawdown_pct: float = 10.0,
    force: bool = False,
) -> pd.DataFrame:
    """Read cached peaks when possible, otherwise compute and persist them."""
    from stock_ana.strategies.primitives.peaks import find_macro_peaks

    if not force:
        cached = load_peaks(ticker)
        if cached is not None:
            return cached
    peaks_df = find_macro_peaks(df, min_gap_days=min_gap_days, min_drawdown_pct=min_drawdown_pct)
    save_peaks(ticker, peaks_df)
    return peaks_df
