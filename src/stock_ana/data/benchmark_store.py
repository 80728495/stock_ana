"""Canonical benchmark registry and OpenD-backed daily price store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from stock_ana.config import CACHE_DIR

BENCHMARK_DIR = CACHE_DIR / "benchmarks"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    futu_code: str
    name: str
    home_market: str
    currency: str
    adjusted: bool = False


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "US_QQQ": BenchmarkSpec("US_QQQ", "US.QQQ", "NASDAQ-100 ETF", "US", "USD", adjusted=True),
    "CN_CHINEXT": BenchmarkSpec("CN_CHINEXT", "SZ.399006", "创业板指", "CN", "CNY"),
    "CN_STAR_COMPOSITE": BenchmarkSpec("CN_STAR_COMPOSITE", "SH.000680", "科创综指", "CN", "CNY"),
    "CN_STAR50": BenchmarkSpec("CN_STAR50", "SH.000688", "科创50", "CN", "CNY"),
    "HK_HSI": BenchmarkSpec("HK_HSI", "HK.800000", "恒生指数", "HK", "HKD"),
    "HK_HSTECH": BenchmarkSpec("HK_HSTECH", "HK.800700", "恒生科技指数", "HK", "HKD"),
}


def _benchmark_path(benchmark_id: str) -> Path:
    if benchmark_id not in BENCHMARKS:
        raise KeyError(f"未知 benchmark: {benchmark_id}")
    return BENCHMARK_DIR / f"{benchmark_id}.parquet"


def load_benchmark(benchmark_id: str) -> pd.DataFrame | None:
    path = _benchmark_path(benchmark_id)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.sort_index()


def load_all_benchmarks() -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for benchmark_id in BENCHMARKS:
        df = load_benchmark(benchmark_id)
        if df is not None and not df.empty:
            result[benchmark_id] = df
    return result


def _save_compatibility_copy(spec: BenchmarkSpec, df: pd.DataFrame) -> None:
    """Keep legacy RS readers working while the canonical store is adopted."""
    if spec.benchmark_id == "US_QQQ":
        from stock_ana.data.fetcher import save_local_data

        save_local_data("QQQ", df)
    elif spec.benchmark_id in {"HK_HSI", "HK_HSTECH"}:
        from stock_ana.data.fetcher_hk import save_hk_local

        save_hk_local(spec.futu_code.split(".", 1)[1], df)


def save_benchmark(benchmark_id: str, df: pd.DataFrame) -> None:
    spec = BENCHMARKS[benchmark_id]
    clean = df.copy()
    clean.index = pd.to_datetime(clean.index).normalize()
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()
    clean.index.name = "date"
    clean.to_parquet(_benchmark_path(benchmark_id))
    _save_compatibility_copy(spec, clean)


def update_benchmarks_futu(
    *,
    force: bool = False,
    max_stale_days: int = 1,
    history_start: str = "2020-01-01",
) -> dict[str, object]:
    """Refresh all registered beta benchmarks through one OpenD context."""
    from stock_ana.data.fetcher_futu import (
        fetch_futu_kline_with_ctx,
        quote_context,
    )

    today = pd.Timestamp.now().normalize()
    end_date = today.strftime("%Y-%m-%d")
    updated = 0
    skipped = 0
    failed = 0
    details: list[dict[str, object]] = []
    with quote_context() as ctx:
        for benchmark_id, spec in BENCHMARKS.items():
            local = load_benchmark(benchmark_id)
            if force or local is None or local.empty:
                start_date = history_start
                mode = "full"
            else:
                last_date = pd.Timestamp(local.index.max()).normalize()
                if (today - last_date).days <= max_stale_days:
                    skipped += 1
                    details.append({
                        "benchmark_id": benchmark_id,
                        "status": "skipped",
                        "last_date": last_date.date().isoformat(),
                    })
                    continue
                start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                mode = "incremental"

            try:
                fresh = fetch_futu_kline_with_ctx(
                    ctx,
                    spec.futu_code,
                    start_date,
                    end_date,
                    adjusted=spec.adjusted,
                )
                if fresh.empty:
                    current = local
                    status = "no_new_data"
                elif local is not None and not local.empty and not force:
                    current = pd.concat([local, fresh])
                    current = current[~current.index.duplicated(keep="last")].sort_index()
                    save_benchmark(benchmark_id, current)
                    updated += 1
                    status = "updated"
                else:
                    current = fresh
                    save_benchmark(benchmark_id, current)
                    updated += 1
                    status = "updated"

                last = (
                    None
                    if current is None or current.empty
                    else pd.Timestamp(current.index.max()).date().isoformat()
                )
                details.append({
                    "benchmark_id": benchmark_id,
                    "futu_code": spec.futu_code,
                    "mode": mode,
                    "status": status,
                    "last_date": last,
                })
                logger.info(f"Benchmark {benchmark_id} ({spec.futu_code}): {status}, latest={last}")
            except Exception as exc:
                failed += 1
                details.append({
                    "benchmark_id": benchmark_id,
                    "futu_code": spec.futu_code,
                    "mode": mode,
                    "status": "failed",
                    "error": str(exc),
                })
                logger.error(f"Benchmark {benchmark_id} ({spec.futu_code}) 更新失败: {exc}")

    return {
        "updated": updated,
        "skipped": skipped,
        "failed": failed,
        "total": len(BENCHMARKS),
        "details": details,
    }
