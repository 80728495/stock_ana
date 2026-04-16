"""Typed contracts shared by strategy screen and scan entrypoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd


StrategyKind = Literal["pattern", "stateful_signal"]


@dataclass
class StrategyDecision:
    """Normalized single-symbol decision returned by screen_* interfaces."""

    passed: bool
    strategy_kind: StrategyKind = "stateful_signal"
    score: float = 0.0
    setup_type: str = ""
    trigger_date: pd.Timestamp | None = None
    stop_hint: float | None = None
    invalidation: str | None = None
    reason: str | None = None
    features: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Mirror strategy kind into metadata so downstream consumers see one shape."""
        self.meta.setdefault("strategy_kind", self.strategy_kind)


@dataclass
class ScanHit:
    """One symbol hit in a scan result."""

    symbol: str
    decision: StrategyDecision


@dataclass
class ScanResult:
    """Normalized batch result returned by scan_* interfaces."""

    strategy: str
    market: str
    strategy_kind: StrategyKind | None = None
    hits: list[ScanHit] = field(default_factory=list)
    total: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    params_snapshot: dict[str, Any] = field(default_factory=dict)

    def as_dataframe(self) -> pd.DataFrame:
        """Convert standardized scan result to a flat DataFrame."""
        rows: list[dict[str, Any]] = []
        for hit in self.hits:
            row = {
                "symbol": hit.symbol,
                "passed": hit.decision.passed,
                "strategy_kind": hit.decision.strategy_kind,
                "score": hit.decision.score,
                "setup_type": hit.decision.setup_type,
                "trigger_date": hit.decision.trigger_date,
                "stop_hint": hit.decision.stop_hint,
                "invalidation": hit.decision.invalidation,
                "reason": hit.decision.reason,
            }
            row.update(hit.decision.features)
            rows.append(row)
        return pd.DataFrame(rows)
