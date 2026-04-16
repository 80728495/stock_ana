"""Small regression helpers used by geometric pattern strategies."""

from __future__ import annotations

import numpy as np


def ols_fit(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float]:
    """Return slope, intercept, r2, and max residual pct for a simple OLS fit."""
    if len(xs) < 2:
        return 0.0, 0.0, 0.0, float("inf")
    slope, intercept = np.polyfit(xs, ys, 1)
    y_hat = slope * xs + intercept
    residuals = ys - y_hat
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    y_mean = float(ys.mean())
    max_residual_pct = float(np.max(np.abs(residuals))) / y_mean * 100 if y_mean > 0 else float("inf")
    return float(slope), float(intercept), r2, max_residual_pct


def line_value(slope: float, intercept: float, x: float | np.ndarray) -> float | np.ndarray:
    """Evaluate a line at x."""
    return slope * x + intercept
