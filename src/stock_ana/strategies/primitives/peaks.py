"""Reusable structural peak-detection helpers for strategy implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def find_macro_peaks(
	df: pd.DataFrame,
	min_gap_days: int = 65,
	min_drawdown_pct: float = 10.0,
) -> pd.DataFrame:
	"""Scan left-to-right and confirm macro peaks after sufficient drawdown."""
	highs = df["high"].values.astype(float)
	n = len(highs)
	if n < 2:
		return df.iloc[:0].copy()

	confirmed_indices: list[int] = []
	confirmed_drawdowns: list[float] = []

	cand_idx = 0
	cand_val = highs[0]

	for i in range(1, n):
		high = highs[i]
		if high > cand_val:
			cand_idx = i
			cand_val = high
			continue

		drawdown = (cand_val - high) / cand_val * 100.0
		if drawdown >= min_drawdown_pct:
			if confirmed_indices and (cand_idx - confirmed_indices[-1]) < min_gap_days:
				if cand_val > highs[confirmed_indices[-1]]:
					confirmed_indices[-1] = cand_idx
					confirmed_drawdowns[-1] = drawdown
			else:
				confirmed_indices.append(cand_idx)
				confirmed_drawdowns.append(drawdown)
			cand_idx = i
			cand_val = high

	if confirmed_indices:
		last_confirmed_val = highs[confirmed_indices[-1]]
		if cand_val > last_confirmed_val and (cand_idx - confirmed_indices[-1]) >= min_gap_days:
			tail_min = float(np.min(highs[cand_idx:]))
			tail_drawdown = (cand_val - tail_min) / cand_val * 100.0
			confirmed_indices.append(cand_idx)
			confirmed_drawdowns.append(tail_drawdown)

	peaks_df = df.iloc[confirmed_indices].copy()
	peaks_df["drawdown_pct"] = confirmed_drawdowns
	return peaks_df
