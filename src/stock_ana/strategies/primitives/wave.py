"""Reusable wave-structure primitives shared across strategies and research tools."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ana.strategies.primitives.pivots import zigzag_points
from stock_ana.strategies.primitives.vegas_zones import vegas_ema_series


def detect_ema8_swings(
	df: pd.DataFrame,
	min_swing_pct: float = 3.0,
) -> list[dict]:
	"""Detect alternating swing pivots from EMA8-driven zigzag structure."""
	x = df.copy()
	x.columns = [c.lower() for c in x.columns]
	if len(x) < 12:
		return []

	close = x["close"].astype(float)
	ema8 = close.ewm(span=8, adjust=False).mean()

	threshold_pct = max(1.5, min_swing_pct * 0.75)
	raw_pivots = zigzag_points(
		ema8.values.astype(float),
		ema8.values.astype(float),
		threshold_pct=threshold_pct,
	)
	if not raw_pivots:
		return []

	alternating: list[dict] = [raw_pivots[0]]
	for pivot in raw_pivots[1:]:
		prev = alternating[-1]
		if pivot["type"] == prev["type"]:
			if pivot["type"] == "H" and pivot["value"] >= prev["value"]:
				alternating[-1] = pivot
			elif pivot["type"] == "L" and pivot["value"] <= prev["value"]:
				alternating[-1] = pivot
		else:
			alternating.append(pivot)

	monotonic: list[dict] = [alternating[0]]
	for pivot in alternating[1:]:
		prev = monotonic[-1]
		if pivot["iloc"] <= prev["iloc"]:
			if pivot["type"] == "H" and pivot["value"] > prev["value"]:
				monotonic[-1] = pivot
			elif pivot["type"] == "L" and pivot["value"] < prev["value"]:
				monotonic[-1] = pivot
		else:
			monotonic.append(pivot)

	if min_swing_pct > 0 and len(monotonic) >= 3:
		filtered: list[dict] = [monotonic[0]]
		for pivot in monotonic[1:]:
			prev = filtered[-1]
			amplitude = abs(pivot["value"] - prev["value"]) / prev["value"] * 100 if prev["value"] > 0 else 999
			if amplitude < min_swing_pct:
				if pivot["type"] == "H" and pivot["value"] > prev["value"]:
					filtered[-1] = pivot
				elif pivot["type"] == "L" and pivot["value"] < prev["value"]:
					filtered[-1] = pivot
			else:
				filtered.append(pivot)
		monotonic = filtered

	final: list[dict] = [monotonic[0]]
	for pivot in monotonic[1:]:
		prev = final[-1]
		if pivot["type"] == prev["type"]:
			if pivot["type"] == "H" and pivot["value"] >= prev["value"]:
				final[-1] = pivot
			elif pivot["type"] == "L" and pivot["value"] <= prev["value"]:
				final[-1] = pivot
		else:
			if pivot["iloc"] > prev["iloc"]:
				final.append(pivot)
			else:
				if pivot["type"] == "H" and pivot["value"] > prev["value"]:
					final[-1] = pivot
				elif pivot["type"] == "L" and pivot["value"] < prev["value"]:
					final[-1] = pivot

	idx = x.index
	for pivot in final:
		pivot["date"] = str(idx[pivot["iloc"]].date()) if hasattr(idx[pivot["iloc"]], "date") else str(idx[pivot["iloc"]])

	return final


def analyze_wave_structure(
	df: pd.DataFrame,
	swing_window: int = 3,
	swing_min_pct: float = 3.0,
	long_vegas_margin_pct: float = 3.0,
	mid_vegas_margin_pct: float = 2.0,
	ema_warmup_bars: int = 200,
) -> dict:
	"""Build multi-wave structure from EMA8 swings and Vegas support zones.

	Args:
		swing_window: 保留参数（历史兼容），当前 ZigZag 实现不使用。
		ema_warmup_bars: EMA200 暖机期长度。此前的低点不参与 Long Vegas
			触碰判定——ewm(adjust=False) 早期 EMA 贴着价格走，"触碰"无意义。

	浪连续性（wave_number 递增）判定基于 boundary id：相邻两浪在
	refinement 之前共享同一个 Long Vegas 触点，wave["end_boundary_id"] ==
	next_wave["start_boundary_id"] 即为连续交接。synthetic 截断（深破 LV）
	会清空 end_boundary_id，天然打断连续性。
	"""
	x = df.copy()
	x.columns = [c.lower() for c in x.columns]
	if len(x) < 50:
		return {
			"major_waves": [],
			"current_wave_number": 0,
			"current_sub_wave": 0,
			"current_status": "insufficient_data",
			"all_pivots": [],
		}

	close = x["close"].astype(float)
	# 真实 low：LV 触碰判定需要（EMA8 枢轴值平滑滞后，深下影触碰会漏检）
	low_s = x["low"].astype(float) if "low" in x.columns else close
	_emas = vegas_ema_series(close)
	ema34 = _emas[34]
	ema55 = _emas[55]
	ema144 = _emas[144]
	ema169 = _emas[169]
	ema200 = _emas[200]
	# LV 上沿逐 bar 序列（触碰判定 / 刺破检查共用）
	_lv_upper_arr = np.maximum(np.maximum(ema144.values, ema169.values), ema200.values)
	_low_arr = low_s.values.astype(float)

	pivots = detect_ema8_swings(df, min_swing_pct=swing_min_pct)
	if len(pivots) < 4:
		return {
			"major_waves": [],
			"current_wave_number": 0,
			"current_sub_wave": 0,
			"current_status": "insufficient_pivots",
			"all_pivots": pivots,
		}

	def _touches_long_vegas(pivot: dict) -> bool:
		"""L 枢轴是否触及 Long Vegas 上沿（margin 容差内）。

		两个判据取并集：
		  1. EMA8 枢轴值 ≤ LV×(1+margin) —— 原判据，收盘级别的回踩；
		  2. 枢轴附近窗口 [i-7, i+2] 内任一 bar 的真实 low ≤ 当日
		     LV×(1+margin) —— 捕获深下影线触碰。EMA8 在 V 型底处比
		     真实低点高 4~10%，仅用判据 1 会漏掉大量真实触碰
		     （实测华虹 2025-06/2026-04、心动 2024-09/2025-04 均属此类）。
		"""
		i = pivot["iloc"]
		margin = 1 + long_vegas_margin_pct / 100
		upper_at_pivot = float(_lv_upper_arr[i])
		if upper_at_pivot > 0 and pivot["value"] <= upper_at_pivot * margin:
			return True
		w_start = max(0, i - 7)
		w_end = min(len(_low_arr), i + 3)
		lv_win = _lv_upper_arr[w_start:w_end]
		lo_win = _low_arr[w_start:w_end]
		mask = lv_win > 0
		return bool(np.any(lo_win[mask] <= lv_win[mask] * margin))

	all_highs = [pivot for pivot in pivots if pivot["type"] == "H"]
	all_lows = [pivot for pivot in pivots if pivot["type"] == "L"]
	# EMA 暖机期内的"LV 触碰"是伪触碰（早期 EMA200 贴着价格走），不作浪边界
	long_touch_lows = [
		pivot for pivot in pivots
		if pivot["type"] == "L"
		and pivot["iloc"] >= ema_warmup_bars
		and _touches_long_vegas(pivot)
	]

	if not long_touch_lows:
		return {
			"major_waves": [],
			"current_wave_number": 0,
			"current_sub_wave": 0,
			"current_status": "no_long_touch",
			"all_pivots": pivots,
		}

	rally_clear_pct = 15.0
	complete_wave_peak_pct = 40.0
	lv_breach_days = 3  # consecutive bars < LV*0.97 = LV boundary broken

	def _has_lv_breach(from_iloc: int, to_iloc: int) -> bool:
		"""Check if price closed below LV*0.97 for >= lv_breach_days consecutive bars."""
		consec = 0
		for _bi in range(from_iloc, to_iloc + 1):
			_lv = max(float(ema144.iloc[_bi]), float(ema169.iloc[_bi]), float(ema200.iloc[_bi]))
			if _lv > 0 and float(close.iloc[_bi]) < _lv * 0.97:
				consec += 1
				if consec >= lv_breach_days:
					return True
			else:
				consec = 0
		return False

	merged_touches: list[dict] = [long_touch_lows[0]]
	for long_touch in long_touch_lows[1:]:
		prev = merged_touches[-1]
		seg_highs = [high for high in all_highs if prev["iloc"] < high["iloc"] < long_touch["iloc"]]
		if seg_highs:
			peak = max(seg_highs, key=lambda high: high["value"])
			peak_rise_pct = (peak["value"] / prev["value"] - 1) * 100 if prev["value"] > 0 else 0
			end_rise_pct = (long_touch["value"] / prev["value"] - 1) * 100 if prev["value"] > 0 else -999
			is_rising_wave = (
				prev["value"] < long_touch["value"] < peak["value"]
				and end_rise_pct >= rally_clear_pct
			)
			if is_rising_wave:
				merged_touches.append(long_touch)
				continue
			if peak_rise_pct >= complete_wave_peak_pct and peak["value"] > long_touch["value"]:
				merged_touches.append(long_touch)
				continue
		# Before rolling forward, check if LV was breached between the two
		# touches.  If so, they belong to different segments — prev is a real
		# boundary and the current touch starts a new search.
		if _has_lv_breach(prev["iloc"], long_touch["iloc"]):
			merged_touches.append(long_touch)
		else:
			merged_touches[-1] = long_touch

	major_waves: list[dict] = []
	for i in range(len(merged_touches)):
		start_touch = merged_touches[i]
		end_touch = merged_touches[i + 1] if i + 1 < len(merged_touches) else None
		end_iloc = end_touch["iloc"] if end_touch else (len(x) - 1)

		# Boundary id = refinement 之前的原始共享触点 iloc。相邻两浪由
		# 同一个 merged touch 交接：本浪 end_boundary_id == 下一浪
		# start_boundary_id。refinement 会把两侧 pivot 的 iloc 推开，
		# 但 boundary id 保持同源，是浪连续性的唯一判据。
		start_boundary_id = start_touch["iloc"]
		end_boundary_id = end_touch["iloc"] if end_touch else None

		# Refine start: advance to the LAST bar where close <= LV * 1.03
		# before price departs upward.  This gives the true wave launch point.
		_refined_iloc = start_touch["iloc"]
		for _si in range(start_touch["iloc"] + 1, end_iloc):
			_lv_si = max(float(ema144.iloc[_si]), float(ema169.iloc[_si]), float(ema200.iloc[_si]))
			if _lv_si > 0 and float(close.iloc[_si]) <= _lv_si * 1.03:
				_refined_iloc = _si
			elif float(close.iloc[_si]) > _lv_si * 1.10:
				break  # price has clearly departed
		if _refined_iloc != start_touch["iloc"]:
			_rd = str(x.index[_refined_iloc].date()) if hasattr(x.index[_refined_iloc], "date") else str(x.index[_refined_iloc])
			start_touch = {
				"type": "L",
				"iloc": _refined_iloc,
				"value": float(close.iloc[_refined_iloc]),
				"date": _rd,
			}

		# Refine end: same logic — find the LAST bar where close <= LV * 1.03
		# searching backward from end_touch toward the peak.
		if end_touch is not None:
			_ref_end = end_touch["iloc"]
			for _ei in range(end_touch["iloc"] - 1, start_touch["iloc"], -1):
				_lv_ei = max(float(ema144.iloc[_ei]), float(ema169.iloc[_ei]), float(ema200.iloc[_ei]))
				if _lv_ei > 0 and float(close.iloc[_ei]) <= _lv_ei * 1.03:
					_ref_end = _ei
					break
				elif float(close.iloc[_ei]) > _lv_ei * 1.10:
					break  # still well above LV, stop searching
			if _ref_end != end_touch["iloc"]:
				_ed = str(x.index[_ref_end].date()) if hasattr(x.index[_ref_end], "date") else str(x.index[_ref_end])
				end_touch = {
					"type": "L",
					"iloc": _ref_end,
					"value": float(close.iloc[_ref_end]),
					"date": _ed,
				}
			end_iloc = end_touch["iloc"]

		seg_highs = [high for high in all_highs if start_touch["iloc"] <= high["iloc"] <= end_iloc]
		seg_lows = [low for low in all_lows if start_touch["iloc"] <= low["iloc"] <= end_iloc]
		if not seg_highs:
			continue
		peak = max(seg_highs, key=lambda high: high["value"])
		if peak["value"] <= start_touch["value"]:
			continue
		# Long Vegas must be in an uptrend from wave start to peak —
		# reject waves where LV is flat or declining (sideways oscillation).
		_lv_start = max(float(ema144.iloc[start_touch["iloc"]]), float(ema169.iloc[start_touch["iloc"]]), float(ema200.iloc[start_touch["iloc"]]))
		_lv_peak = max(float(ema144.iloc[peak["iloc"]]), float(ema169.iloc[peak["iloc"]]), float(ema200.iloc[peak["iloc"]]))
		if _lv_peak <= _lv_start:
			continue
		_wave = _build_major_wave_v2(
			len(major_waves) + 1,
			start_touch,
			end_touch,
			peak,
			seg_lows,
			seg_highs,
			ema34,
			ema55,
			ema144,
			ema169,
			ema200,
			mid_vegas_margin_pct,
		)
		_wave["start_boundary_id"] = start_boundary_id
		_wave["end_boundary_id"] = end_boundary_id
		major_waves.append(_wave)

	# ── Post-peak sustained LV breach → truncate wave end ────────────────────
	# If after the peak the price stays below Long Vegas for >= N consecutive
	# bars, the wave has definitively ended there — don't let a much-later LV
	# touch extend the wave through years of consolidation/downtrend.
	post_peak_breach_days = 3  # consecutive bars below LV * 0.97 = wave over
	for w in major_waves:
		peak_iloc = w["peak_pivot"]["iloc"]
		end_iloc = w["end_pivot"]["iloc"] if w["end_pivot"] else (len(x) - 1)
		consec = 0
		for _bi in range(peak_iloc + 1, end_iloc + 1):
			_lv = max(float(ema144.iloc[_bi]), float(ema169.iloc[_bi]), float(ema200.iloc[_bi]))
			if _lv > 0 and float(close.iloc[_bi]) < _lv * 0.97:
				consec += 1
				if consec >= post_peak_breach_days:
					_breach_start = _bi - post_peak_breach_days + 1
					_bd = str(x.index[_breach_start].date()) if hasattr(x.index[_breach_start], "date") else str(x.index[_breach_start])
					w["end_pivot"] = {
						"type": "L",
						"iloc": _breach_start,
						"value": float(close.iloc[_breach_start]),
						"date": _bd,
						"synthetic": True,
					}
					w["end_boundary_id"] = None  # 深破截断，浪不再终于共享触点
					w["sub_waves"] = [
						s for s in w["sub_waves"]
						if s["end_pivot"] is not None and s["end_pivot"]["iloc"] <= _breach_start
					]
					w["sub_wave_count"] = len(w["sub_waves"])
					break
			else:
				consec = 0

	# ── Wave filtering & numbering ───────────────────────────────────────────
	# Requirements:
	# 1. Wave boundaries = Long Vegas touches; price must not break below LV
	#    during the wave.
	# 2. Each wave lasts at least ~2 months (min_wave_bars).
	# 3. Each wave has at least min_wave_rise_pct% rise from start to peak.
	# 4. Consecutive waves: 相邻两个「合规浪」之间价格未深破 Long Vegas
	#    （是一段不间断的上升结构），且本浪起点价 >= 前浪起点价。

	min_wave_bars = 40          # ~2 months of trading days
	min_wave_rise_pct = 15.0    # 15% minimum rise from start to peak

	# Step 1: filter out waves that don't qualify
	qualified: list[dict] = []
	for w in major_waves:
		# Duration check: peak must be at least min_wave_bars from start
		duration = w["peak_pivot"]["iloc"] - w["start_pivot"]["iloc"]
		if duration < min_wave_bars:
			continue
		# Rise check
		if w["rise_pct"] < min_wave_rise_pct:
			# Allow the current ongoing wave (no end_pivot) to pass with
			# any positive rise — it's still building.
			if w["end_pivot"] is not None:
				continue
		# Long Vegas integrity: during the wave body (start→peak, after warmup),
		# price must not close below LV*0.97 for 3 consecutive days.
		start_bar = w["start_pivot"]["iloc"]
		peak_bar  = w["peak_pivot"]["iloc"]
		duration = peak_bar - start_bar
		warmup = min(20, duration // 4)
		check_start = start_bar + warmup
		if check_start < peak_bar:
			consec_breach = 0
			failed = False
			for _ci in range(check_start, peak_bar + 1):
				_lv = max(float(ema144.iloc[_ci]), float(ema169.iloc[_ci]), float(ema200.iloc[_ci]))
				if _lv > 0 and float(close.iloc[_ci]) < _lv * 0.97:
					consec_breach += 1
					if consec_breach >= 3:
						failed = True
						break
				else:
					consec_breach = 0
			if failed:
				continue
		qualified.append(w)

	# Step 2: renumber the surviving (qualified) waves by real continuity.
	# 连续 = 相邻两个合规浪之间价格未深破 Long Vegas（≥3 日收盘 < LV*0.97
	# 即断裂）且趋势未破坏（本浪起点价 >= 前浪起点价）。基于「浪间是否深破」
	# 而非 boundary id 相等：中间被过滤掉的弱回踩不会打断编号（不跳号），
	# 而真实断裂处（大缺口 + 深破，如两段独立行情）会正确重置为 W1。
	# 每个浪记录 connected_prev 布尔，供 backward_consecutive_count 等下游
	# 无需再访问 EMA 序列即可回溯连续链。
	for idx, w in enumerate(qualified):
		if idx == 0:
			w["connected_prev"] = False
			w["wave_number"] = 1
			continue
		prev = qualified[idx - 1]
		prev_end = prev["end_pivot"]
		gap_breach = (
			prev_end is None
			or bool(prev_end.get("synthetic"))
			or _has_lv_breach(prev_end["iloc"], w["start_pivot"]["iloc"])
		)
		rising = w["start_pivot"]["value"] >= prev["start_pivot"]["value"]
		connected = (not gap_breach) and rising
		w["connected_prev"] = connected
		w["wave_number"] = prev["wave_number"] + 1 if connected else 1

	major_waves = qualified
	curr_wave_num = major_waves[-1]["wave_number"] if major_waves else 0
	curr_sub = 0
	curr_status = "unknown"

	if major_waves:
		last_wave = major_waves[-1]
		curr_sub = len(last_wave["sub_waves"])
		last_close = float(close.iloc[-1])
		last_ema169 = float(ema169.iloc[-1])
		last_ema144 = float(ema144.iloc[-1])
		last_ema200 = float(ema200.iloc[-1])
		last_ema55 = float(ema55.iloc[-1])
		long_upper = max(last_ema144, last_ema169, last_ema200)

		if last_wave["end_pivot"] is not None:
			end_iloc = last_wave["end_pivot"]["iloc"]
			post_highs = [high for high in all_highs if high["iloc"] > end_iloc]
			if post_highs:
				post_peak = max(post_highs, key=lambda high: high["value"])
				if post_peak["value"] > last_wave["end_pivot"]["value"]:
					curr_status = "rising_from_long"
				else:
					curr_status = "consolidating"
			else:
				curr_status = "ended"
		elif long_upper > 0 and last_close < long_upper * 0.97:
			curr_status = "broken"
			# Synthesize end_pivot at the first bar after the peak where close
			# fell below Long Vegas (97% threshold) — the wave has definitively ended.
			peak_iloc = last_wave["peak_pivot"]["iloc"]
			for _bi in range(peak_iloc + 1, len(close)):
				_lv = max(float(ema144.iloc[_bi]), float(ema169.iloc[_bi]), float(ema200.iloc[_bi]))
				if _lv > 0 and float(close.iloc[_bi]) < _lv * 0.97:
					_bd = str(x.index[_bi].date()) if hasattr(x.index[_bi], "date") else str(x.index[_bi])
					last_wave["end_pivot"] = {
						"type": "L",
						"iloc": _bi,
						"value": float(close.iloc[_bi]),
						"date": _bd,
						"synthetic": True,
					}
					last_wave["end_boundary_id"] = None  # 深破截断
					break
		elif long_upper > 0 and last_close <= long_upper * (1 + long_vegas_margin_pct / 100):
			curr_status = "long_pullback"
		elif last_ema55 > 0 and last_close <= last_ema55 * (1 + mid_vegas_margin_pct / 100):
			curr_status = "mid_pullback"
		else:
			curr_status = "rising"
	else:
		curr_status = "no_rally"

	return {
		"major_waves": major_waves,
		"current_wave_number": curr_wave_num,
		"current_sub_wave": curr_sub,
		"current_status": curr_status,
		"all_pivots": pivots,
	}


def _build_major_wave_v2(
	wave_number: int,
	start_pivot: dict,
	end_pivot: dict | None,
	peak_pivot: dict,
	wave_lows: list[dict],
	wave_highs: list[dict],
	ema34: pd.Series,
	ema55: pd.Series,
	ema144: pd.Series,
	ema169: pd.Series,
	ema200: pd.Series,
	mid_margin: float,
	min_sub_wave_rise: float = 5.0,
) -> dict:
	"""Build one major wave bounded by long-Vegas touch pivots.

	Args:
		min_sub_wave_rise: Minimum % rise from sub_start to seg_peak required to
			count a trough as a real sub-wave pullback.  Troughs that follow a
			sideways consolidation (peak rise < this threshold) are ignored,
			preventing noise in slow-grinding or range-bound waves from inflating
			the sub_wave / sub_number count.
	"""

	def _touches_mid(pivot: dict) -> bool:
		e55 = float(ema55.iloc[pivot["iloc"]])
		return e55 > 0 and pivot["value"] <= e55 * (1 + mid_margin / 100)

	def _mid_band(pivot: dict) -> str:
		e34 = float(ema34.iloc[pivot["iloc"]])
		e55 = float(ema55.iloc[pivot["iloc"]])
		bands = {"ema34": e34, "ema55": e55}
		return min(bands, key=lambda name: abs(pivot["value"] - bands[name]))

	def _long_band(pivot: dict) -> str:
		e144 = float(ema144.iloc[pivot["iloc"]])
		e169 = float(ema169.iloc[pivot["iloc"]])
		e200 = float(ema200.iloc[pivot["iloc"]])
		bands = {"ema144": e144, "ema169": e169, "ema200": e200}
		return min(bands, key=lambda name: abs(pivot["value"] - bands[name]))

	wave_end_iloc = end_pivot["iloc"] if end_pivot else (wave_highs[-1]["iloc"] + 9999 if wave_highs else 9999)
	relevant_highs = [high for high in wave_highs if start_pivot["iloc"] <= high["iloc"] <= wave_end_iloc]

	sub_waves: list[dict] = []
	sub_start = start_pivot
	sub_number = 0

	for wave_low in wave_lows:
		if wave_low["iloc"] <= sub_start["iloc"]:
			continue
		if end_pivot is not None and wave_low["iloc"] == end_pivot["iloc"]:
			continue
		if not _touches_mid(wave_low):
			continue
		segment_highs = [high for high in relevant_highs if sub_start["iloc"] <= high["iloc"] <= wave_low["iloc"]]
		if segment_highs:
			seg_peak = max(segment_highs, key=lambda high: high["value"])
			# Require a meaningful rise from sub_start to the segment peak.
			# If the peak barely exceeds sub_start (sideways consolidation),
			# don't count this trough as a new sub-wave — just update sub_start.
			rise_pct = (seg_peak["value"] / sub_start["value"] - 1) * 100 if sub_start["value"] > 0 else 0
			if rise_pct < min_sub_wave_rise:
				# Absorb this trough into the current sub-wave without incrementing
				sub_start = wave_low
				continue
			sub_number += 1
			sub_waves.append(
				{
					"sub_number": sub_number,
					"start_pivot": sub_start,
					"peak_pivot": seg_peak,
					"end_pivot": wave_low,
					"pullback_type": "mid_vegas",
					"pullback_band": _mid_band(wave_low),
					"rise_pct": round((seg_peak["value"] / sub_start["value"] - 1) * 100, 2) if sub_start["value"] > 0 else 0,
				}
			)
			sub_start = wave_low

	tail_highs = [high for high in relevant_highs if high["iloc"] >= sub_start["iloc"]]
	if tail_highs:
		tail_peak = max(tail_highs, key=lambda high: high["value"])
		sub_number += 1
		sub_waves.append(
			{
				"sub_number": sub_number,
				"start_pivot": sub_start,
				"peak_pivot": tail_peak,
				"end_pivot": end_pivot,
				"pullback_type": "long_vegas" if end_pivot is not None else "none",
				"pullback_band": _long_band(end_pivot) if end_pivot is not None else "",
				"rise_pct": round((tail_peak["value"] / sub_start["value"] - 1) * 100, 2) if sub_start["value"] > 0 else 0,
			}
		)

	rise_pct = round((peak_pivot["value"] / start_pivot["value"] - 1) * 100, 2) if start_pivot["value"] > 0 else 0

	return {
		"wave_number": wave_number,
		"start_pivot": start_pivot,
		"end_pivot": end_pivot,
		"peak_pivot": peak_pivot,
		"sub_waves": sub_waves,
		"sub_wave_count": len(sub_waves),
		"mid_pullback_count": sum(1 for sub_wave in sub_waves if sub_wave["pullback_type"] == "mid_vegas"),
		"rise_pct": rise_pct,
		"duration_days": peak_pivot["iloc"] - start_pivot["iloc"],
	}
