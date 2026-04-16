"""Low-level plotting implementations — the single place where matplotlib/mplfinance
is called.  All scan and backtest modules delegate here; none of them import
matplotlib or mplfinance directly.

Font config is set once at module-import time so every chart produced by this
project uses a consistent CJK-capable font stack without per-call repetition.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplfinance as mpf
import numpy as np
import pandas as pd
from loguru import logger

# ── Canonical font config (applied once at import time) ──────────────────────
plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang HK", "STHeiti", "Songti SC",
                                    "Arial Unicode MS", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ── Shared style constants ────────────────────────────────────────────────────
SIGNAL_STYLE = {
    "STRONG_BUY": {"marker": "^", "color": "#00CC00", "edge": "#006400", "size": 200},
    "BUY":        {"marker": "^", "color": "#66DD66", "edge": "#228B22", "size": 160},
    "HOLD":       {"marker": "o", "color": "#FFD700", "edge": "#B8860B", "size": 100},
    "AVOID":      {"marker": "v", "color": "#FF6666", "edge": "#CC0000", "size": 120},
}

_EMA_SPANS = {
    "EMA34": 34, "EMA55": 55, "EMA60": 60,
    "EMA144": 144, "EMA169": 169, "EMA200": 200,
}

_EMA_COLORS = {
    "EMA34":  "#4FC3F7",
    "EMA55":  "#29B6F6",
    "EMA60":  "#0288D1",
    "EMA144": "#FF8A65",
    "EMA169": "#FF7043",
    "EMA200": "#E64A19",
}

# CJK font probe (runtime, cached)
_CJK_FONTS = ["PingFang HK", "Heiti TC", "Arial Unicode MS"]


@lru_cache(maxsize=1)
def _find_cjk_font() -> str | None:
    """Return the first available CJK font name, or None."""
    from matplotlib.font_manager import fontManager  # lazy import — slow first call
    available = {f.name for f in fontManager.ttflist}
    for f in _CJK_FONTS:
        if f in available:
            return f
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Vegas Mid-Vegas scan chart
# ═════════════════════════════════════════════════════════════════════════════

def plot_vegas_mid_scan_chart(
    sym: str,
    market: str,
    name: str,
    df_price: pd.DataFrame,
    signal_info: dict,
    out_dir: Path,
    context_bars: int = 252,
    name_en: str = "",
    sector: str = "",
    industry: str = "",
    biz_summary: str = "",
) -> Path | None:
    """Render a focused Vegas Mid-Vegas scan signal chart.

    Args:
        sym: Ticker symbol.
        market: "US" or "HK".
        name: Display name (may be Chinese).
        df_price: Full OHLCV DataFrame.
        signal_info: Dict returned by scan_one (entry_date, signal, score, …).
        out_dir: Directory to write the PNG to.
        context_bars: How many trailing bars to show.
        name_en: ASCII name used for the filename (falls back to sym).

    Returns:
        Path to the saved PNG, or None on failure.
    """
    df = df_price.copy()
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    df = df.sort_index()[["open", "high", "low", "close", "volume"]]

    # Build view window centred around the entry date so historical backtest
    # charts always show the correct period rather than the most-recent bars.
    entry_date_str = signal_info["entry_date"].split("(")[0]
    entry_date = pd.Timestamp(entry_date_str)
    post_bars = 20  # bars to show after the signal
    entry_pos = df.index.searchsorted(entry_date, side="left")
    if entry_pos >= len(df):
        entry_pos = len(df) - 1
    view_end = min(entry_pos + post_bars + 1, len(df))
    view_start = max(view_end - context_bars, 0)
    df_view = df.iloc[view_start:view_end].copy()
    if len(df_view) < 20:
        return None

    emas_full = {
        f"EMA{s}": df["close"].ewm(span=s, adjust=False).mean()
        for s in [34, 55, 60, 144, 169, 200]
    }
    add_plots = []
    for ema_name, ema_s in emas_full.items():
        ema_view = ema_s.reindex(df_view.index)
        add_plots.append(mpf.make_addplot(
            ema_view, color=_EMA_COLORS[ema_name], width=1.2,
            linestyle="-" if any(x in ema_name for x in ["144", "169", "200"]) else "--",
        ))

    sig_type = signal_info["signal"]
    style = SIGNAL_STYLE.get(sig_type, SIGNAL_STYLE["BUY"])

    marker_series = pd.Series(np.nan, index=df_view.index)
    # entry_pos within the view
    idx_pos = df_view.index.searchsorted(entry_date, side="left")
    if idx_pos >= len(df_view):
        idx_pos = len(df_view) - 1
    marker_series.iloc[idx_pos] = float(df_view.iloc[idx_pos]["low"]) * 0.97

    if marker_series.notna().any():
        add_plots.append(mpf.make_addplot(
            marker_series, type="scatter", markersize=style["size"],
            marker=style["marker"], color=style["color"],
            edgecolors=style["edge"], linewidths=1.5,
        ))

    mc = mpf.make_marketcolors(
        up="#CC3333", down="#00AA00", edge="inherit", wick="inherit", volume="in",
    )
    mpf_style = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=":", gridcolor="#E0E0E0",
        rc={"font.sans-serif": plt.rcParams["font.sans-serif"],
            "axes.unicode_minus": False},
    )

    score = signal_info["score"]
    sig_color = {
        "STRONG_BUY": "#006400", "BUY": "#228B22",
        "HOLD": "#B8860B", "AVOID": "#CC0000",
    }.get(sig_type, "#333333")

    fig, axes = mpf.plot(
        df_view, type="candle", volume=True, style=mpf_style,
        addplot=add_plots, figsize=(20, 9), returnfig=True,
        tight_layout=False,
        scale_padding={"left": 0.05, "right": 0.38, "top": 0.6, "bottom": 0.5},
    )
    ax = axes[0]

    # ── Warmup boundary: gray line where detection becomes active ──────────
    # The detector skips the first 200 bars (EMA warm-up). Any signal whose
    # touch_bar falls before that date is historical context only — the
    # strategy was not live for those bars.  When bar-200 falls inside the
    # view window we draw a dashed gray divider so the reader knows the dates
    # to its left were never evaluated by the algorithm.
    _DETECTOR_WARMUP = 200
    if len(df) > _DETECTOR_WARMUP:
        _warmup_date = df.index[_DETECTOR_WARMUP]
        if df_view.index[0] <= _warmup_date <= df_view.index[-1]:
            _wpos = int(df_view.index.searchsorted(_warmup_date, side="left"))
            ax.axvline(_wpos - 0.5, color="#AAAAAA", linewidth=1.4,
                       linestyle="--", alpha=0.85, zorder=1)
            ax.text(
                _wpos - 0.6, 0.99, "← 策略检测起点",
                transform=ax.get_xaxis_transform(),
                fontsize=7, color="#888888", va="top", ha="right",
                rotation=90, style="italic",
            )
            ax.text(
                _wpos + 0.4, 0.99, "策略有效 →",
                transform=ax.get_xaxis_transform(),
                fontsize=7, color="#888888", va="top", ha="left",
                rotation=90, style="italic",
            )

    fig.suptitle(
        f"{market}:{sym}  {name}    [{sig_type}  score={score:+d}]",
        fontsize=17, fontweight="bold", y=0.98, color=sig_color,
    )
    ax.set_title(
        f"{signal_info['support_band']}  @ {signal_info['entry_date']}",
        fontsize=11, color="#555555", pad=4,
    )

    if idx_pos < len(df_view):
        low_val = float(df_view.iloc[idx_pos]["low"])
        sig_short = {"STRONG_BUY": "SB", "BUY": "B", "HOLD": "H", "AVOID": "A"}[sig_type]
        ax.annotate(f"{sig_short}{score:+d}", xy=(idx_pos, low_val * 0.94),
                    fontsize=10, fontweight="bold", color=style["edge"],
                    ha="center", va="top")

    # ── Detailed info panel (right side) ─────────────────────────────────────
    def _yn(v) -> str:
        return "[Y]" if v else "[N]"

    si = signal_info
    wave_number   = si.get("wave_number", 0)
    sub_number    = si.get("sub_number", 0)
    touch_seq     = si.get("touch_seq", 0)
    wave_rise     = si.get("wave_rise_pct", 0.0)
    consec        = si.get("consec_waves", 0)
    gap_pct       = si.get("mid_long_gap_pct", 0.0)
    slope_pct     = si.get("long_slope_pct", 0.0)
    orderly       = si.get("orderly_pullback", False)
    struct_passed = si.get("structure_passed", False)

    # score factor labels
    _factor_labels = [
        ("factor_mkt",        f"市场({market})"),
        ("factor_sub_pos",    f"回踩位置(第{sub_number}次)"),
        ("factor_wave_rise",  f"涨幅({wave_rise:+.0f}%)"),
        ("factor_three_wave", f"连续升浪({consec}浪)"),
        ("factor_wave_num",   f"第{wave_number}浪"),
        ("factor_ml_gap",     f"Mid/Long间距({gap_pct:.0f}%)"),
        ("factor_orderly",    "有序回踩"),
    ]
    score_lines = []
    for key, label in _factor_labels:
        val = si.get(key, 0)
        score_lines.append(f"  {label:<18} {val:+d}")

    _struct_label = "[Y] 通过" if struct_passed else "[N] 未通过 -> AVOID"
    panel_lines = [
        "--- 波浪结构 ---",
        f"  第{wave_number}浪  第{sub_number}次回踩  连续{consec}浪",
        f"  起点涨幅: {wave_rise:+.1f}%  回踩序号: #{touch_seq}",
        "",
        "--- 结构条件 ---",
        f"  Mid > Long:     {_yn(si.get('mid_above_long'))}",
        f"  价格 > Long:    {_yn(si.get('price_above_long'))}",
        f"  价格(3M)>Long:  {_yn(si.get('price_above_long_3m'))}",
        f"  Long上升:       {_yn(si.get('long_rising'))}",
        f"  Mid/Long间距:   {gap_pct:.1f}%  {_yn(si.get('gap_enough'))}(5~25%)",
        f"  Long斜率:       {slope_pct:.2f}%  {_yn(si.get('long_slope_strong'))}(>=2%)",
        f"  有序回踩:       {_yn(orderly)}",
        f"  浪内回踩序号:   #{touch_seq}  {_yn(si.get('touch_seq_ok'))}(<=3)",
        "",
        "--- 评分明细 ---",
    ] + score_lines + [
        "  " + "-" * 22,
        f"  总分: {score:+d} -> {sig_type}",
        f"  结构: {_struct_label}",
    ]

    # 行业 / 主营业务
    if sector or industry or biz_summary:
        panel_lines.append("")
        panel_lines.append("--- 公司信息 ---")
        if sector:
            panel_lines.append(f"  板块: {sector}")
        if industry:
            panel_lines.append(f"  行业: {industry}")
        if biz_summary:
            # 截取前200字并按约40字换行
            _summary = biz_summary[:200].replace("\n", " ")
            import textwrap
            for _line in textwrap.wrap(_summary, width=40):
                panel_lines.append(f"  {_line}")

    panel_text = "\n".join(panel_lines)
    ax.text(
        1.01, 0.99, panel_text,
        transform=ax.transAxes,
        fontsize=7.5, ha="left", va="top",
        fontfamily=plt.rcParams["font.sans-serif"][0],
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFDE7", alpha=0.92,
                  edgecolor="#CCCC88", linewidth=0.8),
    )

    legend_elements = []
    for ema_name in ["EMA34", "EMA60", "EMA144", "EMA200"]:
        legend_elements.append(
            Line2D([0], [0], color=_EMA_COLORS[ema_name], linewidth=1.5, label=ema_name)
        )
    legend_elements.append(
        Line2D([0], [0], marker=style["marker"], color="w",
               markerfacecolor=style["color"], markeredgecolor=style["edge"],
               markersize=10, label=sig_type)
    )
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.8)

    out_dir.mkdir(parents=True, exist_ok=True)
    # For HK/CN: include Chinese name directly in filename (macOS supports Unicode paths).
    # For US: keep the ASCII name_en approach.
    _UNSAFE = set('/\\:*?"<>|')
    if market in ("HK", "CN"):
        safe_name = "".join(c for c in name if c not in _UNSAFE).strip()[:20] or sym
        if name_en:
            safe_name = name_en[:20] + "_" + safe_name
    else:
        _name_for_file = name_en or name
        safe_name = "".join(
            c if c.isascii() and (c.isalnum() or c in "-_") else "" for c in _name_for_file
        ).strip("-_") or sym
        safe_name = safe_name[:30]
    out_path = out_dir / f"{sig_type}_{market}_{sym}_{safe_name}_{entry_date_str}.png"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Glyph.*missing from font")
        fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_stock_all_signals_chart(
    sym: str,
    market: str,
    name: str,
    df_price: pd.DataFrame,
    signals: list[dict],
    out_dir: Path,
    lead_bars: int = 60,
    trail_bars: int = 40,
) -> Path | None:
    """Render one chart per stock showing ALL backtest signals on the same canvas.

    Each signal is marked with a styled arrow/dot at its entry bar.  Different
    signal types (STRONG_BUY / BUY / HOLD / AVOID) get different colours and
    shapes.  Touch-strategy signals are annotated with a 'T' suffix.

    Args:
        sym: Ticker symbol.
        market: "US" or "HK".
        name: Human-readable display name (may contain CJK).
        df_price: Full OHLCV DataFrame.
        signals: List of signal dicts (each must have entry_date, signal, score,
                 support_band, touch_strategy).
        out_dir: Output directory — chart saved as ``{market}_{sym}.png``.
        lead_bars: Bars of context before the earliest signal.
        trail_bars: Bars of context after the latest signal.

    Returns:
        Path to the saved PNG, or None on failure.
    """
    if not signals:
        return None

    df = df_price.copy()
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    df = df.sort_index()[["open", "high", "low", "close", "volume"]]

    # Determine view window spanning all signals
    entry_dates = []
    for sig in signals:
        try:
            entry_dates.append(pd.Timestamp(sig["entry_date"].split("(")[0]))
        except Exception:
            pass
    if not entry_dates:
        return None

    first_pos = df.index.searchsorted(min(entry_dates), side="left")
    last_pos  = df.index.searchsorted(max(entry_dates), side="left")
    view_start = max(first_pos - lead_bars, 0)
    view_end   = min(last_pos + trail_bars + 1, len(df))
    df_view = df.iloc[view_start:view_end].copy()
    if len(df_view) < 20:
        return None

    # EMAs computed on full df, reindexed to the view window
    emas_full = {
        f"EMA{s}": df["close"].ewm(span=s, adjust=False).mean()
        for s in [34, 55, 60, 144, 169, 200]
    }
    add_plots = []
    for ema_name, ema_s in emas_full.items():
        add_plots.append(mpf.make_addplot(
            ema_s.reindex(df_view.index),
            color=_EMA_COLORS[ema_name],
            width=1.0,
            linestyle="-" if any(x in ema_name for x in ["144", "169", "200"]) else "--",
        ))

    # One scatter series per signal type (so each type gets its own style)
    sig_type_series: dict[str, pd.Series] = {
        t: pd.Series(np.nan, index=df_view.index) for t in SIGNAL_STYLE
    }
    valid_sigs: list[tuple[int, dict]] = []   # (iloc, sig)
    for sig in signals:
        try:
            ed = pd.Timestamp(sig["entry_date"].split("(")[0])
        except Exception:
            continue
        idx_pos = df_view.index.searchsorted(ed, side="left")
        if idx_pos >= len(df_view):
            continue
        sig_type = sig.get("signal", "HOLD")
        if sig_type not in sig_type_series:
            sig_type = "HOLD"
        low_val = float(df_view.iloc[idx_pos]["low"])
        # Place marker slightly below the low; avoid duplicate at same bar by
        # taking min (place furthest-down marker).
        existing = sig_type_series[sig_type].iloc[idx_pos]
        candidate = low_val * 0.965
        if np.isnan(existing) or candidate < existing:
            sig_type_series[sig_type].iloc[idx_pos] = candidate
        valid_sigs.append((idx_pos, sig))

    for sig_type, series in sig_type_series.items():
        if series.notna().any():
            style = SIGNAL_STYLE[sig_type]
            add_plots.append(mpf.make_addplot(
                series, type="scatter",
                markersize=style["size"],
                marker=style["marker"],
                color=style["color"],
                edgecolors=style["edge"],
                linewidths=1.2,
            ))

    mc = mpf.make_marketcolors(
        up="#CC3333", down="#00AA00", edge="inherit", wick="inherit", volume="in",
    )
    mpf_style = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=":", gridcolor="#E0E0E0",
        rc={"font.sans-serif": plt.rcParams["font.sans-serif"],
            "axes.unicode_minus": False},
    )

    n_bars = len(df_view)
    fig_width = max(24, n_bars * 0.14)   # wider canvas for dense history
    fig, axes = mpf.plot(
        df_view, type="candle", volume=True, style=mpf_style,
        addplot=add_plots, figsize=(fig_width, 10), returnfig=True,
        tight_layout=False,
        scale_padding={"left": 0.05, "right": 0.1, "top": 0.7, "bottom": 0.5},
    )
    ax = axes[0]

    # Annotate each signal with a small label
    SIG_SHORT = {"STRONG_BUY": "SB", "BUY": "B", "HOLD": "H", "AVOID": "A"}
    for idx_pos, sig in valid_sigs:
        sig_type = sig.get("signal", "HOLD")
        score    = sig.get("score", 0)
        ts       = sig.get("touch_strategy", "")
        style    = SIGNAL_STYLE.get(sig_type, SIGNAL_STYLE["HOLD"])
        low_val  = float(df_view.iloc[idx_pos]["low"])
        label    = f"{SIG_SHORT.get(sig_type, sig_type[:2])}{score:+d}"
        if ts == "touch":
            label += "T"
        ax.annotate(
            label,
            xy=(idx_pos, low_val * 0.935),
            fontsize=7, fontweight="bold", color=style["edge"],
            ha="center", va="top",
        )

    # Warmup boundary
    _DETECTOR_WARMUP = 200
    if len(df) > _DETECTOR_WARMUP:
        _warmup_date = df.index[_DETECTOR_WARMUP]
        if df_view.index[0] <= _warmup_date <= df_view.index[-1]:
            _wpos = int(df_view.index.searchsorted(_warmup_date, side="left"))
            ax.axvline(_wpos - 0.5, color="#AAAAAA", linewidth=1.2,
                       linestyle="--", alpha=0.85, zorder=1)

    # Title and signal-count summary
    fig.suptitle(f"{market}:{sym}  {name}", fontsize=16, fontweight="bold",
                 y=0.99, color="#1a1a1a")
    sig_counts: dict[str, int] = {}
    for sig in signals:
        st = sig.get("signal", "HOLD")
        sig_counts[st] = sig_counts.get(st, 0) + 1
    counts_str = "  ".join(
        f"{k}×{v}" for k, v in sig_counts.items() if v
    )
    ax.set_title(f"共 {len(signals)} 个信号: {counts_str}", fontsize=10, pad=4)

    # Legend
    legend_elements = []
    for ema_name in ["EMA34", "EMA60", "EMA144", "EMA200"]:
        legend_elements.append(
            Line2D([0], [0], color=_EMA_COLORS[ema_name], linewidth=1.2,
                   label=ema_name)
        )
    for sig_type, style in SIGNAL_STYLE.items():
        legend_elements.append(
            Line2D([0], [0], marker=style["marker"], color="w",
                   markerfacecolor=style["color"], markeredgecolor=style["edge"],
                   markersize=9, label=sig_type)
        )
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.8)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{market}_{sym}.png"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Glyph.*missing from font")
        fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_wave_structure_chart(
    sym: str,
    market: str,
    name: str,
    df_price: pd.DataFrame,
    waves: list[dict],
    out_dir: Path,
) -> Path | None:
    """Draw full-history candlestick chart with wave structure annotations.

    Each wave is shaded with a colour band and labelled with its start/end
    prices, peak, rise %, and sub-wave count.

    Args:
        sym: Ticker symbol.
        market: Market label (US / HK / CN).
        name: Human-readable display name.
        df_price: Full OHLCV DataFrame.
        waves: ``major_waves`` list from ``analyze_wave_structure``.
        out_dir: Output directory.

    Returns:
        Path to the saved PNG, or None on failure.
    """
    df = df_price.copy()
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    df = df.sort_index()[["open", "high", "low", "close", "volume"]]

    if len(df) < 50 or not waves:
        return None

    # EMAs on full data
    emas_full = {
        f"EMA{s}": df["close"].ewm(span=s, adjust=False).mean()
        for s in [34, 55, 60, 144, 169, 200]
    }
    add_plots = []
    for ema_name, ema_s in emas_full.items():
        add_plots.append(mpf.make_addplot(
            ema_s, color=_EMA_COLORS[ema_name], width=1.0,
            linestyle="-" if any(x in ema_name for x in ["144", "169", "200"]) else "--",
        ))

    mc = mpf.make_marketcolors(
        up="#CC3333", down="#00AA00", edge="inherit", wick="inherit", volume="in",
    )
    mpf_style = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=":", gridcolor="#E0E0E0",
        rc={"font.sans-serif": plt.rcParams["font.sans-serif"],
            "axes.unicode_minus": False},
    )

    n_bars = len(df)
    fig_width = max(28, n_bars * 0.05)
    fig, axes = mpf.plot(
        df, type="candle", volume=True, style=mpf_style,
        addplot=add_plots,
        figsize=(fig_width, 10), returnfig=True, tight_layout=False,
        warn_too_much_data=n_bars + 1,
        scale_padding={"left": 0.03, "right": 0.03, "top": 0.7, "bottom": 0.5},
    )
    ax = axes[0]

    _WAVE_COLORS = [
        "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
        "#00BCD4", "#F44336", "#8BC34A", "#E91E63",
    ]

    _cjk_font = plt.rcParams["font.sans-serif"][0]

    for w in waves:
        wn = w["wave_number"]
        sp = w["start_pivot"]
        ep = w.get("end_pivot")
        pk = w["peak_pivot"]
        rise = w.get("rise_pct", 0)
        subs = w.get("sub_wave_count", 0)

        x_start = sp["iloc"]
        x_end = ep["iloc"] if ep else len(df) - 1
        color = _WAVE_COLORS[(wn - 1) % len(_WAVE_COLORS)]

        # Shaded region for this wave
        ax.axvspan(x_start, x_end, alpha=0.08, color=color, zorder=0)

        # Start vertical line + label
        ax.axvline(x_start, color=color, linewidth=1.5, linestyle="--", alpha=0.6, zorder=1)
        start_date = df.index[x_start].strftime("%y/%m/%d")
        ax.annotate(
            f"W{wn}\u2191\n{start_date}\n@{sp['value']:.1f}",
            xy=(x_start, sp["value"]),
            xytext=(x_start + 2, sp["value"] * 0.93),
            fontsize=7.5, fontweight="bold", color=color,
            fontfamily=_cjk_font,
            ha="left", va="top",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor=color),
        )

        # Peak label
        peak_date = df.index[pk["iloc"]].strftime("%y/%m/%d")
        ax.annotate(
            f"W{wn} peak\n{peak_date}\n@{pk['value']:.1f}\n+{rise:.0f}%",
            xy=(pk["iloc"], pk["value"]),
            xytext=(pk["iloc"], pk["value"] * 1.04),
            fontsize=7, color=color, fontfamily=_cjk_font,
            ha="center", va="bottom",
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor=color),
        )

        # End label (if wave is closed)
        if ep:
            ax.axvline(x_end, color=color, linewidth=1.5, linestyle="--", alpha=0.6, zorder=1)
            end_date = df.index[x_end].strftime("%y/%m/%d")
            ax.annotate(
                f"W{wn}\u2193\n{end_date}\n@{ep['value']:.1f}\nsubs={subs}",
                xy=(x_end, ep["value"]),
                xytext=(x_end + 2, ep["value"] * 0.93),
                fontsize=7, color=color, fontfamily=_cjk_font,
                ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor=color),
            )

    # Title
    fig.suptitle(f"{market}:{sym}  {name}  — 浪结构分析", fontsize=16, fontweight="bold", y=0.99)
    wave_desc = "  |  ".join(
        f"W{w['wave_number']}(+{w.get('rise_pct',0):.0f}%)"
        for w in waves
    )
    ax.set_title(wave_desc, fontsize=10, pad=4)

    # Legend
    legend_elements = []
    for ema_name in ["EMA34", "EMA60", "EMA144", "EMA200"]:
        legend_elements.append(
            Line2D([0], [0], color=_EMA_COLORS[ema_name], linewidth=1.2, label=ema_name)
        )
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.8)

    out_dir.mkdir(parents=True, exist_ok=True)
    _UNSAFE = set('/\\:*?"<>|')
    if market in ("HK", "CN"):
        safe_name = "".join(c for c in name if c not in _UNSAFE).strip()[:20] or sym
    else:
        safe_name = "".join(c if c.isascii() and (c.isalnum() or c in "-_") else "" for c in name).strip("-_")[:20] or sym
    out_path = out_dir / f"{market}_{sym}_{safe_name}_wave.png"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Glyph.*missing from font")
        fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# MA Squeeze scan chart
# ═════════════════════════════════════════════════════════════════════════════

def plot_ma_squeeze_chart(
    df: pd.DataFrame,
    ticker: str,
    display_name: str,
    stage: str,
    details: dict,
    market: str,
    chart_dir: Path,
) -> Path:
    """Render a candlestick chart with MA30/60/200 for a MA-Squeeze scan hit.

    Args:
        df: Full OHLCV DataFrame (lower-case columns).
        ticker: Ticker / code string.
        display_name: Human-readable display name.
        stage: "stage1" or "stage2".
        details: Signal details dict (ma_squeeze_ratio, prange_20d, prange_60d, …).
        market: "us" or "hk" — determines sub-directory under chart_dir.
        chart_dir: Root output directory; charts are saved to chart_dir/market/.

    Returns:
        Path to the saved PNG.
    """
    plot_df = df.iloc[-120:].copy()
    close = df["close"]
    for span, col in [(30, "ma30"), (60, "ma60"), (200, "ma200")]:
        plot_df[col] = close.rolling(span).mean().iloc[-120:]

    add_plots = [
        mpf.make_addplot(plot_df["ma30"],  panel=0, color="#FF6600", width=1.2),
        mpf.make_addplot(plot_df["ma60"],  panel=0, color="#0066FF", width=1.2),
        mpf.make_addplot(plot_df["ma200"], panel=0, color="#9900CC", width=1.5,
                         linestyle="dashed"),
    ]

    stage_label = "S1" if stage == "stage1" else "S2"
    squeeze_r = details.get("ma_squeeze_ratio", "")
    pr20 = details.get("prange_20d", "")
    pr60 = details.get("prange_60d", "")
    title = (f"{display_name}  [{stage_label}]  "
             f"squeeze={squeeze_r}  pr20d={pr20}%  pr60d={pr60}%")

    rc: dict = {}
    cjk_font = _find_cjk_font()
    if cjk_font:
        rc["font.sans-serif"] = [cjk_font]
        rc["axes.unicode_minus"] = False

    style = mpf.make_mpf_style(base_mpf_style="charles", rc=rc)
    sub_dir = chart_dir / market
    sub_dir.mkdir(parents=True, exist_ok=True)
    save_path = sub_dir / f"{ticker}_{stage_label}.png"

    mpf.plot(
        plot_df, type="candle", volume=True, title=title, style=style,
        figscale=1.5, figratio=(16, 9), addplot=add_plots,
        savefig=str(save_path),
    )
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# HK RS scan charts
# ═════════════════════════════════════════════════════════════════════════════

def plot_hk_rs_charts(
    rs_data: dict[str, pd.DataFrame],
    code_bench_map: dict[str, str],
    code_name_map: dict[str, str],
    out_dir: Path,
    days: int = 0,
) -> Path:
    """Render RS Z-Score trend charts for every stock in rs_data.

    Args:
        rs_data: {code: rs_df} as returned by hk_rs_scan.compute_all_rs_series().
        code_bench_map: Maps stock code → benchmark code ("800000" or "800700").
        code_name_map: Maps stock code → Chinese display name.
        out_dir: Directory to write PNGs to.
        days: If >0, only show the last N days of data.

    Returns:
        out_dir path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_name_map = {"800000": "恒生指数", "800700": "恒生科技"}
    total = len(rs_data)
    for i, (code, rs_df) in enumerate(rs_data.items(), 1):
        name = code_name_map.get(code, "")
        bench = bench_name_map.get(code_bench_map.get(code, ""), "")
        plot_df = rs_df.iloc[-days:] if days > 0 else rs_df
        try:
            _draw_single_rs_chart(code, name, bench, plot_df, out_dir)
            if i % 10 == 0 or i == total:
                logger.info(f"[{i}/{total}] RS 图表生成中 ...")
        except Exception as e:
            logger.error(f"{code} {name} 图表生成失败: {e}")
    plt.close("all")
    return out_dir


def _draw_single_rs_chart(
    code: str,
    name: str,
    bench: str,
    rs_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Render one three-panel RS Z-Score chart and save to out_dir."""
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 11),
        gridspec_kw={"height_ratios": [1, 1, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.08)

    dates = rs_df.index
    close = rs_df["close"]
    rs_z = rs_df["rs_zscore"]
    rs_fast = rs_df["rs_zscore_fast"]
    rs_z_21 = rs_df["rs_zscore_21d"]
    rs_fast_21 = rs_df["rs_zscore_21d_fast"]

    ax1.plot(dates, close, color="#2196F3", linewidth=1.2, label="收盘价")
    if len(close) >= 50:
        ax1.plot(dates, close.rolling(50).mean(), color="#FF9800",
                 linewidth=0.8, alpha=0.7, label="MA50")
    if len(close) >= 200:
        ax1.plot(dates, close.rolling(200).mean(), color="#9C27B0",
                 linewidth=0.8, alpha=0.7, label="MA200")
    ax1.set_ylabel("价格", fontsize=11)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{code} {name}  |  基准: {bench}", fontsize=14, fontweight="bold")

    _draw_rs_panel(ax2, dates, rs_z, rs_fast,
                   panel_label="RS Z-Score 63d",
                   slow_label="RS 63d (EMA10)", fast_label="RS 63d 快线 (EMA5)",
                   color_slow="#E91E63", color_fast="#FF9800")
    _draw_rs_panel(ax3, dates, rs_z_21, rs_fast_21,
                   panel_label="RS Z-Score 21d",
                   slow_label="RS 21d (EMA10)", fast_label="RS 21d 快线 (EMA5)",
                   color_slow="#1565C0", color_fast="#26A69A")

    ax3.set_xlabel("日期", fontsize=11)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30)

    safe_name = name.replace("/", "_").replace(" ", "")
    filepath = out_dir / f"{code}_{safe_name}.png"
    fig.savefig(filepath, dpi=120, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def _draw_rs_panel(
    ax,
    dates,
    rs_z: pd.Series,
    rs_fast: pd.Series | None,
    panel_label: str,
    slow_label: str,
    fast_label: str,
    color_slow: str,
    color_fast: str,
) -> None:
    """Draw one RS Z-Score panel on an existing Axes object."""
    for y_val, style, alpha in [
        (0, "-", 0.6), (1, ":", 0.5), (-1, ":", 0.5), (2, "--", 0.4), (-2, "--", 0.4),
    ]:
        color = "gray" if y_val == 0 else ("#4CAF50" if y_val > 0 else "#F44336")
        ax.axhline(y=y_val, color=color, linewidth=0.6 if abs(y_val) else 1.0,
                   linestyle=style, alpha=alpha)

    valid = rs_z.notna()
    ax.fill_between(dates, 0, rs_z, where=(rs_z >= 0) & valid,
                    alpha=0.15, color="#4CAF50", interpolate=True)
    ax.fill_between(dates, 0, rs_z, where=(rs_z < 0) & valid,
                    alpha=0.15, color="#F44336", interpolate=True)

    ax.plot(dates, rs_z, color=color_slow, linewidth=1.5, label=slow_label)
    if rs_fast is not None and rs_fast.notna().any():
        ax.plot(dates, rs_fast, color=color_fast, linewidth=0.9, alpha=0.6, label=fast_label)

    valid_z = rs_z.dropna()
    if len(valid_z) > 0:
        last_rs = valid_z.iloc[-1]
        last_date = valid_z.index[-1]
        ax.annotate(f"{last_rs:+.2f}\u03c3", xy=(last_date, last_rs),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=9, color=color_slow, fontweight="bold")
        for y_val, label in [(2, "显著强"), (1, "偏强"), (-1, "偏弱"), (-2, "显著弱")]:
            ax.text(1.01, y_val, label, transform=ax.get_yaxis_transform(),
                    fontsize=7, color="gray", va="center")

        if last_rs > 1.5:
            trend_label, trend_color = "↑ 显著跑赢", "#4CAF50"
        elif last_rs > 0.5:
            trend_label, trend_color = "↗ 跑赢", "#8BC34A"
        elif last_rs > -0.5:
            trend_label, trend_color = "→ 接近", "#9E9E9E"
        elif last_rs > -1.5:
            trend_label, trend_color = "↘ 跑输", "#FF9800"
        else:
            trend_label, trend_color = "↓ 显著跑输", "#F44336"

        dir_label = ""
        if len(valid_z) >= 5:
            dir_label = f"5日变化: {last_rs - valid_z.iloc[-5]:+.2f}\u03c3"
        ax.text(0.02, 0.95,
                f"{trend_label}  ({last_rs:+.2f}\u03c3)  {dir_label}",
                transform=ax.transAxes, fontsize=10, color=trend_color,
                fontweight="bold", verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_ylabel(f"{panel_label} (\u03c3)", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    valid_vals = rs_z.dropna()
    y_abs_max = (max(abs(valid_vals.min()), abs(valid_vals.max()), 3) * 1.05
                 if len(valid_vals) > 0 else 3)
    ax.set_ylim(-y_abs_max, y_abs_max)





# ═══════════════════════════════════════════════════════════════════════════
# Migrated from backtest/chart.py — all strategy chart rendering lives here
# ═══════════════════════════════════════════════════════════════════════════

_STRATEGY_NAMES_CN = {
    "vegas": "Vegas 通道回弹",
    "triangle": "上升三角形(OLS)",
    "triangle_kde": "上升三角形(KDE)",
    "parallel": "上升平行通道",
    "wedge": "上升楔形",
    "vcp": "VCP 波动率收缩",
    "rs_strict": "RS加速(严格)",
    "rs_loose": "RS加速(宽松)",
    "rs_trap_strict": "RS陷阱预警(严格)",
    "rs_trap_loose": "RS陷阱预警(宽松)",
    "ma_squeeze": "MA Squeeze",
    "momentum": "Momentum 异动",
}

def _plot_single_entry_batch(
    hits: list[dict],
    output_dir: str | Path,
    title_builder,
    filename_builder,
    addplot_builder,
    last_n_days: int,
) -> None:
    """批量绘制单点信号图（每票一图，统一渲染流程）。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not hits:
        logger.warning("没有找到符合条件的股票，无图可绘")
        return

    style = mpf.make_mpf_style(base_mpf_style="charles")
    for item in hits:
        ticker = item["ticker"]
        df = item["df"].copy().iloc[-last_n_days:]

        kwargs = {
            "type": "candle",
            "volume": True,
            "title": title_builder(ticker, item),
            "style": style,
            "figscale": 1.5,
            "figratio": (16, 9),
            "savefig": str(out_dir / filename_builder(ticker, item)),
        }
        add_plots = addplot_builder(df, item)
        if add_plots:
            kwargs["addplot"] = add_plots

        mpf.plot(df, **kwargs)


def plot_strategy_hits(
    hits: list[dict],
    plot_mode: str,
    output_dir: str | Path | None = None,
    **kwargs,
) -> None:
    """统一图形入口：按 plot_mode 分发到对应绘图实现。"""
    if plot_mode == "single_signal_macd":
        plot_macd_cross_results(
            hits,
            output_dir=output_dir,
            last_n_days=kwargs.get("last_n_days", 60),
        )
        return

    if plot_mode == "single_signal_vegas":
        plot_vegas_touch_results(
            hits,
            output_dir=output_dir,
            last_n_days=kwargs.get("last_n_days", 120),
        )
        return

    if plot_mode == "pattern_triangle":
        plot_ascending_triangle_results(hits, output_dir=output_dir)
        return

    if plot_mode == "pattern_vcp":
        plot_vcp_results(hits, output_dir=output_dir)
        return

    if plot_mode == "pattern_cup_handle":
        plot_cup_handle_annotated(hits, output_dir=output_dir)
        return

    raise ValueError(f"未知 plot_mode: {plot_mode}")


def plot_candlestick(
    df: pd.DataFrame,
    title: str = "K线图",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    绘制带成交量的K线图

    Args:
        df: 包含 open, high, low, close, volume 的 DataFrame（index 为日期）
        title: 图表标题
        save_path: 保存路径（可选）
        show: 是否显示图表
    """
    style = mpf.make_mpf_style(base_mpf_style="charles", rc={"font.size": 10})

    kwargs = {
        "type": "candle",
        "volume": True,
        "title": title,
        "style": style,
        "figscale": 1.2,
    }

    if save_path:
        kwargs["savefig"] = str(save_path)
    if show:
        kwargs["show_nontrading"] = False

    mpf.plot(df, **kwargs)


def plot_with_indicators(
    df: pd.DataFrame,
    title: str = "技术分析",
    indicators: list[str] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """
    绘制带技术指标的K线图

    Args:
        df: 带有技术指标列的 DataFrame
        title: 图表标题
        indicators: 要绘制的指标列名列表
        save_path: 保存路径
    """
    if indicators is None:
        indicators = ["sma_5", "sma_20", "sma_60"]

    add_plots = []

    # 主图叠加指标（均线、布林带）
    main_indicators = [col for col in indicators if col in df.columns and col.startswith(("sma_", "ema_", "bb_"))]
    for col in main_indicators:
        add_plots.append(mpf.make_addplot(df[col], panel=0))

    # 副图指标
    if "rsi" in indicators and "rsi" in df.columns:
        add_plots.append(mpf.make_addplot(df["rsi"], panel=2, ylabel="RSI"))

    if "macd" in indicators and "macd" in df.columns:
        add_plots.append(mpf.make_addplot(df["macd"], panel=3, color="blue", ylabel="MACD"))
        if "macd_signal" in df.columns:
            add_plots.append(mpf.make_addplot(df["macd_signal"], panel=3, color="orange"))
        if "macd_hist" in df.columns:
            add_plots.append(mpf.make_addplot(df["macd_hist"], panel=3, type="bar", color="gray"))

    style = mpf.make_mpf_style(base_mpf_style="charles")

    kwargs = {
        "type": "candle",
        "volume": True,
        "title": title,
        "style": style,
        "addplot": add_plots if add_plots else None,
        "figscale": 1.5,
        "figratio": (16, 9),
    }

    if save_path:
        kwargs["savefig"] = str(save_path)

    mpf.plot(df, **kwargs)


def plot_macd_cross_results(
    hits: list[dict],
    output_dir: str | Path | None = None,
    last_n_days: int = 60,
) -> None:
    """
    为 MACD 金叉扫描结果批量绘制 K 线图（含 MACD 副图）

    Args:
        hits: scan_macd_cross 返回的结果列表
        output_dir: 图片保存目录，默认为 data/output
        last_n_days: 图中显示最近多少个交易日
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "output"

    def _macd_addplots(df: pd.DataFrame, _item: dict) -> list:
        """Build MACD overlay panels for a single MACD scan chart."""
        add_plots = []
        if "macd" in df.columns:
            add_plots.append(mpf.make_addplot(df["macd"], panel=2, color="blue", ylabel="MACD"))
        if "macd_signal" in df.columns:
            add_plots.append(mpf.make_addplot(df["macd_signal"], panel=2, color="orange"))
        if "macd_hist" in df.columns:
            colors = ["green" if v >= 0 else "red" for v in df["macd_hist"]]
            add_plots.append(mpf.make_addplot(df["macd_hist"], panel=2, type="bar", color=colors))
        return add_plots

    _plot_single_entry_batch(
        hits=hits,
        output_dir=output_dir,
        title_builder=lambda ticker, _item: f"{ticker} - MACD Golden Cross",
        filename_builder=lambda ticker, _item: f"{ticker}_macd_cross.png",
        addplot_builder=_macd_addplots,
        last_n_days=last_n_days,
    )


def plot_vegas_touch_results(
    hits: list[dict],
    output_dir: str | Path | None = None,
    last_n_days: int = 120,
) -> None:
    """
    为 Vegas 通道回踩扫描结果批量绘制 K 线图（含 EMA144/169 通道）

    Args:
        hits: 统一策略扫描输出转换后的结果列表
        output_dir: 图片保存目录，默认为 data/output
        last_n_days: 图中显示最近多少个交易日（默认 120 天看半年趋势）
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "output"

    def _vegas_addplots(df: pd.DataFrame, _item: dict) -> list:
        """Build EMA144/169 overlay lines for a single Vegas touch chart."""
        add_plots = []
        if "ema_144" in df.columns:
            add_plots.append(mpf.make_addplot(
                df["ema_144"], panel=0, color="blue", width=1.5,
                linestyle="dashed", secondary_y=False,
            ))
        if "ema_169" in df.columns:
            add_plots.append(mpf.make_addplot(
                df["ema_169"], panel=0, color="purple", width=1.5,
                linestyle="dashed", secondary_y=False,
            ))
        return add_plots

    _plot_single_entry_batch(
        hits=hits,
        output_dir=output_dir,
        title_builder=lambda ticker, _item: f"{ticker} - Vegas Channel (EMA144/169) Touch",
        filename_builder=lambda ticker, _item: f"{ticker}_vegas_touch.png",
        addplot_builder=_vegas_addplots,
        last_n_days=last_n_days,
    )


def plot_ascending_triangle_results(
    hits: list[dict],
    output_dir: str | Path | None = None,
) -> None:
    """
    为上升三角形/楔形扫描结果绘制 K 线图（含趋势线）

    Args:
        hits: 统一策略扫描输出转换后的结果列表（需包含 pattern_info）
        output_dir: 图片保存目录
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hits:
        logger.warning("没有找到符合条件的股票，无图可绘")
        return

    for item in hits:
        ticker = item["ticker"]
        info = item["pattern_info"]
        df_full = item["df"].copy()

        period = info["period"]
        window_start = info["window_start"]

        # 显示区间：形态开始前留 10 天上下文
        chart_start = max(0, window_start - 10)
        df = df_full.iloc[chart_start:]

        # 在 df 坐标系下重新计算趋势线
        res = info["resistance"]
        sup = info["support"]

        # 趋势线 y = slope * x + intercept，x 是相对于 window 起始的位置
        # 对整个显示区间生成趋势线值
        offset = window_start - chart_start  # df 坐标中形态起点偏移
        n_display = len(df)

        resistance_line = pd.Series(np.nan, index=df.index)
        support_line = pd.Series(np.nan, index=df.index)

        for i in range(n_display):
            x = i - offset  # 相对于 window 起始的 x 坐标
            if 0 <= x < period:
                resistance_line.iloc[i] = res["slope"] * x + res["intercept"]
                support_line.iloc[i] = sup["slope"] * x + sup["intercept"]

        add_plots = [
            mpf.make_addplot(resistance_line, panel=0, color="red", width=2.0,
                             linestyle="dashed", secondary_y=False),
            mpf.make_addplot(support_line, panel=0, color="green", width=2.0,
                             linestyle="dashed", secondary_y=False),
        ]

        _PATTERN_EN = {
            "ascending_triangle": "Ascending Triangle",
            "rising_wedge": "Rising Wedge",
            "symmetrical_triangle": "Symmetrical Triangle",
            "descending_wedge": "Descending Wedge",
        }
        pattern_name = _PATTERN_EN.get(info["pattern"], info["pattern"])

        angle = info.get("convergence_angle_deg", 0)
        res_touches = res.get("touches", 0)
        sup_touches = sup.get("touches", 0)
        status = info.get("convergence_status", "")
        dtc = info.get("days_to_convergence", 0)
        status_label = "CONVERGED" if status == "converged" else f"~{dtc:.0f}d"

        style = mpf.make_mpf_style(base_mpf_style="charles")
        save_path = output_dir / f"{ticker}_triangle.png"

        kwargs = {
            "type": "candle",
            "volume": True,
            "title": (f"{ticker} - {pattern_name} [{status_label}] "
                      f"(angle={angle:.1f}° "
                      f"res_touch={res_touches} sup_touch={sup_touches})"),
            "style": style,
            "figscale": 1.5,
            "figratio": (16, 9),
            "addplot": add_plots,
            "savefig": str(save_path),
        }

        mpf.plot(df, **kwargs)
        logger.info(f"已保存 {ticker} {pattern_name} 图 → {save_path}")


def plot_vcp_results(
    hits: list[dict],
    output_dir: str | Path | None = None,
) -> None:
    """
    为 VCP 扫描结果绘制 K 线图（约 1 年数据，VCP 区域在右侧）。

    Args:
        hits: 统一策略扫描输出转换后的结果列表（需包含 vcp_info）
        output_dir: 图片保存目录
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hits:
        logger.warning("没有找到符合 VCP 条件的股票，无图可绘")
        return

    for item in hits:
        ticker = item["ticker"]
        info = item["vcp_info"]
        df_full = item["df"].copy()

        # 计算基底起点 iloc
        current_iloc = len(df_full) - 1
        base_high_iloc = current_iloc - info["base_days"]

        # 显示范围：基底前 ~200 天 + 信号后 5 天（约 1 年数据）
        pre_days = 200
        view_start = max(0, base_high_iloc - pre_days)
        view_end = min(len(df_full), current_iloc + 5)
        df = df_full.iloc[view_start:view_end].copy()

        if len(df) < 20:
            continue

        # SMA150 / SMA200（基于全量数据计算再截取）
        sma150 = df_full["close"].rolling(150).mean().iloc[view_start:view_end]
        sma200 = df_full["close"].rolling(200).mean().iloc[view_start:view_end]

        add_plots = []
        if not sma150.isna().all():
            add_plots.append(
                mpf.make_addplot(sma150, panel=0, color="blue", width=1.0,
                                 linestyle="dashed", secondary_y=False)
            )
        if not sma200.isna().all():
            add_plots.append(
                mpf.make_addplot(sma200, panel=0, color="purple", width=1.0,
                                 linestyle="dashed", secondary_y=False)
            )

        # 收缩区域标注：为每个收缩画高低水平线
        contractions = info.get("waves", [])
        if isinstance(contractions, list) and contractions and isinstance(contractions[0], dict):
            # waves 中的 idx 是相对 base_period (从 base_high_iloc 开始) 的偏移
            for ci, c in enumerate(contractions):
                hi_pos = base_high_iloc + c["high_idx"] - view_start
                lo_pos = base_high_iloc + c["low_idx"] - view_start
                hi_val = c["high_val"]
                lo_val = c["low_val"]

                hi_line = pd.Series(np.nan, index=df.index)
                lo_line = pd.Series(np.nan, index=df.index)

                start_pos = max(0, hi_pos)
                end_pos = min(len(df) - 1, lo_pos + 5)

                for i in range(start_pos, end_pos + 1):
                    if i < len(hi_line):
                        hi_line.iloc[i] = hi_val
                        lo_line.iloc[i] = lo_val

                color = "red" if ci == 0 else ("orange" if ci == 1 else "gray")
                if hi_line.notna().any():
                    add_plots.append(
                        mpf.make_addplot(hi_line, panel=0, color=color, width=1.2,
                                         linestyle="dotted", secondary_y=False)
                    )
                if lo_line.notna().any():
                    add_plots.append(
                        mpf.make_addplot(lo_line, panel=0, color=color, width=1.2,
                                         linestyle="dotted", secondary_y=False)
                    )

        # Base High 水平参考线
        base_high_val = info.get("base_high", 0)
        if base_high_val > 0:
            pivot_line = pd.Series(base_high_val, index=df.index)
            add_plots.append(
                mpf.make_addplot(pivot_line, panel=0, color="magenta", width=1.5,
                                 linestyle="--", secondary_y=False)
            )

        # 绿色圆圈：VCP 起点（Base High）
        start_marker = pd.Series(np.nan, index=df.index)
        bh_view_pos = base_high_iloc - view_start
        if 0 <= bh_view_pos < len(df):
            start_marker.iloc[bh_view_pos] = df_full.iloc[base_high_iloc]["high"]
        if start_marker.notna().any():
            add_plots.append(
                mpf.make_addplot(start_marker, type="scatter", markersize=200,
                                 marker="o", color="lime", edgecolors="green",
                                 linewidths=2, alpha=0.9)
            )

        # 红色圆圈：VCP 终点（当前信号日）
        end_marker = pd.Series(np.nan, index=df.index)
        sig_view_pos = current_iloc - view_start
        if 0 <= sig_view_pos < len(df):
            end_marker.iloc[sig_view_pos] = df_full.iloc[current_iloc]["close"]
        if end_marker.notna().any():
            add_plots.append(
                mpf.make_addplot(end_marker, type="scatter", markersize=200,
                                 marker="o", color="red", edgecolors="darkred",
                                 linewidths=2, alpha=0.9)
            )

        # 标题
        depths = info.get("depths", [])
        depths_str = "→".join(f"{d:.0f}%" for d in depths) if depths else "N/A"
        pattern_name = info.get("pattern", "VCP")

        style = mpf.make_mpf_style(base_mpf_style="charles")
        save_path = output_dir / f"{ticker}_vcp.png"

        kwargs = {
            "type": "candle",
            "volume": True,
            "title": (f"{ticker} - {pattern_name} "
                      f"(T: {depths_str}, "
                      f"vol={info['vol_ratio']:.0%}, "
                      f"base={info['base_days']}d)"),
            "style": style,
            "figscale": 1.5,
            "figratio": (16, 9),
            "addplot": add_plots,
            "savefig": str(save_path),
            "warn_too_much_data": len(df) + 1,
        }

        mpf.plot(df, **kwargs)
        logger.info(f"已保存 {ticker} {pattern_name} 图 → {save_path}")


def plot_cup_handle_annotated(
    hits: list[dict],
    output_dir: str | Path | None = None,
) -> None:
    """
    为杯柄 / 平底形态扫描结果绘制标注图，清晰标出：
      - 🟢 左杯口（base_high，左侧高点）  → 绿色上三角
      - 🔵 杯底（最低点）                → 蓝色下三角
      - 🟠 右杯口（手柄起点，右侧回升位置）→ 橙色上三角
      - 紫/品红色水平虚线：枢轴（左杯口价格参考线）

    图表窗口：左杯口前 ~60 根 K 线 + 完整基底 + 手柄区域。

    Args:
        hits: 统一策略扫描输出转换后的结果列表（需包含 vcp_info）
        output_dir: 图片保存目录，默认 data/output/vcp/
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "output" / "vcp"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hits:
        logger.warning("没有找到符合条件的股票，无图可绘")
        return

    for item in hits:
        ticker = item["ticker"]
        info = item["vcp_info"]
        df_full = item["df"].copy()

        base_start_iloc = info["base_start_iloc"]      # 左杯口绝对位置
        cup_bottom_iloc = info["cup_bottom_iloc"]      # 杯底绝对位置
        handle_start_iloc = info["handle_start_iloc"]  # 右杯口绝对位置
        current_iloc = len(df_full) - 1

        # 视图窗口：左杯口前 60 根 + 当前 + 5 根
        pre_days = 60
        view_start = max(0, base_start_iloc - pre_days)
        view_end = min(len(df_full), current_iloc + 5)
        df = df_full.iloc[view_start:view_end].copy()

        if len(df) < 20:
            continue

        # SMA150 / SMA200（基于全量数据计算再截取）
        sma150 = df_full["close"].rolling(150).mean().iloc[view_start:view_end]
        sma200 = df_full["close"].rolling(200).mean().iloc[view_start:view_end]

        add_plots = []
        if not sma150.isna().all():
            add_plots.append(mpf.make_addplot(
                sma150, panel=0, color="steelblue", width=1.0,
                linestyle="dashed", secondary_y=False,
            ))
        if not sma200.isna().all():
            add_plots.append(mpf.make_addplot(
                sma200, panel=0, color="purple", width=1.0,
                linestyle="dashed", secondary_y=False,
            ))

        # 枢轴参考线（左杯口价格水平线）
        base_high_val = info["base_high"]
        pivot_line = pd.Series(base_high_val, index=df.index)
        add_plots.append(mpf.make_addplot(
            pivot_line, panel=0, color="magenta", width=1.5,
            linestyle="--", secondary_y=False,
        ))

        def _view_pos(abs_iloc: int) -> int:
            """Translate an absolute iloc in the full DataFrame into the chart window."""
            return abs_iloc - view_start

        # ── 🟢 左杯口标注（上三角，高出最高价 2%） ──
        lrim_pos = _view_pos(base_start_iloc)
        left_rim_marker = pd.Series(np.nan, index=df.index)
        if 0 <= lrim_pos < len(df):
            left_rim_marker.iloc[lrim_pos] = df_full.iloc[base_start_iloc]["high"] * 1.03
        if left_rim_marker.notna().any():
            add_plots.append(mpf.make_addplot(
                left_rim_marker, type="scatter", markersize=200,
                marker="^", color="lime", edgecolors="green", linewidths=2,
            ))

        # ── 🔵 杯底标注（下三角，低于最低价 2%） ──
        bot_pos = _view_pos(cup_bottom_iloc)
        bottom_marker = pd.Series(np.nan, index=df.index)
        if 0 <= bot_pos < len(df):
            bottom_marker.iloc[bot_pos] = df_full.iloc[cup_bottom_iloc]["low"] * 0.97
        if bottom_marker.notna().any():
            add_plots.append(mpf.make_addplot(
                bottom_marker, type="scatter", markersize=200,
                marker="v", color="deepskyblue", edgecolors="navy", linewidths=2,
            ))

        # ── 🟠 右杯口（手柄起点）标注（上三角，高出最高价 2%） ──
        rrim_pos = _view_pos(handle_start_iloc)
        right_rim_marker = pd.Series(np.nan, index=df.index)
        if 0 <= rrim_pos < len(df):
            right_rim_marker.iloc[rrim_pos] = df_full.iloc[handle_start_iloc]["high"] * 1.03
        if right_rim_marker.notna().any():
            add_plots.append(mpf.make_addplot(
                right_rim_marker, type="scatter", markersize=200,
                marker="^", color="orange", edgecolors="darkorange", linewidths=2,
            ))

        # 标题
        depths = info.get("depths", [])
        depths_str = "→".join(f"{d:.0f}%" for d in depths) if depths else "N/A"
        pattern_name = info.get("pattern", "VCP")

        style = mpf.make_mpf_style(base_mpf_style="charles")
        save_path = output_dir / f"{ticker}_cup_handle.png"

        kwargs = {
            "type": "candle",
            "volume": True,
            "title": (
                f"{ticker} - {pattern_name} "
                f"(深度:{info['base_depth_pct']:.1f}%  收缩:{depths_str}  "
                f"量缩:{info['vol_ratio']:.0%}  基底:{info['base_days']}d)  "
                f"▲绿=左杯口  ▼蓝=杯底  ▲橙=右杯口  --枢轴"
            ),
            "style": style,
            "figscale": 1.5,
            "figratio": (18, 9),
            "addplot": add_plots,
            "savefig": str(save_path),
            "warn_too_much_data": len(df) + 1,
        }

        mpf.plot(df, **kwargs)
        logger.info(f"已保存 {ticker} 杯柄标注图 → {save_path}")


# ═══════════════════════════════════════════════════════════
# Backtest strategy chart renderers
# ═══════════════════════════════════════════════════════════

_STRATEGY_NAMES_EN = {
    "vegas": "Vegas Channel Touchback",
    "triangle": "Ascending Triangle (OLS)",
    "triangle_kde": "Ascending Triangle (KDE)",
    "parallel": "Parallel Channel",
    "wedge": "Rising Wedge",
    "vcp": "VCP (Volatility Contraction)",
    "rs_strict": "RS Acceleration (Strict)",
    "rs_loose": "RS Acceleration (Loose)",
    "rs_trap_strict": "RS Trap Warning (Strict)",
    "rs_trap_loose": "RS Trap Warning (Loose)",
    "ma_squeeze": "MA Squeeze",
    "momentum": "Momentum Detection",
}

_TRIANGLE_CN = {
    "ascending_triangle": "上升三角形",
    "rising_wedge": "上升楔形",
}


def _normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw OHLCV frame to lower-case columns and sorted datetime index."""
    x = df.copy()
    x.columns = [c.lower() for c in x.columns]
    x.index = pd.to_datetime(x.index)
    x.index.name = "date"
    return x.sort_index()


def plot_triangle_backtest_signals(
    signals: list[dict],
    stock_data: dict[str, pd.DataFrame],
    output_dir: Path,
    max_charts: int = 30,
) -> None:
    """为每个三角形/楔形信号绘制 K 线图（含阻力线 / 支撑线）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    to_plot = sorted(signals, key=lambda x: x["signal_date"])[:max_charts]
    chart_count = 0

    for s in to_plot:
        ticker = s["ticker"]
        if ticker not in stock_data:
            continue

        full_df = stock_data[ticker]
        sig_iloc = s["signal_iloc"]
        ws_iloc = s["window_start_iloc"]
        period = s["period"]

        view_start = max(0, ws_iloc - 50)
        view_end = min(len(full_df), sig_iloc + 40)
        df_view = full_df.iloc[view_start:view_end].copy()

        if len(df_view) < 20:
            continue

        add_plots = []
        closes_full = full_df["close"]
        for ma_len, color in [(50, "green"), (150, "blue"), (200, "purple")]:
            ma = closes_full.rolling(ma_len).mean().iloc[view_start:view_end]
            if not ma.isna().all():
                add_plots.append(mpf.make_addplot(ma, color=color, width=0.8))

        offset = ws_iloc - view_start
        n_display = len(df_view)
        res_line = pd.Series(np.nan, index=df_view.index)
        sup_line = pd.Series(np.nan, index=df_view.index)

        for i in range(n_display):
            x = i - offset
            if 0 <= x < period:
                res_line.iloc[i] = s["resistance_slope"] * x + s["resistance_intercept"]
                sup_line.iloc[i] = s["support_slope"] * x + s["support_intercept"]

        add_plots.append(mpf.make_addplot(res_line, color="red", width=2.0, linestyle="dashed"))
        add_plots.append(mpf.make_addplot(sup_line, color="lime", width=2.0, linestyle="dashed"))

        sig_marker = pd.Series(np.nan, index=df_view.index)
        if view_start <= sig_iloc < view_end:
            sig_date = full_df.index[sig_iloc]
            sig_marker.loc[sig_date] = full_df.loc[sig_date, "close"]
        if sig_marker.notna().any():
            add_plots.append(mpf.make_addplot(
                sig_marker, type="scatter", markersize=200, marker="o",
                color="red", edgecolors="darkred", linewidths=2, alpha=0.9,
            ))

        ret_parts = []
        for pk, label in [("5d", "1W"), ("21d", "1M"), ("63d", "3M")]:
            info = s.get(pk, {})
            r = info.get("return_pct")
            if r is not None:
                ret_parts.append(f"{label}:{r:+.1f}%")

        cn = _TRIANGLE_CN.get(s["pattern"], s["pattern"])
        title = (
            f"{ticker} {cn} ({s['period']}d)  "
            f"收敛{s['convergence_ratio']:.0%} "
            f"波幅缩{1 - s['spread_contraction']:.0%}  |  "
            f"{'  '.join(ret_parts)}"
        )

        style = mpf.make_mpf_style(base_mpf_style="charles", rc={"font.size": 9})
        save_path = output_dir / f"{ticker}_tri_{s['signal_date']}.png"

        try:
            mpf.plot(
                df_view, type="candle", volume=True, title=title, style=style,
                figscale=1.5, figratio=(16, 9), addplot=add_plots,
                savefig=str(save_path), warn_too_much_data=len(df_view) + 1,
            )
            plt.close("all")
            chart_count += 1
        except Exception as e:
            logger.debug(f"{ticker}: 绘图失败 - {e}")

    logger.info(f"  已保存 {chart_count} 张信号图 → {output_dir}")


def plot_vcp_backtest_signals(
    signals: list[dict],
    stock_data: dict[str, pd.DataFrame],
    output_dir: Path,
    pre_year_days: int = 200,
    post_days: int = 40,
    max_charts: int = 30,
) -> None:
    """为每个 VCP 信号绘制约 1 年的 K 线图。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    to_plot = sorted(signals, key=lambda x: x["signal_date"])[:max_charts]
    chart_count = 0

    for s in to_plot:
        ticker = s["ticker"]
        if ticker not in stock_data:
            continue

        full_df = stock_data[ticker]
        sig_iloc = s["signal_iloc"]
        base_start_iloc = sig_iloc - s["base_days"]

        view_start = max(0, base_start_iloc - pre_year_days)
        view_end = min(len(full_df), sig_iloc + post_days)
        df_view = full_df.iloc[view_start:view_end].copy()

        if len(df_view) < 20:
            continue

        add_plots = []
        closes_full = full_df["close"]
        for ma_len, color in [(50, "green"), (150, "blue"), (200, "purple")]:
            ma = closes_full.rolling(ma_len).mean().iloc[view_start:view_end]
            if not ma.isna().all():
                add_plots.append(mpf.make_addplot(ma, color=color, width=0.8, linestyle="-"))

        start_marker = pd.Series(np.nan, index=df_view.index)
        if view_start <= base_start_iloc < view_end:
            start_date = full_df.index[base_start_iloc]
            start_marker.loc[start_date] = full_df.loc[start_date, "high"]

        end_marker = pd.Series(np.nan, index=df_view.index)
        if view_start <= sig_iloc < view_end:
            sig_date = full_df.index[sig_iloc]
            end_marker.loc[sig_date] = full_df.loc[sig_date, "close"]

        if start_marker.notna().any():
            add_plots.append(mpf.make_addplot(
                start_marker, type="scatter", markersize=200, marker="o",
                color="lime", edgecolors="green", linewidths=2, alpha=0.9,
            ))
        if end_marker.notna().any():
            add_plots.append(mpf.make_addplot(
                end_marker, type="scatter", markersize=200, marker="o",
                color="red", edgecolors="darkred", linewidths=2, alpha=0.9,
            ))

        base_high_line = pd.Series(s["base_high"], index=df_view.index)
        add_plots.append(mpf.make_addplot(base_high_line, color="magenta", width=0.7, linestyle=":"))

        ret_parts = []
        for pk, label in [("5d", "1W"), ("21d", "1M"), ("63d", "3M")]:
            info = s.get(pk, {})
            r = info.get("return_pct")
            if r is not None:
                ret_parts.append(f"{label}:{r:+.1f}%")
        ret_str = "  ".join(ret_parts)

        depths_str = "->".join(f"{d:.0f}%" for d in s["depths"])
        title = (
            f"{ticker} VCP ({s['wave_count']} waves: {depths_str})  "
            f"Base:{s['base_days']}d  |  {ret_str}"
        )

        style = mpf.make_mpf_style(base_mpf_style="charles", rc={"font.size": 9})
        save_path = output_dir / f"{ticker}_vcp_{s['signal_date']}.png"

        try:
            mpf.plot(
                df_view, type="candle", volume=True, title=title, style=style,
                figscale=1.5, figratio=(16, 9), addplot=add_plots,
                savefig=str(save_path), warn_too_much_data=len(df_view) + 1,
            )
            plt.close("all")
            chart_count += 1
        except Exception as e:
            logger.debug(f"{ticker}: 绘图失败 - {e}")
            continue

    logger.info(f"  已保存 {chart_count} 张 VCP 信号图 → {output_dir}")


def plot_main_rally_pullback_signals(
    all_hits: list[dict],
    market_data: dict[str, dict],
    chart_dir: Path,
    pre_bars: int = 120,
    post_bars: int = 63,
) -> None:
    """为回调信号绘制 K 线图，叠加 EMA34/55/144/169 及信号标记。"""
    chart_dir.mkdir(parents=True, exist_ok=True)
    style = mpf.make_mpf_style(base_mpf_style="charles", rc={"font.size": 9})
    saved = 0

    for hit in all_hits:
        market = hit["market"]
        symbol = hit["symbol"]
        key = f"{market}:{symbol}"
        info = market_data.get(key)
        if info is None:
            continue

        full_df = _normalize_price_frame(info["df"])
        sig_iloc = hit["signal_iloc"]
        entry_iloc = hit["entry_iloc"]

        view_start = max(0, sig_iloc - pre_bars)
        view_end = min(len(full_df), sig_iloc + post_bars + 1)
        df_view = full_df.iloc[view_start:view_end].copy()
        if len(df_view) < 20:
            continue

        close_full = full_df["close"].astype(float)
        ema34_view  = close_full.ewm(span=34,  adjust=False).mean().iloc[view_start:view_end]
        ema55_view  = close_full.ewm(span=55,  adjust=False).mean().iloc[view_start:view_end]
        ema60_view  = close_full.ewm(span=60,  adjust=False).mean().iloc[view_start:view_end]
        ema144_view = close_full.ewm(span=144, adjust=False).mean().iloc[view_start:view_end]
        ema169_view = close_full.ewm(span=169, adjust=False).mean().iloc[view_start:view_end]
        ema200_view = close_full.ewm(span=200, adjust=False).mean().iloc[view_start:view_end]

        add_plots = []
        for ema_view, color, width, ls in [
            (ema34_view,  "#2E8B57", 1.4, "solid"),
            (ema55_view,  "#FF6600", 1.8, "solid"),
            (ema60_view,  "#CC3399", 1.2, "solid"),
            (ema144_view, "#0066CC", 1.2, "dashed"),
            (ema169_view, "#9900CC", 1.2, "dashed"),
            (ema200_view, "#666666", 1.4, "dashed"),
        ]:
            if not ema_view.isna().all():
                add_plots.append(mpf.make_addplot(
                    ema_view, panel=0, color=color, width=width,
                    linestyle=ls, secondary_y=False,
                ))

        sig_marker = pd.Series(np.nan, index=df_view.index)
        sig_offset = sig_iloc - view_start
        if 0 <= sig_offset < len(df_view):
            sig_marker.iloc[sig_offset] = float(full_df.iloc[sig_iloc]["low"]) * 0.97
        if sig_marker.notna().any():
            add_plots.append(mpf.make_addplot(
                sig_marker, type="scatter", markersize=180,
                marker="^", color="lime", edgecolors="green", linewidths=1.5,
            ))

        entry_marker = pd.Series(np.nan, index=df_view.index)
        entry_offset = entry_iloc - view_start
        if 0 <= entry_offset < len(df_view):
            entry_marker.iloc[entry_offset] = float(full_df.iloc[entry_iloc]["close"])
        if entry_marker.notna().any():
            add_plots.append(mpf.make_addplot(
                entry_marker, type="scatter", markersize=120,
                marker="o", color="red", edgecolors="darkred", linewidths=1.5,
            ))

        ret_parts = []
        for d in [5, 10, 21, 63]:
            val = hit.get(f"ret_{d}d")
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                sign = "+" if float(val) >= 0 else ""
                ret_parts.append(f"{d}d:{sign}{val:.1f}%")
        ret_str = "  ".join(ret_parts) if ret_parts else "N/A"

        pullback = hit.get("pullback_pct_from_high", 0)
        support_type = hit.get("support_type", "unknown")
        support_band = hit.get("support_band", "")
        close_vs = hit.get("close_vs_ema55_pct", 0)
        title = (
            f"{market}:{symbol}  [{hit['signal_date']}]\n"
            f"pullback {pullback:.1f}% | support {support_type}/{support_band} | "
            f"close vs ema55 {close_vs:+.1f}% | "
            f"green ema34  orange ema55  pink ema60  blue ema144  purple ema169  gray ema200 | {ret_str}"
        )

        fname = f"{market}_{symbol}_{hit['signal_date'].replace('-', '')}.png"
        save_path = chart_dir / fname
        try:
            mpf.plot(
                df_view, type="candle", volume=True,
                title=title, style=style, figscale=1.4, figratio=(18, 9),
                addplot=add_plots, savefig=str(save_path),
                warn_too_much_data=len(df_view) + 1,
            )
            plt.close("all")
            saved += 1
        except Exception as e:
            logger.debug(f"{key} 绘图失败: {e}")

    logger.info(f"已保存 {saved} 张信号图 → {chart_dir}")


def _build_vegas_overlays(
    df_view: pd.DataFrame,
    full_df: pd.DataFrame,
    view_start: int,
    view_end: int,
) -> list:
    """为 Vegas 策略生成 EMA144/EMA169 辅助线。"""
    from stock_ana.data.indicators import add_vegas_channel
    df_full_ema = add_vegas_channel(full_df.copy())
    ema144 = df_full_ema["ema_144"].iloc[view_start:view_end]
    ema169 = df_full_ema["ema_169"].iloc[view_start:view_end]
    return [
        mpf.make_addplot(ema144, color="blue", width=1.5, linestyle="-"),
        mpf.make_addplot(ema169, color="purple", width=1.5, linestyle="-"),
    ]


def _build_triangle_overlays(
    df_view: pd.DataFrame,
    signal_info: dict,
    full_df: pd.DataFrame,
    view_start: int,
    view_end: int,
) -> list:
    """为三角形策略生成上轨/下轨辅助线。"""
    resistance = signal_info.get("resistance")
    support = signal_info.get("support")
    window_start = signal_info.get("window_start")
    period = signal_info.get("period")

    if not resistance or not support or window_start is None or period is None:
        return []

    res_slope = resistance["slope"]
    res_intercept = resistance["intercept"]
    sup_slope = support["slope"]
    sup_intercept = support["intercept"]

    res_series = pd.Series(np.nan, index=df_view.index)
    sup_series = pd.Series(np.nan, index=df_view.index)

    for i, _date in enumerate(df_view.index):
        abs_idx = view_start + i
        rel_x = abs_idx - window_start
        if rel_x >= -5 and rel_x <= period + 20:
            res_series.iloc[i] = res_slope * rel_x + res_intercept
            sup_series.iloc[i] = sup_slope * rel_x + sup_intercept

    return [
        mpf.make_addplot(res_series, color="red", width=1.2, linestyle="--"),
        mpf.make_addplot(sup_series, color="green", width=1.2, linestyle="--"),
    ]


def _build_vcp_overlays(
    df_view: pd.DataFrame,
    signal_info: dict,
    full_df: pd.DataFrame,
    view_start: int,
    view_end: int,
) -> list:
    """为 VCP 策略生成收缩高低点连线、枢轴价线、SMA。"""
    contractions = signal_info.get("contractions", [])
    window_start = signal_info.get("window_start")
    pivot_price = signal_info.get("pivot_price")

    if not contractions or window_start is None:
        return []

    overlays = []
    high_series = pd.Series(np.nan, index=df_view.index)
    low_series = pd.Series(np.nan, index=df_view.index)

    for c in contractions:
        hi_abs = window_start + c["high_idx"]
        lo_abs = window_start + c["low_idx"]
        if view_start <= hi_abs < view_end:
            high_series.iloc[hi_abs - view_start] = c["high_val"]
        if view_start <= lo_abs < view_end:
            low_series.iloc[lo_abs - view_start] = c["low_val"]

    overlays.append(mpf.make_addplot(high_series.interpolate(limit_area="inside"), color="red", width=1.0, linestyle="--"))
    overlays.append(mpf.make_addplot(low_series.interpolate(limit_area="inside"), color="orange", width=1.0, linestyle="--"))

    if pivot_price:
        overlays.append(mpf.make_addplot(pd.Series(pivot_price, index=df_view.index), color="magenta", width=0.8, linestyle=":"))

    closes_full = full_df["close"]
    sma150 = closes_full.rolling(150).mean().iloc[view_start:view_end]
    sma200 = closes_full.rolling(200).mean().iloc[view_start:view_end]
    if not sma150.isna().all():
        overlays.append(mpf.make_addplot(sma150, color="blue", width=1.0, linestyle="-"))
    if not sma200.isna().all():
        overlays.append(mpf.make_addplot(sma200, color="purple", width=1.0, linestyle="-"))

    return overlays


def _build_rs_overlays(
    df_view: pd.DataFrame,
    full_df: pd.DataFrame,
    view_start: int,
    view_end: int,
) -> list:
    """为 RS 策略生成 SMA200/SMA50 辅助线。"""
    overlays = []
    closes_full = full_df["close"]
    sma200 = closes_full.rolling(200).mean().iloc[view_start:view_end]
    sma50 = closes_full.rolling(50).mean().iloc[view_start:view_end]
    if not sma200.isna().all():
        overlays.append(mpf.make_addplot(sma200, color="purple", width=1.0, linestyle="-"))
    if not sma50.isna().all():
        overlays.append(mpf.make_addplot(sma50, color="blue", width=0.8, linestyle="-"))
    return overlays


def plot_multi_strategy_backtest_signals(
    trades: list[dict],
    stock_data: dict[str, pd.DataFrame],
    strategy: str,
    output_dir: Path,
) -> None:
    """为回测信号绘制 K 线图，含红圈标注信号日 + 各策略辅助线。"""
    strat_dir = output_dir / strategy
    strat_dir.mkdir(parents=True, exist_ok=True)

    ticker_trades: dict[str, list[dict]] = {}
    for t in trades:
        ticker_trades.setdefault(t["ticker"], []).append(t)

    name = _STRATEGY_NAMES_CN.get(strategy, strategy)

    for ticker, t_list in ticker_trades.items():
        if ticker not in stock_data:
            continue
        full_df = stock_data[ticker]

        signal_dates = []
        for t in t_list:
            cut_date = pd.Timestamp(t["cutoff_date"])
            idx_pos = full_df.index.searchsorted(cut_date)
            if idx_pos >= len(full_df):
                idx_pos = len(full_df) - 1
            signal_dates.append(idx_pos)

        earliest = min(signal_dates)
        latest = max(signal_dates)
        year_start = max(0, latest - 251)
        view_start = min(year_start, max(0, earliest - 10))
        view_end = min(len(full_df), latest + 40)
        df_view = full_df.iloc[view_start:view_end].copy()

        if len(df_view) < 5:
            continue

        marker_red = pd.Series(np.nan, index=df_view.index)
        for idx_pos in signal_dates:
            if view_start <= idx_pos < view_end:
                date = full_df.index[idx_pos]
                if date in marker_red.index:
                    marker_red.loc[date] = full_df.loc[date, "close"]

        last_trade = t_list[-1]
        signal_info = last_trade.get("signal_info", {})

        marker_green = pd.Series(np.nan, index=df_view.index)
        if strategy == "vcp":
            ws = signal_info.get("window_start")
            green_rel = signal_info.get("green_idx_rel")
            red_rel = signal_info.get("red_idx_rel")
            if ws is not None and green_rel is not None:
                green_abs = ws + green_rel
                if view_start <= green_abs < view_end:
                    date = full_df.index[green_abs]
                    if date in marker_green.index:
                        marker_green.loc[date] = full_df.loc[date, "high"]
            if ws is not None and red_rel is not None:
                red_abs = ws + red_rel
                if view_start <= red_abs < view_end:
                    marker_red = pd.Series(np.nan, index=df_view.index)
                    date = full_df.index[red_abs]
                    if date in marker_red.index:
                        marker_red.loc[date] = full_df.loc[date, "high"]

        add_plots = []
        try:
            if strategy == "vegas":
                add_plots.extend(_build_vegas_overlays(df_view, full_df, view_start, view_end))
            elif strategy in ("triangle", "triangle_kde", "parallel", "wedge"):
                add_plots.extend(_build_triangle_overlays(df_view, signal_info, full_df, view_start, view_end))
            elif strategy == "vcp":
                add_plots.extend(_build_vcp_overlays(df_view, signal_info, full_df, view_start, view_end))
            elif strategy in ("rs_strict", "rs_loose", "rs_trap_strict", "rs_trap_loose"):
                add_plots.extend(_build_rs_overlays(df_view, full_df, view_start, view_end))
        except Exception as e:
            logger.debug(f"{ticker}: 辅助线绘制异常 {e}")

        if not marker_green.isna().all():
            add_plots.append(mpf.make_addplot(
                marker_green, type="scatter", markersize=200, marker="o", color="green", alpha=0.8,
            ))
        add_plots.append(mpf.make_addplot(
            marker_red, type="scatter", markersize=200, marker="o", color="red", alpha=0.8,
        ))

        ret_text = []
        for pk, label in [("5d", "1W"), ("10d", "2W"), ("21d", "1M"), ("to_end", "End")]:
            t_info = last_trade.get(pk, {})
            r = t_info.get("return_pct")
            if r is not None:
                ret_text.append(f"{label}:{r:+.1f}%")
        ret_str = "  ".join(ret_text)

        name_en = _STRATEGY_NAMES_EN.get(strategy, strategy)
        if strategy in ("triangle", "triangle_kde", "parallel", "wedge"):
            pat = signal_info.get("pattern", "")
            method = signal_info.get("method", "ols")
            suffix = "(KDE)" if method == "kde" else ""
            _pat_names = {
                "ascending_triangle": "Ascending Triangle",
                "parallel_channel": "Parallel Channel",
                "rising_wedge": "Rising Wedge",
            }
            pat_label = _pat_names.get(pat, pat)
            name_en = f"{pat_label} {suffix}".strip() if suffix else pat_label
        if strategy in ("rs_strict", "rs_loose"):
            rs_r = signal_info.get("rs_rank", 0)
            accel = signal_info.get("acceleration", 0)
            name_en = f"{name_en}  RS:{rs_r:.0f}%  Accel:{accel:+.1f}%"
        elif strategy in ("rs_trap_strict", "rs_trap_loose"):
            rs_r = signal_info.get("rs_rank", 0)
            outperf = signal_info.get("outperform", 0)
            rs63 = signal_info.get("rs_chg_63d", 0)
            name_en = f"{name_en}  RS:{rs_r:.0f}%  Out:{outperf:+.1f}ppt  RS63d:{rs63:+.1f}%"
        title = f"{ticker} - {name_en}  |  {ret_str}"

        style = mpf.make_mpf_style(base_mpf_style="charles", rc={"font.size": 9})

        vlines_dates = [full_df.index[p] for p in signal_dates if view_start <= p < view_end]
        vlines_kwargs = {}
        if vlines_dates:
            vlines_kwargs = {
                "vlines": dict(
                    vlines=vlines_dates, colors="red",
                    linewidths=0.8, linestyle="--", alpha=0.5,
                ),
            }

        if strategy in ("triangle", "triangle_kde", "parallel", "wedge"):
            pat = signal_info.get("pattern", strategy)
            save_path = strat_dir / f"{ticker}_{pat}.png"
        else:
            save_path = strat_dir / f"{ticker}_{strategy}.png"

        mpf.plot(
            df_view, type="candle", volume=True, title=title, style=style,
            figscale=1.5, figratio=(16, 9), addplot=add_plots,
            savefig=str(save_path), **vlines_kwargs,
        )
        plt.close("all")

    logger.info(f"  已保存 {len(ticker_trades)} 张 {name} 信号图 → {strat_dir}")


def plot_vegas_wave_summary(df: pd.DataFrame, out_dir: Path) -> None:
    """绘制 Vegas 波段策略汇总图（score vs return 散点图 + 信号表现柱状图）。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # score vs return scatter
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, d in zip(axes, [21, 63]):
        col = f"ret_{d}d"
        vals = pd.to_numeric(df[col], errors="coerce")
        mask = vals.notna()
        x = df.loc[mask, "score"]
        y = vals[mask]
        colors = ["green" if v > 0 else "red" for v in y]
        ax.scatter(x, y, c=colors, alpha=0.4, s=30, edgecolors="none")
        means = df.loc[mask].groupby("score")[col].apply(
            lambda s: pd.to_numeric(s, errors="coerce").mean()
        )
        ax.plot(means.index, means.values, "b-o", linewidth=2, markersize=6, label="avg return")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(4, color="orange", linewidth=1, linestyle=":", label="BUY threshold")
        ax.axvline(7, color="red", linewidth=1, linestyle=":", label="STRONG_BUY threshold")
        ax.set_xlabel("Score")
        ax.set_ylabel(f"Return {d}d (%)")
        ax.set_title(f"Score vs {d}-day Return")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "score_vs_return.png", dpi=150)
    plt.close()

    # signal breakdown bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    signal_order = ["STRONG_BUY", "BUY", "HOLD", "AVOID"]
    colors_bar = ["#00AA00", "#66CC66", "#CCCC00", "#CC3333"]
    for ax, d in zip(axes, [21, 63]):
        col = f"ret_{d}d"
        wr_vals, avg_vals, labels = [], [], []
        for sig in signal_order:
            sub = df[df["signal"] == sig]
            v = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(v) == 0:
                continue
            labels.append(f"{sig}\n(n={len(v)})")
            wr_vals.append((v > 0).mean() * 100)
            avg_vals.append(v.mean())
        x_pos = range(len(labels))
        bar_colors = colors_bar[:len(labels)]
        ax.bar(x_pos, wr_vals, color=bar_colors, alpha=0.7, label="win rate %")
        ax2 = ax.twinx()
        ax2.plot(x_pos, avg_vals, "ko-", linewidth=2, label="avg return %")
        ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Win Rate (%)")
        ax2.set_ylabel("Avg Return (%)")
        ax.set_title(f"{d}-day Performance by Signal")
        ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_dir / "signal_performance.png", dpi=150)
    plt.close()
    logger.info(f"汇总图已保存 → {out_dir}")


def plot_triangle_vcp_backtest_signals(
    signals: list[dict],
    stock_data: dict[str, pd.DataFrame],
    output_dir: Path,
    max_charts: int = 50,
) -> None:
    """为 VCP 三角形信号绘制 K 线图（含阻力线/支撑线/前高标记）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    to_plot = sorted(signals, key=lambda x: x["signal_date"])[:max_charts]
    chart_count = 0

    _CN_TVCP = {
        "ascending_triangle": "上升三角形(VCP)",
        "descending_triangle": "下降三角形(VCP)",
    }

    for s in to_plot:
        ticker = s["ticker"]
        if ticker not in stock_data:
            continue

        full_df = stock_data[ticker]
        sig_iloc = s["signal_iloc"]
        ws_iloc = s["window_start_iloc"]
        period = s["period"]
        peak_iloc = s["peak_iloc"]

        view_start = max(0, peak_iloc - 30)
        view_end = min(len(full_df), sig_iloc + 40)
        df_view = full_df.iloc[view_start:view_end].copy()

        if len(df_view) < 20:
            continue

        add_plots = []

        closes_full = full_df["close"]
        for ma_len, color in [(50, "green"), (150, "blue"), (200, "purple")]:
            ma = closes_full.rolling(ma_len).mean().iloc[view_start:view_end]
            if not ma.isna().all():
                add_plots.append(mpf.make_addplot(ma, color=color, width=0.8))

        offset = ws_iloc - view_start
        n_display = len(df_view)
        res_line = pd.Series(np.nan, index=df_view.index)
        sup_line = pd.Series(np.nan, index=df_view.index)

        for i in range(n_display):
            x = i - offset
            if 0 <= x < period:
                res_line.iloc[i] = s["resistance_slope"] * x + s["resistance_intercept"]
                sup_line.iloc[i] = s["support_slope"] * x + s["support_intercept"]

        add_plots.append(mpf.make_addplot(res_line, color="red", width=2.0, linestyle="dashed"))
        add_plots.append(mpf.make_addplot(sup_line, color="lime", width=2.0, linestyle="dashed"))

        peak_marker = pd.Series(np.nan, index=df_view.index)
        if view_start <= peak_iloc < view_end:
            pk_date = full_df.index[peak_iloc]
            peak_marker.loc[pk_date] = full_df.loc[pk_date, "high"]
        if peak_marker.notna().any():
            add_plots.append(mpf.make_addplot(
                peak_marker, type="scatter", markersize=250,
                marker="v", color="blue", edgecolors="darkblue",
                linewidths=2, alpha=0.9,
            ))

        sig_marker = pd.Series(np.nan, index=df_view.index)
        if view_start <= sig_iloc < view_end:
            sig_date = full_df.index[sig_iloc]
            sig_marker.loc[sig_date] = full_df.loc[sig_date, "close"]
        if sig_marker.notna().any():
            add_plots.append(mpf.make_addplot(
                sig_marker, type="scatter", markersize=200,
                marker="o", color="red", edgecolors="darkred",
                linewidths=2, alpha=0.9,
            ))

        ret_parts = []
        for pk, label in [("5d", "1W"), ("21d", "1M"), ("63d", "3M")]:
            info = s.get(pk, {})
            r = info.get("return_pct")
            if r is not None:
                ret_parts.append(f"{label}:{r:+.1f}%")

        cn = _CN_TVCP.get(s["pattern"], s["pattern"])
        title = (
            f"{ticker} {cn} ({s['period']}d from peak {s['peak_date']})  "
            f"收敛{s['convergence_ratio']:.0%} "
            f"波幅缩{1 - s['spread_contraction']:.0%}  |  "
            f"{'  '.join(ret_parts)}"
        )

        style = mpf.make_mpf_style(base_mpf_style="charles", rc={"font.size": 9})
        save_path = output_dir / f"{ticker}_tri_vcp_{s['signal_date']}.png"

        try:
            mpf.plot(
                df_view, type="candle", volume=True,
                title=title, style=style,
                figscale=1.5, figratio=(16, 9),
                addplot=add_plots,
                savefig=str(save_path),
                warn_too_much_data=len(df_view) + 1,
            )
            plt.close("all")
            chart_count += 1
        except Exception as e:
            logger.debug(f"{ticker}: 绘图失败 - {e}")

    logger.info(f"  已保存 {chart_count} 张信号图 → {output_dir}")
