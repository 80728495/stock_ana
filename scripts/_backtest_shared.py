"""
_backtest_shared.py
顶部反转系列回测脚本的共用基础设施。
- CJK 字体检测（用 FontProperties(fname=...) 避免名称查找失效）
- _load_ohlcv / _parse_holding_md
- _make_ema_addplots
- _stats_block（标准化汇总打印）
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

# ── matplotlib 字体配置 ──────────────────────────────────────────────────────
_CJK_CANDIDATES = [
    "Heiti TC", "PingFang HK", "PingFang SC",
    "STHeiti", "Songti SC", "Arial Unicode MS",
    "WenQuanYi Micro Hei", "Noto Sans CJK SC",
]


def _find_cjk_font_path() -> str | None:
    """从系统字体中找到第一个支持 CJK 的字体文件路径。"""
    name_to_path: dict[str, str] = {f.name: f.fname for f in fm.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in name_to_path:
            return name_to_path[name]
    return None


_CJK_FONT_PATH: str | None = _find_cjk_font_path()

# 全局 rcParams（fallback 用，不依赖 fontfamily 参数）
plt.rcParams["font.sans-serif"] = _CJK_CANDIDATES + ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def cjk_font_prop(size: float = 8.0) -> fm.FontProperties:
    """返回一个支持 CJK 的 FontProperties 对象。优先使用实际文件路径。"""
    if _CJK_FONT_PATH:
        return fm.FontProperties(fname=_CJK_FONT_PATH, size=size)
    return fm.FontProperties(size=size)


# ── EMA 颜色配置 ──────────────────────────────────────────────────────────────
EMA_COLORS: dict[int, str] = {
    34: "#4FC3F7",
    55: "#29B6F6",
    144: "#FF8A65",
    200: "#E64A19",
}


def make_ema_addplots(df: pd.DataFrame, df_plot: pd.DataFrame) -> list:
    """生成 EMA 叠加图层列表（供 mplfinance addplot 参数使用）。"""
    plots = []
    for span, color in EMA_COLORS.items():
        ema = df["close"].ewm(span=span, adjust=False).mean().reindex(df_plot.index)
        plots.append(mpf.make_addplot(
            ema, color=color, width=1.0,
            linestyle="-" if span >= 144 else "--",
        ))
    return plots


# ── 数据加载 ─────────────────────────────────────────────────────────────────

def load_ohlcv(sym: str, market: str, cache_dir_map: dict[str, Path]) -> pd.DataFrame | None:
    path = cache_dir_map.get(market.upper(), next(iter(cache_dir_map.values()))) / f"{sym}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df.sort_index()
    except Exception:
        return None


def parse_holding_md(holding_path: Path) -> list[dict]:
    """解析 holding.md，返回所有标的（持仓+关注+观察）。"""
    text = holding_path.read_text(encoding="utf-8")
    items: list[dict] = []
    current_section = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            current_section = stripped
            continue
        if not stripped.startswith("|"):
            continue
        parts = [p.strip() for p in stripped.strip("|").split("|")]
        if len(parts) < 2:
            continue
        sym = parts[0]
        if not sym or re.fullmatch(r"[-: ]+", sym) or sym in ("代码", "Code"):
            continue
        name = parts[2] if len(parts) > 2 else parts[1]
        if re.search(r"[A-Za-z]+\d{4}$", sym):  # 期货
            continue

        if "持仓" in current_section or "Holdings" in current_section:
            market = parts[1] if len(parts) > 1 else "US"
            section = "holdings"
        elif "关注" in current_section or "Focus" in current_section:
            market = parts[1] if len(parts) > 1 else "CN"
            section = "focus"
        elif "观察" in current_section or "Watch" in current_section:
            market = parts[1] if len(parts) > 1 else "US"
            section = "watch"
        else:
            continue
        items.append({"sym": sym, "market": market.upper(), "name": name, "section": section})

    seen: set[str] = set()
    deduped: list[dict] = []
    for it in items:
        key = f"{it['market']}:{it['sym']}"
        if key not in seen:
            seen.add(key)
            deduped.append(it)
    return deduped


# ── 汇总统计打印 ──────────────────────────────────────────────────────────────

def print_stats(sub: pd.DataFrame, label: str) -> None:
    """标准化打印一组信号的回测统计。"""
    if sub.empty:
        return
    n = len(sub)
    a5  = (sub["fwd_min_5d"]  <= -2.0).sum()
    a10 = (sub["fwd_ret_10d"] <= -3.0).sum()
    a20 = (sub["fwd_ret_20d"] <= -5.0).sum()
    r5  = sub["fwd_ret_5d"].mean()
    m5  = sub["fwd_min_5d"].mean()
    r20 = sub["fwd_ret_20d"].mean()
    print(f"\n  [{label}]  n={n}")
    print(f"    准确率  5d>2%跌: {a5/n*100:5.1f}%  10d>3%跌: {a10/n*100:5.1f}%  20d>5%跌: {a20/n*100:5.1f}%")
    print(f"    均值    5d收盘: {r5:+.1f}%  5d最低: {m5:+.1f}%  20d收盘: {r20:+.1f}%")


def print_per_sym_table(filt: pd.DataFrame) -> None:
    """打印各标的信号明细表。"""
    print("\n\n  各标的信号明细：")
    print(f"  {'市场':<4} {'代码':<8} {'名称':<12} {'信号数':>4} {'5d准确%':>7} {'20d准确%':>8}  {'5d均值':>7} {'20d均值':>8}")
    print("  " + "-" * 68)
    per_sym = (
        filt.groupby(["market", "sym", "name"])
        .apply(lambda g: pd.Series({
            "n":     len(g),
            "acc5":  (g["fwd_min_5d"]  <= -2.0).mean() * 100,
            "acc20": (g["fwd_ret_20d"] <= -5.0).mean() * 100,
            "ret5":  g["fwd_ret_5d"].mean(),
            "ret20": g["fwd_ret_20d"].mean(),
        }), include_groups=False)
        .reset_index()
        .sort_values("acc5", ascending=False)
    )
    for _, r in per_sym.iterrows():
        print(
            f"  {r['market']:<4} {r['sym']:<8} {str(r['name'])[:12]:<12}"
            f"  {int(r['n']):>4}  {r['acc5']:>6.0f}%  {r['acc20']:>7.0f}%"
            f"  {r['ret5']:>+6.1f}%  {r['ret20']:>+7.1f}%"
        )


# ── 图表基础构建 ──────────────────────────────────────────────────────────────

def build_base_chart(
    df: pd.DataFrame,
    df_plot: pd.DataFrame,
    add_plots: list,
    marker_series: pd.Series,
) -> tuple:
    """用 mplfinance 构建基础 K 线图，返回 (fig, axes)。

    marker_series: 与 df_plot 同 index 的 Series，有信号的位置填 high 值，其余 NaN。
    """
    all_add = list(add_plots)
    if marker_series.notna().any():
        all_add.append(mpf.make_addplot(
            marker_series, type="scatter", marker="v",
            markersize=80, color="#FF4444",
        ))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        fig, axes = mpf.plot(
            df_plot, type="candle", style="yahoo",
            addplot=all_add if all_add else None,
            volume=True,
            figratio=(22, 10), figscale=1.2,
            tight_layout=False,
            returnfig=True,
            warn_too_much_data=9999,
        )
    return fig, axes


def add_chart_legend(ax, strategy_label: str = "Day-1 信号") -> None:
    """添加 EMA 图例 + 信号标记图例。"""
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=c, linewidth=1.5, label=f"EMA{s}")
        for s, c in EMA_COLORS.items()
    ]
    handles.append(Line2D(
        [0], [0], marker="v", color="w",
        markerfacecolor="#FF4444", markeredgecolor="#990000",
        markersize=9, label=strategy_label,
    ))
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.8)


def add_stat_panel(
    ax,
    panel_lines: list[str],
    bg_color: str = "#FFF3E0",
    edge_color: str = "#FFCC80",
) -> None:
    """在图表右侧添加统计信息文本框（使用 CJK 字体属性）。"""
    ax.text(
        1.005, 0.99, "\n".join(panel_lines),
        transform=ax.transAxes,
        ha="left", va="top",
        fontproperties=cjk_font_prop(size=7.5),
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=bg_color, alpha=0.93,
            edgecolor=edge_color, linewidth=0.8,
        ),
    )


def set_chart_title(fig, title: str) -> None:
    """设置图表大标题（使用 CJK 字体）。"""
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98,
                 fontproperties=cjk_font_prop(size=15))


def safe_save(fig, path: Path, dpi: int = 130) -> None:
    """保存图表，抑制字体缺字形警告。"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Glyph.*missing from font")
        warnings.filterwarnings("ignore")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
