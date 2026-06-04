"""为 Futu 自选股生成带评分的 SMC Order Block 历史图表。

与每日扫描不同，此脚本保留所有 OB（包括已被突破的），
以直观验证评分高低与实际支撑/阻力效果的对应关系。

  - 活跃 OB：实线矩形，按分数着色（绿→黄→橙→红 四档）
  - 已突破 OB：虚线矩形，低透明度，以灰色显示
  - 同方向重叠 OB 的 zone_score 叠加标注

用法:
    python scripts/gen_ob_score_charts.py                   # futu 自选全量
    python scripts/gen_ob_score_charts.py NVDA AAOI 00700   # 指定股票
    python scripts/gen_ob_score_charts.py --lookback 300    # 自定义回看
    python scripts/gen_ob_score_charts.py --market us       # 仅美股

输出目录: data/output/ob_score_charts/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mplfinance as mpf
import numpy as np
import pandas as pd
from loguru import logger

# ── 路径引导 ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import os
os.environ.setdefault("SMC_CREDIT", "0")
from smartmoneyconcepts import smc as upstream_smc          # noqa: E402

from stock_ana.config import CACHE_DIR, OUTPUT_DIR          # noqa: E402
from stock_ana.strategies.impl.smc import (                 # noqa: E402
    _ob_causal,
    ob_quality_rating,
    ob_quality_score,
)
from stock_ana.data.list_manager import parse_watchlist      # noqa: E402

# 触发 CJK 字体注册
import stock_ana.utils.plot_renderers  # noqa: F401

# ── 常量 ─────────────────────────────────────────────────────────────────────
OUT_DIR = OUTPUT_DIR / "ob_score_charts"
LOOKBACK_BARS = 500
SWING_LENGTH = 5

# 评分颜色映射（bull / bear 分别有不同色系）
# (score_min, facecolor, edgecolor)
_BULL_TIERS = [
    (70, "#006400B0", "#004400"),   # 极强 — 深绿
    (50, "#009900A0", "#006600"),   # 强   — 绿
    (30, "#66BB6A90", "#338833"),   # 中   — 浅绿
    ( 0, "#A5D6A780", "#558855"),   # 弱   — 很浅绿
]
_BEAR_TIERS = [
    (70, "#B71C1CB0", "#880000"),   # 极强 — 深红
    (50, "#E53935A0", "#AA0000"),   # 强   — 红
    (30, "#EF9A9A90", "#883333"),   # 中   — 浅红
    ( 0, "#FFCDD280", "#885555"),   # 弱   — 很浅红
]
_MITIGATED_FC = "#CCCCCC40"
_MITIGATED_EC = "#999999"


def _tier_colors(score: float, is_bull: bool) -> tuple[str, str]:
    """根据分数和方向返回 (facecolor, edgecolor)。"""
    tiers = _BULL_TIERS if is_bull else _BEAR_TIERS
    for min_score, fc, ec in tiers:
        if score >= min_score:
            return fc, ec
    return tiers[-1][1], tiers[-1][2]


# ═════════════════════════════════════════════════════════════════════════════
# 数据加载
# ═════════════════════════════════════════════════════════════════════════════

def _load_cache(sym: str, market: str) -> pd.DataFrame | None:
    path = CACHE_DIR / market / f"{sym}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 计算所有 OB 并评分
# ═════════════════════════════════════════════════════════════════════════════

def _compute_scored_obs(
    df: pd.DataFrame,
    swing_length: int = 5,
) -> list[dict]:
    """用因果 OB 检测 + 评分，返回全部 OB（含已突破）。"""
    swing_hl = upstream_smc.swing_highs_lows(df, swing_length=swing_length)
    ob_df = _ob_causal(df, swing_hl, swing_length=swing_length)

    obs: list[dict] = []
    for bar_idx in range(len(ob_df)):
        row = ob_df.iloc[bar_idx]
        if pd.isna(row.get("OB")) or row["OB"] == 0:
            continue

        direction = int(row["OB"])
        ts = df.index[bar_idx]
        formed_date = str(ts.date()) if hasattr(ts, "date") else str(ts)

        mit_idx = row["MitigatedIndex"]
        is_mitigated = pd.notna(mit_idx) and int(mit_idx) != 0
        mitigated_bar: int | None = None
        if is_mitigated:
            mitigated_bar = int(mit_idx)

        score, detail = ob_quality_rating(df, ob_df, bar_idx)

        obs.append({
            "bar_idx":       bar_idx,
            "direction":     direction,
            "top":           float(row["Top"]),
            "bottom":        float(row["Bottom"]),
            "formed_date":   formed_date,
            "score":         score,
            "score_detail":  detail,
            "is_mitigated":  is_mitigated,
            "mitigated_bar": mitigated_bar,
        })

    # Zone score 叠加（仅活跃 OB 参与）
    _apply_zone_scores(obs)
    return obs


def _apply_zone_scores(obs: list[dict]) -> None:
    """对同方向、价格重叠的活跃 OB 叠加 zone_score。"""
    for direction in (1, -1):
        active = [o for o in obs if o["direction"] == direction and not o["is_mitigated"]]
        if not active:
            continue
        active.sort(key=lambda o: o["bottom"])

        zones: list[list[dict]] = []
        cur_zone = [active[0]]
        cur_top = active[0]["top"]
        for o in active[1:]:
            if o["bottom"] <= cur_top:
                cur_zone.append(o)
                cur_top = max(cur_top, o["top"])
            else:
                zones.append(cur_zone)
                cur_zone = [o]
                cur_top = o["top"]
        zones.append(cur_zone)

        for zone in zones:
            ztotal = round(sum(o["score"] for o in zone), 1)
            for o in zone:
                o["zone_score"] = ztotal

    # 已突破 OB 的 zone_score = 自身 score
    for o in obs:
        if "zone_score" not in o:
            o["zone_score"] = o["score"]


# ═════════════════════════════════════════════════════════════════════════════
# 绘图
# ═════════════════════════════════════════════════════════════════════════════

_MPF_STYLE = mpf.make_mpf_style(
    base_mpf_style="charles",
    rc={
        "font.sans-serif": plt.rcParams["font.sans-serif"],
        "axes.unicode_minus": False,
    },
)


def _plot_ob_scored(
    sym: str,
    market: str,
    name: str,
    df: pd.DataFrame,
    lookback: int,
    swing_length: int,
    out_dir: Path,
) -> Path | None:
    """生成带评分的 SMC OB 图表，所有 OB（含已突破）都画出来。"""
    # 全量数据计算 OB（因果检测需要完整历史）
    all_obs = _compute_scored_obs(df, swing_length=swing_length)
    if not all_obs:
        logger.info(f"{sym}: 无 OB，跳过")
        return None

    # 截取显示窗口
    n_total = len(df)
    view_start = max(0, n_total - lookback)
    df_view = df.iloc[view_start:].copy()
    n_view = len(df_view)
    if n_view < 30:
        return None

    close_last = float(df_view["close"].iloc[-1])
    date_last = df_view.index[-1].strftime("%Y-%m-%d")

    # 筛选在显示窗口内生成 或 矩形仍延伸到窗口内的 OB
    vis_obs = []
    for o in all_obs:
        ob_start = o["bar_idx"]
        ob_end = o["mitigated_bar"] if o["is_mitigated"] else n_total - 1
        if ob_end is None:
            ob_end = n_total - 1
        # OB 矩形是否与视窗交叠
        if ob_end >= view_start and ob_start < n_total:
            vis_obs.append(o)

    if not vis_obs:
        logger.info(f"{sym}: 窗口内无 OB，跳过")
        return None

    # ── mplfinance 主图 ──────────────────────────────────────────────────────
    fig, axes = mpf.plot(
        df_view,
        type="candle",
        volume=True,
        style=_MPF_STYLE,
        figsize=(22, 10),
        returnfig=True,
        tight_layout=False,
        scale_padding={"left": 0.05, "right": 0.12, "top": 0.7, "bottom": 0.5},
    )
    ax: plt.Axes = axes[0]

    # ── 叠加 OB 矩形 ────────────────────────────────────────────────────────
    bull_active = bull_mit = bear_active = bear_mit = 0
    is_bull = None

    for o in vis_obs:
        is_bull = o["direction"] == 1
        top = o["top"]
        bot = o["bottom"]
        score = o["score"]
        zone_score = o["zone_score"]

        # 矩形坐标（相对于 view 的 x 位置）
        rect_start = max(o["bar_idx"] - view_start, 0)
        if o["is_mitigated"] and o["mitigated_bar"] is not None:
            rect_end = min(o["mitigated_bar"] - view_start, n_view)
        else:
            rect_end = n_view

        rect_width = max(rect_end - rect_start, 1)

        if o["is_mitigated"]:
            fc, ec = _MITIGATED_FC, _MITIGATED_EC
            ls = ":"
            lw = 0.6
            if is_bull:
                bull_mit += 1
            else:
                bear_mit += 1
        else:
            fc, ec = _tier_colors(score, is_bull)
            ls = "-"
            lw = 1.2
            if is_bull:
                bull_active += 1
            else:
                bear_active += 1

        rect = mpatches.FancyBboxPatch(
            (rect_start - 0.4, bot),
            rect_width + 0.8,
            top - bot,
            boxstyle="round,pad=0",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            linestyle=ls,
            zorder=2,
        )
        ax.add_patch(rect)

        # 分数标注（在矩形右端）
        label_x = min(rect_start + rect_width + 0.5, n_view + 0.5)
        label_y = (top + bot) / 2
        tag = "▲" if is_bull else "▼"

        if o["is_mitigated"]:
            label = f"{tag}{score:.0f}"
            ax.text(label_x, label_y, label,
                    fontsize=5.5, va="center", ha="left",
                    color="#999999", clip_on=True)
        else:
            # 活跃 OB — 如果 zone_score > score 说明有叠加
            if zone_score > score + 0.5:
                label = f"{tag}{score:.0f} z={zone_score:.0f}"
            else:
                label = f"{tag}{score:.0f}"
            ax.text(label_x, label_y, label,
                    fontsize=7, va="center", ha="left",
                    color=ec, fontweight="bold", clip_on=True)

    # ── 当前价格线 ───────────────────────────────────────────────────────────
    ax.axhline(close_last, color="#888888", linewidth=0.8,
               linestyle="--", zorder=1, alpha=0.7)
    ax.text(n_view * 0.01, close_last, f"  {close_last:.2f}",
            fontsize=8, va="bottom", color="#555555")

    # ── 图例 ─────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor="#006400B0", edgecolor="#004400", label="Bull ≥70 (极强)"),
        mpatches.Patch(facecolor="#009900A0", edgecolor="#006600", label="Bull ≥50 (强)"),
        mpatches.Patch(facecolor="#66BB6A90", edgecolor="#338833", label="Bull ≥30 (中)"),
        mpatches.Patch(facecolor="#A5D6A780", edgecolor="#558855", label="Bull <30 (弱)"),
        mpatches.Patch(facecolor="#B71C1CB0", edgecolor="#880000", label="Bear ≥70"),
        mpatches.Patch(facecolor="#E53935A0", edgecolor="#AA0000", label="Bear ≥50"),
        mpatches.Patch(facecolor="#EF9A9A90", edgecolor="#883333", label="Bear ≥30"),
        mpatches.Patch(facecolor="#CCCCCC40", edgecolor="#999999", label="已突破"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=7,
              ncol=4, framealpha=0.8)

    # ── 标题 ─────────────────────────────────────────────────────────────────
    market_tag = market.upper()
    title_main = f"{market_tag}:{sym}  {name}    SMC Order Block 评分图"
    title_sub = (
        f"swing_length={swing_length}  |  "
        f"看涨: {bull_active}活跃 + {bull_mit}已突破  "
        f"看跌: {bear_active}活跃 + {bear_mit}已突破  |  "
        f"最新收盘={close_last:.2f}  @{date_last}"
    )
    fig.suptitle(title_main, fontsize=15, fontweight="bold", y=0.99)
    ax.set_title(title_sub, fontsize=9, color="#555555", pad=3)

    # ── 保存 ─────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    mkt_prefix = {"hk": "HK_", "cn": "CN_"}.get(market.lower(), "")
    save_path = out_dir / f"{mkt_prefix}{sym}_ob_score.png"
    fig.savefig(str(save_path), dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# 主入口
# ═════════════════════════════════════════════════════════════════════════════

def _build_target_list(args) -> list[tuple[str, str, str]]:
    """从 futu 自选或命令行参数构建 (symbol, market, name) 列表。"""
    wl = parse_watchlist()  # 自动读 futu_watchlist.md

    targets: list[tuple[str, str, str]] = []
    for mkt in ("us", "hk", "cn"):
        if args.market and args.market.lower() != mkt:
            continue
        for item in wl.get(mkt, []):
            sym = item["symbol"]
            name = item.get("name", sym)
            targets.append((sym, mkt, name))

    if args.symbols:
        sym_set = {s.upper() for s in args.symbols}
        # 先从 watchlist 里找
        filtered = [t for t in targets if t[0].upper() in sym_set]
        found = {t[0].upper() for t in filtered}
        # 找不到的从缓存目录探测
        for raw in args.symbols:
            if raw.upper() in found:
                continue
            for mkt in ("us", "hk", "cn"):
                p = CACHE_DIR / mkt / f"{raw}.parquet"
                if not p.exists() and mkt == "hk":
                    p = CACHE_DIR / mkt / f"{raw.zfill(5)}.parquet"
                    if p.exists():
                        filtered.append((raw.zfill(5), mkt, raw.zfill(5)))
                        found.add(raw.upper())
                        break
                if p.exists():
                    filtered.append((raw, mkt, raw))
                    found.add(raw.upper())
                    break
        targets = filtered

    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Futu 自选 SMC OB 评分图表")
    parser.add_argument("symbols", nargs="*", help="指定股票代码（不填则全部自选）")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_BARS,
                        help=f"显示最近 N 根 K 线（默认 {LOOKBACK_BARS}）")
    parser.add_argument("--swing_length", type=int, default=SWING_LENGTH)
    parser.add_argument("--market", type=str, default="",
                        help="只处理指定市场（us/hk/cn）")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="输出目录")
    args = parser.parse_args()

    targets = _build_target_list(args)
    if not targets:
        logger.error("未找到目标股票")
        sys.exit(1)

    logger.info(f"共 {len(targets)} 只股票，lookback={args.lookback}")
    logger.info(f"输出目录: {args.out}")

    ok = fail = skip = 0
    for i, (sym, market, name) in enumerate(targets, 1):
        logger.info(f"[{i}/{len(targets)}] {market.upper()}:{sym} {name}")
        df = _load_cache(sym, market)
        if df is None:
            logger.debug(f"  缓存不存在，跳过")
            skip += 1
            continue
        path = _plot_ob_scored(sym, market, name, df, args.lookback,
                               args.swing_length, args.out)
        if path:
            logger.info(f"  → {path.name}")
            ok += 1
        else:
            fail += 1

    logger.info(f"\n完成: {ok} 张图  跳过: {skip}  失败: {fail}")
    logger.info(f"输出目录: {args.out}")


if __name__ == "__main__":
    main()
