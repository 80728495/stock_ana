"""为持仓股生成 SMC Order Block 图表。

用法:
    python scripts/gen_smc_ob_charts.py            # 默认：持仓股
    python scripts/gen_smc_ob_charts.py --lookback 500   # 自定义 K 线回看数
    python scripts/gen_smc_ob_charts.py NVDA AAOI  # 只生成指定股票

输出目录: data/output/smc_holding/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from loguru import logger

# ── 路径引导 ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_ana.config import CACHE_DIR, OUTPUT_DIR            # noqa: E402
from stock_ana.strategies.impl.smc import compute_smc_full   # noqa: E402

# ── 字体注册：让 matplotlib 能渲染中文 ──────────────────────────────────────
# plot_renderers 在导入时已设置 plt.rcParams，这里仅做保险式补充
import stock_ana.utils.plot_renderers  # noqa: F401  side-effect: sets rcParams

# ── 常量 ─────────────────────────────────────────────────────────────────────
OUT_DIR = OUTPUT_DIR / "smc_holding"
LOOKBACK_BARS = 500          # 显示最近 N 根 K 线（约 2 年）
SWING_LENGTH = 5             # 摆动点灵敏度

# 持仓列表（排除无缓存的 MSFU、SOXS）
HOLDINGS: list[tuple[str, str, str]] = [
    # (symbol, market, name)
    ("00981", "hk", "中芯国际"),
    ("01347", "hk", "华虹半导体"),
    ("01810", "hk", "小米集团-W"),
    ("02400", "hk", "心动公司"),
    ("02631", "hk", "天岳先进"),
    ("03690", "hk", "美团-W"),
    ("03896", "hk", "金山云"),
    ("06869", "hk", "长飞光纤光缆"),
    ("09988", "hk", "阿里巴巴-W"),
    ("09992", "hk", "泡泡玛特"),
    ("AAOI", "us", "Applied Optoelectronics"),
    ("AMD",  "us", "超微半导体"),
    ("AXTI", "us", "AXT Inc"),
    ("BWXT", "us", "BWX Technologies"),
    ("DOCN", "us", "DigitalOcean"),
    ("DT",   "us", "Dynatrace"),
    ("GLW",  "us", "康宁"),
    ("GOOG", "us", "谷歌-C"),
    ("MSFT", "us", "微软"),
    ("MU",   "us", "美光科技"),
    ("NVDA", "us", "英伟达"),
    ("PDD",  "us", "拼多多"),
    ("TEM",  "us", "Tempus AI"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 加载缓存数据
# ─────────────────────────────────────────────────────────────────────────────

def _load_cache(sym: str, market: str) -> pd.DataFrame | None:
    path = CACHE_DIR / market / f"{sym}.parquet"
    if not path.exists():
        logger.warning(f"缓存不存在: {path}")
        return None
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        logger.warning(f"{sym} 缺少必要列: {required - set(df.columns)}")
        return None
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────────────────────────────────────

_MPF_STYLE = mpf.make_mpf_style(
    base_mpf_style="charles",
    rc={
        "font.sans-serif": plt.rcParams["font.sans-serif"],
        "axes.unicode_minus": False,
    },
)


def _plot_smc_ob(
    sym: str,
    market: str,
    name: str,
    df: pd.DataFrame,
    lookback: int,
    swing_length: int,
    out_dir: Path,
) -> Path | None:
    """生成单只股票的 SMC Order Block K 线图，返回保存路径。"""
    df_view = df.iloc[-lookback:].copy()
    if len(df_view) < swing_length * 2 + 20:
        logger.warning(f"{sym} 数据不足，跳过")
        return None

    # ── 计算 SMC 指标 ──────────────────────────────────────────────────────
    try:
        smc_res = compute_smc_full(df_view, swing_length=swing_length)
    except Exception as e:
        logger.error(f"{sym} SMC 计算失败: {e}")
        return None

    ob_df   = smc_res["ob"]
    bos_df  = smc_res["bos_choch"]
    shl_df  = smc_res["swing_hl"]

    n = len(df_view)
    close_last = float(df_view["close"].iloc[-1])
    date_last  = df_view.index[-1].strftime("%Y-%m-%d")

    # ── 准备 addplot：摆动高低点标记 ────────────────────────────────────────
    add_plots = []

    # 摆动高点（绿三角）/ 低点（红三角）
    swing_high_vals = pd.Series(np.nan, index=range(n))
    swing_low_vals  = pd.Series(np.nan, index=range(n))
    for idx, row in shl_df.iterrows():
        if pd.notna(row.get("HighLow")) and row["HighLow"] == 1:
            swing_high_vals.iloc[int(idx)] = float(row["Level"]) * 1.005
        elif pd.notna(row.get("HighLow")) and row["HighLow"] == -1:
            swing_low_vals.iloc[int(idx)] = float(row["Level"]) * 0.995

    if swing_high_vals.notna().any():
        add_plots.append(mpf.make_addplot(
            swing_high_vals, type="scatter", markersize=40,
            marker="v", color="#CC0000", panel=0,
        ))
    if swing_low_vals.notna().any():
        add_plots.append(mpf.make_addplot(
            swing_low_vals, type="scatter", markersize=40,
            marker="^", color="#009900", panel=0,
        ))

    # ── 绘制蜡烛图主图 ───────────────────────────────────────────────────────
    # 注意：不传 title= 给 mpf.plot()，避免 fig.suptitle 绕过字体配置
    fig, axes = mpf.plot(
        df_view,
        type="candle",
        volume=True,
        style=_MPF_STYLE,
        addplot=add_plots if add_plots else None,
        figsize=(20, 9),
        returnfig=True,
        tight_layout=False,
        scale_padding={"left": 0.05, "right": 0.05, "top": 0.7, "bottom": 0.5},
    )
    ax: plt.Axes = axes[0]

    # ── 叠加 Order Block 矩形 ────────────────────────────────────────────────
    bull_count = bear_count = 0
    for bar_idx, row in ob_df.iterrows():
        if pd.isna(row.get("OB")) or row["OB"] == 0:
            continue
        is_bull = row["OB"] == 1
        top  = float(row["Top"])
        bot  = float(row["Bottom"])
        is_broken = (
            pd.notna(row.get("MitigatedIndex")) and row["MitigatedIndex"] != 0
        )
        # 已消除的 OB 不画（避免干扰视线）
        if is_broken:
            continue
        if is_bull:
            fc = "#00880050"
            ec = "#006600"
            bull_count += 1
        else:
            fc = "#CC000050"
            ec = "#880000"
            bear_count += 1

        rect = plt.Rectangle(
            (int(bar_idx) - 0.4, bot),
            width=n - int(bar_idx) + 0.8,
            height=top - bot,
            facecolor=fc,
            edgecolor=ec,
            linewidth=0.8,
            zorder=2,
        )
        ax.add_patch(rect)

        # 价格标注
        label = f"{'↑OB' if is_bull else '↓OB'} {top:.2f}/{bot:.2f}"
        ax.text(
            n + 0.5, (top + bot) / 2,
            label,
            fontsize=6.5, va="center", ha="left",
            color=ec, clip_on=False,
        )

    # ── 当前价水平线 ─────────────────────────────────────────────────────────
    ax.axhline(close_last, color="#888888", linewidth=0.8,
               linestyle="--", zorder=1, alpha=0.7)
    ax.text(
        n * 0.01, close_last,
        f"  {close_last:.2f}",
        fontsize=8, va="bottom", color="#555555",
    )

    # ── 标题（用 fig.suptitle，已通过 rcParams 注册中文字体）───────────────
    market_tag = market.upper()
    title_main = f"{market_tag}:{sym}  {name}    SMC Order Block"
    title_sub  = (
        f"swing_length={swing_length}  |  "
        f"看涨OB={bull_count}  看跌OB={bear_count}  |  "
        f"最新收盘={close_last:.2f}  @{date_last}"
    )
    fig.suptitle(title_main, fontsize=15, fontweight="bold", y=0.99)
    ax.set_title(title_sub, fontsize=9, color="#555555", pad=3)

    # ── 保存 ────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{sym}_smc_ob.png"
    save_path = out_dir / fname
    fig.savefig(str(save_path), dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="生成持仓 SMC OB 图表")
    parser.add_argument(
        "symbols", nargs="*",
        help="指定股票代码（不填则处理全部持仓）",
    )
    parser.add_argument(
        "--lookback", type=int, default=LOOKBACK_BARS,
        help=f"显示最近 N 根 K 线（默认 {LOOKBACK_BARS}）",
    )
    parser.add_argument(
        "--swing_length", type=int, default=SWING_LENGTH,
        help=f"摆动点灵敏度（默认 {SWING_LENGTH}）",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_DIR,
        help="输出目录",
    )
    args = parser.parse_args()

    # 过滤目标列表
    targets = HOLDINGS
    if args.symbols:
        sym_set = {s.upper() for s in args.symbols}
        targets = [h for h in HOLDINGS if h[0].upper() in sym_set]
        # 未在 HOLDINGS 中的符号 → 自动从 cache 目录探测市场
        found_syms = {t[0].upper() for t in targets}
        for raw_sym in args.symbols:
            sym = raw_sym.upper()
            if sym in found_syms:
                continue
            detected: tuple[str, str, str] | None = None
            for mkt in ("us", "hk", "cn"):
                p = CACHE_DIR / mkt / f"{raw_sym}.parquet"
                if not p.exists() and mkt == "hk":
                    # HK 代码可能需要补零
                    p = CACHE_DIR / mkt / f"{raw_sym.zfill(5)}.parquet"
                    if p.exists():
                        detected = (raw_sym.zfill(5), mkt, raw_sym.zfill(5))
                        break
                if p.exists():
                    detected = (raw_sym, mkt, raw_sym)
                    break
            if detected:
                targets.append(detected)
            else:
                logger.warning(f"找不到缓存，跳过: {raw_sym}")
        if not targets:
            logger.error(f"未找到指定股票: {args.symbols}")
            sys.exit(1)

    logger.info(f"共 {len(targets)} 只股票，lookback={args.lookback}，swing_length={args.swing_length}")
    logger.info(f"输出目录: {args.out}")

    ok = fail = skip = 0
    for i, (sym, market, name) in enumerate(targets, 1):
        logger.info(f"[{i}/{len(targets)}] {market.upper()}:{sym} {name}")
        df = _load_cache(sym, market)
        if df is None:
            skip += 1
            continue
        path = _plot_smc_ob(sym, market, name, df, args.lookback, args.swing_length, args.out)
        if path:
            logger.info(f"  → {path.name}")
            ok += 1
        else:
            fail += 1

    logger.info(f"\n完成: {ok} 张图表  跳过: {skip}  失败: {fail}")
    logger.info(f"输出目录: {args.out}")


if __name__ == "__main__":
    main()
