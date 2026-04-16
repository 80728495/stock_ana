"""
每周板块异动分析报告

完整周度流水线：
  1. 更新美股价格数据
  2. 扫描本周异动股票
  3. 按板块聚合异动信号
  4. 送 Gemini 深度分析，生成周报

用法:
    python -m stock_ana.workflows.weekly_sector_report                  # 完整流程
    python -m stock_ana.workflows.weekly_sector_report --preview        # 仅预览数据
    python -m stock_ana.workflows.weekly_sector_report --skip-update    # 跳过价格更新
    python -m stock_ana.workflows.weekly_sector_report --lookback 5     # 自定义回看天数（默认一周）
"""

import asyncio
from datetime import date, timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.config import DATA_DIR, CACHE_DIR, OUTPUT_DIR
from stock_ana.data.market_data import load_symbol_data
from stock_ana.data.fetcher import update_us_price_data
from stock_ana.strategies.impl.momentum_detector import (
    scan_universe,
)

from gemini_webapi import GeminiClient

# ──────── 配置 ────────

ANALYSIS_MODEL = "gemini-3.0-pro"
PROFILES_FILE = DATA_DIR / "us_sec_profiles.csv"
REPORT_DIR = OUTPUT_DIR / "weekly_sector"

# 默认回看 5 个交易日（一周）
DEFAULT_LOOKBACK = 5
DEFAULT_MIN_BREADTH = 2


# ──────── 周次计算 ────────

def _week_label(d: date | None = None) -> str:
    """返回 'YYYY年第WW周' 格式的周标签"""
    d = d or date.today()
    year, week, _ = d.isocalendar()
    return f"{year}年第{week}周"


# ──────── Step 1: 扫描本周异动 ────────

def scan_weekly_momentum(
    lookback: int = DEFAULT_LOOKBACK,
    update: bool = True,
) -> pd.DataFrame:
    """
    扫描全市场本周异动股票。

    Args:
        lookback: 回看交易日数
        update: 是否先更新价格数据

    Returns:
        异动股票 DataFrame（score >= 3.0）
    """
    if update:
        logger.info("Step 1/3: 更新美股价格数据...")
        update_us_price_data()

    logger.info(f"Step 2/3: 扫描全市场异动 (lookback={lookback}d)...")
    result = scan_universe(lookback=lookback, min_score=3.0)
    logger.info(f"检测到 {len(result)} 只异动股票")
    return result


# ──────── Step 2: 板块聚合 ────────

def _build_weekly_sector_text(
    scan_result: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    min_breadth: int = DEFAULT_MIN_BREADTH,
) -> str:
    """
    将本周扫描结果按板块聚合，生成供 Gemini 分析的文本。
    """
    if scan_result.empty:
        return "本周未检测到板块级异动。"

    profiles = pd.read_csv(PROFILES_FILE, encoding="utf-8-sig")
    prof = profiles[["ticker", "company_name", "sector", "sic_code",
                     "sic_description", "sub_label"]].copy()
    prof["sub_label"] = prof["sub_label"].fillna("")
    prof["group_label"] = prof.apply(
        lambda r: r["sub_label"] if r["sub_label"] else r["sic_description"], axis=1
    )
    name_map = prof.set_index("ticker")["company_name"].to_dict()
    group_map = prof.set_index("ticker")["group_label"].to_dict()
    sector_map = prof.set_index("ticker")["sector"].to_dict()

    # 为扫描结果补充 group_label
    df = scan_result.copy()
    df["group_label"] = df["ticker"].map(group_map).fillna("Unknown")
    if "sector" not in df.columns:
        df["sector"] = df["ticker"].map(sector_map).fillna("")

    # 按板块聚合
    group_stats = (
        df.groupby(["sector", "group_label"])
        .agg(
            unique_tickers=("ticker", "nunique"),
            tickers=("ticker", lambda x: sorted(set(x))),
            avg_score=("score", "mean"),
            max_score=("score", "max"),
        )
        .reset_index()
    )

    sector_moves = group_stats[
        group_stats["unique_tickers"] >= min_breadth
    ].sort_values(["unique_tickers", "avg_score"], ascending=[False, False])

    if sector_moves.empty:
        return "本周未检测到板块级异动（同一板块 ≥2 只股票同时异动）。"

    # 计算区间涨跌幅的辅助函数
    def _calc_return(ticker: str) -> str:
        """Return the lookback-period percentage move for a ticker as display text."""
        price_df = load_symbol_data(ticker, universe="us+ndx100")
        if price_df is None or len(price_df) < lookback + 1:
            return "N/A"
        ret = (price_df["close"].iloc[-1] / price_df["close"].iloc[-lookback - 1] - 1) * 100
        return f"{ret:+.1f}%"

    today = date.today()
    week_label = _week_label(today)

    lines = [
        f"报告周期: {week_label}",
        f"扫描窗口: 最近 {lookback} 个交易日",
        f"全市场异动股票: {len(scan_result)} 只",
        f"板块级异动: {len(sector_moves)} 个板块\n",
    ]

    for _, row in sector_moves.iterrows():
        tickers_list = row["tickers"]

        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"【{row['sector']}】{row['group_label']}")
        lines.append(
            f"  板块概况: {row['unique_tickers']} 只股票异动 | "
            f"板块均分: {row['avg_score']:.1f} | 最高分: {row['max_score']:.1f}"
        )

        # 逐股票明细
        lines.append("  个股明细:")
        for t in tickers_list:
            company = name_map.get(t, t)
            t_row = df[df["ticker"] == t].iloc[0]
            score = t_row["score"]

            # 信号拆解
            sig_parts = []
            vol_r = t_row.get("vol_ratio", 0)
            z_sc = t_row.get("z_score", 0)
            brk = t_row.get("breakout", "")
            gap = t_row.get("gap_pct", 0)
            acc = t_row.get("accum_days", 0)

            if vol_r and vol_r >= 1.8:
                sig_parts.append(f"量比{vol_r:.1f}x")
            if z_sc and abs(z_sc) >= 1.5:
                sig_parts.append(f"Z={z_sc:.1f}")
            if brk and str(brk) not in ("", "nan"):
                sig_parts.append(f"突破{brk}")
            if gap and gap >= 2:
                sig_parts.append(f"跳空{gap:.1f}%")
            if acc and acc >= 2:
                sig_parts.append(f"放量阳线{int(acc)}天")
            sig_detail = ", ".join(sig_parts) if sig_parts else "综合触发"

            ret_str = _calc_return(t)

            lines.append(
                f"    {t} ({company}): "
                f"得分 {score:.1f} | 周涨跌 {ret_str} | 信号: {sig_detail}"
            )

        lines.append("")

    return "\n".join(lines)


# ──────── Step 3: Gemini 分析 ────────

_WEEKLY_PROMPT = """角色设定：你是一位顶级宏观策略分析师和行业研究员，擅长从量化异动信号中发现板块性投资机会。

## 背景

这是 **{week_label}** 的美股板块异动周度分析。我通过量化模型对美股 ~1500 只股票进行了扫描，使用 6 维异动信号检测模型，当同一细分行业内多只股票同时触发异动信号时，判定为**板块级行情**。

### 异动评分体系说明（总分 0-10 分，≥3 分触发异动）

| 信号维度 | 分值 | 含义 |
|---------|------|------|
| 量能放大 (vol_ratio) | 0-2 分 | 近 5 日均量 vs 过去 50 日均量的比值。1.8x 得 1 分，3x 得 2 分。仅在价格同期不跌时计分 |
| 超预期涨幅 (z_score) | 0-2 分 | 近 5 日收益率对历史波动率的 Z-score。Z≥1.5 得 1 分，Z≥3.0 得 2 分 |
| 创新高突破 (breakout) | 0-2 分 | 突破之前 20 日最高价得 1 分，突破 60 日最高价得 2 分 |
| 跳空高开 (gap_pct) | 0-1 分 | 窗口期内最大跳空缺口。≥2% 开始计分，4% 满分 |
| 均线突破 (ma_breakout) | 0-1 分 | 上穿 50 日均线得 0.5 分，上穿 200 日均线再加 0.5 分 |
| 量价共振 (accum_days) | 0-2 分 | 放量阳线天数。≥2 天得 0.5 分，≥3 天得 1 分，全部放量阳线得 2 分 |

## 本周数据

{sector_data}

## 任务

请对本周板块异动进行深度分析，按以下结构输出。请调用 Google Search 检索最新的宏观和行业新闻。

### Part 1：逐板块深度解读

对每个异动板块：
1. **行情驱动因素**：为什么这个板块本周在异动？结合最新宏观政策（美联储利率、关税政策等）、行业催化剂（财报、并购、技术突破、政策利好、供需变化、地缘事件）进行分析
2. **持续性判断**：基于量化指标（参与股票数、分数高低、信号类型）+ 基本面逻辑，判断该行情属于「趋势性行情」还是「事件驱动脉冲」
3. **板块内推荐标的**：推荐 1-2 只最值得关注的股票及理由

### Part 2：跨板块联动与市场主线
- 哪些板块之间存在产业链联动？
- 本周市场的核心投资主线是什么？

### Part 3：风险提示与操作建议
- 哪些板块可能是昙花一现，不宜追高？
- 给出 TOP 5 可布局板块排序及操作建议（立即布局 / 等回调 / 观望 / 回避）

请用中文输出，分析要结合具体数据和最新新闻，不要泛泛而谈。"""


async def _init_client() -> GeminiClient:
    """Initialize the Gemini client used for weekly sector commentary."""
    client = GeminiClient()
    await client.init(
        timeout=180,
        auto_close=False,
        auto_refresh=True,
        verbose=False,
    )
    logger.info(f"Gemini 客户端初始化成功 (模型: {ANALYSIS_MODEL})")
    return client


async def analyze_weekly(
    sector_text: str,
    model: str = ANALYSIS_MODEL,
) -> str:
    """
    将本周板块异动数据送 Gemini 分析。

    Returns:
        Gemini 返回的分析文本
    """
    week_label = _week_label()
    prompt = _WEEKLY_PROMPT.format(
        week_label=week_label,
        sector_data=sector_text,
    )

    logger.info(f"Prompt 长度: {len(prompt)} 字符")
    logger.info("正在调用 Gemini 进行板块分析...")

    client = await _init_client()
    try:
        response = await client.generate_content(prompt, model=model)
        text = response.text or ""
        logger.success(f"Gemini 分析完成，返回 {len(text)} 字符")
        return text
    finally:
        await client.close()


# ──────── 报告输出 ────────

def _save_report(text: str, sector_text: str) -> Path:
    """保存周报为 Markdown"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    today = date.today()
    week_label = _week_label(today)
    year, week, _ = today.isocalendar()
    filename = f"weekly_sector_{year}_W{week:02d}.md"
    path = REPORT_DIR / filename

    header = (
        f"# {week_label}美股板块异动分析\n\n"
        f"**生成日期**: {today.isoformat()}\n"
        f"**分析模型**: {ANALYSIS_MODEL}\n\n"
        f"---\n\n"
        f"## 异动数据摘要\n\n"
        f"```\n{sector_text}\n```\n\n"
        f"---\n\n"
        f"## Gemini 深度分析\n\n"
    )

    path.write_text(header + text, encoding="utf-8")
    return path


def _save_weekly_summary(
    scan_result: pd.DataFrame,
    lookback: int,
    min_breadth: int,
    report_path: Path | None,
    status: str,
    error: str = "",
) -> Path:
    """保存周报摘要 JSON，供 clawbot 等外部系统读取。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    today = date.today()
    week_label = _week_label(today)
    year, week, _ = today.isocalendar()
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)

    top = []
    if not scan_result.empty:
        top_df = scan_result.sort_values("score", ascending=False).head(20)
        for _, row in top_df.iterrows():
            top.append({
                "ticker": str(row.get("ticker", "")),
                "score": float(row.get("score", 0.0)),
                "sector": str(row.get("sector", "")),
            })

    summary = {
        "workflow": "weekly_sector_report",
        "week_label": week_label,
        "year": int(year),
        "week": int(week),
        "period_start": week_start.isoformat(),
        "period_end": week_end.isoformat(),
        "generated_at": today.isoformat(),
        "lookback_days": int(lookback),
        "window_desc": f"最近 {lookback} 个交易日",
        "min_breadth": int(min_breadth),
        "momentum_count": int(len(scan_result)),
        "report_path": str(report_path) if report_path else None,
        "status": status,
        "error": error,
        "top_tickers": top,
    }

    summary_path = REPORT_DIR / f"weekly_sector_{year}_W{week:02d}_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


# ──────── CLI ────────

async def _main():
    """Execute the weekly sector workflow CLI, including scan, analysis, and save."""
    import argparse

    parser = argparse.ArgumentParser(description="每周板块异动分析报告")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK,
                        help=f"回看交易日数 (默认 {DEFAULT_LOOKBACK})")
    parser.add_argument("--min-breadth", type=int, default=DEFAULT_MIN_BREADTH,
                        help=f"最少异动股票数 (默认 {DEFAULT_MIN_BREADTH})")
    parser.add_argument("--skip-update", action="store_true",
                        help="跳过价格数据更新")
    parser.add_argument("--preview", action="store_true",
                        help="仅预览板块数据，不调用 Gemini")
    args = parser.parse_args()

    week_label = _week_label()
    logger.info(f"=== {week_label}美股板块异动分析 ===")

    # Step 1 & 2: 扫描
    scan_result = scan_weekly_momentum(
        lookback=args.lookback,
        update=not args.skip_update,
    )

    if scan_result.empty:
        logger.warning("本周未检测到异动股票，退出")
        summary_path = _save_weekly_summary(
            scan_result=scan_result,
            lookback=args.lookback,
            min_breadth=args.min_breadth,
            report_path=None,
            status="empty",
            error="本周未检测到异动股票",
        )
        logger.info(f"周报摘要已保存: {summary_path}")
        return

    # Step 3: 聚合
    logger.info("Step 3/3: 聚合板块数据 & Gemini 分析...")
    sector_text = _build_weekly_sector_text(
        scan_result,
        lookback=args.lookback,
        min_breadth=args.min_breadth,
    )

    if args.preview:
        summary_path = _save_weekly_summary(
            scan_result=scan_result,
            lookback=args.lookback,
            min_breadth=args.min_breadth,
            report_path=None,
            status="preview",
        )
        print(f"\n{'='*60}")
        print(f"  {week_label}美股板块异动数据")
        print(f"{'='*60}\n")
        print(sector_text)
        print(f"\n摘要JSON: {summary_path}")
        return

    # Step 4: Gemini 分析
    result = await analyze_weekly(sector_text)

    # 保存
    path = _save_report(result, sector_text)
    summary_path = _save_weekly_summary(
        scan_result=scan_result,
        lookback=args.lookback,
        min_breadth=args.min_breadth,
        report_path=path,
        status="ok",
    )
    logger.info(f"周报已保存: {path}")
    logger.info(f"周报摘要已保存: {summary_path}")

    print(f"\n{'='*60}")
    print(f"  {week_label}美股板块异动分析")
    print(f"{'='*60}\n")
    print(result)
    print(f"\n{'='*60}")
    print(f"周报已保存: {path}")
    print(f"周报摘要已保存: {summary_path}")


def main():
    """Entry point for launching the async weekly sector workflow."""
    asyncio.run(_main())


if __name__ == "__main__":
    main()
