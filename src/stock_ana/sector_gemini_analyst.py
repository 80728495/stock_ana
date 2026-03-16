"""
板块行情 Gemini 深度解读模块

将板块级异动信号（多只股票同时异动的板块）送给 Gemini 进行综合解读，
包括：行情驱动因素、持续性判断、产业链联动、投资建议等。

用法:
    python -m stock_ana.sector_gemini_analyst
    python -m stock_ana.sector_gemini_analyst --days 10 --min-breadth 3
"""

import asyncio
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from stock_ana.config import DATA_DIR, CACHE_DIR, OUTPUT_DIR

# ──────── 复用 gemini_analyst 的客户端初始化 ────────
from gemini_webapi import GeminiClient

# 使用最强分析模型
ANALYSIS_MODEL = "gemini-3.0-pro"


# ──────── Prompt ────────

_SECTOR_PROMPT = """角色设定：你是一位顶级宏观策略分析师和行业研究员，擅长从量化异动信号中发现板块性投资机会。

## 背景

我通过量化模型对美股 ~1500 只股票进行了逐日扫描，使用 6 维异动信号检测模型，当同一细分行业（子板块）内多只股票同时触发异动信号时，判定为**板块级行情**。

### 异动评分体系说明（总分 0-10 分，≥3 分触发异动）

| 信号维度 | 分值 | 含义 |
|---------|------|------|
| 量能放大 (vol_ratio) | 0-2 分 | 近 5 日均量 vs 过去 50 日均量的比值。1.8 倍得 1 分，3 倍得 2 分。仅在价格同期不跌时计分 |
| 超预期涨幅 (z_score) | 0-2 分 | 近 5 日收益率对历史波动率的 Z-score。Z≥1.5 得 1 分，Z≥3.0 得 2 分 |
| 创新高突破 (breakout) | 0-2 分 | 突破之前 20 日最高价得 1 分，突破 60 日最高价得 2 分 |
| 跳空高开 (gap_pct) | 0-1 分 | 窗口期内最大跳空缺口。≥2% 开始计分，4% 满分 |
| 均线突破 (ma_breakout) | 0-1 分 | 上穿 50 日均线得 0.5 分，上穿 200 日均线再加 0.5 分 |
| 量价共振 (accum_days) | 0-2 分 | 放量阳线(量>1.2倍均量且收阳)天数。≥2 天得 0.5 分，≥3 天得 1 分，全部放量阳线得 2 分 |

**板块异动判定**：同一子板块内 ≥2 只股票在同一周内同时触发异动（得分 ≥3），认定为板块行情。

## 数据（最近 {days} 个交易日）

{sector_data}

## 任务

请对每个异动板块逐一进行深度分析，按以下结构输出。请调用 Google Search 检索最新的宏观和行业新闻。

### Part 1：逐板块深度解读

对每个异动板块：
1. **行情驱动因素**：为什么这个板块近期在异动？结合最新的宏观政策（如美联储利率、关税政策）、行业催化剂（财报、产品/技术突破、政策利好、供需变化、地缘事件）进行分析
2. **持续性判断**：基于量化指标（参与股票数、活跃天数、分数高低）+ 基本面逻辑，判断该行情属于「趋势性行情」还是「事件驱动脉冲」
3. **板块内推荐标的**：推荐 1-2 只该板块内最值得关注的股票及理由

### Part 2：跨板块联动与市场主线
- 哪些板块之间存在产业链联动？
- 当前市场的核心投资主线是什么？

### Part 3：风险提示与操作建议
- 哪些板块可能是昙花一现，不宜追高？
- 给出 TOP 5 可布局板块排序及操作建议（立即布局 / 等回调 / 观望 / 回避）

请用中文输出，分析要结合具体数据和最新新闻，不要泛泛而谈。"""


# ──────── 数据准备 ────────

def _build_sector_signal_text(days: int = 5, min_breadth: int = 2) -> str:
    """
    从回测结果构建板块异动摘要文本（含个股明细），供 Gemini 分析。
    """
    detail_file = sorted(
        [f for f in OUTPUT_DIR.glob("daily_momentum_*.csv")
         if "summary" not in f.name],
        key=lambda f: f.name,
    )
    if not detail_file:
        raise FileNotFoundError("未找到 daily_momentum 回测文件，请先运行 backtest_momentum")

    detail = pd.read_csv(detail_file[-1], parse_dates=["date"])
    profiles = pd.read_csv(DATA_DIR / "us_sec_profiles.csv", encoding="utf-8-sig")

    # 构造 group_label 和 company_name 映射
    prof = profiles[["ticker", "company_name", "sector", "sic_code",
                     "sic_description", "sub_label"]].copy()
    prof["sub_label"] = prof["sub_label"].fillna("")
    prof["group_label"] = prof.apply(
        lambda r: r["sub_label"] if r["sub_label"] else r["sic_description"], axis=1
    )
    name_map = prof.set_index("ticker")["company_name"].to_dict()
    sector_map = prof.set_index("ticker")["sector"].to_dict()
    group_map = prof.set_index("ticker")["group_label"].to_dict()

    # 最近 N 个交易日
    all_dates = sorted(detail["date"].unique())
    cutoff_dates = all_dates[-days:]
    cutoff = cutoff_dates[0]

    recent = detail[detail["date"] >= cutoff].copy()
    recent["group_label"] = recent["ticker"].map(group_map).fillna("Unknown")
    recent["sector"] = recent["ticker"].map(sector_map).fillna("")

    date_range = (f"{cutoff_dates[0].strftime('%Y-%m-%d')} ~ "
                  f"{cutoff_dates[-1].strftime('%Y-%m-%d')}")

    # 聚合：按 sub_label 分组
    group_stats = (
        recent.groupby(["sector", "group_label"])
        .agg(
            unique_tickers=("ticker", "nunique"),
            total_hits=("ticker", "size"),
            tickers=("ticker", lambda x: sorted(set(x))),
            avg_score=("score", "mean"),
            max_score=("score", "max"),
            days_active=("date", "nunique"),
        )
        .reset_index()
    )

    sector_moves = group_stats[group_stats["unique_tickers"] >= min_breadth].sort_values(
        ["unique_tickers", "total_hits"], ascending=[False, False]
    )

    # 构建输出文本
    lines = [f"分析区间: {date_range} ({len(cutoff_dates)} 个交易日)"]
    lines.append(f"共 {len(sector_moves)} 个板块级异动:\n")

    for _, row in sector_moves.iterrows():
        tickers_list = row["tickers"]

        lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"【{row['sector']}】{row['group_label']}")
        lines.append(
            f"  板块概况: {row['unique_tickers']} 只股票异动 | "
            f"总异动天次: {row['total_hits']} | "
            f"活跃天数: {row['days_active']}/{len(cutoff_dates)} | "
            f"板块平均分: {row['avg_score']:.1f} | 最高分: {row['max_score']:.1f}"
        )

        # 逐股票明细
        lines.append(f"  个股明细:")
        for t in tickers_list:
            company = name_map.get(t, t)
            # 该股票在此区间的异动记录
            t_hits = recent[(recent["ticker"] == t)
                            & (recent["group_label"] == row["group_label"])]
            hit_days = len(t_hits)
            t_avg_score = t_hits["score"].mean() if len(t_hits) > 0 else 0
            t_max_score = t_hits["score"].max() if len(t_hits) > 0 else 0

            # 各维度拆分（取最近一次的信号明细）
            if len(t_hits) > 0:
                last = t_hits.sort_values("date").iloc[-1]
                vol_r = last.get("vol_ratio", 0)
                z_sc = last.get("z_score", 0)
                brk = last.get("breakout", "")
                gap = last.get("gap_pct", 0)
                acc = last.get("accum_days", 0)
                sig_parts = []
                if vol_r and vol_r >= 1.8:
                    sig_parts.append(f"量比{vol_r:.1f}x")
                if z_sc and abs(z_sc) >= 1.5:
                    sig_parts.append(f"Z={z_sc:.1f}")
                if brk and brk != "nan" and str(brk) != "nan":
                    sig_parts.append(f"突破{brk}")
                if gap and gap >= 2:
                    sig_parts.append(f"跳空{gap:.1f}%")
                if acc and acc >= 2:
                    sig_parts.append(f"放量阳线{int(acc)}天")
                sig_detail = ", ".join(sig_parts) if sig_parts else "综合触发"
            else:
                sig_detail = ""

            # 区间涨跌幅
            ret_str = "N/A"
            for cache_dir in [CACHE_DIR / "us", CACHE_DIR / "ndx100"]:
                f = cache_dir / f"{t}.parquet"
                if f.exists():
                    df = pd.read_parquet(f)
                    df.columns = [c.lower() for c in df.columns]
                    mask = (df.index >= cutoff) & (df.index <= cutoff_dates[-1])
                    sub = df.loc[mask]
                    if len(sub) >= 2:
                        ret = (sub["close"].iloc[-1] / sub["close"].iloc[0] - 1) * 100
                        ret_str = f"{ret:+.1f}%"
                    break

            lines.append(
                f"    {t} ({company}): "
                f"异动{hit_days}天, 均分{t_avg_score:.1f}, 最高{t_max_score:.1f} | "
                f"涨跌幅{ret_str} | 信号: {sig_detail}"
            )

        lines.append("")  # 空行分隔

    return "\n".join(lines)


# ──────── Gemini 调用 ────────

async def _init_client() -> GeminiClient:
    """初始化 Gemini Web 客户端"""
    client = GeminiClient()
    await client.init(
        timeout=180,
        auto_close=False,
        auto_refresh=True,
        verbose=False,
    )
    logger.info(f"Gemini 客户端初始化成功 (模型: {ANALYSIS_MODEL})")
    return client


async def analyze_sector_momentum(
    days: int = 5,
    min_breadth: int = 2,
    model: str = ANALYSIS_MODEL,
) -> str:
    """
    将板块异动信号送给 Gemini 进行综合分析。

    Args:
        days: 分析最近 N 个交易日
        min_breadth: 最少异动股票数（板块级定义）
        model: Gemini 模型

    Returns:
        Gemini 返回的分析文本
    """
    logger.info(f"准备板块异动数据 (最近 {days} 天, 广度 >= {min_breadth})...")
    sector_data = _build_sector_signal_text(days=days, min_breadth=min_breadth)

    prompt = _SECTOR_PROMPT.format(days=days, sector_data=sector_data)

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


def _save_report(text: str, days: int) -> Path:
    """保存分析报告为 Markdown 文件"""
    output_dir = OUTPUT_DIR / "sector_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    filename = f"sector_momentum_{today}_{days}d.md"
    path = output_dir / filename

    header = (
        f"# 板块行情深度解读报告\n\n"
        f"**分析日期**: {today}\n"
        f"**分析窗口**: 最近 {days} 个交易日\n"
        f"**分析模型**: {ANALYSIS_MODEL}\n\n"
        f"---\n\n"
    )

    path.write_text(header + text, encoding="utf-8")
    return path


# ──────── CLI ────────

async def _main():
    import argparse

    parser = argparse.ArgumentParser(description="板块行情 Gemini 深度解读")
    parser.add_argument("--days", type=int, default=5, help="分析最近 N 个交易日 (默认 5)")
    parser.add_argument("--min-breadth", type=int, default=2, help="最少异动股票数 (默认 2)")
    parser.add_argument("--preview", action="store_true", help="只预览数据，不调用 Gemini")
    args = parser.parse_args()

    if args.preview:
        text = _build_sector_signal_text(days=args.days, min_breadth=args.min_breadth)
        print(text)
        return

    result = await analyze_sector_momentum(
        days=args.days,
        min_breadth=args.min_breadth,
    )

    path = _save_report(result, args.days)
    logger.info(f"报告已保存: {path}")

    print(f"\n{'='*60}")
    print(result)
    print(f"{'='*60}")
    print(f"\n报告已保存: {path}")


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
