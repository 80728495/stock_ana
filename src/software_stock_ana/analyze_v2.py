"""
软件板块 AI 业务促进分析 v2（新维度框架）

Prompt：核心业务引擎 / AI影响分析 / 财务韧性 / 结论（4维度）

用法：
    # 测试5只股票
    python scripts/analyze_software_v2.py --test
    python scripts/analyze_software_v2.py --test --tickers ADBE DDOG FROG VEEV CRM

    # 全量运行（18组）
    python scripts/analyze_software_v2.py
    python scripts/analyze_software_v2.py --resume   # 断点续传
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from datetime import date
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

PROFILE_CSV   = ROOT / "data" / "us_sec_profiles.csv"
OUT_BASE      = ROOT / "data" / "output" / "software_sector_analysis" / "ai_potential_v2"
DEFAULT_MODEL = "gemini-3.0-pro"
GROUP_SIZE    = 5
DELAY_MIN     = 250
DELAY_MAX     = 300

# ── 新 Prompt 模板 ────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
你是一个经验丰富的股票分析师。过去一段时间软件板块整体遭受大幅抛售，但是crwd、ddog等最近表现强势，原因是从业务模型上，大语言模型的使用越多，反应公司核心业务指标的当量就越大。下面对于股票进行分析，分析的维度格式如下，请严格按照维度进行内容的输出：

本次标的 共 {stock_count} 只。
{stock_list}

请递进分析每只股票是否符合AI对业务大规模促进的条件：

**核心业务引擎**
该公司的核心业务是什么，核心业务的收入模型是什么，核心业务的核心指标是什么（有的是按使用次数付费，有的是按照订阅量或者坐席量付费，而toc的则还有跟DAU相关的）

**AI影响分析**
如果该业务在AI大语言模型的时代，对于其核心业务指标是会随着大语言模型用量的增加明显增加，还是几乎不变。比如云资源的使用量，DDOG和安全防护会线性增加，坐席则不会，内容平台也不会（因为用户的时长有限）

**财务韧性**
在用量激增背景下，其 Non-GAAP 毛利率是否仍然可维持？

**结论**
如果该业务在AI大语言模型大量增加使用的情况下，对于业务核心指标有明显线性的促进作用，且利润率可维持，则符合，否则不符合。
"""


def load_software_stocks() -> list[dict]:
    df = pd.read_csv(PROFILE_CSV)
    sw = df[df["sic_description"] == "Services-Prepackaged Software"].copy()
    sw = sw.sort_values(["sub_label", "ticker"]).reset_index(drop=True)
    return [
        {"ticker": r["ticker"], "company": r["company_name"], "sub_label": r["sub_label"]}
        for _, r in sw.iterrows()
    ]


def build_prompt(stocks: list[dict]) -> str:
    lines = [f"{i}. {s['ticker']}（{s['company']}）" for i, s in enumerate(stocks, 1)]
    return PROMPT_TEMPLATE.format(
        stock_count=len(stocks),
        stock_list="\n".join(lines),
    )


async def call_gemini(client, prompt: str, model: str) -> str:
    response = await client.generate_content(prompt, model=model)
    return response.text or ""


async def run_groups(
    client,
    groups: list[list[dict]],
    out_dir: Path,
    model: str,
    resume: bool,
) -> None:
    total = len(groups)
    merged_path = out_dir / f"{date.today().isoformat()}_software_ai_v2.md"
    merged_lines: list[str] = []

    if merged_path.exists() and resume:
        merged_lines = merged_path.read_text(encoding="utf-8").splitlines(keepends=True)

    for idx, group in enumerate(groups, 1):
        tickers = [s["ticker"] for s in group]
        slug    = "_".join(tickers[:3])
        fname   = f"group{idx:02d}_{slug}.md"
        fpath   = out_dir / fname

        if resume and fpath.exists():
            logger.info(f"[{idx:02d}/{total}] 跳过（已存在）: {fname}")
            continue

        logger.info(f"[{idx:02d}/{total}] 开始分析: {', '.join(tickers)}")
        prompt = build_prompt(group)

        try:
            text = await call_gemini(client, prompt, model)
        except Exception as e:
            logger.error(f"[{idx:02d}/{total}] Gemini 调用失败: {e}")
            continue

        logger.success(f"Gemini 返回 {len(text)} 字符")

        labels = ", ".join(f'{s["ticker"]}（{s["company"]}）' for s in group)
        header = (
            f"# AI 业务促进分析（v2）— 第 {idx} 组\n\n"
            f"**分析日期**: {date.today().isoformat()}  \n"
            f"**标的**: {labels}\n\n"
            "---\n\n"
        )
        fpath.write_text(header + text, encoding="utf-8")
        logger.info(f"已保存 → {fname}")

        if idx < total:
            delay = random.randint(DELAY_MIN, DELAY_MAX)
            logger.info(f"等待 {delay} 秒后继续下一组...")
            await asyncio.sleep(delay)

    # 合并
    all_files = sorted(out_dir.glob("group*.md"))
    combined  = [f"# 软件板块 AI 业务促进分析总报告（v2）\n\n"
                 f"**生成日期**: {date.today().isoformat()}  |  **共 90 只股票，18 组**\n\n\n"]
    for gf in all_files:
        combined.append(gf.read_text(encoding="utf-8"))
        combined.append("\n\n")
    merged_path.write_text("".join(combined), encoding="utf-8")
    logger.success(f"合并报告已写入 → {merged_path}")


async def main(args: argparse.Namespace) -> None:
    from gemini_webapi import GeminiClient

    psid   = os.environ.get("GEMINI_PSID", "").strip()
    psidts = os.environ.get("GEMINI_PSIDTS", "").strip()
    client = GeminiClient(secure_1psid=psid, secure_1psidts=psidts or None) if psid else GeminiClient()
    await client.init(timeout=180, auto_close=False, auto_refresh=True, verbose=False)
    logger.success(f"Gemini 客户端初始化成功（模型：{args.model}）")

    out_dir = OUT_BASE / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.test:
        # ── 测试模式：只跑指定5只 ────────────────────────────────────────────
        all_stocks = load_software_stocks()
        lookup = {s["ticker"]: s for s in all_stocks}

        test_group: list[dict] = []
        for tk in args.tickers:
            if tk in lookup:
                test_group.append(lookup[tk])
            else:
                # ticker 不在 profile 里，用占位数据
                test_group.append({"ticker": tk, "company": tk, "sub_label": "Unknown"})

        logger.info(f"测试模式：{', '.join(s['ticker'] for s in test_group)}")
        prompt = build_prompt(test_group)
        logger.info(f"Prompt 长度: {len(prompt)} 字符")

        text = await call_gemini(client, prompt, args.model)
        logger.success(f"Gemini 返回 {len(text)} 字符")
        print("\n" + "=" * 60)
        print(text)
        print("=" * 60)

        slug     = "_".join(s["ticker"] for s in test_group)
        out_path = out_dir / f"test_{slug}.md"
        header   = (
            f"# AI 业务促进分析测试（v2）— {slug}\n\n"
            f"**分析日期**: {date.today().isoformat()}\n\n---\n\n"
        )
        out_path.write_text(header + text, encoding="utf-8")
        logger.info(f"已保存 → {out_path}")

    else:
        # ── 全量模式：18组 ───────────────────────────────────────────────────
        all_stocks = load_software_stocks()
        groups = [all_stocks[i: i + GROUP_SIZE] for i in range(0, len(all_stocks), GROUP_SIZE)]
        logger.info(f"共 {len(all_stocks)} 只软件股，{len(groups)} 组")
        await run_groups(client, groups, out_dir, args.model, resume=args.resume)

    await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="软件板块 AI 业务促进分析 v2")
    parser.add_argument("--model",  default=DEFAULT_MODEL, help="Gemini 模型名")
    parser.add_argument("--resume", action="store_true",   help="断点续传（跳过已完成组）")
    parser.add_argument("--test",   action="store_true",   help="测试模式（只分析指定股票）")
    parser.add_argument(
        "--tickers", nargs="+",
        default=["ADBE", "DDOG", "FROG", "VEEV", "CRM"],
        help="测试模式下要分析的股票（默认5只）",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
