"""
软件板块 AI 时代影响分析

对 Services-Prepackaged Software 90 只股票，每 5 只一组（共 18 组），
依次调用 Gemini，分析 AI Coding 爆发对每只股票的正负面影响并打分。

每组请求返回后随机暂停 250-300 秒，防止被限流。

用法：
    python scripts/analyze_software_sector.py
    python scripts/analyze_software_sector.py --resume  # 跳过已有结果的组
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

# ── 项目路径 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

# ── 常量 ──────────────────────────────────────────────────────────────────────
PROFILE_CSV  = ROOT / "data" / "us_sec_profiles.csv"
OUT_DIR      = ROOT / "data" / "output" / "software_sector_analysis"
DEFAULT_MODEL = "gemini-3.0-pro"
GROUP_SIZE   = 5
DELAY_MIN    = 250   # 秒
DELAY_MAX    = 300   # 秒

PROMPT_TEMPLATE = """\
你是一个经验丰富的股票分析师。过去半年因为anthropic在ai coding上的爆发，让软件股持续承压，很多股票甚至录的半年腰斩。然后泥沙俱下时，有些股票会受到ai coding爆发的冲击，但有些在ai时代反倒会有更大的机会。请针对以下股票进行分析：

标的 共 {stock_count} 只。
{stock_list}

请针对每一支股票按照下面的方式分析并输出：
1. 该股票的主营业务是什么，面向什么样的客户，以什么样的技术解决客户什么样的问题；
2. 面对ai coding的爆发，对于其核心业务的正面影响是什么，逻辑是什么；
3. 面对ai coding的爆发，对其核心业务产生的负面影响是什么，逻辑是什么？
4. 最近一次财报如何，对于利空或者利好是否有佐证？佐证是什么？
5. 如果以10分制打分，10分为极端利好，0分为极端利空，结合上面的分析，你打几分。
"""

# ── 股票列表加载 ───────────────────────────────────────────────────────────────

def load_software_stocks() -> list[dict]:
    df = pd.read_csv(PROFILE_CSV)
    sw = df[df["sic_description"] == "Services-Prepackaged Software"].copy()
    sw = sw.sort_values(["sub_label", "ticker"]).reset_index(drop=True)
    return [
        {"ticker": r["ticker"], "company": r["company_name"], "sub_label": r["sub_label"]}
        for _, r in sw.iterrows()
    ]


def build_prompt(stocks: list[dict]) -> str:
    lines = []
    for i, s in enumerate(stocks, 1):
        lines.append(f"{i}. {s['ticker']}（{s['company']}）")
    return PROMPT_TEMPLATE.format(
        stock_count=len(stocks),
        stock_list="\n".join(lines),
    )


# ── Gemini 客户端 ─────────────────────────────────────────────────────────────

async def _init_client(model: str):
    from gemini_webapi import GeminiClient

    psid   = os.environ.get("GEMINI_PSID", "").strip()
    psidts = os.environ.get("GEMINI_PSIDTS", "").strip()

    if psid:
        client = GeminiClient(secure_1psid=psid, secure_1psidts=psidts or None)
        logger.info("Gemini 客户端：使用环境变量 Cookie 初始化")
    else:
        client = GeminiClient()
        logger.info("Gemini 客户端：使用 browser-cookie3 自动读取 Cookie")

    await client.init(timeout=180, auto_close=False, auto_refresh=True, verbose=False)
    logger.info(f"Gemini 客户端初始化成功（模型：{model}）")
    return client


async def _call_gemini(prompt: str, client, model: str, max_retries: int = 2) -> str:
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            response = await client.generate_content(prompt, model=model)
            text = response.text or ""
            logger.success(f"Gemini 返回 {len(text)} 字符")
            return text
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                wait = 20 * (attempt + 1)
                logger.warning(f"请求失败: {str(e)[:80]}，{wait}s 后重试...")
                await asyncio.sleep(wait)
            else:
                logger.error(f"请求失败（已重试 {max_retries} 次）: {str(e)[:200]}")
    raise last_err  # type: ignore[misc]


# ── 输出保存 ───────────────────────────────────────────────────────────────────

def group_out_path(out_dir: Path, group_idx: int, tickers: list[str]) -> Path:
    label = "_".join(tickers[:3])
    return out_dir / f"group{group_idx:02d}_{label}.md"


def save_group(out_dir: Path, group_idx: int, stocks: list[dict], text: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = [s["ticker"] for s in stocks]
    path = group_out_path(out_dir, group_idx, tickers)
    stock_labels = ", ".join(f"{s['ticker']}（{s['company']}）" for s in stocks)
    header = (
        f"# 软件板块 AI 影响分析 — 第 {group_idx} 组\n\n"
        f"**分析日期**: {date.today().isoformat()}  \n"
        f"**标的**: {stock_labels}\n\n"
        f"---\n\n"
    )
    path.write_text(header + text, encoding="utf-8")
    logger.info(f"已保存 → {path.name}")
    return path


def merge_all(out_dir: Path, stocks: list[dict]) -> Path:
    """将所有组的分析结果合并为一个总报告。"""
    all_files = sorted(out_dir.glob("group*.md"))
    merged_lines = [
        f"# 美股软件板块 AI 时代影响分析总报告\n",
        f"**生成日期**: {date.today().isoformat()}  |  **共 {len(stocks)} 只股票，{len(all_files)} 组**\n",
        f"\n---\n",
    ]
    for f in all_files:
        merged_lines.append(f.read_text(encoding="utf-8"))
        merged_lines.append("\n\n---\n\n")

    merged_path = out_dir / f"{date.today().isoformat()}_software_ai_analysis.md"
    merged_path.write_text("\n".join(merged_lines), encoding="utf-8")
    logger.success(f"总报告已保存 → {merged_path}")
    return merged_path


# ── 主流程 ─────────────────────────────────────────────────────────────────────

async def main(resume: bool, model: str) -> None:
    stocks = load_software_stocks()
    logger.info(f"共 {len(stocks)} 只软件股，每组 {GROUP_SIZE} 只，共 {-(-len(stocks)//GROUP_SIZE)} 组")

    out_dir = OUT_DIR / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 分组
    groups = [stocks[i:i + GROUP_SIZE] for i in range(0, len(stocks), GROUP_SIZE)]

    client = await _init_client(model)
    try:
        for idx, group in enumerate(groups, 1):
            tickers = [s["ticker"] for s in group]
            group_path = group_out_path(out_dir, idx, tickers)

            if resume and group_path.exists():
                logger.info(f"[{idx:02d}/{len(groups)}] 跳过（已存在）: {group_path.name}")
                continue

            logger.info(f"[{idx:02d}/{len(groups)}] 开始分析: {', '.join(tickers)}")
            prompt = build_prompt(group)

            text = await _call_gemini(prompt, client, model=model)
            save_group(out_dir, idx, group, text)

            if idx < len(groups):
                delay = random.randint(DELAY_MIN, DELAY_MAX)
                logger.info(f"等待 {delay} 秒后继续下一组...")
                await asyncio.sleep(delay)

    finally:
        await client.close()

    logger.info("所有组请求完成，合并报告...")
    merged = merge_all(out_dir, stocks)
    logger.success(f"完成！总报告: {merged}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="软件板块 AI 影响批量分析")
    parser.add_argument("--resume", action="store_true", help="跳过已有结果的组，断点续传")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini 模型名")
    args = parser.parse_args()

    asyncio.run(main(resume=args.resume, model=args.model))
