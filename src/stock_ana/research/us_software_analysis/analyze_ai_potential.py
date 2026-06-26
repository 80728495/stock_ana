"""
软件板块 AI 潜力分析（Consumption-based 筛选维度）

对 Services-Prepackaged Software 90 只股票，每 5 只一组（共 18 组），
依次调用 Gemini，从5个维度分析每只股票是否符合"AI用量线性增长"的受益模型。

用法：
    python src/stock_ana/research/us_software_analysis/analyze_ai_potential.py
    python src/stock_ana/research/us_software_analysis/analyze_ai_potential.py --resume   # 断点续传
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


def _find_project_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError("Cannot find project root containing pyproject.toml")


ROOT = _find_project_root()
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

PROFILE_CSV  = ROOT / "data" / "us_sec_profiles.csv"
OUT_DIR      = ROOT / "data" / "output" / "software_sector_analysis" / "ai_potential"
DEFAULT_MODEL = "gemini-3.0-pro"
GROUP_SIZE   = 5
DELAY_MIN    = 250
DELAY_MAX    = 300

PROMPT_TEMPLATE = """\
你是一个经验丰富的股票分析师。过去一段时间软件板块遭到资本大幅抛售。但是过去几个交易日，DDOG发生了跳涨，核心原因是，DDOG的客户需求因为AI大语言模型的爆发也随着增加，也就是DDOG本身的业务体量和云一样，与大语言模型的使用体量是线性关系，而非软件SaaS坐席类，因为大语言模型而通缩。现在，你要对下面的每只股票，进行分析，分析的维度格式如下：

本次标的 共 {stock_count} 只。
{stock_list}

请基于以下五个维度分析每只股票的 AI 潜力：

**维度1：定价模式**
是否为完全的 Consumption-based？其计费因子是否随 AI 产出（如代码量、数据量）线性增长？

**维度2：NRR 归因**
其 NRR 的增长主要来自客户数增加，还是单个客户的用量爆发？

**维度3：核心卡位**
它是否解决了 AI 带来的"代码爆炸"或"系统复杂性剧增"的痛点？

**维度4：资产广度**
它是否已具备管理 AI 模型（Weights/Artifacts）或 AI 推理过程的能力？

**维度5：财务韧性**
在用量激增背景下，其 Non-GAAP 毛利率是否仍然可维持？

**结论**
如果该业务模型符合上述维度特征，则结论为"符合"，否则结论为"不符合"。

请对每只股票逐一分析，按上述五个维度 + 结论的格式输出。
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


def group_out_path(out_dir: Path, group_idx: int, tickers: list[str]) -> Path:
    label = "_".join(tickers[:3])
    return out_dir / f"group{group_idx:02d}_{label}.md"


def save_group(out_dir: Path, group_idx: int, stocks: list[dict], text: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = [s["ticker"] for s in stocks]
    path = group_out_path(out_dir, group_idx, tickers)
    stock_labels = ", ".join(f"{s['ticker']}（{s['company']}）" for s in stocks)
    header = (
        f"# AI 潜力分析 — 第 {group_idx} 组\n\n"
        f"**分析日期**: {date.today().isoformat()}  \n"
        f"**标的**: {stock_labels}\n\n"
        f"---\n\n"
    )
    path.write_text(header + text, encoding="utf-8")
    logger.info(f"已保存 → {path.name}")
    return path


def merge_all(out_dir: Path, total_stocks: int) -> Path:
    all_files = sorted(out_dir.glob("group*.md"))
    merged_lines = [
        f"# 美股软件板块 AI 潜力分析总报告\n",
        f"**生成日期**: {date.today().isoformat()}  |  **共 {total_stocks} 只股票，{len(all_files)} 组**\n",
        f"\n---\n",
    ]
    for f in all_files:
        merged_lines.append(f.read_text(encoding="utf-8"))
        merged_lines.append("\n\n---\n\n")

    merged_path = out_dir / f"{date.today().isoformat()}_software_ai_potential.md"
    merged_path.write_text("\n".join(merged_lines), encoding="utf-8")
    logger.success(f"总报告已保存 → {merged_path}")
    return merged_path


async def _init_client(model: str):
    from gemini_webapi import GeminiClient

    psid   = os.environ.get("GEMINI_PSID", "").strip()
    psidts = os.environ.get("GEMINI_PSIDTS", "").strip()
    client = GeminiClient(secure_1psid=psid, secure_1psidts=psidts or None) if psid else GeminiClient()
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


async def main(resume: bool, model: str) -> None:
    stocks = load_software_stocks()
    groups = [stocks[i:i + GROUP_SIZE] for i in range(0, len(stocks), GROUP_SIZE)]
    logger.info(f"共 {len(stocks)} 只软件股，{len(groups)} 组")

    out_dir = OUT_DIR / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

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
    merge_all(out_dir, len(stocks))
    logger.success("全量 AI 潜力分析完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="软件板块 AI 潜力批量分析")
    parser.add_argument("--resume", action="store_true", help="断点续传，跳过已有文件")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini 模型名")
    args = parser.parse_args()
    asyncio.run(main(resume=args.resume, model=args.model))
