"""
扫描信号基本面分析模块

将 vegas_mid_scan 等扫描模块输出的信号列表，
结合 data/scan_signal_prompt.md 模板构造**单个**批量 prompt，
一次请求发给 Gemini，获得所有标的的基本面 + 估值分析，
结果保存为一个 .md 文件。

用法：
    import asyncio
    from stock_ana.utils.scan_analyst import analyze_signals
    asyncio.run(analyze_signals(signals_list))
"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

from loguru import logger

from stock_ana.config import OUTPUT_DIR

# ─── 路径常量 ───────────────────────────────────────────────────────────────
PROMPT_TEMPLATE_PATH = Path(__file__).parents[3] / "data" / "scan_signal_prompt.md"
DEFAULT_OUT_DIR = OUTPUT_DIR / "scan_analysis"
DEFAULT_MODEL = "gemini-3.0-pro"

# ─── Prompt 构建 ─────────────────────────────────────────────────────────────

def _load_template() -> str:
    """读取 Markdown prompt 模板文件（去掉顶部注释块）。"""
    if not PROMPT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Prompt 模板不存在: {PROMPT_TEMPLATE_PATH}")
    text = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip() == "---":
            start = i + 1
            break
    return "\n".join(lines[start:]).strip()


def _format_stock_block(i: int, signal: dict) -> str:
    """将单个信号简化为一行：序号. 代号（公司名）。"""
    sym  = signal.get("symbol", "")
    name = signal.get("name", sym)
    return f"{i}. {sym}（{name}）"


def build_prompt(signals: list[dict]) -> str:
    """
    将多个信号 dict 构造为单个批量分析 prompt。

    Args:
        signals: 信号列表，每个元素来自 run_scan() 返回值

    Returns:
        完整 prompt 字符串
    """
    template = _load_template()

    stock_blocks = "\n\n".join(
        _format_stock_block(i, sig) for i, sig in enumerate(signals, 1)
    )

    return template.format(
        scan_date        = date.today().isoformat(),
        stock_count      = len(signals),
        stock_list_block = stock_blocks,
    )


# ─── Gemini 调用 ─────────────────────────────────────────────────────────────

async def _init_client(model: str = DEFAULT_MODEL):
    """初始化 Gemini 客户端。

    优先从环境变量 GEMINI_PSID / GEMINI_PSIDTS 读取 Cookie，
    避免 cron 等非 GUI session 下 browser-cookie3 无法访问 Keychain 的问题。
    若环境变量未设置，回退到 browser-cookie3 自动读取（仅限交互式 session）。
    """
    import os
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


async def _call_gemini(
    prompt: str,
    client,
    model: str = DEFAULT_MODEL,
    max_retries: int = 2,
) -> str:
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            response = await client.generate_content(prompt, model=model)
            text = response.text or ""
            logger.success(f"Gemini 分析完成，共 {len(text)} 字符")
            return text
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                wait = 15 * (attempt + 1)
                logger.warning(f"请求失败: {str(e)[:80]}，{wait}s 后重试...")
                await asyncio.sleep(wait)
            else:
                logger.error(f"分析失败（重试{max_retries}次）: {str(e)[:200]}")
    raise last_err  # type: ignore[misc]


# ─── 结果保存 ─────────────────────────────────────────────────────────────────

def _save_result(signals: list[dict], analysis_text: str, out_dir: Path) -> Path:
    """将批量分析结果保存为一个 Markdown 文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    today      = date.today().isoformat()
    syms       = "_".join(s.get("symbol", "") for s in signals[:5])
    suffix     = f"_plus{len(signals)-5}more" if len(signals) > 5 else ""
    filename   = f"{today}_{syms}{suffix}.md"
    path       = out_dir / filename

    # 文件头：信号汇总表
    header_lines = [
        f"# 扫描信号基本面分析报告",
        f"",
        f"**扫描日期**: {today}  |  **标的数**: {len(signals)}",
        f"",
        f"| 代码 | 公司 | 信号 | 评分 | 入场日 |",
        f"|------|------|------|------|--------|",
    ]
    for s in signals:
        header_lines.append(
            f"| {s.get('symbol','')} | {s.get('name','')} "
            f"| {s.get('signal','')} | {s.get('score',0):+d} "
            f"| {s.get('entry_date','').split('(')[0]} |"
        )
    header_lines += ["", "---", ""]

    path.write_text("\n".join(header_lines) + "\n" + analysis_text, encoding="utf-8")
    logger.info(f"分析报告已保存 → {path}")
    return path


# ─── 公开接口 ─────────────────────────────────────────────────────────────────

async def analyze_signals(
    signals: list[dict],
    model: str = DEFAULT_MODEL,
    out_dir: Path | None = None,
    min_signal: str | None = None,
) -> Path:
    """
    批量分析股票列表，一次请求完成所有标的，保存为一个 .md 报告。

    Args:
        signals:    股票列表，每个元素至少包含 symbol、name 字段。
                    也可以是 run_scan() 的返回值（会自动按 min_signal 过滤）。
        model:      Gemini 模型名
        out_dir:    输出目录，默认 data/output/scan_analysis/YYYY-MM-DD/
        min_signal: 当 signals 来自扫描结果时，可指定最低等级过滤
                    （STRONG_BUY/BUY/HOLD）；None 则不过滤，全部分析。

    Returns:
        生成的 .md 文件路径
    """
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR / date.today().isoformat()

    if min_signal:
        rank = {"STRONG_BUY": 4, "BUY": 3, "HOLD": 2, "AVOID": 1}
        min_rank = rank.get(min_signal, 2)
        targets = [s for s in signals if rank.get(s.get("signal", ""), 0) >= min_rank]
    else:
        targets = list(signals)

    if not targets:
        logger.info("没有符合条件的标的，跳过分析")
        raise ValueError("没有符合条件的标的")

    logger.info(f"构建批量 Prompt，共 {len(targets)} 个标的...")
    prompt = build_prompt(targets)
    logger.info(f"Prompt 长度: {len(prompt)} 字符")

    client = await _init_client(model)
    try:
        text = await _call_gemini(prompt, client, model=model)
        path = _save_result(targets, text, out_dir)
    finally:
        await client.close()

    return path

