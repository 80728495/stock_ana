"""
LLM 行业分类标注模块

读取 taxonomy_v2.yaml 和 us_sec_profiles.csv，
对需要 sub_label 的股票按 SIC 分组批量调用 LLM，
将结果回写到 CSV 的 sub_label 列。

用法:
    python -m stock_ana.labeler [--dry-run] [--batch N] [--sic SIC_CODE]
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from openai import OpenAI

from stock_ana.config import DATA_DIR

# ═══════════════════ 配置 ═══════════════════

_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
_API_KEY = "34081167-83fa-43c5-9c30-632e640fba9c"
_MODEL = "kimi-k2.5"
_BATCH_SIZE = 10          # 每次 API 调用处理的股票数
_REQUEST_INTERVAL = 1.0   # API 调用间隔（秒）
_MAX_RETRIES = 2          # 单批次最大重试次数

TAXONOMY_FILE = DATA_DIR / "taxonomy_v2.yaml"
PROFILES_FILE = DATA_DIR / "us_sec_profiles.csv"


# ═══════════════════ Taxonomy 加载 ═══════════════════

def _load_taxonomy() -> dict:
    """加载 taxonomy_v2.yaml，返回 {sic_code: {sic_name, labels: [...]}}"""
    with open(TAXONOMY_FILE, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("sub_labels", {})


def _get_allowed_labels(taxonomy: dict, sic_code: int) -> list[str]:
    """获取某个 SIC 允许的 sub_label 列表"""
    info = taxonomy.get(sic_code, {})
    return [item["label"] for item in info.get("labels", [])]


def _build_label_desc(taxonomy: dict, sic_code: int) -> str:
    """构造某个 SIC 下所有可选标签的描述文本（用于 prompt）"""
    info = taxonomy.get(sic_code, {})
    lines = []
    for i, item in enumerate(info.get("labels", []), 1):
        examples = ", ".join(str(e) for e in item.get("examples", []))
        ex_str = f" (如: {examples})" if examples else ""
        lines.append(f'{i}. "{item["label"]}" — {item["description"]}{ex_str}')
    return "\n".join(lines)


# ═══════════════════ Prompt 构造 ═══════════════════

_SYSTEM_PROMPT = """你是一个股票行业分类专家。你的任务是为股票选择最匹配的细分行业标签（sub_label）。

规则：
1. 你只能从给定的候选标签列表中选择，不能自己创造标签
2. 每只股票选且只选 1 个标签
3. 根据公司名称、业务描述来判断
4. 输出严格的 JSON 格式，不要有其他文字"""


def _build_batch_prompt(
    sic_code: int,
    sic_name: str,
    label_desc: str,
    stocks: list[dict],
) -> str:
    """为一批股票构造 user prompt"""
    stock_lines = []
    for s in stocks:
        bs = str(s.get("business_summary", ""))[:300]
        bs = bs if bs and bs != "nan" else "无描述"
        stock_lines.append(
            f'- ticker: {s["ticker"]}, '
            f'公司: {s["company_name"]}, '
            f'业务: {bs}'
        )
    stocks_text = "\n".join(stock_lines)

    return f"""SIC {sic_code}: {sic_name}

候选标签：
{label_desc}

请为以下 {len(stocks)} 只股票各选 1 个最匹配的标签：
{stocks_text}

输出 JSON 数组，格式如下（不要输出其他内容）：
[{{"ticker": "XXX", "sub_label": "标签名"}}, ...]"""


# ═══════════════════ API 调用 ═══════════════════

def _create_client() -> OpenAI:
    return OpenAI(base_url=_BASE_URL, api_key=_API_KEY)


def _call_llm(client: OpenAI, system: str, user: str) -> str:
    """调用 LLM，返回 content 文本"""
    resp = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=1024,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def _parse_response(text: str, allowed: list[str]) -> dict[str, str]:
    """
    解析 LLM 返回的 JSON，返回 {ticker: sub_label}
    对不在 allowed 列表中的标签标记为 INVALID
    """
    # 提取 JSON 部分（LLM 可能包裹在 ```json ... ``` 中）
    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if not json_match:
        logger.warning(f"无法从 LLM 输出中提取 JSON: {text[:200]}")
        return {}

    try:
        items = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 解析失败: {e}, 原文: {text[:200]}")
        return {}

    result = {}
    allowed_set = set(allowed)
    for item in items:
        ticker = item.get("ticker", "")
        label = item.get("sub_label", "")
        if ticker and label:
            if label in allowed_set:
                result[ticker] = label
            else:
                logger.warning(f"{ticker}: LLM 返回非法标签 '{label}'，允许: {allowed}")
                result[ticker] = f"INVALID:{label}"
    return result


# ═══════════════════ 批量标注引擎 ═══════════════════

def run_labeling(
    dry_run: bool = False,
    batch_size: int = _BATCH_SIZE,
    target_sic: int | None = None,
) -> pd.DataFrame:
    """
    主入口：批量标注所有需要 sub_label 且 sub_label 为空的股票

    Args:
        dry_run: 仅打印 prompt，不调用 API
        batch_size: 每次 API 调用的股票数
        target_sic: 仅处理指定 SIC 代码（调试用）

    Returns:
        更新后的 DataFrame
    """
    taxonomy = _load_taxonomy()
    df = pd.read_csv(PROFILES_FILE, encoding="utf-8-sig")

    # 筛选待标注股票：根据 taxonomy 中定义的 SIC 组来判断
    sub_label_sics = set(int(s) for s in taxonomy.keys())
    mask = df["sic_code"].apply(lambda x: pd.notna(x) and int(x) in sub_label_sics) & (
        df["sub_label"].isna() | (df["sub_label"] == "")
    )
    if target_sic is not None:
        mask = mask & (df["sic_code"] == target_sic)

    pending = df[mask].copy()
    if pending.empty:
        logger.info("没有待标注的股票")
        return df

    logger.info(f"待标注股票: {len(pending)} 只，分布在 "
                f"{pending['sic_code'].nunique()} 个 SIC 组")

    client = _create_client() if not dry_run else None
    total_labeled = 0
    total_invalid = 0

    # 按 SIC 分组处理
    for sic_code, group in pending.groupby("sic_code"):
        sic_int = int(sic_code)
        info = taxonomy.get(sic_int, {})
        if not info:
            logger.warning(f"SIC {sic_int} 未在 taxonomy 中定义，跳过")
            continue

        sic_name = info["sic_name"]
        allowed = _get_allowed_labels(taxonomy, sic_int)
        label_desc = _build_label_desc(taxonomy, sic_int)

        stocks = group.to_dict("records")
        logger.info(f"SIC {sic_int} ({sic_name}): {len(stocks)} 只待标注, "
                     f"{len(allowed)} 个候选标签")

        # 分批处理
        for i in range(0, len(stocks), batch_size):
            batch = stocks[i : i + batch_size]
            tickers_in_batch = [s["ticker"] for s in batch]

            prompt = _build_batch_prompt(sic_int, sic_name, label_desc, batch)

            if dry_run:
                logger.info(f"  [DRY RUN] batch {i // batch_size + 1}: "
                            f"{tickers_in_batch}")
                logger.debug(f"  prompt 长度: {len(prompt)} chars")
                continue

            # 调用 LLM（带重试）
            result = {}
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    raw = _call_llm(client, _SYSTEM_PROMPT, prompt)
                    result = _parse_response(raw, allowed)
                    break
                except Exception as e:
                    logger.warning(f"  API 调用失败 (attempt {attempt + 1}): {e}")
                    if attempt < _MAX_RETRIES:
                        time.sleep(2 ** attempt)

            # 写入结果
            batch_labeled = 0
            batch_invalid = 0
            for ticker in tickers_in_batch:
                label = result.get(ticker, "")
                if label and not label.startswith("INVALID:"):
                    df.loc[df["ticker"] == ticker, "sub_label"] = label
                    batch_labeled += 1
                elif label.startswith("INVALID:"):
                    batch_invalid += 1

            total_labeled += batch_labeled
            total_invalid += batch_invalid
            logger.info(f"  batch {i // batch_size + 1}/{-(-len(stocks) // batch_size)}: "
                        f"标注 {batch_labeled}/{len(batch)}, "
                        f"无效 {batch_invalid}")

            # 每批次写入 CSV（断点续传）
            df.to_csv(PROFILES_FILE, index=False, encoding="utf-8-sig")

            time.sleep(_REQUEST_INTERVAL)

    logger.info(f"标注完成: 成功 {total_labeled}, 无效 {total_invalid}, "
                f"总待标注 {len(pending)}")
    return df


# ═══════════════════ CLI 入口 ═══════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLM 行业分类标注")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅预览 prompt，不调用 API")
    parser.add_argument("--batch", type=int, default=_BATCH_SIZE,
                        help=f"每批次股票数 (默认: {_BATCH_SIZE})")
    parser.add_argument("--sic", type=int, default=None,
                        help="仅处理指定 SIC 代码")
    args = parser.parse_args()

    run_labeling(
        dry_run=args.dry_run,
        batch_size=args.batch,
        target_sic=args.sic,
    )


if __name__ == "__main__":
    main()
