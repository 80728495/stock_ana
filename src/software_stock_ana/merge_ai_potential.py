"""
从 AI Potential 分析的 group*.md 文件解析每只股票的五维度内容和结论，
写入 data/lists/software_all.csv：
  - AI Potential 结论列（符合/不符合）：插入 年度涨跌幅 之后
  - 定价模式 / NRR归因 / 核心卡位 / 资产广度 / 财务韧性：插入 主营业务与客户画像 之后
"""

import re
import pathlib
import pandas as pd

ROOT      = pathlib.Path(__file__).parents[2]
GROUP_DIR = ROOT / "data" / "output" / "software_sector_analysis" / "ai_potential" / "2026-05-09"
CSV_PATH  = pathlib.Path(__file__).parent / "data" / "software_all.csv"


# ── 解析单只股票块内某维度的文本 ────────────────────────────────────────────
def _extract_dimension(block: str, dim_kw: str) -> str:
    """
    从一只股票的文本块中提取指定维度关键词后的内容。
    dim_kw 示例: "定价模式", "NRR 归因", "核心卡位", "资产广度", "财务韧性"
    """
    # 匹配 **维度N：XXX** 后紧跟的文本，直到下一个 **维度/结论/---/# 或空行后接分隔
    pattern = (
        r"\*{1,2}\s*\*{0,2}\s*维度\d[：:]\s*" + re.escape(dim_kw)
        + r"[\s\S]*?\*{0,2}\s*\n"                        # 结尾 **
        r"([\s\S]+?)"                                     # 内容（捕获组）
        r"(?=\n\s*[\*\-]\s*\*{1,2}\s*(?:维度|结论)|---|\Z)"
    )
    m = re.search(pattern, block, re.IGNORECASE)
    if m:
        return _clean(m.group(1))

    # 宽松模式：按行抓维度标题后的第一段
    lines = block.splitlines()
    for i, line in enumerate(lines):
        if re.search(r"维度\d[：:]\s*" + re.escape(dim_kw), line, re.IGNORECASE):
            # 收集后续非空行，直到下一个维度/结论/---
            content_lines = []
            for j in range(i + 1, len(lines)):
                nxt = lines[j]
                if re.search(r"维度\d[：:]|结论|^---", nxt, re.IGNORECASE):
                    break
                content_lines.append(nxt)
            return _clean("\n".join(content_lines))

    return ""


def _extract_conclusion(block: str) -> str:
    """提取结论：符合 / 不符合"""
    # 模式1: **结论：符合** 或 **结论：不符合**（一行内，冒号后紧跟结果）
    m = re.search(r"\*{1,2}\s*结论[：:]\s*(符合|不符合)\s*\*{0,2}", block)
    if m:
        return m.group(1)

    # 模式2: **结论：** 不符合（粗体在冒号后关闭，结果在后面同行）
    m = re.search(r"\*{1,2}\s*结论[：:]\s*\*{0,2}\s*(符合|不符合)", block)
    if m:
        return m.group(1)

    # 模式3: **结论：** **不符合**（结果被单独加粗）
    m = re.search(r"\*{1,2}\s*结论[：:]\s*\*{0,2}[\s\n]+\*{1,2}\s*(符合|不符合)\s*\*{0,2}", block)
    if m:
        return m.group(1)

    # 模式4: 结论独立行(**结论** 或 **结论**)，后续行含符合/不符合
    m = re.search(r"\*{1,2}\s*结论\s*\*{0,2}\s*\n+([\s\S]+?)(?=\n\s*[\*\-#]|\Z)", block)
    if m:
        text = m.group(1).strip()
        text = re.sub(r"\*+", "", text).strip(".。 ")
        if "不符合" in text:
            return "不符合"
        if "符合" in text:
            return "符合"

    # 模式5: 裸搜索 — 最后一道防线，在块末尾区域找"符合/不符合"
    # 只在 **结论** 标题之后的文字里找
    if "结论" in block:
        tail = block[block.rfind("结论"):]
        tail_clean = re.sub(r"\*+", "", tail)
        if "不符合" in tail_clean:
            return "不符合"
        if "符合" in tail_clean:
            return "符合"

    return ""


def _clean(text: str) -> str:
    """去掉多余空白和 markdown 修饰"""
    text = re.sub(r"\*+", "", text)          # 去 **
    text = re.sub(r"`+", "", text)           # 去 ``
    text = re.sub(r"^\s*[\-\*]\s*", "", text, flags=re.MULTILINE)  # 去行首 - *
    text = re.sub(r"\n{3,}", "\n\n", text)   # 合并多余空行
    return text.strip()


# ── 解析单个 group 文件 ──────────────────────────────────────────────────────
# 维度关键词（同时兼容中文空格变体）
DIM_KEYS = ["定价模式", "NRR 归因", "核心卡位", "资产广度", "财务韧性"]
DIM_KEYS_ALT = ["定价模式", "NRR归因", "核心卡位", "资产广度", "财务韧性"]


def _dim_text(block: str, kw: str, kw_alt: str) -> str:
    text = _extract_dimension(block, kw)
    if not text:
        text = _extract_dimension(block, kw_alt)
    return text


def parse_group_file(path: pathlib.Path) -> dict:
    """
    返回 {TICKER: {定价模式, NRR归因, 核心卡位, 资产广度, 财务韧性, AI Potential}}
    """
    content = path.read_text(encoding="utf-8")
    results = {}

    # 切分股票块：匹配  ### N. TICKER  /  #### N. TICKER  /  **N. TICKER**
    splitter = re.compile(
        r"(?:^#{2,4}\s+\d+\.\s+([A-Z][A-Z0-9]*)\s*[（(]|^\*{1,2}\d+\.\s+([A-Z][A-Z0-9]*)\s*[（(])",
        re.MULTILINE,
    )

    matches = list(splitter.finditer(content))
    if not matches:
        print(f"  [WARN] {path.name}: 未找到股票标题")
        return results

    for idx, match in enumerate(matches):
        ticker = match.group(1) or match.group(2)
        start  = match.start()
        end    = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        block  = content[start:end]

        pricing     = _dim_text(block, "定价模式", "定价模式")
        nrr         = _dim_text(block, "NRR 归因", "NRR归因")
        position    = _dim_text(block, "核心卡位", "核心卡位")
        assets      = _dim_text(block, "资产广度", "资产广度")
        finance     = _dim_text(block, "财务韧性", "财务韧性")
        conclusion  = _extract_conclusion(block)

        results[ticker] = {
            "定价模式":   pricing,
            "NRR归因":    nrr,
            "核心卡位":   position,
            "资产广度":   assets,
            "财务韧性":   finance,
            "AI Potential": conclusion,
        }

    return results


def main():
    # 1. 读取所有 group 文件
    all_data: dict = {}
    for gf in sorted(GROUP_DIR.glob("group*.md")):
        data = parse_group_file(gf)
        print(f"  {gf.name}: 解析出 {len(data)} 只股票 → {', '.join(data.keys())}")
        all_data.update(data)

    print(f"\n共解析 {len(all_data)} 只股票")

    # 2. 读 CSV
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    print(f"CSV 原始列: {list(df.columns)}")
    print(f"CSV 行数: {len(df)}")

    # 3. 填充新列数据
    for col in ["AI Potential", "定价模式", "NRR归因", "核心卡位", "资产广度", "财务韧性"]:
        df[col] = df["股票"].map(lambda t: all_data.get(t, {}).get(col, ""))

    # 4. 重排列顺序
    # 目标顺序:
    # 股票, 公司名, sub_label, 年度涨跌幅, AI Potential,
    # 评分, 主营业务与客户画像, 定价模式, NRR归因, 核心卡位, 资产广度, 财务韧性,
    # AI Coding 正面影响, AI Coding 负面影响, 新财报与佐证, 评分叙述
    target_cols = [
        "股票", "公司名", "sub_label", "年度涨跌幅", "AI Potential",
        "评分", "主营业务与客户画像",
        "定价模式", "NRR归因", "核心卡位", "资产广度", "财务韧性",
        "AI Coding 正面影响", "AI Coding 负面影响", "新财报与佐证", "评分叙述",
    ]
    # 保留原始中可能有但 target 里没有的列（追加到末尾）
    extra_cols = [c for c in df.columns if c not in target_cols]
    final_cols = target_cols + extra_cols
    df = df[final_cols]

    # 5. 写回
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已写回 {CSV_PATH}")
    print(f"   最终列顺序: {list(df.columns)}")

    # 6. 验证 AI Potential 列
    summary = df["AI Potential"].value_counts(dropna=False)
    print(f"\n【AI Potential 结论分布】\n{summary.to_string()}")
    missing = df[df["AI Potential"] == ""]["股票"].tolist()
    if missing:
        print(f"\n⚠ 以下 {len(missing)} 只股票结论为空: {missing}")
    else:
        print("\n✅ 所有股票均有结论，无缺失")


if __name__ == "__main__":
    main()
