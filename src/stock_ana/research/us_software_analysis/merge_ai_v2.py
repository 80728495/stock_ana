"""
从 ai_potential_v2 group*.md 解析四维度内容，生成：
  1. data/lists/software_v2.csv  — 新文件，基于 ytd.md + v2分析
  2. 追加到 data/lists/software_all.csv — 新增四列 + 是否符合列
"""

import re
import pathlib
import pandas as pd


def _find_project_root() -> pathlib.Path:
    for path in pathlib.Path(__file__).resolve().parents:
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError("Cannot find project root containing pyproject.toml")


ROOT      = _find_project_root()
GROUP_DIR = ROOT / "data" / "output" / "software_sector_analysis" / "ai_potential_v2" / "2026-05-10"
ALL_CSV   = pathlib.Path(__file__).parent / "data" / "software_all.csv"
V2_CSV    = pathlib.Path(__file__).parent / "data" / "software_v2.csv"

# Gemini 有时使用旧 ticker，需要别名映射
TICKER_ALIAS = {
    "SQ": "XYZ",   # Block, Inc. 新代码为 XYZ
}

# ── 解析工具 ─────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"^\s*[\-\*]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


SECTION_HEADERS = {"核心业务引擎", "AI影响分析", "财务韧性", "结论"}

def _is_section_header(line: str) -> bool:
    """判断一行是否是四个维度之一的标题行"""
    return any(h in line for h in SECTION_HEADERS) and bool(re.search(r"\*{1,2}", line))


def _extract_section(block: str, kw: str) -> str:
    """逐行提取指定维度（kw）的内容，直到下一个维度标题或分隔线"""
    lines = block.splitlines()
    collecting = False
    content_lines = []

    for line in lines:
        if not collecting:
            # 找到包含关键词的标题行
            if kw in line and re.search(r"\*{1,2}", line):
                collecting = True
        else:
            # 遇到下一个维度标题或分隔线则停止
            if _is_section_header(line) or re.match(r"\s*---+\s*$", line):
                break
            content_lines.append(line)

    return _clean("\n".join(content_lines))


def _extract_conclusion(block: str) -> str:
    """提取结论：符合 / 不符合"""
    # 在结论标题后的区域搜
    tail = ""
    m = re.search(r"\*{1,2}\s*结论\s*\*{0,2}([\s\S]{0,300})", block)
    if m:
        tail = m.group(1)

    if not tail:
        tail = block[-400:]  # fallback: 末尾区域

    tail_clean = re.sub(r"\*+", "", tail)
    if "不符合" in tail_clean:
        return "不符合"
    if "符合" in tail_clean:
        return "符合"
    return ""


def parse_group_file(path: pathlib.Path) -> dict:
    """返回 {TICKER: {核心业务引擎, AI影响分析, 财务韧性, 结论}}"""
    content = path.read_text(encoding="utf-8")
    results = {}

    # 切股票块：### N. TICKER（  或  **N. TICKER（
    splitter = re.compile(
        r"(?:^#{2,4}\s+\d+\.\s+([A-Z][A-Z0-9]*)\s*[（(]"
        r"|^\*{1,2}\d+\.\s+([A-Z][A-Z0-9]*)\s*[（(])",
        re.MULTILINE,
    )
    matches = list(splitter.finditer(content))
    if not matches:
        print(f"  [WARN] {path.name}: 未找到股票标题")
        return results

    for idx, match in enumerate(matches):
        ticker = match.group(1) or match.group(2)
        ticker = TICKER_ALIAS.get(ticker, ticker)   # 别名映射
        start  = match.start()
        end    = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        block  = content[start:end]

        engine     = _extract_section(block, "核心业务引擎")
        ai_impact  = _extract_section(block, "AI影响分析")
        fin        = _extract_section(block, "财务韧性")
        conclusion = _extract_conclusion(block)

        results[ticker] = {
            "核心业务引擎": engine,
            "AI影响分析":  ai_impact,
            "财务韧性v2":  fin,
            "结论v2":      conclusion,
        }
    return results


# ── 解析 software_ytd.md → DataFrame ─────────────────────────────────────────

def parse_ytd_md(path: pathlib.Path) -> pd.DataFrame:
    """从 ytd.md 解析出 (股票, 公司名, sub_label, 年度涨跌幅)"""
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    current_label = ""
    for line in lines:
        # sub_label 标题行
        if line.startswith("## "):
            current_label = line[3:].strip()
            continue
        # 数据行：非空、非分隔线、非表头
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        # 跳过表头行（含 "股票"）
        if stripped.startswith("股票"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        ticker = parts[0]
        # 年度涨跌幅是最后一列
        ytd = parts[-1]
        # 公司名是中间部分（去掉 ticker 和最后的 ytd）
        company = " ".join(parts[1:-1]).strip()
        if not re.match(r"^[A-Z][A-Z0-9]{0,9}$", ticker):
            continue
        rows.append({
            "股票":     ticker,
            "公司名":    company,
            "sub_label": current_label,
            "年度涨跌幅": ytd,
        })
    return pd.DataFrame(rows)


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    # 1. 解析所有 group 文件
    all_data: dict = {}
    for gf in sorted(GROUP_DIR.glob("group*.md")):
        data = parse_group_file(gf)
        print(f"  {gf.name}: {len(data)} 只 → {', '.join(data.keys())}")
        all_data.update(data)
    print(f"\n共解析 {len(all_data)} 只股票")

    # 2. 以 software_all.csv 作为基础（已有 股票/公司名/sub_label/年度涨跌幅）
    df_all = pd.read_csv(ALL_CSV, encoding="utf-8-sig")
    print(f"software_all.csv 行数: {len(df_all)}")

    # 3. 填充 v2 分析列到 all.csv
    for col_key, col_name in [("核心业务引擎", "核心业务引擎"),
                               ("AI影响分析", "AI影响分析"),
                               ("财务韧性v2", "财务韧性v2"),
                               ("结论v2", "结论v2")]:
        df_all[col_name] = df_all["股票"].map(lambda t: all_data.get(t, {}).get(col_key, ""))

    df_all["是否符合"] = df_all["结论v2"].map(lambda x: 1 if x == "符合" else 0)

    # 写回 software_all.csv（把 是否符合 放 结论v2 之前，财务韧性v2 改名）
    df_all = df_all.rename(columns={"财务韧性v2": "财务韧性v2（AI）", "结论v2": "结论v2"})
    df_all.to_csv(ALL_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 已追加到 {ALL_CSV}")
    print(f"   最终列: {list(df_all.columns)}")

    # 4. 生成 software_v2.csv（精简版：股票/公司名/sub_label/ytd + v2分析）
    base_cols = ["股票", "公司名", "sub_label", "年度涨跌幅"]
    df_v2 = df_all[base_cols].copy()
    df_v2["是否符合"]    = df_all["是否符合"]
    df_v2["核心业务引擎"] = df_all["核心业务引擎"]
    df_v2["AI影响分析"]  = df_all["AI影响分析"]
    df_v2["财务韧性"]    = df_all["财务韧性v2（AI）"]
    df_v2["结论"]        = df_all["结论v2"]

    df_v2.to_csv(V2_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 已写入 {V2_CSV}")
    print(f"   列: {list(df_v2.columns)}")

    # 验证
    符合_count  = int((df_v2["是否符合"] == 1).sum())
    empty_count = int((df_v2["结论"] == "").sum())
    print(f"\n【结论分布】符合={符合_count} / 不符合={len(df_v2)-符合_count-empty_count} / 空={empty_count}")
    if empty_count:
        print("  结论为空:", df_v2[df_v2["结论"] == ""]["股票"].tolist())

    # 5. 打印符合清单
    符合_df = df_v2[df_v2["是否符合"] == 1][["股票", "公司名", "sub_label", "年度涨跌幅"]]
    print(f"\n=== 符合 AI 业务促进条件（共 {len(符合_df)} 只）===")
    print(符合_df.to_string(index=False))


if __name__ == "__main__":
    main()
