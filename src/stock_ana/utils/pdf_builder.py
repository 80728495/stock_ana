"""
PDF 报告生成工具

将扫描结果按股票组织：每只股票先放图表（PNG），再放 Gemini 分析文本。
使用 reportlab，中文字体优先使用系统字体（Windows 微软雅黑 / macOS PingFang）。
"""

from __future__ import annotations

import io
import platform
import re
from pathlib import Path

from loguru import logger

# ── 字体查找 ──────────────────────────────────────────────────────────────────

def _find_chinese_font() -> str | None:
    """返回可用的中文 TTF 字体路径，找不到返回 None。"""
    sys = platform.system()
    if sys == "Windows":
        candidates = [
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simsun.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
        ]
    elif sys == "Darwin":
        candidates = [
            Path("/System/Library/Fonts/PingFang.ttc"),
            Path("/Library/Fonts/Arial Unicode MS.ttf"),
        ]
    else:
        candidates = [
            Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _register_font() -> str:
    """注册中文字体，返回字体名称。无中文字体时返回 'Helvetica'。"""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_path = _find_chinese_font()
    if not font_path:
        logger.warning("未找到中文字体，PDF 中文可能显示为方框")
        return "Helvetica"
    try:
        pdfmetrics.registerFont(TTFont("CJK", font_path))
        return "CJK"
    except Exception as e:
        logger.warning(f"字体注册失败 ({font_path}): {e}")
        return "Helvetica"


# ── Markdown 解析 ─────────────────────────────────────────────────────────────

def _split_md_by_symbol(md: str) -> dict[str, str]:
    """
    将完整 Markdown 报告按 `## SYMBOL — ...` 分节，
    返回 {symbol_upper: section_md} 字典。
    """
    sections: dict[str, str] = {}
    current_sym: str | None = None
    current_lines: list[str] = []

    for line in md.splitlines(keepends=True):
        m = re.match(r"^## ([A-Z0-9\.]+)\s*[—–-]", line)
        if m:
            if current_sym:
                sections[current_sym] = "".join(current_lines).strip()
            current_sym = m.group(1).upper()
            current_lines = [line]
        elif current_sym:
            current_lines.append(line)

    if current_sym and current_lines:
        sections[current_sym] = "".join(current_lines).strip()

    return sections


def _parse_md_blocks(md: str) -> list[tuple[str, str]]:
    """将 Markdown 文本解析为 (type, text) 块列表。"""
    blocks: list[tuple[str, str]] = []
    for line in md.splitlines():
        stripped = line.rstrip()
        if stripped.startswith("### "):
            blocks.append(("h3", stripped[4:].strip()))
        elif stripped.startswith("## "):
            blocks.append(("h2", stripped[3:].strip()))
        elif stripped.startswith("# "):
            blocks.append(("h1", stripped[2:].strip()))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            text = stripped[2:].strip().replace("**", "").replace("*", "").replace("`", "")
            blocks.append(("bullet", text))
        elif stripped == "" or stripped == "---":
            blocks.append(("blank", ""))
        else:
            text = stripped.replace("**", "").replace("*", "").replace("`", "")
            blocks.append(("body", text))
    return blocks


# ── PDF 生成 ──────────────────────────────────────────────────────────────────

def build_scan_pdf(
    md_content: str,
    chart_paths: list[Path],
    signal_labels: list[str] | None = None,
    title: str = "每日扫描报告",
    signals: list[dict] | None = None,
) -> bytes:
    """
    将扫描报告生成 PDF：每只股票先放图表，再放 Gemini 分析文本。

    Args:
        md_content:    完整 Gemini Markdown 报告（含所有股票分节）
        chart_paths:   图表 PNG 路径列表（与 signals 顺序一致）
        signal_labels: 每张图表的标题（如 "NVDA (NVIDIA)"），与 chart_paths 等长
        title:         PDF 标题
        signals:       summary JSON 中的 signals 列表（含 symbol 字段），用于匹配图文
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
        HRFlowable, PageBreak,
    )
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    font_name = _register_font()

    def style(name, size, color=colors.black, space_before=0, space_after=4,
              align=TA_LEFT, leading=None):
        return ParagraphStyle(
            name, fontName=font_name, fontSize=size, textColor=color,
            spaceBefore=space_before, spaceAfter=space_after,
            alignment=align, leading=leading or size * 1.4,
        )

    s_title  = style("Title", 18, color=colors.HexColor("#1a1a2e"), space_after=8, align=TA_CENTER)
    s_h1     = style("H1",    15, color=colors.HexColor("#16213e"), space_before=10, space_after=4)
    s_h2     = style("H2",    13, color=colors.HexColor("#0f3460"), space_before=8,  space_after=3)
    s_h3     = style("H3",    11, color=colors.HexColor("#533483"), space_before=6,  space_after=2)
    s_body   = style("Body",   9, space_before=1, space_after=2)
    s_bullet = style("Bullet", 9, space_before=1, space_after=1)
    s_label  = style("Label", 11, color=colors.HexColor("#1a1a2e"), space_before=4, space_after=3, align=TA_CENTER)

    buf = io.BytesIO()
    page_w, page_h = A4
    margin = 18 * mm
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin - 15 * mm

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin,
        title=title,
    )

    story = []

    # ── 封面：标题 + 汇总表格 ──
    story.append(Paragraph(title, s_title))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 6))

    # 解析 Markdown 前缀部分（H1 + 汇总表格，位于第一个 ## 之前）
    first_section_idx = re.search(r"^## ", md_content, re.MULTILINE)
    preamble = md_content[:first_section_idx.start()].strip() if first_section_idx else md_content
    for btype, btext in _parse_md_blocks(preamble):
        if btype == "blank":
            story.append(Spacer(1, 3))
        elif btype == "h1":
            story.append(Paragraph(btext, s_h1))
        elif btype == "body":
            story.append(Paragraph(btext, s_body))

    # ── 按股票循环：图表 → 分析文本 ──
    sym_to_md = _split_md_by_symbol(md_content)

    # 构建 (symbol, chart_path, label) 列表
    entries: list[tuple[str, Path | None, str]] = []
    if signals:
        for i, sig in enumerate(signals):
            sym = str(sig.get("symbol", "")).upper()
            cp = Path(chart_paths[i]) if i < len(chart_paths) else None
            if cp and not cp.exists():
                cp = None
            label = (signal_labels[i] if signal_labels and i < len(signal_labels)
                     else f"{sym}  {sig.get('name','')}")
            entries.append((sym, cp, label))
    else:
        for i, cp in enumerate(chart_paths):
            label = signal_labels[i] if signal_labels and i < len(signal_labels) else cp.stem
            sym = label.split()[0].upper() if label else cp.stem.upper()
            entries.append((sym, cp if cp.exists() else None, label))

    for sym, cp, label in entries:
        story.append(PageBreak())

        # 图表
        story.append(Paragraph(label, s_label))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd")))
        story.append(Spacer(1, 4))

        if cp and cp.exists():
            from PIL import Image as PILImage
            try:
                with PILImage.open(cp) as pil_img:
                    iw, ih = pil_img.size
            except Exception:
                iw, ih = 1200, 800
            scale = min(usable_w / iw, usable_h / ih)
            story.append(RLImage(str(cp), width=iw * scale, height=ih * scale))
        else:
            story.append(Paragraph(f"（{sym} 图表不可用）", s_body))

        # Gemini 分析文本
        sec_md = sym_to_md.get(sym, "")
        if sec_md:
            story.append(Spacer(1, 8))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#eeeeee")))
            for btype, btext in _parse_md_blocks(sec_md):
                if btype == "blank":
                    story.append(Spacer(1, 3))
                elif btype == "h2":
                    story.append(Paragraph(btext, s_h2))
                elif btype == "h3":
                    story.append(Paragraph(btext, s_h3))
                elif btype == "bullet":
                    story.append(Paragraph(f"• {btext}", s_bullet))
                elif btype == "body":
                    story.append(Paragraph(btext, s_body))

    doc.build(story)
    return buf.getvalue()

