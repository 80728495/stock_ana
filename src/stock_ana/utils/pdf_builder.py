"""
PDF 报告生成工具

将 Markdown 分析文本 + 图表 PNG 合并为一份 PDF 文件。
使用 reportlab，中文字体优先使用系统字体（Windows 微软雅黑 / macOS PingFang）。
"""

from __future__ import annotations

import io
import platform
from pathlib import Path

from loguru import logger

# ── 字体查找 ──────────────────────────────────────────────────────────────────

def _find_chinese_font() -> str | None:
    """返回可用的中文 TTF 字体路径，找不到返回 None。"""
    candidates: list[Path] = []
    sys = platform.system()
    if sys == "Windows":
        win_fonts = Path("C:/Windows/Fonts")
        candidates = [
            win_fonts / "msyh.ttc",      # 微软雅黑
            win_fonts / "msyhbd.ttc",
            win_fonts / "simsun.ttc",    # 宋体
            win_fonts / "simhei.ttf",    # 黑体
        ]
    elif sys == "Darwin":
        candidates = [
            Path("/System/Library/Fonts/PingFang.ttc"),
            Path("/Library/Fonts/Arial Unicode MS.ttf"),
        ]
    else:
        candidates = [
            Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
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


# ── Markdown 简单解析 ──────────────────────────────────────────────────────────

def _parse_md_blocks(md: str) -> list[tuple[str, str]]:
    """
    将 Markdown 文本解析为 (type, text) 块列表：
      type: "h1" | "h2" | "h3" | "body" | "bullet" | "blank"
    """
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
            blocks.append(("bullet", stripped[2:].strip()))
        elif stripped == "":
            blocks.append(("blank", ""))
        else:
            # 去掉简单的 **bold** / *italic* 标记
            text = stripped.replace("**", "").replace("*", "").replace("`", "")
            blocks.append(("body", text))
    return blocks


# ── PDF 生成 ──────────────────────────────────────────────────────────────────

def build_scan_pdf(
    md_content: str,
    chart_paths: list[Path],
    signal_labels: list[str] | None = None,
    title: str = "每日扫描报告",
) -> bytes:
    """
    将 Markdown 分析文本和图表合并为 PDF，返回 bytes。

    Args:
        md_content:    Gemini 分析 Markdown 文本
        chart_paths:   图表 PNG 路径列表
        signal_labels: 每张图表的标题（与 chart_paths 等长）
        title:         PDF 标题（显示在第一页页眉）
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

    # 样式定义
    def style(name, size, bold=False, color=colors.black, space_before=0, space_after=4,
              align=TA_LEFT, leading=None):
        return ParagraphStyle(
            name,
            fontName=font_name,
            fontSize=size,
            textColor=color,
            spaceBefore=space_before,
            spaceAfter=space_after,
            alignment=align,
            leading=leading or size * 1.4,
        )

    s_title  = style("Title",  18, color=colors.HexColor("#1a1a2e"), space_before=0, space_after=8, align=TA_CENTER)
    s_h1     = style("H1",     15, color=colors.HexColor("#16213e"), space_before=10, space_after=4)
    s_h2     = style("H2",     13, color=colors.HexColor("#0f3460"), space_before=8,  space_after=3)
    s_h3     = style("H3",     11, color=colors.HexColor("#533483"), space_before=6,  space_after=2)
    s_body   = style("Body",    9, space_before=1, space_after=2)
    s_bullet = style("Bullet",  9, space_before=1, space_after=1)
    s_label  = style("Label",  10, color=colors.HexColor("#333333"), space_before=8, space_after=2, align=TA_CENTER)

    buf = io.BytesIO()
    page_w, page_h = A4
    margin = 18 * mm

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin,
        title=title,
    )

    story = []

    # 标题
    story.append(Paragraph(title, s_title))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 6))

    # Markdown 文本
    blocks = _parse_md_blocks(md_content)
    for btype, btext in blocks:
        if btype == "blank":
            story.append(Spacer(1, 3))
        elif btype == "h1":
            story.append(Paragraph(btext, s_h1))
        elif btype == "h2":
            story.append(Paragraph(btext, s_h2))
        elif btype == "h3":
            story.append(Paragraph(btext, s_h3))
        elif btype == "bullet":
            story.append(Paragraph(f"• {btext}", s_bullet))
        else:
            story.append(Paragraph(btext, s_body))

    # 图表：每张独占一页（横向充满）
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin - 20 * mm  # 留标签空间

    for i, cp in enumerate(chart_paths):
        if not cp.exists():
            continue
        story.append(PageBreak())
        label = (signal_labels[i] if signal_labels and i < len(signal_labels) else cp.stem)
        story.append(Paragraph(label, s_label))
        story.append(Spacer(1, 3))

        # 保持图片宽高比，最大填满页面
        from PIL import Image as PILImage
        try:
            with PILImage.open(cp) as pil_img:
                iw, ih = pil_img.size
        except Exception:
            iw, ih = 1200, 800  # 默认比例

        scale = min(usable_w / iw, usable_h / ih)
        draw_w, draw_h = iw * scale, ih * scale
        story.append(RLImage(str(cp), width=draw_w, height=draw_h))

    doc.build(story)
    return buf.getvalue()
