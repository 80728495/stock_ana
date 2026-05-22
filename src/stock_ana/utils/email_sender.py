"""
邮件发送工具

从 macOS Keychain 读取 SMTP 授权码，将 Markdown 内容渲染为 HTML 后发送邮件。

配置（Keychain）：
    security add-generic-password -a "80728495@qq.com" -s "stock-ana-smtp" -w "<授权码>" -U

用法：
    from stock_ana.utils.email_sender import send_markdown_email
    send_markdown_email(subject="每日扫描报告", md_content="# 标题\\n...")
"""

import smtplib
import subprocess
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from loguru import logger

# ── SMTP 配置 ──
SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 587
SMTP_USER = "80728495@qq.com"
KEYCHAIN_SERVICE = "stock-ana-smtp"

# 收件人默认与发件人相同（发给自己）
DEFAULT_TO = "80728495@qq.com"


def _get_password() -> str:
    """读取 SMTP 授权码。
    优先级：环境变量 SMTP_PASSWORD → .env 文件 → macOS Keychain。
    Windows 上请在 .env 中设置 SMTP_PASSWORD=<授权码>。
    """
    import os
    # 1. 环境变量
    pw = os.environ.get("SMTP_PASSWORD", "").strip()
    if pw:
        return pw

    # 2. .env 文件（项目根 = src/stock_ana/utils 的 parents[3]）
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("SMTP_PASSWORD="):
                pw = line.split("=", 1)[1].strip().strip('"').strip("'")
                if pw:
                    return pw

    # 3. macOS Keychain（仅 macOS）
    result = subprocess.run(
        ["security", "find-generic-password", "-a", SMTP_USER, "-s", KEYCHAIN_SERVICE, "-w"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()

    raise RuntimeError(
        "未找到 SMTP 授权码，请在 .env 中设置：\n"
        "SMTP_PASSWORD=<QQ邮箱授权码>\n"
        "或 macOS Keychain：\n"
        f'security add-generic-password -a "{SMTP_USER}" -s "{KEYCHAIN_SERVICE}" -w "<授权码>" -U'
    )


def _md_to_html(md_content: str) -> str:
    """将 Markdown 转为带简单样式的 HTML。优先用 markdown 库，否则降级为 pre 包裹。"""
    try:
        import markdown
        body = markdown.markdown(
            md_content,
            extensions=["tables", "fenced_code", "nl2br"],
        )
    except ImportError:
        # 降级：原样显示
        escaped = md_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body = f"<pre style='font-family:monospace;white-space:pre-wrap'>{escaped}</pre>"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: -apple-system, "PingFang SC", "Helvetica Neue", sans-serif;
         font-size: 14px; line-height: 1.7; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
  h1, h2, h3 {{ color: #1a1a1a; border-bottom: 1px solid #eee; padding-bottom: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  code {{ background: #f0f0f0; padding: 2px 5px; border-radius: 3px; font-size: 13px; }}
  pre {{ background: #f6f8fa; padding: 12px; border-radius: 6px; overflow-x: auto; }}
  blockquote {{ border-left: 3px solid #ccc; margin: 0; padding-left: 14px; color: #666; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def send_report_with_charts(
    subject: str,
    md_content: str,
    chart_paths: list[Path],
    signal_labels: list[str] | None = None,
    to: list[str] | None = None,
    sender_name: str = "Stock-Ana",
    signals: list[dict] | None = None,
) -> bool:
    """
    将 Markdown 报告 + 图表打包为 PDF 附件发送邮件。
    邮件正文为 HTML 摘要，PDF 附件包含完整分析文本和所有图表（每张独页）。

    Args:
        subject:       邮件主题
        md_content:    Markdown 格式正文（Gemini 分析结果）
        chart_paths:   图表文件路径列表（PNG）
        signal_labels: 每张图表对应的标签，与 chart_paths 等长
        to:            收件人列表，默认发给 DEFAULT_TO
        sender_name:   发件人显示名称
    """
    import smtplib
    from email.mime.application import MIMEApplication

    recipients = to or [DEFAULT_TO]
    try:
        password = _get_password()
    except RuntimeError as e:
        logger.error(f"邮件发送失败（密码读取）: {e}")
        return False

    # 生成 PDF
    pdf_bytes: bytes | None = None
    try:
        from stock_ana.utils.pdf_builder import build_scan_pdf
        valid_charts = [cp for cp in chart_paths if cp.exists()]
        pdf_bytes = build_scan_pdf(
            md_content=md_content,
            chart_paths=valid_charts,
            signal_labels=signal_labels,
            title=subject,
            signals=signals,
        )
    except Exception as e:
        logger.warning(f"PDF 生成失败，将退回纯 HTML 邮件: {e}")

    html_body = _md_to_html(md_content)

    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"]    = f"{sender_name} <{SMTP_USER}>"
    msg["To"]      = ", ".join(recipients)

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(md_content, "plain", "utf-8"))
    alt.attach(MIMEText(html_body, "html", "utf-8"))
    msg.attach(alt)

    if pdf_bytes:
        pdf_attach = MIMEApplication(pdf_bytes, _subtype="pdf")
        from datetime import date
        pdf_name = f"scan_report_{date.today().isoformat()}.pdf"
        pdf_attach.add_header("Content-Disposition", "attachment", filename=pdf_name)
        msg.attach(pdf_attach)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, password)
            server.sendmail(SMTP_USER, recipients, msg.as_bytes())
        pdf_note = f"含PDF附件({len(valid_charts)}张图表)" if pdf_bytes else "无PDF"
        logger.success(f"邮件已发送（{pdf_note}）：{subject} → {recipients}")
        return True
    except Exception as e:
        logger.error(f"邮件发送失败: {e}")
        return False


def send_pdf_attachments(
    subject: str,
    pdfs: list[tuple[bytes, str]],
    to: list[str] | None = None,
    sender_name: str = "Stock-Ana",
) -> bool:
    """
    将多个 PDF 作为附件合并到一封邮件发送。

    Args:
        subject: 邮件主题
        pdfs:    [(pdf_bytes, filename), ...] 每个市场一份 PDF
        to:      收件人列表，默认发给 DEFAULT_TO
    """
    import smtplib
    from email.mime.application import MIMEApplication

    recipients = to or [DEFAULT_TO]
    try:
        password = _get_password()
    except RuntimeError as e:
        logger.error(f"邮件发送失败（密码读取）: {e}")
        return False

    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = f"{sender_name} <{SMTP_USER}>"
    msg["To"] = ", ".join(recipients)

    alt = MIMEMultipart("alternative")
    body = f"请查收附件中的每日扫描报告（共 {len(pdfs)} 份）。"
    alt.attach(MIMEText(body, "plain", "utf-8"))
    alt.attach(MIMEText(f"<p>{body}</p>", "html", "utf-8"))
    msg.attach(alt)

    for pdf_bytes, filename in pdfs:
        attach = MIMEApplication(pdf_bytes, _subtype="pdf")
        attach.add_header("Content-Disposition", "attachment", filename=filename)
        msg.attach(attach)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, password)
            server.sendmail(SMTP_USER, recipients, msg.as_bytes())
        logger.success(f"合并邮件已发送（{len(pdfs)} 份 PDF）：{subject} → {recipients}")
        return True
    except Exception as e:
        logger.error(f"合并邮件发送失败: {e}")
        return False


def send_markdown_email(
    subject: str,
    md_content: str,
    to: str | None = None,
    sender_name: str = "Stock-Ana",
) -> bool:
    """
    将 Markdown 内容渲染为 HTML 后通过 QQ SMTP 发送邮件。

    Args:
        subject:     邮件主题
        md_content:  Markdown 格式正文
        to:          收件人，默认发给自己
        sender_name: 发件人显示名称

    Returns:
        True 表示发送成功，False 表示失败
    """
    to = to or DEFAULT_TO
    try:
        password = _get_password()
    except RuntimeError as e:
        logger.error(f"邮件发送失败（Keychain）: {e}")
        return False

    html_content = _md_to_html(md_content)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"{sender_name} <{SMTP_USER}>"
    msg["To"]      = to

    msg.attach(MIMEText(md_content, "plain", "utf-8"))
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, password)
            server.sendmail(SMTP_USER, [to], msg.as_bytes())
        logger.success(f"邮件已发送：{subject} → {to}")
        return True
    except Exception as e:
        logger.error(f"邮件发送失败: {e}")
        return False
