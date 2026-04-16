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
    """从 macOS Keychain 读取 SMTP 授权码。"""
    result = subprocess.run(
        ["security", "find-generic-password", "-a", SMTP_USER, "-s", KEYCHAIN_SERVICE, "-w"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Keychain 读取失败，请先运行：\n"
            f'security add-generic-password -a "{SMTP_USER}" -s "{KEYCHAIN_SERVICE}" -w "<授权码>" -U'
        )
    return result.stdout.strip()


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
) -> bool:
    """
    将 Markdown 报告渲染为 HTML，并将图表以 CID 内嵌方式附加到邮件正文中。

    Args:
        subject:       邮件主题
        md_content:    Markdown 格式正文
        chart_paths:   图表文件路径列表（PNG）
        signal_labels: 每张图表对应的标签（如 "NVDA (NVIDIA)"），与 chart_paths 等长
        to:            收件人列表，默认发给 DEFAULT_TO
        sender_name:   发件人显示名称
    """
    import smtplib
    from email.mime.image import MIMEImage

    recipients = to or [DEFAULT_TO]
    try:
        password = _get_password()
    except RuntimeError as e:
        logger.error(f"邮件发送失败（Keychain）: {e}")
        return False

    html_body = _md_to_html(md_content)

    # 在 HTML 末尾拼入图表（CID 内嵌）
    chart_cid_map: dict[str, Path] = {}
    if chart_paths:
        charts_html = "<hr><h2>图表</h2>"
        for i, cp in enumerate(chart_paths):
            if not cp.exists():
                continue
            cid = f"chart_{i}_{cp.stem}"
            chart_cid_map[cid] = cp
            label = (signal_labels[i] if signal_labels and i < len(signal_labels) else cp.stem)
            charts_html += f"<h3>{label}</h3>"
            charts_html += f'<img src="cid:{cid}" style="max-width:100%;"><br><br>'
        html_body = html_body.replace("</body>", charts_html + "</body>")

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"]    = f"{sender_name} <{SMTP_USER}>"
    msg["To"]      = ", ".join(recipients)

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(md_content, "plain", "utf-8"))
    alt.attach(MIMEText(html_body, "html", "utf-8"))
    msg.attach(alt)

    for cid, cp in chart_cid_map.items():
        img = MIMEImage(cp.read_bytes(), "png")
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline", filename=cp.name)
        msg.attach(img)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, password)
            server.sendmail(SMTP_USER, recipients, msg.as_bytes())
        logger.success(f"邮件已发送（含{len(chart_cid_map)}张图表）：{subject} → {recipients}")
        return True
    except Exception as e:
        logger.error(f"邮件发送失败: {e}")
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
