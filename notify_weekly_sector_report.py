#!/usr/bin/env python3
"""Send weekly sector report summary and report file to main agent (Feishu)."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WEEKLY_DIR = PROJECT_ROOT / "data" / "output" / "weekly_sector"

FEISHU_APP_ID = "cli_a924285ae7f85cc7"
FEISHU_APP_SECRET = "53hrIbxJYHGGAI8qbndwofOzltJAkah0"
FEISHU_USER_OPEN_ID = "ou_5489407346c5c13bc4687a83859d619b"
FEISHU_API = "https://open.feishu.cn/open-apis"

_feishu_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _latest_summary() -> Path | None:
    if not WEEKLY_DIR.exists():
        return None
    files = sorted(WEEKLY_DIR.glob("weekly_sector_*_summary.json"))
    return files[-1] if files else None


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_tenant_token() -> str | None:
    url = f"{FEISHU_API}/auth/v3/tenant_access_token/internal"
    body = json.dumps({"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with _feishu_opener.open(req, timeout=15) as resp:
        result = json.loads(resp.read())
    return result.get("tenant_access_token")


def send_post_message(token: str, title: str, blocks: list[list[dict]]) -> bool:
    url = f"{FEISHU_API}/im/v1/messages?receive_id_type=open_id"
    payload = {
        "receive_id": FEISHU_USER_OPEN_ID,
        "msg_type": "post",
        "content": json.dumps({"zh_cn": {"title": title, "content": blocks}}, ensure_ascii=False),
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {token}",
        },
    )
    with _feishu_opener.open(req, timeout=20) as resp:
        result = json.loads(resp.read())
    return result.get("code") == 0


def send_file_message(token: str, file_key: str) -> bool:
    url = f"{FEISHU_API}/im/v1/messages?receive_id_type=open_id"
    payload = {
        "receive_id": FEISHU_USER_OPEN_ID,
        "msg_type": "file",
        "content": json.dumps({"file_key": file_key}, ensure_ascii=False),
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {token}",
        },
    )
    with _feishu_opener.open(req, timeout=20) as resp:
        result = json.loads(resp.read())
    return result.get("code") == 0


def upload_report_file(token: str, report_path: Path) -> str | None:
    if not report_path.exists():
        return None

    import io

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = io.BytesIO()

    def _part(name: str, value: str) -> None:
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.write(value.encode("utf-8"))
        body.write(b"\r\n")

    _part("file_type", "stream")
    _part("file_name", report_path.name)

    body.write(f"--{boundary}\r\n".encode())
    body.write(
        f'Content-Disposition: form-data; name="file"; filename="{report_path.name}"\r\n'.encode()
    )
    body.write(b"Content-Type: text/markdown\r\n\r\n")
    body.write(report_path.read_bytes())
    body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())

    url = f"{FEISHU_API}/im/v1/files"
    req = urllib.request.Request(
        url,
        data=body.getvalue(),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with _feishu_opener.open(req, timeout=30) as resp:
            result = json.loads(resp.read())
        return result.get("data", {}).get("file_key")
    except Exception:
        return None


def build_blocks(summary: dict, workflow_exit_code: int) -> tuple[str, list[list[dict]]]:
    week_label = summary.get("week_label", "未知周")
    status = summary.get("status", "unknown")
    momentum_count = summary.get("momentum_count", 0)
    window_desc = summary.get("window_desc", "")
    report_path = summary.get("report_path") or "N/A"

    title = f"📈 周度板块异动报告 {week_label}"
    if workflow_exit_code != 0:
        title += " ⚠️"

    lines = [
        f"状态: {status}",
        f"窗口: {window_desc}",
        f"异动股票数: {momentum_count}",
        f"本地报告路径: {report_path}",
    ]

    blocks: list[list[dict]] = [[{"tag": "text", "text": "\n".join(lines)}]]

    top = summary.get("top_tickers", [])
    if top:
        top_lines = ["Top 标的:"]
        for row in top[:10]:
            top_lines.append(
                f"- {row.get('ticker', '')} score={float(row.get('score', 0)):.2f}  {row.get('sector', '')}"
            )
        blocks.append([{"tag": "text", "text": "\n".join(top_lines)}])

    if workflow_exit_code != 0:
        blocks.insert(0, [{"tag": "text", "text": f"⚠️ 周报流程退出码: {workflow_exit_code}"}])

    return title, blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Notify weekly sector report to main agent")
    parser.add_argument("--workflow-exit-code", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    token = get_tenant_token()
    if not token:
        print("❌ 获取飞书 token 失败")
        return 1

    summary_path = _latest_summary()
    if not summary_path:
        ok = send_post_message(
            token,
            "📈 周度板块异动报告（未找到摘要）",
            [[{"tag": "text", "text": "⚠️ 未找到 weekly summary.json，请检查 weekly workflow 日志。"}]],
        )
        return 0 if ok else 1

    summary = _read_json(summary_path)
    if not summary:
        ok = send_post_message(
            token,
            "📈 周度板块异动报告（摘要解析失败）",
            [[{"tag": "text", "text": f"⚠️ 无法解析摘要文件: {summary_path}"}]],
        )
        return 0 if ok else 1

    title, blocks = build_blocks(summary, args.workflow_exit_code)
    ok = send_post_message(token, title, blocks)

    # For clickable viewing in Feishu: upload markdown as file and send file card.
    report_path_raw = summary.get("report_path")
    if report_path_raw:
        report_path = Path(report_path_raw)
        file_key = upload_report_file(token, report_path)
        if file_key:
            send_post_message(
                token,
                "📎 周报正文文件",
                [[{"tag": "text", "text": "已附上本周 Markdown 报告文件，点击下方文件卡片可直接查看。"}]],
            )
            send_file_message(token, file_key)

    # ── 邮件发送：周报 MD 文件 ──
    if report_path_raw:
        report_path = Path(report_path_raw)
        if report_path.exists():
            try:
                from stock_ana.utils.email_sender import send_markdown_email
                md_content = report_path.read_text(encoding="utf-8")
                week_label = summary.get("week_label", "") if summary else ""
                email_sent = send_markdown_email(
                    subject=f"📈 每周行业报告 {week_label}",
                    md_content=md_content,
                )
                if not email_sent:
                    print("⚠️ 邮件发送失败")
            except Exception as e:
                print(f"⚠️ 邮件发送异常: {e}")

    if ok:
        print("✅ weekly report 通知发送完成")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
