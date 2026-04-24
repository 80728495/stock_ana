#!/usr/bin/env python3
"""Send daily update + Vegas scan summary to main agent via Feishu."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.request
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

DAILY_UPDATE_DIR = OUTPUT_DIR / "daily_update"
DAILY_SCAN_DIR = OUTPUT_DIR / "daily_scan"

FEISHU_APP_ID = "cli_a924285ae7f85cc7"
FEISHU_APP_SECRET = "53hrIbxJYHGGAI8qbndwofOzltJAkah0"
FEISHU_USER_OPEN_ID = "ou_5489407346c5c13bc4687a83859d619b"
FEISHU_API = "https://open.feishu.cn/open-apis"

_feishu_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _find_latest_file(root: Path, filename: str) -> Path | None:
    if not root.exists():
        return None
    candidates = sorted(root.glob(f"*/{filename}"))
    if not candidates:
        return None
    return candidates[-1]


def _find_today_or_latest(root: Path, filename: str) -> Path | None:
    today_file = root / date.today().isoformat() / filename
    if today_file.exists():
        return today_file
    return _find_latest_file(root, filename)


def get_tenant_token() -> str | None:
    url = f"{FEISHU_API}/auth/v3/tenant_access_token/internal"
    data = json.dumps({"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET}).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    req = urllib.request.Request(url, data=data, headers=headers)
    with _feishu_opener.open(req, timeout=15) as resp:
        result = json.loads(resp.read())
    return result.get("tenant_access_token")


def send_post_message(token: str, title: str, blocks: list[list[dict]]) -> bool:
    url = f"{FEISHU_API}/im/v1/messages?receive_id_type=open_id"
    post_body = {"zh_cn": {"title": title, "content": blocks}}
    payload = {
        "receive_id": FEISHU_USER_OPEN_ID,
        "msg_type": "post",
        "content": json.dumps(post_body, ensure_ascii=False),
    }
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {token}",
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
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

    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        f"{FEISHU_API}/im/v1/files",
        "-H",
        f"Authorization: Bearer {token}",
        "-F",
        "file_type=stream",
        "-F",
        f"file_name={report_path.name}",
        "-F",
        f"file=@{report_path}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=40, check=False)
        if proc.returncode != 0:
            return None
        result = json.loads(proc.stdout)
        return result.get("data", {}).get("file_key")
    except Exception:
        return None


def upload_chart_to_feishu(token: str, image_path: Path) -> str | None:
    if not image_path.exists():
        return None

    import io

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="image_type"\r\n\r\n')
    body.write(b"message\r\n")
    body.write(f"--{boundary}\r\n".encode())
    body.write(
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'.encode()
    )
    body.write(b"Content-Type: image/png\r\n\r\n")
    body.write(image_path.read_bytes())
    body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())

    url = f"{FEISHU_API}/im/v1/images"
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Authorization": f"Bearer {token}",
    }
    req = urllib.request.Request(url, data=body.getvalue(), headers=headers)
    try:
        with _feishu_opener.open(req, timeout=30) as resp:
            result = json.loads(resp.read())
        return result.get("data", {}).get("image_key")
    except Exception:
        return None


def _step_line(step: dict) -> str:
    ok = bool(step.get("ok", False))
    icon = "✅" if ok else "❌"
    name = step.get("name", "未知步骤")
    elapsed = step.get("elapsed", 0)

    parts: list[str] = []
    if "updated" in step or "skipped" in step or "failed" in step:
        parts.append(f"更新{step.get('updated', 0)}")
        parts.append(f"跳过{step.get('skipped', 0)}")
        parts.append(f"失败{step.get('failed', 0)}")
    elif "count" in step:
        parts.append(f"{step.get('count', 0)}只")
    elif "ok_count" in step or "skip_count" in step or "fail_count" in step:
        parts.append(f"成功{step.get('ok_count', 0)}")
        parts.append(f"跳过{step.get('skip_count', 0)}")
        parts.append(f"失败{step.get('fail_count', 0)}")

    detail = " ".join(parts).strip()
    if detail:
        return f"{icon} {name}  {detail}  {elapsed}s"
    return f"{icon} {name}  {elapsed}s"


def build_update_blocks(status: dict | None) -> tuple[str, list[list[dict]]]:
    if not status:
        return (
            "🗄 数据更新报告（未找到结果）",
            [[{"tag": "text", "text": "⚠️ 未找到 status.json，请检查 07:00 数据更新任务是否成功运行。"}]],
        )

    update_date = status.get("update_date", "unknown")
    all_ok = bool(status.get("all_ok", False))
    steps = status.get("steps", [])
    total_elapsed = status.get("total_elapsed", 0)

    title = f"🗄 数据更新报告 {update_date} {'✅' if all_ok else '⚠️'}"
    lines = [_step_line(s) for s in steps]
    lines.append(f"总耗时: {total_elapsed}s")

    error_lines = []
    for s in steps:
        if not s.get("ok", False) and s.get("error"):
            error_lines.append(f"{s.get('name', '步骤')}: {s.get('error')}")

    blocks: list[list[dict]] = [[{"tag": "text", "text": "\n".join(lines)}]]
    if error_lines:
        blocks.append([{"tag": "text", "text": "错误详情:\n" + "\n".join(error_lines[:5])}])
    return title, blocks


def _group_signals(signals: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {"STRONG_BUY": [], "BUY": [], "HOLD": [], "AVOID": []}
    for s in signals:
        sig = s.get("signal", "HOLD")
        if sig not in grouped:
            sig = "HOLD"
        grouped[sig].append(s)
    return grouped


def build_scan_blocks(summary: dict | None, token: str) -> tuple[str, list[list[dict]]]:
    if not summary:
        return (
            "📊 Vegas 扫描报告（未找到结果）",
            [[{"tag": "text", "text": "⚠️ 未找到 summary.json，请检查 08:00 扫描任务是否成功运行。"}]],
        )

    scan_date = summary.get("scan_date", "unknown")
    market_label = summary.get("market_label") or "Vegas 扫描"
    total_scanned = summary.get("total_scanned", 0)
    signals_found = summary.get("signals_found", 0)
    has_gemini = bool(summary.get("has_gemini_analysis", False))
    report_path = summary.get("gemini_report_path") or "N/A"
    signals = summary.get("signals", [])

    title = f"📊 {market_label} {scan_date}"
    head = [
        f"扫描总数: {total_scanned}",
        f"触发信号: {signals_found}",
        f"Gemini分析: {'完成' if has_gemini else '未完成'}",
        f"报告: {report_path}",
    ]

    grouped = _group_signals(signals)
    blocks: list[list[dict]] = [[{"tag": "text", "text": "\n".join(head)}]]

    order = ["STRONG_BUY", "BUY", "HOLD", "AVOID"]
    icon = {"STRONG_BUY": "🟢", "BUY": "🔵", "HOLD": "🟡", "AVOID": "🔴"}

    for sig in order:
        rows = grouped[sig]
        if not rows:
            continue
        lines = [f"{icon[sig]} {sig}"]
        for r in rows:
            symbol = r.get("symbol", "")
            name = r.get("name", "")
            score = r.get("score", "")
            entry = r.get("entry_date", "")
            gm = r.get("gemini_conclusion", "")
            gm_f = r.get("gemini_fundamental_score", "")
            gm_v = r.get("gemini_valuation_score", "")
            gm_t = r.get("gemini_technical_score", "")

            if isinstance(score, (int, float)):
                score_text = f"{score:+.0f}" if float(score).is_integer() else f"{score:+.2f}"
            else:
                score_text = str(score)
            line = f"• {symbol} ({name}) score={score_text} 入场:{entry}"
            lines.append(line)
            if gm:
                score_detail = []
                if gm_f:
                    score_detail.append(f"基本面{gm_f}")
                if gm_v:
                    score_detail.append(f"估值{gm_v}")
                if gm_t:
                    score_detail.append(f"技术{gm_t}")
                tail = f" ({', '.join(score_detail)})" if score_detail else ""
                lines.append(f"  Gemini: {gm}{tail}")

        blocks.append([{"tag": "text", "text": "\n".join(lines)}])

    # Push chart images for STRONG_BUY / BUY / HOLD / AVOID.
    for sig in ("STRONG_BUY", "BUY", "HOLD", "AVOID"):
        for r in grouped[sig]:
            chart_path = r.get("chart_path", "")
            if not chart_path:
                continue
            p = Path(chart_path)
            image_key = upload_chart_to_feishu(token, p)
            if image_key:
                blocks.append([
                    {
                        "tag": "text",
                        "text": f"{sig} 图表: {r.get('symbol', '')}",
                    }
                ])
                blocks.append([
                    {
                        "tag": "img",
                        "image_key": image_key,
                    }
                ])

    table = summary.get("gemini_summary_table", "")
    if table:
        clipped = table[:2800]
        blocks.append([{"tag": "text", "text": "Gemini 汇总表:\n" + clipped}])

    return title, blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Notify daily update + Vegas scan result to main agent.")
    parser.add_argument("--scan-exit-code", type=int, default=0, help="Exit code of vegas_mid_daily_scan.py")
    parser.add_argument("--no-email", action="store_true", help="跳过邮件发送，仅发飞书消息")
    parser.add_argument(
        "--market",
        choices=["us", "hk", "combined"],
        default="combined",
        help="指定发哪个市场的扫描通知（默认 combined = 全部）",
    )
    return parser.parse_args()


def _send_scan_notification(token: str, summary: dict | None, scan_exit_code: int = 0, no_email: bool = False) -> bool:
    """发送单个市场的扫描通知（飞书 + 邮件）。返回是否成功。"""
    ok = True
    title2, blocks2 = build_scan_blocks(summary, token)
    if scan_exit_code != 0:
        blocks2.insert(
            0,
            [{"tag": "text", "text": f"⚠️ 扫描脚本退出码: {scan_exit_code}，请检查日志。"}],
        )
        title2 = f"{title2} ⚠️"

    if not send_post_message(token, title2, blocks2):
        print(f"❌ 扫描消息发送失败（{title2}）")
        ok = False

    report_path_raw = summary.get("gemini_report_path") if summary else None
    if report_path_raw:
        report_path = Path(report_path_raw)
        file_key = upload_report_file(token, report_path)
        if file_key:
            market_label = (summary or {}).get("market_label") or "每日扫描"
            send_post_message(
                token,
                f"📎 {market_label} Gemini 报告文件",
                [[{"tag": "text", "text": "已附上 Gemini Markdown 报告文件。"}]],
            )
            if not send_file_message(token, file_key):
                print("⚠️ 报告文件卡片发送失败")

    # 邮件
    if not no_email and report_path_raw:
        report_path = Path(report_path_raw)
        if report_path.exists():
            try:
                from stock_ana.utils.email_sender import send_report_with_charts
                md_content = report_path.read_text(encoding="utf-8")
                scan_date_str = (summary or {}).get("scan_date", date.today().isoformat())
                signals_found = (summary or {}).get("signals_found", 0)
                signals = (summary or {}).get("signals", [])
                market_label = (summary or {}).get("market_label") or "每日扫描"

                chart_paths, chart_labels = [], []
                for s in signals:
                    cp = Path(s.get("chart_path", ""))
                    if cp.exists():
                        chart_paths.append(cp)
                        chart_labels.append(f"{s.get('symbol','')} ({s.get('name','')})")

                email_sent = send_report_with_charts(
                    subject=f"📊 {market_label} {scan_date_str}（{signals_found} 只信号）",
                    md_content=md_content,
                    chart_paths=chart_paths,
                    signal_labels=chart_labels,
                    to=["99772120@qq.com", "80728495@qq.com"],
                )
                if not email_sent:
                    print("⚠️ 邮件发送失败")
            except Exception as e:
                print(f"⚠️ 邮件发送异常: {e}")

    return ok


def main() -> int:
    args = parse_args()

    status_path = _find_today_or_latest(DAILY_UPDATE_DIR, "status.json")
    status = _read_json(status_path) if status_path else None

    token = get_tenant_token()
    if not token:
        print("❌ 获取飞书 token 失败")
        return 1

    ok = True

    # Notification 1: update report
    title1, blocks1 = build_update_blocks(status)
    if not send_post_message(token, title1, blocks1):
        print("❌ 数据更新消息发送失败")
        ok = False

    # Notification 2 & 3: 美股 + 港股扫描（combined 模式）；如仅有一个则单独发
    summary_us_path = _find_today_or_latest(DAILY_SCAN_DIR, "summary_us.json")
    summary_hk_path = _find_today_or_latest(DAILY_SCAN_DIR, "summary_hk.json")
    summary_legacy_path = _find_today_or_latest(DAILY_SCAN_DIR, "summary.json")

    summary_us = _read_json(summary_us_path) if summary_us_path else None
    summary_hk = _read_json(summary_hk_path) if summary_hk_path else None
    summary_legacy = _read_json(summary_legacy_path) if summary_legacy_path else None

    send_us = args.market in ("us", "combined")
    send_hk = args.market in ("hk", "combined")

    if summary_us or summary_hk:
        # 按 --market 参数决定发哪个市场
        if send_us and summary_us:
            if not _send_scan_notification(token, summary_us, args.scan_exit_code, args.no_email):
                ok = False
        elif send_us and not summary_us:
            print("⚠️ 未找到 summary_us.json，跳过美股通知")
        if send_hk and summary_hk:
            if not _send_scan_notification(token, summary_hk, 0, args.no_email):
                ok = False
        elif send_hk and not summary_hk:
            print("⚠️ 未找到 summary_hk.json，跳过港股通知")
    else:
        # 向后兼容：读旧 summary.json
        if not _send_scan_notification(token, summary_legacy, args.scan_exit_code, args.no_email):
            ok = False

    if ok:
        print("✅ main agent 通知发送完成")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
