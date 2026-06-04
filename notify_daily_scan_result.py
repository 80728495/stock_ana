#!/usr/bin/env python3
"""Send daily update + Vegas scan summary to main agent via Feishu."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from datetime import date
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

DAILY_UPDATE_DIR = OUTPUT_DIR / "daily_update"
DAILY_SCAN_DIR = OUTPUT_DIR / "daily_scan"

FEISHU_APP_ID = "cli_a924285ae7f85cc7"
FEISHU_APP_SECRET = "53hrIbxJYHGGAI8qbndwofOzltJAkah0"
FEISHU_USER_OPEN_ID = "ou_5489407346c5c13bc4687a83859d619b"
FEISHU_API = "https://open.feishu.cn/open-apis"
DAILY_VEGUS_EMAIL_RECIPIENTS = [
    "99772120@qq.com",
    "80728495@qq.com",
    "185182@qq.com",
    "tsy_fever@163.COM",
]

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


def _find_today_file(root: Path, filename: str) -> Path | None:
    today_file = root / date.today().isoformat() / filename
    return today_file if today_file.exists() else None


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
            "⚠️ 数据更新状态未知",
            [[{"tag": "text", "text": "未找到 status.json。\n请检查每日数据更新任务是否已运行。"}]],
        )

    update_date = status.get("update_date", "unknown")
    all_ok = bool(status.get("all_ok", False))
    steps = status.get("steps", [])
    total_elapsed = status.get("total_elapsed", 0)

    failed_steps = [s for s in steps if not s.get("ok", False)]

    if all_ok:
        title = f"✅ 数据更新完成 {update_date}"
    else:
        failed_names = "、".join(s.get("name", "?") for s in failed_steps[:3])
        title = f"❌ 数据更新失败 {update_date}（{failed_names}）"

    lines = [_step_line(s) for s in steps]
    lines.append(f"\n总耗时: {total_elapsed}s")
    blocks: list[list[dict]] = [[{"tag": "text", "text": "\n".join(lines)}]]

    if failed_steps:
        err_lines = []
        for s in failed_steps:
            if s.get("error"):
                err_lines.append(f"❌ {s.get('name', '?')}: {str(s.get('error', ''))[:120]}")
        if err_lines:
            blocks.append([{"tag": "text", "text": "失败详情:\n" + "\n".join(err_lines)}])
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


def _format_signals_text(signals: list[dict]) -> str:
    """将信号列表格式化为简洁文本，用于飞书卡片摘要。"""
    if not signals:
        return "无触发信号"
    icon = {"STRONG_BUY": "🟢", "BUY": "🔵", "HOLD": "🟡", "AVOID": "🔴"}
    grouped: dict[str, list] = {"STRONG_BUY": [], "BUY": [], "HOLD": [], "AVOID": []}
    for s in signals:
        sig = s.get("signal", "HOLD")
        grouped.get(sig, grouped["HOLD"]).append(s)
    lines = []
    for sig in ("STRONG_BUY", "BUY", "HOLD", "AVOID"):
        rows = grouped[sig]
        if not rows:
            continue
        lines.append(f"{icon.get(sig, '●')} {sig}")
        for r in rows:
            sym = r.get("symbol", "")
            name = r.get("name", "")
            score = r.get("score", "")
            entry = r.get("entry_date", "")
            band = r.get("support_band", "")
            score_text = f"{score:+.0f}" if isinstance(score, (int, float)) else str(score)
            lines.append(f"  • {sym} ({name})  score={score_text}  {band}  入场:{entry}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Notify daily update + Vegas scan result to main agent.")
    parser.add_argument("--scan-exit-code", type=int, default=0, help="Exit code of vegas_mid_daily_scan.py")
    parser.add_argument("--no-email", action="store_true", help="跳过邮件发送，仅发飞书消息")
    parser.add_argument(
        "--market",
        choices=["us", "hk", "cn", "combined", "combined_cn"],
        default="combined",
        help="指定发哪个市场的扫描通知（默认 combined = 美股+港股；combined_cn = 美股+港股+A股）",
    )
    parser.add_argument(
        "--skip-update",
        action="store_true",
        help="跳过每日数据更新状态通知（分步发送时避免重复）",
    )
    parser.add_argument(
        "--no-new-data",
        action="store_true",
        help="数据未更新场景：只发飞书数据未更新通知，跳过所有扫描通知",
    )
    parser.add_argument(
        "--send-combined-email",
        action="store_true",
        help="汇总所有市场 PDF 发送一封合并邮件（各市场飞书已单独发送，此步仅做邮件合并）",
    )
    return parser.parse_args()


def _build_pdf_for_summary(summary: dict) -> bytes | None:
    """为一个市场的 summary 生成 PDF bytes，仅在 has_gemini_analysis=True 时有效。"""
    market_label = summary.get("market_label") or "每日扫描"
    scan_date_str = summary.get("scan_date", date.today().isoformat())
    signals_found = summary.get("signals_found", 0)
    signals = summary.get("signals", [])
    report_path_raw = summary.get("gemini_report_path")

    chart_paths: list[Path] = []
    chart_labels: list[str] = []
    for s in signals:
        cp = Path(s.get("chart_path", ""))
        if cp.exists():
            chart_paths.append(cp)
            chart_labels.append(f"{s.get('symbol','')} ({s.get('name','')})")

    title = f"📊 {market_label} {scan_date_str}（{signals_found} 只信号）"
    md_content = ""
    if report_path_raw:
        rp = Path(report_path_raw)
        if rp.exists():
            md_content = rp.read_text(encoding="utf-8")

    if not md_content:
        return None

    try:
        from stock_ana.utils.pdf_builder import build_scan_pdf
        return build_scan_pdf(
            md_content=md_content,
            chart_paths=chart_paths,
            signal_labels=chart_labels,
            title=title,
            signals=signals,
        )
    except Exception as e:
        print(f"⚠️ PDF 生成失败 ({market_label}): {e}")
        return None


def _scan_pdf_name(market_slug: str, scan_date_str: str) -> str:
    slug = (market_slug or "unknown").strip().lower()
    return f"vegus_{slug}_{scan_date_str}.pdf"


def _write_scan_pdf(pdf_bytes: bytes, market_slug: str, scan_date_str: str) -> Path:
    out_dir = DAILY_SCAN_DIR / scan_date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / _scan_pdf_name(market_slug, scan_date_str)
    pdf_path.write_bytes(pdf_bytes)
    return pdf_path


def _infer_market_slug(summary: dict | None, fallback: str = "unknown") -> str:
    if not summary:
        return fallback
    label = str(summary.get("market_label") or "").lower()
    report_path = str(summary.get("gemini_report_path") or "").lower()
    if "cn" in report_path or "a股" in label or "高新" in label:
        return "cn"
    if "hk" in report_path or "港股" in label:
        return "hk"
    if "us" in report_path or "美股" in label:
        return "us"
    return fallback


def _send_scan_notification(
    token: str,
    summary: dict | None,
    scan_exit_code: int = 0,
    no_email: bool = False,
    market_slug: str = "unknown",
) -> bool:
    """
    扫描通知三分支：
      A. scan 退出码非0：只发错误卡片
      B. scan OK + Gemini 失败：飞书摘要卡片 + 逐张图表图片，不发 PDF，不发邮件
      C. scan OK + Gemini 成功：飞书摘要卡片 + PDF 文件，邮件发 PDF，不单独发图片
    """
    ok = True
    market_label = (summary or {}).get("market_label") or "每日扫描"
    scan_date_str = (summary or {}).get("scan_date", date.today().isoformat())
    total_scanned = (summary or {}).get("total_scanned", 0)
    signals_found = (summary or {}).get("signals_found", 0)
    signals       = (summary or {}).get("signals", [])
    has_gemini    = bool((summary or {}).get("has_gemini_analysis", False))
    gemini_status = str((summary or {}).get("gemini_status") or "").lower()
    report_path_raw = (summary or {}).get("gemini_report_path")

    # 收集有效图表路径
    chart_paths:  list[Path] = []
    chart_labels: list[str]  = []
    for s in signals:
        cp = Path(s.get("chart_path", ""))
        if cp.exists():
            chart_paths.append(cp)
            chart_labels.append(f"{s.get('symbol','')} ({s.get('name','')})")

    # ── 分支 A：扫描本身失败 ────────────────────────────────────────────
    if scan_exit_code != 0:
        send_post_message(
            token,
            f"❌ {market_label} 扫描失败 {scan_date_str}",
            [[{"tag": "text", "text": f"扫描脚本退出码: {scan_exit_code}，请检查日志。"}]],
        )
        return False

    title = f"📊 {market_label} {scan_date_str}（{signals_found} 只信号）"
    sig_text = _format_signals_text(signals)
    head_text = f"扫描：{total_scanned} 只  |  信号：{signals_found} 只"

    # ── 分支 B：只有 Gemini 最终失败才发半成品图文消息 ──
    if not has_gemini:
        if gemini_status != "failed":
            status_text = gemini_status or "unknown"
            print(
                f"⚠️ {market_label} {scan_date_str} Gemini 未完成但非最终失败 "
                f"(status={status_text})，跳过飞书半成品消息"
            )
            return ok

        summary_text = head_text + "  |  Gemini：最终失败\n\n" + sig_text
        img_blocks: list[list[dict]] = []
        for cp, label in zip(chart_paths, chart_labels):
            image_key = upload_chart_to_feishu(token, cp)
            if image_key:
                img_blocks.append([{"tag": "text", "text": label}])
                img_blocks.append([{"tag": "img", "image_key": image_key}])
        all_blocks = [[{"tag": "text", "text": summary_text}]] + img_blocks
        if not send_post_message(token, title, all_blocks):
            print(f"❌ 扫描半成品消息发送失败（{title}）")
            ok = False
        return ok

    # ── 分支 C：Gemini 成功 → 直接发 PDF（飞书 + 邮件），不发文字卡 ──────
    md_content = ""
    if report_path_raw:
        rp = Path(report_path_raw)
        if rp.exists():
            md_content = rp.read_text(encoding="utf-8")

    # 生成 PDF
    pdf_bytes: bytes | None = None
    try:
        from stock_ana.utils.pdf_builder import build_scan_pdf
        pdf_bytes = build_scan_pdf(
            md_content=md_content,
            chart_paths=chart_paths,
            signal_labels=chart_labels,
            title=title,
            signals=signals,
        )
    except Exception as e:
        print(f"⚠️ PDF 生成失败: {e}")
        ok = False

    # 飞书：发送 PDF 文件
    if pdf_bytes:
        pdf_slug = _infer_market_slug(summary, market_slug)
        pdf_path = _write_scan_pdf(pdf_bytes, pdf_slug, scan_date_str)
        file_key = upload_report_file(token, pdf_path)
        if file_key:
            if send_file_message(token, file_key):
                print(f"✅ 飞书 PDF 已发送：{pdf_path.name}")
            else:
                print("⚠️ 飞书 PDF 文件消息发送失败")
                ok = False
        else:
            print("⚠️ 飞书 PDF 上传失败")
            ok = False

    # 邮件：发送 PDF 附件
    if not no_email and md_content and pdf_bytes:
        try:
            from stock_ana.utils.email_sender import send_report_with_charts
            email_sent = send_report_with_charts(
                subject=title,
                md_content=md_content,
                chart_paths=chart_paths,
                signal_labels=chart_labels,
                to=DAILY_VEGUS_EMAIL_RECIPIENTS,
                signals=signals,
            )
            if not email_sent:
                print("⚠️ 邮件发送失败")
        except Exception as e:
            print(f"⚠️ 邮件发送异常: {e}")

    return ok


def main() -> int:
    args = parse_args()

    status_path = _find_today_file(DAILY_UPDATE_DIR, "status.json")
    status = _read_json(status_path) if status_path else None

    token = get_tenant_token()
    if not token:
        print("❌ 获取飞书 token 失败")
        return 1

    ok = True

    # ── --send-combined-email：三市场合并邮件（各市场飞书已单独发完，只发邮件）────
    if getattr(args, "send_combined_email", False):
        summary_us_path = _find_today_file(DAILY_SCAN_DIR, "summary_us.json")
        summary_hk_path = _find_today_file(DAILY_SCAN_DIR, "summary_hk.json")
        summary_cn_path = _find_today_file(DAILY_SCAN_DIR, "summary_cn.json")
        pdfs: list[tuple[bytes, str]] = []
        for mkt, path in [("us", summary_us_path), ("hk", summary_hk_path), ("cn", summary_cn_path)]:
            summary = _read_json(path) if path else None
            if not summary or not summary.get("has_gemini_analysis"):
                print(f"⚠️ {mkt} 无 Gemini 分析，跳过 PDF")
                continue
            pdf_bytes = _build_pdf_for_summary(summary)
            if pdf_bytes:
                scan_date_str = summary.get("scan_date", date.today().isoformat())
                pdfs.append((pdf_bytes, _scan_pdf_name(mkt, scan_date_str)))
        if pdfs:
            from stock_ana.utils.email_sender import send_pdf_attachments
            today_str = date.today().isoformat()
            email_sent = send_pdf_attachments(
                subject=f"📊 每日扫描综合报告 {today_str}（{len(pdfs)} 份）",
                pdfs=pdfs,
                to=DAILY_VEGUS_EMAIL_RECIPIENTS,
            )
            if email_sent:
                print(f"✅ 合并邮件已发送，共 {len(pdfs)} 份 PDF")
                return 0
            else:
                print("⚠️ 合并邮件发送失败")
                return 1
        else:
            print("⚠️ 无可用 PDF（所有市场 Gemini 均未完成），跳过邮件")
            return 0

    # ── --no-new-data：今日缓存无更新，直接发通知后退出，不发扫描结果 ──
    if getattr(args, "no_new_data", False):
        title_nd = "⚠️ 今日数据未更新，扫描已跳过"
        text_nd  = "检测到缓存文件未刷新（daily_update 可能未运行或失败）。\n今日美股/港股扫描已跳过，不重复发送昨日结果。"
        send_post_message(token, title_nd, [[{"tag": "text", "text": text_nd}]])
        if status:
            title1, blocks1 = build_update_blocks(status)
            send_post_message(token, title1, blocks1)
        print("⚠️ 今日数据未更新，已发飞书通知，退出。")
        return 0

    # Notification 1: update report（可通过 --skip-update 跳过，避免分步发送时重复）
    if not args.skip_update:
        title1, blocks1 = build_update_blocks(status)
        if not send_post_message(token, title1, blocks1):
            print("❌ 数据更新消息发送失败")
            ok = False

    # Notification 2 & 3: 美股 + 港股扫描（combined 模式）；如仅有一个则单独发
    summary_us_path = _find_today_file(DAILY_SCAN_DIR, "summary_us.json")
    summary_hk_path = _find_today_file(DAILY_SCAN_DIR, "summary_hk.json")
    summary_cn_path = _find_today_file(DAILY_SCAN_DIR, "summary_cn.json")
    summary_legacy_path = _find_today_file(DAILY_SCAN_DIR, "summary.json")

    summary_us = _read_json(summary_us_path) if summary_us_path else None
    summary_hk = _read_json(summary_hk_path) if summary_hk_path else None
    summary_cn = _read_json(summary_cn_path) if summary_cn_path else None
    summary_legacy = _read_json(summary_legacy_path) if summary_legacy_path else None

    send_us = args.market in ("us", "combined", "combined_cn")
    send_hk = args.market in ("hk", "combined", "combined_cn")
    send_cn = args.market in ("cn", "combined_cn")

    if summary_us or summary_hk or summary_cn:
        # 按 --market 参数决定发哪个市场
        if send_us and summary_us:
            if not _send_scan_notification(token, summary_us, args.scan_exit_code, args.no_email, market_slug="us"):
                ok = False
        elif send_us and not summary_us:
            print("⚠️ 未找到 summary_us.json，跳过美股通知")
        if send_hk and summary_hk:
            if summary_hk.get("data_stale"):
                print("[daily-scan] 港股数据未更新（data_stale=True），跳过港股通知")
            elif not _send_scan_notification(token, summary_hk, 0, args.no_email, market_slug="hk"):
                ok = False
        elif send_hk and not summary_hk:
            print("⚠️ 未找到 summary_hk.json，跳过港股通知")
        if send_cn and summary_cn:
            if summary_cn.get("data_stale"):
                print("[daily-scan] A股数据未更新（data_stale=True），跳过A股通知")
            elif not _send_scan_notification(token, summary_cn, 0, args.no_email, market_slug="cn"):
                ok = False
        elif send_cn and not summary_cn:
            print("⚠️ 未找到 summary_cn.json，跳过A股高新技术通知")
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
