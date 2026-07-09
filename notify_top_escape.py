#!/usr/bin/env python3
"""
持仓顶部逃顶信号日报飞书通知。

读取 daily_update.py 落盘的 top_candidate_research/escape_events_{date}.json，
将每日「发现 / 取消 / 下跌结构确认」事件单独发送。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TOP_OUT_DIR = PROJECT_ROOT / "data" / "output" / "top_candidate_research"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

FEISHU_APP_ID = "cli_a924285ae7f85cc7"
FEISHU_APP_SECRET = "53hrIbxJYHGGAI8qbndwofOzltJAkah0"
FEISHU_USER_OPEN_ID = "ou_5489407346c5c13bc4687a83859d619b"
FEISHU_API = "https://open.feishu.cn/open-apis"

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get_token() -> str | None:
    url = f"{FEISHU_API}/auth/v3/tenant_access_token/internal"
    data = json.dumps({"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json; charset=utf-8"})
    with _opener.open(req, timeout=15) as resp:
        return json.loads(resp.read()).get("tenant_access_token")


def _send_post(token: str, title: str, blocks: list[list[dict]]) -> bool:
    url = f"{FEISHU_API}/im/v1/messages?receive_id_type=open_id"
    body = {"zh_cn": {"title": title, "content": blocks}}
    payload = {
        "receive_id": FEISHU_USER_OPEN_ID,
        "msg_type": "post",
        "content": json.dumps(body, ensure_ascii=False),
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {token}",
        },
    )
    with _opener.open(req, timeout=20) as resp:
        result = json.loads(resp.read())
    ok = result.get("code") == 0
    if not ok:
        print(f"[top-escape-notify] 发送失败: {result}")
    return ok


def _events_path(day: str) -> Path:
    return TOP_OUT_DIR / f"escape_events_{day}.json"


def _state_path() -> Path:
    return TOP_OUT_DIR / "escape_signal_state.json"


def _load_events(day: str) -> list[dict]:
    path = _events_path(day)
    if not path.exists():
        print(f"[top-escape-notify] 未找到事件文件: {path.name}")
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


def _load_state_summary() -> tuple[int, list[str]]:
    path = _state_path()
    if not path.exists():
        return 0, []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        signals = raw.get("signals", raw if isinstance(raw, dict) else {})
        watching = [
            f"{v.get('market')}:{v.get('sym')}"
            for v in signals.values()
            if isinstance(v, dict) and v.get("state") == "watching"
        ]
        return len(watching), sorted(watching)
    except Exception as exc:
        print(f"[top-escape-notify] 状态文件读取失败: {exc}")
        return 0, []


def _fmt_num(value: object, suffix: str = "") -> str:
    try:
        x = float(value)
    except Exception:
        return "-"
    if x != x:
        return "-"
    return f"{x:.2f}{suffix}"


def _event_line(e: dict) -> str:
    market = str(e.get("market", ""))
    sym = str(e.get("sym", ""))
    top_date = str(e.get("top_date", "-"))
    day = str(e.get("date", "-"))
    msg = str(e.get("message", "")).strip()
    if e.get("kind") in ("new_signal", "alert"):
        strength = _fmt_num(e.get("strength"))
        return f"  {day} {market}:{sym:<8} 顶={top_date} strength={strength}  {msg}"
    if e.get("kind") == "confirm_down":
        drop = _fmt_num(e.get("drop_pct"), "%")
        return f"  {day} {market}:{sym:<8} 顶={top_date} 距顶={drop}  {msg}"
    return f"  {day} {market}:{sym:<8} 顶={top_date}  {msg}"


def _build_blocks(day: str, events: list[dict], always_send: bool) -> tuple[str, list[list[dict]], bool]:
    counts = Counter(str(e.get("kind", "")) for e in events)
    watching_count, watching = _load_state_summary()
    send = bool(events) or always_send

    if not events:
        title = f"持仓顶部逃顶日报 {day} | 今日无事件"
        blocks = [[{"tag": "text", "text": f"今日无新增逃顶事件。当前观察队列 {watching_count} 只。"}]]
        if watching:
            blocks.append([{"tag": "text", "text": "观察中：" + "、".join(watching[:30])}])
        return title, blocks, send

    parts = []
    discovery = counts.get("new_signal", 0) + counts.get("alert", 0)
    if discovery:
        parts.append(f"发现 {discovery}")
    if counts.get("confirm_down", 0):
        parts.append(f"二次确认 {counts['confirm_down']}")
    if counts.get("cancel", 0):
        parts.append(f"取消 {counts['cancel']}")
    title = f"持仓顶部逃顶日报 {day} | " + "  ".join(parts)

    blocks: list[list[dict]] = []
    blocks.append([{"tag": "text", "text": f"当前观察队列 {watching_count} 只。"}])

    groups = [
        ("confirm_down", "【顶部二次确认】"),
        ("cancel", "【信号取消】"),
        ("alert", "【强提醒】"),
        ("new_signal", "【新发现】"),
    ]
    for kind, label in groups:
        items = [e for e in events if e.get("kind") == kind]
        if not items:
            continue
        lines = [f"{label}{len(items)} 个"]
        lines.extend(_event_line(e) for e in sorted(items, key=lambda x: (x.get("market", ""), x.get("sym", ""))))
        blocks.append([{"tag": "text", "text": "\n".join(lines)}])

    if watching:
        blocks.append([{"tag": "text", "text": "观察中：" + "、".join(watching[:30])}])
    return title, blocks, send


def main() -> None:
    parser = argparse.ArgumentParser(description="持仓顶部逃顶信号日报飞书通知")
    parser.add_argument("--date", default=date.today().isoformat(), help="事件文件日期（默认今日）")
    parser.add_argument("--always-send", action="store_true", help="即使无事件也发送")
    args = parser.parse_args()

    events = _load_events(args.date)
    title, blocks, should_send = _build_blocks(args.date, events, args.always_send)
    print(f"[top-escape-notify] events={len(events)} title={title}")

    if not should_send:
        print("[top-escape-notify] 今日无逃顶事件，跳过发送（可用 --always-send 强制发送）")
        return

    token = _get_token()
    if not token:
        print("[top-escape-notify] 获取飞书 token 失败")
        return

    if _send_post(token, title, blocks):
        print("[top-escape-notify] 已发送飞书")
    else:
        print("[top-escape-notify] 飞书发送失败")


if __name__ == "__main__":
    main()
