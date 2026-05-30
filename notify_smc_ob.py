#!/usr/bin/env python3
"""
SMC OB 日报飞书通知

读取 daily_update.py 落盘的 smc_ob_scan/{date}_futu_events.json，
将当日所有 OB 事件（新生成 / OB 失效 / 价格刺入）汇总为一条飞书消息发出。

用法:
    python notify_smc_ob.py                   # 自动读取今日结果
    python notify_smc_ob.py --date 2026-05-25  # 指定日期
    python notify_smc_ob.py --always-send      # 无事件时也发送（确认正常运行）
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SMC_OUT_DIR  = PROJECT_ROOT / "data" / "output" / "smc_ob_scan"

FEISHU_APP_ID       = "cli_a924285ae7f85cc7"
FEISHU_APP_SECRET   = "53hrIbxJYHGGAI8qbndwofOzltJAkah0"
FEISHU_USER_OPEN_ID = "ou_5489407346c5c13bc4687a83859d619b"
FEISHU_API          = "https://open.feishu.cn/open-apis"

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ─────────────────────── 飞书工具 ────────────────────────────────────────────

def _get_token() -> str | None:
    url  = f"{FEISHU_API}/auth/v3/tenant_access_token/internal"
    data = json.dumps({"app_id": FEISHU_APP_ID,
                       "app_secret": FEISHU_APP_SECRET}).encode("utf-8")
    req  = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json; charset=utf-8"})
    with _opener.open(req, timeout=15) as resp:
        return json.loads(resp.read()).get("tenant_access_token")


def _send_post(token: str, title: str, blocks: list[list[dict]]) -> bool:
    url      = f"{FEISHU_API}/im/v1/messages?receive_id_type=open_id"
    body     = {"zh_cn": {"title": title, "content": blocks}}
    payload  = {
        "receive_id": FEISHU_USER_OPEN_ID,
        "msg_type":   "post",
        "content":    json.dumps(body, ensure_ascii=False),
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type":  "application/json; charset=utf-8",
            "Authorization": f"Bearer {token}",
        },
    )
    with _opener.open(req, timeout=20) as resp:
        result = json.loads(resp.read())
    ok = result.get("code") == 0
    if not ok:
        print(f"[feishu] 发送失败: {result}")
    return ok


# ─────────────────────── 消息构建 ────────────────────────────────────────────

def _dir_tag(direction: int) -> str:
    return "↓看跌" if direction == -1 else "↑看涨"


def _build_blocks(payload: dict) -> tuple[str, list[list[dict]]]:
    today      = payload.get("date", date.today().isoformat())
    new_obs    = payload.get("new_ob", [])
    mitigated  = payload.get("mitigated", [])
    touched    = payload.get("touched", [])
    total      = payload.get("total", len(new_obs) + len(mitigated) + len(touched))

    # ── 标题 ──────────────────────────────────────────────────────────────
    if total == 0:
        title = f"📋 SMC OB 日报  {today}  |  今日无事件"
    else:
        parts = []
        if new_obs:    parts.append(f"新OB {len(new_obs)}")
        if mitigated:  parts.append(f"失效 {len(mitigated)}")
        if touched:    parts.append(f"触碰 {len(touched)}")
        title = f"🔔 SMC OB 日报  {today}  |  " + "  ".join(parts)

    blocks: list[list[dict]] = []

    if total == 0:
        blocks.append([{"tag": "text", "text": "富途自选股今日无 OB 事件。"}])
        return title, blocks

    # ── 新生成 OB ──────────────────────────────────────────────────────────
    if new_obs:
        lines = [f"【新生成 OB】{len(new_obs)} 个"]
        for e in sorted(new_obs, key=lambda x: (x["direction"], x["market"], x["symbol"])):
            tag = _dir_tag(e["direction"])
            lines.append(
                f"  {e['market']}:{e['symbol']:<8}  {tag}  "
                f"[{e['bottom']:.2f} ~ {e['top']:.2f}]  "
                f"形成={e['formed_date']}  强度={e.get('percentage', 0):.0f}%"
            )
        blocks.append([{"tag": "text", "text": "\n".join(lines)}])

    # ── OB 失效 ────────────────────────────────────────────────────────────
    if mitigated:
        lines = [f"【OB 失效（被打破）】{len(mitigated)} 个"]
        for e in sorted(mitigated, key=lambda x: (x["direction"], x["market"], x["symbol"])):
            tag = _dir_tag(e["direction"])
            lines.append(
                f"  {e['market']}:{e['symbol']:<8}  {tag}  "
                f"[{e['bottom']:.2f} ~ {e['top']:.2f}]  "
                f"形成={e['formed_date']}  消除={e.get('mitigated_date', '?')}"
            )
        blocks.append([{"tag": "text", "text": "\n".join(lines)}])

    # ── 价格刺入 OB ────────────────────────────────────────────────────────
    if touched:
        # 看跌优先排序（看跌=更危险，放前面）
        sorted_touched = sorted(
            touched,
            key=lambda x: (0 if x["direction"] == -1 else 1, x["market"], x["symbol"])
        )
        lines = [f"【价格刺入 OB】{len(touched)} 个"]
        for e in sorted_touched:
            tag = _dir_tag(e["direction"])
            lines.append(
                f"  {e['market']}:{e['symbol']:<8}  {tag}  "
                f"[{e['bottom']:.2f} ~ {e['top']:.2f}]  "
                f"收盘={e.get('current_close', 0):.2f}  "
                f"(H={e.get('current_high', 0):.2f} L={e.get('current_low', 0):.2f})  "
                f"OB形成={e['formed_date']}"
            )
        blocks.append([{"tag": "text", "text": "\n".join(lines)}])

    return title, blocks


# ─────────────────────── 读取事件文件 ────────────────────────────────────────

def _find_events_file(target_date: str) -> Path | None:
    path = SMC_OUT_DIR / f"{target_date}_futu_events.json"
    if path.exists():
        return path
    # fallback: 找最近一个文件
    candidates = sorted(SMC_OUT_DIR.glob("*_futu_events.json"))
    return candidates[-1] if candidates else None


# ─────────────────────── 主入口 ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SMC OB 日报飞书通知")
    parser.add_argument("--date",        default=date.today().isoformat(),
                        help="事件文件日期（默认今日）")
    parser.add_argument("--always-send", action="store_true",
                        help="即使无事件也发送（确认流程正常运行）")
    args = parser.parse_args()

    # ── 读取事件 ─────────────────────────────────────────────────────────
    events_path = _find_events_file(args.date)
    if not events_path:
        print(f"[smc-notify] 未找到事件文件 ({args.date}_futu_events.json)，退出")
        return

    with open(events_path, encoding="utf-8") as f:
        payload = json.load(f)

    total = payload.get("total", 0)
    print(f"[smc-notify] 读取事件文件: {events_path.name}  total={total}")

    # ── 无事件时根据参数决定是否发送 ─────────────────────────────────────
    if total == 0 and not args.always_send:
        print("[smc-notify] 今日无 OB 事件，跳过发送（可用 --always-send 强制发送）")
        return

    # ── 构建消息 ─────────────────────────────────────────────────────────
    title, blocks = _build_blocks(payload)
    print(f"[smc-notify] 标题: {title}")

    # ── 获取 token ────────────────────────────────────────────────────────
    token = _get_token()
    if not token:
        print("[smc-notify] 获取飞书 token 失败")
        return

    # ── 发送消息 ─────────────────────────────────────────────────────────
    ok = _send_post(token, title, blocks)
    if ok:
        print("[smc-notify] ✅ 飞书通知已发送")
    else:
        print("[smc-notify] ❌ 飞书通知发送失败")


if __name__ == "__main__":
    main()
