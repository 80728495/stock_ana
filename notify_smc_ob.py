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
import importlib.util
import json
import re
import subprocess
import sys
import textwrap
import urllib.request
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SMC_OUT_DIR  = PROJECT_ROOT / "data" / "output" / "smc_ob_scan"
HOLDING_PATH = PROJECT_ROOT / "data" / "lists" / "holding.md"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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


def _send_file(token: str, file_key: str) -> bool:
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
    with _opener.open(req, timeout=20) as resp:
        result = json.loads(resp.read())
    ok = result.get("code") == 0
    if not ok:
        print(f"[feishu] 文件消息发送失败: {result}")
    return ok


def _upload_file(token: str, file_path: Path) -> str | None:
    if not file_path.exists():
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
        f"file_name={file_path.name}",
        "-F",
        f"file=@{file_path}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        if proc.returncode != 0:
            print(f"[feishu] 文件上传失败: {proc.stderr[:200]}")
            return None
        result = json.loads(proc.stdout)
        if result.get("code") != 0:
            print(f"[feishu] 文件上传失败: {result}")
            return None
        return result.get("data", {}).get("file_key")
    except Exception as exc:
        print(f"[feishu] 文件上传异常: {exc}")
        return None


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


def _normalize_symbol(market: str, symbol: str) -> tuple[str, str] | None:
    market_norm = str(market).strip().upper()
    symbol_norm = str(symbol).strip().upper()
    if not market_norm or not symbol_norm:
        return None
    if market_norm == "HK":
        symbol_norm = symbol_norm.zfill(5)
    elif market_norm == "CN":
        symbol_norm = symbol_norm.zfill(6)
    elif market_norm != "US":
        return None
    return market_norm, symbol_norm


def _load_holding_symbols(path: Path = HOLDING_PATH) -> set[tuple[str, str]]:
    """读取 holding.md 的全部标的（持仓 + 关注 + 观察），返回 (market, symbol) 集合。"""
    if not path.exists():
        print(f"[smc-notify] holding filter missing: {path}")
        return set()

    symbols: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        parts = [p.strip() for p in stripped.strip("|").split("|")]
        if len(parts) < 2:
            continue
        symbol = parts[0]
        market = parts[1]
        if not symbol or re.fullmatch(r"[-: ]+", symbol) or symbol.lower() in {"代码", "code"}:
            continue
        key = _normalize_symbol(market, symbol)
        if key is not None:
            symbols.add(key)
    return symbols


def _filter_payload_to_holding(payload: dict, holding_symbols: set[tuple[str, str]]) -> dict:
    """仅保留 holding.md 范围内的 OB 事件。"""
    filtered = dict(payload)
    total_before = sum(len(payload.get(k, [])) for k in ("new_ob", "mitigated", "touched"))

    def keep(event: dict) -> bool:
        key = _normalize_symbol(event.get("market", ""), event.get("symbol", ""))
        return key in holding_symbols if key is not None else False

    for key in ("new_ob", "mitigated", "touched"):
        filtered[key] = [event for event in payload.get(key, []) if keep(event)]

    filtered["source_total"] = payload.get("total", total_before)
    filtered["total"] = sum(len(filtered.get(k, [])) for k in ("new_ob", "mitigated", "touched"))
    filtered["notify_scope"] = "holding"
    return filtered


# ─────────────────────── PDF 图文报告 ────────────────────────────────────────

def _chart_events(payload: dict) -> list[dict]:
    """返回需要配图的事件股票；同一股票只生成一张图。"""
    best: dict[tuple[str, str], tuple[int, dict]] = {}
    priorities = {"new_ob": 0, "touched": 1, "mitigated": 2}
    for event_name in ("new_ob", "touched", "mitigated"):
        for event in payload.get(event_name, []):
            key = _normalize_symbol(event.get("market", ""), event.get("symbol", ""))
            if key is None:
                continue
            candidate = dict(event)
            candidate["event"] = event.get("event") or event_name
            priority = priorities[event_name]
            prev = best.get(key)
            if prev is None or priority < prev[0]:
                best[key] = (priority, candidate)
    return [
        item[1]
        for item in sorted(best.values(), key=lambda x: (x[0], str(x[1].get("market", "")), str(x[1].get("symbol", ""))))
    ]


def _load_chart_module():
    script_path = PROJECT_ROOT / "scripts" / "gen_ob_score_charts.py"
    spec = importlib.util.spec_from_file_location("gen_ob_score_charts", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载图表脚本: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_event_chart_images(
    payload: dict,
    lookback: int = 500,
    swing_length: int = 5,
) -> list[Path]:
    events = _chart_events(payload)
    if not events:
        return []

    today = payload.get("date", date.today().isoformat())
    chart_dir = SMC_OUT_DIR / f"{today}_pdf_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    chart_mod = _load_chart_module()
    chart_paths: list[Path] = []

    for event in events:
        market = str(event["market"]).lower()
        symbol = str(event["symbol"])
        name = str(event.get("name") or symbol)
        df = chart_mod._load_cache(symbol, market)
        if df is None:
            print(f"[smc-notify] 图表跳过，缓存不存在: {market.upper()}:{symbol}")
            continue
        chart_path = chart_mod._plot_ob_scored(
            symbol,
            market,
            name,
            df,
            lookback,
            swing_length,
            chart_dir,
        )
        if chart_path:
            chart_paths.append(chart_path)

    print(f"[smc-notify] OB 图表已生成: {len(chart_paths)} 张")
    return chart_paths


def _find_pdf_font_path() -> str | None:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/Library/Fonts/Arial Unicode MS.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _direction_label(direction: int) -> str:
    return "看跌" if int(direction) == -1 else "看涨"


def _format_event_line(event: dict, extra: str = "") -> str:
    direction = _direction_label(int(event.get("direction", 0)))
    bottom = float(event.get("bottom", 0))
    top = float(event.get("top", 0))
    score = float(event.get("zone_score", event.get("score", 0)))
    formed = event.get("formed_date", "?")
    suffix = f"  {extra}" if extra else ""
    return (
        f"  {event.get('market')}:{event.get('symbol'):<8}  {direction}  "
        f"[{bottom:.2f} ~ {top:.2f}]  形成={formed}  评分={score:.0f}{suffix}"
    )


def _build_report_lines(payload: dict) -> list[str]:
    today = payload.get("date", date.today().isoformat())
    new_obs = payload.get("new_ob", [])
    mitigated = payload.get("mitigated", [])
    touched = payload.get("touched", [])
    total = payload.get("total", len(new_obs) + len(mitigated) + len(touched))
    source_total = payload.get("source_total")

    lines = [
        f"SMC OB 日报  {today}",
        "通知范围：holding.md（持仓 + 关注 + 观察）",
    ]
    if source_total is not None and source_total != total:
        lines.append(f"事件过滤：Futu watchlist 全量 {source_total} 条 -> Holding 通知 {total} 条")
    else:
        lines.append(f"事件总数：{total} 条")
    lines.append(f"新 OB={len(new_obs)}  失效={len(mitigated)}  触碰={len(touched)}")
    lines.append("")

    if total == 0:
        lines.append("今日 holding 范围内无 OB 事件。")
        return lines

    if new_obs:
        lines.append(f"【新生成 OB】{len(new_obs)} 个")
        for event in sorted(new_obs, key=lambda x: (x["direction"], x["market"], x["symbol"])):
            lines.append(_format_event_line(event, f"强度={float(event.get('percentage', 0)):.0f}%"))
        lines.append("")

    if mitigated:
        lines.append(f"【OB 失效 / 被突破】{len(mitigated)} 个")
        for event in sorted(mitigated, key=lambda x: (x["direction"], x["market"], x["symbol"])):
            lines.append(_format_event_line(event, f"失效={event.get('mitigated_date', '?')}"))
        lines.append("")

    if touched:
        lines.append(f"【价格触碰 OB】{len(touched)} 个")
        sorted_touched = sorted(touched, key=lambda x: (0 if x["direction"] == -1 else 1, x["market"], x["symbol"]))
        for event in sorted_touched:
            extra = (
                f"收盘={float(event.get('current_close', 0)):.2f}  "
                f"H={float(event.get('current_high', 0)):.2f}  L={float(event.get('current_low', 0)):.2f}"
            )
            lines.append(_format_event_line(event, extra))
        lines.append("")

    return lines


def _wrapped_lines(lines: list[str], width: int = 98) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        indent = "  " if line.startswith("  ") else ""
        parts = textwrap.wrap(line, width=width, subsequent_indent=indent, break_long_words=False)
        wrapped.extend(parts or [line])
    return wrapped


def _build_smc_pdf(payload: dict, chart_paths: list[Path]) -> Path:
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.font_manager import FontProperties
    from PIL import Image as PILImage

    today = payload.get("date", date.today().isoformat())
    pdf_path = SMC_OUT_DIR / f"smc_{today}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    font_path = _find_pdf_font_path()
    fp_title = FontProperties(fname=font_path, size=16) if font_path else FontProperties(size=16)
    fp_body = FontProperties(fname=font_path, size=9.5) if font_path else FontProperties(size=9.5)
    fp_small = FontProperties(fname=font_path, size=8.5) if font_path else FontProperties(size=8.5)

    lines = _wrapped_lines(_build_report_lines(payload))
    lines_per_page = 44

    with PdfPages(pdf_path) as pdf:
        for page_idx in range(0, max(1, len(lines)), lines_per_page):
            chunk = lines[page_idx:page_idx + lines_per_page]
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            header = f"SMC OB 日报  {today}" if page_idx == 0 else f"SMC OB 日报  {today}（续）"
            ax.text(0.06, 0.965, header, fontproperties=fp_title, va="top", color="#16213e")
            y = 0.925
            for line in chunk:
                ax.text(0.06, y, line, fontproperties=fp_body, va="top", color="#222222")
                y -= 0.0205 if line else 0.014
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        for chart_path in chart_paths:
            if not chart_path.exists():
                continue
            fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
            ax = fig.add_axes([0.035, 0.06, 0.93, 0.86])
            ax.axis("off")
            fig.text(0.5, 0.965, chart_path.stem, ha="center", va="top", fontproperties=fp_small, color="#222222")
            try:
                with PILImage.open(chart_path) as img:
                    ax.imshow(img.convert("RGB"))
            except Exception as exc:
                ax.text(0.5, 0.5, f"图表加载失败: {chart_path.name}\n{exc}", ha="center", va="center", fontproperties=fp_body)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"[smc-notify] PDF 已生成: {pdf_path.name} ({len(chart_paths)} 张图)")
    return pdf_path


# ─────────────────────── 主入口 ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SMC OB 日报飞书通知")
    parser.add_argument("--date",        default=date.today().isoformat(),
                        help="事件文件日期（默认今日）")
    parser.add_argument("--always-send", action="store_true",
                        help="即使无事件也发送（确认流程正常运行）")
    parser.add_argument("--no-charts", "--no-bear-charts", dest="no_charts", action="store_true",
                        help="PDF 中只放文字，不附图（--no-bear-charts 为旧参数别名）")
    parser.add_argument("--chart-lookback", "--bear-chart-lookback", dest="chart_lookback",
                        type=int, default=500,
                        help="PDF 图表显示最近 N 根 K 线（默认 500）")
    args = parser.parse_args()

    # ── 读取事件 ─────────────────────────────────────────────────────────
    events_path = _find_events_file(args.date)
    if not events_path:
        print(f"[smc-notify] 未找到事件文件 ({args.date}_futu_events.json)，退出")
        return

    with open(events_path, encoding="utf-8") as f:
        payload = json.load(f)

    source_total = payload.get("total", 0)
    print(f"[smc-notify] 读取事件文件: {events_path.name}  total={source_total}")

    holding_symbols = _load_holding_symbols()
    payload = _filter_payload_to_holding(payload, holding_symbols)
    total = payload.get("total", 0)
    print(
        f"[smc-notify] holding filter: symbols={len(holding_symbols)}  "
        f"events={source_total}->{total}"
    )

    # ── 无事件时根据参数决定是否发送 ─────────────────────────────────────
    if total == 0 and not args.always_send:
        print("[smc-notify] 今日无 OB 事件，跳过发送（可用 --always-send 强制发送）")
        return

    chart_paths: list[Path] = []
    if not args.no_charts and total > 0:
        try:
            chart_paths = _build_event_chart_images(
                payload,
                lookback=args.chart_lookback,
                swing_length=5,
            )
        except Exception as exc:
            print(f"[smc-notify] OB 图表生成失败，将仅发送文字 PDF: {exc}")

    try:
        pdf_path = _build_smc_pdf(payload, chart_paths)
    except Exception as exc:
        print(f"[smc-notify] PDF 生成失败: {exc}")
        return

    # ── 获取 token ───────────────────────────────────────────────────────
    token = _get_token()
    if not token:
        print("[smc-notify] 获取飞书 token 失败")
        return

    file_key = _upload_file(token, pdf_path)
    if not file_key:
        print("[smc-notify] ❌ SMC PDF 上传失败")
        return

    if _send_file(token, file_key):
        print("[smc-notify] ✅ SMC PDF 已发送飞书")
    else:
        print("[smc-notify] ❌ SMC PDF 文件消息发送失败")


if __name__ == "__main__":
    main()
