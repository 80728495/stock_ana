#!/usr/bin/env python3
"""
周线 Vegas Short 扫描 + Gemini 批量分析 + 飞书 PDF 推送

流程：
  1. 运行周线 Vegas Short 扫描，找出近期 touch 信号（scan 内置图表渲染）
  2. 每最多 3 只一组发送 LLM；每批结果返回后等 180s 再发下一批
  3. 合并所有批次 LLM 文本，生成统一 PDF（封面汇总表 + 逐只图表 + 分析）
  4. 上传 PDF 到飞书并发送消息

用法：
    python weekly_vegas_short_notify.py                   # 扫描自选列表（watchlist.md）
    python weekly_vegas_short_notify.py --list us         # 美股科技列表
    python weekly_vegas_short_notify.py --list hk         # 港股宇宙池
    python weekly_vegas_short_notify.py --lookback 2      # 最近 2 周
    python weekly_vegas_short_notify.py --scan-only       # 仅扫描，不调 LLM，不发通知
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loguru import logger

LOG_DIR = PROJECT_ROOT / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(
    LOG_DIR / "w_vegas_notify_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    encoding="utf-8",
    enqueue=True,
)

# ── Feishu 配置（与 notify_daily_scan_result.py 相同）───────────────────────
FEISHU_APP_ID       = "cli_a924285ae7f85cc7"
FEISHU_APP_SECRET   = "53hrIbxJYHGGAI8qbndwofOzltJAkah0"
FEISHU_USER_OPEN_ID = "ou_5489407346c5c13bc4687a83859d619b"
FEISHU_API          = "https://open.feishu.cn/open-apis"
_feishu_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

BATCH_SIZE  = 3    # 每批最多几只
BATCH_DELAY = 180  # 批次间等待秒数

W_SCAN_OUT_DIR     = PROJECT_ROOT / "data" / "output" / "w_vegas_scan"
ANALYSIS_OUT_DIR   = PROJECT_ROOT / "data" / "output" / "scan_analysis"
SEND_GUARD_PATH    = LOG_DIR / ".w_vegas_last_sent.json"


def _build_send_fingerprint(signals: list[dict], list_mode: str, lookback: int) -> str:
    """Build a stable fingerprint to avoid duplicate sends in the same day."""
    key_parts = [f"list={list_mode}", f"lookback={lookback}"]
    for s in sorted(signals, key=lambda x: (x.get("symbol", ""), x.get("entry_date", ""), x.get("signal", ""))):
        key_parts.append(
            "|".join(
                [
                    str(s.get("symbol", "")),
                    str(s.get("entry_date", "")).split("(")[0],
                    str(s.get("signal", "")),
                    str(s.get("score", "")),
                    str(s.get("support_band", "")),
                ]
            )
        )
    return hashlib.sha256("\n".join(key_parts).encode("utf-8")).hexdigest()


def _load_send_guard() -> dict:
    if not SEND_GUARD_PATH.exists():
        return {}
    try:
        return json.loads(SEND_GUARD_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_send_guard(payload: dict) -> None:
    SEND_GUARD_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ════════════════════════════════════════════════════════════════════
#  周线指标刷新（仅对本次扫描用到的列表）
# ════════════════════════════════════════════════════════════════════

def refresh_weekly_indicators_for_lists() -> None:
    """对美股科技列表 + 港股高科技列表刷新日线 + 周线指标。"""
    from stock_ana.data.indicators_store import update_indicators_for_symbols
    from stock_ana.config import CACHE_DIR, DATA_DIR
    from stock_ana.data.list_manager import load_us_tech_list, _read_md_table

    # ── 美股科技 (us_tech_list.md) ─────────────────────────────────
    logger.info("刷新美股科技（us_tech_list.md）周线指标...")
    us_entries = load_us_tech_list()
    us_symbols  = [
        e["ticker"] for e in us_entries
        if e.get("ticker") and (CACHE_DIR / "us" / f"{e['ticker']}.parquet").exists()
    ]
    logger.info(f"  美股科技：{len(us_symbols)} 只有缓存")
    update_indicators_for_symbols(us_symbols, "us")

    # ── 港股高科技 (hk_techman.md) ────────────────────────────────
    logger.info("刷新港股高科技（hk_techman.md）周线指标...")
    techman_path = DATA_DIR / "lists" / "hk_techman.md"
    rows = _read_md_table(techman_path) if techman_path.exists() else []
    hk_symbols = [
        r[1].strip().zfill(5) for r in rows
        if len(r) >= 2 and (CACHE_DIR / "hk" / f"{r[1].strip().zfill(5)}.parquet").exists()
    ]
    logger.info(f"  港股高科技：{len(hk_symbols)} 只有缓存")
    update_indicators_for_symbols(hk_symbols, "hk")

    logger.success("周线指标刷新完成")


# ════════════════════════════════════════════════════════════════════
#  合并 watchlist（US tech + HK techman）
# ════════════════════════════════════════════════════════════════════

def _build_combined_watchlist() -> dict:
    """美股科技列表 + 港股高科技列表合并。"""
    from stock_ana.scan.w_vegas_short_scan import _build_us_universe_watchlist, _build_hk_techman_watchlist
    wl: dict = {}
    wl.update(_build_us_universe_watchlist())
    wl.update(_build_hk_techman_watchlist())
    logger.info(f"合并 watchlist：{len(wl)} 只（US tech + HK techman）")
    return wl


# ════════════════════════════════════════════════════════════════════
#  飞书工具（与 notify_daily_scan_result 相同逻辑，独立副本）
# ════════════════════════════════════════════════════════════════════

def get_tenant_token() -> str | None:
    url  = f"{FEISHU_API}/auth/v3/tenant_access_token/internal"
    data = json.dumps({"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET}).encode("utf-8")
    req  = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json; charset=utf-8"})
    with _feishu_opener.open(req, timeout=15) as resp:
        return json.loads(resp.read()).get("tenant_access_token")


def send_post_message(token: str, title: str, blocks: list[list[dict]]) -> bool:
    url      = f"{FEISHU_API}/im/v1/messages?receive_id_type=open_id"
    post_body = {"zh_cn": {"title": title, "content": blocks}}
    payload  = {
        "receive_id": FEISHU_USER_OPEN_ID,
        "msg_type":   "post",
        "content":    json.dumps(post_body, ensure_ascii=False),
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8",
                 "Authorization": f"Bearer {token}"},
    )
    with _feishu_opener.open(req, timeout=20) as resp:
        return json.loads(resp.read()).get("code") == 0


def send_file_message(token: str, file_key: str) -> bool:
    url     = f"{FEISHU_API}/im/v1/messages?receive_id_type=open_id"
    payload = {
        "receive_id": FEISHU_USER_OPEN_ID,
        "msg_type":   "file",
        "content":    json.dumps({"file_key": file_key}, ensure_ascii=False),
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8",
                 "Authorization": f"Bearer {token}"},
    )
    with _feishu_opener.open(req, timeout=20) as resp:
        return json.loads(resp.read()).get("code") == 0


def upload_report_file(token: str, report_path: Path) -> str | None:
    if not report_path.exists():
        return None
    cmd = [
        "curl", "-sS", "-X", "POST",
        f"{FEISHU_API}/im/v1/files",
        "-H", f"Authorization: Bearer {token}",
        "-F", "file_type=stream",
        "-F", f"file_name={report_path.name}",
        "-F", f"file=@{report_path}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        if proc.returncode != 0:
            logger.error(f"curl 上传失败: {proc.stderr[:200]}")
            return None
        result = json.loads(proc.stdout)
        return result.get("data", {}).get("file_key")
    except Exception as e:
        logger.error(f"upload_report_file error: {e}")
        return None


# ════════════════════════════════════════════════════════════════════
#  LLM 批量分析（每批 ≤ BATCH_SIZE 只，批间等 BATCH_DELAY 秒）
# ════════════════════════════════════════════════════════════════════

async def run_batched_gemini(signals: list[dict], llm_backend: str = "codex") -> str:
    """
    将信号分成每批 ≤ BATCH_SIZE 只，依次发给 LLM。
    每批结果返回后等待 BATCH_DELAY 秒再发下一批。
    返回所有批次合并的完整 Markdown 文本。
    """
    from stock_ana.utils.scan_analyst import DEFAULT_MODEL, _call_codex, _call_gemini, _init_client, build_prompt

    batches = [signals[i : i + BATCH_SIZE] for i in range(0, len(signals), BATCH_SIZE)]
    logger.info(
        f"{llm_backend} 批量分析：{len(signals)} 只，分 {len(batches)} 批"
        f"（每批 ≤{BATCH_SIZE} 只，批间等待 {BATCH_DELAY}s）"
    )

    client = None
    if llm_backend == "gemini":
        client = await _init_client(DEFAULT_MODEL)
    elif llm_backend != "codex":
        raise ValueError(f"不支持的 LLM backend: {llm_backend}，可选 gemini/codex")

    all_texts: list[str] = []

    try:
        for idx, batch in enumerate(batches, 1):
            syms = [s.get("symbol", "?") for s in batch]
            logger.info(f"  第 {idx}/{len(batches)} 批：{syms}")
            try:
                prompt = build_prompt(batch)
                if llm_backend == "codex":
                    text = await _call_codex(prompt)
                else:
                    text = await _call_gemini(prompt, client, model=DEFAULT_MODEL, max_retries=1)
                all_texts.append(text.strip())
                logger.success(f"  第 {idx} 批完成，{len(text)} 字符")
            except Exception as e:
                logger.error(f"  第 {idx} 批 {llm_backend} 调用失败 [{type(e).__name__}]: {e}")
                # 继续尝试下一批，而不是中断整个流程
                all_texts.append(
                    f"<!-- 第 {idx} 批（{', '.join(syms)}）{llm_backend} 分析失败: {e} -->"
                )

            if idx < len(batches):
                logger.info(f"  等待 {BATCH_DELAY}s 再发下一批...")
                await asyncio.sleep(BATCH_DELAY)
    finally:
        if client is not None:
            await client.close()

    return "\n\n---\n\n".join(all_texts)


# ════════════════════════════════════════════════════════════════════
#  报告保存
# ════════════════════════════════════════════════════════════════════

def save_weekly_report(signals: list[dict], gemini_text: str, out_dir: Path) -> Path:
    """将汇总表 + LLM 合并文本保存为单一 .md 文件。"""
    today  = date.today().isoformat()
    syms   = "_".join(s.get("symbol", "") for s in signals[:5])
    suffix = f"_plus{len(signals) - 5}more" if len(signals) > 5 else ""
    out_dir.mkdir(parents=True, exist_ok=True)
    path   = out_dir / f"W_{today}_{syms}{suffix}.md"

    header = [
        "# 周线 Vegas Short 信号分析报告",
        "",
        f"**日期**: {today}  |  **信号数**: {len(signals)}",
        "",
        "| 代码 | 名称 | 信号 | 评分 | 入场日 | 策略 |",
        "|------|------|------|------|--------|------|",
    ]
    for s in signals:
        header.append(
            f"| {s.get('symbol','')} | {s.get('name','')} "
            f"| {s.get('signal','')} | {s.get('score',0):+d} "
            f"| {s.get('entry_date','').split('(')[0]} "
            f"| {s.get('touch_strategy','')} |"
        )
    header += ["", "---", ""]

    path.write_text("\n".join(header) + "\n" + gemini_text, encoding="utf-8")
    logger.success(f"周线报告已保存 → {path}")
    return path


# ════════════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════════════

async def main_async(
    lookback: int = 1,
    scan_only: bool = False,
    list_mode: str = "combined",
    skip_indicators: bool = False,
    force_send: bool = False,
    llm_backend: str = "codex",
) -> None:
    t0    = datetime.now()
    today = date.today().isoformat()

    logger.info("=" * 60)
    logger.info(f"  周线 Vegas Short 流水线 — {t0:%Y-%m-%d %H:%M:%S}")
    logger.info("=" * 60)

    # ── Step 0: 刷新周线指标 ──────────────────────────────────────────────────
    if not skip_indicators and list_mode in ("combined", "us", "hk"):
        logger.info("【0/3】刷新周线指标...")
        refresh_weekly_indicators_for_lists()
    else:
        logger.info("【0/3】跳过指标刷新")

    # ── Step 1: 扫描（含图表渲染）────────────────────────────────────────────
    logger.info(f"【1/3】运行周线扫描（list={list_mode}, lookback={lookback}w）")
    from stock_ana.scan.w_vegas_short_scan import (
        run_scan,
        _build_hk_techman_watchlist,
        _build_us_full_watchlist,
        _build_us_universe_watchlist,
    )
    from stock_ana.data.market_data import build_watchlist

    if list_mode == "combined":
        watchlist = _build_combined_watchlist()
    elif list_mode == "hk":
        watchlist = _build_hk_techman_watchlist()
    elif list_mode == "us":
        watchlist = _build_us_universe_watchlist()
    elif list_mode == "us-full":
        watchlist = _build_us_full_watchlist()
    else:
        watchlist = build_watchlist()

    signals_raw = run_scan(
        watchlist=watchlist,
        lookback=lookback,
        min_signal="BUY",
        touch_only=True,   # 只要 touch 策略信号
    )
    logger.success(f"扫描完成：{len(watchlist)} 只 → {len(signals_raw)} 个原始信号")

    # ── 按 symbol 去重（保留评分最高的那条）────────────────────────────────
    seen: dict[str, dict] = {}
    for s in signals_raw:
        sym = s.get("symbol", "")
        if sym not in seen or s["score"] > seen[sym]["score"]:
            seen[sym] = s
    signals = sorted(seen.values(), key=lambda s: s["score"], reverse=True)
    logger.info(f"去重后：{len(signals)} 只")

    # ── 排除已在富途自选股中的 symbol（不需要 LLM）────────────────────────
    # 优先读 futu_watched_symbols.json（sync_futu_watchlist.py 每日保存）
    # 若不存在则 fallback 到 watchlist.md
    _futu_cache = PROJECT_ROOT / "data" / "lists" / "futu_watched_symbols.json"
    if _futu_cache.exists():
        personal_syms = set(json.loads(_futu_cache.read_text(encoding="utf-8")).get("symbols", []))
        logger.info(f"富途自选股缓存：{len(personal_syms)} 只（来自 futu_watched_symbols.json）")
    else:
        from stock_ana.data.market_data import build_watchlist as _build_personal_wl
        personal_syms = set(_build_personal_wl().keys())
        logger.warning("未找到 futu_watched_symbols.json，fallback 到 watchlist.md")
    watchlist_signals = [s for s in signals if s.get("symbol", "") in personal_syms]
    new_signals       = [s for s in signals if s.get("symbol", "") not in personal_syms]
    if watchlist_signals:
        logger.info(
            f"已在关注列表，跳过LLM（{len(watchlist_signals)} 只）: "
            + ", ".join(s.get("symbol","") for s in watchlist_signals)
        )
    logger.info(f"需要LLM分析：{len(new_signals)} 只")

    if not signals:
        logger.info("本周无信号，流水线结束")
        token = get_tenant_token()
        if token:
            send_post_message(
                token,
                f"📈 周线 Vegas Short {today}",
                [[{"tag": "text", "text": f"本周（lookback={lookback}w）无 touch 信号触发"}]],
            )
        return

    # 收集图表路径
    chart_paths:  list[Path] = []
    chart_labels: list[str]  = []
    for s in signals:
        cp = Path(s.get("chart_path", ""))
        if cp.exists():
            chart_paths.append(cp)
            chart_labels.append(f"{s.get('symbol','')} ({s.get('name','')})")

    # ── Step 2: LLM 批量分析 ───────────────────────────────────────────────
    report_path: Path | None = None
    gemini_text = ""

    if scan_only:
        logger.info("【2/3】--scan-only 模式，跳过 LLM 分析")
    elif not new_signals:
        logger.info("【2/3】无需 LLM 分析的新信号，跳过")
    else:
        logger.info(f"【2/3】{llm_backend} 批量分析（{len(new_signals)} 只，每批 ≤{BATCH_SIZE} 只）")
        out_dir = ANALYSIS_OUT_DIR / today
        try:
            gemini_text = await run_batched_gemini(new_signals, llm_backend=llm_backend)
            report_path = save_weekly_report(signals, gemini_text, out_dir)
        except Exception as e:
            logger.error(f"{llm_backend} 分析整体失败: {e}")

    # 统一产出可用于 PDF 的 markdown 报告（即使 LLM 未执行/失败）
    if report_path is None:
        out_dir = ANALYSIS_OUT_DIR / today
        if scan_only:
            gemini_text = "## 说明\n\n本次以 --scan-only 模式运行，未执行 LLM 解读。"
        elif not new_signals:
            gemini_text = "## 说明\n\n本次信号均已在关注列表中，按规则跳过 LLM 解读。"
        elif not gemini_text.strip():
            gemini_text = "## 说明\n\nLLM 解读执行失败，请查看日志获取错误详情。"
        report_path = save_weekly_report(signals, gemini_text, out_dir)

    # ── Step 3: 生成 PDF + 发飞书 ────────────────────────────────────────────
    logger.info("【3/3】生成 PDF + 发送飞书")

    fingerprint = _build_send_fingerprint(signals, list_mode, lookback)
    guard = _load_send_guard()
    last_date = str(guard.get("date", ""))
    last_fp = str(guard.get("fingerprint", ""))
    if (not force_send) and last_date == today and last_fp == fingerprint:
        logger.warning("检测到当日同一批信号已发送，跳过重复发送（可用 --force-send 强制重发）")
        return

    token = get_tenant_token()
    if not token:
        logger.error("飞书 token 获取失败，终止")
        return

    title = f"📈 周线 Vegas Short 信号 {today}（{len(signals)} 只）"
    md_content = ""
    if report_path and report_path.exists():
        md_content = report_path.read_text(encoding="utf-8")
    if not md_content.strip():
        logger.error("未生成有效周线报告，回退发送文字错误通知")
        send_post_message(token, title, [[{"tag": "text", "text": "周线报告为空，未能生成 PDF，请检查日志。"}]])
        return

    # 生成 PDF
    from stock_ana.utils.pdf_builder import build_scan_pdf
    try:
        pdf_bytes = build_scan_pdf(
            md_content=md_content,
            chart_paths=chart_paths,
            signal_labels=chart_labels,
            title=title,
            signals=signals,
        )
    except Exception as e:
        logger.error(f"PDF 生成失败: {e}")
        send_post_message(token, title, [[{"tag": "text", "text": f"PDF 生成失败: {e}"}]])
        return

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False,
                                     prefix=f"w_scan_{today}_") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        file_key = upload_report_file(token, tmp_path)
        if file_key and send_file_message(token, file_key):
            logger.success(f"✅ 飞书 PDF 已发送：{title}")
            _save_send_guard(
                {
                    "date": today,
                    "fingerprint": fingerprint,
                    "list_mode": list_mode,
                    "lookback": lookback,
                    "signals_count": len(signals),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
        else:
            logger.error("飞书 PDF 上传或发送失败")
            # fallback：发文字通知
            send_post_message(token, title,
                              [[{"tag": "text", "text": "PDF 上传失败，请查看日志"}]])
    finally:
        tmp_path.unlink(missing_ok=True)

    elapsed = int((datetime.now() - t0).total_seconds())
    logger.info(f"周线流水线完成，总耗时 {elapsed}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="周线 Vegas Short 扫描 + Gemini + 飞书 PDF")
    parser.add_argument("--lookback", type=int, default=1,
                        help="最近几周（默认 1 = 仅当周）")
    parser.add_argument("--scan-only", action="store_true",
                        help="仅扫描，不调 LLM，不发飞书通知")
    parser.add_argument("--list", dest="list_mode", default="combined",
                        choices=["combined", "shawn", "us", "hk", "us-full"],
                        help="扫描标的列表（默认 combined = US tech + HK techman）")
    parser.add_argument("--skip-indicators", action="store_true",
                        help="跳过周线指标刷新（当日已刷新过时使用）")
    parser.add_argument("--force-send", action="store_true",
                        help="忽略去重保护，强制再次发送飞书 PDF")
    parser.add_argument(
        "--llm-backend",
        choices=["gemini", "codex"],
        default=(
            os.environ.get("STOCK_ANA_WEEKLY_LLM_BACKEND")
            or os.environ.get("STOCK_ANA_SCAN_LLM_BACKEND")
            or "codex"
        ).strip().lower(),
        help="LLM 后端：gemini（默认）或 codex（通过本地 CLIProxyAPI 调 gpt-5.5）",
    )
    args = parser.parse_args()
    asyncio.run(main_async(
        lookback=args.lookback,
        scan_only=args.scan_only,
        list_mode=args.list_mode,
        skip_indicators=args.skip_indicators,
        force_send=args.force_send,
        llm_backend=args.llm_backend,
    ))


if __name__ == "__main__":
    main()
