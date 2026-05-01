#!/usr/bin/env python3
"""
watchlist_vegas_scan.py — 每日扫描 watchlist.md 所有标的的 Vegas 触碰情况

检测两类信号：
  - Mid Vegas touch：最新收盘价触及或下穿 EMA34/55/60 通道（回踩信号）
  - Long Vegas touch：最新收盘价触及或下穿 EMA144/169/200 通道（大级别支撑）

有信号则通过飞书推送通知。

用法：
    python watchlist_vegas_scan.py              # 扫描全部，有信号才推送
    python watchlist_vegas_scan.py --lookback 3 # 最近3个交易日内有触碰即算
    python watchlist_vegas_scan.py --dry-run    # 只打印，不推送飞书
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from loguru import logger

# ─── 飞书配置（与 notify_daily_scan_result.py 保持一致）────────────────────
FEISHU_APP_ID = os.environ.get("FEISHU_APP_ID") or "cli_a924285ae7f85cc7"
FEISHU_APP_SECRET = os.environ.get("FEISHU_APP_SECRET") or "53hrIbxJYHGGAI8qbndwofOzltJAkah0"
FEISHU_USER_OPEN_ID = os.environ.get("FEISHU_USER_OPEN_ID") or "ou_5489407346c5c13bc4687a83859d619b"
FEISHU_API = "https://open.feishu.cn/open-apis"

_feishu_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# ─── EMA 参数 ──────────────────────────────────────────────────────────────
MID_SPANS = [34, 55, 60]    # Mid Vegas
LONG_SPANS = [144, 169, 200]  # Long Vegas


# ═══════════════════════════════════════════════════════
#  核心检测逻辑
# ═══════════════════════════════════════════════════════

def _compute_emas(close_s: pd.Series) -> dict[int, np.ndarray]:
    spans = MID_SPANS + LONG_SPANS
    return {s: close_s.ewm(span=s, adjust=False).mean().values for s in spans}


def _detect_vegas_touch(
    sym: str,
    market: str,
    name: str,
    df: pd.DataFrame,
) -> list[dict]:
    """
    检测最新一个交易日是否有 Mid 或 Long Vegas 首次触碰。

    触碰条件（仅检查最新一根 K 线）：
      1. 当日低价 <= Vegas 通道最高 EMA（严格穿入通道，无缓冲）
      2. 前一日收盘价 > 前一日通道最高 EMA（前日在通道之上，当日为首次触碰）

    Returns:
        list of dicts，每条代表一个触碰事件
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if len(df) < max(LONG_SPANS) + 10:
        return []

    close_s = df["close"].astype(float)
    low_s = df["low"].astype(float)
    emas = _compute_emas(close_s)

    # 只检查最新一根 K 线
    bar_idx = len(df) - 1
    bar_date = df.index[bar_idx]
    close_val = float(close_s.iloc[bar_idx])
    low_val = float(low_s.iloc[bar_idx])
    prev_close = float(close_s.iloc[bar_idx - 1])

    results = []

    for vegas_type, spans in [("mid", MID_SPANS), ("long", LONG_SPANS)]:
        ema_vals = {s: float(emas[s][bar_idx]) for s in spans}
        prev_ema_vals = {s: float(emas[s][bar_idx - 1]) for s in spans}

        channel_top = max(ema_vals.values())
        channel_bot = min(ema_vals.values())
        prev_channel_top = max(prev_ema_vals.values())

        # 条件1：当日低价严格穿入/触及通道（无缓冲）
        touched = low_val <= channel_top
        # 条件2：前一日收盘在通道之上（确保是首次触碰，而非已在通道内）
        prev_above = prev_close > prev_channel_top

        if touched and prev_above:
            results.append({
                "sym": sym,
                "market": market,
                "name": name,
                "date": bar_date.strftime("%Y-%m-%d"),
                "vegas_type": vegas_type,
                "close": round(close_val, 4),
                "low": round(low_val, 4),
                "prev_close": round(prev_close, 4),
                "prev_channel_top": round(prev_channel_top, 4),
                "ema_values": {str(s): round(v, 4) for s, v in ema_vals.items()},
                "channel_top": round(channel_top, 4),
                "channel_bot": round(channel_bot, 4),
            })

    return results


# ═══════════════════════════════════════════════════════
#  Watchlist 加载
# ═══════════════════════════════════════════════════════

def _load_watchlist() -> list[dict]:
    """读取 data/lists/watchlist.md，返回 [{sym, market, name, cache_path}]。"""
    from stock_ana.config import CACHE_DIR, DATA_DIR
    from stock_ana.data.list_manager import parse_watchlist

    wl = parse_watchlist()  # {"us": [...], "hk": [...], "cn": [...]}

    market_cache = {
        "us": CACHE_DIR / "us",
        "hk": CACHE_DIR / "hk",
        "cn": CACHE_DIR / "cn",
    }

    items = []
    for mkt_key, entries in wl.items():
        cache_dir = market_cache.get(mkt_key)
        if cache_dir is None:
            continue
        for e in entries:
            sym = e["symbol"]
            name = e.get("name") or e.get("name_cn") or sym
            path = cache_dir / f"{sym}.parquet"
            items.append({
                "sym": sym,
                "market": mkt_key.upper(),
                "name": name,
                "path": path,
            })
    return items


# ═══════════════════════════════════════════════════════
#  飞书推送
# ═══════════════════════════════════════════════════════

def _get_tenant_token() -> str | None:
    url = f"{FEISHU_API}/auth/v3/tenant_access_token/internal"
    data = json.dumps({"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET}).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with _feishu_opener.open(req, timeout=15) as resp:
        return json.loads(resp.read()).get("tenant_access_token")


def _send_feishu_post(token: str, title: str, blocks: list[list[dict]]) -> bool:
    url = f"{FEISHU_API}/im/v1/messages?receive_id_type=open_id"
    post_body = {"zh_cn": {"title": title, "content": blocks}}
    payload = {
        "receive_id": FEISHU_USER_OPEN_ID,
        "msg_type": "post",
        "content": json.dumps(post_body, ensure_ascii=False),
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {token}",
        },
    )
    with _feishu_opener.open(req, timeout=20) as resp:
        result = json.loads(resp.read())
    ok = result.get("code") == 0
    if not ok:
        logger.warning(f"飞书推送失败: {result}")
    return ok


def _format_and_send(signals: list[dict], dry_run: bool = False) -> None:
    """将信号列表格式化为飞书消息并推送。"""
    if not signals:
        logger.info("无触碰信号，不推送")
        return

    today = date.today().isoformat()

    # 按 vegas_type 分组
    mid_signals = [s for s in signals if s["vegas_type"] == "mid"]
    long_signals = [s for s in signals if s["vegas_type"] == "long"]

    blocks: list[list[dict]] = []

    def _txt(text: str) -> list[dict]:
        return [{"tag": "text", "text": text}]

    blocks.append(_txt(f"扫描日期：{today}  |  触碰标的共 {len(signals)} 条"))
    blocks.append(_txt(""))

    if long_signals:
        blocks.append(_txt(f"【Long Vegas 触碰 (EMA144/169/200)】{len(long_signals)} 只"))
        for s in long_signals:
            ema_str = " / ".join(
                f"EMA{e}={s['ema_values'][str(e)]}" for e in LONG_SPANS
            )
            blocks.append(_txt(
                f"  {s['market']}:{s['sym']} {s['name']}  "
                f"收盘={s['close']}  低={s['low']}\n"
                f"  {ema_str}"
            ))

    if mid_signals:
        if long_signals:
            blocks.append(_txt(""))
        blocks.append(_txt(f"【Mid Vegas 触碰 (EMA34/55/60)】{len(mid_signals)} 只"))
        for s in mid_signals:
            ema_str = " / ".join(
                f"EMA{e}={s['ema_values'][str(e)]}" for e in MID_SPANS
            )
            blocks.append(_txt(
                f"  {s['market']}:{s['sym']} {s['name']}  "
                f"收盘={s['close']}  低={s['low']}\n"
                f"  {ema_str}"
            ))

    # 打印
    for b in blocks:
        logger.info("".join(item.get("text", "") for item in b))

    if dry_run:
        logger.info("[dry-run] 跳过飞书推送")
        return

    try:
        token = _get_tenant_token()
        if not token:
            logger.error("获取飞书 token 失败")
            return
        ok = _send_feishu_post(token, f"Vegas 触碰提醒 {today}", blocks)
        if ok:
            logger.success("飞书消息推送成功")
    except Exception as e:
        logger.error(f"飞书推送异常: {e}")


# ═══════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="watchlist Vegas 触碰每日扫描")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印，不推送飞书")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Watchlist Vegas 扫描 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("策略：当日低价严格穿入通道 + 前日收盘在通道上方")
    logger.info("=" * 60)

    items = _load_watchlist()
    logger.info(f"共 {len(items)} 只标的（US + HK + CN）")

    all_signals: list[dict] = []
    missing = 0

    for item in items:
        path: Path = item["path"]
        if not path.exists():
            missing += 1
            logger.debug(f"{item['market']}:{item['sym']} 缺少缓存，跳过")
            continue
        try:
            df = pd.read_parquet(path)
            signals = _detect_vegas_touch(
                item["sym"], item["market"], item["name"], df,
            )
            all_signals.extend(signals)
        except Exception as e:
            logger.warning(f"{item['market']}:{item['sym']} 扫描失败: {e}")

    if missing:
        logger.warning(f"{missing} 只标的缺少缓存数据（未拉取或尚未更新）")

    mid_n = sum(1 for s in all_signals if s["vegas_type"] == "mid")
    long_n = sum(1 for s in all_signals if s["vegas_type"] == "long")
    logger.info(f"扫描完成：Mid Vegas 触碰 {mid_n} 只，Long Vegas 触碰 {long_n} 只")

    # 保存 JSON
    out_dir = PROJECT_ROOT / "data" / "output" / "watchlist_vegas_scan" / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "signals.json"
    out_path.write_text(json.dumps(all_signals, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"结果已保存 → {out_path}")

    _format_and_send(all_signals, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
