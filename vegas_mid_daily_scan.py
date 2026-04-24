#!/usr/bin/env python3
"""
每日扫描 + 基本面分析流水线

执行顺序：
  1. 运行 Vegas Mid touch 策略扫描（美股科技板块，lookback=1）
  2. 针对每只信号股票生成 K 线图（集成在 run_scan 内）
  3. 将所有 HOLD 以上信号发送 Gemini 批量基本面分析
  4. 提取 Gemini 汇总表中每只股票的综合建议
  5. 输出 summary.json（供 clawbot 读取并推送消息）

输出目录（每次运行覆盖当天）：
  data/output/daily_scan/{YYYY-MM-DD}/
    summary.json          ← 精简摘要 + 结论（clawbot 消费入口）
  data/output/vegas_scan/{YYYY-MM-DD}/
    signals.json          ← 原始扫描信号
    signals_full.json     ← 含 base64 图表
    *.png                 ← 各标的 K 线图
  data/output/scan_analysis/{YYYY-MM-DD}/
    *.md                  ← Gemini 完整分析报告

用法：
    python daily_scan.py               # 完整流程（扫描 + Gemini）
    python daily_scan.py --scan-only   # 仅扫描，不调 Gemini
    python daily_scan.py --lookback 3  # 扩大回看窗口（排查漏出的信号）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loguru import logger

# ─── 日志 ─────────────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(
    LOG_DIR / "daily_scan_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    encoding="utf-8",
)

DAILY_SCAN_DIR = PROJECT_ROOT / "data" / "output" / "daily_scan"


# ═══════════════════════════════════════════════════════
#  Step 1-2：运行扫描 + 画图
# ═══════════════════════════════════════════════════════

def run_daily_scan(lookback: int = 1, list_mode: str = "tech") -> tuple[list[dict], int]:
    """运行 Vegas Mid touch 扫描，返回 (signals, total_scanned)。

    Args:
        list_mode: "tech"（科技/通信，339只）| "full"（全量，~1550只）| "hk"（港股宇宙池，575只）
    """
    from stock_ana.scan.vegas_mid_scan import (
        run_scan,
        _build_us_universe_watchlist,
        _build_us_full_watchlist,
        _build_hk_universe_watchlist,
    )

    if list_mode == "full":
        logger.info("=" * 60)
        logger.info("【1/3】构建美股全量 watchlist ...")
        logger.info("=" * 60)
        watchlist = _build_us_full_watchlist()
    elif list_mode == "hk":
        logger.info("=" * 60)
        logger.info("【1/3】构建港股宇宙池 watchlist ...")
        logger.info("=" * 60)
        watchlist = _build_hk_universe_watchlist()
    else:
        logger.info("=" * 60)
        logger.info("【1/3】构建美股科技板块 watchlist ...")
        logger.info("=" * 60)
        watchlist = _build_us_universe_watchlist()
    total = len(watchlist)
    logger.info(f"共 {total} 只标的载入（list_mode={list_mode}）")

    logger.info("=" * 60)
    logger.info(f"【2/3】运行 Vegas Mid touch 扫描（lookback={lookback}）...")
    logger.info("=" * 60)
    signals = run_scan(
        watchlist=watchlist,
        lookback=lookback,
        min_signal="HOLD",
        touch_only=True,
    )
    logger.success(f"扫描完成，触发信号 {len(signals)} 个")
    return signals, total


# ═══════════════════════════════════════════════════════
#  Step 3：Gemini 基本面分析
# ═══════════════════════════════════════════════════════

async def run_gemini_analysis(signals: list[dict]) -> Path | None:
    """对所有通过结构检验的信号发起 Gemini 批量分析，返回报告路径。

    包含 STRONG_BUY / BUY / HOLD，以及 structure_passed=True 的 AVOID
    （即评分不足但结构本身合格的标的）。
    纯结构不过关的 AVOID 已在 run_scan 中提前丢弃，不会出现在此处。
    """
    from stock_ana.utils.scan_analyst import analyze_signals

    # 所有出现在 signals 列表里的标的都已通过结构检验，直接全量发送
    targets = list(signals)
    if not targets:
        logger.info("无信号，跳过 Gemini 分析")
        return None

    logger.info("=" * 60)
    logger.info(f"【3/3】Gemini 基本面分析，共 {len(targets)} 只 ...")
    logger.info("=" * 60)
    try:
        path = await analyze_signals(targets, min_signal=None)
        logger.success(f"Gemini 分析完成 → {path}")
        return path
    except Exception as e:
        logger.error(f"Gemini 分析失败: {e}")
        return None


# ═══════════════════════════════════════════════════════
#  Step 4：从报告提取综合结论
# ═══════════════════════════════════════════════════════

def _extract_gemini_conclusions(report_text: str) -> dict[str, dict]:
    """解析 Gemini 报告末尾汇总表，返回 {TICKER: {conclusion, fundamental_score, ...}}。"""
    conclusions: dict[str, dict] = {}
    lines = report_text.splitlines()
    col_idx: dict[str, int] = {}
    header_found = False
    in_table = False

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if in_table:
                break
            continue

        cells = [c.strip() for c in stripped.strip("|").split("|")]

        # 检测表头行（含"代号"或"综合建议"）
        if not header_found:
            has_ticker  = any("代号" in c or "ticker" in c.lower() for c in cells)
            has_suggest = any("综合建议" in c or "建议" in c for c in cells)
            if has_ticker and has_suggest:
                for i, h in enumerate(cells):
                    h_clean = re.sub(r"\*+", "", h).strip()
                    if "代号" in h_clean or "ticker" in h_clean.lower():
                        col_idx["symbol"] = i
                    elif "综合建议" in h_clean:
                        col_idx["conclusion"] = i
                    elif "基本面" in h_clean:
                        col_idx["fundamental"] = i
                    elif "估值" in h_clean:
                        col_idx["valuation"] = i
                    elif "技术" in h_clean and "信号" not in h_clean:
                        col_idx["technical"] = i
                header_found = True
            continue

        # 跳过分隔行（|---|---|）
        if all(re.fullmatch(r"[-:| ]+", c) for c in cells if c.strip()):
            continue

        if header_found and len(cells) > max(col_idx.values(), default=0):
            in_table = True
            sym_raw = cells[col_idx.get("symbol", 0)]
            sym = re.sub(r"\*+", "", sym_raw).strip().upper()
            if not sym:
                continue
            get = lambda key: re.sub(r"\*+", "", cells[col_idx[key]]).strip() if key in col_idx and col_idx[key] < len(cells) else ""
            conclusions[sym] = {
                "conclusion":        get("conclusion"),
                "fundamental_score": get("fundamental"),
                "valuation_score":   get("valuation"),
                "technical_score":   get("technical"),
            }

    return conclusions


def _extract_summary_table(report_text: str) -> str:
    """提取报告中汇总表格原文（Markdown 格式）。"""
    lines = report_text.splitlines()
    table_lines: list[str] = []
    collecting = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and ("综合建议" in stripped or "代号" in stripped):
            collecting = True
        if collecting:
            if stripped.startswith("|"):
                table_lines.append(line)
            elif table_lines:
                break

    return "\n".join(table_lines)


# ═══════════════════════════════════════════════════════
#  Step 5：保存 summary.json
# ═══════════════════════════════════════════════════════

def save_summary(
    signals: list[dict],
    total_scanned: int,
    gemini_path: Path | None,
    lookback: int,
    market_label: str = "",
    filename: str = "summary.json",
) -> Path:
    """生成 summary JSON，为 clawbot 提供消费入口。"""
    today = date.today().isoformat()
    out_dir = DAILY_SCAN_DIR / today
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取 Gemini 结论
    gemini_conclusions: dict[str, dict] = {}
    gemini_summary_table = ""
    if gemini_path and gemini_path.exists():
        report_text = gemini_path.read_text(encoding="utf-8")
        gemini_conclusions = _extract_gemini_conclusions(report_text)
        gemini_summary_table = _extract_summary_table(report_text)

    # 组装每只标的摘要
    summary_signals = []
    for s in signals:
        sym = s.get("symbol", "")
        gem = gemini_conclusions.get(sym, {})
        summary_signals.append({
            "symbol":                  sym,
            "name":                    s.get("name", ""),
            "signal":                  s.get("signal", ""),
            "score":                   s.get("score", 0),
            "entry_date":              s.get("entry_date", "").split("(")[0],
            "support_band":            s.get("support_band", ""),
            "chart_path":              s.get("chart_path", ""),
            "gemini_conclusion":       gem.get("conclusion", ""),
            "gemini_fundamental_score": gem.get("fundamental_score", ""),
            "gemini_valuation_score":  gem.get("valuation_score", ""),
            "gemini_technical_score":  gem.get("technical_score", ""),
        })

    summary = {
        "scan_date":            today,
        "generated_at":         datetime.now().isoformat(timespec="seconds"),
        "market_label":         market_label,
        "lookback_days":        lookback,
        "total_scanned":        total_scanned,
        "signals_found":        len(signals),
        "has_gemini_analysis":  gemini_path is not None and gemini_path.exists(),
        "gemini_report_path":   str(gemini_path) if gemini_path else None,
        "gemini_summary_table": gemini_summary_table,
        "signals":              summary_signals,
    }

    path = out_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.success(f"Summary 已保存 → {path}")
    return path


# ═══════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════

async def _run_single_market(
    lookback: int,
    scan_only: bool,
    list_mode: str,
    market_label: str,
    filename: str,
) -> Path:
    """扫描单个市场并保存 summary，返回 summary 路径。"""
    signals, total_scanned = run_daily_scan(lookback=lookback, list_mode=list_mode)

    gemini_path: Path | None = None
    if not scan_only and signals:
        gemini_path = await run_gemini_analysis(signals)

    return save_summary(
        signals, total_scanned, gemini_path, lookback,
        market_label=market_label, filename=filename,
    )


async def _main_async(lookback: int, scan_only: bool, list_mode: str = "tech") -> None:
    t0 = datetime.now()
    logger.info("=" * 60)
    logger.info(f"  每日扫描流水线 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    if list_mode == "combined":
        # 先美股，再港股
        logger.info("【组合模式】先扫描美股科技板块，再扫描港股宇宙池")
        await _run_single_market(lookback, scan_only, "tech",  "每日美股扫描", "summary_us.json")
        await _run_single_market(lookback, scan_only, "hk",   "每日港股扫描", "summary_hk.json")
        elapsed = (datetime.now() - t0).seconds
        logger.info("=" * 60)
        logger.info(f"  组合流水线完成 — 总耗时 {elapsed}s")
        logger.info("=" * 60)
        return

    # 单市场模式
    _label_map = {"tech": "每日美股扫描", "full": "每日美股扫描（全量）", "hk": "每日港股扫描"}
    _file_map  = {"tech": "summary_us.json", "full": "summary_us.json", "hk": "summary_hk.json"}
    market_label = _label_map.get(list_mode, "每日扫描")
    filename     = _file_map.get(list_mode, "summary.json")

    # Step 1-2：扫描 + 画图
    signals, total_scanned = run_daily_scan(lookback=lookback, list_mode=list_mode)

    # Step 3：Gemini 分析
    gemini_path: Path | None = None
    if not scan_only and signals:
        gemini_path = await run_gemini_analysis(signals)

    # Step 4-5：保存 summary
    summary_path = save_summary(
        signals, total_scanned, gemini_path, lookback,
        market_label=market_label, filename=filename,
    )

    elapsed = (datetime.now() - t0).seconds
    logger.info("=" * 60)
    logger.info(f"  流水线完成 — 总耗时 {elapsed}s")
    logger.info(f"  信号数：{len(signals)}")
    logger.info(f"  Gemini：{'完成' if gemini_path else '跳过'}")
    logger.info(f"  摘要：{summary_path}")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="每日 Vegas Mid 扫描 + Gemini 分析")
    parser.add_argument("--lookback",  type=int, default=1, help="回看天数（默认 1）")
    parser.add_argument("--scan-only", action="store_true", help="仅扫描，不调 Gemini")
    parser.add_argument(
        "--list",
        dest="list_mode",
        choices=["tech", "full", "hk", "combined"],
        default="tech",
        help="扫描标的池：tech（科技/通信339只，默认）| full（全量~1550只）| hk（港股宇宙池575只）| combined（先美股tech再港股）",
    )
    args = parser.parse_args()

    asyncio.run(_main_async(lookback=args.lookback, scan_only=args.scan_only, list_mode=args.list_mode))


if __name__ == "__main__":
    main()
