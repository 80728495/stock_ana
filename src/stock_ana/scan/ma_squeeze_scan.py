#!/usr/bin/env python3
"""
多线压缩策略 — 每日扫描 (stock_ana.scan.ma_squeeze_scan)

扫描美股 (~1400) + 港股 (~90) 最新一个交易日是否触发多线压缩信号（S1 发现点 / S2 确认点），
触发则画出 K 线图（含 MA30/60/200 三线叠加），并输出结构化 JSON 供下游消费。

════════════════════════════════════════════════════════════════════════
用法:
    python -m stock_ana.scan.ma_squeeze_scan              # 扫描 US + HK，输出文本 + JSON + 图表
    python -m stock_ana.scan.ma_squeeze_scan --us         # 仅美股
    python -m stock_ana.scan.ma_squeeze_scan --hk         # 仅港股

════════════════════════════════════════════════════════════════════════
输出文件 (在 data/output/ma_squeeze_today/ 下):

  1. scan_result_YYYY-MM-DD.json   — 结构化扫描结果
     格式:
     {
       "scan_date": "2026-03-21",         // 扫描执行日期
       "us_data_date": "2026-03-20",      // 美股数据最后交易日
       "hk_data_date": "2026-03-21",      // 港股数据最后交易日
       "signals": [
         {
           "market": "us",                // "us" | "hk"
           "ticker": "PEG",               // 代码
           "name": "PUBLIC SERVICE ...",   // 公司名称
           "description": "Utilities · Electric & Other Services Combined",  // 业务简介
                                           // 美股: sub_label（优先）或 sic_description + sector
                                           // 港股: 暂无（空字符串）
           "data_date": "2026-03-20",     // 该股数据最后交易日
           "stage": "S1",                 // "S1"(发现点) | "S2"(确认点)
           "close": 92.35,               // 最新收盘价
           "details": {                   // S1: 含 ma_squeeze_ratio / prange_* 等
             "ma_squeeze_ratio": 1.03,    // S2: 含 score / confirm_signals 等
             ...
           },
           "chart": "us/PEG_S1.png"       // 图表相对路径 (相对 chart_dir)
         },
         ...
       ]
     }

  2. us/<TICKER>_S1.png / us/<TICKER>_S2.png  — 美股图表
  3. hk/<CODE>_S1.png / hk/<CODE>_S2.png      — 港股图表

════════════════════════════════════════════════════════════════════════
外部集成 (openclaw 飞书推送):

  JSON 文件路径固定为:
    <PROJECT_ROOT>/data/output/ma_squeeze_today/scan_result_YYYY-MM-DD.json

  图表目录:
    <PROJECT_ROOT>/data/output/ma_squeeze_today/us/
    <PROJECT_ROOT>/data/output/ma_squeeze_today/hk/

  外部项目只需:
    1. 读取当天的 scan_result_*.json
    2. 遍历 signals 数组，组装飞书消息卡片
    3. 图表以 chart 字段的相对路径拼接 chart_dir 绝对路径即可发送
════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import sys
import time
from pathlib import Path


import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stock_ana.config import CACHE_DIR, DATA_DIR, OUTPUT_DIR
from stock_ana.strategies.impl.ma_squeeze import detect_stage1, detect_stage2, _MIN_HISTORY
from stock_ana.utils.plot_renderers import plot_ma_squeeze_chart

# 输出目录 — 每日扫描结果统一放这里
SCAN_DIR = OUTPUT_DIR / "ma_squeeze_today"
SCAN_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR = SCAN_DIR  # 图表子目录 us/ hk/ 在此之下

# CJK 字体


def _load_us_stocks() -> dict[str, tuple[pd.DataFrame, str, str]]:
    """加载美股数据，返回 {ticker: (df, display_name, description)}

    description 优先用 sub_label（LLM 分配的细分标签），其次 sic_description（SIC 行业描述）。
    """
    profiles_file = DATA_DIR / "us_sec_profiles.csv"
    if not profiles_file.exists():
        logger.warning("未找到 us_sec_profiles.csv，跳过美股")
        return {}

    profiles = pd.read_csv(profiles_file, encoding="utf-8-sig")
    tickers = profiles["ticker"].tolist()
    name_map = profiles.set_index("ticker")["company_name"].to_dict()

    # description = sub_label（优先） or sic_description，再加 sector
    def _make_desc(row) -> str:
        """Build a short display description from sector and the best available label."""
        label = row.get("sub_label", "")
        if pd.isna(label) or not str(label).strip():
            label = row.get("sic_description", "")
        sector = row.get("sector", "")
        label = str(label).strip() if pd.notna(label) else ""
        sector = str(sector).strip() if pd.notna(sector) else ""
        if label and sector:
            return f"{sector} · {label}"
        return label or sector

    desc_map = {row["ticker"]: _make_desc(row) for _, row in profiles.iterrows()}

    us_dir = CACHE_DIR / "us"
    ndx_dir = CACHE_DIR / "ndx100"

    result = {}
    for t in tickers:
        df = None
        for d in [us_dir, ndx_dir]:
            f = d / f"{t}.parquet"
            if f.exists():
                df = pd.read_parquet(f)
                df.columns = [c.lower() for c in df.columns]
                break
        if df is not None and len(df) >= _MIN_HISTORY:
            result[t] = (df, name_map.get(t, t), desc_map.get(t, ""))
    return result


def _load_hk_stocks() -> dict[str, tuple[pd.DataFrame, str, str]]:
    """加载港股数据，返回 {code: (df, display_name, description)}

    港股暂无业务描述数据，description 为空字符串。
    """
    hk_dir = CACHE_DIR / "hk"
    if not hk_dir.exists():
        logger.warning("未找到港股缓存目录，跳过港股")
        return {}

    # 名称映射
    name_map = {}
    hk_list_file = DATA_DIR / "hk_list.txt"
    if hk_list_file.exists():
        with open(hk_list_file, "r") as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    name_map[parts[0].strip().zfill(5)] = parts[1].strip()

    result = {}
    for f in hk_dir.glob("*.parquet"):
        code = f.stem
        df = pd.read_parquet(f)
        df.columns = [c.lower() for c in df.columns]
        if "volume" not in df.columns and "turnover" in df.columns:
            df["volume"] = df["turnover"]
        if "volume" not in df.columns or len(df) < _MIN_HISTORY:
            continue
        display = f"{code} {name_map.get(code, '')}"
        result[code] = (df, display, "")
    return result



def scan_market(stocks: dict[str, tuple[pd.DataFrame, str, str]], market: str) -> list[dict]:
    """扫描一个市场，返回触发的信号列表（含 data_date、close、description）"""
    hits = []
    total = len(stocks)
    t0 = time.time()

    for idx, (ticker, (df, display_name, description)) in enumerate(stocks.items()):
        data_date = df.index[-1]
        last_close = float(df["close"].iloc[-1])

        # S1 检测
        s1 = detect_stage1(df)
        if s1["triggered"]:
            path = plot_ma_squeeze_chart(df, ticker, display_name, "stage1",
                                         s1["details"], market, CHART_DIR)
            hits.append({
                "market": market,
                "ticker": ticker,
                "name": display_name,
                "description": description,
                "data_date": data_date.strftime("%Y-%m-%d"),
                "stage": "S1",
                "close": last_close,
                "details": s1["details"],
                "chart": str(Path(market) / path.name),
            })

        # S2 检测
        s2 = detect_stage2(df)
        if s2["triggered"]:
            s2_details = s2.get("details", {})
            s2_details["score"] = s2.get("score", 0)
            path = plot_ma_squeeze_chart(df, ticker, display_name, "stage2",
                                         s2_details, market, CHART_DIR)
            hits.append({
                "market": market,
                "ticker": ticker,
                "name": display_name,
                "description": description,
                "data_date": data_date.strftime("%Y-%m-%d"),
                "stage": "S2",
                "close": last_close,
                "details": s2_details,
                "chart": str(Path(market) / path.name),
            })

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            logger.info(f"  [{market}] {idx+1}/{total} ... {elapsed:.0f}s")

    elapsed = time.time() - t0
    logger.info(f"  [{market}] 扫描完成: {total} 只, 触发 {len(hits)} 个信号, {elapsed:.0f}s")
    return hits


def _get_data_date(stocks: dict[str, tuple[pd.DataFrame, str, str]]) -> str:
    """获取一批股票中最大的数据日期"""
    if not stocks:
        return ""
    dates = [df.index[-1] for df, _, _ in stocks.values() if len(df) > 0]
    return max(dates).strftime("%Y-%m-%d") if dates else ""


def _serialize_details(details: dict) -> dict:
    """将 details 中的 numpy/bool 类型转为 JSON 可序列化类型"""
    out = {}
    for k, v in details.items():
        if hasattr(v, 'item'):  # numpy scalar
            out[k] = v.item()
        elif isinstance(v, bool):
            out[k] = v
        else:
            out[k] = v
    return out


def main():
    """Run the daily MA squeeze scan CLI and persist JSON and chart outputs."""
    parser = argparse.ArgumentParser(description="多线压缩策略每日扫描")
    parser.add_argument("--us", action="store_true", help="仅扫描美股")
    parser.add_argument("--hk", action="store_true", help="仅扫描港股")
    args = parser.parse_args()

    run_all = not (args.us or args.hk)
    scan_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    all_hits = []
    us_data_date = ""
    hk_data_date = ""

    if run_all or args.us:
        logger.info("加载美股数据 ...")
        us_stocks = _load_us_stocks()
        logger.info(f"美股: {len(us_stocks)} 只可用")
        us_data_date = _get_data_date(us_stocks)
        us_hits = scan_market(us_stocks, "us")
        all_hits.extend(us_hits)

    if run_all or args.hk:
        logger.info("加载港股数据 ...")
        hk_stocks = _load_hk_stocks()
        logger.info(f"港股: {len(hk_stocks)} 只可用")
        hk_data_date = _get_data_date(hk_stocks)
        hk_hits = scan_market(hk_stocks, "hk")
        all_hits.extend(hk_hits)

    # ── 写 JSON ──
    json_signals = []
    for h in all_hits:
        json_signals.append({
            "market": h["market"],
            "ticker": h["ticker"],
            "name": h["name"],
            "description": h.get("description", ""),
            "data_date": h["data_date"],
            "stage": h["stage"],
            "close": h["close"],
            "details": _serialize_details(h["details"]),
            "chart": h["chart"],
        })

    result_obj = {
        "scan_date": scan_date,
        "us_data_date": us_data_date,
        "hk_data_date": hk_data_date,
        "chart_dir": str(SCAN_DIR),
        "signal_count": len(json_signals),
        "signals": json_signals,
    }

    json_path = SCAN_DIR / f"scan_result_{scan_date}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_obj, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON 已保存: {json_path}")

    # ── 打印文本摘要 ──
    print(f"\n{'='*75}")
    print(f"  多线压缩策略扫描结果 — {scan_date}")
    print(f"  美股数据日期: {us_data_date}  |  港股数据日期: {hk_data_date}")
    print(f"{'='*75}")

    s1_hits = [h for h in all_hits if h["stage"] == "S1"]
    s2_hits = [h for h in all_hits if h["stage"] == "S2"]

    if s1_hits:
        print(f"\n  ── 第一阶段（发现点）: {len(s1_hits)} 只 ──")
        for h in s1_hits:
            d = h["details"]
            print(f"    {h['data_date']}  {h['ticker']:>8s}  {h['name']:20s}  "
                  f"close={h['close']:<9.2f}  "
                  f"squeeze={d.get('ma_squeeze_ratio',''):>6}  "
                  f"pr10d={d.get('prange_10d',''):>5}%  "
                  f"pr20d={d.get('prange_20d',''):>5}%  "
                  f"pr60d={d.get('prange_60d',''):>5}%")

    if s2_hits:
        print(f"\n  ── 第二阶段（确认点）: {len(s2_hits)} 只 ──")
        for h in s2_hits:
            d = h["details"]
            signals = ", ".join(d.get("confirm_signals", []))
            print(f"    {h['data_date']}  {h['ticker']:>8s}  {h['name']:20s}  "
                  f"close={h['close']:<9.2f}  "
                  f"确认 {d.get('score', 0)}/5  [{signals}]")

    if not all_hits:
        print("\n  今日无信号触发")

    print(f"\n  JSON: {json_path}")
    print(f"  图表: {SCAN_DIR}/")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
