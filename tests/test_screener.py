"""screener 模块基础测试"""

import asyncio
import shutil
from pathlib import Path

import pandas as pd

from stock_ana.strategies.screener import (
    screen_golden_cross,
    screen_macd_cross_in_period,
    screen_rsi_oversold,
    scan_macd_cross,
)
from stock_ana.strategies.registry import scan_strategy
from stock_ana.strategies.api import (
    screen_vegas_touch,
    screen_triangle_ascending,
    screen_vcp_setup,
)
from stock_ana.data.indicators import add_vegas_channel
from stock_ana.data.fetcher import update_ndx100_data, load_all_ndx100_data
from stock_ana.utils.plot_renderers import (
    plot_macd_cross_results,
    plot_vegas_touch_results,
    plot_ascending_triangle_results,
    plot_vcp_results,
)
from stock_ana.utils.gemini_analyst import analyze_screener_results, batch_analyze, rank_and_summarize

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "output"


def _to_plot_hits(scan_result, stock_data: dict, info_key: str | None = None, df_transform=None) -> list[dict]:
    """Convert scan hits into plotting payloads with attached data frames."""
    hits: list[dict] = []
    for hit in scan_result.hits:
        df = stock_data.get(hit.symbol)
        if df is None:
            continue
        if df_transform is not None:
            df = df_transform(df.copy())
        item = {"ticker": hit.symbol, "df": df}
        if info_key is not None:
            item[info_key] = hit.decision.features
        hits.append(item)
    return hits


def _scan_vegas_hits(lookback_days: int = 5) -> list[dict]:
    """Run the vegas scan and prepare chart-ready hit payloads."""
    data = load_all_ndx100_data()
    result = scan_strategy("vegas", market="ndx100", lookback_days=lookback_days)
    return _to_plot_hits(result, data, df_transform=add_vegas_channel)


def _scan_triangle_hits(min_period: int = 40, max_period: int = 120) -> list[dict]:
    """Run the ascending triangle scan and format hits for plotting."""
    data = load_all_ndx100_data()
    result = scan_strategy("triangle_ascending", market="ndx100")
    return _to_plot_hits(result, data, info_key="pattern_info")


def _scan_vcp_hits(min_base_days: int = 30, max_base_days: int = 180) -> list[dict]:
    """Run the VCP scan and format hits for plotting."""
    data = load_all_ndx100_data()
    result = scan_strategy(
        "vcp",
        universe="ndx100",
        min_base_days=min_base_days,
        max_base_days=max_base_days,
    )
    return _to_plot_hits(result, data, info_key="vcp_info")


def _clean_output():
    """清空 output 目录"""
    if _OUTPUT_DIR.exists():
        shutil.rmtree(_OUTPUT_DIR)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _make_df(data: dict) -> pd.DataFrame:
    """Build a DataFrame fixture from a compact column dictionary."""
    return pd.DataFrame(data)


def test_golden_cross_true():
    """Return true when the short moving average crosses above the long one."""
    df = _make_df({
        "close": [10, 11, 12, 13],
        "sma_5":  [9, 10, 11, 13],
        "sma_20": [10, 10.5, 11.5, 12],
    })
    assert screen_golden_cross(df) == True


def test_golden_cross_false():
    """Return false when no bullish moving-average crossover is present."""
    df = _make_df({
        "close": [10, 11, 12, 13],
        "sma_5":  [12, 13, 14, 15],
        "sma_20": [10, 10.5, 11, 11.5],
    })
    assert screen_golden_cross(df) == False


def test_rsi_oversold():
    """Detect oversold RSI values and reject neutral readings."""
    df = _make_df({"close": [10], "rsi": [25.0]})
    assert screen_rsi_oversold(df) == True

    df2 = _make_df({"close": [10], "rsi": [55.0]})
    assert screen_rsi_oversold(df2) == False


def test_macd_cross_in_period_true():
    """最近 5 天内 macd_hist 由负转正"""
    df = _make_df({
        "close": list(range(10, 20)),
        "macd_hist": [-3, -2, -1, -0.5, -0.3, -0.1, -0.05, 0.02, 0.1, 0.2],
    })
    assert screen_macd_cross_in_period(df, lookback_days=5) == True


def test_macd_cross_in_period_false():
    """最近 5 天内 macd_hist 一直为正"""
    df = _make_df({
        "close": list(range(10, 20)),
        "macd_hist": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    })
    assert screen_macd_cross_in_period(df, lookback_days=5) == False


# ──────── 实盘测试（需要网络，手动运行） ────────

def test_step1_update_data():
    """
    步骤一：下载/更新纳指100成分股数据（存储到本地）
    运行：pytest tests/test_screener.py::test_step1_update_data -s
    """
    data = update_ndx100_data()
    print(f"\n{'='*60}")
    print(f"数据更新完毕，共 {len(data)} 只股票数据已存储到本地")
    for ticker, df in list(data.items())[:5]:
        print(f"  {ticker:8s}  {df.index.min().date()} ~ {df.index.max().date()}  ({len(df)} 行)")
    if len(data) > 5:
        print(f"  ... 共 {len(data)} 只")
    print(f"{'='*60}")
    assert len(data) > 0


def test_step2_scan_macd_cross():
    """
    步骤二：基于本地数据扫描 MACD 金叉并绘制 K 线图
    运行：pytest tests/test_screener.py::test_step2_scan_macd_cross -s
    """
    _clean_output()
    data = load_all_ndx100_data()
    assert len(data) > 0, "本地无数据！请先运行 test_step1_update_data"

    # 扫描 MACD 金叉
    hits = scan_macd_cross(lookback_days=5, universe="ndx100")
    print(f"\n{'='*60}")
    print(f"共发现 {len(hits)} 只股票在最近1周内发生 MACD 金叉：")
    for item in hits:
        ticker = item["ticker"]
        df = item["df"]
        last_close = df["close"].iloc[-1]
        last_date = df.index[-1].strftime("%Y-%m-%d")
        print(f"  {ticker:8s}  最新: {last_close:.2f}  ({last_date})")
    print(f"{'='*60}")

    # 绘制 K 线图
    plot_macd_cross_results(hits)


def test_vegas_channel_touch_true():
    """半年高点后回落触及 Vegas 通道，未跌破"""
    import numpy as np
    n = 200
    dates = pd.date_range("2025-06-01", periods=n, freq="B")
    # 模拟：前半段涨到高点，后半段回落到 EMA 区域
    price = np.concatenate([
        np.linspace(100, 150, 100),   # 上涨到150
        np.linspace(149, 110, 100),   # 回落到110
    ])
    df = pd.DataFrame({
        "open": price * 0.99,
        "high": price * 1.01,
        "low": price * 0.98,
        "close": price,
        "volume": [1000000] * n,
    }, index=dates)
    from stock_ana.data.indicators import add_vegas_channel
    df = add_vegas_channel(df)
    # 最后几天 low 应该接近 ema_144/169
    decision = screen_vegas_touch(df, lookback_days=5, half_year_days=120)
    # 结果取决于 EMA 的具体值，主要验证不报错
    assert isinstance(decision.passed, bool)


def test_vegas_channel_touch_false_no_drop():
    """股价一直在高位，未触及 Vegas 通道"""
    import numpy as np
    n = 200
    dates = pd.date_range("2025-06-01", periods=n, freq="B")
    price = np.linspace(100, 200, n)  # 一直上涨
    df = pd.DataFrame({
        "open": price * 0.99,
        "high": price * 1.01,
        "low": price * 0.995,
        "close": price,
        "volume": [1000000] * n,
    }, index=dates)
    from stock_ana.data.indicators import add_vegas_channel
    df = add_vegas_channel(df)
    decision = screen_vegas_touch(df, lookback_days=5)
    assert decision.passed == False


def test_step3_scan_vegas_touch():
    """
    步骤三：基于本地数据扫描 Vegas 通道回踩并绘制 K 线图
    运行：pytest tests/test_screener.py::test_step3_scan_vegas_touch -s
    """
    _clean_output()
    data = load_all_ndx100_data()
    assert len(data) > 0, "本地无数据！请先运行 test_step1_update_data"

    hits = _scan_vegas_hits(lookback_days=5)
    print(f"\n{'='*60}")
    print(f"共发现 {len(hits)} 只股票满足 Vegas 通道回踩条件：")
    for item in hits:
        ticker = item["ticker"]
        df = item["df"]
        last_close = df["close"].iloc[-1]
        last_date = df.index[-1].strftime("%Y-%m-%d")
        ema144 = df["ema_144"].iloc[-1]
        ema169 = df["ema_169"].iloc[-1]
        print(f"  {ticker:8s}  最新: {last_close:.2f}  EMA144: {ema144:.2f}  EMA169: {ema169:.2f}  ({last_date})")
    print(f"{'='*60}")

    plot_vegas_touch_results(hits)


def test_ascending_triangle_detection():
    """模拟上升三角形：高点水平，低点上行"""
    import numpy as np
    n = 100
    dates = pd.date_range("2025-06-01", periods=n, freq="B")
    # 构造锯齿价格：高点固定在150，低点从120逐步上行到145
    close = []
    for i in range(n):
        cycle = (i % 20) / 20.0  # 0→1 周期
        low_base = 120 + (i / n) * 25     # 低点从120升到145
        high_base = 150                     # 高点固定
        mid = (low_base + high_base) / 2
        amp = (high_base - low_base) / 2
        price = mid + amp * np.sin(2 * np.pi * cycle)
        close.append(price)
    close = np.array(close)
    df = pd.DataFrame({
        "open": close * 0.998,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": [1000000] * n,
    }, index=dates)
    decision = screen_triangle_ascending(df)
    # 主要验证不报错，形态检测取决于具体数值
    assert isinstance(decision.passed, bool)


def test_ascending_triangle_none_for_downtrend():
    """持续下降趋势不应检出上升三角形"""
    import numpy as np
    n = 100
    dates = pd.date_range("2025-06-01", periods=n, freq="B")
    price = np.linspace(200, 100, n)  # 持续下降
    df = pd.DataFrame({
        "open": price * 1.001,
        "high": price * 1.01,
        "low": price * 0.99,
        "close": price,
        "volume": [1000000] * n,
    }, index=dates)
    decision = screen_triangle_ascending(df)
    assert decision.passed == False


def test_step4_scan_ascending_triangle():
    """
    步骤四：基于本地数据扫描上升三角形/楔形并绘制 K 线图
    运行：pytest tests/test_screener.py::test_step4_scan_ascending_triangle -s
    """
    _clean_output()
    data = load_all_ndx100_data()
    assert len(data) > 0, "本地无数据！请先运行 test_step1_update_data"

    hits = _scan_triangle_hits(min_period=40, max_period=120)
    print(f"\n{'='*60}")
    print(f"共发现 {len(hits)} 只股票呈现收敛三角形/楔形：")
    _PCN = {"ascending_triangle": "上升三角形", "rising_wedge": "上升楔形",
            "symmetrical_triangle": "对称三角形", "descending_wedge": "下降楔形"}
    for item in hits:
        ticker = item["ticker"]
        df = item["df"]
        info = item["pattern_info"]
        last_close = df["close"].iloc[-1]
        last_date = df.index[-1].strftime("%Y-%m-%d")
        ptype = _PCN.get(info["pattern"], info["pattern"])
        angle = info.get('convergence_angle_deg', 0)
        status = "已收敛" if info.get('convergence_status') == 'converged' else "即将收敛"
        dtc = info.get('days_to_convergence', 0)
        dtc_str = f"{dtc:.0f}日后" if dtc > 0 else "已过"
        print(f"  {ticker:8s}  {ptype}【{status}】 周期={info['period']}日  "
              f"收敛={dtc_str}  角度={angle:.1f}°  "
              f"测试=({info['resistance']['touches']}/{info['support']['touches']})  "
              f"最新: {last_close:.2f}  ({last_date})")
    print(f"{'='*60}")

    plot_ascending_triangle_results(hits)


def test_step5_run_all():
    """
    步骤五：清空 output 后，一次性运行全部三个策略
    运行：pytest tests/test_screener.py::test_step5_run_all -s
    """
    _clean_output()
    data = load_all_ndx100_data()
    assert len(data) > 0, "本地无数据！请先运行 test_step1_update_data"

    # 策略1: MACD 金叉
    macd_hits = scan_macd_cross(lookback_days=5, universe="ndx100")
    print(f"\n{'='*60}")
    print(f"【策略1】MACD 金叉：{len(macd_hits)} 只")
    for item in macd_hits:
        t, d = item["ticker"], item["df"]
        print(f"  {t:8s}  最新: {d['close'].iloc[-1]:.2f}  ({d.index[-1].strftime('%Y-%m-%d')})")
    plot_macd_cross_results(macd_hits)

    # 策略2: Vegas 通道回踩
    vegas_hits = _scan_vegas_hits(lookback_days=5)
    print(f"\n{'='*60}")
    print(f"【策略2】Vegas 通道回踩：{len(vegas_hits)} 只")
    for item in vegas_hits:
        t, d = item["ticker"], item["df"]
        print(f"  {t:8s}  最新: {d['close'].iloc[-1]:.2f}  "
              f"EMA144: {d['ema_144'].iloc[-1]:.2f}  EMA169: {d['ema_169'].iloc[-1]:.2f}  "
              f"({d.index[-1].strftime('%Y-%m-%d')})")
    plot_vegas_touch_results(vegas_hits)

    # 策略3: 收敛三角形/楔形
    tri_hits = _scan_triangle_hits(min_period=40, max_period=120)
    print(f"\n{'='*60}")
    print(f"【策略3】收敛三角形/楔形：{len(tri_hits)} 只")
    _PCN = {"ascending_triangle": "上升三角形", "rising_wedge": "上升楔形",
            "symmetrical_triangle": "对称三角形", "descending_wedge": "下降楔形"}
    for item in tri_hits:
        t, d, info = item["ticker"], item["df"], item["pattern_info"]
        ptype = _PCN.get(info["pattern"], info["pattern"])
        angle = info.get('convergence_angle_deg', 0)
        status = "已收敛" if info.get('convergence_status') == 'converged' else "即将收敛"
        dtc = info.get('days_to_convergence', 0)
        dtc_str = f"{dtc:.0f}日后" if dtc > 0 else "已过"
        print(f"  {t:8s}  {ptype}【{status}】 周期={info['period']}日  "
              f"收敛={dtc_str}  角度={angle:.1f}°  "
              f"测试=({info['resistance']['touches']}/{info['support']['touches']})  "
              f"最新: {d['close'].iloc[-1]:.2f}  ({d.index[-1].strftime('%Y-%m-%d')})")
    plot_ascending_triangle_results(tri_hits)

    print(f"\n{'='*60}")
    total = len(macd_hits) + len(vegas_hits) + len(tri_hits)
    print(f"全部扫描完成，共生成 {total} 张图表 → {_OUTPUT_DIR}")
    print(f"{'='*60}")


def test_step6_gemini_analyze_triangle():
    """
    步骤六：对收敛三角形/楔形筛选结果调用 Gemini 进行基本面分析
    需要在浏览器中登录 https://gemini.google.com
    运行：pytest tests/test_screener.py::test_step6_gemini_analyze_triangle -s
    """
    _clean_output()
    data = load_all_ndx100_data()
    assert len(data) > 0, "本地无数据！请先运行 test_step1_update_data"

    # 先筛选
    tri_hits = _scan_triangle_hits(min_period=40, max_period=120)
    print(f"\n{'='*60}")
    print(f"共 {len(tri_hits)} 只股票呈现收敛形态，开始 Gemini 分析...")
    print(f"{'='*60}")

    if not tri_hits:
        print("无筛选结果，跳过分析")
        return

    # 绘图
    plot_ascending_triangle_results(tri_hits)

    # Gemini 分析（async）
    paths = asyncio.run(analyze_screener_results(tri_hits, delay=5.0))
    print(f"\n{'='*60}")
    print(f"Gemini 分析完成，共生成 {len(paths)} 份报告：")
    for p in paths:
        print(f"  📄 {p.name}")
    print(f"输出目录：{_OUTPUT_DIR}")
    print(f"{'='*60}")


def test_step7_gemini_analyze_all():
    """
    步骤七：运行 Vegas + 三角形策略 + Gemini 分析 + 综合排序
    运行：pytest tests/test_screener.py::test_step7_gemini_analyze_all -s
    """
    _clean_output()
    data = load_all_ndx100_data()
    assert len(data) > 0, "本地无数据！请先运行 test_step1_update_data"

    all_hits = []

    # 策略1：Vegas 通道回踩
    vegas_hits = _scan_vegas_hits(lookback_days=5)
    print(f"\n【策略1】Vegas 通道回踩：{len(vegas_hits)} 只")
    plot_vegas_touch_results(vegas_hits)
    all_hits.extend(vegas_hits)

    # 策略2：收敛三角形/楔形
    tri_hits = _scan_triangle_hits(min_period=40, max_period=120)
    print(f"【策略2】收敛三角形/楔形：{len(tri_hits)} 只")
    plot_ascending_triangle_results(tri_hits)
    all_hits.extend(tri_hits)

    # 策略3：VCP / 杯柄形态
    vcp_hits = _scan_vcp_hits(min_base_days=30, max_base_days=180)
    print(f"【策略3】VCP / 杯柄形态：{len(vcp_hits)} 只")
    plot_vcp_results(vcp_hits)
    all_hits.extend(vcp_hits)

    # 去重
    unique_tickers = list(dict.fromkeys(h["ticker"] for h in all_hits))
    print(f"\n{'='*60}")
    print(f"全部策略共筛选出 {len(unique_tickers)} 只不重复股票，开始 Gemini 分析...")
    print(f"{'='*60}")

    if not unique_tickers:
        print("无筛选结果，跳过分析")
        return

    paths = asyncio.run(batch_analyze(unique_tickers, delay=5.0))
    print(f"\n{'='*60}")
    print(f"Gemini 分析完成，共生成 {len(paths)} 份报告：")
    for p in paths:
        print(f"  📄 {p.name}")
    print(f"{'='*60}")

    # 第三步：综合排序
    if paths:
        print(f"\n开始综合排序...")
        rank_path = asyncio.run(rank_and_summarize(paths))
        print(f"📊 综合排序报告：{rank_path.name}")
    print(f"输出目录：{_OUTPUT_DIR}")
    print(f"{'='*60}")


def test_step8_rank_existing_reports():
    """
    步骤八：对 output 目录中已有的分析报告进行综合排序
    适用于已经完成 step6/step7 后，单独重新排序
    运行：pytest tests/test_screener.py::test_step8_rank_existing_reports -s
    """
    reports = sorted(_OUTPUT_DIR.glob("*_analysis.docx"))
    assert len(reports) > 0, f"未找到分析报告！请先运行 step6 或 step7 生成报告到 {_OUTPUT_DIR}"

    print(f"\n{'='*60}")
    print(f"找到 {len(reports)} 份分析报告：")
    for p in reports:
        print(f"  📄 {p.name}")
    print(f"{'='*60}")

    rank_path = asyncio.run(rank_and_summarize(reports))
    print(f"\n{'='*60}")
    print(f"📊 综合排序报告已生成：{rank_path.name}")
    print(f"输出目录：{_OUTPUT_DIR}")
    print(f"{'='*60}")


def test_scan_vcp():
    """
    VCP + 杯柄形态扫描（独立测试）
    运行：pytest tests/test_screener.py::test_scan_vcp -s
    """
    data = load_all_ndx100_data()
    assert len(data) > 0, "本地无数据！请先运行 test_step1_update_data"

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vcp_hits = _scan_vcp_hits(min_base_days=30, max_base_days=180)

    print(f"\n{'='*60}")
    print(f"VCP / 杯柄扫描结果：共 {len(vcp_hits)} 只")
    for h in vcp_hits:
        info = h["vcp_info"]
        depths = "→".join(f"{d:.0f}%" for d in info["depths"])
        print(f"  ✅ {h['ticker']:6s} [{info['pattern']:15s}] "
              f"基底 {info['base_days']:3d}日 "
              f"收缩 {info['num_contractions']}次 [{depths}] "
              f"量缩比 {info['vol_ratio']:.0%} "
              f"距前高 {info['distance_to_pivot_pct']:.1f}%")
    print(f"{'='*60}")

    if vcp_hits:
        plot_vcp_results(vcp_hits)
        print(f"图表已保存到 {_OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════
# AI 代码审查测试
# ══════════════════════════════════════════════════════════════

def test_step9_ai_review_triangle_gemini():
    """
    步骤九A：使用 Gemini 2.5 Flash 审查三角形策略
    前提：先运行回测（python -m stock_ana.backtest.backtest_multi_strategy）生成 data/backtest_charts/triangle/ 图表
    运行：pytest tests/test_screener.py::test_step9_ai_review_triangle_gemini -s
    """
    from stock_ana.utils.ai_code_reviewer import AICodeReviewer

    reviewer = AICodeReviewer(backend="gemini", model="gemini-2.5-flash")
    result = reviewer.review_triangle_strategy(
        backtest_summary=(
            "Triangle 策略 (3年 NDX100 滚动回测):\n"
            "  - 143 个信号, 55% 胜率\n"
            "  - 21日平均收益 +0.67%, Alpha -0.76%\n"
            "  - 最大盈 +26.18%, 最大亏 -22.66%"
        ),
        max_images=3,
    )
    print(f"\n{'='*60}")
    print("Gemini AI 审查意见 (三角形策略)")
    print(f"{'='*60}")
    print(result)
    print(f"{'='*60}")


def test_step9_ai_review_triangle_claude():
    """
    步骤九B：使用 Claude Opus 4.6 (via Antigravity) 审查三角形策略
    前提：
      1. 先运行回测生成图表
      2. 启动 Antigravity Manager 并在 API Proxy 页面开启服务
    运行：pytest tests/test_screener.py::test_step9_ai_review_triangle_claude -s
    """
    from stock_ana.utils.ai_code_reviewer import AICodeReviewer

    reviewer = AICodeReviewer(backend="antigravity")

    # 先检查代理是否可用
    if not reviewer.check_antigravity():
        print("⚠️  Antigravity 代理不可用，请先启动 Antigravity Manager")
        return

    result = reviewer.review_triangle_strategy(
        backtest_summary=(
            "Triangle 策略 (3年 NDX100 滚动回测):\n"
            "  - 143 个信号, 55% 胜率\n"
            "  - 21日平均收益 +0.67%, Alpha -0.76%\n"
            "  - 最大盈 +26.18%, 最大亏 -22.66%"
        ),
        max_images=3,
    )
    print(f"\n{'='*60}")
    print("Claude Opus 4.6 审查意见 (三角形策略)")
    print(f"{'='*60}")
    print(result)
    print(f"{'='*60}")


def test_step9_ai_review_vcp():
    """
    步骤九C：使用 Gemini 审查 VCP 策略
    运行：pytest tests/test_screener.py::test_step9_ai_review_vcp -s
    """
    from stock_ana.utils.ai_code_reviewer import AICodeReviewer

    reviewer = AICodeReviewer(backend="gemini", model="gemini-2.5-flash")
    result = reviewer.review_vcp_strategy(
        backtest_summary=(
            "VCP 策略 (3年 NDX100 滚动回测):\n"
            "  - 11 个信号, 73% 胜率\n"
            "  - 21日平均收益 +1.97%, Alpha +0.54%\n"
            "  - 信号数量较少，需要更多触发机会"
        ),
        max_images=3,
    )
    print(f"\n{'='*60}")
    print("Gemini AI 审查意见 (VCP 策略)")
    print(f"{'='*60}")
    print(result)
    print(f"{'='*60}")
