"""
港股 RS（相对强度）每日计算与报告输出

用法:
    python -m stock_ana.scan.hk_rs_scan

计算每只港股标的相对于其所属基准指数的 RS 值：
- 若标的属于恒生科技指数 → 基准为恒生科技 (800700)
- 否则 → 基准为恒生指数 (800000)
- 同时属于两者 → 优先用恒生科技

RS 定义：
  RS_raw  = 股票收盘价 / 基准收盘价（按日对齐后相除）
  RS_daily = RS_raw 归一化为起始日 = 100 的序列
  RS_rank  = 63 日收益率在同基准全体中的百分位排名 (0~100)
"""

from datetime import datetime
from pathlib import Path

from stock_ana.config import OUTPUT_DIR
from stock_ana.data.fetcher_hk import (
    load_hk_list,
    load_hk_local,
    _INDEX_MAP,
)
from stock_ana.utils.plot_renderers import plot_hk_rs_charts as _render_hk_rs_charts

# 输出目录
HK_RS_DIR = OUTPUT_DIR / "hk_rs"
HK_RS_DIR.mkdir(parents=True, exist_ok=True)

# RS 每日序列存储目录
HK_RS_SERIES_DIR = OUTPUT_DIR / "hk_rs_series"
HK_RS_SERIES_DIR.mkdir(parents=True, exist_ok=True)

# RS 图表存储目录
HK_RS_CHARTS_DIR = OUTPUT_DIR / "hk_rs_charts"
HK_RS_CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────── RS 计算 ───────────────────


def _compute_rs_line(df_stock: pd.DataFrame,
                     df_market: pd.DataFrame) -> pd.Series | None:
    """
    计算 RS Line = Stock Close / Market Close，归一化首日 = 100。
    """
    common_idx = df_stock.index.intersection(df_market.index)
    if len(common_idx) < 60:
        return None

    stock_close = df_stock.loc[common_idx, "close"]
    market_close = df_market.loc[common_idx, "close"]

    rs_raw = stock_close / market_close
    # 归一化
    rs_line = rs_raw / rs_raw.iloc[0] * 100
    return rs_line


def _compute_rs_momentum(
    df_stock: pd.DataFrame,
    df_market: pd.DataFrame,
    lookback: int = 63,
    smooth: int = 10,
) -> pd.Series | None:
    """
    计算 RS Z-Score = 价格比的滚动标准化偏差。

    ratio  = Stock Close / Bench Close
    zscore = (ratio − SMA(ratio, N)) / StdDev(ratio, N)
    RS     = EMA(zscore, smooth)

    以 0 为中轴：
      > 0   当前相对表现优于近 N 日平均
      = 0   与近 N 日平均一致
      < 0   当前相对表现弱于近 N 日平均

    单位：标准差 (σ)，±1 = 较明显偏离，±2 = 显著偏离。
    """
    common_idx = df_stock.index.intersection(df_market.index)
    if len(common_idx) < lookback + smooth:
        return None

    stock_close = df_stock.loc[common_idx, "close"]
    bench_close = df_market.loc[common_idx, "close"]

    # 价格比
    ratio = stock_close / bench_close

    # 滚动均值和标准差
    ratio_sma = ratio.rolling(lookback).mean()
    ratio_std = ratio.rolling(lookback).std()

    # Z-Score
    zscore = (ratio - ratio_sma) / ratio_std.replace(0, np.nan)

    # EMA 平滑
    rs_mom = zscore.ewm(span=smooth).mean()
    return rs_mom


def _compute_rs_rank(stock_data: dict[str, pd.DataFrame],
                     df_market: pd.DataFrame,
                     lookback: int = 63) -> dict[str, float]:
    """
    计算每只股票 N 日收益率在同组中的百分位排名 (0~100)。
    """
    returns: dict[str, float] = {}

    for code, df in stock_data.items():
        common_idx = df.index.intersection(df_market.index)
        if len(common_idx) < lookback:
            continue
        aligned_close = df.loc[common_idx, "close"]
        if len(aligned_close) < lookback:
            continue
        ret = (aligned_close.iloc[-1] / aligned_close.iloc[-lookback] - 1) * 100
        returns[code] = ret

    if not returns:
        return {}

    all_rets = np.array(list(returns.values()))
    ranks = {}
    for code, ret in returns.items():
        pct = float(np.mean(all_rets <= ret)) * 100
        ranks[code] = round(pct, 1)
    return ranks


def compute_hk_rs_daily() -> pd.DataFrame:
    """
    计算全部港股标的的每日 RS 指标。

    Returns:
        DataFrame: columns = [
            code, name, benchmark, rs_rank_63d, rs_rank_21d,
            rs_momentum, rs_momentum_21d,
            rs_latest, rs_ema21, rs_chg_5d, rs_chg_21d, rs_chg_63d,
            close, pct_5d, pct_21d, strength
        ]
    """
    hk_list = load_hk_list()

    # 加载基准指数
    df_hsi = load_hk_local("800000")
    df_hstech = load_hk_local("800700")

    if df_hsi is None or df_hstech is None:
        raise RuntimeError("指数数据缺失，请先运行 update_hk_data()")

    # 按基准分组：hstech=True → 恒生科技，否则 → 恒生指数
    stocks_by_bench: dict[str, list[str]] = {"800700": [], "800000": []}
    code_bench_map: dict[str, str] = {}

    for _, row in hk_list.iterrows():
        code = row["code"]
        if code in _INDEX_MAP:
            continue  # 跳过指数自身
        if row["hstech"]:
            bench = "800700"
        else:
            bench = "800000"
        stocks_by_bench[bench].append(code)
        code_bench_map[code] = bench

    bench_df_map = {"800000": df_hsi, "800700": df_hstech}
    bench_name_map = {"800000": "恒生指数", "800700": "恒生科技"}

    # 加载个股数据
    all_stock_data: dict[str, pd.DataFrame] = {}
    for code in code_bench_map:
        df = load_hk_local(code)
        if df is not None and not df.empty:
            all_stock_data[code] = df

    # 按基准分组计算 RS 排名
    rs_ranks_63d: dict[str, float] = {}
    rs_ranks_21d: dict[str, float] = {}

    for bench_code, codes in stocks_by_bench.items():
        df_bench = bench_df_map[bench_code]
        group_data = {c: all_stock_data[c] for c in codes if c in all_stock_data}
        if not group_data:
            continue
        r63 = _compute_rs_rank(group_data, df_bench, lookback=63)
        r21 = _compute_rs_rank(group_data, df_bench, lookback=21)
        rs_ranks_63d.update(r63)
        rs_ranks_21d.update(r21)

    # 逐只计算 RS 详细指标
    code_name_map = dict(zip(hk_list["code"], hk_list["name"]))
    rows = []

    for code, df_stock in all_stock_data.items():
        bench_code = code_bench_map.get(code)
        if bench_code is None:
            continue
        df_bench = bench_df_map[bench_code]

        rs_line = _compute_rs_line(df_stock, df_bench)
        if rs_line is None or len(rs_line) < 21:
            continue

        # RS 动量 (Z-Score)
        rs_mom = _compute_rs_momentum(df_stock, df_bench, lookback=63, smooth=10)
        rs_mom_val = round(float(rs_mom.iloc[-1]), 2) if rs_mom is not None and len(rs_mom.dropna()) > 0 else np.nan

        rs_mom_21 = _compute_rs_momentum(df_stock, df_bench, lookback=21, smooth=10)
        rs_mom_21_val = round(float(rs_mom_21.iloc[-1]), 2) if rs_mom_21 is not None and len(rs_mom_21.dropna()) > 0 else np.nan

        # RS 指标
        rs_latest = rs_line.iloc[-1]
        rs_ema21 = rs_line.ewm(span=21).mean().iloc[-1]
        rs_chg_5d = (rs_line.iloc[-1] / rs_line.iloc[-6] - 1) * 100 if len(rs_line) >= 6 else np.nan
        rs_chg_21d = (rs_line.iloc[-1] / rs_line.iloc[-22] - 1) * 100 if len(rs_line) >= 22 else np.nan
        rs_chg_63d = (rs_line.iloc[-1] / rs_line.iloc[-64] - 1) * 100 if len(rs_line) >= 64 else np.nan

        # 价格指标
        close = df_stock["close"].iloc[-1]
        close_5 = df_stock["close"].iloc[-6] if len(df_stock) >= 6 else np.nan
        close_21 = df_stock["close"].iloc[-22] if len(df_stock) >= 22 else np.nan
        pct_5d = (close / close_5 - 1) * 100 if not np.isnan(close_5) else np.nan
        pct_21d = (close / close_21 - 1) * 100 if not np.isnan(close_21) else np.nan

        # 强弱判定
        rank_63 = rs_ranks_63d.get(code, np.nan)
        strength = _classify_strength(rank_63)

        rows.append({
            "code": code,
            "name": code_name_map.get(code, ""),
            "benchmark": bench_name_map[bench_code],
            "rs_rank_63d": rank_63,
            "rs_rank_21d": rs_ranks_21d.get(code, np.nan),
            "rs_momentum": rs_mom_val,
            "rs_momentum_21d": rs_mom_21_val,
            "rs_latest": round(rs_latest, 2),
            "rs_ema21": round(rs_ema21, 2),
            "rs_chg_5d": round(rs_chg_5d, 2) if not np.isnan(rs_chg_5d) else np.nan,
            "rs_chg_21d": round(rs_chg_21d, 2) if not np.isnan(rs_chg_21d) else np.nan,
            "rs_chg_63d": round(rs_chg_63d, 2) if not np.isnan(rs_chg_63d) else np.nan,
            "close": round(close, 3),
            "pct_5d": round(pct_5d, 2) if not np.isnan(pct_5d) else np.nan,
            "pct_21d": round(pct_21d, 2) if not np.isnan(pct_21d) else np.nan,
            "strength": strength,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("rs_rank_63d", ascending=False).reset_index(drop=True)

    logger.info(f"港股 RS 计算完成：{len(result)} 只标的")
    return result


def _classify_strength(rank: float) -> str:
    """根据 63 日 RS 排名给出强弱分类"""
    if np.isnan(rank):
        return "N/A"
    if rank >= 90:
        return "极强"
    elif rank >= 70:
        return "强势"
    elif rank >= 50:
        return "中等偏强"
    elif rank >= 30:
        return "中等偏弱"
    elif rank >= 10:
        return "弱势"
    else:
        return "极弱"


# ─────────────────── 报告输出 ───────────────────


_REPORT_HEADER = """\
================================================================================
                      港股核心标的 RS（相对强度）每日报告
================================================================================
生成时间: {timestamp}
数据截止: {data_date}

┌─────────────────────────────────────────────────────────────────────────────┐
│  RS 排名 (rs_rank_63d) 含义 — 63日收益率在同基准组内的百分位排名             │
│                                                                             │
│  90~100  极强     该股近63日涨幅超过组内90%的股票，处于领涨地位               │
│  70~89   强势     明显跑赢大多数同组股票，趋势向上                           │
│  50~69   中等偏强  略优于中位数，表现尚可                                     │
│  30~49   中等偏弱  略弱于中位数，缺乏动能                                    │
│  10~29   弱势     明显跑输大多数同组股票                                      │
│   0~9    极弱     近63日涨幅垫底，处于最弱梯队                               │
│                                                                             │
│  基准选择规则：属于恒生科技成份股 → 对比恒生科技指数                          │
│              其余 → 对比恒生指数                                              │
│                                                                             │
│  RS 动量: Z-Score of (股价/基准价) vs N日均值和标准差                      │
│    63d: 中期趋势视角 | 21d: 短期灵敏视角                                  │
│    > 0: 相对表现优于近N日平均 | = 0: 与平均一致 | < 0: 弱于平均      │
│    ±1σ: 较明显偏离 | ±2σ: 显著偏离                                        │
│  rs_chg_Nd: RS Line 近N日变化率(%)，正值=相对走强                            │
│  rs_ema21: RS Line 的21日指数均线，RS > EMA21 = 短期趋势向上                 │
└─────────────────────────────────────────────────────────────────────────────┘

恒生指数组: {hsi_count} 只  |  恒生科技组: {hstech_count} 只
================================================================================

"""


def generate_hk_rs_report() -> Path:
    """
    计算全部港股 RS 并输出 txt 报告到 data/output/hk_rs/。

    Returns:
        输出文件路径
    """
    df = compute_hk_rs_daily()
    if df.empty:
        logger.warning("无有效 RS 数据")
        return HK_RS_DIR

    # 数据截止日
    sample_code = df["code"].iloc[0]
    sample_df = load_hk_local(sample_code)
    data_date = sample_df.index.max().strftime("%Y-%m-%d") if sample_df is not None else "unknown"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hsi_count = len(df[df["benchmark"] == "恒生指数"])
    hstech_count = len(df[df["benchmark"] == "恒生科技"])

    header = _REPORT_HEADER.format(
        timestamp=timestamp,
        data_date=data_date,
        hsi_count=hsi_count,
        hstech_count=hstech_count,
    )

    lines = [header]

    # ── 全部标的排名表 ──
    lines.append("【一、RS 排名总览（按 rs_rank_63d 降序）】\n")
    lines.append(
        f"{'排名':>4}  {'代码':<6}  {'名称':<12}  {'基准':<6}  "
        f"{'RS排名63d':>9}  {'RS排名21d':>9}  {'RS动量63d':>9}  {'RS动量21d':>9}  "
        f"{'RS变化5d':>8}  {'RS变化21d':>9}  {'RS变化63d':>9}  "
        f"{'收盘价':>8}  {'涨跌5d':>7}  {'涨跌21d':>8}  {'强弱':>4}"
    )
    lines.append("-" * 165)

    for i, row in df.iterrows():
        lines.append(
            f"{i+1:>4}  {row['code']:<6}  {row['name']:<12}  {row['benchmark']:<6}  "
            f"{row['rs_rank_63d']:>9.1f}  {row['rs_rank_21d']:>9.1f}  "
            f"{_fmt_pp(row['rs_momentum']):>9}  "
            f"{_fmt_pp(row['rs_momentum_21d']):>9}  "
            f"{_fmt_pct(row['rs_chg_5d']):>8}  {_fmt_pct(row['rs_chg_21d']):>9}  "
            f"{_fmt_pct(row['rs_chg_63d']):>9}  "
            f"{row['close']:>8.3f}  {_fmt_pct(row['pct_5d']):>7}  "
            f"{_fmt_pct(row['pct_21d']):>8}  {row['strength']:>4}"
        )

    # ── 分组展示 ──
    lines.append("\n")
    lines.append("=" * 80)
    lines.append("【二、按强弱分组】\n")

    for level in ["极强", "强势", "中等偏强", "中等偏弱", "弱势", "极弱"]:
        group = df[df["strength"] == level]
        if group.empty:
            continue
        lines.append(f"── {level} ({len(group)} 只) ──")
        for _, row in group.iterrows():
            lines.append(
                f"  {row['code']} {row['name']:<10}  "
                f"RS排名={row['rs_rank_63d']:.1f}%  "
                f"RS动量63d={_fmt_pp(row['rs_momentum'])}  "
                f"RS动量21d={_fmt_pp(row['rs_momentum_21d'])}  "
                f"收盘={row['close']:.3f}  [{row['benchmark']}]"
            )
        lines.append("")

    # ── 写入文件 ──
    today_str = datetime.now().strftime("%Y%m%d")
    filename = f"hk_rs_{today_str}.txt"
    filepath = HK_RS_DIR / filename
    filepath.write_text("\n".join(lines), encoding="utf-8")

    logger.info(f"港股 RS 报告已生成: {filepath}")
    return filepath


def _fmt_pct(val) -> str:
    """格式化百分比值"""
    if pd.isna(val):
        return "N/A"
    return f"{val:+.2f}%"


def _fmt_pp(val) -> str:
    """格式化 Z-Score 值"""
    if pd.isna(val):
        return "N/A"
    return f"{val:+.2f}σ"


# ─────────────────── RS 每日序列 & 图表 ───────────────────


def _prepare_benchmark_groups() -> tuple[
    dict[str, str],           # code_bench_map
    dict[str, pd.DataFrame],  # bench_df_map
    dict[str, str],           # code_name_map
    dict[str, pd.DataFrame],  # all_stock_data
]:
    """准备基准分组和数据加载（公共逻辑提取）"""
    hk_list = load_hk_list()

    df_hsi = load_hk_local("800000")
    df_hstech = load_hk_local("800700")
    if df_hsi is None or df_hstech is None:
        raise RuntimeError("指数数据缺失，请先运行 update_hk_data()")

    code_bench_map: dict[str, str] = {}
    for _, row in hk_list.iterrows():
        code = row["code"]
        if code in _INDEX_MAP:
            continue
        code_bench_map[code] = "800700" if row["hstech"] else "800000"

    bench_df_map = {"800000": df_hsi, "800700": df_hstech}
    code_name_map = dict(zip(hk_list["code"], hk_list["name"]))

    all_stock_data: dict[str, pd.DataFrame] = {}
    for code in code_bench_map:
        df = load_hk_local(code)
        if df is not None and not df.empty:
            all_stock_data[code] = df

    return code_bench_map, bench_df_map, code_name_map, all_stock_data


def compute_all_rs_series() -> dict[str, pd.DataFrame]:
    """
    计算每只港股标的的完整每日 RS 序列。

    RS = Z-Score of (Stock/Bench ratio) vs rolling mean & std。
    以 0 为中轴，单位为标准差，±1 = 较明显偏离，±2 = 显著偏离。

    两个时间窗口：
    - 63d：中期趋势（rs_zscore / rs_zscore_fast）
    - 21d：短期灵敏（rs_zscore_21d / rs_zscore_21d_fast）

    Returns:
        {code: DataFrame} 其中 DataFrame columns =
            [rs_zscore, rs_zscore_fast, rs_zscore_21d, rs_zscore_21d_fast,
             close, bench_close]
        index = date
    """
    code_bench_map, bench_df_map, code_name_map, all_stock_data = _prepare_benchmark_groups()

    result: dict[str, pd.DataFrame] = {}

    for code, df_stock in all_stock_data.items():
        bench_code = code_bench_map.get(code)
        if bench_code is None:
            continue
        df_bench = bench_df_map[bench_code]

        # ── 63d 窗口：慢线 EMA10 和快线 EMA5 ──
        rs_slow = _compute_rs_momentum(df_stock, df_bench, lookback=63, smooth=10)
        if rs_slow is None:
            continue
        rs_fast = _compute_rs_momentum(df_stock, df_bench, lookback=63, smooth=5)

        # ── 21d 窗口：慢线 EMA10 和快线 EMA5 ──
        rs_slow_21 = _compute_rs_momentum(df_stock, df_bench, lookback=21, smooth=10)
        rs_fast_21 = _compute_rs_momentum(df_stock, df_bench, lookback=21, smooth=5)

        common_idx = rs_slow.dropna().index
        if len(common_idx) < 30:
            continue

        stock_close = df_stock.loc[common_idx, "close"]
        bench_close = df_bench.loc[common_idx, "close"]

        rs_df = pd.DataFrame({
            "rs_zscore": rs_slow.loc[common_idx],
            "rs_zscore_fast": rs_fast.loc[common_idx] if rs_fast is not None else np.nan,
            "rs_zscore_21d": rs_slow_21.reindex(common_idx) if rs_slow_21 is not None else np.nan,
            "rs_zscore_21d_fast": rs_fast_21.reindex(common_idx) if rs_fast_21 is not None else np.nan,
            "close": stock_close,
            "bench_close": bench_close,
        })
        rs_df.index.name = "date"
        result[code] = rs_df

    logger.info(f"RS 序列计算完成：{len(result)} 只标的")
    return result


def save_rs_series(rs_data: dict[str, pd.DataFrame] | None = None) -> Path:
    """
    将每只股票的每日 RS 序列保存为 CSV 文件。

    文件保存在 data/output/hk_rs_series/ 目录下，
    每只股票一个文件：{code}_{name}.csv

    Returns:
        输出目录路径
    """
    if rs_data is None:
        rs_data = compute_all_rs_series()

    _, _, code_name_map, _ = _prepare_benchmark_groups()
    code_bench_map, _, _, _ = _prepare_benchmark_groups()
    bench_name_map = {"800000": "恒生指数", "800700": "恒生科技"}

    for code, rs_df in rs_data.items():
        name = code_name_map.get(code, "")
        bench = bench_name_map.get(code_bench_map.get(code, ""), "")
        safe_name = name.replace("/", "_").replace(" ", "")
        filename = f"{code}_{safe_name}.csv"

        # 保存时添加元信息作为注释行
        filepath = HK_RS_SERIES_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {code} {name} | 基准: {bench}\n")
            f.write(f"# rs_zscore: Z-Score of (股价/基准价) vs 63日均值和标准差, EMA10平滑\n")
            f.write(f"# rs_zscore_fast: 同上但用EMA5平滑(快线)\n")
            f.write(f"# rs_zscore_21d: Z-Score of (股价/基准价) vs 21日均值和标准差, EMA10平滑(更灵敏)\n")
            f.write(f"# rs_zscore_21d_fast: 同上但用EMA5平滑(快线)\n")
            f.write(f"#   > 0 相对表现优于近N日平均 | = 0 与平均一致 | < 0 相对表现弱于平均\n")
            f.write(f"# 单位: 标准差(σ), ±1=较明显偏离, ±2=显著偏离\n")
            rs_df.round(4).to_csv(f)

    logger.info(f"RS 序列已保存：{len(rs_data)} 个文件 → {HK_RS_SERIES_DIR}")
    return HK_RS_SERIES_DIR




def plot_rs_charts(rs_data=None, days=0):
    """Thin wrapper: resolves maps, delegates rendering to plot_renderers."""
    if rs_data is None:
        rs_data = compute_all_rs_series()
    code_bench_map, _, code_name_map, _ = _prepare_benchmark_groups()
    return _render_hk_rs_charts(rs_data, code_bench_map, code_name_map,
                               HK_RS_CHARTS_DIR, days)


def main() -> None:
    """每日港股 RS 计算入口：计算全量 RS 指标、保存序列 CSV、生成图表。"""
    from loguru import logger

    logger.info("=== 港股 RS 每日计算 ===")
    logger.info("Step 1/3: 计算每日 RS 指标...")
    rs_daily = compute_hk_rs_daily()
    logger.info(f"共计算 {len(rs_daily)} 只标的")

    logger.info("Step 2/3: 保存 RS 序列 CSV...")
    save_rs_series_csv()

    logger.info("Step 3/3: 生成 RS 图表...")
    plot_rs_charts()

    logger.success("港股 RS 计算完成。")


if __name__ == "__main__":
    main()
