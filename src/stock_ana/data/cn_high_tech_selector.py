"""沪深高新技术选股工作流。

用法:
    python -m stock_ana.data.cn_high_tech_selector
    python -m stock_ana.data.cn_high_tech_selector --report-date 20260331
    python -m stock_ana.data.cn_high_tech_selector --max-asset-liability-ratio 60
    python -m stock_ana.data.cn_high_tech_selector --rnd-csv data/cn_rnd_ratio.csv --min-rnd-ratio 8

说明:
  - 默认执行行业/概念主题过滤 + 流动性过滤 + 基本面过滤。
  - 不包含技术面和资金面趋势过滤。
  - 研发费用率当前没有稳定的 akshare 批量来源，因此保留为可选外部 CSV 扩展位。
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
from loguru import logger

from stock_ana.config import OUTPUT_DIR
from stock_ana.data.fetcher_hk import _bypass_proxy

REPORT_DIR = OUTPUT_DIR / "cn_high_tech"
LISTS_DIR = Path(__file__).resolve().parents[3] / "data" / "lists"

DEFAULT_INDUSTRY_KEYWORDS: tuple[str, ...] = (
    "计算机",
    "软件",
    "互联网",
    "通信",
    "电子",
    "半导体",
    "元件",
    "自动化",
    "机器人",
    "专用设备",
    "专业工程",
)

DEFAULT_CONCEPT_KEYWORDS: tuple[str, ...] = (
    "人工智能",
    "AI",
    "机器人",
    "机器视觉",
    "半导体",
    "芯片",
    "算力",
    "信创",
    "国产软件",
    "工业软件",
    "数据中心",
    "服务器",
    "光模块",
)

DEFAULT_MIN_AVG_TURNOVER = 100_000_000.0
DEFAULT_MIN_PRICE = 3.0
DEFAULT_MIN_REVENUE_YOY = 15.0
DEFAULT_MIN_DEDUPED_PROFIT_YOY = 10.0
DEFAULT_MIN_OPERATING_CASHFLOW_PS = 0.0
DEFAULT_MAX_SUSPENSION_DAYS_60 = 2
DEFAULT_MAX_ASSET_LIABILITY_RATIO = 65.0
DEFAULT_MIN_GROSS_MARGIN = 0.0
DEFAULT_TURNOVER_WINDOW = 20
DEFAULT_SUSPENSION_WINDOW = 60
DEFAULT_LIMIT_WINDOW = 120
DEFAULT_HISTORY_CALENDAR_DAYS = 260
DEFAULT_REQUEST_PAUSE_SEC = 0.15


@dataclass(slots=True)
class ScreenConfig:
    industry_keywords: tuple[str, ...] = DEFAULT_INDUSTRY_KEYWORDS
    concept_keywords: tuple[str, ...] = DEFAULT_CONCEPT_KEYWORDS
    report_date: str | None = None
    min_avg_turnover_20d: float = DEFAULT_MIN_AVG_TURNOVER
    min_price: float = DEFAULT_MIN_PRICE
    min_revenue_yoy: float = DEFAULT_MIN_REVENUE_YOY
    min_deduped_profit_yoy: float = DEFAULT_MIN_DEDUPED_PROFIT_YOY
    min_operating_cashflow_ps: float = DEFAULT_MIN_OPERATING_CASHFLOW_PS
    max_suspension_days_60: int = DEFAULT_MAX_SUSPENSION_DAYS_60
    max_limit_like_days_120: int | None = None
    max_asset_liability_ratio: float | None = DEFAULT_MAX_ASSET_LIABILITY_RATIO
    min_gross_margin: float = DEFAULT_MIN_GROSS_MARGIN
    min_rnd_ratio: float | None = None
    rnd_csv: Path | None = None
    turnover_window: int = DEFAULT_TURNOVER_WINDOW
    suspension_window: int = DEFAULT_SUSPENSION_WINDOW
    limit_window: int = DEFAULT_LIMIT_WINDOW
    history_calendar_days: int = DEFAULT_HISTORY_CALENDAR_DAYS
    request_pause_sec: float = DEFAULT_REQUEST_PAUSE_SEC

    def threshold_snapshot(self) -> dict[str, object]:
        return {
            "report_date": self.report_date,
            "industry_keywords": list(self.industry_keywords),
            "concept_keywords": list(self.concept_keywords),
            "min_avg_turnover_20d": self.min_avg_turnover_20d,
            "min_price": self.min_price,
            "min_revenue_yoy": self.min_revenue_yoy,
            "min_deduped_profit_yoy": self.min_deduped_profit_yoy,
            "min_gross_margin": self.min_gross_margin,
            "min_operating_cashflow_ps": self.min_operating_cashflow_ps,
            "max_suspension_days_60": self.max_suspension_days_60,
            "max_limit_like_days_120": self.max_limit_like_days_120,
            "max_asset_liability_ratio": self.max_asset_liability_ratio,
            "min_rnd_ratio": self.min_rnd_ratio,
            "rnd_csv": str(self.rnd_csv) if self.rnd_csv else None,
        }


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [str(col).replace(" ", "").strip() for col in data.columns]
    return data


def _normalize_code(value: object) -> str:
    text = str(value or "").strip()
    match = re.search(r"(\d{6})", text)
    return match.group(1) if match else text.zfill(6)


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _contains_any(text: object, keywords: Iterable[str]) -> bool:
    normalized = _normalize_text(text)
    return any(keyword in normalized for keyword in keywords)


def _join_tags(values: Iterable[str]) -> str:
    tags = sorted({_normalize_text(value) for value in values if _normalize_text(value)})
    return "、".join(tags)


def _format_amount_in_yi(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "-"
    return f"{numeric / 1e8:.2f}"


def _format_pct(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "-"
    return f"{numeric:.1f}"


def _call_akshare(
    label: str,
    func: Callable[[], pd.DataFrame],
    retries: int = 3,
    pause_sec: float = 1.0,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with _bypass_proxy():
                return func()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(f"{label}: 第 {attempt}/{retries} 次失败 - {exc}")
            if attempt < retries:
                time.sleep(pause_sec * attempt)
    raise RuntimeError(f"{label} 失败: {last_error}") from last_error


def candidate_report_dates(anchor: date | None = None, years_back: int = 3) -> list[str]:
    anchor = anchor or date.today()
    dates: list[str] = []
    for year in range(anchor.year, anchor.year - years_back, -1):
        for month, day in ((12, 31), (9, 30), (6, 30), (3, 31)):
            quarter_end = date(year, month, day)
            if quarter_end <= anchor:
                dates.append(quarter_end.strftime("%Y%m%d"))
    return dates


def load_latest_earnings_report(report_date: str | None = None) -> tuple[str, pd.DataFrame]:
    import akshare as ak

    report_candidates = [report_date] if report_date else candidate_report_dates()
    last_error: Exception | None = None

    for candidate in report_candidates:
        try:
            raw_df = _call_akshare(
                f"业绩报表 {candidate}",
                lambda candidate=candidate: ak.stock_yjbb_em(date=candidate),
                retries=2,
                pause_sec=1.5,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

        if raw_df.empty:
            continue

        df = _normalize_columns(raw_df)
        rename_map = {
            "股票代码": "ticker",
            "股票简称": "company_name",
            "营业总收入-营业总收入": "revenue",
            "营业总收入-同比增长": "revenue_yoy",
            "净利润-净利润": "net_profit",
            "净利润-同比增长": "net_profit_yoy",
            "净资产收益率": "roe",
            "每股经营现金流量": "operating_cashflow_ps",
            "销售毛利率": "gross_margin",
            "所处行业": "industry_name",
            "最新公告日期": "notice_date",
        }
        missing = [column for column in rename_map if column not in df.columns]
        if missing:
            raise KeyError(f"业绩报表缺少必要列: {missing}")

        report_df = df[list(rename_map.keys())].rename(columns=rename_map)
        report_df["ticker"] = report_df["ticker"].map(_normalize_code)
        report_df["company_name"] = report_df["company_name"].map(_normalize_text)
        report_df["industry_name"] = report_df["industry_name"].map(_normalize_text)
        report_df = report_df[report_df["ticker"].str.match(r"^(0|3|6)\d{5}$")].copy()

        numeric_cols = [
            "revenue",
            "revenue_yoy",
            "net_profit",
            "net_profit_yoy",
            "roe",
            "operating_cashflow_ps",
            "gross_margin",
        ]
        for column in numeric_cols:
            report_df[column] = pd.to_numeric(report_df[column], errors="coerce")

        industry_median = (
            report_df.groupby("industry_name", dropna=False)["gross_margin"]
            .median()
            .rename("industry_gross_margin_median")
            .reset_index()
        )
        report_df = report_df.merge(industry_median, on="industry_name", how="left")
        logger.info(f"已加载业绩报表: {candidate} ({len(report_df)} 只沪深A股)")
        return candidate, report_df

    raise RuntimeError(f"无法加载可用业绩报表: {last_error}") from last_error


def _select_matching_boards(board_df: pd.DataFrame, keywords: Iterable[str]) -> pd.DataFrame:
    df = _normalize_columns(board_df)
    if "板块名称" not in df.columns or "板块代码" not in df.columns:
        raise KeyError("板块数据缺少 板块名称/板块代码 列")
    matched = df[df["板块名称"].map(lambda text: _contains_any(text, keywords))].copy()
    if matched.empty:
        return matched
    return matched[["板块名称", "板块代码"]].drop_duplicates().sort_values("板块名称").reset_index(drop=True)


def fetch_board_candidates(kind: str, keywords: Iterable[str], pause_sec: float) -> tuple[pd.DataFrame, dict[str, object]]:
    import akshare as ak

    meta: dict[str, object] = {
        "matched_boards": [],
        "failed_boards": [],
        "error": "",
    }
    if not list(keywords):
        return pd.DataFrame(), meta

    board_name_loader: Callable[[], pd.DataFrame]
    board_constituent_loader: Callable[..., pd.DataFrame]
    if kind == "industry":
        board_name_loader = ak.stock_board_industry_name_em
        board_constituent_loader = ak.stock_board_industry_cons_em
    elif kind == "concept":
        board_name_loader = ak.stock_board_concept_name_em
        board_constituent_loader = ak.stock_board_concept_cons_em
    else:
        raise ValueError(f"未知板块类型: {kind}")

    try:
        board_names = _call_akshare(f"{kind}板块列表", board_name_loader, retries=2, pause_sec=1.5)
        matched_boards = _select_matching_boards(board_names, keywords)
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)
        logger.warning(f"{kind} 板块拉取失败，跳过该维度: {exc}")
        return pd.DataFrame(), meta

    meta["matched_boards"] = matched_boards["板块名称"].tolist()
    records: dict[str, dict[str, object]] = {}

    _consecutive_failures = 0
    _FAST_FAIL_THRESHOLD = 3  # 连续失败超过此次数则放弃整个板块维度

    for _, board in matched_boards.iterrows():
        if _consecutive_failures >= _FAST_FAIL_THRESHOLD:
            logger.warning(f"{kind} 板块成分连续 {_consecutive_failures} 次失败，放弃板块维度改用 fallback")
            return pd.DataFrame(), meta

        board_name = _normalize_text(board["板块名称"])
        try:
            constituent_df = _call_akshare(
                f"{kind}板块成分 {board_name}",
                lambda board_name=board_name: board_constituent_loader(symbol=board_name),
                retries=2,
                pause_sec=1.5,
            )
            _consecutive_failures = 0  # 成功则重置计数
        except Exception as exc:  # noqa: BLE001
            _consecutive_failures += 1
            meta["failed_boards"].append(board_name)
            logger.warning(f"{kind} 板块 {board_name} 拉取失败: {exc}")
            continue

        df = _normalize_columns(constituent_df)
        if not {"代码", "名称"}.issubset(df.columns):
            meta["failed_boards"].append(board_name)
            logger.warning(f"{kind} 板块 {board_name} 缺少成分列，已跳过")
            continue

        for _, row in df.iterrows():
            ticker = _normalize_code(row.get("代码"))
            if not re.match(r"^(0|3|6)\d{5}$", ticker):
                continue
            entry = records.setdefault(
                ticker,
                {
                    "ticker": ticker,
                    "company_name": _normalize_text(row.get("名称")),
                    "board_latest_price": pd.NA,
                    "matched_industry_boards": set(),
                    "matched_concept_boards": set(),
                    "candidate_sources": set(),
                },
            )
            if kind == "industry":
                entry["matched_industry_boards"].add(board_name)
                entry["candidate_sources"].add("industry_board")
            else:
                entry["matched_concept_boards"].add(board_name)
                entry["candidate_sources"].add("concept_board")

            latest_price = pd.to_numeric(row.get("最新价"), errors="coerce")
            if pd.notna(latest_price):
                entry["board_latest_price"] = float(latest_price)

        time.sleep(pause_sec)

    rows: list[dict[str, object]] = []
    for record in records.values():
        rows.append(
            {
                "ticker": record["ticker"],
                "company_name": record["company_name"],
                "board_latest_price": record["board_latest_price"],
                "matched_industry_boards": _join_tags(record["matched_industry_boards"]),
                "matched_concept_boards": _join_tags(record["matched_concept_boards"]),
                "industry_match_count": len(record["matched_industry_boards"]),
                "concept_match_count": len(record["matched_concept_boards"]),
                "candidate_sources": _join_tags(record["candidate_sources"]),
            }
        )
    return pd.DataFrame(rows), meta


def build_candidate_universe(report_df: pd.DataFrame, config: ScreenConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    industry_board_df, industry_meta = fetch_board_candidates("industry", config.industry_keywords, config.request_pause_sec)
    concept_board_df, concept_meta = fetch_board_candidates("concept", config.concept_keywords, config.request_pause_sec)

    records: dict[str, dict[str, object]] = {}

    def touch(ticker: str) -> dict[str, object]:
        return records.setdefault(
            ticker,
            {
                "ticker": ticker,
                "company_name": "",
                "board_latest_price": pd.NA,
                "matched_industry_boards": set(),
                "matched_concept_boards": set(),
                "candidate_sources": set(),
            },
        )

    def absorb_board_df(df: pd.DataFrame) -> None:
        if df.empty:
            return
        for _, row in df.iterrows():
            ticker = _normalize_code(row.get("ticker"))
            entry = touch(ticker)
            entry["company_name"] = entry["company_name"] or _normalize_text(row.get("company_name"))
            latest_price = pd.to_numeric(row.get("board_latest_price"), errors="coerce")
            if pd.notna(latest_price):
                entry["board_latest_price"] = float(latest_price)

            for board_name in _normalize_text(row.get("matched_industry_boards")).split("、"):
                if board_name:
                    entry["matched_industry_boards"].add(board_name)
            for board_name in _normalize_text(row.get("matched_concept_boards")).split("、"):
                if board_name:
                    entry["matched_concept_boards"].add(board_name)
            for source in _normalize_text(row.get("candidate_sources")).split("、"):
                if source:
                    entry["candidate_sources"].add(source)

    absorb_board_df(industry_board_df)
    absorb_board_df(concept_board_df)

    fallback_df = report_df[report_df["industry_name"].map(lambda text: _contains_any(text, config.industry_keywords))].copy()
    for _, row in fallback_df.iterrows():
        ticker = _normalize_code(row.get("ticker"))
        entry = touch(ticker)
        entry["company_name"] = entry["company_name"] or _normalize_text(row.get("company_name"))
        industry_name = _normalize_text(row.get("industry_name"))
        if industry_name:
            entry["matched_industry_boards"].add(industry_name)
        entry["candidate_sources"].add("report_industry")

    candidate_rows: list[dict[str, object]] = []
    for record in records.values():
        candidate_rows.append(
            {
                "ticker": record["ticker"],
                "company_name": record["company_name"],
                "board_latest_price": record["board_latest_price"],
                "matched_industry_boards": _join_tags(record["matched_industry_boards"]),
                "matched_concept_boards": _join_tags(record["matched_concept_boards"]),
                "industry_match_count": len(record["matched_industry_boards"]),
                "concept_match_count": len(record["matched_concept_boards"]),
                "candidate_sources": _join_tags(record["candidate_sources"]),
            }
        )

    candidate_df = pd.DataFrame(candidate_rows)
    if candidate_df.empty:
        candidate_df = pd.DataFrame(columns=["ticker"])

    candidate_df = candidate_df.merge(report_df, on="ticker", how="left", suffixes=("", "_report"))
    if "company_name_report" in candidate_df.columns:
        candidate_df["company_name"] = candidate_df["company_name"].where(
            candidate_df["company_name"].map(bool), candidate_df["company_name_report"]
        )
        candidate_df = candidate_df.drop(columns=["company_name_report"])

    meta = {
        "industry": industry_meta,
        "concept": concept_meta,
        "fallback_industry_match_count": int(len(fallback_df)),
    }
    logger.info(f"主题候选池构建完成: {len(candidate_df)} 只")
    return candidate_df, meta


def apply_batch_filters(candidate_df: pd.DataFrame, config: ScreenConfig) -> pd.DataFrame:
    df = candidate_df.copy()
    if df.empty:
        return df

    # 只保留沪深主板：沪市 600xxx/601xxx/603xxx/605xxx，深市 000xxx/001xxx/002xxx/003xxx
    # 排除科创板 688xxx 和创业板 300xxx/301xxx
    df["is_mainland_a_share"] = df["ticker"].map(
        lambda code: bool(re.match(r"^(60[013]|605|00[0-3])\d{3}$", str(code)))
    )
    df["name_ok"] = ~df["company_name"].fillna("").str.upper().str.contains(r"ST|退")
    df["theme_match_ok"] = (df["industry_match_count"].fillna(0) > 0) | (df["concept_match_count"].fillna(0) > 0)
    df["revenue_yoy_ok"] = pd.to_numeric(df["revenue_yoy"], errors="coerce") >= config.min_revenue_yoy
    df["gross_margin_ok"] = (
        pd.to_numeric(df["gross_margin"], errors="coerce") >= config.min_gross_margin
    )
    df["operating_cashflow_ok"] = True  # 不再作为过滤条件，保留列供诊断参考
    df["prefilter_passed"] = df[
        [
            "is_mainland_a_share",
            "name_ok",
            "theme_match_ok",
            "revenue_yoy_ok",
            "gross_margin_ok",
        ]
    ].all(axis=1)
    return df


def load_spot_snapshot() -> pd.DataFrame:
    """一次批量调用获取全市场沪深A股实时快照（最新价、当日成交额）。"""
    import akshare as ak

    raw_df = _call_akshare("A股现货快照", ak.stock_zh_a_spot_em, retries=3, pause_sec=2.0)
    df = _normalize_columns(raw_df)
    if not {"代码", "最新价", "成交额"}.issubset(df.columns):
        raise KeyError(f"现货快照缺少必要列，当前列: {list(df.columns)}")
    df = df[["代码", "最新价", "成交额"]].copy()
    df["代码"] = df["代码"].map(_normalize_code)
    df["最新价"] = pd.to_numeric(df["最新价"], errors="coerce")
    df["成交额"] = pd.to_numeric(df["成交额"], errors="coerce")
    df = df.rename(columns={"代码": "ticker", "最新价": "latest_close", "成交额": "avg_turnover_20d"})
    logger.info(f"现货快照加载完成: {len(df)} 只")
    return df


def add_snapshot_metrics(df: pd.DataFrame, config: ScreenConfig) -> pd.DataFrame:
    """用一次批量现货快照替代逐股历史拉取，填充价格和成交额过滤列。
    若快照拉取失败（如夜间维护窗口），两条过滤默认放行，财务过滤照常执行。
    """
    try:
        snapshot = load_spot_snapshot()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"现货快照拉取失败，价格和成交额过滤本次跳过: {exc}")
        enriched = df.copy()
        enriched["latest_close"] = pd.NA
        enriched["avg_turnover_20d"] = pd.NA
        enriched["price_ok"] = True
        enriched["avg_turnover_ok"] = True
        enriched["suspension_ok"] = True
        enriched["limit_move_ok"] = True
        enriched["suspension_days_60"] = pd.NA
        enriched["limit_like_days_120"] = pd.NA
        enriched["history_passed"] = True
        return enriched
    enriched = df.merge(snapshot, on="ticker", how="left")
    enriched["price_ok"] = pd.to_numeric(enriched["latest_close"], errors="coerce") >= config.min_price
    enriched["avg_turnover_ok"] = (
        pd.to_numeric(enriched["avg_turnover_20d"], errors="coerce") >= config.min_avg_turnover_20d
    )
    enriched["suspension_ok"] = True
    enriched["limit_move_ok"] = True
    enriched["suspension_days_60"] = pd.NA
    enriched["limit_like_days_120"] = pd.NA
    enriched["history_passed"] = enriched[["price_ok", "avg_turnover_ok"]].all(axis=1)
    return enriched


def _metric_row(abstract_df: pd.DataFrame, metric_names: Iterable[str]) -> pd.Series | None:
    df = _normalize_columns(abstract_df)
    if df.empty or "指标" not in df.columns:
        return None

    normalized_metric = df["指标"].astype(str).str.replace(" ", "")
    for metric_name in metric_names:
        exact = df[normalized_metric == metric_name]
        if not exact.empty:
            return exact.iloc[0]
    for metric_name in metric_names:
        partial = df[normalized_metric.str.contains(metric_name, regex=False)]
        if not partial.empty:
            return partial.iloc[0]
    return None


def _resolve_report_column(date_columns: list[str], target_report_date: str | None) -> str | None:
    if not date_columns:
        return None
    if target_report_date and target_report_date in date_columns:
        return target_report_date
    if target_report_date:
        earlier = [column for column in date_columns if column <= target_report_date]
        if earlier:
            return max(earlier)
    return max(date_columns)


def extract_metric_from_abstract(
    abstract_df: pd.DataFrame,
    metric_names: Iterable[str],
    target_report_date: str | None = None,
) -> tuple[float | None, str | None]:
    df = _normalize_columns(abstract_df)
    row = _metric_row(df, [metric_name.replace(" ", "") for metric_name in metric_names])
    if row is None:
        return None, None

    date_columns = [str(column) for column in df.columns if re.fullmatch(r"\d{8}", str(column))]
    current_column = _resolve_report_column(date_columns, target_report_date)
    if current_column is None:
        return None, None

    value = pd.to_numeric(row.get(current_column), errors="coerce")
    if pd.isna(value):
        return None, current_column
    return float(value), current_column


def calc_metric_yoy_from_abstract(
    abstract_df: pd.DataFrame,
    metric_names: Iterable[str],
    target_report_date: str | None = None,
) -> tuple[float | None, str | None]:
    df = _normalize_columns(abstract_df)
    row = _metric_row(df, [metric_name.replace(" ", "") for metric_name in metric_names])
    if row is None:
        return None, None

    date_columns = [str(column) for column in df.columns if re.fullmatch(r"\d{8}", str(column))]
    current_column = _resolve_report_column(date_columns, target_report_date)
    if current_column is None:
        return None, None

    previous_column = f"{int(current_column[:4]) - 1}{current_column[4:]}"
    if previous_column not in row.index:
        return None, current_column

    current_value = pd.to_numeric(row.get(current_column), errors="coerce")
    previous_value = pd.to_numeric(row.get(previous_column), errors="coerce")
    if pd.isna(current_value) or pd.isna(previous_value) or previous_value == 0:
        return None, current_column

    yoy = (current_value / previous_value - 1) * 100.0
    return float(yoy), current_column


def load_rnd_ratio_file(path: Path) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(path))
    code_column = next((column for column in ("股票代码", "代码", "ticker", "symbol") if column in df.columns), None)
    ratio_column = next((column for column in ("研发费用率", "研发费率", "rnd_ratio") if column in df.columns), None)
    if code_column is None or ratio_column is None:
        raise KeyError("研发费用率文件必须包含 代码列 和 研发费用率列")

    result = df[[code_column, ratio_column]].copy()
    result.columns = ["ticker", "rnd_ratio"]
    result["ticker"] = result["ticker"].map(_normalize_code)
    result["rnd_ratio"] = pd.to_numeric(result["rnd_ratio"], errors="coerce")
    return result.dropna(subset=["ticker"])


def add_abstract_metrics(df: pd.DataFrame, config: ScreenConfig, report_date: str) -> pd.DataFrame:
    import akshare as ak

    rows: list[dict[str, object]] = []
    total = len(df)
    for index, ticker in enumerate(df["ticker"].tolist(), start=1):
        try:
            abstract_df = _call_akshare(
                f"{ticker} 财务摘要",
                lambda ticker=ticker: ak.stock_financial_abstract(symbol=ticker),
                retries=2,
                pause_sec=1.5,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{ticker} 财务摘要拉取失败: {exc}")
            abstract_df = pd.DataFrame()

        deduped_yoy, used_report_date = calc_metric_yoy_from_abstract(abstract_df, ["扣非净利润"], report_date)
        asset_liability_ratio, asset_report_date = extract_metric_from_abstract(
            abstract_df, ["资产负债率"], report_date
        )
        rows.append(
            {
                "ticker": ticker,
                "deduped_net_profit_yoy": deduped_yoy,
                "asset_liability_ratio": asset_liability_ratio,
                "abstract_report_date": used_report_date or asset_report_date,
            }
        )

        if index % 20 == 0 or index == total:
            logger.info(f"逐股财务摘要补齐进度: {index}/{total}")
        time.sleep(config.request_pause_sec)

    enriched = df.drop(
        columns=[c for c in ["deduped_net_profit_yoy", "asset_liability_ratio", "abstract_report_date"] if c in df.columns],
        errors="ignore",
    ).merge(pd.DataFrame(rows), on="ticker", how="left")
    if config.rnd_csv is not None:
        rnd_df = load_rnd_ratio_file(config.rnd_csv)
        enriched = enriched.merge(rnd_df, on="ticker", how="left")
    else:
        enriched["rnd_ratio"] = pd.NA

    enriched["deduped_net_profit_yoy_ok"] = (
        pd.to_numeric(enriched["deduped_net_profit_yoy"], errors="coerce") >= config.min_deduped_profit_yoy
    )
    if config.max_asset_liability_ratio is None:
        enriched["asset_liability_ok"] = True
    else:
        enriched["asset_liability_ok"] = (
            pd.to_numeric(enriched["asset_liability_ratio"], errors="coerce") <= config.max_asset_liability_ratio
        )
    if config.min_rnd_ratio is None:
        enriched["rnd_ratio_ok"] = True
    else:
        enriched["rnd_ratio_ok"] = pd.to_numeric(enriched["rnd_ratio"], errors="coerce") >= config.min_rnd_ratio
    return enriched


def _failure_reasons(row: pd.Series) -> str:
    ordered_rules = [
        ("name_ok", "名称含ST/退市标记"),
        ("theme_match_ok", "未命中目标行业/概念"),
        ("revenue_yoy_ok", "营收同比不达标"),
        ("gross_margin_ok", "毛利率为负"),
        ("operating_cashflow_ok", "每股经营现金流为负"),
        ("price_ok", "股价低于门槛"),
        ("avg_turnover_ok", "20日平均成交额不足"),
        ("suspension_ok", "近60日停牌天数过多"),
        ("limit_move_ok", "120日类涨跌停天数过多"),
        ("deduped_net_profit_yoy_ok", "扣非净利润同比不达标"),
        ("asset_liability_ok", "资产负债率超阈值"),
        ("rnd_ratio_ok", "研发费用率不达标"),
    ]
    reasons: list[str] = []
    for column, label in ordered_rules:
        value = row.get(column)
        if pd.isna(value):
            continue
        if not bool(value):
            reasons.append(label)
    return "；".join(reasons)


def _run_dir() -> Path:
    path = REPORT_DIR / date.today().strftime("%Y-%m-%d")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _markdown_table(df: pd.DataFrame) -> list[str]:
    headers = [
        "代码",
        "名称",
        "行业",
        "匹配概念",
        "20日均成交额(亿)",
        "营收同比%",
        "扣非净利同比%",
        "毛利率%",
        "资产负债率%",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(getattr(row, "ticker", "-")),
                    str(getattr(row, "company_name", "-")),
                    str(getattr(row, "industry_name", "-")),
                    str(getattr(row, "matched_concept_boards", "-")) or "-",
                    _format_amount_in_yi(getattr(row, "avg_turnover_20d", pd.NA)),
                    _format_pct(getattr(row, "revenue_yoy", pd.NA)),
                    _format_pct(getattr(row, "deduped_net_profit_yoy", pd.NA)),
                    _format_pct(getattr(row, "gross_margin", pd.NA)),
                    _format_pct(getattr(row, "asset_liability_ratio", pd.NA)),
                ]
            )
            + " |"
        )
    return lines


def write_outputs(df: pd.DataFrame, config: ScreenConfig, meta: dict[str, object]) -> dict[str, Path]:
    run_dir = _run_dir()

    diagnostics = df.copy()
    diagnostics["failure_reasons"] = diagnostics.apply(_failure_reasons, axis=1)
    diagnostics = diagnostics.sort_values(
        ["final_passed", "concept_match_count", "industry_match_count", "revenue_yoy", "deduped_net_profit_yoy"],
        ascending=[False, False, False, False, False],
    )

    survivors = diagnostics[diagnostics["final_passed"]].copy()
    survivors = survivors.sort_values(
        ["concept_match_count", "industry_match_count", "revenue_yoy", "deduped_net_profit_yoy", "avg_turnover_20d"],
        ascending=[False, False, False, False, False],
    )

    diagnostics_path = run_dir / "cn_high_tech_diagnostics.csv"
    survivors_path = run_dir / "cn_high_tech_candidates.csv"
    report_path = run_dir / "cn_high_tech_candidates.md"
    meta_path = run_dir / "cn_high_tech_meta.json"

    # 同步写入 data/lists/（纳入 git 追踪）
    lists_csv_path = LISTS_DIR / "cn_hightech.csv"
    lists_md_path = LISTS_DIR / "cn_hightech.md"

    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    survivors.to_csv(survivors_path, index=False, encoding="utf-8-sig")
    survivors.to_csv(lists_csv_path, index=False, encoding="utf-8-sig")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 沪深高新技术选股结果",
        "",
        f"> 运行日期：{date.today().strftime('%Y-%m-%d')}",
        f"> 财报期：{meta['report_date']}",
        f"> 主题候选池：{meta['candidate_count']} 只",
        f"> 基本面预筛后：{meta['prefilter_count']} 只",
        f"> 流动性过滤后：{meta['history_count']} 只",
        f"> 全部规则通过：{meta['final_count']} 只",
        "",
        "## 筛选口径",
        "",
        f"- 行业关键词：{_join_tags(config.industry_keywords)}",
        f"- 概念关键词：{_join_tags(config.concept_keywords)}",
        f"- 近20日平均成交额 ≥ {config.min_avg_turnover_20d / 1e8:.2f} 亿",
        f"- 最新价 ≥ {config.min_price:.2f}",
        f"- 营收同比 ≥ {config.min_revenue_yoy:.1f}%",        f"- 毛利率下限：{config.min_gross_margin:.1f}%",        f"- 扣非净利润同比 ≥ {config.min_deduped_profit_yoy:.1f}%",
        f"- 资产负债率上限：{config.max_asset_liability_ratio if config.max_asset_liability_ratio is not None else '未启用'}",
        f"- 研发费用率下限：{config.min_rnd_ratio if config.min_rnd_ratio is not None else '未启用'}",
        "",
        "## 板块命中情况",
        "",
        f"- 行业板块命中：{_join_tags(meta['board_meta']['industry']['matched_boards']) or '无'}",
        f"- 概念板块命中：{_join_tags(meta['board_meta']['concept']['matched_boards']) or '无'}",
        f"- 行业板块失败：{_join_tags(meta['board_meta']['industry']['failed_boards']) or '无'}",
        f"- 概念板块失败：{_join_tags(meta['board_meta']['concept']['failed_boards']) or '无'}",
        "",
        "## 候选预览",
        "",
    ]
    if survivors.empty:
        lines.append("本次没有股票通过全部硬过滤。")
    else:
        lines.extend(_markdown_table(survivors.head(30)))

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    lists_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "diagnostics_csv": diagnostics_path,
        "candidates_csv": survivors_path,
        "report_md": report_path,
        "lists_csv": lists_csv_path,
        "lists_md": lists_md_path,
        "meta_json": meta_path,
    }


def run_cn_high_tech_selector(config: ScreenConfig) -> tuple[pd.DataFrame, dict[str, Path], dict[str, object]]:
    report_date, report_df = load_latest_earnings_report(config.report_date)
    effective_config = replace(config, report_date=report_date)

    candidate_df, board_meta = build_candidate_universe(report_df, effective_config)
    filtered_df = apply_batch_filters(candidate_df, effective_config)

    history_columns = [
        "latest_close",
        "avg_turnover_20d",
        "suspension_days_60",
        "limit_like_days_120",
        "price_ok",
        "avg_turnover_ok",
        "suspension_ok",
        "limit_move_ok",
        "history_passed",
    ]
    for column in history_columns:
        filtered_df[column] = pd.NA

    prefiltered = filtered_df[filtered_df["prefilter_passed"]].copy()
    if not prefiltered.empty:
        snapshot_df = add_snapshot_metrics(prefiltered, effective_config)
        filtered_df = filtered_df.drop(columns=history_columns).merge(
            snapshot_df[["ticker", *history_columns]], on="ticker", how="left"
        )

    abstract_columns = [
        "deduped_net_profit_yoy",
        "asset_liability_ratio",
        "abstract_report_date",
        "rnd_ratio",
        "deduped_net_profit_yoy_ok",
        "asset_liability_ok",
        "rnd_ratio_ok",
    ]
    for column in abstract_columns:
        filtered_df[column] = pd.NA

    history_passed = filtered_df[
        filtered_df["prefilter_passed"].fillna(False) & filtered_df["history_passed"].fillna(False)
    ].copy()
    if not history_passed.empty:
        abstract_df = add_abstract_metrics(history_passed, effective_config, report_date)
        filtered_df = filtered_df.drop(columns=abstract_columns).merge(
            abstract_df[["ticker", *abstract_columns]], on="ticker", how="left"
        )

    asset_ok = filtered_df["asset_liability_ok"].fillna(True if effective_config.max_asset_liability_ratio is None else False)
    rnd_ok = filtered_df["rnd_ratio_ok"].fillna(True if effective_config.min_rnd_ratio is None else False)
    filtered_df["final_passed"] = (
        filtered_df["prefilter_passed"].fillna(False)
        & filtered_df["history_passed"].fillna(False)
        & filtered_df["deduped_net_profit_yoy_ok"].fillna(False)
        & asset_ok
        & rnd_ok
    )

    meta = {
        "report_date": report_date,
        "candidate_count": int(len(candidate_df)),
        "prefilter_count": int(filtered_df["prefilter_passed"].fillna(False).sum()),
        "history_count": int(
            (filtered_df["prefilter_passed"].fillna(False) & filtered_df["history_passed"].fillna(False)).sum()
        ),
        "final_count": int(filtered_df["final_passed"].fillna(False).sum()),
        "board_meta": board_meta,
        "thresholds": effective_config.threshold_snapshot(),
    }
    output_paths = write_outputs(filtered_df, effective_config, meta)  # defined in write_outputs
    return filtered_df, output_paths, meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="沪深高新技术选股")
    parser.add_argument("--report-date", help="指定财报期，例如 20260331")
    parser.add_argument(
        "--industry-keywords",
        nargs="+",
        default=list(DEFAULT_INDUSTRY_KEYWORDS),
        help="行业关键词列表",
    )
    parser.add_argument(
        "--concept-keywords",
        nargs="+",
        default=list(DEFAULT_CONCEPT_KEYWORDS),
        help="概念关键词列表",
    )
    parser.add_argument("--min-avg-turnover", type=float, default=DEFAULT_MIN_AVG_TURNOVER, help="近20日平均成交额下限")
    parser.add_argument("--min-price", type=float, default=DEFAULT_MIN_PRICE, help="股价下限")
    parser.add_argument("--min-revenue-yoy", type=float, default=DEFAULT_MIN_REVENUE_YOY, help="营收同比下限")
    parser.add_argument(
        "--min-deduped-profit-yoy",
        type=float,
        default=DEFAULT_MIN_DEDUPED_PROFIT_YOY,
        help="扣非净利润同比下限",
    )
    parser.add_argument(
        "--min-operating-cashflow-ps",
        type=float,
        default=DEFAULT_MIN_OPERATING_CASHFLOW_PS,
        help="每股经营现金流下限",
    )
    parser.add_argument(
        "--max-suspension-days-60",
        type=int,
        default=DEFAULT_MAX_SUSPENSION_DAYS_60,
        help="近60个交易日允许的最大停牌天数",
    )
    parser.add_argument(
        "--max-limit-like-days-120",
        type=int,
        default=None,
        help="近120个交易日类涨跌停天数上限；默认不启用",
    )
    parser.add_argument(
        "--max-asset-liability-ratio",
        type=float,
        default=DEFAULT_MAX_ASSET_LIABILITY_RATIO,
        help="资产负债率上限；传空值请改代码或显式调整",
    )
    parser.add_argument("--rnd-csv", type=Path, default=None, help="可选的研发费用率 CSV")
    parser.add_argument("--min-rnd-ratio", type=float, default=None, help="研发费用率下限；默认不启用")
    parser.add_argument("--min-gross-margin", type=float, default=DEFAULT_MIN_GROSS_MARGIN, help="毛利率下限（百分比）；默认不启用")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = ScreenConfig(
        industry_keywords=tuple(args.industry_keywords),
        concept_keywords=tuple(args.concept_keywords),
        report_date=args.report_date,
        min_avg_turnover_20d=args.min_avg_turnover,
        min_price=args.min_price,
        min_revenue_yoy=args.min_revenue_yoy,
        min_deduped_profit_yoy=args.min_deduped_profit_yoy,
        min_operating_cashflow_ps=args.min_operating_cashflow_ps,
        max_suspension_days_60=args.max_suspension_days_60,
        max_limit_like_days_120=args.max_limit_like_days_120,
        max_asset_liability_ratio=args.max_asset_liability_ratio,
        min_gross_margin=args.min_gross_margin,
        min_rnd_ratio=args.min_rnd_ratio,
        rnd_csv=args.rnd_csv,
    )

    result_df, output_paths, meta = run_cn_high_tech_selector(config)
    logger.info(f"选股完成：{meta['final_count']} 只通过全部条件")
    for name, path in output_paths.items():
        logger.info(f"{name}: {path}")
    logger.info(f"诊断样本数：{len(result_df)}")


if __name__ == "__main__":
    main()
