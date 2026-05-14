"""cn_high_tech_selector 选股核心逻辑测试。"""

from __future__ import annotations

import pandas as pd

from stock_ana.data.cn_high_tech_selector import (
    ScreenConfig,
    apply_batch_filters,
    build_candidate_universe,
    calc_metric_yoy_from_abstract,
    extract_metric_from_abstract,
)


def test_apply_batch_filters_marks_expected_rules() -> None:
    config = ScreenConfig(
        min_revenue_yoy=15.0,
        min_operating_cashflow_ps=0.0,
    )
    df = pd.DataFrame(
        [
            {
                "ticker": "600001",
                "company_name": "科技样本",
                "industry_match_count": 1,
                "concept_match_count": 2,
                "revenue_yoy": 28.0,
                "gross_margin": 35.0,
                "industry_gross_margin_median": 30.0,
                "operating_cashflow_ps": 0.5,
            },
            {
                "ticker": "600002",
                "company_name": "ST科技",
                "industry_match_count": 1,
                "concept_match_count": 0,
                "revenue_yoy": 20.0,
                "gross_margin": 32.0,
                "industry_gross_margin_median": 30.0,
                "operating_cashflow_ps": 0.3,
            },
            {
                "ticker": "600003",
                "company_name": "低毛利样本",
                "industry_match_count": 1,
                "concept_match_count": 0,
                "revenue_yoy": 18.0,
                "gross_margin": 18.0,
                "industry_gross_margin_median": 25.0,
                "operating_cashflow_ps": -0.1,
            },
        ]
    )

    result = apply_batch_filters(df, config).set_index("ticker")

    assert bool(result.loc["600001", "prefilter_passed"]) is True
    assert bool(result.loc["600002", "name_ok"]) is False
    assert bool(result.loc["600002", "prefilter_passed"]) is False
    # gross_margin_ok 用固定下限（默认0%），18% > 0% 因此通过
    assert bool(result.loc["600003", "gross_margin_ok"]) is True
    # operating_cashflow_ok 已移除过滤，始终为 True
    assert bool(result.loc["600003", "operating_cashflow_ok"]) is True
    # 600003: gross_margin_ok=True, revenue_yoy_ok=True, industry_match_count=1 → prefilter_passed=True
    assert bool(result.loc["600003", "prefilter_passed"]) is True


def test_snapshot_metrics_mock_replaces_history() -> None:
    """快照模式下：价格和成交额从快照拿，停牌字段直接为 NA。"""
    from unittest.mock import patch

    from stock_ana.data.cn_high_tech_selector import add_snapshot_metrics

    config = ScreenConfig(min_price=5.0, min_avg_turnover_20d=100_000_000.0)
    candidate_df = pd.DataFrame(
        [
            {"ticker": "600001", "company_name": "A"},
            {"ticker": "600002", "company_name": "B"},
        ]
    )
    fake_snapshot = pd.DataFrame(
        [
            {"ticker": "600001", "latest_close": 10.0, "avg_turnover_20d": 200_000_000.0},
            {"ticker": "600002", "latest_close": 2.0, "avg_turnover_20d": 50_000_000.0},
        ]
    )
    with patch(
        "stock_ana.data.cn_high_tech_selector.load_spot_snapshot",
        return_value=fake_snapshot,
    ):
        enriched = add_snapshot_metrics(candidate_df, config).set_index("ticker")

    assert bool(enriched.loc["600001", "history_passed"]) is True
    assert bool(enriched.loc["600002", "history_passed"]) is False
    assert bool(enriched.loc["600001", "suspension_ok"]) is True
    assert pd.isna(enriched.loc["600001", "suspension_days_60"])


def test_abstract_metric_helpers_respect_target_report_date() -> None:
    abstract_df = pd.DataFrame(
        {
            "选项": ["常用指标", "常用指标"],
            "指标": ["扣非净利润", "资产负债率"],
            "20250331": [110.0, 45.0],
            "20240331": [100.0, 40.0],
            "20231231": [95.0, 41.0],
        }
    )

    yoy, yoy_col = calc_metric_yoy_from_abstract(abstract_df, ["扣非净利润"], "20250630")
    liability_ratio, liability_col = extract_metric_from_abstract(abstract_df, ["资产负债率"], "20250630")

    assert yoy_col == "20250331"
    assert round(float(yoy), 2) == 10.0
    assert liability_col == "20250331"
    assert liability_ratio == 45.0


def test_build_candidate_universe_marks_report_industry_fallback_as_theme_match(monkeypatch) -> None:
    report_df = pd.DataFrame(
        [
            {
                "ticker": "600001",
                "company_name": "科技样本",
                "industry_name": "半导体",
                "revenue_yoy": 20.0,
                "gross_margin": 40.0,
                "industry_gross_margin_median": 30.0,
                "operating_cashflow_ps": 0.5,
            }
        ]
    )

    def fake_fetch_board_candidates(kind: str, keywords, pause_sec: float):
        return pd.DataFrame(), {"matched_boards": [], "failed_boards": [], "error": "mocked"}

    monkeypatch.setattr(
        "stock_ana.data.cn_high_tech_selector.fetch_board_candidates",
        fake_fetch_board_candidates,
    )

    candidate_df, _ = build_candidate_universe(report_df, ScreenConfig())

    assert len(candidate_df) == 1
    assert candidate_df.loc[0, "candidate_sources"] == "report_industry"
    assert candidate_df.loc[0, "matched_industry_boards"] == "半导体"
    assert int(candidate_df.loc[0, "industry_match_count"]) == 1
