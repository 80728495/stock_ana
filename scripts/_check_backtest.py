"""快速检查回测数据的时间范围和单只股票明细。"""
import pandas as pd

# 1. 数据时间范围
for sym in ["NVDA", "AAPL", "TSLA", "AMD", "CRWD"]:
    df = pd.read_parquet(f"data/cache/us/{sym}.parquet")
    df.index = pd.to_datetime(df.index)
    print(f"{sym}: {df.index.min().date()} ~ {df.index.max().date()}, {len(df)} bars")

# 2. 回测结果年份分布
ob = pd.read_csv("data/output/backtest_smc_ob/ob_backtest_US_tech.csv")
ob["year"] = pd.to_datetime(ob["formed_date"]).dt.year
print("\n=== OB 生成年份分布 & 各年胜率 ===")
for d in [1, -1]:
    label = "看涨OB" if d == 1 else "看跌OB"
    sub = ob[ob["direction"] == d]
    stats = sub.groupby("year")["success"].agg(["count", "mean"])
    stats.columns = ["样本", "胜率"]
    stats["胜率"] = (stats["胜率"] * 100).round(1)
    print(f"\n{label}:")
    print(stats.to_string())

# 3. Touch 年份分布
touch = pd.read_csv("data/output/backtest_smc_ob/touch_backtest_US_tech.csv")
touch["year"] = pd.to_datetime(touch["touch_date"]).dt.year
print("\n=== Touch 年份分布 & 各年胜率 ===")
for d in [1, -1]:
    label = "看涨OB_touch" if d == 1 else "看跌OB_touch"
    sub = touch[touch["direction"] == d]
    stats = sub.groupby("year")["touch_success"].agg(["count", "mean"])
    stats.columns = ["样本", "胜率"]
    stats["胜率"] = (stats["胜率"] * 100).round(1)
    print(f"\n{label}:")
    print(stats.to_string())

# 4. 打印 CRWD 的全部 OB 回测明细
print("\n\n=== CRWD 全部 Touch 回测明细 ===")
crwd_touch = touch[touch["symbol"] == "CRWD"].sort_values("touch_date")
tcols = ["formed_date", "touch_date", "direction", "top", "bottom",
         "days_to_touch", "ret_end_pct", "touch_success", "reversal", "eventually_mitigated"]
print(crwd_touch[tcols].to_string(index=False))
