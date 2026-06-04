"""
深度分析 OB 特征分桶，寻找高胜率组合。
目标：宁缺毋滥，信号少但准确。
"""
import pandas as pd
import numpy as np

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)

ob = pd.read_csv("data/output/backtest_smc_ob/ob_backtest_US_tech.csv")
touch = pd.read_csv("data/output/backtest_smc_ob/touch_backtest_US_tech.csv")

features = ["ob_width_pct", "body_ratio", "vol_ratio", "trend_before_5d", "ob_atr_ratio", "percentage"]


def bucket_analysis(df, target_col, direction, label):
    """对每个特征分 5 桶，看胜率分布。"""
    sub = df[df["direction"] == direction].copy()
    print(f"\n{'='*80}")
    print(f"  {label} — 样本 {len(sub)}, 基线胜率 {sub[target_col].mean()*100:.1f}%")
    print(f"{'='*80}")

    for col in features + (["days_to_touch"] if "days_to_touch" in sub.columns else []):
        if col not in sub.columns:
            continue
        sub["_bucket"] = pd.qcut(sub[col], q=5, duplicates="drop")
        stats = sub.groupby("_bucket", observed=True)[target_col].agg(["count", "mean"])
        stats.columns = ["n", "胜率"]
        stats["胜率"] = (stats["胜率"] * 100).round(1)
        print(f"\n  {col}:")
        for idx, row in stats.iterrows():
            bar = "█" * int(row["胜率"] / 2)
            print(f"    {str(idx):<35} n={int(row['n']):>4}  胜率={row['胜率']:>5.1f}%  {bar}")


# 方向 1&2
for d, lbl in [(1, "看涨OB生成后"), (-1, "看跌OB生成后")]:
    bucket_analysis(ob, "success", d, lbl)

# 方向 3&4
for d, lbl in [(1, "看涨OB Touch后"), (-1, "看跌OB Touch后")]:
    bucket_analysis(touch, "touch_success", d, lbl)

# ── 组合筛选 grid search ──────────────────────────────────────────────────────
print(f"\n\n{'='*80}")
print("  组合筛选 Grid Search — 寻找高胜率组合")
print(f"{'='*80}")

def grid_search(df, target_col, direction, label):
    sub = df[df["direction"] == direction].copy()
    base_wr = sub[target_col].mean()
    base_n = len(sub)

    # 定义每个特征的候选阈值（分位数）
    thresholds = {}
    for col in features + (["days_to_touch"] if "days_to_touch" in sub.columns else []):
        if col not in sub.columns:
            continue
        qs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        thresholds[col] = [(q, sub[col].quantile(q)) for q in qs]

    # 两两组合搜索
    results = []
    cols = list(thresholds.keys())
    for i, col1 in enumerate(cols):
        for q1, t1 in thresholds[col1]:
            for dir1 in ["ge", "le"]:
                m1 = sub[col1] >= t1 if dir1 == "ge" else sub[col1] <= t1
                n1 = m1.sum()
                if n1 < max(50, base_n * 0.1):
                    continue
                wr1 = sub.loc[m1, target_col].mean()
                if wr1 <= base_wr + 0.03:
                    continue

                # 单条件结果
                results.append({
                    "条件": f"{col1} {'≥' if dir1=='ge' else '≤'} {t1:.3f}",
                    "样本": int(n1),
                    "胜率": round(wr1 * 100, 1),
                    "保留率": round(n1/base_n*100, 1),
                })

                # 加第二个条件
                for j, col2 in enumerate(cols):
                    if j <= i:
                        continue
                    for q2, t2 in thresholds[col2]:
                        for dir2 in ["ge", "le"]:
                            m2 = sub[col2] >= t2 if dir2 == "ge" else sub[col2] <= t2
                            combined = m1 & m2
                            nc = combined.sum()
                            if nc < max(30, base_n * 0.05):
                                continue
                            wrc = sub.loc[combined, target_col].mean()
                            if wrc <= base_wr + 0.05:
                                continue
                            results.append({
                                "条件": f"{col1} {'≥' if dir1=='ge' else '≤'} {t1:.3f} + {col2} {'≥' if dir2=='ge' else '≤'} {t2:.3f}",
                                "样本": int(nc),
                                "胜率": round(wrc * 100, 1),
                                "保留率": round(nc/base_n*100, 1),
                            })

    if results:
        rdf = pd.DataFrame(results).sort_values("胜率", ascending=False).head(20)
        print(f"\n  {label} (基线: n={base_n}, 胜率={base_wr*100:.1f}%)")
        print(f"  {'条件':<70} {'样本':>5} {'胜率':>6} {'保留':>6}")
        print(f"  {'─'*70} {'─'*5} {'─'*6} {'─'*6}")
        for _, r in rdf.iterrows():
            print(f"  {r['条件']:<70} {r['样本']:>5} {r['胜率']:>5.1f}% {r['保留率']:>5.1f}%")


for d, lbl in [(1, "看涨OB"), (-1, "看跌OB")]:
    grid_search(ob, "success", d, lbl)

for d, lbl in [(1, "看涨Touch"), (-1, "看跌Touch")]:
    grid_search(touch, "touch_success", d, lbl)
