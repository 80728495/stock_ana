"""
港股高科技与制造业选股工具。

从 hk_universe.csv 出发，通过富途 OpenD 的行业板块 API 为每只股票打上
行业标签，然后保留属于科技/制造/医疗/新能源白名单行业的股票。

策略：
  - 调用 get_plate_stock 拉取 KEEP_PLATES 中每个行业的成分股映射
  - 只保留至少命中一个保留行业的股票；其余全部剔除
  - 行业来源：get_plate_list(HK, INDUSTRY) 的完整官方分类

用法:
    python -m stock_ana.data.hk_tech_selector
    python -m stock_ana.data.hk_tech_selector --min-cap-yi 50
    python -m stock_ana.data.hk_tech_selector --no-futu   # 用本地缓存，不连富途
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger

from stock_ana.config import DATA_DIR
from stock_ana.data.fetcher_futu import quote_context

# ─────────────────────── 路径 ───────────────────────

HK_UNIVERSE_FILE = DATA_DIR / "hk_universe.csv"
LISTS_DIR = Path(__file__).resolve().parents[3] / "data" / "lists"
INDUSTRY_CACHE_FILE = DATA_DIR / "cache" / "hk_industry_map.csv"
OUT_CSV = LISTS_DIR / "hk_techman.csv"
OUT_MD = LISTS_DIR / "hk_techman.md"

# ─────────────────────── 行业白名单 ───────────────────────
# 来源：get_plate_list(Market.HK, Plate.INDUSTRY) 返回的完整官方行业列表
# 只保留以下行业的股票，其余全部剔除

KEEP_PLATES: dict[str, str] = {
    # ── 科技/半导体/软件 ──
    "HK.LIST1013": "半导体",
    "HK.LIST1360": "半导体设备与材料",
    "HK.LIST1274": "电子零件",
    "HK.LIST1052": "消费电子产品",
    "HK.LIST1053": "电脑及周边器材",
    "HK.LIST1100": "应用软件",
    "HK.LIST1359": "游戏软件",
    "HK.LIST23360": "互动媒体及服务",
    "HK.LIST23363": "数码解决方案服务",
    "HK.LIST23364": "互联网服务及基础设施",
    "HK.LIST23362": "支付服务",
    # ── 电信/通信基础设施 ──
    "HK.LIST1054": "电讯服务",
    "HK.LIST1055": "消费性电讯设备",
    "HK.LIST1014": "卫星及无线通讯",
    "HK.LIST23851": "电讯网路基建设施",
    # ── 医疗/生物 ──
    "HK.LIST1012": "医疗设备及用品",
    "HK.LIST1050": "生物技术",
    "HK.LIST1067": "药品",
    "HK.LIST1086": "医疗及医学美容服务",
    "HK.LIST1284": "中医药",
    "HK.LIST1357": "药品分销",
    # ── 新能源/储能/核能 ──
    "HK.LIST1016": "非传统/可再生能源",
    "HK.LIST1033": "新能源物料",
    "HK.LIST1354": "能源储存装置",
    "HK.LIST1358": "核能",
    # ── 汽车/零件 ──
    "HK.LIST1040": "汽车",
    "HK.LIST1041": "汽车零件",
    "HK.LIST1017": "商业用车及货车",
    # ── 工业/机械/制造 ──
    "HK.LIST1025": "工业零件及器材",
    "HK.LIST1074": "重型机械",
    "HK.LIST23846": "轨道与列车设备",
    "HK.LIST1063": "航空航天与国防",
    # ── 化工/材料 ──
    "HK.LIST1046": "特殊化工用品",
    # ── 家电/智能家居 ──
    "HK.LIST1022": "家庭电器",
    # ── 环保 ──
    "HK.LIST1271": "环保工程",
}

# API 防限频：每个板块请求间隔（秒）
_PLATE_DELAY: float = 0.3


# ─────────────────────── 行业数据获取 ───────────────────────


def fetch_industry_map(use_cache: bool = False) -> pd.DataFrame:
    """
    通过富途 OpenD 拉取 KEEP_PLATES 中每个行业的成分股，
    返回 DataFrame，列：futu_code, industry（多行业时用顿号合并）。

    Args:
        use_cache: True 时读取本地缓存（data/cache/hk_industry_map.csv）
    """
    if use_cache and INDUSTRY_CACHE_FILE.exists():
        logger.info(f"读取行业缓存: {INDUSTRY_CACHE_FILE}")
        return pd.read_csv(INDUSTRY_CACHE_FILE, dtype={"futu_code": str})

    from futu import RET_OK

    logger.info(f"从富途拉取 {len(KEEP_PLATES)} 个行业板块成分股 ...")
    rows: list[dict] = []
    plates = list(KEEP_PLATES.items())
    failed: list[tuple[str, str]] = []  # 记录失败的板块，稍后重试

    with quote_context() as ctx:
        for i, (plate_code, plate_name) in enumerate(plates):
            # 富途限频：每30秒最多10次；每9次后等31秒
            if i > 0 and i % 9 == 0:
                logger.info(f"  限频等待 31 秒 ...")
                time.sleep(31)

            ret, data = ctx.get_plate_stock(plate_code)
            if ret != RET_OK:
                logger.warning(f"  {plate_name} ({plate_code}) 获取失败，稍后重试: {data}")
                failed.append((plate_code, plate_name))
                continue

            logger.info(f"  [{i+1}/{len(plates)}] {plate_name}: {len(data)} 只")
            for _, row in data.iterrows():
                rows.append({"futu_code": row["code"], "industry": plate_name})

            if i < len(plates) - 1:
                time.sleep(_PLATE_DELAY)

        # ── 重试失败的板块 ────────────────────────────
        if failed:
            logger.info(f"  重试 {len(failed)} 个失败板块，先等待 31 秒 ...")
            time.sleep(31)
            for j, (plate_code, plate_name) in enumerate(failed):
                if j > 0 and j % 9 == 0:
                    logger.info(f"  重试限频等待 31 秒 ...")
                    time.sleep(31)
                ret, data = ctx.get_plate_stock(plate_code)
                if ret != RET_OK:
                    logger.error(f"  重试失败: {plate_name} ({plate_code}): {data}")
                    continue
                logger.info(f"  [retry {j+1}/{len(failed)}] {plate_name}: {len(data)} 只")
                for _, row in data.iterrows():
                    rows.append({"futu_code": row["code"], "industry": plate_name})
                if j < len(failed) - 1:
                    time.sleep(_PLATE_DELAY)

    if not rows:
        raise RuntimeError("所有行业板块均获取失败，请检查 OpenD 连接和行情权限")

    raw = pd.DataFrame(rows)
    # 一只股票可能属于多个行业，合并为顿号分隔
    industry_map = (
        raw.groupby("futu_code")["industry"]
        .apply(lambda x: "、".join(sorted(set(x))))
        .reset_index()
    )
    logger.info(f"行业映射完成：{len(industry_map)} 只股票有行业标签")

    INDUSTRY_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    industry_map.to_csv(INDUSTRY_CACHE_FILE, index=False, encoding="utf-8-sig")
    logger.info(f"行业缓存已写入: {INDUSTRY_CACHE_FILE}")
    return industry_map


# ─────────────────────── 主函数 ───────────────────────


@dataclass
class SelectorConfig:
    universe_file: Path = HK_UNIVERSE_FILE
    min_cap_yi: float = 0.0    # 最低市值（亿港元），0 = 不过滤
    use_cache: bool = False    # True = 使用本地行业缓存，不连富途


def run_hk_tech_selector(config: SelectorConfig) -> pd.DataFrame:
    """
    执行选股，返回通过的 DataFrame（含 industry 列）。
    同时写入 data/lists/hk_techman.csv 和 .md。
    """
    # ── 加载 universe ─────────────────────────
    logger.info(f"加载 universe: {config.universe_file}")
    universe = pd.read_csv(config.universe_file, dtype={"code": str})
    universe["code"] = universe["code"].str.zfill(5)
    if "futu_code" not in universe.columns:
        universe["futu_code"] = "HK." + universe["code"]
    logger.info(f"Universe: {len(universe)} 只")

    if config.min_cap_yi > 0 and "market_cap_yi" in universe.columns:
        before = len(universe)
        universe = universe[
            pd.to_numeric(universe["market_cap_yi"], errors="coerce") >= config.min_cap_yi
        ].copy()
        logger.info(f"市值≥{config.min_cap_yi}亿过滤：{before} → {len(universe)} 只")

    # ── 拉取行业映射 ────────────────────────────
    industry_map = fetch_industry_map(use_cache=config.use_cache)

    # ── join ────────────────────────────────────
    result = universe.merge(industry_map, on="futu_code", how="left")
    keep = result[result["industry"].notna()].copy()
    drop = result[result["industry"].isna()].copy()

    logger.info(f"选股结果：保留 {len(keep)} 只 / 剔除 {len(drop)} 只（无行业命中）")

    name_col = "name_zh" if "name_zh" in drop.columns else "name"
    logger.info(
        f"\n剔除样本（前20条）:\n"
        + drop[["code", name_col]].head(20).to_string(index=False)
    )

    industry_counts = keep["industry"].str.split("、").explode().value_counts()
    logger.info(f"\n行业分布:\n{industry_counts.to_string()}")

    # ── 写输出 ────────────────────────────────
    LISTS_DIR.mkdir(parents=True, exist_ok=True)
    keep.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"CSV → {OUT_CSV}")

    _write_md(keep)
    logger.info(f"MD  → {OUT_MD}")

    return keep


def _write_md(df: pd.DataFrame) -> None:
    name_col = "name_zh" if "name_zh" in df.columns else "name"
    cap_col = "market_cap_yi" if "market_cap_yi" in df.columns else None
    turn_col = "avg_turnover_20d" if "avg_turnover_20d" in df.columns else None

    lines = [
        "# 港股高科技与制造业选股",
        "",
        f"共 **{len(df)}** 只 | 来源：hk_universe.csv | 策略：富途行业板块过滤",
        "",
        "| 代码 | 名称 | 市值(亿HKD) | 均成交额(万/日) | 行业 |",
        "|------|------|------------|----------------|------|",
    ]
    for _, row in df.iterrows():
        code = row.get("code", "")
        name = row.get(name_col, "")
        cap = f"{row[cap_col]:.0f}" if cap_col and pd.notna(row.get(cap_col)) else "-"
        turn = (
            f"{row[turn_col]/10000:.0f}"
            if turn_col and pd.notna(row.get(turn_col))
            else "-"
        )
        industry = row.get("industry", "")
        lines.append(f"| {code} | {name} | {cap} | {turn} | {industry} |")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────── CLI ───────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="港股高科技与制造业选股（富途行业板块过滤）")
    parser.add_argument(
        "--universe", type=Path, default=HK_UNIVERSE_FILE,
        help=f"输入 universe CSV（默认：{HK_UNIVERSE_FILE}）",
    )
    parser.add_argument(
        "--min-cap-yi", type=float, default=0.0,
        help="最低市值门槛（亿港元），0 = 不过滤",
    )
    parser.add_argument(
        "--no-futu", action="store_true",
        help="使用本地缓存，不连富途 OpenD（需已有 cache/hk_industry_map.csv）",
    )
    args = parser.parse_args()

    config = SelectorConfig(
        universe_file=args.universe,
        min_cap_yi=args.min_cap_yi,
        use_cache=args.no_futu,
    )
    keep = run_hk_tech_selector(config)
    logger.info(f"选股完成：{len(keep)} 只港股科技/制造标的")


if __name__ == "__main__":
    main()
