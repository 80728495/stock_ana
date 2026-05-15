"""
港股高科技与制造业选股工具。

从 hk_universe.csv（富途筛选后的流动性池）中剔除传统板块（银行/地产/博彩/
餐饮/传统能源/传统零售等），保留高科技、制造业、医疗、新能源等方向。

分类策略（双层关键词匹配，英文名优先）：
  1. 白名单关键词匹配 → 强制保留（高科技/制造/医疗/新能源）
  2. 黑名单关键词匹配 → 剔除（地产/银行/博彩/传统能源等）
  3. 无匹配 → 保留（保守策略，避免误杀）

用法:
    python -m stock_ana.data.hk_tech_selector
    python -m stock_ana.data.hk_tech_selector --universe data/hk_universe.csv
    python -m stock_ana.data.hk_tech_selector --min-cap-yi 50
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger

from stock_ana.config import DATA_DIR

# ─────────────────────── 输入/输出路径 ───────────────────────

HK_UNIVERSE_FILE = DATA_DIR / "hk_universe.csv"
HK_FULL_LIST_FILE = DATA_DIR / "hk_full_list.csv"

LISTS_DIR = Path(__file__).resolve().parents[3] / "data" / "lists"
OUT_CSV = LISTS_DIR / "hk_techman.csv"
OUT_MD = LISTS_DIR / "hk_techman.md"


# ─────────────────────── 关键词配置 ───────────────────────

# ▶ 白名单（英文）：命中即强制保留，即使也命中黑名单
# 覆盖：科技、芯片、软件、互联网、AI、工业制造、医疗、新能源、汽车、电信
_WHITELIST_EN: list[str] = [
    # 通用科技
    r"\bTECH",  # TECH, TECHTRONIC, TECHNOLOGY 等均覆盖
    r"\bDIGITAL\b", r"\bSOFTWARE\b", r"\bHARDWARE\b",
    r"\bINTERNET\b", r"\bCYBER\b", r"\bCLOUD\b", r"\bAI\b", r"\bROBOT",
    r"\bAUTOMATION\b", r"\bINTELLIG", r"\bPLATFORM\b",
    # 半导体/电子
    r"\bSEMICOND", r"\bCHIP\b", r"\bELECTRON", r"\bCIRCUIT\b", r"\bOPTIC",
    r"\bPHOTON", r"\bSENSOR\b", r"\bLASER\b", r"\bLED\b", r"\bPCB\b",
    r"\bDISPLAY\b", r"\bPANEL\b",
    r"\bFIBREOPT", r"\bFIBER OPT", r"\bOPTICAL FIBRE", r"\bOPTICAL FIBER",  # 光纤
    # 电信/网络
    r"\bTELECOM", r"\bTELESY", r"\b5G\b", r"\bNETWORK\b", r"\bSATELL",
    r"\bWIRELESS\b", r"\bFIBRE\b", r"\bFIBER\b",
    # 医疗/生物
    r"\bPHARMA\b", r"\bBIOTECH", r"\bBIO\b", r"\bMEDICAL\b", r"\bMEDICINE\b",
    r"\bHEALTH\b", r"\bDIAGNOS", r"\bGENOM", r"\bHOSPITAL\b", r"\bDENTAL\b",
    r"\bONCO\b", r"\bIMMUNO\b", r"\bVACCINE\b", r"\bCLINIC\b",
    # 新能源/储能
    r"\bSOLAR\b", r"\bWIND\b", r"\bBATTERY\b", r"\bHYDROGEN\b",
    r"\bCHARGING\b", r"\bPOWER STORAGE\b", r"\bENERGY STORAGE\b",
    r"\bNEW ENERGY\b", r"\bRENEWABLE\b", r"\bSMART ENERGY\b",
    # 汽车/出行
    r"\bELECTRIC VEHICLE", r"\bEV\b", r"\bAUTOMOTIVE\b", r"\bVEHICLE\b",
    r"\bAUTO COMPONENT", r"\bDRIVE\b",
    r"\bAUTO\b",  # CHERY AUTO 等轿车制造商
    r"\bSMART",  # SMARTHOME, SMARTBIZ 等智能家居/家电
    # 工业/机械/制造
    r"\bINDUSTRI", r"\bMANUFACT", r"\bMACHINERY\b", r"\bEQUIPMENT\b",
    r"\bROBOT\b", r"\bPRECISION\b", r"\bINSTRUMENT\b", r"\bENGINEER",
    r"\bCOMPONENT\b", r"\bPARTS\b", r"\bMECHATRON",
    r"\bHEAVY IND", r"\bHEAVY EQUIP", r"\bHEAVY MACH",  # 三一重工/潍柴类
    # 材料/化工（先进材料）
    r"\bMATERIAL\b", r"\bCOMPOSITE\b", r"\bCHEMICAL\b", r"\bSPECIALT",
    r"\bADVANCED MATERIAL", r"\bNANO\b",
    # 航空航天/国防
    r"\bAVIATION TECH", r"\bAEROSPACE\b", r"\bDEFENCE\b", r"\bDEFENSE\b",
    r"\bMISSILE\b", r"\bSATELL",
    # 物流科技/供应链
    r"\bSUPPLY CHAIN\b", r"\bWAREHOUS\b", r"\bDRONE\b",
]

# ▶ 黑名单（英文）：命中且未命中白名单则剔除
_BLOCKLIST_EN: list[str] = [
    # 银行/金融
    r"\bBANK\b", r"\bBANKING\b", r"\bFINANCIAL HOLD", r"\bFINANCE HOLD",
    r"\bINSURANCE\b", r"\bASSURANCE\b", r"\bFUND\b", r"\bTRUST\b",
    r"\bBROKER\b", r"\bSECURITIES\b", r"\bLEASING\b", r"\bCREDIT\b",
    r"\bPAYMENT\b",
    # 知名外资银行/保险缩写
    r"\bHSBC\b", r"\bSCB\b", r"STANDARD CHART", r"HANG SENG BANK",
    r"\bAIA\b", r"LIFE INSUR", r"\bPING AN\b",
    r"\bMANULIFE\b", r"\bBOC HONG KONG",
    # 传统能源/矿业
    r"\bSHENHUA\b", r"CHINA COAL", r"CHINA SHENHUA",
    r"\bCMOC\b", r"MOLYBDENUM", r"ALUMINUM CORP",
    # 地产
    r"\bPROPERTY\b", r"\bREAL ESTATE\b", r"\bREALTY\b", r"\bDEVELOPMENT HOLD",
    r"\bLAND DEV", r"\bHOUSING\b", r"\bREIT\b",
    # 博彩
    r"\bCASINO\b", r"\bGAMING\b", r"\bGAMBL\b", r"\bRESORTS?\b",
    r"\bENTERTAINMENT\b",
    # 餐饮/零售
    r"\bRESTAURANT\b", r"\bCAFE\b", r"\bFOOD & BEV", r"\bSOFT DRINK\b",
    r"\bBEER\b", r"\bWINE\b", r"\bBREWERY\b", r"\bDEPT STORE\b",
    r"\bDEPARTMENT STORE\b", r"\bSUPERMARKET\b", r"\bSHOPPING MALL\b",
    r"\bRETAIL HOLD", r"\bDISCOUNT\b",
    # 传统能源
    r"\bPETROLEUM\b", r"\bOIL\b", r"\bCOAL\b", r"\bMINING\b", r"\bMINE\b",
    r"\bGAS HOLD", r"\bGASHOLD", r"\bGAS DIST", r"\bSTEEL\b", r"\bIRON ORE\b",
    r"\bALUMINIUM\b",
    # 传统公用事业
    r"\bELECTRIC HOLD", r"\bPOWER HOLD", r"\bPOWER ASSETS\b",
    r"\bWATER HOLD", r"\bWATER SUPP", r"\bUTILITY\b",
    # 传统交通
    r"\bFERRY\b", r"\bSHIPPING\b", r"\bSHIP HOLD", r"\bSHIPBUILD", r"\bBUS HOLD",
    r"\bRAILWAY HOLD\b", r"\bAIRWAY\b", r"\bAIRLINE\b", r"\bAVIATION HOLD",
    r"\bHARBOUR\b", r"\bPORT HOLD",
    r"\bMTR\b", r"\bCOSCO SHIP",  # 港铁 / 中远海运
    # 传统媒体
    r"\bMEDIA HOLD", r"\bBROADCAST\b", r"\bNEWSPAPER\b", r"\bTV HOLD\b",
    r"\bPRINT\b",
    # 传统农业/食品
    r"\bAGRI\b", r"\bFARMING\b", r"\bSUGAR\b", r"\bTEA HOLD", r"\bFISH\b",
    r"\bTOBACCO\b", r"\bFOOD HOLD",
    # 纺织/服装
    r"\bTEXTILE\b", r"\bGARMENT\b", r"\bCLOTH\b", r"\bFASHION HOLD",
    # 珠宝
    r"\bJEWELL\b", r"\bJEWELLERY\b", r"\bJEWELRY\b", r"\bWATCH HOLD\b",
    r"\bGOLD\b",
    # 建筑/工程（传统）
    r"\bCONSTRUCTION HOLD", r"\bCIVIL ENG HOLD",
    # 酒店
    r"\bHOTEL\b", r"\bHOSPITALITY HOLD",
]

# ▶ 白名单（中文）：命中即强制保留
_WHITELIST_ZH: list[str] = [
    "科技", "技术", "软件", "数字", "智能", "人工智能", "云",
    "芯片", "半导体", "电子", "光电", "光学", "激光", "传感",
    "互联网", "网络", "通信", "电信", "5G", "卫星",
    "医疗", "医药", "生物", "基因", "健康", "制药", "诊断", "临床",
    "新能源", "光伏", "储能", "电池", "氢能", "充电",
    "锂",  # 赣锋锂业等电池材料供应商
    "光纤", "光缆",  # 长飞光纤等通信基础设施
    "电动", "智能驾驶",
    "工业", "制造", "机械", "设备", "精密", "仪器",
    "材料", "化工", "复合",
    "航空航天", "无人机",
    # 知名科技/制造业品牌（中文名无行业描述词时兜底）
    "腾讯", "阿里", "百度", "美团", "京东", "拼多多", "字节",
    "小米", "华为", "比亚迪", "宁德", "理想", "蔚来", "小鹏",
    "中芯", "华虹", "联发", "海康", "大疆",
    "联通", "电信",  # 中国联通/电信 → 电信运营商保留
    "中国移动",          # 中国移动 → 电信运营商保留
    # 制造业大品牌（机械/家电/汽车）
    "美的", "格力",   # Midea / Gree 制造业
    "吉利", "长安",   # 汽车制造商
    # 生物制药/CRO/进创生物科技
    "药明", "药简", "信达",   # 药明康德 / 药简平弥 / 信达生物
    "百济", "君实", "康方",   # 百济神州 / 君实生物 / 康方生物
    # 半导体/芯片公司具体名称
    "兆易", "向海",   # 兆易创新(GigaDevice) / 向海半导体
    # 知名重工/制造
    "潍柴", "三一", "中联", "徐工", "创科",  # 发动机/工程机械/电动工具
    "奇瑞",  # 奇瑞汽车 Chery Auto
    # 智能家居/电信基础设施
    "海尔智家", "智家",  # Haier Smarthome
    "铁塔",  # 中国铁塔 安装运营商导颂站站点
    # 互联网平台
    "网易", "哔哩哔哩", "快手", "渴望音乐",  # NetEase / Bilibili / Kuaishou / NetEase Music
    "创新",   # 将 创新 作为通用白名单（创新科技/兆易创新等）
]

# ▶ 黑名单（中文）：命中且未命中白名单则剔除
_BLOCKLIST_ZH: list[str] = [
    "银行", "保险", "证券", "融资", "基金", "信托", "租赁",
    "地产", "房地产", "房产", "置业", "物业", "楼宇",
    "博彩", "赌场", "娱乐城",
    "餐饮", "餐厅", "饮食", "酒楼", "茶餐厅", "快餐",
    "石油", "煤炭", "矿业", "矿产", "钢铁", "铝业",
    "电力控股", "水务", "燃气控股",
    "渡轮", "船务", "航运", "巴士", "出租车", "的士",
    "传媒", "广播", "报纸", "印刷",
    "渔业", "糖业", "烟草",
    "纺织", "服装", "珠宝", "钟表", "黄金零售",
    "酒店", "旅游",
    "零售控股", "百货",
    # 知名传统公司品牌（中文名无行业描述词时兜底）
    "汇丰", "渣打", "恒生银行", "东亚银行",  # 外资银行
    "国寿", "平安保", "人寿", "新华保", "太平洋保险",   # 保险集团
    "中国平安", "宏利", "友邦", "中国财险", "中国太保", "保诚",  # 保险品牌
    "神华", "华润",   # 煤/传统综合
    "农业银行", "建设银行", "工商银行", "中国银行",  # 四大行全称
    "中银香港",  # 银行子公司
    "钼业", "铝业", "铅业", "锌业", "铅锌",  # 有色金属矿
    "紫金矿", "宏桥",  # 黄金矿 / 铝制造
    "农夫山泉",  # 饮料，非科技
    "港铁", "铁路控股", "中远", "海通", "交易所",  # 公共交通/航运/券商/交易所
    "长实集团",  # 长和系地产
    "中煤", "海天味", "牧原",  # 煤炭/调味品/生猪养殖
    "黄金",  # 黄金矿业（黄金国际/黄金控股等）
]


# ─────────────────────── 分类逻辑 ───────────────────────

def _compile(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(patterns), re.IGNORECASE)


_WL_EN = _compile(_WHITELIST_EN)
_BL_EN = _compile(_BLOCKLIST_EN)
_WL_ZH = _compile(_WHITELIST_ZH)
_BL_ZH = _compile(_BLOCKLIST_ZH)


def classify_stock(name_en: str, name_zh: str) -> tuple[str, str]:
    """
    返回 (decision, reason)。
    decision: 'keep' | 'drop'
    reason:   说明命中了哪个规则
    """
    en = str(name_en).upper() if pd.notna(name_en) else ""
    zh = str(name_zh) if pd.notna(name_zh) else ""

    # 白名单优先（英文）
    m = _WL_EN.search(en)
    if m:
        return "keep", f"WL_EN:{m.group()}"

    # 白名单（中文）
    m = _WL_ZH.search(zh)
    if m:
        return "keep", f"WL_ZH:{m.group()}"

    # 黑名单（英文）
    m = _BL_EN.search(en)
    if m:
        return "drop", f"BL_EN:{m.group()}"

    # 黑名单（中文）
    m = _BL_ZH.search(zh)
    if m:
        return "drop", f"BL_ZH:{m.group()}"

    # 无匹配 → 保留（保守策略）
    return "keep", "no_match"


# ─────────────────────── 主函数 ───────────────────────

@dataclass
class SelectorConfig:
    universe_file: Path = HK_UNIVERSE_FILE
    full_list_file: Path = HK_FULL_LIST_FILE
    min_cap_yi: float = 0.0       # 最低市值（亿港元），0 = 不过滤
    drop_no_match: bool = False   # True = 无匹配的也剔除（严格模式）


def run_hk_tech_selector(config: SelectorConfig) -> pd.DataFrame:
    """
    执行选股，返回通过的 DataFrame（含 decision/reason 列）。
    同时写入 data/lists/hk_techman.csv 和 .md。
    """
    # ── 加载数据 ──────────────────────────────
    logger.info(f"加载 universe: {config.universe_file}")
    universe = pd.read_csv(config.universe_file, dtype={"code": str})

    # 标准化 code 列（去除 HK. 前缀，确保5位零填充纯数字）
    if "futu_code" in universe.columns and "code" not in universe.columns:
        universe["code"] = universe["futu_code"].str.replace(r"^HK\.", "", regex=True)
    elif "code" in universe.columns:
        universe["code"] = universe["code"].str.zfill(5)

    logger.info(f"Universe: {len(universe)} 只")

    # 可选：市值过滤
    if config.min_cap_yi > 0 and "market_cap_yi" in universe.columns:
        before = len(universe)
        universe = universe[
            pd.to_numeric(universe["market_cap_yi"], errors="coerce") >= config.min_cap_yi
        ].copy()
        logger.info(f"市值≥{config.min_cap_yi}亿过滤：{before} → {len(universe)} 只")

    # ── 补充英文名 ────────────────────────────
    if config.full_list_file.exists():
        logger.info(f"加载英文名: {config.full_list_file}")
        full = pd.read_csv(config.full_list_file, dtype={"code": str})
        full["code"] = full["code"].str.zfill(5)
        # 只取 code + name_en
        name_map = full[["code", "name_en"]].drop_duplicates("code").set_index("code")["name_en"]
        universe["name_en"] = universe["code"].map(name_map)
    else:
        logger.warning(f"未找到 {config.full_list_file}，将仅用中文名过滤")
        universe["name_en"] = ""

    # ── 分类 ─────────────────────────────────
    logger.info("执行双层关键词分类 ...")
    name_zh_col = "name_zh" if "name_zh" in universe.columns else "name"
    results = universe.apply(
        lambda r: classify_stock(r.get("name_en", ""), r.get(name_zh_col, "")),
        axis=1,
        result_type="expand",
    )
    universe["decision"] = results[0]
    universe["reason"] = results[1]

    if config.drop_no_match:
        universe.loc[universe["reason"] == "no_match", "decision"] = "drop"

    # ── 统计 ─────────────────────────────────
    keep = universe[universe["decision"] == "keep"].copy()
    drop = universe[universe["decision"] == "drop"].copy()
    no_match = keep[keep["reason"] == "no_match"]
    logger.info(
        f"分类结果：保留 {len(keep)} 只 / 剔除 {len(drop)} 只 "
        f"（其中 {len(no_match)} 只为 no_match 保守保留）"
    )

    _log_sample(drop, "剔除样本", 10)
    _log_sample(no_match, "no_match 样本（需人工复核）", 15)

    # ── 写入输出 ──────────────────────────────
    LISTS_DIR.mkdir(parents=True, exist_ok=True)
    keep.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"CSV → {OUT_CSV}")

    _write_md(keep, name_zh_col)
    logger.info(f"MD  → {OUT_MD}")

    return keep


def _log_sample(df: pd.DataFrame, label: str, n: int) -> None:
    if df.empty:
        return
    name_col = "name_zh" if "name_zh" in df.columns else ("name" if "name" in df.columns else None)
    cols = ["code"]
    if name_col:
        cols.append(name_col)
    if "name_en" in df.columns:
        cols.append("name_en")
    cols.append("reason")
    sample = df[cols].head(n).to_string(index=False)
    logger.info(f"\n{label}（前{n}条）:\n{sample}")


def _write_md(df: pd.DataFrame, name_zh_col: str) -> None:
    lines = [
        "# 港股高科技与制造业选股",
        "",
        f"共 **{len(df)}** 只 | 来源：hk_universe.csv | 策略：双层关键词过滤",
        "",
        "| 代码 | 名称 | 英文名 | 市值(亿HKD) | 均成交额(万/日) | 分类依据 |",
        "|------|------|--------|------------|----------------|---------|",
    ]
    cap_col = "market_cap_yi" if "market_cap_yi" in df.columns else None
    turn_col = "avg_turnover_20d" if "avg_turnover_20d" in df.columns else None

    for _, row in df.iterrows():
        code = row.get("code", "")
        name_zh = row.get(name_zh_col, row.get("name", ""))
        name_en = row.get("name_en", "")
        cap = f"{row[cap_col]:.0f}" if cap_col and pd.notna(row.get(cap_col)) else "-"
        turn = (
            f"{row[turn_col]/10000:.0f}"
            if turn_col and pd.notna(row.get(turn_col))
            else "-"
        )
        reason = row.get("reason", "")
        lines.append(f"| {code} | {name_zh} | {name_en} | {cap} | {turn} | {reason} |")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────── CLI ───────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="港股高科技与制造业选股")
    parser.add_argument(
        "--universe", type=Path, default=HK_UNIVERSE_FILE,
        help=f"输入 universe CSV（默认：{HK_UNIVERSE_FILE}）",
    )
    parser.add_argument(
        "--full-list", type=Path, default=HK_FULL_LIST_FILE,
        help=f"含 name_en 的全量列表（默认：{HK_FULL_LIST_FILE}）",
    )
    parser.add_argument(
        "--min-cap-yi", type=float, default=0.0,
        help="最低市值门槛（亿港元），0 = 不过滤（默认：0）",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="严格模式：无关键词命中的股票也剔除",
    )
    args = parser.parse_args()

    config = SelectorConfig(
        universe_file=args.universe,
        full_list_file=args.full_list,
        min_cap_yi=args.min_cap_yi,
        drop_no_match=args.strict,
    )
    keep = run_hk_tech_selector(config)
    logger.info(f"选股完成：{len(keep)} 只港股科技/制造标的")


if __name__ == "__main__":
    main()
