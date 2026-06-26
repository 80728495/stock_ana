# 研究代码与 GitHub 上传整理

生成日期：2026-06-23

## 当前仓库状态

- 当前分支：`feat/top-reversal-market-models`
- 远端仓库：`origin https://github.com/80728495/stock_ana.git`
- 仓库总大小约 `2.2G`，其中 `.venv` 约 `855M`，`data/` 约 `1.3G`，`equity_research/` 约 `51M`。
- `.gitignore` 当前已忽略 `.env`、`.venv/`、`data/output/*`、`data/cache`、`logs/`、`src/data/output/`、`/equity_research/`、`/analysis/`。
- 研究型 `equity_research/` 目录目前整体不入 Git，适合作为本地个股/专题研究档案。

## 研究分析目录

| 目录 | 大小 | 作用 | GitHub 建议 |
|---|---:|---|---|
| `equity_research/lite_model` | `1.1M` | LITE 文档抽取、抓取、业务预测、Excel 输出 | 私有保留；若上传只抽脚本和去敏摘要 |
| `equity_research/marvell_model` | `1.8M` | MRVL 产品维度、增长逻辑、2030 预测 | 私有保留；可抽通用建模脚本 |
| `equity_research/docn_model` | `1.7M` | DOCN 业务指标、增长测算、回归验证 | 私有保留；可抽模板 |
| `equity_research/ddog_model` | `7.7M` | DDOG 业务驱动、Agent/日志增长假设、Excel | 私有保留；含较多输出预览 |
| `equity_research/hua_hong_model` | `13M` | 华虹半导体产能、工艺平台、PB/情景预测 | 私有保留；报告和网页内容不建议公开 |
| `equity_research/ai_supply_chain_opportunities` | `25M` | AI 产业链三份报告、57 个来源抓取、19 只股票估值 | 私有保留；原始报告和网页全文不建议公开 |
| `equity_research/source_monitoring` | `688K` | 第三方网站抓取摸排、SeekingAlpha 公共抓取实验 | 可以整理成私有工具；公开前需去掉站点绕过意味的表述 |

## 研究流水线代码

这些是最近研究工作的核心代码，均在 `equity_research/` 下：

- 文档抽取：`extract_docx_reports.py`、`extract_docx_sources.py`
- 来源抓取：`scrape_*_sources.py`、`scrape_and_extract_sources.py`
- 估值抓取：`fetch_valuation_data.py`
- Excel/摘要生成：`build_*_workbook.mjs`、`build_*_outputs.py`
- 网站摸排：`probe_third_party_sites.py`
- SeekingAlpha 公共页抓取实验：`seekingalpha_public_scraper.mjs`

当前问题是这些脚本大多是“单项目脚本”，字段、ticker、输出表结构写死较多；适合本地研究复现，但还没抽象成一个干净的开源工具包。

## 不建议公开上传的内容

- 原始研究报告：`equity_research/**/inputs/*.docx`
- 抓取网页全文和 PDF：`equity_research/**/raw_html/`、`equity_research/**/source_texts/`
- 估值原始响应：`equity_research/**/valuation_raw/`
- Excel 输出、预览图：`equity_research/**/outputs/`
- 个人持仓和自选：`data/lists/holding.md`、`data/lists/watchlist.md`、`data/lists/futu_watchlist.md`、`data/lists/futu_watched_symbols.json`
- 个人账务、收益统计、税务导出、临时调试脚本和结果：`temp_scripts/`（被 git 忽略的一次性研究脚本）、`data/output/` 中相关导出结果
- 本地缓存、日志、大型研究输出：`data/cache/`、`data/output/`、`data/logs/`
- 本地凭据：`.env`、浏览器 cookie、任何 OpenD/飞书/Gemini 相关密钥。

## 可以上传的内容

如果是私有 GitHub 仓库：

- 主程序代码：`src/stock_ana/`
- 稳定命令入口和仓库守卫：`scripts/`
- 顶部研究入口和可复用研究模块：`src/stock_ana/research/top_reversal/`
- 自动化入口：`daily_update.py`、`full_refresh_pipeline.py`
- 测试：`tests/`
- 文档：`README.md`、`docs/`
- 通用配置：`pyproject.toml`、`.env.example`

如果是公开 GitHub 仓库：

- 建议只上传框架代码、接口定义、测试、文档和脱敏示例数据。
- 不建议上传个人持仓列表、交易记录、抓取网页全文、第三方报告原文。
- Futu、飞书、Gemini cookie 相关功能需要写成“用户自行配置凭据”的形式，避免示例里出现真实账户上下文。

## 当前 Git 状态注意点

- 当前有多处未提交修改，主要集中在顶部反转模型、持仓列表和 watchlist。
- `equity_research/` 整体被忽略，所以最近研究模型不会自动进入 commit。
- 根目录 `node_modules` 是符号链接，已补充忽略规则 `node_modules`，避免误入库。
- `src/data/output/ODFL_macd_cross.png` 是已跟踪文件，即使现在目录被忽略，它仍会继续显示修改；如果决定清理历史输出，需要单独 `git rm --cached`。

## 建议上传策略

1. 先把当前仓库当作私有研究仓库，不建议直接公开。
2. 若要公开，建议新建一个干净分支或新仓库，只保留 `src/`、`scripts/` 中通用入口、`tests/`、`docs/`、`pyproject.toml`。
3. 研究报告类内容单独放本地或私有对象存储，只在 Git 中保留“来源链接清单 + 摘要 + 可复现脚本”。
4. 后续可以把 `equity_research/` 中重复逻辑抽象成 `research_pipeline/`：
   - `docx_extract`
   - `source_scrape`
   - `fact_table`
   - `valuation_snapshot`
   - `workbook_export`
5. 对每只股票的项目，只保留 YAML/CSV 配置，而不是复制一套脚本。

## 结论

当前代码“适合上传到私有 GitHub 做版本管理”，但“还不适合直接公开”。如果目标是长期研究资产沉淀，下一步应该是把 `equity_research/` 的方法论代码抽象出来，原始报告、网页全文、Excel 输出继续留在本地或私有存储。
