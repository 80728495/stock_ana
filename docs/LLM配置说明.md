# 大语言模型接入配置说明

本文档记录本项目当前使用的大语言模型接入方式。项目内原先通过旧 Coding Plan 调用的大模型，统一迁移到 DeepSeek 官方 OpenAI-compatible API。

## 当前标准配置

| 项目 | 值 |
|------|----|
| 服务商 | DeepSeek 官方 API |
| Base URL | `https://api.deepseek.com` |
| Chat Completions | `https://api.deepseek.com/chat/completions` |
| 默认模型 | `deepseek-v4-pro` |
| API Key 环境变量 | `DEEPSEEK_API_KEY` |
| 兼容协议 | OpenAI-compatible Chat Completions |

## 项目环境变量

根目录 `.env` / `.env.example` 使用：

```ini
DEEPSEEK_API_KEY=your_key_here
STOCK_ANA_LLM_MODEL=deepseek-v4-pro
STOCK_ANA_LLM_BASE_URL=https://api.deepseek.com
STOCK_ANA_LLM_THINKING=enabled
STOCK_ANA_LLM_REASONING_EFFORT=high
```

`youtube_trans/.env` / `youtube_trans/.env.example` 使用：

```ini
DEEPSEEK_API_KEY=your_key_here
RHINO_LLM_MODEL=deepseek-v4-pro
RHINO_LLM_BASE_URL=https://api.deepseek.com
RHINO_LLM_THINKING=enabled
RHINO_LLM_REASONING_EFFORT=high
```

不要把真实 API key 写入代码或文档；只写入本机 `.env`，并确认 `.env` 被 Git 忽略。

## 已迁移模块

| 模块 | 用途 | 配置前缀 |
|------|------|----------|
| `src/stock_ana/data/labeler.py` | 美股 SEC 行业子标签分类 | `STOCK_ANA_LLM_*` |
| `youtube_trans/rhino_finance_daily.py` | RhinoFinance 视频转写后的结构化摘要 | `RHINO_LLM_*` |
| `youtube_trans/rhino_finance_test.py` | RhinoFinance 测试脚本 | `RHINO_LLM_*` |

Vegas 扫描、周报等分析流程默认使用 Codex/`gpt-5.5`，通过本地
CLIProxyAPI 调用；如需临时回退 Gemini，可传 `--llm-backend gemini`。

## Codex / GPT-5.5 本地代理配置

先启动本地 CLIProxyAPI，并完成 Codex OAuth 登录：

```powershell
cd C:\Users\shawn\stock_ana\CLIProxyAPI
.\scripts\start-local.ps1
.\scripts\codex-login.ps1
```

项目根目录 `.env` 可选配置：

```ini
STOCK_ANA_CODEX_BASE_URL=http://127.0.0.1:8317
STOCK_ANA_CODEX_API_KEY=your_cli_proxy_api_key_here
STOCK_ANA_CODEX_MODEL=gpt-5.5
STOCK_ANA_CODEX_REASONING_EFFORT=high
STOCK_ANA_CODEX_WEB_SEARCH=required
STOCK_ANA_CODEX_TIMEOUT_SEC=1200
STOCK_ANA_CODEX_BATCH_SIZE=3
```

如果未设置 `STOCK_ANA_CODEX_API_KEY`，代码会尝试读取
`CLIProxyAPI/config.local.yaml` 中的第一个 `api-keys`。

`STOCK_ANA_CODEX_REASONING_EFFORT=high` 表示使用 GPT-5.5 的高推理档位。
`STOCK_ANA_CODEX_WEB_SEARCH=required` 表示每次请求都会注入 web search 工具并强制使用。
可选值为 `off`、`auto`、`required`。
`STOCK_ANA_CODEX_TIMEOUT_SEC=1200` 表示单次 Codex 请求最多等待 20 分钟。
`STOCK_ANA_CODEX_BATCH_SIZE=3` 是 Codex 后端的兼容批大小配置。
默认扫描分析使用 `STOCK_ANA_SCAN_LLM_BATCH_SIZE=3` 控制分批。

每日 Vegas Mid 扫描默认：

```powershell
python vegas_mid_daily_scan.py --list tech --llm-backend codex
```

周线 Vegas Short 扫描默认：

```powershell
python weekly_vegas_short_notify.py --list combined --llm-backend codex
```

每周板块异动周报默认：

```powershell
python -m stock_ana.workflows.weekly_sector_report --skip-update --llm-backend codex
```

环境变量默认：

```ini
STOCK_ANA_SCAN_LLM_BACKEND=codex
STOCK_ANA_WEEKLY_LLM_BACKEND=codex
STOCK_ANA_WEEKLY_SECTOR_LLM_BACKEND=codex
STOCK_ANA_GEMINI_MODEL=gemini-3.1-pro
STOCK_ANA_SCAN_LLM_BATCH_SIZE=3
```

Gemini 默认使用 `gemini-3.1-pro`（网页端 Pro Extended/Pro 档位）。
当前 `gemini_webapi` 内置枚举尚未包含该模型名，因此项目通过
`model_header` 自定义模式传入。若 Google 网页端内部 header 变更导致
请求失败，可临时回退：

```ini
STOCK_ANA_GEMINI_MODEL=gemini-3.0-pro
```

临时回退 Gemini：

```powershell
python vegas_mid_daily_scan.py --list tech --llm-backend gemini
```

## Python SDK 示例

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("STOCK_ANA_LLM_BASE_URL", "https://api.deepseek.com"),
)

response = client.chat.completions.create(
    model=os.environ.get("STOCK_ANA_LLM_MODEL", "deepseek-v4-pro"),
    messages=[{"role": "user", "content": "你好"}],
    reasoning_effort=os.environ.get("STOCK_ANA_LLM_REASONING_EFFORT", "high"),
    extra_body={
        "thinking": {"type": os.environ.get("STOCK_ANA_LLM_THINKING", "enabled")},
    },
)

print(response.choices[0].message.content)
```

## HTTP 示例

```bash
curl https://api.deepseek.com/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  -d '{
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "你好"}],
    "thinking": {"type": "enabled"},
    "reasoning_effort": "high",
    "stream": false
  }'
```

## 旧配置清理

旧供应商的 endpoint、模型名和 API key 环境变量不再使用；新增或修改脚本时，只使用本文档上方的 DeepSeek 配置。
