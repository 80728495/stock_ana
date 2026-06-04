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

Vegas 扫描、周报等现有 Gemini 分析流程仍使用 Gemini，不属于本次 LLM 供应商迁移范围。

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
