# 大语言模型接入配置说明

本文档记录当前已验证可用的大语言模型接入方式，供其他项目直接参考配置。

---

## 一、当前使用的接入方案

### 火山引擎 Ark · Coding Plan

- **服务商**：字节跳动火山引擎
- **计费方式**：Coding Plan（包月订阅，按订阅额度使用，非按 token 计费）
- **控制台**：https://console.volcengine.com/ark
- **API Endpoint**：`https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions`
- **API Key**：`34081167-83fa-43c5-9c30-632e640fba9c`
- **协议**：OpenAI 兼容（`/chat/completions`），可直接使用 OpenAI SDK 或 httpx 调用

> ⚠️ 注意：普通按量付费接口的路径是 `/api/v3/`，Coding Plan 的路径是 `/api/coding/v3/`，两者不同，不能混用。

---

## 二、已验证可用的模型

### 2.1 MiniMax M2.5（主力模型，推荐）

| 参数 | 值 |
|------|-----|
| 模型名称（`model` 字段） | `minimax-m2.5` |
| Context 窗口 | 200K tokens |
| 最大输出 | 128K tokens |
| 工具调用（Function Calling） | 支持 |
| 流式输出 | 支持（不返回 stream usage） |

**curl 示例：**

```bash
curl https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 34081167-83fa-43c5-9c30-632e640fba9c" \
  -d '{
    "model": "minimax-m2.5",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": false
  }'
```

**Python 示例（openai SDK）：**

```python
from openai import OpenAI

client = OpenAI(
    api_key="34081167-83fa-43c5-9c30-632e640fba9c",
    base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
)

response = client.chat.completions.create(
    model="minimax-m2.5",
    messages=[{"role": "user", "content": "你好"}],
)
print(response.choices[0].message.content)
```

---

### 2.2 豆包系列（备用）

通过同一个 Coding Plan endpoint 可以访问豆包系列模型，`model` 字段填写对应的接入点 ID 或模型名称。

| 模型 | `model` 字段 | Context |
|------|-------------|---------|
| 豆包标准版 | `doubao-1-5-pro-32k` 或方舟接入点 ID | 32K |
| 豆包长文本版 | `doubao-1-5-pro-256k` 或方舟接入点 ID | 256K |

---

## 三、与 dayu-agent 项目集成

dayu-agent 项目位于 `/Users/wl/fa_gpt`，通过 `llm_models.json` 管理模型配置。

### 配置文件位置

```
workspace/config/llm_models.json   ← 运行时实际读取
dayu/config/llm_models.json        ← 源文件
```

修改源文件后必须同步：

```bash
cp /Users/wl/fa_gpt/dayu/config/llm_models.json \
   /Users/wl/fa_gpt/workspace/config/llm_models.json
```

### minimax-m2.5 在 llm_models.json 中的完整配置块

```json
"minimax-m2.5": {
  "runner_type": "openai_compatible",
  "name": "minimax-m2.5",
  "endpoint_url": "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions",
  "model": "minimax-m2.5",
  "headers": {
    "Authorization": "Bearer {{ARK_API_KEY}}",
    "Content-Type": "application/json"
  },
  "timeout": 3600,
  "stream_idle_timeout": 120.0,
  "stream_idle_heartbeat_sec": 10.0,
  "supports_stream": true,
  "supports_tool_calling": true,
  "supports_usage": true,
  "supports_stream_usage": false,
  "max_context_tokens": 200000,
  "max_output_tokens": 128000
}
```

`{{ARK_API_KEY}}` 在运行时自动替换为同名环境变量，需在 `~/.zshrc` 中设置：

```bash
export ARK_API_KEY="34081167-83fa-43c5-9c30-632e640fba9c"
```

### 指定模型运行

```bash
dayu-cli prompt "你的问题" --ticker AAOI --model-name minimax-m2.5
dayu-cli write --ticker AAOI --model-name minimax-m2.5
dayu-wechat run --base ./workspace --model-name minimax-m2.5
```

---

## 四、其他可接入的服务商（备查）

以下服务商均已内置在 `llm_models.json` 中，只需设置对应环境变量即可使用：

| 服务商 | 环境变量 | 推荐模型名 |
|--------|----------|-----------|
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat` / `deepseek-thinking` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` / `gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-opus` / `claude-sonnet` |
| Google | `GOOGLE_API_KEY` | `gemini-2.5-pro` |
| 阿里云百炼 | `DASHSCOPE_API_KEY` | `qwen-plus` / `qwen-max` |
