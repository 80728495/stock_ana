"""
AI 策略代码审查模块
==================

将回测图表 + 策略源码发送给 AI 模型，获取代码改进建议。

支持两种后端:
  1. Gemini API 直连 (默认) — 使用 .env 中的 GEMINI_API_KEY
     模型: gemini-2.5-flash / gemini-2.5-pro
  2. Antigravity 代理 — 通过本地反代访问 Claude Opus 4.6
     需要先启动 Antigravity Manager，监听 127.0.0.1:8045

使用示例:
    from stock_ana.utils.ai_code_reviewer import AICodeReviewer

    reviewer = AICodeReviewer()                         # Gemini 直连
    reviewer = AICodeReviewer(backend="antigravity")    # Claude via 代理
    result = reviewer.review_triangle_strategy()
"""

from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path

import httpx
from loguru import logger

from stock_ana.config import OUTPUT_DIR, PROJECT_ROOT

# ──────── 常量 ────────

# Antigravity 代理配置
ANTIGRAVITY_BASE_URL = "http://127.0.0.1:8045"
ANTIGRAVITY_API_KEY = "sk-antigravity"
ANTIGRAVITY_MODEL = "claude-opus-4-6"

# Gemini 直连配置
GEMINI_MODEL_DEFAULT = "gemini-2.5-flash"

# 速率限制
MIN_REQUEST_INTERVAL_SEC = 15       # 两次请求最小间隔（秒）
MAX_RETRIES = 3                     # 最大重试次数
BACKOFF_BASE_SEC = 30               # 退避基础时间（秒）
REQUEST_TIMEOUT_SEC = 600           # 单次请求超时（秒）— 长文本+图片需要较长处理
MAX_IMAGES_PER_REQUEST = 3          # 单次请求最多附带图片数（图片越多越耗时）

# 图表目录
BACKTEST_CHART_DIR = PROJECT_ROOT / "data" / "backtest_charts"

# 策略源码文件
STRATEGY_FILES = {
    "triangle": PROJECT_ROOT / "src" / "stock_ana" / "strategy_triangle.py",
    "base": PROJECT_ROOT / "src" / "stock_ana" / "strategy_base.py",
    "vcp": PROJECT_ROOT / "src" / "stock_ana" / "strategy_vcp.py",
    "screener": PROJECT_ROOT / "src" / "stock_ana" / "screener.py",
}

# 审查输出目录
REVIEW_OUTPUT_DIR = OUTPUT_DIR / "ai_reviews"

# ──────── Prompt ────────

_REVIEW_PROMPT = """\
你是一位专业的量化策略研究员和 Python 开发者。

## 任务
我正在开发股票技术分析策略（主要面向纳斯达克 100 成分股）。
请分析以下回测结果图表和策略源代码，给出具体的代码修改建议来提升策略表现。

## 当前回测表现
{backtest_summary}

## 策略源代码

{code_blocks}

## 附带的图表
以下是回测生成的 K 线图，每张图标注了信号触发日（红圈/红虚线）以及后续收益。
请仔细观察:
- 信号触发的位置是否合理
- 是否有明显的假信号模式
- 图表中的辅助线（上轨/下轨/均线）是否有效

## 要求
1. **分析当前策略的主要缺陷**（基于图表中的失败案例）
2. **给出 3-5 个具体的代码修改建议**，每个建议包含:
   - 问题描述
   - 修改思路
   - 具体的代码 diff（可直接应用的 Python 代码片段）
3. **预期影响**: 每个修改预计对胜率/收益的影响
4. 修改必须保持函数签名和返回格式的兼容性

请用中文回答。代码注释可以中英文混合。
"""


# ──────── 工具函数 ────────

def _encode_image_base64(path: Path) -> str:
    """读取图片并编码为 base64 字符串。"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _collect_chart_images(
    chart_dir: Path,
    strategy: str,
    max_images: int = MAX_IMAGES_PER_REQUEST,
) -> list[Path]:
    """
    收集指定策略的回测图表。

    优先选取失败案例（文件名中可能带 negative return），
    其次选取成功案例，确保正反样本均衡。
    """
    strat_dir = chart_dir / strategy
    if not strat_dir.exists():
        logger.warning(f"图表目录不存在: {strat_dir}")
        return []

    all_pngs = sorted(strat_dir.glob("*.png"))
    if not all_pngs:
        logger.warning(f"未找到 {strategy} 图表")
        return []

    # 如果图片不多，全部选取
    if len(all_pngs) <= max_images:
        return all_pngs

    # 否则均匀采样
    step = len(all_pngs) / max_images
    selected = [all_pngs[int(i * step)] for i in range(max_images)]
    return selected


def _read_source_code(strategy: str) -> str:
    """读取策略相关源码，返回格式化后的代码块。"""
    blocks = []

    # 主策略文件
    main_file = STRATEGY_FILES.get(strategy)
    if main_file and main_file.exists():
        code = main_file.read_text(encoding="utf-8")
        blocks.append(f"### {main_file.name}\n```python\n{code}\n```")

    # 始终包含基础模块
    base_file = STRATEGY_FILES.get("base")
    if base_file and base_file.exists() and strategy != "base":
        code = base_file.read_text(encoding="utf-8")
        blocks.append(f"### {base_file.name}\n```python\n{code}\n```")

    return "\n\n".join(blocks)


# ──────── API 客户端 ────────

class _RateLimiter:
    """简单速率限制器，确保请求间隔不低于阈值。"""

    def __init__(self, min_interval: float = MIN_REQUEST_INTERVAL_SEC):
        """Store the minimum spacing between outbound review requests."""
        self._min_interval = min_interval
        self._last_request_time: float = 0

    def wait(self):
        """阻塞等待直到可以发送下一个请求。"""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            sleep_time = self._min_interval - elapsed
            logger.debug(f"速率限制: 等待 {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()


class AICodeReviewer:
    """
    AI 策略代码审查器。

    Args:
        backend: "gemini" (直连) 或 "antigravity" (代理 Claude)
        model: 模型名称，默认根据 backend 自动选择
        api_key: API 密钥，默认从环境变量读取
        base_url: Antigravity 代理地址
        min_interval: 请求最小间隔（秒）
    """

    def __init__(
        self,
        backend: str = "gemini",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        min_interval: float = MIN_REQUEST_INTERVAL_SEC,
    ):
        """Configure the reviewer backend, credentials, and request pacing."""
        self.backend = backend.lower()
        self._rate_limiter = _RateLimiter(min_interval)

        if self.backend == "antigravity":
            self.model = model or ANTIGRAVITY_MODEL
            self.api_key = api_key or ANTIGRAVITY_API_KEY
            self.base_url = (base_url or ANTIGRAVITY_BASE_URL).rstrip("/")
            logger.info(
                f"AI Reviewer: Antigravity 代理模式 "
                f"(model={self.model}, url={self.base_url})"
            )
        elif self.backend == "gemini":
            self.model = model or GEMINI_MODEL_DEFAULT
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
            if not self.api_key:
                raise ValueError(
                    "Gemini API Key 未设置。"
                    "请在 .env 中设置 GEMINI_API_KEY 或传入 api_key 参数"
                )
            self.base_url = ""  # 使用 google-genai SDK
            logger.info(f"AI Reviewer: Gemini 直连模式 (model={self.model})")
        else:
            raise ValueError(f"不支持的 backend: {backend}，可选 'gemini' / 'antigravity'")

    # ─── Antigravity (OpenAI 兼容 API) ───

    def _call_antigravity(
        self,
        prompt: str,
        images: list[Path] | None = None,
    ) -> str:
        """通过 Antigravity 代理调用 Claude/Gemini (OpenAI 兼容协议)。"""
        content: list[dict] = []

        # 文本
        content.append({"type": "text", "text": prompt})

        # 图片 (base64)
        if images:
            for img_path in images:
                b64 = _encode_image_base64(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                    },
                })

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 8192,
            "temperature": 0.3,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/v1/chat/completions"

        for attempt in range(1, MAX_RETRIES + 1):
            self._rate_limiter.wait()
            try:
                logger.info(
                    f"[Antigravity] 发送请求 (尝试 {attempt}/{MAX_RETRIES}, "
                    f"images={len(images) if images else 0})"
                )
                with httpx.Client(timeout=REQUEST_TIMEOUT_SEC) as client:
                    resp = client.post(url, json=payload, headers=headers)

                if resp.status_code == 200:
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"]
                    logger.success(f"[Antigravity] 响应成功 ({len(text)} 字符)")
                    return text

                elif resp.status_code == 429:
                    # 速率限制 — 退避重试
                    wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                    logger.warning(
                        f"[Antigravity] 429 速率限制，等待 {wait}s 后重试..."
                    )
                    time.sleep(wait)
                    continue

                elif resp.status_code == 529:
                    # API 过载
                    wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                    logger.warning(
                        f"[Antigravity] 529 API 过载，等待 {wait}s 后重试..."
                    )
                    time.sleep(wait)
                    continue

                else:
                    error_text = resp.text[:500]
                    logger.error(
                        f"[Antigravity] HTTP {resp.status_code}: {error_text}"
                    )
                    if attempt < MAX_RETRIES:
                        time.sleep(BACKOFF_BASE_SEC)
                        continue
                    raise RuntimeError(
                        f"Antigravity API 错误 {resp.status_code}: {error_text}"
                    )

            except httpx.ConnectError:
                logger.error(
                    f"[Antigravity] 无法连接 {self.base_url}，"
                    "请确认 Antigravity Manager 已启动"
                )
                raise RuntimeError(
                    f"无法连接 Antigravity 代理 ({self.base_url})。\n"
                    "请先启动 Antigravity Manager 并在 API Proxy 页面开启服务。"
                )
            except httpx.TimeoutException:
                logger.warning(
                    f"[Antigravity] 请求超时 ({REQUEST_TIMEOUT_SEC}s)，重试..."
                )
                if attempt < MAX_RETRIES:
                    continue
                raise

        raise RuntimeError(f"Antigravity API 在 {MAX_RETRIES} 次重试后仍然失败")

    # ─── Gemini 直连 ───

    def _call_gemini(
        self,
        prompt: str,
        images: list[Path] | None = None,
    ) -> str:
        """通过 google-genai SDK 直接调用 Gemini 模型。"""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        # 构建多模态内容
        contents: list = []
        if images:
            for img_path in images:
                img_bytes = img_path.read_bytes()
                contents.append(
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                )
        contents.append(prompt)

        config = types.GenerateContentConfig(
            max_output_tokens=8192,
            temperature=0.3,
        )

        for attempt in range(1, MAX_RETRIES + 1):
            self._rate_limiter.wait()
            try:
                logger.info(
                    f"[Gemini] 发送请求 (尝试 {attempt}/{MAX_RETRIES}, "
                    f"model={self.model}, images={len(images) if images else 0})"
                )
                response = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                text = response.text
                if text:
                    logger.success(f"[Gemini] 响应成功 ({len(text)} 字符)")
                    return text
                else:
                    logger.warning("[Gemini] 响应为空，重试...")
                    if attempt < MAX_RETRIES:
                        time.sleep(BACKOFF_BASE_SEC)
                        continue
                    raise RuntimeError("Gemini 返回空响应")

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                    logger.warning(
                        f"[Gemini] 速率限制 (429)，等待 {wait}s 后重试..."
                    )
                    time.sleep(wait)
                    if attempt < MAX_RETRIES:
                        continue
                raise RuntimeError(f"Gemini API 错误: {e}") from e

        raise RuntimeError(f"Gemini API 在 {MAX_RETRIES} 次重试后仍然失败")

    # ─── 统一调用接口 ───

    def _call(self, prompt: str, images: list[Path] | None = None) -> str:
        """根据 backend 选择调用方式。"""
        if self.backend == "antigravity":
            return self._call_antigravity(prompt, images)
        else:
            return self._call_gemini(prompt, images)

    # ─── Antigravity 健康检查 ───

    def check_antigravity(self) -> bool:
        """检查 Antigravity 代理是否可用。"""
        if self.backend != "antigravity":
            return True
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(f"{self.base_url}/v1/models")
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m.get("id", "") for m in data.get("data", [])]
                    claude_models = [m for m in models if "claude" in m.lower()]
                    logger.info(
                        f"Antigravity 可用，{len(models)} 个模型，"
                        f"Claude 模型: {claude_models[:5]}"
                    )
                    return True
                else:
                    logger.warning(f"Antigravity 返回 {resp.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Antigravity 不可用: {e}")
            return False

    # ═══════════════════════════════════════════════════════
    # 策略审查入口
    # ═══════════════════════════════════════════════════════

    def review_triangle_strategy(
        self,
        chart_dir: Path | None = None,
        backtest_summary: str = "",
        max_images: int = MAX_IMAGES_PER_REQUEST,
        save: bool = True,
    ) -> str:
        """
        审查三角形策略代码。

        将回测图表和策略源码一起发送给 AI，获取改进建议。

        Args:
            chart_dir: 回测图表目录，默认 data/backtest_charts
            backtest_summary: 回测结果摘要文本
            max_images: 最多附带几张图表
            save: 是否将结果保存到文件

        Returns:
            AI 返回的审查意见文本
        """
        chart_dir = chart_dir or BACKTEST_CHART_DIR

        # 收集图表
        images = _collect_chart_images(chart_dir, "triangle", max_images)
        logger.info(f"收集到 {len(images)} 张三角形策略图表")

        # 读取源码
        code_blocks = _read_source_code("triangle")

        # 构建 prompt
        if not backtest_summary:
            backtest_summary = (
                "Triangle 策略最近回测 (3年 NDX100 滚动): "
                "143 信号, 55% 胜率, 21日平均 +0.67%, Alpha -0.76%"
            )

        prompt = _REVIEW_PROMPT.format(
            backtest_summary=backtest_summary,
            code_blocks=code_blocks,
        )

        logger.info(
            f"发送审查请求: prompt {len(prompt)} 字符, "
            f"{len(images)} 张图表 → {self.backend}/{self.model}"
        )

        result = self._call(prompt, images if images else None)

        if save:
            self._save_review(result, "triangle")

        return result

    def review_vcp_strategy(
        self,
        chart_dir: Path | None = None,
        backtest_summary: str = "",
        max_images: int = MAX_IMAGES_PER_REQUEST,
        save: bool = True,
    ) -> str:
        """审查 VCP 策略代码。"""
        chart_dir = chart_dir or BACKTEST_CHART_DIR

        images = _collect_chart_images(chart_dir, "vcp", max_images)
        logger.info(f"收集到 {len(images)} 张 VCP 策略图表")

        code_blocks = _read_source_code("vcp")

        if not backtest_summary:
            backtest_summary = (
                "VCP 策略最近回测 (3年 NDX100 滚动): "
                "11 信号, 73% 胜率, 21日平均 +1.97%, Alpha +0.54%"
            )

        prompt = _REVIEW_PROMPT.format(
            backtest_summary=backtest_summary,
            code_blocks=code_blocks,
        )

        result = self._call(prompt, images if images else None)

        if save:
            self._save_review(result, "vcp")

        return result

    def review_strategy(
        self,
        strategy: str,
        chart_dir: Path | None = None,
        backtest_summary: str = "",
        extra_code_files: list[Path] | None = None,
        max_images: int = MAX_IMAGES_PER_REQUEST,
        save: bool = True,
    ) -> str:
        """
        通用策略审查。

        Args:
            strategy: 策略名称 ("triangle" / "vcp" / "vegas")
            chart_dir: 图表目录
            backtest_summary: 回测摘要
            extra_code_files: 额外需要发送的代码文件
            max_images: 最多图片数
            save: 是否保存

        Returns:
            AI 审查意见
        """
        chart_dir = chart_dir or BACKTEST_CHART_DIR

        images = _collect_chart_images(chart_dir, strategy, max_images)
        code_blocks = _read_source_code(strategy)

        # 额外代码文件
        if extra_code_files:
            for fpath in extra_code_files:
                if fpath.exists():
                    code = fpath.read_text(encoding="utf-8")
                    code_blocks += f"\n\n### {fpath.name}\n```python\n{code}\n```"

        prompt = _REVIEW_PROMPT.format(
            backtest_summary=backtest_summary or f"策略: {strategy}",
            code_blocks=code_blocks,
        )

        result = self._call(prompt, images if images else None)

        if save:
            self._save_review(result, strategy)

        return result

    # ─── 保存结果 ───

    def _save_review(self, text: str, strategy: str) -> Path:
        """将审查结果保存为 Markdown 文件。"""
        REVIEW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"review_{strategy}_{self.backend}_{ts}.md"
        path = REVIEW_OUTPUT_DIR / filename

        header = (
            f"# AI 策略审查: {strategy}\n\n"
            f"- **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- **后端**: {self.backend}\n"
            f"- **模型**: {self.model}\n\n"
            f"---\n\n"
        )
        path.write_text(header + text, encoding="utf-8")
        logger.success(f"审查报告已保存: {path}")
        return path


# ──────── 便捷函数 ────────

def review_triangle(
    backend: str = "gemini",
    model: str | None = None,
    **kwargs,
) -> str:
    """
    便捷函数: 审查三角形策略。

    Args:
        backend: "gemini" 或 "antigravity"
        model: 模型名称
        **kwargs: 传递给 review_triangle_strategy 的额外参数

    Returns:
        AI 审查意见
    """
    reviewer = AICodeReviewer(backend=backend, model=model)
    return reviewer.review_triangle_strategy(**kwargs)


def review_vcp(
    backend: str = "gemini",
    model: str | None = None,
    **kwargs,
) -> str:
    """便捷函数: 审查 VCP 策略。"""
    reviewer = AICodeReviewer(backend=backend, model=model)
    return reviewer.review_vcp_strategy(**kwargs)
