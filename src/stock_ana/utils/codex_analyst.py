"""
Codex/GPT analysis client.

This module calls the local CLIProxyAPI service through its OpenAI-compatible
Responses API. It is intentionally small so existing Gemini prompts can be
reused by daily and weekly workflows.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests
import yaml
from loguru import logger

from stock_ana.config import PROJECT_ROOT

DEFAULT_CODEX_BASE_URL = "http://127.0.0.1:8317"
DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_CODEX_REASONING_EFFORT = "xhigh"
DEFAULT_CODEX_WEB_SEARCH = "required"
DEFAULT_TIMEOUT_SEC = 600
MAX_RETRIES = 2
BACKOFF_BASE_SEC = 20
_WEB_SEARCH_TOOL = {"type": "web_search_preview"}


def _normalize_base_url(raw: str) -> str:
    """Return a base URL without a trailing slash or /v1 suffix."""
    base = raw.strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base


def _read_local_cliproxy_key() -> str:
    """Read the first local CLIProxyAPI API key when no env var is provided."""
    config_path = PROJECT_ROOT / "CLIProxyAPI" / "config.local.yaml"
    if not config_path.exists():
        return ""
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.debug(f"读取 CLIProxyAPI 本地配置失败: {exc}")
        return ""

    keys = data.get("api-keys") or []
    if isinstance(keys, list) and keys:
        return str(keys[0]).strip()
    return ""


def get_codex_config() -> tuple[str, str, str]:
    """Return (base_url, api_key, model) for local Codex proxy calls."""
    base_url = (
        os.environ.get("STOCK_ANA_CODEX_BASE_URL")
        or os.environ.get("CODEX_PROXY_BASE_URL")
        or DEFAULT_CODEX_BASE_URL
    )
    api_key = (
        os.environ.get("STOCK_ANA_CODEX_API_KEY")
        or os.environ.get("CODEX_PROXY_API_KEY")
        or os.environ.get("CLI_PROXY_API_KEY")
        or _read_local_cliproxy_key()
    )
    model = (
        os.environ.get("STOCK_ANA_CODEX_MODEL")
        or os.environ.get("CODEX_PROXY_MODEL")
        or DEFAULT_CODEX_MODEL
    )
    return _normalize_base_url(base_url), api_key.strip(), model.strip()


def _extract_response_text(data: dict[str, Any]) -> str:
    """Extract assistant text from OpenAI Responses-compatible JSON."""
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    chunks: list[str] = []
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        chunks.append(text)
            text = item.get("text")
            if isinstance(text, str) and text:
                chunks.append(text)

    if chunks:
        return "\n".join(chunks).strip()
    return ""


def call_codex_prompt(
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    max_output_tokens: int = 8192,
    reasoning_effort: str | None = None,
    web_search: str | None = None,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> str:
    """Call local CLIProxyAPI/Codex with a text prompt and return plain text."""
    cfg_base_url, cfg_api_key, cfg_model = get_codex_config()
    resolved_base_url = _normalize_base_url(base_url or cfg_base_url)
    resolved_api_key = (api_key or cfg_api_key).strip()
    resolved_model = (model or cfg_model).strip()

    if not resolved_api_key:
        raise ValueError(
            "Codex proxy API key 未配置。请设置 STOCK_ANA_CODEX_API_KEY，"
            "或确认 CLIProxyAPI/config.local.yaml 存在 api-keys。"
        )

    input_payload: str | list[dict[str, str]]
    if system_prompt:
        input_payload = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        input_payload = prompt

    payload: dict[str, Any] = {
        "model": resolved_model,
        "input": input_payload,
        "max_output_tokens": max_output_tokens,
    }

    effort = (
        reasoning_effort
        or os.environ.get("STOCK_ANA_CODEX_REASONING_EFFORT")
        or DEFAULT_CODEX_REASONING_EFFORT
    )
    if effort:
        payload["reasoning"] = {"effort": effort}

    search_mode = (
        web_search
        or os.environ.get("STOCK_ANA_CODEX_WEB_SEARCH")
        or DEFAULT_CODEX_WEB_SEARCH
    ).strip().lower()
    if search_mode not in {"off", "auto", "required"}:
        raise ValueError("STOCK_ANA_CODEX_WEB_SEARCH 仅支持 off/auto/required")
    if search_mode != "off":
        payload["tools"] = [_WEB_SEARCH_TOOL]
        if search_mode == "required":
            payload["tool_choice"] = {"type": _WEB_SEARCH_TOOL["type"]}

    headers = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Content-Type": "application/json",
    }
    url = f"{resolved_base_url}/v1/responses"

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            logger.info(
                f"[Codex] 发送请求 (尝试 {attempt}/{MAX_RETRIES + 1}, "
                f"model={resolved_model}, reasoning={effort}, "
                f"web_search={search_mode}, prompt={len(prompt)} chars)"
            )
            resp = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=timeout_sec,
            )
            if resp.status_code == 200:
                data = resp.json()
                text = _extract_response_text(data)
                if not text:
                    raise RuntimeError(f"Codex 返回空文本: {str(data)[:500]}")
                logger.success(f"[Codex] 响应成功 ({len(text)} 字符)")
                return text

            error_text = resp.text[:1000]
            if resp.status_code in (408, 429, 500, 502, 503, 504) and attempt <= MAX_RETRIES:
                wait = BACKOFF_BASE_SEC * attempt
                logger.warning(f"[Codex] HTTP {resp.status_code}，{wait}s 后重试: {error_text[:200]}")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Codex API 错误 {resp.status_code}: {error_text}")

        except requests.RequestException as exc:
            last_error = exc
            if attempt <= MAX_RETRIES:
                wait = BACKOFF_BASE_SEC * attempt
                logger.warning(f"[Codex] 请求异常 {type(exc).__name__}，{wait}s 后重试: {exc}")
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"无法连接 Codex proxy ({resolved_base_url})，请确认 CLIProxyAPI 已启动。"
            ) from exc

    raise RuntimeError(f"Codex API 在重试后仍失败: {last_error}")
