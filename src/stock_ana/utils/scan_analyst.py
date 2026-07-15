"""
扫描信号基本面分析模块

将 vegas_mid_scan 等扫描模块输出的信号列表，
结合 data/scan_signal_prompt.md 模板构造**单个**批量 prompt，
一次请求发给 Gemini，获得所有标的的基本面 + 估值分析，
结果保存为一个 .md 文件。

用法：
    import asyncio
    from stock_ana.utils.scan_analyst import analyze_signals
    asyncio.run(analyze_signals(signals_list))
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import date
from pathlib import Path

from loguru import logger

from stock_ana.config import OUTPUT_DIR

# ─── 路径常量 ───────────────────────────────────────────────────────────────
PROMPT_TEMPLATE_PATH = Path(__file__).parents[3] / "data" / "scan_signal_prompt.md"
DEFAULT_OUT_DIR = OUTPUT_DIR / "scan_analysis"
DEFAULT_MODEL = os.environ.get("STOCK_ANA_GEMINI_MODEL") or "gemini-3.1-pro"
DEFAULT_LLM_BACKEND = "codex"
DEFAULT_LLM_BATCH_SIZE = 3
TRUTHY_ENV = {"1", "true", "yes", "on"}

GEMINI_WEB_MODEL_HEADERS: dict[str, dict[str, str]] = {
    # Gemini web internal Pro mode header. Kept separate from gemini_webapi's
    # built-in enum so we can move faster when the web app exposes a new model.
    "gemini-3.1-pro": {
        "x-goog-ext-525001261-jspb": (
            '[1,null,null,null,"e6fa609c3fa255c0",null,null,0,[4,5,6,8],'
            'null,null,2,null,null,3,1,"09D681E7-26F2-4A94-A465-38386B7AB93B"]'
        )
    },
}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(f"{name}={raw!r} 不是有效整数，使用默认值 {default}")
        return default
    return value if value > 0 else default


def gemini_web_enabled() -> bool:
    """Return whether Gemini Web cookie-based access is explicitly enabled."""
    return os.environ.get("STOCK_ANA_ENABLE_GEMINI_WEB", "").strip().lower() in TRUTHY_ENV


def resolve_gemini_model(model: str | dict) -> str | dict:
    """Return a gemini_webapi-compatible model value."""
    if isinstance(model, dict):
        return model
    model_name = str(model).strip()
    if model_name in GEMINI_WEB_MODEL_HEADERS:
        return {
            "model_name": model_name,
            "model_header": GEMINI_WEB_MODEL_HEADERS[model_name],
        }
    return model_name

# ─── Prompt 构建 ─────────────────────────────────────────────────────────────

def _load_template() -> str:
    """读取 Markdown prompt 模板文件（去掉顶部注释块）。"""
    if not PROMPT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Prompt 模板不存在: {PROMPT_TEMPLATE_PATH}")
    text = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip() == "---":
            start = i + 1
            break
    return "\n".join(lines[start:]).strip()


def _format_stock_block(i: int, signal: dict) -> str:
    """将单个信号简化为一行：序号. 代号（公司名）。"""
    sym  = signal.get("symbol", "")
    name = signal.get("name", sym)
    return f"{i}. {sym}（{name}）"


def build_prompt(signals: list[dict]) -> str:
    """
    将多个信号 dict 构造为单个批量分析 prompt。

    Args:
        signals: 信号列表，每个元素来自 run_scan() 返回值

    Returns:
        完整 prompt 字符串
    """
    template = _load_template()

    stock_blocks = "\n\n".join(
        _format_stock_block(i, sig) for i, sig in enumerate(signals, 1)
    )

    return template.format(
        scan_date        = date.today().isoformat(),
        stock_count      = len(signals),
        stock_list_block = stock_blocks,
    )


# ─── Gemini 调用 ─────────────────────────────────────────────────────────────

def _update_env_cookies(psid: str, psidts: str, env_path: "Path") -> None:
    """将最新的 Cookie 写回 .env 文件，保留文件中其他内容。"""
    import re
    text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    for key, val in [("GEMINI_PSID", psid), ("GEMINI_PSIDTS", psidts)]:
        pattern = rf"^{key}=.*$"
        replacement = f"{key}={val}"
        if re.search(pattern, text, flags=re.MULTILINE):
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        else:
            text = text.rstrip("\n") + f"\n{replacement}\n"
    env_path.write_text(text, encoding="utf-8")


async def _init_client(model: str = DEFAULT_MODEL):
    """初始化 Gemini 客户端。

    优先顺序：
    1. 从 Chrome 直接读取最新 Cookie（自动更新 .env）
    2. 仅当 Chrome 文件被锁（Chrome 正在运行）时，回退到 .env 缓存

    若 PSIDTS 已被 Google 服务端轮换（约每 1-3 小时），会自动打开浏览器访问
    Gemini，等 Chrome 拿到新 Cookie 后重新读取并重试（最多 2 次）。
    """
    if not gemini_web_enabled():
        raise RuntimeError(
            "Gemini Web 访问已关闭。若确需临时使用，请设置 STOCK_ANA_ENABLE_GEMINI_WEB=1。"
        )

    from pathlib import Path

    from gemini_webapi import GeminiClient
    from gemini_webapi.exceptions import AuthError as _AuthError

    env_path = Path(__file__).resolve().parents[3] / ".env"

    async def _read_cookies() -> tuple[str, str]:
        """从 Chrome 读取 PSID / PSIDTS（Chrome 被锁时会自动关闭 Chrome 后读取）。"""
        psid = psidts = ""
        from stock_ana.utils.chrome_cookies import get_gemini_cookies
        chrome = get_gemini_cookies()
        psid   = chrome.get("__Secure-1PSID", "").strip()
        psidts = chrome.get("__Secure-1PSIDTS", "").strip()
        if psid:
            _update_env_cookies(psid, psidts, env_path)
        return psid, psidts

    async def _try_rotate(psid: str, psidts: str) -> str:
        """尝试用 RotateCookies 端点刷新 PSIDTS，写入 gemini_webapi 缓存。"""
        if not psid:
            return psidts
        try:
            from httpx import Cookies as _HxCookies
            from gemini_webapi.utils.rotate_1psidts import rotate_1psidts as _rotate
            _jar = _HxCookies()
            _jar.set("__Secure-1PSID", psid, domain=".google.com")
            if psidts:
                _jar.set("__Secure-1PSIDTS", psidts, domain=".google.com")
            _new, _ = await _rotate(_jar)
            if _new:
                _update_env_cookies(psid, _new, env_path)
                logger.info(f"PSIDTS 已通过 RotateCookies 刷新（{len(_new)} 字符）")
                return _new
        except Exception as _e:
            logger.debug(f"PSIDTS 预刷新失败: {_e}")
        return psidts

    psid, psidts = await _read_cookies()
    logger.info("Gemini 客户端：从 Chrome 读取最新 Cookie 并已更新 .env")
    psidts = await _try_rotate(psid, psidts)

    def _is_cookie_error(exc: Exception) -> bool:
        """判断异常是否属于 Cookie 过期类错误（含 gemini_webapi 内部重试耗尽）。"""
        if isinstance(exc, _AuthError):
            return True
        if isinstance(exc, RuntimeError):
            msg = str(exc).lower()
            return "failed to initialize" in msg or "psidts" in msg or "expired" in msg
        return False

    def _is_network_error(exc: Exception) -> bool:
        """判断是否为网络连接类错误（代理未就绪、ConnectTimeout 等）。"""
        try:
            import httpx as _httpx
            return isinstance(exc, (_httpx.ConnectTimeout, _httpx.ConnectError, _httpx.NetworkError))
        except ImportError:
            return False

    for attempt in range(5):  # 最多重试 4 次（给 Chrome flush 和网络恢复更多机会）
        try:
            client = GeminiClient(
                secure_1psid=psid or None,
                secure_1psidts=psidts or None,
            )
            await client.init(timeout=900, auto_close=False, auto_refresh=True, verbose=False)
            logger.info(f"Gemini 客户端初始化成功（模型：{model}）")
            return client
        except Exception as _exc:
            if _is_network_error(_exc):
                if attempt >= 3:
                    raise
                wait = 30 * (attempt + 1)
                logger.warning(
                    f"Gemini 初始化网络错误（第{attempt+1}次，{type(_exc).__name__}），"
                    f"{wait}s 后重试..."
                )
                await asyncio.sleep(wait)
                continue
            if not _is_cookie_error(_exc):
                raise
            # Mac 下 Cookie 从不过期（用户持续登录），直接抛出
            if sys.platform != "win32" or attempt >= 3:
                raise
            # Windows：PSIDTS 已被 Google 服务端轮换 → 打开 Chrome 刷新后重读
            logger.warning(
                f"Gemini Cookie 已过期（第{attempt+1}次尝试），"
                "正在自动打开 Chrome 刷新 Gemini 登录态，请稍候 55 秒..."
            )
            import subprocess
            _chrome_paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            ]
            for _p in _chrome_paths:
                if Path(_p).exists():
                    subprocess.Popen([_p, "https://gemini.google.com/app"])
                    break
            await asyncio.sleep(55)          # 等 Chrome 将新 Cookie 写入 SQLite（55s 比原 35s 更稳定）
            psid, psidts = await _read_cookies()
            psidts = await _try_rotate(psid, psidts)
            logger.info("已重新读取 Cookie，准备重试初始化...")

    raise RuntimeError("Gemini 客户端初始化失败，已穷尽所有重试")


async def _call_gemini(
    prompt: str,
    client,
    model: str = DEFAULT_MODEL,
    max_retries: int = 0,
) -> str:
    """
    发送 prompt 到 Gemini，返回文本结果。

    gemini_webapi 内部有时会在内容已经完整返回后，因缺少"完成标记"帧
    而抛出 APIError("Stream interrupted or truncated.")。
    此时 generate_content 的异常向上传播，但实际文本已经在最后一个
    yield 的 ModelOutput 里。

    解法：在 _generate 层把 APIError 改成在已有候选文本时直接 return，
    而不是重试或丢弃。我们在这里用 monkey-patch 替换 _generate 的收集逻辑。
    """
    import gemini_webapi.exceptions as _gex

    last_err = None
    resolved_model = resolve_gemini_model(model)
    for attempt in range(max_retries + 1):
        try:
            # 直接 iterate _generate，遇到 APIError("interrupted") 时
            # 如果已经拿到文本就视为成功，否则重新抛出。
            last_output = None
            async for last_output in client._generate(prompt=prompt, model=resolved_model):
                pass
            # 正常完成
            text = (last_output.text if last_output else "") or ""
            logger.success(f"Gemini 分析完成，共 {len(text)} 字符")
            return text
        except _gex.APIError as e:
            msg = str(e)
            if "interrupted" in msg.lower() or "truncated" in msg.lower():
                if last_output is not None:
                    text = last_output.text or ""
                    if text.strip():
                        logger.info(
                            f"Gemini 流标记缺失但内容已完整（{len(text)} 字符），视为成功"
                        )
                        return text
            last_err = e
            if attempt < max_retries:
                wait = 15 * (attempt + 1)
                logger.warning(f"请求失败 [{type(e).__name__}]: {msg[:80]}，{wait}s 后重试...")
                await asyncio.sleep(wait)
            else:
                logger.error(f"分析失败（重试{max_retries}次）[{type(e).__name__}]: {msg[:200]}")
        except (asyncio.TimeoutError, TimeoutError) as e:
            # asyncio.TimeoutError / Python 内置 TimeoutError:
            # 在 Python 3.11+ 两者是同一个类，str(e) 为空字符串。
            # httpx 在 Windows 上某些路径走 asyncio.timeout() 而非 ReadTimeout，
            # 会绕过 gemini_webapi 的 except ReadTimeout，以空消息穿透上来。
            last_err = e
            msg = f"请求超时（timeout=900s），attempt={attempt+1}/{max_retries+1}"
            if attempt < max_retries:
                wait = 30 * (attempt + 1)
                logger.warning(f"{msg}，{wait}s 后重试...")
                await asyncio.sleep(wait)
            else:
                logger.error(f"分析失败（重试{max_retries}次）[TimeoutError]: {msg}")
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                wait = 15 * (attempt + 1)
                logger.warning(f"请求失败 [{type(e).__name__}]: {str(e)[:80]}，{wait}s 后重试...")
                await asyncio.sleep(wait)
            else:
                logger.error(f"分析失败（重试{max_retries}次）[{type(e).__name__}]: {str(e)[:200]}")
    raise last_err  # type: ignore[misc]


async def _call_codex(prompt: str, model: str | None = None) -> str:
    """发送 prompt 到本地 CLIProxyAPI/Codex，返回文本结果。"""
    from stock_ana.utils.codex_analyst import DEFAULT_CODEX_MODEL, call_codex_prompt

    return await asyncio.to_thread(
        call_codex_prompt,
        prompt,
        model=model or DEFAULT_CODEX_MODEL,
    )


def _analysis_batch_size() -> int:
    return _env_int(
        "STOCK_ANA_SCAN_LLM_BATCH_SIZE",
        _env_int("STOCK_ANA_CODEX_BATCH_SIZE", DEFAULT_LLM_BATCH_SIZE),
    )


async def _call_codex_for_targets(targets: list[dict], model: str | None = None) -> str:
    """Analyze targets with Codex, splitting large lists into stable batches."""
    batch_size = _analysis_batch_size()
    if len(targets) <= batch_size:
        prompt = build_prompt(targets)
        logger.info(f"Prompt 长度: {len(prompt)} 字符")
        return await _call_codex(prompt, model=model)

    batches = [targets[i : i + batch_size] for i in range(0, len(targets), batch_size)]
    logger.info(
        f"Codex 分批分析：{len(targets)} 个标的，分 {len(batches)} 批"
        f"（每批 ≤{batch_size} 只）"
    )
    texts: list[str] = []
    for idx, batch in enumerate(batches, 1):
        symbols = ", ".join(str(s.get("symbol", "?")) for s in batch)
        prompt = build_prompt(batch)
        logger.info(
            f"  Codex 第 {idx}/{len(batches)} 批：{symbols}，"
            f"Prompt 长度: {len(prompt)} 字符"
        )
        text = await _call_codex(prompt, model=model)
        texts.append(
            f"## Codex Batch {idx}/{len(batches)} — {symbols}\n\n{text.strip()}"
        )
    return "\n\n---\n\n".join(texts)


async def _call_gemini_for_targets(targets: list[dict], client, model: str) -> str:
    """Analyze targets with Gemini, splitting large lists into stable batches."""
    batch_size = _analysis_batch_size()
    if len(targets) <= batch_size:
        prompt = build_prompt(targets)
        logger.info(f"Prompt 长度: {len(prompt)} 字符")
        return await _call_gemini(prompt, client, model=model, max_retries=1)

    batches = [targets[i : i + batch_size] for i in range(0, len(targets), batch_size)]
    logger.info(
        f"Gemini 分批分析：{len(targets)} 个标的，分 {len(batches)} 批"
        f"（每批 ≤{batch_size} 只）"
    )
    texts: list[str] = []
    for idx, batch in enumerate(batches, 1):
        symbols = ", ".join(str(s.get("symbol", "?")) for s in batch)
        prompt = build_prompt(batch)
        logger.info(
            f"  Gemini 第 {idx}/{len(batches)} 批：{symbols}，"
            f"Prompt 长度: {len(prompt)} 字符"
        )
        text = await _call_gemini(prompt, client, model=model, max_retries=1)
        texts.append(
            f"## Gemini Batch {idx}/{len(batches)} — {symbols}\n\n{text.strip()}"
        )
    return "\n\n---\n\n".join(texts)


# ─── 结果保存 ─────────────────────────────────────────────────────────────────

def _save_result(signals: list[dict], analysis_text: str, out_dir: Path) -> Path:
    """将批量分析结果保存为一个 Markdown 文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    today      = date.today().isoformat()
    syms       = "_".join(s.get("symbol", "") for s in signals[:5])
    suffix     = f"_plus{len(signals)-5}more" if len(signals) > 5 else ""
    filename   = f"{today}_{syms}{suffix}.md"
    path       = out_dir / filename

    # 文件头：信号汇总表
    header_lines = [
        f"# 扫描信号基本面分析报告",
        f"",
        f"**扫描日期**: {today}  |  **标的数**: {len(signals)}",
        f"",
        f"| 代码 | 公司 | 信号 | 评分 | 入场日 |",
        f"|------|------|------|------|--------|",
    ]
    for s in signals:
        header_lines.append(
            f"| {s.get('symbol','')} | {s.get('name','')} "
            f"| {s.get('signal','')} | {s.get('score',0):+d} "
            f"| {s.get('entry_date','').split('(')[0]} |"
        )
    header_lines += ["", "---", ""]

    path.write_text("\n".join(header_lines) + "\n" + analysis_text, encoding="utf-8")
    logger.info(f"分析报告已保存 → {path}")
    return path


# ─── 公开接口 ─────────────────────────────────────────────────────────────────

async def analyze_signals(
    signals: list[dict],
    model: str = DEFAULT_MODEL,
    out_dir: Path | None = None,
    min_signal: str | None = None,
    backend: str | None = None,
) -> Path:
    """
    批量分析股票列表，一次请求完成所有标的，保存为一个 .md 报告。

    Args:
        signals:    股票列表，每个元素至少包含 symbol、name 字段。
                    也可以是 run_scan() 的返回值（会自动按 min_signal 过滤）。
        model:      模型名。Gemini 默认 gemini-3.1-pro；Codex 可传 gpt-5.5
        out_dir:    输出目录，默认 data/output/scan_analysis/YYYY-MM-DD/
        min_signal: 当 signals 来自扫描结果时，可指定最低等级过滤
                    （STRONG_BUY/BUY/HOLD）；None 则不过滤，全部分析。
        backend:    "gemini" 或 "codex"。为 None 时读取 STOCK_ANA_SCAN_LLM_BACKEND，
                    未设置则默认使用 Codex。

    Returns:
        生成的 .md 文件路径
    """
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR / date.today().isoformat()

    if min_signal:
        rank = {"STRONG_BUY": 4, "BUY": 3, "HOLD": 2, "AVOID": 1}
        min_rank = rank.get(min_signal, 2)
        targets = [s for s in signals if rank.get(s.get("signal", ""), 0) >= min_rank]
    else:
        targets = list(signals)

    if not targets:
        logger.info("没有符合条件的标的，跳过分析")
        raise ValueError("没有符合条件的标的")

    resolved_backend = (
        backend
        or os.environ.get("STOCK_ANA_SCAN_LLM_BACKEND")
        or DEFAULT_LLM_BACKEND
    ).strip().lower()

    if resolved_backend == "codex":
        logger.info(f"构建 Codex 分析任务，共 {len(targets)} 个标的...")
        text = await _call_codex_for_targets(targets, model=None if model == DEFAULT_MODEL else model)
        path = _save_result(targets, text, out_dir)
        return path

    if resolved_backend != "gemini":
        raise ValueError(f"不支持的 LLM backend: {resolved_backend}，可选 gemini/codex")

    client = await _init_client(model)
    try:
        logger.info(f"构建 Gemini 分析任务，共 {len(targets)} 个标的...")
        text = await _call_gemini_for_targets(targets, client, model=model)
        path = _save_result(targets, text, out_dir)
    finally:
        await client.close()

    return path

