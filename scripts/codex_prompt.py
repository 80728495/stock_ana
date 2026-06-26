#!/usr/bin/env python3
"""Send an arbitrary prompt to Codex/GPT-5.5 through local CLIProxyAPI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stock_ana.utils.codex_analyst import call_codex_prompt  # noqa: E402


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("请通过 --prompt、--prompt-file 或 stdin 提供 prompt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Codex/GPT-5.5 prompt test")
    parser.add_argument("--prompt", help="直接传入 prompt 文本")
    parser.add_argument("--prompt-file", help="从文件读取 prompt")
    parser.add_argument("--model", default=None, help="模型名，默认读取 STOCK_ANA_CODEX_MODEL 或 gpt-5.5")
    parser.add_argument("--out", help="可选：将响应写入文件")
    args = parser.parse_args()

    text = call_codex_prompt(_read_prompt(args), model=args.model)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
