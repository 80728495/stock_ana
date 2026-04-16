#!/usr/bin/env python3
"""守护脚本：检测策略层架构边界是否被违反。

规则：
1. strategies/impl 内的模块不可互相导入（避免 impl 间耦合）
2. workflows 模块不可导入 impl 的私有符号（下划线开头）

Fail when strategy/workflow imports violate layer boundaries.

Rules:
1. `strategies/impl` modules must not import other `strategies/impl` modules.
2. `workflows` modules must not import private symbols (leading underscore)
   from `strategies/impl` modules.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "stock_ana"


def _is_ignored(path: Path) -> bool:
    parts = set(path.parts)
    return ".venv" in parts or ".git" in parts or "site-packages" in parts


def _scan_file(path: Path) -> list[tuple[Path, int, str]]:
    offenders: list[tuple[Path, int, str]] = []
    rel = path.relative_to(ROOT)
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue

        mod = node.module or ""
        names = [a.name for a in node.names]

        # Rule 1: impl should not depend on impl.
        if "src/stock_ana/strategies/impl/" in str(path).replace("\\", "/"):
            if mod.startswith("stock_ana.strategies.impl"):
                offenders.append((rel, node.lineno, f"impl_to_impl import: from {mod} import {', '.join(names)}"))

        # Rule 2: workflows should not import private impl symbols.
        if "src/stock_ana/workflows/" in str(path).replace("\\", "/"):
            if mod.startswith("stock_ana.strategies.impl"):
                private = [n for n in names if n.startswith("_")]
                if private:
                    offenders.append((rel, node.lineno, f"workflow imports private impl symbol(s): from {mod} import {', '.join(private)}"))

    return offenders


def main() -> int:
    offenders: list[tuple[Path, int, str]] = []

    for path in SRC.rglob("*.py"):
        if _is_ignored(path):
            continue
        offenders.extend(_scan_file(path))

    if offenders:
        print("Found strategy layer boundary violations:")
        for rel, line, msg in sorted(offenders):
            print(f"  {rel}:{line}: {msg}")
        return 1

    print("OK: strategy layer boundaries are clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
