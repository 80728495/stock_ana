#!/usr/bin/env python3
"""守护脚本：检测是否重新引入已删除的旧策略 wrapper 导入路径。

扫描项目源码，若发现内部代码仍在 import 已移除的旧模块（参见 REMOVED_MODULES 列表），
则以非零退出码报错。设计为无额外依赖，可在本地或 CI 轻量运行。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LEGACY = {
    "find_vcp",
    "ma_squeeze",
    "momentum_detector",
    "rs",
    "triangle",
    "triangle_kde",
    "triangle_vcp",
    "vcp",
    "main_rally_pullback",
}

FROM_RE = re.compile(
    r"^\s*from\s+stock_ana\.strategies\.([a-zA-Z0-9_]+)\s+import\s+",
    re.MULTILINE,
)
IMPORT_RE = re.compile(
    r"^\s*import\s+stock_ana\.strategies\.([a-zA-Z0-9_]+)(?:\s|$)",
    re.MULTILINE,
)


def _is_ignored(path: Path) -> bool:
    """Return whether a path should be skipped during import scanning."""
    parts = set(path.parts)
    if ".venv" in parts or ".git" in parts or "site-packages" in parts:
        return True
    return False


def main() -> int:
    """Scan repository Python files and fail on removed legacy imports."""
    offenders: list[tuple[Path, int, str]] = []

    for path in ROOT.rglob("*.py"):
        if _is_ignored(path):
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue

        for regex in (FROM_RE, IMPORT_RE):
            for m in regex.finditer(text):
                mod = m.group(1)
                if mod not in LEGACY:
                    continue
                line = text.count("\n", 0, m.start()) + 1
                offenders.append((path.relative_to(ROOT), line, m.group(0).strip()))

    if offenders:
        print("Found removed legacy strategy imports:")
        for rel, line, stmt in sorted(offenders):
            print(f"  {rel}:{line}: {stmt}")
        print("\nUse stock_ana.strategies.impl.* or stock_ana.strategies.api instead.")
        return 1

    print("OK: no removed legacy strategy imports found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
