"""守护测试：确认仓库内没有重新引入已删除的旧策略 wrapper 导入路径。

Guard test: no internal imports from deprecated strategy wrappers.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_no_legacy_wrapper_imports() -> None:
    """Ensure internal code no longer imports removed strategy wrappers."""
    script = ROOT / "scripts" / "check_legacy_wrapper_imports.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_strategy_layer_boundaries() -> None:
    """Ensure strategy/workflow imports keep the intended layering boundaries."""
    script = ROOT / "scripts" / "check_strategy_layer_boundaries.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
