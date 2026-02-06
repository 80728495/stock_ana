"""
跨平台配置管理
"""

import platform
from pathlib import Path

from dotenv import load_dotenv

# 加载 .env 文件（如有 API 密钥等）
load_dotenv()

# 项目根目录（跨平台兼容）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "output"

# 确保目录存在
for d in [DATA_DIR, CACHE_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_platform_info() -> dict:
    """返回当前平台信息"""
    return {
        "system": platform.system(),        # Windows / Darwin(macOS)
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
