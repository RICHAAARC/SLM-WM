"""运行 Shallow Diffuse 的 SD3.5 方法忠实 baseline 适配器。"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_baseline.primary.shallow_diffuse.adapter.method_faithful_sd35 import run_cli


def _remove_adapter_mode(argv: list[str]) -> list[str]:
    """验证唯一正式模式并从参数中移除调度字段。"""

    cleaned: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--adapter-mode":
            if index + 1 >= len(argv) or argv[index + 1] != "method_faithful_sd35":
                raise SystemExit("adapter 仅支持 method_faithful_sd35 正式模式")
            index += 2
            continue
        if item.startswith("--adapter-mode="):
            if item.split("=", 1)[1] != "method_faithful_sd35":
                raise SystemExit("adapter 仅支持 method_faithful_sd35 正式模式")
            index += 1
            continue
        cleaned.append(item)
        index += 1
    return cleaned


def main() -> None:
    """直接进入真实 SD3.5 方法实现, 不提供替代计算路径。"""

    run_cli(_remove_adapter_mode(sys.argv[1:]))


if __name__ == "__main__":
    main()
