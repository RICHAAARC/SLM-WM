"""运行 gaussian_shading 的 SLM 外部 baseline adapter。

默认模式保留轻量 SD3.5 latent smoke adapter, 用于本地快速测试和命令计划链路检查。
显式传入 `--adapter-mode method_faithful_sd35` 时, 该入口调用 Gaussian Shading
方法忠实 SD3.5 适配器, 在真实 GPU 环境加载 SD3.5 Medium 并执行 message noise
采样、生成、近似反演和 bit voting 检测。
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_baseline.primary.sd35_diffusion_baseline_common import run_contract_or_report_required_adapter
from external_baseline.primary.gaussian_shading.adapter.method_faithful_sd35 import run_cli as run_method_faithful_cli

BASELINE_ID = "gaussian_shading"
MODEL_ALIGNMENT_STATUS = "sd35_medium_method_faithful_adapter_available"
REAL_RUN_UNSUPPORTED_REASON = "gaussian_shading_method_faithful_sd35_mode_requires_explicit_adapter_mode"
METHOD_FAITHFUL_MODE = "method_faithful_sd35"
LATENT_SMOKE_MODE = "latent_smoke"


def _extract_adapter_mode(argv: list[str]) -> tuple[str, list[str]]:
    """从 argv 中提取 adapter mode, 并返回移除该参数后的参数列表。"""

    cleaned: list[str] = []
    mode = LATENT_SMOKE_MODE
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--adapter-mode":
            if index + 1 >= len(argv):
                raise SystemExit("--adapter-mode 需要一个取值")
            mode = argv[index + 1]
            index += 2
            continue
        if item.startswith("--adapter-mode="):
            mode = item.split("=", 1)[1]
            index += 1
            continue
        cleaned.append(item)
        index += 1
    return mode, cleaned


def main() -> None:
    """CLI 入口。"""

    mode, cleaned_argv = _extract_adapter_mode(sys.argv[1:])
    if mode == METHOD_FAITHFUL_MODE:
        run_method_faithful_cli(cleaned_argv)
        return
    if mode != LATENT_SMOKE_MODE:
        raise SystemExit(f"unsupported_gaussian_shading_adapter_mode:{mode}")
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *cleaned_argv]
        run_contract_or_report_required_adapter(
            baseline_id=BASELINE_ID,
            model_alignment_status=MODEL_ALIGNMENT_STATUS,
            real_run_unsupported_reason=REAL_RUN_UNSUPPORTED_REASON,
        )
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
