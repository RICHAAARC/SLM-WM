"""SD3.5 runtime adapter。"""

from __future__ import annotations

from experiments.runtime.diffusion.model_adapter import RuntimeModelConfig, RuntimeProbeBundle
from experiments.runtime.diffusion.sd3_adapter import Sd3RuntimeAdapter


class Sd35RuntimeAdapter(Sd3RuntimeAdapter):
    """为 SD3.5 提供与 SD3 共享边界的 runtime adapter。"""

    unsupported_reason = "real_sd35_backend_unavailable"

    def generate(self, config: RuntimeModelConfig) -> RuntimeProbeBundle:
        """运行 SD3.5 probe, 当前复用 synthetic fallback。"""
        return super().generate(config)
