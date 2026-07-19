"""提供正式二维 LF、HF-tail 与盲内容评分惰性公开接口。"""

from importlib import import_module
from typing import Any

__all__ = [
    "BlindContentScore",
    "HIGH_FREQUENCY_TAIL_PROTOCOL_DIGEST",
    "HighFrequencyTailCarrierTemplate",
    "KEYED_PRG_VERSION",
    "LOW_FREQUENCY_BOUNDARY_MODE",
    "LOW_FREQUENCY_CEIL_MODE",
    "LOW_FREQUENCY_COUNT_INCLUDE_PAD",
    "LOW_FREQUENCY_DIVISOR_OVERRIDE",
    "LOW_FREQUENCY_KERNEL_SIZE",
    "LOW_FREQUENCY_PADDING",
    "LOW_FREQUENCY_STRIDE",
    "LowFrequencyCarrierTemplate",
    "LowFrequencyCarrierConfig",
    "build_low_frequency_template",
    "build_high_frequency_tail_template",
    "compute_blind_content_score",
    "keyed_prg_protocol_record",
    "require_supported_keyed_prg_version",
]


_EXPORT_MODULES = {
    **{
        name: "main.core.keyed_prg"
        for name in (
            "KEYED_PRG_VERSION",
            "keyed_prg_protocol_record",
            "require_supported_keyed_prg_version",
        )
    },
    **{
        name: "main.methods.carrier.blind_content_score"
        for name in ("BlindContentScore", "compute_blind_content_score")
    },
    **{
        name: "main.methods.carrier.high_frequency_tail"
        for name in (
            "HIGH_FREQUENCY_TAIL_PROTOCOL_DIGEST",
            "HighFrequencyTailCarrierTemplate",
            "build_high_frequency_tail_template",
        )
    },
    **{
        name: "main.methods.carrier.low_frequency"
        for name in ("LowFrequencyCarrierTemplate", "build_low_frequency_template")
    },
    **{
        name: "main.methods.carrier.keyed_tensor"
        for name in (
            "LOW_FREQUENCY_BOUNDARY_MODE",
            "LOW_FREQUENCY_CEIL_MODE",
            "LOW_FREQUENCY_COUNT_INCLUDE_PAD",
            "LOW_FREQUENCY_DIVISOR_OVERRIDE",
            "LOW_FREQUENCY_KERNEL_SIZE",
            "LOW_FREQUENCY_PADDING",
            "LOW_FREQUENCY_STRIDE",
            "LowFrequencyCarrierConfig",
        )
    },
}


def __getattr__(name: str) -> Any:
    """仅在公开符号被实际消费时加载对应科学模块。"""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """向交互式调用者公开稳定接口集合。"""

    return sorted((*globals(), *__all__))
