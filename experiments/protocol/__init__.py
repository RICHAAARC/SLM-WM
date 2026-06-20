"""实验协议构建模块。"""

from experiments.protocol.calibration import (
    FixedFprCalibrationConfig,
    FixedFprThreshold,
    calibrated_records,
    empirical_threshold_at_fpr,
    operating_point_metrics,
)
from experiments.protocol.events import build_event_records
from experiments.protocol.prompts import build_prompt_records, load_prompt_records
from experiments.protocol.splits import SAMPLE_ROLES, SPLIT_NAMES

__all__ = [
    "FixedFprCalibrationConfig",
    "FixedFprThreshold",
    "SAMPLE_ROLES",
    "SPLIT_NAMES",
    "build_event_records",
    "build_prompt_records",
    "calibrated_records",
    "empirical_threshold_at_fpr",
    "load_prompt_records",
    "operating_point_metrics",
]
