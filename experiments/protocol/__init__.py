"""实验协议构建模块。"""

from experiments.protocol.attacks import (
    AttackConfig,
    attack_config_digest,
    build_attack_record_digest,
    default_attack_configs,
    resolve_formal_attack_config,
)
from experiments.protocol.calibration import (
    FixedFprCalibrationConfig,
    FixedFprThreshold,
    calibrated_records,
    empirical_threshold_at_fpr,
    operating_point_metrics,
)
from experiments.protocol.dataset_quality import (
    FORMAL_FEATURE_BACKEND,
    FORMAL_FEATURE_EXTRACTOR_ID,
    FORMAL_FID_KID_BLOCKER,
    FORMAL_FID_KID_NUMERIC_BLOCKER,
    FORMAL_FID_KID_SAMPLE_BLOCKER,
    DatasetQualityImageRecord,
    build_dataset_quality_image_records,
    build_dataset_quality_metric_rows,
    build_dataset_quality_summary,
    rebuild_formal_fid_kid_metric_rows,
)
from experiments.protocol.events import build_event_records
from experiments.protocol.prompts import build_prompt_records, load_prompt_records
from experiments.protocol.splits import SAMPLE_ROLES, SPLIT_NAMES

__all__ = [
    "AttackConfig",
    "DatasetQualityImageRecord",
    "FixedFprCalibrationConfig",
    "FixedFprThreshold",
    "FORMAL_FEATURE_BACKEND",
    "FORMAL_FEATURE_EXTRACTOR_ID",
    "FORMAL_FID_KID_BLOCKER",
    "FORMAL_FID_KID_NUMERIC_BLOCKER",
    "FORMAL_FID_KID_SAMPLE_BLOCKER",
    "SAMPLE_ROLES",
    "SPLIT_NAMES",
    "attack_config_digest",
    "build_attack_record_digest",
    "build_dataset_quality_image_records",
    "build_dataset_quality_metric_rows",
    "build_dataset_quality_summary",
    "rebuild_formal_fid_kid_metric_rows",
    "build_event_records",
    "build_prompt_records",
    "calibrated_records",
    "default_attack_configs",
    "empirical_threshold_at_fpr",
    "load_prompt_records",
    "operating_point_metrics",
    "resolve_formal_attack_config",
]
