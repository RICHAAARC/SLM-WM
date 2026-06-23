"""实验协议构建模块。"""

from experiments.protocol.attacks import (
    AttackConfig,
    AttackDetectionRecord,
    AttackEvaluationBoundary,
    attack_config_digest,
    build_attack_detection_record,
    build_attack_detection_records,
    default_attack_configs,
    family_metrics,
    rescue_by_attack_rows,
    score_retention_rows,
    strength_curve,
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
    FORMAL_FID_KID_BLOCKER,
    FORMAL_FID_KID_NUMERIC_BLOCKER,
    FORMAL_FID_KID_SAMPLE_BLOCKER,
    PIXEL_FEATURE_BACKEND,
    DatasetQualityImageRecord,
    build_dataset_quality_image_records,
    build_dataset_quality_metric_rows,
    build_dataset_quality_summary,
    extract_pixel_histogram_feature,
)
from experiments.protocol.events import build_event_records
from experiments.protocol.prompts import build_prompt_records, load_prompt_records
from experiments.protocol.splits import SAMPLE_ROLES, SPLIT_NAMES

__all__ = [
    "AttackConfig",
    "AttackDetectionRecord",
    "AttackEvaluationBoundary",
    "DatasetQualityImageRecord",
    "FixedFprCalibrationConfig",
    "FixedFprThreshold",
    "FORMAL_FEATURE_BACKEND",
    "FORMAL_FID_KID_BLOCKER",
    "FORMAL_FID_KID_NUMERIC_BLOCKER",
    "FORMAL_FID_KID_SAMPLE_BLOCKER",
    "PIXEL_FEATURE_BACKEND",
    "SAMPLE_ROLES",
    "SPLIT_NAMES",
    "attack_config_digest",
    "build_attack_detection_record",
    "build_attack_detection_records",
    "build_dataset_quality_image_records",
    "build_dataset_quality_metric_rows",
    "build_dataset_quality_summary",
    "build_event_records",
    "build_prompt_records",
    "calibrated_records",
    "default_attack_configs",
    "empirical_threshold_at_fpr",
    "extract_pixel_histogram_feature",
    "family_metrics",
    "load_prompt_records",
    "operating_point_metrics",
    "rescue_by_attack_rows",
    "score_retention_rows",
    "strength_curve",
]
