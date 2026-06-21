"""外部 baseline 共同协议接入模块。"""

from experiments.baselines.adapters import (
    BaselineObservation,
    BaselineResultRecord,
    BaselineSpec,
    aggregate_baseline_metrics,
    aggregate_slm_proxy_metrics,
    build_baseline_observations,
    build_baseline_result_index,
    build_comparison_rows,
    default_baseline_specs,
    load_baseline_source_registry,
    normalize_baseline_result_record,
    overlay_specs_with_source_registry,
)
from experiments.baselines.primary_reproduction import (
    PRIMARY_BASELINE_IDS,
    PrimaryBaselineCommandProfile,
    PrimaryBaselineExecutionPlan,
    PrimaryBaselineResultTemplate,
    build_primary_baseline_execution_plans,
    build_primary_baseline_report,
    build_primary_result_templates,
    default_primary_command_profiles,
)

__all__ = [
    "BaselineObservation",
    "BaselineResultRecord",
    "BaselineSpec",
    "PRIMARY_BASELINE_IDS",
    "PrimaryBaselineCommandProfile",
    "PrimaryBaselineExecutionPlan",
    "PrimaryBaselineResultTemplate",
    "aggregate_baseline_metrics",
    "aggregate_slm_proxy_metrics",
    "build_baseline_observations",
    "build_baseline_result_index",
    "build_comparison_rows",
    "build_primary_baseline_execution_plans",
    "build_primary_baseline_report",
    "build_primary_result_templates",
    "default_baseline_specs",
    "default_primary_command_profiles",
    "load_baseline_source_registry",
    "normalize_baseline_result_record",
    "overlay_specs_with_source_registry",
]
