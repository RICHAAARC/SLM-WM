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

__all__ = [
    "BaselineObservation",
    "BaselineResultRecord",
    "BaselineSpec",
    "aggregate_baseline_metrics",
    "aggregate_slm_proxy_metrics",
    "build_baseline_observations",
    "build_baseline_result_index",
    "build_comparison_rows",
    "default_baseline_specs",
    "load_baseline_source_registry",
    "normalize_baseline_result_record",
    "overlay_specs_with_source_registry",
]
