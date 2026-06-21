"""外部 baseline 对比协议模块。"""

from experiments.baselines.adapters import (
    BaselineObservation,
    BaselineSpec,
    aggregate_baseline_metrics,
    aggregate_slm_proxy_metrics,
    build_baseline_observations,
    build_comparison_rows,
    default_baseline_specs,
)

__all__ = [
    "BaselineObservation",
    "BaselineSpec",
    "aggregate_baseline_metrics",
    "aggregate_slm_proxy_metrics",
    "build_baseline_observations",
    "build_comparison_rows",
    "default_baseline_specs",
]
