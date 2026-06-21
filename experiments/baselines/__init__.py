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
from experiments.baselines.command_adapter import (
    BaselineCommandResult,
    BaselineCommandSpec,
    run_baseline_command,
    run_baseline_commands,
)
from experiments.baselines.command_plan import (
    PRIMARY_BASELINE_ADAPTERS,
    build_baseline_command_plan_manifest,
    load_baseline_command_plan,
    selected_primary_baselines,
)
from experiments.baselines.evidence_validator import validate_external_baseline_evidence
from experiments.baselines.observation_io import (
    BaselineExecutionManifest,
    build_baseline_execution_manifest,
    load_baseline_observation_rows,
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
    "BaselineCommandResult",
    "BaselineCommandSpec",
    "BaselineExecutionManifest",
    "BaselineObservation",
    "BaselineResultRecord",
    "BaselineSpec",
    "PRIMARY_BASELINE_ADAPTERS",
    "PRIMARY_BASELINE_IDS",
    "PrimaryBaselineCommandProfile",
    "PrimaryBaselineExecutionPlan",
    "PrimaryBaselineResultTemplate",
    "aggregate_baseline_metrics",
    "aggregate_slm_proxy_metrics",
    "build_baseline_command_plan_manifest",
    "build_baseline_execution_manifest",
    "build_baseline_observations",
    "build_baseline_result_index",
    "build_comparison_rows",
    "build_primary_baseline_execution_plans",
    "build_primary_baseline_report",
    "build_primary_result_templates",
    "default_baseline_specs",
    "default_primary_command_profiles",
    "load_baseline_command_plan",
    "load_baseline_observation_rows",
    "load_baseline_source_registry",
    "normalize_baseline_result_record",
    "overlay_specs_with_source_registry",
    "run_baseline_command",
    "run_baseline_commands",
    "selected_primary_baselines",
    "validate_external_baseline_evidence",
]
