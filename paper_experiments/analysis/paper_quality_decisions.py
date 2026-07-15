"""冻结质量主张语义并执行 Prompt 级非劣效推断。"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.protocol.attacks import (
    attack_config_digest,
    default_attack_configs,
)
from main.core.digest import build_stable_digest
from paper_experiments.analysis.paper_claim_decisions import build_claim_decision


DEFAULT_PAPER_QUALITY_CLAIM_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "paper_quality_claim_protocol.json"
)
QUALITY_SUBCLAIM_IDS = (
    "paired_perceptual_quality_noninferiority",
    "semantic_alignment_noninferiority",
    "distributional_preservation_noninferiority",
)


class PaperQualityDecisionError(ValueError):
    """表示质量协议、Prompt 级推断或三态组合不合法。"""


def load_paper_quality_claim_protocol(
    path: str | Path = DEFAULT_PAPER_QUALITY_CLAIM_PROTOCOL_PATH,
) -> dict[str, Any]:
    """读取质量结论配置并绑定当前正式攻击注册表。"""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperQualityDecisionError("论文质量结论协议无法读取") from exc
    if not isinstance(payload, dict) or payload.get("protocol_schema") != (
        "paper_quality_claim_protocol_v1"
    ):
        raise PaperQualityDecisionError("论文质量结论协议 schema 不受支持")
    if (
        payload.get("primary_sampling_unit") != "prompt"
        or payload.get("nested_sampling_unit")
        != "registered_randomization_repeat_within_prompt"
        or payload.get("bootstrap_bit_generator") != "PCG64"
        or payload.get("bootstrap_quantile_method") != "linear"
        or type(payload.get("bootstrap_resample_count")) is not int
        or int(payload["bootstrap_resample_count"]) < 1000
        or not math.isclose(
            float(payload.get("confidence_level", math.nan)),
            0.95,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise PaperQualityDecisionError("Prompt 聚类推断配置不完整")
    perceptual = payload.get("paired_perceptual_quality_noninferiority")
    semantic = payload.get("semantic_alignment_noninferiority")
    distribution = payload.get("distributional_preservation_noninferiority")
    if not all(isinstance(value, dict) for value in (perceptual, semantic, distribution)):
        raise PaperQualityDecisionError("三类质量非劣效配置必须是对象")
    if (
        not math.isclose(
            float(perceptual.get("minimum_lower_confidence_bound", math.nan)),
            0.99,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(semantic.get("minimum_lower_confidence_bound", math.nan)),
            0.995,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(
                distribution.get(
                    "maximum_prompt_conditional_kid_upper_confidence_bound",
                    math.nan,
                )
            ),
            0.001,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or distribution.get("fid_evidence_role") != "descriptive"
        or distribution.get(
            "generation_quality_against_real_reference_allowed_without_reference"
        )
        is not False
    ):
        raise PaperQualityDecisionError("质量非劣效界限或解释边界发生漂移")
    attack_rows = [
        {
            "attack_id": config.attack_id,
            "attack_config_digest": attack_config_digest(config),
        }
        for config in default_attack_configs()
        if config.enabled
    ]
    resolved = {
        **payload,
        "registered_attack_ids": [row["attack_id"] for row in attack_rows],
        "registered_attack_registry_digest": build_stable_digest(attack_rows),
    }
    return {
        **resolved,
        "paper_quality_claim_protocol_digest": build_stable_digest(resolved),
    }


def build_prompt_cluster_mean_inference(
    prompt_values: Mapping[str, float],
    *,
    analysis_id: str,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """以 Prompt 为唯一重采样单位构造均值和双侧 bootstrap 区间。"""

    resolved_protocol = dict(protocol or load_paper_quality_claim_protocol())
    ordered_items = sorted(
        (str(prompt_id).strip(), float(value))
        for prompt_id, value in prompt_values.items()
    )
    if (
        len(ordered_items) < 2
        or len({prompt_id for prompt_id, _value in ordered_items})
        != len(ordered_items)
        or any(not prompt_id or not math.isfinite(value) for prompt_id, value in ordered_items)
    ):
        raise PaperQualityDecisionError("Prompt 级统计必须至少包含2个唯一有限观测")
    prompt_ids = [prompt_id for prompt_id, _value in ordered_items]
    values = np.asarray([value for _prompt_id, value in ordered_items], dtype=np.float64)
    analysis_name = str(analysis_id).strip()
    if not analysis_name:
        raise PaperQualityDecisionError("analysis_id 不得为空")
    seed_material = {
        "analysis_id": analysis_name,
        "prompt_ids": prompt_ids,
        "prompt_values_digest": build_stable_digest(ordered_items),
        "paper_quality_claim_protocol_digest": resolved_protocol[
            "paper_quality_claim_protocol_digest"
        ],
    }
    seed_digest = build_stable_digest(seed_material)
    generator = np.random.Generator(np.random.PCG64(int(seed_digest[:16], 16)))
    resample_count = int(resolved_protocol["bootstrap_resample_count"])
    prompt_count = len(values)
    bootstrap_means = np.empty(resample_count, dtype=np.float64)
    batch_size = max(1, min(256, 2_000_000 // prompt_count))
    for start in range(0, resample_count, batch_size):
        stop = min(start + batch_size, resample_count)
        sampled_indices = generator.integers(
            0,
            prompt_count,
            size=(stop - start, prompt_count),
        )
        bootstrap_means[start:stop] = values[sampled_indices].mean(axis=1)
    alpha = 1.0 - float(resolved_protocol["confidence_level"])
    ci_low, ci_high = np.quantile(
        bootstrap_means,
        (alpha / 2.0, 1.0 - alpha / 2.0),
        method=str(resolved_protocol["bootstrap_quantile_method"]),
    )
    core = {
        "analysis_id": analysis_name,
        "primary_sampling_unit": "prompt",
        "nested_sampling_unit": resolved_protocol["nested_sampling_unit"],
        "prompt_count": prompt_count,
        "prompt_id_digest": build_stable_digest(prompt_ids),
        "estimate": float(values.mean()),
        "confidence_interval_low": float(ci_low),
        "confidence_interval_high": float(ci_high),
        "confidence_level": float(resolved_protocol["confidence_level"]),
        "bootstrap_resample_count": resample_count,
        "bootstrap_bit_generator": resolved_protocol["bootstrap_bit_generator"],
        "bootstrap_quantile_method": resolved_protocol[
            "bootstrap_quantile_method"
        ],
        "bootstrap_seed_digest_random": seed_digest,
        "paper_quality_claim_protocol_digest": resolved_protocol[
            "paper_quality_claim_protocol_digest"
        ],
    }
    return {**core, "prompt_cluster_inference_digest": build_stable_digest(core)}


def build_quality_preservation_decisions(
    *,
    distributional_inference: Mapping[str, Any],
    paired_perceptual_inference: Mapping[str, Any] | None = None,
    semantic_alignment_inference: Mapping[str, Any] | None = None,
    per_attack_inference: Mapping[str, Mapping[str, Any]] | None = None,
    evidence_artifact_id: str,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """分别形成三类质量、逐攻击、跨攻击和总体质量三态决策。"""

    resolved_protocol = dict(protocol or load_paper_quality_claim_protocol())
    artifact_ids = (str(evidence_artifact_id),)

    def lower_bound_decision(
        claim_id: str,
        inference: Mapping[str, Any] | None,
        minimum: float,
        missing_reason: str,
    ) -> dict[str, Any]:
        if inference is None:
            return build_claim_decision(
                claim_id,
                evidence_complete=False,
                scientific_support=None,
                evidence_artifact_ids=artifact_ids,
                evidence_blockers=(missing_reason,),
            )
        return build_claim_decision(
            claim_id,
            evidence_complete=True,
            scientific_support=float(inference["confidence_interval_low"])
            >= minimum,
            evidence_artifact_ids=artifact_ids,
        )

    perceptual = lower_bound_decision(
        "paired_perceptual_quality_noninferiority",
        paired_perceptual_inference,
        float(
            resolved_protocol["paired_perceptual_quality_noninferiority"][
                "minimum_lower_confidence_bound"
            ]
        ),
        "paired_perceptual_prompt_observations_missing",
    )
    semantic = lower_bound_decision(
        "semantic_alignment_noninferiority",
        semantic_alignment_inference,
        float(
            resolved_protocol["semantic_alignment_noninferiority"][
                "minimum_lower_confidence_bound"
            ]
        ),
        "semantic_alignment_prompt_observations_missing",
    )
    distribution_margin = float(
        resolved_protocol["distributional_preservation_noninferiority"][
            "maximum_prompt_conditional_kid_upper_confidence_bound"
        ]
    )
    distribution = build_claim_decision(
        "distributional_preservation_noninferiority",
        evidence_complete=True,
        scientific_support=float(distributional_inference["confidence_interval_high"])
        <= distribution_margin,
        evidence_artifact_ids=artifact_ids,
    )
    attack_inputs = dict(per_attack_inference or {})
    per_attack_decisions: dict[str, dict[str, Any]] = {}
    for attack_id in resolved_protocol["registered_attack_ids"]:
        attack_components = attack_inputs.get(attack_id)
        if attack_components is None:
            per_attack_decisions[attack_id] = build_claim_decision(
                f"quality_preservation_under_attack:{attack_id}",
                evidence_complete=False,
                scientific_support=None,
                evidence_artifact_ids=artifact_ids,
                evidence_blockers=("per_attack_quality_observations_missing",),
            )
        else:
            expected_component_ids = {
                "paired_perceptual_quality_noninferiority",
                "semantic_alignment_noninferiority",
                "distributional_preservation_noninferiority",
            }
            if set(attack_components) != expected_component_ids:
                raise PaperQualityDecisionError(
                    f"逐攻击质量推断未覆盖三类子主张: {attack_id}"
                )
            per_attack_decisions[attack_id] = build_claim_decision(
                f"quality_preservation_under_attack:{attack_id}",
                evidence_complete=True,
                scientific_support=all(
                    (
                        float(
                            attack_components[
                                "paired_perceptual_quality_noninferiority"
                            ]["confidence_interval_low"]
                        )
                        >= float(
                            resolved_protocol[
                                "paired_perceptual_quality_noninferiority"
                            ]["minimum_lower_confidence_bound"]
                        ),
                        float(
                            attack_components[
                                "semantic_alignment_noninferiority"
                            ]["confidence_interval_low"]
                        )
                        >= float(
                            resolved_protocol[
                                "semantic_alignment_noninferiority"
                            ]["minimum_lower_confidence_bound"]
                        ),
                        float(
                            attack_components[
                                "distributional_preservation_noninferiority"
                            ]["confidence_interval_high"]
                        )
                        <= distribution_margin,
                    )
                ),
                evidence_artifact_ids=artifact_ids,
            )
    attack_complete = all(
        record["evidence_complete"] for record in per_attack_decisions.values()
    )
    cross_attack = build_claim_decision(
        "cross_attack_quality_preservation",
        evidence_complete=attack_complete,
        scientific_support=(
            all(record["scientific_support"] is True for record in per_attack_decisions.values())
            if attack_complete
            else None
        ),
        evidence_artifact_ids=artifact_ids,
        evidence_blockers=(
            () if attack_complete else ("registered_attack_quality_evidence_incomplete",)
        ),
    )
    subclaim_decisions = {
        "paired_perceptual_quality_noninferiority": perceptual,
        "semantic_alignment_noninferiority": semantic,
        "distributional_preservation_noninferiority": distribution,
    }
    required_components = [*subclaim_decisions.values(), cross_attack]
    quality_complete = all(
        record["evidence_complete"] for record in required_components
    )
    quality_decision = build_claim_decision(
        "quality_preservation",
        evidence_complete=quality_complete,
        scientific_support=(
            all(record["scientific_support"] is True for record in required_components)
            if quality_complete
            else None
        ),
        evidence_artifact_ids=artifact_ids,
        evidence_blockers=(
            () if quality_complete else ("quality_component_evidence_incomplete",)
        ),
    )
    return {
        "paper_quality_claim_protocol": resolved_protocol,
        "quality_subclaim_decisions": subclaim_decisions,
        "per_attack_quality_decisions": per_attack_decisions,
        "cross_attack_quality_decision": cross_attack,
        "quality_preservation_claim_decision": quality_decision,
    }
