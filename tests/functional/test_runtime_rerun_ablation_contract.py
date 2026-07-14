"""正式重运行消融15项规范的轻量协议测试."""

from __future__ import annotations

from dataclasses import replace

import pytest

from experiments.ablations.runtime_rerun import (
    FORMAL_RUNTIME_RERUN_ABLATION_IDS,
    FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
    _formal_attack_coverage_ready,
    default_runtime_rerun_ablation_specs,
    runtime_rerun_randomization_plan,
    runtime_rerun_ablation_contract,
)
from experiments.protocol.formal_randomization import (
    formal_runtime_randomization_plan_record,
)
from experiments.protocol.attacks import (
    attack_config_digest,
    default_attack_configs,
    formal_attack_seed_protocol_record,
    formal_attack_seed_random,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    _branch_risk_configs,
    semantic_watermark_runtime_config_payload,
)
from main.methods.semantic import build_branch_risk_fields


@pytest.mark.quick
def test_ablation_manifest_uses_complete_shared_randomization_plan() -> None:
    """消融顶层 manifest 必须声明与主方法相同的9重复计划."""

    plan = runtime_rerun_randomization_plan(
        SemanticWatermarkRuntimeConfig()
    )

    assert plan == formal_runtime_randomization_plan_record(1703)
    assert len(plan["repeat_records"]) == 9
    assert len(plan["watermark_key_records"]) == 3


@pytest.mark.quick
def test_formal_runtime_rerun_ablation_contract_is_exactly_fifteen_items() -> None:
    """正式消融必须覆盖机制开关与四个注意力分量留一对照."""

    contract = runtime_rerun_ablation_contract(default_runtime_rerun_ablation_specs())

    assert len(FORMAL_RUNTIME_RERUN_ABLATION_IDS) == 15
    assert len(set(FORMAL_RUNTIME_RERUN_ABLATION_IDS)) == 15
    assert {
        "shared_global_risk_routing",
        "lf_content_only",
        "tail_robust_only",
        "without_centered_qk_logit",
        "without_differentiable_row_rank",
        "without_attention_probability",
        "without_distance_modulated_probability",
    } <= set(FORMAL_RUNTIME_RERUN_ABLATION_IDS)
    assert contract["expected_ablation_ids"] == list(FORMAL_RUNTIME_RERUN_ABLATION_IDS)
    assert contract["actual_ablation_ids"] == list(FORMAL_RUNTIME_RERUN_ABLATION_IDS)
    assert contract["ablation_spec_digest"] == FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST
    assert contract["ablation_exact_set_ready"] is True


@pytest.mark.quick
def test_formal_runtime_rerun_ablation_contract_rejects_id_or_setting_drift() -> None:
    """缺失正式项或修改同名配置字段都必须使精确集合状态失败。"""

    specs = default_runtime_rerun_ablation_specs()
    six_item_contract = runtime_rerun_ablation_contract(specs[:6])
    changed_setting_contract = runtime_rerun_ablation_contract(
        (*specs[:-1], replace(specs[-1], image_alignment_enabled=True))
    )

    assert six_item_contract["ablation_exact_set_ready"] is False
    assert changed_setting_contract["actual_ablation_ids"] == list(
        FORMAL_RUNTIME_RERUN_ABLATION_IDS
    )
    assert changed_setting_contract["ablation_exact_set_ready"] is False


@pytest.mark.quick
def test_shared_global_risk_ablation_reuses_one_real_risk_field() -> None:
    """共享全局风险对照必须真实改变路由, 不能只更换消融标签。"""

    spec = next(
        item
        for item in default_runtime_rerun_ablation_specs()
        if item.ablation_id == "shared_global_risk_routing"
    )
    config = spec.apply(
        SemanticWatermarkRuntimeConfig(),
        "outputs/formal_mechanism_ablation/probe_paper",
    )
    fields = build_branch_risk_fields(
        semantic_values=(0.1, 0.7),
        texture_values=(0.2, 0.8),
        adjacent_step_stability_values=(0.9, 0.4),
        local_contrast_risk_values=(0.3, 0.6),
        attention_stability_values=(0.8, 0.5),
        configs=_branch_risk_configs(config),
        risk_neutral_texture_value=config.risk_neutral_texture_value,
    )

    assert config.branch_risk_mode == "shared_global"
    assert fields.lf_content.risk_values == fields.tail_robust.risk_values
    assert fields.tail_robust.risk_values == fields.attention_geometry.risk_values


@pytest.mark.quick
def test_single_carrier_ablations_disable_attention_and_other_content_branch() -> None:
    """LF-only 与 Tail-only 必须形成真实单载体重运行配置。"""

    specs = {
        spec.ablation_id: spec
        for spec in default_runtime_rerun_ablation_specs()
    }
    base = SemanticWatermarkRuntimeConfig()
    lf_only = specs["lf_content_only"].apply(base, "outputs/ablations")
    tail_only = specs["tail_robust_only"].apply(base, "outputs/ablations")

    assert (lf_only.lf_enabled, lf_only.tail_robust_enabled) == (True, False)
    assert (tail_only.lf_enabled, tail_only.tail_robust_enabled) == (False, True)
    assert lf_only.attention_geometry_enabled is False
    assert tail_only.attention_geometry_enabled is False
    assert lf_only.image_alignment_enabled is False
    assert tail_only.image_alignment_enabled is False


@pytest.mark.quick
def test_attention_and_alignment_ablations_preserve_dependency_direction() -> None:
    """移除 attention 必须同时移除 alignment, 仅移除 alignment 则保留 raw Q/K."""

    specs = {
        spec.ablation_id: spec
        for spec in default_runtime_rerun_ablation_specs()
    }
    without_attention = specs["without_attention_geometry"]
    without_alignment = specs["without_image_alignment"]

    assert without_attention.attention_geometry_enabled is False
    assert without_attention.image_alignment_enabled is False
    assert without_alignment.attention_geometry_enabled is True
    assert without_alignment.image_alignment_enabled is False


@pytest.mark.quick
def test_four_attention_component_ablations_zero_exactly_one_weight() -> None:
    """四个分量对照必须逐项置零并对其余三个分量重新归一化."""

    component_ids = (
        "without_centered_qk_logit",
        "without_differentiable_row_rank",
        "without_attention_probability",
        "without_distance_modulated_probability",
    )
    specs = {
        spec.ablation_id: spec
        for spec in default_runtime_rerun_ablation_specs()
    }
    base = SemanticWatermarkRuntimeConfig()

    for removed_index, ablation_id in enumerate(component_ids):
        spec = specs[ablation_id]
        weights = spec.attention_relation_component_weights
        runtime_config = spec.apply(base, "outputs/formal_mechanism_ablation")
        assert len(weights) == 4
        assert weights[removed_index] == 0.0
        assert sum(weight == 0.0 for weight in weights) == 1
        assert sum(weights) == pytest.approx(1.0)
        assert all(
            weight == pytest.approx(1.0 / 3.0)
            for index, weight in enumerate(weights)
            if index != removed_index
        )
        assert runtime_config.attention_relation_component_weights == weights
        assert semantic_watermark_runtime_config_payload(runtime_config)[
            "attention_relation_component_weights"
        ] == list(weights)


def _formal_attack_records() -> tuple[dict[str, object], ...]:
    """构造一个 test Prompt 的完整正式攻击身份记录."""

    generation_seed_random = 1703
    attack_seed_protocol_digest = formal_attack_seed_protocol_record()[
        "formal_attack_seed_protocol_digest"
    ]
    return tuple(
        {
            "attack_id": config.attack_id,
            "attack_family": config.attack_family,
            "attack_name": config.attack_name,
            "resource_profile": config.resource_profile,
            "attack_config_digest": attack_config_digest(config),
            "attack_parameters": config.attack_parameters,
            "attack_performed": True,
            "sample_role": sample_role,
            "generation_seed_random": generation_seed_random,
            "attack_seed_random": formal_attack_seed_random(
                generation_seed_random,
                config.attack_id,
            ),
            "formal_attack_seed_protocol_digest": (
                attack_seed_protocol_digest
            ),
            **(
                {
                    "detector_guided_attack_threshold_digest": (
                        FORMAL_ATTACK_THRESHOLD_DIGEST
                    )
                }
                if config.attack_name == "adversarial_removal_attack"
                else {}
            ),
        }
        for config in default_attack_configs()
        if config.enabled and config.resource_profile in {"full_main", "full_extra"}
        for sample_role in ("clean_negative", "positive_source")
    )


FORMAL_ATTACK_THRESHOLD_DIGEST = "d" * 64


@pytest.mark.quick
def test_formal_ablation_attack_coverage_is_exact_on_test_only() -> None:
    """test split 必须覆盖完整攻击笛卡尔积, 其他 split 必须保持无攻击."""

    records = _formal_attack_records()

    assert _formal_attack_coverage_ready(
        records,
        split="test",
        expected_generation_seed_random=1703,
        expected_threshold_digest=FORMAL_ATTACK_THRESHOLD_DIGEST,
    ) is True
    assert _formal_attack_coverage_ready(
        (),
        split="calibration",
        expected_generation_seed_random=1703,
        expected_threshold_digest=FORMAL_ATTACK_THRESHOLD_DIGEST,
    ) is True
    assert _formal_attack_coverage_ready(
        records,
        split="calibration",
        expected_generation_seed_random=1703,
        expected_threshold_digest=FORMAL_ATTACK_THRESHOLD_DIGEST,
    ) is False


@pytest.mark.quick
def test_formal_ablation_attack_coverage_rejects_missing_or_duplicate_record() -> None:
    """任一攻击角色缺失或重复时不得形成正式消融攻击证据."""

    records = _formal_attack_records()

    assert _formal_attack_coverage_ready(
        records[:-1],
        split="test",
        expected_generation_seed_random=1703,
        expected_threshold_digest=FORMAL_ATTACK_THRESHOLD_DIGEST,
    ) is False
    assert _formal_attack_coverage_ready(
        (*records, records[0]),
        split="test",
        expected_generation_seed_random=1703,
        expected_threshold_digest=FORMAL_ATTACK_THRESHOLD_DIGEST,
    ) is False


@pytest.mark.quick
def test_formal_ablation_attack_coverage_rejects_seed_drift() -> None:
    """消融攻击 seed 必须由生成 seed 与攻击 ID 的统一公式重建."""

    records = [dict(record) for record in _formal_attack_records()]
    records[0]["attack_seed_random"] = int(
        records[0]["attack_seed_random"]
    ) + 1

    assert _formal_attack_coverage_ready(
        tuple(records),
        split="test",
        expected_generation_seed_random=1703,
        expected_threshold_digest=FORMAL_ATTACK_THRESHOLD_DIGEST,
    ) is False


@pytest.mark.quick
def test_formal_ablation_attack_coverage_rejects_joint_seed_rewrite() -> None:
    """攻击 seed 与错误生成 seed 同步重算也不得脱离规范样本身份."""

    records = [dict(record) for record in _formal_attack_records()]
    for record in records:
        record["generation_seed_random"] = 1704
        record["attack_seed_random"] = formal_attack_seed_random(
            1704,
            str(record["attack_id"]),
        )

    assert _formal_attack_coverage_ready(
        tuple(records),
        split="test",
        expected_generation_seed_random=1703,
        expected_threshold_digest=FORMAL_ATTACK_THRESHOLD_DIGEST,
    ) is False


@pytest.mark.quick
def test_formal_ablation_attack_coverage_rejects_guided_threshold_drift() -> None:
    """检测器引导移除攻击必须绑定当前消融的冻结阈值摘要."""

    records = [dict(record) for record in _formal_attack_records()]
    guided_record = next(
        record
        for record in records
        if record["attack_name"] == "adversarial_removal_attack"
    )
    guided_record["detector_guided_attack_threshold_digest"] = "e" * 64

    assert _formal_attack_coverage_ready(
        tuple(records),
        split="test",
        expected_generation_seed_random=1703,
        expected_threshold_digest=FORMAL_ATTACK_THRESHOLD_DIGEST,
    ) is False
