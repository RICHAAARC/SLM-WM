"""正式重运行消融11项规范的轻量协议测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from experiments.ablations.runtime_rerun import (
    FORMAL_RUNTIME_RERUN_ABLATION_IDS,
    FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
    _formal_attack_coverage_ready,
    default_runtime_rerun_ablation_specs,
    runtime_rerun_ablation_contract,
)
from experiments.protocol.attacks import attack_config_digest, default_attack_configs
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    _branch_risk_configs,
)
from main.methods.semantic import build_branch_risk_fields


@pytest.mark.quick
def test_formal_runtime_rerun_ablation_contract_is_exactly_eleven_items() -> None:
    """正式消融必须覆盖分支风险、单载体与逐机制移除配置。"""

    contract = runtime_rerun_ablation_contract(default_runtime_rerun_ablation_specs())

    assert len(FORMAL_RUNTIME_RERUN_ABLATION_IDS) == 11
    assert len(set(FORMAL_RUNTIME_RERUN_ABLATION_IDS)) == 11
    assert {
        "shared_global_risk_routing",
        "lf_content_only",
        "tail_robust_only",
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
        stability_values=(0.9, 0.4),
        saliency_values=(0.3, 0.6),
        attention_stability_values=(0.8, 0.5),
        configs=_branch_risk_configs(config),
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


def _formal_attack_records() -> tuple[dict[str, object], ...]:
    """构造一个 test Prompt 的完整正式攻击身份记录."""

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
        }
        for config in default_attack_configs()
        if config.enabled and config.resource_profile in {"full_main", "full_extra"}
        for sample_role in ("clean_negative", "positive_source")
    )


@pytest.mark.quick
def test_formal_ablation_attack_coverage_is_exact_on_test_only() -> None:
    """test split 必须覆盖完整攻击笛卡尔积, 其他 split 必须保持无攻击."""

    records = _formal_attack_records()

    assert _formal_attack_coverage_ready(records, split="test") is True
    assert _formal_attack_coverage_ready((), split="calibration") is True
    assert _formal_attack_coverage_ready(records, split="calibration") is False


@pytest.mark.quick
def test_formal_ablation_attack_coverage_rejects_missing_or_duplicate_record() -> None:
    """任一攻击角色缺失或重复时不得形成正式消融攻击证据."""

    records = _formal_attack_records()

    assert _formal_attack_coverage_ready(records[:-1], split="test") is False
    assert _formal_attack_coverage_ready((*records, records[0]), split="test") is False
