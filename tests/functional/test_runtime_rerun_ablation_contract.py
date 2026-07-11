"""正式重运行消融8项规范的轻量协议测试。"""

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


@pytest.mark.quick
def test_formal_runtime_rerun_ablation_contract_is_exactly_eight_items() -> None:
    """当前正式消融必须完整覆盖8个唯一机制配置。"""

    contract = runtime_rerun_ablation_contract(default_runtime_rerun_ablation_specs())

    assert len(FORMAL_RUNTIME_RERUN_ABLATION_IDS) == 8
    assert len(set(FORMAL_RUNTIME_RERUN_ABLATION_IDS)) == 8
    assert contract["expected_ablation_ids"] == list(FORMAL_RUNTIME_RERUN_ABLATION_IDS)
    assert contract["actual_ablation_ids"] == list(FORMAL_RUNTIME_RERUN_ABLATION_IDS)
    assert contract["ablation_spec_digest"] == FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST
    assert contract["ablation_exact_set_ready"] is True


@pytest.mark.quick
def test_formal_runtime_rerun_ablation_contract_rejects_id_or_setting_drift() -> None:
    """仅保留6项或修改同名配置字段都必须使精确集合状态失败。"""

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
