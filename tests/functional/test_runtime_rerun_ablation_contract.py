"""正式重运行消融8项规范的轻量协议测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from experiments.ablations.runtime_rerun import (
    FORMAL_RUNTIME_RERUN_ABLATION_IDS,
    FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
    default_runtime_rerun_ablation_specs,
    runtime_rerun_ablation_contract,
)


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
