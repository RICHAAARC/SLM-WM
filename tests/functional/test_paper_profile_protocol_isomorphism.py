"""论文 profile 协议同构、科学作用域和流程迁移派生测试。"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from paper_experiments.analysis.paper_claim_decisions import (
    build_claim_decision,
    build_claim_decision_bundle,
    load_paper_claim_registry,
)
from paper_experiments.analysis.paper_profile_protocol_isomorphism import (
    PaperProfileProtocolError,
    build_paper_profile_protocol_isomorphism_report,
    build_paper_profile_protocol_records,
    validate_paper_profile_protocol_isomorphism_report,
)
import scripts.write_paper_profile_protocol_isomorphism_report as writer_module


pytestmark = pytest.mark.quick


def _probe_closure_report(
    *,
    scientific_support: bool,
    code_version: str = "a" * 40,
) -> dict[str, object]:
    """构造流程闭合但科学结论可正可负的最小 probe 报告。"""

    registry = load_paper_claim_registry()
    bundle = build_claim_decision_bundle(
        {
            claim_id: build_claim_decision(
                claim_id,
                evidence_complete=True,
                scientific_support=scientific_support,
                evidence_artifact_ids=(f"{claim_id}_artifact",),
            )
            for claim_id in registry["registered_claim_ids"]
        },
        registry=registry,
    )
    return {
        "paper_claim_scale": "probe_paper",
        "target_fpr": 0.1,
        "common_code_version": code_version,
        "closure_check_count": 12,
        "blocked_check_count": 0,
        "evidence_closure_allowed": True,
        "result_closure_ready": True,
        **bundle,
    }


def test_registered_profiles_are_isomorphic_and_keep_science_scoped() -> None:
    """同一协议可迁移不表示 probe 科学结论外推到更严格 FPR。"""

    report = build_paper_profile_protocol_isomorphism_report(
        _probe_closure_report(scientific_support=True)
    )

    assert report["protocol_isomorphism_ready"] is True
    assert report["artifact_contract_isomorphic"] is True
    assert report["profile_scale_registration_ready"] is True
    assert report["workflow_transfer_ready"] is True
    assert report["scientific_scope_by_profile"]["probe_paper"][
        "registered_claim_set_scientific_support"
    ] is True
    for profile_id, target_fpr in (("pilot_paper", 0.01), ("full_paper", 0.001)):
        scope = report["scientific_scope_by_profile"][profile_id]
        assert scope["target_fpr"] == target_fpr
        assert scope["registered_claim_set_decision"] == "evidence_incomplete"
        assert scope["registered_claim_set_scientific_support"] is None
        assert scope["scientific_support_transferred_from_probe"] is False
    validate_paper_profile_protocol_isomorphism_report(report)


def test_measured_negative_probe_still_proves_workflow_transfer() -> None:
    """真实负结果不应否决已闭合且同构的代码与协议迁移能力。"""

    report = build_paper_profile_protocol_isomorphism_report(
        _probe_closure_report(scientific_support=False)
    )

    assert report["probe_workflow_closed"] is True
    assert report["scientific_scope_by_profile"]["probe_paper"][
        "registered_claim_set_decision"
    ] == "measured_not_supported"
    assert report["workflow_transfer_ready"] is True


@pytest.mark.parametrize(
    ("drift_kind", "expected_path"),
    (
        ("method", "core_method.formal_method_config.guidance_scale"),
        ("attack", "attack_definitions.attack_records[0].attack_strength"),
        (
            "baseline",
            "baseline_definitions.primary_baseline_records[0].baseline_name",
        ),
        ("gate", "gate_roles[0].gate_role"),
    ),
)
def test_semantic_drift_breaks_protocol_isomorphism(
    drift_kind: str,
    expected_path: str,
) -> None:
    """方法、攻击、baseline 或 gate 的任一 profile 分叉都必须失败。"""

    records = build_paper_profile_protocol_records()
    pilot = records["pilot_paper"]["protocol_contract"]
    if drift_kind == "method":
        pilot["core_method"]["formal_method_config"]["guidance_scale"] = 99.0
    elif drift_kind == "attack":
        pilot["attack_definitions"]["attack_records"][0]["attack_strength"] = 99.0
    elif drift_kind == "baseline":
        pilot["baseline_definitions"]["primary_baseline_records"][0][
            "baseline_name"
        ] = "drifted-baseline"
    else:
        pilot["gate_roles"][0]["gate_role"] = "drifted_gate"

    report = build_paper_profile_protocol_isomorphism_report(
        _probe_closure_report(scientific_support=True),
        profile_records=records,
    )

    assert report["protocol_isomorphism_ready"] is False
    assert report["workflow_transfer_ready"] is False
    assert expected_path in report["protocol_difference_paths"]["pilot_paper"]


def test_artifact_contract_drift_blocks_transfer_independently() -> None:
    """产物 schema 漂移必须由独立契约门禁阻断流程迁移。"""

    records = build_paper_profile_protocol_records()
    records["full_paper"]["artifact_contract"][0]["file_names"][0] = (
        "drifted_records.jsonl"
    )

    report = build_paper_profile_protocol_isomorphism_report(
        _probe_closure_report(scientific_support=True),
        profile_records=records,
    )

    assert report["protocol_isomorphism_ready"] is True
    assert report["artifact_contract_isomorphic"] is False
    assert report["workflow_transfer_ready"] is False


def test_allowed_scale_change_does_not_change_normalized_protocol() -> None:
    """Prompt 数量等允许字段只影响登记核验, 不伪造协议语义差异。"""

    records = build_paper_profile_protocol_records()
    scale = records["pilot_paper"]["scale_contract"]
    scale["prompt_count"] = 701
    scale["record_count_derivation"]["prompt_primary_unit_count"] = 701

    report = build_paper_profile_protocol_isomorphism_report(
        _probe_closure_report(scientific_support=True),
        profile_records=records,
    )

    assert report["protocol_isomorphism_ready"] is True
    assert report["artifact_contract_isomorphic"] is True
    assert report["profile_scale_registration_ready"] is False
    assert report["scale_registration_difference_paths"]["pilot_paper"]


def test_report_validator_rejects_forged_transfer_state() -> None:
    """workflow_transfer_ready 只能由三个独立门禁重新派生。"""

    report = build_paper_profile_protocol_isomorphism_report(
        _probe_closure_report(scientific_support=False)
    )
    forged = {**report, "workflow_transfer_ready": False}

    with pytest.raises(PaperProfileProtocolError, match="重算结果不一致"):
        validate_paper_profile_protocol_isomorphism_report(forged)


def test_writer_materializes_report_and_manifest_under_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """独立 CLI writer 在脱离 Notebook 时写出同一报告与 manifest。"""

    code_version = "b" * 40
    source_report = _probe_closure_report(
        scientific_support=False,
        code_version=code_version,
    )
    source_path = tmp_path / "probe_result_closure_report.json"
    source_path.write_text(
        json.dumps(source_report, ensure_ascii=False),
        encoding="utf-8",
    )
    records = build_paper_profile_protocol_records()
    monkeypatch.setattr(writer_module, "resolve_code_version", lambda _root: code_version)
    monkeypatch.setattr(
        writer_module,
        "build_paper_profile_protocol_records",
        lambda _root: deepcopy(records),
    )

    manifest = writer_module.write_paper_profile_protocol_isomorphism_report(
        root=tmp_path,
        probe_result_closure_report_path=source_path,
    )

    output_dir = tmp_path / "outputs" / "paper_profile_protocol_isomorphism"
    report = json.loads(
        (output_dir / writer_module.REPORT_FILE_NAME).read_text(encoding="utf-8")
    )
    assert report["workflow_transfer_ready"] is True
    assert manifest["metadata"]["workflow_transfer_ready"] is True
    assert (output_dir / writer_module.MANIFEST_FILE_NAME).is_file()
