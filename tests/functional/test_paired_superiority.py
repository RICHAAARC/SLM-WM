"""验证 Prompt-clustered 配对优势统计."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.write_paired_superiority_outputs as paired_writer
from experiments.protocol.attacks import default_attack_configs
from experiments.protocol.pilot_paper_fixed_fpr import (
    build_pilot_paper_attack_matrix_rows,
)
from main.core.digest import build_stable_digest
from paper_experiments.analysis.fixed_fpr_threshold_audit import (
    build_fixed_fpr_threshold_audit_report,
)
from paper_experiments.analysis.paired_superiority import (
    BOOTSTRAP_ANALYSIS_SCHEMA,
    BOOTSTRAP_BIT_GENERATOR,
    BOOTSTRAP_QUANTILE_METHOD,
    CLAIM_P_VALUE_METHOD,
    DEFAULT_BOOTSTRAP_RESAMPLE_COUNT,
    PRIMARY_BASELINE_IDS,
    SHARP_NULL_DIAGNOSTIC_METHOD,
    THRESHOLD_AUDIT_FIELDS,
    THRESHOLD_AUDIT_METHOD_IDS,
    PairedSuperiorityError,
    build_paired_outcome_set_digest,
    build_paired_outcomes,
    build_paired_superiority_protocol_digest,
    build_paired_superiority_rows,
    build_paired_superiority_summary,
    canonical_attack_registry_rows,
    canonical_threshold_audit_rows,
)


pytestmark = pytest.mark.quick

PROPOSED_THRESHOLD_DIGEST = "a" * 64
BASELINE_THRESHOLD_DIGESTS = {
    baseline_id: f"{index + 1:x}" * 64
    for index, baseline_id in enumerate(PRIMARY_BASELINE_IDS)
}
UNIT_ATTACK_REGISTRY = (
    {
        "attack_id": "jpeg_compression_main",
        "attack_family": "standard_distortion",
        "attack_name": "jpeg_compression",
        "resource_profile": "full_main",
        "attack_config_digest": "b" * 64,
    },
    {
        "attack_id": "gaussian_noise_main",
        "attack_family": "standard_distortion",
        "attack_name": "gaussian_noise",
        "resource_profile": "full_main",
        "attack_config_digest": "c" * 64,
    },
)


def observation_rows(
    *,
    baseline_id: str = "",
    baseline_positive_prompt_ids: set[str] | None = None,
    prompt_count: int = 3,
    attack_registry: tuple[dict[str, str], ...] = UNIT_ATTACK_REGISTRY,
) -> list[dict[str, object]]:
    """构造可配对的 Prompt x 正式攻击观测."""

    positive_ids = baseline_positive_prompt_ids or set()
    rows = []
    for prompt_index in range(prompt_count):
        prompt_id = f"prompt_{prompt_index}"
        for attack in attack_registry:
            decision = prompt_id in positive_ids if baseline_id else True
            row: dict[str, object] = {
                "prompt_id": prompt_id,
                "split": "test",
                "sample_role": (
                    "attacked_positive" if baseline_id else "positive_source"
                ),
                "attack_id": attack["attack_id"],
                "attack_family": attack["attack_family"],
                "attack_name": attack["attack_name"],
                "resource_profile": attack["resource_profile"],
                "attack_config_digest": attack["attack_config_digest"],
            }
            if baseline_id:
                row.update(
                    {
                        "baseline_id": baseline_id,
                        "threshold_digest": BASELINE_THRESHOLD_DIGESTS[
                            baseline_id
                        ],
                        "detection_decision": decision,
                    }
                )
            else:
                row.update(
                    {
                        "frozen_threshold_digest": PROPOSED_THRESHOLD_DIGEST,
                        "formal_evidence_positive": decision,
                    }
                )
            rows.append(row)
    return rows


def paired_outcomes(
    *,
    prompt_count: int = 3,
    baseline_positive_prompt_ids: set[str] | None = None,
) -> tuple[dict[str, object], ...]:
    """构造精确覆盖4个 baseline 的配对 outcome."""

    proposed = observation_rows(prompt_count=prompt_count)
    return tuple(
        outcome
        for baseline_id in PRIMARY_BASELINE_IDS
        for outcome in build_paired_outcomes(
            proposed,
            observation_rows(
                baseline_id=baseline_id,
                baseline_positive_prompt_ids=baseline_positive_prompt_ids,
                prompt_count=prompt_count,
            ),
            baseline_id=baseline_id,
            proposed_method_threshold_digest=PROPOSED_THRESHOLD_DIGEST,
            baseline_method_threshold_digest=BASELINE_THRESHOLD_DIGESTS[
                baseline_id
            ],
            attack_registry_rows=UNIT_ATTACK_REGISTRY,
        )
    )


def file_sha256(path: Path) -> str:
    """计算测试输入文件的原始字节 SHA-256."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_paired_outcomes_require_exact_prompt_attack_set() -> None:
    """baseline 缺少任一 Prompt x attack 键时必须阻断."""

    proposed = observation_rows()
    baseline = observation_rows(
        baseline_id="tree_ring",
        baseline_positive_prompt_ids={"prompt_0"},
    )
    baseline.pop()
    with pytest.raises(PairedSuperiorityError, match="配对集合不一致"):
        build_paired_outcomes(
            proposed,
            baseline,
            baseline_id="tree_ring",
            proposed_method_threshold_digest=PROPOSED_THRESHOLD_DIGEST,
            baseline_method_threshold_digest=BASELINE_THRESHOLD_DIGESTS[
                "tree_ring"
            ],
            attack_registry_rows=UNIT_ATTACK_REGISTRY,
        )


def test_paired_outcomes_only_accept_audited_decision_fields() -> None:
    """汇总判定不得替代主方法或 baseline 的正式审计判定字段."""

    proposed = observation_rows()
    baseline = observation_rows(baseline_id="tree_ring")
    proposed[0]["final_decision"] = proposed[0].pop("formal_evidence_positive")
    with pytest.raises(PairedSuperiorityError, match="formal_evidence_positive"):
        build_paired_outcomes(
            proposed,
            baseline,
            baseline_id="tree_ring",
            proposed_method_threshold_digest=PROPOSED_THRESHOLD_DIGEST,
            baseline_method_threshold_digest=BASELINE_THRESHOLD_DIGESTS[
                "tree_ring"
            ],
            attack_registry_rows=UNIT_ATTACK_REGISTRY,
        )

    proposed = observation_rows()
    baseline = observation_rows(baseline_id="tree_ring")
    baseline[0]["final_decision"] = baseline[0].pop("detection_decision")
    with pytest.raises(PairedSuperiorityError, match="detection_decision"):
        build_paired_outcomes(
            proposed,
            baseline,
            baseline_id="tree_ring",
            proposed_method_threshold_digest=PROPOSED_THRESHOLD_DIGEST,
            baseline_method_threshold_digest=BASELINE_THRESHOLD_DIGESTS[
                "tree_ring"
            ],
            attack_registry_rows=UNIT_ATTACK_REGISTRY,
        )

    baseline = observation_rows(baseline_id="tree_ring")
    baseline[0]["final_decision"] = not baseline[0]["detection_decision"]
    with pytest.raises(PairedSuperiorityError, match="不一致"):
        build_paired_outcomes(
            observation_rows(),
            baseline,
            baseline_id="tree_ring",
            proposed_method_threshold_digest=PROPOSED_THRESHOLD_DIGEST,
            baseline_method_threshold_digest=BASELINE_THRESHOLD_DIGESTS[
                "tree_ring"
            ],
            attack_registry_rows=UNIT_ATTACK_REGISTRY,
        )


def test_paired_outcomes_bind_thresholds_and_formal_attack_registry() -> None:
    """每条 outcome 必须绑定两方法阈值与正式攻击配置身份."""

    outcomes = build_paired_outcomes(
        observation_rows(),
        observation_rows(baseline_id="tree_ring"),
        baseline_id="tree_ring",
        proposed_method_threshold_digest=PROPOSED_THRESHOLD_DIGEST,
        baseline_method_threshold_digest=BASELINE_THRESHOLD_DIGESTS["tree_ring"],
        attack_registry_rows=UNIT_ATTACK_REGISTRY,
    )
    first = outcomes[0]
    assert first["proposed_method_threshold_digest"] == PROPOSED_THRESHOLD_DIGEST
    assert (
        first["baseline_method_threshold_digest"]
        == BASELINE_THRESHOLD_DIGESTS["tree_ring"]
    )
    registry_by_id = {row["attack_id"]: row for row in UNIT_ATTACK_REGISTRY}
    assert first["resource_profile"] == registry_by_id[first["attack_id"]][
        "resource_profile"
    ]
    assert first["attack_config_digest"] == registry_by_id[first["attack_id"]][
        "attack_config_digest"
    ]


def test_clustered_superiority_discloses_non_superior_result() -> None:
    """完整统计可披露不优于 baseline 的结果, 但不得通过优势门禁."""

    outcomes = paired_outcomes(
        baseline_positive_prompt_ids={"prompt_0", "prompt_1", "prompt_2"}
    )
    rows = build_paired_superiority_rows(
        outcomes,
        protocol_digest="d" * 64,
    )
    assert all(
        row["one_sided_bounded_hoeffding_mean_p_value"] == 1.0
        for row in rows
    )
    assert all(row["paired_superiority_ready"] is False for row in rows)
    summary = build_paired_superiority_summary(rows, paired_outcomes=outcomes)
    assert summary["overall_paired_superiority_ready"] is False


def test_bounded_mean_claim_and_exact_sharp_null_diagnostic() -> None:
    """正式 claim 使用 Hoeffding, exact sign-flip 只提供 sharp-null 诊断."""

    outcomes = paired_outcomes()
    first = build_paired_superiority_rows(
        outcomes,
        protocol_digest="e" * 64,
    )
    second = build_paired_superiority_rows(
        outcomes,
        protocol_digest="e" * 64,
    )
    assert first == second
    assert all(row["mean_paired_difference_ci_low"] == 1.0 for row in first)
    assert all(
        row["claim_p_value_method"] == CLAIM_P_VALUE_METHOD
        for row in first
    )
    assert all(
        row["sharp_null_diagnostic_method"] == SHARP_NULL_DIAGNOSTIC_METHOD
        for row in first
    )
    assert all(
        row["exact_prompt_cluster_sign_flip_p_value_is_diagnostic"] is True
        for row in first
    )
    assert all(
        row["one_sided_exact_prompt_cluster_sign_flip_p_value"] == 0.125
        for row in first
    )
    assert all(
        row["one_sided_bounded_hoeffding_mean_p_value"]
        == pytest.approx(0.22313016014842982)
        for row in first
    )
    assert all(
        row["bootstrap_resample_count"] == DEFAULT_BOOTSTRAP_RESAMPLE_COUNT
        for row in first
    )
    assert all(row["bootstrap_analysis_schema"] == BOOTSTRAP_ANALYSIS_SCHEMA for row in first)
    assert all(row["bootstrap_bit_generator"] == BOOTSTRAP_BIT_GENERATOR for row in first)
    assert all(row["bootstrap_quantile_method"] == BOOTSTRAP_QUANTILE_METHOD for row in first)
    assert all("permutation_resample_count" not in row for row in first)
    assert all("permutation_seed_digest_random" not in row for row in first)
    assert all(
        row["holm_adjusted_p_value"]
        >= row["one_sided_bounded_hoeffding_mean_p_value"]
        for row in first
    )
    assert all(row["paired_superiority_ready"] is False for row in first)

    outcome_set_digest = build_paired_outcome_set_digest(outcomes)
    prompt_id_digest = build_stable_digest(["prompt_0", "prompt_1", "prompt_2"])
    attack_registry_digest = build_stable_digest(
        list(canonical_attack_registry_rows(UNIT_ATTACK_REGISTRY))
    )
    expected_tree_seed = build_stable_digest(
        {
            "analysis_schema": BOOTSTRAP_ANALYSIS_SCHEMA,
            "baseline_id": "tree_ring",
            "paired_test_prompt_id_digest": prompt_id_digest,
            "paired_attack_registry_digest": attack_registry_digest,
            "paired_outcome_set_digest": outcome_set_digest,
            "confidence_level": 0.95,
            "bootstrap_resample_count": DEFAULT_BOOTSTRAP_RESAMPLE_COUNT,
            "bootstrap_bit_generator": BOOTSTRAP_BIT_GENERATOR,
            "bootstrap_quantile_method": BOOTSTRAP_QUANTILE_METHOD,
        }
    )
    tree_row = next(row for row in first if row["baseline_id"] == "tree_ring")
    assert tree_row["bootstrap_seed_digest_random"] == expected_tree_seed
    assert tree_row["paired_test_prompt_id_digest"] == prompt_id_digest
    assert tree_row["paired_attack_registry_digest"] == attack_registry_digest
    assert tree_row["paired_outcome_set_digest"] == outcome_set_digest
    assert tree_row["holm_adjusted_p_value"] == pytest.approx(
        min(1.0, 4.0 * tree_row["one_sided_bounded_hoeffding_mean_p_value"])
    )

    changed_protocol = build_paired_superiority_rows(
        outcomes,
        protocol_digest="f" * 64,
    )
    assert [row["bootstrap_seed_digest_random"] for row in changed_protocol] == [
        row["bootstrap_seed_digest_random"] for row in first
    ]
    assert [row["protocol_digest"] for row in changed_protocol] != [
        row["protocol_digest"] for row in first
    ]

    changed_outcomes = paired_outcomes(
        baseline_positive_prompt_ids={"prompt_0"}
    )
    changed_data = build_paired_superiority_rows(
        changed_outcomes,
        protocol_digest="e" * 64,
    )
    assert [row["bootstrap_seed_digest_random"] for row in changed_data] != [
        row["bootstrap_seed_digest_random"] for row in first
    ]

    with pytest.raises(PairedSuperiorityError, match="不得小于100000"):
        build_paired_superiority_rows(
            outcomes,
            protocol_digest="e" * 64,
            bootstrap_resample_count=99_999,
        )


def test_formal_writer_rejects_noncanonical_resample_counts(tmp_path: Path) -> None:
    """正式 writer 必须固定 bootstrap CI 精度和跨运行可比性."""

    with pytest.raises(ValueError, match="固定使用100000次"):
        paired_writer.write_paired_superiority_outputs(
            root=tmp_path,
            bootstrap_resample_count=20_000,
        )


def test_summary_requires_exact_four_baselines_and_binds_prompt_ids() -> None:
    """总体摘要同时约束 baseline exact set 与规范 test Prompt 摘要."""

    outcomes = paired_outcomes()
    rows = [
        {
            "baseline_id": baseline_id,
            "paired_superiority_ready": True,
        }
        for baseline_id in PRIMARY_BASELINE_IDS[:-1]
    ]
    summary = build_paired_superiority_summary(rows, paired_outcomes=outcomes)
    assert summary["paired_superiority_exact_set_ready"] is False
    assert summary["overall_paired_superiority_ready"] is False
    assert summary["paired_test_prompt_count"] == 3
    assert summary["paired_test_prompt_id_digest"] == build_stable_digest(
        ["prompt_0", "prompt_1", "prompt_2"]
    )


def test_protocol_digest_covers_canonical_threshold_rows_and_report() -> None:
    """threshold 行或报告任一事实变化都必须改变配对统计协议摘要."""

    rows = []
    for index, method_id in enumerate(reversed(THRESHOLD_AUDIT_METHOD_IDS)):
        rows.append(
            {
                "method_id": method_id,
                "threshold_source": "calibration_clean_negative_conformal",
                "target_fpr": "0.1",
                "calibration_clean_negative_count": "36",
                "test_clean_negative_count": "34",
                "calibrated_detection_threshold": "0.5",
                "threshold_digest": f"{index + 1:x}" * 64,
                "observation_source_sha256": f"{index + 6:x}" * 64,
                "protocol_target_ready": "True",
                "protocol_value_ready": "True",
                "detection_decision_ready": "True",
                "split_count_ready": "True",
                "fixed_fpr_threshold_ready": "True",
                "supports_paper_claim": "False",
            }
        )
    normalized_rows = canonical_threshold_audit_rows(rows)
    report = build_fixed_fpr_threshold_audit_report(
        normalized_rows,
        paper_run_name="probe_paper",
        target_fpr=0.1,
    )
    first = build_paired_superiority_protocol_digest(report, rows, "f" * 64)
    second = build_paired_superiority_protocol_digest(
        {**report, "target_fpr": 0.2},
        rows,
        "f" * 64,
    )
    assert first != second
    assert canonical_threshold_audit_rows(rows) == canonical_threshold_audit_rows(
        reversed(rows)
    )


def test_writer_closes_probe_scale_with_exact_raw_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writer 必须消费34个 test Prompt 的全部原始配对判定与审计字节绑定."""

    paper_run = SimpleNamespace(
        run_name="probe_paper",
        prompt_set="probe_paper",
        target_fpr=0.1,
        prompt_count=70,
    )
    fixed_fpr_config = SimpleNamespace(
        attack_resource_profiles=("full_main", "full_extra"),
        result_scope="probe_paper_common_protocol",
    )
    monkeypatch.setattr(paired_writer, "build_paper_run_config", lambda _root: paper_run)
    monkeypatch.setattr(
        paired_writer,
        "build_paper_fixed_fpr_config",
        lambda _root: fixed_fpr_config,
    )
    monkeypatch.setattr(
        paired_writer,
        "resolve_code_version",
        lambda _root: "a" * 40,
    )
    attack_registry = canonical_attack_registry_rows(
        build_pilot_paper_attack_matrix_rows(
            default_attack_configs(),
            fixed_fpr_config,
        )
    )

    proposed_path = (
        tmp_path
        / "outputs"
        / "image_only_dataset_runtime"
        / "probe_paper"
        / "image_only_detection_records.jsonl"
    )
    proposed_path.parent.mkdir(parents=True)
    proposed_rows = []
    for prompt_index in range(34):
        for attack in attack_registry:
            proposed_rows.append(
                {
                    "prompt_id": f"prompt_{prompt_index}",
                    "split": "test",
                    "sample_role": "positive_source",
                    "attack_id": attack["attack_id"],
                    "attack_family": attack["attack_family"],
                    "attack_name": attack["attack_name"],
                    "resource_profile": attack["resource_profile"],
                    "attack_config_digest": attack["attack_config_digest"],
                    "frozen_threshold_digest": PROPOSED_THRESHOLD_DIGEST,
                    "formal_evidence_positive": True,
                }
            )
    proposed_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in proposed_rows),
        encoding="utf-8",
    )

    collection_root = (
        tmp_path
        / "outputs"
        / "external_baseline_method_faithful"
        / "probe_paper"
    )
    split_root = collection_root / "split_observations"
    split_root.mkdir(parents=True)
    baseline_rows_by_id = {}
    observation_paths: dict[str, Path] = {"slm_wm": proposed_path}
    for baseline_id in PRIMARY_BASELINE_IDS:
        rows = [
            {
                "prompt_id": proposed["prompt_id"],
                "split": "test",
                "sample_role": "attacked_positive",
                "attack_id": proposed["attack_id"],
                "attack_family": proposed["attack_family"],
                "attack_name": proposed["attack_name"],
                "resource_profile": proposed["resource_profile"],
                "attack_config_digest": proposed["attack_config_digest"],
                "baseline_id": baseline_id,
                "threshold_digest": BASELINE_THRESHOLD_DIGESTS[baseline_id],
                "detection_decision": False,
            }
            for proposed in proposed_rows
        ]
        baseline_rows_by_id[baseline_id] = rows
        if baseline_id != "t2smark":
            path = split_root / f"{baseline_id}_baseline_observations.json"
            path.write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")
            observation_paths[baseline_id] = path
    t2_path = (
        tmp_path
        / "outputs"
        / "t2smark_formal_reproduction"
        / "probe_paper"
        / "t2smark_adapter"
        / "baseline_observations.json"
    )
    t2_path.parent.mkdir(parents=True)
    t2_path.write_text(
        json.dumps(baseline_rows_by_id["t2smark"], sort_keys=True),
        encoding="utf-8",
    )
    observation_paths["t2smark"] = t2_path
    observation_sha_map = {
        method_id: file_sha256(path)
        for method_id, path in sorted(observation_paths.items())
    }

    threshold_digest_map = {
        "slm_wm": PROPOSED_THRESHOLD_DIGEST,
        **BASELINE_THRESHOLD_DIGESTS,
    }
    threshold_rows = [
        {
            "method_id": method_id,
            "threshold_source": "calibration_clean_negative_conformal",
            "target_fpr": 0.1,
            "calibration_clean_negative_count": 36,
            "test_clean_negative_count": 34,
            "calibrated_detection_threshold": 0.5,
            "threshold_digest": threshold_digest_map[method_id],
            "observation_source_sha256": observation_sha_map[method_id],
            "protocol_target_ready": True,
            "protocol_value_ready": True,
            "detection_decision_ready": True,
            "split_count_ready": True,
            "fixed_fpr_threshold_ready": True,
            "supports_paper_claim": False,
        }
        for method_id in THRESHOLD_AUDIT_METHOD_IDS
    ]
    canonical_threshold_rows = canonical_threshold_audit_rows(threshold_rows)
    threshold_rows_digest = build_stable_digest(list(canonical_threshold_rows))
    threshold_root = (
        tmp_path / "outputs" / "fixed_fpr_threshold_audit" / "probe_paper"
    )
    threshold_root.mkdir(parents=True)
    with (threshold_root / "threshold_audit_rows.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=THRESHOLD_AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(threshold_rows)
    threshold_report = {
        "paper_claim_scale": "probe_paper",
        "target_fpr": 0.1,
        "expected_method_ids": list(THRESHOLD_AUDIT_METHOD_IDS),
        "audited_method_ids": list(THRESHOLD_AUDIT_METHOD_IDS),
        "audited_method_count": len(THRESHOLD_AUDIT_METHOD_IDS),
        "method_observation_source_sha256_map": observation_sha_map,
        "method_threshold_digest_map": threshold_digest_map,
        "threshold_audit_rows_digest": threshold_rows_digest,
        "method_identity_ready": True,
        "all_method_thresholds_ready": True,
        "threshold_observation_binding_ready": True,
        "fixed_fpr_threshold_audit_ready": True,
        "supports_paper_claim": True,
    }
    (threshold_root / "threshold_audit_report.json").write_text(
        json.dumps(threshold_report),
        encoding="utf-8",
    )
    threshold_manifest = {
        "artifact_id": "fixed_fpr_threshold_audit_manifest",
        "config_digest": build_stable_digest(
            paired_writer.build_fixed_fpr_threshold_manifest_config(
                threshold_report
            )
        ),
        "metadata": threshold_report,
    }
    (threshold_root / "manifest.local.json").write_text(
        json.dumps(threshold_manifest),
        encoding="utf-8",
    )

    manifest = paired_writer.write_paired_superiority_outputs(
        root=tmp_path,
        require_pass=True,
    )
    summary = manifest["metadata"]
    assert summary["paired_superiority_scale_ready"] is True
    assert summary["overall_paired_superiority_ready"] is True
    assert summary["paired_outcome_count"] == 34 * len(attack_registry) * 4
    assert summary["paired_test_prompt_count"] == 34
    assert len(summary["paired_test_prompt_id_digest"]) == 64
    assert summary["method_threshold_digest_map"] == threshold_digest_map
    assert summary["claim_p_value_method"] == CLAIM_P_VALUE_METHOD
    assert summary["sharp_null_diagnostic_method"] == SHARP_NULL_DIAGNOSTIC_METHOD
    assert summary["bootstrap_analysis_schema"] == BOOTSTRAP_ANALYSIS_SCHEMA
    assert summary["bootstrap_bit_generator"] == BOOTSTRAP_BIT_GENERATOR
    assert summary["bootstrap_quantile_method"] == BOOTSTRAP_QUANTILE_METHOD
    assert summary["bootstrap_resample_count"] == DEFAULT_BOOTSTRAP_RESAMPLE_COUNT
    assert summary["confidence_level"] == 0.95
    assert (
        summary["method_observation_source_sha256_map"] == observation_sha_map
    )
    assert (
        "outputs/image_only_dataset_runtime/probe_paper/"
        "image_only_detection_records.jsonl"
        in manifest["input_paths"]
    )
    assert (
        "outputs/fixed_fpr_threshold_audit/probe_paper/threshold_audit_rows.csv"
        in manifest["input_paths"]
    )
    outcome = json.loads(
        (
            tmp_path
            / "outputs"
            / "paired_superiority_analysis"
            / "probe_paper"
            / "paired_outcomes.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert outcome["proposed_method_threshold_digest"] == PROPOSED_THRESHOLD_DIGEST
    assert len(outcome["attack_config_digest"]) == 64

    table_path = (
        tmp_path
        / "outputs"
        / "paired_superiority_analysis"
        / "probe_paper"
        / "paired_superiority_table.csv"
    )
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        table_reader = csv.DictReader(handle)
        table_rows = list(table_reader)
        table_fields = tuple(table_reader.fieldnames or ())
    assert len(table_rows) == len(PRIMARY_BASELINE_IDS)
    assert "one_sided_bounded_hoeffding_mean_p_value" in table_fields
    assert "one_sided_exact_prompt_cluster_sign_flip_p_value" in table_fields
    assert "permutation_resample_count" not in table_fields
    assert "permutation_seed_digest_random" not in table_fields

    expected_manifest_config = {
        "paper_claim_scale": "probe_paper",
        "target_fpr": 0.1,
        "bootstrap_resample_count": DEFAULT_BOOTSTRAP_RESAMPLE_COUNT,
        "confidence_level": 0.95,
        "bootstrap_analysis_schema": BOOTSTRAP_ANALYSIS_SCHEMA,
        "bootstrap_bit_generator": BOOTSTRAP_BIT_GENERATOR,
        "bootstrap_quantile_method": BOOTSTRAP_QUANTILE_METHOD,
        "claim_p_value_method": CLAIM_P_VALUE_METHOD,
        "sharp_null_diagnostic_method": SHARP_NULL_DIAGNOSTIC_METHOD,
        "paired_outcome_set_digest": summary["paired_outcome_set_digest"],
        "paired_superiority_rows_digest": summary[
            "paired_superiority_rows_digest"
        ],
        "paired_superiority_protocol_digest": summary[
            "paired_superiority_protocol_digest"
        ],
        "paired_test_prompt_count": summary["paired_test_prompt_count"],
        "paired_test_prompt_id_digest": summary["paired_test_prompt_id_digest"],
        "paired_attack_registry_digest": summary["paired_attack_registry_digest"],
        "method_threshold_digest_map": threshold_digest_map,
        "method_observation_source_sha256_map": observation_sha_map,
        "method_observation_source_path_map": summary[
            "method_observation_source_path_map"
        ],
        "threshold_audit_rows_digest": summary["threshold_audit_rows_digest"],
    }
    assert manifest["config_digest"] == build_stable_digest(expected_manifest_config)

    tree_path = observation_paths["tree_ring"]
    tree_path.write_bytes(tree_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="字节摘要"):
        paired_writer.write_paired_superiority_outputs(
            root=tmp_path,
            output_dir="outputs/paired_superiority_analysis/tampered",
        )
