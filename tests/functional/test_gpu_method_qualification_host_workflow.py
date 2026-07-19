"""验证 fresh-host 到隔离 GPU 方法资格化入口的完整编排接线."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.protocol.gpu_method_qualification import (
    GPU_METHOD_QUALIFICATION_SCHEMA,
)
from experiments.protocol.image_only_evidence import (
    FrozenEvidenceProtocol,
    apply_frozen_evidence_protocol,
)
from experiments.runners.image_only_dataset_runtime import (
    calibrate_complete_evidence_protocol,
)
from main.core.digest import build_stable_digest
from scripts import formal_workflow_entry
from scripts import gpu_method_qualification_host_workflow as workflow
from scripts import run_formal_workflow_host as host
from tests.helpers.formal_detection_record import bind_formal_detection_record


def _sha256(path: Path) -> str:
    """计算测试资格化报告的真实文件摘要."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualification_report(
    *,
    root: Path,
    repository_commit: str,
    paper_run_name: str,
    prompt_id: str,
    operator_ready: bool,
    protocol: FrozenEvidenceProtocol,
    protocol_path: Path,
    formal_detection_path: Path,
    formal_detection_count: int,
) -> dict[str, object]:
    """构造满足宿主交叉复验规则的最小资格化报告."""

    report: dict[str, object] = {
        "qualification_report_schema": GPU_METHOD_QUALIFICATION_SCHEMA,
        "qualification_binding": {
            "code_version": repository_commit,
            "dependency_profile_id": workflow.SCIENTIFIC_PROFILE_ID,
            "input_summary": {
                "paper_run_name": paper_run_name,
                "prompt_id": prompt_id,
            },
            "frozen_evidence_protocol_identity": {
                "source_path": protocol_path.as_posix(),
                "source_file_sha256": _sha256(protocol_path),
                "threshold_digest": protocol.threshold_digest,
                "image_only_measurement_config_digest": (
                    protocol.image_only_measurement_config_digest
                ),
            },
            "formal_detection_records_identity": {
                "path": formal_detection_path.relative_to(root).as_posix(),
                "file_sha256": _sha256(formal_detection_path),
                "record_count": formal_detection_count,
            },
        },
        "gpu_operator_preflight_ready": operator_ready,
        "gpu_resource_budget_ready": False,
        "supports_paper_claim": False,
    }
    report["qualification_report_digest"] = build_stable_digest(report)
    return report


def _frozen_protocol_evidence(
    root: Path,
    report_dir: Path,
) -> tuple[FrozenEvidenceProtocol, Path, Path, int]:
    """构造真实完整协议及其正式检测记录，供宿主证据测试消费。"""

    raw_records = tuple(
        bind_formal_detection_record(
            {
                "prompt_id": f"calibration-{index}",
                "split": "calibration",
                "sample_role": "clean_negative",
                "detection_key_role": "registered_watermark_key",
                "attack_id": "",
                "content_score": score,
                "aligned_content_score": score,
                "attention_geometry_score": 0.0,
                "registration_confidence": 0.0,
                "attention_sync_score": 0.0,
                "geometry_reliable": False,
                "alignment": {"registration_geometry_reliable": False},
            }
        )
        for index, score in enumerate((0.1, 0.2, 0.3))
    )
    protocol = calibrate_complete_evidence_protocol(
        raw_records,
        target_fpr=0.25,
    )
    input_dir = root / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = (input_dir / "frozen_evidence_protocol.json").resolve()
    protocol_path.write_text(
        json.dumps(protocol.to_dict(), ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    formal_records = apply_frozen_evidence_protocol(raw_records, protocol)
    formal_detection_path = (
        report_dir / "formal_image_only_detection_records.jsonl"
    ).resolve()
    formal_detection_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in formal_records
        ),
        encoding="utf-8",
    )
    return protocol, protocol_path, formal_detection_path, len(formal_records)


@pytest.mark.quick
@pytest.mark.parametrize("operator_ready", (False, True))
def test_host_workflow_uses_exact_sd35_child_and_operator_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operator_ready: bool,
) -> None:
    """资源超限不得篡改方法结论, 方法门禁必须传回宿主状态."""

    root = tmp_path / "repository"
    report_dir = root / "outputs/gpu_method_qualification/runtime_run"
    report_dir.mkdir(parents=True)
    repository_commit = "a" * 40
    paper_run_name = "probe_paper"
    prompt_id = "probe_prompt_0001"
    (
        protocol,
        protocol_path,
        formal_detection_path,
        formal_detection_count,
    ) = _frozen_protocol_evidence(root, report_dir)
    qualification_report = _qualification_report(
        root=root,
        repository_commit=repository_commit,
        paper_run_name=paper_run_name,
        prompt_id=prompt_id,
        operator_ready=operator_ready,
        protocol=protocol,
        protocol_path=protocol_path,
        formal_detection_path=formal_detection_path,
        formal_detection_count=formal_detection_count,
    )
    qualification_report_path = (
        report_dir / "gpu_method_qualification_report.json"
    )
    qualification_report_path.write_text(
        json.dumps(qualification_report, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    invocation = {
        "report_schema": workflow.QUALIFICATION_INVOCATION_RESULT_SCHEMA,
        "schema_version": 1,
        "gpu_method_qualification_report_path": (
            qualification_report_path.relative_to(root).as_posix()
        ),
        "gpu_method_qualification_report_sha256": _sha256(
            qualification_report_path
        ),
        "gpu_method_qualification_report_digest": qualification_report[
            "qualification_report_digest"
        ],
        "gpu_operator_preflight_ready": operator_ready,
        "gpu_resource_budget_ready": False,
        "frozen_threshold_digest": protocol.threshold_digest,
        "formal_detection_record_path": formal_detection_path.relative_to(
            root
        ).as_posix(),
        "formal_detection_record_sha256": _sha256(formal_detection_path),
        "supports_paper_claim": False,
    }
    captured: dict[str, object] = {}

    def fake_execute_isolated(
        profile_id,
        child_argv,
        *,
        execution_report_path,
        repository_root,
    ):
        """物化真实格式的隔离执行报告, 不伪装为 CUDA 科学证据."""

        captured.update(
            {
                "profile_id": profile_id,
                "child_argv": list(child_argv),
                "repository_root": Path(repository_root),
            }
        )
        isolated_report = {
            "report_schema": "isolated_scientific_execution_report",
            "schema_version": 1,
            "profile_id": profile_id,
            "profile_digest": "b" * 64,
            "direct_requirements_digest": "c" * 64,
            "complete_hash_lock_digest": "d" * 64,
            "complete_hash_lock_dependency_count": 10,
            "dependency_environment_report_path": (
                "outputs/dependency_profiles/sd35/report.json"
            ),
            "dependency_environment_report_digest": "e" * 64,
            "dependency_environment_report_valid": True,
            "python_executable_path": "/managed/python",
            "python_executable_sha256": "f" * 64,
            "formal_execution_commit": repository_commit,
            "formal_execution_lock_ready": True,
            "formal_execution_lock_revalidated_before_child": True,
            "formal_execution_lock_revalidated_after_child": True,
            "python_executable_revalidated_before_child": True,
            "python_executable_revalidated_after_child": True,
            "dependency_environment_report_revalidated_before_child": True,
            "dependency_environment_report_revalidated_after_child": True,
            "execution": {
                "return_code": 0 if operator_ready else 1,
                "stdout": "runtime log\n" + json.dumps(invocation),
                "stderr": "",
            },
            "decision": "pass" if operator_ready else "fail",
            "failure_reasons": (
                [] if operator_ready else ["scientific_child_command_failed"]
            ),
            "supports_paper_claim": False,
        }
        persisted_path = Path(execution_report_path)
        persisted_path.parent.mkdir(parents=True, exist_ok=True)
        persisted_path.write_text(
            json.dumps(isolated_report, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return isolated_report, persisted_path

    monkeypatch.setattr(
        workflow,
        "execute_isolated_scientific_command",
        fake_execute_isolated,
    )
    result = workflow.run_gpu_method_qualification_host_workflow(
        root=root,
        repository_commit=repository_commit,
        paper_run_name=paper_run_name,
        prompt_id=prompt_id,
        result_path="outputs/host/qualification_result.json",
        known_answer="configs/keyed_prg_cross_platform_known_answer.json",
        qualification_output_root="outputs/gpu_method_qualification",
        frozen_evidence_protocol=protocol_path,
        reference_gradient=1.0,
        reference_response=0.5,
        reference_sensitivity=0.25,
    )

    assert captured["profile_id"] == "sd35_method_runtime_gpu"
    assert captured["repository_root"] == root.resolve()
    child_argv = captured["child_argv"]
    assert child_argv[0] == str(
        root.resolve() / "scripts/run_gpu_method_qualification.py"
    )
    assert child_argv[child_argv.index("--prompt-id") + 1] == prompt_id
    assert child_argv[child_argv.index("--frozen-evidence-protocol") + 1] == str(
        protocol_path
    )
    assert child_argv[child_argv.index("--reference-gradient") + 1] == "1.0"
    assert result["decision"] == ("pass" if operator_ready else "fail")
    assert result["return_code"] == (0 if operator_ready else 1)
    assert result["workflow_summary"]["gpu_resource_budget_ready"] is False
    assert result["supports_paper_claim"] is False


@pytest.mark.quick
@pytest.mark.parametrize(
    "mutation",
    (
        "formal_path_outside_qualification_root",
        "formal_record_missing",
        "formal_record_sha",
        "invocation_missing_formal_sha",
        "invocation_threshold_digest",
        "report_protocol_threshold_digest",
        "formal_record_threshold_digest",
    ),
)
def test_host_rejects_formal_detection_and_protocol_identity_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    """宿主必须独立复验正式记录路径、文件与冻结阈值身份。"""

    root = tmp_path / "repository"
    report_dir = root / "outputs/gpu_method_qualification/runtime_run"
    report_dir.mkdir(parents=True)
    (
        protocol,
        protocol_path,
        formal_detection_path,
        formal_detection_count,
    ) = _frozen_protocol_evidence(root, report_dir)
    report = _qualification_report(
        root=root,
        repository_commit="a" * 40,
        paper_run_name="probe_paper",
        prompt_id="probe_prompt_0001",
        operator_ready=True,
        protocol=protocol,
        protocol_path=protocol_path,
        formal_detection_path=formal_detection_path,
        formal_detection_count=formal_detection_count,
    )
    report_path = report_dir / "gpu_method_qualification_report.json"
    invocation: dict[str, object] = {
        "report_schema": workflow.QUALIFICATION_INVOCATION_RESULT_SCHEMA,
        "schema_version": 1,
        "gpu_method_qualification_report_path": report_path.relative_to(
            root
        ).as_posix(),
        "gpu_method_qualification_report_sha256": "",
        "gpu_method_qualification_report_digest": "",
        "gpu_operator_preflight_ready": True,
        "gpu_resource_budget_ready": False,
        "frozen_threshold_digest": protocol.threshold_digest,
        "formal_detection_record_path": formal_detection_path.relative_to(
            root
        ).as_posix(),
        "formal_detection_record_sha256": _sha256(formal_detection_path),
        "supports_paper_claim": False,
    }

    def persist_report() -> None:
        report.pop("qualification_report_digest", None)
        report["qualification_report_digest"] = build_stable_digest(report)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        invocation["gpu_method_qualification_report_sha256"] = _sha256(
            report_path
        )
        invocation["gpu_method_qualification_report_digest"] = report[
            "qualification_report_digest"
        ]

    if mutation == "formal_path_outside_qualification_root":
        outside = root / "outputs/other/formal.jsonl"
        outside.parent.mkdir(parents=True)
        outside.write_bytes(formal_detection_path.read_bytes())
        invocation["formal_detection_record_path"] = outside.relative_to(
            root
        ).as_posix()
    elif mutation == "formal_record_missing":
        formal_detection_path.unlink()
    elif mutation == "formal_record_sha":
        invocation["formal_detection_record_sha256"] = "0" * 64
    elif mutation == "invocation_missing_formal_sha":
        del invocation["formal_detection_record_sha256"]
    elif mutation == "invocation_threshold_digest":
        invocation["frozen_threshold_digest"] = "0" * 64
    elif mutation == "report_protocol_threshold_digest":
        report["qualification_binding"]["frozen_evidence_protocol_identity"][
            "threshold_digest"
        ] = "0" * 64
    elif mutation == "formal_record_threshold_digest":
        rows = [
            json.loads(line)
            for line in formal_detection_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        rows[0]["frozen_threshold_digest"] = "0" * 64
        formal_detection_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        changed_sha = _sha256(formal_detection_path)
        invocation["formal_detection_record_sha256"] = changed_sha
        report["qualification_binding"]["formal_detection_records_identity"][
            "file_sha256"
        ] = changed_sha
    persist_report()
    isolated_report = {
        "report_schema": "isolated_scientific_execution_report",
        "schema_version": 1,
        "profile_id": workflow.SCIENTIFIC_PROFILE_ID,
        "profile_digest": "b" * 64,
        "direct_requirements_digest": "c" * 64,
        "complete_hash_lock_digest": "d" * 64,
        "complete_hash_lock_dependency_count": 10,
        "dependency_environment_report_valid": True,
        "dependency_environment_report_digest": "e" * 64,
        "python_executable_sha256": "f" * 64,
        "formal_execution_commit": "a" * 40,
        "formal_execution_lock_ready": True,
        "formal_execution_lock_revalidated_before_child": True,
        "formal_execution_lock_revalidated_after_child": True,
        "python_executable_revalidated_before_child": True,
        "python_executable_revalidated_after_child": True,
        "dependency_environment_report_revalidated_before_child": True,
        "dependency_environment_report_revalidated_after_child": True,
        "execution": {
            "return_code": 0,
            "stdout": json.dumps(invocation),
            "stderr": "",
        },
        "decision": "pass",
        "supports_paper_claim": False,
    }

    with pytest.raises(ValueError):
        workflow._validate_qualification_evidence(
            root=root,
            repository_commit="a" * 40,
            paper_run_name="probe_paper",
            prompt_id="probe_prompt_0001",
            qualification_output_root=(
                root / "outputs/gpu_method_qualification"
            ).resolve(),
            isolated_report=isolated_report,
            frozen_evidence_protocol=protocol,
            frozen_evidence_protocol_path=protocol_path,
        )


@pytest.mark.quick
def test_formal_host_builds_qualification_child_command() -> None:
    """公开宿主子命令必须进入统一 formal_workflow_entry, 而非旁路执行科学代码."""

    arguments = host.build_parser().parse_args(
        [
            "--repository-commit",
            "a" * 40,
            "qualification",
            "--paper-run-name",
            "probe_paper",
            "--prompt-id",
            "probe_prompt_0001",
            "--frozen-evidence-protocol",
            "inputs/frozen_evidence_protocol.json",
            "--reference-gradient",
            "1.0",
            "--reference-response",
            "0.5",
            "--reference-sensitivity",
            "0.25",
            "--registered-budget",
            "configs/gpu_budget.json",
            "--result-path",
            "outputs/host/qualification_result.json",
        ]
    )
    bootstrap_identity = {
        "profile_id": "workflow_orchestrator",
        "python_version": "3.12.13",
        "complete_hash_lock_digest": "b" * 64,
        "python_executable": "/managed/python",
        "python_executable_sha256": "c" * 64,
    }
    command = host.build_child_command(
        arguments,
        Path("/managed/python"),
        Path("/repository"),
        bootstrap_identity,
    )

    assert command[:3] == [
        str(Path("/managed/python")),
        "-I",
        str(Path("/repository/scripts/formal_workflow_entry.py")),
    ]
    assert command[3] == "qualification"
    assert command[command.index("--prompt-id") + 1] == "probe_prompt_0001"
    assert command[command.index("--registered-budget") + 1] == (
        "configs/gpu_budget.json"
    )
    assert command[command.index("--frozen-evidence-protocol") + 1] == (
        "inputs/frozen_evidence_protocol.json"
    )
    assert command[command.index("--reference-gradient") + 1] == "1.0"
    assert command[command.index("--reference-response") + 1] == "0.5"
    assert command[command.index("--reference-sensitivity") + 1] == "0.25"
    assert "--workflow" not in command
    assert "--randomization-repeat-id" not in command


@pytest.mark.quick
@pytest.mark.parametrize(
    "missing_option",
    (
        "--frozen-evidence-protocol",
        "--reference-gradient",
        "--reference-response",
        "--reference-sensitivity",
    ),
)
def test_formal_host_requires_protocol_and_three_references(
    missing_option: str,
) -> None:
    """公开qualification CLI不得为协议或reference提供隐式默认值。"""

    arguments = [
        "--repository-commit",
        "a" * 40,
        "qualification",
        "--paper-run-name",
        "probe_paper",
        "--prompt-id",
        "probe_prompt_0001",
        "--frozen-evidence-protocol",
        "inputs/frozen_evidence_protocol.json",
        "--reference-gradient",
        "1.0",
        "--reference-response",
        "0.5",
        "--reference-sensitivity",
        "0.25",
        "--result-path",
        "outputs/host/qualification_result.json",
    ]
    index = arguments.index(missing_option)
    del arguments[index : index + 2]

    with pytest.raises(SystemExit):
        host.build_parser().parse_args(arguments)


@pytest.mark.quick
def test_formal_host_builds_reproducible_content_runtime_smoke_command() -> None:
    """The real smoke has a named clean-detached host route, not python -c."""

    arguments = host.build_parser().parse_args(
        [
            "--repository-commit",
            "a" * 40,
            "content_runtime_smoke",
            "--paper-run-name",
            "probe_paper",
            "--prompt-id",
            "probe_prompt_0001",
            "--reference-gradient",
            "1.0",
            "--reference-response",
            "0.5",
            "--reference-sensitivity",
            "0.25",
            "--result-path",
            "outputs/host/content_runtime_smoke.json",
        ]
    )
    command = host.build_child_command(
        arguments,
        Path("/managed/python"),
        Path("/repository"),
        {
            "profile_id": "workflow_orchestrator",
            "python_version": "3.12.13",
            "complete_hash_lock_digest": "b" * 64,
            "python_executable": "/managed/python",
            "python_executable_sha256": "c" * 64,
        },
    )
    assert command[2] == "/repository/scripts/formal_workflow_entry.py"
    assert command[3] == "content_runtime_smoke"
    assert "-c" not in command
    assert command[command.index("--reference-gradient") + 1] == "1.0"


@pytest.mark.quick
@pytest.mark.parametrize(
    "mutation",
    (
        None,
        "lf_zero",
        "hf_zero",
        "geometry_zero",
        "current_decode_count",
        "probe_decode_count",
        "capture_count",
        "callback_count",
        "write_count",
        "combined_budget",
        "strict_score",
        "image_missing",
        "image_sha",
    ),
)
def test_content_runtime_smoke_host_revalidates_clean_detached_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str | None,
) -> None:
    """Smoke success requires the same before/after lock evidence as qualification."""

    root = tmp_path / "repository"
    report_dir = root / "outputs/content_runtime_smoke/content_runtime_smoke/prompt"
    report_dir.mkdir(parents=True)
    image_path = report_dir / "watermarked.png"
    image_path.write_bytes(b"real-image-bytes")
    diagnostic = {
        "method_role": "full_dual_chain",
        "captured_previous_index": 9,
        "captured_previous_count": 1,
        "callback_write_index": 10,
        "callback_write_count": 1,
        "current_image_decode_count": 1,
        "public_probe_additional_decode_count": 1,
        "actual_dtype_single_write_count": 1,
        "common_gamma": 0.5,
        "lf_effective_l2": 0.001,
        "hf_tail_effective_l2": 0.001,
        "geometry_effective_l2": 0.001,
        "combined_effective_l2": 0.004,
        "combined_effective_l2_limit": 0.005,
        "combined_effective_l2_ready": True,
        "actual_dtype_single_write_digest": "a" * 64,
        "content_only_postwrite_qk_score": 0.1,
        "final_postwrite_qk_score": 0.2,
        "post_write_qk_strict_ready": True,
        "legacy_semantic_feature_operator_present": False,
    }
    if mutation == "lf_zero":
        diagnostic["lf_effective_l2"] = 0.0
    elif mutation == "hf_zero":
        diagnostic["hf_tail_effective_l2"] = 0.0
    elif mutation == "geometry_zero":
        diagnostic["geometry_effective_l2"] = 0.0
    elif mutation == "current_decode_count":
        diagnostic["current_image_decode_count"] = 2
    elif mutation == "probe_decode_count":
        diagnostic["public_probe_additional_decode_count"] = 2
    elif mutation == "capture_count":
        diagnostic["captured_previous_count"] = 2
    elif mutation == "callback_count":
        diagnostic["callback_write_count"] = 2
    elif mutation == "write_count":
        diagnostic["actual_dtype_single_write_count"] = 2
    elif mutation == "combined_budget":
        diagnostic["combined_effective_l2"] = 0.006
    elif mutation == "strict_score":
        diagnostic["final_postwrite_qk_score"] = 0.1
    report = {
        "report_schema": "content_runtime_gpu_smoke_v1",
        "schema_version": 1,
        "content_runtime_smoke_ready": True,
        "runtime_diagnostic": diagnostic,
        "image_path": image_path.relative_to(root).as_posix(),
        "image_sha256": _sha256(image_path),
        "supports_paper_claim": False,
    }
    if mutation == "image_sha":
        report["image_sha256"] = "0" * 64
    report["content_runtime_smoke_digest"] = build_stable_digest(report)
    report_path = report_dir / "content_runtime_smoke.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    invocation = {
        "report_schema": workflow.CONTENT_RUNTIME_SMOKE_INVOCATION_SCHEMA,
        "schema_version": 1,
        "content_runtime_smoke_report_path": report_path.relative_to(root).as_posix(),
        "content_runtime_smoke_report_sha256": _sha256(report_path),
        "content_runtime_smoke_digest": report["content_runtime_smoke_digest"],
        "content_runtime_smoke_ready": True,
        "supports_paper_claim": False,
    }
    repository_commit = "a" * 40

    def fake_execute(_profile: str, _argv: list[str], *, execution_report_path: Path, repository_root: Path):
        isolated = {
            "report_schema": "isolated_scientific_execution_report",
            "schema_version": 1,
            "profile_id": workflow.SCIENTIFIC_PROFILE_ID,
            "profile_digest": "b" * 64,
            "direct_requirements_digest": "c" * 64,
            "complete_hash_lock_digest": "d" * 64,
            "complete_hash_lock_dependency_count": 1,
            "dependency_environment_report_valid": True,
            "dependency_environment_report_digest": "e" * 64,
            "python_executable_sha256": "f" * 64,
            "formal_execution_commit": repository_commit,
            "formal_execution_lock_ready": True,
            "formal_execution_lock_revalidated_before_child": True,
            "formal_execution_lock_revalidated_after_child": True,
            "python_executable_revalidated_before_child": True,
            "python_executable_revalidated_after_child": True,
            "dependency_environment_report_revalidated_before_child": True,
            "dependency_environment_report_revalidated_after_child": True,
            "execution": {"return_code": 0, "stdout": json.dumps(invocation), "stderr": ""},
            "decision": "pass",
            "supports_paper_claim": False,
        }
        path = Path(execution_report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(isolated), encoding="utf-8")
        return isolated, path

    monkeypatch.setattr(workflow, "execute_isolated_scientific_command", fake_execute)
    if mutation == "image_missing":
        image_path.unlink()
    call = lambda: workflow.run_content_runtime_smoke_host_workflow(
        root=root,
        repository_commit=repository_commit,
        paper_run_name="probe_paper",
        prompt_id="prompt",
        result_path="outputs/host/smoke.json",
        smoke_output_root="outputs/content_runtime_smoke",
        reference_gradient=1.0,
        reference_response=0.5,
        reference_sensitivity=0.25,
    )
    if mutation is None:
        result = call()
        assert result["decision"] == "pass"
        assert result["workflow_summary"]["content_runtime_smoke_ready"] is True
    else:
        with pytest.raises(ValueError):
            call()


@pytest.mark.quick
@pytest.mark.parametrize("operator_ready", (False, True))
def test_formal_entry_propagates_qualification_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operator_ready: bool,
) -> None:
    """父编排结果必须由方法门禁决定, 资源门禁不得改变成功状态."""

    bootstrap_identity = {
        "profile_id": "workflow_orchestrator",
        "python_version": "3.12.13",
        "complete_hash_lock_digest": "b" * 64,
        "python_executable": "/managed/python",
        "python_executable_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        formal_workflow_entry,
        "_validate_bootstrap_identity",
        lambda _arguments: bootstrap_identity,
    )
    monkeypatch.setattr(
        formal_workflow_entry,
        "build_formal_execution_lock",
        lambda _root, commit: {"formal_execution_commit": commit},
    )
    monkeypatch.setattr(
        formal_workflow_entry,
        "publish_formal_execution_lock",
        lambda lock: lock,
    )
    monkeypatch.setattr(
        formal_workflow_entry,
        "_require_workflow_orchestrator_environment",
        lambda _root: {
            "profile_id": "workflow_orchestrator",
            "complete_hash_lock_digest": "b" * 64,
        },
    )
    captured_qualification: dict[str, object] = {}

    def fake_run_qualification(**kwargs: object) -> dict[str, object]:
        captured_qualification.update(kwargs)
        return {
            "workflow_summary": {
                "workflow_completion_state": (
                    "gpu_operator_preflight_ready"
                    if operator_ready
                    else "gpu_operator_preflight_failed"
                ),
                "gpu_operator_preflight_ready": operator_ready,
                "gpu_resource_budget_ready": False,
                "supports_paper_claim": False,
            },
            "workflow_environment": {
                "scientific_profile_id": "sd35_method_runtime_gpu"
            },
            "decision": "pass" if operator_ready else "fail",
            "failure_reasons": (
                [] if operator_ready else ["gpu_operator_preflight_not_ready"]
            ),
            "supports_paper_claim": False,
        }

    monkeypatch.setattr(
        formal_workflow_entry,
        "run_gpu_method_qualification_host_workflow",
        fake_run_qualification,
    )
    arguments = argparse.Namespace(
        operation="qualification",
        root=str(tmp_path),
        repository_commit="a" * 40,
        paper_run_name="probe_paper",
        result_path="outputs/host/qualification_result.json",
        orchestrator_profile_id="workflow_orchestrator",
        orchestrator_python_version="3.12.13",
        orchestrator_lock_digest="b" * 64,
        orchestrator_python_executable="/managed/python",
        orchestrator_python_executable_sha256="c" * 64,
        workflow=None,
        persistent_output_dir="",
        package_search_root="",
        randomization_repeat_id="",
        prompt_id="probe_prompt_0001",
        known_answer="configs/keyed_prg_cross_platform_known_answer.json",
        registered_budget="",
        qualification_output_root="outputs/gpu_method_qualification",
        frozen_evidence_protocol="inputs/frozen_evidence_protocol.json",
        reference_gradient=1.0,
        reference_response=0.5,
        reference_sensitivity=0.25,
    )

    payload = formal_workflow_entry.execute(arguments)

    assert payload["decision"] == ("pass" if operator_ready else "fail")
    assert payload["session_execution_decision"] == payload["decision"]
    assert payload["workflow_name"] == "gpu_method_qualification"
    assert payload["workflow_summary"]["gpu_resource_budget_ready"] is False
    assert payload["supports_paper_claim"] is False
    assert captured_qualification["frozen_evidence_protocol"] == (
        "inputs/frozen_evidence_protocol.json"
    )
    assert captured_qualification["reference_gradient"] == 1.0
    assert captured_qualification["reference_response"] == 0.5
    assert captured_qualification["reference_sensitivity"] == 0.25


@pytest.mark.quick
def test_host_workflow_rejects_report_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """宿主不得只信任子进程返回码或 stdout 中的报告路径."""

    root = tmp_path / "repository"
    report_path = (
        root
        / "outputs/gpu_method_qualification/runtime_run"
        / "gpu_method_qualification_report.json"
    )
    report_path.parent.mkdir(parents=True)
    (
        protocol,
        protocol_path,
        formal_detection_path,
        formal_detection_count,
    ) = _frozen_protocol_evidence(root, report_path.parent)
    report = _qualification_report(
        root=root,
        repository_commit="a" * 40,
        paper_run_name="probe_paper",
        prompt_id="probe_prompt_0001",
        operator_ready=True,
        protocol=protocol,
        protocol_path=protocol_path,
        formal_detection_path=formal_detection_path,
        formal_detection_count=formal_detection_count,
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    invocation = {
        "report_schema": workflow.QUALIFICATION_INVOCATION_RESULT_SCHEMA,
        "schema_version": 1,
        "gpu_method_qualification_report_path": report_path.relative_to(
            root
        ).as_posix(),
        "gpu_method_qualification_report_sha256": "0" * 64,
        "gpu_method_qualification_report_digest": report[
            "qualification_report_digest"
        ],
        "gpu_operator_preflight_ready": True,
        "gpu_resource_budget_ready": False,
        "frozen_threshold_digest": protocol.threshold_digest,
        "formal_detection_record_path": formal_detection_path.relative_to(
            root
        ).as_posix(),
        "formal_detection_record_sha256": _sha256(formal_detection_path),
        "supports_paper_claim": False,
    }

    def fake_execute(*_args, execution_report_path, **_kwargs):
        isolated = {
            "report_schema": "isolated_scientific_execution_report",
            "schema_version": 1,
            "profile_id": workflow.SCIENTIFIC_PROFILE_ID,
            "profile_digest": "b" * 64,
            "direct_requirements_digest": "c" * 64,
            "complete_hash_lock_digest": "d" * 64,
            "complete_hash_lock_dependency_count": 10,
            "dependency_environment_report_valid": True,
            "dependency_environment_report_digest": "e" * 64,
            "python_executable_sha256": "f" * 64,
            "formal_execution_commit": "a" * 40,
            "formal_execution_lock_ready": True,
            "formal_execution_lock_revalidated_before_child": True,
            "formal_execution_lock_revalidated_after_child": True,
            "python_executable_revalidated_before_child": True,
            "python_executable_revalidated_after_child": True,
            "dependency_environment_report_revalidated_before_child": True,
            "dependency_environment_report_revalidated_after_child": True,
            "execution": {
                "return_code": 0,
                "stdout": json.dumps(invocation),
                "stderr": "",
            },
            "decision": "pass",
            "supports_paper_claim": False,
        }
        path = Path(execution_report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(isolated), encoding="utf-8")
        return isolated, path

    monkeypatch.setattr(
        workflow,
        "execute_isolated_scientific_command",
        fake_execute,
    )
    with pytest.raises(ValueError, match="报告文件摘要不一致"):
        workflow.run_gpu_method_qualification_host_workflow(
            root=root,
            repository_commit="a" * 40,
            paper_run_name="probe_paper",
            prompt_id="probe_prompt_0001",
            result_path="outputs/host/result.json",
            known_answer="configs/keyed_prg_cross_platform_known_answer.json",
            qualification_output_root="outputs/gpu_method_qualification",
            frozen_evidence_protocol=protocol_path,
            reference_gradient=1.0,
            reference_response=0.5,
            reference_sensitivity=0.25,
        )


@pytest.mark.quick
def test_host_reports_scientific_child_exception_before_invocation_exists() -> None:
    """科学子进程提前失败时, 宿主必须透传真实异常而非只报告索引缺失."""

    isolated_report = {
        "report_schema": "isolated_scientific_execution_report",
        "schema_version": 1,
        "profile_id": workflow.SCIENTIFIC_PROFILE_ID,
        "profile_digest": "b" * 64,
        "direct_requirements_digest": "c" * 64,
        "complete_hash_lock_digest": "d" * 64,
        "complete_hash_lock_dependency_count": 10,
        "dependency_environment_report_valid": True,
        "dependency_environment_report_digest": "e" * 64,
        "python_executable_sha256": "f" * 64,
        "formal_execution_commit": "a" * 40,
        "formal_execution_lock_ready": True,
        "formal_execution_lock_revalidated_before_child": True,
        "formal_execution_lock_revalidated_after_child": True,
        "python_executable_revalidated_before_child": True,
        "python_executable_revalidated_after_child": True,
        "dependency_environment_report_revalidated_before_child": True,
        "dependency_environment_report_revalidated_after_child": True,
        "execution": {
            "return_code": 1,
            "stdout": "模型加载日志\n",
            "stderr": (
                "Traceback (most recent call last):\n"
                "  File \"runtime.py\", line 62, in decode_latent\n"
                "RuntimeError: aten._local_scalar_dense.default\n"
            ),
        },
        "decision": "fail",
        "supports_paper_claim": False,
    }

    with pytest.raises(
        ValueError,
        match=r"科学子进程失败原因: RuntimeError: .*_local_scalar_dense",
    ):
        workflow._validate_qualification_evidence(
            root=Path.cwd(),
            repository_commit="a" * 40,
            paper_run_name="probe_paper",
            prompt_id="probe_prompt_0001",
            qualification_output_root=Path.cwd() / "outputs",
            isolated_report=isolated_report,
            frozen_evidence_protocol=SimpleNamespace(
                threshold_digest="1" * 64,
                image_only_measurement_config_digest="2" * 64,
            ),
            frozen_evidence_protocol_path=Path.cwd() / "protocol.json",
        )


@pytest.mark.quick
def test_formal_entry_main_returns_nonzero_for_operator_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """资格化方法门禁失败必须穿透父编排并形成非零退出码."""

    monkeypatch.setattr(
        formal_workflow_entry,
        "execute",
        lambda _arguments: {
            "decision": "fail",
            "session_execution_decision": "fail",
            "workflow_completion_state": "gpu_operator_preflight_failed",
            "workflow_name": "gpu_method_qualification",
            "paper_run_name": "probe_paper",
            "profile_id": "workflow_orchestrator",
            "orchestrator_bootstrap_identity": {
                "complete_hash_lock_digest": "b" * 64,
                "python_executable_sha256": "c" * 64,
            },
            "supports_paper_claim": False,
        },
    )
    monkeypatch.setattr(
        formal_workflow_entry,
        "_write_result",
        lambda _root, path, _payload: Path(path),
    )
    exit_code = formal_workflow_entry.main(
        [
            "qualification",
            "--root",
            str(tmp_path),
            "--repository-commit",
            "a" * 40,
            "--paper-run-name",
            "probe_paper",
            "--result-path",
            "outputs/host/result.json",
            "--orchestrator-profile-id",
            "workflow_orchestrator",
            "--orchestrator-python-version",
            "3.12.13",
            "--orchestrator-lock-digest",
            "b" * 64,
            "--orchestrator-python-executable",
            "/managed/python",
            "--orchestrator-python-executable-sha256",
            "c" * 64,
            "--prompt-id",
            "probe_prompt_0001",
            "--frozen-evidence-protocol",
            "inputs/frozen_evidence_protocol.json",
            "--reference-gradient",
            "1.0",
            "--reference-response",
            "0.5",
            "--reference-sensitivity",
            "0.25",
        ]
    )

    assert exit_code == 1
