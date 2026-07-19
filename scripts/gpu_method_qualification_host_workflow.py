"""在精确父编排环境中启动隔离的 GPU 方法资格化子进程."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


from experiments.protocol.gpu_method_qualification_schema import (
    GPU_METHOD_QUALIFICATION_INVOCATION_RESULT_SCHEMA,
    GPU_METHOD_QUALIFICATION_SCHEMA,
)
from experiments.runtime.isolated_scientific_execution import (
    execute_isolated_scientific_command,
)
from main.core.digest import build_stable_digest


SCIENTIFIC_PROFILE_ID = "sd35_method_runtime_gpu"
QUALIFICATION_WORKFLOW_NAME = "gpu_method_qualification"
QUALIFICATION_INVOCATION_RESULT_SCHEMA = (
    GPU_METHOD_QUALIFICATION_INVOCATION_RESULT_SCHEMA
)
CONTENT_RUNTIME_SMOKE_WORKFLOW_NAME = "content_runtime_smoke"
CONTENT_RUNTIME_SMOKE_INVOCATION_SCHEMA = "content_runtime_smoke_invocation_v1"


def _file_sha256(path: Path) -> str:
    """流式计算资格化报告或隔离执行报告的 SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    """判断一个值是否为规范小写 SHA-256 文本."""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_report_float(value: Any, *, positive: bool = False) -> bool:
    """Accept only exact finite JSON floats, optionally requiring positivity."""

    return (
        type(value) is float
        and math.isfinite(value)
        and (not positive or value > 0.0)
    )


def _resolve_under_outputs(root: Path, value: str | Path, field_name: str) -> Path:
    """解析宿主输出路径, 并阻止资格化证据写出仓库 outputs/ 边界."""

    requested = Path(value).expanduser()
    resolved = (
        requested.resolve()
        if requested.is_absolute()
        else (root / requested).resolve()
    )
    try:
        resolved.relative_to((root / "outputs").resolve())
    except ValueError as exc:
        raise ValueError(f"{field_name} 必须位于仓库 outputs/ 下") from exc
    return resolved


def _read_json_mapping(path: Path) -> dict[str, Any]:
    """读取受治理 JSON 映射, 拒绝数组或标量替代正式报告."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"资格化 JSON 顶层必须是映射: {path}")
    return payload


def _invocation_record(stdout: str) -> dict[str, Any] | None:
    """从科学子进程最后一条结构化输出读取资格化报告索引."""

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _scientific_child_failure_diagnostic(execution: Mapping[str, Any]) -> str:
    """提取科学子进程最后一条异常, 避免宿主错误掩盖真实失败原因."""

    stderr_lines = [
        line.strip()
        for line in str(execution.get("stderr", "")).splitlines()
        if line.strip()
    ]
    for line in reversed(stderr_lines):
        exception_name, separator, _detail = line.partition(":")
        normalized_name = exception_name.rsplit(".", 1)[-1]
        if separator and normalized_name.endswith(("Error", "Exception")):
            return line[-2000:]
    if stderr_lines:
        return stderr_lines[-1][-2000:]
    return f"return_code={execution.get('return_code')}"


def _validate_qualification_evidence(
    *,
    root: Path,
    repository_commit: str,
    paper_run_name: str,
    prompt_id: str,
    qualification_output_root: Path,
    isolated_report: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """交叉复验子进程索引、资格化报告和隔离执行进程证据."""

    execution = isolated_report.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("隔离科学执行报告缺少 execution 映射")
    if not all(
        (
            isolated_report.get("report_schema")
            == "isolated_scientific_execution_report",
            isolated_report.get("schema_version") == 1,
            isolated_report.get("profile_id") == SCIENTIFIC_PROFILE_ID,
            _is_sha256(isolated_report.get("profile_digest")),
            _is_sha256(isolated_report.get("direct_requirements_digest")),
            _is_sha256(isolated_report.get("complete_hash_lock_digest")),
            isinstance(
                isolated_report.get("complete_hash_lock_dependency_count"),
                int,
            ),
            isolated_report.get("complete_hash_lock_dependency_count", 0) > 0,
            isolated_report.get("dependency_environment_report_valid") is True,
            _is_sha256(
                isolated_report.get("dependency_environment_report_digest")
            ),
            _is_sha256(isolated_report.get("python_executable_sha256")),
            isolated_report.get("formal_execution_commit")
            == repository_commit,
            isolated_report.get("formal_execution_lock_ready") is True,
            isolated_report.get(
                "formal_execution_lock_revalidated_before_child"
            )
            is True,
            isolated_report.get(
                "formal_execution_lock_revalidated_after_child"
            )
            is True,
            isolated_report.get("python_executable_revalidated_before_child")
            is True,
            isolated_report.get("python_executable_revalidated_after_child")
            is True,
            isolated_report.get(
                "dependency_environment_report_revalidated_before_child"
            )
            is True,
            isolated_report.get(
                "dependency_environment_report_revalidated_after_child"
            )
            is True,
            isolated_report.get("supports_paper_claim") is False,
        )
    ):
        raise ValueError("隔离科学执行报告身份或执行前后复验证据无效")
    invocation = _invocation_record(str(execution.get("stdout", "")))
    if invocation is None:
        diagnostic = _scientific_child_failure_diagnostic(execution)
        raise ValueError(
            "GPU 方法资格化子进程未输出结构化报告索引; "
            f"科学子进程失败原因: {diagnostic}"
        )
    expected_invocation_fields = {
        "report_schema",
        "schema_version",
        "gpu_method_qualification_report_path",
        "gpu_method_qualification_report_sha256",
        "gpu_method_qualification_report_digest",
        "gpu_operator_preflight_ready",
        "gpu_resource_budget_ready",
        "supports_paper_claim",
    }
    if set(invocation) != expected_invocation_fields:
        raise ValueError("GPU 方法资格化子进程报告索引字段集合不一致")
    if (
        invocation["report_schema"] != QUALIFICATION_INVOCATION_RESULT_SCHEMA
        or invocation["schema_version"] != 1
        or invocation["supports_paper_claim"] is not False
    ):
        raise ValueError("GPU 方法资格化子进程报告索引身份无效")

    report_path = _resolve_under_outputs(
        root,
        str(invocation["gpu_method_qualification_report_path"]),
        "gpu_method_qualification_report_path",
    )
    try:
        report_path.relative_to(qualification_output_root)
    except ValueError as exc:
        raise ValueError("GPU 方法资格化报告不属于请求的输出根") from exc
    if not report_path.is_file():
        raise ValueError("GPU 方法资格化报告文件不存在")
    if (
        _file_sha256(report_path)
        != invocation["gpu_method_qualification_report_sha256"]
    ):
        raise ValueError("GPU 方法资格化报告文件摘要不一致")

    report = _read_json_mapping(report_path)
    report_digest = report.get("qualification_report_digest")
    digest_input = dict(report)
    digest_input.pop("qualification_report_digest", None)
    binding = report.get("qualification_binding")
    binding_input = (
        binding.get("input_summary") if isinstance(binding, Mapping) else None
    )
    if not all(
        (
            report.get("qualification_report_schema")
            == GPU_METHOD_QUALIFICATION_SCHEMA,
            isinstance(report_digest, str),
            build_stable_digest(digest_input) == report_digest,
            report_digest
            == invocation["gpu_method_qualification_report_digest"],
            report.get("supports_paper_claim") is False,
            isinstance(report.get("gpu_operator_preflight_ready"), bool),
            isinstance(report.get("gpu_resource_budget_ready"), bool),
            report.get("gpu_operator_preflight_ready")
            == invocation["gpu_operator_preflight_ready"],
            report.get("gpu_resource_budget_ready")
            == invocation["gpu_resource_budget_ready"],
            isinstance(binding, Mapping),
            binding.get("code_version") == repository_commit,
            binding.get("dependency_profile_id") == SCIENTIFIC_PROFILE_ID,
            isinstance(binding_input, Mapping),
            binding_input.get("paper_run_name") == paper_run_name,
            binding_input.get("prompt_id") == prompt_id,
        )
    ):
        raise ValueError("GPU 方法资格化报告未通过宿主证据复验")

    operator_ready = report["gpu_operator_preflight_ready"]
    expected_return_code = 0 if operator_ready else 1
    expected_isolated_decision = "pass" if operator_ready else "fail"
    if (
        execution.get("return_code") != expected_return_code
        or isolated_report.get("decision") != expected_isolated_decision
        or isolated_report.get("supports_paper_claim") is not False
    ):
        raise ValueError("资格化门禁、科学子进程状态码和隔离执行结论不一致")
    return report, report_path, invocation


def _workflow_environment(
    isolated_report: Mapping[str, Any],
    isolated_report_path: Path,
) -> dict[str, Any]:
    """提取宿主结果需要持久绑定的科学解释器与依赖身份."""

    return {
        "scientific_profile_id": isolated_report.get("profile_id"),
        "scientific_profile_digest": isolated_report.get("profile_digest"),
        "direct_requirements_digest": isolated_report.get(
            "direct_requirements_digest"
        ),
        "complete_hash_lock_digest": isolated_report.get(
            "complete_hash_lock_digest"
        ),
        "dependency_environment_report_path": isolated_report.get(
            "dependency_environment_report_path"
        ),
        "dependency_environment_report_digest": isolated_report.get(
            "dependency_environment_report_digest"
        ),
        "python_executable_path": isolated_report.get("python_executable_path"),
        "python_executable_sha256": isolated_report.get(
            "python_executable_sha256"
        ),
        "scientific_execution_report_path": isolated_report_path.as_posix(),
        "scientific_execution_report_digest": _file_sha256(
            isolated_report_path
        ),
    }


def run_gpu_method_qualification_host_workflow(
    *,
    root: str | Path,
    repository_commit: str,
    paper_run_name: str,
    prompt_id: str,
    result_path: str | Path,
    known_answer: str | Path,
    qualification_output_root: str | Path,
    registered_budget: str | Path | None = None,
) -> dict[str, Any]:
    """在受治理隔离环境中运行单 Prompt 资格化并重建宿主结论."""

    root_path = Path(root).resolve()
    resolved_result_path = _resolve_under_outputs(
        root_path,
        result_path,
        "result_path",
    )
    resolved_output_root = _resolve_under_outputs(
        root_path,
        qualification_output_root,
        "qualification_output_root",
    )
    execution_report_path = resolved_result_path.with_name(
        resolved_result_path.stem
        + "_gpu_method_qualification_scientific_execution.json"
    )
    child_argv = [
        str(root_path / "scripts/run_gpu_method_qualification.py"),
        "--root",
        str(root_path),
        "--paper-run-name",
        paper_run_name,
        "--prompt-id",
        prompt_id,
        "--known-answer",
        str(known_answer),
        "--output-root",
        str(resolved_output_root),
    ]
    if registered_budget is not None:
        child_argv.extend(["--registered-budget", str(registered_budget)])

    isolated_report, persisted_execution_report_path = (
        execute_isolated_scientific_command(
            SCIENTIFIC_PROFILE_ID,
            child_argv,
            execution_report_path=execution_report_path,
            repository_root=root_path,
        )
    )
    persisted_execution_report_path = Path(
        persisted_execution_report_path
    ).resolve()
    if (
        not persisted_execution_report_path.is_file()
        or _read_json_mapping(persisted_execution_report_path) != isolated_report
        or isolated_report.get("profile_id") != SCIENTIFIC_PROFILE_ID
    ):
        raise ValueError("隔离科学执行报告未通过持久化身份复验")

    report, report_path, invocation = _validate_qualification_evidence(
        root=root_path,
        repository_commit=repository_commit,
        paper_run_name=paper_run_name,
        prompt_id=prompt_id,
        qualification_output_root=resolved_output_root,
        isolated_report=isolated_report,
    )
    operator_ready = report["gpu_operator_preflight_ready"]
    failure_reasons = [] if operator_ready else ["gpu_operator_preflight_not_ready"]
    workflow_summary = {
        "workflow_name": QUALIFICATION_WORKFLOW_NAME,
        "paper_run_name": paper_run_name,
        "prompt_id": prompt_id,
        "gpu_method_qualification_report_path": report_path.relative_to(
            root_path
        ).as_posix(),
        "gpu_method_qualification_report_sha256": invocation[
            "gpu_method_qualification_report_sha256"
        ],
        "gpu_method_qualification_report_digest": invocation[
            "gpu_method_qualification_report_digest"
        ],
        "gpu_operator_preflight_ready": operator_ready,
        "gpu_resource_budget_ready": report["gpu_resource_budget_ready"],
        "workflow_completion_state": (
            "gpu_operator_preflight_ready"
            if operator_ready
            else "gpu_operator_preflight_failed"
        ),
        "failure_reasons": failure_reasons,
        "supports_paper_claim": False,
    }
    return {
        "workflow_summary": workflow_summary,
        "workflow_environment": _workflow_environment(
            isolated_report,
            persisted_execution_report_path,
        ),
        "archive_record": None,
        "return_code": 0 if operator_ready else 1,
        "decision": "pass" if operator_ready else "fail",
        "failure_reasons": failure_reasons,
        "supports_paper_claim": False,
    }


def run_content_runtime_smoke_host_workflow(
    *,
    root: str | Path,
    repository_commit: str,
    paper_run_name: str,
    prompt_id: str,
    result_path: str | Path,
    smoke_output_root: str | Path,
    reference_gradient: float,
    reference_response: float,
    reference_sensitivity: float,
) -> dict[str, Any]:
    """Run the reproducible clean-detached content runtime GPU smoke."""

    root_path = Path(root).resolve()
    resolved_result_path = _resolve_under_outputs(root_path, result_path, "result_path")
    resolved_output_root = _resolve_under_outputs(
        root_path,
        smoke_output_root,
        "smoke_output_root",
    )
    execution_report_path = resolved_result_path.with_name(
        resolved_result_path.stem + "_content_runtime_smoke_execution.json"
    )
    child_argv = [
        str(root_path / "scripts/run_gpu_method_qualification.py"),
        "--root",
        str(root_path),
        "--paper-run-name",
        paper_run_name,
        "--prompt-id",
        prompt_id,
        "--output-root",
        str(resolved_output_root),
        "--content-runtime-smoke",
        "--reference-gradient",
        repr(reference_gradient),
        "--reference-response",
        repr(reference_response),
        "--reference-sensitivity",
        repr(reference_sensitivity),
    ]
    isolated_report, persisted_path = execute_isolated_scientific_command(
        SCIENTIFIC_PROFILE_ID,
        child_argv,
        execution_report_path=execution_report_path,
        repository_root=root_path,
    )
    persisted_path = Path(persisted_path).resolve()
    if not persisted_path.is_file() or _read_json_mapping(persisted_path) != isolated_report:
        raise ValueError("content runtime smoke execution report identity mismatch")
    execution = isolated_report.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("content runtime smoke execution is missing")
    if not all(
        (
            isolated_report.get("report_schema")
            == "isolated_scientific_execution_report",
            isolated_report.get("schema_version") == 1,
            isolated_report.get("profile_id") == SCIENTIFIC_PROFILE_ID,
            _is_sha256(isolated_report.get("profile_digest")),
            _is_sha256(isolated_report.get("direct_requirements_digest")),
            _is_sha256(isolated_report.get("complete_hash_lock_digest")),
            isinstance(
                isolated_report.get("complete_hash_lock_dependency_count"),
                int,
            ),
            isolated_report.get("complete_hash_lock_dependency_count", 0) > 0,
            isolated_report.get("dependency_environment_report_valid") is True,
            _is_sha256(
                isolated_report.get("dependency_environment_report_digest")
            ),
            _is_sha256(isolated_report.get("python_executable_sha256")),
            isolated_report.get("formal_execution_commit") == repository_commit,
            isolated_report.get("formal_execution_lock_ready") is True,
            isolated_report.get("formal_execution_lock_revalidated_before_child")
            is True,
            isolated_report.get("formal_execution_lock_revalidated_after_child")
            is True,
            isolated_report.get("python_executable_revalidated_before_child") is True,
            isolated_report.get("python_executable_revalidated_after_child") is True,
            isolated_report.get(
                "dependency_environment_report_revalidated_before_child"
            )
            is True,
            isolated_report.get(
                "dependency_environment_report_revalidated_after_child"
            )
            is True,
            isolated_report.get("supports_paper_claim") is False,
        )
    ):
        raise ValueError("content runtime smoke isolated environment identity is invalid")
    invocation = _invocation_record(str(execution.get("stdout", "")))
    if invocation is None:
        raise ValueError(
            "content runtime smoke did not produce an invocation: "
            + _scientific_child_failure_diagnostic(execution)
        )
    expected = {
        "report_schema",
        "schema_version",
        "content_runtime_smoke_report_path",
        "content_runtime_smoke_report_sha256",
        "content_runtime_smoke_digest",
        "content_runtime_smoke_ready",
        "supports_paper_claim",
    }
    if set(invocation) != expected or not all(
        (
            invocation.get("report_schema") == CONTENT_RUNTIME_SMOKE_INVOCATION_SCHEMA,
            invocation.get("schema_version") == 1,
            invocation.get("content_runtime_smoke_ready") is True,
            invocation.get("supports_paper_claim") is False,
            _is_sha256(invocation.get("content_runtime_smoke_report_sha256")),
            _is_sha256(invocation.get("content_runtime_smoke_digest")),
            execution.get("return_code") == 0,
            isolated_report.get("decision") == "pass",
            isolated_report.get("formal_execution_commit") == repository_commit,
        )
    ):
        raise ValueError("content runtime smoke invocation or isolated identity is invalid")
    report_path = _resolve_under_outputs(
        root_path,
        invocation["content_runtime_smoke_report_path"],
        "content_runtime_smoke_report_path",
    )
    try:
        report_path.relative_to(resolved_output_root)
    except ValueError as exc:
        raise ValueError("content runtime smoke report escaped its output root") from exc
    if not report_path.is_file() or _file_sha256(report_path) != invocation[
        "content_runtime_smoke_report_sha256"
    ]:
        raise ValueError("content runtime smoke report file identity mismatch")
    report = _read_json_mapping(report_path)
    digest_input = dict(report)
    digest = digest_input.pop("content_runtime_smoke_digest", None)
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    diagnostic = report.get("runtime_diagnostic")
    if type(diagnostic) is not dict:
        raise ValueError("content runtime smoke diagnostic is missing")
    branch_l2_fields = (
        "lf_effective_l2",
        "hf_tail_effective_l2",
        "geometry_effective_l2",
    )
    combined_l2 = diagnostic.get("combined_effective_l2")
    combined_limit = diagnostic.get("combined_effective_l2_limit")
    score_before = diagnostic.get("content_only_postwrite_qk_score")
    score_after = diagnostic.get("final_postwrite_qk_score")
    combined_ready = (
        _finite_report_float(combined_l2, positive=True)
        and _finite_report_float(combined_limit, positive=True)
        and combined_l2 <= combined_limit
    )
    score_ready = (
        _finite_report_float(score_before)
        and _finite_report_float(score_after)
        and score_after > score_before
    )
    gamma = diagnostic.get("common_gamma")
    gamma_ready = _finite_report_float(gamma, positive=True) and gamma <= 1.0
    if not all(
        (
            digest == invocation["content_runtime_smoke_digest"],
            build_stable_digest(digest_input) == digest,
            report.get("content_runtime_smoke_ready") is True,
            report.get("supports_paper_claim") is False,
            diagnostic.get("method_role") == "full_dual_chain",
            diagnostic.get("captured_previous_index") == 9,
            diagnostic.get("captured_previous_count") == 1,
            diagnostic.get("callback_write_index") == 10,
            diagnostic.get("callback_write_count") == 1,
            diagnostic.get("current_image_decode_count") == 1,
            diagnostic.get("public_probe_additional_decode_count") == 1,
            diagnostic.get("actual_dtype_single_write_count") == 1,
            all(
                _finite_report_float(diagnostic.get(field), positive=True)
                for field in branch_l2_fields
            ),
            combined_ready,
            diagnostic.get("combined_effective_l2_ready") is True,
            gamma_ready,
            _is_sha256(diagnostic.get("actual_dtype_single_write_digest")),
            score_ready,
            diagnostic.get("post_write_qk_strict_ready") is True,
            "semantic_feature_operator_contract" not in serialized,
            "COMPLETE_FEATURE_WIDTH" not in serialized,
        )
    ):
        raise ValueError("content runtime smoke report did not close the new-chain boundary")
    image_path_value = report.get("image_path")
    image_sha256 = report.get("image_sha256")
    if type(image_path_value) is not str or not _is_sha256(image_sha256):
        raise ValueError("content runtime smoke image identity is missing")
    image_path = _resolve_under_outputs(
        root_path,
        image_path_value,
        "content_runtime_smoke_image_path",
    )
    try:
        image_path.relative_to(resolved_output_root)
    except ValueError as exc:
        raise ValueError("content runtime smoke image escaped its output root") from exc
    if not image_path.is_file() or _file_sha256(image_path) != image_sha256:
        raise ValueError("content runtime smoke image file identity mismatch")
    return {
        "workflow_summary": {
            "workflow_name": CONTENT_RUNTIME_SMOKE_WORKFLOW_NAME,
            "paper_run_name": paper_run_name,
            "prompt_id": prompt_id,
            "content_runtime_smoke_report_path": report_path.relative_to(root_path).as_posix(),
            "content_runtime_smoke_report_sha256": invocation[
                "content_runtime_smoke_report_sha256"
            ],
            "content_runtime_smoke_digest": digest,
            "content_runtime_smoke_ready": True,
            "workflow_completion_state": "content_runtime_smoke_ready",
            "supports_paper_claim": False,
        },
        "workflow_environment": _workflow_environment(isolated_report, persisted_path),
        "archive_record": None,
        "return_code": 0,
        "decision": "pass",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }


__all__ = [
    "CONTENT_RUNTIME_SMOKE_INVOCATION_SCHEMA",
    "CONTENT_RUNTIME_SMOKE_WORKFLOW_NAME",
    "QUALIFICATION_INVOCATION_RESULT_SCHEMA",
    "QUALIFICATION_WORKFLOW_NAME",
    "SCIENTIFIC_PROFILE_ID",
    "run_gpu_method_qualification_host_workflow",
    "run_content_runtime_smoke_host_workflow",
]
