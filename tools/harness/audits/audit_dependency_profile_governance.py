"""审计正式依赖 profile、哈希锁门禁和单一准备路径."""

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runtime.dependency_profiles import (  # noqa: E402
    REQUIRED_DEPENDENCY_PROFILE_NAMES,
    load_dependency_profile_registry,
    parse_exact_requirement_spec,
)
from tools.harness.lib.json_report import build_report, exit_with_report  # noqa: E402


REQUIRED_PATHS = (
    "configs/dependency_profile_registry.json",
    "configs/dependency_profiles",
    "experiments/runtime/dependency_profiles.py",
    "experiments/runtime/dependency_preparation.py",
    "experiments/runtime/isolated_dependency_environment.py",
    "experiments/runtime/isolated_scientific_execution.py",
    "experiments/runtime/repository_environment.py",
    "scripts/prepare_dependency_profile.py",
    "scripts/prepare_isolated_dependency_environment.py",
    "scripts/materialize_dependency_lock_candidate.py",
    "scripts/write_dependency_lock_review_bundle.py",
    "scripts/write_reviewed_dependency_hash_lock.py",
    "configs/dependency_profiles/dependency_qualification_uv_linux_x86_64_lock.txt",
    "docs/field_registry.md",
    "docs/builds/formal_dependency_environment.md",
)
FORBIDDEN_PATHS = ("configs/colab_sd35_runtime_constraints.txt",)
BUSINESS_SCAN_ROOTS = (
    "main",
    "experiments",
    "paper_experiments",
    "scripts",
    "paper_workflow",
)
BUSINESS_TEXT_SUFFIXES = frozenset({".py", ".ipynb", ".sh"})
FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "manual_version_pins",
        "pip_install_command",
        "pair_perceptual_dependency_install_command",
        "recommended_python",
    }
)
REQUIRED_FIELD_NAMES = frozenset(
    {
        "profile_id",
        "profile_digest",
        "profile_summary_digest",
        "execution_role",
        "direct_dependency_input_contract",
        "complete_hash_lock_contract",
        "direct_requirements_path",
        "direct_requirements",
        "direct_requirements_digest",
        "locked_requirements",
        "complete_hash_lock_path",
        "complete_hash_lock_present",
        "complete_hash_lock_digest",
        "complete_hash_lock_dependency_count",
        "formal_ready",
        "readiness_blockers",
        "expected_environment",
        "observed_environment",
        "locked_dependencies",
        "environment_match",
        "mismatches",
        "inspection_digest",
        "repository_commit_state",
        "installation",
        "pip_check",
        "compatibility_check_required",
        "runtime_comparison",
        "failure_reasons",
        "operation_kind",
        "target_complete_hash_lock_ready",
        "environment_root",
        "managed_python_root",
        "isolated_environment_path",
        "uv_distribution_version",
        "uv_distribution_record_path",
        "uv_distribution_record_sha256",
        "uv_distribution_executable_record_path",
        "uv_distribution_executable_record_sha256",
        "uv_executable_path",
        "uv_executable_sha256",
        "uv_reported_version",
        "python_executable_path",
        "python_executable_sha256",
        "python_executable_sha256_after_preparation",
        "command_results",
        "uv_commands",
        "argv",
        "environment_overrides",
        "orchestrator_profile_digest",
        "orchestrator_complete_hash_lock_digest",
        "orchestrator_inspection",
        "provisioned",
        "provision_report_path",
        "provision_report_digest",
        "provision_report",
        "dependency_preparation_command",
        "dependency_preparation_report_path",
        "dependency_preparation_report_digest",
        "dependency_preparation_report",
        "formal_preparation_completed",
        "dependency_environment_report_path",
        "dependency_environment_report_digest",
        "dependency_environment_report_actual_digest",
        "dependency_environment_report_valid",
        "dependency_environment_validation_errors",
        "python_executable_revalidated_before_child",
        "python_executable_revalidated_after_child",
        "dependency_environment_report_revalidated_before_child",
        "dependency_environment_report_revalidated_after_child",
        "formal_execution_lock_revalidated_before_child",
        "formal_execution_lock_revalidated_after_child",
        "child_argv_tail",
        "execution_report_path",
        "execution_completed",
        "isolated_scientific_context_required",
        "isolated_scientific_context_ready",
        "isolated_scientific_context",
        "reported_profile_digest",
        "reported_complete_hash_lock_digest",
        "reported_formal_execution_lock_digest",
        "reported_python_executable",
        "reported_python_executable_sha256",
        "current_python_executable",
        "current_python_executable_sha256",
        "pip_resolver_report_path",
        "candidate_lock_path",
        "candidate_lock_logical_digest",
        "candidate_lock_dependency_count",
        "candidate_hash_source",
        "review_execution_mode",
        "orchestrator_preparation",
        "isolated_python_provision",
        "candidate_materialization",
        "qualification_tool_lock_path",
        "qualification_tool_lock_digest",
        "qualification_report_path",
    }
)
REQUIRED_PREPARATION_TOKENS = (
    "require_published_formal_execution_lock",
    "require_dependency_profile_ready",
    "inspect_dependency_profile_environment",
    '"--require-hashes"',
    '"--only-binary=:all:"',
    "if ready_profile.pytorch_index_url is not None",
    '"pip", "check"',
    "outputs/dependency_profiles",
)
REQUIRED_ISOLATED_PREPARATION_TOKENS = (
    'UV_DISTRIBUTION_VERSION = "0.11.28"',
    "provision_isolated_dependency_python",
    "prepare_isolated_dependency_environment",
    "require_published_formal_execution_lock",
    '"--install-dir"',
    '"--clear"',
    '"--managed-python"',
    '"ensurepip"',
    '"uv_executable_sha256"',
    "_inspect_uv_executable_distribution_source",
    '"uv_distribution_record_sha256"',
    '"uv_distribution_executable_record_sha256"',
    '"python_executable_sha256"',
    '"python_executable_sha256_after_preparation"',
    '"experiments.runtime.dependency_preparation"',
    "Path(tempfile.gettempdir())",
)
REQUIRED_ISOLATED_SCIENTIFIC_CONTEXT_TOKENS = (
    "ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH_ENVIRONMENT_KEY",
    "ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_DIGEST_ENVIRONMENT_KEY",
    "SCIENTIFIC_DEPENDENCY_PROFILE_IDS",
    "_inspect_isolated_scientific_context",
    '"isolated_scientific_context_required"',
    '"isolated_scientific_context_ready"',
    '"isolated_context_environment_report_path_missing"',
    '"isolated_context_complete_hash_lock_mismatch"',
    '"isolated_context_formal_execution_lock_mismatch"',
    '"isolated_context_python_executable_digest_mismatch"',
)
THIN_SCRIPT_IMPORTS = {
    "scripts/prepare_dependency_profile.py": (
        "from experiments.runtime.dependency_preparation import main",
    ),
    "scripts/prepare_isolated_dependency_environment.py": (
        "from experiments.runtime.isolated_dependency_environment import main",
    ),
}
FORBIDDEN_THIN_SCRIPT_IMPLEMENTATION_TOKENS = (
    "subprocess",
    "get_dependency_profile",
    "require_dependency_profile_ready",
    "pip install",
    "uv venv",
)
REQUIRED_LOCK_REVIEW_TOKENS = (
    "CURRENT_INTERPRETER_PROFILE_ID = WORKFLOW_ORCHESTRATOR_PROFILE_ID",
    "if profile_id != CURRENT_INTERPRETER_PROFILE_ID",
    "provision_isolated_dependency_python",
    "child_command_runner(command, repository_root)",
    "candidate_materializer.load_resolved_wheels",
    "candidate_materializer.candidate_lock_text",
    "candidate_materializer.candidate_lock_logical_digest",
    "launch_dependency_lock_qualification",
    '"--require-hashes"',
    '"--only-binary=:all:"',
    "_require_qualification_child_interpreter",
    "_validate_written_review_bundle",
)
REQUIRED_LOCK_ACCEPTANCE_TOKENS = (
    "validate_formal_execution_lock_record",
    "candidate_materializer.load_resolved_wheels",
    "candidate_materializer.candidate_lock_text",
    "candidate_materializer.candidate_lock_logical_digest",
    'target_path.open("xb")',
    "complete_hash_lock_already_present",
    "lock_written_for_commit",
)
QUALIFICATION_TOOL_LOCK_TEXT = (
    "uv==0.11.28 "
    "--hash=sha256:49fe42df9f42056037473f3876adec1615709b57d3470ed39178ff420f3afb9f"
)
REQUIRED_CANDIDATE_MATERIALIZER_TOKENS = (
    "if profile.pytorch_index_url is not None",
    "candidate_lock_text",
    "candidate_lock_logical_digest",
    "load_resolved_wheels",
)
INNER_LAYER_SCAN_ROOTS = ("main", "experiments", "paper_experiments")
WORKFLOW_ORCHESTRATOR_REQUIRED_DIRECT_REQUIREMENTS = frozenset(
    {"huggingface_hub==1.20.1", "uv==0.11.28"}
)
SCIENCE_REQUIRED_INSTALLER_REQUIREMENTS = frozenset(
    {"pip==24.3.1", "setuptools==75.3.0", "wheel==0.45.1"}
)
_FREE_DEPENDENCY_OVERRIDE_PATTERN = re.compile(
    r"\b(?:python_version|torch_specs|torchvision_specs|package_specs|pytorch_index_url)"
    r"\b\s*=\s*(?:os\.environ|os\.getenv)",
    re.IGNORECASE,
)
_DEPENDENCY_LATEST_PATTERN = re.compile(
    r"(?:[A-Za-z0-9_.-]+\s*==\s*latest\b|/(?:latest)(?:\b|/)|"
    r"(?:pip|uv|micromamba|curl|wget)[^\r\n]{0,160}\blatest\b)",
    re.IGNORECASE,
)


def _field_names(field_registry_path: Path) -> set[str]:
    """读取 Markdown 表中的字段名集合."""

    names: set[str] = set()
    for line in field_registry_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith("| ") or line.startswith("| field_name") or line.startswith("| ---"):
            continue
        names.add(line.split("|", maxsplit=2)[1].strip())
    return names


def _append_text_violation(
    violations: list[dict[str, Any]],
    *,
    path: str,
    reason: str,
    text: str,
    token: str,
) -> None:
    """按首次命中位置生成稳定文本违规记录."""

    lowered_text = text.lower()
    line_number = lowered_text[: lowered_text.find(token.lower())].count("\n") + 1
    violations.append({"path": path, "reason": reason, "line_number": line_number})


def _scan_business_paths(
    root_path: Path,
    violations: list[dict[str, Any]],
    checked_paths: list[str],
) -> None:
    """检查 Notebook 与业务模块没有第二套动态依赖规则."""

    for relative_root in BUSINESS_SCAN_ROOTS:
        scan_root = root_path / relative_root
        if not scan_root.is_dir():
            continue
        for path in sorted(scan_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in BUSINESS_TEXT_SUFFIXES:
                continue
            relative_path = path.relative_to(root_path).as_posix()
            checked_paths.append(relative_path)
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                violations.append({"path": relative_path, "reason": "dependency_path_not_utf8"})
                continue
            lowered = text.lower()
            if "%pip" in lowered:
                _append_text_violation(
                    violations,
                    path=relative_path,
                    reason="notebook_pip_magic_forbidden",
                    text=text,
                    token="%pip",
                )
            if "--upgrade" in lowered:
                _append_text_violation(
                    violations,
                    path=relative_path,
                    reason="dependency_upgrade_option_forbidden",
                    text=text,
                    token="--upgrade",
                )
            latest_match = _DEPENDENCY_LATEST_PATTERN.search(text)
            if latest_match is not None:
                violations.append(
                    {
                        "path": relative_path,
                        "reason": "mutable_latest_dependency_forbidden",
                        "line_number": text[: latest_match.start()].count("\n") + 1,
                    }
                )
            override_match = _FREE_DEPENDENCY_OVERRIDE_PATTERN.search(text)
            if override_match is not None:
                violations.append(
                    {
                        "path": relative_path,
                        "reason": "free_dependency_spec_override_forbidden",
                        "line_number": text[: override_match.start()].count("\n") + 1,
                    }
                )


def _scan_inner_layer_references(
    root_path: Path,
    violations: list[dict[str, Any]],
    checked_paths: list[str],
) -> None:
    """阻止方法与实验内层反向引用外层 scripts CLI."""

    forbidden_tokens = (
        "scripts.prepare_dependency_profile",
        "scripts.prepare_isolated_dependency_environment",
        "scripts/prepare_dependency_profile.py",
        "scripts/prepare_isolated_dependency_environment.py",
    )
    for relative_root in INNER_LAYER_SCAN_ROOTS:
        scan_root = root_path / relative_root
        if not scan_root.is_dir():
            continue
        for path in sorted(scan_root.rglob("*.py")):
            relative_path = path.relative_to(root_path).as_posix()
            checked_paths.append(relative_path)
            text = path.read_text(encoding="utf-8-sig")
            for token in forbidden_tokens:
                if token not in text:
                    continue
                _append_text_violation(
                    violations,
                    path=relative_path,
                    reason="inner_layer_references_outer_dependency_script",
                    text=text,
                    token=token,
                )
                break


def _audit_registry(
    root_path: Path,
    violations: list[dict[str, Any]],
    checked_paths: list[str],
) -> dict[str, int]:
    """解析六个 profile 并确认缺锁状态保持 fail-closed."""

    registry_path = root_path / "configs/dependency_profile_registry.json"
    profile_count = 0
    direct_dependency_count = 0
    missing_lock_count = 0
    fail_closed_missing_lock_count = 0
    ready_lock_count = 0
    try:
        profiles = load_dependency_profile_registry(registry_path)
    except (FileNotFoundError, ValueError) as exc:
        violations.append(
            {
                "path": "configs/dependency_profile_registry.json",
                "reason": "dependency_profile_registry_invalid",
                "error": str(exc),
            }
        )
        return {
            "profile_count": 0,
            "direct_dependency_count": 0,
            "missing_lock_count": 0,
            "fail_closed_missing_lock_count": 0,
            "ready_lock_count": 0,
        }

    profile_count = len(profiles)
    if tuple(profiles) != REQUIRED_DEPENDENCY_PROFILE_NAMES:
        violations.append(
            {
                "path": "configs/dependency_profile_registry.json",
                "reason": "dependency_profile_set_mismatch",
            }
        )
    for profile_name in REQUIRED_DEPENDENCY_PROFILE_NAMES:
        profile = profiles[profile_name]
        checked_paths.append(profile.direct_requirements_path)
        direct_dependency_count += len(profile.direct_requirements)
        required_installer_requirements = (
            WORKFLOW_ORCHESTRATOR_REQUIRED_DIRECT_REQUIREMENTS
            if profile_name == "workflow_orchestrator"
            else SCIENCE_REQUIRED_INSTALLER_REQUIREMENTS
        )
        missing_installer_requirements = sorted(
            required_installer_requirements - set(profile.direct_requirements)
        )
        if missing_installer_requirements:
            violations.append(
                {
                    "path": profile.direct_requirements_path,
                    "reason": "dependency_profile_installer_pin_missing",
                    "requirements": missing_installer_requirements,
                }
            )
        for specification in profile.direct_requirements:
            try:
                dependency = parse_exact_requirement_spec(specification)
            except ValueError as exc:
                violations.append(
                    {
                        "path": profile.direct_requirements_path,
                        "reason": "direct_dependency_not_exact",
                        "error": str(exc),
                    }
                )
                continue
            if dependency.specification != specification:
                violations.append(
                    {
                        "path": profile.direct_requirements_path,
                        "reason": "direct_dependency_not_canonical",
                        "specification": specification,
                    }
                )

        lock_path = root_path / profile.complete_hash_lock_path
        checked_paths.append(profile.complete_hash_lock_path)
        if lock_path.is_file():
            if not profile.formal_ready or profile.readiness_blockers:
                violations.append(
                    {
                        "path": profile.complete_hash_lock_path,
                        "reason": "valid_hash_lock_not_ready",
                    }
                )
            elif (
                not isinstance(profile.complete_hash_lock_digest, str)
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    profile.complete_hash_lock_digest,
                )
                is None
                or isinstance(profile.complete_hash_lock_dependency_count, bool)
                or not isinstance(
                    profile.complete_hash_lock_dependency_count,
                    int,
                )
                or profile.complete_hash_lock_dependency_count <= 0
            ):
                violations.append(
                    {
                        "path": profile.complete_hash_lock_path,
                        "reason": "ready_hash_lock_identity_invalid",
                    }
                )
            else:
                ready_lock_count += 1
            continue
        missing_lock_count += 1
        if (
            profile.formal_ready is False
            and profile.complete_hash_lock_present is False
            and profile.complete_hash_lock_digest is None
            and profile.readiness_blockers == ("complete_hash_lock_missing",)
        ):
            fail_closed_missing_lock_count += 1
        else:
            violations.append(
                {
                    "path": profile.complete_hash_lock_path,
                    "reason": "missing_hash_lock_not_fail_closed",
                }
            )

    return {
        "profile_count": profile_count,
        "direct_dependency_count": direct_dependency_count,
        "missing_lock_count": missing_lock_count,
        "fail_closed_missing_lock_count": fail_closed_missing_lock_count,
        "ready_lock_count": ready_lock_count,
    }


def run_audit(root: str | Path) -> dict[str, Any]:
    """执行正式依赖环境结构与门禁审计."""

    root_path = Path(root)
    violations: list[dict[str, Any]] = []
    checked_paths: list[str] = []

    for relative_path in REQUIRED_PATHS:
        checked_paths.append(relative_path)
        if not (root_path / relative_path).exists():
            violations.append({"path": relative_path, "reason": "required_dependency_path_missing"})
    for relative_path in FORBIDDEN_PATHS:
        checked_paths.append(relative_path)
        if (root_path / relative_path).exists():
            violations.append({"path": relative_path, "reason": "obsolete_dependency_path_present"})

    field_registry_path = root_path / "docs/field_registry.md"
    if field_registry_path.is_file():
        try:
            registered_fields = _field_names(field_registry_path)
        except UnicodeDecodeError:
            violations.append(
                {
                    "path": "docs/field_registry.md",
                    "reason": "field_registry_not_utf8",
                }
            )
        else:
            missing_fields = sorted(REQUIRED_FIELD_NAMES - registered_fields)
            forbidden_fields = sorted(FORBIDDEN_FIELD_NAMES & registered_fields)
            if missing_fields:
                violations.append(
                    {
                        "path": "docs/field_registry.md",
                        "reason": "dependency_evidence_fields_missing",
                        "fields": missing_fields,
                    }
                )
            if forbidden_fields:
                violations.append(
                    {
                        "path": "docs/field_registry.md",
                        "reason": "dynamic_dependency_fields_forbidden",
                        "fields": forbidden_fields,
                    }
                )

    preparation_path = root_path / "experiments/runtime/dependency_preparation.py"
    if preparation_path.is_file():
        preparation_text = preparation_path.read_text(encoding="utf-8-sig")
        for token in REQUIRED_PREPARATION_TOKENS:
            if token not in preparation_text:
                violations.append(
                    {
                        "path": "experiments/runtime/dependency_preparation.py",
                        "reason": "dependency_preparation_contract_missing",
                        "token": token,
                    }
                )

    isolated_preparation_path = (
        root_path / "experiments/runtime/isolated_dependency_environment.py"
    )
    if isolated_preparation_path.is_file():
        isolated_preparation_text = isolated_preparation_path.read_text(
            encoding="utf-8-sig"
        )
        for token in REQUIRED_ISOLATED_PREPARATION_TOKENS:
            if token not in isolated_preparation_text:
                violations.append(
                    {
                        "path": "experiments/runtime/isolated_dependency_environment.py",
                        "reason": "isolated_dependency_preparation_contract_missing",
                        "token": token,
                    }
                )

    for relative_path, required_imports in THIN_SCRIPT_IMPORTS.items():
        script_path = root_path / relative_path
        if not script_path.is_file():
            continue
        script_text = script_path.read_text(encoding="utf-8-sig")
        for required_import in required_imports:
            if required_import not in script_text:
                violations.append(
                    {
                        "path": relative_path,
                        "reason": "dependency_script_forwarder_missing",
                        "token": required_import,
                    }
                )
        for token in FORBIDDEN_THIN_SCRIPT_IMPLEMENTATION_TOKENS:
            if token in script_text:
                violations.append(
                    {
                        "path": relative_path,
                        "reason": "dependency_script_contains_inner_implementation",
                        "token": token,
                    }
                )

    review_bundle_path = root_path / "scripts/write_dependency_lock_review_bundle.py"
    if review_bundle_path.is_file():
        review_bundle_text = review_bundle_path.read_text(encoding="utf-8-sig")
        for token in REQUIRED_LOCK_REVIEW_TOKENS:
            if token not in review_bundle_text:
                violations.append(
                    {
                        "path": "scripts/write_dependency_lock_review_bundle.py",
                        "reason": "dependency_lock_review_isolation_contract_missing",
                        "token": token,
                    }
                )

    candidate_materializer_path = (
        root_path / "scripts/materialize_dependency_lock_candidate.py"
    )
    if candidate_materializer_path.is_file():
        candidate_materializer_text = candidate_materializer_path.read_text(
            encoding="utf-8-sig"
        )
        for token in REQUIRED_CANDIDATE_MATERIALIZER_TOKENS:
            if token not in candidate_materializer_text:
                violations.append(
                    {
                        "path": "scripts/materialize_dependency_lock_candidate.py",
                        "reason": "dependency_lock_materializer_contract_missing",
                        "token": token,
                    }
                )

    acceptance_path = root_path / "scripts/write_reviewed_dependency_hash_lock.py"
    if acceptance_path.is_file():
        acceptance_text = acceptance_path.read_text(encoding="utf-8-sig")
        for token in REQUIRED_LOCK_ACCEPTANCE_TOKENS:
            if token not in acceptance_text:
                violations.append(
                    {
                        "path": "scripts/write_reviewed_dependency_hash_lock.py",
                        "reason": "dependency_lock_acceptance_contract_missing",
                        "token": token,
                    }
                )

    qualification_tool_lock_path = (
        root_path
        / (
            "configs/dependency_profiles/"
            "dependency_qualification_uv_linux_x86_64_lock.txt"
        )
    )
    if qualification_tool_lock_path.is_file():
        qualification_tool_lock_lines = [
            line.strip()
            for line in qualification_tool_lock_path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        ]
        if qualification_tool_lock_lines != [QUALIFICATION_TOOL_LOCK_TEXT]:
            violations.append(
                {
                    "path": (
                        "configs/dependency_profiles/"
                        "dependency_qualification_uv_linux_x86_64_lock.txt"
                    ),
                    "reason": "dependency_qualification_tool_lock_mismatch",
                }
            )

    repository_environment_path = (
        root_path / "experiments/runtime/repository_environment.py"
    )
    if repository_environment_path.is_file():
        repository_environment_text = repository_environment_path.read_text(
            encoding="utf-8-sig"
        )
        for token in REQUIRED_ISOLATED_SCIENTIFIC_CONTEXT_TOKENS:
            if token not in repository_environment_text:
                violations.append(
                    {
                        "path": "experiments/runtime/repository_environment.py",
                        "reason": "isolated_scientific_context_contract_missing",
                        "token": token,
                    }
                )
    registry_summary = _audit_registry(root_path, violations, checked_paths)
    _scan_business_paths(root_path, violations, checked_paths)
    _scan_inner_layer_references(root_path, violations, checked_paths)

    report = build_report(
        "audit_dependency_profile_governance",
        "fail" if violations else "pass",
        violations,
        checked_paths,
    )
    report["summary"].update(registry_summary)
    return report


def main() -> None:
    """执行审计并用退出码表达结果."""

    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
