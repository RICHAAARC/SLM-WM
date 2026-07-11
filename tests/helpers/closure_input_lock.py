"""为需要 CPU 结果闭合来源的功能测试写出精确输入锁."""

from __future__ import annotations

import json
from pathlib import Path

from main.core.digest import build_stable_digest
from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
    LOCK_FILENAME,
    LOCK_MANIFEST_FILENAME,
    LOCK_OUTPUT_ROOT,
)
from tests.helpers.formal_execution_lock import build_test_formal_execution_lock


def _scientific_lock_fields(spec: object, index: int) -> dict[str, object]:
    """为共享闭合锁夹具构造与 family 契约一致的科学执行摘要."""

    binding = getattr(spec, "scientific_execution_binding")
    dependency = getattr(spec, "dependency_environment_evidence")
    profile_id = (
        binding.profile_id
        if binding is not None
        else dependency.profile_id
        if dependency is not None
        else ""
    )
    if not profile_id:
        return {
            "scientific_profile_id": "",
            "scientific_profile_digest": "",
            "scientific_direct_requirements_digest": "",
            "scientific_complete_hash_lock_digest": "",
            "scientific_complete_hash_lock_dependency_count": 0,
            "scientific_python_executable_digest": "",
            "scientific_execution_report_digest": "",
            "scientific_command_dispatch_report_digest": "",
            "scientific_command_sequence_digest": "",
            "scientific_execution_binding_digest": "",
            "scientific_dependency_evidence_digest": "",
        }

    package_family = str(getattr(spec, "package_family"))
    semantic_family = package_family in {
        "image_only_dataset_runtime",
        "runtime_rerun_ablation",
        "dataset_level_quality",
    }

    def digest(offset: int) -> str:
        """生成规范小写 SHA-256 形状的确定性测试值."""

        value = ((index + offset) % 15) + 1
        return format(value, "x") * 64

    profile_digest = "1" * 64 if semantic_family else digest(1)
    direct_digest = "2" * 64 if semantic_family else digest(2)
    complete_digest = "3" * 64 if semantic_family else digest(3)
    python_digest = "4" * 64 if semantic_family else digest(4)
    execution_digest = "5" * 64 if semantic_family else digest(5)
    dependency_digest = "7" * 64 if semantic_family else digest(7)
    return {
        "scientific_profile_id": profile_id,
        "scientific_profile_digest": profile_digest,
        "scientific_direct_requirements_digest": direct_digest,
        "scientific_complete_hash_lock_digest": complete_digest,
        "scientific_complete_hash_lock_dependency_count": 17,
        "scientific_python_executable_digest": python_digest,
        "scientific_execution_report_digest": (
            execution_digest if binding is not None else ""
        ),
        "scientific_command_dispatch_report_digest": (
            digest(8) if binding is not None else ""
        ),
        "scientific_command_sequence_digest": (
            "6" * 64 if semantic_family else ""
        ),
        "scientific_execution_binding_digest": (
            digest(9) if binding is not None else ""
        ),
        "scientific_dependency_evidence_digest": dependency_digest,
    }


def build_test_closure_input_lock_payloads(
    *,
    paper_run_name: str,
    target_fpr: float,
    common_code_version: str = "a" * 40,
    package_root: str | Path = "D:/drive",
) -> tuple[dict[str, object], dict[str, object]]:
    """构造不读取大型 ZIP, 但满足下游身份复验的10类测试锁."""

    execution_lock = build_test_formal_execution_lock(common_code_version)
    run_lock_digest = execution_lock["formal_execution_lock_digest"]
    package_lock_digest = execution_lock["formal_execution_lock_digest"]
    resolved_package_root = Path(package_root)
    records = []
    for index, spec in enumerate(CLOSURE_PACKAGE_FAMILY_SPECS):
        records.append(
            {
                "package_family": spec.package_family,
                "package_path": (
                    resolved_package_root / f"{spec.package_family}.zip"
                ).as_posix(),
                "package_sha256": f"{index + 1:x}" * 64,
                "paper_run_name": paper_run_name,
                "target_fpr": target_fpr,
                "code_version": common_code_version,
                "formal_execution_run_lock_digest": run_lock_digest,
                "formal_execution_package_lock_digest": package_lock_digest,
                "generated_at": "2026-07-11T00:00:00+00:00",
                **_scientific_lock_fields(spec, index),
            }
        )
    lock_payload: dict[str, object] = {
        "paper_run_name": paper_run_name,
        "target_fpr": target_fpr,
        "common_code_version": common_code_version,
        "closure_input_package_count": len(records),
        "closure_input_packages": records,
        "formal_execution_run_lock_digests": {
            record["package_family"]: record["formal_execution_run_lock_digest"]
            for record in records
        },
        "formal_execution_package_lock_digests": {
            record["package_family"]: record["formal_execution_package_lock_digest"]
            for record in records
        },
    }
    lock_payload["closure_input_lock_digest"] = build_stable_digest(lock_payload)
    manifest = {
        "artifact_id": f"{paper_run_name}_closure_input_lock_manifest",
        "artifact_type": "local_manifest",
        "input_paths": [record["package_path"] for record in records],
        "output_paths": [
            f"{LOCK_OUTPUT_ROOT.as_posix()}/{paper_run_name}/{LOCK_FILENAME}",
            f"{LOCK_OUTPUT_ROOT.as_posix()}/{paper_run_name}/{LOCK_MANIFEST_FILENAME}",
        ],
        "config_digest": "a" * 64,
        "code_version": common_code_version,
        "rebuild_command": "test_closure_input_lock",
        "config": {},
        "metadata": {
            "closure_input_lock_ready": True,
            "closure_input_package_count": len(records),
            "closure_input_packages": records,
            "closure_input_lock_digest": lock_payload["closure_input_lock_digest"],
            "paper_run_name": paper_run_name,
            "target_fpr": target_fpr,
            "common_code_version": common_code_version,
        },
    }
    return lock_payload, manifest


def write_test_closure_input_lock(
    root: Path,
    *,
    paper_run_name: str,
    target_fpr: float,
    common_code_version: str = "a" * 40,
) -> tuple[Path, Path]:
    """把共享闭合锁夹具写入当前测试仓库的受治理路径."""

    lock_payload, manifest = build_test_closure_input_lock_payloads(
        paper_run_name=paper_run_name,
        target_fpr=target_fpr,
        common_code_version=common_code_version,
        package_root=root / "drive",
    )
    output_dir = root / LOCK_OUTPUT_ROOT / paper_run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / LOCK_FILENAME
    manifest_path = output_dir / LOCK_MANIFEST_FILENAME
    lock_path.write_text(
        json.dumps(lock_payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return lock_path, manifest_path
