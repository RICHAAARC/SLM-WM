"""当前论文运行层级完整结果包的轻量功能测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_STORED, ZipFile

import pytest

from experiments.runtime import repository_environment
from main.core.digest import build_stable_digest
from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
    ClosurePackageSelectionError,
    validate_closure_input_lock_payloads,
)
from scripts.write_pilot_paper_complete_result_package import (
    PACKAGE_EXTRA_PATHS,
    REQUIRED_OUTPUT_DIR_NAMES,
    build_closure_input_lock_status,
    build_dependency_lock_status,
    build_parser,
    build_required_output_dirs,
    write_archive_with_progress,
    write_pilot_paper_complete_result_package_outputs,
)
from tests.helpers.formal_execution_lock import build_test_formal_execution_lock


class _ClosureCandidate:
    """为完整包测试返回与输入锁逐字段相等的动态候选记录."""

    def __init__(self, record: dict[str, Any]) -> None:
        self._record = dict(record)

    def to_lock_record(self) -> dict[str, Any]:
        """返回未硬编码候选字段集合的测试锁记录."""

        return dict(self._record)


def write_json(path: Path, value: object) -> None:
    """写出测试 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def build_test_lock_record(
    package_path: Path,
    *,
    package_family: str,
    paper_run_name: str,
    target_fpr: float,
) -> dict[str, Any]:
    """从实际测试 ZIP 构造 selector 候选会返回的精确记录."""

    family_index = next(
        index
        for index, spec in enumerate(CLOSURE_PACKAGE_FAMILY_SPECS)
        if spec.package_family == package_family
    )
    return {
        "package_family": package_family,
        "package_path": package_path.resolve().as_posix(),
        "package_sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
        "paper_run_name": paper_run_name,
        "target_fpr": target_fpr,
        "code_version": "a" * 40,
        "generated_at": f"2026-01-01T00:00:{family_index:02d}+00:00",
    }


def inspect_test_closure_package(
    package_path: Path,
    *,
    spec: Any,
    paper_run_name: str,
    target_fpr: float,
) -> _ClosureCandidate:
    """模拟已由 selector 深度检查并返回动态锁字段的候选包."""

    return _ClosureCandidate(
        build_test_lock_record(
            Path(package_path),
            package_family=spec.package_family,
            paper_run_name=paper_run_name,
            target_fpr=target_fpr,
        )
    )


def configure_paper_run(monkeypatch: pytest.MonkeyPatch, root: Path, paper_run_name: str) -> Path:
    """配置一个最小当前论文运行层级与对应 Prompt 文件。"""

    prompt_path = root / "configs" / f"paper_main_{paper_run_name}_prompts.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(f"a {paper_run_name} prompt\n", encoding="utf-8")
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", paper_run_name)
    monkeypatch.setenv("SLM_WM_PROMPT_SET", paper_run_name)
    monkeypatch.setenv("SLM_WM_PROMPT_FILE", prompt_path.relative_to(root).as_posix())
    monkeypatch.setenv("SLM_WM_PAPER_RUN_SAMPLE_COUNT", "all")
    monkeypatch.setattr(
        "scripts.write_pilot_paper_complete_result_package.resolve_code_version",
        lambda _root_path: "a" * 40,
    )
    execution_lock = build_test_formal_execution_lock("a" * 40)
    monkeypatch.setattr(
        repository_environment,
        "require_published_formal_execution_lock",
        lambda _root: dict(execution_lock),
    )
    monkeypatch.setattr(
        "scripts.write_pilot_paper_complete_result_package.build_dependency_lock_status",
        lambda _root, *, archive_entries: {
            "dependency_profile_count": 6,
            "dependency_profile_records": [],
            "dependency_hash_lock_count": 6,
            "dependency_hash_lock_archive_entries_ready": True,
            "dependency_profile_inputs_archive_entries_ready": True,
            "dependency_hash_locks_ready": True,
            "dependency_hash_lock_failure_reason": "",
        },
    )
    monkeypatch.setattr(
        "scripts.write_pilot_paper_complete_result_package.validate_closure_input_lock_payloads",
        lambda _lock, _manifest, *, paper_run_name, target_fpr: {
            "paper_run_name": paper_run_name,
            "target_fpr": target_fpr,
        },
    )
    monkeypatch.setattr(
        "scripts.write_pilot_paper_complete_result_package.inspect_closure_package",
        inspect_test_closure_package,
    )
    monkeypatch.setattr(
        "scripts.write_pilot_paper_complete_result_package._validate_candidate_repository_profile",
        lambda _candidate, *, root_path: None,
    )
    return prompt_path


def create_required_outputs(
    root: Path,
    paper_run_name: str,
    *,
    paper_run_claim_ready: bool = True,
    result_closure_ready: bool = True,
    omitted_directory_name: str = "",
) -> tuple[str, ...]:
    """创建带 manifest 的当前运行层级输出目录。"""

    required_dirs = build_required_output_dirs(paper_run_name)
    governed_source_paths: list[Path] = []
    for index, relative_dir in enumerate(required_dirs):
        if relative_dir.endswith(f"/{omitted_directory_name}/{paper_run_name}"):
            continue
        output_dir = root / relative_dir
        output_dir.mkdir(parents=True)
        sample_manifest_path = output_dir / f"sample_manifest_{index}.json"
        write_json(
            sample_manifest_path,
            {"paper_claim_scale": paper_run_name, "index": index},
        )
        governed_source_paths.append(sample_manifest_path)
    common_summary = (
        root
        / "outputs"
        / "pilot_paper_fixed_fpr_common_protocol"
        / paper_run_name
        / "pilot_paper_common_protocol_summary.json"
    )
    if common_summary.parent.is_dir():
        write_json(
            common_summary,
            {
                "paper_run_claim_ready": paper_run_claim_ready,
                "paper_run_claim_type": paper_run_name.replace("_paper", "_claim"),
                "probe_claim_ready": paper_run_name == "probe_paper" and paper_run_claim_ready,
                "pilot_claim_ready": paper_run_name == "pilot_paper" and paper_run_claim_ready,
                "full_claim_ready": paper_run_name == "full_paper" and paper_run_claim_ready,
            },
        )
    gate_dir = root / "outputs" / "result_closure_gate" / paper_run_name
    if gate_dir.is_dir():
        target_fpr = {
            "probe_paper": 0.1,
            "pilot_paper": 0.01,
            "full_paper": 0.001,
        }[paper_run_name]
        gate_report_path = gate_dir / "result_closure_gate_report.json"
        gate_manifest_path = gate_dir / "manifest.local.json"
        closure_source_file_sha256 = {
            path.relative_to(root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in [common_summary, *governed_source_paths]
        }
        closure_source_file_digest = build_stable_digest(closure_source_file_sha256)
        expected_counts = {
            "probe_paper": (70, 34),
            "pilot_paper": (700, 340),
            "full_paper": (7000, 3400),
        }[paper_run_name]
        gate_report = {
            "paper_claim_scale": paper_run_name,
            "target_fpr": target_fpr,
            "expected_prompt_count": expected_counts[0],
            "expected_test_count": expected_counts[1],
            "expected_prompt_id_digest": "c" * 64,
            "result_closure_ready": result_closure_ready,
            "closure_decision": "pass" if result_closure_ready else "blocked",
            "evidence_closure_allowed": result_closure_ready,
            "blocked_check_count": 0 if result_closure_ready else 1,
            "supports_paper_claim": result_closure_ready,
            "source_artifact_digests": {},
            "closure_source_file_sha256": closure_source_file_sha256,
            "closure_source_file_digest": closure_source_file_digest,
        }
        write_json(gate_report_path, gate_report)
        input_bundle_digest = "b" * 64
        report_digest = hashlib.sha256(gate_report_path.read_bytes()).hexdigest()
        gate_config = {
            "paper_claim_scale": paper_run_name,
            "target_fpr": target_fpr,
            "expected_prompt_count": expected_counts[0],
            "expected_test_count": expected_counts[1],
            "expected_prompt_id_digest": "c" * 64,
            "input_bundle_digest": input_bundle_digest,
            "report_digest": report_digest,
            "source_artifact_digests": {},
            "closure_source_file_sha256": closure_source_file_sha256,
            "closure_source_file_digest": closure_source_file_digest,
        }
        write_json(
            gate_manifest_path,
            {
                "artifact_id": f"{paper_run_name}_result_closure_gate_manifest",
                "artifact_type": "local_manifest",
                "input_paths": sorted(closure_source_file_sha256),
                "output_paths": [
                    gate_report_path.relative_to(root).as_posix(),
                    gate_manifest_path.relative_to(root).as_posix(),
                ],
                "config_digest": build_stable_digest(gate_config),
                "code_version": "a" * 40,
                "metadata": {
                    "paper_claim_scale": paper_run_name,
                    "target_fpr": target_fpr,
                    "result_closure_ready": result_closure_ready,
                    "closure_decision": "pass" if result_closure_ready else "blocked",
                    "evidence_closure_allowed": result_closure_ready,
                    "expected_prompt_id_digest": "c" * 64,
                    "input_bundle_digest": input_bundle_digest,
                    "report_digest": report_digest,
                    "source_artifact_digests": {},
                    "closure_source_file_sha256": closure_source_file_sha256,
                    "closure_source_file_digest": closure_source_file_digest,
                },
            },
        )
    return required_dirs


def create_closure_input_lock(root: Path, paper_run_name: str) -> tuple[Path, ...]:
    """创建10个小型显式 zip 与内容绑定的当前运行层级输入锁。"""

    target_fpr_by_run = {
        "probe_paper": 0.1,
        "pilot_paper": 0.01,
        "full_paper": 0.001,
    }
    target_fpr = target_fpr_by_run[paper_run_name]
    package_root = root / "drive" / "locked_packages" / paper_run_name
    package_root.mkdir(parents=True, exist_ok=True)
    package_paths: list[Path] = []
    lock_records: list[dict[str, object]] = []
    for spec in CLOSURE_PACKAGE_FAMILY_SPECS:
        package_path = package_root / f"{spec.package_family}.zip"
        output_prefix = spec.allowed_output_prefix_templates[0].format(
            paper_run=paper_run_name,
            baseline=spec.baseline_id or "",
        )
        source_marker_name = output_prefix + "locked_source_marker.json"
        source_marker_payload = json.dumps(
            {"package_family": spec.package_family},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        with ZipFile(package_path, "w") as archive:
            archive.writestr(
                f"governance/{spec.package_family}.json",
                json.dumps({"package_family": spec.package_family}),
            )
            archive.writestr(source_marker_name, source_marker_payload)
        source_marker_path = root / source_marker_name
        source_marker_path.parent.mkdir(parents=True, exist_ok=True)
        source_marker_path.write_bytes(source_marker_payload)
        package_paths.append(package_path)
        lock_records.append(
            build_test_lock_record(
                package_path,
                package_family=spec.package_family,
                paper_run_name=paper_run_name,
                target_fpr=target_fpr,
            )
        )
    lock_payload: dict[str, object] = {
        "paper_run_name": paper_run_name,
        "target_fpr": target_fpr,
        "common_code_version": "a" * 40,
        "closure_input_package_count": len(lock_records),
        "closure_input_packages": lock_records,
    }
    lock_payload["closure_input_lock_digest"] = build_stable_digest(lock_payload)
    closure_dir = root / "outputs" / "paper_result_closure" / paper_run_name
    closure_dir.mkdir(parents=True, exist_ok=True)
    lock_path = closure_dir / "closure_input_lock.json"
    manifest_path = closure_dir / "input_lock_manifest.local.json"
    write_json(lock_path, lock_payload)
    manifest_config = {
        "paper_run_name": paper_run_name,
        "target_fpr": target_fpr,
        "common_code_version": "a" * 40,
        "closure_input_packages": lock_records,
    }
    write_json(
        manifest_path,
        {
            "artifact_id": f"{paper_run_name}_closure_input_lock_manifest",
            "artifact_type": "local_manifest",
            "input_paths": [str(row["package_path"]) for row in lock_records],
            "output_paths": [
                lock_path.relative_to(root).as_posix(),
                manifest_path.relative_to(root).as_posix(),
            ],
            "config_digest": build_stable_digest(manifest_config),
            "metadata": {
                "closure_input_lock_ready": True,
                "closure_input_package_count": len(lock_records),
                "closure_input_packages": lock_records,
                "closure_input_lock_digest": lock_payload["closure_input_lock_digest"],
                "paper_run_name": paper_run_name,
                "target_fpr": target_fpr,
                "common_code_version": "a" * 40,
            },
        },
    )
    return tuple(package_paths)


@pytest.mark.quick
def test_complete_result_package_collects_only_current_paper_run_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整结果包应覆盖当前运行层级, 且排除其他层级的结果与 Prompt。"""

    paper_run_name = "pilot_paper"
    prompt_path = configure_paper_run(monkeypatch, tmp_path, paper_run_name)
    required_dirs = create_required_outputs(tmp_path, paper_run_name)
    package_paths = create_closure_input_lock(tmp_path, paper_run_name)
    stale_same_run_path = (
        tmp_path
        / "outputs"
        / "attack_matrix"
        / paper_run_name
        / "stale_same_run.json"
    )
    stale_same_run_path.write_text("{}\n", encoding="utf-8")
    for other_run_name in ("probe_paper", "full_paper"):
        other_prompt_path = tmp_path / "configs" / f"paper_main_{other_run_name}_prompts.txt"
        other_prompt_path.write_text(f"a {other_run_name} prompt\n", encoding="utf-8")
        other_output = tmp_path / "outputs" / "attack_matrix" / other_run_name
        other_output.mkdir(parents=True)
        (other_output / "must_not_be_packaged.json").write_text("{}\n", encoding="utf-8")

    drive_dir = tmp_path / "drive" / "SLM" / "pilot_paper_results" / "complete_result_package"
    receipt = write_pilot_paper_complete_result_package_outputs(
        root=tmp_path,
        drive_output_dir=str(drive_dir),
        package_paths=package_paths,
    )
    output_dir = tmp_path / "outputs" / "pilot_paper_complete_result_package"
    archive_path = tmp_path / receipt["archive_path"]
    receipt_path = output_dir / "pilot_paper_complete_result_package_archive_receipt.json"
    internal_manifest_path = output_dir / "pilot_paper_complete_package_manifest.local.json"

    assert receipt["metadata"]["pilot_paper_complete_result_package_ready"] is True
    assert archive_path.is_file()
    assert receipt_path.is_file()
    assert (drive_dir / archive_path.name).is_file()
    assert (drive_dir / receipt_path.name).is_file()
    internal_manifest = json.loads(internal_manifest_path.read_text(encoding="utf-8"))
    assert internal_manifest["artifact_id"] == "pilot_paper_complete_result_package_manifest"

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        compression_types = {entry.compress_type for entry in archive.infolist()}
        source_provenance_name = (
            "outputs/pilot_paper_complete_result_package/"
            "pilot_paper_complete_package_source_provenance.json"
        )
        source_provenance = json.loads(archive.read(source_provenance_name))
        archived_source_digests = {
            package_record["archived_source_path"]: hashlib.sha256(
                archive.read(package_record["archived_source_path"])
            ).hexdigest()
            for package_record in source_provenance["input_packages"]
        }
    attack_dir = f"outputs/attack_matrix/{paper_run_name}"
    attack_index = required_dirs.index(attack_dir)
    assert f"{attack_dir}/sample_manifest_{attack_index}.json" in names
    assert f"outputs/fixed_fpr_threshold_audit/{paper_run_name}" in "\n".join(names)
    assert f"outputs/primary_baseline_evidence/{paper_run_name}" in "\n".join(names)
    assert f"outputs/paper_artifact_evidence_audit/{paper_run_name}" in "\n".join(names)
    assert f"outputs/submission_readiness/{paper_run_name}" in "\n".join(names)
    assert f"outputs/evidence_closure_entry_review/{paper_run_name}" in "\n".join(names)
    assert f"outputs/paper_result_closure/{paper_run_name}" in "\n".join(names)
    assert f"outputs/result_closure_gate/{paper_run_name}" in "\n".join(names)
    assert prompt_path.relative_to(tmp_path).as_posix() in names
    assert stale_same_run_path.relative_to(tmp_path).as_posix() not in names
    assert source_provenance["decision"] == "pass"
    assert source_provenance["input_package_count"] == len(
        CLOSURE_PACKAGE_FAMILY_SPECS
    )
    assert len(source_provenance["input_packages"]) == len(
        CLOSURE_PACKAGE_FAMILY_SPECS
    )
    for package_record in source_provenance["input_packages"]:
        archived_source_path = package_record["archived_source_path"]
        assert archived_source_path in names
        assert archived_source_digests[archived_source_path] == (
            package_record["source_package_sha256"]
        )
        assert package_record["archived_source_sha256"] == package_record[
            "source_package_sha256"
        ]
        assert package_record["member_count"] == 2
    assert "configs/paper_main_probe_paper_prompts.txt" not in names
    assert "configs/paper_main_full_paper_prompts.txt" not in names
    assert not any("/probe_paper/" in name or "/full_paper/" in name for name in names)
    assert receipt_path.relative_to(tmp_path).as_posix() not in names
    assert compression_types == {ZIP_STORED}


@pytest.mark.quick
def test_complete_result_package_internal_metadata_is_not_overwritten_after_archiving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """最终摘要应只写入包外 receipt, 归档内 metadata 应与本地文件完全一致。"""

    configure_paper_run(monkeypatch, tmp_path, "probe_paper")
    create_required_outputs(tmp_path, "probe_paper")
    package_paths = create_closure_input_lock(tmp_path, "probe_paper")
    receipt = write_pilot_paper_complete_result_package_outputs(
        root=tmp_path,
        drive_output_dir="",
        package_paths=package_paths,
    )
    output_dir = tmp_path / "outputs" / "pilot_paper_complete_result_package"
    archive_path = tmp_path / receipt["archive_path"]
    summary_path = output_dir / "probe_paper_complete_package_readiness_summary.json"
    manifest_path = output_dir / "probe_paper_complete_package_manifest.local.json"

    with ZipFile(archive_path) as archive:
        archived_summary = archive.read(summary_path.relative_to(tmp_path).as_posix())
        archived_manifest = archive.read(manifest_path.relative_to(tmp_path).as_posix())
        archive_names = set(archive.namelist())

    assert archived_summary == summary_path.read_bytes()
    assert archived_manifest == manifest_path.read_bytes()
    assert not any(name.endswith("_archive_receipt.json") for name in archive_names)
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == receipt["archive_digest"]
    assert receipt["metadata"]["archive_digest_scope"] == "final_zip_bytes_external_sidecar"
    assert receipt["metadata"]["final_archive_digest_available_in_sidecar"] is True


@pytest.mark.quick
def test_complete_result_package_rejects_written_zip_member_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ZIP 写后成员集合偏离冻结清单时必须删除归档并失败."""

    configure_paper_run(monkeypatch, tmp_path, "probe_paper")
    create_required_outputs(tmp_path, "probe_paper")
    package_paths = create_closure_input_lock(tmp_path, "probe_paper")

    def write_with_unexpected_member(
        root_path: Path,
        archive_path: Path,
        entries: Any,
        compression_method: int,
    ) -> None:
        write_archive_with_progress(
            root_path,
            archive_path,
            entries,
            compression_method,
        )
        with ZipFile(archive_path, "a") as archive:
            archive.writestr("outputs/unexpected.json", "{}\n")

    monkeypatch.setattr(
        "scripts.write_pilot_paper_complete_result_package.write_archive_with_progress",
        write_with_unexpected_member,
    )

    with pytest.raises(RuntimeError, match="写后成员路径集合"):
        write_pilot_paper_complete_result_package_outputs(
            root=tmp_path,
            drive_output_dir="",
            package_paths=package_paths,
            materialize_packages=False,
        )

    assert not (
        tmp_path
        / "outputs"
        / "pilot_paper_complete_result_package"
        / "probe_paper_complete_result_package.zip"
    ).exists()


@pytest.mark.quick
def test_complete_result_package_rechecks_locked_package_hashes_after_zip_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ZIP 写入期间输入包发生漂移时最终摘要重验必须阻止交付."""

    configure_paper_run(monkeypatch, tmp_path, "probe_paper")
    create_required_outputs(tmp_path, "probe_paper")
    package_paths = create_closure_input_lock(tmp_path, "probe_paper")

    def write_then_mutate_input(
        root_path: Path,
        archive_path: Path,
        entries: Any,
        compression_method: int,
    ) -> None:
        write_archive_with_progress(
            root_path,
            archive_path,
            entries,
            compression_method,
        )
        package_paths[0].write_bytes(b"changed after readiness")

    monkeypatch.setattr(
        "scripts.write_pilot_paper_complete_result_package.write_archive_with_progress",
        write_then_mutate_input,
    )

    with pytest.raises(RuntimeError, match="闭合输入包路径或摘要发生漂移"):
        write_pilot_paper_complete_result_package_outputs(
            root=tmp_path,
            drive_output_dir="",
            package_paths=package_paths,
            materialize_packages=False,
        )

    assert not (
        tmp_path
        / "outputs"
        / "pilot_paper_complete_result_package"
        / "probe_paper_complete_result_package.zip"
    ).exists()


@pytest.mark.quick
def test_complete_result_package_rejects_not_ready_run_before_zip_or_drive_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """共同协议 readiness 为 false 时不得创建 zip 或 Drive 目录。"""

    configure_paper_run(monkeypatch, tmp_path, "pilot_paper")
    create_required_outputs(tmp_path, "pilot_paper", paper_run_claim_ready=False)
    package_paths = create_closure_input_lock(tmp_path, "pilot_paper")
    output_dir = tmp_path / "outputs" / "pilot_paper_complete_result_package"
    output_dir.mkdir(parents=True)
    stale_archive = output_dir / "pilot_paper_complete_result_package.zip"
    stale_archive.write_bytes(b"stale")
    drive_dir = tmp_path / "drive" / "complete_result_package"

    with pytest.raises(RuntimeError, match="readiness 未通过"):
        write_pilot_paper_complete_result_package_outputs(
            root=tmp_path,
            drive_output_dir=str(drive_dir),
            package_paths=package_paths,
        )

    assert not stale_archive.exists()
    assert not drive_dir.exists()
    readiness_path = output_dir / "pilot_paper_complete_package_readiness_summary.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert readiness["paper_run_complete_result_package_ready"] is False
    assert not (output_dir / "pilot_paper_complete_result_package_archive_receipt.json").exists()


@pytest.mark.quick
def test_complete_result_package_requires_semantic_result_closure_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """closure gate 未通过时, 目录齐全也不得创建完整结果包。"""

    configure_paper_run(monkeypatch, tmp_path, "pilot_paper")
    create_required_outputs(tmp_path, "pilot_paper", result_closure_ready=False)
    package_paths = create_closure_input_lock(tmp_path, "pilot_paper")

    with pytest.raises(RuntimeError, match="result_closure_ready=False"):
        write_pilot_paper_complete_result_package_outputs(
            root=tmp_path,
            drive_output_dir="",
            package_paths=package_paths,
        )

    readiness_path = (
        tmp_path
        / "outputs"
        / "pilot_paper_complete_result_package"
        / "pilot_paper_complete_package_readiness_summary.json"
    )
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert readiness["result_closure_ready"] is False
    assert readiness["paper_run_complete_result_package_ready"] is False


@pytest.mark.quick
def test_complete_result_package_requires_exact_locked_explicit_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式包集合少于输入锁或包内容被替换时必须 fail-closed。"""

    configure_paper_run(monkeypatch, tmp_path, "pilot_paper")
    create_required_outputs(tmp_path, "pilot_paper")
    package_paths = create_closure_input_lock(tmp_path, "pilot_paper")

    with pytest.raises(RuntimeError, match="closure_input_lock_ready=False"):
        write_pilot_paper_complete_result_package_outputs(
            root=tmp_path,
            drive_output_dir="",
            package_paths=package_paths[:-1],
            materialize_packages=False,
        )

    package_paths[0].write_bytes(b"lock digest mismatch")
    with pytest.raises(RuntimeError, match="closure_input_lock_ready=False"):
        write_pilot_paper_complete_result_package_outputs(
            root=tmp_path,
            drive_output_dir="",
            package_paths=package_paths,
            materialize_packages=False,
        )


@pytest.mark.quick
def test_closure_input_status_revalidates_semantics_packages_and_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整包状态必须重新调用语义,逐包和仓库 profile 三层校验."""

    configure_paper_run(monkeypatch, tmp_path, "probe_paper")
    package_paths = create_closure_input_lock(tmp_path, "probe_paper")
    calls = {"semantic": 0, "inspection": 0, "profile": 0}

    def semantic_validator(
        _lock: dict[str, Any],
        _manifest: dict[str, Any],
        *,
        paper_run_name: str,
        target_fpr: float,
    ) -> dict[str, Any]:
        calls["semantic"] += 1
        return {
            "paper_run_name": paper_run_name,
            "target_fpr": target_fpr,
        }

    def package_inspector(
        package_path: Path,
        *,
        spec: Any,
        paper_run_name: str,
        target_fpr: float,
    ) -> _ClosureCandidate:
        calls["inspection"] += 1
        return inspect_test_closure_package(
            package_path,
            spec=spec,
            paper_run_name=paper_run_name,
            target_fpr=target_fpr,
        )

    def profile_validator(_candidate: Any, *, root_path: Path) -> None:
        assert root_path == tmp_path.resolve()
        calls["profile"] += 1

    monkeypatch.setattr(
        "scripts.write_pilot_paper_complete_result_package.validate_closure_input_lock_payloads",
        semantic_validator,
    )
    monkeypatch.setattr(
        "scripts.write_pilot_paper_complete_result_package.inspect_closure_package",
        package_inspector,
    )
    monkeypatch.setattr(
        "scripts.write_pilot_paper_complete_result_package._validate_candidate_repository_profile",
        profile_validator,
    )

    status = build_closure_input_lock_status(
        tmp_path.resolve(),
        paper_run_name="probe_paper",
        target_fpr=0.1,
        explicit_packages=package_paths,
    )

    assert status["closure_input_lock_ready"] is True
    assert status["closure_input_lock_semantic_validation_ready"] is True
    assert status["closure_input_package_inspection_ready"] is True
    assert status["closure_input_repository_profiles_ready"] is True
    assert status["failure_reasons"] == []
    assert calls == {"semantic": 1, "inspection": 10, "profile": 10}


@pytest.mark.quick
def test_closure_input_status_rejects_semantically_incomplete_lock_without_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """自摘要正确但缺少科学字段的旧式锁必须返回非 ready 状态."""

    configure_paper_run(monkeypatch, tmp_path, "probe_paper")
    package_paths = create_closure_input_lock(tmp_path, "probe_paper")
    monkeypatch.setattr(
        "scripts.write_pilot_paper_complete_result_package.validate_closure_input_lock_payloads",
        validate_closure_input_lock_payloads,
    )

    status = build_closure_input_lock_status(
        tmp_path.resolve(),
        paper_run_name="probe_paper",
        target_fpr=0.1,
        explicit_packages=package_paths,
    )

    assert status["closure_input_lock_semantic_validation_ready"] is False
    assert status["closure_input_package_inspection_ready"] is True
    assert status["closure_input_lock_ready"] is False
    assert any(
        reason.startswith("closure_input_lock_semantic_validation_failed:")
        for reason in status["failure_reasons"]
    )


@pytest.mark.quick
def test_closure_input_status_rejects_package_inspection_failure_without_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任一 family 深度检查失败时状态 API 应 fail-closed 而不是抛出异常."""

    configure_paper_run(monkeypatch, tmp_path, "probe_paper")
    package_paths = create_closure_input_lock(tmp_path, "probe_paper")

    def rejecting_inspector(
        package_path: Path,
        *,
        spec: Any,
        paper_run_name: str,
        target_fpr: float,
    ) -> _ClosureCandidate:
        if spec.package_family == CLOSURE_PACKAGE_FAMILY_SPECS[0].package_family:
            raise ClosurePackageSelectionError("测试包检查失败")
        return inspect_test_closure_package(
            package_path,
            spec=spec,
            paper_run_name=paper_run_name,
            target_fpr=target_fpr,
        )

    monkeypatch.setattr(
        "scripts.write_pilot_paper_complete_result_package.inspect_closure_package",
        rejecting_inspector,
    )

    status = build_closure_input_lock_status(
        tmp_path.resolve(),
        paper_run_name="probe_paper",
        target_fpr=0.1,
        explicit_packages=package_paths,
    )

    assert status["closure_input_package_inspection_ready"] is False
    assert status["closure_input_repository_profiles_ready"] is False
    assert status["closure_input_lock_ready"] is False
    assert any(
        reason.startswith("closure_input_package_inspection_failed:")
        for reason in status["failure_reasons"]
    )


@pytest.mark.quick
def test_complete_result_package_rejects_mixed_code_version_even_when_lock_is_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """打包器必须重验 common_code_version, 不能只相信锁文件自身摘要."""

    configure_paper_run(monkeypatch, tmp_path, "pilot_paper")
    create_required_outputs(tmp_path, "pilot_paper")
    package_paths = create_closure_input_lock(tmp_path, "pilot_paper")
    lock_dir = tmp_path / "outputs" / "paper_result_closure" / "pilot_paper"
    lock_path = lock_dir / "closure_input_lock.json"
    manifest_path = lock_dir / "input_lock_manifest.local.json"
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_payload["closure_input_packages"][0]["code_version"] = "b" * 40
    digest_payload = dict(lock_payload)
    digest_payload.pop("closure_input_lock_digest", None)
    lock_payload["closure_input_lock_digest"] = build_stable_digest(digest_payload)
    write_json(lock_path, lock_payload)
    lock_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock_manifest["metadata"]["closure_input_packages"] = lock_payload[
        "closure_input_packages"
    ]
    lock_manifest["metadata"]["closure_input_lock_digest"] = lock_payload[
        "closure_input_lock_digest"
    ]
    lock_manifest["config_digest"] = build_stable_digest(
        {
            "paper_run_name": "pilot_paper",
            "target_fpr": 0.01,
            "common_code_version": "a" * 40,
            "closure_input_packages": lock_payload["closure_input_packages"],
        }
    )
    write_json(manifest_path, lock_manifest)

    with pytest.raises(RuntimeError, match="closure_input_lock_ready=False"):
        write_pilot_paper_complete_result_package_outputs(
            root=tmp_path,
            drive_output_dir="",
            package_paths=package_paths,
            materialize_packages=False,
        )


@pytest.mark.quick
def test_complete_result_package_rejects_gate_source_changed_after_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """门禁通过后任一实际输入文件发生字节变化都不得生成 ZIP 或 Drive 副本."""

    configure_paper_run(monkeypatch, tmp_path, "pilot_paper")
    create_required_outputs(tmp_path, "pilot_paper")
    package_paths = create_closure_input_lock(tmp_path, "pilot_paper")
    common_summary = (
        tmp_path
        / "outputs"
        / "pilot_paper_fixed_fpr_common_protocol"
        / "pilot_paper"
        / "pilot_paper_common_protocol_summary.json"
    )
    common_summary.write_text(
        common_summary.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    drive_dir = tmp_path / "drive" / "complete"

    with pytest.raises(RuntimeError, match="result_closure_ready=False"):
        write_pilot_paper_complete_result_package_outputs(
            root=tmp_path,
            drive_output_dir=str(drive_dir),
            package_paths=package_paths,
            materialize_packages=False,
        )

    archive_path = (
        tmp_path
        / "outputs"
        / "pilot_paper_complete_result_package"
        / "pilot_paper_complete_result_package.zip"
    )
    assert not archive_path.exists()
    assert not drive_dir.exists()


@pytest.mark.quick
def test_complete_result_package_rejects_tampered_gate_report_and_code_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """报告摘要和 gate 代码版本任一失配都必须在归档前阻断."""

    configure_paper_run(monkeypatch, tmp_path, "probe_paper")
    create_required_outputs(tmp_path, "probe_paper")
    package_paths = create_closure_input_lock(tmp_path, "probe_paper")
    gate_dir = tmp_path / "outputs" / "result_closure_gate" / "probe_paper"
    report_path = gate_dir / "result_closure_gate_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    original_report = dict(report)
    report["unbound_change"] = True
    write_json(report_path, report)

    with pytest.raises(RuntimeError, match="result_closure_ready=False"):
        write_pilot_paper_complete_result_package_outputs(
            root=tmp_path,
            drive_output_dir="",
            package_paths=package_paths,
            materialize_packages=False,
        )

    write_json(report_path, original_report)
    gate_manifest_path = gate_dir / "manifest.local.json"
    gate_manifest = json.loads(gate_manifest_path.read_text(encoding="utf-8"))
    gate_manifest["code_version"] = "b" * 40
    write_json(gate_manifest_path, gate_manifest)
    with pytest.raises(RuntimeError, match="result_closure_ready=False"):
        write_pilot_paper_complete_result_package_outputs(
            root=tmp_path,
            drive_output_dir="",
            package_paths=package_paths,
            materialize_packages=False,
        )


@pytest.mark.quick
def test_complete_result_package_rejects_dirty_current_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """打包当下仓库必须仍是输入锁绑定的同一 clean Git 提交."""

    configure_paper_run(monkeypatch, tmp_path, "probe_paper")
    create_required_outputs(tmp_path, "probe_paper")
    package_paths = create_closure_input_lock(tmp_path, "probe_paper")
    monkeypatch.setattr(
        "scripts.write_pilot_paper_complete_result_package.resolve_code_version",
        lambda _root_path: "abc1234-dirty",
    )

    with pytest.raises(RuntimeError, match="result_closure_ready=False"):
        write_pilot_paper_complete_result_package_outputs(
            root=tmp_path,
            drive_output_dir="",
            package_paths=package_paths,
            materialize_packages=False,
        )


@pytest.mark.quick
def test_complete_result_package_rejects_missing_required_evidence_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任一关键证据目录缺失都必须阻断归档。"""

    configure_paper_run(monkeypatch, tmp_path, "pilot_paper")
    create_required_outputs(
        tmp_path,
        "pilot_paper",
        omitted_directory_name="fixed_fpr_threshold_audit",
    )
    package_paths = create_closure_input_lock(tmp_path, "pilot_paper")

    with pytest.raises(RuntimeError, match="fixed_fpr_threshold_audit"):
        write_pilot_paper_complete_result_package_outputs(
            root=tmp_path,
            drive_output_dir="",
            package_paths=package_paths,
        )

    assert not (
        tmp_path
        / "outputs"
        / "pilot_paper_complete_result_package"
        / "pilot_paper_complete_result_package.zip"
    ).exists()


@pytest.mark.quick
def test_complete_result_package_rejects_missing_dependency_hash_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """六套完整依赖锁缺失时不得生成任何论文完整结果归档."""

    paper_run_name = "probe_paper"
    configure_paper_run(monkeypatch, tmp_path, paper_run_name)
    create_required_outputs(tmp_path, paper_run_name)
    package_paths = create_closure_input_lock(tmp_path, paper_run_name)
    monkeypatch.setattr(
        "scripts.write_pilot_paper_complete_result_package.build_dependency_lock_status",
        build_dependency_lock_status,
    )

    with pytest.raises(RuntimeError, match="dependency_hash_locks_ready=False"):
        write_pilot_paper_complete_result_package_outputs(
            root=tmp_path,
            drive_output_dir="",
            package_paths=package_paths,
        )

    summary_path = (
        tmp_path
        / "outputs"
        / "pilot_paper_complete_result_package"
        / "probe_paper_complete_package_readiness_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["dependency_hash_locks_ready"] is False
    assert summary["dependency_hash_lock_count"] == 0
    assert summary["paper_run_complete_result_package_ready"] is False
    assert not (
        tmp_path
        / "outputs"
        / "pilot_paper_complete_result_package"
        / "probe_paper_complete_result_package.zip"
    ).exists()


@pytest.mark.quick
def test_complete_result_package_uses_only_explicit_package_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """打包器只消费显式 zip, 且跳过物化时不得读取相邻目录中的 zip。"""

    configure_paper_run(monkeypatch, tmp_path, "pilot_paper")
    create_required_outputs(tmp_path, "pilot_paper")
    package_paths = create_closure_input_lock(tmp_path, "pilot_paper")
    ignored_package = tmp_path / "drive" / "nested" / "ignored.zip"
    ignored_package.parent.mkdir()
    with ZipFile(ignored_package, "w") as archive:
        archive.writestr("outputs/ignored_result/result.json", '{"ignored": true}\n')

    receipt = write_pilot_paper_complete_result_package_outputs(
        root=tmp_path,
        drive_output_dir="",
        package_paths=package_paths,
        materialize_packages=False,
    )

    materialization = receipt["metadata"]["materialization_report"]
    assert not (tmp_path / "outputs" / "ignored_result" / "result.json").exists()
    assert materialization["materialization_skipped"] is True
    assert materialization["input_package_count"] == len(CLOSURE_PACKAGE_FAMILY_SPECS)
    assert materialization["input_package_paths"] == [path.as_posix() for path in package_paths]
    assert "--package-search-root" not in build_parser().format_help()


@pytest.mark.quick
def test_complete_result_package_required_directories_cover_evidence_closure_chain() -> None:
    """关键科学结果、审计与投稿门禁目录必须全部进入动态目录集合。"""

    required_names = set(REQUIRED_OUTPUT_DIR_NAMES)
    assert {
        "attack_matrix",
        "fixed_fpr_threshold_audit",
        "primary_baseline_method_faithful_adapter_protocol",
        "primary_baseline_evidence",
        "official_reference_fidelity_evidence",
        "paired_superiority_analysis",
        "paper_artifact_evidence_audit",
        "submission_readiness",
        "evidence_closure_entry_review",
        "paper_result_closure",
        "result_closure_gate",
    } <= required_names
    assert "scripts/write_external_baseline_comparison_outputs.py" in PACKAGE_EXTRA_PATHS
    assert "scripts/write_official_reference_fidelity_evidence_outputs.py" in PACKAGE_EXTRA_PATHS
    assert "scripts/write_paired_superiority_outputs.py" in PACKAGE_EXTRA_PATHS
    assert "scripts/write_paper_artifact_evidence_audit_outputs.py" in PACKAGE_EXTRA_PATHS
    assert "scripts/write_result_closure_gate_outputs.py" in PACKAGE_EXTRA_PATHS
    assert "paper_experiments/analysis/result_closure_gate.py" in PACKAGE_EXTRA_PATHS
    assert "scripts/write_submission_readiness_outputs.py" in PACKAGE_EXTRA_PATHS
    assert "scripts/write_evidence_closure_entry_review_outputs.py" in PACKAGE_EXTRA_PATHS
    assert "configs/dependency_profiles/workflow_orchestrator_lock.txt" in PACKAGE_EXTRA_PATHS
    assert "experiments/runtime/dependency_preparation.py" in PACKAGE_EXTRA_PATHS
    assert "experiments/runtime/isolated_dependency_environment.py" in PACKAGE_EXTRA_PATHS
    assert "experiments/runtime/isolated_scientific_execution.py" in PACKAGE_EXTRA_PATHS
    assert "experiments/runtime/scientific_execution_binding.py" in PACKAGE_EXTRA_PATHS
    assert (
        "experiments/runtime/semantic_watermark_scientific_session.py"
        in PACKAGE_EXTRA_PATHS
    )
    assert "experiments/runtime/repository_environment.py" in PACKAGE_EXTRA_PATHS
    assert "experiments/runners/image_only_dataset_workload.py" in PACKAGE_EXTRA_PATHS
    assert "experiments/ablations/mechanism_ablation_workload.py" in PACKAGE_EXTRA_PATHS
    assert "scripts/paper_result_closure.py" in PACKAGE_EXTRA_PATHS
    assert "scripts/semantic_watermark_scientific_workflow.py" in PACKAGE_EXTRA_PATHS
    assert "scripts/run_semantic_watermark_scientific_session.py" in PACKAGE_EXTRA_PATHS
    assert (
        "paper_experiments/runners/isolated_scientific_workflow.py"
        in PACKAGE_EXTRA_PATHS
    )
    assert "paper_experiments/baselines/command_plan_builder.py" in PACKAGE_EXTRA_PATHS
    assert "paper_experiments/baselines/command_plan_execution.py" in PACKAGE_EXTRA_PATHS
    assert "paper_experiments/baselines/evidence_validation_cli.py" in PACKAGE_EXTRA_PATHS
    assert (
        "paper_experiments/runners/official_reference_dependency_environment.py"
        in PACKAGE_EXTRA_PATHS
    )
    assert "paper_experiments/runners/t2smark_formal_reproduction.py" in PACKAGE_EXTRA_PATHS
    assert (
        "paper_experiments/runners/external_baseline_method_faithful.py"
        in PACKAGE_EXTRA_PATHS
    )
    assert not any(path.startswith("paper_workflow/") for path in PACKAGE_EXTRA_PATHS)
    assert build_required_output_dirs("probe_paper") != build_required_output_dirs("pilot_paper")
