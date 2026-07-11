from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from zipfile import ZipFile

import pytest

from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
    LOCK_FILENAME,
    LOCK_MANIFEST_FILENAME,
    LOCK_OUTPUT_ROOT,
    ClosurePackageFamilySpec,
    ClosurePackageSelectionError,
    JsonFieldSource,
    build_closure_input_selection_report,
    inspect_closure_package,
    select_and_lock_closure_input_packages,
)
from tests.helpers.formal_execution_lock import build_test_formal_execution_lock


pytestmark = pytest.mark.quick


PAPER_RUN_NAME = "probe_paper"
TARGET_FPR = 0.1
CODE_VERSION = "a" * 40
GENERATED_AT = "2026-07-11T08:00:00+00:00"


def _render(template: str, spec: ClosurePackageFamilySpec, paper_run_name: str) -> str:
    return template.format(
        paper_run=paper_run_name,
        baseline=spec.baseline_id or "",
    )


def _assign(payload: dict[str, Any], field_path: tuple[str, ...], value: Any) -> None:
    current = payload
    for field_name in field_path[:-1]:
        nested = current.setdefault(field_name, {})
        assert isinstance(nested, dict)
        current = nested
    current[field_path[-1]] = value


def _assign_source(
    documents: dict[str, dict[str, Any]],
    spec: ClosurePackageFamilySpec,
    source: JsonFieldSource,
    value: Any,
    *,
    paper_run_name: str,
) -> None:
    member_name = _render(source.member_template, spec, paper_run_name)
    payload = documents.setdefault(member_name, {})
    _assign(payload, source.field_path, value)


def _assign_execution_locks(
    documents: dict[str, dict[str, Any]],
    spec: ClosurePackageFamilySpec,
    commit: str,
    *,
    paper_run_name: str,
) -> None:
    """让测试 manifest 的运行锁和打包锁共同绑定指定 commit."""

    manifest_name = _render(
        spec.manifest_member_template,
        spec,
        paper_run_name,
    )
    manifest = documents.setdefault(manifest_name, {})
    manifest["formal_execution_run_lock"] = build_test_formal_execution_lock(commit)
    manifest["formal_execution_package_lock"] = build_test_formal_execution_lock(commit)


def _valid_member_payloads(
    spec: ClosurePackageFamilySpec,
    *,
    paper_run_name: str,
    target_fpr: float,
    generated_at: str,
    mutate: Callable[[dict[str, dict[str, Any]], ClosurePackageFamilySpec], None] | None = None,
) -> dict[str, bytes]:
    required_members = {
        _render(template, spec, paper_run_name)
        for template in spec.required_member_templates
    }
    documents: dict[str, dict[str, Any]] = {
        member_name: {}
        for member_name in required_members
        if member_name.endswith(".json")
    }
    manifest_member = _render(spec.manifest_member_template, spec, paper_run_name)
    manifest = documents.setdefault(manifest_member, {})
    manifest["artifact_id"] = _render(
        spec.manifest_artifact_id_template,
        spec,
        paper_run_name,
    )
    manifest["artifact_type"] = "local_manifest"
    _assign_execution_locks(
        documents,
        spec,
        CODE_VERSION,
        paper_run_name=paper_run_name,
    )
    for source in spec.paper_run_sources:
        _assign_source(
            documents,
            spec,
            source,
            paper_run_name,
            paper_run_name=paper_run_name,
        )
    for source in spec.target_fpr_sources:
        _assign_source(
            documents,
            spec,
            source,
            target_fpr,
            paper_run_name=paper_run_name,
        )
    for source in spec.baseline_sources:
        _assign_source(
            documents,
            spec,
            source,
            spec.baseline_id,
            paper_run_name=paper_run_name,
        )
    for source in spec.code_version_sources:
        _assign_source(
            documents,
            spec,
            source,
            CODE_VERSION,
            paper_run_name=paper_run_name,
        )
    _assign_source(
        documents,
        spec,
        spec.generated_at_source,
        generated_at,
        paper_run_name=paper_run_name,
    )
    for requirement in spec.value_requirements:
        _assign_source(
            documents,
            spec,
            requirement.source,
            requirement.expected_value,
            paper_run_name=paper_run_name,
        )
    if mutate is not None:
        mutate(documents, spec)

    member_payloads: dict[str, bytes] = {}
    baseline_rows_members = {
        _render(source.member_template, spec, paper_run_name)
        for source in spec.baseline_rows_sources
    }
    for member_name in required_members:
        if member_name in baseline_rows_members:
            row = {"baseline_id": spec.baseline_id}
            member_payloads[member_name] = (
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8")
        elif member_name in documents:
            member_payloads[member_name] = (
                json.dumps(documents[member_name], ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8")
        elif member_name.endswith(".jsonl"):
            member_payloads[member_name] = b'{"record_id":"fixture"}\n'
        elif member_name.endswith(".csv"):
            member_payloads[member_name] = b"metric_name,metric_value\nfixture,1\n"
        else:
            member_payloads[member_name] = b"fixture\n"
    return member_payloads


def _write_family_package(
    package_root: Path,
    spec: ClosurePackageFamilySpec,
    *,
    token: str,
    generated_at: str = GENERATED_AT,
    mutate: Callable[[dict[str, dict[str, Any]], ClosurePackageFamilySpec], None] | None = None,
    extra_members: dict[str, bytes] | None = None,
) -> Path:
    package_path = package_root / spec.filename_pattern.replace("*", token)
    package_path.parent.mkdir(parents=True, exist_ok=True)
    members = _valid_member_payloads(
        spec,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        generated_at=generated_at,
        mutate=mutate,
    )
    members.update(extra_members or {})
    with ZipFile(package_path, "w") as archive:
        for member_name, payload in sorted(members.items()):
            archive.writestr(member_name, payload)
    return package_path


def _write_all_family_packages(package_root: Path, *, token: str = "current") -> list[Path]:
    return [
        _write_family_package(package_root, spec, token=token)
        for spec in CLOSURE_PACKAGE_FAMILY_SPECS
    ]


def test_dry_run_selects_exact_ten_families_without_mixing_unrelated_archives(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "drive"
    expected_paths = _write_all_family_packages(package_root)
    with ZipFile(package_root / "probe_paper_complete_result_package_prior.zip", "w") as archive:
        archive.writestr("outputs/paper_result/summary.json", "{}")
    with ZipFile(package_root / "unrelated_evidence.zip", "w") as archive:
        archive.writestr("outputs/unrelated/value.json", "{}")
    matching_directory = package_root / CLOSURE_PACKAGE_FAMILY_SPECS[0].filename_pattern.replace(
        "*",
        "directory",
    )
    matching_directory.mkdir()

    report = build_closure_input_selection_report(
        package_root,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        root=tmp_path,
    )

    assert report["closure_input_selection_ready"] is True
    assert report["closure_input_lock_written"] is False
    assert report["closure_input_package_count"] == 10
    assert len(report["selected_package_paths"]) == 10
    assert set(report["selected_package_paths"]) == {
        path.resolve().as_posix() for path in expected_paths
    }
    assert not (tmp_path / LOCK_OUTPUT_ROOT / PAPER_RUN_NAME).exists()


def test_formal_selection_writes_run_scoped_lock_and_independent_manifest(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "drive"
    _write_all_family_packages(package_root)

    selected_paths = select_and_lock_closure_input_packages(
        package_root,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        root=tmp_path,
    )

    output_dir = tmp_path / LOCK_OUTPUT_ROOT / PAPER_RUN_NAME
    lock_path = output_dir / LOCK_FILENAME
    manifest_path = output_dir / LOCK_MANIFEST_FILENAME
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(selected_paths) == 10
    assert lock_payload["closure_input_package_count"] == 10
    assert lock_payload["paper_run_name"] == PAPER_RUN_NAME
    assert lock_payload["target_fpr"] == TARGET_FPR
    assert lock_payload["common_code_version"] == CODE_VERSION
    assert [row["package_path"] for row in lock_payload["closure_input_packages"]] == list(
        selected_paths
    )
    assert all(len(row["package_sha256"]) == 64 for row in lock_payload["closure_input_packages"])
    assert all(
        len(row["formal_execution_run_lock_digest"]) == 64
        and len(row["formal_execution_package_lock_digest"]) == 64
        for row in lock_payload["closure_input_packages"]
    )
    assert lock_payload["formal_execution_run_lock_digests"] == {
        row["package_family"]: row["formal_execution_run_lock_digest"]
        for row in lock_payload["closure_input_packages"]
    }
    assert lock_payload["formal_execution_package_lock_digests"] == {
        row["package_family"]: row["formal_execution_package_lock_digest"]
        for row in lock_payload["closure_input_packages"]
    }
    digest_payload = dict(lock_payload)
    stored_digest = digest_payload.pop("closure_input_lock_digest")
    canonical = json.dumps(
        digest_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert stored_digest == hashlib.sha256(canonical).hexdigest()
    assert manifest["artifact_id"] == f"{PAPER_RUN_NAME}_closure_input_lock_manifest"
    assert manifest["metadata"]["closure_input_lock_ready"] is True
    assert manifest["metadata"]["closure_input_package_count"] == 10
    assert manifest["metadata"]["closure_input_packages"] == lock_payload[
        "closure_input_packages"
    ]
    assert manifest["metadata"]["closure_input_lock_digest"] == stored_digest
    assert manifest["metadata"]["common_code_version"] == CODE_VERSION
    assert manifest["output_paths"] == [
        f"outputs/paper_result_closure/{PAPER_RUN_NAME}/{LOCK_FILENAME}",
        f"outputs/paper_result_closure/{PAPER_RUN_NAME}/{LOCK_MANIFEST_FILENAME}",
    ]


def test_multiple_candidates_use_governed_generated_at_instead_of_path_name(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "drive"
    for spec in CLOSURE_PACKAGE_FAMILY_SPECS[1:]:
        _write_family_package(package_root, spec, token="current")
    selected_spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]
    _write_family_package(
        package_root,
        selected_spec,
        token="zzz",
        generated_at="2026-07-10T08:00:00+00:00",
    )
    latest_path = _write_family_package(
        package_root,
        selected_spec,
        token="aaa",
        generated_at="2026-07-12T08:00:00+00:00",
    )

    report = build_closure_input_selection_report(
        package_root,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        root=tmp_path,
    )

    runtime_record = next(
        row
        for row in report["closure_input_packages"]
        if row["package_family"] == selected_spec.package_family
    )
    assert runtime_record["package_path"] == latest_path.resolve().as_posix()
    assert runtime_record["generated_at"] == "2026-07-12T08:00:00+00:00"


def test_internal_identity_rejects_wrong_run_fpr_baseline_and_ready_flag(
    tmp_path: Path,
) -> None:
    cases: list[
        tuple[
            ClosurePackageFamilySpec,
            Callable[[dict[str, dict[str, Any]], ClosurePackageFamilySpec], None],
        ]
    ] = []

    runtime_spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]

    def wrong_run(documents: dict[str, dict[str, Any]], spec: ClosurePackageFamilySpec) -> None:
        _assign_source(
            documents,
            spec,
            spec.paper_run_sources[0],
            "pilot_paper",
            paper_run_name=PAPER_RUN_NAME,
        )

    cases.append((runtime_spec, wrong_run))
    ablation_spec = CLOSURE_PACKAGE_FAMILY_SPECS[1]

    def wrong_fpr(documents: dict[str, dict[str, Any]], spec: ClosurePackageFamilySpec) -> None:
        _assign_source(
            documents,
            spec,
            spec.target_fpr_sources[0],
            0.01,
            paper_run_name=PAPER_RUN_NAME,
        )

    cases.append((ablation_spec, wrong_fpr))
    baseline_spec = CLOSURE_PACKAGE_FAMILY_SPECS[3]

    def wrong_baseline(
        documents: dict[str, dict[str, Any]],
        spec: ClosurePackageFamilySpec,
    ) -> None:
        _assign_source(
            documents,
            spec,
            spec.baseline_sources[0],
            "gaussian_shading",
            paper_run_name=PAPER_RUN_NAME,
        )

    cases.append((baseline_spec, wrong_baseline))
    official_spec = CLOSURE_PACKAGE_FAMILY_SPECS[6]

    def ready_false(documents: dict[str, dict[str, Any]], spec: ClosurePackageFamilySpec) -> None:
        requirement = next(
            requirement
            for requirement in spec.value_requirements
            if requirement.expected_value is True
        )
        _assign_source(
            documents,
            spec,
            requirement.source,
            False,
            paper_run_name=PAPER_RUN_NAME,
        )

    cases.append((official_spec, ready_false))

    for case_index, (spec, mutate) in enumerate(cases):
        package_path = _write_family_package(
            tmp_path / f"case_{case_index}",
            spec,
            token="invalid",
            mutate=mutate,
        )
        with pytest.raises(ClosurePackageSelectionError):
            inspect_closure_package(
                package_path,
                spec=spec,
                paper_run_name=PAPER_RUN_NAME,
                target_fpr=TARGET_FPR,
            )


@pytest.mark.parametrize(
    "missing_lock_field",
    ("formal_execution_run_lock", "formal_execution_package_lock"),
)
def test_package_rejects_missing_formal_execution_lock(
    tmp_path: Path,
    missing_lock_field: str,
) -> None:
    """每个闭合包 manifest 必须同时携带运行锁和打包锁."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]

    def mutate(
        documents: dict[str, dict[str, Any]],
        current_spec: ClosurePackageFamilySpec,
    ) -> None:
        manifest_name = _render(
            current_spec.manifest_member_template,
            current_spec,
            PAPER_RUN_NAME,
        )
        documents[manifest_name].pop(missing_lock_field)

    package_path = _write_family_package(
        tmp_path,
        spec,
        token=f"missing_{missing_lock_field}",
        mutate=mutate,
    )
    with pytest.raises(ClosurePackageSelectionError, match="缺少 formal_execution"):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_package_rejects_forged_formal_execution_lock_digest(tmp_path: Path) -> None:
    """任一执行锁摘要被篡改后都不得进入正式闭合."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]

    def mutate(
        documents: dict[str, dict[str, Any]],
        current_spec: ClosurePackageFamilySpec,
    ) -> None:
        manifest_name = _render(
            current_spec.manifest_member_template,
            current_spec,
            PAPER_RUN_NAME,
        )
        package_lock = documents[manifest_name]["formal_execution_package_lock"]
        assert isinstance(package_lock, dict)
        package_lock["formal_execution_lock_digest"] = "0" * 64

    package_path = _write_family_package(
        tmp_path,
        spec,
        token="forged_lock_digest",
        mutate=mutate,
    )
    with pytest.raises(ClosurePackageSelectionError, match="严格复验"):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_package_rejects_run_and_package_lock_commit_drift(tmp_path: Path) -> None:
    """运行锁与打包锁即使各自有效也必须绑定同一个 commit."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]

    def mutate(
        documents: dict[str, dict[str, Any]],
        current_spec: ClosurePackageFamilySpec,
    ) -> None:
        manifest_name = _render(
            current_spec.manifest_member_template,
            current_spec,
            PAPER_RUN_NAME,
        )
        documents[manifest_name]["formal_execution_package_lock"] = (
            build_test_formal_execution_lock("b" * 40)
        )

    package_path = _write_family_package(
        tmp_path,
        spec,
        token="lock_commit_drift",
        mutate=mutate,
    )
    with pytest.raises(ClosurePackageSelectionError, match="运行锁与打包锁 commit 不一致"):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_package_rejects_execution_lock_and_code_version_drift(tmp_path: Path) -> None:
    """两个执行锁一致时仍必须与全部 code_version 来源完全一致."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]

    def mutate(
        documents: dict[str, dict[str, Any]],
        current_spec: ClosurePackageFamilySpec,
    ) -> None:
        manifest_name = _render(
            current_spec.manifest_member_template,
            current_spec,
            PAPER_RUN_NAME,
        )
        drifted_lock = build_test_formal_execution_lock("b" * 40)
        documents[manifest_name]["formal_execution_run_lock"] = drifted_lock
        documents[manifest_name]["formal_execution_package_lock"] = drifted_lock

    package_path = _write_family_package(
        tmp_path,
        spec,
        token="lock_code_version_drift",
        mutate=mutate,
    )
    with pytest.raises(ClosurePackageSelectionError, match="code_version 来源不一致"):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


@pytest.mark.parametrize(
    "invalid_code_version",
    (
        f"{CODE_VERSION}-dirty",
        "git_version_unavailable",
        "main",
        "abc1234",
        "A" * 40,
    ),
)
def test_package_rejects_non_clean_git_code_version(
    tmp_path: Path,
    invalid_code_version: str,
) -> None:
    """单包必须拒绝 dirty, 降级值, 分支名, 短 SHA 与大写 SHA."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]

    def mutate(documents: dict[str, dict[str, Any]], current_spec: ClosurePackageFamilySpec) -> None:
        for source in current_spec.code_version_sources:
            _assign_source(
                documents,
                current_spec,
                source,
                invalid_code_version,
                paper_run_name=PAPER_RUN_NAME,
            )

    package_path = _write_family_package(tmp_path, spec, token="invalid_code", mutate=mutate)
    with pytest.raises(ClosurePackageSelectionError, match="code_version"):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_selection_requires_one_common_code_version_across_all_families(tmp_path: Path) -> None:
    """即使各包内部自洽, 10个 family 的 clean Git 提交也必须完全相同."""

    package_root = tmp_path / "drive"
    for spec in CLOSURE_PACKAGE_FAMILY_SPECS[1:]:
        _write_family_package(package_root, spec, token="current")
    selected_spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]

    def mutate(documents: dict[str, dict[str, Any]], current_spec: ClosurePackageFamilySpec) -> None:
        for source in current_spec.code_version_sources:
            _assign_source(
                documents,
                current_spec,
                source,
                "b" * 40,
                paper_run_name=PAPER_RUN_NAME,
            )
        _assign_execution_locks(
            documents,
            current_spec,
            "b" * 40,
            paper_run_name=PAPER_RUN_NAME,
        )

    _write_family_package(package_root, selected_spec, token="different", mutate=mutate)
    with pytest.raises(ClosurePackageSelectionError, match="共享同一"):
        build_closure_input_selection_report(
            package_root,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            root=tmp_path,
        )


def test_full_lowercase_clean_commit_is_accepted(tmp_path: Path) -> None:
    """精确40位小写 clean 提交可以进入闭合输入候选."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]
    full_commit = "b" * 40

    def mutate(documents: dict[str, dict[str, Any]], current_spec: ClosurePackageFamilySpec) -> None:
        for source in current_spec.code_version_sources:
            _assign_source(
                documents,
                current_spec,
                source,
                full_commit,
                paper_run_name=PAPER_RUN_NAME,
            )
        _assign_execution_locks(
            documents,
            current_spec,
            full_commit,
            paper_run_name=PAPER_RUN_NAME,
        )

    package_path = _write_family_package(tmp_path, spec, token="full_commit", mutate=mutate)
    candidate = inspect_closure_package(
        package_path,
        spec=spec,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
    )
    assert candidate.code_version == full_commit
    expected_lock_digest = build_test_formal_execution_lock(full_commit)[
        "formal_execution_lock_digest"
    ]
    assert candidate.formal_execution_run_lock_digest == expected_lock_digest
    assert candidate.formal_execution_package_lock_digest == expected_lock_digest


def test_extended_clean_short_commit_is_rejected(tmp_path: Path) -> None:
    """任何不足40位的 clean 提交前缀都不得进入正式闭合."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]
    extended_short_commit = "abc12345"

    def mutate(documents: dict[str, dict[str, Any]], current_spec: ClosurePackageFamilySpec) -> None:
        for source in current_spec.code_version_sources:
            _assign_source(
                documents,
                current_spec,
                source,
                extended_short_commit,
                paper_run_name=PAPER_RUN_NAME,
            )

    package_path = _write_family_package(tmp_path, spec, token="extended_short", mutate=mutate)
    with pytest.raises(ClosurePackageSelectionError, match="code_version"):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_matching_filename_cannot_replace_internal_family_identity(tmp_path: Path) -> None:
    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]

    def wrong_artifact(
        documents: dict[str, dict[str, Any]],
        current_spec: ClosurePackageFamilySpec,
    ) -> None:
        manifest_name = _render(
            current_spec.manifest_member_template,
            current_spec,
            PAPER_RUN_NAME,
        )
        documents[manifest_name]["artifact_id"] = "unrelated_artifact"

    package_path = _write_family_package(
        tmp_path,
        spec,
        token="masquerading",
        mutate=wrong_artifact,
    )
    with pytest.raises(ClosurePackageSelectionError):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_rejects_directory_empty_damaged_traversal_and_non_output_members(
    tmp_path: Path,
) -> None:
    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]
    candidate_paths: list[Path] = []

    directory_path = tmp_path / "directory.zip"
    directory_path.mkdir()
    candidate_paths.append(directory_path)

    empty_file_path = tmp_path / "empty_file.zip"
    empty_file_path.write_bytes(b"")
    candidate_paths.append(empty_file_path)

    empty_zip_path = tmp_path / "empty_archive.zip"
    with ZipFile(empty_zip_path, "w"):
        pass
    candidate_paths.append(empty_zip_path)

    damaged_path = tmp_path / "damaged.zip"
    damaged_path.write_bytes(b"not-a-zip")
    candidate_paths.append(damaged_path)

    traversal_path = tmp_path / "traversal.zip"
    with ZipFile(traversal_path, "w") as archive:
        archive.writestr("../escape.json", "{}")
    candidate_paths.append(traversal_path)

    non_output_path = tmp_path / "non_output.zip"
    with ZipFile(non_output_path, "w") as archive:
        archive.writestr("README.md", "not allowed")
    candidate_paths.append(non_output_path)

    for candidate_path in candidate_paths:
        with pytest.raises(ClosurePackageSelectionError):
            inspect_closure_package(
                candidate_path,
                spec=spec,
                paper_run_name=PAPER_RUN_NAME,
                target_fpr=TARGET_FPR,
            )


def test_package_input_member_digests_bind_declared_bytes_and_exact_member_set(
    tmp_path: Path,
) -> None:
    """存在逐成员摘要声明时, 内容变化或额外成员都必须被拒绝."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[3]
    assert spec.package_input_manifest_template is not None
    members = _valid_member_payloads(
        spec,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        generated_at=GENERATED_AT,
    )
    package_input_member = _render(spec.package_input_manifest_template, spec, PAPER_RUN_NAME)
    declared_member = next(
        name
        for name in sorted(members)
        if name != package_input_member and name.endswith("_summary.json")
    )
    package_input = json.loads(members[package_input_member].decode("utf-8"))
    package_input["entry_paths"] = [declared_member]
    package_input["entry_sha256"] = {
        declared_member: hashlib.sha256(members[declared_member]).hexdigest()
    }
    members[package_input_member] = (
        json.dumps(package_input, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")

    valid_path = tmp_path / "declared_valid.zip"
    with ZipFile(valid_path, "w") as archive:
        for member_name, payload in sorted(members.items()):
            archive.writestr(member_name, payload)
    inspect_closure_package(
        valid_path,
        spec=spec,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
    )

    changed_path = tmp_path / "declared_changed.zip"
    with ZipFile(changed_path, "w") as archive:
        for member_name, payload in sorted(members.items()):
            archive.writestr(
                member_name,
                b"changed\n" if member_name == declared_member else payload,
            )
    with pytest.raises(ClosurePackageSelectionError, match="成员摘要不匹配"):
        inspect_closure_package(
            changed_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )

    extra_path = tmp_path / "declared_extra.zip"
    allowed_extra_member = _render(
        spec.allowed_output_prefix_templates[0], spec, PAPER_RUN_NAME
    ) + "undeclared_extra.json"
    with ZipFile(extra_path, "w") as archive:
        for member_name, payload in sorted(members.items()):
            archive.writestr(member_name, payload)
        archive.writestr(allowed_extra_member, b"{}\n")
    with pytest.raises(ClosurePackageSelectionError, match="精确成员集合不一致"):
        inspect_closure_package(
            extra_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )
