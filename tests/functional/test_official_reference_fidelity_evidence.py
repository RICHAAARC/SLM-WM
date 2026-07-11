"""验证三个官方参考方法忠实度的独立证据 builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from paper_experiments.baselines.official_reference_fidelity_evidence import (
    ARCHIVE_GOVERNANCE_SCOPE,
    OFFICIAL_REFERENCE_BASELINE_IDS,
    OfficialReferenceFidelityEvidenceError,
    build_official_reference_fidelity_summary,
)
from scripts.write_official_reference_fidelity_evidence_outputs import (
    write_official_reference_fidelity_evidence_outputs,
)


CODE_VERSION = "8" * 40
PAPER_RUN_NAME = "probe_paper"
TARGET_FPR = 0.1


def _write_json(path: Path, payload: object) -> None:
    """写出测试使用的稳定 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    """计算测试文件 SHA-256."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path, root: Path) -> str:
    """返回测试仓库相对路径."""

    return path.relative_to(root).as_posix()


def _write_official_reference_family(root: Path, baseline_id: str) -> Path:
    """构造符合当前 producer 语义的最小 official-reference family."""

    output_dir = (
        root
        / "outputs"
        / f"{baseline_id}_official_reference"
        / PAPER_RUN_NAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{baseline_id}_official_reference_summary.json"
    run_manifest_path = output_dir / "manifest.local.json"
    records_path = output_dir / f"{baseline_id}_official_reference_records.jsonl"
    validation_path = output_dir / f"{baseline_id}_official_reference_validation_report.json"
    package_input_path = output_dir / f"{baseline_id}_official_reference_package_input_manifest.json"
    archive_summary_path = output_dir / f"{baseline_id}_official_reference_archive_summary.json"
    archive_manifest_path = output_dir / f"{baseline_id}_official_reference_archive_manifest.local.json"
    environment_path = output_dir / f"{baseline_id}_official_reference_environment_report.json"
    schema_path = output_dir / f"{baseline_id}_official_reference_schema.json"

    record = {
        "reference_record_id": f"{baseline_id}_reference_record",
        "reference_record_digest": f"{baseline_id}_digest",
        "baseline_id": baseline_id,
        "reference_protocol_name": f"{baseline_id}_official_reference_protocol",
        "supplemental_table_role": "supplemental_method_fidelity_reference",
        "main_table_eligible": False,
        "supports_paper_claim": False,
    }
    records_path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = {
        "reference_protocol_name": record["reference_protocol_name"],
        "input_record_count": 1,
        "accepted_reference_record_count": 1,
        "rejected_reference_record_count": 0,
        "reference_issue_count": 0,
        "reference_import_ready": True,
        "accepted_records": [record],
        "issues": [],
        "main_table_eligible": False,
        "supports_paper_claim": False,
    }
    _write_json(validation_path, validation)
    _write_json(environment_path, {"runtime_ready": True})
    _write_json(schema_path, {"baseline_id": baseline_id})
    summary = {
        "generated_at": "2026-07-11T00:00:00+00:00",
        "baseline_id": baseline_id,
        "paper_claim_scale": PAPER_RUN_NAME,
        "target_fpr": TARGET_FPR,
        "run_decision": "pass",
        f"{baseline_id}_official_reference_ready": True,
        "reference_import_ready": True,
        "governed_reference_record_count": 1,
        "main_table_eligible": False,
        "supports_paper_claim": False,
        "summary_path": _relative(summary_path, root),
        "reference_records_path": _relative(records_path, root),
        "reference_validation_path": _relative(validation_path, root),
        "metadata": {"validation": validation},
    }
    _write_json(summary_path, summary)
    run_outputs = [
        _relative(summary_path, root),
        _relative(environment_path, root),
        _relative(schema_path, root),
        _relative(records_path, root),
        _relative(validation_path, root),
        _relative(run_manifest_path, root),
    ]
    _write_json(
        run_manifest_path,
        {
            "artifact_id": f"{baseline_id}_official_reference_manifest",
            "artifact_type": "local_manifest",
            "input_paths": [f"external_baseline/primary/{baseline_id}/source"],
            "output_paths": run_outputs,
            "config_digest": f"{baseline_id}_config",
            "code_version": CODE_VERSION,
            "rebuild_command": f"调用 {baseline_id} official reference runner",
            "metadata": {
                "run_decision": "pass",
                f"{baseline_id}_official_reference_ready": True,
                "main_table_eligible": False,
                "supports_paper_claim": False,
            },
        },
    )

    dynamic_files = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.suffix != ".zip"
    )
    entry_paths = [_relative(path, root) for path in dynamic_files]
    entry_sha256 = {_relative(path, root): _sha256(path) for path in dynamic_files}
    archive_name = f"external_baseline_official_reference_package_{baseline_id}_test.zip"
    archive_path = output_dir / archive_name
    _write_json(
        package_input_path,
        {
            "generated_at": "2026-07-11T00:01:00+00:00",
            "paper_run_name": PAPER_RUN_NAME,
            "target_fpr": TARGET_FPR,
            "baseline_id": baseline_id,
            "entry_paths": entry_paths,
            "entry_sha256": entry_sha256,
            "entry_count": len(entry_paths) + 3,
            "embedded_digest_scope": ARCHIVE_GOVERNANCE_SCOPE,
        },
    )
    _write_json(
        archive_summary_path,
        {
            "archive_path": _relative(archive_path, root),
            "archive_digest": "",
            "archive_entry_count": len(entry_paths) + 3,
            "drive_archive_path": f"/content/drive/{archive_name}",
            "drive_archive_digest": "",
            "metadata": {
                "drive_output_dir": "/content/drive",
                "embedded_digest_scope": ARCHIVE_GOVERNANCE_SCOPE,
                "generated_at": "2026-07-11T00:01:01+00:00",
            },
        },
    )
    _write_json(
        archive_manifest_path,
        {
            "artifact_id": f"{baseline_id}_official_reference_archive_manifest",
            "artifact_type": "local_manifest",
            "input_paths": [*entry_paths, _relative(package_input_path, root)],
            "output_paths": [
                _relative(archive_path, root),
                _relative(archive_summary_path, root),
                _relative(archive_manifest_path, root),
            ],
            "config_digest": f"{baseline_id}_archive_config",
            "code_version": CODE_VERSION,
            "rebuild_command": f"调用 {baseline_id} official reference runner",
            "metadata": {
                "embedded_digest_scope": ARCHIVE_GOVERNANCE_SCOPE,
                "generated_at": "2026-07-11T00:01:02+00:00",
                "paper_run_name": PAPER_RUN_NAME,
                "target_fpr": TARGET_FPR,
                "baseline_id": baseline_id,
                "main_table_eligible": False,
            },
        },
    )
    return output_dir


def _write_all_sources(root: Path) -> dict[str, Path]:
    """写出精确三个 official-reference family."""

    return {
        baseline_id: _write_official_reference_family(root, baseline_id)
        for baseline_id in OFFICIAL_REFERENCE_BASELINE_IDS
    }


@pytest.mark.quick
def test_official_reference_fidelity_builder_writes_exact_supplemental_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """精确三个已闭合输入应生成独立补充证据, 且不得声明主表优势."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", PAPER_RUN_NAME)
    source_dirs = _write_all_sources(tmp_path)

    manifest = write_official_reference_fidelity_evidence_outputs(
        root=tmp_path,
        tree_ring_output_dir=source_dirs["tree_ring"],
        gaussian_shading_output_dir=source_dirs["gaussian_shading"],
        shallow_diffuse_output_dir=source_dirs["shallow_diffuse"],
        repository_code_version=CODE_VERSION,
        require_pass=True,
    )

    output_dir = (
        tmp_path / "outputs" / "official_reference_fidelity_evidence" / PAPER_RUN_NAME
    )
    records = [
        json.loads(line)
        for line in (
            output_dir / "official_reference_fidelity_evidence_records.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads(
        (output_dir / "official_reference_fidelity_evidence_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert [record["baseline_id"] for record in records] == list(
        OFFICIAL_REFERENCE_BASELINE_IDS
    )
    assert all(record["official_reference_fidelity_evidence_ready"] for record in records)
    assert all(record["supports_main_table_superiority_claim"] is False for record in records)
    assert all(len(record["official_reference_source_artifact_digests"]) == 7 for record in records)
    assert summary["official_reference_exact_set_ready"] is True
    assert summary["official_reference_fidelity_ready_count"] == 3
    assert summary["common_code_version"] == CODE_VERSION
    assert summary["official_reference_fidelity_evidence_ready"] is True
    assert summary["supports_main_table_superiority_claim"] is False
    assert manifest["artifact_id"] == "official_reference_fidelity_evidence_manifest"
    assert manifest["code_version"] == CODE_VERSION
    assert len(manifest["input_paths"]) == 21


@pytest.mark.quick
def test_official_reference_fidelity_summary_rejects_duplicate_identity() -> None:
    """重复方法身份不得满足精确三方法集合."""

    records = [
        {
            "baseline_id": baseline_id,
            "code_version": CODE_VERSION,
            "official_reference_fidelity_evidence_ready": True,
            "supplemental_method_fidelity_evidence_ready": True,
            "source_code_version_consistent_ready": True,
            "supports_main_table_superiority_claim": False,
            "main_table_eligible": False,
        }
        for baseline_id in ("tree_ring", "tree_ring", "shallow_diffuse")
    ]

    summary = build_official_reference_fidelity_summary(records)

    assert summary["official_reference_exact_set_ready"] is False
    assert summary["duplicate_official_reference_baseline_ids"] == ["tree_ring"]
    assert summary["missing_official_reference_baseline_ids"] == ["gaussian_shading"]
    assert summary["official_reference_fidelity_evidence_ready"] is False


@pytest.mark.quick
def test_official_reference_fidelity_builder_rejects_tampered_declared_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """package input 声明后被修改的动态文件必须阻断证据生成."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", PAPER_RUN_NAME)
    source_dirs = _write_all_sources(tmp_path)
    tampered_path = (
        source_dirs["tree_ring"]
        / "tree_ring_official_reference_environment_report.json"
    )
    _write_json(tampered_path, {"runtime_ready": False})

    with pytest.raises(
        OfficialReferenceFidelityEvidenceError,
        match="package entry 摘要不匹配",
    ):
        write_official_reference_fidelity_evidence_outputs(
            root=tmp_path,
            repository_code_version=CODE_VERSION,
        )


@pytest.mark.quick
def test_official_reference_fidelity_builder_rejects_validation_issue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validation 中任一拒绝或 issue 都必须阻断正式补充证据."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", PAPER_RUN_NAME)
    source_dirs = _write_all_sources(tmp_path)
    validation_path = (
        source_dirs["gaussian_shading"]
        / "gaussian_shading_official_reference_validation_report.json"
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["reference_import_ready"] = False
    validation["accepted_reference_record_count"] = 0
    validation["rejected_reference_record_count"] = 1
    validation["reference_issue_count"] = 1
    validation["accepted_records"] = []
    validation["issues"] = [
        {"row_index": 0, "field_name": "baseline_id", "reason": "identity_mismatch"}
    ]
    _write_json(validation_path, validation)

    with pytest.raises(
        OfficialReferenceFidelityEvidenceError,
        match="validation 存在拒绝记录",
    ):
        write_official_reference_fidelity_evidence_outputs(
            root=tmp_path,
            repository_code_version=CODE_VERSION,
        )


@pytest.mark.quick
def test_official_reference_fidelity_builder_rejects_dirty_code_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """official-reference manifest 的 dirty 代码版本不得进入证据链."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", PAPER_RUN_NAME)
    source_dirs = _write_all_sources(tmp_path)
    run_manifest_path = source_dirs["shallow_diffuse"] / "manifest.local.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["code_version"] = f"{CODE_VERSION}-dirty"
    _write_json(run_manifest_path, run_manifest)

    with pytest.raises(
        OfficialReferenceFidelityEvidenceError,
        match="精确40位小写 clean Git 提交 SHA",
    ):
        write_official_reference_fidelity_evidence_outputs(
            root=tmp_path,
            repository_code_version=CODE_VERSION,
        )


@pytest.mark.quick
def test_official_reference_fidelity_builder_rejects_repository_version_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU 审计代码提交与 GPU 输入提交不同必须阻断 manifest 写出."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", PAPER_RUN_NAME)
    _write_all_sources(tmp_path)

    with pytest.raises(
        OfficialReferenceFidelityEvidenceError,
        match="CPU 审计仓库代码版本",
    ):
        write_official_reference_fidelity_evidence_outputs(
            root=tmp_path,
            repository_code_version="9" * 40,
        )
