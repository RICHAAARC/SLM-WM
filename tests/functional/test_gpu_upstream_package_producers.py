"""验证主方法 GPU 上游包与 CPU 闭合选择契约完全一致。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.runtime import repository_environment
from experiments.runtime.package_input_manifest import (
    validate_exact_package_archive,
)
from experiments.ablations.runtime_rerun import (
    FORMAL_RUNTIME_RERUN_ABLATION_IDS,
    FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
    package_runtime_rerun_ablations,
)
from experiments.artifacts.dataset_level_quality_outputs import (
    canonical_prompt_ids_for_paper_run,
    package_dataset_level_quality_outputs,
    path_digest,
)
from experiments.protocol.prompts import PROMPT_FILES
from main.core.digest import build_stable_digest
from experiments.runners.image_only_dataset_runtime import (
    package_image_only_dataset_runtime,
)
from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
    ClosurePackageSelectionError,
    inspect_closure_package,
)
from tests.helpers.formal_execution_lock import build_test_formal_execution_lock
from tests.helpers.scientific_execution_binding import (
    write_test_scientific_execution_binding,
)


PAPER_RUN_NAME = "pilot_paper"
TARGET_FPR = 0.01
PROMPT_COUNT = 700
GENERATED_AT = "2026-07-11T00:00:00+00:00"
FORMAL_EXECUTION_LOCK = build_test_formal_execution_lock()


@pytest.fixture(autouse=True)
def _publish_formal_execution_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """把临时输出目录绑定到确定性正式执行锁."""

    monkeypatch.setattr(
        repository_environment,
        "require_published_formal_execution_lock",
        lambda _root: dict(FORMAL_EXECUTION_LOCK),
    )


def _write_json(path: Path, payload: object) -> None:
    """写出稳定 JSON 测试产物。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_required_files(directory: Path, filenames: tuple[str, ...]) -> None:
    """写出不承担字段身份的必要文件。"""

    directory.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        path = directory / filename
        path.write_text("{}\n", encoding="utf-8")


def _prepare_image_runtime(root: Path) -> Path:
    """构造仅图像运行打包门禁所需的最小正式形状。"""

    directory = root / "outputs" / "image_only_dataset_runtime" / PAPER_RUN_NAME
    _write_required_files(
        directory,
        (
            "runtime_results.jsonl",
            "image_only_detection_records.jsonl",
            "watermark_quality_image_registry.jsonl",
            "frozen_evidence_protocol.json",
            "test_detection_metrics.csv",
            "score_distribution_table.csv",
            "roc_curve_points.csv",
            "det_curve_points.csv",
        ),
    )
    _write_json(
        directory / "dataset_runtime_summary.json",
        {
            "generated_at": GENERATED_AT,
            "paper_run_name": PAPER_RUN_NAME,
            "target_fpr": TARGET_FPR,
            "protocol_decision": "pass",
            "full_method_claim_ready": True,
            "detection_curve_data_ready": True,
            "supports_paper_claim": True,
        },
    )
    _write_json(
        directory / "manifest.local.json",
        {
            "artifact_id": f"{PAPER_RUN_NAME}_image_only_dataset_runtime_manifest",
            "artifact_type": "local_manifest",
            "code_version": "b" * 40,
            "formal_execution_run_lock": FORMAL_EXECUTION_LOCK,
            "output_paths": [
                (directory / filename).relative_to(root).as_posix()
                for filename in (
                    "runtime_results.jsonl",
                    "image_only_detection_records.jsonl",
                    "watermark_quality_image_registry.jsonl",
                    "frozen_evidence_protocol.json",
                    "test_detection_metrics.csv",
                    "score_distribution_table.csv",
                    "roc_curve_points.csv",
                    "det_curve_points.csv",
                    "dataset_runtime_summary.json",
                    "manifest.local.json",
                )
            ],
            "config": {
                "paper_run": {
                    "run_name": PAPER_RUN_NAME,
                    "target_fpr": TARGET_FPR,
                }
            },
        },
    )
    write_test_scientific_execution_binding(
        repository_root=root,
        artifact_dir=directory,
        artifact_role="image_only_dataset_runtime",
        paper_run_name=PAPER_RUN_NAME,
        profile_id="sd35_method_runtime_gpu",
        summary_file_name="dataset_runtime_summary.json",
        manifest_file_name="manifest.local.json",
        formal_execution_lock=FORMAL_EXECUTION_LOCK,
        execution_route="semantic_watermark_session",
    )
    return package_image_only_dataset_runtime(PAPER_RUN_NAME, root=root)


def _prepare_ablation(root: Path) -> Path:
    """构造正式重运行消融打包门禁所需的最小正式形状。"""

    directory = root / "outputs" / "formal_mechanism_ablation" / PAPER_RUN_NAME
    _write_required_files(
        directory,
        (
            "runtime_rerun_records.jsonl",
            "formal_detection_records.jsonl",
            "mechanism_pairwise_delta.csv",
        ),
    )
    _write_json(
        directory / "per_ablation_frozen_protocols.json",
        {ablation_id: {} for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS},
    )
    (directory / "mechanism_ablation_metrics.csv").write_text(
        "ablation_id\n"
        + "".join(f"{ablation_id}\n" for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        encoding="utf-8",
    )
    ablation_contract = {
        "expected_ablation_ids": list(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "actual_ablation_ids": list(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "ablation_spec_digest": FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
        "ablation_exact_set_ready": True,
    }
    _write_json(
        directory / "ablation_claim_summary.json",
        {
            "generated_at": GENERATED_AT,
            "paper_run_name": PAPER_RUN_NAME,
            "target_fpr": TARGET_FPR,
            **ablation_contract,
            "protocol_decision": "pass",
            "ablation_claim_gate_ready": True,
            "supports_paper_claim": True,
        },
    )
    _write_json(
        directory / "manifest.local.json",
        {
            "artifact_id": "formal_mechanism_ablation_manifest",
            "artifact_type": "local_manifest",
            "code_version": "b" * 40,
            "formal_execution_run_lock": FORMAL_EXECUTION_LOCK,
            "output_paths": [
                (directory / filename).relative_to(root).as_posix()
                for filename in (
                    "runtime_rerun_records.jsonl",
                    "formal_detection_records.jsonl",
                    "per_ablation_frozen_protocols.json",
                    "mechanism_ablation_metrics.csv",
                    "mechanism_pairwise_delta.csv",
                    "ablation_claim_summary.json",
                    "manifest.local.json",
                )
            ],
            "config": {"target_fpr": TARGET_FPR, **ablation_contract},
            "metadata": ablation_contract,
        },
    )
    write_test_scientific_execution_binding(
        repository_root=root,
        artifact_dir=directory,
        artifact_role="runtime_rerun_ablation",
        paper_run_name=PAPER_RUN_NAME,
        profile_id="sd35_method_runtime_gpu",
        summary_file_name="ablation_claim_summary.json",
        manifest_file_name="manifest.local.json",
        formal_execution_lock=FORMAL_EXECUTION_LOCK,
        execution_route="semantic_watermark_ablation_session",
    )
    return package_runtime_rerun_ablations(PAPER_RUN_NAME, root=root)


def _prepare_dataset_quality(root: Path) -> Path:
    """构造正式 FID/KID 打包门禁所需的最小正式形状。"""

    directory = root / "outputs" / "dataset_level_quality" / PAPER_RUN_NAME
    _write_required_files(
        directory,
        (
            "dataset_quality_image_records.jsonl",
            "dataset_quality_image_resolution_records.jsonl",
        ),
    )
    feature_records_path = directory / "dataset_quality_formal_feature_records.jsonl"
    feature_records_path.write_bytes(b"{}\n")
    feature_sha256 = path_digest(feature_records_path)
    canonical_ids = canonical_prompt_ids_for_paper_run(
        root_path=root,
        prompt_set=PAPER_RUN_NAME,
        prompt_file=PROMPT_FILES[PAPER_RUN_NAME],
    )
    prompt_digest = build_stable_digest(sorted(canonical_ids))
    coverage = {
        "canonical_prompt_id_digest": prompt_digest,
        "registry_prompt_id_digest": prompt_digest,
        "prompt_registry_exact_set_ready": True,
        "accepted_feature_pair_count": PROMPT_COUNT,
        "missing_feature_pair_count": 0,
        "feature_issue_count": 0,
        "formal_feature_record_count": PROMPT_COUNT * 2,
        "formal_feature_records_sha256": feature_sha256,
    }
    _write_json(
        directory / "dataset_quality_formal_feature_import_report.json",
        {
            "paper_run_name": PAPER_RUN_NAME,
            "target_fpr": TARGET_FPR,
            "expected_feature_pair_count": PROMPT_COUNT,
            **coverage,
        },
    )
    (directory / "dataset_quality_metrics.csv").write_text(
        "quality_metric_name,metric_status,source_image_count,comparison_image_count,sample_pair_count\n"
        f"fid,measured,{PROMPT_COUNT},{PROMPT_COUNT},{PROMPT_COUNT}\n"
        f"kid,measured,{PROMPT_COUNT},{PROMPT_COUNT},{PROMPT_COUNT}\n",
        encoding="utf-8",
    )
    _write_json(
        directory / "dataset_quality_summary.json",
        {
            "generated_at": GENERATED_AT,
            "paper_run_name": PAPER_RUN_NAME,
            "target_fpr": TARGET_FPR,
            "expected_prompt_count": PROMPT_COUNT,
            "registry_prompt_count": PROMPT_COUNT,
            "sample_pair_count": PROMPT_COUNT,
            **coverage,
            "formal_feature_backend_ready": True,
            "formal_sample_scale_ready": True,
            "canonical_formal_feature_extractor_ready": True,
            "formal_fid_kid_claim_gate_ready": True,
        },
    )
    _write_json(
        directory / "manifest.local.json",
        {
            "artifact_id": "dataset_level_quality_manifest",
            "artifact_type": "local_manifest",
            "code_version": "b" * 40,
            "formal_execution_run_lock": FORMAL_EXECUTION_LOCK,
            "output_paths": [
                (directory / filename).relative_to(root).as_posix()
                for filename in (
                    "dataset_quality_image_records.jsonl",
                    "dataset_quality_image_resolution_records.jsonl",
                    "dataset_quality_formal_feature_records.jsonl",
                    "dataset_quality_formal_feature_import_report.json",
                    "dataset_quality_metrics.csv",
                    "dataset_quality_summary.json",
                    "manifest.local.json",
                )
            ],
            "metadata": {
                "paper_run_name": PAPER_RUN_NAME,
                "target_fpr": TARGET_FPR,
                **coverage,
            },
            "config": coverage,
        },
    )
    write_test_scientific_execution_binding(
        repository_root=root,
        artifact_dir=directory,
        artifact_role="dataset_level_quality",
        paper_run_name=PAPER_RUN_NAME,
        profile_id="sd35_method_runtime_gpu",
        summary_file_name="dataset_quality_summary.json",
        manifest_file_name="manifest.local.json",
        formal_execution_lock=FORMAL_EXECUTION_LOCK,
        execution_route="semantic_watermark_session",
    )
    return package_dataset_level_quality_outputs(PAPER_RUN_NAME, root=root)


@pytest.mark.quick
def test_primary_gpu_package_producers_pass_strict_closure_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """三个主方法上游包应为 run-scoped 且可直接通过精确闭合选择器。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", PAPER_RUN_NAME)
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)
    sentinel = tmp_path / "outside_family.txt"
    sentinel.write_text("不得归档\n", encoding="utf-8")
    archives = (
        _prepare_image_runtime(tmp_path),
        _prepare_ablation(tmp_path),
        _prepare_dataset_quality(tmp_path),
    )
    for archive_path, spec in zip(archives, CLOSURE_PACKAGE_FAMILY_SPECS[:3]):
        candidate = inspect_closure_package(
            archive_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )
        assert candidate.package_family == spec.package_family
        with ZipFile(archive_path) as archive:
            archive_names = set(archive.namelist())
            assert archive_names
            assert all(
                name.startswith("outputs/") and f"/{PAPER_RUN_NAME}/" in name
                for name in archive_names
            )
            assert sentinel.name not in archive_names
            assert spec.package_input_manifest_template is not None
            package_input_member = spec.package_input_manifest_template.format(
                paper_run=PAPER_RUN_NAME,
                baseline=spec.baseline_id or "",
            )
            package_input = json.loads(archive.read(package_input_member))
            declared_paths = package_input["entry_paths"]
            assert package_input["entry_count"] == len(declared_paths)
            assert set(declared_paths) == archive_names - {package_input_member}
            assert package_input["entry_sha256"] == {
                member_name: hashlib.sha256(archive.read(member_name)).hexdigest()
                for member_name in declared_paths
            }


@pytest.mark.quick
def test_primary_package_ignores_stale_file_and_selector_rejects_undeclared_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """打包器不得吸收遗留文件, 选择器也必须拒绝归档后追加的同前缀成员."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", PAPER_RUN_NAME)
    archive_path = _prepare_image_runtime(tmp_path)
    archive_path.unlink()
    output_dir = (
        tmp_path
        / "outputs"
        / "image_only_dataset_runtime"
        / PAPER_RUN_NAME
    )
    stale_path = output_dir / "stale_same_prefix.json"
    stale_path.write_text("{}\n", encoding="utf-8")

    archive_path = package_image_only_dataset_runtime(PAPER_RUN_NAME, root=tmp_path)
    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]
    with ZipFile(archive_path) as archive:
        assert stale_path.relative_to(tmp_path).as_posix() not in archive.namelist()

    undeclared_member = (
        f"outputs/image_only_dataset_runtime/{PAPER_RUN_NAME}/"
        "undeclared_same_prefix.json"
    )
    with ZipFile(archive_path, "a") as archive:
        archive.writestr(undeclared_member, b"{}\n")
    assert spec.package_input_manifest_template is not None
    package_input_path = tmp_path / spec.package_input_manifest_template.format(
        paper_run=PAPER_RUN_NAME,
        baseline=spec.baseline_id or "",
    )
    with pytest.raises(RuntimeError, match="写后成员集合"):
        validate_exact_package_archive(
            archive_path,
            repository_root=tmp_path,
            package_input_manifest_path=package_input_path,
        )
    with pytest.raises(ClosurePackageSelectionError, match="精确成员集合不一致"):
        inspect_closure_package(
            archive_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


@pytest.mark.quick
def test_primary_gpu_package_producers_reject_non_ready_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """上游 summary 未通过时不得先生成可被误选的 ZIP。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", PAPER_RUN_NAME)
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)
    archive_path = _prepare_image_runtime(tmp_path)
    archive_path.unlink()
    summary_path = (
        tmp_path
        / "outputs"
        / "image_only_dataset_runtime"
        / PAPER_RUN_NAME
        / "dataset_runtime_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["supports_paper_claim"] = False
    _write_json(summary_path, summary)
    with pytest.raises(RuntimeError, match="ready 门禁"):
        package_image_only_dataset_runtime(PAPER_RUN_NAME, root=tmp_path)
    assert not tuple(summary_path.parent.glob("image_only_dataset_runtime_package_*.zip"))


@pytest.mark.quick
def test_ablation_and_quality_packages_reject_inexact_scientific_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """消融非8项或质量特征缺配对时不得生成新的正式 ZIP。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", PAPER_RUN_NAME)
    ablation_archive = _prepare_ablation(tmp_path)
    ablation_archive.unlink()
    ablation_summary_path = (
        tmp_path
        / "outputs/formal_mechanism_ablation"
        / PAPER_RUN_NAME
        / "ablation_claim_summary.json"
    )
    ablation_summary = json.loads(ablation_summary_path.read_text(encoding="utf-8"))
    ablation_summary["actual_ablation_ids"] = list(FORMAL_RUNTIME_RERUN_ABLATION_IDS[:6])
    _write_json(ablation_summary_path, ablation_summary)
    with pytest.raises(RuntimeError, match="精确8项规范"):
        package_runtime_rerun_ablations(PAPER_RUN_NAME, root=tmp_path)

    quality_archive = _prepare_dataset_quality(tmp_path)
    quality_archive.unlink()
    quality_report_path = (
        tmp_path
        / "outputs/dataset_level_quality"
        / PAPER_RUN_NAME
        / "dataset_quality_formal_feature_import_report.json"
    )
    quality_report = json.loads(quality_report_path.read_text(encoding="utf-8"))
    quality_report["accepted_feature_pair_count"] = PROMPT_COUNT - 1
    quality_report["missing_feature_pair_count"] = 1
    _write_json(quality_report_path, quality_report)
    with pytest.raises(RuntimeError, match="精确 Prompt/特征覆盖"):
        package_dataset_level_quality_outputs(PAPER_RUN_NAME, root=tmp_path)


@pytest.mark.quick
def test_package_removes_archive_when_final_execution_lock_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """归档写出后的 Git 锁漂移必须删除尚未交付的 ZIP."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", PAPER_RUN_NAME)
    archive_path = _prepare_image_runtime(tmp_path)
    archive_path.unlink()
    changed_lock = build_test_formal_execution_lock("c" * 40)
    lock_records = iter((FORMAL_EXECUTION_LOCK, changed_lock))
    monkeypatch.setattr(
        repository_environment,
        "require_published_formal_execution_lock",
        lambda _root: dict(next(lock_records)),
    )

    with pytest.raises(repository_environment.FormalExecutionLockError):
        package_image_only_dataset_runtime(PAPER_RUN_NAME, root=tmp_path)

    output_dir = tmp_path / "outputs" / "image_only_dataset_runtime" / PAPER_RUN_NAME
    assert not tuple(output_dir.glob("image_only_dataset_runtime_package_*.zip"))
