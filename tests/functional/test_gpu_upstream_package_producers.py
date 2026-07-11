"""验证主方法 GPU 上游包与 CPU 闭合选择契约完全一致。"""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

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
    inspect_closure_package,
)


PAPER_RUN_NAME = "pilot_paper"
TARGET_FPR = 0.01
PROMPT_COUNT = 700
GENERATED_AT = "2026-07-11T00:00:00+00:00"


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
            "supports_paper_claim": True,
        },
    )
    _write_json(
        directory / "manifest.local.json",
        {
            "artifact_id": f"{PAPER_RUN_NAME}_image_only_dataset_runtime_manifest",
            "artifact_type": "local_manifest",
            "code_version": "b370425",
            "config": {
                "paper_run": {
                    "run_name": PAPER_RUN_NAME,
                    "target_fpr": TARGET_FPR,
                }
            },
        },
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
            "code_version": "b370425",
            "config": {"target_fpr": TARGET_FPR, **ablation_contract},
            "metadata": ablation_contract,
        },
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
            "code_version": "b370425",
            "metadata": {
                "paper_run_name": PAPER_RUN_NAME,
                "target_fpr": TARGET_FPR,
                **coverage,
            },
            "config": coverage,
        },
    )
    return package_dataset_level_quality_outputs(PAPER_RUN_NAME, root=root)


@pytest.mark.quick
def test_primary_gpu_package_producers_pass_strict_closure_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """三个主方法上游包应为 run-scoped 且可直接通过精确闭合选择器。"""

    monkeypatch.delenv("SLM_WM_PAPER_RUN_NAME", raising=False)
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
            assert archive.namelist()
            assert all(
                name.startswith("outputs/") and f"/{PAPER_RUN_NAME}/" in name
                for name in archive.namelist()
            )
            assert sentinel.name not in archive.namelist()


@pytest.mark.quick
def test_primary_gpu_package_producers_reject_non_ready_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """上游 summary 未通过时不得先生成可被误选的 ZIP。"""

    monkeypatch.delenv("SLM_WM_PAPER_RUN_NAME", raising=False)
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

    monkeypatch.delenv("SLM_WM_PAPER_RUN_NAME", raising=False)
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
