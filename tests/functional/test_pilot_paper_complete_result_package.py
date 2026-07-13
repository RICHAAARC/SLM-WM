"""论文完整结果包公开门禁与通用归档原语测试."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import pytest

from paper_experiments.runners.paper_claim_provenance import (
    PaperClaimAggregateRequiredError,
)
from scripts.write_pilot_paper_complete_result_package import (
    REQUIRED_OUTPUT_DIR_NAMES,
    _validated_zip_member_name,
    build_parser,
    build_required_output_dirs,
    resolve_explicit_package_paths,
    write_archive_with_progress,
    write_pilot_paper_complete_result_package_outputs,
)


pytestmark = pytest.mark.quick


def test_public_complete_result_package_rejects_missing_exact9_aggregate_before_output(
    tmp_path: Path,
) -> None:
    """公开打包入口缺少精确9重复来源时不得创建 outputs 目录."""

    with pytest.raises(
        PaperClaimAggregateRequiredError,
        match="版本化精确9重复聚合证据",
    ):
        write_pilot_paper_complete_result_package_outputs(root=tmp_path)

    assert not (tmp_path / "outputs").exists()


def test_explicit_package_resolution_never_scans_neighboring_archives(
    tmp_path: Path,
) -> None:
    """通用来源解析只接受显式路径, 不从相邻目录推断输入."""

    selected = tmp_path / "selected.zip"
    ignored = tmp_path / "ignored.zip"
    selected.write_bytes(b"selected")
    ignored.write_bytes(b"ignored")

    resolved = resolve_explicit_package_paths(tmp_path, (selected.name,))

    assert resolved == (selected.resolve(),)
    assert ignored.resolve() not in resolved
    assert "--package-search-root" not in build_parser().format_help()


def test_zip_member_validator_rejects_noncanonical_paths() -> None:
    """归档成员必须使用不可逃逸的规范 POSIX 相对路径."""

    assert _validated_zip_member_name("outputs/result.json") == "outputs/result.json"
    for member_name in (
        "../result.json",
        "/outputs/result.json",
        "outputs\\result.json",
        "outputs/./result.json",
    ):
        with pytest.raises(RuntimeError, match="非规范成员路径"):
            _validated_zip_member_name(member_name)


def test_archive_writer_preserves_explicit_member_bytes(tmp_path: Path) -> None:
    """通用归档写入应保持显式文件的路径与字节内容."""

    source = tmp_path / "outputs" / "evidence" / "record.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b'{"ready": false}\n')
    archive_path = tmp_path / "outputs" / "package.zip"

    write_archive_with_progress(
        tmp_path,
        archive_path,
        (source,),
        compression_method=ZIP_STORED,
    )

    with ZipFile(archive_path) as archive:
        assert archive.namelist() == ["outputs/evidence/record.json"]
        assert archive.read("outputs/evidence/record.json") == source.read_bytes()


def test_complete_result_package_required_directories_cover_evidence_closure_chain() -> None:
    """关键科学结果、审计与投稿门禁目录必须进入动态目录集合."""

    required_names = set(REQUIRED_OUTPUT_DIR_NAMES)
    assert {
        "image_only_dataset_runtime",
        "formal_mechanism_ablation",
        "dataset_level_quality",
        "external_baseline_comparison",
        "attack_matrix",
        "fixed_fpr_threshold_audit",
        "paired_superiority_analysis",
        "paper_artifact_evidence_audit",
        "submission_readiness",
        "evidence_closure_entry_review",
        "result_closure_gate",
    } <= required_names
    assert all(
        path.endswith("/probe_paper")
        for path in build_required_output_dirs("probe_paper")
    )
