"""数据集级图像质量证据链路测试。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest

import scripts.write_dataset_level_quality_outputs as dataset_quality_writer
from experiments.protocol import (
    FORMAL_FEATURE_BACKEND,
    FORMAL_FID_KID_BLOCKER,
    FORMAL_FID_KID_SAMPLE_BLOCKER,
    PIXEL_FEATURE_BACKEND,
    build_dataset_quality_image_records,
    build_dataset_quality_metric_rows,
    build_dataset_quality_summary,
)
from scripts.write_dataset_level_quality_outputs import write_dataset_level_quality_outputs
from paper_workflow.colab_utils.dataset_level_quality import (
    dataset_level_quality_claim_boundary,
    run_default_dataset_level_quality_from_drive_plan,
)


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    """写入小尺寸 RGB 图像 fixture。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def file_digest(path: Path) -> str:
    """计算测试图像文件的 SHA-256 摘要."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def registry_rows(root_path: Path) -> list[dict[str, object]]:
    """构造两个 source / attacked 图像对。"""

    source_a = root_path / "outputs" / "images" / "source_a.png"
    source_b = root_path / "outputs" / "images" / "source_b.png"
    attacked_a = root_path / "outputs" / "images" / "attacked_a.png"
    attacked_b = root_path / "outputs" / "images" / "attacked_b.png"
    write_image(source_a, (255, 0, 0))
    write_image(source_b, (0, 255, 0))
    write_image(attacked_a, (240, 10, 10))
    write_image(attacked_b, (10, 240, 10))
    return [
        {
            "attack_name": "img2img_regeneration",
            "source_image_path": source_a.relative_to(root_path).as_posix(),
            "attacked_image_path": attacked_a.relative_to(root_path).as_posix(),
        },
        {
            "attack_name": "sdedit_regeneration",
            "source_image_path": source_b.relative_to(root_path).as_posix(),
            "attacked_image_path": attacked_b.relative_to(root_path).as_posix(),
        },
    ]


@pytest.mark.quick
def test_dataset_quality_protocol_keeps_formal_fid_kid_unsupported(tmp_path: Path) -> None:
    """小样本 proxy 可测量, 但正式 FID / KID 仍必须保持 unsupported。"""

    rows = registry_rows(tmp_path)
    records = build_dataset_quality_image_records(rows, tmp_path)
    metric_rows = build_dataset_quality_metric_rows(records, tmp_path)
    summary = build_dataset_quality_summary(records, metric_rows)
    rows_by_name = {row["quality_metric_name"]: row for row in metric_rows}

    assert len(records) == 2
    assert rows_by_name["fid"]["metric_status"] == FORMAL_FID_KID_BLOCKER
    assert rows_by_name["kid"]["metric_status"] == FORMAL_FID_KID_BLOCKER
    assert rows_by_name["fid_pixel_feature_proxy"]["metric_status"] == "measured_small_sample_proxy"
    assert rows_by_name["kid_pixel_feature_proxy"]["metric_status"] == "measured_small_sample_proxy"
    assert summary["feature_backend"] == PIXEL_FEATURE_BACKEND
    assert summary["dataset_level_quality_proxy_ready"] is True
    assert summary["formal_fid_kid_ready"] is False
    assert summary["supports_paper_claim"] is False


@pytest.mark.quick
def test_dataset_quality_writer_outputs_rebuildable_artifacts(tmp_path: Path) -> None:
    """写出脚本应从真实攻击 registry 重建数据集级质量治理产物。"""

    rows = registry_rows(tmp_path)
    registry_path = tmp_path / "outputs" / "real_attack_evaluation" / "real_attacked_image_registry.jsonl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    manifest = write_dataset_level_quality_outputs(root=tmp_path)
    output_dir = tmp_path / "outputs" / "dataset_level_quality"
    metric_rows = list(csv.DictReader((output_dir / "dataset_quality_metrics.csv").open(encoding="utf-8")))
    summary = json.loads((output_dir / "dataset_quality_summary.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "dataset_level_quality_manifest"
    assert {row["quality_metric_name"] for row in metric_rows} == {
        "fid",
        "kid",
        "fid_pixel_feature_proxy",
        "kid_pixel_feature_proxy",
    }
    assert summary["dataset_level_quality_proxy_ready"] is True
    assert summary["formal_fid_kid_ready"] is False
    assert all(str(path).startswith("outputs/") for path in manifest["output_paths"])


@pytest.mark.quick
def test_dataset_quality_writer_materializes_missing_images_from_input_package(tmp_path: Path) -> None:
    """当前序图像只存在于 ZIP 包中时, 写出脚本应只物化所需图像并完成小样本 proxy 计算。"""

    source_path = tmp_path / "outputs" / "aligned_rescoring" / "aligned_images" / "source_from_package.png"
    attacked_path = tmp_path / "outputs" / "real_attack_evaluation" / "attacked_images" / "attacked_existing.png"
    write_image(source_path, (120, 10, 10))
    write_image(attacked_path, (100, 20, 20))

    package_path = tmp_path / "outputs" / "input_packages" / "aligned_rescoring_package.zip"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package_path, "w") as archive:
        archive.write(source_path, source_path.relative_to(tmp_path).as_posix())
    source_path.unlink()

    registry_path = tmp_path / "outputs" / "real_attack_evaluation" / "real_attacked_image_registry.jsonl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "attack_name": "img2img_regeneration",
                "source_image_path": "outputs/aligned_rescoring/aligned_images/source_from_package.png",
                "attacked_image_path": attacked_path.relative_to(tmp_path).as_posix(),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = write_dataset_level_quality_outputs(root=tmp_path, input_package_paths=(package_path,))
    output_dir = tmp_path / "outputs" / "dataset_level_quality"
    summary = json.loads((output_dir / "dataset_quality_summary.json").read_text(encoding="utf-8"))
    resolution_rows = [
        json.loads(line)
        for line in (output_dir / "dataset_quality_image_resolution_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["dataset_level_quality_proxy_ready"] is True
    assert summary["formal_fid_kid_ready"] is False
    assert summary["materialized_image_input_count"] == 1
    assert any(row["resolution_status"] == "materialized_from_input_package" for row in resolution_rows)
    assert any("materialized_image_inputs" in path for path in manifest["output_paths"])


@pytest.mark.quick
def test_dataset_quality_materialization_caches_package_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """物化多张图像时应按包缓存 ZIP 摘要, 避免重复扫描大文件。"""

    image_entries = (
        Path("outputs/images/source_a.png"),
        Path("outputs/images/attacked_a.png"),
        Path("outputs/images/source_b.png"),
        Path("outputs/images/attacked_b.png"),
    )
    colors = ((10, 20, 30), (11, 21, 31), (40, 50, 60), (41, 51, 61))
    for entry, color in zip(image_entries, colors, strict=True):
        write_image(tmp_path / entry, color)

    records = build_dataset_quality_image_records(
        [
            {
                "attack_name": "jpeg_compression",
                "source_image_path": image_entries[0].as_posix(),
                "attacked_image_path": image_entries[1].as_posix(),
            },
            {
                "attack_name": "gaussian_blur",
                "source_image_path": image_entries[2].as_posix(),
                "attacked_image_path": image_entries[3].as_posix(),
            },
        ],
        tmp_path,
    )
    package_path = tmp_path / "outputs" / "input_packages" / "dataset_quality_images.zip"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package_path, "w") as archive:
        for entry in image_entries:
            archive.write(tmp_path / entry, entry.as_posix())

    original_path_digest = dataset_quality_writer.path_digest
    package_digest_calls: list[str] = []

    def counting_path_digest(path: Path) -> str:
        """统计 ZIP 摘要计算次数, 保持原始摘要语义不变。"""

        if Path(path).suffix == ".zip":
            package_digest_calls.append(Path(path).name)
        return original_path_digest(path)

    monkeypatch.setattr(dataset_quality_writer, "path_digest", counting_path_digest)
    materialized_records = dataset_quality_writer.materialize_images_from_input_packages(
        records=records,
        materialized_root=tmp_path / "outputs" / "dataset_level_quality" / "materialized_image_inputs",
        input_package_paths=(package_path,),
    )

    assert len(materialized_records) == 4
    assert package_digest_calls == [package_path.name]


@pytest.mark.quick
def test_dataset_quality_formal_feature_import_keeps_small_sample_blocked(tmp_path: Path) -> None:
    """即使导入了 Inception 特征, 小样本也不能被升级为正式 FID / KID 结论。"""

    rows = registry_rows(tmp_path)
    registry_path = tmp_path / "outputs" / "real_attack_evaluation" / "real_attacked_image_registry.jsonl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    records = build_dataset_quality_image_records(rows, tmp_path)
    feature_records_path = tmp_path / "outputs" / "dataset_level_quality" / "dataset_quality_formal_feature_records.jsonl"
    feature_records_path.parent.mkdir(parents=True, exist_ok=True)
    feature_lines = []
    for index, record in enumerate(records):
        source_vector = [float(index), 0.0, 1.0]
        comparison_vector = [float(index), 0.2, 0.8]
        for image_role, vector in (("source", source_vector), ("comparison", comparison_vector)):
            feature_lines.append(
                json.dumps(
                    {
                        "dataset_quality_record_id": record.dataset_quality_record_id,
                        "dataset_quality_image_role": image_role,
                        "feature_backend": FORMAL_FEATURE_BACKEND,
                        "feature_vector": vector,
                    },
                    ensure_ascii=False,
                )
            )
    feature_records_path.write_text("\n".join(feature_lines) + "\n", encoding="utf-8")

    manifest = write_dataset_level_quality_outputs(
        root=tmp_path,
        formal_feature_records_path=feature_records_path,
        formal_min_sample_count=10,
    )
    output_dir = tmp_path / "outputs" / "dataset_level_quality"
    metric_rows = list(csv.DictReader((output_dir / "dataset_quality_metrics.csv").open(encoding="utf-8")))
    rows_by_name = {row["quality_metric_name"]: row for row in metric_rows}
    summary = json.loads((output_dir / "dataset_quality_summary.json").read_text(encoding="utf-8"))
    import_report = json.loads((output_dir / "dataset_quality_formal_feature_import_report.json").read_text(encoding="utf-8"))

    assert rows_by_name["fid"]["feature_backend"] == FORMAL_FEATURE_BACKEND
    assert rows_by_name["fid"]["metric_status"] == FORMAL_FID_KID_SAMPLE_BLOCKER
    assert rows_by_name["kid"]["metric_status"] == FORMAL_FID_KID_SAMPLE_BLOCKER
    assert import_report["formal_feature_backend_ready"] is True
    assert import_report["formal_sample_scale_ready"] is False
    assert summary["formal_fid_kid_ready"] is False
    assert summary["formal_feature_backend_ready"] is True
    assert summary["formal_sample_scale_ready"] is False
    assert any(path.endswith("dataset_quality_formal_feature_import_report.json") for path in manifest["output_paths"])


@pytest.mark.quick
def test_dataset_quality_formal_feature_import_measures_when_scale_is_ready(tmp_path: Path) -> None:
    """样本规模达到门槛时, 正式 Inception 特征应完成 FID / KID 后处理。"""

    rows = registry_rows(tmp_path)
    records = build_dataset_quality_image_records(rows, tmp_path)
    formal_source_features = [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
    formal_comparison_features = [
        [0.1, 0.0, 0.9],
        [0.9, 0.1, 0.0],
    ]

    metric_rows = build_dataset_quality_metric_rows(
        records,
        tmp_path,
        formal_source_features=formal_source_features,
        formal_comparison_features=formal_comparison_features,
        formal_min_sample_count=2,
    )
    summary = build_dataset_quality_summary(records, metric_rows)
    rows_by_name = {row["quality_metric_name"]: row for row in metric_rows}

    assert rows_by_name["fid"]["metric_status"] == "measured"
    assert rows_by_name["kid"]["metric_status"] == "measured"
    assert isinstance(rows_by_name["fid"]["quality_metric_value"], float)
    assert isinstance(rows_by_name["kid"]["quality_metric_value"], float)
    assert summary["formal_fid_kid_ready"] is True
    assert (
        dataset_level_quality_claim_boundary(summary)
        == "formal_fid_kid_measured_but_paper_claim_requires_evidence_closure"
    )


@pytest.mark.quick
def test_dataset_quality_drive_plan_imports_mock_formal_features(tmp_path: Path) -> None:
    """Drive 前序包链路应能生成正式特征记录, 但小样本仍保持 FID / KID 阻断."""

    staging_dir = tmp_path / "staging"
    source_entry = Path("outputs/aligned_rescoring/aligned_images/source.png")
    attacked_entry = Path("outputs/real_attack_evaluation/attacked_images/attacked.png")
    source_file = staging_dir / source_entry
    attacked_file = staging_dir / attacked_entry
    write_image(source_file, (30, 60, 90))
    write_image(attacked_file, (35, 65, 95))

    real_attack_drive_dir = tmp_path / "drive" / "real_attack_evaluation"
    aligned_drive_dir = tmp_path / "drive" / "aligned_rescoring"
    real_attack_drive_dir.mkdir(parents=True)
    aligned_drive_dir.mkdir(parents=True)
    registry_entry = Path("outputs/real_attack_evaluation/real_attacked_image_registry.jsonl")
    registry_file = staging_dir / registry_entry
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_file.write_text(
        json.dumps(
            {
                "attack_name": "img2img_regeneration",
                "source_image_path": source_entry.as_posix(),
                "source_image_digest": file_digest(source_file),
                "attacked_image_path": attacked_entry.as_posix(),
                "attacked_image_digest": file_digest(attacked_file),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    real_attack_package = real_attack_drive_dir / "real_attack_evaluation_package_20260623.zip"
    aligned_package = aligned_drive_dir / "aligned_rescoring_package_20260623.zip"
    with zipfile.ZipFile(real_attack_package, "w") as archive:
        archive.write(registry_file, registry_entry.as_posix())
        archive.write(attacked_file, attacked_entry.as_posix())
    with zipfile.ZipFile(aligned_package, "w") as archive:
        archive.write(source_file, source_entry.as_posix())

    def mock_feature_extractor(image_path: Path) -> list[float]:
        """测试用轻量特征后端, 避免默认测试下载 Inception 权重."""

        return [float(len(image_path.name)), 0.5, 1.0]

    result = run_default_dataset_level_quality_from_drive_plan(
        root=tmp_path,
        real_attack_evaluation_drive_dir=str(real_attack_drive_dir),
        aligned_rescoring_drive_dir=str(aligned_drive_dir),
        formal_min_sample_count=10,
        feature_extractor=mock_feature_extractor,
        environment_report={"feature_backend": FORMAL_FEATURE_BACKEND, "package_versions": {}},
    )
    output_dir = tmp_path / "outputs" / "dataset_level_quality"
    feature_rows = [
        json.loads(line)
        for line in (output_dir / "dataset_quality_formal_feature_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    import_report = json.loads((output_dir / "dataset_quality_formal_feature_import_report.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "dataset_quality_summary.json").read_text(encoding="utf-8"))
    resolution_rows = [
        json.loads(line)
        for line in (output_dir / "dataset_quality_image_resolution_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result["run_decision"] == "pass"
    assert result["formal_feature_backend_ready"] is True
    assert result["formal_fid_kid_ready"] is False
    assert result["unsupported_reason"] == FORMAL_FID_KID_SAMPLE_BLOCKER
    assert result["metadata"]["claim_boundary"] == "formal_feature_backend_ready_but_formal_fid_kid_blocked"
    assert len(feature_rows) == 2
    assert {row["dataset_quality_image_role"] for row in feature_rows} == {"source", "comparison"}
    assert import_report["accepted_feature_pair_count"] == 1
    assert import_report["formal_sample_scale_ready"] is False
    assert summary["formal_feature_backend_ready"] is True
    assert summary["materialized_image_input_count"] == 2
    assert all(row["materialized_image_input"] is True for row in resolution_rows)
    assert all("outputs/dataset_level_quality/materialized_image_inputs/" in row["resolved_image_path"] for row in resolution_rows)
    assert (output_dir / "dataset_level_quality_colab_manifest.local.json").exists()
