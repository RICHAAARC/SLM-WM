"""数据集级图像质量证据链路测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest

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


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    """写入小尺寸 RGB 图像 fixture。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


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
