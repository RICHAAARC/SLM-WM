"""数据集级正式 Inception FID / KID 证据链路测试。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest

import experiments.artifacts.dataset_level_quality_outputs as dataset_quality_writer
from experiments.artifacts.dataset_level_quality_outputs import write_dataset_level_quality_outputs
from experiments.protocol import (
    FORMAL_FEATURE_BACKEND,
    FORMAL_FID_KID_BLOCKER,
    FORMAL_FID_KID_SAMPLE_BLOCKER,
    build_dataset_quality_image_records,
    build_dataset_quality_metric_rows,
    build_dataset_quality_summary,
)


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    """写入小尺寸 RGB 图像夹具。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def file_digest(path: Path) -> str:
    """计算测试图像文件的 SHA-256 摘要。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def registry_rows(root_path: Path) -> list[dict[str, object]]:
    """构造两个真实 source / comparison 图像对。"""

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
            "source_image_digest": file_digest(source_a),
            "attacked_image_path": attacked_a.relative_to(root_path).as_posix(),
            "attacked_image_digest": file_digest(attacked_a),
        },
        {
            "attack_name": "sdedit_regeneration",
            "source_image_path": source_b.relative_to(root_path).as_posix(),
            "source_image_digest": file_digest(source_b),
            "attacked_image_path": attacked_b.relative_to(root_path).as_posix(),
            "attacked_image_digest": file_digest(attacked_b),
        },
    ]


def write_registry(root_path: Path, rows: list[dict[str, object]]) -> Path:
    """写出数据集质量入口注册表。"""

    path = root_path / "outputs" / "real_attack_evaluation" / "real_attacked_image_registry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.mark.quick
def test_dataset_quality_protocol_requires_formal_inception_features(tmp_path: Path) -> None:
    """缺少 Inception 特征时只能形成明确阻断, 不得计算替代指标。"""

    records = build_dataset_quality_image_records(registry_rows(tmp_path), tmp_path)
    metric_rows = build_dataset_quality_metric_rows(records, tmp_path)
    summary = build_dataset_quality_summary(records, metric_rows)
    rows_by_name = {row["quality_metric_name"]: row for row in metric_rows}

    assert len(records) == 2
    assert all(record.feature_backend == FORMAL_FEATURE_BACKEND for record in records)
    assert rows_by_name["fid"]["metric_status"] == FORMAL_FID_KID_BLOCKER
    assert rows_by_name["kid"]["metric_status"] == FORMAL_FID_KID_BLOCKER
    assert summary["feature_backend"] == FORMAL_FEATURE_BACKEND
    assert summary["formal_fid_kid_ready"] is False
    assert summary["formal_fid_kid_claim_blocker"] == FORMAL_FID_KID_BLOCKER
    assert all("proxy" not in key for key in summary)


@pytest.mark.quick
def test_dataset_quality_writer_outputs_only_formal_metrics(tmp_path: Path) -> None:
    """产物构建器不得写出非正式质量指标文件。"""

    write_registry(tmp_path, registry_rows(tmp_path))
    manifest = write_dataset_level_quality_outputs(root=tmp_path)
    output_dir = tmp_path / "outputs" / "dataset_level_quality"
    metric_rows = list(csv.DictReader((output_dir / "dataset_quality_metrics.csv").open(encoding="utf-8")))
    summary = json.loads((output_dir / "dataset_quality_summary.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "dataset_level_quality_manifest"
    assert {row["quality_metric_name"] for row in metric_rows} == {"fid", "kid"}
    assert not (output_dir / "dataset_quality_diagnostic_metrics.csv").exists()
    assert summary["dataset_quality_formal_metrics_path"] == "outputs/dataset_level_quality/dataset_quality_metrics.csv"
    assert summary["formal_fid_kid_ready"] is False
    assert all("proxy" not in key for key in summary)
    assert all(str(path).startswith("outputs/") for path in manifest["output_paths"])


@pytest.mark.quick
def test_dataset_quality_writer_materializes_required_images_from_package(tmp_path: Path) -> None:
    """图像只存在于结果包时应按摘要物化正式特征所需文件。"""

    source_path = tmp_path / "outputs" / "runtime" / "source.png"
    attacked_path = tmp_path / "outputs" / "runtime" / "attacked.png"
    write_image(source_path, (120, 10, 10))
    write_image(attacked_path, (100, 20, 20))
    source_digest = file_digest(source_path)
    attacked_digest = file_digest(attacked_path)
    package_path = tmp_path / "outputs" / "input_packages" / "runtime_package.zip"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package_path, "w") as archive:
        archive.write(source_path, source_path.relative_to(tmp_path).as_posix())
        archive.write(attacked_path, attacked_path.relative_to(tmp_path).as_posix())
    source_path.unlink()
    attacked_path.unlink()
    write_registry(
        tmp_path,
        [
            {
                "attack_name": "watermark_embedding",
                "source_image_path": "outputs/runtime/source.png",
                "source_image_digest": source_digest,
                "attacked_image_path": "outputs/runtime/attacked.png",
                "attacked_image_digest": attacked_digest,
            }
        ],
    )

    manifest = write_dataset_level_quality_outputs(root=tmp_path, input_package_paths=(package_path,))
    summary = json.loads(
        (tmp_path / "outputs" / "dataset_level_quality" / "dataset_quality_summary.json").read_text(encoding="utf-8")
    )
    assert summary["materialized_image_input_count"] == 2
    assert any("materialized_image_inputs" in path for path in manifest["output_paths"])


@pytest.mark.quick
def test_dataset_quality_materialization_caches_package_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """物化多张图像时应按包缓存 ZIP 摘要。"""

    entries = tuple(Path(f"outputs/images/image_{index}.png") for index in range(4))
    for index, entry in enumerate(entries):
        write_image(tmp_path / entry, (10 + index, 20 + index, 30 + index))
    records = build_dataset_quality_image_records(
        [
            {
                "attack_name": "jpeg_compression",
                "source_image_path": entries[0].as_posix(),
                "attacked_image_path": entries[1].as_posix(),
            },
            {
                "attack_name": "gaussian_blur",
                "source_image_path": entries[2].as_posix(),
                "attacked_image_path": entries[3].as_posix(),
            },
        ],
        tmp_path,
    )
    package_path = tmp_path / "outputs" / "input_packages" / "images.zip"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package_path, "w") as archive:
        for entry in entries:
            archive.write(tmp_path / entry, entry.as_posix())
    original = dataset_quality_writer.path_digest
    calls: list[str] = []

    def counting_digest(path: Path) -> str:
        """统计 ZIP 摘要计算次数并保持原始摘要语义。"""

        if path.suffix == ".zip":
            calls.append(path.name)
        return original(path)

    monkeypatch.setattr(dataset_quality_writer, "path_digest", counting_digest)
    materialized = dataset_quality_writer.materialize_images_from_input_packages(
        records=records,
        materialized_root=tmp_path / "outputs" / "dataset_level_quality" / "materialized_image_inputs",
        input_package_paths=(package_path,),
    )
    assert len(materialized) == 4
    assert calls == [package_path.name]


@pytest.mark.quick
def test_dataset_quality_formal_feature_import_keeps_small_sample_blocked(tmp_path: Path) -> None:
    """Inception 特征样本数不足时不得形成正式 FID / KID 结论。"""

    rows = registry_rows(tmp_path)
    write_registry(tmp_path, rows)
    records = build_dataset_quality_image_records(rows, tmp_path)
    feature_path = tmp_path / "outputs" / "dataset_level_quality" / "dataset_quality_formal_feature_records.jsonl"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_rows = []
    for index, record in enumerate(records):
        for role, vector in (("source", [float(index), 0.0, 1.0]), ("comparison", [float(index), 0.2, 0.8])):
            feature_rows.append(
                {
                    "dataset_quality_record_id": record.dataset_quality_record_id,
                    "dataset_quality_image_role": role,
                    "feature_backend": FORMAL_FEATURE_BACKEND,
                    "feature_extractor_id": dataset_quality_writer.FORMAL_FEATURE_EXTRACTOR_ID,
                    "feature_vector": vector,
                }
            )
    feature_path.write_text("".join(json.dumps(row) + "\n" for row in feature_rows), encoding="utf-8")

    write_dataset_level_quality_outputs(
        root=tmp_path,
        formal_feature_records_path=feature_path,
        formal_min_sample_count=10,
    )
    output_dir = tmp_path / "outputs" / "dataset_level_quality"
    metric_rows = list(csv.DictReader((output_dir / "dataset_quality_metrics.csv").open(encoding="utf-8")))
    summary = json.loads((output_dir / "dataset_quality_summary.json").read_text(encoding="utf-8"))

    assert {row["metric_status"] for row in metric_rows} == {FORMAL_FID_KID_SAMPLE_BLOCKER}
    assert summary["formal_feature_backend_ready"] is True
    assert summary["formal_sample_scale_ready"] is False
    assert summary["formal_fid_kid_claim_gate_ready"] is False


@pytest.mark.quick
def test_dataset_quality_formal_features_measure_fid_and_kid_at_required_scale(tmp_path: Path) -> None:
    """样本规模达到门槛时应从正式特征计算 FID 与 KID。"""

    records = build_dataset_quality_image_records(registry_rows(tmp_path), tmp_path)
    metric_rows = build_dataset_quality_metric_rows(
        records,
        tmp_path,
        formal_source_features=[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        formal_comparison_features=[[0.1, 0.0, 0.9], [0.9, 0.1, 0.0]],
        formal_min_sample_count=2,
    )
    summary = build_dataset_quality_summary(records, metric_rows)
    rows_by_name = {row["quality_metric_name"]: row for row in metric_rows}

    assert rows_by_name["fid"]["metric_status"] == "measured"
    assert rows_by_name["kid"]["metric_status"] == "measured"
    assert isinstance(rows_by_name["fid"]["quality_metric_value"], float)
    assert isinstance(rows_by_name["kid"]["quality_metric_value"], float)
    assert summary["formal_fid_kid_ready"] is True
    assert summary["feature_backend"] == FORMAL_FEATURE_BACKEND
    assert all("proxy" not in key for key in summary)
