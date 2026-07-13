"""数据集级正式 Inception FID / KID 证据链路测试。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

from PIL import Image
import pytest

import experiments.artifacts.dataset_level_quality_outputs as dataset_quality_writer
from experiments.runtime import repository_environment
from experiments.runtime.scientific_unit_provenance import (
    build_scientific_unit_provenance,
)
from experiments.artifacts.dataset_level_quality_outputs import write_dataset_level_quality_outputs
from experiments.protocol import (
    FORMAL_DATASET_QUALITY_METRIC_NAMES,
    FORMAL_FEATURE_BACKEND,
    FORMAL_FID_KID_BLOCKER,
    build_dataset_quality_image_records,
    build_dataset_quality_metric_rows,
    build_dataset_quality_summary,
)
from experiments.protocol.prompts import PROMPT_FILES, build_prompt_records, read_prompt_file
from tests.helpers.formal_execution_lock import build_test_formal_execution_lock

PAPER_RUN_NAME = "probe_paper"
TARGET_FPR = 0.1
PROMPT_COUNT = 70
FORMAL_EXECUTION_LOCK = build_test_formal_execution_lock()
TEST_PROFILE_DIGEST = "1" * 64
TEST_COMPLETE_HASH_LOCK_DIGEST = "3" * 64


class _FakeCuda:
    """提供正式特征来源夹具需要的最小 CUDA 接口."""

    def is_available(self) -> bool:
        return True

    def current_device(self) -> int:
        return 0

    def device_count(self) -> int:
        return 1

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return "NVIDIA T4"

    def get_device_capability(self, index: int) -> tuple[int, int]:
        assert index == 0
        return (7, 5)


class _FakeTorch:
    """固定测试 PyTorch 和 CUDA build 身份."""

    __version__ = "2.7.1+cu128"
    version = SimpleNamespace(cuda="12.8")
    cuda = _FakeCuda()


def build_feature_provenance(
    unit_id: str,
    config_digest: str,
) -> dict[str, object]:
    """构造字段完整的 GPU 特征完成单元来源记录."""

    return build_scientific_unit_provenance(
        scientific_unit_id=unit_id,
        scientific_unit_config_digest=config_digest,
        runtime_environment={
            "dependency_environment_ready": True,
            "formal_execution_lock_ready": True,
            "isolated_scientific_context_ready": True,
            "dependency_profile_id": "sd35_method_runtime_gpu",
            "dependency_profile_digest": TEST_PROFILE_DIGEST,
            "direct_requirements_digest": "2" * 64,
            "complete_hash_lock_digest": TEST_COMPLETE_HASH_LOCK_DIGEST,
            "formal_execution_commit": FORMAL_EXECUTION_LOCK[
                "formal_execution_commit"
            ],
            "formal_execution_lock_digest": FORMAL_EXECUTION_LOCK[
                "formal_execution_lock_digest"
            ],
            "python_version": "3.12.11",
            "package_versions": {"torch": "2.7.1+cu128"},
            "cuda_version": "12.8",
            "device_count": 1,
            "gpu_name": "NVIDIA T4",
            "isolated_scientific_context": {
                "dependency_environment_report_actual_digest": "6" * 64,
                "current_python_executable_sha256": "7" * 64,
            },
        },
        execution_device_name="cuda:0",
        torch_module=_FakeTorch(),
        random_identity_random={
            "feature_extraction_seed_random": "not_used_deterministic_eval"
        },
    )


@pytest.fixture(autouse=True)
def configure_probe_paper_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """让 writer 测试始终使用 probe_paper 的冻结运行身份。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", PAPER_RUN_NAME)
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)
    monkeypatch.delenv("SLM_WM_RESUME_CHECKPOINT_DIR", raising=False)
    monkeypatch.setattr(
        repository_environment,
        "require_published_formal_execution_lock",
        lambda _root: dict(FORMAL_EXECUTION_LOCK),
    )
    monkeypatch.setattr(
        dataset_quality_writer,
        "require_dependency_profile_ready",
        lambda *_args, **_kwargs: SimpleNamespace(
            profile_digest=TEST_PROFILE_DIGEST,
            complete_hash_lock_digest=TEST_COMPLETE_HASH_LOCK_DIGEST,
        ),
    )


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    """写入小尺寸 RGB 图像夹具。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def file_digest(path: Path) -> str:
    """计算测试图像文件的 SHA-256 摘要。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def formal_feature_vector(index: int, role: str) -> list[float]:
    """构造维度正确且避免退化秩的确定性测试特征."""

    role_offset = 1 if role == "source" else PROMPT_COUNT + 1
    multiplier = index + role_offset
    return [
        float((multiplier * (dimension + 1) + 17 * dimension * dimension) % 10007)
        / 10007.0
        for dimension in range(2048)
    ]


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
            "run_id": "quality_run_a",
            "prompt_id": "quality_prompt_a",
            "attack_name": "watermark_embedding",
            "image_pair_role": "clean_to_watermarked",
            "source_image_path": source_a.relative_to(root_path).as_posix(),
            "source_image_digest": file_digest(source_a),
            "attacked_image_path": attacked_a.relative_to(root_path).as_posix(),
            "attacked_image_digest": file_digest(attacked_a),
            "supports_paper_claim": False,
        },
        {
            "run_id": "quality_run_b",
            "prompt_id": "quality_prompt_b",
            "attack_name": "watermark_embedding",
            "image_pair_role": "clean_to_watermarked",
            "source_image_path": source_b.relative_to(root_path).as_posix(),
            "source_image_digest": file_digest(source_b),
            "attacked_image_path": attacked_b.relative_to(root_path).as_posix(),
            "attacked_image_digest": file_digest(attacked_b),
            "supports_paper_claim": False,
        },
    ]


def canonical_prompt_ids() -> tuple[str, ...]:
    """读取 probe_paper 当前冻结 Prompt 集的稳定标识。"""

    prompt_path = Path(__file__).resolve().parents[2] / PROMPT_FILES[PAPER_RUN_NAME]
    return tuple(
        record.prompt_id
        for record in build_prompt_records(
            PAPER_RUN_NAME,
            read_prompt_file(prompt_path),
        )
    )


def canonical_registry_rows(root_path: Path) -> list[dict[str, object]]:
    """构造恰好一条图像对对应一个受治理 Prompt 的 registry。"""

    rows: list[dict[str, object]] = []
    for index, prompt_id in enumerate(canonical_prompt_ids()):
        source_path = (
            root_path
            / "outputs/images/canonical_quality"
            / f"clean_{index:05d}.png"
        )
        comparison_path = (
            root_path
            / "outputs/images/canonical_quality"
            / f"watermarked_{index:05d}.png"
        )
        write_image(
            source_path,
            ((index * 17) % 256, (index * 29) % 256, (index * 43) % 256),
        )
        write_image(
            comparison_path,
            (
                (index * 17 + 3) % 256,
                (index * 29 + 5) % 256,
                (index * 43 + 7) % 256,
            ),
        )
        rows.append(
            {
                "run_id": f"quality_run_{index:05d}",
                "prompt_id": prompt_id,
                "attack_name": "watermark_embedding",
                "image_pair_role": "clean_to_watermarked",
                "source_image_path": source_path.relative_to(
                    root_path
                ).as_posix(),
                "source_image_digest": file_digest(source_path),
                "attacked_image_path": comparison_path.relative_to(
                    root_path
                ).as_posix(),
                "attacked_image_digest": file_digest(comparison_path),
                "supports_paper_claim": False,
            }
        )
    return rows


def write_registry(root_path: Path, rows: list[dict[str, object]]) -> Path:
    """写出数据集质量入口注册表。"""

    path = (
        root_path
        / "outputs"
        / "image_only_dataset_runtime"
        / PAPER_RUN_NAME
        / "watermark_quality_image_registry.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def formal_feature_import_fixture(
    root_path: Path,
) -> tuple[
    tuple[object, ...],
    tuple[dict[str, object], ...],
    list[dict[str, object]],
]:
    """构造一对具有实际图像解析身份和科学来源的正式特征行."""

    records = build_dataset_quality_image_records(
        registry_rows(root_path)[:1],
        root_path,
    )
    resolution_records = (
        dataset_quality_writer.build_image_resolution_records(
            records=records,
            root_path=root_path,
            image_search_roots=(),
            materialized_root=(
                root_path / "outputs/dataset_level_quality/materialized"
            ),
            materialized_records=(),
        )
    )
    rows: list[dict[str, object]] = []
    for role, value in (("source", 0.0), ("comparison", 0.25)):
        record = records[0]
        row = {
            "dataset_quality_record_id": record.dataset_quality_record_id,
            "dataset_quality_image_role": role,
            "feature_backend": FORMAL_FEATURE_BACKEND,
            "feature_extractor_id": (
                dataset_quality_writer.FORMAL_FEATURE_EXTRACTOR_ID
            ),
            "feature_dimension": 2048,
            "image_path": getattr(record, f"{role}_image_path"),
            "image_digest": getattr(record, f"{role}_image_digest"),
            "feature_vector": [value] * 2048,
            "supports_paper_claim": False,
        }
        item_identity = [
            {
                field_name: row[field_name]
                for field_name in (
                    "dataset_quality_record_id",
                    "dataset_quality_image_role",
                    "image_path",
                    "image_digest",
                )
            }
        ]
        unit_id = (
            "feature_batch_"
            + dataset_quality_writer.build_stable_digest(
                [
                    (
                        row["dataset_quality_record_id"],
                        row["dataset_quality_image_role"],
                    )
                ]
            )[:16]
        )
        row["scientific_unit_provenance"] = build_feature_provenance(
            unit_id,
            dataset_quality_writer._inception_batch_config_digest(
                item_identity
            ),
        )
        rows.append(row)
    return records, resolution_records, rows


@pytest.mark.quick
@pytest.mark.parametrize(
    ("mutation_id", "expected_issue_field"),
    (
        ("feature_extractor_id", "feature_extractor_id"),
        ("declared_feature_dimension", "feature_dimension"),
        ("feature_vector_length", "feature_dimension"),
        ("supports_paper_claim", "supports_paper_claim"),
        ("resolved_image_path", "image_digest"),
    ),
)
def test_formal_feature_import_rejects_resigned_row_schema_drift(
    tmp_path: Path,
    mutation_id: str,
    expected_issue_field: str,
) -> None:
    """逐行字段即使同步重签科学来源也不得进入正式 FID/KID."""

    records, resolution_records, rows = formal_feature_import_fixture(
        tmp_path
    )
    forged = dict(rows[0])
    if mutation_id == "feature_extractor_id":
        forged["feature_extractor_id"] = "forged_extractor"
    elif mutation_id == "declared_feature_dimension":
        forged["feature_dimension"] = 1024
    elif mutation_id == "feature_vector_length":
        forged["feature_vector"] = [0.0] * 2047
    elif mutation_id == "supports_paper_claim":
        forged["supports_paper_claim"] = True
    elif mutation_id == "resolved_image_path":
        forged["image_path"] = "outputs/images/forged.png"
    else:  # pragma: no cover - 参数集合由测试本身冻结.
        raise AssertionError("未知测试变体")
    item_identity = [
        {
            field_name: forged[field_name]
            for field_name in (
                "dataset_quality_record_id",
                "dataset_quality_image_role",
                "image_path",
                "image_digest",
            )
        }
    ]
    forged["scientific_unit_provenance"] = build_feature_provenance(
        str(forged["scientific_unit_provenance"]["scientific_unit_id"]),
        dataset_quality_writer._inception_batch_config_digest(item_identity),
    )
    rows[0] = forged

    payload = dataset_quality_writer.build_formal_feature_import_payload(
        records=records,
        image_resolution_records=resolution_records,
        feature_rows=rows,
        formal_feature_records_path=tmp_path / "features.jsonl",
        root_path=tmp_path,
        formal_min_sample_count=1,
        formal_feature_records_sha256="f" * 64,
    )

    assert payload["report"]["formal_feature_backend_ready"] is False
    assert expected_issue_field in {
        issue["field_name"] for issue in payload["report"]["issues"]
    }


@pytest.mark.quick
def test_formal_feature_import_rehashes_actual_resolution_image(
    tmp_path: Path,
) -> None:
    """解析记录生成后实际图像字节漂移必须在特征导入层阻断."""

    records, resolution_records, rows = formal_feature_import_fixture(
        tmp_path
    )
    source_path = tmp_path / records[0].source_image_path
    source_path.write_bytes(b"forged-image-bytes")

    payload = dataset_quality_writer.build_formal_feature_import_payload(
        records=records,
        image_resolution_records=resolution_records,
        feature_rows=rows,
        formal_feature_records_path=tmp_path / "features.jsonl",
        root_path=tmp_path,
        formal_min_sample_count=1,
        formal_feature_records_sha256="f" * 64,
    )

    assert payload["report"]["image_resolution_identity_ready"] is False
    assert payload["report"]["formal_feature_backend_ready"] is False


@pytest.mark.quick
@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    (
        ("attack_name", "jpeg_compression"),
        ("image_pair_role", "clean_to_attacked"),
        ("supports_paper_claim", True),
    ),
)
def test_dataset_quality_records_require_clean_to_watermarked_semantics(
    tmp_path: Path,
    field_name: str,
    forged_value: object,
) -> None:
    """真实路径和 SHA 有效也不能把其他攻击对冒充 clean/watermarked."""

    rows = registry_rows(tmp_path)
    rows[0] = {**rows[0], field_name: forged_value}

    with pytest.raises(ValueError, match="clean-to-watermarked"):
        build_dataset_quality_image_records(rows, tmp_path)


@pytest.mark.quick
def test_dataset_quality_records_reject_resolved_path_alias_reuse(
    tmp_path: Path,
) -> None:
    """不同文本路径解析到同一文件时不得重复计入正式样本规模."""

    rows = registry_rows(tmp_path)
    source_path = str(rows[0]["source_image_path"])
    rows[1] = {
        **rows[1],
        "source_image_path": (
            f"{Path(source_path).parent.as_posix()}/./"
            f"{Path(source_path).name}"
        ),
        "source_image_digest": rows[0]["source_image_digest"],
    }

    with pytest.raises(ValueError, match="实际文件路径"):
        build_dataset_quality_image_records(rows, tmp_path)


def write_inception_checkpoint_fixture(
    root_path: Path,
) -> tuple[SimpleNamespace, Path, Path, list[dict[str, object]]]:
    """写出无需加载 GPU 模型即可验证恢复路径的完整特征 shard."""

    source_path = root_path / "outputs" / "images" / "checkpoint_source.png"
    comparison_path = (
        root_path / "outputs" / "images" / "checkpoint_comparison.png"
    )
    write_image(source_path, (40, 50, 60))
    write_image(comparison_path, (45, 55, 65))
    record = SimpleNamespace(
        dataset_quality_record_id="quality_record_0001",
        source_image_path=source_path.relative_to(root_path).as_posix(),
        source_image_digest=file_digest(source_path),
        comparison_image_path=comparison_path.relative_to(root_path).as_posix(),
        comparison_image_digest=file_digest(comparison_path),
    )
    output_dir = (
        root_path / "outputs" / "dataset_level_quality" / PAPER_RUN_NAME
    )
    checkpoint_dir = output_dir / "inception_feature_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset_quality_formal_feature_records.jsonl"
    identities = [
        {
            "dataset_quality_record_id": record.dataset_quality_record_id,
            "dataset_quality_image_role": image_role,
            "image_path": image_path.relative_to(root_path).as_posix(),
            "image_digest": image_digest,
        }
        for image_role, image_path, image_digest in (
            ("source", source_path, record.source_image_digest),
            ("comparison", comparison_path, record.comparison_image_digest),
        )
    ]
    context_path = checkpoint_dir / "feature_checkpoint_context.json"
    context_path.write_text(
        json.dumps(
            {
                "report_schema": "inception_feature_checkpoint_context",
                "schema_version": 1,
                "feature_backend": FORMAL_FEATURE_BACKEND,
                "feature_extractor_id": (
                    dataset_quality_writer.FORMAL_FEATURE_EXTRACTOR_ID
                ),
                "item_count": len(identities),
                "item_identity_digest": (
                    dataset_quality_writer.build_stable_digest(identities)
                ),
                "formal_execution_lock": FORMAL_EXECUTION_LOCK,
                "evidence_eligibility": "intermediate_state_only",
                "supports_paper_claim": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = [
        {
            **identity,
            "feature_backend": FORMAL_FEATURE_BACKEND,
            "feature_extractor_id": (
                dataset_quality_writer.FORMAL_FEATURE_EXTRACTOR_ID
            ),
            "feature_dimension": 2048,
            "feature_vector": [float(index)] * 2048,
            "supports_paper_claim": False,
        }
        for index, identity in enumerate(identities)
    ]
    shard_unit_id = "feature_batch_" + dataset_quality_writer.build_stable_digest(
        [
            (
                identity["dataset_quality_record_id"],
                identity["dataset_quality_image_role"],
            )
            for identity in identities
        ]
    )[:16]
    shard_provenance = build_feature_provenance(
        shard_unit_id,
        dataset_quality_writer._inception_batch_config_digest(identities),
    )
    rows = [
        {**row, "scientific_unit_provenance": shard_provenance}
        for row in rows
    ]
    (checkpoint_dir / f"{shard_unit_id}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return record, output_path, context_path, rows


@pytest.mark.quick
def test_inception_checkpoint_rejects_declared_image_digest_drift(
    tmp_path: Path,
) -> None:
    """图像文件字节与记录摘要不一致时不得复用或生成正式特征."""

    record, output_path, _, _ = write_inception_checkpoint_fixture(tmp_path)
    record.source_image_digest = "0" * 64

    with pytest.raises(RuntimeError, match="图像摘要与实际文件不一致"):
        dataset_quality_writer.extract_formal_inception_feature_rows(
            records=(record,),
            root_path=tmp_path,
            image_search_roots=(),
            output_path=output_path,
        )


@pytest.mark.quick
def test_inception_checkpoint_rejects_context_identity_drift(
    tmp_path: Path,
) -> None:
    """Prompt 图像集合或执行锁身份变化后不得消费旧特征 shard."""

    record, output_path, context_path, _ = write_inception_checkpoint_fixture(
        tmp_path
    )
    context = json.loads(context_path.read_text(encoding="utf-8"))
    context["item_identity_digest"] = "f" * 64
    context_path.write_text(
        json.dumps(context, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="检查点身份与当前运行不一致"):
        dataset_quality_writer.extract_formal_inception_feature_rows(
            records=(record,),
            root_path=tmp_path,
            image_search_roots=(),
            output_path=output_path,
        )


@pytest.mark.quick
def test_inception_checkpoint_rejects_conflicting_shards(tmp_path: Path) -> None:
    """两个 shard 对同一图像声明不同特征时必须停止而非覆盖."""

    record, output_path, context_path, rows = (
        write_inception_checkpoint_fixture(tmp_path)
    )
    conflicting_row = dict(rows[0])
    conflicting_row["feature_vector"] = [9.0] * 2048
    conflicting_identity = [
        {
            field_name: conflicting_row[field_name]
            for field_name in (
                "dataset_quality_record_id",
                "dataset_quality_image_role",
                "image_path",
                "image_digest",
            )
        }
    ]
    conflicting_unit_id = (
        "feature_batch_"
        + dataset_quality_writer.build_stable_digest(
            [
                (
                    conflicting_identity[0]["dataset_quality_record_id"],
                    conflicting_identity[0]["dataset_quality_image_role"],
                )
            ]
        )[:16]
    )
    conflicting_row["scientific_unit_provenance"] = build_feature_provenance(
        conflicting_unit_id,
        dataset_quality_writer._inception_batch_config_digest(
            conflicting_identity
        ),
    )
    (context_path.parent / f"{conflicting_unit_id}.jsonl").write_text(
        json.dumps(conflicting_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="包含冲突记录"):
        dataset_quality_writer.extract_formal_inception_feature_rows(
            records=(record,),
            root_path=tmp_path,
            image_search_roots=(),
            output_path=output_path,
        )


@pytest.mark.quick
@pytest.mark.parametrize("invalid_value", ["1.0", True, float("inf")])
def test_inception_checkpoint_rejects_nonfinite_or_nonnumeric_features(
    tmp_path: Path,
    invalid_value: object,
) -> None:
    """恢复的2048维特征必须全部是有限数值且不得接受 bool 或字符串."""

    record, output_path, context_path, rows = (
        write_inception_checkpoint_fixture(tmp_path)
    )
    invalid_row = dict(rows[0])
    invalid_vector = list(invalid_row["feature_vector"])
    invalid_vector[0] = invalid_value
    invalid_row["feature_vector"] = invalid_vector
    shard_path = context_path.parent / "feature_batch_complete.jsonl"
    shard_path.write_text(
        json.dumps(invalid_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="内容未通过身份校验"):
        dataset_quality_writer.extract_formal_inception_feature_rows(
            records=(record,),
            root_path=tmp_path,
            image_search_roots=(),
            output_path=output_path,
        )


@pytest.mark.quick
def test_inception_checkpoint_completion_writes_canonical_rows_and_clears_progress(
    tmp_path: Path,
) -> None:
    """完整 shard 恢复只生成规范特征文件, 并清除中间 progress 入口."""

    record, output_path, _, expected_rows = write_inception_checkpoint_fixture(
        tmp_path
    )
    progress_path = output_path.parent / "inception_feature_progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "protocol_decision": "resume_required",
                "evidence_eligibility": "intermediate_state_only",
                "supports_paper_claim": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = dataset_quality_writer.extract_formal_inception_feature_rows(
        records=(record,),
        root_path=tmp_path,
        image_search_roots=(),
        output_path=output_path,
    )

    assert rows == expected_rows
    assert output_path.is_file()
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 2
    assert not progress_path.exists()
    assert not list(output_path.parent.rglob("*.partial"))


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
    assert rows_by_name["kid_mean"]["metric_status"] == FORMAL_FID_KID_BLOCKER
    assert rows_by_name["kid_std"]["metric_status"] == FORMAL_FID_KID_BLOCKER
    assert summary["feature_backend"] == FORMAL_FEATURE_BACKEND
    assert summary["formal_fid_kid_ready"] is False
    assert summary["formal_fid_kid_component_blocker"] == FORMAL_FID_KID_BLOCKER
    assert all("proxy" not in key for key in summary)


@pytest.mark.quick
def test_dataset_quality_writer_outputs_only_formal_metrics(tmp_path: Path) -> None:
    """产物构建器不得写出非正式质量指标文件。"""

    write_registry(tmp_path, canonical_registry_rows(tmp_path))
    manifest = write_dataset_level_quality_outputs(
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        root=tmp_path,
    )
    output_dir = tmp_path / "outputs" / "dataset_level_quality" / PAPER_RUN_NAME
    metric_rows = list(csv.DictReader((output_dir / "dataset_quality_metrics.csv").open(encoding="utf-8")))
    summary = json.loads((output_dir / "dataset_quality_summary.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "dataset_level_quality_manifest"
    assert [row["quality_metric_name"] for row in metric_rows] == list(
        FORMAL_DATASET_QUALITY_METRIC_NAMES
    )
    assert not (output_dir / "dataset_quality_diagnostic_metrics.csv").exists()
    assert summary["dataset_quality_formal_metrics_path"] == (
        "outputs/dataset_level_quality/probe_paper/dataset_quality_metrics.csv"
    )
    assert summary["paper_run_name"] == PAPER_RUN_NAME
    assert summary["target_fpr"] == TARGET_FPR
    assert summary["formal_fid_kid_ready"] is False
    assert all("proxy" not in key for key in summary)
    assert all(str(path).startswith("outputs/") for path in manifest["output_paths"])


@pytest.mark.quick
def test_dataset_quality_writer_rejects_partial_run_sample_floor(tmp_path: Path) -> None:
    """正式 FID/KID 样本门槛不得低于当前论文层级完整 Prompt 数量。"""

    with pytest.raises(ValueError, match="完整 Prompt 数量"):
        write_dataset_level_quality_outputs(
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            root=tmp_path,
            formal_min_sample_count=69,
        )


@pytest.mark.quick
def test_dataset_quality_writer_rejects_noncanonical_prompt_registry(tmp_path: Path) -> None:
    """registry 缺少任一当前 Prompt 或包含重复 Prompt 时必须立即拒绝。"""

    rows = canonical_registry_rows(tmp_path)
    rows[-1] = {**rows[-1], "prompt_id": rows[0]["prompt_id"]}
    write_registry(tmp_path, rows)

    with pytest.raises(ValueError, match="精确覆盖"):
        write_dataset_level_quality_outputs(
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            root=tmp_path,
        )


@pytest.mark.quick
def test_dataset_quality_writer_copies_external_features_and_binds_exact_coverage(
    tmp_path: Path,
) -> None:
    """外部正式特征必须复制到 run 目录并覆盖全部70个 Prompt 图像对。"""

    rows = canonical_registry_rows(tmp_path)
    write_registry(tmp_path, rows)
    records = build_dataset_quality_image_records(rows, tmp_path)
    external_feature_path = tmp_path / "outputs/imported_features/features.jsonl"
    external_feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_rows = [
        {
            "dataset_quality_record_id": record.dataset_quality_record_id,
            "dataset_quality_image_role": role,
            "feature_backend": FORMAL_FEATURE_BACKEND,
            "feature_extractor_id": dataset_quality_writer.FORMAL_FEATURE_EXTRACTOR_ID,
            "feature_dimension": 2048,
            "feature_vector": formal_feature_vector(index, role),
            "image_path": (
                record.source_image_path
                if role == "source"
                else record.comparison_image_path
            ),
            "image_digest": (
                record.source_image_digest
                if role == "source"
                else record.comparison_image_digest
            ),
            "supports_paper_claim": False,
        }
        for index, record in enumerate(records)
        for role in ("source", "comparison")
    ]
    feature_rows = [
        {
            **row,
            "scientific_unit_provenance": build_feature_provenance(
                "feature_batch_"
                + dataset_quality_writer.build_stable_digest(
                    [
                        (
                            row["dataset_quality_record_id"],
                            row["dataset_quality_image_role"],
                        )
                    ]
                )[:16],
                dataset_quality_writer._inception_batch_config_digest(
                    [
                        {
                            field_name: row[field_name]
                            for field_name in (
                                "dataset_quality_record_id",
                                "dataset_quality_image_role",
                                "image_path",
                                "image_digest",
                            )
                        }
                    ]
                ),
            ),
        }
        for row in feature_rows
    ]
    external_feature_path.write_text(
        "".join(json.dumps(row) + "\n" for row in feature_rows),
        encoding="utf-8",
    )

    manifest = write_dataset_level_quality_outputs(
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        root=tmp_path,
        formal_feature_records_path=external_feature_path,
    )
    output_dir = tmp_path / "outputs/dataset_level_quality" / PAPER_RUN_NAME
    canonical_feature_path = output_dir / "dataset_quality_formal_feature_records.jsonl"
    summary = json.loads((output_dir / "dataset_quality_summary.json").read_text(encoding="utf-8"))
    report = json.loads(
        (output_dir / "dataset_quality_formal_feature_import_report.json").read_text(
            encoding="utf-8"
        )
    )
    metric_rows = list(
        csv.DictReader((output_dir / "dataset_quality_metrics.csv").open(encoding="utf-8"))
    )

    assert canonical_feature_path.is_file()
    assert len(canonical_feature_path.read_text(encoding="utf-8").splitlines()) == PROMPT_COUNT * 2
    assert summary["prompt_registry_exact_set_ready"] is True
    assert summary["accepted_feature_pair_count"] == PROMPT_COUNT
    assert summary["missing_feature_pair_count"] == 0
    assert summary["feature_issue_count"] == 0
    assert summary["formal_feature_record_count"] == PROMPT_COUNT * 2
    assert summary["kid_effective_subset_size"] == PROMPT_COUNT
    assert report["formal_feature_records_sha256"] == summary["formal_feature_records_sha256"]
    assert [row["quality_metric_name"] for row in metric_rows] == list(
        FORMAL_DATASET_QUALITY_METRIC_NAMES
    )
    assert {row["metric_status"] for row in metric_rows} == {"measured"}
    assert all(int(row["sample_pair_count"]) == PROMPT_COUNT for row in metric_rows)
    assert canonical_feature_path.relative_to(tmp_path).as_posix() in manifest["output_paths"]
    assert external_feature_path.relative_to(tmp_path).as_posix() in manifest["input_paths"]


@pytest.mark.quick
def test_dataset_quality_writer_materializes_required_images_from_package(tmp_path: Path) -> None:
    """图像只存在于结果包时应按摘要物化正式特征所需文件。"""

    package_path = tmp_path / "outputs" / "input_packages" / "runtime_package.zip"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    registry: list[dict[str, object]] = []
    with zipfile.ZipFile(package_path, "w") as archive:
        for index, prompt_id in enumerate(canonical_prompt_ids()):
            source_path = (
                tmp_path / "outputs/runtime" / f"clean_{index:05d}.png"
            )
            comparison_path = (
                tmp_path
                / "outputs/runtime"
                / f"watermarked_{index:05d}.png"
            )
            write_image(source_path, ((index * 3) % 256, 10, 20))
            write_image(comparison_path, ((index * 3 + 1) % 256, 12, 22))
            source_digest = file_digest(source_path)
            comparison_digest = file_digest(comparison_path)
            archive.write(
                source_path,
                source_path.relative_to(tmp_path).as_posix(),
            )
            archive.write(
                comparison_path,
                comparison_path.relative_to(tmp_path).as_posix(),
            )
            registry.append(
                {
                    "run_id": f"materialized_run_{index:05d}",
                    "prompt_id": prompt_id,
                    "attack_name": "watermark_embedding",
                    "image_pair_role": "clean_to_watermarked",
                    "source_image_path": source_path.relative_to(
                        tmp_path
                    ).as_posix(),
                    "source_image_digest": source_digest,
                    "attacked_image_path": comparison_path.relative_to(
                        tmp_path
                    ).as_posix(),
                    "attacked_image_digest": comparison_digest,
                    "supports_paper_claim": False,
                }
            )
            source_path.unlink()
            comparison_path.unlink()
    write_registry(tmp_path, registry)

    manifest = write_dataset_level_quality_outputs(
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        root=tmp_path,
        input_package_paths=(package_path,),
    )
    summary = json.loads(
        (
            tmp_path
            / "outputs"
            / "dataset_level_quality"
            / PAPER_RUN_NAME
            / "dataset_quality_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["materialized_image_input_count"] == PROMPT_COUNT * 2
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
                "run_id": "materialization_run_0",
                "prompt_id": "materialization_prompt_0",
                "attack_name": "watermark_embedding",
                "image_pair_role": "clean_to_watermarked",
                "source_image_path": entries[0].as_posix(),
                "attacked_image_path": entries[1].as_posix(),
                "supports_paper_claim": False,
            },
            {
                "run_id": "materialization_run_1",
                "prompt_id": "materialization_prompt_1",
                "attack_name": "watermark_embedding",
                "image_pair_role": "clean_to_watermarked",
                "source_image_path": entries[2].as_posix(),
                "attacked_image_path": entries[3].as_posix(),
                "supports_paper_claim": False,
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
def test_dataset_quality_formal_feature_import_keeps_incomplete_coverage_blocked(tmp_path: Path) -> None:
    """Inception 特征未覆盖全部 Prompt 时不得形成正式 FID / KID 结论。"""

    rows = canonical_registry_rows(tmp_path)
    write_registry(tmp_path, rows)
    records = build_dataset_quality_image_records(rows, tmp_path)
    feature_path = tmp_path / "outputs" / "dataset_level_quality" / "dataset_quality_formal_feature_records.jsonl"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_rows = []
    for index, record in enumerate(records[:2]):
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
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        root=tmp_path,
        formal_feature_records_path=feature_path,
        formal_min_sample_count=PROMPT_COUNT,
    )
    output_dir = tmp_path / "outputs" / "dataset_level_quality" / PAPER_RUN_NAME
    metric_rows = list(csv.DictReader((output_dir / "dataset_quality_metrics.csv").open(encoding="utf-8")))
    summary = json.loads((output_dir / "dataset_quality_summary.json").read_text(encoding="utf-8"))

    assert {row["metric_status"] for row in metric_rows} == {FORMAL_FID_KID_BLOCKER}
    assert summary["formal_feature_backend_ready"] is False
    assert summary["formal_sample_scale_ready"] is False
    assert summary["formal_fid_kid_component_ready"] is False
    assert summary["accepted_feature_pair_count"] == 0
    assert summary["missing_feature_pair_count"] == PROMPT_COUNT
    assert summary["feature_issue_count"] == 5


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
    assert rows_by_name["kid_mean"]["metric_status"] == "measured"
    assert rows_by_name["kid_std"]["metric_status"] == "measured"
    assert isinstance(rows_by_name["fid"]["quality_metric_value"], float)
    assert isinstance(rows_by_name["kid_mean"]["quality_metric_value"], float)
    assert rows_by_name["kid_std"]["quality_metric_value"] == 0.0
    assert summary["formal_fid_kid_ready"] is True
    assert summary["formal_quality_metric_count"] == 3
    assert summary["kid_effective_subset_size"] == 2
    assert summary["feature_backend"] == FORMAL_FEATURE_BACKEND
    assert all("proxy" not in key for key in summary)
