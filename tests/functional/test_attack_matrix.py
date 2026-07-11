"""验证攻击矩阵只消费真实 attacked image 与冻结后的仅图像检测记录。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
import pytest

from experiments.protocol.attacks import attack_config_digest, default_attack_configs
from experiments.runtime.repository_environment import file_digest
from scripts.write_attack_matrix_outputs import write_attack_matrix_outputs


def _json_line(value: dict[str, object]) -> str:
    """把测试记录写成 JSONL 单行。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def _write_runtime_fixture(root: Path, *, attack_prompt_count: int = 1, blind_detector: bool = True) -> Path:
    """写出最小但结构完整的真实攻击运行夹具。"""

    runtime_dir = root / "outputs" / "image_only_dataset_runtime" / "probe_paper"
    image_dir = runtime_dir / "runs" / "fixture"
    image_dir.mkdir(parents=True)
    clean_path = image_dir / "clean.png"
    watermarked_path = image_dir / "watermarked.png"
    attacked_clean_path = image_dir / "attacked_clean.png"
    attacked_positive_path = image_dir / "attacked_positive.png"
    Image.new("RGB", (8, 8), (20, 30, 40)).save(clean_path)
    Image.new("RGB", (8, 8), (190, 120, 70)).save(watermarked_path)
    Image.new("RGB", (8, 8), (24, 34, 44)).save(attacked_clean_path)
    Image.new("RGB", (8, 8), (184, 116, 72)).save(attacked_positive_path)

    metadata = {
        "detector_input_access_mode": "image_key_public_model_only",
        "blind_image_detector": blind_detector,
        "generation_latent_trace_required": False,
    }
    records: list[dict[str, object]] = []
    for sample_role, source_path, content_score in (
        ("clean_negative", clean_path, 0.12),
        ("positive_source", watermarked_path, 0.88),
    ):
        records.append(
            {
                "run_id": "run_fixture",
                "prompt_id": "prompt_fixture",
                "split": "test",
                "sample_role": sample_role,
                "content_score": content_score,
                "lf_score": content_score,
                "tail_robust_score": content_score,
                "aligned_content_score": None,
                "formal_evidence_positive": sample_role == "positive_source",
                "formal_rescue_applied": False,
                "formal_metric_status": "measured_image_only_detection",
                "detector_digest": f"source_{sample_role}",
                "metadata": metadata,
            }
        )
    configs = tuple(
        config
        for config in default_attack_configs()
        if config.enabled and config.resource_profile in {"full_main", "full_extra"}
    )
    for config in configs:
        for sample_role, source_path, attacked_path, content_score in (
            ("clean_negative", clean_path, attacked_clean_path, 0.10),
            ("positive_source", watermarked_path, attacked_positive_path, 0.76),
        ):
            records.append(
                {
                    "run_id": "run_fixture",
                    "prompt_id": "prompt_fixture",
                    "split": "test",
                    "sample_role": sample_role,
                    "attack_id": config.attack_id,
                    "attack_family": config.attack_family,
                    "attack_name": config.attack_name,
                    "resource_profile": config.resource_profile,
                    "attack_parameters": config.attack_parameters,
                    "attack_performed": True,
                    "source_image_path": source_path.relative_to(root).as_posix(),
                    "source_image_digest": file_digest(source_path),
                    "attacked_image_path": attacked_path.relative_to(root).as_posix(),
                    "attacked_image_digest": file_digest(attacked_path),
                    "content_score": content_score,
                    "lf_score": content_score,
                    "tail_robust_score": content_score,
                    "aligned_content_score": None,
                    "formal_evidence_positive": sample_role == "positive_source",
                    "formal_rescue_applied": False,
                    "formal_metric_status": "measured_image_only_detection",
                    "frozen_threshold_digest": "threshold_fixture",
                    "detector_digest": f"{config.attack_id}_{sample_role}",
                    "geometry_reliable": False,
                    "metadata": metadata,
                }
            )
    (runtime_dir / "image_only_detection_records.jsonl").write_text(
        "".join(_json_line(record) for record in records),
        encoding="utf-8",
    )
    (runtime_dir / "dataset_runtime_summary.json").write_text(
        json.dumps(
            {
                "paper_run_name": "probe_paper",
                "protocol_decision": "pass",
                "attack_prompt_count": attack_prompt_count,
                "formal_attack_detection_ready": True,
                "attacked_image_evidence_chain_ready": True,
                "full_method_claim_ready": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (runtime_dir / "frozen_evidence_protocol.json").write_text(
        json.dumps(
            {
                "content_threshold": 0.5,
                "geometry_score_threshold": 0.0,
                "rescue_margin_low": -0.05,
                "target_fpr": 0.1,
                "threshold_digest": "threshold_fixture",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (runtime_dir / "manifest.local.json").write_text(
        json.dumps({"artifact_id": "probe_runtime_manifest"}),
        encoding="utf-8",
    )
    return runtime_dir


@pytest.mark.quick
def test_attack_config_digest_is_stable() -> None:
    """相同正式攻击配置必须得到相同摘要。"""

    config = default_attack_configs()[1]
    assert attack_config_digest(config) == attack_config_digest(config)


@pytest.mark.quick
def test_attack_matrix_aggregates_only_measured_image_records(tmp_path: Path) -> None:
    """攻击矩阵应从真实图像文件和冻结盲检记录重建全部统计。"""

    runtime_dir = _write_runtime_fixture(tmp_path)
    manifest = write_attack_matrix_outputs(
        root=tmp_path,
        paper_run_name="probe_paper",
        dataset_runtime_dir=runtime_dir,
    )
    output_dir = tmp_path / "outputs" / "attack_matrix" / "probe_paper"
    attack_manifest = json.loads((output_dir / "attack_manifest.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (output_dir / "attack_detection_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    family_rows = list(csv.DictReader((output_dir / "attack_family_metrics.csv").open(encoding="utf-8")))

    expected_attack_count = sum(
        config.enabled and config.resource_profile in {"full_main", "full_extra"}
        for config in default_attack_configs()
    )
    assert len(records) == expected_attack_count * 2
    assert attack_manifest["attack_record_coverage_ready"] is True
    assert attack_manifest["formal_attack_detection_ready"] is True
    assert attack_manifest["full_method_claim_ready"] is True
    assert attack_manifest["detector_input_access_mode"] == "image_key_public_model_only"
    assert attack_manifest["blind_image_detector"] is True
    assert attack_manifest["generation_latent_trace_required"] is False
    assert all(record["attacked_image_available"] for record in records)
    assert all(record["metric_status"] == "measured_real_attacked_image_image_only_detection" for record in records)
    assert all(record["quality_ssim"] > 0.0 for record in records)
    assert all("proxy" not in key for record in records for key in record)
    assert family_rows
    assert "quality_score_mean" in family_rows[0]
    assert all("proxy" not in key for key in family_rows[0])
    assert manifest["metadata"]["protocol_decision"] == "pass"


@pytest.mark.quick
def test_attack_matrix_blocks_incomplete_split_role_coverage(tmp_path: Path) -> None:
    """实际角色记录数与运行摘要不一致时不得形成正式攻击结论。"""

    runtime_dir = _write_runtime_fixture(tmp_path, attack_prompt_count=2)
    manifest = write_attack_matrix_outputs(
        root=tmp_path,
        paper_run_name="probe_paper",
        dataset_runtime_dir=runtime_dir,
    )
    attack_manifest = json.loads(
        (tmp_path / "outputs" / "attack_matrix" / "probe_paper" / "attack_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert attack_manifest["attack_record_coverage_ready"] is False
    assert attack_manifest["supports_paper_claim"] is False
    assert manifest["metadata"]["protocol_decision"] == "fail"


@pytest.mark.quick
def test_attack_matrix_rejects_non_blind_detection_records(tmp_path: Path) -> None:
    """读取 prompt 或生成轨迹的检测记录不得进入正式攻击矩阵。"""

    runtime_dir = _write_runtime_fixture(tmp_path, blind_detector=False)
    with pytest.raises(ValueError, match="仅图像盲检"):
        write_attack_matrix_outputs(
            root=tmp_path,
            paper_run_name="probe_paper",
            dataset_runtime_dir=runtime_dir,
        )
