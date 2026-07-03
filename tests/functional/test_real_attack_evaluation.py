"""验证真实攻击闭环 helper 的轻量行为."""

from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image

from paper_workflow.colab_utils import conventional_geometric_attack_evaluation, real_attack_evaluation
from experiments.protocol.attacks import default_attack_configs
from external_baseline.primary.sd35_method_faithful_common import supported_formal_image_attack_names
from paper_workflow.colab_utils.conventional_geometric_attack_evaluation import conventional_attack_configs
from paper_workflow.colab_utils.real_attack_evaluation import (
    PRIMARY_MODEL_FAMILY,
    PRIMARY_MODEL_ID,
    RealAttackEvaluationConfig,
    package_real_attack_evaluation_outputs,
    materialize_drive_package_inputs,
    run_default_real_attack_evaluation_from_drive_plan,
    default_attack_specs,
    write_real_attack_evaluation_outputs,
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """读取测试中生成的 JSONL 记录."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.quick
def test_slm_attack_workflows_cover_paper_formal_attack_matrix() -> None:
    """SLM 两类攻击 workflow 应共同覆盖 paper 共同协议的 12 类正式攻击。"""

    paper_attack_names = {
        config.attack_name
        for config in default_attack_configs()
        if config.resource_profile in {"full_main", "full_extra"}
    }
    conventional_names = {config.attack_name for config in conventional_attack_configs()}
    regeneration_names = {spec.attack_name for spec in default_attack_specs()}

    assert len(paper_attack_names) == 12
    assert conventional_names == {
        "jpeg_compression",
        "gaussian_noise",
        "gaussian_blur",
        "resize",
        "crop",
        "rotation",
        "crop_resize",
        "composite_geometric_attacks",
    }
    assert regeneration_names == {
        "img2img_regeneration",
        "ddim_inversion_regeneration",
        "sdedit_regeneration",
        "diffusion_purification",
    }
    assert conventional_names.isdisjoint(regeneration_names)
    assert conventional_names | regeneration_names == paper_attack_names
    assert paper_attack_names == set(supported_formal_image_attack_names())
    assert all(config.resource_profile == "full_main" for config in conventional_attack_configs())
    assert conventional_geometric_attack_evaluation.load_detector_pipeline is real_attack_evaluation.load_detector_pipeline
    assert not hasattr(conventional_geometric_attack_evaluation, "load_img2img_pipeline")


@pytest.mark.quick
def test_detector_loader_falls_back_to_vae_subfolder(monkeypatch: pytest.MonkeyPatch) -> None:
    """完整 SD3 pipeline 导入失败时, detector loader 应退到 VAE 子模块路径。"""

    class FakeVae:
        """模拟测试用 VAE, 避免加载真实模型。"""

        def to(self, device_name: str) -> "FakeVae":
            self.device_name = device_name
            return self

        def eval(self) -> None:
            self.eval_called = True

    def fake_sd3_loader(config: RealAttackEvaluationConfig, torch_module: object) -> object:
        raise RuntimeError("stable_diffusion_3_import_failed")

    def fake_vae_loader(config: RealAttackEvaluationConfig, torch_module: object) -> real_attack_evaluation.DetectorPipeline:
        return real_attack_evaluation.DetectorPipeline(
            vae=FakeVae(),
            image_processor=real_attack_evaluation.SimpleVaeImageProcessor(),
            loader_name="vae_subfolder",
        )

    monkeypatch.setattr(real_attack_evaluation, "_load_detector_pipeline_from_sd3_pipeline", fake_sd3_loader)
    monkeypatch.setattr(real_attack_evaluation, "_load_detector_pipeline_from_vae", fake_vae_loader)
    config = RealAttackEvaluationConfig(
        model_family=PRIMARY_MODEL_FAMILY,
        model_id=PRIMARY_MODEL_ID,
        seed=20260621,
        prompt="test prompt",
        negative_prompt="low quality",
        width=32,
        height=32,
        inference_steps=2,
        guidance_scale=1.0,
        device_name="cpu",
        torch_dtype="float32",
    )

    detector_pipeline, runtime_versions = real_attack_evaluation.load_detector_pipeline(config)

    assert isinstance(detector_pipeline, real_attack_evaluation.DetectorPipeline)
    assert runtime_versions["detector_loader_name"] == "vae_subfolder"
    assert "stable_diffusion_3_import_failed" in runtime_versions["detector_loader_fallback_reason"]
    assert runtime_versions["runtime_environment"]["detector_loader_name"] == "vae_subfolder"


@pytest.mark.quick
def test_transformers_dinov2_registers_compatibility_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """transformers 缺少 registers 导出时应由兼容补丁补齐。"""

    fake_transformers = types.SimpleNamespace(
        Dinov2Config=type("Dinov2Config", (), {}),
        Dinov2Model=type("Dinov2Model", (), {}),
        Dinov2PreTrainedModel=type("Dinov2PreTrainedModel", (), {}),
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

    report = real_attack_evaluation.patch_transformers_for_diffusers_autoencoder_import()

    assert report["transformers_dinov2_registers_patch_applied"] is True
    assert set(report["patched_transformers_exports"]) == {
        "Dinov2WithRegistersConfig",
        "Dinov2WithRegistersModel",
        "Dinov2WithRegistersPreTrainedModel",
    }
    assert fake_transformers.Dinov2WithRegistersConfig is fake_transformers.Dinov2Config
    assert fake_transformers.Dinov2WithRegistersModel is fake_transformers.Dinov2Model


@pytest.mark.quick
def test_numpy_umath_center_compatibility_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """NumPy 字符串居中导出缺失时应由兼容补丁补齐。"""

    fake_umath = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "numpy._core.umath", fake_umath)

    report = real_attack_evaluation.patch_numpy_core_umath_string_center_compatibility()

    assert report["numpy_umath_center_patch_applied"] is True
    assert hasattr(fake_umath, "_center")


@pytest.mark.quick
def test_pillow_typing_ink_compatibility_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pillow `_Ink` 类型导出缺失时应由兼容补丁补齐。"""

    fake_pil_typing = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "PIL._typing", fake_pil_typing)

    report = real_attack_evaluation.patch_pillow_typing_ink_compatibility()

    assert report["pillow_typing_ink_patch_applied"] is True
    assert hasattr(fake_pil_typing, "_Ink")


@pytest.mark.quick
def test_conventional_gaussian_noise_attack_is_deterministic_without_numpy() -> None:
    """常规高斯噪声攻击应由 torch 生成, 避免依赖 Colab 中易错的 NumPy 重载状态。"""

    attack_config = next(config for config in conventional_attack_configs() if config.attack_name == "gaussian_noise")
    source_image = Image.new("RGB", (8, 8), color=(120, 90, 70))

    first = conventional_geometric_attack_evaluation.apply_conventional_geometric_attack(
        source_image, attack_config, seed=20260703
    )
    second = conventional_geometric_attack_evaluation.apply_conventional_geometric_attack(
        source_image, attack_config, seed=20260703
    )

    assert first.mode == "RGB"
    assert first.size == source_image.size
    assert first.tobytes() == second.tobytes()
    assert first.tobytes() != source_image.tobytes()


@pytest.mark.quick
def test_materialize_drive_package_inputs_extracts_only_governed_outputs(tmp_path: Path) -> None:
    """前序结果应来自 Drive 包, 且不能把包内代码文件覆盖回工作区。"""
    aligned_drive = tmp_path / "drive" / "aligned_rescoring"
    threshold_drive = tmp_path / "drive" / "threshold_calibration"
    aligned_drive.mkdir(parents=True)
    threshold_drive.mkdir(parents=True)
    aligned_package = aligned_drive / "aligned_rescoring_package_20260621t000000z_abcdef0.zip"
    threshold_package = threshold_drive / "threshold_calibration_package_20260621t000000z_abcdef0.zip"
    with ZipFile(aligned_package, mode="w") as archive:
        archive.writestr("outputs/aligned_rescoring/aligned_images/sample.png", b"image-bytes")
        archive.writestr("outputs/prompt_event_protocol/prompt_records.jsonl", "{}\n")
        archive.writestr("paper_workflow/colab_utils/aligned_rescoring.py", "should_not_extract = True\n")
    with ZipFile(threshold_package, mode="w") as archive:
        archive.writestr("outputs/threshold_calibration/calibration_thresholds.json", "{}\n")
        archive.writestr("outputs/threshold_calibration/threshold_degeneracy_report.json", "{}\n")
        archive.writestr("paper_workflow/colab_utils/threshold_helper.py", "should_not_extract = True\n")

    manifest = materialize_drive_package_inputs(
        root=tmp_path,
        aligned_rescoring_drive_dir=str(aligned_drive),
        threshold_calibration_drive_dir=str(threshold_drive),
    )

    assert manifest["aligned_extracted_entry_count"] == 2
    assert manifest["threshold_extracted_entry_count"] == 2
    assert (tmp_path / "outputs" / "aligned_rescoring" / "aligned_images" / "sample.png").exists()
    assert (tmp_path / "outputs" / "threshold_calibration" / "calibration_thresholds.json").exists()
    assert not (tmp_path / "paper_workflow" / "colab_utils" / "aligned_rescoring.py").exists()
    assert not (tmp_path / "paper_workflow" / "colab_utils" / "threshold_helper.py").exists()


@pytest.mark.quick
def test_drive_workflow_writes_failure_summary_when_required_package_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive 前序包缺失时也应写出可打包诊断产物。"""
    monkeypatch.setenv("SLM_WM_REAL_ATTACK_OUTPUT_DIR", "outputs/real_attack_evaluation")
    summary = run_default_real_attack_evaluation_from_drive_plan(
        root=tmp_path,
        aligned_rescoring_drive_dir=str(tmp_path / "missing_aligned"),
        threshold_calibration_drive_dir=str(tmp_path / "missing_threshold"),
    )
    output_dir = tmp_path / "outputs" / "real_attack_evaluation"

    assert summary["run_decision"] == "fail"
    assert "drive_package_missing" in summary["unsupported_reason"]
    assert (output_dir / "real_attack_run_summary.json").exists()
    assert (output_dir / "real_attack_environment_report.json").exists()
    assert (output_dir / "real_attack_manifest.local.json").exists()

    archive_record = package_real_attack_evaluation_outputs(
        root=tmp_path,
        drive_output_dir=str(tmp_path / "drive_mirror"),
    )
    assert (tmp_path / archive_record.archive_path).exists()
    assert archive_record.drive_archive_digest == archive_record.archive_digest


@pytest.mark.quick
def test_real_attack_evaluation_writes_image_registry_and_detection_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """真实攻击闭环 helper 应记录 source / attacked digest 并重跑攻击后检测."""
    source_dir = tmp_path / "outputs" / "aligned_rescoring" / "aligned_images"
    clean_dir = tmp_path / "outputs" / "aligned_rescoring" / "clean_images"
    source_dir.mkdir(parents=True)
    clean_dir.mkdir(parents=True)
    source_path = source_dir / "sample_aligned.png"
    clean_source_path = clean_dir / "sample_clean.png"
    Image.new("RGB", (32, 32), color=(120, 90, 70)).save(source_path)
    Image.new("RGB", (32, 32), color=(80, 100, 120)).save(clean_source_path)
    prompt_dir = tmp_path / "outputs" / "prompt_event_protocol"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "prompt_records.jsonl").write_text(
        '{"prompt_id":"prompt_a","prompt_text":"a ceramic bird","supports_paper_claim":false}\n',
        encoding="utf-8",
    )
    aligned_dir = tmp_path / "outputs" / "aligned_rescoring"
    (aligned_dir / "aligned_rescoring_quality_metrics.csv").write_text(
        "prompt_id,clean_image_path,aligned_image_path\n"
        "prompt_a,outputs/aligned_rescoring/clean_images/sample_clean.png,"
        "outputs/aligned_rescoring/aligned_images/sample_aligned.png\n",
        encoding="utf-8",
    )
    (aligned_dir / "aligned_rescoring_records.jsonl").write_text(
        '{"aligned_rescoring_record_id":"aligned_a","prompt_id":"prompt_a","real_raw_content_score":0.8,'
        '"real_aligned_content_score":0.82,"aligned_rescoring_ready":true,'
        '"split":"test","sample_role":"positive_source","supports_paper_claim":false}\n'
        '{"aligned_rescoring_record_id":"clean_a","prompt_id":"prompt_a","real_raw_content_score":0.1,'
        '"real_aligned_content_score":0.12,"aligned_rescoring_ready":true,'
        '"split":"test","sample_role":"clean_negative","supports_paper_claim":false}\n'
        '{"aligned_rescoring_record_id":"attacked_a","prompt_id":"prompt_a","real_raw_content_score":0.2,'
        '"real_aligned_content_score":0.22,"aligned_rescoring_ready":true,'
        '"split":"test","sample_role":"attacked_negative","supports_paper_claim":false}\n',
        encoding="utf-8",
    )
    threshold_dir = tmp_path / "outputs" / "threshold_calibration"
    threshold_dir.mkdir(parents=True)
    (threshold_dir / "calibration_thresholds.json").write_text(
        '{"threshold_value":0.5,"target_fpr":0.05,"supports_paper_claim":false}\n',
        encoding="utf-8",
    )
    (threshold_dir / "threshold_degeneracy_report.json").write_text(
        '{"calibrated_content_threshold":0.5,"target_fpr":0.05,"rescue_margin_low":-0.05,'
        '"allowed_fail_reasons":["geometry_suspected","low_confidence"],"supports_paper_claim":false}\n',
        encoding="utf-8",
    )

    def fake_load_pipeline(config: RealAttackEvaluationConfig) -> tuple[object, dict[str, object]]:
        return object(), {"runtime_environment": {"cuda_available": True, "gpu_name": "fake_gpu"}}

    def fake_run_pipeline_attack(
        pipeline: object,
        source_image: Image.Image,
        spec: real_attack_evaluation.RealAttackSpec,
        config: RealAttackEvaluationConfig,
        seed: int,
        prompt_text: str,
    ) -> Image.Image:
        channels = {
            "img2img_regeneration": (126, 92, 73),
            "sdedit_regeneration": (138, 100, 82),
            "diffusion_purification": (124, 91, 72),
        }
        return Image.new("RGB", (source_image.width * 2, source_image.height * 2), color=channels[spec.attack_name])

    def fake_run_strict_ddim(
        source_image: Image.Image,
        spec: real_attack_evaluation.RealAttackSpec,
        config: RealAttackEvaluationConfig,
        seed: int,
        prompt_text: str,
    ) -> Image.Image:
        return Image.new("RGB", (source_image.width * 2, source_image.height * 2), color=(130, 96, 78))

    def fake_rescore_attacked_image_with_detector(
        detector_pipeline: object,
        attacked_image: Image.Image,
        source_context: dict[str, object],
        boundary: dict[str, object],
        config: RealAttackEvaluationConfig,
    ) -> dict[str, object]:
        raw_score = 0.74 if source_context["sample_role"] == "positive_source" else 0.10
        aligned_score = 0.80 if source_context["sample_role"] == "positive_source" else 0.12
        return {
            "raw_content_score_after": raw_score,
            "aligned_content_score_after": aligned_score,
            "threshold_score_after": aligned_score,
            "raw_content_margin_after": raw_score - float(boundary["content_threshold"]),
            "aligned_content_margin_after": aligned_score - float(boundary["content_threshold"]),
            "positive_by_content": raw_score >= float(boundary["content_threshold"]),
            "geometry_reliable": True,
            "registration_confidence": 0.82,
            "anchor_inlier_ratio": 0.75,
            "recovered_sync_consistency": 0.88,
            "alignment_residual": 0.18,
            "rescue_eligible": False,
            "rescue_applied": False,
            "evidence_decision": aligned_score >= float(boundary["content_threshold"]),
            "formal_detection_decision": aligned_score >= float(boundary["content_threshold"]),
            "attacked_image_rescore_performed": True,
            "formal_detection_proxy": False,
            "detection_score_source": "attacked_image_vae_latent_projection_watermark_rescore",
            "latent_projection_mode": "periodic_slot_pooled_content_carrier",
            "attacked_latent_projection_digest": "fake_projection_digest",
            "watermark_coordinate": 1.0,
            "bounded_watermark_coordinate": 1.0,
        }

    monkeypatch.setattr(real_attack_evaluation, "load_img2img_pipeline", fake_load_pipeline)
    monkeypatch.setattr(real_attack_evaluation, "load_detector_pipeline", fake_load_pipeline)
    monkeypatch.setattr(real_attack_evaluation, "run_pipeline_attack", fake_run_pipeline_attack)
    monkeypatch.setattr(real_attack_evaluation, "run_strict_ddim_inversion_attack", fake_run_strict_ddim)
    monkeypatch.setattr(real_attack_evaluation, "rescore_attacked_image_with_detector", fake_rescore_attacked_image_with_detector)
    config = RealAttackEvaluationConfig(
        model_family=PRIMARY_MODEL_FAMILY,
        model_id=PRIMARY_MODEL_ID,
        seed=20260621,
        prompt="test prompt",
        negative_prompt="low quality",
        width=32,
        height=32,
        inference_steps=2,
        guidance_scale=1.0,
        source_image_dir="outputs/aligned_rescoring/aligned_images",
    )

    summary = write_real_attack_evaluation_outputs(config=config, root=tmp_path)
    output_dir = tmp_path / "outputs" / "real_attack_evaluation"
    records = read_jsonl(output_dir / "real_attack_detection_records.jsonl")
    registry_rows = read_jsonl(output_dir / "real_attacked_image_registry.jsonl")

    assert summary["run_decision"] == "pass"
    assert summary["real_attacked_image_closed_loop_ready"] is True
    assert summary["regeneration_attack_gpu_validation_ready"] is True
    assert summary["attack_detection_rerun_ready"] is True
    assert summary["formal_attack_detection_ready"] is True
    assert summary["attacked_image_rescore_ready"] is True
    assert summary["proxy_formal_record_count"] == 0
    assert summary["real_attacked_image_count"] == 8
    assert len(records) == 8
    assert len(registry_rows) == 8
    assert all(record["metric_status"] == "measured_from_real_attacked_image_watermark_rescore" for record in records)
    assert all(record["source_image_digest"] for record in records)
    assert all(record["attacked_image_digest"] for record in records)
    assert all(record["supports_paper_claim"] is False for record in records)
    assert all((tmp_path / str(row["attacked_image_path"])).exists() for row in registry_rows)
    for row in registry_rows:
        with Image.open(tmp_path / str(row["attacked_image_path"])) as attacked_image:
            assert attacked_image.size == (32, 32)
    assert (output_dir / "real_attack_family_metrics.csv").read_text(encoding="utf-8").count("\n") >= 5
    formal_records = read_jsonl(output_dir / "formal_attack_detection_records.jsonl")
    family_rows = list(csv.DictReader((output_dir / "real_attack_family_metrics.csv").open(encoding="utf-8")))
    assert len(formal_records) == 8
    assert all(
        record["metric_status"] == "measured_from_real_attacked_image_watermark_rescore_formal_protocol"
        for record in formal_records
    )
    assert all(record["metadata"]["formal_detection_proxy"] is False for record in formal_records)
    assert all(record["metadata"]["attacked_image_rescore_performed"] is True for record in formal_records)
    assert all(record["metadata"]["attacked_image_rescore_required_for_claim"] is True for record in formal_records)
    assert {record["sample_role"] for record in formal_records} == {"positive_source", "clean_negative"}
    for family_row in family_rows:
        positive_records = [
            record
            for record in formal_records
            if record["attack_name"] == family_row["attack_name"] and record["sample_role"] == "positive_source"
        ]
        clean_negative_records = [
            record
            for record in formal_records
            if record["attack_name"] == family_row["attack_name"] and record["sample_role"] == "clean_negative"
        ]
        expected_positive_rate = sum(1 for record in positive_records if record["evidence_decision"]) / len(positive_records)
        expected_clean_fpr = sum(1 for record in clean_negative_records if record["evidence_decision"]) / len(
            clean_negative_records
        )
        assert family_row["metrics_source"] == "formal_attack_detection_records"
        assert float(family_row["detection_positive_rate"]) == pytest.approx(expected_positive_rate)
        assert float(family_row["formal_clean_false_positive_rate"]) == pytest.approx(expected_clean_fpr)


@pytest.mark.quick
def test_real_attack_evaluation_package_contains_audit_files(tmp_path: Path) -> None:
    """真实攻击闭环压缩包应纳入核对文件和 attacked image 文件."""
    attack_dir = tmp_path / "outputs" / "real_attack_evaluation"
    attacked_dir = attack_dir / "attacked_images"
    attacked_dir.mkdir(parents=True)
    (attack_dir / "real_attack_run_summary.json").write_text('{"run_decision":"pass"}\n', encoding="utf-8")
    (attack_dir / "real_attack_detection_records.jsonl").write_text('{"attack_performed":true}\n', encoding="utf-8")
    (attack_dir / "formal_attack_detection_records.jsonl").write_text('{"attack_performed":true}\n', encoding="utf-8")
    (attack_dir / "real_attacked_image_registry.jsonl").write_text('{"attacked_image_digest":"digest"}\n', encoding="utf-8")
    (attack_dir / "real_attack_family_metrics.csv").write_text("attack_name,measured_record_count\nimg2img_regeneration,1\n", encoding="utf-8")
    (attack_dir / "real_attack_environment_report.json").write_text('{"cuda_available":true}\n', encoding="utf-8")
    (attack_dir / "real_attack_manifest.local.json").write_text('{"artifact_id":"real_attack_evaluation_manifest"}\n', encoding="utf-8")
    (attacked_dir / "sample_attacked.png").write_bytes(b"fake_png_bytes")

    drive_dir = tmp_path / "drive_mirror"
    archive_record = package_real_attack_evaluation_outputs(root=tmp_path, drive_output_dir=str(drive_dir))

    assert (tmp_path / archive_record.archive_path).exists()
    assert (drive_dir / "real_attack_evaluation_package.zip").exists()
    assert archive_record.archive_digest == archive_record.drive_archive_digest
    assert archive_record.archive_entry_count >= 8
