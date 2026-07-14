"""验证正式 baseline 只使用负样本校准、实测质量和真实 SD3.5 模式。"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from experiments.runtime.image_metrics import compute_image_quality_metrics, measured_image_ssim
from experiments.protocol.attacks import (
    attack_config_digest,
    formal_attack_seed_protocol_record,
    formal_attack_seed_random,
    resolve_formal_attack_config,
)
from experiments.protocol.image_only_evidence import (
    partition_calibration_prompt_ids,
)
from external_baseline.primary.sd35_method_faithful_common import (
    DEFAULT_SD35_MODEL_REVISION,
    derive_threshold,
    load_sd3_pipeline as load_common_sd3_pipeline,
)
from external_baseline.primary.gaussian_shading.adapter.method_faithful_sd35 import (
    build_observation as build_gaussian_shading_observation,
)
from external_baseline.primary.shallow_diffuse.adapter.method_faithful_sd35 import (
    build_observation as build_shallow_diffuse_observation,
)
from external_baseline.primary.t2smark.adapter.run_slm_eval import (
    _auto_threshold,
    build_t2smark_observations,
)
from external_baseline.primary.tree_ring.adapter.method_faithful_sd35 import (
    build_observation as build_tree_ring_observation,
    load_sd3_pipeline as load_tree_ring_sd3_pipeline,
)
from paper_experiments.baselines.command_plan_builder import build_parser, build_plan


T2SMARK_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"


@pytest.mark.quick
def test_method_faithful_threshold_uses_calibration_negatives_only() -> None:
    """改变阳性分数不得改变 fixed-FPR 阈值。"""

    negatives = [
        {
            "prompt_id": f"calibration_{index}",
            "split": "calibration",
            "sample_role": "clean_negative",
            "attack_family": "clean",
            "score": value,
        }
        for index, value in enumerate((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    ]
    first, source = derive_threshold(
        (*negatives, {"split": "calibration", "sample_role": "positive_source", "score": 0.6}),
        0.1,
    )
    second, _ = derive_threshold(
        (*negatives, {"split": "calibration", "sample_role": "positive_source", "score": 100.0}),
        0.1,
    )
    _, freeze_ids, _ = partition_calibration_prompt_ids(
        row["prompt_id"] for row in negatives
    )
    freeze_id_set = set(freeze_ids)

    assert first == second
    assert source == "nested_calibration_threshold_freeze_conformal"
    assert sum(
        row["score"] >= first
        for row in negatives
        if row["prompt_id"] in freeze_id_set
    ) == 0


@pytest.mark.quick
def test_common_backbone_loaders_pass_exact_model_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """共享 loader 与 Tree-Ring loader 都必须把不可变 commit 传给 Diffusers。"""

    captured: list[dict[str, object]] = []

    class Component:
        def eval(self) -> None:
            """模拟 Diffusers 子模块的推理模式切换。"""

    class FakePipeline:
        transformer = Component()
        vae = Component()

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> "FakePipeline":
            """记录模型来源参数并返回无需权重的测试 pipeline。"""

            captured.append({"model_id": model_id, **kwargs})
            return cls()

        def to(self, _device: str) -> "FakePipeline":
            """模拟设备迁移并保持对象身份。"""

            return self

        def set_progress_bar_config(self, **_kwargs: object) -> None:
            """模拟关闭 Diffusers 进度条。"""

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(StableDiffusion3Pipeline=FakePipeline),
    )
    common_pipe = load_common_sd3_pipeline(
        model_id="stabilityai/stable-diffusion-3.5-medium",
        model_revision=DEFAULT_SD35_MODEL_REVISION,
        device="cpu",
        torch_dtype_name="float32",
        adapter_class_name="TestCommonPipeline",
    )
    tree_pipe = load_tree_ring_sd3_pipeline(
        model_id="stabilityai/stable-diffusion-3.5-medium",
        model_revision=DEFAULT_SD35_MODEL_REVISION,
        device="cpu",
        torch_dtype_name="float32",
    )

    assert common_pipe is not None
    assert tree_pipe is not None
    assert [row["revision"] for row in captured] == [
        DEFAULT_SD35_MODEL_REVISION,
        DEFAULT_SD35_MODEL_REVISION,
    ]
    with pytest.raises(ValueError, match="40位小写十六进制"):
        load_common_sd3_pipeline(
            model_id="stabilityai/stable-diffusion-3.5-medium",
            model_revision="main",
            device="cpu",
            torch_dtype_name="float32",
            adapter_class_name="InvalidRevisionPipeline",
        )


@pytest.mark.parametrize(
    ("builder", "extra_fields"),
    (
        (build_tree_ring_observation, {}),
        (build_gaussian_shading_observation, {}),
        (build_shallow_diffuse_observation, {"injection_mode": "complex"}),
    ),
)
def test_common_backbone_producers_bind_formal_attack_identity(
    builder: object,
    extra_fields: dict[str, object],
) -> None:
    """三个 common-backbone producer 都必须在写行前校验攻击身份."""

    attack_config = resolve_formal_attack_config(
        attack_family="standard_distortion",
        attack_name="jpeg_compression",
    )
    kwargs = {
        "event_id": "event_0001",
        "score": 0.9,
        "threshold": 0.5,
        "threshold_source": "nested_calibration_threshold_freeze_conformal",
        "row": {"split": "test", "prompt_id": "prompt_0001"},
        "index": 1,
        "sample_role": "attacked_positive",
        "attack_family": attack_config.attack_family,
        "attack_condition": attack_config.attack_name,
        "image_id": "image_0001",
        "image_path": "outputs/test/image_0001.png",
        "image_digest": "1" * 64,
        "latent_shape": (1, 16, 64, 64),
        "execution_device": "cuda",
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "model_revision": DEFAULT_SD35_MODEL_REVISION,
        "quality_score": 0.9,
        "score_retention": 0.8,
        "attack_id": attack_config.attack_id,
        "resource_profile": attack_config.resource_profile,
        "attack_config_digest_value": attack_config_digest(attack_config),
        **extra_fields,
    }

    row = builder(**kwargs)  # type: ignore[operator]

    assert row["attack_id"] == attack_config.attack_id
    assert row["resource_profile"] == attack_config.resource_profile
    assert row["attack_config_digest"] == attack_config_digest(attack_config)
    assert row["generation_model_revision"] == DEFAULT_SD35_MODEL_REVISION

    kwargs["attack_config_digest_value"] = "0" * 64
    with pytest.raises(ValueError, match="AttackConfig"):
        builder(**kwargs)  # type: ignore[operator]


@pytest.mark.quick
def test_t2smark_threshold_uses_only_calibration_clean_scores() -> None:
    """T2SMark 适配阈值必须遵循同一负样本校准协议。"""

    pairs = [
        {"prompt_id": "calibration_0", "split": "calibration"},
        {"prompt_id": "calibration_1", "split": "calibration"},
        {"prompt_id": "calibration_2", "split": "calibration"},
        {"prompt_id": "test_0", "split": "test"},
    ]
    results = {
        0: {"image_only_detection": {"clean_score": 0.1, "watermarked_score": 0.9}},
        1: {"image_only_detection": {"clean_score": 0.2, "watermarked_score": -100.0}},
        2: {"image_only_detection": {"clean_score": 0.3, "watermarked_score": -200.0}},
        3: {"image_only_detection": {"clean_score": 999.0, "watermarked_score": 999.0}},
    }

    threshold, source = _auto_threshold(results, pairs, 0.1)

    assert threshold > 0.1
    assert source == "nested_calibration_threshold_freeze_conformal"


@pytest.mark.quick
def test_t2smark_adapter_rejects_incomplete_calibration_before_threshold() -> None:
    """缺少任一 Prompt 单元时不得用部分 calibration 分数冻结阈值."""

    pairs = [
        {"split": "calibration"},
        {"split": "calibration"},
        {"split": "test"},
    ]
    incomplete_results = {
        "0": {
            "image_only_detection": {
                "clean_score": 0.1,
                "watermarked_score": 0.9,
            }
        },
        "2": {
            "image_only_detection": {
                "clean_score": 0.2,
                "watermarked_score": 0.8,
            }
        },
    }

    with pytest.raises(ValueError, match="完整 Prompt 单元集合"):
        build_t2smark_observations(
            image_pairs=pairs,
            t2smark_results=incomplete_results,
            model_id=T2SMARK_MODEL_ID,
            model_revision=DEFAULT_SD35_MODEL_REVISION,
            target_fpr=0.1,
        )


@pytest.mark.quick
def test_t2smark_formal_attacks_use_distinct_clean_and_watermarked_images(
    tmp_path: Path,
) -> None:
    """攻击后阴性必须来自真实 clean image, 不能复用水印图像的错误密钥分数。"""

    clean_path = tmp_path / "clean.png"
    watermarked_path = tmp_path / "watermarked.png"
    attacked_negative_path = tmp_path / "attacked_negative.png"
    attacked_positive_path = tmp_path / "attacked_positive.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(clean_path)
    Image.new("RGB", (16, 16), (40, 50, 60)).save(watermarked_path)
    Image.new("RGB", (16, 16), (15, 25, 35)).save(attacked_negative_path)
    Image.new("RGB", (16, 16), (45, 55, 65)).save(attacked_positive_path)
    rows = [
        {
            "image_id": "image_0001",
            "prompt_id": "prompt_0001",
            "prompt_text": "a ceramic fox",
            "randomization_repeat_id": "seed_00_key_00",
            "generation_seed_index": 0,
            "generation_seed_offset": 0,
            "generation_seed_random": 1703,
            "watermark_key_index": 0,
            "watermark_key_seed_random": 1729,
            "watermark_key_material_digest_random": "1" * 64,
            "formal_randomization_protocol_digest": "2" * 64,
            "formal_randomization_identity_digest_random": "3" * 64,
            "base_latent_content_digest_random": "4" * 64,
            "base_latent_identity_digest_random": "5" * 64,
            "split": "test",
            "clean_image_path": str(clean_path),
            "clean_image_digest": "clean_digest",
            "watermarked_image_path": str(watermarked_path),
            "watermarked_image_digest": "watermarked_digest",
            "strict_pair_quality_ready": True,
        }
    ]
    calibration_rows = [
        {
            **rows[0],
            "image_id": f"calibration_image_{index}",
            "prompt_id": f"calibration_prompt_{index}",
            "prompt_text": f"calibration prompt {index}",
            "split": "calibration",
        }
        for index in range(3)
    ]
    rows = calibration_rows + rows
    attack_seed = formal_attack_seed_random(1703, "jpeg_compression_main")
    attack_seed_protocol_digest = formal_attack_seed_protocol_record()[
        "formal_attack_seed_protocol_digest"
    ]
    results = {
        **{
            str(index): {
                "robustness": {
                    "norm1_no_w": 0.2,
                    "norm1_w": 0.9,
                    "acc_msg": 1.0,
                },
                "image_only_detection": {
                    "clean_score": 0.1 + index * 0.01,
                    "watermarked_score": 0.9,
                },
            }
            for index in range(3)
        },
        "3": {
            "robustness": {"norm1_no_w": 0.2, "norm1_w": 0.9, "acc_msg": 1.0},
            "image_only_detection": {"clean_score": 0.1, "watermarked_score": 0.9},
            "formal_attacks": {
                "jpeg_compression": {
                    "attack_id": "jpeg_compression_main",
                    "attack_family": "standard_distortion",
                    "attack_name": "jpeg_compression",
                    "attack_condition": "jpeg_compression",
                    "resource_profile": "full_main",
                    "attack_config_digest": attack_config_digest(
                        resolve_formal_attack_config(
                            attack_family="standard_distortion",
                            attack_name="jpeg_compression",
                        )
                    ),
                    "attack_seed_random": attack_seed,
                    "formal_attack_seed_protocol_digest": (
                        attack_seed_protocol_digest
                    ),
                    "attacked_negative": {
                        "attack_id": "jpeg_compression_main",
                        "resource_profile": "full_main",
                        "attack_config_digest": attack_config_digest(
                            resolve_formal_attack_config(
                                attack_family="standard_distortion",
                                attack_name="jpeg_compression",
                            )
                        ),
                        "attack_seed_random": attack_seed,
                        "formal_attack_seed_protocol_digest": (
                            attack_seed_protocol_digest
                        ),
                        "detection_score": 0.12,
                        "attacked_image_path": str(attacked_negative_path),
                        "attacked_image_digest": "attacked_negative_digest",
                    },
                    "attacked_positive": {
                        "attack_id": "jpeg_compression_main",
                        "resource_profile": "full_main",
                        "attack_config_digest": attack_config_digest(
                            resolve_formal_attack_config(
                                attack_family="standard_distortion",
                                attack_name="jpeg_compression",
                            )
                        ),
                        "attack_seed_random": attack_seed,
                        "formal_attack_seed_protocol_digest": (
                            attack_seed_protocol_digest
                        ),
                        "detection_score": 0.75,
                        "attacked_image_path": str(attacked_positive_path),
                        "attacked_image_digest": "attacked_positive_digest",
                    },
                }
            },
        }
    }

    observations, _ = build_t2smark_observations(
        image_pairs=rows,
        t2smark_results=results,
        model_id=T2SMARK_MODEL_ID,
        model_revision=DEFAULT_SD35_MODEL_REVISION,
        target_fpr=0.1,
    )

    assert all(
        row["generation_model_id"] == T2SMARK_MODEL_ID
        and row["generation_model_revision"] == DEFAULT_SD35_MODEL_REVISION
        for row in observations
    )
    attacked_rows = [row for row in observations if str(row["sample_role"]).startswith("attacked_")]
    assert len(attacked_rows) == 2
    by_role = {row["sample_role"]: row for row in attacked_rows}
    assert by_role["attacked_negative"]["image_path"] == str(attacked_negative_path)
    assert by_role["attacked_positive"]["image_path"] == str(attacked_positive_path)
    assert by_role["attacked_negative"]["score"] == pytest.approx(0.12)
    assert by_role["attacked_positive"]["score"] == pytest.approx(0.75)

    del results["3"]["formal_attacks"]["jpeg_compression"]["attacked_positive"][
        "attack_config_digest"
    ]
    with pytest.raises(ValueError, match="AttackConfig"):
        build_t2smark_observations(
            image_pairs=rows,
            t2smark_results=results,
            model_id=T2SMARK_MODEL_ID,
            model_revision=DEFAULT_SD35_MODEL_REVISION,
            target_fpr=0.1,
        )


@pytest.mark.quick
def test_measured_ssim_reads_image_content() -> None:
    """质量分数必须随实际像素变化, 不能固定为 1。"""

    white = Image.new("RGB", (32, 32), (255, 255, 255))
    black = Image.new("RGB", (32, 32), (0, 0, 0))

    assert measured_image_ssim(white, white) == pytest.approx(1.0, abs=1e-6)
    assert measured_image_ssim(white, black) < 0.1


@pytest.mark.quick
def test_paired_image_metrics_reject_implicit_resize() -> None:
    """正式成对质量统计不得在指标函数内部静默改变候选图像尺寸。"""

    reference = Image.new("RGB", (32, 32), (255, 255, 255))
    different_size = Image.new("RGB", (16, 16), (255, 255, 255))

    with pytest.raises(ValueError, match="相同尺寸"):
        compute_image_quality_metrics(reference, different_size)


@pytest.mark.quick
def test_external_baseline_plan_requires_target_fpr_and_real_mode(tmp_path: Path) -> None:
    """命令计划必须显式传递论文级 FPR 并选择唯一真实模式。"""

    adapter = tmp_path / "external_baseline/primary/tree_ring/adapter/run_slm_eval.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("print('adapter')\n", encoding="utf-8")
    prompt_plan = tmp_path / "prompt_plan.json"
    prompt_plan.write_text('[{"prompt_text":"a fox","split":"calibration"}]\n', encoding="utf-8")
    base_args = [
        "--root", str(tmp_path),
        "--methods", "tree_ring",
        "--output-root", str(tmp_path / "outputs/baselines"),
        "--prompt-plan", str(prompt_plan),
    ]
    parser = build_parser()
    with pytest.raises(ValueError, match="target-fpr"):
        build_plan(parser.parse_args(base_args))

    plan = build_plan(parser.parse_args([*base_args, "--target-fpr", "0.1"]))
    command = plan[0]["command"]
    assert command[command.index("--adapter-mode") + 1] == "method_faithful_sd35"
    assert command[command.index("--target-fpr") + 1] == "0.1"
    assert command[command.index("--model-revision") + 1] == DEFAULT_SD35_MODEL_REVISION


@pytest.mark.quick
def test_generic_command_plan_rejects_t2smark_duplicate_formal_entry(tmp_path: Path) -> None:
    """T2SMark 只能由独立 formal runner 启动，不能进入 generic command plan。"""

    prompt_plan = tmp_path / "prompt_plan.json"
    prompt_plan.write_text('[{"prompt_text":"a fox","split":"calibration"}]\n', encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--methods",
            "t2smark",
            "--output-root",
            str(tmp_path / "outputs/baselines"),
            "--prompt-plan",
            str(prompt_plan),
            "--target-fpr",
            "0.1",
        ]
    )

    with pytest.raises(ValueError, match="未登记的 primary baseline adapter"):
        build_plan(args)
