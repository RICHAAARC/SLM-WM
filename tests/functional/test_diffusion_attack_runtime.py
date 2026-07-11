"""验证共同扩散攻击只通过真实 img2img、inpaint 与反演算子执行。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from experiments.runtime.diffusion.regeneration_attacks import (
    DiffusionAttackExecution,
    DiffusionAttackRuntime,
    DiffusionAttackRuntimeConfig,
    default_diffusion_attack_specs,
    diffusion_attack_spec,
)
from external_baseline.primary.sd35_method_faithful_common import (
    apply_regeneration_image_attack,
    normalize_attack_request,
)


class RecordingImagePipeline:
    """记录 Diffusers 风格调用参数并返回可区分的真实 PIL 图像。"""

    def __init__(self, color: tuple[int, int, int] = (220, 10, 10)) -> None:
        self.calls: list[dict[str, object]] = []
        self.color = color

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        """保存调用并返回与输入图像同尺寸的结果。"""

        self.calls.append(dict(kwargs))
        source = kwargs.get("image")
        size = source.size if isinstance(source, Image.Image) else (16, 16)
        image = Image.new("RGB", size, self.color)
        image.candidate_index = len(self.calls)
        return SimpleNamespace(images=[image])


def build_runtime() -> tuple[DiffusionAttackRuntime, RecordingImagePipeline, RecordingImagePipeline]:
    """构造不加载模型的 CPU 记录运行时。"""

    text_to_image = RecordingImagePipeline((10, 220, 10))
    img2img = RecordingImagePipeline()
    inpaint = RecordingImagePipeline()
    runtime = DiffusionAttackRuntime(
        text_to_image_pipeline=text_to_image,
        img2img_pipeline=img2img,
        inpaint_pipeline=inpaint,
        config=DiffusionAttackRuntimeConfig(device_name="cpu", height=16, width=16),
    )
    return runtime, img2img, inpaint


@pytest.mark.quick
def test_diffusion_attack_specs_are_exact_and_current() -> None:
    """共同规格应精确覆盖8个真实 GPU 攻击且不保留错误的 DDIM 名称。"""

    specs = default_diffusion_attack_specs()
    names = {spec.attack_name for spec in specs}

    assert len(specs) == 8
    assert names == {
        "img2img_regeneration",
        "flow_matching_inversion_regeneration",
        "sdedit_regeneration",
        "diffusion_purification",
        "global_editing_attack",
        "local_editing_attack",
        "visual_paraphrase_attack",
        "adversarial_removal_attack",
    }
    assert diffusion_attack_spec("local_editing_attack").attack_implementation == (
        "sd3_inpainting_local_edit"
    )


@pytest.mark.quick
def test_local_editing_uses_inpaint_mask_and_preserves_unmasked_pixels() -> None:
    """局部编辑必须调用 inpaint, 且 mask 外像素保持为源图像。"""

    runtime, img2img, inpaint = build_runtime()
    source = Image.new("RGB", (20, 10), (5, 10, 200))

    execution = runtime.apply(
        source,
        diffusion_attack_spec("local_editing_attack"),
        seed=17,
        prompt_text="a ceramic fox",
    )

    assert not img2img.calls
    assert len(inpaint.calls) == 1
    call = inpaint.calls[0]
    assert call["image"] is source
    mask = call["mask_image"]
    assert isinstance(mask, Image.Image)
    assert mask.getbbox() is not None
    assert mask.getbbox() != (0, 0, mask.width, mask.height)
    assert execution.local_edit_mask_digest
    assert execution.local_edit_mask_area_ratio == pytest.approx(0.36, abs=0.08)
    pixels = list(execution.image.getdata())
    assert (220, 10, 10) in pixels
    assert (5, 10, 200) in pixels


@pytest.mark.quick
def test_img2img_attack_variants_consume_registered_protocol_parameters() -> None:
    """全图扩散攻击必须传入 source image 并消费各自冻结参数。"""

    runtime, img2img, _ = build_runtime()
    source = Image.new("RGB", (16, 16), (20, 30, 40))
    attack_names = (
        "img2img_regeneration",
        "sdedit_regeneration",
        "diffusion_purification",
        "global_editing_attack",
        "visual_paraphrase_attack",
    )

    for attack_index, attack_name in enumerate(attack_names):
        runtime.apply(
            source,
            diffusion_attack_spec(attack_name),
            seed=100 + attack_index,
            prompt_text="a lighthouse at sunrise",
        )

    assert len(img2img.calls) == len(attack_names)
    assert all(call["image"] is source for call in img2img.calls)
    assert img2img.calls[0]["strength"] == pytest.approx(0.35)
    assert img2img.calls[1]["strength"] == pytest.approx(0.45)
    assert img2img.calls[2]["prompt"] == ""
    assert img2img.calls[2]["num_inference_steps"] == 20
    assert img2img.calls[2]["guidance_scale"] == pytest.approx(1.0)
    assert "changed global style and lighting" in str(img2img.calls[3]["prompt"])
    assert "different visual composition" in str(img2img.calls[4]["prompt"])


@pytest.mark.quick
def test_detector_guided_removal_returns_lowest_measured_score_candidate() -> None:
    """对抗去水印必须逐候选调用真实分数函数并返回最低分图像。"""

    runtime, img2img, _ = build_runtime()
    source = Image.new("RGB", (16, 16), (20, 30, 40))
    source.candidate_index = 0
    measured_indices: list[int] = []

    def detection_score(image: Image.Image) -> float:
        candidate_index = int(getattr(image, "candidate_index", 0))
        measured_indices.append(candidate_index)
        return -float(candidate_index)

    execution = runtime.apply(
        source,
        diffusion_attack_spec("adversarial_removal_attack"),
        seed=500,
        prompt_text="a greenhouse",
        detection_score=detection_score,
    )

    assert len(img2img.calls) == 8
    assert measured_indices == list(range(9))
    assert int(execution.image.candidate_index) == 8
    assert len(execution.detector_query_trace) == 9
    assert execution.detector_query_trace[-1]["detection_score"] == pytest.approx(-8.0)


@pytest.mark.quick
def test_external_baseline_wrapper_delegates_to_shared_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """baseline 包装层不得保留第二套 latent 混合攻击实现。"""

    source = Image.new("RGB", (16, 16), (1, 2, 3))
    generated = Image.new("RGB", (16, 16), (4, 5, 6))
    captured: dict[str, object] = {}

    class FakeRuntime:
        def apply(self, image: Image.Image, spec: object, **kwargs: object) -> DiffusionAttackExecution:
            captured["image"] = image
            captured["spec"] = spec
            captured["kwargs"] = kwargs
            return DiffusionAttackExecution(
                image=generated,
                attack_name="img2img_regeneration",
                attack_implementation="sd3_img2img",
                attack_seed_random=int(kwargs["seed"]),
                effective_parameters={"denoise_strength": 0.35},
            )

    monkeypatch.setattr(
        DiffusionAttackRuntime,
        "from_text_to_image_pipeline",
        classmethod(lambda cls, pipeline, config: FakeRuntime()),
    )
    pipe = SimpleNamespace()

    attacked, implementation, trace = apply_regeneration_image_attack(
        source,
        attack_family="img2img_regeneration",
        seed=31,
        pipe=pipe,
        prompt="a ceramic fox",
        size=16,
        device="cpu",
    )

    assert attacked.tobytes() == generated.tobytes()
    assert implementation == "sd3_img2img"
    assert trace["attack_seed_random"] == 31
    assert captured["image"].size == source.size
    assert "latents" not in captured["kwargs"]


@pytest.mark.quick
def test_attack_name_normalization_rejects_non_protocol_aliases() -> None:
    """正式入口只接受当前协议名称, 不接受历史简称。"""

    assert normalize_attack_request("rotation") == "rotation"
    with pytest.raises(ValueError, match="unsupported_sd35_adapter_attack"):
        normalize_attack_request("rotate")
