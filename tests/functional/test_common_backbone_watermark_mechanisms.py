"""验证 common-backbone 外部基线载体和 Gaussian ChaCha20 科学算子."""

from __future__ import annotations

import inspect
from typing import Any, Callable

import pytest

from external_baseline.primary.gaussian_shading.adapter.method_faithful_sd35 import (
    GaussianShadingWatermark,
    chacha20_decrypt,
    chacha20_encrypt,
)
from external_baseline.primary.shallow_diffuse.adapter.method_faithful_sd35 import (
    build_fixed_watermark_carrier as build_shallow_diffuse_carrier,
    fuse_shallow_diffuse_watermark_channels,
    generate_shallow_diffuse_latent_pair,
    invert_flow_matching_to_edit_timestep,
    resolve_shallow_diffuse_edit_timestep,
    run_shallow_diffuse_method_faithful_adapter,
)
from external_baseline.primary.tree_ring.adapter.method_faithful_sd35 import (
    build_fixed_watermark_carrier as build_tree_ring_carrier,
    run_tree_ring_method_faithful_adapter,
)


pytestmark = pytest.mark.quick


class FakeFlowMatchScheduler:
    """提供可精确检查 schedule index 的确定性 FlowMatch Euler schedule."""

    def __init__(self, torch_module: Any) -> None:
        """保存 torch 模块并初始化 scheduler 配置."""

        self._torch = torch_module
        self.config: dict[str, object] = {
            "use_dynamic_shifting": False,
            "stochastic_sampling": False,
        }
        self.timesteps = torch_module.tensor([])
        self.sigmas = torch_module.tensor([])

    def set_timesteps(
        self,
        step_count: int,
        *,
        device: str,
        **_kwargs: object,
    ) -> None:
        """构造从高噪声到数据端的线性测试 schedule."""

        self.timesteps = self._torch.arange(
            int(step_count),
            0,
            -1,
            device=device,
            dtype=self._torch.float32,
        )
        self.sigmas = self._torch.linspace(
            1.0,
            0.0,
            int(step_count) + 1,
            device=device,
            dtype=self._torch.float32,
        )


class FakeFlowMatchTransformer:
    """记录真实两段调用的 timestep 与 Prompt conditioning batch."""

    def __init__(self, torch_module: Any) -> None:
        """初始化配置和调用记录."""

        self._torch = torch_module
        self.config = type("TransformerConfig", (), {"patch_size": 1})()
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        hidden_states: Any,
        timestep: Any,
        encoder_hidden_states: Any,
        pooled_projections: Any,
        joint_attention_kwargs: object,
        return_dict: bool,
    ) -> tuple[Any]:
        """以 conditioning 值构造速度场并记录真实调用边界."""

        del pooled_projections, joint_attention_kwargs, return_dict
        conditioning = encoder_hidden_states[:, 0, 0]
        self.calls.append(
            {
                "timestep": float(timestep[0].item()),
                "latent_batch": int(hidden_states.shape[0]),
                "conditioning": [
                    float(value) for value in conditioning.detach().cpu().tolist()
                ],
            }
        )
        velocity = conditioning.reshape(-1, 1, 1, 1).expand_as(hidden_states)
        return (velocity.to(dtype=hidden_states.dtype),)


class FakeFlowMatchPipeline:
    """提供无需模型权重的 SD3 Prompt 编码与 FlowMatch 组件."""

    def __init__(self, torch_module: Any) -> None:
        """初始化 scheduler, transformer 与 CPU 执行设备."""

        self._torch = torch_module
        self._execution_device = "cpu"
        self._joint_attention_kwargs = None
        self.scheduler = FakeFlowMatchScheduler(torch_module)
        self.transformer = FakeFlowMatchTransformer(torch_module)

    def encode_prompt(
        self,
        *,
        prompt: str,
        do_classifier_free_guidance: bool,
        **_kwargs: object,
    ) -> tuple[Any, Any, Any, Any]:
        """为正文 Prompt 和空 Prompt 返回可区分的确定性 embedding."""

        positive_value = 1.0 if prompt else 0.25
        positive = self._torch.tensor([[[positive_value]]], dtype=self._torch.float32)
        pooled_positive = self._torch.tensor(
            [[positive_value]],
            dtype=self._torch.float32,
        )
        if not do_classifier_free_guidance:
            return positive, None, pooled_positive, None
        negative = self._torch.zeros_like(positive)
        pooled_negative = self._torch.zeros_like(pooled_positive)
        return positive, negative, pooled_positive, pooled_negative


@pytest.mark.parametrize(
    ("builder", "parameters"),
    (
        (
            build_tree_ring_carrier,
            {
                "pattern": "ring",
                "channel": 0,
                "radius": 3,
            },
        ),
        (
            build_shallow_diffuse_carrier,
            {
                "mask_shape": "circle",
                "radius": 3,
                "inner_radius": 1,
                "channel": 0,
                "pattern": "complex_rand",
            },
        ),
    ),
)
def test_fixed_carrier_digest_is_shared_by_all_prompts_and_resume(
    builder: Callable[..., tuple[Any, Any, str]],
    parameters: dict[str, object],
) -> None:
    """固定 watermark seed 必须在所有 Prompt 和恢复会话重建同一载体."""

    torch = pytest.importorskip("torch")
    shape = (1, 4, 8, 8)
    first_mask, first_carrier, first_digest = builder(
        shape,
        watermark_seed=1729,
        device="cpu",
        **parameters,
    )
    resumed_mask, resumed_carrier, resumed_digest = builder(
        shape,
        watermark_seed=1729,
        device="cpu",
        **parameters,
    )
    _, changed_carrier, changed_digest = builder(
        shape,
        watermark_seed=1730,
        device="cpu",
        **parameters,
    )

    assert torch.equal(first_mask, resumed_mask)
    assert torch.equal(first_carrier, resumed_carrier)
    assert first_digest == resumed_digest
    assert len(first_digest) == 64
    assert not torch.equal(first_carrier, changed_carrier)
    assert first_digest != changed_digest
    prompt_provenance = [
        {
            "generation_seed_random": prompt_index,
            "watermark_seed_random": 1729,
            "watermark_carrier_digest_random": first_digest,
        }
        for prompt_index in range(4)
    ]
    assert {row["watermark_carrier_digest_random"] for row in prompt_provenance} == {
        first_digest
    }


@pytest.mark.parametrize(
    "runner",
    (
        run_tree_ring_method_faithful_adapter,
        run_shallow_diffuse_method_faithful_adapter,
    ),
)
def test_fixed_carrier_is_scheduled_before_prompt_loop(runner: object) -> None:
    """正式 adapter 必须在 Prompt 循环外构造一次固定载体."""

    source = inspect.getsource(runner)
    carrier_position = source.index("build_fixed_watermark_carrier(")
    prompt_loop_position = source.index("for index, row in enumerate(prompt_rows")

    assert carrier_position < prompt_loop_position
    assert source.count('"watermark_carrier_digest_random"') >= 3
    assert "int(args.watermark_seed) + index - 1" not in source


def test_shallow_diffuse_edit_timestep_uses_official_floor_semantics() -> None:
    """edit timestep 必须按官方 int(fraction * steps) 计算, 不能四舍五入."""

    assert resolve_shallow_diffuse_edit_timestep(20, 0.24) == (4, 16)
    with pytest.raises(ValueError, match="完整 schedule 内部"):
        resolve_shallow_diffuse_edit_timestep(20, 0.0)
    with pytest.raises(ValueError, match="完整 schedule 内部"):
        resolve_shallow_diffuse_edit_timestep(20, 1.0)
    runner_source = inspect.getsource(run_shallow_diffuse_method_faithful_adapter)
    assert "int(args.num_inversion_steps) != int(args.num_inference_steps)" in runner_source
    assert "callback_on_step_end" not in runner_source


def test_shallow_diffuse_split_protocol_and_channel_fusion() -> None:
    """同一 base latent 必须在 edit 注入后以 guidance=1 完成并仅保留水印通道."""

    torch = pytest.importorskip("torch")
    pipe = FakeFlowMatchPipeline(torch)
    base_latents = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
    mask = torch.zeros_like(base_latents, dtype=torch.bool)
    mask[:, 1] = True
    patch = torch.full_like(base_latents, 5.0)

    clean_latents, watermarked_latents, protocol = (
        generate_shallow_diffuse_latent_pair(
            pipe,
            "a ceramic fox",
            base_latents=base_latents,
            mask=mask,
            patch=patch,
            num_inference_steps=10,
            guidance_scale=4.0,
            edit_fraction=0.2,
            injection="seed",
            watermark_channel=1,
        )
    )

    assert protocol == {
        "injection_mode": "edit_timestep_split_flow_matching",
        "edit_timestep": 2,
        "edit_schedule_index": 8,
        "pre_edit_guidance_scale": 4.0,
        "post_edit_guidance_scale": 1.0,
        "watermark_channel": 1,
        "channel_fusion": "watermark_channel_from_watermarked_branch_other_channels_from_clean_branch",
    }
    assert torch.equal(base_latents, torch.zeros_like(base_latents))
    assert torch.equal(watermarked_latents[:, 0], clean_latents[:, 0])
    assert torch.equal(watermarked_latents[:, 2:], clean_latents[:, 2:])
    assert not torch.equal(watermarked_latents[:, 1], clean_latents[:, 1])
    assert len(pipe.transformer.calls) == 12
    assert [call["timestep"] for call in pipe.transformer.calls[:8]] == [
        10.0,
        9.0,
        8.0,
        7.0,
        6.0,
        5.0,
        4.0,
        3.0,
    ]
    assert [call["timestep"] for call in pipe.transformer.calls[8:10]] == [
        2.0,
        1.0,
    ]
    assert [call["timestep"] for call in pipe.transformer.calls[10:]] == [
        2.0,
        1.0,
    ]
    assert all(
        call["latent_batch"] == 2
        and call["conditioning"] == [0.0, 1.0]
        for call in pipe.transformer.calls[:8]
    )
    assert all(
        call["latent_batch"] == 1 and call["conditioning"] == [1.0]
        for call in pipe.transformer.calls[8:]
    )
    assert "callback_on_step_end" not in inspect.getsource(
        generate_shallow_diffuse_latent_pair
    )


def test_shallow_diffuse_detection_stops_at_same_edit_timestep() -> None:
    """仅图像检测反演必须执行 edit_timestep 个逆向 step 后停止."""

    torch = pytest.importorskip("torch")
    pipe = FakeFlowMatchPipeline(torch)
    image_latents = torch.zeros((1, 4, 2, 2), dtype=torch.float32)

    recovered = invert_flow_matching_to_edit_timestep(
        pipe,
        image_latents,
        num_inference_steps=10,
        edit_timestep=2,
    )

    assert len(pipe.transformer.calls) == 2
    assert [call["timestep"] for call in pipe.transformer.calls] == [1.0, 2.0]
    assert all(
        call["latent_batch"] == 1 and call["conditioning"] == [0.25]
        for call in pipe.transformer.calls
    )
    assert torch.allclose(recovered, torch.full_like(recovered, 0.05))


def test_shallow_diffuse_channel_fusion_supports_declared_channel_modes() -> None:
    """单通道模式保留 clean 其余通道, 全通道模式使用完整水印分支."""

    torch = pytest.importorskip("torch")
    clean = torch.zeros((1, 4, 2, 2))
    watermarked = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)

    single_channel = fuse_shallow_diffuse_watermark_channels(
        clean,
        watermarked,
        watermark_channel=2,
    )
    all_channels = fuse_shallow_diffuse_watermark_channels(
        clean,
        watermarked,
        watermark_channel=-1,
    )

    assert torch.equal(single_channel[:, :2], clean[:, :2])
    assert torch.equal(single_channel[:, 2], watermarked[:, 2])
    assert torch.equal(single_channel[:, 3], clean[:, 3])
    assert torch.equal(all_channels, watermarked)


def test_chacha20_matches_pycryptodome_counter_and_ietf_cipher_vectors() -> None:
    """ChaCha20 必须匹配 PyCryptodome 默认 counter=0 和 IETF counter=1."""

    default_expected = bytes.fromhex(
        "76b8e0ada0f13d90405d6ae55386bd28"
        "bdd219b8a08ded1aa836efcc8b770dc7"
        "da41597c5157488d7724e03fb8d84a37"
        "6a43b8f41518a11cc387b669b2ee6586"
    )
    assert chacha20_encrypt(
        bytes(64),
        key=bytes(32),
        nonce=bytes(12),
    ) == default_expected

    key = bytes(range(32))
    nonce = bytes.fromhex("000000090000004a00000000")
    expected = bytes.fromhex(
        "10f1e7e4d13b5915500fdd1fa32071c4"
        "c7d1f4c733c068030422aa9ac3d46c4e"
        "d2826446079faa0914c2d705d98b02a2"
        "b5129cd1de164eb9cbd083e8a2503c4e"
    )

    ciphertext = chacha20_encrypt(
        bytes(64),
        key=key,
        nonce=nonce,
        initial_counter=1,
    )
    assert ciphertext == expected
    assert chacha20_decrypt(
        ciphertext,
        key=key,
        nonce=nonce,
        initial_counter=1,
    ) == bytes(64)


def test_gaussian_shading_strict_pair_reuses_exact_base_magnitude() -> None:
    """Gaussian strict pair 必须复用唯一 base latent, 并执行 ChaCha20 解密 voting."""

    torch = pytest.importorskip("torch")
    shape = (1, 4, 8, 8)
    watermark_seed_random = 9001
    first = GaussianShadingWatermark(
        latent_shape=shape,
        channel_copy=1,
        hw_copy=2,
        generator=torch.Generator(device="cpu").manual_seed(
            watermark_seed_random
        ),
        device="cpu",
    )
    resumed = GaussianShadingWatermark(
        latent_shape=shape,
        channel_copy=1,
        hw_copy=2,
        generator=torch.Generator(device="cpu").manual_seed(
            watermark_seed_random
        ),
        device="cpu",
    )
    clean_latents = torch.randn(
        shape,
        generator=torch.Generator(device="cpu").manual_seed(37),
        dtype=torch.float32,
    )
    watermarked_latents = first.create_strict_paired_latents(clean_latents)
    resumed_latents = resumed.create_strict_paired_latents(clean_latents)

    assert torch.equal(clean_latents.abs(), watermarked_latents.abs())
    assert torch.equal(watermarked_latents, resumed_latents)
    assert torch.unique(first.watermark).numel() == 2
    assert torch.equal(
        first.expanded_watermark(),
        first.watermark.repeat(1, first.channel_copy, first.hw_copy, first.hw_copy),
    )
    assert first.score_latents(watermarked_latents) == pytest.approx(1.0)
    assert first.secret_material_digest_random == resumed.secret_material_digest_random
    assert first.chacha_message_digest_random == resumed.chacha_message_digest_random
    assert list(inspect.signature(first.create_strict_paired_latents).parameters) == [
        "clean_latents"
    ]

    identity = first.build_random_identity_random(
        generation_seed_random=37,
        watermark_seed_random=watermark_seed_random,
        base_latent_content_digest_random="a" * 64,
    )
    assert identity["generation_seed_random"] == 37
    assert identity["watermark_seed_random"] == watermark_seed_random
    assert identity["base_latent_content_digest_random"] == "a" * 64
    assert len(identity["gaussian_chacha_secret_material_digest_random"]) == 64
    assert len(identity["gaussian_chacha_message_digest_random"]) == 64
    assert not any("key" in field or "nonce" in field for field in identity)
