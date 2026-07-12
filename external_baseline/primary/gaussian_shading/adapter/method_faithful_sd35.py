"""Gaussian Shading 在 SD3.5 Medium 上的方法忠实适配器.

该模块位于 external baseline 适配层. watermark bit 经 ChaCha20 key/nonce 加密为
latent sign message. clean / watermarked 路线共享同一 Gaussian 样本的逐坐标幅值,
图像反演恢复 noise sign 后使用 ChaCha20 解密与 block voting 得到检测分数.

项目特定写法:
- SD3.5 Medium 使用 16-channel latent, 因此 message 与 watermark 的重复映射从
  官方 4-channel latent 显式推广到可配置通道数.
- observation 标记为 method-faithful adapter, 但仍不直接声明论文主张.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from experiments.runtime.image_metrics import measured_image_ssim, measured_score_retention
from experiments.protocol.attacks import attack_config_digest
from main.core.digest import build_stable_digest

from external_baseline.primary.sd35_method_faithful_common import (
    DEFAULT_SD35_MODEL_ID,
    DEFAULT_SD35_MODEL_REVISION,
    METHOD_FAITHFUL_ADAPTER_BOUNDARY,
    apply_formal_image_attack,
    canonical_attack_family,
    canonical_attack_name,
    derive_threshold,
    emit_adapter_progress,
    file_digest,
    formal_image_attack_config,
    load_prompt_rows,
    load_sd3_pipeline,
    observation_digest,
    prompt_text,
    row_id,
    safe_file_stem,
    score_image_latents,
    select_prompt_rows,
    split_name,
    validate_model_revision,
    validated_observation_attack_identity,
    write_json,
)
from external_baseline.primary.sd35_method_faithful_units import (
    aggregate_method_faithful_unit_records,
    apply_frozen_threshold,
    build_irreversible_random_material_digest,
    build_method_faithful_unit_context,
    build_method_faithful_unit_spec,
    load_completed_method_faithful_unit,
    repository_relative_method_faithful_path,
    resolve_method_faithful_output_path,
    threshold_independent_observation,
    write_completed_method_faithful_unit,
)

BASELINE_ID = "gaussian_shading"
DEFAULT_SCORE_NAME = "gaussian_shading_bit_vote_accuracy"
CHACHA20_PROTOCOL = "chacha20_ietf_256_bit_key_96_bit_nonce"


def _rotate_left32(value: int, count: int) -> int:
    """执行 ChaCha20 定义的32位循环左移."""

    masked = value & 0xFFFFFFFF
    return ((masked << count) & 0xFFFFFFFF) | (masked >> (32 - count))


def _chacha20_quarter_round(
    state: list[int],
    first: int,
    second: int,
    third: int,
    fourth: int,
) -> None:
    """就地执行一个 ChaCha20 quarter round."""

    state[first] = (state[first] + state[second]) & 0xFFFFFFFF
    state[fourth] = _rotate_left32(state[fourth] ^ state[first], 16)
    state[third] = (state[third] + state[fourth]) & 0xFFFFFFFF
    state[second] = _rotate_left32(state[second] ^ state[third], 12)
    state[first] = (state[first] + state[second]) & 0xFFFFFFFF
    state[fourth] = _rotate_left32(state[fourth] ^ state[first], 8)
    state[third] = (state[third] + state[fourth]) & 0xFFFFFFFF
    state[second] = _rotate_left32(state[second] ^ state[third], 7)


def chacha20_encrypt(
    plaintext: bytes,
    *,
    key: bytes,
    nonce: bytes,
    initial_counter: int = 0,
) -> bytes:
    """实现官方 use_chacha 路线使用的32字节 key 与12字节 nonce ChaCha20."""

    if len(key) != 32:
        raise ValueError("ChaCha20 key 必须恰好为32字节")
    if len(nonce) != 12:
        raise ValueError("ChaCha20 nonce 必须恰好为12字节")
    if not 0 <= int(initial_counter) <= 0xFFFFFFFF:
        raise ValueError("ChaCha20 initial_counter 超出32位范围")
    constants = b"expand 32-byte k"
    prefix = [
        int.from_bytes(constants[offset : offset + 4], "little")
        for offset in range(0, 16, 4)
    ]
    key_words = [
        int.from_bytes(key[offset : offset + 4], "little")
        for offset in range(0, 32, 4)
    ]
    nonce_words = [
        int.from_bytes(nonce[offset : offset + 4], "little")
        for offset in range(0, 12, 4)
    ]
    output = bytearray()
    for block_index, offset in enumerate(range(0, len(plaintext), 64)):
        counter = int(initial_counter) + block_index
        if counter > 0xFFFFFFFF:
            raise ValueError("ChaCha20 counter 已耗尽")
        state = [*prefix, *key_words, counter, *nonce_words]
        working = list(state)
        for _ in range(10):
            _chacha20_quarter_round(working, 0, 4, 8, 12)
            _chacha20_quarter_round(working, 1, 5, 9, 13)
            _chacha20_quarter_round(working, 2, 6, 10, 14)
            _chacha20_quarter_round(working, 3, 7, 11, 15)
            _chacha20_quarter_round(working, 0, 5, 10, 15)
            _chacha20_quarter_round(working, 1, 6, 11, 12)
            _chacha20_quarter_round(working, 2, 7, 8, 13)
            _chacha20_quarter_round(working, 3, 4, 9, 14)
        key_stream = b"".join(
            ((working[index] + state[index]) & 0xFFFFFFFF).to_bytes(4, "little")
            for index in range(16)
        )
        block = plaintext[offset : offset + 64]
        output.extend(value ^ key_stream[index] for index, value in enumerate(block))
    return bytes(output)


def chacha20_decrypt(
    ciphertext: bytes,
    *,
    key: bytes,
    nonce: bytes,
    initial_counter: int = 0,
) -> bytes:
    """使用相同 ChaCha20 keystream 解密 ciphertext."""

    return chacha20_encrypt(
        ciphertext,
        key=key,
        nonce=nonce,
        initial_counter=initial_counter,
    )


def _pack_binary_tensor(value: Any) -> bytes:
    """按官方 numpy.packbits 的 big-endian 位顺序压缩二值 tensor."""

    import numpy as np
    import torch

    array = (
        value.detach()
        .to(device="cpu", dtype=torch.uint8)
        .contiguous()
        .numpy()
        .reshape(-1)
    )
    return np.packbits(array, bitorder="big").tobytes()


def _unpack_binary_bytes(value: bytes, *, latent_shape: tuple[int, int, int, int], device: str) -> Any:
    """按官方 numpy.unpackbits 的 big-endian 位顺序恢复二值 tensor."""

    import numpy as np
    import torch

    bit_count = math.prod(latent_shape)
    unpacked = np.unpackbits(
        np.frombuffer(value, dtype=np.uint8),
        bitorder="big",
    )[:bit_count].copy()
    return torch.from_numpy(unpacked).reshape(latent_shape).to(
        device=device,
        dtype=torch.int64,
    )


class GaussianShadingWatermark:
    """保存 SD3.5 Gaussian Shading 的 ChaCha20 message 与 voting 状态."""

    def __init__(
        self,
        *,
        latent_shape: tuple[int, int, int, int],
        channel_copy: int,
        hw_copy: int,
        generator: Any,
        device: str,
    ) -> None:
        """初始化 watermark, ChaCha20 key/nonce 与重复映射参数."""

        import torch

        batch, channels, height, width = latent_shape
        if batch != 1:
            raise ValueError("gaussian_shading_sd35_adapter_currently_requires_batch_one")
        if channels % int(channel_copy) != 0 or height % int(hw_copy) != 0 or width % int(hw_copy) != 0:
            raise ValueError("gaussian_shading_copy_factors_must_divide_sd35_latent_shape")
        self.latent_shape = latent_shape
        self.channel_copy = int(channel_copy)
        self.hw_copy = int(hw_copy)
        self.device = device
        key_values = torch.randint(
            0,
            256,
            (32,),
            generator=generator,
            device="cpu",
            dtype=torch.int64,
        )
        nonce_values = torch.randint(
            0,
            256,
            (12,),
            generator=generator,
            device="cpu",
            dtype=torch.int64,
        )
        self._key_material = bytes(int(value) for value in key_values.tolist())
        self._nonce_material = bytes(int(value) for value in nonce_values.tolist())
        self.watermark = torch.randint(
            0,
            2,
            (batch, channels // self.channel_copy, height // self.hw_copy, width // self.hw_copy),
            generator=generator,
            device="cpu",
            dtype=torch.int64,
        ).to(device=device)
        self.vote_threshold = max(1, self.channel_copy * self.hw_copy * self.hw_copy // 2)
        encrypted_message = chacha20_encrypt(
            _pack_binary_tensor(self.expanded_watermark()),
            key=self._key_material,
            nonce=self._nonce_material,
        )
        self.encrypted_message = _unpack_binary_bytes(
            encrypted_message,
            latent_shape=self.latent_shape,
            device=device,
        )
        self.secret_material_digest_random = build_irreversible_random_material_digest(
            self._key_material,
            self._nonce_material,
            self.watermark,
        )
        self.chacha_message_digest_random = build_irreversible_random_material_digest(
            encrypted_message
        )

    def expanded_watermark(self) -> Any:
        """按官方 tensor.repeat 调度把低维 watermark 平铺到完整 latent."""

        return self.watermark.repeat(
            1,
            self.channel_copy,
            self.hw_copy,
            self.hw_copy,
        )

    def create_strict_paired_latents(self, clean_latents: Any) -> Any:
        """以 clean latent 的同一幅值和 ChaCha20 message 符号构造条件采样."""

        import torch

        if tuple(clean_latents.shape) != self.latent_shape:
            raise ValueError("Gaussian Shading strict pair 的 latent 形状不匹配")
        signs = self.encrypted_message.to(dtype=torch.float32) * 2.0 - 1.0
        return (signs * torch.abs(clean_latents.float())).to(dtype=clean_latents.dtype)

    def build_random_identity_random(
        self,
        *,
        generation_seed_random: int,
        watermark_seed_random: int,
        clean_base_latent_digest_random: str,
    ) -> dict[str, Any]:
        """构造只含 seed 和不可逆摘要的逐 Prompt 随机来源."""

        return {
            "generation_seed_random": int(generation_seed_random),
            "watermark_seed_random": int(watermark_seed_random),
            "clean_base_latent_digest_random": str(
                clean_base_latent_digest_random
            ),
            "gaussian_chacha_secret_material_digest_random": (
                self.secret_material_digest_random
            ),
            "gaussian_chacha_message_digest_random": (
                self.chacha_message_digest_random
            ),
        }

    def decode_recovered_watermark(self, reversed_latents: Any) -> Any:
        """从反演 latent 的 sign 恢复 watermark bit 并执行 block voting。"""

        import torch

        reversed_message = (reversed_latents.float() > 0).to(dtype=torch.int64)
        decrypted_message = chacha20_decrypt(
            _pack_binary_tensor(reversed_message),
            key=self._key_material,
            nonce=self._nonce_material,
        )
        decoded = _unpack_binary_bytes(
            decrypted_message,
            latent_shape=self.latent_shape,
            device=self.device,
        )
        batch, channels, height, width = self.latent_shape
        split_dim1 = torch.cat(torch.split(decoded, channels // self.channel_copy, dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, height // self.hw_copy, dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, width // self.hw_copy, dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        return (vote > self.vote_threshold).to(dtype=torch.int64).unsqueeze(0)

    def score_latents(self, reversed_latents: Any) -> float:
        """计算 recovered watermark 与原 watermark 的 bit accuracy。"""

        recovered = self.decode_recovered_watermark(reversed_latents)
        return float((recovered == self.watermark).float().mean().detach().cpu().item())


def build_observation(
    *,
    event_id: str,
    score: float,
    threshold: float,
    threshold_source: str,
    row: dict[str, Any],
    index: int,
    sample_role: str,
    attack_family: str,
    attack_condition: str,
    image_id: str,
    image_path: str,
    image_digest: str,
    latent_shape: tuple[int, int, int, int],
    execution_device: str,
    model_id: str,
    model_revision: str,
    quality_score: float,
    score_retention: float,
    attack_id: str = "",
    resource_profile: str = "",
    attack_config_digest_value: str = "",
) -> dict[str, Any]:
    """构造统一 baseline observation。"""

    detection_decision = bool(float(score) >= float(threshold))
    attack_identity = validated_observation_attack_identity(
        sample_role=sample_role,
        attack_family=attack_family,
        attack_name=attack_condition,
        attack_id=attack_id,
        resource_profile=resource_profile,
        attack_config_digest_value=attack_config_digest_value,
    )
    return observation_digest(
        {
            "event_id": event_id,
            "baseline_id": BASELINE_ID,
            "score": float(score),
            "threshold": float(threshold),
            "score_name": DEFAULT_SCORE_NAME,
            "higher_is_positive": True,
            "detection_decision": detection_decision,
            "final_decision": detection_decision,
            "split": split_name(row),
            "sample_role": sample_role,
            "attack_family": attack_family,
            "attack_name": attack_condition,
            "attack_condition": attack_condition,
            **attack_identity,
            "prompt_id": row_id(row, index, "prompt_id", "prompt"),
            "prompt_text": prompt_text(row),
            "image_id": image_id,
            "image_path": image_path,
            "image_digest": image_digest,
            "producer_id": "gaussian_shading_method_faithful_sd35_adapter",
            "producer_role": "external_baseline_method_faithful_adapter",
            "adapter_boundary": METHOD_FAITHFUL_ADAPTER_BOUNDARY,
            "threshold_source": threshold_source,
            "formal_result_claim": False,
            "supports_paper_claim": False,
            "generation_model_id": model_id,
            "generation_model_revision": validate_model_revision(model_revision),
            "latent_shape": list(latent_shape),
            "execution_device": execution_device,
            "quality_score": float(quality_score),
            "score_retention": float(score_retention),
        }
    )


def score_image(pipe: Any, image: Any, *, size: int, device: str, watermark: GaussianShadingWatermark, num_inversion_steps: int) -> float:
    """把图像重新编码、反演并计算 Gaussian Shading bit vote 分数。"""

    reversed_latents = score_image_latents(pipe, image, size=int(size), device=device, num_inversion_steps=int(num_inversion_steps))
    return watermark.score_latents(reversed_latents)


def run_gaussian_shading_method_faithful_adapter(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """执行 Gaussian Shading SD3.5 方法忠实适配流程。"""

    import torch
    from PIL import Image

    prompt_rows = select_prompt_rows(load_prompt_rows(args.prompt_plan), args.max_samples)
    attack_families = [item.strip() for item in str(args.attack_families or "").split(",") if item.strip()]
    test_prompt_count = sum(split_name(row) == "test" for row in prompt_rows)
    progress_total = max(1, 1 + len(prompt_rows) + len(attack_families) * test_prompt_count * 2)
    progress_completed = 0
    emit_adapter_progress(
        baseline_id=BASELINE_ID,
        operation="load_sd3_pipeline",
        completed=progress_completed,
        total=progress_total,
        profile=f"operation=load_sd3_pipeline prompts={len(prompt_rows)} attacks={len(attack_families)}",
    )
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    if args.require_cuda and device != "cuda":
        raise RuntimeError("gaussian_shading_method_faithful_sd35_requires_cuda")

    pipe = load_sd3_pipeline(
        model_id=args.model_id,
        model_revision=args.model_revision,
        device=device,
        torch_dtype_name=args.torch_dtype,
        adapter_class_name="GaussianShadingInversionStableDiffusion3Pipeline",
    )
    progress_completed += 1
    emit_adapter_progress(
        baseline_id=BASELINE_ID,
        operation="source_pair_generation",
        completed=progress_completed,
        total=progress_total,
        profile=f"operation=source_pair_generation prompt=0/{len(prompt_rows)}",
    )
    artifact_root = Path(args.artifact_root) if args.artifact_root else Path(args.out).resolve().parent / "artifacts"
    clean_dir = artifact_root / "images" / "clean"
    watermarked_dir = artifact_root / "images" / "watermarked"
    attacked_dir = artifact_root / "images" / "attacked"
    clean_dir.mkdir(parents=True, exist_ok=True)
    watermarked_dir.mkdir(parents=True, exist_ok=True)
    attacked_dir.mkdir(parents=True, exist_ok=True)

    latent_shape = (1, int(args.latent_channels), max(1, int(args.height) // 8), max(1, int(args.width) // 8))
    run_config = {
        "adapter_boundary": METHOD_FAITHFUL_ADAPTER_BOUNDARY,
        "model_id": args.model_id,
        "model_revision": validate_model_revision(args.model_revision),
        "torch_dtype": str(args.torch_dtype),
        "latent_shape": list(latent_shape),
        "height": int(args.height),
        "width": int(args.width),
        "num_inference_steps": int(args.num_inference_steps),
        "num_inversion_steps": int(args.num_inversion_steps),
        "guidance_scale": float(args.guidance_scale),
        "target_fpr": float(args.target_fpr),
        "explicit_threshold": args.threshold,
        "attack_families": list(attack_families),
        "attack_execution_split": "test",
        "prompt_count": len(prompt_rows),
        "test_prompt_count": test_prompt_count,
        "seed": int(args.seed),
        "watermark_seed": int(args.watermark_seed),
        "channel_copy": int(args.channel_copy),
        "hw_copy": int(args.hw_copy),
        "message_encryption": CHACHA20_PROTOCOL,
        "strict_pair_shared_magnitude": True,
        "prompt_plan_digest": build_stable_digest(prompt_rows),
    }
    unit_context = build_method_faithful_unit_context(
        baseline_id=BASELINE_ID,
        artifact_root=artifact_root,
        run_config=run_config,
        execution_device=device,
        torch_module=torch,
    )
    observations_without_threshold: list[dict[str, Any]] = []
    image_pairs: list[dict[str, Any]] = []
    runtime_keys: dict[str, dict[str, Any]] = {}
    source_unit_specs: list[Any] = []
    source_unit_records: list[dict[str, Any]] = []

    for index, row in enumerate(prompt_rows, start=1):
        current_prompt = prompt_text(row)
        prompt_id = row_id(row, index, "prompt_id", "prompt")
        image_id = row_id(row, index, "image_id", "gaussian_shading_image")
        file_stem = safe_file_stem(image_id, f"gaussian_shading_image_{index:05d}")
        generation_seed_random = int(args.seed) + index - 1
        watermark_seed_random = int(args.watermark_seed) + index - 1
        latent_generator = torch.Generator(device=device).manual_seed(
            generation_seed_random
        )
        watermark_generator = torch.Generator(device="cpu").manual_seed(
            watermark_seed_random
        )

        watermark = GaussianShadingWatermark(
            latent_shape=latent_shape,
            channel_copy=int(args.channel_copy),
            hw_copy=int(args.hw_copy),
            generator=watermark_generator,
            device=device,
        )
        clean_latents = torch.randn(
            latent_shape,
            generator=latent_generator,
            device=device,
            dtype=pipe.transformer.dtype,
        )
        clean_base_latent_digest_random = build_irreversible_random_material_digest(
            clean_latents
        )
        source_random_identity_random = watermark.build_random_identity_random(
            generation_seed_random=generation_seed_random,
            watermark_seed_random=watermark_seed_random,
            clean_base_latent_digest_random=clean_base_latent_digest_random,
        )
        source_unit_spec = build_method_faithful_unit_spec(
            unit_context,
            unit_kind="source_pair",
            row=row,
            index=index,
            random_identity_random=source_random_identity_random,
            unit_parameters={
                "image_id": image_id,
                "latent_shape": list(latent_shape),
                "message_encryption": CHACHA20_PROTOCOL,
                "strict_pair_shared_magnitude": True,
            },
        )
        source_unit_specs.append(source_unit_spec)
        completed_source_unit = load_completed_method_faithful_unit(
            unit_context,
            source_unit_spec,
        )
        if completed_source_unit is not None:
            unit_data = completed_source_unit["unit_data"]
            image_pair = dict(unit_data["image_pair"])
            source_observations = [
                dict(item) for item in unit_data["observations_without_threshold"]
            ]
            if len(source_observations) != 2:
                raise ValueError("Gaussian Shading Prompt 源图完成单元必须包含两条连续分数")
            image_pairs.append(image_pair)
            observations_without_threshold.extend(source_observations)
            runtime_keys[image_id] = {
                "watermark": watermark,
                "row": row,
                "row_index": index,
                "clean_score": float(source_observations[0]["score"]),
                "watermarked_score": float(source_observations[1]["score"]),
                "source_random_identity_random": source_random_identity_random,
            }
            source_unit_records.append(completed_source_unit)
            progress_completed += 1
            emit_adapter_progress(
                baseline_id=BASELINE_ID,
                operation="source_pair_resume",
                completed=progress_completed,
                total=progress_total,
                profile=f"operation=source_pair_resume prompt={index}/{len(prompt_rows)} image_id={image_id}",
            )
            continue
        watermarked_latents = watermark.create_strict_paired_latents(clean_latents)
        if not torch.equal(clean_latents.abs(), watermarked_latents.abs()):
            raise RuntimeError("Gaussian Shading strict pair 的逐坐标幅值不一致")

        clean_image = pipe(
            current_prompt,
            guidance_scale=float(args.guidance_scale),
            num_inference_steps=int(args.num_inference_steps),
            height=int(args.height),
            width=int(args.width),
            latents=clean_latents,
        ).images[0]
        watermarked_image = pipe(
            current_prompt,
            guidance_scale=float(args.guidance_scale),
            num_inference_steps=int(args.num_inference_steps),
            height=int(args.height),
            width=int(args.width),
            latents=watermarked_latents,
        ).images[0]

        clean_path = clean_dir / f"{file_stem}_clean.png"
        watermarked_path = watermarked_dir / f"{file_stem}_gaussian_shading.png"
        clean_image.save(clean_path)
        watermarked_image.save(watermarked_path)
        clean_digest = file_digest(clean_path)
        watermarked_digest = file_digest(watermarked_path)
        clean_score = score_image(
            pipe,
            clean_image,
            size=int(args.height),
            device=device,
            watermark=watermark,
            num_inversion_steps=int(args.num_inversion_steps),
        )
        watermarked_score = score_image(
            pipe,
            watermarked_image,
            size=int(args.height),
            device=device,
            watermark=watermark,
            num_inversion_steps=int(args.num_inversion_steps),
        )
        pair_quality = measured_image_ssim(clean_image, watermarked_image)

        runtime_keys[image_id] = {
            "watermark": watermark,
            "row": row,
            "row_index": index,
            "clean_score": clean_score,
            "watermarked_score": watermarked_score,
            "source_random_identity_random": source_random_identity_random,
        }
        image_pairs.append(
            {
                "event_id": image_id,
                "image_id": image_id,
                "prompt_id": prompt_id,
                "prompt_text": current_prompt,
                "split": split_name(row),
                "clean_image_path": str(clean_path),
                "clean_image_digest": clean_digest,
                "watermarked_image_path": str(watermarked_path),
                "watermarked_image_digest": watermarked_digest,
                "baseline_id": BASELINE_ID,
                "generation_model_id": args.model_id,
                "generation_model_revision": args.model_revision,
                "latent_shape": list(latent_shape),
                "strict_pair_shared_magnitude": True,
                "clean_base_latent_digest_random": clean_base_latent_digest_random,
                "gaussian_chacha_secret_material_digest_random": (
                    watermark.secret_material_digest_random
                ),
                "gaussian_chacha_message_digest_random": (
                    watermark.chacha_message_digest_random
                ),
            }
        )
        observations_without_threshold.append(
            build_observation(
                event_id=f"{image_id}__clean_negative",
                score=clean_score,
                threshold=0.0,
                threshold_source="pending",
                row=row,
                index=index,
                sample_role="clean_negative",
                attack_family="clean",
                attack_condition="clean_none",
                image_id=image_id,
                image_path=str(clean_path),
                image_digest=clean_digest,
                latent_shape=latent_shape,
                execution_device=device,
                model_id=args.model_id,
                model_revision=args.model_revision,
                quality_score=1.0,
                score_retention=1.0,
            )
        )
        observations_without_threshold.append(
            build_observation(
                event_id=f"{image_id}__positive_source",
                score=watermarked_score,
                threshold=0.0,
                threshold_source="pending",
                row=row,
                index=index,
                sample_role="positive_source",
                attack_family="clean",
                attack_condition="clean_none",
                image_id=image_id,
                image_path=str(watermarked_path),
                image_digest=watermarked_digest,
                latent_shape=latent_shape,
                execution_device=device,
                model_id=args.model_id,
                model_revision=args.model_revision,
                quality_score=pair_quality,
                score_retention=1.0,
            )
        )
        source_observations = [
            threshold_independent_observation(item)
            for item in observations_without_threshold[-2:]
        ]
        observations_without_threshold[-2:] = source_observations
        completed_source_unit = write_completed_method_faithful_unit(
            unit_context,
            source_unit_spec,
            unit_data={
                "image_pair": image_pairs[-1],
                "observations_without_threshold": source_observations,
            },
            artifact_paths=(clean_path, watermarked_path),
        )
        canonical_source_data = completed_source_unit["unit_data"]
        image_pairs[-1] = dict(canonical_source_data["image_pair"])
        observations_without_threshold[-2:] = [
            dict(item)
            for item in canonical_source_data["observations_without_threshold"]
        ]
        source_unit_records.append(completed_source_unit)
        if device == "cuda":
            torch.cuda.empty_cache()
        progress_completed += 1
        emit_adapter_progress(
            baseline_id=BASELINE_ID,
            operation="source_pair_generation",
            completed=progress_completed,
            total=progress_total,
            profile=f"operation=source_pair_generation prompt={index}/{len(prompt_rows)} image_id={image_id}",
        )

    threshold, threshold_source = derive_threshold(observations_without_threshold, args.threshold, args.target_fpr)
    attacked_records: list[dict[str, Any]] = []
    attack_observations_without_threshold: list[dict[str, Any]] = []
    attack_unit_specs: list[Any] = []
    attack_unit_records: list[dict[str, Any]] = []
    for attack_family in attack_families:
        formal_attack_config = formal_image_attack_config(attack_family)
        attack_matrix_family = canonical_attack_family(attack_family)
        attack_matrix_name = canonical_attack_name(attack_family)
        formal_attack_digest = attack_config_digest(formal_attack_config)
        for pair_index, pair in enumerate(image_pairs, start=1):
            if str(pair.get("split", "")) != "test":
                continue
            image_id = str(pair["image_id"])
            runtime = runtime_keys[image_id]
            for role_name, source_path_field, source_digest_field, sample_role in (
                ("clean", "clean_image_path", "clean_image_digest", "attacked_negative"),
                ("watermarked", "watermarked_image_path", "watermarked_image_digest", "attacked_positive"),
            ):
                attack_seed = int(args.seed) + pair_index
                attack_unit_spec = build_method_faithful_unit_spec(
                    unit_context,
                    unit_kind=f"formal_attack_{attack_matrix_name}_{role_name}",
                    row=runtime["row"],
                    index=int(runtime["row_index"]),
                    random_identity_random={
                        **dict(runtime["source_random_identity_random"]),
                        "attack_seed_random": attack_seed,
                    },
                    unit_parameters={
                        "image_id": image_id,
                        "source_role": role_name,
                        "sample_role": sample_role,
                        "attack_id": formal_attack_config.attack_id,
                        "attack_config_digest": formal_attack_digest,
                        "frozen_threshold_digest": build_stable_digest(
                            {
                                "threshold": float(threshold),
                                "threshold_source": threshold_source,
                            }
                        ),
                    },
                )
                attack_unit_specs.append(attack_unit_spec)
                completed_attack_unit = load_completed_method_faithful_unit(
                    unit_context,
                    attack_unit_spec,
                )
                if completed_attack_unit is not None:
                    unit_data = completed_attack_unit["unit_data"]
                    attack_observations_without_threshold.append(
                        dict(unit_data["observation_without_threshold"])
                    )
                    attacked_records.append(dict(unit_data["attacked_record"]))
                    attack_unit_records.append(completed_attack_unit)
                    progress_completed += 1
                    emit_adapter_progress(
                        baseline_id=BASELINE_ID,
                        operation="formal_image_attack_resume",
                        completed=progress_completed,
                        total=progress_total,
                        profile=(
                            "operation=formal_image_attack_resume "
                            f"attack={attack_matrix_name} pair={pair_index}/{len(image_pairs)} role={role_name}"
                        ),
                        attack_name=attack_matrix_name,
                        image_id=image_id,
                    )
                    continue
                source_image_path = resolve_method_faithful_output_path(
                    unit_context,
                    pair[source_path_field],
                )
                with Image.open(source_image_path) as source_image:
                    attacked_image, attack_transform_name, attack_execution = apply_formal_image_attack(
                        source_image,
                        attack_family=attack_family,
                        seed=attack_seed,
                        pipe=pipe,
                        prompt=str(pair["prompt_text"]),
                        size=int(args.height),
                        device=device,
                        detection_score=lambda candidate: score_image(
                            pipe,
                            candidate,
                            size=int(args.height),
                            device=device,
                            watermark=runtime["watermark"],
                            num_inversion_steps=int(args.num_inversion_steps),
                        ),
                    )
                    attack_quality = measured_image_ssim(source_image, attacked_image)
                attacked_stem = safe_file_stem(f"{image_id}_{role_name}_{attack_matrix_name}", f"attacked_{pair_index:05d}")
                attacked_path = attacked_dir / f"{attacked_stem}.png"
                attacked_image.save(attacked_path)
                attacked_digest = file_digest(attacked_path)
                attacked_id = f"{image_id}__{role_name}__{attack_matrix_name}"
                score = score_image(
                    pipe,
                    attacked_image,
                    size=int(args.height),
                    device=device,
                    watermark=runtime["watermark"],
                    num_inversion_steps=int(args.num_inversion_steps),
                )
                attack_observation = threshold_independent_observation(
                    build_observation(
                        event_id=attacked_id,
                        score=score,
                        threshold=0.0,
                        threshold_source="pending",
                        row=runtime["row"],
                        index=int(runtime["row_index"]),
                        sample_role=sample_role,
                        attack_family=attack_matrix_family,
                        attack_condition=attack_matrix_name,
                        image_id=image_id,
                        image_path=str(attacked_path),
                        image_digest=attacked_digest,
                        latent_shape=latent_shape,
                        execution_device=device,
                        model_id=args.model_id,
                        model_revision=args.model_revision,
                        quality_score=attack_quality,
                        score_retention=measured_score_retention(
                            float(runtime[f"{role_name}_score"]),
                            score,
                        ),
                        attack_id=formal_attack_config.attack_id,
                        resource_profile=formal_attack_config.resource_profile,
                        attack_config_digest_value=formal_attack_digest,
                    )
                )
                attacked_record = {
                    "attacked_image_id": attacked_id,
                    "source_image_id": image_id,
                    "source_role": role_name,
                    "sample_role": sample_role,
                    "source_image_path": pair[source_path_field],
                    "source_image_digest": pair[source_digest_field],
                    "attacked_image_path": str(attacked_path),
                    "attacked_image_digest": attacked_digest,
                    "attack_family": attack_matrix_family,
                    "attack_name": attack_matrix_name,
                    "attack_condition": attack_matrix_name,
                    "attack_id": formal_attack_config.attack_id,
                    "resource_profile": formal_attack_config.resource_profile,
                    "attack_config_digest": formal_attack_digest,
                    "attack_transform_name": attack_transform_name,
                    "attack_execution": attack_execution,
                    "generation_model_id": args.model_id,
                    "generation_model_revision": args.model_revision,
                }
                attack_observations_without_threshold.append(attack_observation)
                attacked_records.append(attacked_record)
                completed_attack_unit = write_completed_method_faithful_unit(
                    unit_context,
                    attack_unit_spec,
                    unit_data={
                        "observation_without_threshold": attack_observation,
                        "attacked_record": attacked_record,
                    },
                    artifact_paths=(attacked_path,),
                )
                canonical_attack_data = completed_attack_unit["unit_data"]
                attack_observations_without_threshold[-1] = dict(
                    canonical_attack_data["observation_without_threshold"]
                )
                attacked_records[-1] = dict(canonical_attack_data["attacked_record"])
                attack_unit_records.append(completed_attack_unit)
                if device == "cuda":
                    torch.cuda.empty_cache()
                progress_completed += 1
                emit_adapter_progress(
                    baseline_id=BASELINE_ID,
                    operation="formal_image_attack",
                    completed=progress_completed,
                    total=progress_total,
                    profile=(
                        "operation=formal_image_attack "
                        f"attack={attack_matrix_name} pair={pair_index}/{len(image_pairs)} role={role_name}"
                    ),
                    attack_name=attack_matrix_name,
                    image_id=image_id,
                )

    unit_aggregate = aggregate_method_faithful_unit_records(
        unit_context,
        (*source_unit_records, *attack_unit_records),
        expected_specs=(*source_unit_specs, *attack_unit_specs),
    )
    observations = apply_frozen_threshold(
        (*observations_without_threshold, *attack_observations_without_threshold),
        threshold=threshold,
        threshold_source=threshold_source,
    )
    image_pairs_path = write_json(artifact_root / "gaussian_shading_image_pairs.json", image_pairs)
    attacked_manifest_path = write_json(
        artifact_root / "attacked_image_manifest.json",
        {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "attacked_images": attacked_records,
            "attacked_image_count": len(attacked_records),
        },
    )
    manifest = {
        "artifact_name": "gaussian_shading_method_faithful_sd35_adapter_manifest.json",
        "producer_id": "gaussian_shading_method_faithful_sd35_adapter",
        "baseline_id": BASELINE_ID,
        "adapter_boundary": METHOD_FAITHFUL_ADAPTER_BOUNDARY,
        "adapter_status": "method_faithful_sd35_adapter_ready",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "prompt_plan_path": repository_relative_method_faithful_path(
            unit_context, args.prompt_plan
        ),
        "baseline_observations_path": repository_relative_method_faithful_path(
            unit_context, args.out
        ),
        "artifact_root": repository_relative_method_faithful_path(
            unit_context, artifact_root
        ),
        "image_pairs_path": repository_relative_method_faithful_path(
            unit_context, image_pairs_path
        ),
        "attacked_image_manifest_path": repository_relative_method_faithful_path(
            unit_context, attacked_manifest_path
        ),
        "image_pair_count": len(image_pairs),
        "attacked_image_count": len(attacked_records),
        "test_prompt_count": test_prompt_count,
        "expected_formal_attack_unit_count": test_prompt_count * len(attack_families) * 2,
        "observation_count": len(observations),
        "latent_shape": list(latent_shape),
        "execution_device": device,
        "generation_protocol": {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "num_inference_steps": int(args.num_inference_steps),
            "guidance_scale": float(args.guidance_scale),
            "height": int(args.height),
            "width": int(args.width),
        },
        "detection_protocol": {
            "input_access_mode": "image_only",
            "num_inversion_steps": int(args.num_inversion_steps),
            "target_fpr": float(args.target_fpr),
        },
        "watermark_parameters": {
            "channel_copy": int(args.channel_copy),
            "hw_copy": int(args.hw_copy),
            "watermark_seed": int(args.watermark_seed),
            "message_encryption": CHACHA20_PROTOCOL,
            "message_mapping": "chacha20_sign_conditioned_gaussian_with_repeated_bit_voting",
            "strict_pair_shared_magnitude": True,
        },
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "run_config": unit_context.run_config,
        "run_config_digest": unit_context.run_config_digest,
        "stable_scientific_execution_identity": unit_context.stable_execution_identity,
        "stable_scientific_execution_identity_digest": unit_context.stable_execution_identity_digest,
        "method_faithful_source_identity": unit_context.source_identity,
        "method_faithful_source_identity_digest": unit_context.source_identity[
            "method_faithful_source_identity_digest"
        ],
        **unit_aggregate,
        "formal_result_claim": False,
        "supports_paper_claim": False,
    }
    manifest["adapter_digest"] = build_stable_digest(manifest)
    emit_adapter_progress(
        baseline_id=BASELINE_ID,
        operation="write_outputs",
        completed=progress_total,
        total=progress_total,
        profile=f"operation=write_outputs observations={len(observations)} attacked_images={len(attacked_records)}",
    )
    return observations, manifest


def build_parser() -> argparse.ArgumentParser:
    """构造 Gaussian Shading 方法忠实适配器参数解析器。"""

    parser = argparse.ArgumentParser(description="运行 Gaussian Shading SD3.5 方法忠实 external baseline adapter")
    parser.add_argument("--prompt-plan", required=True, help="共同 prompt 计划 JSON 路径")
    parser.add_argument("--out", required=True, help="baseline observations JSON 输出路径")
    parser.add_argument("--artifact-root", default=None, help="图像、攻击结果和 manifest 输出目录")
    parser.add_argument("--model-id", default=DEFAULT_SD35_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_SD35_MODEL_REVISION)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--num-inversion-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--watermark-seed", type=int, default=20260622)
    parser.add_argument("--channel-copy", type=int, default=1)
    parser.add_argument("--hw-copy", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--target-fpr", type=float, required=True)
    parser.add_argument("--attack-families", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def run_cli(argv: list[str] | None = None) -> None:
    """命令行入口, 写出 observations 与 manifest。"""

    args = build_parser().parse_args(argv)
    observations, manifest = run_gaussian_shading_method_faithful_adapter(args)
    write_json(args.out, observations)
    manifest_path = Path(args.out).with_name("gaussian_shading_method_faithful_sd35_adapter_manifest.json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    run_cli()
