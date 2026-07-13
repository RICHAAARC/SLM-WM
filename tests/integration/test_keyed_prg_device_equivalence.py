"""验证密钥 PRG 模板在 CPU 与 CUDA 间保持字节等价."""

from __future__ import annotations

import pytest
import torch

from experiments.protocol.formal_randomization import (
    build_canonical_sd35_base_latent,
)
from main.core.digest import tensor_content_sha256
from main.core.keyed_prg import KEYED_PRG_VERSION
from main.core.normal_quantile_table import NORMAL_QUANTILE_TABLE_SHA256
from main.methods.carrier import (
    build_low_frequency_template,
    build_tail_robust_template,
)
from main.methods.geometry.differentiable_attention import keyed_relation_signs


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 设备")
def test_keyed_prg_templates_are_exactly_device_independent() -> None:
    """同一公开输入在 CPU 与 CUDA 上必须重建完全相同的模板."""

    cpu_reference = torch.zeros((1, 2, 8, 8), dtype=torch.float16)
    cuda_reference = cpu_reference.to("cuda")

    cpu_lf = build_low_frequency_template(
        cpu_reference,
        "device-independent-key",
        "registered-model@revision",
    )
    cuda_lf = build_low_frequency_template(
        cuda_reference,
        "device-independent-key",
        "registered-model@revision",
    )
    cpu_tail, cpu_threshold, cpu_retained = build_tail_robust_template(
        cpu_reference,
        "device-independent-key",
        "registered-model@revision",
        0.20,
    )
    cuda_tail, cuda_threshold, cuda_retained = build_tail_robust_template(
        cuda_reference,
        "device-independent-key",
        "registered-model@revision",
        0.20,
    )

    assert torch.equal(cpu_lf, cuda_lf.cpu())
    assert torch.equal(cpu_tail, cuda_tail.cpu())
    assert cpu_threshold == cuda_threshold
    assert cpu_retained == cuda_retained

    cpu_attention = torch.zeros((8, 8), dtype=torch.float32)
    cuda_attention = cpu_attention.to("cuda")
    assert torch.equal(
        keyed_relation_signs(
            cpu_attention,
            "device-independent-key",
            "transformer_blocks.0.attn",
            KEYED_PRG_VERSION,
        ),
        keyed_relation_signs(
            cuda_attention,
            "device-independent-key",
            "transformer_blocks.0.attn",
            KEYED_PRG_VERSION,
        ).cpu(),
    )


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 设备")
def test_formal_base_latent_cpu_to_cuda_preserves_frozen_identity() -> None:
    """Colab 必须复验正式 shape 的 CPU 规范生成和 CUDA 搬运逐字节一致性."""

    common = {
        "shape": (1, 16, 64, 64),
        "generation_seed_random": 1703,
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "model_revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
        "dtype": torch.float16,
    }
    cpu_latent, cpu_identity = build_canonical_sd35_base_latent(
        device="cpu",
        **common,
    )
    cuda_latent, cuda_identity = build_canonical_sd35_base_latent(
        device="cuda",
        **common,
    )

    assert NORMAL_QUANTILE_TABLE_SHA256 == (
        "70abf440a7f3670147965ffa52f5aaa639dab97f6282b68f3a9a1b1ce5e6cf5a"
    )
    assert cpu_identity == cuda_identity
    assert torch.equal(cpu_latent, cuda_latent.cpu())
    assert tensor_content_sha256(cpu_latent) == (
        "389678342d98601962a78b3fd03d576a7462ab294b1a6faf9d49e3d71cd1fdb1"
    )
    assert cpu_identity["base_latent_identity_digest_random"] == (
        "3f246e7ff53f50bba08bd93c2677bddbf2fd511ec35067a28453901adb321116"
    )
