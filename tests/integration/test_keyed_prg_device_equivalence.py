"""验证密钥 PRG 模板在 CPU 与 CUDA 间保持字节等价."""

from __future__ import annotations

import pytest
import torch

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
        ),
        keyed_relation_signs(
            cuda_attention,
            "device-independent-key",
            "transformer_blocks.0.attn",
        ).cpu(),
    )
