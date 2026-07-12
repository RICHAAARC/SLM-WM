"""验证正式 FID/KID 在 CPU 与 CUDA float64 后端之间数值一致。"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.protocol.dataset_quality import (
    _sqrt_trace_fid_from_covariances,
    _torch_gaussian_fid,
    _torch_unbiased_polynomial_mmd_exact,
)


pytestmark = pytest.mark.integration


def _cpu_unbiased_polynomial_mmd(
    source: np.ndarray,
    comparison: np.ndarray,
) -> float:
    """在 CPU 上独立计算正式 KID 单子集无偏公式。"""

    feature_dimension = int(source.shape[1])
    source_kernel = (source @ source.T / feature_dimension + 1.0) ** 3
    comparison_kernel = (
        comparison @ comparison.T / feature_dimension + 1.0
    ) ** 3
    cross_kernel = (source @ comparison.T / feature_dimension + 1.0) ** 3
    sample_count = int(source.shape[0])
    denominator = sample_count * (sample_count - 1)
    return float(
        (source_kernel.sum() - np.trace(source_kernel)) / denominator
        + (comparison_kernel.sum() - np.trace(comparison_kernel))
        / denominator
        - 2.0 * cross_kernel.mean()
    )


def test_dataset_quality_float64_cpu_cuda_equivalence() -> None:
    """同一冻结特征在 CPU 与 CUDA 上必须落入闭合容差。"""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("该集成测试需要 CUDA")
    random = np.random.default_rng(2020)
    source = random.normal(size=(32, 8)).astype(np.float64)
    comparison = (
        source * 0.93 + random.normal(scale=0.05, size=(32, 8))
    ).astype(np.float64)

    cpu_fid = _sqrt_trace_fid_from_covariances(
        source.mean(axis=0),
        comparison.mean(axis=0),
        np.cov(source, rowvar=False),
        np.cov(comparison, rowvar=False),
    )
    cuda_fid = _torch_gaussian_fid(
        source,
        comparison,
        allow_high_dimensional_cpu=True,
    )
    cpu_kid = _cpu_unbiased_polynomial_mmd(source, comparison)
    cuda_kid = _torch_unbiased_polynomial_mmd_exact(source, comparison)

    assert cuda_fid is not None
    assert cuda_kid is not None
    assert cuda_fid == pytest.approx(cpu_fid, rel=1e-8, abs=1e-10)
    assert cuda_kid == pytest.approx(cpu_kid, rel=1e-8, abs=1e-10)
