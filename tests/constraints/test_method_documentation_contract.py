"""验证方法文档与高斯幅值尾部截断实现保持同一术语边界。"""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
METHOD_DOCUMENT = ROOT / "docs" / "builds" / "method_section_semantic_conditioned_latent_manifold_watermark.md"
PRIMITIVE_DOCUMENT = ROOT / "docs" / "builds" / "algorithm_primitives_semantic_conditioned_latent_manifold_watermark.md"
TAIL_CARRIER_SOURCE = ROOT / "main" / "methods" / "carrier" / "keyed_tensor.py"


@pytest.mark.constraint
def test_formal_method_uses_tail_mathematical_branch() -> None:
    """正式方法公式和正文必须只描述当前 tail 分支。"""

    text = METHOD_DOCUMENT.read_text(encoding="utf-8")

    assert "高斯幅值尾部截断分支只比较高斯模板元素的绝对幅值与分位点" in text
    assert "正式分支标识为 `tail_robust`" in text
    assert "\\Delta z_t^{\\mathrm{tail}}" in text
    assert "\\widetilde\\nu_{\\mathrm{tail},i}" in text


@pytest.mark.constraint
def test_algorithm_primitive_defines_amplitude_tail_without_frequency_band() -> None:
    """算法原语必须分别定义幅值分位点与空间频带边界。"""

    text = PRIMITIVE_DOCUMENT.read_text(encoding="utf-8")

    assert "### （二）高斯幅值尾部截断" in text
    assert "与二维空间频谱无关" in text
    assert "tail_fraction" in text
    assert "不是频率截止值" in text


@pytest.mark.constraint
def test_tail_primitive_metadata_uses_amplitude_domain_semantics() -> None:
    """尾部载体元数据必须登记幅值域语义和频带不适用标识。"""

    text = TAIL_CARRIER_SOURCE.read_text(encoding="utf-8")

    assert '"tail_branch_semantics": "gaussian_amplitude_tail_truncation"' in text
    assert "quantile" in text
    assert "abs()" in text
