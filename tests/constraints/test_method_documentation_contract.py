"""验证目标方法文档冻结内容路由与内容-几何双链边界。"""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
METHOD_DOCUMENT = ROOT / "docs" / "builds" / "method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md"
PRIMITIVE_DOCUMENT = ROOT / "docs" / "builds" / "algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md"
CONSTRUCTION_DOCUMENT = ROOT / "docs" / "builds" / "project_construction_state.md"
TAIL_CARRIER_SOURCE = ROOT / "main" / "methods" / "carrier" / "keyed_tensor.py"


@pytest.mark.constraint
def test_formal_method_uses_tail_mathematical_branch() -> None:
    """目标文档必须冻结 S/T/R/Q 路由、HF-tail 和单次写回。"""

    method_text = METHOD_DOCUMENT.read_text(encoding="utf-8")
    primitive_text = PRIMITIVE_DOCUMENT.read_text(encoding="utf-8")

    assert "M_{\\mathrm{LF}}=A\\odot(1-T)" in primitive_text
    assert "M_{\\mathrm{HF-tail}}=A\\odot T" in primitive_text
    assert "callback 索引10" in primitive_text
    assert "build_adjacent_latent_response_map" in method_text
    assert "build_public_probe_local_sensitivity_map" in method_text
    assert "compose_dual_chain_update_once" in method_text
    assert "仓库快照状态属于开发治理信息，不是本规范的一部分" in method_text
    assert "不记录任何一次仓库快照的实现" in method_text
    assert "GitNexus 影响面" not in method_text


@pytest.mark.constraint
def test_algorithm_primitive_defines_amplitude_tail_without_frequency_band() -> None:
    """算法原语必须冻结先高通再幅值 tail 的 HF-tail。"""

    text = PRIMITIVE_DOCUMENT.read_text(encoding="utf-8")

    assert "## 7. 原语四：HF-tail 密钥载体" in text
    assert "H(U)=U-L(U)" in text
    assert "绝对值降序、展平索引升序" in text
    assert "\\operatorname{TopAbs}_{0.20}" in text
    assert "仅对原始高斯幅值做 TopAbs" in text
    assert "不符合该原语" in text


@pytest.mark.constraint
def test_tail_primitive_metadata_uses_amplitude_domain_semantics() -> None:
    """纯幅值 tail 的实现差距只能登记到有状态构建文档。"""

    source_text = TAIL_CARRIER_SOURCE.read_text(encoding="utf-8")
    primitive_text = PRIMITIVE_DOCUMENT.read_text(encoding="utf-8")
    construction_text = CONSTRUCTION_DOCUMENT.read_text(encoding="utf-8")

    assert (
        '"tail_branch_semantics": "gaussian_amplitude_tail_truncation"'
        in source_text
    )
    assert "ranked_indices" in source_text
    assert "abs(flat_values[index])" in source_text
    assert "任何只对原始高斯幅值执行 TopAbs" in primitive_text
    assert "已存在确定性幅值 tail，但缺少高通前置步骤" in construction_text
