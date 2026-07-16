"""冻结目标算法原语和方法机制文档的唯一来源。"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_specification_inventory import BUILD_SPECIFICATION_PATHS


ROOT = Path(__file__).resolve().parents[2]
BUILDS = ROOT / "docs" / "builds"
PRIMITIVE_DOCUMENT = (
    BUILDS / "algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md"
)
MECHANISM_DOCUMENT = (
    BUILDS / "method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md"
)
CONSTRUCTION_DOCUMENT = BUILDS / "project_construction_state.md"
LEGACY_TRACE_DOCUMENT = ROOT / "docs" / "legacy" / "method_semantic_invariants.md"
OLD_METHOD_DOCUMENT_NAMES = {
    "algorithm_primitives_semantic_conditioned_latent_manifold_watermark.md",
    "method_module_technical_route_semantic_conditioned_latent_manifold_watermark.md",
    "method_section_semantic_conditioned_latent_manifold_watermark.md",
    "method_conformance_report.md",
    "real_scientific_operator_implementation.md",
    "single_model_branch_risk_parameter_sensitivity.md",
    "slm_wm_core_first_construction_guide.md",
}


@pytest.mark.constraint
def test_target_method_has_one_primitive_one_mechanism_and_one_state_document() -> None:
    """目标方法必须分离公式、软件结构和项目状态。"""

    assert PRIMITIVE_DOCUMENT.is_file()
    assert MECHANISM_DOCUMENT.is_file()
    assert CONSTRUCTION_DOCUMENT.is_file()
    assert not OLD_METHOD_DOCUMENT_NAMES.intersection(
        path.name for path in BUILDS.iterdir() if path.is_file()
    )

    primitive_text = PRIMITIVE_DOCUMENT.read_text(encoding="utf-8")
    mechanism_text = MECHANISM_DOCUMENT.read_text(encoding="utf-8")
    assert primitive_text.count(
        "# 算法原语：语义显著性自适应内容-几何双链潜空间水印"
    ) == 1
    assert mechanism_text.count(
        "# 方法机制设计：语义显著性自适应内容-几何双链潜空间水印"
    ) == 1


@pytest.mark.constraint
def test_target_primitive_freezes_executable_minimum_method() -> None:
    """算法原语必须冻结内容路由、几何链、写回和评测边界。"""

    text = PRIMITIVE_DOCUMENT.read_text(encoding="utf-8")
    required_fragments = {
        "A=\n\\sqrt[3]",
        "M_{\\mathrm{LF}}=A\\odot(1-T)",
        "M_{\\mathrm{HF-tail}}=A\\odot T",
        "callback 索引10",
        "\\Delta z_{\\mathrm{LF}}=0.0025",
        "=0.0050n_z",
        "s_{\\mathrm{raw}}(y,K)",
        "crop 与 crop-rescale 捕获边界",
        "`watermarked_positive`",
        "`attacked_negative`",
        "不得重新单位化整个分支",
        "检测器不得访问",
        "禁止代理与静默回退",
        "明确禁止主张",
    }
    assert all(fragment in text for fragment in required_fragments)


@pytest.mark.constraint
def test_target_documents_separate_stateless_rules_from_project_state() -> None:
    """两份方法规范不得混入仓库状态，状态文档必须承担迁移事实。"""

    primitive_text = PRIMITIVE_DOCUMENT.read_text(encoding="utf-8")
    mechanism_text = MECHANISM_DOCUMENT.read_text(encoding="utf-8")
    construction_text = CONSTRUCTION_DOCUMENT.read_text(encoding="utf-8")
    state_only_fragments = {
        "GitNexus 影响面",
        "保留清单",
        "修改清单",
        "移除清单",
        "有序实施计划",
        "core_documents_frozen",
        "document_ecosystem_synchronized",
        "not_implemented",
        "injection_step_indices: [6, 10, 14]",
    }
    assert not state_only_fragments.intersection(primitive_text)
    assert not state_only_fragments.intersection(mechanism_text)
    assert all(fragment in construction_text for fragment in state_only_fragments)

    required_mechanism_fragments = {
        "build_prompt_conditioned_semantic_saliency",
        "build_public_probe_local_sensitivity_map",
        "compose_dual_chain_update_once",
        "measure_dual_chain_watermark",
        "不记录任何一次仓库快照的实现",
    }
    assert all(fragment in mechanism_text for fragment in required_mechanism_fragments)


@pytest.mark.constraint
def test_legacy_trace_is_explicitly_non_authoritative() -> None:
    """迁移前登记只能识别旧实现，不能覆盖目标算法。"""

    text = LEGACY_TRACE_DOCUMENT.read_text(encoding="utf-8")
    assert "迁移前方法语义不变量兼容记录" in text
    assert "不再是目标方法的权威来源" in text
    assert "不能让新旧不变量长期并存" in text


@pytest.mark.constraint
def test_project_contract_blocks_paper_production_during_method_migration() -> None:
    """项目契约必须阻止旧实现生产新方法论文结果。"""

    text = (ROOT / ".codex" / "project_contract.md").read_text(encoding="utf-8")
    assert "`project_unit`: `document_ecosystem_synchronization`" in text
    assert "`target_construction_unit`: `core_method_runtime_construction`" in text
    assert "二者都不表示完成状态" in text
    assert "正式论文结果生产处于阻断状态" in text
    assert "历史 `c6139ced` 结果" in text


@pytest.mark.constraint
def test_complete_package_uses_target_method_documents() -> None:
    """完整结果包清单必须携带三份中心文档并排除已删除文档。"""

    registered = set(BUILD_SPECIFICATION_PATHS)
    assert str(PRIMITIVE_DOCUMENT.relative_to(ROOT)).replace("\\", "/") in registered
    assert str(MECHANISM_DOCUMENT.relative_to(ROOT)).replace("\\", "/") in registered
    assert str(CONSTRUCTION_DOCUMENT.relative_to(ROOT)).replace("\\", "/") in registered
    assert not OLD_METHOD_DOCUMENT_NAMES.intersection(
        Path(path).name for path in registered
    )
