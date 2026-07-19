"""约束现行构建规范的唯一清单、职责边界和实施入口。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.write_paper_complete_result_package import BUILD_SPECIFICATION_PATHS


ROOT = Path(__file__).resolve().parents[2]
BUILDS = ROOT / "docs" / "builds"
BUILD_INDEX = BUILDS / "README.md"


@pytest.mark.constraint
def test_build_document_inventory_matches_result_package_registry() -> None:
    """真实目录与完整结果包只能共享一套构建文档清单。"""

    registered = {ROOT / relative_path for relative_path in BUILD_SPECIFICATION_PATHS}
    actual = set(BUILDS.glob("*.md"))
    assert actual == registered


@pytest.mark.constraint
def test_build_index_names_every_registered_document_once() -> None:
    """构建索引必须完整列出登记文档且不能生成第二份同名规范。"""

    text = BUILD_INDEX.read_text(encoding="utf-8")
    for path in BUILD_SPECIFICATION_PATHS:
        name = Path(path).name
        if name == "README.md":
            continue
        assert text.count(f"`{name}`") == 1


@pytest.mark.constraint
def test_registry_points_to_current_build_specification() -> None:
    """正式方法追踪不得继续把历史协议当作当前定义来源。"""

    assert not (BUILDS / "method_semantic_invariants.md").exists()
    assert not (BUILDS / "external_gpu_workflow_persistence.md").exists()
    assert (ROOT / "docs" / "legacy" / "method_semantic_invariants.md").is_file()
    assert (
        ROOT / "docs" / "runtime" / "external_gpu_workflow_persistence.md"
    ).is_file()

    registry = json.loads(
        (ROOT / "configs" / "method_semantic_registry.json").read_text(
            encoding="utf-8"
        )
    )
    pointers = {item["definition_pointer"] for item in registry["invariants"]}
    assert pointers
    assert all(
        pointer.startswith(
            "docs/builds/method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md#"
        )
        for pointer in pointers
    )


@pytest.mark.constraint
def test_build_documents_expose_machine_sources_and_ordered_changes() -> None:
    """构建规范必须能定位机器事实并给出有序项目变更。"""

    index_text = BUILD_INDEX.read_text(encoding="utf-8")
    required_index_fragments = {
        "BUILD_SPECIFICATION_PATHS",
        "可执行项目变更顺序",
        "核心方法迁移",
        "环境迁移",
        "实验协议迁移",
        "质量与论文结论闭合",
        "统一完成判据",
    }
    assert all(fragment in index_text for fragment in required_index_fragments)

    claim_text = (BUILDS / "paper_claim_decision_governance.md").read_text(
        encoding="utf-8"
    )
    quality_text = (BUILDS / "paper_quality_claim_governance.md").read_text(
        encoding="utf-8"
    )
    profile_text = (BUILDS / "paper_profile_protocol_isomorphism.md").read_text(
        encoding="utf-8"
    )
    assert "paper_profile_protocol_isomorphism.md" in claim_text
    assert "paper_quality_claim_governance.md" in claim_text
    assert "build_quality_preservation_decisions" in quality_text
    assert "方法迁移时的更新顺序" in profile_text


@pytest.mark.constraint
def test_first_method_version_rejects_saliency_scope_drift() -> None:
    """目标显著性必须冻结为 Prompt 条件 patch 相关性并限制主张边界。"""

    primitive_text = (
        BUILDS
        / "algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md"
    ).read_text(encoding="utf-8")
    mechanism_text = (
        BUILDS
        / "method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md"
    ).read_text(encoding="utf-8")
    assert "它不是人类注视真值" in primitive_text
    assert "不得使用全图单一 CLIP cosine 广播成空间图" in primitive_text
    assert "class SemanticSaliencyResult" in mechanism_text
    assert "CLIP patch-text 相关性" in mechanism_text
