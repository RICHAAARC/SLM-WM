from __future__ import annotations

import ast
import json
import posixpath
from pathlib import Path
import re

import pytest

from scripts.extract_release_package import (
    CORE_METHOD_SPECIFICATION_PATHS,
    BUILD_SPECIFICATION_PATHS,
    PROFILES,
    extract_profile,
)


ROOT = Path(__file__).resolve().parents[2]


def _first_party_markdown_targets(
    text: str,
    *,
    package_path: str,
) -> set[str]:
    """解析第一方 Markdown 引用, 用包内目标路径而不是开发源路径解算."""

    raw_targets = set(re.findall(r"\[[^\]]+\]\(([^)]+)\)", text))
    raw_targets.update(
        target
        for target in re.findall(r"`([^`\n]+\.md(?:#[^`\n]+)?)`", text)
        if "/" in target or "\\" in target
    )
    resolved: set[str] = set()
    root_prefixes = ("docs/", "main/", "configs/", "experiments/", "scripts/")
    for raw_target in raw_targets:
        target = raw_target.strip().strip("<>").split("#", maxsplit=1)[0]
        if not target.endswith(".md") or "://" in target:
            continue
        if target.startswith(root_prefixes):
            normalized = posixpath.normpath(target)
        else:
            normalized = posixpath.normpath(
                posixpath.join(posixpath.dirname(package_path), target)
            )
        if not normalized.startswith("../"):
            resolved.add(normalized)
    return resolved


def _resolve_field_lifecycle(
    field_name: str,
    *,
    target_fields: set[str],
    registry: dict[str, object],
) -> str:
    """按机器登记的冻结优先级解析一个字段的唯一生命周期."""

    if field_name in target_fields:
        return "target_required"
    if field_name in set(registry["legacy_only_exact_fields"]):
        return "legacy_only"
    if any(
        field_name.startswith(prefix)
        for prefix in registry["legacy_only_prefixes"]
    ):
        return "legacy_only"
    if any(token in field_name for token in registry["legacy_only_contains"]):
        return "legacy_only"
    return "active_shared"


@pytest.mark.constraint
def test_release_profiles_carry_their_authoritative_documents() -> None:
    """抽离包必须携带其 README 引用的权威规范。"""

    minimal = set(PROFILES["minimal_method_package"].include_paths)
    assert set(CORE_METHOD_SPECIFICATION_PATHS) <= minimal
    assert "docs/builds/project_construction_state.md" not in minimal

    for profile_name in (
        "paper_artifact_rebuild_package",
        "paper_experiment_execution_package",
    ):
        included = set(PROFILES[profile_name].include_paths)
        assert set(BUILD_SPECIFICATION_PATHS) <= included

    minimal_readme = (ROOT / "docs/core_method_package_readme.md").read_text(
        encoding="utf-8-sig"
    )
    assert "docs/builds/project_construction_state.md" not in minimal_readme
    for relative_path in CORE_METHOD_SPECIFICATION_PATHS:
        assert relative_path in minimal_readme


@pytest.mark.constraint
def test_release_profile_markdown_references_are_self_contained(tmp_path: Path) -> None:
    """真实 dry-run 清单中的第一方 Markdown 引用不得指向包外缺失文档."""

    for profile_name in PROFILES:
        manifest = extract_profile(
            ROOT,
            tmp_path / profile_name,
            profile_name,
            dry_run=True,
        )
        copied = set(manifest["copied_files"])
        missing: list[str] = []
        for record in manifest["copied_file_records"]:
            package_path = str(record["path"])
            if not package_path.endswith(".md"):
                continue
            source_path = ROOT / str(record["source_path"])
            for target in _first_party_markdown_targets(
                source_path.read_text(encoding="utf-8-sig"),
                package_path=package_path,
            ):
                if target not in copied:
                    missing.append(f"{package_path} -> {target}")
        assert missing == [], f"{profile_name} 存在包内文档断链: {missing}"


@pytest.mark.constraint
def test_release_profile_machine_definition_pointers_are_self_contained(
    tmp_path: Path,
) -> None:
    """外层包复制机器方法登记时必须同时携带其定义指针目标."""

    for profile_name in (
        "paper_artifact_rebuild_package",
        "paper_experiment_execution_package",
    ):
        manifest = extract_profile(
            ROOT,
            tmp_path / profile_name,
            profile_name,
            dry_run=True,
        )
        copied = set(manifest["copied_files"])
        registry = json.loads(
            (ROOT / "configs/method_semantic_registry.json").read_text(
                encoding="utf-8-sig"
            )
        )
        pointers = {
            str(record["definition_pointer"]).split("#", maxsplit=1)[0]
            for record in registry["invariants"]
        }
        assert pointers <= copied


@pytest.mark.constraint
def test_claim_registry_excludes_parameter_sensitivity_diagnostics() -> None:
    """参数敏感性只能是诊断证据，不能进入正式主张或产物 gate。"""

    claim_registry = json.loads(
        (ROOT / "configs/paper_claim_registry.json").read_text(encoding="utf-8-sig")
    )
    expected_claims = {
        "fixed_fpr_detection",
        "baseline_superiority",
        "quality_preservation",
        "mechanism_necessity",
    }
    assert set(claim_registry["registered_claim_ids"]) == expected_claims
    assert set(claim_registry["required_claims"]) == expected_claims
    assert claim_registry["optional_claims"] == []

    profile_registry = json.loads(
        (ROOT / "configs/paper_profile_protocol_registry.json").read_text(
            encoding="utf-8-sig"
        )
    )
    serialized = json.dumps(profile_registry, ensure_ascii=False, sort_keys=True)
    assert "parameter_robustness" not in serialized
    claim_gate_text = json.dumps(
        profile_registry["gate_roles"],
        ensure_ascii=False,
        sort_keys=True,
    )
    assert "branch_risk_parameter_sensitivity" not in claim_gate_text


@pytest.mark.constraint
def test_current_documents_do_not_restore_retired_method_or_profile_names() -> None:
    """非 legacy 文档不得恢复旧方法语义、旧 profile 名称或旧 GPU 状态名。"""

    excluded_roots = {
        ROOT / "docs/legacy",
        ROOT / "outputs",
    }
    state_document = ROOT / "docs/builds/project_construction_state.md"
    checked_paths: list[Path] = []
    for path in ROOT.rglob("*.md"):
        if any(root == path or root in path.parents for root in excluded_roots):
            continue
        if path == state_document:
            continue
        if "source" in path.parts and "external_baseline" in path.parts:
            continue
        checked_paths.append(path)

    combined = "\n".join(
        path.read_text(encoding="utf-8-sig") for path in sorted(checked_paths)
    )
    assert "probe_claim" not in combined
    assert "pilot_claim" not in combined
    assert "full_claim" not in combined
    assert "gpu_verified" not in combined
    assert "没有空间频带定义" not in combined
    assert "只定义幅值域筛选" not in combined


@pytest.mark.constraint
def test_field_registry_declares_target_legacy_lifecycle_precedence() -> None:
    """字段表必须能确定性地区分目标、共享与迁移前字段。"""

    text = (ROOT / "docs/field_registry.md").read_text(encoding="utf-8-sig")
    for lifecycle in ("target_required", "legacy_only", "active_shared"):
        assert lifecycle in text
    assert "生命周期规则高于旧字段行中的 `allowed_in_claims`" in text
    assert "目标协议不得读取该字段" in text
    lifecycle_registry = json.loads(
        (ROOT / "configs/field_lifecycle_registry.json").read_text(
            encoding="utf-8-sig"
        )
    )
    legacy_fields = set(lifecycle_registry["legacy_only_exact_fields"])
    for field_name in (
        "branch",
        "method_definition",
        "texture_threshold",
        "tail_fraction",
        "tail_score",
        "tail_update_values",
        "registration_threshold",
        "resolved_branch_risk_configs",
        "semantic_feature_protocol_schema",
        "injection_step_indices",
        "tail_robust_detection_score_weight",
        "tail_robust_score",
        "torch_func_compatibility",
        "ph" "ase_status",
    ):
        assert field_name in legacy_fields

    target_section = text.split("## 冻结目标方法新增字段", maxsplit=1)[1].split(
        "## 字段登记表",
        maxsplit=1,
    )[0]
    target_fields = set(
        re.findall(r"^\| ([A-Za-z0-9_]+) \|", target_section, flags=re.MULTILINE)
    ) - {"field_name"}
    for field_name in (
        "semantic_feature_protocol_schema",
        "injection_step_indices",
        "tail_robust_score",
        "torch_func_compatibility",
        "ph" "ase_status",
    ):
        assert _resolve_field_lifecycle(
            field_name,
            target_fields=target_fields,
            registry=lifecycle_registry,
        ) == "legacy_only"
    for field_name in (
        "measurement_status",
        "lf_mask",
        "hf_tail_mask",
        "content_threshold",
        "rescue_margin_low",
        "registration_confidence",
    ):
        assert _resolve_field_lifecycle(
            field_name,
            target_fields=target_fields,
            registry=lifecycle_registry,
        ) == "target_required"
    assert _resolve_field_lifecycle(
        "code_version",
        target_fields=target_fields,
        registry=lifecycle_registry,
    ) == "active_shared"
    assert _resolve_field_lifecycle(
        "threshold_source",
        target_fields=target_fields,
        registry=lifecycle_registry,
    ) == "active_shared"

    registered_names = re.findall(
        r"^\| ([A-Za-z0-9_]+) \|",
        text,
        flags=re.MULTILINE,
    )
    registered_names = [
        name for name in registered_names if name not in {"field", "field_name"}
    ]
    assert len(registered_names) == len(set(registered_names))


@pytest.mark.constraint
def test_target_method_dataclass_fields_have_one_governed_lifecycle() -> None:
    """目标方法接口字段必须完整登记, 且不能落入迁移前生命周期."""

    method_text = (
        ROOT
        / "docs/builds/method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md"
    ).read_text(encoding="utf-8-sig")
    dataclass_fields: set[str] = set()
    for code_block in re.findall(r"```python\n(.*?)\n```", method_text, flags=re.DOTALL):
        syntax_tree = ast.parse(code_block)
        for node in syntax_tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            is_dataclass = any(
                (isinstance(decorator, ast.Name) and decorator.id == "dataclass")
                or (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "dataclass"
                )
                for decorator in node.decorator_list
            )
            if not is_dataclass:
                continue
            dataclass_fields.update(
                statement.target.id
                for statement in node.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
            )

    lifecycle_registry = json.loads(
        (ROOT / "configs/field_lifecycle_registry.json").read_text(
            encoding="utf-8-sig"
        )
    )
    target_fields = set(lifecycle_registry["target_required_exact_fields"])
    legacy_fields = set(lifecycle_registry["legacy_only_exact_fields"])
    assert target_fields.isdisjoint(legacy_fields)
    assert dataclass_fields <= target_fields
    assert "local_sensitivity_map" in dataclass_fields
    assert "local_sensitivity_map_digest" in dataclass_fields
    assert "sensitivity_map" not in dataclass_fields
    assert "sensitivity_map_digest" not in dataclass_fields

    field_registry_text = (ROOT / "docs/field_registry.md").read_text(
        encoding="utf-8-sig"
    )
    registered_names = re.findall(
        r"^\| ([A-Za-z0-9_]+) \|",
        field_registry_text,
        flags=re.MULTILINE,
    )
    registered_names = [
        name for name in registered_names if name not in {"field", "field_name"}
    ]
    assert len(registered_names) == len(set(registered_names))
    assert target_fields <= set(registered_names)

    field_record_permissions = {
        match.group("field_name"): match.group("allowed_in_records") == "true"
        for match in re.finditer(
            r"^\| (?P<field_name>[A-Za-z0-9_]+) \| [^|]+ \| [^|]+ \| "
            r"(?P<allowed_in_records>true|false) \|",
            field_registry_text,
            flags=re.MULTILINE,
        )
    }
    runtime_only_fields = {
        "aligned_image",
        "geometry",
        "geometry_update",
        "hf_tail_mask",
        "lf_mask",
        "local_sensitivity_map",
        "patch_relevance",
        "raw",
        "response_map",
        "saliency_map",
        "template",
        "texture_map",
        "writable_capacity_map",
        "written_latent",
    }
    assert all(
        field_record_permissions[field_name] is False
        for field_name in runtime_only_fields
    )
    serializable_identity_fields = {
        "aligned_height",
        "aligned_image_member_path",
        "aligned_image_sha256",
        "aligned_width",
        "geometry_measurement",
        "geometry_update_digest",
        "layer_candidate_summaries",
        "local_sensitivity_map_digest",
        "response_map_digest",
        "routing_identity_digest",
        "saliency_map_digest",
        "template_digest",
        "texture_map_digest",
        "write_identity_digest",
    }
    assert all(
        field_record_permissions[field_name] is True
        for field_name in serializable_identity_fields
    )

    target_section = field_registry_text.split(
        "## 冻结目标方法新增字段",
        maxsplit=1,
    )[1].split("## 字段登记表", maxsplit=1)[0]
    target_governance_fields = set(
        re.findall(
            r"^\| ([A-Za-z0-9_]+) \|",
            target_section,
            flags=re.MULTILINE,
        )
    ) - {"field_name"}
    assert target_fields == dataclass_fields | target_governance_fields


@pytest.mark.constraint
def test_independent_visual_quality_claim_does_not_assert_prompt_alignment() -> None:
    """DINOv2 图像-图像指标只能支持配对视觉内容保持."""

    protocol = json.loads(
        (ROOT / "configs/paper_quality_claim_protocol.json").read_text(
            encoding="utf-8-sig"
        )
    )
    claim = protocol["independent_visual_content_preservation_noninferiority"]
    assert claim["estimand_interpretation"] == (
        "paired_clean_image_visual_content_preservation_only"
    )
    assert claim["prompt_text_encoded"] is False
    serialized = json.dumps(protocol, ensure_ascii=False)
    assert "semantic_alignment_noninferiority" not in serialized


@pytest.mark.constraint
def test_profile_document_freezes_five_repeats_and_four_claims() -> None:
    """三档必须固定5重复，且输出位置只能是操作派生事实。"""

    text = (
        ROOT / "docs/builds/paper_profile_protocol_isomorphism.md"
    ).read_text(encoding="utf-8-sig")
    assert "固定5重复" in text
    assert "固定9重复" not in text
    assert "四项正式论文主张" in text
    assert "操作存储位置，不是科学协议变化字段" in text
    assert "五项论文主张" not in text
    assert "五项登记主张" not in text
    assert "五类统计产物" not in text
