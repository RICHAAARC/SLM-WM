"""解析并验证正式内容双链的最小方法语义追踪登记。"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

import yaml


REGISTRY_RELATIVE_PATH = PurePosixPath("configs/method_semantic_registry.json")
METHOD_CONFIG_RELATIVE_PATH = PurePosixPath("configs/model_sd35.yaml")
REGISTRY_SCHEMA = "slm_wm_method_semantic_trace_registry"
REGISTRY_SCOPE = "normative_traceability_without_scientific_conformance_decision"
EXPECTED_INVARIANT_IDS = (
    "formal_content_observation_routing",
    "formal_lf_hf_tail_carriers",
    "direct_qk_geometry_sync",
    "common_gamma_actual_dtype_single_write",
    "formal_image_only_blind_detection",
)
EXPECTED_NORMATIVE_TRACE_DIGEST = (
    "0707ab90d6759a34b0618a616cd9c1177ec0728e8044e3ce0add73acdbd07cee"
)
REQUIRED_INVARIANT_FIELDS = frozenset(
    {
        "invariant_id",
        "definition_pointer",
        "formal_expression",
        "configuration_fields",
        "method_implementation_symbols",
        "runtime_binding_symbols",
        "runtime_evidence_fields",
        "fail_closed_conditions",
        "forbidden_substitutes",
        "cpu_property_id",
        "specification_test_nodes",
        "cpu_property_test_nodes",
        "gpu_atomic_roles",
        "gpu_observation_requirement",
        "claim_boundary",
    }
)
_NORMATIVE_TRACE_FIELDS = (
    "invariant_id",
    "definition_pointer",
    "formal_expression",
    "configuration_fields",
    "runtime_evidence_fields",
    "fail_closed_conditions",
    "forbidden_substitutes",
    "gpu_atomic_roles",
    "gpu_observation_requirement",
    "claim_boundary",
)
_SELF_ASSERTION_KEYS = frozenset(
    {
        "pass",
        "passed",
        "ready",
        "verified",
        "decision",
        "status",
        "supports_method_claim",
        "supports_paper_claim",
        "cpu_verified",
        "gpu_verified",
    }
)
_TEST_NODE = re.compile(
    r"^(tests/[a-z0-9_/]*test_[a-z0-9_]+\.py)::(test_[a-z0-9_]+)$"
)
_FIELD_ROW = re.compile(r"^\| ([a-z][a-z0-9_]*) \|", re.MULTILINE)


def load_method_semantic_registry(root: str | Path) -> dict[str, Any]:
    """从固定位置读取精确 JSON object。"""

    path = Path(root) / REGISTRY_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise ValueError("method semantic registry 顶层必须为精确 object")
    return payload


def method_semantic_normative_trace_digest(payload: Mapping[str, Any]) -> str:
    """仅对人类可读规范职责计算稳定身份。"""

    invariants = payload.get("invariants")
    if type(invariants) is not list:
        raise ValueError("invariants 必须为精确 list")
    trace = {
        "registry_schema": payload.get("registry_schema"),
        "registry_scope": payload.get("registry_scope"),
        "method_definition_schema": payload.get("method_definition_schema"),
        "method_definition_digest": payload.get("method_definition_digest"),
        "invariants": [
            {
                field_name: invariant.get(field_name)
                for field_name in _NORMATIVE_TRACE_FIELDS
            }
            for invariant in invariants
            if type(invariant) is dict
        ],
    }
    encoded = json.dumps(
        trace,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _violation(rule: str, message: str) -> dict[str, str]:
    return {"rule": rule, "message": message}


def _exact_string_list(value: Any) -> bool:
    return (
        type(value) is list
        and bool(value)
        and all(type(item) is str and bool(item) for item in value)
        and len(value) == len(set(value))
    )


def _symbol_exists(path: Path, symbol: str) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == symbol
        for node in tree.body
    )


def _binding_list_ready(
    root: Path,
    value: Any,
    *,
    prefix: str,
) -> bool:
    if type(value) is not list or not value:
        return False
    seen: set[tuple[str, str]] = set()
    for binding in value:
        if type(binding) is not dict or set(binding) != {"path", "symbol"}:
            return False
        path_value = binding.get("path")
        symbol = binding.get("symbol")
        if (
            type(path_value) is not str
            or type(symbol) is not str
            or not path_value.startswith(prefix)
            or ".." in PurePosixPath(path_value).parts
        ):
            return False
        identity = (path_value, symbol)
        if identity in seen or not _symbol_exists(root / path_value, symbol):
            return False
        seen.add(identity)
    return True


def _test_nodes_ready(root: Path, value: Any) -> bool:
    if not _exact_string_list(value):
        return False
    for node in value:
        match = _TEST_NODE.fullmatch(node)
        if match is None:
            return False
        relative_path, function_name = match.groups()
        if relative_path == "tests/constraints/test_method_semantic_registry_contract.py":
            return False
        if not _symbol_exists(root / relative_path, function_name):
            return False
    return True


def _config_path_exists(payload: Mapping[str, Any], dot_path: str) -> bool:
    current: Any = payload
    for part in dot_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def _contains_self_assertion(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key) in _SELF_ASSERTION_KEYS
            or _contains_self_assertion(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_self_assertion(item) for item in value)
    return False


def validate_method_semantic_registry(
    root: str | Path,
    payload: Mapping[str, Any],
    *,
    expected_method_definition_schema: str,
    expected_method_definition_digest: str,
) -> list[dict[str, str]]:
    """验证当前双链登记的身份、引用闭包和反自证边界。"""

    root_path = Path(root)
    violations: list[dict[str, str]] = []
    if type(payload) is not dict:
        return [_violation("registry_type", "registry 必须为精确 object")]
    if payload.get("registry_schema") != REGISTRY_SCHEMA:
        violations.append(_violation("registry_schema", "registry schema 漂移"))
    if payload.get("registry_scope") != REGISTRY_SCOPE:
        violations.append(_violation("registry_scope", "registry scope 漂移"))
    if payload.get("method_definition_schema") != expected_method_definition_schema:
        violations.append(_violation("method_definition_schema", "方法 schema 漂移"))
    if payload.get("method_definition_digest") != expected_method_definition_digest:
        violations.append(_violation("method_definition_digest", "方法摘要漂移"))
    if _contains_self_assertion(payload):
        violations.append(_violation("self_asserted_conformance", "登记不得自证通过"))

    invariants = payload.get("invariants")
    if type(invariants) is not list:
        return violations + [_violation("invariant_exact_set", "invariants 必须为 list")]
    ids = tuple(
        item.get("invariant_id") if type(item) is dict else None
        for item in invariants
    )
    if ids != EXPECTED_INVARIANT_IDS:
        violations.append(_violation("invariant_exact_set", "不变量集合或顺序漂移"))

    try:
        config = yaml.safe_load(
            (root_path / METHOD_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError):
        config = {}
    field_names = set(
        _FIELD_ROW.findall(
            (root_path / "docs/field_registry.md").read_text(encoding="utf-8")
        )
    )
    for item in invariants:
        if type(item) is not dict or set(item) != REQUIRED_INVARIANT_FIELDS:
            violations.append(_violation("invariant_fields", "不变量字段集合漂移"))
            continue
        invariant_id = str(item["invariant_id"])
        for field_name in (
            "formal_expression",
            "configuration_fields",
            "runtime_evidence_fields",
            "fail_closed_conditions",
            "forbidden_substitutes",
            "gpu_atomic_roles",
        ):
            if not _exact_string_list(item[field_name]):
                violations.append(_violation(field_name, f"{invariant_id} 的 {field_name} 非精确非空列表"))
        if not all(
            type(item[field_name]) is str and bool(item[field_name])
            for field_name in (
                "definition_pointer",
                "cpu_property_id",
                "gpu_observation_requirement",
                "claim_boundary",
            )
        ):
            violations.append(_violation("invariant_text", f"{invariant_id} 文本职责缺失"))
        pointer_path = str(item["definition_pointer"]).split("#", 1)[0]
        if not pointer_path.startswith("docs/") or not (root_path / pointer_path).is_file():
            violations.append(_violation("definition_anchor", f"{invariant_id} 定义路径无效"))
        if not _binding_list_ready(root_path, item["method_implementation_symbols"], prefix="main/"):
            violations.append(_violation("method_implementation_symbols", f"{invariant_id} 方法绑定无效"))
        if not _binding_list_ready(root_path, item["runtime_binding_symbols"], prefix="experiments/"):
            violations.append(_violation("runtime_binding_symbols", f"{invariant_id} runtime绑定无效"))
        if not _test_nodes_ready(root_path, item["specification_test_nodes"]):
            violations.append(_violation("specification_test_nodes", f"{invariant_id} 规范测试绑定无效"))
        if not _test_nodes_ready(root_path, item["cpu_property_test_nodes"]):
            violations.append(_violation("cpu_property_test_nodes", f"{invariant_id} CPU测试绑定无效"))
        if not all(_config_path_exists(config, path) for path in item["configuration_fields"]):
            violations.append(_violation("configuration_fields", f"{invariant_id} 配置路径无效"))
        if not set(item["runtime_evidence_fields"]).issubset(field_names):
            violations.append(_violation("field_registry", f"{invariant_id} 使用未登记证据字段"))

    if method_semantic_normative_trace_digest(payload) != EXPECTED_NORMATIVE_TRACE_DIGEST:
        violations.append(_violation("normative_trace_digest", "规范追踪摘要漂移"))
    return violations


__all__ = [
    "EXPECTED_INVARIANT_IDS",
    "EXPECTED_NORMATIVE_TRACE_DIGEST",
    "METHOD_CONFIG_RELATIVE_PATH",
    "REGISTRY_RELATIVE_PATH",
    "REGISTRY_SCHEMA",
    "REGISTRY_SCOPE",
    "load_method_semantic_registry",
    "method_semantic_normative_trace_digest",
    "validate_method_semantic_registry",
]
