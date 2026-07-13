"""审计核心方法规范登记与冻结方法身份的一致性."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.methods.method_definition import (
    METHOD_DEFINITION_SCHEMA,
    semantic_conditioned_latent_method_definition_digest,
)
from tools.harness.lib.json_report import build_report, exit_with_report
from tools.harness.lib.method_semantic_registry import (
    REGISTRY_RELATIVE_PATH,
    load_method_semantic_registry,
    validate_method_semantic_registry,
)


EXPECTED_METHOD_DEFINITION_SCHEMA = "slm_wm_constructive_local_tangent_v9"
EXPECTED_METHOD_DEFINITION_DIGEST = (
    "2b7ab51c952abf74d145fd23694790b9de71fc6712b319fb10b3f46c9db0a0fe"
)


def run_audit(root: str | Path) -> dict[str, Any]:
    """检查规范追踪完整性, 但不生成方法科学通过结论."""

    root_path = Path(root)
    checked_paths = [
        str(REGISTRY_RELATIVE_PATH),
        "configs/model_sd35.yaml",
        "docs/builds/method_semantic_invariants.md",
        "docs/field_registry.md",
        "main/methods/method_definition.py",
    ]
    violations: list[dict[str, Any]] = []
    if METHOD_DEFINITION_SCHEMA != EXPECTED_METHOD_DEFINITION_SCHEMA:
        violations.append(
            {
                "path": "main/methods/method_definition.py",
                "reason": "method_definition_schema_drift",
            }
        )
    actual_digest = semantic_conditioned_latent_method_definition_digest()
    if actual_digest != EXPECTED_METHOD_DEFINITION_DIGEST:
        violations.append(
            {
                "path": "main/methods/method_definition.py",
                "reason": "method_definition_digest_drift",
                "expected": EXPECTED_METHOD_DEFINITION_DIGEST,
                "actual": actual_digest,
            }
        )
    try:
        payload = load_method_semantic_registry(root_path)
        registry_violations = validate_method_semantic_registry(
            root_path,
            payload,
            expected_method_definition_schema=EXPECTED_METHOD_DEFINITION_SCHEMA,
            expected_method_definition_digest=EXPECTED_METHOD_DEFINITION_DIGEST,
        )
    except (OSError, ValueError, TypeError) as error:
        registry_violations = [
            {
                "rule": "registry_load",
                "message": f"方法语义登记无法解析: {error}",
            }
        ]
    violations.extend(
        {
            "path": str(REGISTRY_RELATIVE_PATH),
            "reason": item.get("rule", "registry_violation"),
            "message": item.get("message", ""),
        }
        for item in registry_violations
    )
    return build_report(
        "audit_method_semantic_registry",
        "fail" if violations else "pass",
        violations,
        checked_paths,
    )


def main() -> None:
    """执行审计并返回稳定退出码."""

    exit_with_report(run_audit(ROOT))


if __name__ == "__main__":
    main()
