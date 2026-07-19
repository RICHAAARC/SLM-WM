"""验证方法语义规范追踪已经接入统一 harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.harness.audits.audit_method_semantic_registry import run_audit
from tools.harness.run_all_audits import AUDIT_MODULE_NAMES


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.constraint
def test_method_semantic_registry_audit_passes_for_repository() -> None:
    """权威规范、配置、方法镜像和登记表必须保持同一身份."""

    report = run_audit(ROOT)

    assert report["decision"] == "pass", report["violations"]
    assert "experiments/runners/semantic_watermark_runtime.py" in {
        Path(path).as_posix() for path in report["checked_paths"]
    }


@pytest.mark.constraint
def test_method_semantic_registry_audit_is_not_optional() -> None:
    """统一审计入口不得漏掉方法规范追踪检查."""

    assert "tools.harness.audits.audit_method_semantic_registry" in (
        AUDIT_MODULE_NAMES
    )
