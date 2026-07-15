"""约束 GPU 资格化逻辑只位于服务器脚本及其内层协议."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.extract_release_package import PROFILES


pytestmark = pytest.mark.constraint


def test_gpu_qualification_is_server_entrypoint_not_notebook_implementation() -> None:
    """Notebook 不得定义方法、资格化事实或资源预算门禁."""

    root = Path.cwd()
    entrypoint = root / "scripts/run_gpu_method_qualification.py"
    assert entrypoint.is_file()
    source = entrypoint.read_text(encoding="utf-8")
    assert "write_semantic_watermark_runtime_outputs" in source
    assert "build_gpu_method_qualification_report" in source
    assert (
        "scripts/run_gpu_method_qualification.py"
        in PROFILES["paper_experiment_execution_package"].required_entrypoints
    )
    for notebook_path in (root / "paper_workflow").rglob("*.ipynb"):
        notebook_text = notebook_path.read_text(encoding="utf-8")
        assert "build_gpu_method_qualification_report" not in notebook_text
        assert "gpu_operator_preflight_ready" not in notebook_text
        assert "gpu_resource_budget_ready" not in notebook_text
