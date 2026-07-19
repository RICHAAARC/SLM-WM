"""约束 GPU 资格化逻辑只位于服务器脚本及其内层协议."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

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


def test_formal_host_owns_fresh_host_qualification_entrypoint() -> None:
    """fresh-host 公开入口必须先进入父编排, 不得直接加载科学方法."""

    root = Path.cwd()
    host_source = (root / "scripts/run_formal_workflow_host.py").read_text(
        encoding="utf-8"
    )
    formal_entry_source = (root / "scripts/formal_workflow_entry.py").read_text(
        encoding="utf-8"
    )
    assert 'subparsers.add_parser("qualification")' in host_source
    assert "run_gpu_method_qualification_host_workflow" in formal_entry_source
    assert (
        "scripts/run_formal_workflow_host.py"
        in PROFILES["paper_experiment_execution_package"].required_entrypoints
    )


def test_orchestrator_qualification_import_does_not_load_gpu_runtime() -> None:
    """CPU 父解释器导入资格化编排时不得提前导入 torch 或科学 runner."""

    root = Path.cwd().resolve()
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(root)!r}); "
        "import scripts.gpu_method_qualification_host_workflow; "
        "assert 'torch' not in sys.modules; "
        "assert 'experiments.protocol.gpu_method_qualification' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_content_runtime_smoke_has_real_entry_and_no_legacy_precondition() -> None:
    """正式入口不得在新链前执行旧变换或固定 registry 工作。"""

    root = Path.cwd()
    entry_source = (root / "scripts/run_gpu_method_qualification.py").read_text(
        encoding="utf-8"
    )
    host_source = (root / "scripts/run_formal_workflow_host.py").read_text(
        encoding="utf-8"
    )
    assert 'subparsers.add_parser("content_runtime_smoke")' in host_source
    assert "run_content_runtime_smoke" in entry_source
    smoke_branch = entry_source.index(
        "if args.content_runtime_smoke:",
        entry_source.index("torch.cuda"),
    )
    formal_writer = entry_source.index(
        "write_semantic_watermark_runtime_outputs(",
        smoke_branch,
    )
    assert smoke_branch < formal_writer
    assert "_evaluate_torch_func_compatibility" not in entry_source
    assert "torch.func.linearize" not in entry_source
    assert "torch.func.vjp" not in entry_source
    assert "load_content_routing_reference_registry" not in entry_source
    assert "explicit_smoke_only_unqualified" in entry_source
