"""验证服务器执行面脱离 Notebook 并共享受治理 CLI 契约。"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


SERVER_ENTRYPOINTS = (
    "scripts/run_formal_workflow_host.py",
    "scripts/run_gpu_server_workflow.py",
    "scripts/run_gpu_server_result_closure.py",
    "scripts/write_paper_profile_protocol_isomorphism_report.py",
)


@pytest.mark.constraint
@pytest.mark.parametrize("relative_path", SERVER_ENTRYPOINTS)
def test_server_entrypoint_starts_without_notebook_or_pythonpath(
    relative_path: str,
    tmp_path: Path,
) -> None:
    """服务器入口只依赖代码包内层, 不借用开发仓库 PYTHONPATH。"""

    repository_root = Path.cwd()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(repository_root / relative_path),
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "paper_workflow" not in completed.stderr


@pytest.mark.constraint
def test_notebook_layer_is_absent_from_server_release_profiles() -> None:
    """服务器执行包和论文重建包都必须显式排除 Colab 外层。"""

    from scripts.extract_release_package import PROFILES

    for profile_name in (
        "paper_artifact_rebuild_package",
        "paper_experiment_execution_package",
    ):
        profile = PROFILES[profile_name]
        assert "paper_workflow" in profile.exclude_parts
        assert all(
            not path.startswith("paper_workflow")
            for path in profile.include_paths
        )
        assert (
            "scripts/write_paper_profile_protocol_isomorphism_report.py"
            in profile.required_entrypoints
        )
