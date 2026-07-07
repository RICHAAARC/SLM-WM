from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Iterator

import pytest

from scripts.run_gpu_server_result_closure import configure_closure_environment
from scripts.run_gpu_server_workflow import (
    configure_common_server_environment,
    resolve_workflow_selection,
    workflow_publish_dir,
)

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def isolate_slm_environment() -> Iterator[None]:
    """隔离直接写入 os.environ 的服务器脚本配置。"""

    previous_values = {key: value for key, value in os.environ.items() if key.startswith("SLM_WM_")}
    for key in tuple(os.environ):
        if key.startswith("SLM_WM_"):
            os.environ.pop(key, None)
    yield
    for key in tuple(os.environ):
        if key.startswith("SLM_WM_"):
            os.environ.pop(key, None)
    os.environ.update(previous_values)


def test_gpu_server_workflow_keeps_local_result_root(tmp_path: Path) -> None:
    """服务器 workflow 配置应使用本地结果根目录, 不回退到 Colab Drive 路径。"""

    result_root = tmp_path / "server_results"
    report = configure_common_server_environment(
        root=Path.cwd(),
        paper_run_name="pilot_paper",
        result_root=result_root,
        sample_count_token="5",
        target_fpr_override="",
    )

    assert report["paper_run"]["drive_result_root"] == result_root.as_posix()
    assert report["paper_run"]["sample_count"] == 5
    assert "/content/drive" not in report["paper_run"]["drive_result_root"]


def test_gpu_server_workflow_configures_probe_paper_without_notebook_logic_fork(tmp_path: Path) -> None:
    """服务器 workflow 应能只切换运行层级进入 probe_paper 前置验证配置。"""

    result_root = tmp_path / "probe_results"
    report = configure_common_server_environment(
        root=Path.cwd(),
        paper_run_name="probe_paper",
        result_root=result_root,
        sample_count_token="all",
        target_fpr_override="",
    )

    assert report["paper_run"]["run_name"] == "probe_paper"
    assert report["paper_run"]["prompt_set"] == "probe_paper"
    assert report["paper_run"]["prompt_file"].endswith("configs/paper_main_probe_paper_prompts.txt")
    assert report["paper_run"]["sample_count"] == 60
    assert report["paper_run"]["target_fpr"] == 0.1
    assert report["paper_run"]["minimum_clean_negative_count"] == 10
    assert report["paper_run"]["dataset_level_quality_minimum_count"] == 60
    assert report["paper_run"]["protocol_profile"] == "probe_paper_fixed_fpr_0_1"
    assert report["paper_run"]["drive_result_root"] == result_root.as_posix()
    assert os.environ["SLM_WM_PAPER_RUN_NAME"] == "probe_paper"
    assert os.environ["SLM_WM_PAPER_RUN_EXPECTED_SAMPLE_COUNT"] == "60"
    assert os.environ["SLM_WM_PAPER_RUN_MINIMUM_CLEAN_NEGATIVE_COUNT"] == "10"


def test_gpu_server_external_baseline_alias_resolves_single_method() -> None:
    """单方法 baseline alias 应解析为统一 method-faithful workflow 与对应 baseline。"""

    selection = resolve_workflow_selection("external_baseline_tree_ring")

    assert selection.workflow_name == "external_baseline_method_faithful"
    assert selection.baseline_id == "tree_ring"


def test_gpu_server_publish_dir_uses_semantic_result_bucket(tmp_path: Path) -> None:
    """服务器发布目录应与结果闭合预检使用的语义目录保持一致。"""

    publish_dir = workflow_publish_dir(tmp_path, "official_reference_tree_ring")

    assert publish_dir == tmp_path / "external_baseline_official_reference"


def test_gpu_server_closure_uses_local_package_root(tmp_path: Path) -> None:
    """汇总服务器闭合配置应使用本地交换目录作为 package search root。"""

    report = configure_closure_environment(
        root=Path.cwd(),
        paper_run_name="pilot_paper",
        package_search_root=tmp_path,
        target_fpr_override="",
    )

    assert report["package_search_root"] == tmp_path.as_posix()
    assert report["paper_run"]["drive_result_root"] == tmp_path.as_posix()
    assert "/content/drive" not in report["package_search_root"]


def test_gpu_server_result_closure_does_not_depend_on_colab_layer() -> None:
    """汇总服务器结果闭合入口不得依赖 Colab 运行层。"""

    script_text = Path("scripts/run_gpu_server_result_closure.py").read_text(encoding="utf-8")

    assert "paper_experiments.runners.paper_result_closure" in script_text
    assert "paper_workflow.colab_utils.paper_result_closure" not in script_text
