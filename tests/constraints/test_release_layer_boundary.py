"""验证当前五层发布边界和依赖方向。"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.constraint
def test_main_contains_only_core_method_modules() -> None:
    """main 只允许方法定义、最小数学工具、正式组件和唯一合成算子。"""

    assert {path.name for path in Path("main").iterdir() if path.name != "__pycache__"} == {
        "__init__.py",
        "README.md",
        "core",
        "methods",
    }
    assert {path.name for path in Path("main/core").glob("*.py")} == {
        "__init__.py",
        "digest.py",
        "keyed_prg.py",
    }
    assert {path.name for path in Path("main/methods").iterdir() if path.name != "__pycache__"} == {
        "__init__.py",
        "carrier",
        "detection",
        "geometry",
        "method_definition.py",
        "semantic",
        "subspace",
        "update_composition.py",
    }


@pytest.mark.constraint
def test_formal_experiment_runners_live_outside_main() -> None:
    """数据集、消融和运行编排必须位于 experiments。"""

    required = {
        Path("experiments/runners/semantic_watermark_runtime.py"),
        Path("experiments/runners/image_only_dataset_runtime.py"),
        Path("experiments/ablations/runtime_rerun.py"),
    }
    assert all(path.is_file() for path in required)


@pytest.mark.constraint
def test_paper_analysis_and_baselines_live_outside_experiments() -> None:
    """论文证据审计和 baseline 聚合必须位于 paper_experiments。"""

    assert Path("paper_experiments/analysis/paper_evidence_audit.py").is_file()
    assert Path("paper_experiments/baselines/formal_import.py").is_file()


@pytest.mark.constraint
def test_notebook_workflow_archive_registry_stays_in_outer_layer() -> None:
    """Notebook 与 baseline 归档词表不得由 experiments 内层维护。"""

    inner_source = Path(
        "experiments/runtime/archive_naming.py"
    ).read_text(encoding="utf-8")
    outer_path = Path(
        "paper_workflow/notebook_utils/workflow_archive_naming.py"
    )
    outer_source = outer_path.read_text(encoding="utf-8")

    assert "WORKFLOW_ARCHIVE_PREFIXES" not in inner_source
    assert "external_baseline_method_faithful" not in inner_source
    assert "official_reference_tree_ring" not in inner_source
    assert outer_path.is_file()
    assert "WORKFLOW_ARCHIVE_PREFIXES" in outer_source
    assert "external_baseline_method_faithful" in outer_source


@pytest.mark.constraint
def test_formal_experiment_surface_excludes_component_entrypoints() -> None:
    """正式实验表面不得包含绕过完整方法 runner 的分量级入口。"""

    forbidden = (
        "experiments/runners/aligned_rescoring.py",
        "experiments/runners/attention_geometry_capture.py",
        "experiments/runners/attention_latent_injection.py",
        "experiments/runners/minimal_latent_injection.py",
        "experiments/runners/threshold_calibration.py",
        "scripts/run_core_smoke.py",
        "scripts/write_internal_ablation_outputs.py",
    )
    assert all(not Path(path).exists() for path in forbidden)
