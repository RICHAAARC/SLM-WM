"""验证独立科学入口统一采用配置层定义的论文运行默认值."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.runtime import semantic_watermark_scientific_session as dispatcher
from paper_experiments.runners.external_baseline_method_faithful import (
    ExternalBaselineMethodFaithfulConfig,
)
from paper_experiments.runners import t2smark_formal_reproduction as t2smark
from scripts import semantic_watermark_scientific_workflow as scientific_workflow


pytestmark = pytest.mark.quick


def test_scientific_dispatcher_uses_probe_default_without_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """环境未指定层级时, 子解释器调度路径必须落入 probe_paper."""

    progress_path = (
        tmp_path
        / "outputs"
        / "image_only_dataset_runtime"
        / "probe_paper"
        / "dataset_runtime_progress.json"
    )
    progress_path.parent.mkdir(parents=True)
    progress_path.write_text(
        json.dumps({"protocol_decision": "resume_required"}),
        encoding="utf-8",
    )
    monkeypatch.delenv("SLM_WM_PAPER_RUN_NAME", raising=False)
    monkeypatch.setattr(dispatcher, "ROOT", tmp_path)
    monkeypatch.setattr(
        dispatcher,
        "_run_child",
        lambda command_tail: {
            "argv": list(command_tail),
            "return_code": 0,
            "stdout": "",
            "stderr": "",
        },
    )

    report = dispatcher.run_scientific_commands(run_formal_ablation=False)

    assert report["paper_run_name"] == "probe_paper"
    assert report["artifact_state"]["runtime_progress_path"].endswith(
        "probe_paper/dataset_runtime_progress.json"
    )


def test_outer_scientific_entry_uses_resolved_config_for_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """外层入口必须先解析配置, 再以同一层级身份搜索闭合归档."""

    captured: dict[str, object] = {}
    paper_run = SimpleNamespace(run_name="probe_paper", target_fpr=0.1)
    monkeypatch.delenv("SLM_WM_PAPER_RUN_NAME", raising=False)
    monkeypatch.setattr(
        scientific_workflow,
        "build_paper_run_config",
        lambda root: paper_run,
    )

    def recover(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "all_expected_roles_recovered": True,
            "recovered_archive_roles": sorted(kwargs["expected_roles"]),
        }

    monkeypatch.setattr(scientific_workflow, "_recover_closed_archives", recover)

    summary = scientific_workflow.run_semantic_watermark_image_only_session(
        tmp_path
    )

    assert captured["paper_run_name"] == "probe_paper"
    assert captured["target_fpr"] == 0.1
    assert summary["paper_run_name"] == "probe_paper"
    assert summary["workflow_decision"] == "closed_archives_recovered"


def test_external_baseline_config_defaults_match_probe_scale() -> None:
    """公开 baseline 配置默认值必须与统一 probe_paper 入口一致."""

    config = ExternalBaselineMethodFaithfulConfig(primary_baseline_id="tree_ring")

    assert config.prompt_set == "probe_paper"
    assert Path(config.prompt_file).name == "paper_main_probe_paper_prompts.txt"
    assert config.target_fpr == 0.1
    assert config.primary_baseline_max_samples == 70


@pytest.mark.parametrize(
    ("paper_run_name", "expected_prompt_count", "expected_target_fpr"),
    (
        ("probe_paper", 70, 0.1),
        ("pilot_paper", 700, 0.01),
        ("full_paper", 7000, 0.001),
    ),
)
def test_t2smark_default_identity_tracks_paper_run_scale(
    paper_run_name: str,
    expected_prompt_count: int,
    expected_target_fpr: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2SMark 默认目录身份、Prompt 规模和 FPR 必须随统一配置切换."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", paper_run_name)
    for environment_name in (
        "SLM_WM_PROMPT_SET",
        "SLM_WM_PROMPT_FILE",
        "SLM_WM_PAPER_RUN_SAMPLE_COUNT",
        "SLM_WM_T2SMARK_FORMAL_RUN_NAME",
        "SLM_WM_T2SMARK_FORMAL_PROMPT_FILE",
        "SLM_WM_T2SMARK_FORMAL_PROMPT_LIMIT",
        "SLM_WM_T2SMARK_FORMAL_TARGET_FPR",
    ):
        monkeypatch.delenv(environment_name, raising=False)

    config = t2smark.build_default_config()

    assert config.prompt_set == paper_run_name
    assert config.t2smark_run_name == f"t2smark_sd35_medium_{paper_run_name}"
    assert config.prompt_limit == expected_prompt_count
    assert config.minimum_prompt_protocol_count == expected_prompt_count
    assert config.target_fpr == expected_target_fpr


def test_t2smark_protocol_rejects_cross_scale_run_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式协议不得把 probe_paper 数据写入带 pilot_paper 身份的运行目录."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)
    config = replace(
        t2smark.build_default_config(),
        t2smark_run_name="t2smark_sd35_medium_pilot_paper",
    )

    with pytest.raises(ValueError, match="run name"):
        t2smark.validate_t2smark_formal_protocol_config(
            config,
            root_path=Path(".").resolve(),
        )
