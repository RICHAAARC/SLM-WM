"""验证论文运行配置的集中解析边界."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.protocol.paper_run_config import (
    DEFAULT_CONTENT_BASIS_RANK,
    DEFAULT_CONTENT_VECTOR_WIDTH,
    DEFAULT_DRIVE_ROOT,
    build_paper_run_config,
    parse_record_limit,
    resolve_count_from_environment,
    shared_method_settings,
)


def write_prompt_file(path: Path, count: int) -> None:
    """写出受测试控制的 prompt 文件, 用于避免依赖仓库外部状态."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"a controlled prompt {index}" for index in range(count)) + "\n", encoding="utf-8")


@pytest.mark.constraint
def test_paper_run_config_resolves_pilot_paper_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """默认论文运行层级应解析为 pilot_paper, 并使用该层级全部 prompt."""

    write_prompt_file(tmp_path / "configs" / "paper_main_pilot_paper_prompts.txt", 7)
    monkeypatch.delenv("SLM_WM_PAPER_RUN_NAME", raising=False)
    monkeypatch.delenv("SLM_WM_DRIVE_RESULT_ROOT", raising=False)
    monkeypatch.delenv("SLM_WM_PAPER_RUN_SAMPLE_COUNT", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)

    config = build_paper_run_config(root=tmp_path)

    assert config.run_name == "pilot_paper"
    assert config.prompt_set == "pilot_paper"
    assert config.prompt_count == 7
    assert config.sample_count == 7
    assert config.drive_result_root == f"{DEFAULT_DRIVE_ROOT}/pilot_paper_results"
    assert config.protocol_profile == "pilot_paper_fixed_fpr_0_01"
    assert config.target_fpr == 0.01
    assert config.content_vector_width == DEFAULT_CONTENT_VECTOR_WIDTH
    assert config.content_basis_rank == DEFAULT_CONTENT_BASIS_RANK
    assert config.drive_dir("aligned_rescoring").endswith("/pilot_paper_results/aligned_rescoring")


@pytest.mark.constraint
def test_paper_run_config_switches_to_full_paper_without_notebook_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """切换环境变量即可让同一入口使用 full_paper prompt 与 Drive 根目录."""

    write_prompt_file(tmp_path / "configs" / "paper_main_full_paper_prompts.txt", 11)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "full_paper")
    monkeypatch.delenv("SLM_WM_DRIVE_RESULT_ROOT", raising=False)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_SAMPLE_COUNT", "all")
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)

    config = build_paper_run_config(root=tmp_path)

    assert config.run_name == "full_paper"
    assert config.prompt_set == "full_paper"
    assert config.prompt_count == 11
    assert config.sample_count == 11
    assert config.drive_result_root == f"{DEFAULT_DRIVE_ROOT}/full_paper_results"
    assert config.protocol_profile == "full_paper_fixed_fpr_0_001"
    assert config.target_fpr == 0.001
    assert config.content_vector_width == DEFAULT_CONTENT_VECTOR_WIDTH
    assert config.content_basis_rank == DEFAULT_CONTENT_BASIS_RANK
    assert config.drive_dir("threshold_calibration").endswith("/full_paper_results/threshold_calibration")


@pytest.mark.constraint
def test_paper_run_config_resolves_probe_paper_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """probe_paper 应使用完整论文入口, 但采用较小 prompt 数量和较宽 fixed-FPR。"""

    write_prompt_file(tmp_path / "configs" / "paper_main_probe_paper_prompts.txt", 5)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    monkeypatch.delenv("SLM_WM_DRIVE_RESULT_ROOT", raising=False)
    monkeypatch.delenv("SLM_WM_PAPER_RUN_SAMPLE_COUNT", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)

    config = build_paper_run_config(root=tmp_path)

    assert config.run_name == "probe_paper"
    assert config.prompt_set == "probe_paper"
    assert config.prompt_count == 5
    assert config.sample_count == 5
    assert config.drive_result_root == f"{DEFAULT_DRIVE_ROOT}/probe_paper_results"
    assert config.protocol_profile == "probe_paper_fixed_fpr_0_1"
    assert config.target_fpr == 0.1
    assert config.content_vector_width == DEFAULT_CONTENT_VECTOR_WIDTH
    assert config.content_basis_rank == DEFAULT_CONTENT_BASIS_RANK
    assert config.drive_dir("aligned_rescoring").endswith("/probe_paper_results/aligned_rescoring")


@pytest.mark.constraint
def test_paper_run_levels_share_method_settings_except_claim_fpr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """三类论文运行层级共享方法参数, 但使用不同样本规模和 fixed-FPR。"""

    write_prompt_file(tmp_path / "configs" / "paper_main_probe_paper_prompts.txt", 5)
    write_prompt_file(tmp_path / "configs" / "paper_main_pilot_paper_prompts.txt", 7)
    write_prompt_file(tmp_path / "configs" / "paper_main_full_paper_prompts.txt", 11)
    monkeypatch.delenv("SLM_WM_DRIVE_RESULT_ROOT", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)
    monkeypatch.delenv("SLM_WM_PAPER_RUN_SAMPLE_COUNT", raising=False)

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    probe_config = build_paper_run_config(root=tmp_path)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "pilot_paper")
    pilot_config = build_paper_run_config(root=tmp_path)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "full_paper")
    full_config = build_paper_run_config(root=tmp_path)

    assert shared_method_settings(probe_config) == shared_method_settings(pilot_config)
    assert shared_method_settings(pilot_config) == shared_method_settings(full_config)
    assert probe_config.target_fpr == 0.1
    assert pilot_config.target_fpr == 0.01
    assert full_config.target_fpr == 0.001
    assert {probe_config.sample_count, pilot_config.sample_count, full_config.sample_count} == {5, 7, 11}
    assert len({probe_config.prompt_file, pilot_config.prompt_file, full_config.prompt_file}) == 3


@pytest.mark.constraint
def test_record_limit_parser_uses_prompt_count_for_unbounded_tokens() -> None:
    """all、none、unlimited 和非正数均应回落到当前 prompt 数量."""

    assert parse_record_limit("all", prompt_count=600) == 600
    assert parse_record_limit("none", prompt_count=600) == 600
    assert parse_record_limit("unlimited", prompt_count=600) == 600
    assert parse_record_limit("0", prompt_count=600) == 600
    assert parse_record_limit("17", prompt_count=600) == 17


@pytest.mark.constraint
def test_count_environment_resolver_inherits_current_paper_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """业务 helper 读取单项计数环境变量时应复用统一论文运行配置."""

    write_prompt_file(tmp_path / "configs" / "paper_main_pilot_paper_prompts.txt", 13)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "pilot_paper")
    monkeypatch.setenv("SLM_WM_EXAMPLE_COUNT", "all")
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)

    assert resolve_count_from_environment("SLM_WM_EXAMPLE_COUNT", root=tmp_path) == 13


@pytest.mark.constraint
def test_paper_run_config_rejects_stale_prompt_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """切换到 full_paper 时不得静默沿用 pilot_paper prompt 环境变量."""

    write_prompt_file(tmp_path / "configs" / "paper_main_full_paper_prompts.txt", 11)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "full_paper")
    monkeypatch.setenv("SLM_WM_PROMPT_SET", "pilot_paper")
    monkeypatch.setenv("SLM_WM_PROMPT_FILE", "configs/paper_main_pilot_paper_prompts.txt")

    with pytest.raises(ValueError, match="SLM_WM_PROMPT_SET"):
        build_paper_run_config(root=tmp_path)

    monkeypatch.setenv("SLM_WM_PROMPT_SET", "full_paper")
    with pytest.raises(ValueError, match="SLM_WM_PROMPT_FILE"):
        build_paper_run_config(root=tmp_path)
