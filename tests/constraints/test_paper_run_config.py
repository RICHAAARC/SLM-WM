"""验证论文运行配置的集中解析边界."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from experiments.protocol.paper_run_config import (
    DEFAULT_DRIVE_ROOT,
    PaperRunPromptContract,
    build_paper_run_config,
    derive_dataset_level_quality_minimum_count,
    derive_minimum_clean_negative_count,
    parse_record_limit,
    resolve_count_from_environment,
    shared_experiment_settings,
    shared_method_settings,
    validate_frozen_paper_run_target_fpr,
)
from paper_workflow.colab_utils.paper_run_environment import (
    _resolve_paper_run_name,
)


def write_method_config(root: Path) -> None:
    """把唯一正式方法配置显式复制到受测试目标根目录。"""

    source = Path(__file__).resolve().parents[2] / "configs" / "model_sd35.yaml"
    target = root / "configs" / "model_sd35.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())


def write_prompt_file(path: Path, count: int) -> None:
    """写出受测试控制的 prompt 文件, 用于避免依赖仓库外部状态."""

    path.parent.mkdir(parents=True, exist_ok=True)
    write_method_config(path.parent.parent)
    path.write_text("\n".join(f"a controlled prompt {index}" for index in range(count)) + "\n", encoding="utf-8")


def write_prompt_contract(
    root: Path,
    run_name: str,
    count: int,
) -> PaperRunPromptContract:
    """显式构造测试 Prompt 依赖, 不冒充正式注册表输入。"""

    write_method_config(root)
    relative_path = Path("configs") / f"paper_main_{run_name}_prompts.txt"
    path = root / relative_path
    write_prompt_file(path, count)
    return PaperRunPromptContract(
        run_name=run_name,
        prompt_file=relative_path.as_posix(),
        expected_prompt_count=count,
        prompt_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


@pytest.mark.constraint
def test_colab_environment_resolves_probe_paper_without_explicit_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Colab 环境 helper 与协议解析层必须共享 probe_paper 默认值."""

    monkeypatch.delenv("SLM_WM_PAPER_RUN_NAME", raising=False)

    assert _resolve_paper_run_name() == "probe_paper"


@pytest.mark.constraint
def test_paper_run_config_resolves_probe_paper_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无显式运行层级时应唯一解析为 probe_paper 并使用全部 Prompt."""

    prompt_contract = write_prompt_contract(tmp_path, "probe_paper", 7)
    monkeypatch.delenv("SLM_WM_PAPER_RUN_NAME", raising=False)
    monkeypatch.delenv("SLM_WM_DRIVE_RESULT_ROOT", raising=False)
    monkeypatch.delenv("SLM_WM_PAPER_RUN_SAMPLE_COUNT", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)

    config = build_paper_run_config(root=tmp_path, prompt_contract=prompt_contract)

    assert config.run_name == "probe_paper"
    assert config.prompt_set == "probe_paper"
    assert config.prompt_count == 7
    assert config.sample_count == 7
    assert config.drive_result_root == f"{DEFAULT_DRIVE_ROOT}/probe_paper_results"
    assert config.protocol_profile == "probe_paper_fixed_fpr_0_1"
    assert config.target_fpr == 0.1
    assert config.minimum_clean_negative_count == 34
    assert config.dataset_level_quality_minimum_count == 70
    assert config.drive_dir("aligned_rescoring").endswith(
        "/probe_paper_results/randomization_repeats/seed_00_key_00/aligned_rescoring"
    )


@pytest.mark.constraint
def test_paper_run_config_switches_to_full_paper_without_notebook_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """切换环境变量即可让同一入口使用 full_paper prompt 与 Drive 根目录."""

    prompt_contract = write_prompt_contract(tmp_path, "full_paper", 11)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "full_paper")
    monkeypatch.delenv("SLM_WM_DRIVE_RESULT_ROOT", raising=False)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_SAMPLE_COUNT", "all")
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)

    config = build_paper_run_config(root=tmp_path, prompt_contract=prompt_contract)

    assert config.run_name == "full_paper"
    assert config.prompt_set == "full_paper"
    assert config.prompt_count == 11
    assert config.sample_count == 11
    assert config.drive_result_root == f"{DEFAULT_DRIVE_ROOT}/full_paper_results"
    assert config.protocol_profile == "full_paper_fixed_fpr_0_001"
    assert config.target_fpr == 0.001
    assert config.minimum_clean_negative_count == 3400
    assert config.dataset_level_quality_minimum_count == 7000
    assert config.drive_dir("threshold_calibration").endswith(
        "/full_paper_results/randomization_repeats/seed_00_key_00/threshold_calibration"
    )


@pytest.mark.constraint
def test_paper_run_config_switches_to_pilot_paper_with_explicit_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式选择 pilot_paper 时应使用其完整 Prompt 协议与统计强度."""

    prompt_contract = write_prompt_contract(tmp_path, "pilot_paper", 700)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "pilot_paper")
    monkeypatch.delenv("SLM_WM_DRIVE_RESULT_ROOT", raising=False)
    monkeypatch.delenv("SLM_WM_PAPER_RUN_SAMPLE_COUNT", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)

    config = build_paper_run_config(root=tmp_path, prompt_contract=prompt_contract)

    assert config.run_name == "pilot_paper"
    assert config.prompt_set == "pilot_paper"
    assert config.prompt_count == 700
    assert config.sample_count == 700
    assert config.drive_result_root == f"{DEFAULT_DRIVE_ROOT}/pilot_paper_results"
    assert config.protocol_profile == "pilot_paper_fixed_fpr_0_01"
    assert config.target_fpr == 0.01
    assert config.minimum_clean_negative_count == 340
    assert config.dataset_level_quality_minimum_count == 700
    assert config.drive_dir("aligned_rescoring").endswith(
        "/pilot_paper_results/randomization_repeats/seed_00_key_00/aligned_rescoring"
    )


@pytest.mark.constraint
def test_paper_run_levels_share_method_settings_except_protocol_scale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """三类论文运行层级共享方法参数, 门禁规模只能由样本规模和 fixed-FPR 派生。"""

    contracts = {
        run_name: write_prompt_contract(tmp_path, run_name, count)
        for run_name, count in (
            ("probe_paper", 70),
            ("pilot_paper", 700),
            ("full_paper", 7000),
        )
    }
    monkeypatch.delenv("SLM_WM_DRIVE_RESULT_ROOT", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)
    monkeypatch.delenv("SLM_WM_PAPER_RUN_SAMPLE_COUNT", raising=False)

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    probe_config = build_paper_run_config(
        root=tmp_path,
        prompt_contract=contracts["probe_paper"],
    )
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "pilot_paper")
    pilot_config = build_paper_run_config(
        root=tmp_path,
        prompt_contract=contracts["pilot_paper"],
    )
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "full_paper")
    full_config = build_paper_run_config(
        root=tmp_path,
        prompt_contract=contracts["full_paper"],
    )

    assert shared_method_settings(probe_config) == shared_method_settings(pilot_config)
    assert shared_method_settings(pilot_config) == shared_method_settings(full_config)
    assert shared_experiment_settings(probe_config) == shared_experiment_settings(
        pilot_config
    )
    assert shared_experiment_settings(pilot_config) == shared_experiment_settings(
        full_config
    )
    assert shared_method_settings(probe_config)["attention_module_names"] == (
        "transformer_blocks.0.attn",
        "transformer_blocks.23.attn",
    )
    assert shared_method_settings(probe_config)[
        "attention_coordinate_convention"
    ] == "normalized_xy_token_centers_corner_endpoints_v1"
    assert shared_method_settings(probe_config)[
        "attention_grid_align_corners"
    ] is True
    frozen_settings = shared_method_settings(probe_config)
    assert len(frozen_settings["formal_method_config_digest"]) == 64
    assert frozen_settings["pipeline_class_name"].endswith(
        ".StableDiffusion3Pipeline"
    )
    assert frozen_settings["vae_class_name"].endswith(".AutoencoderKL")
    assert frozen_settings["transformer_class_name"].endswith(
        ".SD3Transformer2DModel"
    )
    assert frozen_settings["scheduler_class_name"].endswith(
        ".FlowMatchEulerDiscreteScheduler"
    )
    assert (
        frozen_settings["vae_scaling_factor"],
        frozen_settings["vae_shift_factor"],
        frozen_settings["latent_torch_dtype"],
        frozen_settings["vision_torch_dtype"],
    ) == (1.5305, 0.0609, "float16", "float32")
    assert frozen_settings["public_detection_schedule_index"] == 7
    assert frozen_settings["public_detection_noise_prg_protocol"] == (
        "sha256_counter_box_muller_float32_v1"
    )
    assert frozen_settings["public_detection_noise_domain"] == (
        "public_image_only_qk_detection_noise_v1"
    )
    assert frozen_settings["public_detection_condition_text"] == ""
    assert frozen_settings["risk_signal_calibration_protocol"] == (
        "analytic_bounded_branch_signals_v1"
    )
    assert frozen_settings["risk_eligibility_comparison"] == "strict_less_than"
    assert frozen_settings["risk_neutral_texture_value"] == 0.5
    assert frozen_settings["risk_budget_broadcast_protocol"] == (
        "per_sample_hw_repeat_channels_nchw_v1"
    )
    assert frozen_settings["risk_zero_support_protocol"] == (
        "exact_zero_direction_or_fail_closed"
    )
    assert frozen_settings["lf_content_risk_config"][
        "eligibility_threshold"
    ] == 0.55
    assert frozen_settings["tail_robust_risk_config"][
        "texture_preference"
    ] == "prefer"
    assert frozen_settings["attention_geometry_risk_config"][
        "attention_instability_weight"
    ] == 0.30
    assert frozen_settings["qr_reference_solve_protocol"] == (
        "right_upper_triangular_solve_without_explicit_inverse_v1"
    )
    assert frozen_settings["quantized_branch_composition_protocol"] == (
        "float32_ordered_branch_sum_add_float32_latent_single_cast_v1"
    )
    assert frozen_settings["quantized_branch_composition_order"] == (
        "lf_content",
        "tail_robust",
        "attention_geometry",
    )
    assert frozen_settings["combined_budget_envelope_rule"] == (
        "sum_active_branch_envelopes"
    )
    assert frozen_settings[
        "quantized_budget_envelope_backtracking_maximum_steps"
    ] == 24
    assert probe_config.target_fpr == 0.1
    assert pilot_config.target_fpr == 0.01
    assert full_config.target_fpr == 0.001
    assert probe_config.minimum_clean_negative_count == 34
    assert pilot_config.minimum_clean_negative_count == 340
    assert full_config.minimum_clean_negative_count == 3400
    assert probe_config.dataset_level_quality_minimum_count == 70
    assert pilot_config.dataset_level_quality_minimum_count == 700
    assert full_config.dataset_level_quality_minimum_count == 7000
    assert {probe_config.sample_count, pilot_config.sample_count, full_config.sample_count} == {70, 700, 7000}
    assert len({probe_config.prompt_file, pilot_config.prompt_file, full_config.prompt_file}) == 3


@pytest.mark.constraint
def test_paper_run_selects_registered_crossed_randomization_repeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """环境变量只能选择登记的生成种子与密钥交叉重复."""

    prompt_contract = write_prompt_contract(tmp_path, "probe_paper", 7)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    monkeypatch.setenv("SLM_WM_RANDOMIZATION_REPEAT_ID", "seed_02_key_01")
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)

    config = build_paper_run_config(
        root=tmp_path,
        prompt_contract=prompt_contract,
    )

    assert config.randomization_repeat_id == "seed_02_key_01"
    assert config.generation_seed_index == 2
    assert config.generation_seed_offset == 2_000_003
    assert config.watermark_key_index == 1
    assert config.formal_randomization_repeat_count == 9
    assert config.drive_dir("runtime").endswith(
        "/randomization_repeats/seed_02_key_01/runtime"
    )


@pytest.mark.constraint
def test_paper_run_gate_counts_are_derived_from_scale_and_fixed_fpr() -> None:
    """门禁计数应由样本规模和 fixed-FPR 标准派生, 不应成为独立协议分叉。"""

    assert derive_minimum_clean_negative_count(70, 0.1) == 34
    assert derive_minimum_clean_negative_count(700, 0.01) == 340
    assert derive_minimum_clean_negative_count(7000, 0.001) == 3400
    assert derive_dataset_level_quality_minimum_count(70) == 70
    assert derive_dataset_level_quality_minimum_count(700) == 700
    assert derive_dataset_level_quality_minimum_count(7000) == 7000


@pytest.mark.constraint
def test_record_limit_parser_uses_prompt_count_for_unbounded_tokens() -> None:
    """all、none、unlimited 和非正数均应回落到当前 prompt 数量."""

    assert parse_record_limit("all", prompt_count=700) == 700
    assert parse_record_limit("none", prompt_count=700) == 700
    assert parse_record_limit("unlimited", prompt_count=700) == 700
    assert parse_record_limit("0", prompt_count=700) == 700
    assert parse_record_limit("17", prompt_count=700) == 17


@pytest.mark.constraint
def test_count_environment_resolver_inherits_current_paper_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """业务 helper 读取单项计数环境变量时应复用统一论文运行配置."""

    prompt_contract = write_prompt_contract(tmp_path, "pilot_paper", 13)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "pilot_paper")
    monkeypatch.setenv("SLM_WM_EXAMPLE_COUNT", "all")
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)

    assert resolve_count_from_environment(
        "SLM_WM_EXAMPLE_COUNT",
        root=tmp_path,
        prompt_contract=prompt_contract,
    ) == 13


@pytest.mark.constraint
def test_paper_run_config_rejects_stale_prompt_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """切换到 full_paper 时不得静默沿用 pilot_paper prompt 环境变量."""

    prompt_contract = write_prompt_contract(tmp_path, "full_paper", 11)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "full_paper")
    monkeypatch.setenv("SLM_WM_PROMPT_SET", "pilot_paper")
    monkeypatch.setenv("SLM_WM_PROMPT_FILE", "configs/paper_main_pilot_paper_prompts.txt")

    with pytest.raises(ValueError, match="SLM_WM_PROMPT_SET"):
        build_paper_run_config(root=tmp_path, prompt_contract=prompt_contract)

    monkeypatch.setenv("SLM_WM_PROMPT_SET", "full_paper")
    with pytest.raises(ValueError, match="SLM_WM_PROMPT_FILE"):
        build_paper_run_config(root=tmp_path, prompt_contract=prompt_contract)


@pytest.mark.constraint
def test_formal_prompt_contract_accepts_repository_governed_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式入口必须能核验仓库内规范 Prompt 的数量和字节摘要。"""

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)

    config = build_paper_run_config(root=root)

    assert config.prompt_count == 70


@pytest.mark.constraint
def test_formal_prompt_contract_uses_packaged_source_for_artifact_only_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """仅包含产物的隔离根目录应复用当前提交内受治理 Prompt。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)
    write_method_config(tmp_path)

    config = build_paper_run_config(root=tmp_path)

    assert config.prompt_count == 70
    assert config.prompt_file == "configs/paper_main_probe_paper_prompts.txt"


@pytest.mark.constraint
def test_formal_prompt_contract_rejects_unregistered_root_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """隔离根目录一旦提供规范路径文件, 就必须通过当前摘要校验。"""

    prompt_path = tmp_path / "configs" / "paper_main_probe_paper_prompts.txt"
    write_prompt_file(prompt_path, 70)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)

    with pytest.raises(ValueError, match="SHA-256"):
        build_paper_run_config(root=tmp_path)


@pytest.mark.constraint
def test_formal_prompt_contract_rejects_external_same_name_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同名外部文件不得通过仅比较 basename 的旧边界。"""

    root = Path(__file__).resolve().parents[2]
    external_path = tmp_path / "paper_main_probe_paper_prompts.txt"
    write_prompt_file(external_path, 70)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.setenv("SLM_WM_PROMPT_FILE", str(external_path))

    with pytest.raises(ValueError, match="精确匹配"):
        build_paper_run_config(root=root)


@pytest.mark.constraint
def test_formal_prompt_contract_rejects_count_and_digest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式入口必须分别拒绝数量漂移和保持数量时的内容漂移。"""

    repository_root = Path(__file__).resolve().parents[2]
    write_method_config(tmp_path)
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    registry_source = repository_root / "configs" / "prompt_source_registry.json"
    manifest_source = repository_root / "configs" / "prompt_selection_manifest.jsonl"
    prompt_source = repository_root / "configs" / "paper_main_probe_paper_prompts.txt"
    (configs_dir / registry_source.name).write_bytes(registry_source.read_bytes())
    (configs_dir / manifest_source.name).write_bytes(manifest_source.read_bytes())
    prompt_path = configs_dir / prompt_source.name
    prompt_path.write_bytes(prompt_source.read_bytes())
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)

    lines = prompt_path.read_text(encoding="utf-8").splitlines()
    prompt_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="逐字节重建"):
        build_paper_run_config(root=tmp_path)

    prompt_path.write_bytes(prompt_source.read_bytes())
    drifted_lines = prompt_path.read_text(encoding="utf-8").splitlines()
    drifted_lines[0] = drifted_lines[0] + " content drift"
    prompt_path.write_text("\n".join(drifted_lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="逐字节重建"):
        build_paper_run_config(root=tmp_path)

@pytest.mark.parametrize(
    "paper_run_name,target_fpr",
    (
        ("probe_paper", 0.1),
        ("pilot_paper", 0.01),
        ("full_paper", 0.001),
    ),
)
def test_frozen_paper_run_target_fpr_accepts_only_registered_working_point(
    paper_run_name: str,
    target_fpr: float,
) -> None:
    """共享协议边界必须返回每个运行层级唯一冻结的统计工作点."""

    assert (
        validate_frozen_paper_run_target_fpr(paper_run_name, target_fpr)
        == target_fpr
    )
    with pytest.raises(ValueError, match="必须使用冻结值"):
        validate_frozen_paper_run_target_fpr(paper_run_name, 0.05)
    with pytest.raises(TypeError, match="必须是有限数值"):
        validate_frozen_paper_run_target_fpr(paper_run_name, True)
