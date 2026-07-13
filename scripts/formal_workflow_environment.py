"""集中配置可脱离 Notebook 执行的正式论文工作流环境变量.

该模块位于 scripts 层, 为 GPU 服务器入口与外层 Colab 包装提供同一配置实现。
它只发布运行层级、模型身份、固定 FPR、持久化目录和科学执行参数, 不承担
Notebook 状态记录、Drive 挂载或真实模型执行。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any

from experiments.protocol.paper_run_config import (
    PROBE_PAPER_RUN_NAME,
    RUN_DEFAULTS,
    build_paper_run_config,
    normalize_paper_run_name,
)
from experiments.runtime.repository_environment import (
    require_published_formal_execution_lock,
)
from experiments.runtime.model_sources import get_model_source
from paper_experiments.runners.model_snapshot_runtime import (
    DEFAULT_SHARED_HUGGING_FACE_SNAPSHOT_ROOT,
    build_shared_hugging_face_snapshot_dir,
)

FORMAL_IMAGE_ATTACK_FAMILIES = (
    "jpeg_compression,gaussian_noise,gaussian_blur,rotation,resize,crop,crop_resize,"
    "composite_geometric_attacks,photometric_distortion_attack,img2img_regeneration,"
    "flow_matching_inversion_regeneration,sdedit_regeneration,diffusion_purification,"
    "global_editing_attack,local_editing_attack,visual_paraphrase_attack,adversarial_removal_attack"
)
CROSS_REPEAT_INVARIANT_WORKFLOW_NAMES = frozenset(
    {
        "official_reference_tree_ring",
        "official_reference_gaussian_shading",
        "official_reference_shallow_diffuse",
    }
)
SEMANTIC_WATERMARK_REPEAT_DRIVE_ROOT_ENVIRONMENT_KEY = (
    "SLM_WM_SEMANTIC_WATERMARK_REPEAT_DRIVE_ROOT"
)
_SD35_MODEL_SOURCE = get_model_source("stabilityai_stable_diffusion_3_5_medium")
_OFFICIAL_REFERENCE_DIFFUSION_MODEL_SOURCE = get_model_source("manojb_stable_diffusion_2_1_base")
_CLIP_MODEL_SOURCE = get_model_source("openai_clip_vit_base_patch32")
_OFFICIAL_REFERENCE_DIFFUSION_SNAPSHOT_DIR = build_shared_hugging_face_snapshot_dir(
    _OFFICIAL_REFERENCE_DIFFUSION_MODEL_SOURCE.repository_id,
    _OFFICIAL_REFERENCE_DIFFUSION_MODEL_SOURCE.revision,
)


@dataclass(frozen=True)
class FormalWorkflowEnvironment:
    """记录正式工作流已发布到环境变量中的论文运行配置."""

    workflow_name: str
    paper_run_name: str
    protocol_profile: str
    prompt_set: str
    prompt_file: str
    drive_result_root: str
    resume_checkpoint_dir: str
    target_fpr: str
    sample_count_token: str
    expected_sample_count: int
    minimum_clean_negative_count: str
    dataset_quality_minimum_count: str
    randomization_repeat_id: str
    generation_seed_index: int
    generation_seed_offset: int
    watermark_key_index: int
    formal_randomization_repeat_count: int
    formal_randomization_protocol_digest: str
    selected_baseline_id: str
    formal_execution_commit: str
    formal_execution_lock_digest: str
    configured_environment_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """转换为服务器或 Colab 外层可直接记录的配置摘要."""

        return asdict(self)


def _set_env(name: str, value: str | int | float) -> None:
    """无条件写入环境变量, 用于消除 Colab 内核中残留的旧配置。"""

    os.environ[name] = str(value)


def _set_default_env(name: str, value: str | int | float) -> None:
    """仅在用户未显式覆盖时写入默认环境变量。"""

    os.environ.setdefault(name, str(value))


def _target_fpr_text(value: float) -> str:
    """把目标 FPR 转为稳定短文本, 供子进程环境统一使用。"""

    return f"{float(value):g}"


def _repeat_drive_dir(paper_run: Any, artifact_role: str) -> str:
    """把活动随机化产物隔离到当前 seed-key repeat 的持久化目录."""

    # PaperRunConfig.drive_dir 已经把 repeat 身份写入规范路径, 此处不得再次追加。
    return Path(paper_run.drive_dir(artifact_role)).as_posix()


def _invariant_drive_dir(paper_run: Any, artifact_role: str) -> str:
    """为只运行一次的跨 repeat 不变证据构造独立持久化目录."""

    return (
        Path(paper_run.drive_result_root)
        / "cross_repeat_invariant"
        / artifact_role
    ).as_posix()


def _resolve_paper_run_name() -> str:
    """解析外层传入的论文运行层级, 无显式输入时使用 probe_paper."""

    return normalize_paper_run_name(
        os.environ.get("SLM_WM_PAPER_RUN_NAME", PROBE_PAPER_RUN_NAME)
    )


def _configure_common_paper_run_environment(
    repository_root: str | Path,
) -> tuple[Any, str, str]:
    """配置所有论文 workflow 共享的环境变量。"""

    paper_run_name = _resolve_paper_run_name()
    defaults = RUN_DEFAULTS[paper_run_name]
    sample_count_token = os.environ.get("SLM_WM_PAPER_RUN_SAMPLE_COUNT", "all")

    # 这些字段必须从当前运行层级重新派生, 避免 Colab 同一内核重跑时沿用旧值。
    _set_env("SLM_WM_PAPER_RUN_NAME", paper_run_name)
    _set_env("SLM_WM_PROMPT_SET", defaults["prompt_set"])
    _set_env("SLM_WM_PROMPT_FILE", defaults["prompt_file"])
    _set_env("SLM_WM_DRIVE_RESULT_ROOT", defaults["drive_result_root"])
    _set_env("SLM_WM_PAPER_RUN_SAMPLE_COUNT", sample_count_token)
    _set_env("SLM_WM_PAPER_RUN_TARGET_FPR", defaults["target_fpr"])
    os.environ.pop("SLM_WM_PAPER_RUN_MINIMUM_CLEAN_NEGATIVE_COUNT", None)
    os.environ.pop("SLM_WM_PAPER_RUN_DATASET_QUALITY_MINIMUM_COUNT", None)
    os.environ.pop("SLM_WM_RESUME_CHECKPOINT_DIR", None)

    paper_run = build_paper_run_config(repository_root)
    _set_env(
        "SLM_WM_RANDOMIZATION_REPEAT_ID",
        paper_run.randomization_repeat_id,
    )
    target_fpr_text = _target_fpr_text(paper_run.target_fpr)
    _set_env("SLM_WM_PAPER_RUN_TARGET_FPR", target_fpr_text)
    _set_env("SLM_WM_PROTOCOL_PROFILE", paper_run.protocol_profile)
    _set_env("SLM_WM_PAPER_RUN_EXPECTED_SAMPLE_COUNT", paper_run.sample_count)
    _set_env("SLM_WM_PAPER_RUN_MINIMUM_CLEAN_NEGATIVE_COUNT", paper_run.minimum_clean_negative_count)
    _set_env("SLM_WM_PAPER_RUN_DATASET_QUALITY_MINIMUM_COUNT", paper_run.dataset_level_quality_minimum_count)
    return paper_run, sample_count_token, target_fpr_text


def _configure_attention_geometry(paper_run: Any, sample_count_token: str, target_fpr_text: str) -> None:
    _set_env("SLM_WM_DRIVE_OUTPUT_DIR", paper_run.drive_dir("attention_geometry"))
    _set_default_env("SLM_WM_ATTENTION_CAPTURE_COUNT", "16")
    _set_default_env("SLM_WM_ATTENTION_TOKEN_COUNT", "32")


def _configure_semantic_watermark_image_only(
    paper_run: Any,
    sample_count_token: str,
    target_fpr_text: str,
) -> None:
    """配置真实科学算子、仅图像检测和正式消融的 Colab 续跑入口。"""

    image_only_drive_dir = _repeat_drive_dir(
        paper_run,
        "image_only_dataset_runtime",
    )
    _set_env(
        SEMANTIC_WATERMARK_REPEAT_DRIVE_ROOT_ENVIRONMENT_KEY,
        Path(image_only_drive_dir).parent.as_posix(),
    )
    _set_env("SLM_WM_DRIVE_OUTPUT_DIR", image_only_drive_dir)
    _set_env("SLM_WM_IMAGE_ONLY_RUNTIME_DRIVE_DIR", image_only_drive_dir)
    _set_env(
        "SLM_WM_DATASET_QUALITY_DRIVE_DIR",
        _repeat_drive_dir(paper_run, "dataset_level_quality"),
    )
    _set_env(
        "SLM_WM_RUNTIME_RERUN_ABLATION_DRIVE_DIR",
        _repeat_drive_dir(paper_run, "runtime_rerun_ablation"),
    )
    _set_env(
        "SLM_WM_RESUME_CHECKPOINT_DIR",
        _repeat_drive_dir(paper_run, "semantic_watermark_resume_checkpoint"),
    )
    _set_default_env("SLM_WM_MAX_NEW_PROMPTS_PER_SESSION", "5")
    _set_default_env("SLM_WM_MAX_NEW_ABLATION_RUNS_PER_SESSION", "5")
    _set_default_env("SLM_WM_INCEPTION_BATCH_SIZE", "32")
    _set_default_env("SLM_WM_ENABLE_DIFFUSION_ATTACKS", "1")
    _set_default_env("SLM_WM_DDIM_MODEL_CPU_OFFLOAD", "1")


def _configure_attention_latent_injection(paper_run: Any, sample_count_token: str, target_fpr_text: str) -> None:
    _set_env("SLM_WM_DRIVE_OUTPUT_DIR", paper_run.drive_dir("attention_latent_injection"))
    _set_env("SLM_WM_ATTENTION_GEOMETRY_DRIVE_DIR", paper_run.drive_dir("attention_geometry"))
    _set_env("SLM_WM_ATTENTION_SUBSPACE_RECORDS", sample_count_token)
    _set_default_env("SLM_WM_ATTENTION_RUNTIME_STRENGTH", "0.025")


def _configure_aligned_rescoring(paper_run: Any, sample_count_token: str, target_fpr_text: str) -> None:
    _set_env("SLM_WM_DRIVE_OUTPUT_DIR", paper_run.drive_dir("aligned_rescoring"))
    _set_env("SLM_WM_ATTENTION_GEOMETRY_DRIVE_DIR", paper_run.drive_dir("attention_geometry"))
    _set_env("SLM_WM_ALIGNED_RESCORING_SUBSPACE_RECORDS", sample_count_token)
    _set_env("SLM_WM_ALIGNED_RESCORING_CARRIER_COUNT", sample_count_token)
    _set_default_env("SLM_WM_ATTENTION_RUNTIME_STRENGTH", "0.025")
    _set_default_env("SLM_WM_ENABLE_PAIR_PERCEPTUAL_METRICS", "1")
    _set_default_env("SLM_WM_REQUIRE_PAIR_PERCEPTUAL_METRICS", "1")
    _set_env("SLM_WM_CLIP_MODEL_ID", _CLIP_MODEL_SOURCE.repository_id)
    _set_env("SLM_WM_CLIP_MODEL_REVISION", _CLIP_MODEL_SOURCE.revision)
    _set_default_env("SLM_WM_LPIPS_NETWORK", "alex")
    _set_default_env("SLM_WM_PERCEPTUAL_METRIC_DEVICE", "cpu")
    _set_default_env("SLM_WM_ENABLE_PIPELINE_PROGRESS_BAR", "0")
    _set_default_env("SLM_WM_ENABLE_CARRIER_PROGRESS_BAR", "1")


def _configure_threshold_calibration(paper_run: Any, sample_count_token: str, target_fpr_text: str) -> None:
    _set_env("SLM_WM_THRESHOLD_CALIBRATION_DRIVE_DIR", paper_run.drive_dir("threshold_calibration"))
    _set_env("SLM_WM_ATTENTION_INJECTION_DRIVE_DIR", paper_run.drive_dir("attention_latent_injection"))
    _set_env("SLM_WM_ALIGNED_RESCORING_DRIVE_DIR", paper_run.drive_dir("aligned_rescoring"))
    _set_env("SLM_WM_THRESHOLD_TARGET_FPR", target_fpr_text)
    _set_env("SLM_WM_THRESHOLD_MAX_CONTENT_RECORDS", "all")
    _set_env("SLM_WM_THRESHOLD_MINIMUM_CLEAN_NEGATIVE_COUNT", paper_run.minimum_clean_negative_count)
    _set_default_env("SLM_WM_ATTENTION_INJECTION_PACKAGE_PATTERN", "attention_latent_injection_package_*.zip")
    _set_default_env("SLM_WM_ALIGNED_RESCORING_PACKAGE_PATTERN", "aligned_rescoring_package_*.zip")


def _configure_real_attack_evaluation(paper_run: Any, sample_count_token: str, target_fpr_text: str) -> None:
    _set_env("SLM_WM_DRIVE_OUTPUT_DIR", paper_run.drive_dir("real_attack_evaluation"))
    _set_env("SLM_WM_ALIGNED_RESCORING_DRIVE_DIR", paper_run.drive_dir("aligned_rescoring"))
    _set_env("SLM_WM_THRESHOLD_CALIBRATION_DRIVE_DIR", paper_run.drive_dir("threshold_calibration"))
    _set_default_env("SLM_WM_REAL_ATTACK_SOURCE_IMAGE_DIR", "outputs/aligned_rescoring/aligned_images")
    _set_env("SLM_WM_REAL_ATTACK_SOURCE_COUNT", sample_count_token)
    _set_default_env("SLM_WM_REQUIRE_ALL_REGEN_ATTACKS", "1")
    _set_default_env("SLM_WM_ENABLE_PIPELINE_PROGRESS_BAR", "0")
    _set_default_env("SLM_WM_ENABLE_ATTACK_PROGRESS_BAR", "1")


def _configure_conventional_geometric_attack_evaluation(
    paper_run: Any,
    sample_count_token: str,
    target_fpr_text: str,
) -> None:
    _set_env(
        "SLM_WM_CONVENTIONAL_GEOMETRIC_ATTACK_DRIVE_DIR",
        paper_run.drive_dir("conventional_geometric_attack_evaluation"),
    )
    _set_env("SLM_WM_ALIGNED_RESCORING_DRIVE_DIR", paper_run.drive_dir("aligned_rescoring"))
    _set_env("SLM_WM_THRESHOLD_CALIBRATION_DRIVE_DIR", paper_run.drive_dir("threshold_calibration"))
    _set_default_env("SLM_WM_REAL_ATTACK_SOURCE_IMAGE_DIR", "outputs/aligned_rescoring/aligned_images")
    _set_env("SLM_WM_CONVENTIONAL_GEOMETRIC_ATTACK_SOURCE_COUNT", sample_count_token)
    _set_default_env("SLM_WM_ENABLE_ATTACK_PROGRESS_BAR", "1")


def _configure_external_baseline_method_faithful(
    paper_run: Any,
    sample_count_token: str,
    target_fpr_text: str,
    baseline_id: str,
) -> None:
    _set_env("SLM_WM_PRIMARY_BASELINE_ID", baseline_id)
    _set_env(
        "SLM_WM_EXTERNAL_BASELINE_DRIVE_OUTPUT_DIR",
        _repeat_drive_dir(paper_run, "external_baseline_method_faithful"),
    )
    _set_env("SLM_WM_EXTERNAL_BASELINE_TARGET_FPR", target_fpr_text)
    _set_env("SLM_WM_EXTERNAL_BASELINE_MODEL_ID", _SD35_MODEL_SOURCE.repository_id)
    _set_env("SLM_WM_EXTERNAL_BASELINE_MODEL_REVISION", _SD35_MODEL_SOURCE.revision)
    _set_env("SLM_WM_EXTERNAL_BASELINE_NUM_INFERENCE_STEPS", str(paper_run.inference_steps))
    _set_env("SLM_WM_EXTERNAL_BASELINE_NUM_INVERSION_STEPS", str(paper_run.inference_steps))
    _set_env("SLM_WM_EXTERNAL_BASELINE_GUIDANCE_SCALE", str(paper_run.guidance_scale))
    _set_default_env("SLM_WM_EXTERNAL_BASELINE_REQUIRE_CUDA", "1")
    _set_env("SLM_WM_PRIMARY_BASELINE_MAX_SAMPLES", sample_count_token)
    _set_default_env("SLM_WM_TREE_RING_ADAPTER_MODE", "method_faithful_sd35")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_ADAPTER_MODE", "method_faithful_sd35")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_ADAPTER_MODE", "method_faithful_sd35")
    _set_default_env("SLM_WM_TREE_RING_ATTACK_FAMILIES", FORMAL_IMAGE_ATTACK_FAMILIES)
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_ATTACK_FAMILIES", FORMAL_IMAGE_ATTACK_FAMILIES)
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_ATTACK_FAMILIES", FORMAL_IMAGE_ATTACK_FAMILIES)
    _set_default_env("SLM_WM_ENABLE_WORKFLOW_PROGRESS_BAR", "1")


def _configure_official_reference_common(paper_run: Any, prefix: str, sample_count_token: str) -> None:
    _set_env(
        f"SLM_WM_{prefix}_OFFICIAL_DRIVE_OUTPUT_DIR",
        _invariant_drive_dir(
            paper_run,
            "external_baseline_official_reference",
        ),
    )
    _set_env(f"SLM_WM_{prefix}_OFFICIAL_SAMPLE_COUNT", sample_count_token)
    _set_default_env(f"SLM_WM_{prefix}_OFFICIAL_RUN_COMMAND", "1")
    _set_default_env(f"SLM_WM_{prefix}_OFFICIAL_REQUIRE_CUDA", "1")
    _set_default_env(f"SLM_WM_{prefix}_OFFICIAL_TIMEOUT_SECONDS", "86400")
    _set_env("SLM_WM_OPENCLIP_CACHE_ROOT", DEFAULT_SHARED_HUGGING_FACE_SNAPSHOT_ROOT)
    _set_default_env("SLM_WM_ENABLE_WORKFLOW_PROGRESS_BAR", "1")
    _set_default_env("WANDB_MODE", "disabled")


def _configure_official_reference_tree_ring(paper_run: Any, sample_count_token: str, target_fpr_text: str) -> None:
    _configure_official_reference_common(paper_run, "TREE_RING", sample_count_token)
    _set_default_env("SLM_WM_TREE_RING_OFFICIAL_OUTPUT_DIR", "outputs/tree_ring_official_reference")
    _set_default_env("SLM_WM_TREE_RING_OFFICIAL_SOURCE_DIR", "external_baseline/primary/tree_ring/source")
    _set_default_env("SLM_WM_TREE_RING_OFFICIAL_RUN_NAME", "tree_ring_official_reference")
    _set_default_env("SLM_WM_TREE_RING_OFFICIAL_START_INDEX", "0")
    _set_env("SLM_WM_TREE_RING_OFFICIAL_MODEL_ID", _OFFICIAL_REFERENCE_DIFFUSION_MODEL_SOURCE.repository_id)
    _set_env("SLM_WM_TREE_RING_OFFICIAL_MODEL_REVISION", _OFFICIAL_REFERENCE_DIFFUSION_MODEL_SOURCE.revision)
    _set_default_env("SLM_WM_TREE_RING_UPSTREAM_OFFICIAL_MODEL_ID", "stabilityai/stable-diffusion-2-1-base")
    _set_default_env("SLM_WM_TREE_RING_PATCH_MODEL_REPOSITORY_LAYOUT", "1")
    _set_default_env("SLM_WM_TREE_RING_PREPARE_LOCAL_MODEL_REPOSITORY", "1")
    _set_env("SLM_WM_TREE_RING_LOCAL_MODEL_REPOSITORY_DIR", _OFFICIAL_REFERENCE_DIFFUSION_SNAPSHOT_DIR)
    _set_default_env("SLM_WM_TREE_RING_PATCH_MODEL_INDEX_FOR_PINNED_TRANSFORMERS", "1")


def _configure_official_reference_gaussian_shading(paper_run: Any, sample_count_token: str, target_fpr_text: str) -> None:
    _configure_official_reference_common(paper_run, "GAUSSIAN_SHADING", sample_count_token)
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_OUTPUT_DIR", "outputs/gaussian_shading_official_reference")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_SOURCE_DIR", "external_baseline/primary/gaussian_shading/source")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_RUN_NAME", "gaussian_shading_official_reference")
    _set_env("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_MODEL_ID", _OFFICIAL_REFERENCE_DIFFUSION_MODEL_SOURCE.repository_id)
    _set_env("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_MODEL_REVISION", _OFFICIAL_REFERENCE_DIFFUSION_MODEL_SOURCE.revision)
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_UPSTREAM_OFFICIAL_MODEL_ID", "stabilityai/stable-diffusion-2-1-base")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_PATCH_MODEL_REPOSITORY_LAYOUT", "1")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_PREPARE_LOCAL_MODEL_REPOSITORY", "1")
    _set_default_env(
        "SLM_WM_GAUSSIAN_SHADING_LOCAL_MODEL_REPOSITORY_DIR",
        _OFFICIAL_REFERENCE_DIFFUSION_SNAPSHOT_DIR,
    )
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_PATCH_MODEL_INDEX_FOR_PINNED_TRANSFORMERS", "1")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_NUM_INFERENCE_STEPS", "50")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_NUM_INVERSION_STEPS", "50")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_USE_CHACHA", "1")


def _configure_official_reference_shallow_diffuse(paper_run: Any, sample_count_token: str, target_fpr_text: str) -> None:
    _configure_official_reference_common(paper_run, "SHALLOW_DIFFUSE", sample_count_token)
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_OUTPUT_DIR", "outputs/shallow_diffuse_official_reference")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_SOURCE_DIR", "external_baseline/primary/shallow_diffuse/source")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_RUN_NAME", "shallow_diffuse_official_reference")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_START_INDEX", "0")
    _set_env("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_MODEL_ID", _OFFICIAL_REFERENCE_DIFFUSION_MODEL_SOURCE.repository_id)
    _set_env("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_MODEL_REVISION", _OFFICIAL_REFERENCE_DIFFUSION_MODEL_SOURCE.revision)
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_UPSTREAM_OFFICIAL_MODEL_ID", "stabilityai/stable-diffusion-2-1-base")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_DATASET", "Gustavosta/Stable-Diffusion-Prompts")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_IMAGE_LENGTH", "512")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_GUIDANCE_SCALE", "7.5")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_NUM_INFERENCE_STEPS", "50")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_EDIT_TIME_LIST", "0.3")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_W_SEED", "42")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_W_CHANNEL", "3")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_W_PATTERN", "complex2_ring")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_W_MASK_SHAPE", "circle")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_W_RADIUS", "10")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_W_MEASUREMENT", "l1_complex2")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_W_INJECTION", "complex2")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_ATTACKER_NAMES", "none")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_PATCH_MODEL_REPOSITORY_LAYOUT", "1")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_PREPARE_LOCAL_MODEL_REPOSITORY", "1")
    _set_default_env(
        "SLM_WM_SHALLOW_DIFFUSE_LOCAL_MODEL_REPOSITORY_DIR",
        _OFFICIAL_REFERENCE_DIFFUSION_SNAPSHOT_DIR,
    )
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_PATCH_MODEL_INDEX_FOR_PINNED_TRANSFORMERS", "1")


def _configure_official_reference_t2smark(paper_run: Any, sample_count_token: str, target_fpr_text: str) -> None:
    _set_env("SLM_WM_T2SMARK_FORMAL_PROMPT_FILE", paper_run.prompt_file)
    _set_env(
        "SLM_WM_T2SMARK_FORMAL_DRIVE_OUTPUT_DIR",
        _repeat_drive_dir(paper_run, "external_baseline_official_reference"),
    )
    _set_env("SLM_WM_T2SMARK_MODEL_ID", _SD35_MODEL_SOURCE.repository_id)
    _set_env("SLM_WM_T2SMARK_MODEL_REVISION", _SD35_MODEL_SOURCE.revision)
    _set_env("SLM_WM_T2SMARK_FORMAL_PAIR_CLIP_MODEL_ID", _CLIP_MODEL_SOURCE.repository_id)
    _set_env("SLM_WM_T2SMARK_FORMAL_PAIR_CLIP_MODEL_REVISION", _CLIP_MODEL_SOURCE.revision)
    _set_env("SLM_WM_T2SMARK_FORMAL_RUN_NAME", f"t2smark_sd35_medium_{paper_run.run_name}")
    _set_env("SLM_WM_T2SMARK_FORMAL_PROMPT_LIMIT", sample_count_token)
    _set_env("SLM_WM_T2SMARK_FORMAL_TARGET_FPR", target_fpr_text)
    _set_env("SLM_WM_T2SMARK_FORMAL_NUM_INFERENCE_STEPS", str(paper_run.inference_steps))
    _set_env("SLM_WM_T2SMARK_FORMAL_NUM_INVERSION_STEPS", str(paper_run.inference_steps))
    _set_env("SLM_WM_T2SMARK_FORMAL_GUIDANCE_SCALE", str(paper_run.guidance_scale))
    _set_default_env("SLM_WM_T2SMARK_FORMAL_REQUIRE_CUDA", "1")
    _set_default_env("SLM_WM_ENABLE_WORKFLOW_PROGRESS_BAR", "1")


WORKFLOW_CONFIGURERS = {
    "semantic_watermark_image_only": _configure_semantic_watermark_image_only,
    "attention_geometry": _configure_attention_geometry,
    "attention_latent_injection": _configure_attention_latent_injection,
    "aligned_rescoring": _configure_aligned_rescoring,
    "threshold_calibration": _configure_threshold_calibration,
    "real_attack_evaluation": _configure_real_attack_evaluation,
    "conventional_geometric_attack_evaluation": _configure_conventional_geometric_attack_evaluation,
    "official_reference_tree_ring": _configure_official_reference_tree_ring,
    "official_reference_gaussian_shading": _configure_official_reference_gaussian_shading,
    "official_reference_shallow_diffuse": _configure_official_reference_shallow_diffuse,
    "official_reference_t2smark": _configure_official_reference_t2smark,
}


def configure_formal_workflow_environment(
    workflow_name: str,
    *,
    baseline_id: str = "",
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    """配置服务器与 Colab 共同使用的正式论文运行环境.

    该函数不运行真实模型, 不生成正式结果。GPU 服务器可以直接调用该函数,
    外层 Notebook 只需在调用前记录会话起点并在调用后处理 Drive 展示。
    """

    formal_execution_lock = require_published_formal_execution_lock(
        repository_root
    )
    formal_execution_commit = formal_execution_lock["formal_execution_commit"]
    formal_execution_lock_digest = formal_execution_lock[
        "formal_execution_lock_digest"
    ]

    paper_run, sample_count_token, target_fpr_text = (
        _configure_common_paper_run_environment(repository_root)
    )
    invariant_workflow = workflow_name in CROSS_REPEAT_INVARIANT_WORKFLOW_NAMES
    if invariant_workflow:
        # 官方原环境忠实度只运行一次, 子进程不得继承任一活动 repeat 身份。
        os.environ.pop("SLM_WM_RANDOMIZATION_REPEAT_ID", None)
    if workflow_name == "external_baseline_method_faithful":
        if not baseline_id:
            raise ValueError("external_baseline_method_faithful 需要 baseline_id")
        _configure_external_baseline_method_faithful(paper_run, sample_count_token, target_fpr_text, baseline_id)
    else:
        try:
            configurer = WORKFLOW_CONFIGURERS[workflow_name]
        except KeyError as exc:
            raise ValueError(f"未知正式 workflow: {workflow_name}") from exc
        configurer(paper_run, sample_count_token, target_fpr_text)

    tracked_keys = tuple(sorted(key for key in os.environ if key.startswith("SLM_WM_")))
    return FormalWorkflowEnvironment(
        workflow_name=workflow_name,
        paper_run_name=paper_run.run_name,
        protocol_profile=os.environ["SLM_WM_PROTOCOL_PROFILE"],
        prompt_set=os.environ["SLM_WM_PROMPT_SET"],
        prompt_file=os.environ["SLM_WM_PROMPT_FILE"],
        drive_result_root=os.environ["SLM_WM_DRIVE_RESULT_ROOT"],
        resume_checkpoint_dir=os.environ.get(
            "SLM_WM_RESUME_CHECKPOINT_DIR",
            "",
        ),
        target_fpr=os.environ["SLM_WM_PAPER_RUN_TARGET_FPR"],
        sample_count_token=os.environ["SLM_WM_PAPER_RUN_SAMPLE_COUNT"],
        expected_sample_count=int(os.environ["SLM_WM_PAPER_RUN_EXPECTED_SAMPLE_COUNT"]),
        minimum_clean_negative_count=os.environ["SLM_WM_PAPER_RUN_MINIMUM_CLEAN_NEGATIVE_COUNT"],
        dataset_quality_minimum_count=os.environ["SLM_WM_PAPER_RUN_DATASET_QUALITY_MINIMUM_COUNT"],
        randomization_repeat_id=(
            "" if invariant_workflow else paper_run.randomization_repeat_id
        ),
        generation_seed_index=paper_run.generation_seed_index,
        generation_seed_offset=paper_run.generation_seed_offset,
        watermark_key_index=paper_run.watermark_key_index,
        formal_randomization_repeat_count=(
            paper_run.formal_randomization_repeat_count
        ),
        formal_randomization_protocol_digest=(
            paper_run.formal_randomization_protocol_digest
        ),
        selected_baseline_id=baseline_id,
        formal_execution_commit=formal_execution_commit,
        formal_execution_lock_digest=formal_execution_lock_digest,
        configured_environment_keys=tracked_keys,
    ).to_dict()
