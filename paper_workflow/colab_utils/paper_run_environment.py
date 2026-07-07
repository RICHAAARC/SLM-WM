"""集中配置 Colab 论文运行入口的环境变量。

该模块的作用是把 Notebook 中重复出现的运行层级、prompt、目标 FPR、
Google Drive 输出目录和 workflow 专用默认值收敛到一个 helper 中。
Notebook 只保留挂载 Drive、选择运行层级、拉取仓库和调用 helper 的入口逻辑。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any

from experiments.protocol.paper_run_config import (
    PILOT_PAPER_RUN_NAME,
    RUN_DEFAULTS,
    build_paper_run_config,
    normalize_paper_run_name,
)
from paper_workflow.notebook_utils.notebook_runtime import mark_notebook_runtime_start

FORMAL_IMAGE_ATTACK_FAMILIES = (
    "jpeg_compression,gaussian_noise,gaussian_blur,rotation,resize,crop,crop_resize,"
    "composite_geometric_attacks,photometric_distortion_attack,img2img_regeneration,"
    "ddim_inversion_regeneration,sdedit_regeneration,diffusion_purification,"
    "global_editing_attack,local_editing_attack,visual_paraphrase_attack,adversarial_removal_attack"
)


@dataclass(frozen=True)
class PaperRunEnvironment:
    """记录 Notebook 已发布到环境变量中的论文运行配置。"""

    workflow_name: str
    paper_run_name: str
    protocol_profile: str
    prompt_set: str
    prompt_file: str
    drive_result_root: str
    target_fpr: str
    sample_count_token: str
    expected_sample_count: int
    minimum_clean_negative_count: str
    dataset_quality_minimum_count: str
    selected_baseline_id: str
    configured_environment_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """转换为 Notebook 可直接打印的配置摘要。"""

        return asdict(self)


def _set_env(name: str, value: str | int | float) -> None:
    """无条件写入环境变量, 用于消除 Colab 内核中残留的旧配置。"""

    os.environ[name] = str(value)


def _set_default_env(name: str, value: str | int | float) -> None:
    """仅在用户未显式覆盖时写入默认环境变量。"""

    os.environ.setdefault(name, str(value))


def _target_fpr_text(value: float) -> str:
    """把目标 FPR 转为稳定短文本, 便于生成 profile 名称。"""

    return f"{float(value):g}"


def _protocol_profile(paper_run_name: str, target_fpr_text: str) -> str:
    """根据运行层级和目标 FPR 构造协议 profile 名称。"""

    return f"{paper_run_name}_fixed_fpr_{target_fpr_text.replace('.', '_')}"


def _resolve_paper_run_name() -> str:
    """解析 Notebook 传入的论文运行层级, 默认保持 pilot_paper。"""

    return normalize_paper_run_name(os.environ.get("SLM_WM_PAPER_RUN_NAME", PILOT_PAPER_RUN_NAME))


def _configure_common_paper_run_environment() -> tuple[Any, str, str]:
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

    paper_run = build_paper_run_config(".")
    target_fpr_text = _target_fpr_text(paper_run.target_fpr)
    profile = _protocol_profile(paper_run.run_name, target_fpr_text)
    _set_env("SLM_WM_PAPER_RUN_TARGET_FPR", target_fpr_text)
    _set_env("SLM_WM_PROTOCOL_PROFILE", profile)
    _set_env("SLM_WM_PAPER_RUN_EXPECTED_SAMPLE_COUNT", paper_run.sample_count)
    _set_env("SLM_WM_PAPER_RUN_MINIMUM_CLEAN_NEGATIVE_COUNT", paper_run.minimum_clean_negative_count)
    _set_env("SLM_WM_PAPER_RUN_DATASET_QUALITY_MINIMUM_COUNT", paper_run.dataset_level_quality_minimum_count)
    return paper_run, sample_count_token, target_fpr_text


def _configure_attention_geometry(paper_run: Any, sample_count_token: str, target_fpr_text: str) -> None:
    _set_env("SLM_WM_DRIVE_OUTPUT_DIR", paper_run.drive_dir("attention_geometry"))
    _set_default_env("SLM_WM_ATTENTION_CAPTURE_COUNT", "16")
    _set_default_env("SLM_WM_ATTENTION_TOKEN_COUNT", "32")


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
    _set_default_env("SLM_WM_CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
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
    _set_default_env("SLM_WM_DDIM_ATTACK_MODEL_ID", "runwayml/stable-diffusion-v1-5")
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


def _configure_dataset_level_quality(paper_run: Any, sample_count_token: str, target_fpr_text: str) -> None:
    _set_env("SLM_WM_DATASET_QUALITY_DRIVE_DIR", paper_run.drive_dir("dataset_level_quality"))
    _set_env("SLM_WM_REAL_ATTACK_EVALUATION_DRIVE_DIR", paper_run.drive_dir("real_attack_evaluation"))
    _set_env("SLM_WM_ALIGNED_RESCORING_DRIVE_DIR", paper_run.drive_dir("aligned_rescoring"))
    _set_env("SLM_WM_FORMAL_MIN_SAMPLE_COUNT", paper_run.dataset_level_quality_minimum_count)
    _set_default_env("SLM_WM_REAL_ATTACK_EVALUATION_PACKAGE_PATTERN", "real_attack_evaluation_package_*.zip")
    _set_default_env("SLM_WM_ALIGNED_RESCORING_PACKAGE_PATTERN", "aligned_rescoring_package_*.zip")


def _configure_external_baseline_method_faithful(
    paper_run: Any,
    sample_count_token: str,
    target_fpr_text: str,
    baseline_id: str,
) -> None:
    _set_env("SLM_WM_PRIMARY_BASELINE_METHODS", baseline_id)
    _set_env("SLM_WM_EXTERNAL_BASELINE_DRIVE_OUTPUT_DIR", paper_run.drive_dir("external_baseline_method_faithful"))
    _set_env("SLM_WM_EXTERNAL_BASELINE_PRIOR_DRIVE_DIR", paper_run.drive_dir("external_baseline_method_faithful"))
    _set_default_env("SLM_WM_T2SMARK_MODEL_ID", "stabilityai/stable-diffusion-3.5-medium")
    _set_default_env("SLM_WM_T2SMARK_RUN_NAME", "t2smark_sd35_medium_method_faithful")
    _set_env("SLM_WM_T2SMARK_ROBUST_TEST_NUM", sample_count_token)
    _set_default_env("SLM_WM_T2SMARK_CLIP_TEST_NUM", "0")
    _set_default_env("SLM_WM_T2SMARK_NUM_INFERENCE_STEPS", "8")
    _set_default_env("SLM_WM_T2SMARK_NUM_INVERSION_STEPS", "3")
    _set_default_env("SLM_WM_T2SMARK_GUIDANCE_SCALE", "4.0")
    _set_default_env("SLM_WM_EXTERNAL_BASELINE_REUSE_EXISTING", "1")
    _set_default_env("SLM_WM_EXTERNAL_BASELINE_REUSE_DRIVE", "1")
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
    _set_env(f"SLM_WM_{prefix}_OFFICIAL_DRIVE_OUTPUT_DIR", paper_run.drive_dir("external_baseline_official_reference"))
    _set_env(f"SLM_WM_{prefix}_OFFICIAL_SAMPLE_COUNT", sample_count_token)
    _set_default_env(f"SLM_WM_{prefix}_OFFICIAL_RUN_COMMAND", "1")
    _set_default_env(f"SLM_WM_{prefix}_OFFICIAL_REQUIRE_CUDA", "1")
    _set_default_env(f"SLM_WM_{prefix}_OFFICIAL_SUMMARY_IMPORT_PATH", "")
    _set_default_env(f"SLM_WM_{prefix}_OFFICIAL_LOG_IMPORT_PATH", "")
    _set_default_env(f"SLM_WM_{prefix}_OFFICIAL_TIMEOUT_SECONDS", "86400")
    _set_default_env("SLM_WM_ENABLE_WORKFLOW_PROGRESS_BAR", "1")
    _set_default_env("WANDB_MODE", "disabled")


def _configure_official_reference_tree_ring(paper_run: Any, sample_count_token: str, target_fpr_text: str) -> None:
    _configure_official_reference_common(paper_run, "TREE_RING", sample_count_token)
    _set_default_env("SLM_WM_TREE_RING_OFFICIAL_OUTPUT_DIR", "outputs/tree_ring_official_reference")
    _set_default_env("SLM_WM_TREE_RING_OFFICIAL_SOURCE_DIR", "external_baseline/primary/tree_ring/source")
    _set_default_env("SLM_WM_TREE_RING_OFFICIAL_RUN_NAME", "tree_ring_official_legacy_reference")
    _set_default_env("SLM_WM_TREE_RING_OFFICIAL_START_INDEX", "0")
    _set_default_env("SLM_WM_TREE_RING_OFFICIAL_MODEL_ID", "Manojb/stable-diffusion-2-1-base")
    _set_default_env("SLM_WM_TREE_RING_UPSTREAM_OFFICIAL_MODEL_ID", "stabilityai/stable-diffusion-2-1-base")
    _set_default_env("SLM_WM_TREE_RING_PATCH_MODEL_REPOSITORY_LAYOUT", "1")
    _set_default_env("SLM_WM_TREE_RING_PREPARE_LOCAL_MODEL_REPOSITORY", "1")
    _set_default_env("SLM_WM_TREE_RING_LOCAL_MODEL_REPOSITORY_DIR", "/content/tree_ring_model_repository/stable_diffusion_2_1_base")
    _set_default_env("SLM_WM_TREE_RING_PATCH_MODEL_INDEX_FOR_LEGACY_TRANSFORMERS", "1")
    _set_default_env("SLM_WM_TREE_RING_OFFICIAL_PYTHON_EXECUTABLE", "")
    _set_default_env("SLM_WM_TREE_RING_PREPARE_LEGACY_ENV", "1")
    _set_default_env("SLM_WM_TREE_RING_LEGACY_ENV_PREFIX", "/content/tree_ring_legacy_env")
    _set_default_env("SLM_WM_TREE_RING_MICROMAMBA_PATH", "/content/bin/micromamba")
    _set_default_env("SLM_WM_TREE_RING_LEGACY_PYTHON_VERSION", "3.9")
    _set_default_env("SLM_WM_TREE_RING_LEGACY_TORCH_SPECS", "torch==1.13.0+cu117 torchvision==0.14.0+cu117")
    _set_default_env("SLM_WM_TREE_RING_LEGACY_PYTORCH_INDEX_URL", "https://download.pytorch.org/whl/cu117")
    _set_default_env(
        "SLM_WM_TREE_RING_LEGACY_PACKAGE_SPECS",
        "transformers==4.23.1 diffusers==0.11.1 huggingface_hub==0.10.1 datasets==2.6.1 "
        "pyarrow<13 fsspec==2022.10.0 numpy<2 scikit-learn scipy tqdm wandb open_clip_torch==2.7.0 ftfy regex",
    )


def _configure_official_reference_gaussian_shading(paper_run: Any, sample_count_token: str, target_fpr_text: str) -> None:
    _configure_official_reference_common(paper_run, "GAUSSIAN_SHADING", sample_count_token)
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_OUTPUT_DIR", "outputs/gaussian_shading_official_reference")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_SOURCE_DIR", "external_baseline/primary/gaussian_shading/source")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_RUN_NAME", "gaussian_shading_official_legacy_reference")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_MODEL_ID", "Manojb/stable-diffusion-2-1-base")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_UPSTREAM_OFFICIAL_MODEL_ID", "stabilityai/stable-diffusion-2-1-base")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_PATCH_MODEL_REPOSITORY_LAYOUT", "1")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_PREPARE_LOCAL_MODEL_REPOSITORY", "1")
    _set_default_env(
        "SLM_WM_GAUSSIAN_SHADING_LOCAL_MODEL_REPOSITORY_DIR",
        "/content/gaussian_shading_model_repository/stable_diffusion_2_1_base",
    )
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_PATCH_MODEL_INDEX_FOR_LEGACY_TRANSFORMERS", "1")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_PYTHON_EXECUTABLE", "")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_PREPARE_LEGACY_ENV", "1")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_LEGACY_ENV_PREFIX", "/content/gaussian_shading_legacy_env")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_MICROMAMBA_PATH", "/content/bin/micromamba")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_LEGACY_PYTHON_VERSION", "3.8")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_LEGACY_TORCH_SPECS", "torch==1.13.0+cu117 torchvision==0.14.0+cu117")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_STRICT_OFFICIAL_ENV", "1")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_ALLOW_COMPATIBLE_ENV_FALLBACK", "1")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_LEGACY_PYTORCH_INDEX_URL", "https://download.pytorch.org/whl/cu117")
    _set_default_env(
        "SLM_WM_GAUSSIAN_SHADING_LEGACY_PACKAGE_SPECS",
        "transformers==4.23.1 diffusers==0.11.1 huggingface_hub==0.10.1 datasets==2.6.1 "
        "pyarrow<13 fsspec==2022.10.0 numpy==1.24.4 scipy==1.10.1 Pillow==9.5.0 tqdm==4.66.2 "
        "pycryptodome==3.20.0 open_clip_torch==2.7.0 ftfy==6.2.0 regex==2023.12.25 Requests==2.31.0 "
        "omegaconf==2.3.0 einops==0.4.1 kornia==0.6.4 matplotlib==3.7.5 timm==0.5.4",
    )
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_NUM_INFERENCE_STEPS", "50")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_NUM_INVERSION_STEPS", "50")
    _set_default_env("SLM_WM_GAUSSIAN_SHADING_USE_CHACHA", "1")


def _configure_official_reference_shallow_diffuse(paper_run: Any, sample_count_token: str, target_fpr_text: str) -> None:
    _configure_official_reference_common(paper_run, "SHALLOW_DIFFUSE", sample_count_token)
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_OUTPUT_DIR", "outputs/shallow_diffuse_official_reference")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_SOURCE_DIR", "external_baseline/primary/shallow_diffuse/source")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_RUN_NAME", "shallow_diffuse_official_legacy_reference")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_START_INDEX", "0")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_MODEL_ID", "Manojb/stable-diffusion-2-1-base")
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
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_REFERENCE_MODEL", "ViT-g-14")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_REFERENCE_MODEL_PRETRAIN", "laion2b_s12b_b42k")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_ATTACKER_NAMES", "none")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_PATCH_MODEL_REPOSITORY_LAYOUT", "1")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_PREPARE_LOCAL_MODEL_REPOSITORY", "1")
    _set_default_env(
        "SLM_WM_SHALLOW_DIFFUSE_LOCAL_MODEL_REPOSITORY_DIR",
        "/content/shallow_diffuse_model_repository/stable_diffusion_2_1_base",
    )
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_PATCH_MODEL_INDEX_FOR_LEGACY_TRANSFORMERS", "1")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_PYTHON_EXECUTABLE", "")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_PREPARE_LEGACY_ENV", "1")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_LEGACY_ENV_PREFIX", "/content/shallow_diffuse_legacy_env")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_MICROMAMBA_PATH", "/content/bin/micromamba")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_LEGACY_PYTHON_VERSION", "3.9")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_LEGACY_TORCH_SPECS", "torch==1.13.0+cu117 torchvision==0.14.0+cu117")
    _set_default_env("SLM_WM_SHALLOW_DIFFUSE_LEGACY_PYTORCH_INDEX_URL", "https://download.pytorch.org/whl/cu117")
    _set_default_env(
        "SLM_WM_SHALLOW_DIFFUSE_LEGACY_PACKAGE_SPECS",
        "transformers==4.23.1 diffusers==0.11.1 huggingface_hub==0.10.1 datasets==2.6.1 "
        "pyarrow<13 fsspec==2022.10.0 numpy==1.24.4 scipy==1.10.1 Pillow==9.5.0 tqdm==4.66.2 "
        "scikit-learn==1.3.2 wandb==0.16.6 open_clip_torch==2.7.0 ftfy==6.2.0 regex==2023.12.25 "
        "Requests==2.31.0 omegaconf==2.3.0 einops==0.4.1 matplotlib==3.7.5 timm==0.5.4 "
        "opencv-python-headless==4.9.0.80",
    )


def _configure_official_reference_t2smark(paper_run: Any, sample_count_token: str, target_fpr_text: str) -> None:
    _set_env("SLM_WM_T2SMARK_FULL_MAIN_PROMPT_FILE", paper_run.prompt_file)
    _set_env("SLM_WM_T2SMARK_FULL_MAIN_DRIVE_OUTPUT_DIR", paper_run.drive_dir("external_baseline_official_reference"))
    _set_default_env("SLM_WM_T2SMARK_MODEL_ID", "stabilityai/stable-diffusion-3.5-medium")
    _set_env("SLM_WM_T2SMARK_FULL_MAIN_RUN_NAME", f"t2smark_sd35_medium_{paper_run.run_name}")
    _set_env("SLM_WM_T2SMARK_FULL_MAIN_PROMPT_LIMIT", sample_count_token)
    _set_env("SLM_WM_T2SMARK_FULL_MAIN_TARGET_FPR", target_fpr_text)
    _set_env("SLM_WM_T2SMARK_FULL_MAIN_FIXED_FPR_READY", "1")
    _set_env("SLM_WM_T2SMARK_FULL_MAIN_ATTACK_MATRIX_READY", "1")
    _set_default_env("SLM_WM_T2SMARK_FULL_MAIN_REQUIRE_CUDA", "1")
    _set_default_env("SLM_WM_ENABLE_WORKFLOW_PROGRESS_BAR", "1")


def _configure_result_closure(paper_run: Any, sample_count_token: str, target_fpr_text: str) -> None:
    _set_env("SLM_WM_PAPER_RUN_PACKAGE_SEARCH_ROOT", paper_run.drive_result_root)
    _set_env("SLM_WM_PAPER_RUN_COMPLETE_DRIVE_OUTPUT_DIR", paper_run.drive_dir("complete_result_package"))


WORKFLOW_CONFIGURERS = {
    "attention_geometry": _configure_attention_geometry,
    "attention_latent_injection": _configure_attention_latent_injection,
    "aligned_rescoring": _configure_aligned_rescoring,
    "threshold_calibration": _configure_threshold_calibration,
    "real_attack_evaluation": _configure_real_attack_evaluation,
    "conventional_geometric_attack_evaluation": _configure_conventional_geometric_attack_evaluation,
    "dataset_level_quality": _configure_dataset_level_quality,
    "official_reference_tree_ring": _configure_official_reference_tree_ring,
    "official_reference_gaussian_shading": _configure_official_reference_gaussian_shading,
    "official_reference_shallow_diffuse": _configure_official_reference_shallow_diffuse,
    "official_reference_t2smark": _configure_official_reference_t2smark,
    "paper_result_closure": _configure_result_closure,
}


def configure_paper_run_environment(
    workflow_name: str,
    *,
    baseline_id: str = "",
) -> dict[str, Any]:
    """配置某个 Notebook 入口需要的论文运行环境。

    该函数属于 Notebook 入口治理层。它不运行真实模型, 不生成正式结果,
    只把 Notebook 过去重复维护的环境变量写入收敛到一个可测试位置。
    """

    mark_notebook_runtime_start(
        workflow_name=workflow_name,
        baseline_id=baseline_id,
        source="configure_paper_run_environment",
    )
    paper_run, sample_count_token, target_fpr_text = _configure_common_paper_run_environment()
    if workflow_name == "external_baseline_method_faithful":
        if not baseline_id:
            raise ValueError("external_baseline_method_faithful 需要 baseline_id")
        _configure_external_baseline_method_faithful(paper_run, sample_count_token, target_fpr_text, baseline_id)
    else:
        try:
            configurer = WORKFLOW_CONFIGURERS[workflow_name]
        except KeyError as exc:
            raise ValueError(f"未知 Notebook workflow: {workflow_name}") from exc
        configurer(paper_run, sample_count_token, target_fpr_text)

    tracked_keys = tuple(sorted(key for key in os.environ if key.startswith("SLM_WM_")))
    return PaperRunEnvironment(
        workflow_name=workflow_name,
        paper_run_name=paper_run.run_name,
        protocol_profile=os.environ["SLM_WM_PROTOCOL_PROFILE"],
        prompt_set=os.environ["SLM_WM_PROMPT_SET"],
        prompt_file=os.environ["SLM_WM_PROMPT_FILE"],
        drive_result_root=os.environ["SLM_WM_DRIVE_RESULT_ROOT"],
        target_fpr=os.environ["SLM_WM_PAPER_RUN_TARGET_FPR"],
        sample_count_token=os.environ["SLM_WM_PAPER_RUN_SAMPLE_COUNT"],
        expected_sample_count=int(os.environ["SLM_WM_PAPER_RUN_EXPECTED_SAMPLE_COUNT"]),
        minimum_clean_negative_count=os.environ["SLM_WM_PAPER_RUN_MINIMUM_CLEAN_NEGATIVE_COUNT"],
        dataset_quality_minimum_count=os.environ["SLM_WM_PAPER_RUN_DATASET_QUALITY_MINIMUM_COUNT"],
        selected_baseline_id=baseline_id,
        configured_environment_keys=tracked_keys,
    ).to_dict()

