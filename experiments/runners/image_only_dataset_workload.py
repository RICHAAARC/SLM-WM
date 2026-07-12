"""执行当前论文规模的真实 SLM-WM 图像盲检数据集工作负载."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

from experiments.artifacts.dataset_level_quality_outputs import (
    package_dataset_level_quality_outputs,
    write_dataset_level_quality_outputs,
)
from experiments.protocol.method_runtime_config import (
    load_formal_method_runtime_config,
    require_formal_method_environment_consistency,
)
from experiments.protocol.paper_run_config import (
    build_paper_run_config,
    shared_method_settings,
)
from experiments.runners.image_only_dataset_runtime import (
    package_image_only_dataset_runtime,
    run_image_only_dataset_runtime,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
)


ROOT = Path(__file__).resolve().parents[2]


def build_method_config(
    root: str | Path = ".",
) -> SemanticWatermarkRuntimeConfig:
    """从唯一方法 YAML 与运行环境控制项构造真实方法配置."""

    method = load_formal_method_runtime_config(root)
    require_formal_method_environment_consistency(method)
    paper_run = build_paper_run_config(root)
    if shared_method_settings(paper_run) != method.paper_method_settings():
        raise RuntimeError("论文运行配置没有完整继承 configs/model_sd35.yaml")
    return SemanticWatermarkRuntimeConfig(
        model_family=method.model_family,
        model_id=method.model_id,
        model_revision=method.model_revision,
        vision_model_id=method.vision_model_id,
        vision_model_revision=method.vision_model_revision,
        device_name=os.environ.get("SLM_WM_DEVICE", "cuda"),
        torch_dtype=os.environ.get("SLM_WM_TORCH_DTYPE", "float16"),
        vision_torch_dtype=os.environ.get("SLM_WM_VISION_TORCH_DTYPE", "float32"),
        prompt=method.prompt,
        negative_prompt=method.negative_prompt,
        key_material=os.environ.get("SLM_WM_KEY_MATERIAL", "slm_wm_paper_key"),
        seed=method.seed,
        width=method.width,
        height=method.height,
        inference_steps=method.inference_steps,
        guidance_scale=method.guidance_scale,
        injection_step_indices=method.injection_step_indices,
        candidate_count=method.jacobian_candidate_count,
        null_rank=method.null_space_rank,
        lf_relative_strength=method.lf_relative_strength,
        tail_relative_strength=method.tail_relative_strength,
        attention_relative_strength=method.attention_relative_strength,
        attention_stable_token_fraction=(
            method.attention_stable_token_fraction
        ),
        attention_unstable_pair_weight=method.attention_unstable_pair_weight,
        minimum_final_image_attention_score_gain=(
            method.minimum_final_image_attention_score_gain
        ),
        tail_fraction=method.tail_fraction,
        minimum_projection_energy_retention=(
            method.minimum_projection_energy_retention
        ),
        maximum_relative_response_residual=method.maximum_relative_response_residual,
        maximum_quantized_write_relative_jacobian_response=(
            method.maximum_quantized_write_relative_jacobian_response
        ),
        keyed_prg_version=method.keyed_prg_version,
        null_space_cg_max_iterations=method.null_space_cg_max_iterations,
        null_space_cg_relative_tolerance=(
            method.null_space_cg_relative_tolerance
        ),
        minimum_semantic_preservation_cosine=(
            method.minimum_semantic_preservation_cosine
        ),
        maximum_handcrafted_structure_feature_relative_drift=(
            method.maximum_handcrafted_structure_feature_relative_drift
        ),
        max_attention_tokens=method.max_attention_tokens,
        attention_module_names=method.attention_module_names,
        attention_coordinate_convention=(
            method.attention_coordinate_convention
        ),
        attention_grid_align_corners=(
            method.attention_grid_align_corners
        ),
        diffusion_attacks_enabled=method.diffusion_attacks_enabled,
    )


def run_image_only_dataset_workload(
    root: str | Path = ROOT,
) -> dict[str, object]:
    """运行生成、盲检、攻击和正式 FID/KID, 并返回可序列化结果."""

    root_path = Path(root).resolve()
    paper_run = build_paper_run_config(root_path)
    summary = run_image_only_dataset_runtime(
        build_method_config(root_path),
        root=root_path,
        paper_run=paper_run,
        max_new_prompts_per_session=int(
            os.environ.get("SLM_WM_MAX_NEW_PROMPTS_PER_SESSION", "0")
        ),
    )
    if summary.get("protocol_decision") == "resume_required":
        return {"summary": summary}

    # 主方法模型已经离开调用作用域, 先回收显存再加载正式 Inception 模型.
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    packaging_deferred = (
        os.environ.get("SLM_WM_DEFER_SCIENTIFIC_PACKAGING", "0") == "1"
    )
    archive_path = None
    quality_manifest = write_dataset_level_quality_outputs(
        paper_run_name=paper_run.run_name,
        target_fpr=paper_run.target_fpr,
        root=root_path,
        quality_image_registry_path=(
            "outputs/image_only_dataset_runtime/"
            f"{paper_run.run_name}/watermark_quality_image_registry.jsonl"
        ),
        formal_min_sample_count=paper_run.dataset_level_quality_minimum_count,
        auto_extract_formal_features=True,
        inception_device_name=os.environ.get("SLM_WM_INCEPTION_DEVICE"),
        inception_batch_size=int(
            os.environ.get("SLM_WM_INCEPTION_BATCH_SIZE", "32")
        ),
    )
    quality_archive_path = (
        None
        if packaging_deferred
        else package_dataset_level_quality_outputs(
            paper_run.run_name,
            root=root_path,
        )
    )
    if not packaging_deferred:
        archive_path = package_image_only_dataset_runtime(
            paper_run.run_name,
            root=root_path,
        )
    return {
        "summary": summary,
        "archive_path": None if archive_path is None else str(archive_path),
        "packaging_deferred": packaging_deferred,
        "quality_manifest": quality_manifest,
        "quality_archive_path": (
            None if quality_archive_path is None else str(quality_archive_path)
        ),
    }


def main() -> None:
    """命令行入口."""

    print(
        json.dumps(
            run_image_only_dataset_workload(ROOT),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
