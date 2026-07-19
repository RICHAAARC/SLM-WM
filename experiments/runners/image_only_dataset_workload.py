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
from experiments.protocol.formal_randomization import (
    formal_watermark_key_material,
    formal_watermark_key_seed_random,
    resolve_formal_randomization_repeat,
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
    repeat = resolve_formal_randomization_repeat(
        paper_run.randomization_repeat_id
    )
    root_key_material = os.environ.get(
        "SLM_WM_KEY_MATERIAL",
        "slm_wm_paper_key",
    )
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
        key_material=formal_watermark_key_material(
            root_key_material,
            repeat,
        ),
        seed=method.seed + repeat.generation_seed_offset,
        randomization_repeat_id=repeat.randomization_repeat_id,
        generation_seed_index=repeat.generation_seed_index,
        generation_seed_offset=repeat.generation_seed_offset,
        watermark_key_index=repeat.watermark_key_index,
        watermark_key_seed_random=formal_watermark_key_seed_random(
            root_key_material,
            repeat,
        ),
        formal_randomization_protocol_digest=(
            paper_run.formal_randomization_protocol_digest
        ),
        width=method.width,
        height=method.height,
        inference_steps=method.inference_steps,
        guidance_scale=method.guidance_scale,
        injection_step_indices=method.injection_step_indices,
        lf_relative_strength=method.lf_relative_strength,
        tail_relative_strength=method.tail_relative_strength,
        attention_relative_strength=method.attention_relative_strength,
        lf_kernel_size=method.lf_kernel_size,
        lf_stride=method.lf_stride,
        lf_padding=method.lf_padding,
        lf_boundary_mode=method.lf_boundary_mode,
        lf_ceil_mode=method.lf_ceil_mode,
        lf_count_include_pad=method.lf_count_include_pad,
        lf_divisor_override=method.lf_divisor_override,
        lf_detection_score_weight=method.lf_detection_score_weight,
        tail_robust_detection_score_weight=(
            method.tail_robust_detection_score_weight
        ),
        attention_stable_token_fraction=(
            method.attention_stable_token_fraction
        ),
        attention_unstable_pair_weight=method.attention_unstable_pair_weight,
        attention_relation_component_weights=(
            method.attention_relation_component_weights
        ),
        attention_anchor_count=method.attention_anchor_count,
        attention_residual_threshold=(
            method.attention_residual_threshold
        ),
        attention_minimum_inlier_ratio=(
            method.attention_minimum_inlier_ratio
        ),
        minimum_final_image_attention_score_gain=(
            method.minimum_final_image_attention_score_gain
        ),
        tail_fraction=method.tail_fraction,
        keyed_prg_version=method.keyed_prg_version,
        minimum_semantic_preservation_cosine=(
            method.minimum_semantic_preservation_cosine
        ),
        maximum_handcrafted_structure_feature_relative_drift=(
            method.maximum_handcrafted_structure_feature_relative_drift
        ),
        attention_operator_schedule_index=(
            method.attention_operator_schedule_index
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
        attack_quality_registry_path=(
            "outputs/image_only_dataset_runtime/"
            f"{paper_run.run_name}/attack_conditioned_quality_image_records.jsonl"
        ),
        formal_min_sample_count=paper_run.dataset_level_quality_minimum_count,
        auto_extract_formal_features=True,
        inception_device_name=os.environ.get("SLM_WM_INCEPTION_DEVICE"),
        inception_batch_size=int(
            os.environ.get("SLM_WM_INCEPTION_BATCH_SIZE", "32")
        ),
        clip_device_name=os.environ.get("SLM_WM_CLIP_DEVICE"),
        clip_batch_size=int(
            os.environ.get("SLM_WM_CLIP_BATCH_SIZE", "32")
        ),
        independent_semantic_device_name=os.environ.get(
            "SLM_WM_INDEPENDENT_SEMANTIC_DEVICE"
        ),
        independent_semantic_batch_size=int(
            os.environ.get("SLM_WM_INDEPENDENT_SEMANTIC_BATCH_SIZE", "32")
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
