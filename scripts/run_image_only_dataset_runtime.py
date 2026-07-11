"""运行当前论文规模的真实 SLM-WM 图像盲检数据集实验。"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol.method_runtime_config import (
    load_formal_method_runtime_config,
    require_formal_method_environment_consistency,
)
from experiments.protocol.paper_run_config import build_paper_run_config, shared_method_settings
from experiments.artifacts.dataset_level_quality_outputs import (
    package_dataset_level_quality_outputs,
    write_dataset_level_quality_outputs,
)
from experiments.runners.image_only_dataset_runtime import (
    package_image_only_dataset_runtime,
    run_image_only_dataset_runtime,
)
from experiments.runners.semantic_watermark_runtime import SemanticWatermarkRuntimeConfig


def build_method_config(root: str | Path = ".") -> SemanticWatermarkRuntimeConfig:
    """从唯一方法 YAML 与运行环境控制项构造真实方法配置。"""

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
        tail_fraction=method.tail_fraction,
        minimum_projection_energy_retention=method.minimum_projection_energy_retention,
        maximum_relative_response_residual=method.maximum_relative_response_residual,
        max_attention_tokens=method.max_attention_tokens,
        attention_module_count=method.attention_module_count,
        diffusion_attacks_enabled=method.diffusion_attacks_enabled,
    )


def main() -> None:
    """命令行入口。"""

    paper_run = build_paper_run_config(ROOT)
    summary = run_image_only_dataset_runtime(
        build_method_config(ROOT),
        root=ROOT,
        paper_run=paper_run,
        max_new_prompts_per_session=int(os.environ.get("SLM_WM_MAX_NEW_PROMPTS_PER_SESSION", "0")),
    )
    if summary.get("protocol_decision") == "resume_required":
        print(json.dumps({"summary": summary}, ensure_ascii=False, sort_keys=True))
        return
    # 主方法上下文已经离开作用域; 先释放模型引用和 CUDA 缓存, 再加载正式 Inception。
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    archive_path = package_image_only_dataset_runtime(paper_run.run_name, root=ROOT)
    quality_manifest = None
    quality_archive_path = None
    if os.environ.get("SLM_WM_SKIP_DATASET_QUALITY", "0") != "1":
        quality_manifest = write_dataset_level_quality_outputs(
            paper_run_name=paper_run.run_name,
            target_fpr=paper_run.target_fpr,
            root=ROOT,
            real_attack_registry_path=(
                f"outputs/image_only_dataset_runtime/{paper_run.run_name}/watermark_quality_image_registry.jsonl"
            ),
            formal_min_sample_count=paper_run.dataset_level_quality_minimum_count,
            auto_extract_formal_features=True,
            inception_device_name=os.environ.get("SLM_WM_INCEPTION_DEVICE"),
            inception_batch_size=int(os.environ.get("SLM_WM_INCEPTION_BATCH_SIZE", "32")),
        )
        quality_archive_path = package_dataset_level_quality_outputs(paper_run.run_name, root=ROOT)
    print(
        json.dumps(
            {
                "summary": summary,
                "archive_path": str(archive_path),
                "quality_manifest": quality_manifest,
                "quality_archive_path": None if quality_archive_path is None else str(quality_archive_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
