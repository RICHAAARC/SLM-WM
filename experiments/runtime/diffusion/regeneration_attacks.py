"""复用已加载模型执行真实再扩散、编辑和反演攻击。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from experiments.runners.real_attack_evaluation import (
    RealAttackEvaluationConfig,
    RealAttackSpec,
    run_pipeline_attack,
    run_strict_ddim_inversion_attack,
)


@dataclass
class DiffusionAttackRuntime:
    """保存 SD3 img2img 和 legacy DDIM 反演攻击的共享运行时。"""

    img2img_pipeline: Any
    evaluation_config: RealAttackEvaluationConfig

    @classmethod
    def from_text_to_image_pipeline(cls, pipeline: Any, config: Any) -> "DiffusionAttackRuntime":
        """从已加载 SD3/SD3.5 pipeline 共享组件创建 img2img 运行时。"""

        from diffusers import StableDiffusion3Img2ImgPipeline

        img2img = StableDiffusion3Img2ImgPipeline.from_pipe(pipeline)
        img2img = img2img.to(config.device_name)
        img2img.set_progress_bar_config(disable=True)
        evaluation_config = RealAttackEvaluationConfig(
            model_family=config.model_family,
            model_id=config.model_id,
            seed=config.seed,
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            width=config.width,
            height=config.height,
            inference_steps=config.inference_steps,
            guidance_scale=config.guidance_scale,
            device_name=config.device_name,
            torch_dtype=config.torch_dtype,
            hf_token_env=config.hf_token_env,
            enable_pipeline_progress_bar=False,
            enable_attack_progress_bar=False,
        )
        return cls(img2img_pipeline=img2img, evaluation_config=evaluation_config)

    def apply(self, source_image: Any, spec: RealAttackSpec, seed: int, prompt_text: str) -> Any:
        """根据攻击协议执行真实图像生成攻击。"""

        if spec.attack_name == "ddim_inversion_regeneration":
            return run_strict_ddim_inversion_attack(
                source_image,
                spec,
                self.evaluation_config,
                seed,
                prompt_text,
            )
        return run_pipeline_attack(
            self.img2img_pipeline,
            source_image,
            spec,
            self.evaluation_config,
            seed,
            prompt_text,
        )
