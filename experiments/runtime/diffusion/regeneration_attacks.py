"""使用已加载 SD3.5 模型执行共同扩散攻击与检测器引导去水印。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from experiments.protocol.attacks import default_attack_configs


@dataclass(frozen=True)
class DiffusionAttackSpec:
    """描述一个必须在 GPU 上真实执行的共同攻击。"""

    attack_id: str
    attack_family: str
    attack_name: str
    attack_strength: float
    attack_parameters: dict[str, Any]
    attack_implementation: str


def default_diffusion_attack_specs() -> tuple[DiffusionAttackSpec, ...]:
    """从唯一共同攻击配置生成扩散攻击规格。"""

    implementations = {
        "img2img_regeneration": "sd3_img2img",
        "ddim_inversion_regeneration": "sd3_flow_matching_inverse_integration",
        "sdedit_regeneration": "sd3_img2img_sdedit",
        "diffusion_purification": "sd3_img2img_purification",
        "global_editing_attack": "sd3_img2img_global_edit",
        "local_editing_attack": "sd3_img2img_local_edit",
        "visual_paraphrase_attack": "sd3_img2img_visual_paraphrase",
        "adversarial_removal_attack": "detector_guided_black_box_img2img_search",
    }
    return tuple(
        DiffusionAttackSpec(
            attack_id=config.attack_id,
            attack_family=config.attack_family,
            attack_name=config.attack_name,
            attack_strength=config.attack_strength,
            attack_parameters=dict(config.attack_parameters),
            attack_implementation=implementations[config.attack_name],
        )
        for config in default_attack_configs()
        if config.enabled and config.requires_gpu and config.resource_profile == "full_extra"
    )


@dataclass
class DiffusionAttackRuntime:
    """保存 SD3.5 text-to-image、img2img 与共同运行配置。"""

    text_to_image_pipeline: Any
    img2img_pipeline: Any
    config: Any

    @classmethod
    def from_text_to_image_pipeline(cls, pipeline: Any, config: Any) -> "DiffusionAttackRuntime":
        """从主方法已加载 pipeline 共享权重创建 img2img 运行时。"""

        from diffusers import StableDiffusion3Img2ImgPipeline

        img2img = StableDiffusion3Img2ImgPipeline.from_pipe(pipeline).to(config.device_name)
        img2img.set_progress_bar_config(disable=True)
        return cls(pipeline, img2img, config)

    def _img2img(
        self,
        source_image: Any,
        prompt_text: str,
        strength: float,
        seed: int,
    ) -> Any:
        """以确定性种子执行一次真实 SD3.5 img2img。"""

        import torch

        generator = torch.Generator(device=self.config.device_name).manual_seed(int(seed))
        return self.img2img_pipeline(
            prompt=prompt_text,
            negative_prompt=self.config.negative_prompt,
            image=source_image,
            strength=float(strength),
            num_inference_steps=self.config.inference_steps,
            guidance_scale=self.config.guidance_scale,
            generator=generator,
        ).images[0]

    def _flow_matching_inversion(self, source_image: Any, prompt_text: str) -> Any:
        """通过 SD3 scheduler 的反向 Euler 积分恢复高噪声 latent。"""

        import torch

        pipeline = self.text_to_image_pipeline
        dtype = next(pipeline.vae.parameters()).dtype
        pixels = pipeline.image_processor.preprocess(source_image).to(
            device=pipeline._execution_device,
            dtype=dtype,
        )
        with torch.inference_mode():
            encoding = pipeline.vae.encode(pixels).latent_dist.mode()
            shift = float(getattr(pipeline.vae.config, "shift_factor", 0.0) or 0.0)
            scale = float(getattr(pipeline.vae.config, "scaling_factor", 1.0) or 1.0)
            latents = (encoding - shift) * scale
            pipeline.scheduler.set_timesteps(
                self.config.inference_steps,
                device=pipeline._execution_device,
            )
            prompt_embeds, _, pooled, _ = pipeline.encode_prompt(
                prompt=prompt_text,
                prompt_2=None,
                prompt_3=None,
                device=pipeline._execution_device,
                do_classifier_free_guidance=False,
            )
            timesteps = pipeline.scheduler.timesteps
            sigmas = pipeline.scheduler.sigmas
            for schedule_index in range(len(timesteps) - 1, -1, -1):
                timestep = timesteps[schedule_index]
                prediction = pipeline.transformer(
                    latents,
                    timestep=timestep.expand(latents.shape[0]),
                    pooled_projections=pooled,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                sigma_current = sigmas[schedule_index + 1].to(latents)
                sigma_next = sigmas[schedule_index].to(latents)
                latents = latents + (sigma_next - sigma_current) * prediction
        return latents

    def _run_inversion_regeneration(
        self,
        source_image: Any,
        prompt_text: str,
        seed: int,
    ) -> Any:
        """从真实反演 latent 重新执行 SD3.5 生成。"""

        import torch

        latents = self._flow_matching_inversion(source_image, prompt_text)
        generator = torch.Generator(device=self.config.device_name).manual_seed(int(seed))
        return self.text_to_image_pipeline(
            prompt=prompt_text,
            negative_prompt=self.config.negative_prompt,
            latents=latents,
            height=self.config.height,
            width=self.config.width,
            num_inference_steps=self.config.inference_steps,
            guidance_scale=self.config.guidance_scale,
            generator=generator,
        ).images[0]

    def _run_detector_guided_removal(
        self,
        source_image: Any,
        spec: DiffusionAttackSpec,
        seed: int,
        prompt_text: str,
        detection_score: Callable[[Any], float],
    ) -> Any:
        """在固定查询预算下最小化实际盲检连续分数。"""

        query_count = int(spec.attack_parameters["query_count"])
        strength_min = float(spec.attack_parameters["strength_min"])
        strength_max = float(spec.attack_parameters["strength_max"])
        if query_count < 2:
            raise ValueError("对抗去水印至少需要2次候选查询")
        best_image = source_image
        best_score = float(detection_score(source_image))
        for query_index in range(query_count):
            fraction = query_index / (query_count - 1)
            strength = strength_min + (strength_max - strength_min) * fraction
            candidate = self._img2img(source_image, prompt_text, strength, seed + query_index)
            candidate_score = float(detection_score(candidate))
            if candidate_score < best_score:
                best_image = candidate
                best_score = candidate_score
        return best_image

    def apply(
        self,
        source_image: Any,
        spec: DiffusionAttackSpec,
        seed: int,
        prompt_text: str,
        detection_score: Callable[[Any], float] | None = None,
    ) -> Any:
        """按共同协议执行指定真实扩散攻击。"""

        if spec.attack_name == "ddim_inversion_regeneration":
            return self._run_inversion_regeneration(source_image, prompt_text, seed)
        if spec.attack_name == "adversarial_removal_attack":
            if detection_score is None:
                raise ValueError("对抗去水印攻击必须接收实际盲检连续分数函数")
            return self._run_detector_guided_removal(
                source_image,
                spec,
                seed,
                prompt_text,
                detection_score,
            )
        prompt = prompt_text
        if spec.attack_name == "global_editing_attack":
            prompt = f"{prompt_text}, {spec.attack_parameters['edit_prompt_suffix']}"
        elif spec.attack_name == "visual_paraphrase_attack":
            prompt = f"{prompt_text}, {spec.attack_parameters['paraphrase_prompt_suffix']}"
        strength = float(
            spec.attack_parameters.get(
                "denoise_strength",
                spec.attack_parameters.get("noise_level", spec.attack_strength),
            )
        )
        return self._img2img(source_image, prompt, strength, seed)
