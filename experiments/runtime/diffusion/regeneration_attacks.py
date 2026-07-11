"""使用已加载 SD3.5 模型执行共同扩散攻击与检测器引导去水印。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
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


@dataclass(frozen=True)
class DiffusionAttackExecution:
    """保存真实攻击图像及支持复核的执行轨迹。"""

    image: Any
    attack_name: str
    attack_implementation: str
    attack_seed_random: int
    effective_parameters: dict[str, Any]
    local_edit_mask_digest: str = ""
    local_edit_mask_area_ratio: float | None = None
    detector_query_trace: tuple[dict[str, Any], ...] = ()

    def to_record(self) -> dict[str, Any]:
        """转换为不包含图像对象的受治理记录。"""

        return {
            "attack_name": self.attack_name,
            "attack_implementation": self.attack_implementation,
            "attack_seed_random": self.attack_seed_random,
            "effective_parameters": dict(self.effective_parameters),
            "local_edit_mask_digest": self.local_edit_mask_digest,
            "local_edit_mask_area_ratio": self.local_edit_mask_area_ratio,
            "detector_query_trace": [dict(item) for item in self.detector_query_trace],
        }


def default_diffusion_attack_specs() -> tuple[DiffusionAttackSpec, ...]:
    """从唯一共同攻击配置生成扩散攻击规格。"""

    implementations = {
        "img2img_regeneration": "sd3_img2img",
        "flow_matching_inversion_regeneration": "sd3_flow_matching_inverse_euler_regeneration",
        "sdedit_regeneration": "sd3_img2img_sdedit",
        "diffusion_purification": "sd3_img2img_purification",
        "global_editing_attack": "sd3_img2img_global_edit",
        "local_editing_attack": "sd3_inpainting_local_edit",
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


def diffusion_attack_spec(attack_name: str) -> DiffusionAttackSpec:
    """按正式名称返回唯一扩散攻击规格。"""

    matches = tuple(spec for spec in default_diffusion_attack_specs() if spec.attack_name == attack_name)
    if len(matches) != 1:
        raise ValueError(f"扩散攻击必须唯一登记: {attack_name}")
    return matches[0]


@dataclass(frozen=True)
class DiffusionAttackRuntimeConfig:
    """定义可在主方法与 baseline 之间共享的扩散攻击运行边界。"""

    device_name: str = "cuda"
    negative_prompt: str = "low quality, blurry"
    inference_steps: int = 28
    guidance_scale: float = 4.5
    height: int = 512
    width: int = 512

    def __post_init__(self) -> None:
        """集中校验 GPU 扩散攻击所需的数值边界。"""

        if self.inference_steps <= 0:
            raise ValueError("扩散攻击推理步数必须为正数")
        if self.guidance_scale < 0.0:
            raise ValueError("扩散攻击 guidance_scale 不得小于 0")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("扩散攻击图像尺寸必须为正数")

    @classmethod
    def from_runtime_config(cls, config: Any) -> "DiffusionAttackRuntimeConfig":
        """从项目运行配置提取攻击所需字段, 避免依赖外层配置类型。"""

        return cls(
            device_name=str(config.device_name),
            negative_prompt=str(config.negative_prompt),
            inference_steps=int(config.inference_steps),
            guidance_scale=float(config.guidance_scale),
            height=int(config.height),
            width=int(config.width),
        )


@dataclass
class DiffusionAttackRuntime:
    """保存 SD3.5 text-to-image、img2img、inpaint 与共同运行配置。"""

    text_to_image_pipeline: Any
    img2img_pipeline: Any
    config: DiffusionAttackRuntimeConfig
    inpaint_pipeline: Any | None = None

    @classmethod
    def from_text_to_image_pipeline(cls, pipeline: Any, config: Any) -> "DiffusionAttackRuntime":
        """从主方法已加载 pipeline 共享权重创建 img2img 运行时。"""

        from diffusers import StableDiffusion3Img2ImgPipeline

        runtime_config = (
            config
            if isinstance(config, DiffusionAttackRuntimeConfig)
            else DiffusionAttackRuntimeConfig.from_runtime_config(config)
        )
        img2img = StableDiffusion3Img2ImgPipeline.from_pipe(pipeline).to(runtime_config.device_name)
        img2img.set_progress_bar_config(disable=True)
        return cls(pipeline, img2img, runtime_config)

    def _get_inpaint_pipeline(self) -> Any:
        """按需共享 SD3.5 权重创建真实 inpainting pipeline。"""

        if self.inpaint_pipeline is None:
            from diffusers import AutoPipelineForInpainting

            self.inpaint_pipeline = AutoPipelineForInpainting.from_pipe(
                self.text_to_image_pipeline
            ).to(self.config.device_name)
            self.inpaint_pipeline.set_progress_bar_config(disable=True)
        return self.inpaint_pipeline

    @staticmethod
    def _first_image(result: Any) -> Any:
        """读取 Diffusers pipeline 的首幅输出图像。"""

        images = getattr(result, "images", None)
        if not images:
            raise RuntimeError("扩散攻击 pipeline 未返回图像")
        return images[0]

    def _img2img(
        self,
        source_image: Any,
        prompt_text: str,
        strength: float,
        seed: int,
        *,
        inference_steps: int | None = None,
        guidance_scale: float | None = None,
        negative_prompt: str | None = None,
    ) -> tuple[Any, str, float]:
        """以确定性种子执行一次真实 SD3.5 img2img。"""

        import torch

        generator = torch.Generator(device=self.config.device_name).manual_seed(int(seed))
        result = self.img2img_pipeline(
            prompt=prompt_text,
            negative_prompt=self.config.negative_prompt if negative_prompt is None else negative_prompt,
            image=source_image,
            strength=float(strength),
            num_inference_steps=int(inference_steps or self.config.inference_steps),
            guidance_scale=(
                self.config.guidance_scale if guidance_scale is None else float(guidance_scale)
            ),
            generator=generator,
        )
        return self._first_image(result)

    @staticmethod
    def _local_edit_mask(source_image: Any, mask_ratio: float, seed: int) -> Any:
        """生成面积受控且按样本种子复现的局部白色编辑区域。"""

        from PIL import Image, ImageDraw

        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("local_mask_ratio 必须位于 (0, 1)")
        width, height = source_image.size
        side_fraction = math.sqrt(mask_ratio)
        mask_width = min(width, max(1, round(width * side_fraction)))
        mask_height = min(height, max(1, round(height * side_fraction)))
        generator = random.Random(int(seed))
        left = generator.randint(0, max(0, width - mask_width))
        top = generator.randint(0, max(0, height - mask_height))
        mask = Image.new("L", (width, height), color=0)
        ImageDraw.Draw(mask).rectangle(
            (left, top, left + mask_width - 1, top + mask_height - 1),
            fill=255,
        )
        return mask

    def _inpaint(
        self,
        source_image: Any,
        prompt_text: str,
        strength: float,
        mask_ratio: float,
        seed: int,
        *,
        inference_steps: int,
        guidance_scale: float,
        negative_prompt: str,
    ) -> Any:
        """只在白色 mask 区域执行真实 SD3.5 inpainting 局部编辑。"""

        import torch

        generator = torch.Generator(device=self.config.device_name).manual_seed(int(seed))
        mask = self._local_edit_mask(source_image, mask_ratio, seed)
        result = self._get_inpaint_pipeline()(
            prompt=prompt_text,
            negative_prompt=negative_prompt,
            image=source_image,
            mask_image=mask,
            strength=float(strength),
            num_inference_steps=int(inference_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        )
        from PIL import Image

        generated = self._first_image(result).convert("RGB")
        mask_digest = hashlib.sha256(mask.tobytes()).hexdigest()
        mask_area_ratio = sum(1 for value in mask.getdata() if value > 0) / float(
            mask.width * mask.height
        )
        return (
            Image.composite(generated, source_image.convert("RGB"), mask),
            mask_digest,
            mask_area_ratio,
        )

    def _flow_matching_inversion(
        self,
        source_image: Any,
        prompt_text: str,
        *,
        inversion_steps: int,
        guidance_scale: float,
    ) -> tuple[Any, tuple[dict[str, Any], ...]]:
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
                int(inversion_steps),
                device=pipeline._execution_device,
            )
            do_classifier_free_guidance = float(guidance_scale) > 1.0
            prompt_embeds, negative_prompt_embeds, pooled, negative_pooled = pipeline.encode_prompt(
                prompt=prompt_text,
                prompt_2=None,
                prompt_3=None,
                device=pipeline._execution_device,
                negative_prompt=self.config.negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                pooled = torch.cat([negative_pooled, pooled])
            timesteps = pipeline.scheduler.timesteps
            sigmas = pipeline.scheduler.sigmas
            for schedule_index in range(len(timesteps) - 1, -1, -1):
                timestep = timesteps[schedule_index]
                latent_model_input = (
                    torch.cat([latents, latents]) if do_classifier_free_guidance else latents
                )
                prediction = pipeline.transformer(
                    latent_model_input,
                    timestep=timestep.expand(latent_model_input.shape[0]),
                    pooled_projections=pooled,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                if do_classifier_free_guidance:
                    prediction_unconditional, prediction_conditional = prediction.chunk(2)
                    prediction = prediction_unconditional + float(guidance_scale) * (
                        prediction_conditional - prediction_unconditional
                    )
                sigma_current = sigmas[schedule_index + 1].to(latents)
                sigma_next = sigmas[schedule_index].to(latents)
                latents = latents + (sigma_next - sigma_current) * prediction
        return latents

    def _run_inversion_regeneration(
        self,
        source_image: Any,
        prompt_text: str,
        seed: int,
        spec: DiffusionAttackSpec,
    ) -> Any:
        """从真实反演 latent 重新执行 SD3.5 生成。"""

        import torch

        inversion_steps = int(spec.attack_parameters["inversion_steps"])
        reconstruction_steps = int(spec.attack_parameters["reconstruction_steps"])
        guidance_scale = float(spec.attack_parameters["guidance_scale"])
        latents = self._flow_matching_inversion(
            source_image,
            prompt_text,
            inversion_steps=inversion_steps,
            guidance_scale=guidance_scale,
        )
        generator = torch.Generator(device=self.config.device_name).manual_seed(int(seed))
        result = self.text_to_image_pipeline(
            prompt=prompt_text,
            negative_prompt=str(spec.attack_parameters["negative_prompt"]),
            latents=latents,
            height=self.config.height,
            width=self.config.width,
            num_inference_steps=reconstruction_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return self._first_image(result)

    def _run_detector_guided_removal(
        self,
        source_image: Any,
        spec: DiffusionAttackSpec,
        seed: int,
        prompt_text: str,
        detection_score: Callable[[Any], float],
    ) -> Any:
        """在固定查询预算下最小化实际盲检连续分数。"""

        query_count = int(spec.attack_parameters["candidate_query_count"])
        strength_min = float(spec.attack_parameters["strength_min"])
        strength_max = float(spec.attack_parameters["strength_max"])
        if query_count < 2:
            raise ValueError("对抗去水印至少需要2次候选查询")
        best_image = source_image
        best_score = float(detection_score(source_image))
        if not math.isfinite(best_score):
            raise ValueError("对抗去水印检测分数必须是有限数值")
        query_trace: list[dict[str, Any]] = [
            {
                "candidate_index": -1,
                "candidate_strength": 0.0,
                "candidate_seed_random": int(seed),
                "detection_score": best_score,
                "is_source_image": True,
            }
        ]
        for query_index in range(query_count):
            fraction = query_index / (query_count - 1)
            strength = strength_min + (strength_max - strength_min) * fraction
            candidate = self._img2img(
                source_image,
                prompt_text,
                strength,
                seed + query_index,
                inference_steps=int(spec.attack_parameters["inference_steps"]),
                guidance_scale=float(spec.attack_parameters["guidance_scale"]),
                negative_prompt=str(spec.attack_parameters["negative_prompt"]),
            )
            candidate_score = float(detection_score(candidate))
            if not math.isfinite(candidate_score):
                raise ValueError("对抗去水印检测分数必须是有限数值")
            query_trace.append(
                {
                    "candidate_index": query_index,
                    "candidate_strength": strength,
                    "candidate_seed_random": int(seed + query_index),
                    "detection_score": candidate_score,
                    "is_source_image": False,
                }
            )
            if candidate_score < best_score:
                best_image = candidate
                best_score = candidate_score
        return best_image, tuple(query_trace)

    def apply(
        self,
        source_image: Any,
        spec: DiffusionAttackSpec,
        seed: int,
        prompt_text: str,
        detection_score: Callable[[Any], float] | None = None,
    ) -> DiffusionAttackExecution:
        """按共同协议执行指定真实扩散攻击。"""

        if spec.attack_name == "flow_matching_inversion_regeneration":
            image = self._run_inversion_regeneration(source_image, prompt_text, seed, spec)
            return DiffusionAttackExecution(
                image=image,
                attack_name=spec.attack_name,
                attack_implementation=spec.attack_implementation,
                attack_seed_random=int(seed),
                effective_parameters=dict(spec.attack_parameters),
            )
        if spec.attack_name == "adversarial_removal_attack":
            if detection_score is None:
                raise ValueError("对抗去水印攻击必须接收实际盲检连续分数函数")
            image, query_trace = self._run_detector_guided_removal(
                source_image,
                spec,
                seed,
                prompt_text,
                detection_score,
            )
            return DiffusionAttackExecution(
                image=image,
                attack_name=spec.attack_name,
                attack_implementation=spec.attack_implementation,
                attack_seed_random=int(seed),
                effective_parameters=dict(spec.attack_parameters),
                detector_query_trace=query_trace,
            )
        if spec.attack_name == "local_editing_attack":
            prompt = f"{prompt_text}, {spec.attack_parameters['edit_prompt_suffix']}"
            image, mask_digest, mask_area_ratio = self._inpaint(
                source_image,
                prompt,
                float(spec.attack_parameters["denoise_strength"]),
                float(spec.attack_parameters["local_mask_ratio"]),
                seed,
                inference_steps=int(spec.attack_parameters["inference_steps"]),
                guidance_scale=float(spec.attack_parameters["guidance_scale"]),
                negative_prompt=str(spec.attack_parameters["negative_prompt"]),
            )
            return DiffusionAttackExecution(
                image=image,
                attack_name=spec.attack_name,
                attack_implementation=spec.attack_implementation,
                attack_seed_random=int(seed),
                effective_parameters=dict(spec.attack_parameters),
                local_edit_mask_digest=mask_digest,
                local_edit_mask_area_ratio=mask_area_ratio,
            )
        if spec.attack_name not in {
            "img2img_regeneration",
            "sdedit_regeneration",
            "diffusion_purification",
            "global_editing_attack",
            "visual_paraphrase_attack",
        }:
            raise ValueError(f"未登记的扩散攻击实现: {spec.attack_name}")
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
        if spec.attack_name == "diffusion_purification":
            image = self._img2img(
                source_image,
                "",
                strength,
                seed,
                inference_steps=int(spec.attack_parameters["purification_steps"]),
                guidance_scale=float(spec.attack_parameters["guidance_scale"]),
                negative_prompt=str(spec.attack_parameters["negative_prompt"]),
            )
        else:
            image = self._img2img(
                source_image,
                prompt,
                strength,
                seed,
                inference_steps=int(spec.attack_parameters["inference_steps"]),
                guidance_scale=float(spec.attack_parameters["guidance_scale"]),
                negative_prompt=str(spec.attack_parameters["negative_prompt"]),
            )
        return DiffusionAttackExecution(
            image=image,
            attack_name=spec.attack_name,
            attack_implementation=spec.attack_implementation,
            attack_seed_random=int(seed),
            effective_parameters=dict(spec.attack_parameters),
        )
