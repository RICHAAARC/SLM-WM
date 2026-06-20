"""SD3 runtime adapter。"""

from __future__ import annotations

from experiments.runtime.diffusion.attention_capture import build_attention_capture_records
from experiments.runtime.diffusion.latent_estimator import build_initial_latent, build_prompt_delta
from experiments.runtime.diffusion.latent_trace import build_latent_trace_records
from experiments.runtime.diffusion.model_adapter import RuntimeModelConfig, RuntimeProbeBundle, build_generation_record
from experiments.runtime.diffusion.sampler_hook import run_synthetic_sampler


class Sd3RuntimeAdapter:
    """为 SD3 提供可替换的 runtime adapter 边界。"""

    backend_name = "synthetic_latent_adapter"
    runtime_dependency_mode = "synthetic_fallback"
    unsupported_reason = "real_sd3_backend_unavailable"

    def generate(self, config: RuntimeModelConfig) -> RuntimeProbeBundle:
        """运行 SD3 probe, 当前默认使用 synthetic fallback。"""
        initial_latent = build_initial_latent(config.model_id, config.prompt, config.seed, config.latent_width)
        prompt_delta = build_prompt_delta(config.prompt, config.negative_prompt, config.latent_width)
        sampling_trace = run_synthetic_sampler(
            initial_latent=initial_latent,
            prompt_delta=prompt_delta,
            inference_steps=config.inference_steps,
            guidance_scale=config.guidance_scale,
        )
        run_id = config.build_run_id()
        final_latent = sampling_trace.trajectory_vectors[-1]
        latent_records = build_latent_trace_records(
            run_id=run_id,
            model_family=config.model_family,
            model_id=config.model_id,
            backend_name=self.backend_name,
            timesteps=sampling_trace.timesteps,
            trajectory_vectors=sampling_trace.trajectory_vectors,
            unsupported_reason=self.unsupported_reason,
        )
        attention_records = build_attention_capture_records(
            run_id=run_id,
            model_family=config.model_family,
            model_id=config.model_id,
            backend_name=self.backend_name,
            trajectory_vectors=sampling_trace.trajectory_vectors,
            unsupported_reason=self.unsupported_reason,
        )
        generation_record = build_generation_record(
            config=config,
            run_id=run_id,
            backend_name=self.backend_name,
            runtime_dependency_mode=self.runtime_dependency_mode,
            final_latent=final_latent,
            unsupported_reason=self.unsupported_reason,
        )
        return RuntimeProbeBundle(
            generation_record=generation_record,
            latent_trace_records=latent_records,
            attention_capture_records=attention_records,
        )
