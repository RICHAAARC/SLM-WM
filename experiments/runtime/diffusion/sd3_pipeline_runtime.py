"""SD3/SD3.5 pipeline 运行时通用工具。

该模块属于核心方法复现层, 用于把真实模型加载、tensor 摘要辅助和轻量图像质量
计算从 Notebook helper 中抽离出来。这样主方法 runner 可以在服务器和 Colab
入口中复用同一套实现, 不需要反向依赖 `paper_workflow/`。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from experiments.runtime import repository_environment
from experiments.runtime.model_sources import require_registered_model_reference
from experiments.runtime.repository_environment import (
    build_runtime_environment_report,
    flatten_environment_versions,
)


def import_runtime_dependencies() -> tuple[Any, Any, Any, Any]:
    """延迟导入真实模型和图像依赖。"""

    import torch
    import diffusers
    from diffusers import StableDiffusion3Pipeline

    return None, torch, diffusers, StableDiffusion3Pipeline


def _qualified_class_name(value: Any) -> str:
    """返回对象或类的完整模块限定类名。"""

    resolved_class = value if isinstance(value, type) else type(value)
    return f"{resolved_class.__module__}.{resolved_class.__qualname__}"


def _module_dtype_name(module: Any, component_name: str) -> str:
    """读取真实模型组件 dtype, 缺少参数时按 fail-closed 处理。"""

    dtype = getattr(module, "dtype", None)
    if dtype is None:
        parameters = getattr(module, "parameters", None)
        if not callable(parameters):
            raise RuntimeError(f"{component_name} 不提供可核验的 dtype")
        try:
            dtype = next(iter(parameters())).dtype
        except (StopIteration, AttributeError) as exc:
            raise RuntimeError(f"{component_name} 不提供可核验的 dtype") from exc
    dtype_name = str(dtype)
    return dtype_name.removeprefix("torch.")


def _validate_loaded_pipeline(config: Any, pipeline: Any) -> dict[str, Any]:
    """复验实际加载组件的类、VAE 归一化常量和参数 dtype。"""

    components = {
        "pipeline": pipeline,
        "vae": getattr(pipeline, "vae", None),
        "transformer": getattr(pipeline, "transformer", None),
        "scheduler": getattr(pipeline, "scheduler", None),
    }
    if any(component is None for component in components.values()):
        raise RuntimeError("SD3.5 pipeline 缺少 VAE、Transformer 或 scheduler")
    expected_class_names = {
        "pipeline": config.pipeline_class_name,
        "vae": config.vae_class_name,
        "transformer": config.transformer_class_name,
        "scheduler": config.scheduler_class_name,
    }
    actual_class_names = {
        name: _qualified_class_name(component)
        for name, component in components.items()
    }
    if actual_class_names != expected_class_names:
        raise RuntimeError(
            "SD3.5 pipeline 组件类身份不匹配: "
            f"expected={expected_class_names}, actual={actual_class_names}"
        )
    vae_config = getattr(components["vae"], "config", None)
    if vae_config is None:
        raise RuntimeError("SD3.5 VAE 缺少可核验的 config")
    actual_scaling_factor = getattr(vae_config, "scaling_factor", None)
    actual_shift_factor = getattr(vae_config, "shift_factor", None)
    if (
        actual_scaling_factor != config.vae_scaling_factor
        or actual_shift_factor != config.vae_shift_factor
    ):
        raise RuntimeError(
            "SD3.5 VAE scaling_factor 或 shift_factor 与正式配置不匹配"
        )
    component_dtypes = {
        "vae": _module_dtype_name(components["vae"], "VAE"),
        "transformer": _module_dtype_name(
            components["transformer"],
            "Transformer",
        ),
    }
    if any(
        dtype_name != config.latent_torch_dtype
        for dtype_name in component_dtypes.values()
    ):
        raise RuntimeError(
            "SD3.5 latent 组件 dtype 与正式配置不匹配: "
            f"{component_dtypes}"
        )
    return {
        "component_class_names": actual_class_names,
        "vae_scaling_factor": float(actual_scaling_factor),
        "vae_shift_factor": float(actual_shift_factor),
        "latent_component_dtypes": component_dtypes,
    }


def load_pipeline(config: Any) -> tuple[Any, dict[str, Any]]:
    """加载真实 SD3 系列 pipeline 并建立受治理的设备放置。

    `config` 必须提供设备、精确模型 revision、组件类身份、VAE 归一化常量和
    latent dtype。加载后会复验实际对象, 任一身份不匹配都会在科学运行前失败。
    """

    formal_execution_lock = (
        repository_environment.require_published_formal_execution_lock(Path.cwd())
    )
    model_source = require_registered_model_reference(
        config.model_id,
        config.model_revision,
        required_usage_role="primary_diffusion_model",
    )
    _, torch, _, pipeline_class = import_runtime_dependencies()
    if config.device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("gpu_unavailable")
    environment_report = build_runtime_environment_report(
        "sd35_method_runtime_gpu",
        torch_module=torch,
        verified_formal_execution_lock=formal_execution_lock,
    )
    if environment_report["dependency_environment_ready"] is not True:
        blockers = ",".join(environment_report["dependency_readiness_blockers"])
        raise RuntimeError(f"dependency_profile_environment_not_ready:{blockers}")
    if _qualified_class_name(pipeline_class) != config.pipeline_class_name:
        raise RuntimeError("导入的 SD3.5 pipeline 类与正式配置不匹配")
    dtype = getattr(torch, config.latent_torch_dtype)
    token = os.environ.get(config.hf_token_env) or None
    pipeline = pipeline_class.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        torch_dtype=dtype,
        token=token,
    )
    operator_identity = _validate_loaded_pipeline(config, pipeline)
    if config.device_name == "cuda":
        # 模型级 CPU offload 仅改变组件驻留位置, 不改变权重、dtype 或科学算子。
        # 该通用运行时策略避免三个文本编码器与需要自动微分的 Transformer
        # 同时常驻 GPU, 从而为精确 JVP/VJP 和 Q/K 梯度保留显存。
        pipeline.enable_model_cpu_offload(device=config.device_name)
        device_placement = {
            "placement_protocol": "diffusers_model_cpu_offload",
            "execution_device": config.device_name,
            "offload_device": "cpu",
        }
    else:
        pipeline = pipeline.to(config.device_name)
        device_placement = {
            "placement_protocol": "whole_pipeline_single_device",
            "execution_device": config.device_name,
            "offload_device": None,
        }
    pipeline.set_progress_bar_config(disable=False)
    runtime_versions = {
        **flatten_environment_versions(environment_report),
        "runtime_environment": environment_report,
        "diffusion_model_source": model_source.to_dict(),
        "sd35_operator_identity": operator_identity,
        "sd35_device_placement": device_placement,
    }
    return pipeline, runtime_versions


def tensor_norm(tensor: Any) -> float:
    """计算 tensor 的二范数。"""

    return float(tensor.detach().float().norm().item())
