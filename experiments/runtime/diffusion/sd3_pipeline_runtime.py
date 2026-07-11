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


def load_pipeline(config: Any) -> tuple[Any, dict[str, Any]]:
    """加载真实 SD3 系列 pipeline 并移动到目标设备。

    `config` 只要求提供 `device_name`、`torch_dtype`、`hf_token_env` 和
    `model_id` 字段, 因而可被最小机制预检 runner 与正式 latent injection
    runner 共同复用。
    """

    formal_execution_lock = (
        repository_environment.require_published_formal_execution_lock(Path.cwd())
    )
    _, torch, _, pipeline_class = import_runtime_dependencies()
    if config.device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("gpu_unavailable")
    dtype = getattr(torch, config.torch_dtype)
    token = os.environ.get(config.hf_token_env) or None
    pipeline = pipeline_class.from_pretrained(config.model_id, torch_dtype=dtype, token=token)
    pipeline = pipeline.to(config.device_name)
    pipeline.set_progress_bar_config(disable=False)
    environment_report = build_runtime_environment_report(
        torch_module=torch,
        verified_formal_execution_lock=formal_execution_lock,
    )
    runtime_versions = {
        **flatten_environment_versions(environment_report),
        "runtime_environment": environment_report,
    }
    return pipeline, runtime_versions


def tensor_norm(tensor: Any) -> float:
    """计算 tensor 的二范数。"""

    return float(tensor.detach().float().norm().item())
