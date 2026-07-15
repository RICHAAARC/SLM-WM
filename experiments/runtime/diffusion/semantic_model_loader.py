"""加载共同协议登记的语义条件模型。

该模块位于实验运行层, 因为模型注册表、下载来源和设备放置均属于外层运行环境。
核心方法层只接收已经构造的 VAE 与视觉编码器, 因而不会反向依赖实验层。
"""

from __future__ import annotations

from typing import Any

from experiments.runtime.model_sources import require_registered_model_reference


def load_clip_vision_model(
    model_id: str,
    model_revision: str,
    device_name: str,
    torch_dtype: str = "float32",
) -> Any:
    """加载并冻结登记的 CLIP 图像编码器, 供核心语义特征运行时使用。"""

    import torch
    from transformers import CLIPVisionModelWithProjection

    require_registered_model_reference(
        model_id,
        model_revision,
        required_usage_role="semantic_condition_encoder",
    )
    dtype = getattr(torch, torch_dtype)
    model = CLIPVisionModelWithProjection.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    model = model.to(device_name)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
