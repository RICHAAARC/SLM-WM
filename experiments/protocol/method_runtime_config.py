"""加载正式 SLM-WM 方法配置并约束配置来源唯一性。

该模块只解析方法与模型身份, 不解析论文运行规模、输出目录或设备选择。
因此 probe_paper、pilot_paper 和 full_paper 可以共享同一份方法配置, 同时由
``paper_run_config`` 独立控制 Prompt 数量与 fixed-FPR 统计强度。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml

from experiments.runtime.model_sources import require_registered_model_reference


FORMAL_METHOD_CONFIG_RELATIVE_PATH = Path("configs/model_sd35.yaml")


@dataclass(frozen=True)
class FormalMethodRuntimeConfig:
    """保存由 YAML 唯一确定的正式方法参数。"""

    model_family: str
    model_id: str
    model_revision: str
    vision_model_id: str
    vision_model_revision: str
    backend_mode: str
    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int
    inference_steps: int
    guidance_scale: float
    detector_input_access_mode: str
    jacobian_candidate_count: int
    null_space_rank: int
    lf_relative_strength: float
    tail_relative_strength: float
    attention_relative_strength: float
    attention_stable_token_fraction: float
    attention_unstable_pair_weight: float
    tail_fraction: float
    minimum_projection_energy_retention: float
    maximum_relative_response_residual: float
    null_space_cg_max_iterations: int
    null_space_cg_relative_tolerance: float
    minimum_semantic_preservation_cosine: float
    maximum_visual_feature_relative_drift: float
    injection_step_indices: tuple[int, ...]
    max_attention_tokens: int
    attention_module_count: int
    diffusion_attacks_enabled: bool

    def __post_init__(self) -> None:
        """集中校验模型身份和关键方法边界。"""

        require_registered_model_reference(
            self.model_id,
            self.model_revision,
            required_usage_role="primary_diffusion_model",
        )
        require_registered_model_reference(
            self.vision_model_id,
            self.vision_model_revision,
            required_usage_role="semantic_condition_encoder",
        )
        if self.model_family != "sd35" or self.backend_mode != "real_diffusion":
            raise ValueError("正式方法配置必须使用 sd35 real_diffusion 后端")
        if self.detector_input_access_mode != "image_key_public_model_only":
            raise ValueError("正式检测器必须保持仅图像盲检输入制度")
        if self.width <= 0 or self.height <= 0 or self.inference_steps <= 0:
            raise ValueError("图像尺寸和推理步数必须为正整数")
        if self.jacobian_candidate_count < self.null_space_rank or self.null_space_rank <= 0:
            raise ValueError("jacobian_candidate_count 必须不小于正的 null_space_rank")
        if any(index < 0 or index >= self.inference_steps for index in self.injection_step_indices):
            raise ValueError("injection_step_indices 必须位于推理步范围内")
        if not 0.0 < self.tail_fraction <= 1.0:
            raise ValueError("tail_fraction 必须位于 (0, 1]")
        if not 0.0 < self.attention_stable_token_fraction <= 1.0:
            raise ValueError(
                "attention_stable_token_fraction 必须位于 (0, 1]"
            )
        if not 0.0 <= self.attention_unstable_pair_weight < 1.0:
            raise ValueError(
                "attention_unstable_pair_weight 必须位于 [0, 1)"
            )
        if not 0.0 < self.minimum_projection_energy_retention <= 1.0:
            raise ValueError("minimum_projection_energy_retention 必须位于 (0, 1]")
        if not 0.0 < self.maximum_relative_response_residual <= 1.0:
            raise ValueError("maximum_relative_response_residual 必须位于 (0, 1]")
        if self.null_space_cg_max_iterations <= 0:
            raise ValueError("null_space_cg_max_iterations 必须为正整数")
        if not 0.0 < self.null_space_cg_relative_tolerance < 1.0:
            raise ValueError("null_space_cg_relative_tolerance 必须位于 (0, 1)")
        if not 0.0 < self.minimum_semantic_preservation_cosine <= 1.0:
            raise ValueError(
                "minimum_semantic_preservation_cosine 必须位于 (0, 1]"
            )
        if not 0.0 <= self.maximum_visual_feature_relative_drift <= 1.0:
            raise ValueError(
                "maximum_visual_feature_relative_drift 必须位于 [0, 1]"
            )
        if self.max_attention_tokens < 4 or self.attention_module_count < 2:
            raise ValueError("注意力几何配置不能退化为单层或过短 token 近似")
        if not self.diffusion_attacks_enabled:
            raise ValueError("正式方法配置必须启用真实扩散攻击协议")

    def paper_method_settings(self) -> dict[str, Any]:
        """返回需要写入各论文运行层级的共享方法字段。"""

        return {
            "inference_steps": self.inference_steps,
            "guidance_scale": self.guidance_scale,
            "attention_injection_steps": self.injection_step_indices,
            "jacobian_candidate_count": self.jacobian_candidate_count,
            "null_space_rank": self.null_space_rank,
            "lf_relative_strength": self.lf_relative_strength,
            "tail_relative_strength": self.tail_relative_strength,
            "attention_relative_strength": self.attention_relative_strength,
            "attention_stable_token_fraction": (
                self.attention_stable_token_fraction
            ),
            "attention_unstable_pair_weight": (
                self.attention_unstable_pair_weight
            ),
            "tail_fraction": self.tail_fraction,
            "minimum_projection_energy_retention": self.minimum_projection_energy_retention,
            "maximum_relative_response_residual": self.maximum_relative_response_residual,
            "null_space_cg_max_iterations": self.null_space_cg_max_iterations,
            "null_space_cg_relative_tolerance": (
                self.null_space_cg_relative_tolerance
            ),
            "minimum_semantic_preservation_cosine": (
                self.minimum_semantic_preservation_cosine
            ),
            "maximum_visual_feature_relative_drift": (
                self.maximum_visual_feature_relative_drift
            ),
        }


def resolve_formal_method_config_path(root: str | Path = ".") -> Path:
    """优先解析目标仓库配置, 打包测试缺少配置时回落到当前包内配置。"""

    requested_path = (Path(root) / FORMAL_METHOD_CONFIG_RELATIVE_PATH).resolve()
    package_path = (Path(__file__).resolve().parents[2] / FORMAL_METHOD_CONFIG_RELATIVE_PATH).resolve()
    return requested_path if requested_path.is_file() else package_path


def _required_payload(payload: Any, path: Path) -> dict[str, Any]:
    """确认 YAML 根节点与正式字段完整, 避免缺失字段被隐式默认。"""

    if not isinstance(payload, dict):
        raise ValueError(f"正式方法配置必须是 YAML 映射: {path}")
    required_fields = tuple(FormalMethodRuntimeConfig.__dataclass_fields__)
    missing_fields = tuple(field for field in required_fields if field not in payload)
    if missing_fields:
        raise ValueError(f"正式方法配置缺少字段 {missing_fields}: {path}")
    unknown_fields = tuple(sorted(set(payload) - set(required_fields)))
    if unknown_fields:
        raise ValueError(f"正式方法配置包含未消费字段 {unknown_fields}: {path}")
    return payload


def load_formal_method_runtime_config(root: str | Path = ".") -> FormalMethodRuntimeConfig:
    """从 ``configs/model_sd35.yaml`` 构造经过完整校验的方法配置。"""

    path = resolve_formal_method_config_path(root)
    payload = _required_payload(yaml.safe_load(path.read_text(encoding="utf-8")), path)
    normalized = dict(payload)
    normalized["injection_step_indices"] = tuple(int(value) for value in payload["injection_step_indices"])
    return FormalMethodRuntimeConfig(**normalized)


def require_formal_method_environment_consistency(config: FormalMethodRuntimeConfig) -> None:
    """拒绝残留环境变量改变 YAML 中冻结的方法身份或超参数。

    允许环境变量重复声明相同值, 便于 Notebook 显示当前配置; 任何不同值都会
    在模型加载前失败, 从而避免不同 Colab session 形成未登记的方法分叉。
    """

    expected_values = {
        "SLM_WM_MODEL_FAMILY": config.model_family,
        "SLM_WM_MODEL_ID": config.model_id,
        "SLM_WM_MODEL_REVISION": config.model_revision,
        "SLM_WM_VISION_MODEL_ID": config.vision_model_id,
        "SLM_WM_VISION_MODEL_REVISION": config.vision_model_revision,
        "SLM_WM_SEED": str(config.seed),
        "SLM_WM_IMAGE_WIDTH": str(config.width),
        "SLM_WM_IMAGE_HEIGHT": str(config.height),
        "SLM_WM_INFERENCE_STEPS": str(config.inference_steps),
        "SLM_WM_GUIDANCE_SCALE": str(config.guidance_scale),
        "SLM_WM_ATTENTION_INJECTION_STEPS": ",".join(str(value) for value in config.injection_step_indices),
        "SLM_WM_JACOBIAN_CANDIDATE_COUNT": str(config.jacobian_candidate_count),
        "SLM_WM_NULL_SPACE_RANK": str(config.null_space_rank),
        "SLM_WM_LF_RELATIVE_STRENGTH": str(config.lf_relative_strength),
        "SLM_WM_TAIL_RELATIVE_STRENGTH": str(config.tail_relative_strength),
        "SLM_WM_ATTENTION_RELATIVE_STRENGTH": str(config.attention_relative_strength),
        "SLM_WM_ATTENTION_STABLE_TOKEN_FRACTION": str(
            config.attention_stable_token_fraction
        ),
        "SLM_WM_ATTENTION_UNSTABLE_PAIR_WEIGHT": str(
            config.attention_unstable_pair_weight
        ),
        "SLM_WM_TAIL_FRACTION": str(config.tail_fraction),
        "SLM_WM_MINIMUM_PROJECTION_ENERGY_RETENTION": str(config.minimum_projection_energy_retention),
        "SLM_WM_MAXIMUM_RELATIVE_RESPONSE_RESIDUAL": str(config.maximum_relative_response_residual),
        "SLM_WM_NULL_SPACE_CG_MAX_ITERATIONS": str(
            config.null_space_cg_max_iterations
        ),
        "SLM_WM_NULL_SPACE_CG_RELATIVE_TOLERANCE": str(
            config.null_space_cg_relative_tolerance
        ),
        "SLM_WM_MINIMUM_SEMANTIC_PRESERVATION_COSINE": str(
            config.minimum_semantic_preservation_cosine
        ),
        "SLM_WM_MAXIMUM_VISUAL_FEATURE_RELATIVE_DRIFT": str(
            config.maximum_visual_feature_relative_drift
        ),
        "SLM_WM_MAX_ATTENTION_TOKENS": str(config.max_attention_tokens),
        "SLM_WM_ENABLE_DIFFUSION_ATTACKS": "1" if config.diffusion_attacks_enabled else "0",
    }
    drift = {
        name: {"expected": expected, "actual": os.environ[name]}
        for name, expected in expected_values.items()
        if name in os.environ and os.environ[name].strip() != expected
    }
    if drift:
        raise ValueError(f"正式方法环境变量与 configs/model_sd35.yaml 不一致: {drift}")
