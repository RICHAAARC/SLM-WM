"""使用真实 Jacobian-Vector Product 求解语义条件低响应子空间。

通用工程写法: 通过 JVP 计算方向导数, 避免显式构造尺寸为“特征数乘以
latent 元素数”的完整 Jacobian。

项目特定写法: 候选方向先由分支风险预算限制, 再在语义特征和视觉质量特征的
联合响应矩阵上执行 SVD。小奇异值右向量先表示候选系数, 必须乘回候选矩阵才
能得到真正位于 latent 空间中的安全基底。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Sequence

from main.core.digest import build_stable_digest

TensorFeatureFunction = Callable[[Any], Any]


def _torch() -> Any:
    """延迟导入 PyTorch, 使仅使用治理工具时不强制加载运行时。"""

    import torch

    return torch


def _flatten_feature(value: Any) -> Any:
    """把单个 tensor 或 tensor 序列展平并连接。"""

    torch = _torch()
    if isinstance(value, dict):
        tensors = [_flatten_feature(value[key]) for key in sorted(value)]
        return torch.cat(tensors) if tensors else torch.empty(0)
    if isinstance(value, (tuple, list)):
        tensors = [_flatten_feature(item) for item in value]
        return torch.cat(tensors) if tensors else torch.empty(0)
    if not torch.is_tensor(value):
        raise TypeError("特征函数必须返回 tensor、tensor 序列或 tensor 字典")
    return value.reshape(-1)


def _stable_seed(key_material: str, branch_name: str) -> int:
    """从密钥材料生成 PyTorch 可用的确定性随机种子。"""

    digest = hashlib.sha256(f"{key_material}|{branch_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _broadcast_axis_budget(latent: Any, axis_budget: Sequence[float] | Any | None) -> Any:
    """把空间风险预算广播到完整 latent 形状。"""

    torch = _torch()
    if axis_budget is None:
        return torch.ones_like(latent, dtype=torch.float32)
    budget = torch.as_tensor(axis_budget, device=latent.device, dtype=torch.float32)
    if tuple(budget.shape) == tuple(latent.shape):
        return budget
    if budget.numel() == latent.numel():
        return budget.reshape(latent.shape)
    if latent.ndim >= 2 and budget.numel() == math.prod(latent.shape[-2:]):
        view_shape = (1,) * (latent.ndim - 2) + tuple(latent.shape[-2:])
        return budget.reshape(view_shape).expand(latent.shape)
    raise ValueError("axis_budget 无法广播到 latent 形状")


def generate_keyed_candidate_directions(
    latent: Any,
    key_material: str,
    branch_name: str,
    candidate_count: int,
    axis_budget: Sequence[float] | Any | None = None,
    preferred_directions: Sequence[Any] = (),
) -> Any:
    """生成风险加权且正交的密钥化候选方向矩阵。

    返回矩阵形状为 `[latent 元素数, candidate_count]`。每一列都是一个真实
    latent 方向, 后续 SVD 的右奇异向量只负责组合这些列。调用方可以把固定
    盲检模板或真实注意力梯度作为优先方向置于随机候选之前, 避免低秩随机
    子空间几乎完全丢失实际载体能量。
    """

    torch = _torch()
    if candidate_count <= 0:
        raise ValueError("candidate_count 必须为正数")
    element_count = latent.numel()
    if candidate_count > element_count:
        raise ValueError("candidate_count 不得超过 latent 元素数")
    generator_device = latent.device.type if latent.device.type in {"cpu", "cuda"} else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(_stable_seed(key_material, branch_name))
    preferred_values = tuple(preferred_directions)
    if len(preferred_values) > candidate_count:
        raise ValueError("preferred_directions 数量不得超过 candidate_count")
    random_matrix = torch.randn(
        element_count,
        candidate_count - len(preferred_values),
        generator=generator,
        device=latent.device,
        dtype=torch.float32,
    )
    budget = _broadcast_axis_budget(latent, axis_budget).reshape(-1, 1)
    preferred_columns = []
    for direction in preferred_values:
        direction_tensor = torch.as_tensor(direction, device=latent.device, dtype=torch.float32)
        if direction_tensor.numel() != element_count:
            raise ValueError("preferred_directions 必须与 latent 具有相同元素数")
        preferred_columns.append(direction_tensor.reshape(-1, 1))
    raw_matrix = (
        torch.cat((*preferred_columns, random_matrix), dim=1)
        if preferred_columns
        else random_matrix
    )
    weighted = raw_matrix * budget
    if int(torch.linalg.matrix_rank(weighted).item()) < candidate_count:
        raise ValueError("风险预算保留的坐标不足以形成指定数量的独立候选方向")
    candidate_matrix, _ = torch.linalg.qr(weighted, mode="reduced")
    return candidate_matrix


def exact_jvp(feature_function: TensorFeatureFunction, latent: Any, direction: Any) -> tuple[Any, Any]:
    """计算特征函数在给定 latent 和方向上的真实 JVP。

    这里调用 PyTorch autograd 的 JVP, 不使用相邻坐标差分或轨迹摘要替代。
    返回值依次为当前特征和方向导数, 二者都已展平。
    """

    torch = _torch()

    def flattened(candidate: Any) -> Any:
        return _flatten_feature(feature_function(candidate))

    primal, tangent = torch.autograd.functional.jvp(
        flattened,
        (latent,),
        (direction,),
        create_graph=False,
        strict=True,
    )
    return primal.reshape(-1), tangent.reshape(-1)


@dataclass
class ExactJVPLinearization:
    """保存同一 latent 点上的可复用精确 JVP 线性算子。"""

    primal: Any
    apply_function: Callable[[Any], Any]
    linearization_mode: str

    def apply(self, direction: Any) -> Any:
        """返回给定方向的展平精确 JVP。"""

        return _flatten_feature(self.apply_function(direction))


def build_exact_jvp_linearization(
    feature_function: TensorFeatureFunction,
    latent: Any,
) -> ExactJVPLinearization:
    """构造可跨候选方向和分支复用的真实 Jacobian 线性算子。

    `torch.func.linearize` 会在固定 primal 点保留精确线性化, 后续每个方向不再
    重复执行完整 VAE+CLIP primal 前向。若运行时算子不支持该接口, 则回退到
    `torch.autograd.functional.jvp`; 两条路径都计算真实 JVP, 不使用有限差分。
    """

    torch = _torch()

    def autograd_compatibility() -> ExactJVPLinearization:
        """为不支持可复用线性化的算子构造逐方向真实 JVP。"""

        primal = feature_function(latent)

        def autograd_jvp(direction: Any) -> Any:
            """计算一个候选方向的真实 JVP。"""

            _, tangent = torch.autograd.functional.jvp(
                feature_function,
                (latent,),
                (direction,),
                create_graph=False,
                strict=True,
            )
            return tangent

        return ExactJVPLinearization(
            primal=primal,
            apply_function=autograd_jvp,
            linearization_mode="torch_autograd_exact_jvp_compatibility",
        )

    try:
        primal, jvp_function = torch.func.linearize(feature_function, latent)
        return ExactJVPLinearization(
            primal=primal,
            apply_function=jvp_function,
            linearization_mode="torch_func_linearize_exact_jvp",
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        unsupported_forward_ad = any(
            marker in message
            for marker in (
                "forward-mode ad not implemented",
                "forward ad is not implemented",
                "does not support forward ad",
                "jvp is not implemented",
            )
        )
        if not unsupported_forward_ad:
            # 显存不足、形状错误和模型实现缺陷必须直接暴露, 不能伪装成兼容性回退。
            raise
        return autograd_compatibility()
    except NotImplementedError:
        return autograd_compatibility()


def _apply_feature_weights(response: Any, weights: Any | None, field_name: str) -> Any:
    """将特征空间权重应用到一列 JVP 响应。"""

    torch = _torch()
    if weights is None:
        return response
    weight_tensor = torch.as_tensor(weights, device=response.device, dtype=response.dtype).reshape(-1)
    if weight_tensor.numel() not in {1, response.numel()}:
        raise ValueError(f"{field_name} 长度必须为 1 或对应特征维度")
    return response * weight_tensor


@dataclass
class JacobianNullSpaceResult:
    """保存真实 JVP 响应矩阵和 SVD 求得的 latent 安全基底。"""

    branch_name: str
    candidate_matrix: Any
    response_matrix: Any
    coefficient_basis: Any
    latent_basis: Any
    singular_values: tuple[float, ...]
    selected_response_values: tuple[float, ...]
    response_residual: float
    relative_response_residual: float
    orthogonality_error: float
    solver_digest: str
    metadata: dict[str, Any]

    @property
    def basis_rank(self) -> int:
        """返回安全基底列数。"""

        return int(self.latent_basis.shape[1])

    def project(self, tensor: Any) -> Any:
        """把任意同形 latent tensor 正交投影到安全子空间。"""

        flat = tensor.reshape(-1).to(device=self.latent_basis.device, dtype=self.latent_basis.dtype)
        projected = self.latent_basis @ (self.latent_basis.transpose(0, 1) @ flat)
        return projected.reshape(tensor.shape).to(dtype=tensor.dtype)

    def to_record(self) -> dict[str, Any]:
        """返回不包含大型矩阵的受治理摘要记录。"""

        return {
            "branch_name": self.branch_name,
            "basis_rank": self.basis_rank,
            "candidate_count": int(self.candidate_matrix.shape[1]),
            "response_width": int(self.response_matrix.shape[0]),
            "singular_values": list(self.singular_values),
            "selected_response_values": list(self.selected_response_values),
            "response_residual": self.response_residual,
            "relative_response_residual": self.relative_response_residual,
            "orthogonality_error": self.orthogonality_error,
            "solver_digest": self.solver_digest,
            "metadata": self.metadata,
        }


def solve_jacobian_null_space(
    latent: Any,
    semantic_feature_function: TensorFeatureFunction,
    visual_feature_function: TensorFeatureFunction,
    candidate_matrix: Any,
    null_rank: int,
    semantic_weights: Any | None = None,
    visual_weights: Any | None = None,
    visual_response_weight: float = 1.0,
    branch_name: str = "lf_content",
    joint_feature_function: TensorFeatureFunction | None = None,
    joint_feature_linearization: ExactJVPLinearization | None = None,
) -> JacobianNullSpaceResult:
    """通过真实 JVP 和 SVD 求解语义条件 Jacobian 低响应子空间。"""

    torch = _torch()
    if candidate_matrix.ndim != 2 or candidate_matrix.shape[0] != latent.numel():
        raise ValueError("candidate_matrix 必须具有 [latent 元素数, 候选数] 形状")
    candidate_count = int(candidate_matrix.shape[1])
    if not 0 < null_rank <= candidate_count:
        raise ValueError("null_rank 必须位于 [1, candidate_count]")
    if visual_response_weight < 0.0:
        raise ValueError("visual_response_weight 不得为负")

    joint_semantic_width: int | None = None
    joint_visual_width: int | None = None
    if joint_feature_linearization is not None and joint_feature_function is None:
        raise ValueError("joint_feature_linearization 要求同时声明 joint_feature_function")
    if joint_feature_function is not None:
        joint_primal = (
            joint_feature_linearization.primal
            if joint_feature_linearization is not None
            else joint_feature_function(latent)
        )
        if not isinstance(joint_primal, (tuple, list)) or len(joint_primal) != 2:
            raise ValueError("joint_feature_function 必须依次返回语义特征和视觉特征")
        joint_semantic_width = _flatten_feature(joint_primal[0]).numel()
        joint_visual_width = _flatten_feature(joint_primal[1]).numel()

    response_columns = []
    for index in range(candidate_count):
        direction = candidate_matrix[:, index].reshape(latent.shape).to(dtype=latent.dtype)
        if joint_feature_function is None:
            _, semantic_response = exact_jvp(semantic_feature_function, latent, direction)
            _, visual_response = exact_jvp(visual_feature_function, latent, direction)
        else:
            joint_response = (
                joint_feature_linearization.apply(direction)
                if joint_feature_linearization is not None
                else exact_jvp(joint_feature_function, latent, direction)[1]
            )
            semantic_width = int(joint_semantic_width or 0)
            visual_width = int(joint_visual_width or 0)
            if joint_response.numel() != semantic_width + visual_width:
                raise ValueError("joint_feature_function 输出宽度必须等于语义与视觉特征宽度之和")
            semantic_response = joint_response[:semantic_width]
            visual_response = joint_response[semantic_width:]
        semantic_response = _apply_feature_weights(semantic_response, semantic_weights, "semantic_weights")
        visual_response = _apply_feature_weights(visual_response, visual_weights, "visual_weights")
        response_columns.append(
            torch.cat(
                (
                    semantic_response.float(),
                    math.sqrt(visual_response_weight) * visual_response.float(),
                )
            )
        )
    response_matrix = torch.stack(response_columns, dim=1)
    _, singular_values_tensor, right_vectors_transposed = torch.linalg.svd(
        response_matrix,
        full_matrices=True,
    )
    padded_responses = torch.zeros(candidate_count, device=response_matrix.device, dtype=response_matrix.dtype)
    padded_responses[: singular_values_tensor.numel()] = singular_values_tensor
    selected_indices = torch.argsort(padded_responses)[:null_rank]
    coefficient_basis = right_vectors_transposed.transpose(0, 1).index_select(1, selected_indices)
    latent_basis_raw = candidate_matrix.float() @ coefficient_basis.float()
    latent_basis, _ = torch.linalg.qr(latent_basis_raw, mode="reduced")
    response_residual = float(torch.linalg.norm(response_matrix @ coefficient_basis).item())
    response_reference = float(torch.linalg.norm(response_matrix).item())
    relative_response_residual = (
        response_residual / math.sqrt(null_rank)
    ) / max(response_reference / math.sqrt(candidate_count), 1e-12)
    identity = torch.eye(null_rank, device=latent_basis.device, dtype=latent_basis.dtype)
    orthogonality_error = float(
        torch.linalg.norm(latent_basis.transpose(0, 1) @ latent_basis - identity).item()
    )
    singular_values = tuple(float(value) for value in singular_values_tensor.detach().cpu().tolist())
    selected_response_values = tuple(float(padded_responses[index].item()) for index in selected_indices)
    digest_payload = {
        "branch_name": branch_name,
        "candidate_shape": tuple(int(value) for value in candidate_matrix.shape),
        "response_shape": tuple(int(value) for value in response_matrix.shape),
        "null_rank": null_rank,
        "singular_values": [round(value, 12) for value in singular_values],
        "selected_response_values": [round(value, 12) for value in selected_response_values],
        "response_residual": round(response_residual, 12),
        "relative_response_residual": round(relative_response_residual, 12),
        "orthogonality_error": round(orthogonality_error, 12),
    }
    return JacobianNullSpaceResult(
        branch_name=branch_name,
        candidate_matrix=candidate_matrix,
        response_matrix=response_matrix,
        coefficient_basis=coefficient_basis,
        latent_basis=latent_basis,
        singular_values=singular_values,
        selected_response_values=selected_response_values,
        response_residual=response_residual,
        relative_response_residual=relative_response_residual,
        orthogonality_error=orthogonality_error,
        solver_digest=build_stable_digest(digest_payload),
        metadata={
            "jvp_mode": (
                joint_feature_linearization.linearization_mode
                if joint_feature_linearization is not None
                else "torch_autograd_exact_jvp"
            ),
            "solver": "weighted_response_svd",
            "latent_basis_formula": "orth(candidate_matrix @ small_right_singular_vectors)",
            "route_applied_before_svd": True,
            "joint_feature_jvp": joint_feature_function is not None,
        },
    )
