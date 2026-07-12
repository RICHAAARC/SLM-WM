"""使用完整特征 Jacobian 求解风险支持的真实 Null Space。

通用工程写法: 通过 JVP 计算方向导数, 避免显式构造尺寸为“特征数乘以
latent 元素数”的完整 Jacobian。

项目特定写法: 风险预算作为对角算子 ``B`` 进入约束投影。对实际载体方向
``d`` 求解 ``(J B^2 J^T)y = J B d``, 再构造
``u = B d - B^2 J^T y``。实现只在无阻尼 PSD-CG 收敛且完整 Jacobian
逐列响应通过门禁时返回 Null Space 基底。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

from main.core.digest import (
    TENSOR_CONTENT_DIGEST_VERSION,
    build_stable_digest,
    tensor_content_sha256,
)
from main.core.keyed_prg import (
    KEYED_PRG_VERSION,
    build_keyed_gaussian_tensor,
)

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
    prg_version: str = KEYED_PRG_VERSION,
) -> Any:
    """生成可选风险加权且正交的密钥化种子方向矩阵。

    返回矩阵形状为 `[latent 元素数, candidate_count]`。每一列都是一个真实
    latent 种子。正式 Null Space 求解会把分支风险作为显式 ``B`` 传入, 因此
    正式路径在此处使用单位预算。调用方可以把固定盲检模板或真实注意力梯度
    置于密钥方向之前, 保证约束投影直接覆盖实际载体方向。
    """

    torch = _torch()
    if candidate_count <= 0:
        raise ValueError("candidate_count 必须为正数")
    element_count = latent.numel()
    if candidate_count > element_count:
        raise ValueError("candidate_count 不得超过 latent 元素数")
    preferred_values = tuple(preferred_directions)
    if len(preferred_values) > candidate_count:
        raise ValueError("preferred_directions 数量不得超过 candidate_count")
    random_column_count = candidate_count - len(preferred_values)
    random_matrix = (
        build_keyed_gaussian_tensor(
            (element_count, random_column_count),
            key_material,
            {
                "operator": "jacobian_candidate_directions",
                "branch_name": branch_name,
                "latent_shape": tuple(int(value) for value in latent.shape),
            },
            prg_version=prg_version,
        ).to(device=latent.device, dtype=torch.float32)
        if random_column_count > 0
        else torch.empty(
            (element_count, 0),
            device=latent.device,
            dtype=torch.float32,
        )
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
class ExactJacobianLinearization:
    """保存同一 latent 点上的完整 Jacobian JVP 与 VJP 算子。"""

    primal: Any
    jvp_function: Callable[[Any], Any]
    vjp_function: Callable[[Any], Any]
    linearization_mode: str
    latent_shape: tuple[int, ...]

    @property
    def output_width(self) -> int:
        """返回完整特征向量的宽度。"""

        return int(self.primal.numel())

    def apply(self, direction: Any) -> Any:
        """计算 ``J direction`` 并返回展平完整特征响应。"""

        return _flatten_feature(self.jvp_function(direction))

    def transpose_apply(self, cotangent: Any) -> Any:
        """计算 ``J^T cotangent`` 并恢复 latent 形状。"""

        resolved = _flatten_feature(cotangent)
        if resolved.numel() != self.output_width:
            raise ValueError("VJP 余切宽度必须等于完整特征宽度")
        return self.vjp_function(resolved).reshape(self.latent_shape)


def build_exact_jacobian_linearization(
    feature_function: TensorFeatureFunction,
    latent: Any,
) -> ExactJacobianLinearization:
    """构造可复用的完整 Jacobian JVP/VJP 线性算子。

    主路径同时使用 ``torch.func.linearize`` 与 ``torch.func.vjp``。若模型算子
    明确不支持 ``torch.func`` 的前向自动微分, 则回退到逐次重算的 autograd
    JVP/VJP。两条路径都使用真实自动微分, 不使用有限差分或低维特征草图。
    """

    torch = _torch()

    def flattened(candidate: Any) -> Any:
        """把完整语义与视觉特征统一为单个向量。"""

        return _flatten_feature(feature_function(candidate))

    def autograd_compatibility() -> ExactJacobianLinearization:
        """为不支持可复用线性化的算子提供精确重算路径。"""

        primal = flattened(latent)

        def autograd_jvp(direction: Any) -> Any:
            """通过 autograd 计算一个完整特征 JVP。"""

            _, tangent = torch.autograd.functional.jvp(
                flattened,
                (latent,),
                (direction,),
                create_graph=False,
                strict=True,
            )
            return tangent

        def autograd_vjp(cotangent: Any) -> Any:
            """通过 autograd 计算一个完整特征 VJP。"""

            _, gradient = torch.autograd.functional.vjp(
                flattened,
                latent,
                v=cotangent,
                create_graph=False,
                strict=True,
            )
            return gradient

        return ExactJacobianLinearization(
            primal=primal,
            jvp_function=autograd_jvp,
            vjp_function=autograd_vjp,
            linearization_mode="torch_autograd_exact_jvp_vjp_compatibility",
            latent_shape=tuple(int(value) for value in latent.shape),
        )

    try:
        primal, jvp_function = torch.func.linearize(flattened, latent)
        vjp_primal, vjp_closure = torch.func.vjp(flattened, latent)
        if primal.shape != vjp_primal.shape or not bool(torch.allclose(primal, vjp_primal)):
            raise RuntimeError("JVP 与 VJP 线性化的完整特征 primal 不一致")

        def reusable_vjp(cotangent: Any) -> Any:
            """调用冻结 primal 点的 VJP closure。"""

            return vjp_closure(cotangent)[0]

        return ExactJacobianLinearization(
            primal=primal,
            jvp_function=jvp_function,
            vjp_function=reusable_vjp,
            linearization_mode="torch_func_exact_jvp_vjp",
            latent_shape=tuple(int(value) for value in latent.shape),
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


@dataclass(frozen=True)
class PSDConjugateGradientResult:
    """保存无阻尼半正定共轭梯度求解状态。"""

    solution: Any
    converged: bool
    iteration_count: int
    absolute_residual: float
    relative_residual: float


def solve_psd_conjugate_gradient(
    operator: Callable[[Any], Any],
    right_hand_side: Any,
    *,
    maximum_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: float = 1e-10,
) -> PSDConjugateGradientResult:
    """无阻尼求解一致的半正定线性系统。

    该实现不添加会改变 Null Space 约束的 Tikhonov 阻尼。若曲率失效或在最大
    迭代数内未达到残差门禁, 调用方必须阻断当前科学算子。
    """

    torch = _torch()
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations 必须为正数")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance 必须位于 (0, 1)")
    if absolute_tolerance <= 0.0:
        raise ValueError("absolute_tolerance 必须为正数")

    rhs = right_hand_side.detach().float().reshape(-1)
    rhs_norm = float(torch.linalg.norm(rhs).item())
    solution = torch.zeros_like(rhs)
    if rhs_norm <= absolute_tolerance:
        return PSDConjugateGradientResult(
            solution=solution,
            converged=True,
            iteration_count=0,
            absolute_residual=rhs_norm,
            relative_residual=0.0,
        )

    residual = rhs.clone()
    search = residual.clone()
    residual_square = torch.dot(residual, residual)
    absolute_residual = rhs_norm
    relative_residual = 1.0
    iteration_count = 0
    converged = False
    for iteration_count in range(1, maximum_iterations + 1):
        image = operator(search).detach().float().reshape(-1)
        if image.shape != rhs.shape:
            raise ValueError("PSD-CG 算子输出宽度必须与右端项一致")
        curvature = torch.dot(search, image)
        if not bool(torch.isfinite(curvature)) or float(curvature.item()) <= 0.0:
            break
        step_size = residual_square / curvature
        solution = solution + step_size * search
        residual = residual - step_size * image
        next_residual_square = torch.dot(residual, residual)
        absolute_residual = float(torch.sqrt(next_residual_square.clamp_min(0.0)).item())
        relative_residual = absolute_residual / max(rhs_norm, absolute_tolerance)
        if relative_residual <= relative_tolerance or absolute_residual <= absolute_tolerance:
            converged = True
            break
        if not bool(torch.isfinite(next_residual_square)):
            break
        search = residual + (next_residual_square / residual_square) * search
        residual_square = next_residual_square

    return PSDConjugateGradientResult(
        solution=solution,
        converged=converged,
        iteration_count=iteration_count,
        absolute_residual=absolute_residual,
        relative_residual=relative_residual,
    )


_MAXIMUM_ORTHOGONALITY_ERROR = 1e-5


@dataclass(frozen=True)
class _ProjectedNullDirection:
    """保存一个风险支持种子方向的完整 Jacobian 约束投影。"""

    projected: Any
    reference_response_norm: float
    response_norm: float
    relative_response_residual: float
    projection_energy_retention: float
    cg_result: PSDConjugateGradientResult


@dataclass
class JacobianNullSpaceResult:
    """保存通过完整 Jacobian 约束投影求得的 latent Null Space 基底。"""

    branch_name: str
    candidate_matrix: Any
    response_matrix: Any
    latent_basis: Any
    column_response_norms: tuple[float, ...]
    column_relative_response_residuals: tuple[float, ...]
    projection_energy_retentions: tuple[float, ...]
    cg_iteration_counts: tuple[int, ...]
    cg_relative_residuals: tuple[float, ...]
    evaluated_direction_indices: tuple[int, ...]
    response_residual: float
    relative_response_residual: float
    orthogonality_error: float
    candidate_matrix_content_sha256: str
    risk_budget_content_sha256: str
    response_matrix_content_sha256: str
    latent_basis_content_sha256: str
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
            "evaluated_direction_count": (
                max(self.evaluated_direction_indices) + 1
                if self.evaluated_direction_indices
                else 0
            ),
            "evaluated_direction_indices": list(self.evaluated_direction_indices),
            "response_width": int(self.response_matrix.shape[0]),
            "column_response_norms": list(self.column_response_norms),
            "column_relative_response_residuals": list(
                self.column_relative_response_residuals
            ),
            "projection_energy_retentions": list(
                self.projection_energy_retentions
            ),
            "cg_iteration_counts": list(self.cg_iteration_counts),
            "cg_relative_residuals": list(self.cg_relative_residuals),
            "cg_converged": True,
            "response_residual": self.response_residual,
            "relative_response_residual": self.relative_response_residual,
            "orthogonality_error": self.orthogonality_error,
            "candidate_matrix_content_sha256": (
                self.candidate_matrix_content_sha256
            ),
            "risk_budget_content_sha256": self.risk_budget_content_sha256,
            "response_matrix_content_sha256": (
                self.response_matrix_content_sha256
            ),
            "latent_basis_content_sha256": self.latent_basis_content_sha256,
            "solver_digest": self.solver_digest,
            "metadata": self.metadata,
        }


def solve_jacobian_null_space(
    latent: Any,
    candidate_matrix: Any,
    risk_budget: Any,
    null_rank: int,
    joint_feature_linearization: ExactJacobianLinearization,
    branch_name: str = "lf_content",
    maximum_relative_response_residual: float = 1e-4,
    minimum_projection_energy_retention: float = 0.01,
    cg_maximum_iterations: int = 64,
    cg_relative_tolerance: float = 1e-6,
) -> JacobianNullSpaceResult:
    """通过完整 Jacobian JVP/VJP 与无阻尼 PSD-CG 求解 Null Space。

    候选矩阵只提供实际载体和密钥种子方向, 不限制 Jacobian 校正项所在空间。
    风险预算 ``B`` 显式进入 ``J B^2 J^T``。因此返回方向同时保留分支风险
    支持并满足完整特征一阶零响应, 而不是依赖“输出维数小于候选数”制造零值。
    """

    torch = _torch()
    if candidate_matrix.ndim != 2 or candidate_matrix.shape[0] != latent.numel():
        raise ValueError("candidate_matrix 必须具有 [latent 元素数, 候选数] 形状")
    candidate_count = int(candidate_matrix.shape[1])
    if not 0 < null_rank <= candidate_count:
        raise ValueError("null_rank 必须位于 [1, candidate_count]")
    if not 0.0 < maximum_relative_response_residual < 1.0:
        raise ValueError("maximum_relative_response_residual 必须位于 (0, 1)")
    if not 0.0 < minimum_projection_energy_retention <= 1.0:
        raise ValueError("minimum_projection_energy_retention 必须位于 (0, 1]")
    if tuple(joint_feature_linearization.latent_shape) != tuple(latent.shape):
        raise ValueError("完整 Jacobian 线性化的 latent 形状与求解点不一致")

    budget = _broadcast_axis_budget(latent, risk_budget).detach().float()
    if not bool(torch.isfinite(budget).all()) or bool((budget < 0.0).any()):
        raise ValueError("risk_budget 必须为有限非负值")
    if float(torch.linalg.norm(budget).item()) <= 1e-12:
        raise ValueError("risk_budget 不得全部为零")
    budget_square = budget.square()

    def project_direction(direction: Any) -> _ProjectedNullDirection:
        """执行一个方向的风险支持完整 Jacobian 约束投影。"""

        resolved_direction = direction.reshape(latent.shape).detach().float()
        routed_direction = budget * resolved_direction
        routed_norm = float(torch.linalg.norm(routed_direction).item())
        if routed_norm <= 1e-12:
            raise RuntimeError("风险预算后的 Null Space 种子方向能量为零")
        right_hand_side = joint_feature_linearization.apply(routed_direction).float()
        reference_norm = float(torch.linalg.norm(right_hand_side).item())

        def gram_operator(cotangent: Any) -> Any:
            """计算 ``J B^2 J^T cotangent``。"""

            latent_cotangent = joint_feature_linearization.transpose_apply(cotangent)
            return joint_feature_linearization.apply(
                budget_square * latent_cotangent.float()
            )

        cg_result = solve_psd_conjugate_gradient(
            gram_operator,
            right_hand_side,
            maximum_iterations=cg_maximum_iterations,
            relative_tolerance=cg_relative_tolerance,
        )
        if not cg_result.converged:
            raise RuntimeError("完整 Jacobian 无阻尼 PSD-CG 未在冻结迭代预算内收敛")
        correction = (
            budget_square
            * joint_feature_linearization.transpose_apply(cg_result.solution).float()
        ).detach()
        projected = (routed_direction - correction).detach()
        projected_norm = float(torch.linalg.norm(projected).item())
        energy_retention = projected_norm / max(routed_norm, 1e-12)
        response = joint_feature_linearization.apply(projected).float().detach()
        response_norm = float(torch.linalg.norm(response).item())
        relative_residual = response_norm / max(reference_norm, 1e-12)
        if not math.isfinite(relative_residual) or (
            relative_residual > maximum_relative_response_residual
        ):
            raise RuntimeError("约束投影方向的完整 Jacobian 相对响应残差超过正式门禁")
        if not math.isfinite(energy_retention) or (
            energy_retention < minimum_projection_energy_retention
        ):
            raise RuntimeError("约束投影方向的能量保留比例低于正式门禁")
        return _ProjectedNullDirection(
            projected=projected,
            reference_response_norm=reference_norm,
            response_norm=response_norm,
            relative_response_residual=relative_residual,
            projection_energy_retention=energy_retention,
            cg_result=cg_result,
        )

    accepted: list[_ProjectedNullDirection] = []
    accepted_indices: list[int] = []
    independent_columns: list[Any] = []
    for index in range(candidate_count):
        result = project_direction(
            candidate_matrix[:, index].reshape(latent.shape).to(dtype=latent.dtype)
        )
        orthogonal = result.projected.reshape(-1).float()
        for existing in independent_columns:
            orthogonal = orthogonal - existing * torch.dot(existing, orthogonal)
        orthogonal_norm = torch.linalg.norm(orthogonal)
        if float(orthogonal_norm.item()) <= 1e-7:
            continue
        independent_columns.append(orthogonal / orthogonal_norm)
        accepted.append(result)
        accepted_indices.append(index)
        if len(accepted) == null_rank:
            break
    if len(accepted) != null_rank:
        raise RuntimeError("完整 Jacobian 约束投影未形成指定秩的独立 Null Space")

    latent_basis_raw = torch.stack(
        tuple(item.projected.reshape(-1).float() for item in accepted),
        dim=1,
    )
    latent_basis, _ = torch.linalg.qr(latent_basis_raw, mode="reduced")
    latent_basis = latent_basis.detach()
    response_columns = tuple(
        joint_feature_linearization.apply(
            latent_basis[:, index].reshape(latent.shape)
        ).float().detach()
        for index in range(null_rank)
    )
    response_matrix = torch.stack(response_columns, dim=1).detach()
    response_reference = math.sqrt(
        sum(item.reference_response_norm**2 for item in accepted) / null_rank
    )
    column_response_norms = tuple(
        float(torch.linalg.norm(column).item()) for column in response_columns
    )
    column_relative_response_residuals = tuple(
        value / max(response_reference, 1e-12) for value in column_response_norms
    )
    if any(
        not math.isfinite(value) or value > maximum_relative_response_residual
        for value in column_relative_response_residuals
    ):
        raise RuntimeError("QR 后 Null Space 基底的完整 Jacobian 逐列残差超过正式门禁")
    response_residual = float(torch.linalg.norm(response_matrix).item())
    relative_response_residual = max(column_relative_response_residuals)
    identity = torch.eye(null_rank, device=latent_basis.device, dtype=latent_basis.dtype)
    orthogonality_error = float(
        torch.linalg.norm(latent_basis.transpose(0, 1) @ latent_basis - identity).item()
    )
    if not math.isfinite(orthogonality_error) or (
        orthogonality_error > _MAXIMUM_ORTHOGONALITY_ERROR
    ):
        raise RuntimeError("完整 Jacobian Null Space 基底正交误差超过正式门禁")
    projection_energy_retentions = tuple(
        item.projection_energy_retention for item in accepted
    )
    cg_iteration_counts = tuple(item.cg_result.iteration_count for item in accepted)
    cg_relative_residuals = tuple(item.cg_result.relative_residual for item in accepted)
    candidate_matrix_content_sha256 = tensor_content_sha256(candidate_matrix)
    risk_budget_content_sha256 = tensor_content_sha256(budget)
    response_matrix_content_sha256 = tensor_content_sha256(response_matrix)
    latent_basis_content_sha256 = tensor_content_sha256(latent_basis)
    digest_payload = {
        "branch_name": branch_name,
        "candidate_shape": tuple(int(value) for value in candidate_matrix.shape),
        "response_shape": tuple(int(value) for value in response_matrix.shape),
        "null_rank": null_rank,
        "evaluated_direction_indices": accepted_indices,
        "column_response_norms": [round(value, 12) for value in column_response_norms],
        "column_relative_response_residuals": [
            round(value, 12) for value in column_relative_response_residuals
        ],
        "projection_energy_retentions": [
            round(value, 12) for value in projection_energy_retentions
        ],
        "cg_iteration_counts": cg_iteration_counts,
        "cg_relative_residuals": [
            round(value, 12) for value in cg_relative_residuals
        ],
        "response_residual": round(response_residual, 12),
        "relative_response_residual": round(relative_response_residual, 12),
        "orthogonality_error": round(orthogonality_error, 12),
        "candidate_matrix_content_sha256": candidate_matrix_content_sha256,
        "risk_budget_content_sha256": risk_budget_content_sha256,
        "response_matrix_content_sha256": response_matrix_content_sha256,
        "latent_basis_content_sha256": latent_basis_content_sha256,
        "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
    }
    return JacobianNullSpaceResult(
        branch_name=branch_name,
        candidate_matrix=candidate_matrix,
        response_matrix=response_matrix,
        latent_basis=latent_basis,
        column_response_norms=column_response_norms,
        column_relative_response_residuals=column_relative_response_residuals,
        projection_energy_retentions=projection_energy_retentions,
        cg_iteration_counts=cg_iteration_counts,
        cg_relative_residuals=cg_relative_residuals,
        evaluated_direction_indices=tuple(accepted_indices),
        response_residual=response_residual,
        relative_response_residual=relative_response_residual,
        orthogonality_error=orthogonality_error,
        candidate_matrix_content_sha256=candidate_matrix_content_sha256,
        risk_budget_content_sha256=risk_budget_content_sha256,
        response_matrix_content_sha256=response_matrix_content_sha256,
        latent_basis_content_sha256=latent_basis_content_sha256,
        solver_digest=build_stable_digest(digest_payload),
        metadata={
            "jvp_mode": joint_feature_linearization.linearization_mode,
            "solver": "matrix_free_full_jacobian_psd_cg",
            "latent_basis_formula": "qr(Bd - B^2 J^T pinv(J B^2 J^T) J B d)",
            "full_feature_jvp": True,
            "full_feature_vjp": True,
            "cg_damping": 0.0,
            "cg_maximum_iterations": cg_maximum_iterations,
            "cg_relative_tolerance": cg_relative_tolerance,
            "maximum_relative_response_residual": (
                maximum_relative_response_residual
            ),
            "minimum_projection_energy_retention": (
                minimum_projection_energy_retention
            ),
            "maximum_orthogonality_error": _MAXIMUM_ORTHOGONALITY_ERROR,
            "risk_budget_operator": "explicit_diagonal_B",
            "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
        },
    )
