"""语义条件局部切空间的分支投影算子。

该模块把候选方向生成、716维特征 Jacobian 线性化约束和风险预算投影组合为
一个可直接复用的核心方法入口。它只处理单个 latent 与单个载体分支, 不负责
prompt 调度、模型下载、攻击评测或结果文件写入。
"""

from __future__ import annotations

from typing import Any

from main.core.keyed_prg import KEYED_PRG_VERSION, keyed_prg_protocol_record
from main.methods.semantic import DifferentiableSemanticFeatureRuntime
from main.methods.subspace.jacobian_nullspace import (
    ExactJacobianLinearization,
    generate_keyed_candidate_directions,
    solve_jacobian_null_space,
)


def solve_semantic_branch_subspace(
    latent: Any,
    feature_runtime: DifferentiableSemanticFeatureRuntime,
    key_material: str,
    branch_name: str,
    axis_budget: Any,
    candidate_count: int,
    null_rank: int,
    joint_feature_linearization: ExactJacobianLinearization,
    preferred_directions: tuple[Any, ...] = (),
    maximum_relative_response_residual: float = 1e-4,
    minimum_projection_energy_retention: float = 0.01,
    cg_maximum_iterations: int = 64,
    cg_relative_tolerance: float = 1e-6,
    numerical_epsilon: float = 1e-12,
    maximum_qr_condition_number: float = 1e6,
    maximum_orthogonality_error: float = 1e-5,
    qr_reference_solve_protocol: str = (
        "right_upper_triangular_solve_without_explicit_inverse"
    ),
    prg_version: str = KEYED_PRG_VERSION,
) -> Any:
    """求解一个载体分支在当前 latent 点的一阶局部低响应方向。

    这里的“局部切空间”由当前 latent 点处716维联合特征的 Jacobian 决定,
    不表示已经学习全局流形。调用方可以把真实载体或注意力梯度作为
    ``preferred_directions`` 传入, 使求解结果同时满足密钥随机性与分支机制。
    """

    candidates = generate_keyed_candidate_directions(
        latent,
        key_material,
        branch_name,
        candidate_count,
        axis_budget=None,
        preferred_directions=preferred_directions,
        prg_version=prg_version,
    )
    result = solve_jacobian_null_space(
        latent=latent.float(),
        candidate_matrix=candidates,
        risk_budget=axis_budget,
        null_rank=null_rank,
        joint_feature_linearization=joint_feature_linearization,
        branch_name=branch_name,
        maximum_relative_response_residual=maximum_relative_response_residual,
        minimum_projection_energy_retention=minimum_projection_energy_retention,
        cg_maximum_iterations=cg_maximum_iterations,
        cg_relative_tolerance=cg_relative_tolerance,
        numerical_epsilon=numerical_epsilon,
        maximum_qr_condition_number=maximum_qr_condition_number,
        maximum_orthogonality_error=maximum_orthogonality_error,
        qr_reference_solve_protocol=qr_reference_solve_protocol,
    )
    result.metadata["preferred_direction_count"] = len(preferred_directions)
    result.metadata["preferred_direction_role"] = "carrier_or_attention_gradient"
    result.metadata.update(keyed_prg_protocol_record(prg_version))
    result.metadata.update(feature_runtime.feature_schema_record())
    if result.relative_response_residual > maximum_relative_response_residual:
        raise RuntimeError("完整 Jacobian Null Space 的相对响应残差超过正式门禁")
    return result
