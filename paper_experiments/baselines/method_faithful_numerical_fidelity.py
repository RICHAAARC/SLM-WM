"""比较 SD3.5 方法忠实适配器与官方源码关键算子的数值输出.

该模块从来源登记表指定的 Git commit 读取不可变源码 blob, 而不是读取可能已被
运行时补丁修改的工作树文件. Tree-Ring 与 Shallow Diffuse 的 mask、载体、注入和
检测分数直接执行官方函数; Gaussian Shading 则核验 ChaCha20、条件 Gaussian 符号
映射和 block voting. 所有比较均使用小型确定性 CPU Tensor, 不生成论文效果结果.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
from statistics import NormalDist
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from main.core.digest import build_stable_digest


METHOD_FAITHFUL_NUMERICAL_FIDELITY_SCHEMA = (
    "method_faithful_numerical_fidelity_v1"
)
METHOD_FAITHFUL_NUMERICAL_FIDELITY_BASELINE_IDS = (
    "tree_ring",
    "gaussian_shading",
    "shallow_diffuse",
)
DEFAULT_ABSOLUTE_TOLERANCE = 1e-6
_EXPECTED_OPERATOR_IDS = {
    "tree_ring": (
        "tree_ring.mask",
        "tree_ring.ring_key",
        "tree_ring.fourier_injection",
        "tree_ring.negative_l1_detection_score",
    ),
    "gaussian_shading": (
        "gaussian_shading.chacha20_ietf_cipher",
        "gaussian_shading.block_voting",
        "gaussian_shading.conditional_gaussian_sign_mapping",
    ),
    "shallow_diffuse": (
        "shallow_diffuse.ring_mask",
        "shallow_diffuse.complex_random_patch",
        "shallow_diffuse.fourier_injection",
        "shallow_diffuse.negative_l1_detection_score",
        "shallow_diffuse.edit_timestep_floor",
    ),
}
_REFERENCE_MODES = {
    "tree_ring": "executed_official_commit_operator_equivalence",
    "gaussian_shading": (
        "official_source_bound_rfc8439_and_operator_equivalence"
    ),
    "shallow_diffuse": "executed_official_commit_operator_equivalence",
}


class MethodFaithfulNumericalFidelityError(ValueError):
    """表示官方源码身份或任一关键算子数值比较未通过."""


@dataclass(frozen=True)
class _SourceSpec:
    """登记一个 baseline 的官方源码与本地适配器路径."""

    baseline_id: str
    source_dir: str
    source_file: str
    adapter_file: str
    definition_names: tuple[str, ...]


_SOURCE_SPECS = {
    "tree_ring": _SourceSpec(
        baseline_id="tree_ring",
        source_dir="external_baseline/primary/tree_ring/source",
        source_file="optim_utils.py",
        adapter_file=(
            "external_baseline/primary/tree_ring/adapter/"
            "method_faithful_sd35.py"
        ),
        definition_names=(
            "set_random_seed",
            "circle_mask",
            "get_watermarking_mask",
            "get_watermarking_pattern",
            "inject_watermark",
            "eval_watermark",
        ),
    ),
    "gaussian_shading": _SourceSpec(
        baseline_id="gaussian_shading",
        source_dir="external_baseline/primary/gaussian_shading/source",
        source_file="watermark.py",
        adapter_file=(
            "external_baseline/primary/gaussian_shading/adapter/"
            "method_faithful_sd35.py"
        ),
        definition_names=("Gaussian_Shading_chacha",),
    ),
    "shallow_diffuse": _SourceSpec(
        baseline_id="shallow_diffuse",
        source_dir="external_baseline/primary/shallow_diffuse/source",
        source_file="optim_utils.py",
        adapter_file=(
            "external_baseline/primary/shallow_diffuse/adapter/"
            "method_faithful_sd35.py"
        ),
        definition_names=(
            "set_random_seed",
            "circle_mask",
            "get_watermarking_mask",
            "get_watermarking_pattern",
            "inject_watermark",
            "eval_watermark_single",
        ),
    ),
}


def _file_sha256(path: Path) -> str:
    """流式计算普通文件 SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_text(source_dir: Path, arguments: Sequence[str]) -> str:
    """执行只读 Git 命令并返回 UTF-8 文本."""

    completed = subprocess.run(
        ["git", "-C", str(source_dir), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if completed.returncode != 0:
        raise MethodFaithfulNumericalFidelityError(
            "无法读取登记的官方 Git 源码: " + completed.stderr.strip()
        )
    return completed.stdout


def _source_registry_item(root_path: Path, baseline_id: str) -> dict[str, Any]:
    """读取唯一 baseline 来源登记记录."""

    registry_path = root_path / "external_baseline" / "source_registry.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    rows = [
        dict(row)
        for row in payload.get("baseline_sources", ())
        if isinstance(row, Mapping) and row.get("baseline_id") == baseline_id
    ]
    if len(rows) != 1:
        raise MethodFaithfulNumericalFidelityError(
            f"{baseline_id} 必须且只能存在一条官方来源登记"
        )
    return rows[0]


def _official_source_blob(
    root_path: Path,
    spec: _SourceSpec,
) -> tuple[str, str, str]:
    """读取登记 commit 的源码 blob, 并核验本地仓库 HEAD 身份."""

    registry = _source_registry_item(root_path, spec.baseline_id)
    expected_commit = str(registry.get("official_repository_commit", ""))
    if len(expected_commit) != 40 or any(
        character not in "0123456789abcdef" for character in expected_commit
    ):
        raise MethodFaithfulNumericalFidelityError(
            f"{spec.baseline_id} 官方 commit 不是完整小写 Git SHA"
        )
    source_dir = (root_path / spec.source_dir).resolve()
    actual_commit = _git_text(source_dir, ("rev-parse", "HEAD")).strip()
    if actual_commit != expected_commit:
        raise MethodFaithfulNumericalFidelityError(
            f"{spec.baseline_id} 本地官方源码 HEAD 与登记 commit 不一致"
        )
    source_text = _git_text(
        source_dir,
        ("show", f"{expected_commit}:{spec.source_file}"),
    )
    source_blob_sha256 = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    return expected_commit, source_text, source_blob_sha256


def _compile_official_definitions(
    source_text: str,
    names: Sequence[str],
    namespace: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """只编译已登记函数或类, 避免执行官方模块的导入副作用."""

    tree = ast.parse(source_text)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name in set(names)
    ]
    selected_names = {node.name for node in selected}
    if selected_names != set(names):
        missing = sorted(set(names) - selected_names)
        raise MethodFaithfulNumericalFidelityError(
            "官方源码缺少已登记关键算子: " + ",".join(missing)
        )
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    resolved_namespace = dict(namespace)
    exec(compile(module, "<official_commit_blob>", "exec"), resolved_namespace)
    ast_digest = build_stable_digest(
        [
            ast.dump(node, annotate_fields=True, include_attributes=False)
            for node in selected
        ]
    )
    return resolved_namespace, ast_digest


def _tensor_digest(value: Any) -> str:
    """绑定 Tensor 的 dtype、shape 与原始连续字节."""

    tensor = value.detach().cpu().contiguous()
    raw = tensor.numpy().tobytes(order="C")
    return build_stable_digest(
        {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "content_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _numeric_record(
    operator_id: str,
    reference: Any,
    adapter: Any,
    *,
    tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    exact: bool = False,
    reference_origin: str = "executed_official_commit_operator",
) -> dict[str, Any]:
    """比较两个同形 Tensor 并生成可复算门禁记录."""

    import torch

    reference_tensor = torch.as_tensor(reference).detach().cpu()
    adapter_tensor = torch.as_tensor(adapter).detach().cpu()
    shape_ready = tuple(reference_tensor.shape) == tuple(adapter_tensor.shape)
    if shape_ready and reference_tensor.numel():
        if reference_tensor.dtype == torch.bool or adapter_tensor.dtype == torch.bool:
            max_abs_error = float(
                (reference_tensor != adapter_tensor).to(dtype=torch.float32).max().item()
            )
        else:
            max_abs_error = float(
                torch.abs(reference_tensor - adapter_tensor).max().item()
            )
    else:
        max_abs_error = math.inf
    exact_match = bool(shape_ready and torch.equal(reference_tensor, adapter_tensor))
    ready = exact_match if exact else bool(
        shape_ready and math.isfinite(max_abs_error) and max_abs_error <= tolerance
    )
    payload = {
        "operator_id": operator_id,
        "reference_origin": reference_origin,
        "comparison_mode": "exact_tensor" if exact else "absolute_error_tensor",
        "reference_dtype": str(reference_tensor.dtype),
        "adapter_dtype": str(adapter_tensor.dtype),
        "reference_shape": list(reference_tensor.shape),
        "adapter_shape": list(adapter_tensor.shape),
        "element_count": int(reference_tensor.numel()),
        "absolute_tolerance": 0.0 if exact else float(tolerance),
        "max_absolute_error": max_abs_error,
        "exact_match": exact_match,
        "reference_value_digest": _tensor_digest(reference_tensor),
        "adapter_value_digest": _tensor_digest(adapter_tensor),
        "numerical_fidelity_ready": ready,
    }
    payload["comparison_record_digest"] = build_stable_digest(payload)
    return payload


def _scalar_record(
    operator_id: str,
    reference: float,
    adapter: float,
    *,
    tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    reference_origin: str = "executed_official_commit_operator",
) -> dict[str, Any]:
    """比较两个有限标量并生成门禁记录."""

    error = abs(float(reference) - float(adapter))
    ready = bool(
        math.isfinite(float(reference))
        and math.isfinite(float(adapter))
        and error <= tolerance
    )
    payload = {
        "operator_id": operator_id,
        "reference_origin": reference_origin,
        "comparison_mode": "absolute_error_scalar",
        "reference_value": float(reference),
        "adapter_value": float(adapter),
        "absolute_tolerance": float(tolerance),
        "max_absolute_error": error,
        "numerical_fidelity_ready": ready,
    }
    payload["comparison_record_digest"] = build_stable_digest(payload)
    return payload


def _tree_ring_records(official: Mapping[str, Any]) -> list[dict[str, Any]]:
    """比较 Tree-Ring 的 mask、载体、注入和 L1 检测分数."""

    import torch

    from external_baseline.primary.tree_ring.adapter import method_faithful_sd35

    shape = (1, 16, 8, 8)
    args = SimpleNamespace(
        w_mask_shape="circle",
        w_radius=2,
        w_channel=3,
        w_seed=271828,
        w_pattern="ring",
        w_injection="complex",
        w_measurement="l1_complex",
    )
    base = torch.linspace(-1.0, 1.0, math.prod(shape), dtype=torch.float32).reshape(shape)
    official_mask = official["get_watermarking_mask"](base, args, "cpu")
    adapter_mask = method_faithful_sd35.build_watermark_mask(
        shape,
        channel=args.w_channel,
        radius=args.w_radius,
        device="cpu",
    )
    official_key = official["get_watermarking_pattern"](
        None,
        args,
        "cpu",
        shape=shape,
    )
    adapter_key = method_faithful_sd35.build_watermark_key(
        shape,
        pattern=args.w_pattern,
        radius=args.w_radius,
        generator=torch.Generator(device="cpu").manual_seed(args.w_seed),
        device="cpu",
    )
    official_injected = official["inject_watermark"](
        base.clone(),
        official_mask,
        official_key,
        args,
    )
    adapter_injected = method_faithful_sd35.inject_watermark(
        base.clone(),
        adapter_mask,
        adapter_key,
    )
    _, official_distance = official["eval_watermark"](
        base,
        official_injected,
        official_mask,
        official_key,
        args,
    )
    adapter_score = method_faithful_sd35.score_latents(
        adapter_injected,
        adapter_mask,
        adapter_key,
    )
    return [
        _numeric_record("tree_ring.mask", official_mask, adapter_mask, exact=True),
        _numeric_record("tree_ring.ring_key", official_key, adapter_key),
        _numeric_record(
            "tree_ring.fourier_injection",
            official_injected,
            adapter_injected,
        ),
        _scalar_record(
            "tree_ring.negative_l1_detection_score",
            -float(official_distance),
            adapter_score,
        ),
    ]


def _gaussian_shading_records(
    official: Mapping[str, Any],
    source_text: str,
) -> list[dict[str, Any]]:
    """比较 Gaussian Shading 的 cipher、条件采样和 block voting."""

    import torch

    from external_baseline.primary.gaussian_shading.adapter import (
        method_faithful_sd35,
    )

    key = bytes(range(32))
    nonce = bytes.fromhex("000000090000004a00000000")
    plaintext = bytes(64)
    expected_cipher = bytes.fromhex(
        "10f1e7e4d13b5915500fdd1fa32071c4"
        "c7d1f4c733c068030422aa9ac3d46c4e"
        "d2826446079faa0914c2d705d98b02a2"
        "b5129cd1de164eb9cbd083e8a2503c4e"
    )
    adapter_cipher = method_faithful_sd35.chacha20_encrypt(
        plaintext,
        key=key,
        nonce=nonce,
        initial_counter=1,
    )
    cipher_record = _numeric_record(
        "gaussian_shading.chacha20_ietf_cipher",
        torch.tensor(list(expected_cipher), dtype=torch.uint8),
        torch.tensor(list(adapter_cipher), dtype=torch.uint8),
        exact=True,
        reference_origin=(
            "rfc8439_known_answer_bound_to_official_chacha20_source_contract"
        ),
    )
    tree = ast.parse(source_text)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Gaussian_Shading_chacha"
    )
    method_names = {
        node.name for node in class_node.body if isinstance(node, ast.FunctionDef)
    }
    source_cipher_contract_ready = {
        "stream_key_encrypt",
        "stream_key_decrypt",
        "truncSampling",
        "diffusion_inverse",
        "eval_watermark",
    } <= method_names and "ChaCha20.new" in source_text
    cipher_record["official_source_cipher_contract_ready"] = (
        source_cipher_contract_ready
    )
    cipher_record["numerical_fidelity_ready"] = bool(
        cipher_record["numerical_fidelity_ready"]
        and source_cipher_contract_ready
    )
    cipher_record["comparison_record_digest"] = build_stable_digest(
        {
            key_name: value
            for key_name, value in cipher_record.items()
            if key_name != "comparison_record_digest"
        }
    )

    # 官方实现把 latent 空间固定为 4x64x64, 因而该向量必须保留原论文形状.
    shape = (1, 4, 64, 64)
    watermark = method_faithful_sd35.GaussianShadingWatermark(
        latent_shape=shape,
        channel_copy=1,
        hw_copy=2,
        generator=torch.Generator(device="cpu").manual_seed(314159),
        device="cpu",
    )
    strict_latents = watermark.create_strict_paired_latents(torch.ones(shape))
    adapter_vote = watermark.decode_recovered_watermark(strict_latents)
    official_class = official["Gaussian_Shading_chacha"]
    official_instance = official_class.__new__(official_class)
    official_instance.ch = watermark.channel_copy
    official_instance.hw = watermark.hw_copy
    official_instance.threshold = watermark.vote_threshold
    official_vote = official_instance.diffusion_inverse(
        watermark.expanded_watermark()
    ).unsqueeze(0)

    bit_values = [index % 2 for index in range(128)]
    probabilities = [(index + 0.5) / len(bit_values) for index in range(len(bit_values))]
    normal = NormalDist()
    magnitudes = []
    official_quantiles = []
    for bit, probability in zip(bit_values, probabilities):
        if bit:
            quantile = normal.inv_cdf(0.5 + 0.5 * probability)
        else:
            quantile = normal.inv_cdf(0.5 * probability)
        official_quantiles.append(quantile)
        magnitudes.append(abs(quantile))
    repeated_bits = (
        bit_values * (math.prod(shape) // len(bit_values))
    )[: math.prod(shape)]
    repeated_reference = (
        official_quantiles * (math.prod(shape) // len(official_quantiles))
    )[: math.prod(shape)]
    watermark.encrypted_message = torch.tensor(
        repeated_bits,
        dtype=torch.int64,
    ).reshape(shape)
    adapter_quantiles = watermark.create_strict_paired_latents(
        torch.tensor(
            [abs(value) for value in repeated_reference],
            dtype=torch.float32,
        ).reshape(shape)
    )
    return [
        cipher_record,
        _numeric_record(
            "gaussian_shading.block_voting",
            official_vote,
            adapter_vote,
            exact=True,
        ),
        _numeric_record(
            "gaussian_shading.conditional_gaussian_sign_mapping",
            torch.tensor(repeated_reference, dtype=torch.float32).reshape(shape),
            adapter_quantiles,
            exact=True,
            reference_origin="official_conditional_gaussian_definition",
        ),
    ]


def _assignment_digest(
    source_text: str,
    *,
    target_name: str,
    required_text: str,
) -> str:
    """绑定官方入口中的精确赋值公式 AST."""

    if required_text not in source_text:
        raise MethodFaithfulNumericalFidelityError(
            f"官方源码缺少登记公式: {required_text}"
        )
    tree = ast.parse(source_text)
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == target_name
            for target in node.targets
        ):
            candidates.append(node)
    if not candidates:
        raise MethodFaithfulNumericalFidelityError(
            f"官方源码缺少 {target_name} 赋值"
        )
    return build_stable_digest(
        [
            ast.dump(node, annotate_fields=True, include_attributes=False)
            for node in candidates
        ]
    )


def _shallow_diffuse_records(
    official: Mapping[str, Any],
    run_source_text: str,
) -> tuple[list[dict[str, Any]], str]:
    """比较 Shallow Diffuse 的 mask、载体、注入、分数和 edit timestep."""

    import torch

    from external_baseline.primary.shallow_diffuse.adapter import (
        method_faithful_sd35,
    )

    shape = (1, 16, 8, 8)
    radius = 3
    inner_radius = 1
    channel = 5
    seed = 161803
    base = torch.linspace(-0.75, 0.75, math.prod(shape), dtype=torch.float32).reshape(shape)
    official_mask = official["get_watermarking_mask"](
        init_latents_w=base,
        w_mask_shape="ring",
        w_radius=radius,
        w_radius2=inner_radius,
        w_channel=channel,
        device="cpu",
    )
    adapter_mask = method_faithful_sd35.build_watermark_mask(
        shape,
        mask_shape="ring",
        radius=radius,
        inner_radius=inner_radius,
        channel=channel,
        device="cpu",
    )
    args = SimpleNamespace(
        w_seed=seed,
        w_pattern="complex_rand",
        w_radius=radius,
    )
    official_patch = official["get_watermarking_pattern"](
        None,
        args,
        "cpu",
        shape=shape,
    )
    adapter_patch = method_faithful_sd35.build_watermark_patch(
        shape,
        pattern=args.w_pattern,
        radius=radius,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        device="cpu",
    )
    official_injected = official["inject_watermark"](
        base.clone(),
        official_mask,
        official_patch,
        "complex",
    )
    adapter_injected = method_faithful_sd35.inject_watermark(
        base.clone(),
        adapter_mask,
        adapter_patch,
        injection="complex",
    )
    official_score = -float(
        official["eval_watermark_single"](
            official_injected,
            official_mask,
            official_patch,
            "l1_complex",
            channel,
        )["mask_l1diff_mean"]
    )
    adapter_score = method_faithful_sd35.score_latents(
        adapter_injected,
        mask=adapter_mask,
        patch=adapter_patch,
        measurement="l1_complex",
    )
    edit_timestep, edit_schedule_index = (
        method_faithful_sd35.resolve_shallow_diffuse_edit_timestep(20, 0.2)
    )
    edit_formula_digest = _assignment_digest(
        run_source_text,
        target_name="edit_timestep",
        required_text="edit_timestep = int(edit_t * args.num_inference_steps)",
    )
    edit_record = _numeric_record(
        "shallow_diffuse.edit_timestep_floor",
        torch.tensor([4, 16], dtype=torch.int64),
        torch.tensor([edit_timestep, edit_schedule_index], dtype=torch.int64),
        exact=True,
    )
    edit_record["official_edit_timestep_formula_ast_digest"] = edit_formula_digest
    edit_record["comparison_record_digest"] = build_stable_digest(
        {
            key_name: value
            for key_name, value in edit_record.items()
            if key_name != "comparison_record_digest"
        }
    )
    return (
        [
            _numeric_record(
                "shallow_diffuse.ring_mask",
                official_mask,
                adapter_mask,
                exact=True,
            ),
            _numeric_record(
                "shallow_diffuse.complex_random_patch",
                official_patch,
                adapter_patch,
            ),
            _numeric_record(
                "shallow_diffuse.fourier_injection",
                official_injected,
                adapter_injected,
            ),
            _scalar_record(
                "shallow_diffuse.negative_l1_detection_score",
                official_score,
                adapter_score,
            ),
            edit_record,
        ],
        edit_formula_digest,
    )


def build_method_faithful_numerical_fidelity_report(
    root: str | Path,
    baseline_id: str,
) -> dict[str, Any]:
    """构造一个 baseline 的确定性关键算子数值忠实度报告."""

    normalized_id = str(baseline_id).strip().lower()
    if normalized_id not in _SOURCE_SPECS:
        raise MethodFaithfulNumericalFidelityError(
            f"未登记的数值忠实度 baseline: {baseline_id}"
        )
    root_path = Path(root).resolve()
    spec = _SOURCE_SPECS[normalized_id]
    commit, source_text, source_blob_sha256 = _official_source_blob(
        root_path,
        spec,
    )
    import numpy as np
    import torch

    namespace, ast_digest = _compile_official_definitions(
        source_text,
        spec.definition_names,
        {
            "copy": __import__("copy"),
            "np": np,
            "random": random,
            "torch": torch,
        },
    )
    extra_source_digests: dict[str, str] = {}
    if normalized_id == "tree_ring":
        records = _tree_ring_records(namespace)
    elif normalized_id == "gaussian_shading":
        records = _gaussian_shading_records(namespace, source_text)
    else:
        source_dir = (root_path / spec.source_dir).resolve()
        run_source_text = _git_text(
            source_dir,
            ("show", f"{commit}:run_shallow_diffuse_t2i.py"),
        )
        records, edit_formula_digest = _shallow_diffuse_records(
            namespace,
            run_source_text,
        )
        extra_source_digests = {
            "official_entrypoint_blob_sha256": hashlib.sha256(
                run_source_text.encode("utf-8")
            ).hexdigest(),
            "official_edit_timestep_formula_ast_digest": edit_formula_digest,
        }
    adapter_path = (root_path / spec.adapter_file).resolve()
    if not adapter_path.is_file():
        raise MethodFaithfulNumericalFidelityError(
            f"适配器文件不存在: {spec.adapter_file}"
        )
    operator_ids = [str(record["operator_id"]) for record in records]
    payload = {
        "report_schema": METHOD_FAITHFUL_NUMERICAL_FIDELITY_SCHEMA,
        "baseline_id": normalized_id,
        "numerical_fidelity_reference_mode": _REFERENCE_MODES[normalized_id],
        "official_repository_commit": commit,
        "official_source_read_mode": "immutable_git_commit_blob",
        "official_source_file": f"{spec.source_dir}/{spec.source_file}",
        "official_source_blob_sha256": source_blob_sha256,
        "official_operator_ast_digest": ast_digest,
        **extra_source_digests,
        "adapter_file": spec.adapter_file,
        "adapter_file_sha256": _file_sha256(adapter_path),
        "execution_device": "cpu",
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "operator_ids": operator_ids,
        "operator_record_count": len(records),
        "operator_records": records,
        "operator_records_digest": build_stable_digest(records),
        "method_faithful_numerical_fidelity_ready": bool(records)
        and len(operator_ids) == len(set(operator_ids))
        and all(
            record.get("numerical_fidelity_ready") is True
            for record in records
        ),
        "supports_paper_claim": False,
    }
    payload["numerical_fidelity_report_digest"] = build_stable_digest(payload)
    if payload["method_faithful_numerical_fidelity_ready"] is not True:
        raise MethodFaithfulNumericalFidelityError(
            f"{normalized_id} 关键算子数值忠实度未通过"
        )
    return payload


def _is_sha256(value: Any) -> bool:
    """判断字段是否为规范小写 SHA-256."""

    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _finite_number(value: Any) -> float | None:
    """读取有限数值, 布尔值和非法文本返回 None."""

    if isinstance(value, bool):
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def _comparison_record_semantics_ready(record: Mapping[str, Any]) -> bool:
    """从原始数值字段重建单算子判定, 不信任已声明 ready."""

    payload = dict(record)
    digest = str(payload.pop("comparison_record_digest", ""))
    mode = str(payload.get("comparison_mode", ""))
    tolerance = _finite_number(payload.get("absolute_tolerance"))
    max_error = _finite_number(payload.get("max_absolute_error"))
    common_ready = bool(
        str(payload.get("operator_id", ""))
        and str(payload.get("reference_origin", ""))
        and tolerance is not None
        and tolerance >= 0.0
        and max_error is not None
        and max_error >= 0.0
        and digest == build_stable_digest(payload)
    )
    if not common_ready:
        return False

    if mode in {"exact_tensor", "absolute_error_tensor"}:
        reference_shape = payload.get("reference_shape")
        adapter_shape = payload.get("adapter_shape")
        tensor_ready = bool(
            isinstance(reference_shape, list)
            and reference_shape == adapter_shape
            and all(isinstance(value, int) and value >= 0 for value in reference_shape)
            and str(payload.get("reference_dtype", ""))
            == str(payload.get("adapter_dtype", ""))
            and isinstance(payload.get("element_count"), int)
            and int(payload["element_count"]) > 0
            and _is_sha256(payload.get("reference_value_digest"))
            and _is_sha256(payload.get("adapter_value_digest"))
            and isinstance(payload.get("exact_match"), bool)
        )
        if mode == "exact_tensor":
            derived_ready = bool(
                tensor_ready
                and tolerance == 0.0
                and max_error == 0.0
                and payload.get("exact_match") is True
                and payload.get("reference_value_digest")
                == payload.get("adapter_value_digest")
            )
        else:
            derived_ready = bool(tensor_ready and max_error <= tolerance)
    elif mode == "absolute_error_scalar":
        reference = _finite_number(payload.get("reference_value"))
        adapter = _finite_number(payload.get("adapter_value"))
        derived_ready = bool(
            reference is not None
            and adapter is not None
            and math.isclose(
                max_error,
                abs(reference - adapter),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and max_error <= tolerance
        )
    else:
        return False

    operator_id = str(payload.get("operator_id", ""))
    if operator_id == "gaussian_shading.chacha20_ietf_cipher":
        derived_ready = bool(
            derived_ready
            and payload.get("official_source_cipher_contract_ready") is True
        )
    if operator_id == "shallow_diffuse.edit_timestep_floor":
        derived_ready = bool(
            derived_ready
            and _is_sha256(
                payload.get("official_edit_timestep_formula_ast_digest")
            )
        )
    return bool(
        derived_ready
        and payload.get("numerical_fidelity_ready") is derived_ready
    )


def validate_method_faithful_numerical_fidelity_report(
    report: Mapping[str, Any],
    *,
    expected_baseline_id: str,
) -> dict[str, Any]:
    """不执行 Tensor 计算地严格复验已物化报告的字段和稳定摘要."""

    payload = dict(report)
    digest = str(payload.pop("numerical_fidelity_report_digest", ""))
    normalized_baseline_id = str(expected_baseline_id).strip().lower()
    records = payload.get("operator_records")
    operator_ids = payload.get("operator_ids")
    operator_record_count = payload.get("operator_record_count")
    expected_operator_ids = _EXPECTED_OPERATOR_IDS.get(normalized_baseline_id)
    records_ready = bool(
        isinstance(records, list)
        and bool(records)
        and all(
            isinstance(record, Mapping)
            and _comparison_record_semantics_ready(record)
            for record in records
        )
    )
    ready = bool(
        payload.get("report_schema")
        == METHOD_FAITHFUL_NUMERICAL_FIDELITY_SCHEMA
        and expected_operator_ids is not None
        and payload.get("baseline_id") == normalized_baseline_id
        and payload.get("numerical_fidelity_reference_mode")
        == _REFERENCE_MODES.get(normalized_baseline_id)
        and payload.get("official_source_read_mode")
        == "immutable_git_commit_blob"
        and len(str(payload.get("official_repository_commit", ""))) == 40
        and all(
            character in "0123456789abcdef"
            for character in str(payload.get("official_repository_commit", ""))
        )
        and str(payload.get("official_source_file", ""))
        and _is_sha256(payload.get("official_source_blob_sha256"))
        and _is_sha256(payload.get("official_operator_ast_digest"))
        and (
            normalized_baseline_id != "shallow_diffuse"
            or (
                _is_sha256(payload.get("official_entrypoint_blob_sha256"))
                and _is_sha256(
                    payload.get("official_edit_timestep_formula_ast_digest")
                )
            )
        )
        and str(payload.get("adapter_file", ""))
        and _is_sha256(payload.get("adapter_file_sha256"))
        and payload.get("execution_device") == "cpu"
        and str(payload.get("torch_version", ""))
        and str(payload.get("numpy_version", ""))
        and records_ready
        and isinstance(operator_ids, list)
        and tuple(operator_ids) == expected_operator_ids
        and operator_ids == [record.get("operator_id") for record in records]
        and len(operator_ids) == len(set(operator_ids))
        and isinstance(operator_record_count, int)
        and not isinstance(operator_record_count, bool)
        and operator_record_count == len(records)
        and payload.get("operator_records_digest") == build_stable_digest(records)
        and payload.get("method_faithful_numerical_fidelity_ready") is True
        and payload.get("supports_paper_claim") is False
        and digest == build_stable_digest(payload)
    )
    if not ready:
        raise MethodFaithfulNumericalFidelityError(
            f"{expected_baseline_id} 数值忠实度报告无法通过独立复验"
        )
    return {**payload, "numerical_fidelity_report_digest": digest}
