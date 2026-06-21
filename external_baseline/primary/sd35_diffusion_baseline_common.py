"""SD3.5 Medium 外部扩散 baseline 适配器公共工具。

该文件只维护本项目外部 baseline 适配层的输入输出契约, 不把第三方论文算法复制进核心方法包。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.core.digest import build_stable_digest

SCORE_AMPLITUDES: dict[str, tuple[float, float]] = {
    "tree_ring": (0.18, 0.82),
    "gaussian_shading": (0.12, 0.88),
    "shallow_diffuse": (0.22, 0.78),
}

SCORE_NAMES: dict[str, str] = {
    "tree_ring": "tree_ring_sd35_latent_ring_score",
    "gaussian_shading": "gaussian_shading_sd35_latent_key_score",
    "shallow_diffuse": "shallow_diffuse_sd35_shallow_update_score",
}

ADAPTER_BOUNDARY = "sd35_latent_smoke_adapter_not_formal_external_baseline_evidence"


def load_json(path: str | Path) -> Any:
    """读取 JSON 文件, 兼容带 BOM 的 Colab 或 Windows 输出。"""

    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, payload: Any) -> Path:
    """写出 JSON 文件并创建父目录。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def load_prompt_rows(path: str | Path | None) -> list[dict[str, Any]]:
    """读取 prompt 计划并规范化为字典列表。

    该函数兼容列表、`prompts`、`prompt_rows` 和 `records` 这几种常见导出形态, 便于复用已有
    prompt split 产物。
    """

    if path is None:
        return []
    payload = load_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("prompts") or payload.get("prompt_rows") or payload.get("records") or []
    else:
        raise TypeError("prompt plan 必须是 JSON 列表或对象")
    if not isinstance(rows, list):
        raise TypeError("prompt plan 中的 prompt rows 必须是列表")
    return [dict(row) for row in rows]


def selected_prompt_rows(rows: list[dict[str, Any]], max_samples: int | None) -> list[dict[str, Any]]:
    """按可选样本上限截取 prompt rows。"""

    if max_samples is None:
        return rows
    return rows[: max(0, int(max_samples))]


def _prompt_text(row: dict[str, Any]) -> str:
    """从常见 prompt 字段中读取文本, 便于复用不同 prompt 产物。"""

    for field_name in ("prompt_text", "prompt", "caption", "text"):
        value = str(row.get(field_name) or "").strip()
        if value:
            return value
    return "unspecified prompt"


def _prompt_id(row: dict[str, Any], index: int) -> str:
    """读取稳定 prompt 标识, 缺失时使用 adapter 内部顺序标识。"""

    value = str(row.get("prompt_id") or row.get("id") or "").strip()
    return value or f"external_baseline_prompt_{index:05d}"


def _split(row: dict[str, Any]) -> str:
    """读取 prompt split, 缺失时按 gpu_smoke 处理。"""

    value = str(row.get("split") or "gpu_smoke").strip()
    return value or "gpu_smoke"


def _stable_seed(base_seed: int, baseline_id: str, prompt_id: str) -> int:
    """把 baseline、prompt 和用户 seed 映射为可复现的整数种子。"""

    digest = build_stable_digest({"base_seed": int(base_seed), "baseline_id": baseline_id, "prompt_id": prompt_id})
    return int(digest[:12], 16) % (2**31 - 1)


def _latent_shape(args: argparse.Namespace) -> tuple[int, int, int]:
    """按 SD3.5 latent 下采样约定构造轻量 latent 形状。"""

    return (int(args.latent_channels), max(1, int(args.height) // 8), max(1, int(args.width) // 8))


def _torch_device_and_dtype(torch_module: Any, require_cuda: bool) -> tuple[Any, Any]:
    """选择 adapter 张量执行设备, CUDA 要求由上层参数显式控制。"""

    if require_cuda:
        return torch_module.device("cuda"), torch_module.float32
    if torch_module.cuda.is_available():
        return torch_module.device("cuda"), torch_module.float32
    return torch_module.device("cpu"), torch_module.float32


def _torch_generator(torch_module: Any, device: Any, seed: int) -> Any:
    """创建与执行设备兼容的 torch 随机数生成器。"""

    generator_device = "cuda" if getattr(device, "type", str(device)) == "cuda" else "cpu"
    generator = torch_module.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    return generator


def _normalize_direction(torch_module: Any, direction: Any) -> Any:
    """把载体方向归一化, 使不同 baseline 的分数可比较。"""

    norm = torch_module.linalg.vector_norm(direction)
    return direction / torch_module.clamp(norm, min=1e-12)


def _method_direction(torch_module: Any, baseline_id: str, shape: tuple[int, int, int], seed: int, device: Any, dtype: Any) -> Any:
    """构造三类主表 baseline 的 SD3.5 latent 级载体方向。

    该实现属于项目特定的 GPU smoke adapter, 只验证 SD3.5 latent 形状、GPU 张量路径和 observation 落盘链路;
    它不等价于第三方论文的正式官方复现。
    """

    channels, height, width = shape
    generator = _torch_generator(torch_module, device, seed)
    if baseline_id == "tree_ring":
        y_axis = torch_module.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x_axis = torch_module.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch_module.meshgrid(y_axis, x_axis, indexing="ij")
        radius = torch_module.sqrt(xx.square() + yy.square())
        ring = torch_module.exp(-((radius - 0.55).square()) / (2.0 * 0.08**2))
        signs = torch_module.randint(0, 2, (channels, 1, 1), generator=generator, device=device, dtype=dtype) * 2.0 - 1.0
        direction = signs * ring.unsqueeze(0)
    elif baseline_id == "gaussian_shading":
        direction = torch_module.randn(shape, generator=generator, device=device, dtype=dtype)
    elif baseline_id == "shallow_diffuse":
        y_axis = torch_module.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        x_axis = torch_module.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch_module.meshgrid(y_axis, x_axis, indexing="ij")
        base = torch_module.sin(2.0 * torch_module.pi * xx) * torch_module.cos(2.0 * torch_module.pi * yy)
        channel_weights = torch_module.linspace(0.5, 1.5, channels, device=device, dtype=dtype).reshape(channels, 1, 1)
        channel_signs = torch_module.randint(0, 2, (channels, 1, 1), generator=generator, device=device, dtype=dtype) * 2.0 - 1.0
        direction = channel_weights * channel_signs * base.unsqueeze(0)
    else:
        direction = torch_module.randn(shape, generator=generator, device=device, dtype=dtype)
    return _normalize_direction(torch_module, direction)


def _score_latent(torch_module: Any, latent: Any, direction: Any) -> float:
    """计算 latent 在载体方向上的投影分数。"""

    return float(torch_module.sum(latent * direction).detach().cpu().item())


def _torch_latent_scores(
    *,
    baseline_id: str,
    prompt_id: str,
    args: argparse.Namespace,
) -> tuple[float, float, dict[str, Any]]:
    """使用 torch 构造 clean / positive latent 并计算轻量检测分数。"""

    import torch

    device, dtype = _torch_device_and_dtype(torch, bool(args.require_cuda))
    shape = _latent_shape(args)
    seed = _stable_seed(int(args.seed), baseline_id, prompt_id)
    generator = _torch_generator(torch, device, seed + 17)
    direction = _method_direction(torch, baseline_id, shape, seed, device, dtype)
    clean_amplitude, positive_amplitude = SCORE_AMPLITUDES.get(baseline_id, (0.2, 0.8))
    base_latent = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    base_latent = base_latent - torch.sum(base_latent * direction) * direction
    clean_score = _score_latent(torch, base_latent + clean_amplitude * direction, direction)
    positive_score = _score_latent(torch, base_latent + positive_amplitude * direction, direction)
    metadata = {
        "torch_available": True,
        "execution_device": str(device),
        "latent_shape": list(shape),
        "adapter_seed": seed,
    }
    return clean_score, positive_score, metadata


def _fallback_latent_scores(*, baseline_id: str, prompt_id: str, args: argparse.Namespace) -> tuple[float, float, dict[str, Any]]:
    """在未要求 CUDA 且本地缺少 torch 时生成确定性诊断分数。"""

    clean_score, positive_score = SCORE_AMPLITUDES.get(baseline_id, (0.2, 0.8))
    metadata = {
        "torch_available": False,
        "execution_device": "python_scalar_fallback",
        "latent_shape": list(_latent_shape(args)),
        "adapter_seed": _stable_seed(int(args.seed), baseline_id, prompt_id),
    }
    return clean_score, positive_score, metadata


def _latent_scores(*, baseline_id: str, prompt_id: str, args: argparse.Namespace) -> tuple[float, float, dict[str, Any]]:
    """根据运行环境选择 torch 张量路径或本地标量诊断路径。"""

    try:
        return _torch_latent_scores(baseline_id=baseline_id, prompt_id=prompt_id, args=args)
    except ImportError:
        if bool(args.require_cuda):
            raise
        return _fallback_latent_scores(baseline_id=baseline_id, prompt_id=prompt_id, args=args)


def _baseline_observation(
    *,
    baseline_id: str,
    row: dict[str, Any],
    row_index: int,
    score: float,
    threshold: float,
    sample_role: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """构造统一 external baseline observation 行。"""

    prompt_id = _prompt_id(row, row_index)
    payload = {
        "event_id": f"{baseline_id}_{row_index:05d}__{sample_role}",
        "baseline_id": baseline_id,
        "score": float(score),
        "threshold": float(threshold),
        "score_name": SCORE_NAMES.get(baseline_id, f"{baseline_id}_sd35_latent_smoke_score"),
        "higher_is_positive": True,
        "detection_decision": bool(float(score) >= float(threshold)),
        "split": _split(row),
        "sample_role": sample_role,
        "attack_family": "clean",
        "attack_condition": "clean_none",
        "prompt_id": prompt_id,
        "prompt_text": _prompt_text(row),
        "image_id": f"{baseline_id}_{row_index:05d}",
        "threshold_source": "latent_smoke_midpoint_between_clean_and_positive",
        "producer_id": f"{baseline_id}_sd35_latent_smoke_adapter",
        "producer_role": "external_baseline_result_adapter",
        "adapter_boundary": ADAPTER_BOUNDARY,
        "formal_result_claim": False,
        "supports_paper_claim": False,
        "latent_shape": metadata.get("latent_shape", []),
        "execution_device": metadata.get("execution_device", ""),
    }
    payload["baseline_observation_digest"] = build_stable_digest(payload)
    return payload


def build_sd35_latent_smoke_observations(
    *,
    baseline_id: str,
    args: argparse.Namespace,
    prompt_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """生成三类扩散外部 baseline 的 SD3.5 latent 级 smoke observations。"""

    observations: list[dict[str, Any]] = []
    score_metadata: list[dict[str, Any]] = []
    for row_index, row in enumerate(prompt_rows):
        prompt_id = _prompt_id(row, row_index)
        clean_score, positive_score, metadata = _latent_scores(baseline_id=baseline_id, prompt_id=prompt_id, args=args)
        threshold = (float(clean_score) + float(positive_score)) / 2.0
        score_metadata.append({"prompt_id": prompt_id, **metadata})
        observations.append(
            _baseline_observation(
                baseline_id=baseline_id,
                row=row,
                row_index=row_index,
                score=clean_score,
                threshold=threshold,
                sample_role="clean_negative",
                metadata=metadata,
            )
        )
        observations.append(
            _baseline_observation(
                baseline_id=baseline_id,
                row=row,
                row_index=row_index,
                score=positive_score,
                threshold=threshold,
                sample_role="positive_source",
                metadata=metadata,
            )
        )
    summary_metadata = {
        "adapter_boundary": ADAPTER_BOUNDARY,
        "prompt_count": len(prompt_rows),
        "score_metadata": score_metadata,
    }
    return observations, summary_metadata


def require_cuda_if_requested(require_cuda: bool) -> None:
    """在用户明确要求时检查 CUDA 是否可用。"""

    if not require_cuda:
        return
    try:
        import torch
    except Exception as exc:  # pragma: no cover - 本地轻量测试不依赖 torch
        raise RuntimeError("已要求 CUDA, 但无法导入 torch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("已要求 CUDA, 但当前环境未检测到可用 GPU")


def build_common_parser(*, baseline_id: str, description: str) -> argparse.ArgumentParser:
    """构造 SD3.5 外部扩散 baseline adapter 的共同参数解析器。"""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--prompt-plan", default=None, help="共同 prompt 计划 JSON。")
    parser.add_argument("--out", required=True, help="baseline_observations.json 输出路径。")
    parser.add_argument("--artifact-root", default=None, help="诊断 manifest 和外部方法产物目录。")
    parser.add_argument("--contract-only", action="store_true", help="只写出 adapter 契约诊断, 不声明正式结果。")
    parser.add_argument("--require-cuda", action="store_true", help="运行前要求 CUDA 可用。")
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--num-inversion-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.set_defaults(baseline_id=baseline_id)
    return parser


def _artifact_root(args: argparse.Namespace) -> Path:
    """解析 adapter 产物根目录。"""

    if args.artifact_root:
        return Path(args.artifact_root)
    return Path(args.out).with_suffix("").parent / "artifacts"


def write_contract_manifest(
    *,
    baseline_id: str,
    args: argparse.Namespace,
    adapter_status: str,
    model_alignment_status: str,
    observation_count: int,
    unsupported_reason: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """写出 adapter 诊断 manifest。

    该 manifest 是项目特定写法: 它明确记录 SD3.5 主线与外部 baseline 官方实现之间的边界,
    避免把仅完成接口检查的运行误当作论文对比证据。
    """

    artifact_root = _artifact_root(args)
    artifact_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "artifact_name": f"{baseline_id}_slm_adapter_manifest.json",
        "producer_id": f"{baseline_id}_slm_external_baseline_adapter",
        "baseline_id": baseline_id,
        "adapter_status": adapter_status,
        "model_alignment_status": model_alignment_status,
        "model_id": str(args.model_id),
        "torch_dtype": str(args.torch_dtype),
        "height": int(args.height),
        "width": int(args.width),
        "latent_channels": int(args.latent_channels),
        "num_inference_steps": int(args.num_inference_steps),
        "num_inversion_steps": int(args.num_inversion_steps),
        "guidance_scale": float(args.guidance_scale),
        "seed": int(args.seed),
        "prompt_plan_path": str(getattr(args, "prompt_plan", "") or ""),
        "observation_count": int(observation_count),
        "baseline_observations_path": str(Path(args.out)),
        "artifact_root": str(artifact_root),
        "formal_result_claim": False,
        "supports_paper_claim": False,
        "unsupported_reason": unsupported_reason,
    }
    if extra_metadata:
        manifest.update(extra_metadata)
    manifest["adapter_digest"] = build_stable_digest(manifest)
    manifest_path = Path(args.out).with_name(f"{baseline_id}_slm_adapter_manifest.json")
    write_json(manifest_path, manifest)
    write_json(artifact_root / f"{baseline_id}_slm_adapter_manifest.json", manifest)
    return manifest


def run_contract_or_report_required_adapter(
    *,
    baseline_id: str,
    model_alignment_status: str,
    real_run_unsupported_reason: str,
) -> None:
    """运行 SD3.5 latent 级外部扩散 baseline adapter。

    当使用 `--contract-only` 时, 该函数写出空 observation 与诊断 manifest, 用于检查命令编排和
    产物落盘链路。当未使用该选项时, 该函数生成项目治理内的 latent smoke observations。
    该路径只验证 SD3.5 latent 张量、GPU 运行和落盘协议, 不声明第三方 baseline 正式复现结果。
    """

    parser = build_common_parser(
        baseline_id=baseline_id,
        description=f"运行 {baseline_id} 的 SLM 外部 baseline adapter。",
    )
    args = parser.parse_args()
    require_cuda_if_requested(bool(args.require_cuda))
    if args.contract_only:
        write_json(args.out, [])
        manifest = write_contract_manifest(
            baseline_id=baseline_id,
            args=args,
            adapter_status="adapter_contract_ready",
            model_alignment_status=model_alignment_status,
            observation_count=0,
            unsupported_reason=real_run_unsupported_reason,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if not args.prompt_plan:
        raise SystemExit("SD3.5 latent smoke adapter 运行必须提供 --prompt-plan。")
    prompt_rows = selected_prompt_rows(load_prompt_rows(args.prompt_plan), args.max_samples)
    observations, adapter_metadata = build_sd35_latent_smoke_observations(
        baseline_id=baseline_id,
        args=args,
        prompt_rows=prompt_rows,
    )
    write_json(args.out, observations)
    manifest = write_contract_manifest(
        baseline_id=baseline_id,
        args=args,
        adapter_status="sd35_latent_smoke_adapter_ready",
        model_alignment_status=model_alignment_status,
        observation_count=len(observations),
        unsupported_reason=real_run_unsupported_reason,
        extra_metadata={
            **adapter_metadata,
            "formal_result_claim": False,
            "supports_paper_claim": False,
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
