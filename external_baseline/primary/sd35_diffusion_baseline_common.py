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
    """运行尚未完成真实 SD3.5 实装的外部扩散 baseline adapter。

    当使用 `--contract-only` 时, 该函数写出空 observation 与诊断 manifest, 用于检查命令编排和
    产物落盘链路。当未使用该选项时, 它返回非零退出码, 明确提示需要接入真实算法路径。
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
        raise SystemExit("真实 SD3.5 adapter 运行必须提供 --prompt-plan。")
    prompt_rows = selected_prompt_rows(load_prompt_rows(args.prompt_plan), args.max_samples)
    diagnostic = {
        "baseline_id": baseline_id,
        "prompt_count": len(prompt_rows),
        "adapter_status": "real_sd35_adapter_required",
        "model_alignment_status": model_alignment_status,
        "unsupported_reason": real_run_unsupported_reason,
    }
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2), file=sys.stderr)
    raise SystemExit(2)
