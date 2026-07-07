"""生成外部 baseline 适配器命令计划。

该脚本只生成显式 argv 命令计划, 不执行第三方算法。真实运行应由
`scripts/run_external_baseline_command_plan.py` 在 GPU 或 Colab 环境中完成。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.baselines.command_plan import (
    PRIMARY_BASELINE_ADAPTERS,
    build_baseline_command_plan_manifest,
    load_baseline_command_plan,
    selected_primary_baselines,
)

DEFAULT_PLAN_PATH = Path("outputs/external_baseline_command_plan/baseline_command_plan.json")
DEFAULT_ADAPTER_OUTPUT_ROOT = Path("outputs/external_baseline_execution/adapter_outputs")


def _resolve(root: Path, path: str | Path) -> Path:
    """把相对路径解析到仓库根目录。"""

    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _ensure_under_outputs(root: Path, path: Path) -> Path:
    """确保持久化输出路径位于 outputs 目录下。"""

    outputs_root = (root / "outputs").resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError(f"外部 baseline 命令计划输出必须位于 outputs/ 下: {resolved}") from exc
    return resolved


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="生成外部 baseline 适配器命令计划。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--methods", default=",".join(PRIMARY_BASELINE_ADAPTERS), help="逗号分隔的主表 baseline id。")
    parser.add_argument("--out", default=str(DEFAULT_PLAN_PATH), help="命令计划 JSON 输出路径, 必须位于 outputs/ 下。")
    parser.add_argument("--output-root", default=str(DEFAULT_ADAPTER_OUTPUT_ROOT), help="adapter 输出根目录, 必须位于 outputs/ 下。")
    parser.add_argument("--prompt-plan", default=None, help="扩散类 baseline 使用的 prompt 计划 JSON。")
    parser.add_argument("--image-pairs", default=None, help="T2SMark 结果适配使用的 image_pairs JSON。")
    parser.add_argument("--t2smark-results", default=None, help="T2SMark 官方运行产生的 results.json。")
    parser.add_argument("--attacked-image-manifest", default=None, help="可选 attacked image manifest。")
    parser.add_argument("--threshold", type=float, default=None, help="可选显式检测阈值。")
    parser.add_argument("--contract-only", action="store_true", help="只检查 adapter 契约并写出不可支撑论文主张的诊断产物。")
    parser.add_argument("--require-cuda", action="store_true", help="adapter 运行前要求 CUDA 可用。")
    parser.add_argument("--timeout-seconds", type=int, default=86400, help="单个 baseline 命令超时时间。")
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
    parser.add_argument(
        "--tree-ring-adapter-mode",
        default="method_faithful",
        choices=("method_faithful", "method_faithful_sd35"),
        help="Tree-Ring adapter 运行模式。默认保留轻量链路检查, 真实 GPU 运行应使用 method_faithful_sd35。",
    )
    parser.add_argument(
        "--gaussian-shading-adapter-mode",
        default="method_faithful",
        choices=("method_faithful", "method_faithful_sd35"),
        help="Gaussian Shading adapter 运行模式。默认保留轻量链路检查, 真实 GPU 运行应使用 method_faithful_sd35。",
    )
    parser.add_argument(
        "--shallow-diffuse-adapter-mode",
        default="method_faithful",
        choices=("method_faithful", "method_faithful_sd35"),
        help="Shallow Diffuse adapter 运行模式。默认保留轻量链路检查, 真实 GPU 运行应使用 method_faithful_sd35。",
    )
    parser.add_argument("--tree-ring-watermark-seed", type=int, default=999999, help="Tree-Ring key 随机种子。")
    parser.add_argument("--tree-ring-w-channel", type=int, default=0, help="Tree-Ring 写入通道, -1 表示全部通道。")
    parser.add_argument("--tree-ring-w-radius", type=int, default=10, help="Tree-Ring 傅里叶域写入半径。")
    parser.add_argument("--tree-ring-w-pattern", default="ring", choices=("ring", "rand", "zeros"), help="Tree-Ring key 模式。")
    parser.add_argument("--tree-ring-attack-families", default="", help="Tree-Ring 适配器内部执行的轻量图像攻击族。")
    parser.add_argument("--gaussian-shading-watermark-seed", type=int, default=20260622, help="Gaussian Shading message 随机种子。")
    parser.add_argument("--gaussian-shading-channel-copy", type=int, default=1, help="Gaussian Shading 通道重复因子。")
    parser.add_argument("--gaussian-shading-hw-copy", type=int, default=8, help="Gaussian Shading 空间重复因子。")
    parser.add_argument("--gaussian-shading-attack-families", default="", help="Gaussian Shading 适配器内部执行的轻量图像攻击族。")
    parser.add_argument("--shallow-diffuse-watermark-seed", type=int, default=42, help="Shallow Diffuse patch 随机种子。")
    parser.add_argument("--shallow-diffuse-w-channel", type=int, default=0, help="Shallow Diffuse 写入通道, -1 表示全部通道。")
    parser.add_argument("--shallow-diffuse-w-radius", type=int, default=10, help="Shallow Diffuse mask 半径。")
    parser.add_argument("--shallow-diffuse-w-inner-radius", type=int, default=0, help="Shallow Diffuse ring mask 内半径。")
    parser.add_argument(
        "--shallow-diffuse-w-mask-shape",
        default="circle",
        choices=("circle", "ring", "square", "whole", "outercircle"),
        help="Shallow Diffuse mask 形状。",
    )
    parser.add_argument("--shallow-diffuse-w-pattern", default="complex_rand", help="Shallow Diffuse watermark patch 模式。")
    parser.add_argument("--shallow-diffuse-w-injection", default="complex", help="Shallow Diffuse watermark 注入模式。")
    parser.add_argument("--shallow-diffuse-w-measurement", default="l1_complex", help="Shallow Diffuse watermark 检测度量。")
    parser.add_argument("--shallow-diffuse-edit-fraction", type=float, default=0.2, help="Shallow Diffuse 浅层注入采样位置比例。")
    parser.add_argument("--shallow-diffuse-attack-families", default="", help="Shallow Diffuse 适配器内部执行的轻量图像攻击族。")
    return parser


def _append_common_model_args(command: list[str], args: argparse.Namespace) -> None:
    """向扩散类 baseline adapter 命令追加共同模型参数。"""

    command.extend(
        [
            "--model-id",
            str(args.model_id),
            "--torch-dtype",
            str(args.torch_dtype),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--latent-channels",
            str(args.latent_channels),
            "--num-inference-steps",
            str(args.num_inference_steps),
            "--num-inversion-steps",
            str(args.num_inversion_steps),
            "--guidance-scale",
            str(args.guidance_scale),
            "--seed",
            str(args.seed),
        ]
    )
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])


def build_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    """构建外部 baseline adapter 命令计划。"""

    root = _resolve(Path.cwd(), args.root)
    output_root = _ensure_under_outputs(root, _resolve(root, args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    selected = selected_primary_baselines(args.methods)
    if not args.contract_only and any(method != "t2smark" for method in selected) and not args.prompt_plan:
        raise ValueError("运行扩散类 SD3.5 adapter 时必须提供 --prompt-plan, 或使用 --contract-only。")
    if not args.contract_only and "t2smark" in selected and (not args.image_pairs or not args.t2smark_results):
        raise ValueError("运行 T2SMark 结果适配时必须提供 --image-pairs 与 --t2smark-results。")

    rows: list[dict[str, Any]] = []
    for baseline_id in selected:
        adapter_path = root / PRIMARY_BASELINE_ADAPTERS[baseline_id]
        if not adapter_path.is_file():
            raise FileNotFoundError(f"外部 baseline adapter 不存在: {adapter_path}")
        baseline_output_root = output_root / baseline_id
        observation_output = baseline_output_root / "baseline_observations.json"
        artifact_root = baseline_output_root / "artifacts"
        command = [
            sys.executable,
            str(adapter_path),
            "--out",
            str(observation_output),
            "--artifact-root",
            str(artifact_root),
        ]
        if args.contract_only:
            command.append("--contract-only")
        if args.require_cuda:
            command.append("--require-cuda")
        if baseline_id == "t2smark":
            if args.image_pairs:
                command.extend(["--image-pairs", str(_resolve(root, args.image_pairs))])
            if args.t2smark_results:
                command.extend(["--t2smark-results", str(_resolve(root, args.t2smark_results))])
            if args.attacked_image_manifest:
                command.extend(["--attacked-image-manifest", str(_resolve(root, args.attacked_image_manifest))])
            if args.threshold is not None:
                command.extend(["--threshold", str(args.threshold)])
        else:
            if args.prompt_plan:
                command.extend(["--prompt-plan", str(_resolve(root, args.prompt_plan))])
            _append_common_model_args(command, args)
            if baseline_id == "tree_ring":
                command.extend(["--adapter-mode", str(args.tree_ring_adapter_mode)])
                if args.tree_ring_adapter_mode == "method_faithful_sd35":
                    command.extend(
                        [
                            "--watermark-seed",
                            str(args.tree_ring_watermark_seed),
                            "--w-channel",
                            str(args.tree_ring_w_channel),
                            "--w-radius",
                            str(args.tree_ring_w_radius),
                            "--w-pattern",
                            str(args.tree_ring_w_pattern),
                        ]
                    )
                if args.tree_ring_adapter_mode == "method_faithful_sd35" and str(args.tree_ring_attack_families).strip():
                    command.extend(["--attack-families", str(args.tree_ring_attack_families)])
            elif baseline_id == "gaussian_shading":
                command.extend(["--adapter-mode", str(args.gaussian_shading_adapter_mode)])
                if args.gaussian_shading_adapter_mode == "method_faithful_sd35":
                    command.extend(
                        [
                            "--watermark-seed",
                            str(args.gaussian_shading_watermark_seed),
                            "--channel-copy",
                            str(args.gaussian_shading_channel_copy),
                            "--hw-copy",
                            str(args.gaussian_shading_hw_copy),
                        ]
                    )
                if args.gaussian_shading_adapter_mode == "method_faithful_sd35" and str(args.gaussian_shading_attack_families).strip():
                    command.extend(["--attack-families", str(args.gaussian_shading_attack_families)])
            elif baseline_id == "shallow_diffuse":
                command.extend(["--adapter-mode", str(args.shallow_diffuse_adapter_mode)])
                if args.shallow_diffuse_adapter_mode == "method_faithful_sd35":
                    command.extend(
                        [
                            "--watermark-seed",
                            str(args.shallow_diffuse_watermark_seed),
                            "--w-channel",
                            str(args.shallow_diffuse_w_channel),
                            "--w-radius",
                            str(args.shallow_diffuse_w_radius),
                            "--w-inner-radius",
                            str(args.shallow_diffuse_w_inner_radius),
                            "--w-mask-shape",
                            str(args.shallow_diffuse_w_mask_shape),
                            "--w-pattern",
                            str(args.shallow_diffuse_w_pattern),
                            "--w-injection",
                            str(args.shallow_diffuse_w_injection),
                            "--w-measurement",
                            str(args.shallow_diffuse_w_measurement),
                            "--edit-fraction",
                            str(args.shallow_diffuse_edit_fraction),
                        ]
                    )
                if args.shallow_diffuse_adapter_mode == "method_faithful_sd35" and str(args.shallow_diffuse_attack_families).strip():
                    command.extend(["--attack-families", str(args.shallow_diffuse_attack_families)])
        rows.append(
            {
                "baseline_id": baseline_id,
                "command": command,
                "output_path": str(observation_output),
                "working_directory": str(root),
                "timeout_seconds": int(args.timeout_seconds),
            }
        )
    return rows


def main() -> None:
    """CLI 入口。"""

    args = build_parser().parse_args()
    root = _resolve(Path.cwd(), args.root)
    plan_path = _ensure_under_outputs(root, _resolve(root, args.out))
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    specs = load_baseline_command_plan(plan_path)
    manifest_path = plan_path.with_name("baseline_command_plan_manifest.json")
    manifest_path.write_text(
        json.dumps(build_baseline_command_plan_manifest(specs), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"plan_path": str(plan_path), "manifest_path": str(manifest_path), "row_count": len(plan)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

