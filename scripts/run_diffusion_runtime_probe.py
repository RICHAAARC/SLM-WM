"""运行 SD runtime adapter probe 并写出受治理 records。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest

from experiments.runtime.diffusion.model_adapter import RuntimeModelConfig, RuntimeProbeBundle
from experiments.runtime.diffusion.sd35_adapter import Sd35RuntimeAdapter
from experiments.runtime.diffusion.sd3_adapter import Sd3RuntimeAdapter

RUNTIME_UNIT_NAME = "sd_runtime_adapter"
DEFAULT_OUTPUT_DIR = Path("outputs/sd_runtime_adapter")
DEFAULT_CONFIG_PATHS = (Path("configs/model_sd3.yaml"), Path("configs/model_sd35.yaml"))


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定、可读文本。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: Any) -> str:
    """把 JSON 兼容对象转为 JSONL 单行文本。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def parse_config_value(raw_value: str) -> Any:
    """解析当前配置文件使用的最小 YAML 标量子集。"""
    value = raw_value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_runtime_config(path: str | Path) -> RuntimeModelConfig:
    """从轻量 YAML 配置读取 runtime model config。"""
    config_path = Path(path)
    payload: dict[str, Any] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, value = stripped.split(":", 1)
        payload[key.strip()] = parse_config_value(value)
    return RuntimeModelConfig(**payload)


def resolve_code_version(root_path: Path) -> str:
    """读取 Git 提交标识, 若工作区有变更则追加 dirty 标记。"""
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--short"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "git_version_unavailable"
    commit_id = commit_result.stdout.strip()
    if not commit_id:
        return "git_version_unavailable"
    return f"{commit_id}-dirty" if status_result.stdout.strip() else commit_id


def ensure_output_dir_under_outputs(root_path: Path, output_dir: Path) -> Path:
    """确保 runtime adapter 输出目录位于 outputs/ 下。"""
    resolved_output_dir = (root_path / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved_output_dir.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("runtime adapter 输出目录必须位于 outputs/ 下") from exc
    return resolved_output_dir


def build_adapter(model_family: str) -> Any:
    """根据模型族选择 runtime adapter。"""
    adapters = {"sd3": Sd3RuntimeAdapter(), "sd35": Sd35RuntimeAdapter()}
    if model_family not in adapters:
        raise ValueError(f"不支持的模型族: {model_family}")
    return adapters[model_family]


def run_runtime_probes(configs: tuple[RuntimeModelConfig, ...]) -> tuple[RuntimeProbeBundle, ...]:
    """运行一组 runtime adapter probes。"""
    bundles: list[RuntimeProbeBundle] = []
    for config in configs:
        adapter = build_adapter(config.model_family)
        bundles.append(adapter.generate(config))
    return tuple(bundles)


def build_quality_summary(bundles: tuple[RuntimeProbeBundle, ...]) -> dict[str, Any]:
    """构造 runtime adapter 质量与复现摘要。"""
    generation_records = [bundle.generation_record for bundle in bundles]
    latent_records = [record.to_dict() for bundle in bundles for record in bundle.latent_trace_records]
    attention_records = [record.to_dict() for bundle in bundles for record in bundle.attention_capture_records]
    unsupported_reasons = sorted(
        {record["unsupported_reason"] for record in generation_records if record["unsupported_reason"]}
    )
    reproducibility_digest = build_stable_digest(
        {
            "generation_records": generation_records,
            "latent_record_count": len(latent_records),
            "attention_record_count": len(attention_records),
        }
    )
    return {
        "construction_unit_name": RUNTIME_UNIT_NAME,
        "artifact_id": "sd_runtime_adapter_quality_summary",
        "artifact_type": "local_summary",
        "decision": "pass" if generation_records and latent_records and attention_records else "fail",
        "runtime_dependency_mode": "synthetic_fallback",
        "generation_record_count": len(generation_records),
        "latent_trace_record_count": len(latent_records),
        "attention_capture_record_count": len(attention_records),
        "unsupported_reason_count": len(unsupported_reasons),
        "unsupported_reasons": unsupported_reasons,
        "reproducibility_digest": reproducibility_digest,
        "metrics": {
            "mean_quality_score": sum(record["quality_score"] for record in generation_records) / len(generation_records),
        },
        "metadata": {
            "records_are_synthetic": True,
            "supports_paper_claim": False,
        },
    }


def write_runtime_probe_outputs(
    root: str | Path = ".",
    config_paths: tuple[str | Path, ...] = DEFAULT_CONFIG_PATHS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """写出 SD runtime adapter records、summary 和 manifest。"""
    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, Path(output_dir))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_paths = tuple(Path(path) for path in config_paths)
    configs = tuple(load_runtime_config(root_path / path if not path.is_absolute() else path) for path in resolved_config_paths)
    bundles = run_runtime_probes(configs)
    summary = build_quality_summary(bundles)

    generation_records = [bundle.generation_record for bundle in bundles]
    latent_records = [record.to_dict() for bundle in bundles for record in bundle.latent_trace_records]
    attention_records = [record.to_dict() for bundle in bundles for record in bundle.attention_capture_records]

    generation_path = resolved_output_dir / "sd_generation_records.jsonl"
    latent_path = resolved_output_dir / "latent_trace_records.jsonl"
    attention_path = resolved_output_dir / "attention_capture_records.jsonl"
    summary_path = resolved_output_dir / "generation_quality_summary.json"
    manifest_path = resolved_output_dir / "manifest.local.json"

    generation_path.write_text("".join(json_line(record) for record in generation_records), encoding="utf-8")
    latent_path.write_text("".join(json_line(record) for record in latent_records), encoding="utf-8")
    attention_path.write_text("".join(json_line(record) for record in attention_records), encoding="utf-8")
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")

    output_paths = tuple(
        path.relative_to(root_path).as_posix()
        for path in (generation_path, latent_path, attention_path, summary_path, manifest_path)
    )
    manifest = build_artifact_manifest(
        artifact_id="sd_runtime_adapter_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(path.as_posix() for path in resolved_config_paths)
        + (
            "outputs/core_method_synthetic_smoke/manifest.local.json",
            "experiments/runtime/diffusion",
            "scripts/run_diffusion_runtime_probe.py",
            "tests/functional/test_diffusion_runtime_adapter.py",
        ),
        output_paths=output_paths,
        config={
            "construction_unit_name": RUNTIME_UNIT_NAME,
            "config_digests": [build_stable_digest(config.to_dict()) for config in configs],
            "summary_digest": build_stable_digest(summary),
            "generation_record_count": len(generation_records),
            "latent_trace_record_count": len(latent_records),
            "attention_capture_record_count": len(attention_records),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/run_diffusion_runtime_probe.py",
        metadata={
            "construction_unit_name": RUNTIME_UNIT_NAME,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "decision": summary["decision"],
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(description="运行 SD runtime adapter probe。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="模型配置路径, 可重复传入。",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    config_paths = tuple(Path(path) for path in args.config) if args.config else DEFAULT_CONFIG_PATHS
    manifest = write_runtime_probe_outputs(args.root, config_paths=config_paths, output_dir=args.output_dir)
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
