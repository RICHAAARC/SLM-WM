"""运行真实单 Prompt 方法路径并自动形成 GPU 方法资格化报告."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol.formal_randomization import formal_randomization_repeat_ids
from experiments.protocol.gpu_method_qualification import (
    build_gpu_method_qualification_report,
)
from experiments.protocol.gpu_method_qualification_schema import (
    GPU_METHOD_QUALIFICATION_INVOCATION_RESULT_SCHEMA,
)
from experiments.protocol.paper_run_config import (
    RUN_EXPECTED_PROMPT_COUNTS,
    build_paper_run_config,
)
from experiments.protocol.prompts import (
    build_prompt_records,
    read_prompt_file,
)
from experiments.protocol.splits import apply_split_assignments
from experiments.runners.image_only_dataset_workload import build_method_config
from experiments.runners.semantic_watermark_runtime import (
    semantic_watermark_runtime_config_digest,
    write_semantic_watermark_runtime_outputs,
)
from experiments.runtime import repository_environment
from experiments.runtime.dependency_profiles import require_dependency_profile_ready
from experiments.runtime.scientific_unit_provenance import (
    validate_scientific_unit_provenance,
)
from main.core.digest import build_stable_digest


DEFAULT_KNOWN_ANSWER_PATH = Path(
    "configs/keyed_prg_cross_platform_known_answer.json"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/gpu_method_qualification")


def _file_sha256(path: Path) -> str:
    """流式计算资格化输入或运行产物的 SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """读取运行结果或可选资源预算 JSON 映射."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是映射: {path}")
    return payload


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """读取真实运行写出的 JSONL 映射记录."""

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL 第{line_number}行不是映射: {path}")
        rows.append(row)
    return tuple(rows)


def _resolve_under_outputs(root: Path, path: str | Path) -> Path:
    """把资格化持久化目录限制在仓库 outputs/ 下."""

    requested = Path(path).expanduser()
    resolved = (
        requested.resolve()
        if requested.is_absolute()
        else (root / requested).resolve()
    )
    try:
        resolved.relative_to((root / "outputs").resolve())
    except ValueError as exc:
        raise ValueError("GPU 方法资格化输出必须位于 outputs/ 下") from exc
    return resolved


def _registered_prompt(root: Path, paper_run_name: str, prompt_id: str) -> Any:
    """从受治理 Prompt 文件选择一个已登记且已分配 split 的真实输入."""

    paper_run = build_paper_run_config(root)
    if paper_run.run_name != paper_run_name:
        raise ValueError("当前环境中的论文运行层级与命令参数不一致")
    records = apply_split_assignments(
        build_prompt_records(
            paper_run.prompt_set,
            read_prompt_file(root / paper_run.prompt_file),
        )
    )
    matches = [record for record in records if record.prompt_id == prompt_id]
    if len(matches) != 1:
        raise ValueError("--prompt-id 必须唯一存在于当前受治理 Prompt 文件")
    return matches[0]


def _qualification_binding(
    *,
    root: Path,
    runtime_result: Mapping[str, Any],
    config: Any,
    prompt_record: Any,
) -> dict[str, Any]:
    """绑定 Git、依赖锁、模型 revision、Prompt 和真实运行文件摘要."""

    metadata = runtime_result.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    provenance = validate_scientific_unit_provenance(
        metadata["scientific_unit_provenance"]
    )
    environment = provenance["scientific_execution_environment"]
    runtime_paths = {
        field_name: str(runtime_result[field_name])
        for field_name in (
            "update_record_path",
            "detection_record_path",
            "clean_image_path",
            "watermarked_image_path",
            "manifest_path",
        )
    }
    artifact_sha256 = {
        field_name: _file_sha256((root / relative_path).resolve())
        for field_name, relative_path in runtime_paths.items()
    }
    core = {
        "code_version": environment["formal_execution_commit"],
        "dependency_profile_id": environment["dependency_profile_id"],
        "dependency_profile_digest": environment["dependency_profile_digest"],
        "complete_hash_lock_digest": environment[
            "complete_hash_lock_digest"
        ],
        "model_revisions": {
            "sd35_model_id": config.model_id,
            "sd35_model_revision": config.model_revision,
            "vae_model_id": config.model_id,
            "vae_model_revision": config.model_revision,
            "vae_class_name": config.vae_class_name,
            "vae_scaling_factor": config.vae_scaling_factor,
            "vae_shift_factor": config.vae_shift_factor,
            "vision_model_id": config.vision_model_id,
            "vision_model_revision": config.vision_model_revision,
        },
        "input_summary": {
            "paper_run_name": prompt_record.prompt_set,
            "prompt_id": prompt_record.prompt_id,
            "prompt_digest": build_stable_digest({"prompt": config.prompt}),
            "registered_prompt_digest": prompt_record.prompt_digest,
            "split": prompt_record.split,
            "randomization_repeat_id": config.randomization_repeat_id,
            "method_runtime_config_digest": (
                semantic_watermark_runtime_config_digest(config)
            ),
        },
        "runtime_artifact_paths": runtime_paths,
        "runtime_artifact_sha256": artifact_sha256,
    }
    return {**core, "qualification_binding_digest": build_stable_digest(core)}


def build_parser() -> argparse.ArgumentParser:
    """构造可脱离 Notebook 运行的单 Prompt GPU 资格化入口."""

    parser = argparse.ArgumentParser(
        description="运行真实单 Prompt SLM-WM 路径并执行 GPU 方法资格化",
    )
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument(
        "--paper-run-name",
        default="probe_paper",
        choices=tuple(RUN_EXPECTED_PROMPT_COUNTS),
    )
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument(
        "--known-answer",
        default=str(DEFAULT_KNOWN_ANSWER_PATH),
    )
    parser.add_argument("--registered-budget")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """运行真实方法并以方法真实性门禁决定进程状态码."""

    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    output_root = _resolve_under_outputs(root, args.output_root)
    os.environ["SLM_WM_PAPER_RUN_NAME"] = args.paper_run_name
    prompt_record = _registered_prompt(
        root,
        args.paper_run_name,
        args.prompt_id,
    )
    execution_lock = repository_environment.require_published_formal_execution_lock(
        root
    )
    dependency_profile = require_dependency_profile_ready(
        "sd35_method_runtime_gpu"
    )
    if (
        execution_lock["formal_execution_commit"]
        != repository_environment.resolve_code_version(root)
        or not dependency_profile.formal_ready
    ):
        raise RuntimeError("GPU 资格化必须绑定当前 clean detached 正式提交和依赖锁")

    base_config = build_method_config(root)
    config = replace(
        base_config,
        prompt=prompt_record.prompt_text,
        prompt_id=prompt_record.prompt_id,
        split=prompt_record.split,
        standard_attack_profiles=(),
        diffusion_attacks_enabled=False,
        output_dir=(output_root / "runtime_runs").relative_to(root).as_posix(),
    )
    import torch

    if not torch.cuda.is_available() or not config.device_name.startswith("cuda"):
        raise RuntimeError("GPU 方法资格化禁止在 CPU 或伪 CUDA 环境运行")
    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()
    result = write_semantic_watermark_runtime_outputs(config, root=root)
    wall_time_seconds = time.perf_counter() - started_at
    peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())
    runtime_result = result.to_dict()
    update_records = _read_jsonl(root / result.update_record_path)
    detection_records = _read_jsonl(root / result.detection_record_path)
    binding = _qualification_binding(
        root=root,
        runtime_result=runtime_result,
        config=config,
        prompt_record=prompt_record,
    )
    probe_method_only_gpu_hours = (
        wall_time_seconds
        * RUN_EXPECTED_PROMPT_COUNTS["probe_paper"]
        * len(formal_randomization_repeat_ids())
        / 3600.0
    )
    resource_observation = {
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "single_prompt_wall_time_seconds": wall_time_seconds,
        "estimated_probe_total_gpu_hours": probe_method_only_gpu_hours,
        "estimate_scope": "main_method_prompt_generation_only",
    }
    registered_budget = (
        _read_json((root / args.registered_budget).resolve())
        if args.registered_budget
        else None
    )
    known_answer_path = Path(args.known_answer)
    if not known_answer_path.is_absolute():
        known_answer_path = root / known_answer_path
    report = build_gpu_method_qualification_report(
        runtime_result=runtime_result,
        update_records=update_records,
        detection_records=detection_records,
        config=config,
        known_answer_path=known_answer_path,
        resource_observation=resource_observation,
        registered_budget=registered_budget,
        qualification_binding=binding,
    )
    report_dir = output_root / result.run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = report_dir / "gpu_method_qualification_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    invocation_result = {
        "report_schema": GPU_METHOD_QUALIFICATION_INVOCATION_RESULT_SCHEMA,
        "schema_version": 1,
        "gpu_method_qualification_report_path": report_path.relative_to(
            root
        ).as_posix(),
        "gpu_method_qualification_report_sha256": _file_sha256(report_path),
        "gpu_method_qualification_report_digest": report[
            "qualification_report_digest"
        ],
        "gpu_operator_preflight_ready": report[
            "gpu_operator_preflight_ready"
        ],
        "gpu_resource_budget_ready": report["gpu_resource_budget_ready"],
        "supports_paper_claim": False,
    }
    print(json.dumps(invocation_result, ensure_ascii=False, sort_keys=True))
    return 0 if report["gpu_operator_preflight_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
