"""运行真实单 Prompt 方法路径并自动形成 GPU 方法资格化报告."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import struct
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
from experiments.protocol.image_only_evidence import (
    FrozenEvidenceProtocol,
    apply_frozen_evidence_protocol,
    validate_frozen_evidence_protocol_integrity,
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
    _build_image_only_measurement_config,
    semantic_watermark_runtime_config_digest,
    run_content_runtime_smoke,
    write_semantic_watermark_runtime_outputs,
)
from experiments.protocol.content_routing_reference_quantile import (
    ContentRoutingReferenceScalars,
)
from experiments.protocol.content_routing_reference_registry import (
    load_content_routing_reference_registry,
)
from experiments.runtime import repository_environment
from experiments.runtime.dependency_profiles import require_dependency_profile_ready
from experiments.runtime.scientific_unit_provenance import (
    validate_scientific_unit_provenance,
)
from main.core.digest import build_stable_digest
from main.methods.detection import image_only_measurement_config_identity_record


DEFAULT_KNOWN_ANSWER_PATH = Path(
    "configs/keyed_prg_cross_platform_known_answer.json"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/gpu_method_qualification")
CONTENT_RUNTIME_SMOKE_SCHEMA = "content_runtime_gpu_smoke_v1"


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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """写出由正式冻结协议物化的检测记录。"""

    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _load_frozen_evidence_protocol(
    root: Path,
    requested_path: str | None,
) -> tuple[FrozenEvidenceProtocol, Path]:
    """在任何模型调用前读取并复验完整冻结判定协议。"""

    if type(requested_path) is not str or not requested_path:
        raise ValueError("正式GPU资格化必须显式提供--frozen-evidence-protocol")
    candidate = Path(requested_path).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        protocol = FrozenEvidenceProtocol(**_read_json(path))
        validate_frozen_evidence_protocol_integrity(protocol)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("冻结 evidence protocol 缺失或完整性无效") from exc
    return protocol, path


def _require_frozen_protocol_matches_config(
    protocol: FrozenEvidenceProtocol,
    config: Any,
) -> None:
    """在模型运行前绑定冻结阈值与本次正式检测配置。"""

    measurement_identity = image_only_measurement_config_identity_record(
        _build_image_only_measurement_config(config),
        attention_geometry_enabled=config.attention_geometry_enabled,
        image_alignment_enabled=config.image_alignment_enabled,
    )
    expected = {
        "image_only_measurement_config_digest": measurement_identity[
            "image_only_measurement_config_digest"
        ],
        "lf_carrier_protocol_digest": measurement_identity[
            "lf_carrier_protocol_digest"
        ],
        "tail_carrier_protocol_digest": measurement_identity[
            "tail_carrier_protocol_digest"
        ],
        "lf_weight": measurement_identity["lf_weight"],
        "tail_robust_weight": measurement_identity["tail_robust_weight"],
        "tail_fraction": measurement_identity["tail_fraction"],
        "attention_geometry_enabled": config.attention_geometry_enabled,
        "image_alignment_enabled": config.image_alignment_enabled,
    }
    if any(getattr(protocol, name) != value for name, value in expected.items()):
        raise ValueError("冻结 evidence protocol 与本次正式检测配置不一致")


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
    reference_identity: Mapping[str, Any],
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
        "content_routing_reference_identity": dict(reference_identity),
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
    parser.add_argument("--frozen-evidence-protocol")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--content-runtime-smoke", action="store_true")
    parser.add_argument("--reference-gradient", type=float)
    parser.add_argument("--reference-response", type=float)
    parser.add_argument("--reference-sensitivity", type=float)
    parser.add_argument("--expected-reference-registry-digest")
    parser.add_argument("--expected-reference-registry-file-sha256")
    return parser


def _explicit_smoke_references(arguments: argparse.Namespace) -> tuple[
    ContentRoutingReferenceScalars,
    dict[str, Any],
]:
    """Validate required unqualified binary32 reference inputs."""

    values = {
        "reference_gradient": arguments.reference_gradient,
        "reference_response": arguments.reference_response,
        "reference_sensitivity": arguments.reference_sensitivity,
    }
    bits: dict[str, str] = {}
    for name, value in values.items():
        if type(value) is not float or not math.isfinite(value) or not value > 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be explicit and positive")
        try:
            encoded = struct.pack(">f", value)
            decoded = struct.unpack(">f", encoded)[0]
        except (OverflowError, struct.error) as exc:
            raise ValueError(f"{name} must be exact binary32") from exc
        if decoded != value:
            raise ValueError(f"{name} must be exact binary32")
        bits[name] = encoded.hex()
    identity = {
        "reference_input_role": "explicit_smoke_only_unqualified",
        "reference_values": values,
        "reference_binary32_hex": bits,
        "supports_paper_claim": False,
    }
    return ContentRoutingReferenceScalars(**values), {
        **identity,
        "reference_input_digest": build_stable_digest(identity),
    }


def _fixed_registry_references(arguments: argparse.Namespace) -> tuple[
    ContentRoutingReferenceScalars,
    dict[str, Any],
]:
    """Load the unique promoted registry for formal qualification only."""

    if any(
        value is not None
        for value in (
            arguments.reference_gradient,
            arguments.reference_response,
            arguments.reference_sensitivity,
        )
    ):
        raise ValueError("formal qualification must not accept explicit smoke references")
    semantic_digest = arguments.expected_reference_registry_digest
    file_sha256 = arguments.expected_reference_registry_file_sha256
    references = load_content_routing_reference_registry(
        expected_registry_digest=semantic_digest,
        expected_file_sha256=file_sha256,
    )
    identity = {
        "reference_input_role": "fixed_content_routing_reference_registry",
        "content_routing_reference_registry_digest": semantic_digest,
        "content_routing_reference_registry_file_sha256": file_sha256,
        "reference_values": {
            "reference_gradient": references.reference_gradient,
            "reference_response": references.reference_response,
            "reference_sensitivity": references.reference_sensitivity,
        },
        "supports_paper_claim": False,
    }
    return references, identity


def _write_content_runtime_smoke(
    *,
    root: Path,
    output_root: Path,
    config: Any,
    prompt_record: Any,
    execution_lock: Mapping[str, Any],
    references: ContentRoutingReferenceScalars,
    reference_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Run and persist one real image plus its non-claim smoke diagnostic."""

    image, diagnostic = run_content_runtime_smoke(
        config,
        references,
        verified_formal_execution_lock=execution_lock,
        repository_root=root,
    )
    report_dir = output_root / "content_runtime_smoke" / prompt_record.prompt_id
    report_dir.mkdir(parents=True, exist_ok=False)
    image_path = report_dir / "watermarked.png"
    image.save(image_path)
    report = {
        "report_schema": CONTENT_RUNTIME_SMOKE_SCHEMA,
        "schema_version": 1,
        "smoke_scope": "one_prompt_one_image_one_key_real_sd35_cuda",
        "prompt_id": prompt_record.prompt_id,
        "prompt_digest": prompt_record.prompt_digest,
        "key_material_digest": build_stable_digest(
            {"key_material": config.key_material}
        ),
        "reference_input": dict(reference_identity),
        "runtime_diagnostic": diagnostic,
        "image_path": image_path.relative_to(root).as_posix(),
        "image_sha256": _file_sha256(image_path),
        "content_runtime_smoke_ready": True,
        "supports_paper_claim": False,
    }
    report["content_runtime_smoke_digest"] = build_stable_digest(report)
    report_path = report_dir / "content_runtime_smoke.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "report_schema": "content_runtime_smoke_invocation_v1",
        "schema_version": 1,
        "content_runtime_smoke_report_path": report_path.relative_to(root).as_posix(),
        "content_runtime_smoke_report_sha256": _file_sha256(report_path),
        "content_runtime_smoke_digest": report["content_runtime_smoke_digest"],
        "content_runtime_smoke_ready": True,
        "supports_paper_claim": False,
    }


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
    key_material = os.environ.get("SLM_WM_KEY_MATERIAL")
    if type(key_material) is not str or not key_material:
        raise RuntimeError("formal runtime requires SLM_WM_KEY_MATERIAL")
    config = replace(config, key_material=key_material)
    if args.content_runtime_smoke:
        if (
            args.frozen_evidence_protocol is not None
            or args.expected_reference_registry_digest is not None
            or args.expected_reference_registry_file_sha256 is not None
        ):
            raise ValueError("content runtime smoke 必须保持threshold-free")
        frozen_evidence_protocol = None
        frozen_evidence_protocol_path = None
    else:
        if (
            args.expected_reference_registry_digest is None
            or args.expected_reference_registry_file_sha256 is None
        ):
            raise ValueError("正式GPU资格化必须提供fixed registry双摘要")
        (
            frozen_evidence_protocol,
            frozen_evidence_protocol_path,
        ) = _load_frozen_evidence_protocol(
            root,
            args.frozen_evidence_protocol,
        )
        _require_frozen_protocol_matches_config(
            frozen_evidence_protocol,
            config,
        )
    import torch

    if not torch.cuda.is_available() or not config.device_name.startswith("cuda"):
        raise RuntimeError("GPU 方法资格化禁止在 CPU 或伪 CUDA 环境运行")
    if args.content_runtime_smoke:
        references, reference_identity = _explicit_smoke_references(args)
        invocation = _write_content_runtime_smoke(
            root=root,
            output_root=output_root,
            config=config,
            prompt_record=prompt_record,
            execution_lock=execution_lock,
            references=references,
            reference_identity=reference_identity,
        )
        print(json.dumps(invocation, ensure_ascii=False, sort_keys=True))
        return 0
    references, reference_identity = _fixed_registry_references(args)
    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()
    result = write_semantic_watermark_runtime_outputs(
        config,
        root=root,
        references=references,
        verified_formal_execution_lock=execution_lock,
    )
    wall_time_seconds = time.perf_counter() - started_at
    peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())
    runtime_result = result.to_dict()
    update_records = _read_jsonl(root / result.update_record_path)
    raw_detection_records = _read_jsonl(root / result.detection_record_path)
    if frozen_evidence_protocol is None or frozen_evidence_protocol_path is None:
        raise RuntimeError("正式GPU资格化没有冻结 evidence protocol")
    detection_records = apply_frozen_evidence_protocol(
        raw_detection_records,
        frozen_evidence_protocol,
    )
    report_dir = output_root / result.run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    formal_detection_path = report_dir / "formal_image_only_detection_records.jsonl"
    _write_jsonl(formal_detection_path, detection_records)
    binding = _qualification_binding(
        root=root,
        runtime_result=runtime_result,
        config=config,
        prompt_record=prompt_record,
        reference_identity=reference_identity,
    )
    binding_payload = {
        field_name: value
        for field_name, value in binding.items()
        if field_name != "qualification_binding_digest"
    }
    binding_payload["frozen_evidence_protocol_identity"] = {
        "source_path": frozen_evidence_protocol_path.as_posix(),
        "source_file_sha256": _file_sha256(frozen_evidence_protocol_path),
        "threshold_digest": frozen_evidence_protocol.threshold_digest,
        "image_only_measurement_config_digest": (
            frozen_evidence_protocol.image_only_measurement_config_digest
        ),
    }
    binding_payload["formal_detection_records_identity"] = {
        "path": formal_detection_path.relative_to(root).as_posix(),
        "file_sha256": _file_sha256(formal_detection_path),
        "record_count": len(detection_records),
    }
    binding = {
        **binding_payload,
        "qualification_binding_digest": build_stable_digest(binding_payload),
    }
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
        frozen_evidence_protocol=frozen_evidence_protocol,
    )
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
        "frozen_threshold_digest": frozen_evidence_protocol.threshold_digest,
        "formal_detection_record_path": formal_detection_path.relative_to(
            root
        ).as_posix(),
        "formal_detection_record_sha256": _file_sha256(formal_detection_path),
        "supports_paper_claim": False,
    }
    print(json.dumps(invocation_result, ensure_ascii=False, sort_keys=True))
    return 0 if report["gpu_operator_preflight_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
