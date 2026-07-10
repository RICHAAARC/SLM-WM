"""T2SMark full-main 真实 GPU 复现入口的 Colab 辅助函数。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from paper_experiments.baselines import (
    build_fixed_fpr_operating_point,
    build_t2smark_full_main_candidate_records,
    resolve_full_main_prompt_protocol_name,
    validate_primary_baseline_formal_import_rows,
)
from paper_experiments.baselines.t2smark_pair_quality import write_t2smark_strict_pair_quality_outputs
from experiments.protocol.pilot_paper_fixed_fpr import (
    PILOT_PAPER_FIXED_FPR,
)
from experiments.protocol.paper_run_config import build_paper_run_config, resolve_count_from_environment
from experiments.protocol.prompts import build_prompt_records, normalize_prompt_text
from experiments.protocol.splits import apply_split_assignments
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from paper_experiments.runners.external_baseline_method_faithful import (
    DEFAULT_T2SMARK_MODEL_ID,
    DEFAULT_T2SMARK_SOURCE_ENTRY,
    ensure_cuda_if_requested,
    ensure_t2smark_source_available,
    run_command,
)
from experiments.runtime.progress import call_runner_with_progress_status, emit_progress_status, progress_bar, update_progress
from experiments.runtime.repository_environment import (
    build_runtime_environment_report,
    file_digest,
    resolve_code_version,
)

DEFAULT_OUTPUT_DIR = "outputs/t2smark_full_main_reproduction"
DEFAULT_DRIVE_OUTPUT_DIR = ""
DEFAULT_PROMPT_FILE = "configs/paper_main_pilot_paper_prompts.txt"
DEFAULT_RUN_NAME = "t2smark_sd35_medium_pilot_paper"
DEFAULT_TARGET_FPR = PILOT_PAPER_FIXED_FPR
DEFAULT_PROMPT_LIMIT = 700
DEFAULT_T2SMARK_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_T2SMARK_LPIPS_NETWORK = "alex"
PACKAGE_EXTRA_PATHS = (
    "paper_experiments/runners/t2smark_full_main_reproduction.py",
    "paper_experiments/baselines/t2smark_pair_quality.py",
    "paper_experiments/baselines/formal_import.py",
    "scripts/write_primary_baseline_formal_import_protocol.py",
    "external_baseline/primary/t2smark/adapter/run_slm_eval.py",
    "external_baseline/primary/t2smark/source/run_sd35.py",
    "external_baseline/primary/t2smark/source/option.py",
)


@dataclass(frozen=True)
class T2SMarkFullMainReproductionConfig:
    """描述 T2SMark full-main 真实复现所需的最小配置。"""

    output_dir: str = DEFAULT_OUTPUT_DIR
    drive_output_dir: str = field(
        default_factory=lambda: build_paper_run_config(".").drive_dir("external_baseline_official_reference")
    )
    prompt_set: str = "pilot_paper"
    prompt_file: str = DEFAULT_PROMPT_FILE
    t2smark_run_name: str = DEFAULT_RUN_NAME
    model_id: str = DEFAULT_T2SMARK_MODEL_ID
    seed: int = 20260621
    prompt_limit: int = DEFAULT_PROMPT_LIMIT
    clip_test_num: int = 0
    num_inference_steps: int = 28
    num_inversion_steps: int = 28
    guidance_scale: float = 4.0
    target_fpr: float = DEFAULT_TARGET_FPR
    minimum_prompt_protocol_count: int = 100
    fixed_fpr_baseline_calibration_ready: bool = False
    attack_matrix_baseline_detection_ready: bool = False
    reuse_existing: bool = True
    force_generate: bool = False
    save_image: bool = True
    save_clean_pair: bool = True
    enable_pair_perceptual_metrics: bool = True
    pair_clip_model_id: str = DEFAULT_T2SMARK_CLIP_MODEL_ID
    pair_lpips_network: str = DEFAULT_T2SMARK_LPIPS_NETWORK
    pair_perceptual_metric_device_name: str = "cpu"
    require_cuda: bool = True
    timeout_seconds: int = 86400
    enable_workflow_progress_bar: bool = True


@dataclass(frozen=True)
class T2SMarkFullMainArchiveRecord:
    """记录 T2SMark full-main 复现压缩包及 Drive 镜像信息。"""

    archive_path: str
    archive_digest: str
    archive_entry_count: int
    drive_archive_path: str
    drive_archive_digest: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return asdict(self)


def stable_json_text(value: Any) -> str:
    """以稳定顺序序列化 JSON。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: dict[str, Any]) -> str:
    """把字典转换为 JSONL 单行。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def write_json(path: Path, payload: Any) -> None:
    """写出 JSON 文件并创建父目录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json_text(payload), encoding="utf-8")


def read_json(path: Path) -> Any:
    """读取 JSON 文件。"""

    return json.loads(path.read_text(encoding="utf-8-sig"))


def run_command_with_progress_status(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    progress: object | None = None,
    progress_profile: str = "",
) -> dict[str, Any]:
    """调用可被测试替换的命令 runner, 同时在真实 Colab 中保留进度状态。"""

    return call_runner_with_progress_status(
        run_command,
        command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        progress=progress,
        progress_profile=progress_profile,
    )


def synchronize_environment_report_with_device_report(
    environment_report: dict[str, Any],
    device_report: dict[str, Any],
) -> dict[str, Any]:
    """把显式 GPU 检查结果同步到环境报告顶层字段。

    该函数属于项目特定写法: Colab helper 已经在运行入口检查真实 GPU, 因此环境报告顶层字段应直接反映该检查结果,
    避免审计时出现顶层 `cuda_available=null` 但子节点显示 CUDA 可用的歧义。
    """

    merged_report = dict(environment_report)
    merged_report["t2smark_full_main_device_report"] = dict(device_report)
    if "cuda_available" in device_report:
        merged_report["cuda_available"] = bool(device_report["cuda_available"])
    if "device_count" in device_report:
        merged_report["device_count"] = int(device_report["device_count"])
    device_name = str(device_report.get("device_name") or device_report.get("gpu_name") or "")
    if device_name:
        merged_report["gpu_name"] = device_name
    if device_report.get("cuda_version"):
        merged_report["cuda_version"] = str(device_report["cuda_version"])
    return merged_report


def build_t2smark_full_main_environment_report(device_report: dict[str, Any]) -> dict[str, Any]:
    """构造与 T2SMark full-main GPU 检查结果一致的环境报告。"""

    try:
        import torch
    except Exception:  # pragma: no cover - 本地轻量测试不强制安装 torch
        environment_report = build_runtime_environment_report()
    else:
        environment_report = build_runtime_environment_report(torch_module=torch)
    return synchronize_environment_report_with_device_report(environment_report, device_report)


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先记录相对仓库根目录的路径。"""

    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def output_paths(root_path: Path, config: T2SMarkFullMainReproductionConfig) -> dict[str, Path]:
    """集中构造 T2SMark full-main 复现所需路径。"""

    output_dir = (root_path / config.output_dir).resolve()
    official_root = output_dir / "t2smark_official"
    official_run_dir = official_root / config.t2smark_run_name
    adapter_dir = output_dir / "t2smark_adapter"
    return {
        "output_dir": output_dir,
        "official_root": official_root,
        "official_run_dir": official_run_dir,
        "official_results": official_run_dir / "results.json",
        "official_settings": official_run_dir / "settings.json",
        "official_images": official_run_dir / "images",
        "prompt_dataset": output_dir / "t2smark_full_main_prompt_dataset.json",
        "prompt_plan": output_dir / "t2smark_full_main_prompt_plan.json",
        "image_pairs": output_dir / "t2smark_full_main_image_pairs.json",
        "pair_quality_metrics": output_dir / "t2smark_full_main_strict_pair_quality_metrics.csv",
        "pair_quality_summary": output_dir / "t2smark_full_main_strict_pair_quality_summary.json",
        "adapter_observations": adapter_dir / "baseline_observations.json",
        "adapter_manifest": adapter_dir / "t2smark_slm_adapter_manifest.json",
        "candidate_records": output_dir / "t2smark_full_main_formal_import_candidate_records.jsonl",
        "validation_report": output_dir / "t2smark_full_main_formal_import_validation_report.json",
        "environment_report": output_dir / "t2smark_full_main_environment_report.json",
        "summary": output_dir / "t2smark_full_main_reproduction_summary.json",
        "manifest": output_dir / "t2smark_full_main_reproduction_manifest.local.json",
    }


def read_prompt_texts(prompt_file: str | Path) -> tuple[str, ...]:
    """读取 prompt 文件, 忽略空行与注释行。"""

    prompts: list[str] = []
    for line in Path(prompt_file).read_text(encoding="utf-8").splitlines():
        text = normalize_prompt_text(line)
        if text and not text.startswith("#"):
            prompts.append(text)
    return tuple(prompts)


def selected_prompt_texts(prompt_texts: tuple[str, ...], prompt_limit: int) -> tuple[str, ...]:
    """按论文运行上限截取 prompt, 0 表示使用全部 prompt。"""

    if int(prompt_limit) <= 0:
        return prompt_texts
    return prompt_texts[: int(prompt_limit)]


def build_full_main_prompt_rows(prompt_set: str, prompt_texts: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    """构造 T2SMark 运行使用的 prompt 计划。"""

    rows: list[dict[str, Any]] = []
    records = apply_split_assignments(build_prompt_records(prompt_set, prompt_texts))
    for record in records:
        rows.append(
            {
                "prompt_id": record.prompt_id,
                "prompt_index": record.prompt_index,
                "prompt_set": record.prompt_set,
                "split": record.split,
                "prompt_text": record.prompt_text,
                "prompt_digest": record.prompt_digest,
            }
        )
    return tuple(rows)


def write_full_main_prompt_inputs(
    root_path: Path,
    config: T2SMarkFullMainReproductionConfig,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """写出 T2SMark 官方入口与 adapter 共享的 prompt 输入。"""

    prompt_source_path = root_path / config.prompt_file
    all_prompt_texts = read_prompt_texts(prompt_source_path)
    chosen_prompt_texts = selected_prompt_texts(all_prompt_texts, config.prompt_limit)
    prompt_rows = build_full_main_prompt_rows(config.prompt_set, chosen_prompt_texts)
    prompt_protocol_ready = bool(prompt_rows) and len(prompt_rows) >= int(config.minimum_prompt_protocol_count)
    dataset_payload = {
        "annotations": [
            {
                "caption": row["prompt_text"],
                "prompt_id": row["prompt_id"],
                "prompt_index": row["prompt_index"],
            }
            for row in prompt_rows
        ]
    }
    write_json(paths["prompt_dataset"], dataset_payload)
    write_json(paths["prompt_plan"], list(prompt_rows))
    report = {
        "full_main_prompt_source_path": relative_or_absolute(prompt_source_path, root_path),
        "full_main_prompt_count": len(all_prompt_texts),
        "selected_prompt_count": len(prompt_rows),
        "prompt_limit": int(config.prompt_limit),
        "minimum_prompt_protocol_count": int(config.minimum_prompt_protocol_count),
        "full_main_prompt_protocol_ready": prompt_protocol_ready,
        "pilot_paper_prompt_protocol_ready": prompt_protocol_ready,
        "paper_claim_scale": config.prompt_set,
        "prompt_protocol_name": resolve_full_main_prompt_protocol_name(root_path),
        "prompt_protocol_digest": build_stable_digest([row["prompt_digest"] for row in prompt_rows]),
        "prompt_dataset_path": relative_or_absolute(paths["prompt_dataset"], root_path),
        "prompt_plan_path": relative_or_absolute(paths["prompt_plan"], root_path),
    }
    return report


def should_run_official(config: T2SMarkFullMainReproductionConfig, results_path: Path) -> tuple[bool, str]:
    """判断官方 T2SMark full-main 运行是否需要本次生成。"""

    if config.force_generate:
        return True, "force_generate_requested"
    if config.reuse_existing and results_path.is_file():
        if config.save_clean_pair and count_t2smark_pair_quality_items(results_path) < max(1, int(config.prompt_limit)):
            return True, "existing_results_pair_quality_count_insufficient"
        return False, "existing_results_found"
    return True, "results_missing"


def count_t2smark_pair_quality_items(results_path: Path) -> int:
    """统计 T2SMark full-main 结果中严格 pair-level 质量证据数量。"""

    if not results_path.is_file():
        return 0
    try:
        payload = read_json(results_path)
    except Exception:
        return 0
    if not isinstance(payload, dict):
        return 0
    count = 0
    for key, value in payload.items():
        if not str(key).isdigit() or not isinstance(value, dict):
            continue
        pair_quality = value.get("pair_quality")
        if not isinstance(pair_quality, dict):
            continue
        sample_name = f"{int(key):05d}.png"
        clean_path = Path(str(pair_quality.get("clean_image_path") or ""))
        watermarked_path = Path(str(pair_quality.get("watermarked_image_path") or ""))
        clean_exists = clean_path.is_file() or (results_path.parent / "quality_pairs" / "clean" / sample_name).is_file()
        watermarked_exists = watermarked_path.is_file() or (results_path.parent / "images" / sample_name).is_file()
        if (
            pair_quality.get("pair_quality_protocol") == "strict_clean_watermarked_pair"
            and pair_quality.get("clean_image_digest")
            and pair_quality.get("watermarked_image_digest")
            and clean_exists
            and watermarked_exists
        ):
            count += 1
    return count


def run_t2smark_official_if_needed(
    root_path: Path,
    config: T2SMarkFullMainReproductionConfig,
    paths: dict[str, Path],
    prompt_report: dict[str, Any],
    progress: object | None = None,
) -> dict[str, Any]:
    """运行或复用 T2SMark 官方 SD3.5 Medium full-main 结果。"""

    paths["official_root"].mkdir(parents=True, exist_ok=True)
    should_run, reason = should_run_official(config, paths["official_results"])
    if not should_run:
        return {
            "official_result_generated": False,
            "official_result_reused": True,
            "official_generation_reason": reason,
            "official_results_path": relative_or_absolute(paths["official_results"], root_path),
            "official_return_code": 0,
            "official_command": [],
            "source_report": {"source_prepare_skipped": True},
        }
    source_report = ensure_t2smark_source_available(root_path, paths, timeout_seconds=300, progress=progress)
    ensure_cuda_if_requested(config.require_cuda)
    source_entry = root_path / DEFAULT_T2SMARK_SOURCE_ENTRY
    command = [
        sys.executable,
        str(source_entry),
        "--name",
        config.t2smark_run_name,
        "--output_dir",
        str(paths["official_root"]),
        "--seed",
        str(config.seed),
        "--robust_test_num",
        str(prompt_report["selected_prompt_count"]),
        "--clip_test_num",
        str(config.clip_test_num),
        "--dataset_key",
        str(paths["prompt_dataset"]),
        "--model_key",
        config.model_id,
        "--guidance_scale",
        str(config.guidance_scale),
        "--num_inference_steps",
        str(config.num_inference_steps),
        "--num_inversion_steps",
        str(config.num_inversion_steps),
        "--fix_key",
        "--SDv35M",
    ]
    if config.save_clean_pair:
        command.extend(
            [
                "--slm_save_clean_pair",
                "--slm_pair_image_dir",
                str(paths["official_run_dir"] / "quality_pairs"),
            ]
        )
    if config.save_image:
        command.append("--save_image")
    result = run_command_with_progress_status(
        command,
        cwd=root_path,
        timeout_seconds=config.timeout_seconds,
        progress=progress,
        progress_profile=f"operation=t2smark_full_main_official_reference samples={prompt_report['selected_prompt_count']}",
    )
    write_json(paths["output_dir"] / "t2smark_full_main_official_command_result.json", result)
    return {
        "official_result_generated": result["return_code"] == 0,
        "official_result_reused": False,
        "official_generation_reason": reason if result["return_code"] == 0 else "official_command_failed",
        "official_results_path": relative_or_absolute(paths["official_results"], root_path),
        "official_return_code": result["return_code"],
        "official_command": command,
        "source_report": source_report,
    }


def build_t2smark_full_main_image_pairs(
    root_path: Path,
    paths: dict[str, Path],
    prompt_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按 full-main prompt 计划和官方图像目录生成 image_pairs。"""

    rows: list[dict[str, Any]] = []
    for index, prompt_row in enumerate(prompt_rows):
        image_path = paths["official_images"] / f"{index:05d}.png"
        clean_image_path = paths["official_run_dir"] / "quality_pairs" / "clean" / f"{index:05d}.png"
        image_id = f"t2smark_full_main_{index:05d}"
        watermarked_image_digest = file_digest(image_path) if image_path.is_file() else ""
        clean_image_digest = file_digest(clean_image_path) if clean_image_path.is_file() else ""
        rows.append(
            {
                "image_id": image_id,
                "event_id": image_id,
                "prompt_id": str(prompt_row["prompt_id"]),
                "prompt_index": int(prompt_row["prompt_index"]),
                "prompt_set": str(prompt_row.get("prompt_set", "")),
                "split": str(prompt_row.get("split", "test")),
                "baseline_id": "t2smark",
                "generated_image_path": relative_or_absolute(image_path, root_path) if image_path.is_file() else "",
                "generated_image_digest": watermarked_image_digest,
                "clean_image_path": relative_or_absolute(clean_image_path, root_path) if clean_image_path.is_file() else "",
                "clean_image_digest": clean_image_digest,
                "watermarked_image_path": relative_or_absolute(image_path, root_path) if image_path.is_file() else "",
                "watermarked_image_digest": watermarked_image_digest,
                "pair_quality_protocol": "strict_clean_watermarked_pair",
                "strict_pair_quality_ready": bool(clean_image_digest and watermarked_image_digest),
            }
        )
    write_json(paths["image_pairs"], rows)
    return rows


def run_t2smark_adapter(
    root_path: Path,
    config: T2SMarkFullMainReproductionConfig,
    paths: dict[str, Path],
    progress: object | None = None,
) -> dict[str, Any]:
    """把 T2SMark 官方结果转换为项目统一 baseline observations。"""

    adapter_path = root_path / "external_baseline/primary/t2smark/adapter/run_slm_eval.py"
    command = [
        sys.executable,
        str(adapter_path),
        "--image-pairs",
        str(paths["image_pairs"]),
        "--t2smark-results",
        str(paths["official_results"]),
        "--out",
        str(paths["adapter_observations"]),
        "--artifact-root",
        str(paths["adapter_observations"].parent / "artifacts"),
        "--model-id",
        config.model_id,
        "--seed",
        str(config.seed),
    ]
    if config.require_cuda:
        command.append("--require-cuda")
    result = run_command_with_progress_status(
        command,
        cwd=root_path,
        timeout_seconds=config.timeout_seconds,
        progress=progress,
        progress_profile="operation=t2smark_full_main_adapter",
    )
    write_json(paths["output_dir"] / "t2smark_full_main_adapter_command_result.json", result)
    return {
        "adapter_return_code": result["return_code"],
        "adapter_command": command,
        "adapter_observations_path": relative_or_absolute(paths["adapter_observations"], root_path),
    }


def build_candidate_records_and_validation(
    root_path: Path,
    config: T2SMarkFullMainReproductionConfig,
    paths: dict[str, Path],
    prompt_report: dict[str, Any],
) -> dict[str, Any]:
    """从 T2SMark full-main observations 构造正式导入候选并运行 schema 校验。"""

    observations = read_json(paths["adapter_observations"]) if paths["adapter_observations"].is_file() else []
    evidence_paths = [
        relative_or_absolute(paths["official_results"], root_path),
        relative_or_absolute(paths["image_pairs"], root_path),
        relative_or_absolute(paths["adapter_observations"], root_path),
        relative_or_absolute(paths["prompt_dataset"], root_path),
        relative_or_absolute(paths["prompt_plan"], root_path),
    ]
    candidate_records = build_t2smark_full_main_candidate_records(
        observation_rows=observations,
        target_fpr=config.target_fpr,
        baseline_result_source=relative_or_absolute(paths["official_results"], root_path),
        baseline_result_source_digest=file_digest(paths["official_results"]) if paths["official_results"].is_file() else "",
        evidence_paths=evidence_paths,
        prompt_protocol_digest=str(prompt_report["prompt_protocol_digest"]),
        full_main_prompt_protocol_ready=bool(prompt_report["full_main_prompt_protocol_ready"]),
        fixed_fpr_baseline_calibration_ready=bool(config.fixed_fpr_baseline_calibration_ready),
        attack_matrix_baseline_detection_ready=bool(config.attack_matrix_baseline_detection_ready),
    )
    paths["candidate_records"].write_text("".join(json_line(row) for row in candidate_records), encoding="utf-8")
    validation_report = validate_primary_baseline_formal_import_rows(
        candidate_records,
        evidence_root=root_path,
        target_fpr=config.target_fpr,
        require_existing_evidence=True,
    )
    write_json(paths["validation_report"], validation_report)
    return {
        "candidate_record_count": len(candidate_records),
        "validation_report": validation_report,
        "formal_import_candidate_records_path": relative_or_absolute(paths["candidate_records"], root_path),
        "formal_import_validation_report_path": relative_or_absolute(paths["validation_report"], root_path),
    }


def write_failure_outputs(
    root_path: Path,
    config: T2SMarkFullMainReproductionConfig,
    paths: dict[str, Path],
    error: Exception,
) -> dict[str, Any]:
    """在 full-main 复现失败时写出可打包诊断产物。"""

    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    environment_report = build_runtime_environment_report()
    write_json(paths["environment_report"], environment_report)
    summary = {
        "run_decision": "fail",
        "t2smark_full_main_reproduction_ready": False,
        "formal_import_validation_ready": False,
        "unsupported_reason": f"{type(error).__name__}:{error}",
        "supports_paper_claim": False,
        "environment_report_path": relative_or_absolute(paths["environment_report"], root_path),
        "manifest_path": relative_or_absolute(paths["manifest"], root_path),
    }
    write_json(paths["summary"], summary)
    manifest = build_artifact_manifest(
        artifact_id="t2smark_full_main_reproduction_manifest",
        artifact_type="local_manifest",
        input_paths=(),
        output_paths=(relative_or_absolute(paths["summary"], root_path), relative_or_absolute(paths["environment_report"], root_path)),
        config=asdict(config),
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 paper_experiments.runners.t2smark_full_main_reproduction",
        metadata={"run_decision": "fail", "supports_paper_claim": False},
    ).to_dict()
    write_json(paths["manifest"], manifest)
    return summary


def write_t2smark_full_main_reproduction_outputs(
    config: T2SMarkFullMainReproductionConfig,
    root: str | Path = ".",
) -> dict[str, Any]:
    """运行 T2SMark full-main 真实复现路径并写出 summary 与 manifest。"""

    root_path = Path(root).resolve()
    paths = output_paths(root_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    try:
        with progress_bar(8, desc="t2smark full main reproduction", enabled=config.enable_workflow_progress_bar) as run_progress:
            emit_progress_status(run_progress, profile="operation=ensure_cuda status=running")
            device_report = ensure_cuda_if_requested(config.require_cuda)
            update_progress(run_progress, profile="operation=ensure_cuda")
            emit_progress_status(run_progress, profile="operation=write_prompt_inputs status=running")
            prompt_report = write_full_main_prompt_inputs(root_path, config, paths)
            prompt_rows = read_json(paths["prompt_plan"])
            update_progress(run_progress, profile=f"operation=write_prompt_inputs prompts={prompt_report['selected_prompt_count']}")
            official_report = run_t2smark_official_if_needed(root_path, config, paths, prompt_report, progress=run_progress)
            update_progress(run_progress, profile="operation=t2smark_full_main_official_reference")
            emit_progress_status(run_progress, profile="operation=build_image_pairs status=running")
            image_pairs = build_t2smark_full_main_image_pairs(root_path, paths, prompt_rows)
            update_progress(run_progress, profile=f"operation=build_image_pairs pairs={len(image_pairs)}")
            emit_progress_status(run_progress, profile="operation=t2smark_pair_quality status=running")
            pair_quality_report = write_t2smark_strict_pair_quality_outputs(
                root_path=root_path,
                image_pairs_path=paths["image_pairs"],
                metrics_path=paths["pair_quality_metrics"],
                summary_path=paths["pair_quality_summary"],
                enable_pair_perceptual_metrics=config.enable_pair_perceptual_metrics,
                clip_model_id=config.pair_clip_model_id,
                lpips_network=config.pair_lpips_network,
                perceptual_metric_device_name=config.pair_perceptual_metric_device_name,
            )
            update_progress(
                run_progress,
                profile=(
                    "operation=t2smark_pair_quality "
                    f"measured={pair_quality_report['measured_strict_pair_quality_count']}"
                ),
            )
            adapter_report = run_t2smark_adapter(root_path, config, paths, progress=run_progress)
            update_progress(run_progress, profile="operation=t2smark_full_main_adapter")
            emit_progress_status(run_progress, profile="operation=build_candidate_records status=running")
            candidate_report = build_candidate_records_and_validation(root_path, config, paths, prompt_report)
            update_progress(run_progress, profile=f"operation=build_candidate_records records={candidate_report['candidate_record_count']}")
            emit_progress_status(run_progress, profile="operation=write_environment_report status=running")
            environment_report = build_t2smark_full_main_environment_report(device_report)
            write_json(paths["environment_report"], environment_report)
            update_progress(run_progress, profile="operation=write_environment_report")
    except Exception as error:
        return write_failure_outputs(root_path, config, paths, error)

    official_ready = paths["official_results"].is_file() and official_report.get("official_return_code") == 0
    adapter_ready = paths["adapter_observations"].is_file() and adapter_report.get("adapter_return_code") == 0
    pair_quality_ready = bool(pair_quality_report.get("strict_pair_quality_ready", False))
    run_ready = bool(
        official_ready
        and adapter_ready
        and prompt_report["selected_prompt_count"] > 0
        and (not config.save_clean_pair or pair_quality_ready)
    )
    validation_report = candidate_report["validation_report"]
    summary = {
        "run_decision": "pass" if run_ready else "fail",
        "t2smark_full_main_reproduction_ready": run_ready,
        "t2smark_official_result_generated": bool(official_report.get("official_result_generated")),
        "t2smark_official_result_reused": bool(official_report.get("official_result_reused")),
        "full_main_prompt_count": int(prompt_report["full_main_prompt_count"]),
        "paper_prompt_count": int(prompt_report["full_main_prompt_count"]),
        "selected_prompt_count": int(prompt_report["selected_prompt_count"]),
        "prompt_limit": int(config.prompt_limit),
        "full_main_prompt_protocol_ready": bool(prompt_report["full_main_prompt_protocol_ready"]),
        "pilot_paper_prompt_protocol_ready": bool(prompt_report["pilot_paper_prompt_protocol_ready"]),
        "image_pair_count": len(image_pairs),
        "t2smark_strict_pair_quality_ready": pair_quality_ready,
        "t2smark_strict_pair_quality_count": int(pair_quality_report.get("measured_strict_pair_quality_count", 0)),
        "t2smark_strict_pair_quality_metrics_path": relative_or_absolute(paths["pair_quality_metrics"], root_path),
        "t2smark_strict_pair_quality_summary_path": relative_or_absolute(paths["pair_quality_summary"], root_path),
        "formal_import_candidate_record_count": int(candidate_report["candidate_record_count"]),
        "accepted_formal_import_count": int(validation_report.get("accepted_formal_import_count", 0)),
        "rejected_formal_import_count": int(validation_report.get("rejected_formal_import_count", 0)),
        "formal_import_validation_ready": bool(validation_report.get("formal_import_validation_ready", False)),
        "target_fpr": float(config.target_fpr),
        "comparable_operating_point": build_fixed_fpr_operating_point(config.target_fpr),
        "official_results_path": relative_or_absolute(paths["official_results"], root_path),
        "image_pairs_path": relative_or_absolute(paths["image_pairs"], root_path),
        "formal_import_candidate_records_path": candidate_report["formal_import_candidate_records_path"],
        "formal_import_validation_report_path": candidate_report["formal_import_validation_report_path"],
        "environment_report_path": relative_or_absolute(paths["environment_report"], root_path),
        "manifest_path": relative_or_absolute(paths["manifest"], root_path),
        "supports_paper_claim": False,
        "unsupported_reason": "" if run_ready else "t2smark_full_main_reproduction_incomplete",
        "paper_claim_scale": config.prompt_set,
        "metadata": {
            "prompt_report": prompt_report,
            "official_report": official_report,
            "adapter_report": adapter_report,
            "t2smark_pair_quality_report": pair_quality_report,
            "claim_boundary": "pilot_paper_raw_reproduction_requires_formal_import_validation_and_attack_matrix_closure",
        },
    }
    write_json(paths["summary"], summary)
    output_paths_for_manifest = [
        relative_or_absolute(paths["summary"], root_path),
        relative_or_absolute(paths["environment_report"], root_path),
        relative_or_absolute(paths["image_pairs"], root_path),
        relative_or_absolute(paths["candidate_records"], root_path),
        relative_or_absolute(paths["validation_report"], root_path),
        relative_or_absolute(paths["pair_quality_metrics"], root_path),
        relative_or_absolute(paths["pair_quality_summary"], root_path),
    ]
    if paths["adapter_observations"].exists():
        output_paths_for_manifest.append(relative_or_absolute(paths["adapter_observations"], root_path))
    manifest = build_artifact_manifest(
        artifact_id="t2smark_full_main_reproduction_manifest",
        artifact_type="local_manifest",
        input_paths=(relative_or_absolute(root_path / config.prompt_file, root_path),),
        output_paths=tuple(output_paths_for_manifest + [relative_or_absolute(paths["manifest"], root_path)]),
        config=asdict(config),
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 paper_experiments.runners.t2smark_full_main_reproduction",
        metadata={
            "run_decision": summary["run_decision"],
            "t2smark_full_main_reproduction_ready": run_ready,
            "formal_import_validation_ready": summary["formal_import_validation_ready"],
            "t2smark_strict_pair_quality_ready": pair_quality_ready,
            "t2smark_strict_pair_quality_count": summary["t2smark_strict_pair_quality_count"],
            "supports_paper_claim": False,
        },
    ).to_dict()
    write_json(paths["manifest"], manifest)
    return summary


def build_default_config() -> T2SMarkFullMainReproductionConfig:
    """从环境变量构造默认 Colab 运行配置。"""

    paper_run = build_paper_run_config(".")
    return T2SMarkFullMainReproductionConfig(
        output_dir=os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        drive_output_dir=os.environ.get(
            "SLM_WM_T2SMARK_FULL_MAIN_DRIVE_OUTPUT_DIR",
            paper_run.drive_dir("external_baseline_official_reference"),
        ),
        prompt_set=os.environ.get("SLM_WM_PROMPT_SET", paper_run.prompt_set),
        prompt_file=os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_PROMPT_FILE", paper_run.prompt_file),
        t2smark_run_name=os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_RUN_NAME", DEFAULT_RUN_NAME),
        model_id=os.environ.get("SLM_WM_T2SMARK_MODEL_ID", DEFAULT_T2SMARK_MODEL_ID),
        seed=int(os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_SEED", "20260621")),
        prompt_limit=resolve_count_from_environment(
            "SLM_WM_T2SMARK_FULL_MAIN_PROMPT_LIMIT",
            default_value=paper_run.sample_count,
        ),
        clip_test_num=int(os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_CLIP_TEST_NUM", "0")),
        num_inference_steps=int(os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_NUM_INFERENCE_STEPS", "28")),
        num_inversion_steps=int(os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_NUM_INVERSION_STEPS", "28")),
        guidance_scale=float(os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_GUIDANCE_SCALE", "4.0")),
        target_fpr=float(os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_TARGET_FPR", str(paper_run.target_fpr))),
        minimum_prompt_protocol_count=paper_run.minimum_clean_negative_count,
        fixed_fpr_baseline_calibration_ready=os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_FIXED_FPR_READY", "0") == "1",
        attack_matrix_baseline_detection_ready=os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_ATTACK_MATRIX_READY", "0") == "1",
        reuse_existing=os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_REUSE_EXISTING", "1") != "0",
        force_generate=os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_FORCE_GENERATE", "0") == "1",
        save_image=os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_SAVE_IMAGE", "1") != "0",
        save_clean_pair=os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_SAVE_CLEAN_PAIR", "1") != "0",
        enable_pair_perceptual_metrics=os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_PAIR_PERCEPTUAL_METRICS", "1") != "0",
        pair_clip_model_id=os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_PAIR_CLIP_MODEL_ID", DEFAULT_T2SMARK_CLIP_MODEL_ID),
        pair_lpips_network=os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_PAIR_LPIPS_NETWORK", DEFAULT_T2SMARK_LPIPS_NETWORK),
        pair_perceptual_metric_device_name=os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_PAIR_PERCEPTUAL_DEVICE", "cpu"),
        require_cuda=os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_REQUIRE_CUDA", "1") != "0",
        timeout_seconds=int(os.environ.get("SLM_WM_T2SMARK_FULL_MAIN_TIMEOUT_SECONDS", "86400")),
        enable_workflow_progress_bar=os.environ.get("SLM_WM_ENABLE_WORKFLOW_PROGRESS_BAR", "1") != "0",
    )


def run_default_t2smark_full_main_reproduction_plan(root: str | Path = ".") -> dict[str, Any]:
    """运行默认 T2SMark full-main 真实复现计划。"""

    return write_t2smark_full_main_reproduction_outputs(config=build_default_config(), root=root)


def collect_package_entries(root_path: Path, output_dir: Path, archive_path: Path) -> tuple[Path, ...]:
    """收集需要进入压缩包的核对文件。"""

    entries: list[Path] = []
    if output_dir.exists():
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.resolve() != archive_path.resolve() and path.suffix.lower() != ".zip":
                entries.append(path)
    for relative_path in PACKAGE_EXTRA_PATHS:
        path = root_path / relative_path
        if path.exists():
            entries.append(path)
    unique_entries: list[Path] = []
    for entry in entries:
        if entry not in unique_entries:
            unique_entries.append(entry)
    return tuple(unique_entries)


def package_t2smark_full_main_reproduction_outputs(
    root: str | Path = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    drive_output_dir: str | None = None,
    archive_name: str = "external_baseline_official_reference_package_t2smark.zip",
) -> T2SMarkFullMainArchiveRecord:
    """打包 T2SMark full-main 复现产物并镜像到 Google Drive。"""

    root_path = Path(root).resolve()
    resolved_drive_output_dir = drive_output_dir or build_paper_run_config(root_path).drive_dir(
        "external_baseline_official_reference"
    )
    source_dir = (root_path / output_dir).resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_path = source_dir / archive_name
    package_manifest_path = source_dir / "t2smark_full_main_package_input_manifest.json"
    summary_path = source_dir / "t2smark_full_main_archive_summary.json"
    manifest_path = source_dir / "t2smark_full_main_archive_manifest.local.json"
    for stale_path in (package_manifest_path, summary_path, manifest_path):
        if stale_path.exists():
            stale_path.unlink()
    entries = collect_package_entries(root_path, source_dir, archive_path)
    package_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entry_paths": [entry.relative_to(root_path).as_posix() for entry in entries],
        "entry_count": len(entries),
    }
    write_json(package_manifest_path, package_manifest)
    archive_manifest = build_artifact_manifest(
        artifact_id="t2smark_full_main_archive_manifest",
        artifact_type="local_manifest",
        input_paths=tuple([entry.relative_to(root_path).as_posix() for entry in entries] + [package_manifest_path.relative_to(root_path).as_posix()]),
        output_paths=(
            archive_path.relative_to(root_path).as_posix(),
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ),
        config={"archive_name": archive_name, "drive_output_dir": str(Path(resolved_drive_output_dir).expanduser())},
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 paper_experiments.runners.t2smark_full_main_reproduction",
        metadata={"construction_unit_name": "t2smark_full_main_reproduction", "generated_at": datetime.now(timezone.utc).isoformat()},
    ).to_dict()
    write_json(manifest_path, archive_manifest)
    entries = collect_package_entries(root_path, source_dir, archive_path)
    with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.write(entry, entry.relative_to(root_path).as_posix())
    drive_dir = Path(resolved_drive_output_dir).expanduser()
    drive_dir.mkdir(parents=True, exist_ok=True)
    mirrored_path = drive_dir / archive_name
    shutil.copy2(archive_path, mirrored_path)
    record = T2SMarkFullMainArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest=file_digest(archive_path),
        archive_entry_count=len(entries),
        drive_archive_path=str(mirrored_path),
        drive_archive_digest=file_digest(mirrored_path),
        metadata={
            "construction_unit_name": "t2smark_full_main_reproduction",
            "drive_output_dir": str(drive_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(summary_path, record.to_dict())
    archive_manifest.setdefault("metadata", {})["archive_digest"] = record.archive_digest
    archive_manifest.setdefault("metadata", {})["drive_archive_digest"] = record.drive_archive_digest
    write_json(manifest_path, archive_manifest)
    return record




