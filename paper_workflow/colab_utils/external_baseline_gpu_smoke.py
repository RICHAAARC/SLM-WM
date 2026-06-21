"""外部 baseline 真实 GPU smoke 的 Colab 辅助函数。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from main.analysis.artifact_manifest import build_artifact_manifest
from paper_workflow.colab_utils.sd_runtime_cold_start import (
    build_runtime_environment_report,
    file_digest,
    resolve_code_version,
)

DEFAULT_OUTPUT_DIR = "outputs/external_baseline_gpu_smoke"
DEFAULT_DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/SLM/external_baseline_gpu_smoke"
DEFAULT_PRIOR_DRIVE_DIR = DEFAULT_DRIVE_OUTPUT_DIR
DEFAULT_T2SMARK_RUN_NAME = "t2smark_sd35_medium_gpu_smoke"
DEFAULT_T2SMARK_SOURCE_ENTRY = "external_baseline/primary/t2smark/source/run_sd35.py"
DEFAULT_T2SMARK_INVERSION_ENTRY = "external_baseline/primary/t2smark/source/src/inversion/inverse_diffusion3.py"
DEFAULT_SOURCE_REGISTRY_PATH = "external_baseline/source_registry.json"
DEFAULT_T2SMARK_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
DEFAULT_PACKAGE_PATTERN = "external_baseline_gpu_smoke_package_*.zip"
PRIMARY_BASELINE_METHODS = ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
T2SMARK_INVERSION_COMPAT_MARKER = "# SLM-WM 兼容补丁: 为新版 Diffusers 显式补齐注解依赖。"
T2SMARK_INVERSION_COMPAT_BLOCK = f"""{T2SMARK_INVERSION_COMPAT_MARKER}
import torch
from typing import Any, Callable, Dict, List, Optional, Union

try:
    from diffusers.image_processor import PipelineImageInput
except Exception:
    PipelineImageInput = Any
"""
ALLOWED_PRIOR_PREFIXES = ("outputs/external_baseline_gpu_smoke/",)
PACKAGE_EXTRA_PATHS = (
    "paper_workflow/external_baseline_gpu_smoke_run.ipynb",
    "paper_workflow/colab_utils/external_baseline_gpu_smoke.py",
    "external_baseline/README.md",
    "external_baseline/source_registry.json",
    "external_baseline/adaptation_notes/sd35_medium_external_baseline_adaptation.md",
    "external_baseline/primary/sd35_diffusion_baseline_common.py",
    "external_baseline/primary/t2smark/README.md",
    "external_baseline/primary/t2smark/adapter/run_slm_eval.py",
    "external_baseline/primary/tree_ring/adapter/run_slm_eval.py",
    "external_baseline/primary/gaussian_shading/adapter/run_slm_eval.py",
    "external_baseline/primary/shallow_diffuse/adapter/run_slm_eval.py",
    "scripts/build_external_baseline_command_plan.py",
    "scripts/run_external_baseline_command_plan.py",
    "scripts/validate_external_baseline_evidence.py",
)


@dataclass(frozen=True)
class ExternalBaselineGpuSmokeConfig:
    """描述一次外部 baseline 真实 GPU smoke 所需的最小配置。"""

    output_dir: str = DEFAULT_OUTPUT_DIR
    drive_output_dir: str = DEFAULT_DRIVE_OUTPUT_DIR
    prior_drive_dir: str = DEFAULT_PRIOR_DRIVE_DIR
    t2smark_run_name: str = DEFAULT_T2SMARK_RUN_NAME
    model_id: str = DEFAULT_T2SMARK_MODEL_ID
    seed: int = 20260621
    robust_test_num: int = 1
    clip_test_num: int = 0
    num_inference_steps: int = 8
    num_inversion_steps: int = 3
    guidance_scale: float = 4.0
    primary_baseline_max_samples: int = 1
    reuse_existing: bool = True
    reuse_prior_drive_package: bool = True
    force_generate: bool = False
    save_image: bool = True
    require_cuda: bool = True
    timeout_seconds: int = 86400


@dataclass(frozen=True)
class ExternalBaselineGpuSmokeArchiveRecord:
    """记录外部 baseline GPU smoke 压缩包与 Drive 镜像信息。"""

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


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先记录相对仓库根目录的路径。"""

    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_json(path: Path, payload: Any) -> None:
    """写出稳定 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json_text(payload), encoding="utf-8")


def read_json(path: Path) -> Any:
    """读取 JSON 文件。"""

    return json.loads(path.read_text(encoding="utf-8"))


def latest_drive_package(drive_dir: str | Path, pattern: str = DEFAULT_PACKAGE_PATTERN) -> Path | None:
    """从 Google Drive 目录中选择名称排序最新的历史结果包。"""

    candidates = sorted(Path(drive_dir).expanduser().glob(pattern))
    return candidates[-1] if candidates else None


def safe_extract_selected_entries(package_path: Path, root_path: Path, allowed_prefixes: tuple[str, ...]) -> tuple[str, ...]:
    """只解压允许进入工作区的 outputs 输入文件。"""

    extracted: list[str] = []
    with ZipFile(package_path) as archive:
        for member in archive.infolist():
            normalized_name = member.filename.replace("\\", "/")
            if member.is_dir() or not any(normalized_name.startswith(prefix) for prefix in allowed_prefixes):
                continue
            target_path = (root_path / normalized_name).resolve()
            if not target_path.is_relative_to(root_path.resolve()):
                raise RuntimeError(f"unsafe_zip_entry:{normalized_name}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source_handle, target_path.open("wb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)
            extracted.append(normalized_name)
    return tuple(extracted)


def output_paths(root_path: Path, config: ExternalBaselineGpuSmokeConfig) -> dict[str, Path]:
    """集中构造本次 smoke 所需路径。"""

    output_dir = (root_path / config.output_dir).resolve()
    official_root = output_dir / "t2smark_official"
    official_run_dir = official_root / config.t2smark_run_name
    adapter_output_root = output_dir / "adapter_outputs"
    execution_output_dir = output_dir / "execution"
    return {
        "output_dir": output_dir,
        "official_root": official_root,
        "official_run_dir": official_run_dir,
        "official_results": official_run_dir / "results.json",
        "official_settings": official_run_dir / "settings.json",
        "official_images": official_run_dir / "images",
        "t2smark_prompts": output_dir / "t2smark_smoke_prompts.json",
        "primary_prompt_plan": output_dir / "primary_baseline_smoke_prompt_plan.json",
        "image_pairs": output_dir / "t2smark_image_pairs.json",
        "command_plan": output_dir / "baseline_command_plan.json",
        "adapter_output_root": adapter_output_root,
        "execution_output_dir": execution_output_dir,
        "execution_manifest": execution_output_dir / "baseline_execution_manifest.json",
        "command_results": execution_output_dir / "baseline_command_results.json",
        "baseline_observations": execution_output_dir / "baseline_observations.json",
        "environment_report": output_dir / "external_baseline_gpu_smoke_environment_report.json",
        "summary": output_dir / "external_baseline_gpu_smoke_summary.json",
        "manifest": output_dir / "external_baseline_gpu_smoke_manifest.local.json",
    }


def ensure_cuda_if_requested(require_cuda: bool) -> dict[str, Any]:
    """在要求真实 GPU smoke 时检查 CUDA。"""

    try:
        import torch
    except Exception as error:  # pragma: no cover - 本地轻量测试不依赖 torch
        if require_cuda:
            raise RuntimeError("torch_import_failed") from error
        return {"cuda_available": False, "device_count": 0, "device_name": "torch_unavailable"}
    report = {
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }
    if require_cuda and not report["cuda_available"]:
        raise RuntimeError("cuda_unavailable_for_external_baseline_gpu_smoke")
    return report


def materialize_prior_outputs(
    root_path: Path,
    config: ExternalBaselineGpuSmokeConfig,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """如 Drive 中已有历史结果包, 则选择性解压可复用文件。"""

    if not config.reuse_prior_drive_package:
        return {"prior_package_reused": False, "prior_package_path": "", "extracted_entry_count": 0, "extracted_entries": []}
    package_path = latest_drive_package(config.prior_drive_dir)
    if package_path is None:
        return {"prior_package_reused": False, "prior_package_path": "", "extracted_entry_count": 0, "extracted_entries": []}
    extracted_entries = safe_extract_selected_entries(package_path, root_path, ALLOWED_PRIOR_PREFIXES)
    manifest = {
        "prior_package_reused": True,
        "prior_package_path": str(package_path),
        "prior_package_digest": file_digest(package_path),
        "extracted_entry_count": len(extracted_entries),
        "extracted_entries": list(extracted_entries),
    }
    write_json(paths["output_dir"] / "external_baseline_gpu_smoke_prior_package_manifest.json", manifest)
    return manifest


def should_run_t2smark_official(config: ExternalBaselineGpuSmokeConfig, results_path: Path) -> tuple[bool, str]:
    """判断 T2SMark 官方 SD3.5 运行是否需要本次生成。"""

    if config.force_generate:
        return True, "force_generate_requested"
    if config.reuse_existing and results_path.is_file():
        return False, "existing_results_found"
    return True, "results_missing"


def write_t2smark_prompt_input(paths: dict[str, Path]) -> Path:
    """写出官方 T2SMark 入口可直接读取的最小 prompt 文件。"""

    prompt_payload = {
        "annotations": [
            {
                "caption": "a small ceramic fox sitting on a wooden desk under soft studio lighting",
            }
        ]
    }
    write_json(paths["t2smark_prompts"], prompt_payload)
    return paths["t2smark_prompts"]


def write_primary_baseline_prompt_plan(paths: dict[str, Path]) -> Path:
    """写出三类 latent smoke adapter 与 T2SMark 共用的最小 prompt 计划。"""

    prompt_rows = [
        {
            "prompt_id": "primary_baseline_prompt_00000",
            "split": "gpu_smoke",
            "prompt_text": "a small ceramic fox sitting on a wooden desk under soft studio lighting",
        }
    ]
    write_json(paths["primary_prompt_plan"], prompt_rows)
    return paths["primary_prompt_plan"]


def run_command(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict[str, Any]:
    """执行显式 argv 命令并返回可落盘诊断。"""

    completed = subprocess.run(
        command,
        cwd=cwd,
        timeout=timeout_seconds,
        check=False,
        text=True,
        capture_output=True,
    )
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def load_baseline_registry_item(root_path: Path, baseline_id: str) -> dict[str, Any]:
    """从外部 baseline 登记表中读取指定方法的源码缓存描述。"""

    registry_path = root_path / DEFAULT_SOURCE_REGISTRY_PATH
    registry = read_json(registry_path)
    for item in registry.get("baseline_sources", []):
        if item.get("baseline_id") == baseline_id:
            return item
    raise KeyError(f"baseline_registry_item_missing:{baseline_id}")


def normalize_repository_url(repository_url: str) -> str:
    """将常见 SSH 形式转换为无需 SSH key 的 HTTPS 形式。"""

    if repository_url.startswith("git@github.com:"):
        return "https://github.com/" + repository_url.split(":", 1)[1]
    return repository_url


def patch_t2smark_inversion_compatibility(root_path: Path, paths: dict[str, Path]) -> dict[str, Any]:
    """为 T2SMark 官方 SD3.5 inversion 入口补齐新版环境所需导入。"""

    inversion_path = root_path / DEFAULT_T2SMARK_INVERSION_ENTRY
    if not inversion_path.is_file():
        raise FileNotFoundError(f"t2smark_inversion_entry_missing:{inversion_path}")
    source_text = inversion_path.read_text(encoding="utf-8")
    patch_applied = False
    if T2SMARK_INVERSION_COMPAT_MARKER not in source_text:
        import_line = "from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import *\n"
        if import_line in source_text:
            source_text = source_text.replace(import_line, import_line + "\n" + T2SMARK_INVERSION_COMPAT_BLOCK + "\n", 1)
        else:
            source_text = T2SMARK_INVERSION_COMPAT_BLOCK + "\n" + source_text
        inversion_path.write_text(source_text, encoding="utf-8")
        patch_applied = True
    report = {
        "source_patch_applied": patch_applied,
        "source_patch_needed": patch_applied,
        "source_patch_path": relative_or_absolute(inversion_path, root_path),
        "source_patch_reason": "typing_names_required_by_sd35_inversion_entry",
    }
    write_json(paths["output_dir"] / "t2smark_source_compatibility_patch.json", report)
    return report


def ensure_t2smark_source_available(root_path: Path, paths: dict[str, Path], timeout_seconds: int) -> dict[str, Any]:
    """在冷启动环境中按登记表补齐 T2SMark 官方源码缓存。"""

    source_entry = root_path / DEFAULT_T2SMARK_SOURCE_ENTRY
    if source_entry.is_file():
        patch_report = patch_t2smark_inversion_compatibility(root_path, paths)
        return {
            "source_available": True,
            "source_downloaded": False,
            "source_entry_path": relative_or_absolute(source_entry, root_path),
            "source_patch_report": patch_report,
        }

    registry_item = load_baseline_registry_item(root_path, "t2smark")
    source_dir = root_path / str(registry_item["source_dir"])
    repository_url = normalize_repository_url(str(registry_item["official_repository_url"]))
    if source_dir.exists() and any(source_dir.iterdir()):
        raise FileNotFoundError(f"t2smark_source_entry_missing_in_existing_source_cache:{source_entry}")

    source_dir.parent.mkdir(parents=True, exist_ok=True)
    clone_result = run_command(["git", "clone", repository_url, str(source_dir)], cwd=root_path, timeout_seconds=timeout_seconds)
    checkout_result: dict[str, Any] = {"command": [], "return_code": 0, "stdout": "", "stderr": ""}
    if clone_result["return_code"] == 0 and registry_item.get("official_repository_commit"):
        checkout_result = run_command(
            ["git", "checkout", str(registry_item["official_repository_commit"])],
            cwd=source_dir,
            timeout_seconds=300,
        )
    source_report = {
        "source_available": source_entry.is_file() and clone_result["return_code"] == 0 and checkout_result["return_code"] == 0,
        "source_downloaded": clone_result["return_code"] == 0,
        "source_entry_path": relative_or_absolute(source_entry, root_path),
        "source_dir": relative_or_absolute(source_dir, root_path),
        "official_repository_url": repository_url,
        "official_repository_commit": registry_item.get("official_repository_commit", ""),
        "clone_return_code": clone_result["return_code"],
        "checkout_return_code": checkout_result["return_code"],
    }
    write_json(
        paths["output_dir"] / "t2smark_source_prepare_result.json",
        {"source_report": source_report, "clone_result": clone_result, "checkout_result": checkout_result},
    )
    if not source_report["source_available"]:
        raise FileNotFoundError(f"t2smark_source_entry_missing_after_source_prepare:{source_entry}")
    source_report["source_patch_report"] = patch_t2smark_inversion_compatibility(root_path, paths)
    write_json(
        paths["output_dir"] / "t2smark_source_prepare_result.json",
        {"source_report": source_report, "clone_result": clone_result, "checkout_result": checkout_result},
    )
    return source_report


def run_t2smark_official_if_needed(
    root_path: Path,
    config: ExternalBaselineGpuSmokeConfig,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """根据本地和 Drive 结果状态决定是否运行 T2SMark 官方 SD3.5 入口。"""

    paths["official_root"].mkdir(parents=True, exist_ok=True)
    should_run, reason = should_run_t2smark_official(config, paths["official_results"])
    if not should_run:
        return {
            "official_result_generated": False,
            "official_result_reused": True,
            "official_generation_reason": reason,
            "official_results_path": relative_or_absolute(paths["official_results"], root_path),
            "official_return_code": 0,
            "official_command": [],
            "source_report": {
                "source_available": (root_path / DEFAULT_T2SMARK_SOURCE_ENTRY).is_file(),
                "source_downloaded": False,
                "source_prepare_skipped": True,
            },
        }
    source_report = ensure_t2smark_source_available(root_path, paths, timeout_seconds=300)
    source_entry = root_path / DEFAULT_T2SMARK_SOURCE_ENTRY
    ensure_cuda_if_requested(config.require_cuda)
    prompt_input_path = write_t2smark_prompt_input(paths)
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
        str(config.robust_test_num),
        "--clip_test_num",
        str(config.clip_test_num),
        "--dataset_key",
        str(prompt_input_path),
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
    if config.save_image:
        command.append("--save_image")
    result = run_command(command, cwd=root_path, timeout_seconds=config.timeout_seconds)
    write_json(paths["output_dir"] / "t2smark_official_command_result.json", result)
    if result["return_code"] != 0:
        return {
            "official_result_generated": False,
            "official_result_reused": False,
            "official_generation_reason": "official_command_failed",
            "official_results_path": relative_or_absolute(paths["official_results"], root_path),
            "official_return_code": result["return_code"],
            "official_command": command,
            "source_report": source_report,
        }
    if not paths["official_results"].is_file():
        raise FileNotFoundError("t2smark_results_missing_after_official_run")
    return {
        "official_result_generated": True,
        "official_result_reused": False,
        "official_generation_reason": reason,
        "official_results_path": relative_or_absolute(paths["official_results"], root_path),
        "official_return_code": result["return_code"],
        "official_command": command,
        "source_report": source_report,
    }


def build_current_t2smark_image_pairs(
    root_path: Path,
    config: ExternalBaselineGpuSmokeConfig,
    paths: dict[str, Path],
) -> list[dict[str, Any]]:
    """按当前官方图像目录重建 T2SMark adapter 所需的 image_pairs 输入。"""

    image_dir = paths["official_images"]
    rows: list[dict[str, Any]] = []
    for index in range(config.robust_test_num):
        image_path = image_dir / f"{index:05d}.png"
        image_id = f"t2smark_{index:05d}"
        row = {
            "image_id": image_id,
            "event_id": image_id,
            "prompt_id": f"t2smark_prompt_{index:05d}",
            "split": "gpu_smoke",
            "baseline_id": "t2smark",
            "generated_image_path": relative_or_absolute(image_path, root_path) if image_path.is_file() else "",
            "generated_image_digest": file_digest(image_path) if image_path.is_file() else "",
        }
        rows.append(row)
    return rows


def t2smark_image_pairs_are_current(
    image_pairs: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> bool:
    """判断已有 image_pairs 是否已经包含当前图像路径与 digest。"""

    if len(image_pairs) != len(current_rows):
        return False
    for old_row, current_row in zip(image_pairs, current_rows):
        if old_row.get("image_id") != current_row.get("image_id"):
            return False
        if current_row.get("generated_image_digest") and old_row.get("generated_image_digest") != current_row.get("generated_image_digest"):
            return False
        if current_row.get("generated_image_path") and old_row.get("generated_image_path") != current_row.get("generated_image_path"):
            return False
    return True


def build_t2smark_image_pairs(root_path: Path, config: ExternalBaselineGpuSmokeConfig, paths: dict[str, Path]) -> list[dict[str, Any]]:
    """生成或刷新 T2SMark adapter 所需的 image_pairs 输入。"""

    current_rows = build_current_t2smark_image_pairs(root_path, config, paths)
    if paths["image_pairs"].is_file() and config.reuse_existing and not config.force_generate:
        existing_rows = json.loads(paths["image_pairs"].read_text(encoding="utf-8"))
        if t2smark_image_pairs_are_current(existing_rows, current_rows):
            return existing_rows
    write_json(paths["image_pairs"], current_rows)
    return current_rows


def build_and_run_primary_baseline_adapters(
    root_path: Path,
    config: ExternalBaselineGpuSmokeConfig,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """生成命令计划并运行四个主表 external baseline adapter。"""

    prompt_plan_path = write_primary_baseline_prompt_plan(paths)
    build_command = [
        sys.executable,
        "scripts/build_external_baseline_command_plan.py",
        "--root",
        str(root_path),
        "--methods",
        ",".join(PRIMARY_BASELINE_METHODS),
        "--out",
        str(paths["command_plan"]),
        "--output-root",
        str(paths["adapter_output_root"]),
        "--prompt-plan",
        str(prompt_plan_path),
        "--image-pairs",
        str(paths["image_pairs"]),
        "--t2smark-results",
        str(paths["official_results"]),
        "--timeout-seconds",
        str(config.timeout_seconds),
        "--model-id",
        str(config.model_id),
        "--torch-dtype",
        "float16",
        "--height",
        "512",
        "--width",
        "512",
        "--latent-channels",
        "16",
        "--num-inference-steps",
        str(config.num_inference_steps),
        "--num-inversion-steps",
        str(config.num_inversion_steps),
        "--guidance-scale",
        str(config.guidance_scale),
        "--seed",
        str(config.seed),
        "--max-samples",
        str(config.primary_baseline_max_samples),
    ]
    if config.require_cuda:
        build_command.append("--require-cuda")
    build_result = run_command(build_command, cwd=root_path, timeout_seconds=300)
    write_json(paths["output_dir"] / "baseline_command_plan_builder_result.json", build_result)
    if build_result["return_code"] != 0:
        return {"adapter_execution_ready": False, "adapter_unsupported_reason": "command_plan_builder_failed"}

    run_command_args = [
        sys.executable,
        "scripts/run_external_baseline_command_plan.py",
        "--plan",
        str(paths["command_plan"]),
        "--out",
        str(paths["execution_output_dir"]),
        "--require-pass",
    ]
    execution_result = run_command(run_command_args, cwd=root_path, timeout_seconds=config.timeout_seconds)
    write_json(paths["output_dir"] / "baseline_command_plan_runner_result.json", execution_result)
    if execution_result["return_code"] != 0:
        return {"adapter_execution_ready": False, "adapter_unsupported_reason": "command_plan_runner_failed"}

    validation_command = [
        sys.executable,
        "scripts/validate_external_baseline_evidence.py",
        "--baseline-execution-manifest",
        str(paths["execution_manifest"]),
        "--require-pass",
    ]
    validation_result = run_command(validation_command, cwd=root_path, timeout_seconds=300)
    write_json(paths["output_dir"] / "baseline_evidence_validation_result.json", validation_result)
    execution_manifest = read_json(paths["execution_manifest"]) if paths["execution_manifest"].is_file() else {}
    command_results = json.loads(paths["command_results"].read_text(encoding="utf-8")) if paths["command_results"].is_file() else []
    observation_count_by_baseline = {
        str(row.get("baseline_id")): int(row.get("observation_count", 0))
        for row in command_results
        if int(row.get("return_code", 1)) == 0
    }
    ready_baseline_ids = [
        baseline_id
        for baseline_id in PRIMARY_BASELINE_METHODS
        if observation_count_by_baseline.get(baseline_id, 0) > 0
    ]
    primary_ready = set(ready_baseline_ids) == set(PRIMARY_BASELINE_METHODS)
    adapter_execution_ready = validation_result["return_code"] == 0 and primary_ready
    return {
        "adapter_execution_ready": adapter_execution_ready,
        "adapter_unsupported_reason": "" if adapter_execution_ready else "primary_baseline_adapter_smoke_incomplete",
        "adapter_observation_count": int(execution_manifest.get("observation_count", 0)),
        "primary_baseline_adapter_ready": primary_ready,
        "primary_baseline_adapter_count": len(PRIMARY_BASELINE_METHODS),
        "primary_baseline_observation_count": sum(observation_count_by_baseline.values()),
        "primary_baseline_ids": list(PRIMARY_BASELINE_METHODS),
        "ready_primary_baseline_ids": ready_baseline_ids,
        "primary_baseline_observation_count_by_id": observation_count_by_baseline,
        "primary_baseline_prompt_plan_path": relative_or_absolute(prompt_plan_path, root_path),
        "baseline_execution_manifest_path": relative_or_absolute(paths["execution_manifest"], root_path),
        "baseline_observations_path": relative_or_absolute(paths["baseline_observations"], root_path),
        "command_plan_path": relative_or_absolute(paths["command_plan"], root_path),
    }


def build_and_run_t2smark_adapter(
    root_path: Path,
    config: ExternalBaselineGpuSmokeConfig,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """兼容旧调用名称, 实际运行四个主表 external baseline adapter。"""

    return build_and_run_primary_baseline_adapters(root_path, config, paths)


def write_failure_outputs(
    root_path: Path,
    config: ExternalBaselineGpuSmokeConfig,
    paths: dict[str, Path],
    error: Exception,
) -> dict[str, Any]:
    """在真实 GPU smoke 失败时写出可打包诊断产物。"""

    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    environment_report = build_runtime_environment_report()
    write_json(paths["environment_report"], environment_report)
    summary = {
        "run_decision": "fail",
        "external_baseline_gpu_smoke_ready": False,
        "t2smark_real_gpu_smoke_ready": False,
        "adapter_execution_ready": False,
        "adapter_observation_count": 0,
        "primary_baseline_adapter_ready": False,
        "primary_baseline_adapter_count": len(PRIMARY_BASELINE_METHODS),
        "primary_baseline_observation_count": 0,
        "primary_baseline_ids": list(PRIMARY_BASELINE_METHODS),
        "supports_paper_claim": False,
        "unsupported_reason": f"{type(error).__name__}:{error}",
        "environment_report_path": relative_or_absolute(paths["environment_report"], root_path),
        "manifest_path": relative_or_absolute(paths["manifest"], root_path),
    }
    write_json(paths["summary"], summary)
    manifest = build_artifact_manifest(
        artifact_id="external_baseline_gpu_smoke_manifest",
        artifact_type="local_manifest",
        input_paths=(),
        output_paths=(relative_or_absolute(paths["summary"], root_path), relative_or_absolute(paths["environment_report"], root_path)),
        config=asdict(config),
        code_version=resolve_code_version(root_path),
        rebuild_command="运行 paper_workflow/external_baseline_gpu_smoke_run.ipynb",
        metadata={"run_decision": "fail", "supports_paper_claim": False},
    ).to_dict()
    write_json(paths["manifest"], manifest)
    return summary


def write_external_baseline_gpu_smoke_outputs(
    config: ExternalBaselineGpuSmokeConfig,
    root: str | Path = ".",
) -> dict[str, Any]:
    """运行外部 baseline 真实 GPU smoke 并写出 summary、manifest 和 observation。"""

    root_path = Path(root).resolve()
    paths = output_paths(root_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    try:
        prior_manifest = materialize_prior_outputs(root_path, config, paths)
        device_report = ensure_cuda_if_requested(config.require_cuda)
        official_report = run_t2smark_official_if_needed(root_path, config, paths)
        image_pairs = build_t2smark_image_pairs(root_path, config, paths)
        adapter_report = build_and_run_primary_baseline_adapters(root_path, config, paths)
        environment_report = build_runtime_environment_report()
        environment_report["external_baseline_device_report"] = device_report
        write_json(paths["environment_report"], environment_report)
    except Exception as error:
        return write_failure_outputs(root_path, config, paths, error)

    official_ready = paths["official_results"].is_file() and official_report.get("official_return_code") == 0
    adapter_ready = bool(adapter_report.get("adapter_execution_ready"))
    observation_count = int(adapter_report.get("adapter_observation_count", 0))
    primary_ready = bool(adapter_report.get("primary_baseline_adapter_ready"))
    primary_observation_count = int(adapter_report.get("primary_baseline_observation_count", 0))
    run_ready = bool(official_ready and adapter_ready and primary_ready and observation_count > 0)
    unsupported_reason = "" if run_ready else "external_baseline_gpu_smoke_incomplete"
    source_patch_report = official_report.get("source_report", {}).get("source_patch_report", {})
    summary = {
        "run_decision": "pass" if run_ready else "fail",
        "external_baseline_gpu_smoke_ready": run_ready,
        "t2smark_real_gpu_smoke_ready": official_ready,
        "t2smark_official_result_generated": bool(official_report.get("official_result_generated")),
        "t2smark_official_result_reused": bool(official_report.get("official_result_reused")),
        "t2smark_source_available": bool(official_report.get("source_report", {}).get("source_available")),
        "t2smark_source_downloaded": bool(official_report.get("source_report", {}).get("source_downloaded")),
        "t2smark_source_patch_applied": bool(source_patch_report.get("source_patch_applied")),
        "prior_package_reused": bool(prior_manifest.get("prior_package_reused")),
        "image_pair_count": len(image_pairs),
        "adapter_execution_ready": adapter_ready,
        "adapter_observation_count": observation_count,
        "primary_baseline_adapter_ready": primary_ready,
        "primary_baseline_adapter_count": int(adapter_report.get("primary_baseline_adapter_count", len(PRIMARY_BASELINE_METHODS))),
        "primary_baseline_observation_count": primary_observation_count,
        "primary_baseline_ids": list(adapter_report.get("primary_baseline_ids", PRIMARY_BASELINE_METHODS)),
        "ready_primary_baseline_ids": list(adapter_report.get("ready_primary_baseline_ids", [])),
        "primary_baseline_prompt_plan_path": str(adapter_report.get("primary_baseline_prompt_plan_path", "")),
        "supports_paper_claim": False,
        "unsupported_reason": unsupported_reason,
        "official_results_path": relative_or_absolute(paths["official_results"], root_path),
        "image_pairs_path": relative_or_absolute(paths["image_pairs"], root_path),
        "environment_report_path": relative_or_absolute(paths["environment_report"], root_path),
        "manifest_path": relative_or_absolute(paths["manifest"], root_path),
        "metadata": {
            **official_report,
            **adapter_report,
            "prior_manifest": prior_manifest,
            "claim_boundary": "gpu_smoke_not_full_external_baseline_comparison",
        },
    }
    write_json(paths["summary"], summary)
    input_paths = [relative_or_absolute(paths["official_results"], root_path), relative_or_absolute(paths["image_pairs"], root_path)]
    if paths["t2smark_prompts"].exists():
        input_paths.append(relative_or_absolute(paths["t2smark_prompts"], root_path))
    if paths["primary_prompt_plan"].exists():
        input_paths.append(relative_or_absolute(paths["primary_prompt_plan"], root_path))
    output_paths_for_manifest = [
        relative_or_absolute(paths["summary"], root_path),
        relative_or_absolute(paths["environment_report"], root_path),
    ]
    for optional_path in (paths["execution_manifest"], paths["baseline_observations"], paths["command_results"], paths["command_plan"]):
        if optional_path.exists():
            output_paths_for_manifest.append(relative_or_absolute(optional_path, root_path))
    manifest = build_artifact_manifest(
        artifact_id="external_baseline_gpu_smoke_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(input_paths),
        output_paths=tuple(output_paths_for_manifest),
        config=asdict(config),
        code_version=resolve_code_version(root_path),
        rebuild_command="运行 paper_workflow/external_baseline_gpu_smoke_run.ipynb",
        metadata={
            "run_decision": summary["run_decision"],
            "external_baseline_gpu_smoke_ready": run_ready,
            "adapter_observation_count": observation_count,
            "primary_baseline_adapter_ready": primary_ready,
            "primary_baseline_observation_count": primary_observation_count,
            "supports_paper_claim": False,
        },
    ).to_dict()
    write_json(paths["manifest"], manifest)
    return summary


def build_default_config() -> ExternalBaselineGpuSmokeConfig:
    """从环境变量构造默认 Colab 运行配置。"""

    return ExternalBaselineGpuSmokeConfig(
        output_dir=os.environ.get("SLM_WM_EXTERNAL_BASELINE_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        drive_output_dir=os.environ.get("SLM_WM_EXTERNAL_BASELINE_DRIVE_OUTPUT_DIR", DEFAULT_DRIVE_OUTPUT_DIR),
        prior_drive_dir=os.environ.get("SLM_WM_EXTERNAL_BASELINE_PRIOR_DRIVE_DIR", DEFAULT_PRIOR_DRIVE_DIR),
        t2smark_run_name=os.environ.get("SLM_WM_T2SMARK_RUN_NAME", DEFAULT_T2SMARK_RUN_NAME),
        model_id=os.environ.get("SLM_WM_T2SMARK_MODEL_ID", DEFAULT_T2SMARK_MODEL_ID),
        seed=int(os.environ.get("SLM_WM_EXTERNAL_BASELINE_SEED", "20260621")),
        robust_test_num=int(os.environ.get("SLM_WM_T2SMARK_ROBUST_TEST_NUM", "1")),
        clip_test_num=int(os.environ.get("SLM_WM_T2SMARK_CLIP_TEST_NUM", "0")),
        num_inference_steps=int(os.environ.get("SLM_WM_T2SMARK_NUM_INFERENCE_STEPS", "8")),
        num_inversion_steps=int(os.environ.get("SLM_WM_T2SMARK_NUM_INVERSION_STEPS", "3")),
        guidance_scale=float(os.environ.get("SLM_WM_T2SMARK_GUIDANCE_SCALE", "4.0")),
        primary_baseline_max_samples=int(os.environ.get("SLM_WM_PRIMARY_BASELINE_MAX_SAMPLES", "1")),
        reuse_existing=os.environ.get("SLM_WM_EXTERNAL_BASELINE_REUSE_EXISTING", "1") != "0",
        reuse_prior_drive_package=os.environ.get("SLM_WM_EXTERNAL_BASELINE_REUSE_DRIVE", "1") != "0",
        force_generate=os.environ.get("SLM_WM_EXTERNAL_BASELINE_FORCE_GENERATE", "0") == "1",
        save_image=os.environ.get("SLM_WM_T2SMARK_SAVE_IMAGE", "1") != "0",
        require_cuda=os.environ.get("SLM_WM_EXTERNAL_BASELINE_REQUIRE_CUDA", "1") != "0",
        timeout_seconds=int(os.environ.get("SLM_WM_EXTERNAL_BASELINE_TIMEOUT_SECONDS", "86400")),
    )


def run_default_external_baseline_gpu_smoke_plan(root: str | Path = ".") -> dict[str, Any]:
    """运行默认外部 baseline 真实 GPU smoke 计划。"""

    return write_external_baseline_gpu_smoke_outputs(config=build_default_config(), root=root)


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


def package_external_baseline_gpu_smoke_outputs(
    root: str | Path = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    drive_output_dir: str = DEFAULT_DRIVE_OUTPUT_DIR,
    archive_name: str = "external_baseline_gpu_smoke_package.zip",
) -> ExternalBaselineGpuSmokeArchiveRecord:
    """打包外部 baseline GPU smoke 产物并镜像到 Google Drive。"""

    root_path = Path(root).resolve()
    source_dir = (root_path / output_dir).resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_path = source_dir / archive_name
    package_manifest_path = source_dir / "external_baseline_gpu_smoke_package_input_manifest.json"
    summary_path = source_dir / "external_baseline_gpu_smoke_archive_summary.json"
    manifest_path = source_dir / "external_baseline_gpu_smoke_archive_manifest.local.json"
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
    preliminary_record = ExternalBaselineGpuSmokeArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest="",
        archive_entry_count=len(entries) + 3,
        drive_archive_path=str(Path(drive_output_dir).expanduser() / archive_name),
        drive_archive_digest="",
        metadata={
            "construction_unit_name": "external_baseline_gpu_smoke",
            "drive_output_dir": str(Path(drive_output_dir).expanduser()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "embedded_digest_scope": "external_summary_records_final_archive_digest",
        },
    )
    write_json(summary_path, preliminary_record.to_dict())
    archive_manifest = build_artifact_manifest(
        artifact_id="external_baseline_gpu_smoke_archive_manifest",
        artifact_type="local_manifest",
        input_paths=tuple([entry.relative_to(root_path).as_posix() for entry in entries] + [package_manifest_path.relative_to(root_path).as_posix()]),
        output_paths=(
            archive_path.relative_to(root_path).as_posix(),
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ),
        config={"archive_name": archive_name, "drive_output_dir": str(Path(drive_output_dir).expanduser())},
        code_version=resolve_code_version(root_path),
        rebuild_command="运行 paper_workflow/external_baseline_gpu_smoke_run.ipynb",
        metadata={
            "construction_unit_name": "external_baseline_gpu_smoke",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "embedded_digest_scope": "external_summary_records_final_archive_digest",
        },
    ).to_dict()
    write_json(manifest_path, archive_manifest)

    entries = collect_package_entries(root_path, source_dir, archive_path)
    with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.write(entry, entry.relative_to(root_path).as_posix())
    drive_dir = Path(drive_output_dir).expanduser()
    drive_dir.mkdir(parents=True, exist_ok=True)
    mirrored_path = drive_dir / archive_name
    shutil.copy2(archive_path, mirrored_path)
    record = ExternalBaselineGpuSmokeArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest=file_digest(archive_path),
        archive_entry_count=len(entries),
        drive_archive_path=str(mirrored_path),
        drive_archive_digest=file_digest(mirrored_path),
        metadata={
            "construction_unit_name": "external_baseline_gpu_smoke",
            "drive_output_dir": str(drive_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(summary_path, record.to_dict())
    archive_manifest.setdefault("metadata", {})["archive_digest"] = record.archive_digest
    archive_manifest.setdefault("metadata", {})["drive_archive_digest"] = record.drive_archive_digest
    write_json(manifest_path, archive_manifest)
    return record
