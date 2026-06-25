"""Gaussian Shading 官方原始环境复现与 governed import 的 Colab 辅助函数。

该 helper 服务补充表方法忠实度审计。它不把 legacy Stable Diffusion 结果混入 SD3.5 主表,
而是把官方命令、运行日志、环境报告、指标摘要和 governed import 记录统一写入 outputs/。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.protocol.paper_run_config import build_paper_run_config, resolve_count_from_environment
from experiments.baselines import (
    build_gaussian_shading_official_reference_record,
    build_gaussian_shading_official_reference_schema,
    validate_gaussian_shading_official_reference_records,
)
from main.analysis.artifact_manifest import build_artifact_manifest
from paper_workflow.colab_utils.external_baseline_gpu_smoke import (
    ensure_cuda_if_requested,
    load_baseline_registry_item,
    normalize_repository_url,
    run_command,
)
from paper_workflow.colab_utils.progress import (
    call_runner_with_progress_status,
    emit_progress_status,
    progress_bar,
    run_quiet_subprocess_with_progress,
    update_progress,
)
from paper_workflow.colab_utils.sd_runtime_cold_start import (
    build_runtime_environment_report,
    file_digest,
    resolve_code_version,
)

DEFAULT_OUTPUT_DIR = "outputs/gaussian_shading_official_reference"
DEFAULT_DRIVE_OUTPUT_DIR = ""
DEFAULT_SOURCE_DIR = "external_baseline/primary/gaussian_shading/source"
DEFAULT_RUN_NAME = "gaussian_shading_official_legacy_reference"
DEFAULT_SAMPLE_COUNT = 600
DEFAULT_OUTPUT_SUBDIR = "official_output"
DEFAULT_UPSTREAM_OFFICIAL_MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
DEFAULT_OFFICIAL_MODEL_ID = "Manojb/stable-diffusion-2-1-base"
DEFAULT_MODEL_SOURCE_NOTE = (
    "Gaussian Shading 官方 README 默认使用 stabilityai/stable-diffusion-2-1-base; "
    "当该模型不可直接访问时, 默认改用公开镜像并保留该模型来源说明。"
)
DEFAULT_LOCAL_MODEL_REPOSITORY_DIR = "/content/gaussian_shading_model_repository/stable_diffusion_2_1_base"
DEFAULT_LEGACY_ENV_PREFIX = "/content/gaussian_shading_legacy_env"
DEFAULT_MICROMAMBA_PATH = "/content/bin/micromamba"
DEFAULT_LEGACY_PYTHON_VERSION = "3.8"
DEFAULT_LEGACY_TORCH_SPECS = "torch==1.13.0+cu117 torchvision==0.14.0+cu117"
DEFAULT_LEGACY_PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu117"
DEFAULT_STRICT_OFFICIAL_ENVIRONMENT = True
DEFAULT_ALLOW_COMPATIBLE_ENVIRONMENT_FALLBACK = True
DEFAULT_LEGACY_PACKAGE_SPECS = (
    "transformers==4.23.1 diffusers==0.11.1 huggingface_hub==0.10.1 "
    "datasets==2.6.1 pyarrow<13 fsspec==2022.10.0 numpy==1.24.4 scipy==1.10.1 "
    "Pillow==9.5.0 tqdm==4.66.2 pycryptodome==3.20.0 open_clip_torch==2.7.0 "
    "ftfy==6.2.0 regex==2023.12.25 Requests==2.31.0 omegaconf==2.3.0 "
    "einops==0.4.1 kornia==0.6.4 matplotlib==3.7.5 timm==0.5.4"
)
PACKAGE_EXTRA_PATHS = (
    "paper_workflow/gaussian_shading_official_reference_run.ipynb",
    "paper_workflow/colab_utils/gaussian_shading_official_reference.py",
    "experiments/baselines/gaussian_shading_official_reference.py",
    "external_baseline/primary/gaussian_shading/README.md",
    "external_baseline/primary/gaussian_shading/source/README.md",
    "external_baseline/primary/gaussian_shading/source/requirements.txt",
    "external_baseline/source_registry.json",
)


@dataclass(frozen=True)
class GaussianShadingOfficialReferenceConfig:
    """描述 Gaussian Shading 官方原始环境复现与导入所需配置。"""

    output_dir: str = DEFAULT_OUTPUT_DIR
    drive_output_dir: str = field(default_factory=lambda: build_paper_run_config(".").drive_dir("gaussian_shading_official_reference"))
    source_dir: str = DEFAULT_SOURCE_DIR
    run_name: str = DEFAULT_RUN_NAME
    sample_count: int = DEFAULT_SAMPLE_COUNT
    official_output_subdir: str = DEFAULT_OUTPUT_SUBDIR
    official_model_id: str = DEFAULT_OFFICIAL_MODEL_ID
    upstream_official_model_id: str = DEFAULT_UPSTREAM_OFFICIAL_MODEL_ID
    model_source_note: str = DEFAULT_MODEL_SOURCE_NOTE
    dataset_path: str = "Gustavosta/Stable-Diffusion-Prompts"
    fpr: float = 0.000001
    channel_copy: int = 1
    hw_copy: int = 8
    user_number: int = 1000000
    gen_seed: int = 0
    image_length: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    num_inversion_steps: int = 50
    use_chacha: bool = True
    reference_model: str = ""
    reference_model_pretrain: str = ""
    patch_model_repository_layout: bool = True
    prepare_local_model_repository: bool = True
    local_model_repository_dir: str = DEFAULT_LOCAL_MODEL_REPOSITORY_DIR
    patch_model_index_for_legacy_transformers: bool = True
    official_python_executable: str = ""
    prepare_legacy_environment: bool = False
    legacy_environment_prefix: str = DEFAULT_LEGACY_ENV_PREFIX
    micromamba_path: str = DEFAULT_MICROMAMBA_PATH
    legacy_python_version: str = DEFAULT_LEGACY_PYTHON_VERSION
    strict_official_environment: bool = DEFAULT_STRICT_OFFICIAL_ENVIRONMENT
    allow_compatible_environment_fallback: bool = DEFAULT_ALLOW_COMPATIBLE_ENVIRONMENT_FALLBACK
    legacy_torch_specs: str = DEFAULT_LEGACY_TORCH_SPECS
    legacy_pytorch_index_url: str = DEFAULT_LEGACY_PYTORCH_INDEX_URL
    legacy_package_specs: str = DEFAULT_LEGACY_PACKAGE_SPECS
    run_official_command: bool = True
    summary_import_path: str = ""
    log_import_path: str = ""
    require_cuda: bool = True
    timeout_seconds: int = 86400
    enable_workflow_progress_bar: bool = True


@dataclass(frozen=True)
class GaussianShadingOfficialReferenceArchiveRecord:
    """记录 Gaussian Shading 官方参考压缩包与 Drive 镜像信息。"""

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


def write_json(path: Path, payload: Any) -> None:
    """写出 JSON 文件并创建父目录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json_text(payload), encoding="utf-8")


def read_json(path: Path) -> Any:
    """读取 JSON 文件。"""

    return json.loads(path.read_text(encoding="utf-8-sig"))


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先记录相对仓库根目录的路径。"""

    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def split_package_specs(package_specs: str) -> list[str]:
    """解析以空白分隔的 pip package spec 列表。"""

    return [item.strip() for item in str(package_specs).split() if item.strip()]


def command_exception_result(command: Any, error: Exception) -> dict[str, Any]:
    """把环境准备异常转换为可落盘命令诊断。"""

    return_code = 124 if isinstance(error, subprocess.TimeoutExpired) else 98
    return {
        "command": command,
        "return_code": return_code,
        "stdout": str(getattr(error, "stdout", "") or ""),
        "stderr": f"{type(error).__name__}:{error}",
    }


def run_shell_command(
    command: str,
    *,
    cwd: Path,
    timeout_seconds: int,
    progress: object | None = None,
    progress_profile: str = "",
) -> dict[str, Any]:
    """执行 shell 命令并返回可落盘诊断。"""

    try:
        completed = run_quiet_subprocess_with_progress(
            command,
            cwd=cwd,
            shell=True,
            timeout_seconds=timeout_seconds,
            progress=progress,
            progress_profile=progress_profile or "operation=shell_command",
        )
    except Exception as error:
        return command_exception_result(command, error)
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_argv_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    progress: object | None = None,
    progress_profile: str = "",
) -> dict[str, Any]:
    """执行 argv 命令并把失败收敛为可审计诊断。"""

    try:
        return call_runner_with_progress_status(
            run_command,
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            progress=progress,
            progress_profile=progress_profile or "operation=argv_command",
        )
    except Exception as error:
        return command_exception_result(command, error)


def run_shell_command_with_progress_status(
    command: str,
    *,
    cwd: Path,
    timeout_seconds: int,
    progress: object | None = None,
    progress_profile: str = "",
) -> dict[str, Any]:
    """调用可替换 shell runner, 并兼容测试中的轻量 fake。"""

    return call_runner_with_progress_status(
        run_shell_command,
        command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        progress=progress,
        progress_profile=progress_profile,
    )


def run_command_with_progress_status(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    progress: object | None = None,
    progress_profile: str = "",
) -> dict[str, Any]:
    """调用可替换 argv runner, 并兼容测试中的轻量 fake。"""

    return call_runner_with_progress_status(
        run_command,
        command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        progress=progress,
        progress_profile=progress_profile,
    )


def output_paths(root_path: Path, config: GaussianShadingOfficialReferenceConfig) -> dict[str, Path]:
    """集中构造 Gaussian Shading 官方参考 workflow 的输出路径。"""

    output_dir = (root_path / config.output_dir).resolve()
    official_output_dir = output_dir / config.official_output_subdir
    return {
        "output_dir": output_dir,
        "official_output_dir": official_output_dir,
        "official_metric_text": official_output_dir / "Identity.txt",
        "official_command_result": output_dir / "gaussian_shading_official_command_result.json",
        "source_prepare_result": output_dir / "gaussian_shading_official_source_prepare_result.json",
        "source_patch_result": output_dir / "gaussian_shading_official_source_patch_result.json",
        "model_repository_prepare_result": output_dir / "gaussian_shading_model_repository_prepare_result.json",
        "legacy_environment_prepare_result": output_dir / "gaussian_shading_legacy_environment_prepare_result.json",
        "official_stdout": output_dir / "gaussian_shading_official_stdout.txt",
        "official_stderr": output_dir / "gaussian_shading_official_stderr.txt",
        "official_metric_summary": output_dir / "gaussian_shading_official_metric_summary.json",
        "reference_schema": output_dir / "gaussian_shading_official_reference_schema.json",
        "reference_records": output_dir / "gaussian_shading_official_reference_records.jsonl",
        "reference_validation": output_dir / "gaussian_shading_official_reference_validation_report.json",
        "environment_report": output_dir / "gaussian_shading_official_reference_environment_report.json",
        "summary": output_dir / "gaussian_shading_official_reference_summary.json",
        "manifest": output_dir / "manifest.local.json",
    }


def ensure_micromamba_available(
    root_path: Path,
    config: GaussianShadingOfficialReferenceConfig,
    progress: object | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """确保 micromamba 可用, 作为 Colab 隔离环境创建工具。"""

    micromamba_path = Path(config.micromamba_path)
    command_results: list[dict[str, Any]] = []
    if not micromamba_path.is_file():
        micromamba_path.parent.mkdir(parents=True, exist_ok=True)
        command_results.append(
            run_shell_command_with_progress_status(
                f"curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C {micromamba_path.parent.parent} bin/micromamba",
                cwd=root_path,
                timeout_seconds=600,
                progress=progress,
                progress_profile="operation=gaussian_shading_fetch_micromamba",
            )
        )
    return micromamba_path, command_results


def create_python_environment(
    root_path: Path,
    *,
    micromamba_path: Path,
    environment_prefix: Path,
    python_version: str,
    progress: object | None = None,
    progress_profile: str = "",
) -> list[dict[str, Any]]:
    """创建指定 Python 版本的隔离环境, 已存在时保持可复用。"""

    legacy_python = environment_prefix / "bin" / "python"
    if legacy_python.is_file():
        return []
    return [
        run_argv_command(
            [
                str(micromamba_path),
                "create",
                "-y",
                "-p",
                str(environment_prefix),
                f"python={python_version}",
                "pip",
            ],
            cwd=root_path,
            timeout_seconds=900,
            progress=progress,
            progress_profile=progress_profile or "operation=gaussian_shading_create_python_environment",
        )
    ]


def verify_legacy_imports(root_path: Path, legacy_python: Path, progress: object | None = None) -> dict[str, Any]:
    """验证官方脚本所需核心包是否能在 legacy 环境中导入。"""

    verify_command = [
        str(legacy_python),
        "-c",
        "import torch, diffusers, transformers, datasets, open_clip; "
        "print({'torch': torch.__version__, 'diffusers': diffusers.__version__, "
        "'transformers': transformers.__version__, 'datasets': datasets.__version__})",
    ]
    return run_argv_command(
        verify_command,
        cwd=root_path,
        timeout_seconds=300,
        progress=progress,
        progress_profile="operation=gaussian_shading_verify_legacy_imports",
    )


def prepare_strict_official_environment_profile(
    root_path: Path,
    config: GaussianShadingOfficialReferenceConfig,
    *,
    micromamba_path: Path,
    progress: object | None = None,
) -> dict[str, Any]:
    """按官方 README 语义尝试 Python 3.8 + requirements.txt 严格环境。"""

    strict_prefix = Path(config.legacy_environment_prefix) / "official_requirements_strict"
    strict_python = strict_prefix / "bin" / "python"
    requirements_path = (root_path / config.source_dir / "requirements.txt").resolve()
    command_results = create_python_environment(
        root_path,
        micromamba_path=micromamba_path,
        environment_prefix=strict_prefix,
        python_version=config.legacy_python_version,
        progress=progress,
        progress_profile="operation=gaussian_shading_create_strict_environment",
    )
    if requirements_path.is_file():
        command_results.append(
            run_argv_command(
                [str(strict_python), "-m", "pip", "install", "-r", str(requirements_path)],
                cwd=root_path,
                timeout_seconds=3600,
                progress=progress,
                progress_profile="operation=gaussian_shading_install_strict_requirements",
            )
        )
    else:
        command_results.append(
            {
                "command": [str(strict_python), "-m", "pip", "install", "-r", str(requirements_path)],
                "return_code": 96,
                "stdout": "",
                "stderr": "official_requirements_txt_missing",
            }
        )
    command_results.append(verify_legacy_imports(root_path, strict_python, progress=progress))
    ready = strict_python.is_file() and all(int(result.get("return_code", 1)) == 0 for result in command_results)
    return {
        "environment_profile": "official_requirements_strict",
        "environment_prefix": str(strict_prefix),
        "legacy_python_executable": str(strict_python),
        "requirements_path": relative_or_absolute(requirements_path, root_path),
        "environment_ready": ready,
        "command_results": command_results,
    }


def prepare_compatible_environment_profile(
    root_path: Path,
    config: GaussianShadingOfficialReferenceConfig,
    *,
    micromamba_path: Path,
    progress: object | None = None,
) -> dict[str, Any]:
    """在官方 requirements 冲突时准备受治理的 Colab 兼容 fallback 环境。"""

    fallback_prefix = Path(config.legacy_environment_prefix) / "colab_compatible_fallback"
    fallback_python = fallback_prefix / "bin" / "python"
    command_results = create_python_environment(
        root_path,
        micromamba_path=micromamba_path,
        environment_prefix=fallback_prefix,
        python_version=config.legacy_python_version,
        progress=progress,
        progress_profile="operation=gaussian_shading_create_compatible_environment",
    )
    torch_specs = split_package_specs(config.legacy_torch_specs)
    if torch_specs:
        command_results.append(
            run_argv_command(
                [
                    str(fallback_python),
                    "-m",
                    "pip",
                    "install",
                    "--extra-index-url",
                    config.legacy_pytorch_index_url,
                    *torch_specs,
                ],
                cwd=root_path,
                timeout_seconds=1800,
                progress=progress,
                progress_profile="operation=gaussian_shading_install_compatible_torch",
            )
        )
    package_specs = split_package_specs(config.legacy_package_specs)
    if package_specs:
        command_results.append(
            run_argv_command(
                [str(fallback_python), "-m", "pip", "install", *package_specs],
                cwd=root_path,
                timeout_seconds=2400,
                progress=progress,
                progress_profile="operation=gaussian_shading_install_compatible_packages",
            )
        )
    command_results.append(verify_legacy_imports(root_path, fallback_python, progress=progress))
    ready = fallback_python.is_file() and all(int(result.get("return_code", 1)) == 0 for result in command_results)
    return {
        "environment_profile": "colab_compatible_fallback",
        "environment_prefix": str(fallback_prefix),
        "legacy_python_executable": str(fallback_python),
        "environment_ready": ready,
        "legacy_torch_specs": config.legacy_torch_specs,
        "legacy_package_specs": config.legacy_package_specs,
        "command_results": command_results,
    }


def prepare_gaussian_shading_legacy_environment(
    root_path: Path,
    config: GaussianShadingOfficialReferenceConfig,
    paths: dict[str, Path],
    progress: object | None = None,
) -> dict[str, Any]:
    """准备 Gaussian Shading 官方参考所需 legacy Python 环境。"""

    if not config.prepare_legacy_environment:
        report = {
            "legacy_environment_requested": False,
            "legacy_environment_ready": False,
            "legacy_environment_skipped": True,
            "legacy_environment_skip_reason": "prepare_legacy_environment_disabled",
            "legacy_python_executable": config.official_python_executable or sys.executable,
            "legacy_environment_profile": "ambient_python",
        }
        write_json(paths["legacy_environment_prepare_result"], report)
        return report
    if config.official_python_executable:
        report = {
            "legacy_environment_requested": True,
            "legacy_environment_ready": True,
            "legacy_environment_skipped": True,
            "legacy_environment_skip_reason": "explicit_official_python_executable_provided",
            "legacy_python_executable": config.official_python_executable,
            "legacy_environment_profile": "external_python_executable",
        }
        write_json(paths["legacy_environment_prepare_result"], report)
        return report

    micromamba_path, micromamba_results = ensure_micromamba_available(root_path, config, progress=progress)
    profile_reports: list[dict[str, Any]] = []
    selected_profile: dict[str, Any] | None = None
    if config.strict_official_environment:
        strict_report = prepare_strict_official_environment_profile(
            root_path,
            config,
            micromamba_path=micromamba_path,
            progress=progress,
        )
        profile_reports.append(strict_report)
        if strict_report.get("environment_ready"):
            selected_profile = strict_report
    if selected_profile is None and config.allow_compatible_environment_fallback:
        fallback_report = prepare_compatible_environment_profile(
            root_path,
            config,
            micromamba_path=micromamba_path,
            progress=progress,
        )
        profile_reports.append(fallback_report)
        if fallback_report.get("environment_ready"):
            selected_profile = fallback_report

    ready = selected_profile is not None
    selected_profile_name = str(selected_profile.get("environment_profile")) if selected_profile else "none"
    selected_python = str(selected_profile.get("legacy_python_executable")) if selected_profile else str(Path(config.legacy_environment_prefix) / "bin" / "python")
    report = {
        "legacy_environment_requested": True,
        "legacy_environment_ready": ready,
        "legacy_python_executable": selected_python,
        "legacy_environment_prefix": str(Path(config.legacy_environment_prefix)),
        "legacy_environment_profile": selected_profile_name,
        "legacy_python_version": config.legacy_python_version,
        "strict_official_environment_requested": bool(config.strict_official_environment),
        "strict_official_environment_ready": any(
            item.get("environment_profile") == "official_requirements_strict" and item.get("environment_ready")
            for item in profile_reports
        ),
        "compatible_environment_fallback_requested": bool(config.allow_compatible_environment_fallback),
        "compatible_environment_fallback_ready": any(
            item.get("environment_profile") == "colab_compatible_fallback" and item.get("environment_ready")
            for item in profile_reports
        ),
        "legacy_torch_specs": config.legacy_torch_specs,
        "legacy_package_specs": config.legacy_package_specs,
        "micromamba_command_results": micromamba_results,
        "environment_profile_reports": profile_reports,
        "command_results": [*micromamba_results, *(selected_profile.get("command_results", []) if selected_profile else [])],
    }
    write_json(paths["legacy_environment_prepare_result"], report)
    return report


def source_report(root_path: Path, config: GaussianShadingOfficialReferenceConfig) -> dict[str, Any]:
    """检查官方 Gaussian Shading 源码快照和 requirements 是否存在。"""

    source_dir = (root_path / config.source_dir).resolve()
    entrypoint = source_dir / "run_gaussian_shading.py"
    requirements = source_dir / "requirements.txt"
    return {
        "source_dir": relative_or_absolute(source_dir, root_path),
        "official_entrypoint": relative_or_absolute(entrypoint, root_path),
        "requirements_path": relative_or_absolute(requirements, root_path),
        "source_dir_ready": source_dir.is_dir(),
        "official_entrypoint_ready": entrypoint.is_file(),
        "requirements_ready": requirements.is_file(),
        "requirements_text": requirements.read_text(encoding="utf-8") if requirements.is_file() else "",
        "official_python_executable": config.official_python_executable or sys.executable,
    }


def ensure_gaussian_shading_source_available(
    root_path: Path,
    config: GaussianShadingOfficialReferenceConfig,
    paths: dict[str, Path],
    progress: object | None = None,
) -> dict[str, Any]:
    """在 Colab 冷启动环境中按登记表补齐 Gaussian Shading 官方源码。"""

    initial_report = source_report(root_path, config)
    if initial_report["official_entrypoint_ready"]:
        source_prepare_report = {
            **initial_report,
            "source_available": True,
            "source_downloaded": False,
            "source_prepare_reason": "existing_source_entrypoint_found",
        }
        write_json(paths["source_prepare_result"], source_prepare_report)
        return source_prepare_report

    registry_item = load_baseline_registry_item(root_path, "gaussian_shading")
    source_dir = (root_path / config.source_dir).resolve()
    source_entry = source_dir / "run_gaussian_shading.py"
    if source_dir.exists() and any(source_dir.iterdir()):
        source_prepare_report = {
            **initial_report,
            "source_available": False,
            "source_downloaded": False,
            "source_prepare_reason": "source_dir_exists_without_official_entrypoint",
            "official_repository_url": normalize_repository_url(str(registry_item.get("official_repository_url", ""))),
            "official_repository_commit": registry_item.get("official_repository_commit", ""),
        }
        write_json(paths["source_prepare_result"], source_prepare_report)
        return source_prepare_report

    source_dir.parent.mkdir(parents=True, exist_ok=True)
    repository_url = normalize_repository_url(str(registry_item["official_repository_url"]))
    clone_result = run_command_with_progress_status(
        ["git", "clone", repository_url, str(source_dir)],
        cwd=root_path,
        timeout_seconds=300,
        progress=progress,
        progress_profile="operation=gaussian_shading_source_clone",
    )
    checkout_result: dict[str, Any] = {"command": [], "return_code": 0, "stdout": "", "stderr": ""}
    if clone_result["return_code"] == 0 and registry_item.get("official_repository_commit"):
        checkout_result = run_command_with_progress_status(
            ["git", "checkout", str(registry_item["official_repository_commit"])],
            cwd=source_dir,
            timeout_seconds=300,
            progress=progress,
            progress_profile="operation=gaussian_shading_source_checkout",
        )
    refreshed_report = source_report(root_path, config)
    source_prepare_report = {
        **refreshed_report,
        "source_available": source_entry.is_file() and clone_result["return_code"] == 0 and checkout_result["return_code"] == 0,
        "source_downloaded": clone_result["return_code"] == 0,
        "source_prepare_reason": "source_cloned_from_registry",
        "official_repository_url": repository_url,
        "official_repository_commit": registry_item.get("official_repository_commit", ""),
        "clone_return_code": clone_result["return_code"],
        "checkout_return_code": checkout_result["return_code"],
        "clone_result": clone_result,
        "checkout_result": checkout_result,
    }
    write_json(paths["source_prepare_result"], source_prepare_report)
    return source_prepare_report


def patch_gaussian_shading_model_repository_layout(
    root_path: Path,
    config: GaussianShadingOfficialReferenceConfig,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """为公开镜像缺少 fp16 分支的情况应用最小源码入口补丁。"""

    source_dir = (root_path / config.source_dir).resolve()
    entrypoint = source_dir / "run_gaussian_shading.py"
    report = {
        "patch_requested": bool(config.patch_model_repository_layout),
        "patch_applied": False,
        "patch_skipped": False,
        "official_entrypoint": relative_or_absolute(entrypoint, root_path),
        "official_model_id": config.official_model_id,
        "upstream_official_model_id": config.upstream_official_model_id,
        "model_source_note": config.model_source_note,
        "patch_reason": "mirror_has_fp16_weights_on_main_without_fp16_branch",
    }
    if not config.patch_model_repository_layout:
        report.update({"patch_skipped": True, "patch_skip_reason": "patch_model_repository_layout_disabled"})
        write_json(paths["source_patch_result"], report)
        return report
    if not entrypoint.is_file():
        report.update({"patch_skipped": True, "patch_skip_reason": "official_entrypoint_missing"})
        write_json(paths["source_patch_result"], report)
        return report

    source_text = entrypoint.read_text(encoding="utf-8")
    before_digest = file_digest(entrypoint)
    marker = "# SLM-WM: 公开镜像没有 fp16 分支, 因此从 main 分支加载模型权重。"
    target = "            revision='fp16',\n"
    if marker in source_text:
        report.update(
            {
                "patch_skipped": True,
                "patch_skip_reason": "model_repository_layout_patch_already_present",
                "entrypoint_digest_before": before_digest,
                "entrypoint_digest_after": before_digest,
            }
        )
        write_json(paths["source_patch_result"], report)
        return report
    patched_text = source_text.replace(target, f"            {marker}\n")
    if patched_text == source_text:
        report.update(
            {
                "patch_skipped": True,
                "patch_skip_reason": "fp16_revision_line_not_found",
                "entrypoint_digest_before": before_digest,
                "entrypoint_digest_after": before_digest,
            }
        )
        write_json(paths["source_patch_result"], report)
        return report
    entrypoint.write_text(patched_text, encoding="utf-8")
    report.update(
        {
            "patch_applied": True,
            "entrypoint_digest_before": before_digest,
            "entrypoint_digest_after": file_digest(entrypoint),
        }
    )
    write_json(paths["source_patch_result"], report)
    return report


def download_hf_snapshot(
    repo_id: str,
    *,
    local_dir: Path,
    token: str | None,
) -> str:
    """下载 Hugging Face 模型快照到受控运行缓存目录。"""

    from huggingface_hub import snapshot_download

    return str(
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=token or None,
        )
    )


def prepare_gaussian_shading_model_repository(
    root_path: Path,
    config: GaussianShadingOfficialReferenceConfig,
    paths: dict[str, Path],
    progress: object | None = None,
) -> dict[str, Any]:
    """准备本地模型目录并补齐 legacy diffusers 所需的 model_index 兼容项。"""

    local_model_path = Path(config.local_model_repository_dir).expanduser()
    model_index_path = local_model_path / "model_index.json"
    report: dict[str, Any] = {
        "local_model_repository_requested": bool(config.prepare_local_model_repository),
        "local_model_repository_ready": False,
        "local_model_repository_path": str(local_model_path),
        "official_model_id": config.official_model_id,
        "upstream_official_model_id": config.upstream_official_model_id,
        "effective_official_model_id": config.official_model_id,
        "model_index_patch_requested": bool(config.patch_model_index_for_legacy_transformers),
        "model_index_patch_applied": False,
        "model_source_note": config.model_source_note,
    }
    if not config.prepare_local_model_repository:
        report.update({"local_model_repository_skipped": True, "skip_reason": "prepare_local_model_repository_disabled"})
        write_json(paths["model_repository_prepare_result"], report)
        return report

    download_result: dict[str, Any] = {"download_requested": not model_index_path.is_file()}
    if not model_index_path.is_file():
        try:
            emit_progress_status(progress, profile=f"operation=gaussian_shading_model_snapshot_download model={config.official_model_id}")
            local_model_path.mkdir(parents=True, exist_ok=True)
            snapshot_path = download_hf_snapshot(
                config.official_model_id,
                local_dir=local_model_path,
                token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None,
            )
            download_result.update({"return_code": 0, "snapshot_path": snapshot_path})
        except Exception as error:
            download_result.update({"return_code": 98, "error": f"{type(error).__name__}:{error}"})
            report["download_result"] = download_result
            write_json(paths["model_repository_prepare_result"], report)
            return report
    else:
        download_result.update(
            {"return_code": 0, "snapshot_path": str(local_model_path), "download_skipped_reason": "model_index_already_exists"}
        )

    report["download_result"] = download_result
    if not model_index_path.is_file():
        report.update({"local_model_repository_ready": False, "failure_reason": "model_index_missing_after_download"})
        write_json(paths["model_repository_prepare_result"], report)
        return report

    before_digest = file_digest(model_index_path)
    model_index = read_json(model_index_path)
    report["model_index_digest_before"] = before_digest
    feature_extractor = model_index.get("feature_extractor")
    if (
        config.patch_model_index_for_legacy_transformers
        and isinstance(feature_extractor, list)
        and len(feature_extractor) >= 2
        and feature_extractor[0] == "transformers"
        and feature_extractor[1] == "CLIPImageProcessor"
    ):
        feature_extractor[1] = "CLIPFeatureExtractor"
        write_json(model_index_path, model_index)
        report["model_index_patch_applied"] = True
        report["model_index_patch_reason"] = "legacy_diffusers_0_11_1_uses_clip_feature_extractor"
    elif not config.patch_model_index_for_legacy_transformers:
        report["model_index_patch_skip_reason"] = "patch_model_index_for_legacy_transformers_disabled"
    else:
        report["model_index_patch_skip_reason"] = "model_index_feature_extractor_already_compatible"

    report.update(
        {
            "model_index_digest_after": file_digest(model_index_path),
            "model_index_feature_extractor": read_json(model_index_path).get("feature_extractor"),
            "local_model_repository_ready": True,
            "effective_official_model_id": str(local_model_path),
        }
    )
    write_json(paths["model_repository_prepare_result"], report)
    return report


def build_official_command(root_path: Path, config: GaussianShadingOfficialReferenceConfig, paths: dict[str, Path]) -> list[str]:
    """构造 Gaussian Shading 官方 legacy 入口命令。"""

    source_dir = (root_path / config.source_dir).resolve()
    entrypoint = source_dir / "run_gaussian_shading.py"
    python_executable = config.official_python_executable or sys.executable
    output_path = str(paths["official_output_dir"]) + os.sep
    command = [
        python_executable,
        str(entrypoint),
        "--num",
        str(max(1, int(config.sample_count))),
        "--fpr",
        str(config.fpr),
        "--channel_copy",
        str(config.channel_copy),
        "--hw_copy",
        str(config.hw_copy),
        "--user_number",
        str(config.user_number),
        "--gen_seed",
        str(config.gen_seed),
        "--image_length",
        str(config.image_length),
        "--guidance_scale",
        str(config.guidance_scale),
        "--num_inference_steps",
        str(config.num_inference_steps),
        "--num_inversion_steps",
        str(config.num_inversion_steps),
        "--dataset_path",
        str(config.dataset_path),
        "--model_path",
        str(config.official_model_id),
        "--output_path",
        output_path,
    ]
    if config.use_chacha:
        command.append("--chacha")
    if config.reference_model:
        command.extend(["--reference_model", config.reference_model])
    if config.reference_model_pretrain:
        command.extend(["--reference_model_pretrain", config.reference_model_pretrain])
    return command


def parse_metric_text(text: str, sample_count: int) -> dict[str, Any]:
    """从官方 Identity.txt 或日志中解析 Gaussian Shading 指标。"""

    patterns = {
        "detection_true_positive_rate": r"tpr_detection:\s*([0-9eE+\-.]+)",
        "traceability_true_positive_rate": r"tpr_traceability:\s*([0-9eE+\-.]+)",
        "mean_bit_accuracy": r"mean_acc:\s*([0-9eE+\-.]+)",
        "std_bit_accuracy": r"std_acc:\s*([0-9eE+\-.]+)",
        "mean_clip_score": r"mean_clip_score:\s*([0-9eE+\-.]+)",
        "std_clip_score": r"std_clip_score:\s*([0-9eE+\-.]+)",
    }
    metrics: dict[str, Any] = {
        "sample_count": int(sample_count),
        "positive_count": int(sample_count),
        "mean_clip_score": 0.0,
        "std_clip_score": 0.0,
    }
    for field_name, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            metrics[field_name] = float(matches[-1])
    return metrics


def normalize_metric_summary(payload: dict[str, Any], sample_count: int) -> dict[str, Any]:
    """把外部 summary 或日志解析结果规范化为 reference schema 指标。"""

    defaults = {
        "sample_count": int(sample_count),
        "positive_count": int(sample_count),
        "detection_true_positive_rate": 0.0,
        "traceability_true_positive_rate": 0.0,
        "mean_bit_accuracy": 0.0,
        "std_bit_accuracy": 0.0,
        "mean_clip_score": 0.0,
        "std_clip_score": 0.0,
    }
    aliases = {
        "tpr_detection": "detection_true_positive_rate",
        "tpr_traceability": "traceability_true_positive_rate",
        "mean_acc": "mean_bit_accuracy",
        "std_acc": "std_bit_accuracy",
    }
    normalized = dict(defaults)
    for key, value in payload.items():
        target_key = aliases.get(str(key), str(key))
        if target_key in normalized:
            normalized[target_key] = value
    return normalized


def load_imported_metric_summary(
    root_path: Path, config: GaussianShadingOfficialReferenceConfig
) -> tuple[dict[str, Any], list[str]]:
    """从用户提供的 summary 或日志路径导入官方复现指标。"""

    evidence_paths: list[str] = []
    if config.summary_import_path:
        summary_path = Path(config.summary_import_path)
        summary_path = summary_path if summary_path.is_absolute() else root_path / summary_path
        if summary_path.is_file():
            evidence_paths.append(relative_or_absolute(summary_path, root_path))
            return normalize_metric_summary(read_json(summary_path), config.sample_count), evidence_paths
    if config.log_import_path:
        log_path = Path(config.log_import_path)
        log_path = log_path if log_path.is_absolute() else root_path / log_path
        if log_path.is_file():
            evidence_paths.append(relative_or_absolute(log_path, root_path))
            return normalize_metric_summary(
                parse_metric_text(log_path.read_text(encoding="utf-8", errors="ignore"), config.sample_count),
                config.sample_count,
            ), evidence_paths
    return {}, evidence_paths


def run_official_command_if_requested(
    root_path: Path,
    config: GaussianShadingOfficialReferenceConfig,
    paths: dict[str, Path],
    progress: object | None = None,
) -> dict[str, Any]:
    """根据配置执行 Gaussian Shading 官方命令, 并保存 stdout / stderr。"""

    if not config.run_official_command:
        return {
            "official_command_requested": False,
            "official_command": [],
            "return_code": -1,
            "stdout_path": "",
            "stderr_path": "",
        }
    if config.require_cuda:
        try:
            ensure_cuda_if_requested(True)
        except Exception as error:
            paths["official_stdout"].write_text("", encoding="utf-8")
            paths["official_stderr"].write_text(f"{type(error).__name__}:{error}", encoding="utf-8")
            result = {
                "official_command_requested": True,
                "official_command": [],
                "return_code": 97,
                "stdout_path": relative_or_absolute(paths["official_stdout"], root_path),
                "stderr_path": relative_or_absolute(paths["official_stderr"], root_path),
                "error": f"{type(error).__name__}:{error}",
            }
            write_json(paths["official_command_result"], result)
            return result
    source_status = source_report(root_path, config)
    if not source_status["official_entrypoint_ready"]:
        paths["official_stdout"].write_text("", encoding="utf-8")
        paths["official_stderr"].write_text("gaussian_shading_official_source_entrypoint_missing", encoding="utf-8")
        result = {
            "official_command_requested": True,
            "official_command": build_official_command(root_path, config, paths),
            "return_code": 96,
            "stdout_path": relative_or_absolute(paths["official_stdout"], root_path),
            "stderr_path": relative_or_absolute(paths["official_stderr"], root_path),
            "error": "gaussian_shading_official_source_entrypoint_missing",
        }
        write_json(paths["official_command_result"], result)
        return result
    paths["official_output_dir"].mkdir(parents=True, exist_ok=True)
    command = build_official_command(root_path, config, paths)
    source_dir = (root_path / config.source_dir).resolve()
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    try:
        completed = run_quiet_subprocess_with_progress(
            command,
            cwd=source_dir,
            env=env,
            timeout_seconds=int(config.timeout_seconds),
            progress=progress,
            progress_profile=f"operation=gaussian_shading_official_command samples={config.sample_count}",
        )
    except Exception as error:
        paths["official_stdout"].write_text("", encoding="utf-8")
        paths["official_stderr"].write_text(f"{type(error).__name__}:{error}", encoding="utf-8")
        result = {
            "official_command_requested": True,
            "official_command": command,
            "return_code": 98,
            "stdout_path": relative_or_absolute(paths["official_stdout"], root_path),
            "stderr_path": relative_or_absolute(paths["official_stderr"], root_path),
            "error": f"{type(error).__name__}:{error}",
        }
        write_json(paths["official_command_result"], result)
        return result
    paths["official_stdout"].write_text(completed.stdout, encoding="utf-8")
    paths["official_stderr"].write_text(completed.stderr, encoding="utf-8")
    result = {
        "official_command_requested": True,
        "official_command": command,
        "return_code": int(completed.returncode),
        "stdout_path": relative_or_absolute(paths["official_stdout"], root_path),
        "stderr_path": relative_or_absolute(paths["official_stderr"], root_path),
        "official_metric_text_path": relative_or_absolute(paths["official_metric_text"], root_path),
    }
    write_json(paths["official_command_result"], result)
    return result


def write_official_command_skip_result(
    root_path: Path,
    paths: dict[str, Path],
    *,
    return_code: int,
    reason: str,
) -> dict[str, Any]:
    """在前置环境不可用时写出官方命令跳过诊断。"""

    paths["official_stdout"].write_text("", encoding="utf-8")
    paths["official_stderr"].write_text(reason, encoding="utf-8")
    result = {
        "official_command_requested": True,
        "official_command": [],
        "return_code": int(return_code),
        "stdout_path": relative_or_absolute(paths["official_stdout"], root_path),
        "stderr_path": relative_or_absolute(paths["official_stderr"], root_path),
        "error": reason,
    }
    write_json(paths["official_command_result"], result)
    return result


def build_reference_record_report(
    root_path: Path,
    config: GaussianShadingOfficialReferenceConfig,
    paths: dict[str, Path],
    metric_summary: dict[str, Any],
    evidence_paths: list[str],
    source_status: dict[str, Any],
    legacy_environment_report: dict[str, Any],
) -> dict[str, Any]:
    """构造 governed import 记录并写出 schema、records 与 validation report。"""

    schema = build_gaussian_shading_official_reference_schema()
    write_json(paths["reference_schema"], schema)
    if not metric_summary:
        validation = validate_gaussian_shading_official_reference_records([])
        paths["reference_records"].write_text("", encoding="utf-8")
        write_json(paths["reference_validation"], validation)
        return {"record_count": 0, "validation": validation}

    write_json(paths["official_metric_summary"], metric_summary)
    local_evidence_paths = list(evidence_paths)
    for candidate in (
        paths["official_metric_summary"],
        paths["official_command_result"],
        paths["official_stdout"],
        paths["official_stderr"],
        paths["official_metric_text"],
    ):
        if candidate.is_file():
            local_evidence_paths.append(relative_or_absolute(candidate, root_path))
    result_source = relative_or_absolute(paths["official_metric_summary"], root_path)
    result_digest = file_digest(paths["official_metric_summary"])
    record = build_gaussian_shading_official_reference_record(
        official_entrypoint=str(source_status.get("official_entrypoint", "")),
        official_repository_commit="09c678fadc7545acf7be12647ddf2a5e66f6a9dc",
        official_environment_profile=str(
            legacy_environment_report.get("legacy_environment_profile")
            or "legacy_gaussian_shading_official_environment"
        ),
        baseline_result_source=result_source,
        baseline_result_source_digest=result_digest,
        evidence_paths=sorted(set(local_evidence_paths)),
        metric_values=metric_summary,
        ready_flags={
            "official_source_ready": bool(source_status.get("official_entrypoint_ready")),
            "official_environment_report_ready": True,
            "official_result_summary_ready": bool(metric_summary),
            "governed_import_ready": True,
        },
    )
    paths["reference_records"].write_text(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    validation = validate_gaussian_shading_official_reference_records([record])
    write_json(paths["reference_validation"], validation)
    return {"record_count": 1, "validation": validation, "record_digest": record["reference_record_digest"]}


def write_gaussian_shading_official_reference_outputs(
    config: GaussianShadingOfficialReferenceConfig,
    root: str | Path = ".",
) -> dict[str, Any]:
    """执行 Gaussian Shading 官方参考 workflow 并写出 summary、manifest 和 governed import 记录。"""

    root_path = Path(root).resolve()
    paths = output_paths(root_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["official_output_dir"].mkdir(parents=True, exist_ok=True)
    with progress_bar(10, desc="gaussian shading official reference", enabled=config.enable_workflow_progress_bar) as run_progress:
        effective_config = config
        source_status = ensure_gaussian_shading_source_available(root_path, effective_config, paths, progress=run_progress)
        update_progress(run_progress, profile="operation=ensure_gaussian_shading_source")
        emit_progress_status(run_progress, profile="operation=patch_gaussian_shading_source status=running")
        source_patch_report = patch_gaussian_shading_model_repository_layout(root_path, effective_config, paths)
        update_progress(run_progress, profile="operation=patch_gaussian_shading_source")
        legacy_environment_report = prepare_gaussian_shading_legacy_environment(root_path, config, paths, progress=run_progress)
        update_progress(run_progress, profile="operation=prepare_gaussian_shading_legacy_environment")
        if config.prepare_legacy_environment and legacy_environment_report.get("legacy_environment_ready"):
            effective_config = replace(config, official_python_executable=str(legacy_environment_report["legacy_python_executable"]))
        emit_progress_status(run_progress, profile="operation=ensure_cuda status=running")
        try:
            device_report = ensure_cuda_if_requested(effective_config.require_cuda)
        except Exception as error:
            device_report = {"cuda_available": False, "device_error": f"{type(error).__name__}:{error}"}
        update_progress(run_progress, profile="operation=ensure_cuda")
        should_prepare_model_repository = effective_config.run_official_command and not (
            effective_config.prepare_legacy_environment and not legacy_environment_report.get("legacy_environment_ready")
        )
        model_repository_config = effective_config if should_prepare_model_repository else replace(effective_config, prepare_local_model_repository=False)
        model_repository_report = prepare_gaussian_shading_model_repository(root_path, model_repository_config, paths, progress=run_progress)
        update_progress(run_progress, profile="operation=prepare_gaussian_shading_model_repository")
        if model_repository_report.get("local_model_repository_ready"):
            effective_config = replace(effective_config, official_model_id=str(model_repository_report["effective_official_model_id"]))
        emit_progress_status(run_progress, profile="operation=load_imported_metric_summary status=running")
        imported_metrics, imported_evidence = load_imported_metric_summary(root_path, effective_config)
        update_progress(run_progress, profile="operation=load_imported_metric_summary")
        if (
            effective_config.run_official_command
            and effective_config.prepare_legacy_environment
            and not legacy_environment_report.get("legacy_environment_ready")
        ):
            official_report = write_official_command_skip_result(
                root_path,
                paths,
                return_code=95,
                reason="gaussian_shading_legacy_environment_prepare_failed",
            )
        else:
            official_report = run_official_command_if_requested(root_path, effective_config, paths, progress=run_progress)
        update_progress(run_progress, profile="operation=gaussian_shading_official_command")
        emit_progress_status(run_progress, profile="operation=parse_gaussian_shading_metrics status=running")
        command_metrics: dict[str, Any] = {}
        if official_report.get("return_code") == 0 and paths["official_metric_text"].is_file():
            command_metrics = parse_metric_text(
                paths["official_metric_text"].read_text(encoding="utf-8", errors="ignore"),
                effective_config.sample_count,
            )
        metric_summary = normalize_metric_summary({**imported_metrics, **command_metrics}, effective_config.sample_count) if (
            imported_metrics or command_metrics
        ) else {}
        update_progress(run_progress, profile="operation=parse_gaussian_shading_metrics")
        emit_progress_status(run_progress, profile="operation=write_environment_report status=running")
        environment_report = build_runtime_environment_report()
        environment_report["gaussian_shading_official_reference_device_report"] = device_report
        environment_report["gaussian_shading_official_reference_source_report"] = source_status
        environment_report["gaussian_shading_official_reference_source_patch_report"] = source_patch_report
        environment_report["gaussian_shading_official_reference_model_repository_report"] = model_repository_report
        environment_report["gaussian_shading_official_reference_legacy_environment_report"] = legacy_environment_report
        write_json(paths["environment_report"], environment_report)
        update_progress(run_progress, profile="operation=write_environment_report")
        emit_progress_status(run_progress, profile="operation=build_reference_record status=running")
        record_report = build_reference_record_report(
            root_path,
            effective_config,
            paths,
            metric_summary,
            imported_evidence,
            source_status,
            legacy_environment_report,
        )
        update_progress(run_progress, profile=f"operation=build_reference_record records={record_report.get('record_count', 0)}")
    validation = record_report["validation"]
    run_ready = bool(validation.get("reference_import_ready"))
    unsupported_reason = "" if run_ready else "gaussian_shading_official_reference_result_missing_or_invalid"
    if official_report.get("official_command_requested") and int(official_report.get("return_code", 1)) != 0:
        unsupported_reason = "official_command_failed_governed_diagnostics_packaged"
    summary = {
        "run_decision": "pass" if run_ready else "fail",
        "gaussian_shading_official_reference_ready": run_ready,
        "official_command_requested": bool(official_report.get("official_command_requested")),
        "official_command_return_code": int(official_report.get("return_code", -1)),
        "sample_count": int(effective_config.sample_count),
        "paper_claim_scale": build_paper_run_config(root_path).run_name,
        "legacy_environment_requested": bool(legacy_environment_report.get("legacy_environment_requested")),
        "legacy_environment_ready": bool(legacy_environment_report.get("legacy_environment_ready")),
        "legacy_environment_profile": str(legacy_environment_report.get("legacy_environment_profile", "")),
        "strict_official_environment_ready": bool(legacy_environment_report.get("strict_official_environment_ready")),
        "compatible_environment_fallback_ready": bool(legacy_environment_report.get("compatible_environment_fallback_ready")),
        "source_patch_applied": bool(source_patch_report.get("patch_applied")),
        "local_model_repository_ready": bool(model_repository_report.get("local_model_repository_ready")),
        "model_index_patch_applied": bool(model_repository_report.get("model_index_patch_applied")),
        "official_model_id": effective_config.official_model_id,
        "upstream_official_model_id": effective_config.upstream_official_model_id,
        "governed_reference_record_count": int(record_report.get("record_count", 0)),
        "reference_import_ready": bool(validation.get("reference_import_ready")),
        "main_table_eligible": False,
        "supports_paper_claim": False,
        "unsupported_reason": unsupported_reason,
        "summary_path": relative_or_absolute(paths["summary"], root_path),
        "environment_report_path": relative_or_absolute(paths["environment_report"], root_path),
        "reference_records_path": relative_or_absolute(paths["reference_records"], root_path),
        "reference_validation_path": relative_or_absolute(paths["reference_validation"], root_path),
        "official_metric_text_path": relative_or_absolute(paths["official_metric_text"], root_path),
        "metadata": {
            "source_report": source_status,
            "source_patch_report": source_patch_report,
            "model_repository_report": model_repository_report,
            "legacy_environment_report": legacy_environment_report,
            "official_report": official_report,
            "validation": validation,
        },
    }
    write_json(paths["summary"], summary)
    output_paths_for_manifest = [
        relative_or_absolute(paths["summary"], root_path),
        relative_or_absolute(paths["environment_report"], root_path),
        relative_or_absolute(paths["reference_schema"], root_path),
        relative_or_absolute(paths["reference_records"], root_path),
        relative_or_absolute(paths["reference_validation"], root_path),
    ]
    for optional_path in (
        paths["official_command_result"],
        paths["official_stdout"],
        paths["official_stderr"],
        paths["official_metric_summary"],
        paths["official_metric_text"],
        paths["source_prepare_result"],
        paths["source_patch_result"],
        paths["model_repository_prepare_result"],
        paths["legacy_environment_prepare_result"],
    ):
        if optional_path.exists():
            output_paths_for_manifest.append(relative_or_absolute(optional_path, root_path))
    manifest = build_artifact_manifest(
        artifact_id="gaussian_shading_official_reference_manifest",
        artifact_type="local_manifest",
        input_paths=(relative_or_absolute(root_path / config.source_dir, root_path),),
        output_paths=tuple(output_paths_for_manifest + [relative_or_absolute(paths["manifest"], root_path)]),
        config=asdict(effective_config),
        code_version=resolve_code_version(root_path),
        rebuild_command="运行 paper_workflow/gaussian_shading_official_reference_run.ipynb",
        metadata={
            "run_decision": summary["run_decision"],
            "gaussian_shading_official_reference_ready": run_ready,
            "main_table_eligible": False,
            "supports_paper_claim": False,
        },
    ).to_dict()
    write_json(paths["manifest"], manifest)
    return summary


def build_default_config() -> GaussianShadingOfficialReferenceConfig:
    """从环境变量构造默认 Colab 运行配置。"""

    paper_run = build_paper_run_config(".")
    return GaussianShadingOfficialReferenceConfig(
        output_dir=os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        drive_output_dir=os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_DRIVE_OUTPUT_DIR", paper_run.drive_dir("gaussian_shading_official_reference")),
        source_dir=os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_SOURCE_DIR", DEFAULT_SOURCE_DIR),
        run_name=os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_RUN_NAME", DEFAULT_RUN_NAME),
        sample_count=resolve_count_from_environment("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_SAMPLE_COUNT", default_value=paper_run.sample_count),
        official_output_subdir=os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_OUTPUT_SUBDIR", DEFAULT_OUTPUT_SUBDIR),
        official_model_id=os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_MODEL_ID", DEFAULT_OFFICIAL_MODEL_ID),
        upstream_official_model_id=os.environ.get(
            "SLM_WM_GAUSSIAN_SHADING_UPSTREAM_OFFICIAL_MODEL_ID", DEFAULT_UPSTREAM_OFFICIAL_MODEL_ID
        ),
        model_source_note=os.environ.get("SLM_WM_GAUSSIAN_SHADING_MODEL_SOURCE_NOTE", DEFAULT_MODEL_SOURCE_NOTE),
        dataset_path=os.environ.get("SLM_WM_GAUSSIAN_SHADING_DATASET_PATH", "Gustavosta/Stable-Diffusion-Prompts"),
        fpr=float(os.environ.get("SLM_WM_GAUSSIAN_SHADING_FPR", "0.000001")),
        channel_copy=int(os.environ.get("SLM_WM_GAUSSIAN_SHADING_CHANNEL_COPY", "1")),
        hw_copy=int(os.environ.get("SLM_WM_GAUSSIAN_SHADING_HW_COPY", "8")),
        user_number=int(os.environ.get("SLM_WM_GAUSSIAN_SHADING_USER_NUMBER", "1000000")),
        gen_seed=int(os.environ.get("SLM_WM_GAUSSIAN_SHADING_GEN_SEED", "0")),
        image_length=int(os.environ.get("SLM_WM_GAUSSIAN_SHADING_IMAGE_LENGTH", "512")),
        guidance_scale=float(os.environ.get("SLM_WM_GAUSSIAN_SHADING_GUIDANCE_SCALE", "7.5")),
        num_inference_steps=int(os.environ.get("SLM_WM_GAUSSIAN_SHADING_NUM_INFERENCE_STEPS", "50")),
        num_inversion_steps=int(os.environ.get("SLM_WM_GAUSSIAN_SHADING_NUM_INVERSION_STEPS", "50")),
        use_chacha=os.environ.get("SLM_WM_GAUSSIAN_SHADING_USE_CHACHA", "1") != "0",
        reference_model=os.environ.get("SLM_WM_GAUSSIAN_SHADING_REFERENCE_MODEL", ""),
        reference_model_pretrain=os.environ.get("SLM_WM_GAUSSIAN_SHADING_REFERENCE_MODEL_PRETRAIN", ""),
        patch_model_repository_layout=os.environ.get("SLM_WM_GAUSSIAN_SHADING_PATCH_MODEL_REPOSITORY_LAYOUT", "1") != "0",
        prepare_local_model_repository=os.environ.get("SLM_WM_GAUSSIAN_SHADING_PREPARE_LOCAL_MODEL_REPOSITORY", "1") != "0",
        local_model_repository_dir=os.environ.get(
            "SLM_WM_GAUSSIAN_SHADING_LOCAL_MODEL_REPOSITORY_DIR", DEFAULT_LOCAL_MODEL_REPOSITORY_DIR
        ),
        patch_model_index_for_legacy_transformers=os.environ.get(
            "SLM_WM_GAUSSIAN_SHADING_PATCH_MODEL_INDEX_FOR_LEGACY_TRANSFORMERS", "1"
        )
        != "0",
        official_python_executable=os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_PYTHON_EXECUTABLE", ""),
        prepare_legacy_environment=os.environ.get("SLM_WM_GAUSSIAN_SHADING_PREPARE_LEGACY_ENV", "0") == "1",
        legacy_environment_prefix=os.environ.get("SLM_WM_GAUSSIAN_SHADING_LEGACY_ENV_PREFIX", DEFAULT_LEGACY_ENV_PREFIX),
        micromamba_path=os.environ.get("SLM_WM_GAUSSIAN_SHADING_MICROMAMBA_PATH", DEFAULT_MICROMAMBA_PATH),
        legacy_python_version=os.environ.get("SLM_WM_GAUSSIAN_SHADING_LEGACY_PYTHON_VERSION", DEFAULT_LEGACY_PYTHON_VERSION),
        strict_official_environment=os.environ.get("SLM_WM_GAUSSIAN_SHADING_STRICT_OFFICIAL_ENV", "1") != "0",
        allow_compatible_environment_fallback=os.environ.get(
            "SLM_WM_GAUSSIAN_SHADING_ALLOW_COMPATIBLE_ENV_FALLBACK", "1"
        ) != "0",
        legacy_torch_specs=os.environ.get("SLM_WM_GAUSSIAN_SHADING_LEGACY_TORCH_SPECS", DEFAULT_LEGACY_TORCH_SPECS),
        legacy_pytorch_index_url=os.environ.get(
            "SLM_WM_GAUSSIAN_SHADING_LEGACY_PYTORCH_INDEX_URL", DEFAULT_LEGACY_PYTORCH_INDEX_URL
        ),
        legacy_package_specs=os.environ.get("SLM_WM_GAUSSIAN_SHADING_LEGACY_PACKAGE_SPECS", DEFAULT_LEGACY_PACKAGE_SPECS),
        run_official_command=os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_RUN_COMMAND", "1") != "0",
        summary_import_path=os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_SUMMARY_IMPORT_PATH", ""),
        log_import_path=os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_LOG_IMPORT_PATH", ""),
        require_cuda=os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_REQUIRE_CUDA", "1") != "0",
        timeout_seconds=int(os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_TIMEOUT_SECONDS", "86400")),
        enable_workflow_progress_bar=os.environ.get("SLM_WM_ENABLE_WORKFLOW_PROGRESS_BAR", "1") != "0",
    )


def run_default_gaussian_shading_official_reference_plan(root: str | Path = ".") -> dict[str, Any]:
    """运行默认 Gaussian Shading 官方参考 workflow。"""

    return write_gaussian_shading_official_reference_outputs(config=build_default_config(), root=root)


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


def package_gaussian_shading_official_reference_outputs(
    root: str | Path = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    drive_output_dir: str | None = None,
    archive_name: str = "gaussian_shading_official_reference_package.zip",
) -> GaussianShadingOfficialReferenceArchiveRecord:
    """打包 Gaussian Shading 官方参考产物并镜像到 Google Drive。"""

    root_path = Path(root).resolve()
    resolved_drive_output_dir = drive_output_dir or build_paper_run_config(root_path).drive_dir("gaussian_shading_official_reference")
    source_dir = (root_path / output_dir).resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_path = source_dir / archive_name
    package_manifest_path = source_dir / "gaussian_shading_official_reference_package_input_manifest.json"
    summary_path = source_dir / "gaussian_shading_official_reference_archive_summary.json"
    manifest_path = source_dir / "gaussian_shading_official_reference_archive_manifest.local.json"
    for stale_path in (package_manifest_path, summary_path, manifest_path):
        if stale_path.exists():
            stale_path.unlink()
    content_entries = collect_package_entries(root_path, source_dir, archive_path)
    entries = tuple((*content_entries, package_manifest_path, summary_path, manifest_path))
    package_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entry_paths": [entry.relative_to(root_path).as_posix() for entry in entries],
        "entry_count": len(entries),
        "embedded_digest_scope": "external_summary_records_final_archive_digest",
    }
    write_json(package_manifest_path, package_manifest)
    preliminary_record = GaussianShadingOfficialReferenceArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest="",
        archive_entry_count=len(entries),
        drive_archive_path=str(Path(resolved_drive_output_dir).expanduser() / archive_name),
        drive_archive_digest="",
        metadata={
            "drive_output_dir": str(Path(resolved_drive_output_dir).expanduser()),
            "embedded_digest_scope": "external_summary_records_final_archive_digest",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(summary_path, preliminary_record.to_dict())
    archive_manifest = build_artifact_manifest(
        artifact_id="gaussian_shading_official_reference_archive_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(
            [entry.relative_to(root_path).as_posix() for entry in content_entries]
            + [package_manifest_path.relative_to(root_path).as_posix()]
        ),
        output_paths=(
            archive_path.relative_to(root_path).as_posix(),
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ),
        config={"archive_name": archive_name, "drive_output_dir": str(Path(resolved_drive_output_dir).expanduser())},
        code_version=resolve_code_version(root_path),
        rebuild_command="运行 paper_workflow/gaussian_shading_official_reference_run.ipynb",
        metadata={
            "embedded_digest_scope": "external_summary_records_final_archive_digest",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "main_table_eligible": False,
        },
    ).to_dict()
    write_json(manifest_path, archive_manifest)
    with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.write(entry, entry.relative_to(root_path).as_posix())
    drive_dir = Path(resolved_drive_output_dir).expanduser()
    drive_dir.mkdir(parents=True, exist_ok=True)
    mirrored_path = drive_dir / archive_name
    shutil.copy2(archive_path, mirrored_path)
    record = GaussianShadingOfficialReferenceArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest=file_digest(archive_path),
        archive_entry_count=len(entries),
        drive_archive_path=str(mirrored_path),
        drive_archive_digest=file_digest(mirrored_path),
        metadata={"drive_output_dir": str(drive_dir), "generated_at": datetime.now(timezone.utc).isoformat()},
    )
    write_json(summary_path, record.to_dict())
    archive_manifest.setdefault("metadata", {})["archive_digest"] = record.archive_digest
    archive_manifest.setdefault("metadata", {})["drive_archive_digest"] = record.drive_archive_digest
    write_json(manifest_path, archive_manifest)
    return record
