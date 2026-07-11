"""Shallow Diffuse 官方原始环境复现与 governed import 的 Colab 辅助函数。

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
from experiments.runtime import repository_environment
from paper_experiments.baselines import (
    build_shallow_diffuse_official_reference_record,
    build_shallow_diffuse_official_reference_schema,
    validate_shallow_diffuse_official_reference_records,
)
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from paper_experiments.runners.external_source_runtime import (
    ensure_cuda_if_requested,
    load_baseline_registry_item,
    normalize_repository_url,
    run_command,
)
from experiments.runtime.progress import (
    call_runner_with_progress_status,
    emit_progress_status,
    progress_bar,
    run_quiet_subprocess_with_progress,
    update_progress,
)
from experiments.runtime.archive_naming import utc_archive_token
from experiments.runtime.repository_environment import (
    build_runtime_environment_report,
    file_digest,
)

DEFAULT_OUTPUT_DIR = "outputs/shallow_diffuse_official_reference"
DEFAULT_DRIVE_OUTPUT_DIR = ""
DEFAULT_SOURCE_DIR = "external_baseline/primary/shallow_diffuse/source"
DEFAULT_RUN_NAME = "shallow_diffuse_official_legacy_reference"
DEFAULT_SAMPLE_COUNT = 700
DEFAULT_UPSTREAM_OFFICIAL_MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
DEFAULT_OFFICIAL_MODEL_ID = "Manojb/stable-diffusion-2-1-base"
DEFAULT_MODEL_SOURCE_NOTE = (
    "Shallow Diffuse 官方 README 默认使用 stabilityai/stable-diffusion-2-1-base; "
    "当该模型不可直接访问时, 默认改用公开镜像并保留该模型来源说明。"
)
DEFAULT_LOCAL_MODEL_REPOSITORY_DIR = "/content/shallow_diffuse_model_repository/stable_diffusion_2_1_base"
DEFAULT_LEGACY_ENV_PREFIX = "/content/shallow_diffuse_legacy_env"
DEFAULT_MICROMAMBA_PATH = "/content/bin/micromamba"
DEFAULT_LEGACY_PYTHON_VERSION = "3.9"
DEFAULT_LEGACY_TORCH_SPECS = "torch==1.13.0+cu117 torchvision==0.14.0+cu117"
DEFAULT_LEGACY_PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu117"
DEFAULT_LEGACY_PACKAGE_SPECS = (
    "transformers==4.23.1 diffusers==0.11.1 huggingface_hub==0.10.1 "
    "datasets==2.6.1 pyarrow<13 fsspec==2022.10.0 numpy==1.24.4 scipy==1.10.1 "
    "Pillow==9.5.0 tqdm==4.66.2 scikit-learn==1.3.2 wandb==0.16.6 "
    "open_clip_torch==2.7.0 ftfy==6.2.0 regex==2023.12.25 Requests==2.31.0 "
    "omegaconf==2.3.0 einops==0.4.1 matplotlib==3.7.5 timm==0.5.4 "
    "opencv-python-headless==4.9.0.80"
)
DEFAULT_EDIT_TIME_LIST = "0.3"
DEFAULT_ATTACKER_NAMES = "none"
DEFAULT_REFERENCE_MODEL = "ViT-g-14"
DEFAULT_REFERENCE_MODEL_PRETRAIN = "laion2b_s12b_b42k"
@dataclass(frozen=True)
class ShallowDiffuseOfficialReferenceConfig:
    """描述 Shallow Diffuse 官方原始环境复现与导入所需配置。"""

    output_dir: str = DEFAULT_OUTPUT_DIR
    drive_output_dir: str = field(default_factory=lambda: build_paper_run_config(".").drive_dir("external_baseline_official_reference"))
    source_dir: str = DEFAULT_SOURCE_DIR
    run_name: str = DEFAULT_RUN_NAME
    sample_count: int = DEFAULT_SAMPLE_COUNT
    start_index: int = 0
    official_model_id: str = DEFAULT_OFFICIAL_MODEL_ID
    upstream_official_model_id: str = DEFAULT_UPSTREAM_OFFICIAL_MODEL_ID
    model_source_note: str = DEFAULT_MODEL_SOURCE_NOTE
    dataset: str = "Gustavosta/Stable-Diffusion-Prompts"
    image_length: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    edit_time_list: str = DEFAULT_EDIT_TIME_LIST
    w_seed: int = 42
    w_channel: int = 3
    w_pattern: str = "complex2_ring"
    w_mask_shape: str = "circle"
    w_radius: int = 10
    w_measurement: str = "l1_complex2"
    w_injection: str = "complex2"
    reference_model: str = DEFAULT_REFERENCE_MODEL
    reference_model_pretrain: str = DEFAULT_REFERENCE_MODEL_PRETRAIN
    attacker_names: str = DEFAULT_ATTACKER_NAMES
    patch_model_repository_layout: bool = True
    prepare_local_model_repository: bool = True
    local_model_repository_dir: str = DEFAULT_LOCAL_MODEL_REPOSITORY_DIR
    patch_model_index_for_legacy_transformers: bool = True
    official_python_executable: str = ""
    prepare_legacy_environment: bool = False
    legacy_environment_prefix: str = DEFAULT_LEGACY_ENV_PREFIX
    micromamba_path: str = DEFAULT_MICROMAMBA_PATH
    legacy_python_version: str = DEFAULT_LEGACY_PYTHON_VERSION
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
class ShallowDiffuseOfficialReferenceArchiveRecord:
    """记录 Shallow Diffuse 官方参考压缩包与 Drive 镜像信息。"""

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


def parse_edit_time_values(edit_time_list: str) -> list[float]:
    """解析官方脚本使用的逗号分隔 watermark 注入时间比例。"""

    values = [float(item.strip()) for item in str(edit_time_list).split(",") if item.strip()]
    return values or [float(DEFAULT_EDIT_TIME_LIST)]


def primary_edit_timestep(config: ShallowDiffuseOfficialReferenceConfig) -> int:
    """计算默认核对文件所在的官方 timestep 目录。"""

    return int(parse_edit_time_values(config.edit_time_list)[0] * int(config.num_inference_steps))


def command_exception_result(command: Any, error: Exception) -> dict[str, Any]:
    """把环境准备异常转换为可落盘命令诊断, 避免 Notebook 直接中断。"""

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


def output_paths(root_path: Path, config: ShallowDiffuseOfficialReferenceConfig) -> dict[str, Path]:
    """集中构造 Shallow Diffuse 官方参考 workflow 的输出路径。"""

    paper_run = build_paper_run_config(root_path)
    configured_output_root = (root_path / config.output_dir).resolve()
    expected_output_root = (root_path / DEFAULT_OUTPUT_DIR).resolve()
    if configured_output_root != expected_output_root:
        raise ValueError("Shallow Diffuse 官方参考输出根目录必须使用正式 outputs family")
    output_dir = expected_output_root / paper_run.run_name
    official_run_root = output_dir / "output" / config.run_name
    official_timestep_dir = official_run_root / f"timestep{primary_edit_timestep(config)}"
    return {
        "output_dir": output_dir,
        "official_run_root": official_run_root,
        "official_timestep_dir": official_timestep_dir,
        "official_overall_scores": official_timestep_dir / "overall_scores.txt",
        "official_clip_scores": official_timestep_dir / "clip_scores.txt",
        "official_config_log": official_run_root / "config.log",
        "official_command_result": output_dir / "shallow_diffuse_official_command_result.json",
        "source_prepare_result": output_dir / "shallow_diffuse_official_source_prepare_result.json",
        "source_patch_result": output_dir / "shallow_diffuse_official_source_patch_result.json",
        "model_repository_prepare_result": output_dir / "shallow_diffuse_model_repository_prepare_result.json",
        "legacy_environment_prepare_result": output_dir / "shallow_diffuse_legacy_environment_prepare_result.json",
        "official_stdout": output_dir / "shallow_diffuse_official_stdout.txt",
        "official_stderr": output_dir / "shallow_diffuse_official_stderr.txt",
        "official_metric_summary": output_dir / "shallow_diffuse_official_metric_summary.json",
        "reference_schema": output_dir / "shallow_diffuse_official_reference_schema.json",
        "reference_records": output_dir / "shallow_diffuse_official_reference_records.jsonl",
        "reference_validation": output_dir / "shallow_diffuse_official_reference_validation_report.json",
        "environment_report": output_dir / "shallow_diffuse_official_reference_environment_report.json",
        "summary": output_dir / "shallow_diffuse_official_reference_summary.json",
        "manifest": output_dir / "manifest.local.json",
    }


def prepare_shallow_diffuse_legacy_environment(
    root_path: Path,
    config: ShallowDiffuseOfficialReferenceConfig,
    paths: dict[str, Path],
    progress: object | None = None,
) -> dict[str, Any]:
    """在独立 Colab 会话中准备 Shallow Diffuse 官方 legacy Python 环境。"""

    if not config.prepare_legacy_environment:
        report = {
            "legacy_environment_requested": False,
            "legacy_environment_ready": False,
            "legacy_environment_skipped": True,
            "legacy_environment_skip_reason": "prepare_legacy_environment_disabled",
            "legacy_python_executable": config.official_python_executable or sys.executable,
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
        }
        write_json(paths["legacy_environment_prepare_result"], report)
        return report

    micromamba_path = Path(config.micromamba_path)
    legacy_env_prefix = Path(config.legacy_environment_prefix)
    legacy_python = legacy_env_prefix / "bin" / "python"
    command_results: list[dict[str, Any]] = []
    if not micromamba_path.is_file():
        micromamba_path.parent.mkdir(parents=True, exist_ok=True)
        command_results.append(
            run_shell_command_with_progress_status(
                f"curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C {micromamba_path.parent.parent} bin/micromamba",
                cwd=root_path,
                timeout_seconds=600,
                progress=progress,
                progress_profile="operation=shallow_diffuse_fetch_micromamba",
            )
        )
    if not legacy_python.is_file():
        if micromamba_path.is_file():
            command_results.append(
                run_argv_command(
                    [
                        str(micromamba_path),
                        "create",
                        "-y",
                        "-p",
                        str(legacy_env_prefix),
                        "-c",
                        "conda-forge",
                        f"python={config.legacy_python_version}",
                        "pip",
                    ],
                    cwd=root_path,
                    timeout_seconds=1800,
                    progress=progress,
                    progress_profile="operation=shallow_diffuse_create_legacy_environment",
                )
            )
    if legacy_python.is_file():
        command_results.append(
            run_argv_command(
                [str(legacy_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                cwd=root_path,
                timeout_seconds=900,
                progress=progress,
                progress_profile="operation=shallow_diffuse_upgrade_legacy_pip",
            )
        )
        torch_specs = split_package_specs(config.legacy_torch_specs)
        if torch_specs:
            torch_command = [str(legacy_python), "-m", "pip", "install"]
            if config.legacy_pytorch_index_url:
                torch_command.extend(["--extra-index-url", str(config.legacy_pytorch_index_url)])
            torch_command.extend(torch_specs)
            command_results.append(
                run_argv_command(
                    torch_command,
                    cwd=root_path,
                    timeout_seconds=1800,
                    progress=progress,
                    progress_profile="operation=shallow_diffuse_install_legacy_torch",
                )
            )
        package_specs = split_package_specs(config.legacy_package_specs)
        if package_specs:
            command_results.append(
                run_argv_command(
                    [str(legacy_python), "-m", "pip", "install", *package_specs],
                    cwd=root_path,
                    timeout_seconds=1800,
                    progress=progress,
                    progress_profile="operation=shallow_diffuse_install_legacy_packages",
                )
            )
    verify_command = [
        str(legacy_python),
        "-c",
        (
            "import json, torch, transformers, diffusers, datasets, open_clip, sklearn, wandb; "
            "from transformers import CLIPFeatureExtractor; "
            "print(json.dumps({"
            "'torch': torch.__version__, "
            "'cuda_available': bool(torch.cuda.is_available()), "
            "'transformers': transformers.__version__, "
            "'diffusers': diffusers.__version__, "
            "'datasets': datasets.__version__"
            "}))"
        ),
    ]
    verify_result = (
        run_argv_command(
            verify_command,
            cwd=root_path,
            timeout_seconds=300,
            progress=progress,
            progress_profile="operation=shallow_diffuse_verify_legacy_environment",
        )
        if legacy_python.is_file()
        else {"command": verify_command, "return_code": 127, "stdout": "", "stderr": "legacy_python_missing"}
    )
    command_results.append(verify_result)
    report = {
        "legacy_environment_requested": True,
        "legacy_environment_ready": legacy_python.is_file() and verify_result["return_code"] == 0,
        "legacy_environment_skipped": False,
        "legacy_environment_prefix": str(legacy_env_prefix),
        "micromamba_path": str(micromamba_path),
        "legacy_python_executable": str(legacy_python),
        "legacy_python_version": config.legacy_python_version,
        "legacy_torch_specs": split_package_specs(config.legacy_torch_specs),
        "legacy_package_specs": split_package_specs(config.legacy_package_specs),
        "legacy_pytorch_index_url": config.legacy_pytorch_index_url,
        "command_results": command_results,
        "verify_result": verify_result,
    }
    write_json(paths["legacy_environment_prepare_result"], report)
    return report


def source_report(root_path: Path, config: ShallowDiffuseOfficialReferenceConfig) -> dict[str, Any]:
    """检查官方 Shallow Diffuse 源码快照是否存在。"""

    source_dir = (root_path / config.source_dir).resolve()
    entrypoint = source_dir / "run_shallow_diffuse_t2i.py"
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


def ensure_shallow_diffuse_source_available(
    root_path: Path,
    config: ShallowDiffuseOfficialReferenceConfig,
    paths: dict[str, Path],
    progress: object | None = None,
) -> dict[str, Any]:
    """在 Colab 冷启动环境中按登记表补齐 Shallow Diffuse 官方源码。"""

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

    registry_item = load_baseline_registry_item(root_path, "shallow_diffuse")
    source_dir = (root_path / config.source_dir).resolve()
    source_entry = source_dir / "run_shallow_diffuse_t2i.py"
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
        progress_profile="operation=shallow_diffuse_source_clone",
    )
    checkout_result: dict[str, Any] = {"command": [], "return_code": 0, "stdout": "", "stderr": ""}
    if clone_result["return_code"] == 0 and registry_item.get("official_repository_commit"):
        checkout_result = run_command_with_progress_status(
            ["git", "checkout", str(registry_item["official_repository_commit"])],
            cwd=source_dir,
            timeout_seconds=300,
            progress=progress,
            progress_profile="operation=shallow_diffuse_source_checkout",
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



def patch_shallow_diffuse_model_repository_layout(
    root_path: Path,
    config: ShallowDiffuseOfficialReferenceConfig,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """应用官方源码运行补丁, 保持每个补丁都进入可审计报告。"""

    source_dir = (root_path / config.source_dir).resolve()
    entrypoint = source_dir / "run_shallow_diffuse_t2i.py"
    attackers_path = source_dir / "attackers.py"
    optim_utils_path = source_dir / "optim_utils.py"
    report = {
        "patch_requested": bool(config.patch_model_repository_layout),
        "patch_applied": False,
        "patch_skipped": False,
        "official_entrypoint": relative_or_absolute(entrypoint, root_path),
        "attackers_path": relative_or_absolute(attackers_path, root_path),
        "optim_utils_path": relative_or_absolute(optim_utils_path, root_path),
        "official_model_id": config.official_model_id,
        "upstream_official_model_id": config.upstream_official_model_id,
        "model_source_note": config.model_source_note,
        "patch_reason": "mirror_model_layout_and_official_attack_runtime_compatibility",
        "patch_items": [],
    }
    if not config.patch_model_repository_layout:
        report.update({"patch_skipped": True, "patch_skip_reason": "patch_model_repository_layout_disabled"})
        write_json(paths["source_patch_result"], report)
        return report
    if not entrypoint.is_file() or not attackers_path.is_file():
        report.update({"patch_skipped": True, "patch_skip_reason": "official_entrypoint_missing"})
        write_json(paths["source_patch_result"], report)
        return report

    source_text = entrypoint.read_text(encoding="utf-8")
    attackers_text = attackers_path.read_text(encoding="utf-8")
    optim_text = optim_utils_path.read_text(encoding="utf-8") if optim_utils_path.is_file() else ""
    before_digest = file_digest(entrypoint)
    attackers_before_digest = file_digest(attackers_path)
    optim_before_digest = file_digest(optim_utils_path) if optim_utils_path.is_file() else ""
    patched_text = source_text
    patched_attackers_text = attackers_text
    patched_optim_text = optim_text

    revision_marker = "# SLM-WM: 公开镜像没有 fp16 分支, 因此从 main 分支加载模型权重。"
    revision_target = "        revision='fp16',\n"
    if revision_marker not in patched_text and revision_target in patched_text:
        patched_text = patched_text.replace(revision_target, f"        {revision_marker}\n")
        report["patch_items"].append("remove_fp16_revision_branch")

    suffix_marker = "# SLM-WM: 通过环境变量收敛官方攻击器集合, 避免重型攻击器阻断小样本参考复现。"
    suffix_target = (
        "    suffixes = ['none', 'jpeg', 'gaussianblur', 'gaussianstd', 'colorjitter','randomdrop', "
        "'saltandpepper', 'resizerestore','vaebmshj', 'vaecheng', 'diff']\n"
    )
    suffix_replacement = (
        f"    {suffix_marker}\n"
        "    suffixes = [name.strip() for name in os.environ.get('SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_ATTACKER_NAMES', 'none').split(',') if name.strip()]\n"
        "    if not suffixes:\n"
        "        suffixes = ['none']\n"
    )
    if suffix_marker not in patched_text and suffix_target in patched_text:
        patched_text = patched_text.replace(suffix_target, suffix_replacement)
        report["patch_items"].append("environment_controlled_attacker_suffixes")

    attacker_filter_marker = "# SLM-WM: 仅运行本次参考复现声明的攻击器。"
    attacker_filter_anchor = "        'diffpure': image_distortion_diffpure,\n    }\n"
    attacker_filter_replacement = (
        "        'diffpure': image_distortion_diffpure,\n"
        "    }\n"
        f"    {attacker_filter_marker}\n"
        "    attacker_names = [name.strip() for name in os.environ.get('SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_ATTACKER_NAMES', 'none').split(',') if name.strip()]\n"
        "    if not attacker_names:\n"
        "        attacker_names = ['none']\n"
        "    attackers = {name: attackers[name] for name in attacker_names if name in attackers}\n"
    )
    if attacker_filter_marker not in patched_text and attacker_filter_anchor in patched_text:
        patched_text = patched_text.replace(attacker_filter_anchor, attacker_filter_replacement)
        report["patch_items"].append("filter_attacker_dictionary")

    similarity_marker = "# SLM-WM: 官方脚本此处变量名误写为 sim, 应使用上方 measure_similarity 返回的 sims。"
    similarity_target = "        w_no_sim = sim[0].item()\n"
    similarity_replacement = f"        {similarity_marker}\n        w_no_sim = sims[0].item()\n"
    if similarity_marker not in patched_text and similarity_target in patched_text:
        patched_text = patched_text.replace(similarity_target, similarity_replacement, 1)
        report["patch_items"].append("fix_reference_similarity_variable")
    else:
        similarity_target_compact = "    w_no_sim = sim[0].item()\n"
        similarity_replacement_compact = f"    {similarity_marker}\n    w_no_sim = sims[0].item()\n"
        if similarity_marker not in patched_text and similarity_target_compact in patched_text:
            patched_text = patched_text.replace(similarity_target_compact, similarity_replacement_compact, 1)
            report["patch_items"].append("fix_reference_similarity_variable")

    compressai_marker = "# SLM-WM: VAE 攻击器依赖按需导入, 默认 none 攻击不要求 compressai。"
    compressai_target = (
        "from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor\n"
    )
    compressai_replacement = (
        f"{compressai_marker}\n"
        "try:\n"
        "    from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor\n"
        "except Exception:\n"
        "    bmshj2018_factorized = bmshj2018_hyperprior = mbt2018_mean = mbt2018 = cheng2020_anchor = None\n"
    )
    if compressai_marker not in patched_attackers_text and compressai_target in patched_attackers_text:
        patched_attackers_text = patched_attackers_text.replace(compressai_target, compressai_replacement)
        report["patch_items"].append("lazy_compressai_import")

    initialize_marker = "# SLM-WM: 只为被请求的重型攻击器初始化模型。"
    initialize_start = patched_attackers_text.find("def initialize_attackers(args, device):")
    initialize_end = patched_attackers_text.find("\ndef image_distortion_none", initialize_start)
    if initialize_marker not in patched_attackers_text and initialize_start != -1 and initialize_end != -1:
        initialize_replacement = (
            "def initialize_attackers(args, device):\n"
            f"    {initialize_marker}\n"
            "    global vae_attacker1, vae_attacker2, diff_attacker\n"
            "    attacker_names = {name.strip() for name in os.environ.get('SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_ATTACKER_NAMES', 'none').split(',') if name.strip()}\n"
            "    if {'vaebmshj', 'vaecheng'} & attacker_names:\n"
            "        first_model = getattr(args, 'vae_attack_model_name1', getattr(args, 'vae_attack_model_name', 'bmshj2018_hyperprior'))\n"
            "        second_model = getattr(args, 'vae_attack_model_name2', 'cheng2020_anchor')\n"
            "        if bmshj2018_hyperprior is None:\n"
            "            raise RuntimeError('compressai_required_for_vae_attackers')\n"
            "        vae_attacker1 = VAEWMAttacker(first_model, quality=3, metric='mse', device=device)\n"
            "        vae_attacker2 = VAEWMAttacker(second_model, quality=3, metric='mse', device=device)\n"
            "    if 'diff' in attacker_names:\n"
            "        att_pipe = ReSDPipeline.from_pretrained('stabilityai/stable-diffusion-2-1', torch_dtype=torch.float16, revision='fp16')\n"
            "        att_pipe.set_progress_bar_config(disable=True)\n"
            "        att_pipe.to(device)\n"
            "        diff_attacker = DiffWMAttacker(att_pipe, batch_size=1, noise_step=60, captions={})\n"
        )
        patched_attackers_text = (
            patched_attackers_text[:initialize_start] + initialize_replacement + patched_attackers_text[initialize_end:]
        )
        report["patch_items"].append("lazy_heavy_attacker_initialization")

    fft_marker = "# SLM-WM: legacy CUDA half precision FFT 在部分 Colab GPU 上不稳定, 统一用 float32 执行频域变换。"
    fft_target = "torch.fft.fft2("
    fft_replacement = "slm_wm_fft2_float32("
    fft_helper_anchor = "import torch.nn.functional as F\n"
    fft_helper = (
        f"\n{fft_marker}\n"
        "def slm_wm_fft2_float32(tensor):\n"
        "    if tensor.dtype in (torch.float16, torch.bfloat16):\n"
        "        return torch.fft.fft2(tensor.float())\n"
        "    if tensor.is_complex() and str(tensor.dtype) == 'torch.complex32':\n"
        "        return torch.fft.fft2(tensor.to(torch.complex64))\n"
        "    return torch.fft.fft2(tensor)\n"
    )
    if optim_text and fft_marker not in patched_optim_text and fft_target in patched_optim_text:
        patched_optim_text = patched_optim_text.replace(fft_target, fft_replacement)
        if fft_helper_anchor in patched_optim_text:
            patched_optim_text = patched_optim_text.replace(fft_helper_anchor, fft_helper_anchor + fft_helper, 1)
        else:
            patched_optim_text = fft_helper + "\n" + patched_optim_text
        report["patch_items"].append("float32_fft_for_legacy_cuda")

    latent_dtype_marker = (
        "# SLM-WM: 频域注入完成后恢复 latent dtype, 保持与官方 fp16 pipeline 的输入契约一致。"
    )
    inject_function_header = "def inject_watermark(init_latents_w, watermarking_mask, gt_patch, w_injection):\n"
    if optim_text and latent_dtype_marker not in patched_optim_text and inject_function_header in patched_optim_text:
        patched_optim_text = patched_optim_text.replace(
            inject_function_header,
            inject_function_header
            + f"    {latent_dtype_marker}\n"
            + "    slm_wm_init_latents_dtype = init_latents_w.dtype\n",
            1,
        )
        patched_optim_text = patched_optim_text.replace(
            "        init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real\n",
            "        init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real.to(slm_wm_init_latents_dtype)\n",
            1,
        )
        patched_optim_text = patched_optim_text.replace(
            "        init_latents_w = torch.fft.ifft2(init_latents_w_fft).real\n",
            "        init_latents_w = torch.fft.ifft2(init_latents_w_fft).real.to(slm_wm_init_latents_dtype)\n",
            1,
        )
        report["patch_items"].append("preserve_latent_dtype_after_fft_injection")

    if patched_text == source_text and patched_attackers_text == attackers_text and patched_optim_text == optim_text:
        report.update(
            {
                "patch_skipped": True,
                "patch_skip_reason": "source_runtime_patch_already_present_or_targets_missing",
                "entrypoint_digest_before": before_digest,
                "entrypoint_digest_after": before_digest,
                "attackers_digest_before": attackers_before_digest,
                "attackers_digest_after": attackers_before_digest,
                "optim_utils_digest_before": optim_before_digest,
                "optim_utils_digest_after": optim_before_digest,
            }
        )
        write_json(paths["source_patch_result"], report)
        return report
    if patched_text != source_text:
        entrypoint.write_text(patched_text, encoding="utf-8")
    if patched_attackers_text != attackers_text:
        attackers_path.write_text(patched_attackers_text, encoding="utf-8")
    if patched_optim_text != optim_text and optim_utils_path.is_file():
        optim_utils_path.write_text(patched_optim_text, encoding="utf-8")
    report.update(
        {
            "patch_applied": True,
            "entrypoint_digest_before": before_digest,
            "entrypoint_digest_after": file_digest(entrypoint),
            "attackers_digest_before": attackers_before_digest,
            "attackers_digest_after": file_digest(attackers_path),
            "optim_utils_digest_before": optim_before_digest,
            "optim_utils_digest_after": file_digest(optim_utils_path) if optim_utils_path.is_file() else "",
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


def prepare_shallow_diffuse_model_repository(
    root_path: Path,
    config: ShallowDiffuseOfficialReferenceConfig,
    paths: dict[str, Path],
    progress: object | None = None,
) -> dict[str, Any]:
    """准备本地模型目录并补齐 legacy transformers 所需的 model_index 兼容项。"""

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
            emit_progress_status(progress, profile=f"operation=shallow_diffuse_model_snapshot_download model={config.official_model_id}")
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
        download_result.update({"return_code": 0, "snapshot_path": str(local_model_path), "download_skipped_reason": "model_index_already_exists"})

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
        report["model_index_patch_reason"] = "legacy_transformers_4_23_1_requires_clip_feature_extractor"
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


def build_official_command(root_path: Path, config: ShallowDiffuseOfficialReferenceConfig) -> list[str]:
    """构造 Shallow Diffuse 官方 legacy 入口命令。"""

    source_dir = (root_path / config.source_dir).resolve()
    entrypoint = source_dir / "run_shallow_diffuse_t2i.py"
    end_index = int(config.start_index) + max(1, int(config.sample_count))
    python_executable = config.official_python_executable or sys.executable
    return [
        python_executable,
        str(entrypoint),
        "--run_name",
        str(config.run_name),
        "--model_id",
        str(config.official_model_id),
        "--dataset",
        str(config.dataset),
        "--image_length",
        str(config.image_length),
        "--guidance_scale",
        str(config.guidance_scale),
        "--num_inference_steps",
        str(config.num_inference_steps),
        "--w_seed",
        str(config.w_seed),
        "--w_channel",
        str(config.w_channel),
        "--w_pattern",
        str(config.w_pattern),
        "--w_mask_shape",
        str(config.w_mask_shape),
        "--w_radius",
        str(config.w_radius),
        "--w_measurement",
        str(config.w_measurement),
        "--w_injection",
        str(config.w_injection),
        "--reference_model",
        str(config.reference_model),
        "--reference_model_pretrain",
        str(config.reference_model_pretrain),
        "--edit_time_list",
        str(config.edit_time_list),
        "--start",
        str(config.start_index),
        "--end",
        str(end_index),
    ]


def parse_metric_text(text: str, sample_count: int) -> dict[str, Any]:
    """从官方 stdout 或日志中解析 Shallow Diffuse 指标。"""

    patterns = {
        "clip_score_mean": r"(?m)^\s*clip_score_mean\s*:\s*([0-9eE+\-.]+)",
        "watermarked_clip_score_mean": r"(?m)^\s*(?:avg_clip_score_mean|w_clip_score_mean)\s*:\s*([0-9eE+\-.]+)",
        "auc": r"(?m)(?:^|[,\n])\s*auc\s*:\s*([0-9eE+\-.]+)",
        "accuracy": r"(?m)(?:^|[,\n])\s*acc\s*:\s*([0-9eE+\-.]+)",
        "true_positive_rate_at_one_percent_fpr": r"(?m)(?:^|[,\n])\s*TPR@1%FPR\s*:\s*([0-9eE+\-.]+)",
    }
    metrics: dict[str, Any] = {
        "sample_count": int(sample_count),
        "positive_count": int(sample_count),
        "negative_count": int(sample_count),
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
        "negative_count": int(sample_count),
        "auc": 0.0,
        "accuracy": 0.0,
        "true_positive_rate_at_one_percent_fpr": 0.0,
        "clip_score_mean": 0.0,
        "watermarked_clip_score_mean": 0.0,
    }
    aliases = {
        "acc": "accuracy",
        "TPR@1%FPR": "true_positive_rate_at_one_percent_fpr",
        "w_clip_score_mean": "watermarked_clip_score_mean",
        "avg_clip_score_mean": "watermarked_clip_score_mean",
    }
    normalized = dict(defaults)
    for key, value in payload.items():
        target_key = aliases.get(str(key), str(key))
        if target_key in normalized:
            normalized[target_key] = value
    return normalized


def load_imported_metric_summary(root_path: Path, config: ShallowDiffuseOfficialReferenceConfig) -> tuple[dict[str, Any], list[str]]:
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
            return normalize_metric_summary(parse_metric_text(log_path.read_text(encoding="utf-8", errors="ignore"), config.sample_count), config.sample_count), evidence_paths
    return {}, evidence_paths


def run_official_command_if_requested(
    root_path: Path,
    config: ShallowDiffuseOfficialReferenceConfig,
    paths: dict[str, Path],
    progress: object | None = None,
) -> dict[str, Any]:
    """根据配置执行 Shallow Diffuse 官方命令, 并保存 stdout / stderr。"""

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
        paths["official_stderr"].write_text("shallow_diffuse_official_source_entrypoint_missing", encoding="utf-8")
        result = {
            "official_command_requested": True,
            "official_command": build_official_command(root_path, config),
            "return_code": 96,
            "stdout_path": relative_or_absolute(paths["official_stdout"], root_path),
            "stderr_path": relative_or_absolute(paths["official_stderr"], root_path),
            "error": "shallow_diffuse_official_source_entrypoint_missing",
        }
        write_json(paths["official_command_result"], result)
        return result
    command = build_official_command(root_path, config)
    run_dir = paths["output_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "scratch").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env["SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_ATTACKER_NAMES"] = config.attacker_names
    try:
        completed = run_quiet_subprocess_with_progress(
            command,
            cwd=run_dir,
            env=env,
            timeout_seconds=int(config.timeout_seconds),
            progress=progress,
            progress_profile=f"operation=shallow_diffuse_official_command samples={config.sample_count}",
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
        "official_overall_scores_path": relative_or_absolute(paths["official_overall_scores"], root_path),
        "official_clip_scores_path": relative_or_absolute(paths["official_clip_scores"], root_path),
        "official_timestep_dir": relative_or_absolute(paths["official_timestep_dir"], root_path),
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
    config: ShallowDiffuseOfficialReferenceConfig,
    paths: dict[str, Path],
    metric_summary: dict[str, Any],
    evidence_paths: list[str],
    official_report: dict[str, Any],
    source_status: dict[str, Any],
) -> dict[str, Any]:
    """构造 governed import 记录并写出 schema、records 与 validation report。"""

    schema = build_shallow_diffuse_official_reference_schema()
    write_json(paths["reference_schema"], schema)
    if not metric_summary:
        validation = validate_shallow_diffuse_official_reference_records([])
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
        paths["official_overall_scores"],
        paths["official_clip_scores"],
        paths["official_config_log"],
    ):
        if candidate.is_file():
            local_evidence_paths.append(relative_or_absolute(candidate, root_path))
    result_source = relative_or_absolute(paths["official_metric_summary"], root_path)
    result_digest = file_digest(paths["official_metric_summary"])
    record = build_shallow_diffuse_official_reference_record(
        official_entrypoint=str(source_status.get("official_entrypoint", "")),
        official_repository_commit="c80c553fdf66fda8db735d77a9d56538b7a0ade8",
        official_environment_profile="legacy_shallow_diffuse_official_environment",
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
    validation = validate_shallow_diffuse_official_reference_records([record])
    write_json(paths["reference_validation"], validation)
    return {"record_count": 1, "validation": validation, "record_digest": record["reference_record_digest"]}


def write_shallow_diffuse_official_reference_outputs(
    config: ShallowDiffuseOfficialReferenceConfig,
    root: str | Path = ".",
) -> dict[str, Any]:
    """执行 Shallow Diffuse 官方参考 workflow 并写出 summary、manifest 和 governed import 记录。"""

    root_path = Path(root).resolve()
    formal_execution_run_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    paths = output_paths(root_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    with progress_bar(10, desc="shallow diffuse official reference", enabled=config.enable_workflow_progress_bar) as run_progress:
        emit_progress_status(run_progress, profile="operation=prepare_shallow_diffuse_legacy_environment status=running")
        legacy_environment_report = prepare_shallow_diffuse_legacy_environment(root_path, config, paths, progress=run_progress)
        update_progress(run_progress, profile="operation=prepare_shallow_diffuse_legacy_environment")
        effective_config = config
        if config.prepare_legacy_environment and legacy_environment_report.get("legacy_environment_ready"):
            effective_config = replace(config, official_python_executable=str(legacy_environment_report["legacy_python_executable"]))
        device_report: dict[str, Any]
        emit_progress_status(run_progress, profile="operation=ensure_cuda status=running")
        try:
            device_report = ensure_cuda_if_requested(effective_config.require_cuda)
        except Exception as error:
            device_report = {"cuda_available": False, "device_error": f"{type(error).__name__}:{error}"}
        update_progress(run_progress, profile="operation=ensure_cuda")
        source_status = ensure_shallow_diffuse_source_available(root_path, effective_config, paths, progress=run_progress)
        update_progress(run_progress, profile="operation=ensure_shallow_diffuse_source")
        emit_progress_status(run_progress, profile="operation=patch_shallow_diffuse_source status=running")
        source_patch_report = patch_shallow_diffuse_model_repository_layout(root_path, effective_config, paths)
        update_progress(run_progress, profile="operation=patch_shallow_diffuse_source")
        should_prepare_model_repository = effective_config.run_official_command and not (
            effective_config.prepare_legacy_environment and not legacy_environment_report.get("legacy_environment_ready")
        )
        model_repository_config = effective_config if should_prepare_model_repository else replace(effective_config, prepare_local_model_repository=False)
        model_repository_report = prepare_shallow_diffuse_model_repository(root_path, model_repository_config, paths, progress=run_progress)
        update_progress(run_progress, profile="operation=prepare_shallow_diffuse_model_repository")
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
                reason="shallow_diffuse_legacy_environment_prepare_failed",
            )
        else:
            official_report = run_official_command_if_requested(root_path, effective_config, paths, progress=run_progress)
        update_progress(run_progress, profile="operation=shallow_diffuse_official_command")
        emit_progress_status(run_progress, profile="operation=parse_shallow_diffuse_metrics status=running")
        command_metrics: dict[str, Any] = {}
        if official_report.get("return_code") == 0:
            metric_text_parts: list[str] = []
            for metric_path in (paths["official_overall_scores"], paths["official_clip_scores"], paths["official_stdout"]):
                if metric_path.is_file():
                    metric_text_parts.append(metric_path.read_text(encoding="utf-8", errors="ignore"))
            if metric_text_parts:
                command_metrics = parse_metric_text("\n".join(metric_text_parts), effective_config.sample_count)
        metric_summary = normalize_metric_summary({**imported_metrics, **command_metrics}, effective_config.sample_count) if (
            imported_metrics or command_metrics
        ) else {}
        update_progress(run_progress, profile="operation=parse_shallow_diffuse_metrics")
        emit_progress_status(run_progress, profile="operation=write_environment_report status=running")
        environment_report = build_runtime_environment_report(
            verified_formal_execution_lock=formal_execution_run_lock,
        )
        environment_report["shallow_diffuse_official_reference_device_report"] = device_report
        environment_report["shallow_diffuse_official_reference_source_report"] = source_status
        environment_report["shallow_diffuse_official_reference_source_patch_report"] = source_patch_report
        environment_report["shallow_diffuse_official_reference_model_repository_report"] = model_repository_report
        environment_report["shallow_diffuse_official_reference_legacy_environment_report"] = legacy_environment_report
        write_json(paths["environment_report"], environment_report)
        update_progress(run_progress, profile="operation=write_environment_report")
        emit_progress_status(run_progress, profile="operation=build_reference_record status=running")
        record_report = build_reference_record_report(
            root_path,
            effective_config,
            paths,
            metric_summary,
            imported_evidence,
            official_report,
            source_status,
        )
        update_progress(run_progress, profile=f"operation=build_reference_record records={record_report.get('record_count', 0)}")
    validation = record_report["validation"]
    run_ready = bool(validation.get("reference_import_ready"))
    unsupported_reason = "" if run_ready else "shallow_diffuse_official_reference_result_missing_or_invalid"
    if official_report.get("official_command_requested") and int(official_report.get("return_code", 1)) != 0:
        unsupported_reason = "official_command_failed_formal_archive_blocked"
    paper_run = build_paper_run_config(root_path)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_id": "shallow_diffuse",
        "run_decision": "pass" if run_ready else "fail",
        "shallow_diffuse_official_reference_ready": run_ready,
        "official_command_requested": bool(official_report.get("official_command_requested")),
        "official_command_return_code": int(official_report.get("return_code", -1)),
        "sample_count": int(effective_config.sample_count),
        "paper_claim_scale": paper_run.run_name,
        "target_fpr": paper_run.target_fpr,
        "edit_time_list": effective_config.edit_time_list,
        "primary_edit_timestep": primary_edit_timestep(effective_config),
        "attacker_names": effective_config.attacker_names,
        "reference_model": effective_config.reference_model,
        "reference_model_pretrain": effective_config.reference_model_pretrain,
        "legacy_environment_requested": bool(legacy_environment_report.get("legacy_environment_requested")),
        "legacy_environment_ready": bool(legacy_environment_report.get("legacy_environment_ready")),
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
        "official_overall_scores_path": relative_or_absolute(paths["official_overall_scores"], root_path),
        "official_clip_scores_path": relative_or_absolute(paths["official_clip_scores"], root_path),
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
        paths["official_overall_scores"],
        paths["official_clip_scores"],
        paths["official_config_log"],
    ):
        if optional_path.exists():
            output_paths_for_manifest.append(relative_or_absolute(optional_path, root_path))
    if paths["source_prepare_result"].exists():
        output_paths_for_manifest.append(relative_or_absolute(paths["source_prepare_result"], root_path))
    if paths["source_patch_result"].exists():
        output_paths_for_manifest.append(relative_or_absolute(paths["source_patch_result"], root_path))
    if paths["model_repository_prepare_result"].exists():
        output_paths_for_manifest.append(relative_or_absolute(paths["model_repository_prepare_result"], root_path))
    if paths["legacy_environment_prepare_result"].exists():
        output_paths_for_manifest.append(relative_or_absolute(paths["legacy_environment_prepare_result"], root_path))
    formal_execution_run_lock = repository_environment.validate_formal_execution_lock_pair(
        formal_execution_run_lock,
        repository_environment.require_published_formal_execution_lock(root_path),
        formal_execution_run_lock["formal_execution_commit"],
    )
    manifest = build_artifact_manifest(
        artifact_id="shallow_diffuse_official_reference_manifest",
        artifact_type="local_manifest",
        input_paths=(relative_or_absolute(root_path / config.source_dir, root_path),),
        output_paths=tuple(output_paths_for_manifest + [relative_or_absolute(paths["manifest"], root_path)]),
        config=asdict(effective_config),
        code_version=formal_execution_run_lock["formal_execution_commit"],
        rebuild_command="调用 paper_experiments.runners.shallow_diffuse_official_reference",
        metadata={
            "run_decision": summary["run_decision"],
            "shallow_diffuse_official_reference_ready": run_ready,
            "main_table_eligible": False,
            "supports_paper_claim": False,
        },
    ).to_dict()
    manifest["formal_execution_run_lock"] = formal_execution_run_lock
    write_json(paths["manifest"], manifest)
    return summary


def build_default_config() -> ShallowDiffuseOfficialReferenceConfig:
    """从环境变量构造默认 Colab 运行配置。"""

    paper_run = build_paper_run_config(".")
    return ShallowDiffuseOfficialReferenceConfig(
        output_dir=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        drive_output_dir=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_DRIVE_OUTPUT_DIR", paper_run.drive_dir("external_baseline_official_reference")),
        source_dir=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_SOURCE_DIR", DEFAULT_SOURCE_DIR),
        run_name=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_RUN_NAME", DEFAULT_RUN_NAME),
        sample_count=resolve_count_from_environment("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_SAMPLE_COUNT", default_value=paper_run.sample_count),
        start_index=int(os.environ.get("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_START_INDEX", "0")),
        official_model_id=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_MODEL_ID", DEFAULT_OFFICIAL_MODEL_ID),
        upstream_official_model_id=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_UPSTREAM_OFFICIAL_MODEL_ID", DEFAULT_UPSTREAM_OFFICIAL_MODEL_ID),
        model_source_note=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_MODEL_SOURCE_NOTE", DEFAULT_MODEL_SOURCE_NOTE),
        dataset=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_DATASET", "Gustavosta/Stable-Diffusion-Prompts"),
        image_length=int(os.environ.get("SLM_WM_SHALLOW_DIFFUSE_IMAGE_LENGTH", "512")),
        guidance_scale=float(os.environ.get("SLM_WM_SHALLOW_DIFFUSE_GUIDANCE_SCALE", "7.5")),
        num_inference_steps=int(os.environ.get("SLM_WM_SHALLOW_DIFFUSE_NUM_INFERENCE_STEPS", "50")),
        edit_time_list=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_EDIT_TIME_LIST", DEFAULT_EDIT_TIME_LIST),
        w_seed=int(os.environ.get("SLM_WM_SHALLOW_DIFFUSE_W_SEED", "42")),
        w_channel=int(os.environ.get("SLM_WM_SHALLOW_DIFFUSE_W_CHANNEL", "3")),
        w_pattern=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_W_PATTERN", "complex2_ring"),
        w_mask_shape=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_W_MASK_SHAPE", "circle"),
        w_radius=int(os.environ.get("SLM_WM_SHALLOW_DIFFUSE_W_RADIUS", "10")),
        w_measurement=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_W_MEASUREMENT", "l1_complex2"),
        w_injection=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_W_INJECTION", "complex2"),
        reference_model=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_REFERENCE_MODEL", DEFAULT_REFERENCE_MODEL),
        reference_model_pretrain=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_REFERENCE_MODEL_PRETRAIN", DEFAULT_REFERENCE_MODEL_PRETRAIN),
        attacker_names=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_ATTACKER_NAMES", DEFAULT_ATTACKER_NAMES),
        patch_model_repository_layout=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_PATCH_MODEL_REPOSITORY_LAYOUT", "1") != "0",
        prepare_local_model_repository=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_PREPARE_LOCAL_MODEL_REPOSITORY", "1") != "0",
        local_model_repository_dir=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_LOCAL_MODEL_REPOSITORY_DIR", DEFAULT_LOCAL_MODEL_REPOSITORY_DIR),
        patch_model_index_for_legacy_transformers=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_PATCH_MODEL_INDEX_FOR_LEGACY_TRANSFORMERS", "1") != "0",
        official_python_executable=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_PYTHON_EXECUTABLE", ""),
        prepare_legacy_environment=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_PREPARE_LEGACY_ENV", "0") == "1",
        legacy_environment_prefix=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_LEGACY_ENV_PREFIX", DEFAULT_LEGACY_ENV_PREFIX),
        micromamba_path=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_MICROMAMBA_PATH", DEFAULT_MICROMAMBA_PATH),
        legacy_python_version=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_LEGACY_PYTHON_VERSION", DEFAULT_LEGACY_PYTHON_VERSION),
        legacy_torch_specs=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_LEGACY_TORCH_SPECS", DEFAULT_LEGACY_TORCH_SPECS),
        legacy_pytorch_index_url=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_LEGACY_PYTORCH_INDEX_URL", DEFAULT_LEGACY_PYTORCH_INDEX_URL),
        legacy_package_specs=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_LEGACY_PACKAGE_SPECS", DEFAULT_LEGACY_PACKAGE_SPECS),
        run_official_command=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_RUN_COMMAND", "1") != "0",
        summary_import_path=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_SUMMARY_IMPORT_PATH", ""),
        log_import_path=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_LOG_IMPORT_PATH", ""),
        require_cuda=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_REQUIRE_CUDA", "1") != "0",
        timeout_seconds=int(os.environ.get("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_TIMEOUT_SECONDS", "86400")),
        enable_workflow_progress_bar=os.environ.get("SLM_WM_ENABLE_WORKFLOW_PROGRESS_BAR", "1") != "0",
    )


def run_default_shallow_diffuse_official_reference_plan(root: str | Path = ".") -> dict[str, Any]:
    """运行默认 Shallow Diffuse 官方参考 workflow。"""

    return write_shallow_diffuse_official_reference_outputs(config=build_default_config(), root=root)


def collect_package_entries(root_path: Path, output_dir: Path, archive_path: Path) -> tuple[Path, ...]:
    """仅收集当前论文层级的 Shallow Diffuse outputs family 文件。"""

    entries: list[Path] = []
    if output_dir.exists():
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.resolve() != archive_path.resolve() and path.suffix.lower() != ".zip":
                entries.append(path)
    unique_entries: list[Path] = []
    for entry in entries:
        if entry not in unique_entries:
            unique_entries.append(entry)
    return tuple(unique_entries)


def package_shallow_diffuse_official_reference_outputs(
    root: str | Path = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    drive_output_dir: str | None = None,
    archive_name: str | None = None,
) -> ShallowDiffuseOfficialReferenceArchiveRecord:
    """打包 Shallow Diffuse 官方参考产物并镜像到 Google Drive。"""

    root_path = Path(root).resolve()
    formal_execution_package_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    paper_run = build_paper_run_config(root_path)
    resolved_drive_output_dir = drive_output_dir or paper_run.drive_dir("external_baseline_official_reference")
    configured_output_root = (root_path / output_dir).resolve()
    expected_output_root = (root_path / DEFAULT_OUTPUT_DIR).resolve()
    if configured_output_root != expected_output_root:
        raise ValueError("Shallow Diffuse 官方参考打包根目录必须使用正式 outputs family")
    source_dir = expected_output_root / paper_run.run_name
    required_runtime_paths = (
        source_dir / "shallow_diffuse_official_reference_summary.json",
        source_dir / "manifest.local.json",
        source_dir / "shallow_diffuse_official_reference_records.jsonl",
        source_dir / "shallow_diffuse_official_reference_validation_report.json",
    )
    missing_runtime_paths = [path for path in required_runtime_paths if not path.is_file()]
    if missing_runtime_paths:
        raise FileNotFoundError("Shallow Diffuse 正式参考输出不完整, 不得打包")
    run_summary = read_json(required_runtime_paths[0])
    run_manifest = read_json(required_runtime_paths[1])
    formal_execution_run_lock = repository_environment.validate_formal_execution_lock_pair(
        run_manifest.get("formal_execution_run_lock"),
        formal_execution_package_lock,
        run_manifest.get("code_version"),
    )
    if not all(
        (
            run_summary.get("run_decision") == "pass",
            run_summary.get("shallow_diffuse_official_reference_ready") is True,
            run_summary.get("reference_import_ready") is True,
            run_summary.get("baseline_id") == "shallow_diffuse",
            run_summary.get("paper_claim_scale") == paper_run.run_name,
            float(run_summary.get("target_fpr", -1.0)) == paper_run.target_fpr,
        )
    ):
        raise RuntimeError("Shallow Diffuse 正式参考身份或 ready 门禁未通过")
    resolved_archive_name = archive_name or (
        "external_baseline_official_reference_package_shallow_diffuse_"
        f"{utc_archive_token()}_{formal_execution_package_lock['formal_execution_commit'][:7]}.zip"
    )
    if (
        Path(resolved_archive_name).name != resolved_archive_name
        or not resolved_archive_name.startswith(
            "external_baseline_official_reference_package_shallow_diffuse_"
        )
        or Path(resolved_archive_name).suffix.lower() != ".zip"
    ):
        raise ValueError("Shallow Diffuse archive_name 未匹配正式命名")
    archive_path = source_dir / resolved_archive_name
    package_manifest_path = source_dir / "shallow_diffuse_official_reference_package_input_manifest.json"
    summary_path = source_dir / "shallow_diffuse_official_reference_archive_summary.json"
    manifest_path = source_dir / "shallow_diffuse_official_reference_archive_manifest.local.json"
    for stale_path in (package_manifest_path, summary_path, manifest_path):
        if stale_path.exists():
            stale_path.unlink()
    content_entries = collect_package_entries(root_path, source_dir, archive_path)
    entries = tuple((*content_entries, package_manifest_path, summary_path, manifest_path))
    package_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_run_name": paper_run.run_name,
        "target_fpr": paper_run.target_fpr,
        "baseline_id": "shallow_diffuse",
        "formal_execution_run_lock": formal_execution_run_lock,
        "formal_execution_package_lock": formal_execution_package_lock,
        "entry_paths": [entry.relative_to(root_path).as_posix() for entry in content_entries],
        "entry_sha256": {
            entry.relative_to(root_path).as_posix(): file_digest(entry)
            for entry in content_entries
        },
        "entry_count": len(entries),
        "embedded_digest_scope": "external_summary_records_final_archive_digest",
    }
    write_json(package_manifest_path, package_manifest)
    preliminary_record = ShallowDiffuseOfficialReferenceArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest="",
        archive_entry_count=len(entries),
        drive_archive_path=str(Path(resolved_drive_output_dir).expanduser() / resolved_archive_name),
        drive_archive_digest="",
        metadata={
            "drive_output_dir": str(Path(resolved_drive_output_dir).expanduser()),
            "embedded_digest_scope": "external_summary_records_final_archive_digest",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(summary_path, preliminary_record.to_dict())
    archive_manifest = build_artifact_manifest(
        artifact_id="shallow_diffuse_official_reference_archive_manifest",
        artifact_type="local_manifest",
        input_paths=tuple([entry.relative_to(root_path).as_posix() for entry in content_entries] + [package_manifest_path.relative_to(root_path).as_posix()]),
        output_paths=(
            archive_path.relative_to(root_path).as_posix(),
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ),
        config={
            "archive_name": resolved_archive_name,
            "paper_run_name": paper_run.run_name,
            "target_fpr": paper_run.target_fpr,
            "drive_output_dir": str(Path(resolved_drive_output_dir).expanduser()),
        },
        code_version=formal_execution_package_lock["formal_execution_commit"],
        rebuild_command="调用 paper_experiments.runners.shallow_diffuse_official_reference",
        metadata={
            "embedded_digest_scope": "external_summary_records_final_archive_digest",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "paper_run_name": paper_run.run_name,
            "target_fpr": paper_run.target_fpr,
            "baseline_id": "shallow_diffuse",
            "main_table_eligible": False,
        },
    ).to_dict()
    archive_manifest["formal_execution_run_lock"] = formal_execution_run_lock
    archive_manifest["formal_execution_package_lock"] = formal_execution_package_lock
    write_json(manifest_path, archive_manifest)
    with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.write(entry, entry.relative_to(root_path).as_posix())
    try:
        final_package_lock = (
            repository_environment.require_published_formal_execution_lock(root_path)
        )
        repository_environment.validate_formal_execution_lock_pair(
            formal_execution_package_lock,
            final_package_lock,
            formal_execution_package_lock["formal_execution_commit"],
        )
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    drive_dir = Path(resolved_drive_output_dir).expanduser()
    drive_dir.mkdir(parents=True, exist_ok=True)
    mirrored_path = drive_dir / resolved_archive_name
    shutil.copy2(archive_path, mirrored_path)
    record = ShallowDiffuseOfficialReferenceArchiveRecord(
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




