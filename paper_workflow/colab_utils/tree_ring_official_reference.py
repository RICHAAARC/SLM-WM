"""Tree-Ring 官方原始环境复现与 governed import 的 Colab 辅助函数。

该 helper 服务补充表方法忠实度审计。它不把 legacy Stable Diffusion 结果混入 SD3.5 主表,
而是把官方命令、运行日志、环境报告、指标摘要和 governed import 记录统一写入 outputs/。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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

from experiments.baselines import (
    build_tree_ring_official_reference_record,
    build_tree_ring_official_reference_schema,
    validate_tree_ring_official_reference_records,
)
from main.analysis.artifact_manifest import build_artifact_manifest
from paper_workflow.colab_utils.external_baseline_gpu_smoke import (
    ensure_cuda_if_requested,
    load_baseline_registry_item,
    normalize_repository_url,
    run_command,
)
from paper_workflow.colab_utils.sd_runtime_cold_start import (
    build_runtime_environment_report,
    file_digest,
    resolve_code_version,
)

DEFAULT_OUTPUT_DIR = "outputs/tree_ring_official_reference"
DEFAULT_DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/SLM/tree_ring_official_reference"
DEFAULT_SOURCE_DIR = "external_baseline/primary/tree_ring/source"
DEFAULT_RUN_NAME = "tree_ring_official_legacy_reference"
DEFAULT_SAMPLE_COUNT = 5
DEFAULT_UPSTREAM_OFFICIAL_MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
DEFAULT_OFFICIAL_MODEL_ID = "Manojb/stable-diffusion-2-1-base"
DEFAULT_MODEL_SOURCE_NOTE = (
    "官方 Tree-Ring 使用的 stabilityai/stable-diffusion-2-1-base 当前不可直接访问; "
    "默认改用 Hugging Face 上标记为 cloned from stabilityai/stable-diffusion-2-1-base 的公开镜像。"
)
DEFAULT_LOCAL_MODEL_REPOSITORY_DIR = "/content/tree_ring_model_repository/stable_diffusion_2_1_base"
DEFAULT_LEGACY_ENV_PREFIX = "/content/tree_ring_legacy_env"
DEFAULT_MICROMAMBA_PATH = "/content/bin/micromamba"
DEFAULT_LEGACY_PYTHON_VERSION = "3.9"
DEFAULT_LEGACY_TORCH_SPECS = "torch==1.13.0+cu117 torchvision==0.14.0+cu117"
DEFAULT_LEGACY_PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu117"
DEFAULT_LEGACY_PACKAGE_SPECS = (
    "transformers==4.23.1 diffusers==0.11.1 huggingface_hub==0.10.1 "
    "datasets==2.6.1 pyarrow<13 fsspec==2022.10.0 numpy<2 scikit-learn scipy tqdm wandb open_clip_torch==2.7.0 ftfy regex"
)
PACKAGE_EXTRA_PATHS = (
    "paper_workflow/tree_ring_official_reference_run.ipynb",
    "paper_workflow/colab_utils/tree_ring_official_reference.py",
    "experiments/baselines/tree_ring_official_reference.py",
    "external_baseline/primary/tree_ring/README.md",
    "external_baseline/primary/tree_ring/source/README.md",
    "external_baseline/primary/tree_ring/source/requirements.txt",
    "external_baseline/source_registry.json",
)


@dataclass(frozen=True)
class TreeRingOfficialReferenceConfig:
    """描述 Tree-Ring 官方原始环境复现与导入所需配置。"""

    output_dir: str = DEFAULT_OUTPUT_DIR
    drive_output_dir: str = DEFAULT_DRIVE_OUTPUT_DIR
    source_dir: str = DEFAULT_SOURCE_DIR
    run_name: str = DEFAULT_RUN_NAME
    sample_count: int = DEFAULT_SAMPLE_COUNT
    start_index: int = 0
    official_model_id: str = DEFAULT_OFFICIAL_MODEL_ID
    upstream_official_model_id: str = DEFAULT_UPSTREAM_OFFICIAL_MODEL_ID
    model_source_note: str = DEFAULT_MODEL_SOURCE_NOTE
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


@dataclass(frozen=True)
class TreeRingOfficialReferenceArchiveRecord:
    """记录 Tree-Ring 官方参考压缩包与 Drive 镜像信息。"""

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
    """把环境准备异常转换为可落盘命令诊断, 避免 Notebook 直接中断。"""

    return_code = 124 if isinstance(error, subprocess.TimeoutExpired) else 98
    return {
        "command": command,
        "return_code": return_code,
        "stdout": str(getattr(error, "stdout", "") or ""),
        "stderr": f"{type(error).__name__}:{error}",
    }


def run_shell_command(command: str, *, cwd: Path, timeout_seconds: int) -> dict[str, Any]:
    """执行 shell 命令并返回可落盘诊断。"""

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            timeout=timeout_seconds,
            check=False,
            text=True,
            capture_output=True,
            shell=True,
        )
    except Exception as error:
        return command_exception_result(command, error)
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_argv_command(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict[str, Any]:
    """执行 argv 命令并把失败收敛为可审计诊断。"""

    try:
        return run_command(command, cwd=cwd, timeout_seconds=timeout_seconds)
    except Exception as error:
        return command_exception_result(command, error)


def output_paths(root_path: Path, config: TreeRingOfficialReferenceConfig) -> dict[str, Path]:
    """集中构造 Tree-Ring 官方参考 workflow 的输出路径。"""

    output_dir = (root_path / config.output_dir).resolve()
    return {
        "output_dir": output_dir,
        "official_command_result": output_dir / "tree_ring_official_command_result.json",
        "source_prepare_result": output_dir / "tree_ring_official_source_prepare_result.json",
        "source_patch_result": output_dir / "tree_ring_official_source_patch_result.json",
        "model_repository_prepare_result": output_dir / "tree_ring_model_repository_prepare_result.json",
        "legacy_environment_prepare_result": output_dir / "tree_ring_legacy_environment_prepare_result.json",
        "official_stdout": output_dir / "tree_ring_official_stdout.txt",
        "official_stderr": output_dir / "tree_ring_official_stderr.txt",
        "official_metric_summary": output_dir / "tree_ring_official_metric_summary.json",
        "reference_schema": output_dir / "tree_ring_official_reference_schema.json",
        "reference_records": output_dir / "tree_ring_official_reference_records.jsonl",
        "reference_validation": output_dir / "tree_ring_official_reference_validation_report.json",
        "environment_report": output_dir / "tree_ring_official_reference_environment_report.json",
        "summary": output_dir / "tree_ring_official_reference_summary.json",
        "manifest": output_dir / "manifest.local.json",
    }


def prepare_tree_ring_legacy_environment(
    root_path: Path,
    config: TreeRingOfficialReferenceConfig,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """在独立 Colab 会话中准备 Tree-Ring 官方 legacy Python 环境。"""

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
            run_shell_command(
                f"curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C {micromamba_path.parent.parent} bin/micromamba",
                cwd=root_path,
                timeout_seconds=600,
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
                )
            )
    if legacy_python.is_file():
        command_results.append(
            run_argv_command(
                [str(legacy_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                cwd=root_path,
                timeout_seconds=900,
            )
        )
        torch_specs = split_package_specs(config.legacy_torch_specs)
        if torch_specs:
            torch_command = [str(legacy_python), "-m", "pip", "install"]
            if config.legacy_pytorch_index_url:
                torch_command.extend(["--extra-index-url", str(config.legacy_pytorch_index_url)])
            torch_command.extend(torch_specs)
            command_results.append(run_argv_command(torch_command, cwd=root_path, timeout_seconds=1800))
        package_specs = split_package_specs(config.legacy_package_specs)
        if package_specs:
            command_results.append(
                run_argv_command(
                    [str(legacy_python), "-m", "pip", "install", *package_specs],
                    cwd=root_path,
                    timeout_seconds=1800,
                )
            )
    verify_command = [
        str(legacy_python),
        "-c",
        (
            "import json, torch, transformers, diffusers; "
            "from transformers import CLIPFeatureExtractor; "
            "print(json.dumps({"
            "'torch': torch.__version__, "
            "'cuda_available': bool(torch.cuda.is_available()), "
            "'transformers': transformers.__version__, "
            "'diffusers': diffusers.__version__"
            "}))"
        ),
    ]
    verify_result = (
        run_argv_command(verify_command, cwd=root_path, timeout_seconds=300)
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


def source_report(root_path: Path, config: TreeRingOfficialReferenceConfig) -> dict[str, Any]:
    """检查官方 Tree-Ring 源码快照和 requirements 是否存在。"""

    source_dir = (root_path / config.source_dir).resolve()
    entrypoint = source_dir / "run_tree_ring_watermark.py"
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


def ensure_tree_ring_source_available(
    root_path: Path,
    config: TreeRingOfficialReferenceConfig,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """在 Colab 冷启动环境中按登记表补齐 Tree-Ring 官方源码。"""

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

    registry_item = load_baseline_registry_item(root_path, "tree_ring")
    source_dir = (root_path / config.source_dir).resolve()
    source_entry = source_dir / "run_tree_ring_watermark.py"
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
    clone_result = run_command(["git", "clone", repository_url, str(source_dir)], cwd=root_path, timeout_seconds=300)
    checkout_result: dict[str, Any] = {"command": [], "return_code": 0, "stdout": "", "stderr": ""}
    if clone_result["return_code"] == 0 and registry_item.get("official_repository_commit"):
        checkout_result = run_command(
            ["git", "checkout", str(registry_item["official_repository_commit"])],
            cwd=source_dir,
            timeout_seconds=300,
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



def patch_tree_ring_model_repository_layout(
    root_path: Path,
    config: TreeRingOfficialReferenceConfig,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """为公开镜像缺少 fp16 分支的情况应用最小源码入口补丁。"""

    source_dir = (root_path / config.source_dir).resolve()
    entrypoint = source_dir / "run_tree_ring_watermark.py"
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
    target = "        revision='fp16',\n"
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
    patched_text = source_text.replace(target, f"        {marker}\n")
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


def prepare_tree_ring_model_repository(
    root_path: Path,
    config: TreeRingOfficialReferenceConfig,
    paths: dict[str, Path],
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


def build_official_command(root_path: Path, config: TreeRingOfficialReferenceConfig) -> list[str]:
    """构造 Tree-Ring 官方 legacy 入口命令。"""

    source_dir = (root_path / config.source_dir).resolve()
    entrypoint = source_dir / "run_tree_ring_watermark.py"
    end_index = int(config.start_index) + max(1, int(config.sample_count))
    python_executable = config.official_python_executable or sys.executable
    return [
        python_executable,
        str(entrypoint),
        "--run_name",
        str(config.run_name),
        "--model_id",
        str(config.official_model_id),
        "--w_channel",
        "3",
        "--w_pattern",
        "ring",
        "--start",
        str(config.start_index),
        "--end",
        str(end_index),
        "--with_tracking",
    ]


def parse_metric_text(text: str, sample_count: int) -> dict[str, Any]:
    """从官方 stdout 或日志中解析 Tree-Ring 指标。"""

    patterns = {
        "clip_score_mean": r"clip_score_mean:\s*([0-9eE+\-.]+)",
        "watermarked_clip_score_mean": r"w_clip_score_mean:\s*([0-9eE+\-.]+)",
        "auc": r"auc:\s*([0-9eE+\-.]+)",
        "accuracy": r"acc:\s*([0-9eE+\-.]+)",
        "true_positive_rate_at_one_percent_fpr": r"TPR@1%FPR:\s*([0-9eE+\-.]+)",
    }
    metrics: dict[str, Any] = {
        "sample_count": int(sample_count),
        "positive_count": int(sample_count),
        "negative_count": int(sample_count),
    }
    for field_name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[field_name] = float(match.group(1))
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
    }
    normalized = dict(defaults)
    for key, value in payload.items():
        target_key = aliases.get(str(key), str(key))
        if target_key in normalized:
            normalized[target_key] = value
    return normalized


def load_imported_metric_summary(root_path: Path, config: TreeRingOfficialReferenceConfig) -> tuple[dict[str, Any], list[str]]:
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
    config: TreeRingOfficialReferenceConfig,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """根据配置执行 Tree-Ring 官方命令, 并保存 stdout / stderr。"""

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
        paths["official_stderr"].write_text("tree_ring_official_source_entrypoint_missing", encoding="utf-8")
        result = {
            "official_command_requested": True,
            "official_command": build_official_command(root_path, config),
            "return_code": 96,
            "stdout_path": relative_or_absolute(paths["official_stdout"], root_path),
            "stderr_path": relative_or_absolute(paths["official_stderr"], root_path),
            "error": "tree_ring_official_source_entrypoint_missing",
        }
        write_json(paths["official_command_result"], result)
        return result
    command = build_official_command(root_path, config)
    source_dir = (root_path / config.source_dir).resolve()
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    try:
        completed = subprocess.run(
            command,
            cwd=source_dir,
            timeout=int(config.timeout_seconds),
            check=False,
            text=True,
            capture_output=True,
            env=env,
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
    config: TreeRingOfficialReferenceConfig,
    paths: dict[str, Path],
    metric_summary: dict[str, Any],
    evidence_paths: list[str],
    official_report: dict[str, Any],
    source_status: dict[str, Any],
) -> dict[str, Any]:
    """构造 governed import 记录并写出 schema、records 与 validation report。"""

    schema = build_tree_ring_official_reference_schema()
    write_json(paths["reference_schema"], schema)
    if not metric_summary:
        validation = validate_tree_ring_official_reference_records([])
        paths["reference_records"].write_text("", encoding="utf-8")
        write_json(paths["reference_validation"], validation)
        return {"record_count": 0, "validation": validation}

    write_json(paths["official_metric_summary"], metric_summary)
    local_evidence_paths = list(evidence_paths)
    for candidate in (paths["official_metric_summary"], paths["official_command_result"], paths["official_stdout"], paths["official_stderr"]):
        if candidate.is_file():
            local_evidence_paths.append(relative_or_absolute(candidate, root_path))
    result_source = relative_or_absolute(paths["official_metric_summary"], root_path)
    result_digest = file_digest(paths["official_metric_summary"])
    record = build_tree_ring_official_reference_record(
        official_entrypoint=str(source_status.get("official_entrypoint", "")),
        official_repository_commit="3015283d9cf82e90b628f02ad2121bd37408ca9a",
        official_environment_profile="legacy_tree_ring_official_environment",
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
    validation = validate_tree_ring_official_reference_records([record])
    write_json(paths["reference_validation"], validation)
    return {"record_count": 1, "validation": validation, "record_digest": record["reference_record_digest"]}


def write_tree_ring_official_reference_outputs(
    config: TreeRingOfficialReferenceConfig,
    root: str | Path = ".",
) -> dict[str, Any]:
    """执行 Tree-Ring 官方参考 workflow 并写出 summary、manifest 和 governed import 记录。"""

    root_path = Path(root).resolve()
    paths = output_paths(root_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    legacy_environment_report = prepare_tree_ring_legacy_environment(root_path, config, paths)
    effective_config = config
    if config.prepare_legacy_environment and legacy_environment_report.get("legacy_environment_ready"):
        effective_config = replace(config, official_python_executable=str(legacy_environment_report["legacy_python_executable"]))
    device_report: dict[str, Any]
    try:
        device_report = ensure_cuda_if_requested(effective_config.require_cuda)
    except Exception as error:
        device_report = {"cuda_available": False, "device_error": f"{type(error).__name__}:{error}"}
    source_status = ensure_tree_ring_source_available(root_path, effective_config, paths)
    source_patch_report = patch_tree_ring_model_repository_layout(root_path, effective_config, paths)
    should_prepare_model_repository = effective_config.run_official_command and not (
        effective_config.prepare_legacy_environment and not legacy_environment_report.get("legacy_environment_ready")
    )
    model_repository_config = effective_config if should_prepare_model_repository else replace(effective_config, prepare_local_model_repository=False)
    model_repository_report = prepare_tree_ring_model_repository(root_path, model_repository_config, paths)
    if model_repository_report.get("local_model_repository_ready"):
        effective_config = replace(effective_config, official_model_id=str(model_repository_report["effective_official_model_id"]))
    imported_metrics, imported_evidence = load_imported_metric_summary(root_path, effective_config)
    if (
        effective_config.run_official_command
        and effective_config.prepare_legacy_environment
        and not legacy_environment_report.get("legacy_environment_ready")
    ):
        official_report = write_official_command_skip_result(
            root_path,
            paths,
            return_code=95,
            reason="tree_ring_legacy_environment_prepare_failed",
        )
    else:
        official_report = run_official_command_if_requested(root_path, effective_config, paths)
    if official_report.get("return_code") == 0 and paths["official_stdout"].is_file():
        command_metrics = parse_metric_text(paths["official_stdout"].read_text(encoding="utf-8", errors="ignore"), effective_config.sample_count)
        metric_summary = normalize_metric_summary({**imported_metrics, **command_metrics}, effective_config.sample_count)
    else:
        metric_summary = imported_metrics
    environment_report = build_runtime_environment_report()
    environment_report["tree_ring_official_reference_device_report"] = device_report
    environment_report["tree_ring_official_reference_source_report"] = source_status
    environment_report["tree_ring_official_reference_source_patch_report"] = source_patch_report
    environment_report["tree_ring_official_reference_model_repository_report"] = model_repository_report
    environment_report["tree_ring_official_reference_legacy_environment_report"] = legacy_environment_report
    write_json(paths["environment_report"], environment_report)
    record_report = build_reference_record_report(
        root_path,
        effective_config,
        paths,
        metric_summary,
        imported_evidence,
        official_report,
        source_status,
    )
    validation = record_report["validation"]
    run_ready = bool(validation.get("reference_import_ready"))
    unsupported_reason = "" if run_ready else "tree_ring_official_reference_result_missing_or_invalid"
    if official_report.get("official_command_requested") and int(official_report.get("return_code", 1)) != 0:
        unsupported_reason = "official_command_failed_governed_diagnostics_packaged"
    summary = {
        "run_decision": "pass" if run_ready else "fail",
        "tree_ring_official_reference_ready": run_ready,
        "official_command_requested": bool(official_report.get("official_command_requested")),
        "official_command_return_code": int(official_report.get("return_code", -1)),
        "sample_count": int(effective_config.sample_count),
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
    for optional_path in (paths["official_command_result"], paths["official_stdout"], paths["official_stderr"], paths["official_metric_summary"]):
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
    manifest = build_artifact_manifest(
        artifact_id="tree_ring_official_reference_manifest",
        artifact_type="local_manifest",
        input_paths=(relative_or_absolute(root_path / config.source_dir, root_path),),
        output_paths=tuple(output_paths_for_manifest + [relative_or_absolute(paths["manifest"], root_path)]),
        config=asdict(effective_config),
        code_version=resolve_code_version(root_path),
        rebuild_command="运行 paper_workflow/tree_ring_official_reference_run.ipynb",
        metadata={
            "run_decision": summary["run_decision"],
            "tree_ring_official_reference_ready": run_ready,
            "main_table_eligible": False,
            "supports_paper_claim": False,
        },
    ).to_dict()
    write_json(paths["manifest"], manifest)
    return summary


def build_default_config() -> TreeRingOfficialReferenceConfig:
    """从环境变量构造默认 Colab 运行配置。"""

    return TreeRingOfficialReferenceConfig(
        output_dir=os.environ.get("SLM_WM_TREE_RING_OFFICIAL_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        drive_output_dir=os.environ.get("SLM_WM_TREE_RING_OFFICIAL_DRIVE_OUTPUT_DIR", DEFAULT_DRIVE_OUTPUT_DIR),
        source_dir=os.environ.get("SLM_WM_TREE_RING_OFFICIAL_SOURCE_DIR", DEFAULT_SOURCE_DIR),
        run_name=os.environ.get("SLM_WM_TREE_RING_OFFICIAL_RUN_NAME", DEFAULT_RUN_NAME),
        sample_count=int(os.environ.get("SLM_WM_TREE_RING_OFFICIAL_SAMPLE_COUNT", str(DEFAULT_SAMPLE_COUNT))),
        start_index=int(os.environ.get("SLM_WM_TREE_RING_OFFICIAL_START_INDEX", "0")),
        official_model_id=os.environ.get("SLM_WM_TREE_RING_OFFICIAL_MODEL_ID", DEFAULT_OFFICIAL_MODEL_ID),
        upstream_official_model_id=os.environ.get("SLM_WM_TREE_RING_UPSTREAM_OFFICIAL_MODEL_ID", DEFAULT_UPSTREAM_OFFICIAL_MODEL_ID),
        model_source_note=os.environ.get("SLM_WM_TREE_RING_MODEL_SOURCE_NOTE", DEFAULT_MODEL_SOURCE_NOTE),
        patch_model_repository_layout=os.environ.get("SLM_WM_TREE_RING_PATCH_MODEL_REPOSITORY_LAYOUT", "1") != "0",
        prepare_local_model_repository=os.environ.get("SLM_WM_TREE_RING_PREPARE_LOCAL_MODEL_REPOSITORY", "1") != "0",
        local_model_repository_dir=os.environ.get("SLM_WM_TREE_RING_LOCAL_MODEL_REPOSITORY_DIR", DEFAULT_LOCAL_MODEL_REPOSITORY_DIR),
        patch_model_index_for_legacy_transformers=os.environ.get("SLM_WM_TREE_RING_PATCH_MODEL_INDEX_FOR_LEGACY_TRANSFORMERS", "1") != "0",
        official_python_executable=os.environ.get("SLM_WM_TREE_RING_OFFICIAL_PYTHON_EXECUTABLE", ""),
        prepare_legacy_environment=os.environ.get("SLM_WM_TREE_RING_PREPARE_LEGACY_ENV", "0") == "1",
        legacy_environment_prefix=os.environ.get("SLM_WM_TREE_RING_LEGACY_ENV_PREFIX", DEFAULT_LEGACY_ENV_PREFIX),
        micromamba_path=os.environ.get("SLM_WM_TREE_RING_MICROMAMBA_PATH", DEFAULT_MICROMAMBA_PATH),
        legacy_python_version=os.environ.get("SLM_WM_TREE_RING_LEGACY_PYTHON_VERSION", DEFAULT_LEGACY_PYTHON_VERSION),
        legacy_torch_specs=os.environ.get("SLM_WM_TREE_RING_LEGACY_TORCH_SPECS", DEFAULT_LEGACY_TORCH_SPECS),
        legacy_pytorch_index_url=os.environ.get("SLM_WM_TREE_RING_LEGACY_PYTORCH_INDEX_URL", DEFAULT_LEGACY_PYTORCH_INDEX_URL),
        legacy_package_specs=os.environ.get("SLM_WM_TREE_RING_LEGACY_PACKAGE_SPECS", DEFAULT_LEGACY_PACKAGE_SPECS),
        run_official_command=os.environ.get("SLM_WM_TREE_RING_OFFICIAL_RUN_COMMAND", "1") != "0",
        summary_import_path=os.environ.get("SLM_WM_TREE_RING_OFFICIAL_SUMMARY_IMPORT_PATH", ""),
        log_import_path=os.environ.get("SLM_WM_TREE_RING_OFFICIAL_LOG_IMPORT_PATH", ""),
        require_cuda=os.environ.get("SLM_WM_TREE_RING_OFFICIAL_REQUIRE_CUDA", "1") != "0",
        timeout_seconds=int(os.environ.get("SLM_WM_TREE_RING_OFFICIAL_TIMEOUT_SECONDS", "86400")),
    )


def run_default_tree_ring_official_reference_plan(root: str | Path = ".") -> dict[str, Any]:
    """运行默认 Tree-Ring 官方参考 workflow。"""

    return write_tree_ring_official_reference_outputs(config=build_default_config(), root=root)


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


def package_tree_ring_official_reference_outputs(
    root: str | Path = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    drive_output_dir: str = DEFAULT_DRIVE_OUTPUT_DIR,
    archive_name: str = "tree_ring_official_reference_package.zip",
) -> TreeRingOfficialReferenceArchiveRecord:
    """打包 Tree-Ring 官方参考产物并镜像到 Google Drive。"""

    root_path = Path(root).resolve()
    source_dir = (root_path / output_dir).resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_path = source_dir / archive_name
    package_manifest_path = source_dir / "tree_ring_official_reference_package_input_manifest.json"
    summary_path = source_dir / "tree_ring_official_reference_archive_summary.json"
    manifest_path = source_dir / "tree_ring_official_reference_archive_manifest.local.json"
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
        artifact_id="tree_ring_official_reference_archive_manifest",
        artifact_type="local_manifest",
        input_paths=tuple([entry.relative_to(root_path).as_posix() for entry in entries] + [package_manifest_path.relative_to(root_path).as_posix()]),
        output_paths=(
            archive_path.relative_to(root_path).as_posix(),
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ),
        config={"archive_name": archive_name, "drive_output_dir": str(Path(drive_output_dir).expanduser())},
        code_version=resolve_code_version(root_path),
        rebuild_command="运行 paper_workflow/tree_ring_official_reference_run.ipynb",
        metadata={"generated_at": datetime.now(timezone.utc).isoformat(), "main_table_eligible": False},
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
    record = TreeRingOfficialReferenceArchiveRecord(
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
