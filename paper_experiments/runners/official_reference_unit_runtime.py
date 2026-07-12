"""为官方参考复现提供可恢复的原子科学批次协议.

三个官方方法都保留其登记源码与方法算子. 本模块只把长时间运行拆成预注册的
小批次, 并在每个批次真正完成后绑定代码锁、依赖锁、GPU、随机性和原始观测.
任何不完整或身份不一致的批次都会闭锁, 不会被静默删除或部分复用.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import stat
from statistics import mean, stdev
import sys
from typing import Any

from experiments.runtime.progress import run_quiet_subprocess_with_progress
from experiments.runtime.scientific_unit_provenance import (
    aggregate_scientific_unit_provenance,
    build_scientific_unit_provenance,
    validate_scientific_unit_provenance,
)
from main.core.digest import build_stable_digest
from paper_experiments.runners.external_source_runtime import (
    bind_successful_official_command_execution_evidence,
)


OFFICIAL_REFERENCE_UNIT_SCHEMA = "official_reference_scientific_unit"
OFFICIAL_REFERENCE_UNIT_SCHEMA_VERSION = 2
OFFICIAL_REFERENCE_SOURCE_PAYLOAD_SCHEMA = "official_reference_source_unit_payload"
OFFICIAL_REFERENCE_SOURCE_PAYLOAD_SCHEMA_VERSION = 1
OFFICIAL_REFERENCE_COMMAND_IDENTITY_SCHEMA = (
    "official_reference_workspace_independent_command_identity"
)
OFFICIAL_REFERENCE_COMMAND_IDENTITY_SCHEMA_VERSION = 1
DEFAULT_OFFICIAL_REFERENCE_UNIT_BATCH_SIZE = 10
UNIT_CONTEXT_ENV_NAME = "SLM_WM_OFFICIAL_UNIT_CONTEXT"
UNIT_OUTPUT_ENV_NAME = "SLM_WM_OFFICIAL_UNIT_OUTPUT"


def _stable_json_text(value: Any) -> str:
    """使用稳定键顺序生成可复算 JSON 文本."""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _atomic_write_json(path: Path, payload: Any) -> None:
    """通过同目录临时文件原子提交一个完整 JSON 对象."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.writing-{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(_stable_json_text(payload))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _file_sha256(path: Path) -> str:
    """分块计算实际可执行文件摘要."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """读取并要求顶层为 JSON 对象."""

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError("官方参考科学单元必须是 JSON 对象")
    return payload


def _sha256_text(value: str) -> str:
    """计算 UTF-8 文本的 SHA-256."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_official_reference_unit_ranges(
    start_index: int,
    sample_count: int,
    batch_size: int = DEFAULT_OFFICIAL_REFERENCE_UNIT_BATCH_SIZE,
) -> tuple[tuple[int, int], ...]:
    """构造无重叠且完整覆盖的预注册小批次.

    批次大小固定为至少2, 因为三个官方脚本的正式指标包含样本标准差. 默认
    10个 Prompt 能整除三层规模, 同时把 full 运行的模型加载次数限制在700次.
    """

    resolved_start = int(start_index)
    resolved_count = int(sample_count)
    resolved_batch = int(batch_size)
    if resolved_start < 0 or resolved_count <= 0:
        raise ValueError("官方参考科学单元要求非负起点和正样本数")
    if resolved_batch < 2:
        raise ValueError("官方参考原子批次至少包含2个样本")
    if resolved_count % resolved_batch != 0:
        raise ValueError("正式样本数必须被预注册原子批次大小整除")
    return tuple(
        (unit_start, unit_start + resolved_batch)
        for unit_start in range(
            resolved_start,
            resolved_start + resolved_count,
            resolved_batch,
        )
    )


def build_official_reference_config_digest(
    baseline_id: str,
    scientific_config: Mapping[str, Any],
) -> str:
    """计算整次官方参考运行的科学配置摘要."""

    return build_stable_digest(
        {
            "baseline_id": str(baseline_id),
            "scientific_config": dict(scientific_config),
        }
    )


def _path_is_link_or_junction(path: Path) -> bool:
    """不跟随目标地判断路径是否为符号链接或 Windows junction."""

    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def validate_official_reference_package_root_exact_set(
    *,
    output_dir: Path,
    allowed_relative_file_paths: Sequence[str],
    optional_relative_file_paths: Sequence[str] = (),
    unit_directory_name: str = "scientific_units",
) -> tuple[Path, ...]:
    """只用 lstat 检查完整相对路径白名单, 不跟随链接或特殊文件.

    白名单会推导全部合法中间目录, 因而 Gaussian Shading 和 Shallow
    Diffuse 的官方嵌套事实文件可以保留. 科学单元文件名 exact-set 仍由
    后续持久化单元验证器按预注册范围继续核验.
    """

    if _path_is_link_or_junction(output_dir):
        raise RuntimeError("官方参考结果根不得是链接或 junction")
    try:
        output_mode = output_dir.lstat().st_mode
    except FileNotFoundError as error:
        raise FileNotFoundError("官方参考结果根不存在") from error
    if not stat.S_ISDIR(output_mode):
        raise RuntimeError("官方参考结果根不是普通目录")

    allowed_files: set[Path] = set()
    for raw_path in allowed_relative_file_paths:
        relative_path = Path(str(raw_path))
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise ValueError("官方参考结果白名单包含非安全相对路径")
        allowed_files.add(relative_path)
    unit_relative = Path(unit_directory_name)
    if not allowed_files or unit_relative in allowed_files:
        raise ValueError("官方参考结果根白名单配置无效")
    optional_files = {Path(str(path)) for path in optional_relative_file_paths}
    if not optional_files <= allowed_files:
        raise ValueError("官方参考结果根 optional 白名单不是 allowed 子集")
    allowed_directories = {unit_relative}
    for file_path in allowed_files:
        parent = file_path.parent
        while parent != Path("."):
            allowed_directories.add(parent)
            parent = parent.parent

    accepted_files: list[Path] = []

    def inspect_directory(directory: Path, relative_directory: Path) -> None:
        """递归检查一个已由白名单推导出的普通目录."""

        members = sorted(directory.iterdir(), key=lambda item: item.name)
        if not members:
            raise RuntimeError("官方参考结果根不得包含空目录")
        for path in members:
            if _path_is_link_or_junction(path):
                raise RuntimeError("官方参考结果根不得包含链接或 junction")
            relative_path = relative_directory / path.name
            mode = path.lstat().st_mode
            if stat.S_ISREG(mode):
                if relative_directory == unit_relative:
                    accepted_files.append(path)
                    continue
                if relative_path not in allowed_files:
                    raise RuntimeError(
                        "官方参考结果根包含白名单外文件: "
                        + relative_path.as_posix()
                    )
                accepted_files.append(path)
                continue
            if stat.S_ISDIR(mode):
                if relative_directory == unit_relative:
                    raise RuntimeError("官方参考科学单元目录只允许普通文件")
                if relative_path not in allowed_directories:
                    raise RuntimeError(
                        "官方参考结果根包含白名单外目录: "
                        + relative_path.as_posix()
                    )
                inspect_directory(path, relative_path)
                continue
            raise RuntimeError("官方参考结果根不得包含特殊文件")

    inspect_directory(output_dir, Path("."))
    actual_non_unit_files = {
        path.relative_to(output_dir)
        for path in accepted_files
        if unit_relative not in path.relative_to(output_dir).parents
    }
    required_files = allowed_files - optional_files
    missing_files = sorted(
        (required_files - actual_non_unit_files),
        key=lambda item: item.as_posix(),
    )
    if missing_files:
        raise RuntimeError(
            "官方参考结果根缺少白名单必需文件: "
            + ",".join(path.as_posix() for path in missing_files)
        )
    return tuple(accepted_files)


def _unit_id(
    baseline_id: str,
    unit_start: int,
    unit_end: int,
    config_digest: str,
) -> str:
    """构造同时绑定方法、索引范围和配置的稳定单元标识."""

    return (
        f"{baseline_id}:prompt_range:{unit_start:07d}-{unit_end:07d}:"
        f"{config_digest[:16]}"
    )


def _unit_path(unit_dir: Path, unit_start: int, unit_end: int) -> Path:
    """返回一个预注册批次的唯一持久化路径."""

    return unit_dir / f"unit_{unit_start:07d}_{unit_end:07d}.json"


def _required_sha256(value: Any, field_name: str) -> str:
    """集中校验 SHA-256 身份字段."""

    resolved = str(value or "")
    if len(resolved) != 64 or any(character not in "0123456789abcdef" for character in resolved):
        raise ValueError(f"官方参考科学单元的 {field_name} 不是 SHA-256")
    return resolved


def build_official_reference_runtime_context(
    *,
    baseline_id: str,
    formal_execution_lock: Mapping[str, Any],
    dependency_environment_report: Mapping[str, Any],
) -> dict[str, Any]:
    """从已核验报告提取供科学子进程自证的静态运行身份."""

    dependency = dict(dependency_environment_report)
    isolated = dependency.get("isolated_dependency_environment_report")
    profile = dependency.get("dependency_profile")
    if dependency.get("dependency_environment_ready") is not True:
        raise RuntimeError("官方参考科学单元的隔离依赖环境未就绪")
    if not isinstance(isolated, Mapping) or not isinstance(profile, Mapping):
        raise TypeError("官方参考科学单元缺少隔离环境或依赖 profile 报告")
    formal_lock = dict(formal_execution_lock)
    context = {
        "baseline_id": str(baseline_id),
        "dependency_profile_id": str(dependency.get("dependency_profile_id", "")),
        "dependency_profile_digest": _required_sha256(
            dependency.get("dependency_profile_digest"),
            "dependency_profile_digest",
        ),
        "direct_requirements_digest": _required_sha256(
            profile.get("direct_requirements_digest"),
            "direct_requirements_digest",
        ),
        "python_version": str(profile.get("python_version", "")),
        "complete_hash_lock_digest": _required_sha256(
            dependency.get("dependency_lock_digest"),
            "complete_hash_lock_digest",
        ),
        "dependency_environment_report_digest": _required_sha256(
            dependency.get("isolated_dependency_environment_report_digest"),
            "dependency_environment_report_digest",
        ),
        "python_executable_sha256": _required_sha256(
            isolated.get("python_executable_sha256_after_preparation"),
            "python_executable_sha256",
        ),
        "formal_execution_commit": str(formal_lock.get("formal_execution_commit", "")),
        "formal_execution_lock_digest": _required_sha256(
            formal_lock.get("formal_execution_lock_digest"),
            "formal_execution_lock_digest",
        ),
    }
    if len(context["formal_execution_commit"]) != 40:
        raise ValueError("官方参考科学单元的 Git commit 身份无效")
    if not context["dependency_profile_id"]:
        raise ValueError("官方参考科学单元缺少 dependency_profile_id")
    if not context["python_version"]:
        raise ValueError("官方参考科学单元缺少固定 Python 版本")
    return context


def build_irreversible_random_material_digest(*values: Any) -> str:
    """把秘密随机材料转换为不可逆摘要, 不持久化 key、nonce 或张量原文."""

    digest = hashlib.sha256()
    for value in values:
        if value is None:
            digest.update(b"<none>")
            continue
        if isinstance(value, bytes):
            encoded = value
        elif isinstance(value, str):
            encoded = value.encode("utf-8")
        elif hasattr(value, "detach"):
            encoded = value.detach().cpu().contiguous().numpy().tobytes()
        else:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def write_official_reference_source_unit_payload(
    *,
    baseline_id: str,
    observations: Sequence[Mapping[str, Any]],
    random_identity_random: Mapping[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    """由真正执行官方 GPU 算子的子进程原子写出批次观测和来源.

    该函数在外部官方脚本完成本批次后调用. 因此 PyTorch、CUDA 和 GPU 字段
    来自实际科学进程, 而不是由后续 CPU 汇总会话推测.
    """

    context_text = os.environ.get(UNIT_CONTEXT_ENV_NAME, "")
    output_text = os.environ.get(UNIT_OUTPUT_ENV_NAME, "")
    if not context_text or not output_text:
        raise RuntimeError("官方参考科学子进程缺少原子单元上下文")
    context = json.loads(context_text)
    if not isinstance(context, dict) or context.get("baseline_id") != baseline_id:
        raise ValueError("官方参考科学子进程的方法身份不匹配")
    unit_start = int(context["unit_start"])
    unit_end = int(context["unit_end"])
    expected_indices = list(range(unit_start, unit_end))
    normalized_observations = [dict(observation) for observation in observations]
    actual_indices = [int(observation.get("prompt_index", -1)) for observation in normalized_observations]
    if actual_indices != expected_indices:
        raise ValueError("官方参考科学子进程未完整覆盖预注册 Prompt 范围")
    actual_python_version = platform.python_version()
    actual_python_digest = _file_sha256(Path(sys.executable).resolve())
    if actual_python_version != context["python_version"]:
        raise RuntimeError("官方参考科学子进程的 Python 版本与固定 profile 不一致")
    if actual_python_digest != context["python_executable_sha256"]:
        raise RuntimeError("官方参考科学子进程的 Python 可执行文件摘要不一致")

    runtime_environment = {
        "dependency_environment_ready": True,
        "formal_execution_lock_ready": True,
        "isolated_scientific_context_ready": True,
        "dependency_profile_id": context["dependency_profile_id"],
        "dependency_profile_digest": context["dependency_profile_digest"],
        "direct_requirements_digest": context["direct_requirements_digest"],
        "complete_hash_lock_digest": context["complete_hash_lock_digest"],
        "formal_execution_commit": context["formal_execution_commit"],
        "formal_execution_lock_digest": context["formal_execution_lock_digest"],
        "python_version": actual_python_version,
        "package_versions": {"torch": str(torch_module.__version__)},
        "cuda_version": str(torch_module.version.cuda or ""),
        "gpu_name": str(torch_module.cuda.get_device_name(torch_module.cuda.current_device())),
        "device_count": int(torch_module.cuda.device_count()),
        "isolated_scientific_context": {
            "dependency_environment_report_actual_digest": context[
                "dependency_environment_report_digest"
            ],
            "current_python_executable_sha256": actual_python_digest,
        },
    }
    provenance = build_scientific_unit_provenance(
        scientific_unit_id=str(context["scientific_unit_id"]),
        scientific_unit_config_digest=str(context["scientific_unit_config_digest"]),
        runtime_environment=runtime_environment,
        execution_device_name=f"cuda:{int(torch_module.cuda.current_device())}",
        torch_module=torch_module,
        random_identity_random=dict(random_identity_random),
    )
    payload = {
        "report_schema": OFFICIAL_REFERENCE_SOURCE_PAYLOAD_SCHEMA,
        "schema_version": OFFICIAL_REFERENCE_SOURCE_PAYLOAD_SCHEMA_VERSION,
        "baseline_id": baseline_id,
        "unit_start": unit_start,
        "unit_end": unit_end,
        "observations": normalized_observations,
        "observations_digest": build_stable_digest(normalized_observations),
        "scientific_unit_provenance": provenance,
        "supports_paper_claim": False,
    }
    payload["source_payload_digest"] = build_stable_digest(payload)
    _atomic_write_json(Path(output_text), payload)
    return payload


def _numeric(value: Any, field_name: str) -> float:
    """读取有限浮点观测."""

    try:
        resolved = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"官方参考观测的 {field_name} 不是数值") from error
    if not math.isfinite(resolved):
        raise ValueError(f"官方参考观测的 {field_name} 不是有限数")
    return resolved


def _validate_observations(
    baseline_id: str,
    observations: Sequence[Mapping[str, Any]],
    unit_start: int,
    unit_end: int,
) -> list[dict[str, Any]]:
    """按官方方法校验逐 Prompt 原始观测, 不接受仅有聚合值的批次."""

    normalized = [dict(observation) for observation in observations]
    if [int(item.get("prompt_index", -1)) for item in normalized] != list(
        range(unit_start, unit_end)
    ):
        raise ValueError("官方参考批次的 Prompt 索引覆盖不完整")
    for observation in normalized:
        _required_sha256(observation.get("prompt_digest"), "prompt_digest")
        if isinstance(observation.get("prompt_seed_random"), bool) or not isinstance(
            observation.get("prompt_seed_random"), int
        ):
            raise ValueError("官方参考逐 Prompt prompt_seed_random 身份无效")
        if baseline_id == "tree_ring":
            for field_name in ("no_w_metric", "w_metric", "w_no_sim", "w_sim"):
                _numeric(observation.get(field_name), field_name)
        elif baseline_id == "gaussian_shading":
            for field_name in ("bit_accuracy", "clip_score"):
                _numeric(observation.get(field_name), field_name)
            for field_name in ("detection_hit", "traceability_hit"):
                if observation.get(field_name) not in {True, False}:
                    raise ValueError(f"Gaussian Shading 的 {field_name} 必须是布尔值")
            _required_sha256(
                observation.get("random_material_digest_random"),
                "random_material_digest_random",
            )
        elif baseline_id == "shallow_diffuse":
            for field_name in (
                "no_w_metrics_none",
                "avg_metrics_none",
                "clip_scores_no_w",
                "clip_scores_avg",
            ):
                _numeric(observation.get(field_name), field_name)
        else:
            raise ValueError("未知官方参考 baseline_id")
    return normalized


def _validate_source_payload(
    payload: Mapping[str, Any],
    *,
    baseline_id: str,
    unit_start: int,
    unit_end: int,
    scientific_unit_id: str,
    scientific_config_digest: str,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """校验科学子进程写出的完整批次和真实运行来源."""

    resolved = dict(payload)
    if (
        resolved.get("report_schema") != OFFICIAL_REFERENCE_SOURCE_PAYLOAD_SCHEMA
        or resolved.get("schema_version")
        != OFFICIAL_REFERENCE_SOURCE_PAYLOAD_SCHEMA_VERSION
        or resolved.get("baseline_id") != baseline_id
        or int(resolved.get("unit_start", -1)) != unit_start
        or int(resolved.get("unit_end", -1)) != unit_end
        or resolved.get("supports_paper_claim") is not False
    ):
        raise ValueError("官方参考科学子进程 payload 身份不匹配")
    source_digest = _required_sha256(
        resolved.get("source_payload_digest"),
        "source_payload_digest",
    )
    digest_payload = {
        key: value for key, value in resolved.items() if key != "source_payload_digest"
    }
    if build_stable_digest(digest_payload) != source_digest:
        raise ValueError("官方参考科学子进程 payload 自摘要不匹配")
    observations = _validate_observations(
        baseline_id,
        resolved.get("observations", ()),
        unit_start,
        unit_end,
    )
    if build_stable_digest(observations) != resolved.get("observations_digest"):
        raise ValueError("官方参考逐 Prompt 观测摘要不匹配")
    provenance = validate_scientific_unit_provenance(
        resolved.get("scientific_unit_provenance", {}),
        expected_unit_id=scientific_unit_id,
        expected_config_digest=scientific_config_digest,
    )
    if expected_identity is not None:
        execution_environment = provenance["scientific_execution_environment"]
        for field_name in (
            "dependency_profile_id",
            "dependency_profile_digest",
            "direct_requirements_digest",
            "complete_hash_lock_digest",
            "formal_execution_commit",
            "formal_execution_lock_digest",
            "python_version",
            "python_executable_sha256",
            "torch_version",
            "torch_cuda_version",
        ):
            if execution_environment.get(field_name) != expected_identity.get(
                field_name
            ):
                raise ValueError(
                    f"官方参考科学单元来源的 {field_name} 与稳定身份不一致"
                )
    resolved["observations"] = observations
    resolved["scientific_unit_provenance"] = provenance
    return resolved


def _parse_official_command_flags(
    command: Sequence[str],
    boolean_flags: set[str],
) -> dict[str, str | bool]:
    """解析不允许重复键的官方 argv, 保留真实参数文本."""

    if len(command) < 3:
        raise ValueError("官方参考批次命令不完整")
    flags: dict[str, str | bool] = {}
    index = 2
    while index < len(command):
        name = str(command[index])
        if not name.startswith("--") or name in flags:
            raise ValueError("官方参考批次 argv 结构无效")
        if name in boolean_flags:
            flags[name] = True
            index += 1
        else:
            if index + 1 >= len(command):
                raise ValueError("官方参考批次 argv 缺少参数值")
            flags[name] = str(command[index + 1])
            index += 2
    return flags


def _expected_official_command_parameters(
    *,
    baseline_id: str,
    scientific_config: Mapping[str, Any],
    unit_start: int,
    unit_end: int,
) -> tuple[str, dict[str, str | bool], set[str], str, str]:
    """构造不含 workspace 路径的预期科学参数和两个路径参数名."""

    config = dict(scientific_config)
    if baseline_id == "tree_ring":
        parameters: dict[str, str | bool] = {
            "--dataset": str(config["dataset"]),
            "--w_channel": str(config["w_channel"]),
            "--w_pattern": str(config["w_pattern"]),
            "--gen_seed": str(config["gen_seed"]),
            "--w_seed": str(config["w_seed"]),
            "--start": str(unit_start),
            "--end": str(unit_end),
            "--reference_model": str(config["reference_model"]),
            "--with_tracking": True,
        }
        return (
            "run_tree_ring_watermark.py",
            parameters,
            {"--run_name", "--model_id", "--reference_model_pretrain"},
            "--model_id",
            "--reference_model_pretrain",
        )
    if baseline_id == "gaussian_shading":
        parameters = {
            "--num": str(unit_end - unit_start),
            "--fpr": str(config["fpr"]),
            "--channel_copy": str(config["channel_copy"]),
            "--hw_copy": str(config["hw_copy"]),
            "--user_number": str(config["user_number"]),
            "--gen_seed": str(config["gen_seed"]),
            "--image_length": str(config["image_length"]),
            "--guidance_scale": str(config["guidance_scale"]),
            "--num_inference_steps": str(config["num_inference_steps"]),
            "--num_inversion_steps": str(config["num_inversion_steps"]),
            "--dataset_path": str(config["dataset_path"]),
            "--reference_model": str(config["reference_model"]),
        }
        if config["use_chacha"] is True:
            parameters["--chacha"] = True
        return (
            "run_gaussian_shading.py",
            parameters,
            {"--model_path", "--output_path", "--reference_model_pretrain"},
            "--model_path",
            "--reference_model_pretrain",
        )
    if baseline_id == "shallow_diffuse":
        parameters = {
            "--dataset": str(config["dataset"]),
            "--image_length": str(config["image_length"]),
            "--guidance_scale": str(config["guidance_scale"]),
            "--num_inference_steps": str(config["num_inference_steps"]),
            "--w_seed": str(config["w_seed"]),
            "--w_channel": str(config["w_channel"]),
            "--w_pattern": str(config["w_pattern"]),
            "--w_mask_shape": str(config["w_mask_shape"]),
            "--w_radius": str(config["w_radius"]),
            "--w_measurement": str(config["w_measurement"]),
            "--w_injection": str(config["w_injection"]),
            "--reference_model": str(config["reference_model"]),
            "--edit_time_list": str(config["edit_time_list"]),
            "--start": str(unit_start),
            "--end": str(unit_end),
        }
        return (
            "run_shallow_diffuse_t2i.py",
            parameters,
            {"--run_name", "--model_id", "--reference_model_pretrain"},
            "--model_id",
            "--reference_model_pretrain",
        )
    raise ValueError("未知官方参考 baseline_id")


def _validate_official_model_source_bindings(
    *,
    scientific_config: Mapping[str, Any],
    model_repository_report: Mapping[str, Any],
    openclip_report: Mapping[str, Any],
) -> tuple[str, str]:
    """复验稳定内容身份, 仅把绝对路径作为当前执行会话的定位证据."""

    config = dict(scientific_config)
    model_snapshot = model_repository_report.get("model_snapshot_content")
    if not isinstance(model_snapshot, Mapping) or not all(
        (
            model_snapshot.get("snapshot_content_digest")
            == config["model_snapshot_content_digest"],
            model_repository_report.get("official_model_id")
            == config["official_model_id"],
            model_repository_report.get("official_model_revision")
            == config["official_model_revision"],
            openclip_report.get("openclip_model_name")
            == config["reference_model"],
            openclip_report.get("openclip_checkpoint_sha256")
            == config["openclip_checkpoint_sha256"],
            openclip_report.get("openclip_snapshot_content_digest")
            == config["openclip_snapshot_content_digest"],
        )
    ):
        raise ValueError("官方参考规范科学配置与模型快照报告不一致")
    model_locator = str(
        model_repository_report.get("effective_official_model_id", "")
    )
    checkpoint_locator = str(openclip_report.get("openclip_checkpoint_path", ""))
    if not model_locator or not checkpoint_locator:
        raise ValueError("官方参考批次缺少模型或 OpenCLIP 本地定位报告")
    return model_locator, checkpoint_locator


def build_official_reference_canonical_command_identity(
    *,
    baseline_id: str,
    command: Sequence[str],
    unit_start: int,
    unit_end: int,
    scientific_config: Mapping[str, Any],
    runtime_model_locator: str | None = None,
    runtime_openclip_checkpoint_locator: str | None = None,
) -> dict[str, Any]:
    """从原始 argv 生成不包含 workspace 绝对路径的规范命令身份."""

    (
        entrypoint_filename,
        expected_parameters,
        operational_flags,
        model_flag,
        checkpoint_flag,
    ) = _expected_official_command_parameters(
        baseline_id=baseline_id,
        scientific_config=scientific_config,
        unit_start=unit_start,
        unit_end=unit_end,
    )
    boolean_flags = {name for name, value in expected_parameters.items() if value is True}
    flags = _parse_official_command_flags(command, boolean_flags)
    actual_entrypoint_filename = str(command[1]).replace("\\", "/").rsplit(
        "/",
        1,
    )[-1]
    if actual_entrypoint_filename != entrypoint_filename:
        raise ValueError("官方参考批次入口与方法身份不匹配")
    if set(flags) != set(expected_parameters) | operational_flags:
        raise ValueError("官方参考批次 argv 字段集合与规范科学配置不一致")
    if any(flags.get(name) != value for name, value in expected_parameters.items()):
        raise ValueError("官方参考批次 argv 科学参数与规范科学配置不一致")
    model_locator = str(flags.get(model_flag, ""))
    checkpoint_locator = str(flags.get(checkpoint_flag, ""))
    if not model_locator or not checkpoint_locator:
        raise ValueError("官方参考原始 argv 缺少模型路径参数")
    if baseline_id == "gaussian_shading" and not str(
        flags.get("--output_path", "")
    ):
        raise ValueError("Gaussian Shading 原始 argv 缺少隔离输出路径")
    if runtime_model_locator is not None and model_locator != str(runtime_model_locator):
        raise ValueError("官方参考原始 argv 未绑定当前模型快照定位")
    if (
        runtime_openclip_checkpoint_locator is not None
        and checkpoint_locator != str(runtime_openclip_checkpoint_locator)
    ):
        raise ValueError("官方参考原始 argv 未绑定当前 OpenCLIP checkpoint 定位")

    config = dict(scientific_config)
    identity = {
        "report_schema": OFFICIAL_REFERENCE_COMMAND_IDENTITY_SCHEMA,
        "schema_version": OFFICIAL_REFERENCE_COMMAND_IDENTITY_SCHEMA_VERSION,
        "baseline_id": baseline_id,
        "entrypoint_filename": entrypoint_filename,
        "unit_start": int(unit_start),
        "unit_end": int(unit_end),
        "official_scientific_config_digest": (
            build_official_reference_config_digest(baseline_id, config)
        ),
        "official_command_scientific_parameters": expected_parameters,
        "official_model_binding": {
            "official_model_repository_id": str(config["official_model_id"]),
            "official_model_revision": str(config["official_model_revision"]),
            "model_snapshot_content_digest": str(
                config["model_snapshot_content_digest"]
            ),
        },
        "openclip_binding": {
            "openclip_model_name": str(config["reference_model"]),
            "openclip_checkpoint_sha256": str(
                config["openclip_checkpoint_sha256"]
            ),
            "openclip_snapshot_content_digest": str(
                config["openclip_snapshot_content_digest"]
            ),
        },
        "workspace_independent": True,
    }
    identity["official_command_canonical_identity_digest"] = build_stable_digest(
        identity
    )
    return identity


def _expected_unit_identity(
    *,
    baseline_id: str,
    unit_start: int,
    unit_end: int,
    config_digest: str,
    scientific_config: Mapping[str, Any],
    runtime_context: Mapping[str, Any],
    source_status: Mapping[str, Any],
    device_report: Mapping[str, Any],
) -> dict[str, Any]:
    """绑定一个批次在恢复时必须逐项相等的全部静态身份."""

    unit_id = _unit_id(baseline_id, unit_start, unit_end, config_digest)
    identity = {
        "baseline_id": baseline_id,
        "scientific_unit_id": unit_id,
        "scientific_unit_config_digest": config_digest,
        "scientific_config": dict(scientific_config),
        "unit_start": unit_start,
        "unit_end": unit_end,
        "prompt_indices": list(range(unit_start, unit_end)),
        "formal_execution_commit": runtime_context["formal_execution_commit"],
        "formal_execution_lock_digest": runtime_context[
            "formal_execution_lock_digest"
        ],
        "dependency_profile_id": runtime_context["dependency_profile_id"],
        "dependency_profile_digest": runtime_context["dependency_profile_digest"],
        "direct_requirements_digest": runtime_context[
            "direct_requirements_digest"
        ],
        "complete_hash_lock_digest": runtime_context["complete_hash_lock_digest"],
        "python_executable_sha256": runtime_context["python_executable_sha256"],
        "python_version": runtime_context["python_version"],
        "torch_version": str(device_report.get("torch_version", "")),
        "torch_cuda_version": str(device_report.get("torch_cuda_version", "")),
        "official_repository_commit": str(
            source_status.get("official_repository_commit", "")
        ),
        "source_patch_sha256": _required_sha256(
            source_status.get("source_patch_sha256"),
            "source_patch_sha256",
        ),
        "source_worktree_digest": _required_sha256(
            source_status.get("source_worktree_digest"),
            "source_worktree_digest",
        ),
    }
    if not identity["torch_version"] or not identity["torch_cuda_version"]:
        raise ValueError("官方参考科学单元缺少固定 PyTorch 或 CUDA build 身份")
    identity["unit_identity_digest"] = build_stable_digest(identity)
    return identity


def _validate_unit_record(
    record: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """严格复验一个已完成批次, 任一成员损坏都会闭锁."""

    resolved = dict(record)
    if (
        resolved.get("report_schema") != OFFICIAL_REFERENCE_UNIT_SCHEMA
        or resolved.get("schema_version") != OFFICIAL_REFERENCE_UNIT_SCHEMA_VERSION
        or resolved.get("unit_complete") is not True
        or resolved.get("supports_paper_claim") is not False
        or resolved.get("unit_identity") != dict(expected_identity)
    ):
        raise ValueError("已完成官方参考科学单元身份不匹配")
    record_digest = _required_sha256(
        resolved.get("scientific_unit_record_digest"),
        "scientific_unit_record_digest",
    )
    digest_payload = {
        key: value
        for key, value in resolved.items()
        if key != "scientific_unit_record_digest"
    }
    if build_stable_digest(digest_payload) != record_digest:
        raise ValueError("已完成官方参考科学单元自摘要不匹配")
    identity = dict(expected_identity)
    source_payload = _validate_source_payload(
        resolved.get("source_payload", {}),
        baseline_id=str(identity["baseline_id"]),
        unit_start=int(identity["unit_start"]),
        unit_end=int(identity["unit_end"]),
        scientific_unit_id=str(identity["scientific_unit_id"]),
        scientific_config_digest=str(identity["scientific_unit_config_digest"]),
        expected_identity=identity,
    )
    command_evidence = resolved.get("official_command_execution_evidence")
    canonical_command_identity = resolved.get(
        "official_command_canonical_identity"
    )
    cuda_report = resolved.get("cuda_inspection_report")
    if not isinstance(command_evidence, Mapping) or not all(
        (
            command_evidence.get("official_command_execution_evidence_ready") is True,
            command_evidence.get("baseline_id") == identity["baseline_id"],
            command_evidence.get("dependency_python_executable_sha256")
            == identity["python_executable_sha256"],
        )
    ):
        raise ValueError("官方参考科学单元的命令执行证据不完整")
    if not isinstance(canonical_command_identity, Mapping):
        raise ValueError("官方参考科学单元缺少规范命令身份")
    rebuilt_command_identity = build_official_reference_canonical_command_identity(
        baseline_id=str(identity["baseline_id"]),
        command=[str(value) for value in command_evidence["official_command"]],
        unit_start=int(identity["unit_start"]),
        unit_end=int(identity["unit_end"]),
        scientific_config=dict(identity["scientific_config"]),
    )
    if dict(canonical_command_identity) != rebuilt_command_identity:
        raise ValueError("官方参考科学单元规范命令身份与原始 argv 不一致")
    if not isinstance(cuda_report, Mapping) or not all(
        (
            cuda_report.get("decision") == "pass",
            cuda_report.get("cuda_available") is True,
            cuda_report.get("torch_version") == identity["torch_version"],
            cuda_report.get("torch_cuda_version") == identity["torch_cuda_version"],
            cuda_report.get("python_executable_sha256")
            == identity["python_executable_sha256"],
            build_stable_digest(dict(cuda_report))
            == command_evidence.get("cuda_inspection_report_digest"),
        )
    ):
        raise ValueError("官方参考科学单元的 CUDA 检查报告不自洽")
    execution_environment = source_payload["scientific_unit_provenance"][
        "scientific_execution_environment"
    ]
    if (
        execution_environment.get("cuda_device_name")
        != cuda_report.get("gpu_name")
        or execution_environment.get("visible_cuda_device_count")
        != cuda_report.get("device_count")
    ):
        raise ValueError("官方参考科学进程与命令前 CUDA 检查身份不一致")
    resolved["source_payload"] = source_payload
    return resolved


def _load_completed_units(
    unit_dir: Path,
    expected_identities: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """加载有效批次并返回缺失身份; 非预注册文件和损坏文件均直接闭锁."""

    unit_dir.mkdir(parents=True, exist_ok=True)
    expected_paths = {
        _unit_path(
            unit_dir,
            int(identity["unit_start"]),
            int(identity["unit_end"]),
        ): dict(identity)
        for identity in expected_identities
    }
    unexpected_paths: list[Path] = []
    for path in unit_dir.iterdir():
        if path in expected_paths:
            continue
        if path.is_file() and path.name.startswith(".unit_") and ".writing-" in path.name:
            path.unlink()
            continue
        unexpected_paths.append(path)
    if unexpected_paths:
        raise RuntimeError("官方参考科学单元目录包含非预注册批次")
    completed: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for path, identity in expected_paths.items():
        if not path.is_file():
            missing.append(identity)
            continue
        try:
            completed.append(_validate_unit_record(_read_json(path), identity))
        except Exception as error:
            raise RuntimeError(f"官方参考已完成科学单元损坏: {path.name}") from error
    return completed, missing


def _clean_unit_workspace(workspace: Path, workspace_root: Path) -> None:
    """只清理当前 outputs 下已验证归属的未完成批次工作目录."""

    resolved_workspace = workspace.resolve()
    resolved_root = workspace_root.resolve()
    resolved_workspace.relative_to(resolved_root)
    if resolved_workspace.exists():
        shutil.rmtree(resolved_workspace)
    resolved_workspace.mkdir(parents=True, exist_ok=True)


def _clear_workspace_root(workspace_root: Path, unit_dir: Path) -> None:
    """删除不作为事实来源的批次工作区, 并校验其与单元目录同属一输出根."""

    if workspace_root.is_symlink():
        raise RuntimeError("官方参考科学单元工作区不得是符号链接")
    resolved_workspace_root = workspace_root.resolve()
    resolved_unit_dir = unit_dir.resolve()
    if resolved_workspace_root.parent != resolved_unit_dir.parent:
        raise ValueError("官方参考科学单元工作区不属于同一输出根")
    resolved_workspace_root.mkdir(parents=True, exist_ok=True)
    for path in resolved_workspace_root.iterdir():
        if path.is_symlink():
            raise RuntimeError("官方参考科学单元工作区不得包含符号链接")
        path.resolve().relative_to(resolved_workspace_root)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    resolved_workspace_root.rmdir()


def _build_unit_record(
    *,
    identity: Mapping[str, Any],
    source_payload: Mapping[str, Any],
    command_evidence: Mapping[str, Any],
    canonical_command_identity: Mapping[str, Any],
    cuda_inspection_report: Mapping[str, Any],
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    """构造只能在完整子进程 payload 验证后提交的批次记录."""

    record = {
        "report_schema": OFFICIAL_REFERENCE_UNIT_SCHEMA,
        "schema_version": OFFICIAL_REFERENCE_UNIT_SCHEMA_VERSION,
        "unit_complete": True,
        "unit_identity": dict(identity),
        "source_payload": dict(source_payload),
        "official_command_execution_evidence": dict(command_evidence),
        "official_command_canonical_identity": dict(canonical_command_identity),
        "cuda_inspection_report": dict(cuda_inspection_report),
        "stdout": str(stdout),
        "stderr": str(stderr),
        "supports_paper_claim": False,
    }
    record["scientific_unit_record_digest"] = build_stable_digest(record)
    return record


def run_official_reference_unit_schedule(
    *,
    root_path: Path,
    baseline_id: str,
    start_index: int,
    sample_count: int,
    batch_size: int,
    scientific_config: Mapping[str, Any],
    formal_execution_lock: Mapping[str, Any],
    source_status: Mapping[str, Any],
    dependency_environment_report: Mapping[str, Any],
    device_report: Mapping[str, Any],
    model_repository_report: Mapping[str, Any],
    openclip_report: Mapping[str, Any],
    dependency_python_executable: str | Path,
    unit_dir: Path,
    workspace_root: Path,
    timeout_seconds: int,
    command_builder: Callable[[int, int, Path], tuple[list[str], Path, dict[str, str]]],
    progress: object | None = None,
) -> dict[str, Any]:
    """验证并补齐缺失官方小批次, 然后确定性汇总逐 Prompt 观测."""

    ranges = build_official_reference_unit_ranges(start_index, sample_count, batch_size)
    config_digest = build_official_reference_config_digest(
        baseline_id,
        scientific_config,
    )
    runtime_context = build_official_reference_runtime_context(
        baseline_id=baseline_id,
        formal_execution_lock=formal_execution_lock,
        dependency_environment_report=dependency_environment_report,
    )
    runtime_model_locator, runtime_checkpoint_locator = (
        _validate_official_model_source_bindings(
            scientific_config=scientific_config,
            model_repository_report=model_repository_report,
            openclip_report=openclip_report,
        )
    )
    expected_identities = [
        _expected_unit_identity(
            baseline_id=baseline_id,
            unit_start=unit_start,
            unit_end=unit_end,
            config_digest=config_digest,
            scientific_config=scientific_config,
            runtime_context=runtime_context,
            source_status=source_status,
            device_report=device_report,
        )
        for unit_start, unit_end in ranges
    ]
    completed_before, missing = _load_completed_units(unit_dir, expected_identities)
    _clear_workspace_root(workspace_root, unit_dir)
    executed_unit_ids: list[str] = []
    for identity in missing:
        unit_start = int(identity["unit_start"])
        unit_end = int(identity["unit_end"])
        unit_id = str(identity["scientific_unit_id"])
        workspace = workspace_root / _sha256_text(unit_id)[:24]
        _clean_unit_workspace(workspace, workspace_root)
        source_payload_path = workspace / "source_unit_payload.json"
        command, working_directory, extra_environment = command_builder(
            unit_start,
            unit_end,
            workspace,
        )
        canonical_command_identity = (
            build_official_reference_canonical_command_identity(
                baseline_id=baseline_id,
                command=command,
                unit_start=unit_start,
                unit_end=unit_end,
                scientific_config=scientific_config,
                runtime_model_locator=runtime_model_locator,
                runtime_openclip_checkpoint_locator=runtime_checkpoint_locator,
            )
        )
        context = {
            **runtime_context,
            "scientific_unit_id": unit_id,
            "scientific_unit_config_digest": config_digest,
            "unit_start": unit_start,
            "unit_end": unit_end,
        }
        environment = dict(os.environ)
        environment.update(extra_environment)
        existing_python_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            value
            for value in (str(root_path), existing_python_path)
            if value
        )
        environment[UNIT_CONTEXT_ENV_NAME] = json.dumps(
            context,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        environment[UNIT_OUTPUT_ENV_NAME] = str(source_payload_path)
        completed = run_quiet_subprocess_with_progress(
            command,
            cwd=working_directory,
            env=environment,
            timeout_seconds=int(timeout_seconds),
            progress=progress,
            progress_profile=(
                f"operation={baseline_id}_official_unit "
                f"start={unit_start} end={unit_end}"
            ),
        )
        if int(completed.returncode) != 0:
            raise RuntimeError(
                f"{baseline_id} 官方科学单元执行失败: "
                f"{unit_start}-{unit_end}"
            )
        if not source_payload_path.is_file():
            raise RuntimeError("官方科学子进程成功退出但没有提交原子 payload")
        source_payload = _validate_source_payload(
            _read_json(source_payload_path),
            baseline_id=baseline_id,
            unit_start=unit_start,
            unit_end=unit_end,
            scientific_unit_id=unit_id,
            scientific_config_digest=config_digest,
            expected_identity=identity,
        )
        command_result = {
            "official_command_requested": True,
            "official_command": command,
            "return_code": 0,
        }
        command_evidence = bind_successful_official_command_execution_evidence(
            command_result,
            baseline_id=baseline_id,
            command=command,
            working_directory=working_directory,
            dependency_python_executable=dependency_python_executable,
            cuda_inspection_report=dict(device_report),
        )
        record = _build_unit_record(
            identity=identity,
            source_payload=source_payload,
            command_evidence=command_evidence,
            canonical_command_identity=canonical_command_identity,
            cuda_inspection_report=device_report,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        record_path = _unit_path(unit_dir, unit_start, unit_end)
        if record_path.exists():
            raise RuntimeError("官方参考科学单元禁止覆盖已存在记录")
        _atomic_write_json(record_path, record)
        _validate_unit_record(_read_json(record_path), identity)
        executed_unit_ids.append(unit_id)
        shutil.rmtree(workspace)

    completed, still_missing = _load_completed_units(unit_dir, expected_identities)
    if still_missing:
        raise RuntimeError("官方参考科学单元补算后仍存在缺失批次")
    _clear_workspace_root(workspace_root, unit_dir)
    completed.sort(key=lambda item: int(item["unit_identity"]["unit_start"]))
    observations = [
        observation
        for record in completed
        for observation in record["source_payload"]["observations"]
    ]
    expected_indices = list(range(int(start_index), int(start_index) + int(sample_count)))
    if [int(item["prompt_index"]) for item in observations] != expected_indices:
        raise RuntimeError("官方参考科学单元汇总未完整覆盖预注册 Prompt")
    provenances = [
        record["source_payload"]["scientific_unit_provenance"]
        for record in completed
    ]
    provenance_aggregate = aggregate_scientific_unit_provenance(
        provenances,
        expected_reference_count=len(expected_identities),
    )
    coverage_ready = bool(
        provenance_aggregate["scientific_unit_provenance_ready"]
        and len(observations) == int(sample_count)
    )
    command_identities = [
        record["official_command_canonical_identity"] for record in completed
    ]
    return {
        "official_command_requested": True,
        "return_code": 0 if coverage_ready else 99,
        "official_unit_coverage_ready": coverage_ready,
        "official_unit_batch_size": int(batch_size),
        "official_unit_expected_count": len(expected_identities),
        "official_unit_completed_count": len(completed),
        "official_unit_resumed_count": len(completed_before),
        "official_unit_executed_count": len(executed_unit_ids),
        "official_unit_executed_ids": executed_unit_ids,
        "official_unit_ids": [
            str(identity["scientific_unit_id"])
            for identity in expected_identities
        ],
        "official_unit_records_digest": build_stable_digest(completed),
        "official_unit_observations_digest": build_stable_digest(observations),
        "official_unit_command_identities_digest": build_stable_digest(
            command_identities
        ),
        "official_unit_command_identities": command_identities,
        "official_unit_observations": observations,
        "official_scientific_config": dict(scientific_config),
        "official_scientific_config_digest": config_digest,
        "scientific_unit_provenance": provenance_aggregate,
        "official_command_execution_evidence_ready": coverage_ready,
        "supports_paper_claim": False,
    }


def _binary_score_metrics(
    negative_scores: Sequence[float],
    positive_scores: Sequence[float],
) -> dict[str, float]:
    """按官方 sklearn ROC 定义从逐样本分数复算 AUC、准确率和低 FPR TPR."""

    negatives = [float(value) for value in negative_scores]
    positives = [float(value) for value in positive_scores]
    if not negatives or len(negatives) != len(positives):
        raise ValueError("官方 ROC 聚合要求等量正负样本")
    values = sorted(set(negatives + positives), reverse=True)
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for threshold in values:
        false_positive_rate = sum(value >= threshold for value in negatives) / len(
            negatives
        )
        true_positive_rate = sum(value >= threshold for value in positives) / len(
            positives
        )
        points.append((false_positive_rate, true_positive_rate))
    auc = sum(
        (right_fpr - left_fpr) * (left_tpr + right_tpr) / 2.0
        for (left_fpr, left_tpr), (right_fpr, right_tpr) in zip(
            points,
            points[1:],
        )
    )
    accuracy = max(
        1.0 - (false_positive_rate + (1.0 - true_positive_rate)) / 2.0
        for false_positive_rate, true_positive_rate in points
    )
    eligible_tpr = [
        true_positive_rate
        for false_positive_rate, true_positive_rate in points
        if false_positive_rate < 0.01
    ]
    return {
        "auc": auc,
        "accuracy": accuracy,
        "true_positive_rate_at_one_percent_fpr": eligible_tpr[-1],
    }


def aggregate_tree_ring_unit_observations(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """从 Tree-Ring 原始检测距离与 CLIP 分数复算官方指标."""

    resolved = [dict(item) for item in observations]
    roc = _binary_score_metrics(
        [-_numeric(item["no_w_metric"], "no_w_metric") for item in resolved],
        [-_numeric(item["w_metric"], "w_metric") for item in resolved],
    )
    return {
        "sample_count": len(resolved),
        "positive_count": len(resolved),
        "negative_count": len(resolved),
        **roc,
        "clip_score_mean": mean(
            _numeric(item["w_no_sim"], "w_no_sim") for item in resolved
        ),
        "watermarked_clip_score_mean": mean(
            _numeric(item["w_sim"], "w_sim") for item in resolved
        ),
    }


def aggregate_gaussian_shading_unit_observations(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """从 Gaussian Shading 的逐 Prompt 命中与质量观测复算官方指标."""

    resolved = [dict(item) for item in observations]
    accuracies = [_numeric(item["bit_accuracy"], "bit_accuracy") for item in resolved]
    clip_scores = [_numeric(item["clip_score"], "clip_score") for item in resolved]
    if len(resolved) < 2:
        raise ValueError("Gaussian Shading 正式标准差至少需要2个样本")
    return {
        "sample_count": len(resolved),
        "positive_count": len(resolved),
        "detection_true_positive_rate": mean(
            float(bool(item["detection_hit"])) for item in resolved
        ),
        "traceability_true_positive_rate": mean(
            float(bool(item["traceability_hit"])) for item in resolved
        ),
        "mean_bit_accuracy": mean(accuracies),
        "std_bit_accuracy": stdev(accuracies),
        "mean_clip_score": mean(clip_scores),
        "std_clip_score": stdev(clip_scores),
    }


def aggregate_shallow_diffuse_unit_observations(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """从 Shallow Diffuse 原始检测分数与 CLIP 分数复算官方指标."""

    resolved = [dict(item) for item in observations]
    roc = _binary_score_metrics(
        [
            _numeric(item["no_w_metrics_none"], "no_w_metrics_none")
            for item in resolved
        ],
        [
            _numeric(item["avg_metrics_none"], "avg_metrics_none")
            for item in resolved
        ],
    )
    return {
        "sample_count": len(resolved),
        "positive_count": len(resolved),
        "negative_count": len(resolved),
        **roc,
        "clip_score_mean": mean(
            _numeric(item["clip_scores_no_w"], "clip_scores_no_w")
            for item in resolved
        ),
        "watermarked_clip_score_mean": mean(
            _numeric(item["clip_scores_avg"], "clip_scores_avg")
            for item in resolved
        ),
    }


def validate_persisted_official_reference_units(
    *,
    unit_dir: Path,
    baseline_id: str,
    start_index: int,
    sample_count: int,
    batch_size: int,
) -> dict[str, Any]:
    """从持久化批次记录重新验证 exact-set 覆盖并复算科学指标.

    该函数供打包和后续 CPU 证据闭合使用. 它不读取运行摘要中的 ready 标志,
    而是从每个单元的自摘要、逐 Prompt 观测和来源记录重新建立覆盖结论.
    """

    ranges = build_official_reference_unit_ranges(
        start_index,
        sample_count,
        batch_size,
    )
    expected_paths = [
        _unit_path(unit_dir, unit_start, unit_end)
        for unit_start, unit_end in ranges
    ]
    if not unit_dir.is_dir():
        raise FileNotFoundError("官方参考科学单元目录不存在")
    workspace_root = unit_dir.parent / "scientific_unit_workspace"
    if workspace_root.exists() or workspace_root.is_symlink():
        raise RuntimeError("官方参考持久化结果不得保留科学单元工作区")
    actual_paths = sorted(unit_dir.iterdir())
    if any(path.is_symlink() for path in actual_paths):
        raise RuntimeError("官方参考持久化科学单元不得包含符号链接")
    if actual_paths != sorted(expected_paths):
        raise RuntimeError("官方参考持久化科学单元不是预注册 exact-set")

    records: list[dict[str, Any]] = []
    stable_scientific_config: dict[str, Any] | None = None
    stable_identity_values: dict[str, set[Any]] = {
        field_name: set()
        for field_name in (
            "scientific_unit_config_digest",
            "formal_execution_commit",
            "formal_execution_lock_digest",
            "dependency_profile_id",
            "dependency_profile_digest",
            "direct_requirements_digest",
            "complete_hash_lock_digest",
            "python_executable_sha256",
            "python_version",
            "torch_version",
            "torch_cuda_version",
            "official_repository_commit",
            "source_patch_sha256",
            "source_worktree_digest",
        )
    }
    for (unit_start, unit_end), path in zip(ranges, expected_paths):
        raw_record = _read_json(path)
        identity = raw_record.get("unit_identity")
        if not isinstance(identity, Mapping):
            raise TypeError("官方参考持久化科学单元缺少 unit_identity")
        resolved_identity = dict(identity)
        identity_digest = _required_sha256(
            resolved_identity.get("unit_identity_digest"),
            "unit_identity_digest",
        )
        digest_identity = {
            key: value
            for key, value in resolved_identity.items()
            if key != "unit_identity_digest"
        }
        if build_stable_digest(digest_identity) != identity_digest:
            raise ValueError("官方参考持久化科学单元身份摘要不匹配")
        config_digest = _required_sha256(
            resolved_identity.get("scientific_unit_config_digest"),
            "scientific_unit_config_digest",
        )
        scientific_config = resolved_identity.get("scientific_config")
        if not isinstance(scientific_config, Mapping):
            raise TypeError("官方参考持久化科学单元缺少规范科学配置")
        resolved_scientific_config = dict(scientific_config)
        if build_official_reference_config_digest(
            baseline_id,
            resolved_scientific_config,
        ) != config_digest:
            raise ValueError("官方参考科学配置与配置摘要不匹配")
        if stable_scientific_config is None:
            stable_scientific_config = resolved_scientific_config
        elif stable_scientific_config != resolved_scientific_config:
            raise ValueError("官方参考科学配置跨批次漂移")
        expected_unit_id = _unit_id(
            baseline_id,
            unit_start,
            unit_end,
            config_digest,
        )
        if not all(
            (
                resolved_identity.get("baseline_id") == baseline_id,
                resolved_identity.get("scientific_unit_id") == expected_unit_id,
                resolved_identity.get("unit_start") == unit_start,
                resolved_identity.get("unit_end") == unit_end,
                resolved_identity.get("prompt_indices")
                == list(range(unit_start, unit_end)),
            )
        ):
            raise ValueError("官方参考持久化科学单元范围身份不匹配")
        record = _validate_unit_record(raw_record, resolved_identity)
        prompt_seed_offset = int(
            resolved_scientific_config[
                "w_seed" if baseline_id == "shallow_diffuse" else "gen_seed"
            ]
        )
        for observation in record["source_payload"]["observations"]:
            if int(observation["prompt_seed_random"]) != (
                int(observation["prompt_index"]) + prompt_seed_offset
            ):
                raise ValueError("官方参考逐 Prompt 随机种子与科学配置不一致")
        records.append(record)
        for field_name, values in stable_identity_values.items():
            values.add(resolved_identity.get(field_name))

    drift_fields = sorted(
        field_name
        for field_name, values in stable_identity_values.items()
        if len(values) != 1 or None in values or "" in values
    )
    if drift_fields:
        raise ValueError(
            "官方参考科学单元稳定身份跨批次漂移: " + ",".join(drift_fields)
        )
    observations = [
        observation
        for record in records
        for observation in record["source_payload"]["observations"]
    ]
    if baseline_id == "tree_ring":
        metric_summary = aggregate_tree_ring_unit_observations(observations)
    elif baseline_id == "gaussian_shading":
        metric_summary = aggregate_gaussian_shading_unit_observations(observations)
    elif baseline_id == "shallow_diffuse":
        metric_summary = aggregate_shallow_diffuse_unit_observations(observations)
    else:
        raise ValueError("未知官方参考 baseline_id")
    provenance_aggregate = aggregate_scientific_unit_provenance(
        [
            record["source_payload"]["scientific_unit_provenance"]
            for record in records
        ],
        expected_reference_count=len(records),
    )
    command_identities = [
        record["official_command_canonical_identity"] for record in records
    ]
    return {
        "official_unit_coverage_ready": (
            len(records) == len(ranges)
            and len(observations) == int(sample_count)
            and provenance_aggregate["scientific_unit_provenance_ready"] is True
        ),
        "official_unit_expected_count": len(ranges),
        "official_unit_completed_count": len(records),
        "official_unit_records_digest": build_stable_digest(records),
        "official_unit_observations_digest": build_stable_digest(observations),
        "official_unit_command_identities_digest": build_stable_digest(
            command_identities
        ),
        "scientific_unit_provenance": provenance_aggregate,
        "metric_summary": metric_summary,
        "stable_unit_identity": {
            field_name: next(iter(values))
            for field_name, values in stable_identity_values.items()
        },
        "official_scientific_config": stable_scientific_config,
        "official_scientific_config_digest": build_official_reference_config_digest(
            baseline_id,
            stable_scientific_config or {},
        ),
        "official_unit_commands": [
            {
                "unit_start": int(record["unit_identity"]["unit_start"]),
                "unit_end": int(record["unit_identity"]["unit_end"]),
                "command": list(
                    record["official_command_execution_evidence"][
                        "official_command"
                    ]
                ),
                "canonical_identity": dict(
                    record["official_command_canonical_identity"]
                ),
            }
            for record in records
        ],
    }


def validate_official_reference_scientific_config_and_commands(
    *,
    baseline_id: str,
    scientific_config: Mapping[str, Any],
    unit_commands: Sequence[Mapping[str, Any]],
    run_summary: Mapping[str, Any],
    model_repository_report: Mapping[str, Any],
    openclip_report: Mapping[str, Any],
) -> None:
    """验证规范科学配置以及每个批次实际 argv 的科学参数."""

    config = dict(scientific_config)
    required_common = {
        "sample_count": int(run_summary["sample_count"]),
        "start_index": int(run_summary["start_index"]),
        "unit_batch_size": int(run_summary["official_unit_batch_size"]),
        "official_model_id": str(run_summary["model_source_repository_id"]),
        "official_model_revision": str(run_summary["model_source_revision"]),
        "dataset_revision": str(run_summary["prompt_dataset_revision"]),
        "model_snapshot_content_digest": str(
            run_summary["model_snapshot_content_digest"]
        ),
        "openclip_checkpoint_sha256": str(
            run_summary["openclip_checkpoint_sha256"]
        ),
        "openclip_snapshot_content_digest": str(
            run_summary["openclip_snapshot_content_digest"]
        ),
    }
    if any(config.get(name) != value for name, value in required_common.items()):
        raise ValueError("官方参考规范科学配置与运行摘要不一致")
    dataset_field = "dataset_path" if baseline_id == "gaussian_shading" else "dataset"
    if config.get(dataset_field) != run_summary.get("prompt_dataset_repository_id"):
        raise ValueError("官方参考规范科学配置与 Prompt 数据集摘要不一致")
    _validate_official_model_source_bindings(
        scientific_config=config,
        model_repository_report=model_repository_report,
        openclip_report=openclip_report,
    )
    for unit_command in unit_commands:
        unit_start = int(unit_command["unit_start"])
        unit_end = int(unit_command["unit_end"])
        command = [str(value) for value in unit_command["command"]]
        rebuilt_identity = build_official_reference_canonical_command_identity(
            baseline_id=baseline_id,
            command=command,
            unit_start=unit_start,
            unit_end=unit_end,
            scientific_config=config,
        )
        stored_identity = unit_command.get("canonical_identity")
        if not isinstance(stored_identity, Mapping) or dict(
            stored_identity
        ) != rebuilt_identity:
            raise ValueError("官方参考批次规范命令身份与原始 argv 不一致")
