"""运行单个 SD3.5 common-backbone method-faithful baseline。

该模块只负责 Tree-Ring、Gaussian Shading 和 Shallow Diffuse 的公平主表轨道。
T2SMark 使用独立的正式复现 runner，避免同一方法存在两条相互覆盖的正式入口。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from typing import Any, Mapping
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.runtime import repository_environment
from experiments.runtime.model_sources import get_model_source, require_registered_model_reference
from experiments.protocol.fixed_fpr_observation_audit import audit_fixed_fpr_observation_threshold
from experiments.protocol.formal_randomization import (
    DEFAULT_FORMAL_RANDOMIZATION_REPEAT_ID,
    build_formal_randomization_identity,
    formal_runtime_randomization_plan_record,
    formal_watermark_key_seed_random,
    require_formal_watermark_key_plan,
    resolve_formal_randomization_repeat,
)
from experiments.protocol.method_runtime_config import load_formal_method_runtime_config
from experiments.protocol.paper_run_config import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_TARGET_FPR,
    build_paper_run_config,
    normalize_paper_run_name,
    resolve_count_from_environment,
)
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.protocol.splits import apply_split_assignments
from experiments.runtime.progress import (
    PROGRESS_EVENT_ENV_NAME,
    call_runner_with_progress_status,
    progress_bar,
    run_quiet_subprocess_with_progress,
    update_progress,
)
from experiments.runtime.archive_naming import utc_archive_token
from experiments.runtime.repository_environment import (
    build_runtime_environment_report,
    file_digest,
    resolve_code_version,
)
from experiments.runtime.scientific_execution_binding import (
    validate_scientific_execution_binding,
)
from external_baseline.primary.sd35_method_faithful_common import supported_formal_image_attack_names
from external_baseline.primary.sd35_method_faithful_units import (
    validate_method_faithful_adapter_unit_evidence,
)
from paper_experiments.baselines.method_faithful_observation_collection import (
    canonical_prompt_protocol_digest,
    validate_formal_attack_observation_identities,
)
from paper_experiments.baselines.method_faithful_numerical_fidelity import (
    build_method_faithful_numerical_fidelity_report,
    validate_method_faithful_numerical_fidelity_report,
)
from paper_experiments.baselines.observation_io import load_baseline_observation_rows


DEFAULT_OUTPUT_DIR = "outputs/external_baseline_method_faithful"
_COMMON_BACKBONE_SOURCE = get_model_source("stabilityai_stable_diffusion_3_5_medium")
DEFAULT_MODEL_ID = _COMMON_BACKBONE_SOURCE.repository_id
DEFAULT_MODEL_REVISION = _COMMON_BACKBONE_SOURCE.revision
MODEL_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_FORMAL_METHOD_DEFAULTS = load_formal_method_runtime_config(".")
_DEFAULT_RANDOMIZATION_REPEAT = resolve_formal_randomization_repeat(None)
DEFAULT_BASELINE_SEED = _FORMAL_METHOD_DEFAULTS.seed
DEFAULT_WATERMARK_KEY_SEED_RANDOM = formal_watermark_key_seed_random(
    "slm_wm_paper_key",
    _DEFAULT_RANDOMIZATION_REPEAT,
)
DEFAULT_PACKAGE_PATTERN = "external_baseline_method_faithful_package_*.zip"
METHOD_FAITHFUL_BASELINE_IDS = (
    "tree_ring",
    "gaussian_shading",
    "shallow_diffuse",
)
DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES = ",".join(supported_formal_image_attack_names())


@dataclass(frozen=True)
class ExternalBaselineMethodFaithfulConfig:
    """描述一次单 baseline common-backbone 正式运行。"""

    output_dir: str = DEFAULT_OUTPUT_DIR
    drive_output_dir: str = field(
        default_factory=lambda: build_paper_run_config(".").drive_dir("external_baseline_method_faithful")
    )
    prompt_set: str = "probe_paper"
    prompt_file: str = "configs/paper_main_probe_paper_prompts.txt"
    primary_baseline_id: str = "tree_ring"
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = DEFAULT_MODEL_REVISION
    seed: int = DEFAULT_BASELINE_SEED
    randomization_repeat_id: str = DEFAULT_FORMAL_RANDOMIZATION_REPEAT_ID
    generation_seed_index: int = 0
    generation_seed_offset: int = 0
    watermark_key_index: int = 0
    formal_randomization_protocol_digest: str = field(
        default_factory=lambda: build_paper_run_config(
            "."
        ).formal_randomization_protocol_digest
    )
    watermark_key_seed_random: int = DEFAULT_WATERMARK_KEY_SEED_RANDOM
    target_fpr: float = DEFAULT_TARGET_FPR
    num_inference_steps: int = DEFAULT_INFERENCE_STEPS
    num_inversion_steps: int = DEFAULT_INFERENCE_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    primary_baseline_max_samples: int = 70
    tree_ring_adapter_mode: str = "method_faithful_sd35"
    gaussian_shading_adapter_mode: str = "method_faithful_sd35"
    shallow_diffuse_adapter_mode: str = "method_faithful_sd35"
    tree_ring_attack_families: str = DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES
    gaussian_shading_attack_families: str = DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES
    shallow_diffuse_attack_families: str = DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES
    require_cuda: bool = True
    timeout_seconds: int = 86400
    enable_workflow_progress_bar: bool = True

    def __post_init__(self) -> None:
        """在配置边界集中拒绝非正式或不可比较的运行参数。"""

        resolve_primary_baseline_id(self.primary_baseline_id)
        if not 0.0 < float(self.target_fpr) < 1.0:
            raise ValueError("target_fpr 必须位于 (0, 1)")
        if self.num_inference_steps <= 0 or self.num_inversion_steps <= 0:
            raise ValueError("采样步数和反演步数必须为正整数")
        if self.guidance_scale < 0.0:
            raise ValueError("guidance_scale 不得小于 0")
        if self.primary_baseline_max_samples <= 0:
            raise ValueError("primary_baseline_max_samples 必须为正整数")
        if MODEL_REVISION_PATTERN.fullmatch(self.model_revision) is None:
            raise ValueError("model_revision 必须是40位小写十六进制 Git commit")
        require_registered_model_reference(
            self.model_id,
            self.model_revision,
            required_usage_role="common_backbone_baseline_model",
        )


@dataclass(frozen=True)
class ExternalBaselineMethodFaithfulArchiveRecord:
    """记录单 baseline 结果包及其 Drive 镜像。"""

    archive_path: str
    archive_digest: str
    archive_entry_count: int
    drive_archive_path: str
    drive_archive_digest: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return asdict(self)


def baseline_run_identity_manifest_config(
    config: ExternalBaselineMethodFaithfulConfig,
) -> dict[str, Any]:
    """构造 baseline 顶层 manifest 的完整随机身份配置."""

    return {
        **asdict(config),
        "formal_randomization_plan": (
            formal_runtime_randomization_plan_record(
                config.seed - config.generation_seed_offset,
            )
        ),
    }


def resolve_primary_baseline_id(value: str) -> str:
    """解析唯一 baseline id，并拒绝集合、通配符和 T2SMark 重复入口。"""

    baseline_id = str(value or "").strip()
    if baseline_id not in METHOD_FAITHFUL_BASELINE_IDS:
        raise ValueError(f"不支持的 method-faithful baseline id: {baseline_id}")
    return baseline_id


def validate_formal_run_config(
    root_path: Path,
    config: ExternalBaselineMethodFaithfulConfig,
) -> None:
    """要求 GPU runner 与当前论文层级和共同公平预算完全一致。"""

    paper_run = build_paper_run_config(root_path)
    repeat = resolve_formal_randomization_repeat(
        paper_run.randomization_repeat_id
    )
    root_key_material = os.environ.get(
        "SLM_WM_KEY_MATERIAL",
        "slm_wm_paper_key",
    )
    require_formal_watermark_key_plan(root_key_material)
    expected_watermark_key_seed_random = formal_watermark_key_seed_random(
        root_key_material,
        repeat,
    )
    expected_attacks = set(supported_formal_image_attack_names())
    configured_attacks = {
        name.strip()
        for name in _selected_attack_families(config).split(",")
        if name.strip()
    }
    ready = all(
        (
            config.prompt_set == paper_run.prompt_set,
            Path(config.prompt_file).name == Path(paper_run.prompt_file).name,
            config.primary_baseline_max_samples == paper_run.prompt_count,
            config.model_id == DEFAULT_MODEL_ID,
            config.model_revision == DEFAULT_MODEL_REVISION,
            config.seed
            == DEFAULT_BASELINE_SEED + paper_run.generation_seed_offset,
            config.randomization_repeat_id
            == paper_run.randomization_repeat_id,
            config.generation_seed_index == paper_run.generation_seed_index,
            config.generation_seed_offset
            == paper_run.generation_seed_offset,
            config.watermark_key_index == paper_run.watermark_key_index,
            config.formal_randomization_protocol_digest
            == paper_run.formal_randomization_protocol_digest,
            config.watermark_key_seed_random
            == expected_watermark_key_seed_random,
            math.isclose(config.target_fpr, paper_run.target_fpr, rel_tol=0.0, abs_tol=1e-12),
            config.num_inference_steps == paper_run.inference_steps,
            config.num_inversion_steps == paper_run.inference_steps,
            math.isclose(config.guidance_scale, paper_run.guidance_scale, rel_tol=0.0, abs_tol=1e-12),
            configured_attacks == expected_attacks,
            config.require_cuda is True,
        )
    )
    if not ready:
        raise ValueError("common-backbone baseline 配置未匹配当前正式论文协议与公平预算")


def stable_json_text(value: Any) -> str:
    """以确定性字段顺序序列化 JSON。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, payload: Any) -> None:
    """写出稳定 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json_text(payload), encoding="utf-8")


def read_json(path: Path) -> Any:
    """读取 JSON 文件。"""

    return json.loads(path.read_text(encoding="utf-8-sig"))


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先返回仓库相对路径，仓库外路径保留绝对路径。"""

    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def file_sha256(path: Path) -> str:
    """计算文件字节内容的 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_paths(root_path: Path, config: ExternalBaselineMethodFaithfulConfig) -> dict[str, Path]:
    """构造单 baseline 独占运行路径与共享 transfer 路径。"""

    baseline_id = resolve_primary_baseline_id(config.primary_baseline_id)
    paper_run_name = normalize_paper_run_name(config.prompt_set)
    configured_output_root = (root_path / config.output_dir).resolve()
    expected_output_root = (root_path / DEFAULT_OUTPUT_DIR).resolve()
    if configured_output_root != expected_output_root:
        raise ValueError("method-faithful 输出根目录必须使用正式 outputs family")
    output_dir = expected_output_root / paper_run_name
    run_dir = output_dir / "run_records" / baseline_id
    execution_dir = run_dir / "execution"
    adapter_output_root = run_dir / "adapter_outputs"
    split_dir = output_dir / "split_observations"
    return {
        "output_dir": output_dir,
        "run_dir": run_dir,
        "adapter_output_root": adapter_output_root,
        "execution_output_dir": execution_dir,
        "execution_manifest": execution_dir / "baseline_execution_manifest.json",
        "command_results": execution_dir / "baseline_command_results.json",
        "baseline_observations": execution_dir / "baseline_observations.json",
        "command_plan": run_dir / f"{baseline_id}_baseline_command_plan.json",
        "command_plan_builder_result": run_dir / f"{baseline_id}_command_plan_builder_result.json",
        "command_plan_runner_result": run_dir / f"{baseline_id}_command_plan_runner_result.json",
        "evidence_validation_result": run_dir / f"{baseline_id}_evidence_validation_result.json",
        "primary_prompt_plan": run_dir / f"{baseline_id}_prompt_plan.json",
        "split_observation_dir": split_dir,
        "split_observations": split_dir / f"{baseline_id}_baseline_observations.json",
        "split_command_results": split_dir / f"{baseline_id}_baseline_command_results.json",
        "transfer_manifest": split_dir / f"{baseline_id}_baseline_transfer_manifest.json",
        "progress_events": run_dir / f"{baseline_id}_progress_events.jsonl",
        "environment_report": run_dir / f"{baseline_id}_environment_report.json",
        "numerical_fidelity_report": (
            run_dir / f"{baseline_id}_numerical_fidelity_report.json"
        ),
        "summary": run_dir / f"{baseline_id}_summary.json",
        "manifest": run_dir / f"{baseline_id}_manifest.local.json",
        "package_record_dir": run_dir / "package_records",
    }


def prepare_single_baseline_run_directory(paths: Mapping[str, Path]) -> None:
    """保留原子科学单元, 清理当前 baseline 的可重建派生产物。

    完成单元及其图像位于 ``adapter_outputs``. 该目录必须跨 Colab 会话保留,
    其余 runner 文件均可由完成单元重建. 其他 baseline 的共享 transfer 文件
    不会被删除.
    """

    output_dir = paths["output_dir"].resolve()
    run_dir = paths["run_dir"].resolve()
    try:
        relative_run_dir = run_dir.relative_to(output_dir)
    except ValueError as exc:
        raise ValueError("baseline 运行目录必须位于当前输出根目录内") from exc
    if not relative_run_dir.parts:
        raise ValueError("baseline 运行目录不得等于输出根目录")
    if run_dir.is_symlink():
        raise ValueError("baseline 运行目录不得为符号链接")
    run_dir.mkdir(parents=True, exist_ok=True)
    preserved_adapter_root = paths["adapter_output_root"].resolve()
    for child in tuple(run_dir.iterdir()):
        if child.resolve() == preserved_adapter_root:
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
    if preserved_adapter_root.exists():
        current_baseline_root = preserved_adapter_root / run_dir.name
        for child in tuple(preserved_adapter_root.iterdir()):
            if child.resolve() == current_baseline_root.resolve():
                continue
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        if current_baseline_root.exists():
            for child in tuple(current_baseline_root.iterdir()):
                if child.name == "artifacts" and child.is_dir() and not child.is_symlink():
                    continue
                if child.is_symlink() or child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
    for field_name in ("split_observations", "split_command_results", "transfer_manifest"):
        split_path = paths[field_name].resolve()
        try:
            split_path.relative_to(output_dir)
        except ValueError as exc:
            raise ValueError("baseline transfer 文件必须位于当前输出根目录内") from exc
        if split_path.is_file() or split_path.is_symlink():
            split_path.unlink()


def read_paper_prompt_rows(
    root_path: Path,
    config: ExternalBaselineMethodFaithfulConfig,
) -> list[dict[str, Any]]:
    """读取受治理 Prompt 文件并沿用项目统一 split 分配。"""

    prompt_path = Path(config.prompt_file)
    if not prompt_path.is_absolute():
        prompt_path = (root_path / prompt_path).resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"论文 Prompt 文件不存在: {prompt_path}")
    prompt_texts = read_prompt_file(prompt_path)
    requested_count = int(config.primary_baseline_max_samples)
    if requested_count > len(prompt_texts):
        raise ValueError(
            f"请求的 baseline 样本数 {requested_count} 超过 Prompt 文件实际数量 {len(prompt_texts)}"
        )
    records = apply_split_assignments(build_prompt_records(config.prompt_set, prompt_texts))
    selected_records = records[:requested_count]
    root_key_material = os.environ.get(
        "SLM_WM_KEY_MATERIAL",
        "slm_wm_paper_key",
    )
    repeat = resolve_formal_randomization_repeat(
        config.randomization_repeat_id
    )
    return [
        {
            "prompt_id": record.prompt_id,
            "prompt_index": record.prompt_index,
            "prompt_set": record.prompt_set,
            "split": record.split,
            "prompt_text": record.prompt_text,
            "prompt_digest": record.prompt_digest,
            **build_formal_randomization_identity(
                base_seed=DEFAULT_BASELINE_SEED,
                prompt_index=record.prompt_index,
                root_key_material=root_key_material,
                repeat=repeat,
            ),
        }
        for record in selected_records
    ]


def write_primary_baseline_prompt_plan(
    root_path: Path,
    paths: dict[str, Path],
    config: ExternalBaselineMethodFaithfulConfig,
) -> Path:
    """写出当前 baseline 的完整受治理 Prompt 计划。"""

    write_json(paths["primary_prompt_plan"], read_paper_prompt_rows(root_path, config))
    return paths["primary_prompt_plan"]


def ensure_cuda_if_requested(require_cuda: bool) -> dict[str, Any]:
    """在正式 common-backbone 运行前核验 CUDA。"""

    try:
        import torch
    except Exception as error:  # pragma: no cover - 本地轻量测试不依赖 torch
        if require_cuda:
            raise RuntimeError("正式 baseline 运行要求可导入 PyTorch") from error
        return {"torch_available": False, "cuda_available": False, "device": "cpu"}
    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise RuntimeError("正式 baseline 运行要求 CUDA 可用")
    return {
        "torch_available": True,
        "cuda_available": cuda_available,
        "device": "cuda" if cuda_available else "cpu",
        "torch_version": str(torch.__version__),
    }


def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    progress: object | None = None,
    progress_profile: str = "",
    child_progress_path: str | Path | None = None,
) -> dict[str, Any]:
    """执行显式 argv 命令并保留标准输出与错误输出。"""

    command_env = None
    if child_progress_path is not None:
        command_env = dict(os.environ)
        command_env[PROGRESS_EVENT_ENV_NAME] = str(child_progress_path)
    completed = run_quiet_subprocess_with_progress(
        command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        progress=progress,
        progress_profile=progress_profile or "operation=argv_command",
        env=command_env,
        heartbeat_seconds=15.0,
        child_progress_path=child_progress_path,
    )
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_command_with_progress_status(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    progress: object | None = None,
    progress_profile: str = "",
    child_progress_path: str | Path | None = None,
) -> dict[str, Any]:
    """兼容真实进度 runner 与轻量测试替身。"""

    if child_progress_path is None:
        return call_runner_with_progress_status(
            run_command,
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            progress=progress,
            progress_profile=progress_profile,
        )
    try:
        return run_command(
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            progress=progress,
            progress_profile=progress_profile,
            child_progress_path=child_progress_path,
        )
    except TypeError:
        return run_command(command, cwd=cwd, timeout_seconds=timeout_seconds)


def _selected_attack_families(config: ExternalBaselineMethodFaithfulConfig) -> str:
    """返回当前 baseline 的共同攻击矩阵配置。"""

    baseline_id = resolve_primary_baseline_id(config.primary_baseline_id)
    return str(getattr(config, f"{baseline_id}_attack_families"))


def _selected_adapter_mode(config: ExternalBaselineMethodFaithfulConfig) -> str:
    """返回当前 baseline 的 SD3.5 adapter 模式。"""

    baseline_id = resolve_primary_baseline_id(config.primary_baseline_id)
    return str(getattr(config, f"{baseline_id}_adapter_mode"))


def _build_command_plan_command(
    root_path: Path,
    config: ExternalBaselineMethodFaithfulConfig,
    paths: dict[str, Path],
    prompt_plan_path: Path,
) -> list[str]:
    """构造单 baseline 命令计划生成命令。"""

    baseline_id = resolve_primary_baseline_id(config.primary_baseline_id)
    command = [
        sys.executable,
        "-m",
        "paper_experiments.baselines.command_plan_builder",
        "--root",
        str(root_path),
        "--methods",
        baseline_id,
        "--out",
        str(paths["command_plan"]),
        "--output-root",
        str(paths["adapter_output_root"]),
        "--prompt-plan",
        str(prompt_plan_path),
        "--target-fpr",
        str(config.target_fpr),
        "--timeout-seconds",
        str(config.timeout_seconds),
        "--model-id",
        config.model_id,
        "--model-revision",
        config.model_revision,
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
        "--tree-ring-watermark-seed",
        str(config.watermark_key_seed_random),
        "--gaussian-shading-watermark-seed",
        str(config.watermark_key_seed_random),
        "--shallow-diffuse-watermark-seed",
        str(config.watermark_key_seed_random),
        "--max-samples",
        str(config.primary_baseline_max_samples),
        f"--{baseline_id.replace('_', '-')}-adapter-mode",
        _selected_adapter_mode(config),
    ]
    attack_families = _selected_attack_families(config).strip()
    if attack_families:
        command.extend(
            [f"--{baseline_id.replace('_', '-')}-attack-families", attack_families]
        )
    if config.require_cuda:
        command.append("--require-cuda")
    return command


def _adapter_manifest_path(config: ExternalBaselineMethodFaithfulConfig, paths: dict[str, Path]) -> Path:
    """返回当前 adapter 产生的 method-faithful manifest 路径。"""

    baseline_id = resolve_primary_baseline_id(config.primary_baseline_id)
    return (
        paths["adapter_output_root"]
        / baseline_id
        / f"{baseline_id}_method_faithful_sd35_adapter_manifest.json"
    )


def write_baseline_transfer_files(
    root_path: Path,
    config: ExternalBaselineMethodFaithfulConfig,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """验证真实执行计数并写出跨结果包唯一交换面。"""

    baseline_id = resolve_primary_baseline_id(config.primary_baseline_id)
    observations = load_baseline_observation_rows(paths["baseline_observations"])
    command_results = read_json(paths["command_results"])
    execution_manifest = read_json(paths["execution_manifest"])
    if not isinstance(command_results, list) or len(command_results) != 1:
        raise ValueError("单 baseline 运行必须且只能产生一条 command result")
    command_result = dict(command_results[0])
    if str(command_result.get("baseline_id", "")) != baseline_id:
        raise ValueError("command result 的 baseline_id 与当前运行不一致")
    if int(command_result.get("return_code", 1)) != 0:
        raise RuntimeError("baseline adapter 命令未成功完成")
    if not observations:
        raise ValueError("baseline observation 不得为空")
    if any(str(row.get("baseline_id", "")) != baseline_id for row in observations):
        raise ValueError("observation 中存在其他 baseline 记录")
    if any(
        str(row.get("generation_model_id", "")) != config.model_id
        or str(row.get("generation_model_revision", "")) != config.model_revision
        for row in observations
    ):
        raise ValueError("observation 未绑定当前正式模型 id 与 revision")
    event_ids = [str(row.get("event_id", "")).strip() for row in observations]
    if any(not event_id for event_id in event_ids) or len(event_ids) != len(set(event_ids)):
        raise ValueError("observation event_id 必须非空且唯一")
    actual_count = len(observations)
    declared_counts = {
        int(command_result.get("observation_count", -1)),
        int(execution_manifest.get("observation_count", -1)),
    }
    adapter_manifest_path = _adapter_manifest_path(config, paths)
    if not adapter_manifest_path.is_file():
        raise FileNotFoundError(f"adapter manifest 不存在: {adapter_manifest_path}")
    adapter_manifest = read_json(adapter_manifest_path)
    if str(adapter_manifest.get("baseline_id", "")) != baseline_id:
        raise ValueError("adapter manifest 的 baseline_id 与当前运行不一致")
    adapter_generation = adapter_manifest.get("generation_protocol")
    if not isinstance(adapter_generation, Mapping) or not all(
        (
            str(adapter_manifest.get("model_id", "")) == config.model_id,
            str(adapter_manifest.get("model_revision", "")) == config.model_revision,
            str(adapter_generation.get("model_id", "")) == config.model_id,
            str(adapter_generation.get("model_revision", "")) == config.model_revision,
        )
    ):
        raise ValueError("adapter manifest 未绑定当前正式模型 id 与 revision")
    unit_evidence = validate_method_faithful_adapter_unit_evidence(
        manifest=adapter_manifest,
        observation_rows=observations,
        root=root_path,
    )
    if unit_evidence["method_faithful_scientific_unit_resume_ready"] is not True:
        raise RuntimeError("adapter 原子科学完成单元证据未闭合")
    numerical_fidelity_report = read_json(paths["numerical_fidelity_report"])
    validated_fidelity = validate_method_faithful_numerical_fidelity_report(
        numerical_fidelity_report,
        expected_baseline_id=baseline_id,
    )
    declared_counts.add(int(adapter_manifest.get("observation_count", -1)))
    if declared_counts != {actual_count}:
        raise ValueError(
            f"baseline observation 声明计数与实际计数不一致: {sorted(declared_counts)} != {actual_count}"
        )

    prompt_rows = read_json(paths["primary_prompt_plan"])
    expected_calibration_count = sum(
        str(row.get("split", "")) == "calibration" for row in prompt_rows
    )
    threshold_audit = audit_fixed_fpr_observation_threshold(
        observations,
        target_fpr=config.target_fpr,
        expected_calibration_negative_count=expected_calibration_count,
    )
    if not threshold_audit.fixed_fpr_ready:
        raise ValueError("baseline observation 未通过 fixed-FPR 阈值冻结审计")

    configured_attacks = {
        name.strip() for name in _selected_attack_families(config).split(",") if name.strip()
    }
    observed_attacks = {
        str(row.get("attack_name", ""))
        for row in observations
        if str(row.get("attack_family", "")) != "clean"
    }
    if observed_attacks != configured_attacks:
        raise ValueError(
            "baseline observation 攻击集合与正式配置不一致: "
            f"observed={sorted(observed_attacks)} configured={sorted(configured_attacks)}"
        )
    validate_formal_attack_observation_identities(
        observations,
        baseline_id=baseline_id,
    )

    write_json(paths["split_observations"], observations)
    write_json(paths["split_command_results"], command_results)
    collection_root = paths["output_dir"]
    observation_relative_path = paths["split_observations"].relative_to(collection_root).as_posix()
    command_result_relative_path = paths["split_command_results"].relative_to(collection_root).as_posix()
    prompt_plan_relative_path = paths["primary_prompt_plan"].relative_to(collection_root).as_posix()
    transfer_manifest = {
        "artifact_name": f"{baseline_id}_baseline_transfer_manifest.json",
        "baseline_id": baseline_id,
        "baseline_observations_path": observation_relative_path,
        "baseline_observation_count": actual_count,
        "baseline_observations_sha256": file_sha256(paths["split_observations"]),
        "baseline_command_results_path": command_result_relative_path,
        "baseline_command_results_sha256": file_sha256(paths["split_command_results"]),
        "prompt_plan_path": prompt_plan_relative_path,
        "prompt_plan_sha256": file_sha256(paths["primary_prompt_plan"]),
        "prompt_protocol_digest": canonical_prompt_protocol_digest(prompt_rows),
        "adapter_manifest_path": adapter_manifest_path.relative_to(collection_root).as_posix(),
        "adapter_manifest_sha256": file_sha256(adapter_manifest_path),
        "numerical_fidelity_report_path": paths[
            "numerical_fidelity_report"
        ].relative_to(collection_root).as_posix(),
        "numerical_fidelity_report_sha256": file_sha256(
            paths["numerical_fidelity_report"]
        ),
        "numerical_fidelity_report_digest": validated_fidelity[
            "numerical_fidelity_report_digest"
        ],
        "numerical_fidelity_reference_mode": validated_fidelity[
            "numerical_fidelity_reference_mode"
        ],
        "method_faithful_numerical_fidelity_ready": True,
        **unit_evidence,
        "execution_manifest_path": paths["execution_manifest"].relative_to(collection_root).as_posix(),
        "execution_manifest_sha256": file_sha256(paths["execution_manifest"]),
        "paper_run_name": config.prompt_set,
        "prompt_set": config.prompt_set,
        "prompt_count": len(prompt_rows),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "target_fpr": float(config.target_fpr),
        "threshold": threshold_audit.frozen_threshold,
        "threshold_digest": threshold_audit.threshold_digest,
        "generation_protocol": {
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "num_inference_steps": int(config.num_inference_steps),
            "guidance_scale": float(config.guidance_scale),
            "height": 512,
            "width": 512,
        },
        "detection_protocol": {
            "input_access_mode": "image_only",
            "num_inversion_steps": int(config.num_inversion_steps),
        },
        "formal_attack_names": sorted(configured_attacks),
        "code_version": repository_environment.require_published_formal_execution_lock(
            root_path
        )["formal_execution_commit"],
        "transfer_ready": True,
    }
    write_json(paths["transfer_manifest"], transfer_manifest)
    return transfer_manifest


def build_and_run_primary_baseline_adapter(
    root_path: Path,
    config: ExternalBaselineMethodFaithfulConfig,
    paths: dict[str, Path],
    progress: object | None = None,
) -> dict[str, Any]:
    """生成并执行单 baseline adapter 命令计划。"""

    prompt_plan_path = write_primary_baseline_prompt_plan(root_path, paths, config)
    build_command = _build_command_plan_command(root_path, config, paths, prompt_plan_path)
    build_result = run_command_with_progress_status(
        build_command,
        cwd=root_path,
        timeout_seconds=300,
        progress=progress,
        progress_profile="operation=build_single_baseline_command_plan",
    )
    write_json(paths["command_plan_builder_result"], build_result)
    if int(build_result["return_code"]) != 0:
        raise RuntimeError("baseline command plan 生成失败")

    execution_command = [
        sys.executable,
        "-m",
        "paper_experiments.baselines.command_plan_execution",
        "--plan",
        str(paths["command_plan"]),
        "--out",
        str(paths["execution_output_dir"]),
        "--require-pass",
    ]
    execution_result = run_command_with_progress_status(
        execution_command,
        cwd=root_path,
        timeout_seconds=config.timeout_seconds,
        progress=progress,
        progress_profile="operation=run_single_baseline_adapter",
        child_progress_path=paths["progress_events"],
    )
    write_json(paths["command_plan_runner_result"], execution_result)
    if int(execution_result["return_code"]) != 0:
        raise RuntimeError("baseline adapter 执行失败")

    validation_command = [
        sys.executable,
        "-m",
        "paper_experiments.baselines.evidence_validation_cli",
        "--baseline-execution-manifest",
        str(paths["execution_manifest"]),
        "--require-pass",
    ]
    validation_result = run_command_with_progress_status(
        validation_command,
        cwd=root_path,
        timeout_seconds=300,
        progress=progress,
        progress_profile="operation=validate_single_baseline_evidence",
    )
    write_json(paths["evidence_validation_result"], validation_result)
    if int(validation_result["return_code"]) != 0:
        raise RuntimeError("baseline 执行证据校验失败")

    numerical_fidelity_report = (
        build_method_faithful_numerical_fidelity_report(
            root_path,
            config.primary_baseline_id,
        )
    )
    write_json(paths["numerical_fidelity_report"], numerical_fidelity_report)
    if (
        numerical_fidelity_report[
            "method_faithful_numerical_fidelity_ready"
        ]
        is not True
    ):
        raise RuntimeError("baseline 关键算子数值忠实度未通过")

    transfer_manifest = write_baseline_transfer_files(root_path, config, paths)
    unit_evidence = {
        key: value
        for key, value in transfer_manifest.items()
        if key.startswith("method_faithful_") or key.startswith("scientific_")
    }
    return {
        "adapter_execution_ready": True,
        "primary_baseline_adapter_ready": True,
        "primary_baseline_id": config.primary_baseline_id,
        "primary_baseline_observation_count": int(
            transfer_manifest["baseline_observation_count"]
        ),
        "transfer_manifest_path": relative_or_absolute(paths["transfer_manifest"], root_path),
        "threshold_digest": str(transfer_manifest["threshold_digest"]),
        "numerical_fidelity_report_digest": str(
            transfer_manifest["numerical_fidelity_report_digest"]
        ),
        "numerical_fidelity_reference_mode": str(
            transfer_manifest["numerical_fidelity_reference_mode"]
        ),
        "method_faithful_numerical_fidelity_ready": True,
        "unit_evidence": unit_evidence,
    }


def write_failure_outputs(
    root_path: Path,
    config: ExternalBaselineMethodFaithfulConfig,
    paths: dict[str, Path],
    error: Exception,
) -> dict[str, Any]:
    """写出失败诊断，但不生成可被正式闭合选择的结果包。"""

    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    environment_report = build_runtime_environment_report("sd35_method_runtime_gpu")
    write_json(paths["environment_report"], environment_report)
    summary = {
        "run_decision": "fail",
        "external_baseline_method_faithful_ready": False,
        "primary_baseline_id": config.primary_baseline_id,
        "supports_paper_claim": False,
        "unsupported_reason": f"{type(error).__name__}:{error}",
        "environment_report_path": relative_or_absolute(paths["environment_report"], root_path),
    }
    write_json(paths["summary"], summary)
    manifest = build_artifact_manifest(
        artifact_id=f"{config.primary_baseline_id}_method_faithful_manifest",
        artifact_type="local_manifest",
        input_paths=(),
        output_paths=(
            relative_or_absolute(paths["summary"], root_path),
            relative_or_absolute(paths["environment_report"], root_path),
        ),
        config=baseline_run_identity_manifest_config(config),
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 paper_experiments.runners.external_baseline_method_faithful",
        metadata={"run_decision": "fail", "supports_paper_claim": False},
    ).to_dict()
    write_json(paths["manifest"], manifest)
    return summary


def write_external_baseline_method_faithful_outputs(
    config: ExternalBaselineMethodFaithfulConfig,
    root: str | Path = ".",
) -> dict[str, Any]:
    """执行单 baseline 全路径并写出可验证 transfer 产物。"""

    root_path = Path(root).resolve()
    formal_execution_run_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    paths = output_paths(root_path, config)
    validate_formal_run_config(root_path, config)
    prepare_single_baseline_run_directory(paths)
    try:
        with progress_bar(
            4,
            desc=f"method-faithful {config.primary_baseline_id}",
            enabled=config.enable_workflow_progress_bar,
        ) as run_progress:
            device_report = ensure_cuda_if_requested(config.require_cuda)
            update_progress(run_progress, profile="operation=ensure_cuda")
            environment_report = build_runtime_environment_report(
                "sd35_method_runtime_gpu",
                verified_formal_execution_lock=formal_execution_run_lock,
            )
            environment_report["external_baseline_device_report"] = device_report
            write_json(paths["environment_report"], environment_report)
            if environment_report["dependency_environment_ready"] is not True:
                blockers = ",".join(
                    environment_report["dependency_readiness_blockers"]
                )
                raise RuntimeError(
                    f"dependency_profile_environment_not_ready:{blockers}"
                )
            update_progress(run_progress, profile="operation=write_environment_report")
            adapter_report = build_and_run_primary_baseline_adapter(
                root_path,
                config,
                paths,
                progress=run_progress,
            )
            update_progress(run_progress, profile="operation=run_baseline_adapter")
    except Exception as error:
        return write_failure_outputs(root_path, config, paths, error)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_run_name": config.prompt_set,
        "run_decision": "pass",
        "external_baseline_method_faithful_ready": True,
        "primary_baseline_id": config.primary_baseline_id,
        "primary_baseline_adapter_ready": True,
        "primary_baseline_observation_count": adapter_report[
            "primary_baseline_observation_count"
        ],
        "target_fpr": float(config.target_fpr),
        "generation_protocol": {
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "num_inference_steps": int(config.num_inference_steps),
            "guidance_scale": float(config.guidance_scale),
        },
        "detection_protocol": {
            "input_access_mode": "image_only",
            "num_inversion_steps": int(config.num_inversion_steps),
        },
        "transfer_manifest_path": adapter_report["transfer_manifest_path"],
        "threshold_digest": adapter_report["threshold_digest"],
        "numerical_fidelity_report_path": relative_or_absolute(
            paths["numerical_fidelity_report"],
            root_path,
        ),
        "numerical_fidelity_report_digest": adapter_report[
            "numerical_fidelity_report_digest"
        ],
        "numerical_fidelity_reference_mode": adapter_report[
            "numerical_fidelity_reference_mode"
        ],
        "method_faithful_numerical_fidelity_ready": adapter_report[
            "method_faithful_numerical_fidelity_ready"
        ],
        "environment_report_path": relative_or_absolute(paths["environment_report"], root_path),
        "dependency_profile_id": environment_report["dependency_profile_id"],
        "dependency_profile_digest": environment_report[
            "dependency_profile_digest"
        ],
        "dependency_lock_digest": environment_report[
            "complete_hash_lock_digest"
        ],
        "dependency_environment_ready": environment_report[
            "dependency_environment_ready"
        ],
        **adapter_report["unit_evidence"],
        "supports_paper_claim": False,
        "unsupported_reason": "formal_import_and_comparison_required",
    }
    write_json(paths["summary"], summary)
    formal_execution_run_lock = repository_environment.validate_formal_execution_lock_pair(
        formal_execution_run_lock,
        repository_environment.require_published_formal_execution_lock(root_path),
        formal_execution_run_lock["formal_execution_commit"],
    )
    manifest = build_artifact_manifest(
        artifact_id=f"{config.primary_baseline_id}_method_faithful_manifest",
        artifact_type="local_manifest",
        input_paths=(relative_or_absolute(paths["primary_prompt_plan"], root_path),),
        output_paths=(
            relative_or_absolute(paths["summary"], root_path),
            relative_or_absolute(paths["environment_report"], root_path),
            relative_or_absolute(paths["transfer_manifest"], root_path),
            relative_or_absolute(paths["split_observations"], root_path),
            relative_or_absolute(paths["split_command_results"], root_path),
            relative_or_absolute(paths["numerical_fidelity_report"], root_path),
        ),
        config=baseline_run_identity_manifest_config(config),
        code_version=formal_execution_run_lock["formal_execution_commit"],
        rebuild_command="调用 paper_experiments.runners.external_baseline_method_faithful",
        metadata={
            "run_decision": "pass",
            "primary_baseline_id": config.primary_baseline_id,
            "primary_baseline_observation_count": adapter_report[
                "primary_baseline_observation_count"
            ],
            "method_faithful_scientific_unit_resume_ready": adapter_report[
                "unit_evidence"
            ]["method_faithful_scientific_unit_resume_ready"],
            "method_faithful_scientific_unit_records_digest": adapter_report[
                "unit_evidence"
            ]["method_faithful_scientific_unit_records_digest"],
            "numerical_fidelity_report_digest": adapter_report[
                "numerical_fidelity_report_digest"
            ],
            "numerical_fidelity_reference_mode": adapter_report[
                "numerical_fidelity_reference_mode"
            ],
            "method_faithful_numerical_fidelity_ready": adapter_report[
                "method_faithful_numerical_fidelity_ready"
            ],
            "supports_paper_claim": False,
        },
    ).to_dict()
    manifest["formal_execution_run_lock"] = formal_execution_run_lock
    write_json(paths["manifest"], manifest)
    return summary


def build_default_config() -> ExternalBaselineMethodFaithfulConfig:
    """从当前论文运行层级与单 baseline 环境变量构造配置。"""

    paper_run = build_paper_run_config(".")
    repeat = resolve_formal_randomization_repeat(
        paper_run.randomization_repeat_id
    )
    root_key_material = os.environ.get(
        "SLM_WM_KEY_MATERIAL",
        "slm_wm_paper_key",
    )
    require_formal_watermark_key_plan(root_key_material)
    return ExternalBaselineMethodFaithfulConfig(
        output_dir=os.environ.get("SLM_WM_EXTERNAL_BASELINE_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        drive_output_dir=os.environ.get(
            "SLM_WM_EXTERNAL_BASELINE_DRIVE_OUTPUT_DIR",
            paper_run.drive_dir("external_baseline_method_faithful"),
        ),
        prompt_set=os.environ.get("SLM_WM_PROMPT_SET", paper_run.prompt_set),
        prompt_file=os.environ.get("SLM_WM_PROMPT_FILE", paper_run.prompt_file),
        primary_baseline_id=os.environ.get("SLM_WM_PRIMARY_BASELINE_ID", ""),
        model_id=os.environ.get("SLM_WM_EXTERNAL_BASELINE_MODEL_ID", DEFAULT_MODEL_ID),
        model_revision=os.environ.get(
            "SLM_WM_EXTERNAL_BASELINE_MODEL_REVISION",
            DEFAULT_MODEL_REVISION,
        ),
        seed=int(
            os.environ.get(
                "SLM_WM_EXTERNAL_BASELINE_SEED",
                str(DEFAULT_BASELINE_SEED + repeat.generation_seed_offset),
            )
        ),
        randomization_repeat_id=repeat.randomization_repeat_id,
        generation_seed_index=repeat.generation_seed_index,
        generation_seed_offset=repeat.generation_seed_offset,
        watermark_key_index=repeat.watermark_key_index,
        formal_randomization_protocol_digest=(
            paper_run.formal_randomization_protocol_digest
        ),
        watermark_key_seed_random=formal_watermark_key_seed_random(
            root_key_material,
            repeat,
        ),
        target_fpr=float(
            os.environ.get("SLM_WM_EXTERNAL_BASELINE_TARGET_FPR", str(paper_run.target_fpr))
        ),
        num_inference_steps=int(
            os.environ.get("SLM_WM_EXTERNAL_BASELINE_NUM_INFERENCE_STEPS", str(paper_run.inference_steps))
        ),
        num_inversion_steps=int(
            os.environ.get("SLM_WM_EXTERNAL_BASELINE_NUM_INVERSION_STEPS", str(paper_run.inference_steps))
        ),
        guidance_scale=float(
            os.environ.get("SLM_WM_EXTERNAL_BASELINE_GUIDANCE_SCALE", str(paper_run.guidance_scale))
        ),
        primary_baseline_max_samples=resolve_count_from_environment(
            "SLM_WM_PRIMARY_BASELINE_MAX_SAMPLES",
            default_value=paper_run.sample_count,
        ),
        tree_ring_adapter_mode=os.environ.get("SLM_WM_TREE_RING_ADAPTER_MODE", "method_faithful_sd35"),
        gaussian_shading_adapter_mode=os.environ.get(
            "SLM_WM_GAUSSIAN_SHADING_ADAPTER_MODE", "method_faithful_sd35"
        ),
        shallow_diffuse_adapter_mode=os.environ.get(
            "SLM_WM_SHALLOW_DIFFUSE_ADAPTER_MODE", "method_faithful_sd35"
        ),
        tree_ring_attack_families=os.environ.get(
            "SLM_WM_TREE_RING_ATTACK_FAMILIES", DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES
        ),
        gaussian_shading_attack_families=os.environ.get(
            "SLM_WM_GAUSSIAN_SHADING_ATTACK_FAMILIES", DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES
        ),
        shallow_diffuse_attack_families=os.environ.get(
            "SLM_WM_SHALLOW_DIFFUSE_ATTACK_FAMILIES", DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES
        ),
        require_cuda=os.environ.get("SLM_WM_EXTERNAL_BASELINE_REQUIRE_CUDA", "1") != "0",
        timeout_seconds=int(os.environ.get("SLM_WM_EXTERNAL_BASELINE_TIMEOUT_SECONDS", "86400")),
        enable_workflow_progress_bar=os.environ.get("SLM_WM_ENABLE_WORKFLOW_PROGRESS_BAR", "1") != "0",
    )


def run_default_external_baseline_method_faithful_plan(root: str | Path = ".") -> dict[str, Any]:
    """运行当前环境指定的单 baseline 计划。"""

    return write_external_baseline_method_faithful_outputs(config=build_default_config(), root=root)


def _resolve_repository_member(
    root_path: Path,
    relative_value: Any,
    *,
    field_name: str,
    require_file: bool = True,
) -> Path:
    """解析仓库相对 POSIX 路径, 并拒绝逃逸和最终符号链接."""

    relative_text = str(relative_value or "")
    pure_path = PurePosixPath(relative_text)
    if (
        not relative_text
        or "\\" in relative_text
        or pure_path.is_absolute()
        or any(part in {"", ".", ".."} for part in pure_path.parts)
        or pure_path.as_posix() != relative_text
    ):
        raise RuntimeError(f"method-faithful 的 {field_name} 不是安全仓库相对路径")
    resolved_root = root_path.resolve()
    unresolved = root_path / relative_text
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"method-faithful 的 {field_name} 逃逸仓库根目录") from exc
    if unresolved.is_symlink():
        raise RuntimeError(f"method-faithful 的 {field_name} 不得为符号链接")
    if require_file and not resolved.is_file():
        raise FileNotFoundError(f"method-faithful 的 {field_name} 文件不存在: {relative_text}")
    if not require_file and not resolved.is_dir():
        raise FileNotFoundError(f"method-faithful 的 {field_name} 目录不存在: {relative_text}")
    return resolved


def _collect_adapter_unit_files(
    *,
    root_path: Path,
    source_dir: Path,
    run_dir: Path,
    baseline_id: str,
    transfer_manifest: Mapping[str, Any],
) -> set[Path]:
    """从 adapter manifest 和完成单元引用构造精确归档文件集合."""

    adapter_manifest_path = _resolve_transfer_member(
        source_dir,
        transfer_manifest,
        path_field="adapter_manifest_path",
        digest_field="adapter_manifest_sha256",
    )
    expected_adapter_root = (run_dir / "adapter_outputs" / baseline_id).resolve()
    expected_manifest_path = (
        expected_adapter_root
        / f"{baseline_id}_method_faithful_sd35_adapter_manifest.json"
    )
    if adapter_manifest_path != expected_manifest_path:
        raise RuntimeError("baseline transfer 未绑定当前 baseline 的唯一 adapter manifest")
    adapter_manifest = read_json(adapter_manifest_path)
    artifact_root = _resolve_repository_member(
        root_path,
        adapter_manifest.get("artifact_root"),
        field_name="artifact_root",
        require_file=False,
    )
    expected_artifact_root = expected_adapter_root / "artifacts"
    if artifact_root != expected_artifact_root:
        raise RuntimeError("adapter manifest 的 artifact_root 未绑定当前 baseline")

    prompt_plan_path = _resolve_repository_member(
        root_path,
        adapter_manifest.get("prompt_plan_path"),
        field_name="prompt_plan_path",
    )
    transfer_prompt_path = _resolve_transfer_member(
        source_dir,
        transfer_manifest,
        path_field="prompt_plan_path",
        digest_field="prompt_plan_sha256",
    )
    if prompt_plan_path != transfer_prompt_path:
        raise RuntimeError("adapter manifest 与 transfer 未绑定同一 Prompt 计划")

    adapter_files = {adapter_manifest_path}
    for field_name in (
        "baseline_observations_path",
        "image_pairs_path",
        "attacked_image_manifest_path",
        "method_faithful_run_identity_path",
    ):
        adapter_files.add(
            _resolve_repository_member(
                root_path,
                adapter_manifest.get(field_name),
                field_name=field_name,
            )
        )
    record_paths = adapter_manifest.get("method_faithful_scientific_unit_record_paths")
    if not isinstance(record_paths, list) or not record_paths:
        raise RuntimeError("adapter manifest 缺少方法忠实完成单元路径")
    for index, relative_path in enumerate(record_paths):
        record_path = _resolve_repository_member(
            root_path,
            relative_path,
            field_name=f"method_faithful_scientific_unit_record_paths[{index}]",
        )
        try:
            record_path.relative_to(artifact_root)
        except ValueError as exc:
            raise RuntimeError("方法忠实完成单元记录未位于当前 artifact_root") from exc
        adapter_files.add(record_path)
        record = read_json(record_path)
        artifacts = record.get("unit_artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise RuntimeError("方法忠实完成单元缺少真实图像产物")
        for artifact_index, artifact in enumerate(artifacts):
            if not isinstance(artifact, Mapping):
                raise RuntimeError("方法忠实完成单元产物记录必须为 object")
            artifact_path = _resolve_repository_member(
                root_path,
                artifact.get("artifact_path"),
                field_name=f"unit_artifacts[{artifact_index}].artifact_path",
            )
            try:
                artifact_path.relative_to(artifact_root)
            except ValueError as exc:
                raise RuntimeError("方法忠实完成单元图像未位于当前 artifact_root") from exc
            adapter_files.add(artifact_path)

    for path in adapter_files:
        try:
            path.relative_to(expected_adapter_root)
        except ValueError as exc:
            raise RuntimeError("adapter 归档成员逃逸当前 baseline 独占目录") from exc
    adapter_observations_path = _resolve_repository_member(
        root_path,
        adapter_manifest.get("baseline_observations_path"),
        field_name="baseline_observations_path",
    )
    split_observations_path = _resolve_transfer_member(
        source_dir,
        transfer_manifest,
        path_field="baseline_observations_path",
        digest_field="baseline_observations_sha256",
    )
    if read_json(adapter_observations_path) != read_json(split_observations_path):
        raise RuntimeError("adapter 原始 observation 与 transfer observation 不一致")
    return adapter_files


def _inventory_regular_tree(root_dir: Path) -> tuple[set[Path], set[Path]]:
    """递归盘点普通文件和目录, 遇到符号链接或特殊节点立即闭锁."""

    files: set[Path] = set()
    directories: set[Path] = set()
    pending = [root_dir]
    while pending:
        current = pending.pop()
        for child in current.iterdir():
            if child.is_symlink():
                raise RuntimeError(f"method-faithful 运行目录包含符号链接: {child.name}")
            if child.is_file():
                files.add(child.resolve())
            elif child.is_dir():
                resolved_child = child.resolve()
                directories.add(resolved_child)
                pending.append(resolved_child)
            else:
                raise RuntimeError(f"method-faithful 运行目录包含特殊文件: {child.name}")
    return files, directories


def collect_package_entries(
    root_path: Path,
    output_dir: Path,
    archive_path: Path,
    baseline_id: str,
) -> tuple[Path, ...]:
    """按事实引用白名单收集当前 baseline 独占产物."""

    resolved_baseline_id = resolve_primary_baseline_id(baseline_id)
    run_dir = output_dir / "run_records" / resolved_baseline_id
    split_dir = output_dir / "split_observations"
    required_scientific_entries = (
        run_dir / "scientific_execution" / "scientific_workflow_result_envelope.json",
        run_dir / "isolated_scientific_execution_report.json",
        run_dir / "isolated_dependency_environment_report.json",
        run_dir / "scientific_command_dispatch_report.json",
        run_dir / "scientific_execution_binding.json",
    )
    transient_source_report = (
        run_dir
        / "scientific_execution"
        / "source_isolated_scientific_execution_report.json"
    )
    if transient_source_report.exists():
        raise RuntimeError("method-faithful 隔离科学执行证据尚未完成本地绑定")
    missing_scientific_entries = tuple(
        path for path in required_scientific_entries if not path.is_file()
    )
    if missing_scientific_entries:
        raise FileNotFoundError(
            "method-faithful 打包缺少隔离科学执行证据: "
            + ",".join(path.name for path in missing_scientific_entries)
        )
    validate_scientific_execution_binding(
        run_dir / "scientific_execution_binding.json",
        expected_artifact_role="external_baseline_method_faithful",
        expected_paper_run_name=output_dir.name,
        repository_root=root_path,
    )

    transfer_manifest_path = (
        split_dir / f"{resolved_baseline_id}_baseline_transfer_manifest.json"
    )
    transfer_manifest = read_json(transfer_manifest_path)
    transfer_entries = {
        transfer_manifest_path.resolve(),
        _resolve_transfer_member(
            output_dir,
            transfer_manifest,
            path_field="baseline_observations_path",
            digest_field="baseline_observations_sha256",
        ),
        _resolve_transfer_member(
            output_dir,
            transfer_manifest,
            path_field="baseline_command_results_path",
            digest_field="baseline_command_results_sha256",
        ),
        _resolve_transfer_member(
            output_dir,
            transfer_manifest,
            path_field="numerical_fidelity_report_path",
            digest_field="numerical_fidelity_report_sha256",
        ),
    }
    required_runner_entries = {
        run_dir / f"{resolved_baseline_id}_baseline_command_plan.json",
        run_dir / f"{resolved_baseline_id}_command_plan_builder_result.json",
        run_dir / f"{resolved_baseline_id}_command_plan_runner_result.json",
        run_dir / f"{resolved_baseline_id}_evidence_validation_result.json",
        run_dir / f"{resolved_baseline_id}_prompt_plan.json",
        run_dir / f"{resolved_baseline_id}_progress_events.jsonl",
        run_dir / f"{resolved_baseline_id}_environment_report.json",
        run_dir / f"{resolved_baseline_id}_numerical_fidelity_report.json",
        run_dir / f"{resolved_baseline_id}_summary.json",
        run_dir / f"{resolved_baseline_id}_manifest.local.json",
        run_dir / "execution" / "baseline_execution_manifest.json",
        run_dir / "execution" / "baseline_command_results.json",
        run_dir / "execution" / "baseline_observations.json",
        run_dir / "execution" / "baseline_command_plan_manifest.json",
        *required_scientific_entries,
    }
    missing_runner_entries = sorted(
        path for path in required_runner_entries if not path.is_file() or path.is_symlink()
    )
    if missing_runner_entries:
        raise FileNotFoundError(
            "method-faithful 打包缺少正式运行成员: "
            + ",".join(path.name for path in missing_runner_entries)
        )
    adapter_entries = _collect_adapter_unit_files(
        root_path=root_path,
        source_dir=output_dir,
        run_dir=run_dir,
        baseline_id=resolved_baseline_id,
        transfer_manifest=transfer_manifest,
    )

    package_record_dir = run_dir / "package_records"
    package_record_names = {
        f"{resolved_baseline_id}_package_input_manifest.json",
        f"{resolved_baseline_id}_archive_summary.json",
        f"{resolved_baseline_id}_archive_manifest.local.json",
    }
    package_record_files = (
        {path.resolve() for path in package_record_dir.iterdir() if path.is_file()}
        if package_record_dir.is_dir() and not package_record_dir.is_symlink()
        else set()
    )
    if package_record_files and {
        path.name for path in package_record_files
    } != package_record_names:
        raise RuntimeError("method-faithful package_records 必须为空或精确包含三项治理记录")

    expected_run_files = {
        *(path.resolve() for path in required_runner_entries),
        *(path.resolve() for path in adapter_entries),
        *package_record_files,
    }
    actual_run_files, actual_run_directories = _inventory_regular_tree(run_dir)
    expected_run_directories = {run_dir.resolve(), package_record_dir.resolve()}
    for path in expected_run_files:
        parent = path.parent
        while True:
            expected_run_directories.add(parent.resolve())
            if parent.resolve() == run_dir.resolve():
                break
            parent = parent.parent
    if actual_run_files != expected_run_files:
        unexpected = sorted(path.name for path in actual_run_files - expected_run_files)
        missing = sorted(path.name for path in expected_run_files - actual_run_files)
        raise RuntimeError(
            "method-faithful 运行目录文件不满足精确白名单: "
            f"unexpected={unexpected},missing={missing}"
        )
    expected_child_directories = expected_run_directories - {run_dir.resolve()}
    if actual_run_directories != expected_child_directories:
        unexpected = sorted(
            path.name for path in actual_run_directories - expected_child_directories
        )
        missing = sorted(
            path.name for path in expected_child_directories - actual_run_directories
        )
        raise RuntimeError(
            "method-faithful 运行目录结构不满足精确白名单: "
            f"unexpected={unexpected},missing={missing}"
        )
    entries = {
        *expected_run_files,
        *transfer_entries,
    }
    if archive_path.resolve() in entries:
        raise RuntimeError("method-faithful 结果包不得递归包含自身")
    return tuple(sorted(entries, key=lambda path: path.relative_to(root_path).as_posix()))


def _resolve_transfer_member(
    source_dir: Path,
    transfer_manifest: Mapping[str, Any],
    *,
    path_field: str,
    digest_field: str,
) -> Path:
    """解析并复验 transfer manifest 声明的单个仓库内事实文件."""

    relative_text = str(transfer_manifest.get(path_field, ""))
    pure_path = PurePosixPath(relative_text)
    if (
        not relative_text
        or "\\" in relative_text
        or pure_path.is_absolute()
        or any(part in {"", ".", ".."} for part in pure_path.parts)
        or pure_path.as_posix() != relative_text
    ):
        raise RuntimeError(f"baseline transfer 的 {path_field} 不是安全相对路径")
    unresolved = source_dir / relative_text
    resolved = unresolved.resolve()
    resolved.relative_to(source_dir.resolve())
    current = source_dir
    for part in pure_path.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"baseline transfer 的 {path_field} 不得经过符号链接")
    if not resolved.is_file():
        raise FileNotFoundError(f"baseline transfer 成员不存在: {relative_text}")
    if file_sha256(resolved) != str(transfer_manifest.get(digest_field, "")):
        raise RuntimeError(f"baseline transfer 的 {path_field} 字节摘要不一致")
    return resolved


def _validate_package_unit_evidence(
    *,
    root_path: Path,
    source_dir: Path,
    paper_run: Any,
    run_summary: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    transfer_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """在归档前从原子单元重新构造 observation 与来源聚合."""

    member_specs = (
        (
            "baseline_observations_path",
            "baseline_observations_sha256",
        ),
        (
            "baseline_command_results_path",
            "baseline_command_results_sha256",
        ),
        ("prompt_plan_path", "prompt_plan_sha256"),
        ("adapter_manifest_path", "adapter_manifest_sha256"),
        (
            "numerical_fidelity_report_path",
            "numerical_fidelity_report_sha256",
        ),
        ("execution_manifest_path", "execution_manifest_sha256"),
    )
    members = {
        path_field: _resolve_transfer_member(
            source_dir,
            transfer_manifest,
            path_field=path_field,
            digest_field=digest_field,
        )
        for path_field, digest_field in member_specs
    }
    observations = load_baseline_observation_rows(
        members["baseline_observations_path"]
    )
    command_results = read_json(members["baseline_command_results_path"])
    execution_manifest = read_json(members["execution_manifest_path"])
    adapter_manifest = read_json(members["adapter_manifest_path"])
    numerical_fidelity_report = (
        validate_method_faithful_numerical_fidelity_report(
            read_json(members["numerical_fidelity_report_path"]),
            expected_baseline_id=str(transfer_manifest.get("baseline_id", "")),
        )
    )
    unit_evidence = validate_method_faithful_adapter_unit_evidence(
        manifest=adapter_manifest,
        observation_rows=observations,
        root=root_path,
    )
    prompt_rows = read_json(members["prompt_plan_path"])
    if not isinstance(prompt_rows, list) or len(prompt_rows) != paper_run.prompt_count:
        raise RuntimeError("baseline 归档 Prompt 计划未覆盖完整论文运行规模")
    expected_calibration_count = sum(
        str(row.get("split", "")) == "calibration"
        for row in prompt_rows
        if isinstance(row, Mapping)
    )
    expected_test_count = sum(
        str(row.get("split", "")) == "test"
        for row in prompt_rows
        if isinstance(row, Mapping)
    )
    run_config = adapter_manifest.get("run_config")
    formal_attack_names = transfer_manifest.get("formal_attack_names")
    if not isinstance(run_config, Mapping) or not isinstance(formal_attack_names, list):
        raise RuntimeError("baseline 归档缺少方法忠实运行配置或正式攻击集合")
    run_attack_names = [str(name) for name in run_config.get("attack_families", ())]
    transfer_attack_names = [str(name) for name in formal_attack_names]
    if not all(
        (
            int(run_config.get("prompt_count", -1)) == len(prompt_rows),
            int(run_config.get("test_prompt_count", -1)) == expected_test_count,
            len(run_attack_names) == len(set(run_attack_names)),
            len(transfer_attack_names) == len(set(transfer_attack_names)),
            set(run_attack_names) == set(transfer_attack_names),
            transfer_manifest.get("prompt_protocol_digest")
            == canonical_prompt_protocol_digest(prompt_rows),
            transfer_manifest.get("method_faithful_numerical_fidelity_ready")
            is True,
            transfer_manifest.get("numerical_fidelity_report_digest")
            == numerical_fidelity_report["numerical_fidelity_report_digest"],
            transfer_manifest.get("numerical_fidelity_reference_mode")
            == numerical_fidelity_report["numerical_fidelity_reference_mode"],
        )
    ):
        raise RuntimeError("baseline 归档单元计划未绑定完整 Prompt 与正式攻击协议")
    expected_attack_unit_count = expected_test_count * len(transfer_attack_names) * 2
    if not all(
        (
            int(unit_evidence.get("method_faithful_source_prompt_unit_count", -1))
            == len(prompt_rows),
            int(unit_evidence.get("method_faithful_formal_attack_unit_count", -1))
            == expected_attack_unit_count,
            len(observations) == len(prompt_rows) * 2 + expected_attack_unit_count,
        )
    ):
        raise RuntimeError("baseline 归档原子单元未覆盖完整 Prompt 与攻击 exact set")
    if (
        not isinstance(command_results, list)
        or len(command_results) != 1
        or not isinstance(execution_manifest, Mapping)
    ):
        raise RuntimeError("baseline 归档 command result 数量不一致")
    command_result = command_results[0]
    if not isinstance(command_result, Mapping) or not all(
        (
            command_result.get("baseline_id") == transfer_manifest.get("baseline_id"),
            int(command_result.get("return_code", -1)) == 0,
            int(command_result.get("observation_count", -1)) == len(observations),
            int(execution_manifest.get("observation_count", -1)) == len(observations),
        )
    ):
        raise RuntimeError("baseline 归档命令结果或执行 manifest 未绑定 observation")
    threshold_audit = audit_fixed_fpr_observation_threshold(
        observations,
        target_fpr=paper_run.target_fpr,
        expected_calibration_negative_count=expected_calibration_count,
    )
    if not threshold_audit.fixed_fpr_ready:
        raise RuntimeError("baseline 归档 observation 未通过 fixed-FPR 复验")
    if not all(
        (
            int(transfer_manifest.get("baseline_observation_count", -1))
            == len(observations),
            int(run_summary.get("primary_baseline_observation_count", -1))
            == len(observations),
            transfer_manifest.get("threshold") == threshold_audit.frozen_threshold,
            transfer_manifest.get("threshold_digest")
            == threshold_audit.threshold_digest,
            run_summary.get("threshold_digest") == threshold_audit.threshold_digest,
        )
    ):
        raise RuntimeError("baseline 归档 observation 计数或冻结阈值身份不一致")
    for field_name, expected_value in unit_evidence.items():
        if transfer_manifest.get(field_name) != expected_value:
            raise RuntimeError(
                f"baseline transfer 的 {field_name} 未绑定原子单元复算值"
            )
        if run_summary.get(field_name) != expected_value:
            raise RuntimeError(
                f"baseline summary 的 {field_name} 未绑定原子单元复算值"
            )
    metadata = run_manifest.get("metadata")
    if not isinstance(metadata, Mapping) or not all(
        (
            metadata.get("method_faithful_scientific_unit_resume_ready") is True,
            metadata.get("method_faithful_scientific_unit_records_digest")
            == unit_evidence["method_faithful_scientific_unit_records_digest"],
            metadata.get("method_faithful_numerical_fidelity_ready") is True,
            metadata.get("numerical_fidelity_report_digest")
            == numerical_fidelity_report["numerical_fidelity_report_digest"],
            metadata.get("numerical_fidelity_reference_mode")
            == numerical_fidelity_report["numerical_fidelity_reference_mode"],
            run_summary.get("method_faithful_numerical_fidelity_ready") is True,
            run_summary.get("numerical_fidelity_report_digest")
            == numerical_fidelity_report["numerical_fidelity_report_digest"],
            run_summary.get("numerical_fidelity_reference_mode")
            == numerical_fidelity_report["numerical_fidelity_reference_mode"],
        )
    ):
        raise RuntimeError("baseline 运行 manifest 未绑定原子单元证据")
    return unit_evidence


def package_external_baseline_method_faithful_outputs(
    root: str | Path = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    drive_output_dir: str | None = None,
    archive_name: str | None = None,
    baseline_id: str | None = None,
) -> ExternalBaselineMethodFaithfulArchiveRecord:
    """仅在单 baseline 运行通过后打包独占结果并镜像到 Drive。"""

    root_path = Path(root).resolve()
    formal_execution_package_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    resolved_baseline_id = resolve_primary_baseline_id(
        baseline_id or os.environ.get("SLM_WM_PRIMARY_BASELINE_ID", "")
    )
    paper_run = build_paper_run_config(root_path)
    configured_output_root = (root_path / output_dir).resolve()
    expected_output_root = (root_path / DEFAULT_OUTPUT_DIR).resolve()
    if configured_output_root != expected_output_root:
        raise ValueError("method-faithful 打包根目录必须使用正式 outputs family")
    source_dir = expected_output_root / paper_run.run_name
    run_dir = source_dir / "run_records" / resolved_baseline_id
    summary_path = run_dir / f"{resolved_baseline_id}_summary.json"
    run_manifest_path = run_dir / f"{resolved_baseline_id}_manifest.local.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"baseline 运行摘要不存在: {summary_path}")
    run_summary = read_json(summary_path)
    run_manifest = read_json(run_manifest_path)
    run_generation = run_summary.get("generation_protocol")
    run_config = run_manifest.get("config")
    if not isinstance(run_generation, Mapping) or not isinstance(run_config, Mapping):
        raise RuntimeError("baseline 运行摘要或 manifest 未绑定正式模型协议")
    formal_execution_run_lock = repository_environment.validate_formal_execution_lock_pair(
        run_manifest.get("formal_execution_run_lock"),
        formal_execution_package_lock,
        run_manifest.get("code_version"),
    )
    if not all(
        (
            run_summary.get("run_decision") == "pass",
            run_summary.get("external_baseline_method_faithful_ready") is True,
            run_summary.get("primary_baseline_adapter_ready") is True,
            run_summary.get("primary_baseline_id") == resolved_baseline_id,
            run_summary.get("paper_run_name") == paper_run.run_name,
            run_generation.get("model_id") == DEFAULT_MODEL_ID,
            run_generation.get("model_revision") == DEFAULT_MODEL_REVISION,
            run_config.get("model_id") == DEFAULT_MODEL_ID,
            run_config.get("model_revision") == DEFAULT_MODEL_REVISION,
            math.isclose(
                float(run_summary.get("target_fpr", -1.0)),
                paper_run.target_fpr,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
        )
    ):
        raise RuntimeError("失败或不完整的 baseline 运行不得生成正式结果包")
    transfer_manifest_path = (
        source_dir
        / "split_observations"
        / f"{resolved_baseline_id}_baseline_transfer_manifest.json"
    )
    transfer_manifest = read_json(transfer_manifest_path)
    transfer_generation = transfer_manifest.get("generation_protocol")
    if not isinstance(transfer_generation, Mapping):
        raise RuntimeError("baseline transfer manifest 未绑定正式生成协议")
    repository_environment.verify_formal_execution_lock_code_version(
        formal_execution_run_lock,
        transfer_manifest.get("code_version"),
    )
    if not all(
        (
            transfer_manifest.get("baseline_id") == resolved_baseline_id,
            transfer_manifest.get("transfer_ready") is True,
            transfer_manifest.get("paper_run_name") == paper_run.run_name,
            transfer_manifest.get("model_id") == DEFAULT_MODEL_ID,
            transfer_manifest.get("model_revision") == DEFAULT_MODEL_REVISION,
            transfer_generation.get("model_id") == DEFAULT_MODEL_ID,
            transfer_generation.get("model_revision") == DEFAULT_MODEL_REVISION,
            math.isclose(
                float(transfer_manifest.get("target_fpr", -1.0)),
                paper_run.target_fpr,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
        )
    ):
        raise RuntimeError("baseline transfer manifest 未通过")
    package_unit_evidence = _validate_package_unit_evidence(
        root_path=root_path,
        source_dir=source_dir,
        paper_run=paper_run,
        run_summary=run_summary,
        run_manifest=run_manifest,
        transfer_manifest=transfer_manifest,
    )

    resolved_archive_name = archive_name or (
        f"external_baseline_method_faithful_package_{resolved_baseline_id}_"
        f"{utc_archive_token()}_{formal_execution_package_lock['formal_execution_commit'][:7]}.zip"
    )
    expected_archive_prefix = (
        f"external_baseline_method_faithful_package_{resolved_baseline_id}_"
    )
    if (
        Path(resolved_archive_name).name != resolved_archive_name
        or not resolved_archive_name.startswith(expected_archive_prefix)
        or Path(resolved_archive_name).suffix.lower() != ".zip"
    ):
        raise ValueError("method-faithful archive_name 未匹配当前 baseline 正式命名")
    archive_path = source_dir / resolved_archive_name
    package_record_dir = run_dir / "package_records"
    package_record_dir.mkdir(parents=True, exist_ok=True)
    package_input_manifest_path = package_record_dir / f"{resolved_baseline_id}_package_input_manifest.json"
    archive_summary_path = package_record_dir / f"{resolved_baseline_id}_archive_summary.json"
    archive_manifest_path = package_record_dir / f"{resolved_baseline_id}_archive_manifest.local.json"
    for stale_path in (
        package_input_manifest_path,
        archive_summary_path,
        archive_manifest_path,
        archive_path,
    ):
        if stale_path.exists():
            stale_path.unlink()

    entries = collect_package_entries(
        root_path,
        source_dir,
        archive_path,
        resolved_baseline_id,
    )
    if not entries:
        raise RuntimeError("baseline 结果包白名单为空")
    write_json(
        package_input_manifest_path,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "paper_run_name": paper_run.run_name,
            "target_fpr": paper_run.target_fpr,
            "randomization_repeat_id": paper_run.randomization_repeat_id,
            "generation_seed_index": paper_run.generation_seed_index,
            "generation_seed_offset": paper_run.generation_seed_offset,
            "watermark_key_index": paper_run.watermark_key_index,
            "formal_randomization_protocol_digest": (
                paper_run.formal_randomization_protocol_digest
            ),
            "randomization_repeat_identity": {
                "randomization_repeat_id": paper_run.randomization_repeat_id,
                "generation_seed_index": paper_run.generation_seed_index,
                "generation_seed_offset": paper_run.generation_seed_offset,
                "watermark_key_index": paper_run.watermark_key_index,
                "formal_randomization_protocol_digest": (
                    paper_run.formal_randomization_protocol_digest
                ),
            },
            "baseline_id": resolved_baseline_id,
            "model_id": DEFAULT_MODEL_ID,
            "model_revision": DEFAULT_MODEL_REVISION,
            "formal_execution_run_lock": formal_execution_run_lock,
            "formal_execution_package_lock": formal_execution_package_lock,
            "method_faithful_scientific_unit_records_digest": package_unit_evidence[
                "method_faithful_scientific_unit_records_digest"
            ],
            "entry_paths": [entry.relative_to(root_path).as_posix() for entry in entries],
            "entry_sha256": {
                entry.relative_to(root_path).as_posix(): file_sha256(entry) for entry in entries
            },
        },
    )
    resolved_drive_output_dir = drive_output_dir or build_paper_run_config(root_path).drive_dir(
        "external_baseline_method_faithful"
    )
    preliminary_record = ExternalBaselineMethodFaithfulArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest="",
        archive_entry_count=len(entries) + 3,
        drive_archive_path=str(Path(resolved_drive_output_dir).expanduser() / resolved_archive_name),
        drive_archive_digest="",
        metadata={
            "construction_unit_name": "external_baseline_method_faithful",
            "baseline_id": resolved_baseline_id,
            "model_id": DEFAULT_MODEL_ID,
            "model_revision": DEFAULT_MODEL_REVISION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(archive_summary_path, preliminary_record.to_dict())
    archive_manifest = build_artifact_manifest(
        artifact_id=f"{resolved_baseline_id}_method_faithful_archive_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(entry.relative_to(root_path).as_posix() for entry in entries),
        output_paths=(archive_path.relative_to(root_path).as_posix(),),
        config={
            "archive_name": resolved_archive_name,
            "baseline_id": resolved_baseline_id,
            "paper_run_name": paper_run.run_name,
            "target_fpr": paper_run.target_fpr,
            "randomization_repeat_id": paper_run.randomization_repeat_id,
            "generation_seed_index": paper_run.generation_seed_index,
            "generation_seed_offset": paper_run.generation_seed_offset,
            "watermark_key_index": paper_run.watermark_key_index,
            "formal_randomization_protocol_digest": (
                paper_run.formal_randomization_protocol_digest
            ),
            "randomization_repeat_identity": {
                "randomization_repeat_id": paper_run.randomization_repeat_id,
                "generation_seed_index": paper_run.generation_seed_index,
                "generation_seed_offset": paper_run.generation_seed_offset,
                "watermark_key_index": paper_run.watermark_key_index,
                "formal_randomization_protocol_digest": (
                    paper_run.formal_randomization_protocol_digest
                ),
            },
            "model_id": DEFAULT_MODEL_ID,
            "model_revision": DEFAULT_MODEL_REVISION,
            "drive_output_dir": str(Path(resolved_drive_output_dir).expanduser()),
        },
        code_version=formal_execution_package_lock["formal_execution_commit"],
        rebuild_command="调用 paper_experiments.runners.external_baseline_method_faithful",
        metadata={
            "baseline_id": resolved_baseline_id,
            "paper_run_name": paper_run.run_name,
            "target_fpr": paper_run.target_fpr,
            "randomization_repeat_id": paper_run.randomization_repeat_id,
            "generation_seed_index": paper_run.generation_seed_index,
            "generation_seed_offset": paper_run.generation_seed_offset,
            "watermark_key_index": paper_run.watermark_key_index,
            "formal_randomization_protocol_digest": (
                paper_run.formal_randomization_protocol_digest
            ),
            "randomization_repeat_identity": {
                "randomization_repeat_id": paper_run.randomization_repeat_id,
                "generation_seed_index": paper_run.generation_seed_index,
                "generation_seed_offset": paper_run.generation_seed_offset,
                "watermark_key_index": paper_run.watermark_key_index,
                "formal_randomization_protocol_digest": (
                    paper_run.formal_randomization_protocol_digest
                ),
            },
            "model_id": DEFAULT_MODEL_ID,
            "model_revision": DEFAULT_MODEL_REVISION,
            "run_decision": "pass",
        },
    ).to_dict()
    archive_manifest["formal_execution_run_lock"] = formal_execution_run_lock
    archive_manifest["formal_execution_package_lock"] = formal_execution_package_lock
    write_json(archive_manifest_path, archive_manifest)

    final_entries = collect_package_entries(
        root_path,
        source_dir,
        archive_path,
        resolved_baseline_id,
    )
    with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for entry in final_entries:
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
    record = ExternalBaselineMethodFaithfulArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest=file_digest(archive_path),
        archive_entry_count=len(final_entries),
        drive_archive_path=str(mirrored_path),
        drive_archive_digest=file_digest(mirrored_path),
        metadata={
            "construction_unit_name": "external_baseline_method_faithful",
            "baseline_id": resolved_baseline_id,
            "model_id": DEFAULT_MODEL_ID,
            "model_revision": DEFAULT_MODEL_REVISION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(archive_summary_path, record.to_dict())
    return record
