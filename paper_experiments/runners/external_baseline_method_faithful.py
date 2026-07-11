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
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.protocol.fixed_fpr_observation_audit import audit_fixed_fpr_observation_threshold
from experiments.protocol.paper_run_config import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_TARGET_FPR,
    build_paper_run_config,
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
from experiments.runtime.repository_environment import (
    build_runtime_environment_report,
    file_digest,
    resolve_code_version,
)
from external_baseline.primary.sd35_method_faithful_common import supported_formal_image_attack_names
from paper_experiments.baselines.method_faithful_observation_collection import (
    canonical_prompt_protocol_digest,
)
from paper_experiments.baselines.observation_io import load_baseline_observation_rows


DEFAULT_OUTPUT_DIR = "outputs/external_baseline_method_faithful"
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
DEFAULT_BASELINE_SEED = 20260621
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
    prompt_set: str = "pilot_paper"
    prompt_file: str = "configs/paper_main_pilot_paper_prompts.txt"
    primary_baseline_id: str = "tree_ring"
    model_id: str = DEFAULT_MODEL_ID
    seed: int = DEFAULT_BASELINE_SEED
    target_fpr: float = DEFAULT_TARGET_FPR
    num_inference_steps: int = DEFAULT_INFERENCE_STEPS
    num_inversion_steps: int = DEFAULT_INFERENCE_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    primary_baseline_max_samples: int = 700
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
            config.seed == DEFAULT_BASELINE_SEED,
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
    output_dir = (root_path / config.output_dir).resolve()
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
        "summary": run_dir / f"{baseline_id}_summary.json",
        "manifest": run_dir / f"{baseline_id}_manifest.local.json",
        "package_record_dir": run_dir / "package_records",
    }


def prepare_single_baseline_run_directory(paths: Mapping[str, Path]) -> None:
    """清理当前 baseline 的固定运行目录和共享 transfer 文件。

    该函数属于通用可复现运行写法: 每次运行从空的 baseline 独占目录开始，
    避免 probe、pilot、full 或失败重跑遗留的图片和 manifest 混入新结果包。
    其他 baseline 的共享 transfer 文件不会被删除。
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
        run_dir.unlink()
    elif run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
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
    return [
        {
            "prompt_id": record.prompt_id,
            "prompt_index": record.prompt_index,
            "prompt_set": record.prompt_set,
            "split": record.split,
            "prompt_text": record.prompt_text,
            "prompt_digest": record.prompt_digest,
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
        "scripts/build_external_baseline_command_plan.py",
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
        "execution_manifest_path": paths["execution_manifest"].relative_to(collection_root).as_posix(),
        "execution_manifest_sha256": file_sha256(paths["execution_manifest"]),
        "paper_run_name": config.prompt_set,
        "prompt_set": config.prompt_set,
        "prompt_count": len(prompt_rows),
        "target_fpr": float(config.target_fpr),
        "threshold": threshold_audit.frozen_threshold,
        "threshold_digest": threshold_audit.threshold_digest,
        "generation_protocol": {
            "model_id": config.model_id,
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
        "code_version": resolve_code_version(root_path),
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
        "scripts/run_external_baseline_command_plan.py",
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
        "scripts/validate_external_baseline_evidence.py",
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

    transfer_manifest = write_baseline_transfer_files(root_path, config, paths)
    return {
        "adapter_execution_ready": True,
        "primary_baseline_adapter_ready": True,
        "primary_baseline_id": config.primary_baseline_id,
        "primary_baseline_observation_count": int(
            transfer_manifest["baseline_observation_count"]
        ),
        "transfer_manifest_path": relative_or_absolute(paths["transfer_manifest"], root_path),
        "threshold_digest": str(transfer_manifest["threshold_digest"]),
    }


def write_failure_outputs(
    root_path: Path,
    config: ExternalBaselineMethodFaithfulConfig,
    paths: dict[str, Path],
    error: Exception,
) -> dict[str, Any]:
    """写出失败诊断，但不生成可被正式闭合选择的结果包。"""

    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    environment_report = build_runtime_environment_report()
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
        config=asdict(config),
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
    paths = output_paths(root_path, config)
    validate_formal_run_config(root_path, config)
    prepare_single_baseline_run_directory(paths)
    try:
        with progress_bar(
            3,
            desc=f"method-faithful {config.primary_baseline_id}",
            enabled=config.enable_workflow_progress_bar,
        ) as run_progress:
            device_report = ensure_cuda_if_requested(config.require_cuda)
            update_progress(run_progress, profile="operation=ensure_cuda")
            adapter_report = build_and_run_primary_baseline_adapter(
                root_path,
                config,
                paths,
                progress=run_progress,
            )
            update_progress(run_progress, profile="operation=run_baseline_adapter")
            environment_report = build_runtime_environment_report()
            environment_report["external_baseline_device_report"] = device_report
            write_json(paths["environment_report"], environment_report)
            update_progress(run_progress, profile="operation=write_environment_report")
    except Exception as error:
        return write_failure_outputs(root_path, config, paths, error)

    summary = {
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
            "num_inference_steps": int(config.num_inference_steps),
            "guidance_scale": float(config.guidance_scale),
        },
        "detection_protocol": {
            "input_access_mode": "image_only",
            "num_inversion_steps": int(config.num_inversion_steps),
        },
        "transfer_manifest_path": adapter_report["transfer_manifest_path"],
        "threshold_digest": adapter_report["threshold_digest"],
        "environment_report_path": relative_or_absolute(paths["environment_report"], root_path),
        "supports_paper_claim": False,
        "unsupported_reason": "formal_import_and_comparison_required",
    }
    write_json(paths["summary"], summary)
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
        ),
        config=asdict(config),
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 paper_experiments.runners.external_baseline_method_faithful",
        metadata={
            "run_decision": "pass",
            "primary_baseline_id": config.primary_baseline_id,
            "primary_baseline_observation_count": adapter_report[
                "primary_baseline_observation_count"
            ],
            "supports_paper_claim": False,
        },
    ).to_dict()
    write_json(paths["manifest"], manifest)
    return summary


def build_default_config() -> ExternalBaselineMethodFaithfulConfig:
    """从当前论文运行层级与单 baseline 环境变量构造配置。"""

    paper_run = build_paper_run_config(".")
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
        seed=int(os.environ.get("SLM_WM_EXTERNAL_BASELINE_SEED", "20260621")),
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


def collect_package_entries(
    root_path: Path,
    output_dir: Path,
    archive_path: Path,
    baseline_id: str,
) -> tuple[Path, ...]:
    """白名单收集当前 baseline 独占产物。"""

    resolved_baseline_id = resolve_primary_baseline_id(baseline_id)
    run_dir = output_dir / "run_records" / resolved_baseline_id
    split_dir = output_dir / "split_observations"
    entries: list[Path] = []
    for path in sorted(run_dir.rglob("*")) if run_dir.exists() else ():
        if path.is_file() and path.resolve() != archive_path.resolve() and path.suffix.lower() != ".zip":
            entries.append(path)
    for suffix in (
        "baseline_observations.json",
        "baseline_command_results.json",
        "baseline_transfer_manifest.json",
    ):
        path = split_dir / f"{resolved_baseline_id}_{suffix}"
        if path.is_file():
            entries.append(path)
    return tuple(dict.fromkeys(entries))


def package_external_baseline_method_faithful_outputs(
    root: str | Path = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    drive_output_dir: str | None = None,
    archive_name: str = "external_baseline_method_faithful_package.zip",
    baseline_id: str | None = None,
) -> ExternalBaselineMethodFaithfulArchiveRecord:
    """仅在单 baseline 运行通过后打包独占结果并镜像到 Drive。"""

    root_path = Path(root).resolve()
    resolved_baseline_id = resolve_primary_baseline_id(
        baseline_id or os.environ.get("SLM_WM_PRIMARY_BASELINE_ID", "")
    )
    source_dir = (root_path / output_dir).resolve()
    run_dir = source_dir / "run_records" / resolved_baseline_id
    summary_path = run_dir / f"{resolved_baseline_id}_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"baseline 运行摘要不存在: {summary_path}")
    run_summary = read_json(summary_path)
    if run_summary.get("run_decision") != "pass" or not bool(
        run_summary.get("external_baseline_method_faithful_ready")
    ):
        raise RuntimeError("失败或不完整的 baseline 运行不得生成正式结果包")
    transfer_manifest_path = (
        source_dir
        / "split_observations"
        / f"{resolved_baseline_id}_baseline_transfer_manifest.json"
    )
    transfer_manifest = read_json(transfer_manifest_path)
    if transfer_manifest.get("baseline_id") != resolved_baseline_id or not bool(
        transfer_manifest.get("transfer_ready")
    ):
        raise RuntimeError("baseline transfer manifest 未通过")

    archive_path = source_dir / archive_name
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
            "baseline_id": resolved_baseline_id,
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
        drive_archive_path=str(Path(resolved_drive_output_dir).expanduser() / archive_name),
        drive_archive_digest="",
        metadata={
            "construction_unit_name": "external_baseline_method_faithful",
            "baseline_id": resolved_baseline_id,
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
            "archive_name": archive_name,
            "baseline_id": resolved_baseline_id,
            "drive_output_dir": str(Path(resolved_drive_output_dir).expanduser()),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 paper_experiments.runners.external_baseline_method_faithful",
        metadata={"baseline_id": resolved_baseline_id, "run_decision": "pass"},
    ).to_dict()
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
    drive_dir = Path(resolved_drive_output_dir).expanduser()
    drive_dir.mkdir(parents=True, exist_ok=True)
    mirrored_path = drive_dir / archive_name
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
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(archive_summary_path, record.to_dict())
    return record
