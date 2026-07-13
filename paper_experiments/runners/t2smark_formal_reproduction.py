"""T2SMark formal 真实 GPU 复现入口的 Colab 辅助函数。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from paper_experiments.baselines import (
    build_fixed_fpr_operating_point,
    build_t2smark_formal_candidate_records,
    resolve_paper_run_prompt_protocol_name,
    validate_primary_baseline_formal_import_rows,
)
from paper_experiments.baselines.t2smark_pair_quality import (
    DEFAULT_CLIP_MODEL_REVISION,
    write_t2smark_strict_pair_quality_outputs,
)
from experiments.protocol.paper_run_config import (
    DEFAULT_TARGET_FPR as DEFAULT_PAPER_RUN_TARGET_FPR,
    build_paper_run_config,
    normalize_paper_run_name,
    resolve_count_from_environment,
)
from experiments.protocol.formal_randomization import (
    DEFAULT_FORMAL_RANDOMIZATION_REPEAT_ID,
    build_formal_randomization_identity,
    formal_watermark_key_seed_random,
    require_formal_watermark_key_plan,
    resolve_formal_randomization_repeat,
)
from experiments.protocol.method_runtime_config import load_formal_method_runtime_config
from experiments.runtime import repository_environment
from experiments.runtime.model_sources import require_registered_model_reference
from experiments.protocol.fixed_fpr_observation_audit import audit_fixed_fpr_observation_threshold
from experiments.protocol.prompts import build_prompt_records, normalize_prompt_text
from experiments.protocol.splits import apply_split_assignments
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from external_baseline.primary.sd35_method_faithful_common import supported_formal_image_attack_names
from external_baseline.primary.t2smark.adapter.formal_unit_checkpoint import (
    aggregate_t2smark_formal_unit_records,
    build_t2smark_formal_unit_contract,
    inspect_t2smark_formal_unit_records,
    validate_t2smark_formal_unit_contract,
    write_or_validate_t2smark_formal_unit_contract,
)
from external_baseline.primary.t2smark.adapter.run_slm_eval import (
    build_t2smark_observations,
)
from paper_experiments.runners.external_source_runtime import ensure_cuda_if_requested, run_command
from paper_experiments.runners.t2smark_source_runtime import (
    DEFAULT_T2SMARK_MODEL_ID,
    DEFAULT_T2SMARK_MODEL_REVISION,
    DEFAULT_T2SMARK_SOURCE_ENTRY,
    configured_attack_names,
    count_t2smark_formal_attack_items,
    ensure_t2smark_source_available,
)
from experiments.runtime.progress import call_runner_with_progress_status, emit_progress_status, progress_bar, update_progress
from experiments.runtime.archive_naming import utc_archive_token
from experiments.runtime.repository_environment import (
    build_runtime_environment_report,
    file_digest,
    resolve_code_version,
)
from experiments.runtime.scientific_execution_binding import (
    validate_scientific_execution_binding,
)

DEFAULT_OUTPUT_DIR = "outputs/t2smark_formal_reproduction"
DEFAULT_DRIVE_OUTPUT_DIR = ""
DEFAULT_PROMPT_FILE = "configs/paper_main_probe_paper_prompts.txt"
DEFAULT_RUN_NAME = "t2smark_sd35_medium_probe_paper"
DEFAULT_TARGET_FPR = DEFAULT_PAPER_RUN_TARGET_FPR
DEFAULT_PROMPT_LIMIT = 70
DEFAULT_T2SMARK_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_T2SMARK_LPIPS_NETWORK = "alex"
FORMAL_T2SMARK_NUM_INFERENCE_STEPS = 20
FORMAL_T2SMARK_NUM_INVERSION_STEPS = 20
FORMAL_T2SMARK_GUIDANCE_SCALE = 4.5
FORMAL_T2SMARK_IMAGE_SIZE = 512
T2SMARK_FORMAL_PROTOCOL_BINDING_NAME = "t2smark_formal_protocol_binding"
DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES = ",".join(supported_formal_image_attack_names())
_FORMAL_METHOD_DEFAULTS = load_formal_method_runtime_config(".")
DEFAULT_T2SMARK_SEED = _FORMAL_METHOD_DEFAULTS.seed
_DEFAULT_RANDOMIZATION_REPEAT = resolve_formal_randomization_repeat(None)
DEFAULT_T2SMARK_WATERMARK_KEY_SEED_RANDOM = formal_watermark_key_seed_random(
    "slm_wm_paper_key",
    _DEFAULT_RANDOMIZATION_REPEAT,
)
@dataclass(frozen=True)
class T2SMarkFormalReproductionConfig:
    """描述 T2SMark formal 真实复现所需的最小配置。"""

    output_dir: str = DEFAULT_OUTPUT_DIR
    drive_output_dir: str = field(
        default_factory=lambda: build_paper_run_config(".").drive_dir("external_baseline_official_reference")
    )
    prompt_set: str = "probe_paper"
    prompt_file: str = DEFAULT_PROMPT_FILE
    t2smark_run_name: str = DEFAULT_RUN_NAME
    model_id: str = DEFAULT_T2SMARK_MODEL_ID
    model_revision: str = DEFAULT_T2SMARK_MODEL_REVISION
    seed: int = DEFAULT_T2SMARK_SEED
    randomization_repeat_id: str = DEFAULT_FORMAL_RANDOMIZATION_REPEAT_ID
    generation_seed_index: int = 0
    generation_seed_offset: int = 0
    watermark_key_index: int = 0
    formal_randomization_protocol_digest: str = field(
        default_factory=lambda: build_paper_run_config(
            "."
        ).formal_randomization_protocol_digest
    )
    watermark_key_seed_random: int = (
        DEFAULT_T2SMARK_WATERMARK_KEY_SEED_RANDOM
    )
    prompt_limit: int = DEFAULT_PROMPT_LIMIT
    clip_test_num: int = 0
    num_inference_steps: int = FORMAL_T2SMARK_NUM_INFERENCE_STEPS
    num_inversion_steps: int = FORMAL_T2SMARK_NUM_INVERSION_STEPS
    guidance_scale: float = FORMAL_T2SMARK_GUIDANCE_SCALE
    target_fpr: float = DEFAULT_TARGET_FPR
    minimum_prompt_protocol_count: int = DEFAULT_PROMPT_LIMIT
    reuse_existing: bool = True
    force_generate: bool = False
    save_image: bool = True
    save_clean_pair: bool = True
    formal_attack_families: str = DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES
    enable_pair_perceptual_metrics: bool = True
    pair_clip_model_id: str = DEFAULT_T2SMARK_CLIP_MODEL_ID
    pair_clip_model_revision: str = DEFAULT_CLIP_MODEL_REVISION
    pair_lpips_network: str = DEFAULT_T2SMARK_LPIPS_NETWORK
    pair_perceptual_metric_device_name: str = "cpu"
    require_cuda: bool = True
    timeout_seconds: int = 86400
    enable_workflow_progress_bar: bool = True


@dataclass(frozen=True)
class T2SMarkFormalArchiveRecord:
    """记录 T2SMark formal 复现压缩包及 Drive 镜像信息。"""

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
    merged_report["t2smark_formal_device_report"] = dict(device_report)
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


def build_t2smark_formal_environment_report(
    device_report: dict[str, Any],
    *,
    verified_formal_execution_lock: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造与 T2SMark formal GPU 检查结果一致的环境报告。"""

    try:
        import torch
    except Exception:  # pragma: no cover - 本地轻量测试不强制安装 torch
        environment_report = build_runtime_environment_report(
            "t2smark_sd35_gpu",
            verified_formal_execution_lock=verified_formal_execution_lock,
        )
    else:
        environment_report = build_runtime_environment_report(
            "t2smark_sd35_gpu",
            torch_module=torch,
            verified_formal_execution_lock=verified_formal_execution_lock,
        )
    return synchronize_environment_report_with_device_report(environment_report, device_report)


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先记录相对仓库根目录的路径。"""

    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def output_paths(root_path: Path, config: T2SMarkFormalReproductionConfig) -> dict[str, Path]:
    """集中构造 T2SMark formal 复现所需路径。"""

    configured_output_root = (root_path / config.output_dir).resolve()
    expected_output_root = (root_path / DEFAULT_OUTPUT_DIR).resolve()
    if configured_output_root != expected_output_root:
        raise ValueError("T2SMark 输出根目录必须使用正式 outputs family")
    output_dir = expected_output_root / normalize_paper_run_name(config.prompt_set)
    official_root = output_dir / "t2smark_official"
    official_run_dir = official_root / config.t2smark_run_name
    adapter_dir = output_dir / "t2smark_adapter"
    return {
        "output_dir": output_dir,
        "official_root": official_root,
        "official_run_dir": official_run_dir,
        "official_results": official_run_dir / "results.json",
        "official_settings": official_run_dir / "settings.json",
        "official_protocol_binding": official_run_dir / "slm_formal_protocol_binding.json",
        "official_unit_contract": official_run_dir / "slm_formal_unit_contract.json",
        "official_unit_dir": official_run_dir / "slm_formal_units",
        "official_images": official_run_dir / "images",
        "prompt_dataset": output_dir / "t2smark_formal_prompt_dataset.json",
        "prompt_plan": output_dir / "t2smark_formal_prompt_plan.json",
        "image_pairs": output_dir / "t2smark_formal_image_pairs.json",
        "pair_quality_metrics": output_dir / "t2smark_formal_strict_pair_quality_metrics.csv",
        "pair_quality_summary": output_dir / "t2smark_formal_strict_pair_quality_summary.json",
        "adapter_observations": adapter_dir / "baseline_observations.json",
        "adapter_manifest": adapter_dir / "t2smark_slm_adapter_manifest.json",
        "candidate_records": output_dir / "t2smark_formal_import_candidate_records.jsonl",
        "validation_report": output_dir / "t2smark_formal_import_validation_report.json",
        "environment_report": output_dir / "t2smark_formal_environment_report.json",
        "summary": output_dir / "t2smark_formal_reproduction_summary.json",
        "manifest": output_dir / "t2smark_formal_reproduction_manifest.local.json",
        "source_prepare_result": output_dir / "t2smark_source_prepare_result.json",
        "official_command_result": output_dir / "t2smark_formal_official_command_result.json",
        "adapter_command_result": output_dir / "t2smark_formal_adapter_command_result.json",
        "adapter_artifact_manifest": adapter_dir / "artifacts" / "t2smark_slm_adapter_manifest.json",
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


def build_paper_run_prompt_rows(
    prompt_set: str,
    prompt_texts: tuple[str, ...],
    *,
    base_seed: int = DEFAULT_T2SMARK_SEED,
    randomization_repeat_id: str = DEFAULT_FORMAL_RANDOMIZATION_REPEAT_ID,
    root_key_material: str = "slm_wm_paper_key",
) -> tuple[dict[str, Any], ...]:
    """构造 T2SMark 运行使用的 prompt 计划。"""

    rows: list[dict[str, Any]] = []
    repeat = resolve_formal_randomization_repeat(randomization_repeat_id)
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
                **build_formal_randomization_identity(
                    base_seed=base_seed,
                    prompt_index=record.prompt_index,
                    root_key_material=root_key_material,
                    repeat=repeat,
                ),
            }
        )
    return tuple(rows)


def write_paper_run_prompt_inputs(
    root_path: Path,
    config: T2SMarkFormalReproductionConfig,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """写出 T2SMark 官方入口与 adapter 共享的 prompt 输入。"""

    prompt_source_path = root_path / config.prompt_file
    all_prompt_texts = read_prompt_texts(prompt_source_path)
    chosen_prompt_texts = selected_prompt_texts(all_prompt_texts, config.prompt_limit)
    prompt_rows = build_paper_run_prompt_rows(
        config.prompt_set,
        chosen_prompt_texts,
        base_seed=DEFAULT_T2SMARK_SEED,
        randomization_repeat_id=config.randomization_repeat_id,
        root_key_material=os.environ.get(
            "SLM_WM_KEY_MATERIAL",
            "slm_wm_paper_key",
        ),
    )
    prompt_protocol_ready = bool(prompt_rows) and len(prompt_rows) >= int(config.minimum_prompt_protocol_count)
    dataset_payload = {
        "annotations": [
            {
                "caption": row["prompt_text"],
                "prompt_id": row["prompt_id"],
                "prompt_index": row["prompt_index"],
                "prompt_set": row["prompt_set"],
                "split": row["split"],
                "prompt_digest": row["prompt_digest"],
            }
            for row in prompt_rows
        ]
    }
    write_json(paths["prompt_dataset"], dataset_payload)
    write_json(paths["prompt_plan"], list(prompt_rows))
    report = {
        "paper_run_prompt_source_path": relative_or_absolute(prompt_source_path, root_path),
        "paper_run_prompt_count": len(all_prompt_texts),
        "selected_prompt_count": len(prompt_rows),
        "prompt_limit": int(config.prompt_limit),
        "minimum_prompt_protocol_count": int(config.minimum_prompt_protocol_count),
        "paper_run_prompt_protocol_ready": prompt_protocol_ready,
        "paper_claim_scale": config.prompt_set,
        "prompt_protocol_name": resolve_paper_run_prompt_protocol_name(root_path),
        "prompt_protocol_digest": build_stable_digest([row["prompt_digest"] for row in prompt_rows]),
        "prompt_dataset_path": relative_or_absolute(paths["prompt_dataset"], root_path),
        "prompt_plan_path": relative_or_absolute(paths["prompt_plan"], root_path),
    }
    return report


def validate_t2smark_formal_protocol_config(
    config: T2SMarkFormalReproductionConfig,
    *,
    root_path: Path,
) -> Any:
    """要求 T2SMark 使用当前论文级别的完整 Prompt 与冻结公平预算。"""

    if int(config.clip_test_num) != 0:
        raise ValueError("T2SMark formal 必须禁用官方源码中未受治理的 OpenCLIP 测试分支")
    paper_run = build_paper_run_config(root_path)
    repeat = resolve_formal_randomization_repeat(
        paper_run.randomization_repeat_id
    )
    root_key_material = os.environ.get(
        "SLM_WM_KEY_MATERIAL",
        "slm_wm_paper_key",
    )
    require_formal_watermark_key_plan(root_key_material)
    expected_attacks = tuple(supported_formal_image_attack_names())
    actual_attacks = configured_attack_names(config.formal_attack_families)
    if config.prompt_set != paper_run.prompt_set:
        raise ValueError("T2SMark prompt_set 必须与当前论文运行层级一致")
    expected_run_name = f"t2smark_sd35_medium_{paper_run.run_name}"
    if config.t2smark_run_name != expected_run_name:
        raise ValueError("T2SMark run name 必须与当前论文运行层级一致")
    if Path(config.prompt_file).name != Path(paper_run.prompt_file).name:
        raise ValueError("T2SMark Prompt 文件必须使用当前论文运行层级的受治理文件")
    if int(config.prompt_limit) != int(paper_run.prompt_count):
        raise ValueError("T2SMark formal 必须覆盖当前论文运行层级的完整 Prompt 集")
    if int(config.minimum_prompt_protocol_count) != int(paper_run.prompt_count):
        raise ValueError("T2SMark 最小 Prompt 协议计数必须等于完整 Prompt 数量")
    if config.model_id != DEFAULT_T2SMARK_MODEL_ID:
        raise ValueError("T2SMark formal 必须使用冻结的 SD3.5 Medium backbone")
    if config.model_revision != DEFAULT_T2SMARK_MODEL_REVISION:
        raise ValueError("T2SMark formal 必须使用冻结的 SD3.5 Medium revision")
    if (
        config.seed != DEFAULT_T2SMARK_SEED + repeat.generation_seed_offset
        or config.randomization_repeat_id
        != paper_run.randomization_repeat_id
        or config.generation_seed_index != paper_run.generation_seed_index
        or config.generation_seed_offset != paper_run.generation_seed_offset
        or config.watermark_key_index != paper_run.watermark_key_index
        or config.formal_randomization_protocol_digest
        != paper_run.formal_randomization_protocol_digest
        or config.watermark_key_seed_random
        != formal_watermark_key_seed_random(
            root_key_material,
            repeat,
        )
    ):
        raise ValueError("T2SMark formal 随机化身份未匹配共同协议")
    require_registered_model_reference(
        config.model_id,
        config.model_revision,
        required_usage_role="t2smark_diffusion_model",
    )
    if config.pair_clip_model_id != DEFAULT_T2SMARK_CLIP_MODEL_ID:
        raise ValueError("T2SMark formal 必须使用冻结的 CLIP 质量模型")
    if config.pair_clip_model_revision != DEFAULT_CLIP_MODEL_REVISION:
        raise ValueError("T2SMark formal 必须使用冻结的 CLIP revision")
    if int(config.num_inference_steps) != FORMAL_T2SMARK_NUM_INFERENCE_STEPS:
        raise ValueError("T2SMark formal 生成步数必须为20")
    if int(config.num_inversion_steps) != FORMAL_T2SMARK_NUM_INVERSION_STEPS:
        raise ValueError("T2SMark formal 反演步数必须为20")
    if not math.isclose(
        float(config.guidance_scale),
        FORMAL_T2SMARK_GUIDANCE_SCALE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("T2SMark formal guidance_scale 必须为4.5")
    if not math.isclose(float(config.target_fpr), paper_run.target_fpr, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("T2SMark target_fpr 必须与当前论文运行层级一致")
    if actual_attacks != expected_attacks:
        raise ValueError("T2SMark formal 必须使用完整且顺序冻结的共同攻击集合")
    if not config.save_image or not config.save_clean_pair:
        raise ValueError("T2SMark formal 必须保存水印图像与严格 clean pair")
    if not config.require_cuda:
        raise ValueError("T2SMark formal 必须要求 CUDA")
    return paper_run


def build_t2smark_formal_protocol_binding(
    config: T2SMarkFormalReproductionConfig,
    *,
    paper_run: Any,
    prompt_report: dict[str, Any],
    source_report: dict[str, Any],
) -> dict[str, Any]:
    """构造复用前必须逐字段相等的 canonical 协议绑定。"""

    payload = {
        "protocol_binding_name": T2SMARK_FORMAL_PROTOCOL_BINDING_NAME,
        "paper_run_name": paper_run.run_name,
        "t2smark_run_name": config.t2smark_run_name,
        "protocol_profile": paper_run.protocol_profile,
        "prompt_set": config.prompt_set,
        "prompt_protocol_name": str(prompt_report["prompt_protocol_name"]),
        "canonical_prompt_digest": str(prompt_report["prompt_protocol_digest"]),
        "selected_prompt_count": int(prompt_report["selected_prompt_count"]),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "seed": int(config.seed),
        "randomization_repeat_id": config.randomization_repeat_id,
        "generation_seed_index": int(config.generation_seed_index),
        "generation_seed_offset": int(config.generation_seed_offset),
        "watermark_key_index": int(config.watermark_key_index),
        "watermark_key_seed_random": int(config.watermark_key_seed_random),
        "formal_randomization_protocol_digest": (
            config.formal_randomization_protocol_digest
        ),
        "num_inference_steps": int(config.num_inference_steps),
        "num_inversion_steps": int(config.num_inversion_steps),
        "guidance_scale": float(config.guidance_scale),
        "clip_test_num": int(config.clip_test_num),
        "height": FORMAL_T2SMARK_IMAGE_SIZE,
        "width": FORMAL_T2SMARK_IMAGE_SIZE,
        "formal_attack_names": list(configured_attack_names(config.formal_attack_families)),
        "official_repository_commit": str(source_report.get("official_repository_commit", "")),
        "protocol_patch_sha256": str(source_report.get("protocol_patch_sha256", "")),
        "source_worktree_digest": str(source_report.get("source_worktree_digest", "")),
    }
    if not payload["official_repository_commit"] or len(payload["protocol_patch_sha256"]) != 64:
        raise RuntimeError("T2SMark 源码 revision 或正式协议补丁摘要缺失")
    if not bool(source_report.get("source_worktree_exact")) or len(payload["source_worktree_digest"]) != 64:
        raise RuntimeError("T2SMark 源码工作树未通过固定补丁精确性审计")
    payload["protocol_binding_digest"] = build_stable_digest(payload)
    return payload


def build_t2smark_formal_checkpoint_contract(
    config: T2SMarkFormalReproductionConfig,
    *,
    paper_run: Any,
    prompt_rows: list[dict[str, Any]],
    protocol_binding: dict[str, Any],
    source_report: dict[str, Any],
    formal_execution_lock: dict[str, Any],
) -> dict[str, Any]:
    """构造官方源码逐 Prompt 恢复所需的静态完整契约."""

    source_identity = {
        field_name: source_report[field_name]
        for field_name in (
            "official_repository_commit",
            "protocol_patch_sha256",
            "source_worktree_exact",
            "source_worktree_digest",
            "patched_source_sha256",
        )
    }
    paper_run_identity = {
        "run_name": paper_run.run_name,
        "protocol_profile": paper_run.protocol_profile,
        "prompt_set": paper_run.prompt_set,
        "prompt_count": int(paper_run.prompt_count),
        "sample_count": int(paper_run.sample_count),
        "target_fpr": float(paper_run.target_fpr),
    }
    formal_reproduction_config = asdict(config)
    for operational_field in (
        "output_dir",
        "drive_output_dir",
        "prompt_file",
        "reuse_existing",
        "force_generate",
        "require_cuda",
        "timeout_seconds",
        "enable_workflow_progress_bar",
    ):
        formal_reproduction_config.pop(operational_field, None)
    return build_t2smark_formal_unit_contract(
        formal_reproduction_config=formal_reproduction_config,
        paper_run_identity=paper_run_identity,
        prompt_rows=prompt_rows,
        prompt_plan_digest=build_stable_digest(prompt_rows),
        protocol_binding=protocol_binding,
        source_identity=source_identity,
        formal_execution_lock=formal_execution_lock,
    )


def write_t2smark_formal_protocol_binding(
    path: Path,
    expected_binding: dict[str, Any],
    *,
    results_path: Path,
    settings_path: Path,
) -> dict[str, Any]:
    """把 canonical 协议与本次官方输出摘要绑定后落盘。"""

    if not results_path.is_file() or not settings_path.is_file():
        raise FileNotFoundError("T2SMark 官方结果或 settings 缺失, 无法写出协议绑定")
    payload = {
        **expected_binding,
        "official_results_sha256": file_digest(results_path),
        "official_settings_sha256": file_digest(settings_path),
    }
    payload["protocol_evidence_digest"] = build_stable_digest(payload)
    write_json(path, payload)
    return payload


def validate_t2smark_formal_protocol_binding(
    path: Path,
    expected_binding: dict[str, Any],
    *,
    results_path: Path,
    settings_path: Path,
) -> dict[str, Any]:
    """核验复用结果的协议字段及结果、settings 字节摘要。"""

    if not path.is_file() or not results_path.is_file() or not settings_path.is_file():
        raise FileNotFoundError("T2SMark 复用所需协议绑定、结果或 settings 缺失")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise TypeError("T2SMark 正式协议绑定必须是 JSON object")
    for field_name, expected_value in expected_binding.items():
        if payload.get(field_name) != expected_value:
            raise ValueError(f"T2SMark 复用协议字段不一致: {field_name}")
    if payload.get("official_results_sha256") != file_digest(results_path):
        raise ValueError("T2SMark 复用 results 摘要不一致")
    if payload.get("official_settings_sha256") != file_digest(settings_path):
        raise ValueError("T2SMark 复用 settings 摘要不一致")
    evidence_digest = str(payload.get("protocol_evidence_digest", ""))
    digest_payload = dict(payload)
    digest_payload.pop("protocol_evidence_digest", None)
    if evidence_digest != build_stable_digest(digest_payload):
        raise ValueError("T2SMark 复用协议 evidence digest 不一致")
    return dict(payload)


def should_run_official(
    config: T2SMarkFormalReproductionConfig,
    results_path: Path,
    *,
    protocol_binding_path: Path,
    settings_path: Path,
    expected_protocol_binding: dict[str, Any],
    formal_unit_set_ready: bool | None = None,
    expected_formal_attack_sample_count: int | None = None,
) -> tuple[bool, str]:
    """仅在 canonical 协议、源码和结果摘要全部一致时复用官方结果。"""

    if config.force_generate:
        return True, "force_generate_requested"
    if formal_unit_set_ready is False:
        return True, "formal_prompt_units_incomplete"
    if config.reuse_existing and results_path.is_file():
        try:
            validate_t2smark_formal_protocol_binding(
                protocol_binding_path,
                expected_protocol_binding,
                results_path=results_path,
                settings_path=settings_path,
            )
        except (FileNotFoundError, TypeError, ValueError):
            return True, "existing_results_protocol_binding_mismatch"
        attack_names = configured_attack_names(config.formal_attack_families)
        required_attack_count = (
            max(1, int(config.prompt_limit))
            if expected_formal_attack_sample_count is None
            else int(expected_formal_attack_sample_count)
        )
        if attack_names and count_t2smark_formal_attack_items(
            results_path,
            attack_names,
        ) != required_attack_count:
            return True, "existing_results_formal_attack_count_mismatch"
        if config.save_clean_pair and count_t2smark_pair_quality_items(results_path) != max(1, int(config.prompt_limit)):
            return True, "existing_results_pair_quality_count_mismatch"
        return False, "existing_results_found"
    return True, "results_missing"


def count_t2smark_pair_quality_items(results_path: Path) -> int:
    """统计 T2SMark formal 结果中严格 pair-level 质量证据数量。"""

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
    config: T2SMarkFormalReproductionConfig,
    paths: dict[str, Path],
    prompt_report: dict[str, Any],
    prompt_rows: list[dict[str, Any]],
    environment_report: dict[str, Any],
    formal_execution_lock: dict[str, Any],
    progress: object | None = None,
) -> dict[str, Any]:
    """运行或复用 T2SMark 官方 SD3.5 Medium formal 结果。"""

    paths["official_root"].mkdir(parents=True, exist_ok=True)
    source_report = ensure_t2smark_source_available(root_path, paths, timeout_seconds=300, progress=progress)
    paper_run = build_paper_run_config(root_path)
    expected_binding = build_t2smark_formal_protocol_binding(
        config,
        paper_run=paper_run,
        prompt_report=prompt_report,
        source_report=source_report,
    )
    resolved_run_dir = paths["official_run_dir"].resolve()
    resolved_official_root = paths["official_root"].resolve()
    resolved_run_dir.relative_to(resolved_official_root)
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    expected_contract = build_t2smark_formal_checkpoint_contract(
        config,
        paper_run=paper_run,
        prompt_rows=prompt_rows,
        protocol_binding=expected_binding,
        source_report=source_report,
        formal_execution_lock=formal_execution_lock,
    )
    unit_contract = write_or_validate_t2smark_formal_unit_contract(
        paths["official_unit_contract"],
        expected_contract,
    )
    unit_records, missing_unit_indices = inspect_t2smark_formal_unit_records(
        paths["official_unit_dir"],
        contract=unit_contract,
        artifact_root=paths["official_run_dir"],
        runtime_environment=environment_report,
    )
    expected_attack_sample_count = sum(
        str(row.get("split", "")) == "test" for row in prompt_rows
    )
    should_run, reason = should_run_official(
        config,
        paths["official_results"],
        protocol_binding_path=paths["official_protocol_binding"],
        settings_path=paths["official_settings"],
        expected_protocol_binding=expected_binding,
        formal_unit_set_ready=not missing_unit_indices,
        expected_formal_attack_sample_count=expected_attack_sample_count,
    )
    if not should_run:
        rebuilt_results, unit_aggregate = aggregate_t2smark_formal_unit_records(
            unit_records,
            contract=unit_contract,
            artifact_root=paths["official_run_dir"],
            runtime_environment=environment_report,
        )
        official_payload = read_json(paths["official_results"])
        if not isinstance(official_payload, dict) or official_payload != (
            _rebuild_t2smark_official_payload(rebuilt_results, unit_aggregate)
        ):
            raise RuntimeError("T2SMark 官方结果与逐 Prompt 单元聚合不一致")
        protocol_binding = validate_t2smark_formal_protocol_binding(
            paths["official_protocol_binding"],
            expected_binding,
            results_path=paths["official_results"],
            settings_path=paths["official_settings"],
        )
        reuse_result = {
            "command": [],
            "return_code": 0,
            "stdout": "",
            "stderr": "",
            "official_result_reused": True,
            "protocol_binding_digest": expected_binding["protocol_binding_digest"],
        }
        write_json(paths["official_command_result"], reuse_result)
        return {
            "official_result_generated": False,
            "official_result_reused": True,
            "official_generation_reason": reason,
            "official_results_path": relative_or_absolute(paths["official_results"], root_path),
            "official_return_code": 0,
            "official_command": [],
            "source_report": source_report,
            "official_protocol_binding_ready": True,
            "official_protocol_binding": protocol_binding,
            "official_protocol_binding_path": relative_or_absolute(paths["official_protocol_binding"], root_path),
            "formal_unit_aggregate": unit_aggregate,
            "formal_unit_contract": unit_contract,
        }

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
        "--slm_watermark_seed",
        str(config.watermark_key_seed_random),
        "--robust_test_num",
        str(prompt_report["selected_prompt_count"]),
        "--clip_test_num",
        str(config.clip_test_num),
        "--dataset_key",
        str(paths["prompt_dataset"]),
        "--model_key",
        config.model_id,
        "--model_revision",
        config.model_revision,
        "--guidance_scale",
        str(config.guidance_scale),
        "--num_inference_steps",
        str(config.num_inference_steps),
        "--num_inversion_steps",
        str(config.num_inversion_steps),
        "--fix_key",
        "--SDv35M",
        "--slm_unit_contract",
        str(paths["official_unit_contract"]),
        "--slm_runtime_environment_report",
        str(paths["environment_report"]),
    ]
    if config.save_clean_pair:
        command.extend(
            [
                "--slm_save_clean_pair",
                "--slm_pair_image_dir",
                str(paths["official_run_dir"] / "quality_pairs"),
            ]
        )
    if str(config.formal_attack_families).strip():
        command.extend(
            [
                "--slm_attack_families",
                str(config.formal_attack_families),
                "--slm_attack_image_dir",
                str(paths["official_run_dir"] / "formal_attacks"),
            ]
        )
    if config.save_image:
        command.append("--save_image")
    result = run_command_with_progress_status(
        command,
        cwd=root_path,
        timeout_seconds=config.timeout_seconds,
        progress=progress,
        progress_profile=f"operation=t2smark_formal_official_reference samples={prompt_report['selected_prompt_count']}",
    )
    write_json(paths["official_command_result"], result)
    if int(result["return_code"]) != 0:
        raise RuntimeError("T2SMark 官方 formal 命令执行失败")
    unit_records, missing_unit_indices = inspect_t2smark_formal_unit_records(
        paths["official_unit_dir"],
        contract=unit_contract,
        artifact_root=paths["official_run_dir"],
        runtime_environment=environment_report,
    )
    if missing_unit_indices:
        raise RuntimeError("T2SMark 官方命令返回成功但仍缺少逐 Prompt 完成单元")
    rebuilt_results, unit_aggregate = aggregate_t2smark_formal_unit_records(
        unit_records,
        contract=unit_contract,
        artifact_root=paths["official_run_dir"],
        runtime_environment=environment_report,
    )
    official_payload = read_json(paths["official_results"])
    if not isinstance(official_payload, dict) or official_payload != (
        _rebuild_t2smark_official_payload(rebuilt_results, unit_aggregate)
    ):
        raise RuntimeError("T2SMark 官方结果未由完整逐 Prompt 单元确定性重建")
    protocol_binding = write_t2smark_formal_protocol_binding(
        paths["official_protocol_binding"],
        expected_binding,
        results_path=paths["official_results"],
        settings_path=paths["official_settings"],
    )
    return {
        "official_result_generated": True,
        "official_result_reused": False,
        "official_generation_reason": reason,
        "official_results_path": relative_or_absolute(paths["official_results"], root_path),
        "official_return_code": 0,
        "official_command": command,
        "source_report": source_report,
        "official_protocol_binding_ready": True,
        "official_protocol_binding": protocol_binding,
        "official_protocol_binding_path": relative_or_absolute(paths["official_protocol_binding"], root_path),
        "formal_unit_aggregate": unit_aggregate,
        "formal_unit_contract": unit_contract,
    }


def build_t2smark_formal_image_pairs(
    root_path: Path,
    paths: dict[str, Path],
    prompt_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按 当前论文运行层级的完整 Prompt 计划和官方图像目录生成 image_pairs。"""

    official_payload = read_json(paths["official_results"])
    if not isinstance(official_payload, dict):
        raise TypeError("T2SMark 官方结果必须是 JSON 对象")
    rows: list[dict[str, Any]] = []
    for index, prompt_row in enumerate(prompt_rows):
        result = official_payload.get(str(index), official_payload.get(index))
        if not isinstance(result, dict) or not isinstance(
            result.get("pair_quality"),
            dict,
        ):
            raise ValueError("T2SMark 官方结果缺少严格配对随机身份")
        pair_random_identity = dict(result["pair_quality"])
        randomization_fields = (
            "randomization_repeat_id",
            "generation_seed_index",
            "generation_seed_offset",
            "generation_seed_random",
            "watermark_key_index",
            "watermark_key_seed_random",
            "watermark_key_material_digest_random",
            "formal_randomization_protocol_digest",
            "formal_randomization_identity_digest_random",
            "base_latent_content_digest_random",
            "base_latent_identity_digest_random",
        )
        if any(
            field_name not in pair_random_identity
            for field_name in randomization_fields
        ):
            raise ValueError("T2SMark 严格配对结果缺少正式随机化字段")
        for field_name in randomization_fields[:9]:
            if pair_random_identity[field_name] != prompt_row[field_name]:
                raise ValueError("T2SMark 严格配对结果与 Prompt 随机化计划不一致")
        image_path = paths["official_images"] / f"{index:05d}.png"
        clean_image_path = paths["official_run_dir"] / "quality_pairs" / "clean" / f"{index:05d}.png"
        image_id = f"t2smark_formal_{index:05d}"
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
                **{
                    field_name: pair_random_identity[field_name]
                    for field_name in randomization_fields
                },
            }
        )
    write_json(paths["image_pairs"], rows)
    return rows


def run_t2smark_adapter(
    root_path: Path,
    config: T2SMarkFormalReproductionConfig,
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
        "--model-revision",
        config.model_revision,
        "--seed",
        str(config.seed),
        "--target-fpr",
        str(config.target_fpr),
        "--num-inference-steps",
        str(config.num_inference_steps),
        "--num-inversion-steps",
        str(config.num_inversion_steps),
        "--guidance-scale",
        str(config.guidance_scale),
    ]
    if config.require_cuda:
        command.append("--require-cuda")
    result = run_command_with_progress_status(
        command,
        cwd=root_path,
        timeout_seconds=config.timeout_seconds,
        progress=progress,
        progress_profile="operation=t2smark_formal_adapter",
    )
    write_json(paths["adapter_command_result"], result)
    return {
        "adapter_return_code": result["return_code"],
        "adapter_command": command,
        "adapter_observations_path": relative_or_absolute(paths["adapter_observations"], root_path),
    }


def build_candidate_records_and_validation(
    root_path: Path,
    config: T2SMarkFormalReproductionConfig,
    paths: dict[str, Path],
    prompt_report: dict[str, Any],
) -> dict[str, Any]:
    """从 T2SMark formal observations 构造正式导入候选并运行 schema 校验。"""

    observations = read_json(paths["adapter_observations"]) if paths["adapter_observations"].is_file() else []
    prompt_rows = read_json(paths["prompt_plan"])
    expected_calibration_count = sum(
        row.get("split") == "calibration" for row in prompt_rows
    )
    threshold_audit = audit_fixed_fpr_observation_threshold(
        observations,
        target_fpr=config.target_fpr,
        expected_calibration_negative_count=expected_calibration_count,
    )
    fixed_fpr_ready = threshold_audit.fixed_fpr_ready
    from experiments.protocol.attacks import default_attack_configs

    required_attack_names = {
        attack.attack_name
        for attack in default_attack_configs()
        if attack.enabled and attack.resource_profile in {"full_main", "full_extra"}
    }
    actual_attack_names = {
        str(row.get("attack_name") or row.get("attack_condition"))
        for row in observations
        if str(row.get("sample_role", "")).startswith("attacked_")
    }
    evidence_paths = [
        relative_or_absolute(paths["official_results"], root_path),
        relative_or_absolute(paths["image_pairs"], root_path),
        relative_or_absolute(paths["adapter_observations"], root_path),
        relative_or_absolute(paths["prompt_dataset"], root_path),
        relative_or_absolute(paths["prompt_plan"], root_path),
    ]
    candidate_records = build_t2smark_formal_candidate_records(
        observation_rows=observations,
        target_fpr=config.target_fpr,
        baseline_result_source=relative_or_absolute(paths["official_results"], root_path),
        baseline_result_source_digest=file_digest(paths["official_results"]) if paths["official_results"].is_file() else "",
        evidence_paths=evidence_paths,
        prompt_protocol_digest=str(prompt_report["prompt_protocol_digest"]),
        paper_run_prompt_protocol_ready=bool(prompt_report["paper_run_prompt_protocol_ready"]),
        fixed_fpr_baseline_calibration_ready=fixed_fpr_ready,
        attack_matrix_baseline_detection_ready=actual_attack_names == required_attack_names,
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
        "threshold_digest": threshold_audit.threshold_digest,
        "validation_report": validation_report,
        "formal_import_candidate_records_path": relative_or_absolute(paths["candidate_records"], root_path),
        "formal_import_validation_report_path": relative_or_absolute(paths["validation_report"], root_path),
    }


def write_failure_outputs(
    root_path: Path,
    config: T2SMarkFormalReproductionConfig,
    paths: dict[str, Path],
    error: Exception,
) -> dict[str, Any]:
    """在 formal 复现失败时保留诊断产物并阻断正式归档。"""

    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    environment_report = build_runtime_environment_report("t2smark_sd35_gpu")
    write_json(paths["environment_report"], environment_report)
    summary = {
        "run_decision": "fail",
        "t2smark_formal_reproduction_ready": False,
        "formal_import_validation_ready": False,
        "unsupported_reason": f"{type(error).__name__}:{error}",
        "supports_paper_claim": False,
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
        "manifest_path": relative_or_absolute(paths["manifest"], root_path),
    }
    write_json(paths["summary"], summary)
    manifest = build_artifact_manifest(
        artifact_id="t2smark_formal_reproduction_manifest",
        artifact_type="local_manifest",
        input_paths=(),
        output_paths=(relative_or_absolute(paths["summary"], root_path), relative_or_absolute(paths["environment_report"], root_path)),
        config=asdict(config),
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 paper_experiments.runners.t2smark_formal_reproduction",
        metadata={"run_decision": "fail", "supports_paper_claim": False},
    ).to_dict()
    write_json(paths["manifest"], manifest)
    return summary


def build_t2smark_formal_run_readiness(
    *,
    dependency_environment_ready: bool,
    official_ready: bool,
    adapter_ready: bool,
    prompt_ready: bool,
    pair_quality_ready: bool,
    formal_attack_ready: bool,
    formal_import_validation_ready: bool,
    formal_unit_set_ready: bool,
) -> bool:
    """集中判定 T2SMark 运行是否达到可打包的正式边界。"""

    return all(
        (
            dependency_environment_ready,
            official_ready,
            adapter_ready,
            prompt_ready,
            pair_quality_ready,
            formal_attack_ready,
            formal_import_validation_ready,
            formal_unit_set_ready,
        )
    )


def write_t2smark_formal_reproduction_outputs(
    config: T2SMarkFormalReproductionConfig,
    root: str | Path = ".",
) -> dict[str, Any]:
    """运行 T2SMark formal 真实复现路径并写出 summary 与 manifest。"""

    root_path = Path(root).resolve()
    formal_execution_run_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    paths = output_paths(root_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    try:
        paper_run = validate_t2smark_formal_protocol_config(config, root_path=root_path)
        with progress_bar(8, desc="t2smark formal reproduction", enabled=config.enable_workflow_progress_bar) as run_progress:
            emit_progress_status(run_progress, profile="operation=ensure_cuda status=running")
            device_report = ensure_cuda_if_requested(config.require_cuda)
            update_progress(run_progress, profile="operation=ensure_cuda")
            emit_progress_status(run_progress, profile="operation=write_environment_report status=running")
            environment_report = build_t2smark_formal_environment_report(
                device_report,
                verified_formal_execution_lock=formal_execution_run_lock,
            )
            write_json(paths["environment_report"], environment_report)
            if environment_report["dependency_environment_ready"] is not True:
                blockers = ",".join(
                    environment_report["dependency_readiness_blockers"]
                )
                raise RuntimeError(
                    f"dependency_profile_environment_not_ready:{blockers}"
                )
            update_progress(run_progress, profile="operation=write_environment_report")
            emit_progress_status(run_progress, profile="operation=write_prompt_inputs status=running")
            prompt_report = write_paper_run_prompt_inputs(root_path, config, paths)
            if not bool(prompt_report["paper_run_prompt_protocol_ready"]) or int(
                prompt_report["selected_prompt_count"]
            ) != int(paper_run.prompt_count):
                raise RuntimeError("T2SMark canonical Prompt 协议未完整物化")
            prompt_rows = read_json(paths["prompt_plan"])
            update_progress(run_progress, profile=f"operation=write_prompt_inputs prompts={prompt_report['selected_prompt_count']}")
            official_report = run_t2smark_official_if_needed(
                root_path,
                config,
                paths,
                prompt_report,
                prompt_rows,
                environment_report,
                formal_execution_run_lock,
                progress=run_progress,
            )
            update_progress(run_progress, profile="operation=t2smark_formal_official_reference")
            emit_progress_status(run_progress, profile="operation=build_image_pairs status=running")
            image_pairs = build_t2smark_formal_image_pairs(root_path, paths, prompt_rows)
            update_progress(run_progress, profile=f"operation=build_image_pairs pairs={len(image_pairs)}")
            emit_progress_status(run_progress, profile="operation=t2smark_pair_quality status=running")
            pair_quality_report = write_t2smark_strict_pair_quality_outputs(
                root_path=root_path,
                image_pairs_path=paths["image_pairs"],
                metrics_path=paths["pair_quality_metrics"],
                summary_path=paths["pair_quality_summary"],
                enable_pair_perceptual_metrics=config.enable_pair_perceptual_metrics,
                clip_model_id=config.pair_clip_model_id,
                clip_model_revision=config.pair_clip_model_revision,
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
            update_progress(run_progress, profile="operation=t2smark_formal_adapter")
            emit_progress_status(run_progress, profile="operation=build_candidate_records status=running")
            candidate_report = build_candidate_records_and_validation(root_path, config, paths, prompt_report)
            update_progress(run_progress, profile=f"operation=build_candidate_records records={candidate_report['candidate_record_count']}")
    except Exception as error:
        return write_failure_outputs(root_path, config, paths, error)

    validation_report = candidate_report["validation_report"]
    formal_import_ready = bool(validation_report.get("formal_import_validation_ready", False))
    official_ready = bool(
        paths["official_results"].is_file()
        and official_report.get("official_return_code") == 0
        and official_report.get("official_protocol_binding_ready")
        and paths["official_protocol_binding"].is_file()
    )
    formal_unit_aggregate = dict(official_report.get("formal_unit_aggregate", {}))
    formal_unit_set_ready = bool(
        formal_unit_aggregate.get("formal_unit_set_complete") is True
        and formal_unit_aggregate.get("scientific_unit_provenance_ready") is True
        and int(formal_unit_aggregate.get("formal_unit_record_count", 0))
        == int(prompt_report["selected_prompt_count"])
    )
    adapter_ready = paths["adapter_observations"].is_file() and adapter_report.get("adapter_return_code") == 0
    pair_quality_ready = bool(pair_quality_report.get("strict_pair_quality_ready", False))
    formal_attack_names = configured_attack_names(config.formal_attack_families)
    formal_attack_result_count = count_t2smark_formal_attack_items(paths["official_results"], formal_attack_names)
    expected_formal_attack_sample_count = sum(
        str(row.get("split", "")) == "test" for row in prompt_rows
    )
    formal_attack_ready = bool(formal_attack_names) and formal_attack_result_count == int(
        expected_formal_attack_sample_count
    )
    run_ready = build_t2smark_formal_run_readiness(
        dependency_environment_ready=bool(
            environment_report.get("dependency_environment_ready")
        ),
        official_ready=official_ready,
        adapter_ready=adapter_ready,
        prompt_ready=prompt_report["selected_prompt_count"] > 0,
        pair_quality_ready=not config.save_clean_pair or pair_quality_ready,
        formal_attack_ready=formal_attack_ready,
        formal_import_validation_ready=formal_import_ready,
        formal_unit_set_ready=formal_unit_set_ready,
    )
    protocol_binding = dict(official_report.get("official_protocol_binding", {}))
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_id": "t2smark",
        "run_decision": "pass" if run_ready else "fail",
        "t2smark_formal_reproduction_ready": run_ready,
        "t2smark_official_result_generated": bool(official_report.get("official_result_generated")),
        "t2smark_official_result_reused": bool(official_report.get("official_result_reused")),
        "paper_run_prompt_count": int(prompt_report["paper_run_prompt_count"]),
        "paper_prompt_count": int(prompt_report["paper_run_prompt_count"]),
        "selected_prompt_count": int(prompt_report["selected_prompt_count"]),
        "prompt_limit": int(config.prompt_limit),
        "paper_run_prompt_protocol_ready": bool(prompt_report["paper_run_prompt_protocol_ready"]),
        "t2smark_formal_attack_names": list(formal_attack_names),
        "t2smark_formal_attack_result_count": formal_attack_result_count,
        "t2smark_formal_attack_expected_test_count": expected_formal_attack_sample_count,
        "t2smark_formal_attack_ready": formal_attack_ready,
        "t2smark_formal_unit_set_ready": formal_unit_set_ready,
        "t2smark_formal_unit_record_count": int(
            formal_unit_aggregate.get("formal_unit_record_count", 0)
        ),
        "t2smark_formal_unit_records_digest": str(
            formal_unit_aggregate.get("formal_unit_records_digest", "")
        ),
        "t2smark_formal_unit_aggregate_digest": str(
            formal_unit_aggregate.get("formal_unit_aggregate_digest", "")
        ),
        "scientific_unit_provenance_ready": bool(
            formal_unit_aggregate.get("scientific_unit_provenance_ready", False)
        ),
        "image_pair_count": len(image_pairs),
        "t2smark_strict_pair_quality_ready": pair_quality_ready,
        "t2smark_strict_pair_quality_count": int(pair_quality_report.get("measured_strict_pair_quality_count", 0)),
        "t2smark_strict_pair_quality_metrics_path": relative_or_absolute(paths["pair_quality_metrics"], root_path),
        "t2smark_strict_pair_quality_summary_path": relative_or_absolute(paths["pair_quality_summary"], root_path),
        "formal_import_candidate_record_count": int(candidate_report["candidate_record_count"]),
        "accepted_formal_import_count": int(validation_report.get("accepted_formal_import_count", 0)),
        "rejected_formal_import_count": int(validation_report.get("rejected_formal_import_count", 0)),
        "formal_import_validation_ready": bool(validation_report.get("formal_import_validation_ready", False)),
        "official_protocol_binding_ready": bool(official_report.get("official_protocol_binding_ready", False)),
        "official_protocol_binding_digest": str(protocol_binding.get("protocol_binding_digest", "")),
        "official_protocol_evidence_digest": str(protocol_binding.get("protocol_evidence_digest", "")),
        "official_protocol_binding_path": relative_or_absolute(paths["official_protocol_binding"], root_path),
        "target_fpr": float(config.target_fpr),
        "comparable_operating_point": build_fixed_fpr_operating_point(config.target_fpr),
        "official_results_path": relative_or_absolute(paths["official_results"], root_path),
        "image_pairs_path": relative_or_absolute(paths["image_pairs"], root_path),
        "formal_import_candidate_records_path": candidate_report["formal_import_candidate_records_path"],
        "formal_import_validation_report_path": candidate_report["formal_import_validation_report_path"],
        "environment_report_path": relative_or_absolute(paths["environment_report"], root_path),
        "manifest_path": relative_or_absolute(paths["manifest"], root_path),
        "supports_paper_claim": False,
        "unsupported_reason": "" if run_ready else "t2smark_formal_reproduction_incomplete",
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
        relative_or_absolute(paths["official_protocol_binding"], root_path),
        relative_or_absolute(paths["official_unit_contract"], root_path),
        relative_or_absolute(paths["source_prepare_result"], root_path),
        relative_or_absolute(paths["official_command_result"], root_path),
        relative_or_absolute(paths["adapter_command_result"], root_path),
        relative_or_absolute(paths["adapter_manifest"], root_path),
        relative_or_absolute(paths["adapter_artifact_manifest"], root_path),
    ]
    if paths["adapter_observations"].exists():
        output_paths_for_manifest.append(relative_or_absolute(paths["adapter_observations"], root_path))
    formal_execution_run_lock = repository_environment.validate_formal_execution_lock_pair(
        formal_execution_run_lock,
        repository_environment.require_published_formal_execution_lock(root_path),
        formal_execution_run_lock["formal_execution_commit"],
    )
    manifest = build_artifact_manifest(
        artifact_id="t2smark_formal_reproduction_manifest",
        artifact_type="local_manifest",
        input_paths=(relative_or_absolute(root_path / config.prompt_file, root_path),),
        output_paths=tuple(output_paths_for_manifest + [relative_or_absolute(paths["manifest"], root_path)]),
        config=asdict(config),
        code_version=formal_execution_run_lock["formal_execution_commit"],
        rebuild_command="调用 paper_experiments.runners.t2smark_formal_reproduction",
        metadata={
            "run_decision": summary["run_decision"],
            "t2smark_formal_reproduction_ready": run_ready,
            "formal_import_validation_ready": summary["formal_import_validation_ready"],
            "t2smark_strict_pair_quality_ready": pair_quality_ready,
            "t2smark_strict_pair_quality_count": summary["t2smark_strict_pair_quality_count"],
            "t2smark_formal_unit_set_ready": formal_unit_set_ready,
            "t2smark_formal_unit_records_digest": summary[
                "t2smark_formal_unit_records_digest"
            ],
            "supports_paper_claim": False,
        },
    ).to_dict()
    manifest["formal_execution_run_lock"] = formal_execution_run_lock
    write_json(paths["manifest"], manifest)
    return summary


def build_default_config() -> T2SMarkFormalReproductionConfig:
    """从环境变量构造默认 Colab 运行配置。"""

    paper_run = build_paper_run_config(".")
    default_run_name = f"t2smark_sd35_medium_{paper_run.run_name}"
    repeat = resolve_formal_randomization_repeat(
        paper_run.randomization_repeat_id
    )
    root_key_material = os.environ.get(
        "SLM_WM_KEY_MATERIAL",
        "slm_wm_paper_key",
    )
    require_formal_watermark_key_plan(root_key_material)
    return T2SMarkFormalReproductionConfig(
        output_dir=os.environ.get("SLM_WM_T2SMARK_FORMAL_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        drive_output_dir=os.environ.get(
            "SLM_WM_T2SMARK_FORMAL_DRIVE_OUTPUT_DIR",
            paper_run.drive_dir("external_baseline_official_reference"),
        ),
        prompt_set=os.environ.get("SLM_WM_PROMPT_SET", paper_run.prompt_set),
        prompt_file=os.environ.get("SLM_WM_T2SMARK_FORMAL_PROMPT_FILE", paper_run.prompt_file),
        t2smark_run_name=os.environ.get(
            "SLM_WM_T2SMARK_FORMAL_RUN_NAME",
            default_run_name,
        ),
        model_id=os.environ.get("SLM_WM_T2SMARK_MODEL_ID", DEFAULT_T2SMARK_MODEL_ID),
        model_revision=os.environ.get("SLM_WM_T2SMARK_MODEL_REVISION", DEFAULT_T2SMARK_MODEL_REVISION),
        seed=int(
            os.environ.get(
                "SLM_WM_T2SMARK_FORMAL_SEED",
                str(DEFAULT_T2SMARK_SEED + repeat.generation_seed_offset),
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
        prompt_limit=resolve_count_from_environment(
            "SLM_WM_T2SMARK_FORMAL_PROMPT_LIMIT",
            default_value=paper_run.sample_count,
        ),
        clip_test_num=0,
        num_inference_steps=int(
            os.environ.get("SLM_WM_T2SMARK_FORMAL_NUM_INFERENCE_STEPS", str(paper_run.inference_steps))
        ),
        num_inversion_steps=int(
            os.environ.get("SLM_WM_T2SMARK_FORMAL_NUM_INVERSION_STEPS", str(paper_run.inference_steps))
        ),
        guidance_scale=float(
            os.environ.get("SLM_WM_T2SMARK_FORMAL_GUIDANCE_SCALE", str(paper_run.guidance_scale))
        ),
        target_fpr=float(os.environ.get("SLM_WM_T2SMARK_FORMAL_TARGET_FPR", str(paper_run.target_fpr))),
        minimum_prompt_protocol_count=paper_run.prompt_count,
        reuse_existing=os.environ.get("SLM_WM_T2SMARK_FORMAL_REUSE_EXISTING", "1") != "0",
        force_generate=os.environ.get("SLM_WM_T2SMARK_FORMAL_FORCE_GENERATE", "0") == "1",
        save_image=os.environ.get("SLM_WM_T2SMARK_FORMAL_SAVE_IMAGE", "1") != "0",
        save_clean_pair=os.environ.get("SLM_WM_T2SMARK_FORMAL_SAVE_CLEAN_PAIR", "1") != "0",
        formal_attack_families=os.environ.get(
            "SLM_WM_T2SMARK_FORMAL_ATTACK_FAMILIES",
            DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES,
        ),
        enable_pair_perceptual_metrics=os.environ.get("SLM_WM_T2SMARK_FORMAL_PAIR_PERCEPTUAL_METRICS", "1") != "0",
        pair_clip_model_id=os.environ.get("SLM_WM_T2SMARK_FORMAL_PAIR_CLIP_MODEL_ID", DEFAULT_T2SMARK_CLIP_MODEL_ID),
        pair_clip_model_revision=os.environ.get(
            "SLM_WM_T2SMARK_FORMAL_PAIR_CLIP_MODEL_REVISION",
            DEFAULT_CLIP_MODEL_REVISION,
        ),
        pair_lpips_network=os.environ.get("SLM_WM_T2SMARK_FORMAL_PAIR_LPIPS_NETWORK", DEFAULT_T2SMARK_LPIPS_NETWORK),
        pair_perceptual_metric_device_name=os.environ.get("SLM_WM_T2SMARK_FORMAL_PAIR_PERCEPTUAL_DEVICE", "cpu"),
        require_cuda=os.environ.get("SLM_WM_T2SMARK_FORMAL_REQUIRE_CUDA", "1") != "0",
        timeout_seconds=int(os.environ.get("SLM_WM_T2SMARK_FORMAL_TIMEOUT_SECONDS", "86400")),
        enable_workflow_progress_bar=os.environ.get("SLM_WM_ENABLE_WORKFLOW_PROGRESS_BAR", "1") != "0",
    )


def run_default_t2smark_formal_reproduction_plan(root: str | Path = ".") -> dict[str, Any]:
    """运行默认 T2SMark formal 真实复现计划。"""

    return write_t2smark_formal_reproduction_outputs(config=build_default_config(), root=root)


def _read_json_array(path: Path, label: str) -> list[dict[str, Any]]:
    """读取打包门禁所需 JSON 数组。"""

    if not path.is_file():
        raise FileNotFoundError(f"T2SMark 打包缺少 {label}: {path}")
    payload = read_json(path)
    if not isinstance(payload, list):
        raise TypeError(f"T2SMark {label} 必须是 JSON 数组")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _read_jsonl_count(path: Path, label: str) -> int:
    """读取 JSONL 并返回非空记录数。"""

    if not path.is_file():
        raise FileNotFoundError(f"T2SMark 打包缺少 {label}: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"T2SMark {label} 必须只包含 JSON object")
    return len(rows)


def _read_jsonl_rows(path: Path, label: str) -> list[dict[str, Any]]:
    """读取打包门禁所需 JSONL 对象, 并保留正式顺序."""

    if not path.is_file():
        raise FileNotFoundError(f"T2SMark 打包缺少 {label}: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"T2SMark {label} 必须只包含 JSON object")
    return [dict(row) for row in rows]


def _rebuild_t2smark_official_payload(
    results: dict[str, Any],
    unit_aggregate: dict[str, Any],
) -> dict[str, Any]:
    """从逐 Prompt 单元重建官方 results.json 的全部正式派生字段."""

    ordered_indices = tuple(range(len(results)))
    if set(results) != {str(index) for index in ordered_indices}:
        raise RuntimeError("T2SMark 单元聚合结果索引不是连续 exact set")
    ordered_results = [dict(results[str(index)]) for index in ordered_indices]
    no_watermark_scores = [
        float(row["robustness"]["norm1_no_w"]) for row in ordered_results
    ]
    watermarked_scores = [
        float(row["robustness"]["norm1_w"]) for row in ordered_results
    ]
    bit_accuracies = [
        float(row["robustness"]["acc_msg"]) for row in ordered_results
    ]
    if not no_watermark_scores or not watermarked_scores:
        raise RuntimeError("T2SMark 官方派生结果缺少完整正负连续分数")
    maximum_negative = max(no_watermark_scores)
    zero_fpr_tpr = sum(
        score > maximum_negative for score in watermarked_scores
    ) / len(watermarked_scores)
    payload = {str(index): ordered_results[index] for index in ordered_indices}
    payload["tpr"] = zero_fpr_tpr
    payload["bit_accuracy"] = sum(bit_accuracies) / len(bit_accuracies)
    payload["slm_formal_unit_aggregate"] = dict(unit_aggregate)
    return payload


def _expected_indexed_pngs(directory: Path, count: int) -> set[Path]:
    """构造从0开始的完整 Prompt 图像文件集合。"""

    return {directory / f"{index:05d}.png" for index in range(count)}


def _expected_attack_pngs(
    directory: Path,
    prompt_indices: tuple[int, ...],
    attack_names: tuple[str, ...],
) -> set[Path]:
    """构造每个 Prompt、每种攻击和正负角色的完整图像集合。"""

    return {
        directory / f"{index:05d}_{attack_name}_{sample_role}.png"
        for index in prompt_indices
        for attack_name in attack_names
        for sample_role in ("attacked_negative", "attacked_positive")
    }


def _path_is_link_or_reparse(path: Path) -> bool:
    """不跟随目标地识别符号链接、junction 和 Windows reparse point。"""

    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if is_junction is not None and is_junction():
        return True
    try:
        file_attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except FileNotFoundError:
        return False
    reparse_attribute = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(reparse_attribute and file_attributes & reparse_attribute)


def _inventory_t2smark_result_tree(
    output_dir: Path,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """使用 lstat 枚举普通文件与目录, 拒绝根路径和成员中的链接或特殊文件。"""

    if _path_is_link_or_reparse(output_dir):
        raise RuntimeError("T2SMark 结果根不得是链接或 reparse point")
    try:
        root_mode = output_dir.lstat().st_mode
    except FileNotFoundError as error:
        raise FileNotFoundError("T2SMark 结果根不存在") from error
    if not stat.S_ISDIR(root_mode):
        raise RuntimeError("T2SMark 结果根必须是普通目录")

    files: list[Path] = []
    directories: list[Path] = []
    pending = [output_dir]
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if _path_is_link_or_reparse(path):
                raise RuntimeError("T2SMark 结果目录包含链接或 reparse point")
            mode = path.lstat().st_mode
            if stat.S_ISREG(mode):
                files.append(path)
            elif stat.S_ISDIR(mode):
                directories.append(path)
                pending.append(path)
            else:
                raise RuntimeError("T2SMark 结果目录包含非普通文件")
    return tuple(files), tuple(directories)


def collect_package_entries(root_path: Path, output_dir: Path, archive_path: Path) -> tuple[Path, ...]:
    """校验 pass 运行并返回 T2SMark formal 精确白名单文件。"""

    inventory_files, inventory_directories = _inventory_t2smark_result_tree(
        output_dir
    )
    summary_path = output_dir / "t2smark_formal_reproduction_summary.json"
    validation_path = output_dir / "t2smark_formal_import_validation_report.json"
    manifest_path = output_dir / "t2smark_formal_reproduction_manifest.local.json"
    if not summary_path.is_file() or not validation_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("T2SMark 打包要求运行 summary、formal validation 与运行 manifest")
    summary = read_json(summary_path)
    validation = read_json(validation_path)
    run_manifest = read_json(manifest_path)
    if not isinstance(summary, dict) or not isinstance(validation, dict) or not isinstance(run_manifest, dict):
        raise TypeError("T2SMark 打包治理文件必须是 JSON object")
    if summary.get("run_decision") != "pass" or not bool(summary.get("t2smark_formal_reproduction_ready")):
        raise RuntimeError("失败或不完整的 T2SMark 运行不得打包")
    if not bool(summary.get("formal_import_validation_ready")) or not bool(
        validation.get("formal_import_validation_ready")
    ):
        raise RuntimeError("T2SMark formal import validation 未通过, 不得打包")
    if int(validation.get("accepted_formal_import_count", 0)) <= 0:
        raise RuntimeError("T2SMark formal import 没有 accepted record")
    if int(validation.get("rejected_formal_import_count", 0)) != 0 or int(
        validation.get("formal_import_issue_count", 0)
    ) != 0:
        raise RuntimeError("T2SMark formal import 仍含拒绝记录或校验问题")
    if run_manifest.get("metadata", {}).get("run_decision") != "pass" or not bool(
        run_manifest.get("metadata", {}).get("formal_import_validation_ready")
    ):
        raise RuntimeError("T2SMark 运行 manifest 未绑定 pass validation")

    config_payload = run_manifest.get("config")
    if not isinstance(config_payload, dict):
        raise TypeError("T2SMark 运行 manifest 缺少完整 config")
    config = T2SMarkFormalReproductionConfig(**config_payload)
    paper_run = validate_t2smark_formal_protocol_config(config, root_path=root_path)
    paths = output_paths(root_path, config)
    if paths["output_dir"].resolve() != output_dir.resolve():
        raise RuntimeError("T2SMark 打包目录与运行 manifest config 不一致")

    metadata = summary.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("T2SMark summary 缺少运行 metadata")
    prompt_report = metadata.get("prompt_report")
    official_report = metadata.get("official_report")
    if not isinstance(prompt_report, dict) or not isinstance(official_report, dict):
        raise TypeError("T2SMark summary 缺少 Prompt 或 official report")
    source_report = official_report.get("source_report")
    if not isinstance(source_report, dict):
        raise TypeError("T2SMark official report 缺少 source report")
    expected_binding = build_t2smark_formal_protocol_binding(
        config,
        paper_run=paper_run,
        prompt_report=prompt_report,
        source_report=source_report,
    )
    validate_t2smark_formal_protocol_binding(
        paths["official_protocol_binding"],
        expected_binding,
        results_path=paths["official_results"],
        settings_path=paths["official_settings"],
    )
    if summary.get("official_protocol_binding_digest") != expected_binding["protocol_binding_digest"]:
        raise RuntimeError("T2SMark summary 与 canonical 协议绑定摘要不一致")

    prompt_count = int(summary.get("selected_prompt_count", 0))
    if prompt_count != int(paper_run.prompt_count) or prompt_count <= 0:
        raise RuntimeError("T2SMark 打包 Prompt 数量不是当前论文运行层级完整规模")
    prompt_rows = _read_json_array(paths["prompt_plan"], "Prompt plan")
    image_pairs = _read_json_array(paths["image_pairs"], "image pairs")
    observations = _read_json_array(paths["adapter_observations"], "adapter observations")
    attack_names = configured_attack_names(config.formal_attack_families)
    test_prompt_indices = tuple(
        index
        for index, row in enumerate(prompt_rows)
        if str(row.get("split", "")) == "test"
    )
    expected_observation_count = prompt_count * 2 + len(test_prompt_indices) * 2 * len(
        attack_names
    )
    if len(prompt_rows) != prompt_count or len(image_pairs) != prompt_count:
        raise RuntimeError("T2SMark Prompt plan 或 image pair 数量不完整")
    if len(observations) != expected_observation_count:
        raise RuntimeError("T2SMark adapter observation 数量不等于完整 Prompt 与攻击笛卡尔积")
    candidate_count = _read_jsonl_count(paths["candidate_records"], "formal candidate records")
    if candidate_count != int(summary.get("formal_import_candidate_record_count", -1)):
        raise RuntimeError("T2SMark formal candidate 记录数量与 summary 不一致")
    if int(validation.get("accepted_formal_import_count", 0)) != candidate_count:
        raise RuntimeError("T2SMark accepted formal record 未覆盖全部 candidate")
    pair_quality_summary = read_json(paths["pair_quality_summary"])
    if not isinstance(pair_quality_summary, dict) or not bool(pair_quality_summary.get("strict_pair_quality_ready")):
        raise RuntimeError("T2SMark strict pair quality 未通过")
    if int(pair_quality_summary.get("measured_strict_pair_quality_count", 0)) != prompt_count:
        raise RuntimeError("T2SMark strict pair quality 未覆盖完整 Prompt")
    if count_t2smark_formal_attack_items(
        paths["official_results"], attack_names
    ) != len(test_prompt_indices):
        raise RuntimeError("T2SMark 官方结果未覆盖完整 test Prompt 与攻击集合")
    environment_report = read_json(paths["environment_report"])
    if not isinstance(environment_report, dict):
        raise TypeError("T2SMark 打包环境报告必须是 JSON object")
    formal_execution_lock = run_manifest.get("formal_execution_run_lock")
    if not isinstance(formal_execution_lock, dict):
        raise TypeError("T2SMark 运行 manifest 缺少正式代码锁")
    expected_contract = build_t2smark_formal_checkpoint_contract(
        config,
        paper_run=paper_run,
        prompt_rows=prompt_rows,
        protocol_binding=expected_binding,
        source_report=source_report,
        formal_execution_lock=formal_execution_lock,
    )
    unit_contract = validate_t2smark_formal_unit_contract(
        read_json(paths["official_unit_contract"])
    )
    if unit_contract != expected_contract:
        raise RuntimeError("T2SMark 打包单元契约与正式运行身份不一致")
    unit_records, missing_unit_indices = inspect_t2smark_formal_unit_records(
        paths["official_unit_dir"],
        contract=unit_contract,
        artifact_root=paths["official_run_dir"],
        runtime_environment=environment_report,
    )
    if missing_unit_indices:
        raise RuntimeError("T2SMark 打包缺少逐 Prompt 完成单元")
    rebuilt_results, unit_aggregate = aggregate_t2smark_formal_unit_records(
        unit_records,
        contract=unit_contract,
        artifact_root=paths["official_run_dir"],
        runtime_environment=environment_report,
    )
    official_payload = read_json(paths["official_results"])
    expected_official_payload = _rebuild_t2smark_official_payload(
        rebuilt_results,
        unit_aggregate,
    )
    if not isinstance(official_payload, dict) or official_payload != expected_official_payload:
        raise RuntimeError("T2SMark 打包结果与逐 Prompt 单元聚合不一致")
    if not all(
        (
            summary.get("t2smark_formal_unit_set_ready") is True,
            summary.get("t2smark_formal_unit_records_digest")
            == unit_aggregate["formal_unit_records_digest"],
            summary.get("t2smark_formal_unit_aggregate_digest")
            == unit_aggregate["formal_unit_aggregate_digest"],
        )
    ):
        raise RuntimeError("T2SMark summary 未绑定完整逐 Prompt 单元集合")

    adapter_manifest = read_json(paths["adapter_manifest"])
    adapter_artifact_manifest = read_json(paths["adapter_artifact_manifest"])
    if not isinstance(adapter_manifest, dict) or adapter_artifact_manifest != adapter_manifest:
        raise RuntimeError("T2SMark adapter manifest 双份事实不一致")
    adapter_digest = str(adapter_manifest.get("adapter_digest", ""))
    if build_stable_digest(
        {key: value for key, value in adapter_manifest.items() if key != "adapter_digest"}
    ) != adapter_digest:
        raise RuntimeError("T2SMark adapter manifest 自摘要不一致")
    rebuilt_observations, rebuilt_adapter_core = build_t2smark_observations(
        image_pairs=image_pairs,
        t2smark_results=rebuilt_results,
        target_fpr=config.target_fpr,
        evidence_root=root_path,
    )
    if observations != rebuilt_observations:
        raise RuntimeError("T2SMark adapter observations 无法由原子单元确定性重建")
    for field_name, expected_value in rebuilt_adapter_core.items():
        if field_name == "adapter_digest":
            continue
        if adapter_manifest.get(field_name) != expected_value:
            raise RuntimeError(f"T2SMark adapter manifest 的 {field_name} 复算不一致")
    expected_adapter_paths = {
        "baseline_observations_path": paths["adapter_observations"]
        .relative_to(root_path)
        .as_posix(),
        "image_pairs_path": paths["image_pairs"].relative_to(root_path).as_posix(),
        "t2smark_results_path": paths["official_results"]
        .relative_to(root_path)
        .as_posix(),
    }
    for field_name, expected_value in expected_adapter_paths.items():
        if adapter_manifest.get(field_name) != expected_value:
            raise RuntimeError(f"T2SMark adapter manifest 的 {field_name} 不是可迁移路径")
    if adapter_manifest.get("generation_protocol") != {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "num_inference_steps": int(config.num_inference_steps),
        "guidance_scale": float(config.guidance_scale),
    }:
        raise RuntimeError("T2SMark adapter 生成协议与正式配置不一致")
    if adapter_manifest.get("detection_protocol") != {
        "input_access_mode": "image_only",
        "num_inversion_steps": int(config.num_inversion_steps),
        "target_fpr": float(config.target_fpr),
    }:
        raise RuntimeError("T2SMark adapter 检测协议与正式配置不一致")

    expected_calibration_count = sum(
        str(row.get("split", "")) == "calibration" for row in prompt_rows
    )
    threshold_audit = audit_fixed_fpr_observation_threshold(
        observations,
        target_fpr=config.target_fpr,
        expected_calibration_negative_count=expected_calibration_count,
    )
    if not threshold_audit.fixed_fpr_ready:
        raise RuntimeError("T2SMark 打包 observation 未通过 fixed-FPR 复验")
    from experiments.protocol.attacks import default_attack_configs

    required_attack_names = {
        attack.attack_name
        for attack in default_attack_configs()
        if attack.enabled and attack.resource_profile in {"full_main", "full_extra"}
    }
    actual_attack_names = {
        str(row.get("attack_name") or row.get("attack_condition"))
        for row in observations
        if str(row.get("sample_role", "")).startswith("attacked_")
    }
    evidence_paths = [
        relative_or_absolute(paths["official_results"], root_path),
        relative_or_absolute(paths["image_pairs"], root_path),
        relative_or_absolute(paths["adapter_observations"], root_path),
        relative_or_absolute(paths["prompt_dataset"], root_path),
        relative_or_absolute(paths["prompt_plan"], root_path),
    ]
    expected_candidates = build_t2smark_formal_candidate_records(
        observation_rows=observations,
        target_fpr=config.target_fpr,
        baseline_result_source=relative_or_absolute(paths["official_results"], root_path),
        baseline_result_source_digest=file_digest(paths["official_results"]),
        evidence_paths=evidence_paths,
        prompt_protocol_digest=str(prompt_report["prompt_protocol_digest"]),
        paper_run_prompt_protocol_ready=bool(
            prompt_report["paper_run_prompt_protocol_ready"]
        ),
        fixed_fpr_baseline_calibration_ready=threshold_audit.fixed_fpr_ready,
        attack_matrix_baseline_detection_ready=actual_attack_names
        == required_attack_names,
    )
    actual_candidates = _read_jsonl_rows(
        paths["candidate_records"],
        "formal candidate records",
    )
    if actual_candidates != list(expected_candidates):
        raise RuntimeError("T2SMark formal candidate records 无法由 observation 重建")
    rebuilt_validation = validate_primary_baseline_formal_import_rows(
        expected_candidates,
        evidence_root=root_path,
        target_fpr=config.target_fpr,
        require_existing_evidence=True,
    )
    if validation != rebuilt_validation:
        raise RuntimeError("T2SMark formal validation report 无法由候选记录重建")

    required_entries = {
        paths["official_results"],
        paths["official_settings"],
        paths["official_protocol_binding"],
        paths["official_unit_contract"],
        paths["prompt_dataset"],
        paths["prompt_plan"],
        paths["image_pairs"],
        paths["pair_quality_metrics"],
        paths["pair_quality_summary"],
        paths["adapter_observations"],
        paths["adapter_manifest"],
        paths["adapter_artifact_manifest"],
        paths["candidate_records"],
        paths["validation_report"],
        paths["environment_report"],
        paths["summary"],
        paths["manifest"],
        paths["source_prepare_result"],
        paths["official_command_result"],
        paths["adapter_command_result"],
        output_dir / "scientific_execution" / "scientific_workflow_result_envelope.json",
        output_dir / "isolated_scientific_execution_report.json",
        output_dir / "isolated_dependency_environment_report.json",
        output_dir / "scientific_command_dispatch_report.json",
        output_dir / "scientific_execution_binding.json",
    }
    required_entries.update(_expected_indexed_pngs(paths["official_images"], prompt_count))
    required_entries.update(
        _expected_indexed_pngs(paths["official_run_dir"] / "quality_pairs" / "clean", prompt_count)
    )
    required_entries.update(
        _expected_attack_pngs(
            paths["official_run_dir"] / "formal_attacks",
            test_prompt_indices,
            attack_names,
        )
    )
    required_entries.update(
        paths["official_unit_dir"] / f"{index:05d}.json"
        for index in range(prompt_count)
    )
    missing_entries = sorted(path for path in required_entries if not path.is_file())
    if missing_entries:
        raise FileNotFoundError(
            "T2SMark 结果目录缺少白名单文件: "
            + ",".join(path.relative_to(output_dir).as_posix() for path in missing_entries[:20])
        )
    validate_scientific_execution_binding(
        output_dir / "scientific_execution_binding.json",
        expected_artifact_role="t2smark_formal_reproduction",
        expected_paper_run_name=output_dir.name,
        repository_root=root_path,
    )

    actual_output_entries = {
        path
        for path in inventory_files
        if path.resolve() != archive_path.resolve()
    }
    unexpected_entries = sorted(actual_output_entries - required_entries)
    if unexpected_entries:
        raise RuntimeError(
            "T2SMark 结果目录包含旧运行或非白名单文件: "
            + ",".join(path.relative_to(output_dir).as_posix() for path in unexpected_entries[:20])
        )
    expected_directories: set[Path] = set()
    for entry in required_entries:
        parent = entry.parent
        while parent != output_dir:
            parent.resolve().relative_to(output_dir.resolve())
            expected_directories.add(parent)
            parent = parent.parent
    actual_directories = {
        path for path in inventory_directories
    }
    if actual_directories != expected_directories:
        unexpected_directories = sorted(actual_directories - expected_directories)
        missing_directories = sorted(expected_directories - actual_directories)
        raise RuntimeError(
            "T2SMark 结果目录层级不是精确白名单: "
            f"unexpected={[path.relative_to(output_dir).as_posix() for path in unexpected_directories[:20]]},"
            f"missing={[path.relative_to(output_dir).as_posix() for path in missing_directories[:20]]}"
        )

    return tuple(sorted(required_entries, key=lambda path: path.as_posix()))


def package_t2smark_formal_reproduction_outputs(
    root: str | Path = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    drive_output_dir: str | None = None,
    archive_name: str | None = None,
) -> T2SMarkFormalArchiveRecord:
    """仅打包通过 formal validation 且目录精确匹配白名单的 T2SMark 结果。"""

    root_path = Path(root).resolve()
    formal_execution_package_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    paper_run = build_paper_run_config(root_path)
    resolved_drive_output_dir = drive_output_dir or paper_run.drive_dir(
        "external_baseline_official_reference"
    )
    configured_output_root = (root_path / output_dir).resolve()
    expected_output_root = (root_path / DEFAULT_OUTPUT_DIR).resolve()
    if configured_output_root != expected_output_root:
        raise ValueError("T2SMark 打包根目录必须使用正式 outputs family")
    source_dir = expected_output_root / paper_run.run_name
    run_manifest = read_json(
        source_dir / "t2smark_formal_reproduction_manifest.local.json"
    )
    formal_execution_run_lock = repository_environment.validate_formal_execution_lock_pair(
        run_manifest.get("formal_execution_run_lock"),
        formal_execution_package_lock,
        run_manifest.get("code_version"),
    )
    resolved_archive_name = archive_name or (
        "external_baseline_official_reference_package_t2smark_"
        f"{utc_archive_token()}_{formal_execution_package_lock['formal_execution_commit'][:7]}.zip"
    )
    if (
        Path(resolved_archive_name).name != resolved_archive_name
        or not resolved_archive_name.startswith(
            "external_baseline_official_reference_package_t2smark_"
        )
        or Path(resolved_archive_name).suffix.lower() != ".zip"
    ):
        raise ValueError("T2SMark archive_name 未匹配正式命名")
    archive_path = source_dir / resolved_archive_name
    package_manifest_path = source_dir / "t2smark_formal_package_input_manifest.json"
    summary_path = source_dir / "t2smark_formal_archive_summary.json"
    manifest_path = source_dir / "t2smark_formal_archive_manifest.local.json"
    for stale_path in (package_manifest_path, summary_path, manifest_path, archive_path):
        if _path_is_link_or_reparse(stale_path):
            raise RuntimeError("T2SMark 打包目标不得是链接或 reparse point")
        if stale_path.exists():
            if not stale_path.is_file():
                raise RuntimeError("T2SMark 打包目标必须是普通文件")
            stale_path.unlink()
    entries = collect_package_entries(root_path, source_dir, archive_path)
    package_manifest = {
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
        "baseline_id": "t2smark",
        "formal_execution_run_lock": formal_execution_run_lock,
        "formal_execution_package_lock": formal_execution_package_lock,
        "entry_paths": [entry.relative_to(root_path).as_posix() for entry in entries],
        "entry_count": len(entries),
        "entry_sha256": {
            entry.relative_to(root_path).as_posix(): file_digest(entry) for entry in entries
        },
    }
    write_json(package_manifest_path, package_manifest)
    archive_manifest = build_artifact_manifest(
        artifact_id="t2smark_formal_archive_manifest",
        artifact_type="local_manifest",
        input_paths=tuple([entry.relative_to(root_path).as_posix() for entry in entries] + [package_manifest_path.relative_to(root_path).as_posix()]),
        output_paths=(
            archive_path.relative_to(root_path).as_posix(),
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ),
        config={
            "archive_name": resolved_archive_name,
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
            "drive_output_dir": str(Path(resolved_drive_output_dir).expanduser()),
        },
        code_version=formal_execution_package_lock["formal_execution_commit"],
        rebuild_command="调用 paper_experiments.runners.t2smark_formal_reproduction",
        metadata={
            "construction_unit_name": "t2smark_formal_reproduction",
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
            "baseline_id": "t2smark",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    ).to_dict()
    archive_manifest["formal_execution_run_lock"] = formal_execution_run_lock
    archive_manifest["formal_execution_package_lock"] = formal_execution_package_lock
    write_json(manifest_path, archive_manifest)
    preliminary_record = T2SMarkFormalArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest="",
        archive_entry_count=len(entries) + 3,
        drive_archive_path=str(Path(resolved_drive_output_dir).expanduser() / resolved_archive_name),
        drive_archive_digest="",
        metadata={
            "construction_unit_name": "t2smark_formal_reproduction",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(summary_path, preliminary_record.to_dict())
    entries = tuple((*entries, package_manifest_path, summary_path, manifest_path))
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
    record = T2SMarkFormalArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest=file_digest(archive_path),
        archive_entry_count=len(entries),
        drive_archive_path=str(mirrored_path),
        drive_archive_digest=file_digest(mirrored_path),
        metadata={
            "construction_unit_name": "t2smark_formal_reproduction",
            "drive_output_dir": str(drive_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(summary_path, record.to_dict())
    archive_manifest.setdefault("metadata", {})["archive_digest"] = record.archive_digest
    archive_manifest.setdefault("metadata", {})["drive_archive_digest"] = record.drive_archive_digest
    write_json(manifest_path, archive_manifest)
    return record
