"""读取并校验主表 method-faithful baseline 的正式 observation 集合。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping

from experiments.protocol.paper_run_config import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_INFERENCE_STEPS,
    PaperRunConfig,
    build_paper_run_config,
)
from experiments.protocol.attacks import (
    attack_config_digest,
    default_attack_configs,
    resolve_formal_attack_config,
)
from experiments.protocol.fixed_fpr_observation_audit import audit_fixed_fpr_observation_threshold
from experiments.runtime.model_sources import get_model_source, require_registered_model_reference
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.protocol.splits import apply_split_assignments
from main.core.digest import build_stable_digest
from paper_experiments.baselines.method_faithful_numerical_fidelity import (
    validate_method_faithful_numerical_fidelity_report,
)
from paper_experiments.baselines.observation_io import load_baseline_observation_rows

METHOD_FAITHFUL_BASELINE_IDS = (
    "tree_ring",
    "gaussian_shading",
    "shallow_diffuse",
)
DEFAULT_METHOD_FAITHFUL_COLLECTION_ROOT = Path("outputs/external_baseline_method_faithful")
SPLIT_OBSERVATION_DIR_NAME = "split_observations"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMON_BACKBONE_SOURCE = get_model_source("stabilityai_stable_diffusion_3_5_medium")
FORMAL_MODEL_ID = _COMMON_BACKBONE_SOURCE.repository_id
FORMAL_MODEL_REVISION = _COMMON_BACKBONE_SOURCE.revision


@dataclass(frozen=True)
class MethodFaithfulObservationSource:
    """描述一个 baseline 的已校验 observation 与来源 manifest。"""

    baseline_id: str
    observations_path: Path
    transfer_manifest_path: Path
    observations_sha256: str
    prompt_plan_path: Path
    adapter_manifest_path: Path
    execution_manifest_path: Path
    numerical_fidelity_report_path: Path
    numerical_fidelity_report_digest: str
    numerical_fidelity_reference_mode: str
    model_id: str
    model_revision: str
    rows: tuple[dict[str, Any], ...]
    transfer_manifest: dict[str, Any]


@dataclass(frozen=True)
class MethodFaithfulCollectionProtocol:
    """保存 exact-set collection 必须满足的当前论文协议。"""

    paper_run_name: str
    prompt_set: str
    prompt_count: int
    prompt_protocol_digest: str
    target_fpr: float
    model_id: str = FORMAL_MODEL_ID
    model_revision: str = FORMAL_MODEL_REVISION
    num_inference_steps: int = DEFAULT_INFERENCE_STEPS
    num_inversion_steps: int = DEFAULT_INFERENCE_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE

    def __post_init__(self) -> None:
        """将正式 collection 固定到预注册的模型仓库与不可变 commit。"""

        if self.model_id != FORMAL_MODEL_ID or self.model_revision != FORMAL_MODEL_REVISION:
            raise ValueError("method_faithful_collection_model_source_mismatch")
        require_registered_model_reference(
            self.model_id,
            self.model_revision,
            required_usage_role="common_backbone_baseline_model",
        )


def canonical_prompt_protocol_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    """按 Prompt 原始索引构造跨 baseline 共用的规范摘要。"""

    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("canonical_prompt_protocol_rows_empty")
    required_fields = ("prompt_id", "prompt_index", "prompt_set", "split", "prompt_text", "prompt_digest")
    for row_index, row in enumerate(materialized):
        missing = [field for field in required_fields if row.get(field) in {None, ""}]
        if missing:
            raise ValueError(
                f"canonical_prompt_protocol_fields_missing:{row_index}:{','.join(missing)}"
            )
        if isinstance(row["prompt_index"], bool):
            raise ValueError(f"canonical_prompt_protocol_index_invalid:{row_index}")
        try:
            row["prompt_index"] = int(row["prompt_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"canonical_prompt_protocol_index_invalid:{row_index}") from exc
    ordered = sorted(materialized, key=lambda row: int(row["prompt_index"]))
    indices = [int(row["prompt_index"]) for row in ordered]
    if indices != list(range(len(ordered))):
        raise ValueError("canonical_prompt_protocol_indices_not_contiguous")
    prompt_ids = [str(row["prompt_id"]) for row in ordered]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("canonical_prompt_protocol_prompt_id_duplicate")
    return build_stable_digest([str(row["prompt_digest"]) for row in ordered])


def _paper_run_prompt_rows(root_path: Path, paper_run: PaperRunConfig) -> tuple[dict[str, Any], ...]:
    """从当前论文 Prompt 文件重建规范 Prompt 计划。"""

    prompt_path = Path(paper_run.prompt_file)
    if not prompt_path.is_absolute():
        prompt_path = root_path / prompt_path
    records = apply_split_assignments(
        build_prompt_records(paper_run.prompt_set, read_prompt_file(prompt_path))
    )
    return tuple(
        {
            "prompt_id": record.prompt_id,
            "prompt_index": record.prompt_index,
            "prompt_set": record.prompt_set,
            "split": record.split,
            "prompt_text": record.prompt_text,
            "prompt_digest": record.prompt_digest,
        }
        for record in records
    )


def build_method_faithful_collection_protocol(
    project_root: str | Path = ".",
) -> MethodFaithfulCollectionProtocol:
    """从当前论文运行配置构造不可由 transfer manifest 自行声明的协议边界。"""

    root_path = Path(project_root).resolve()
    paper_run = build_paper_run_config(root_path)
    prompt_rows = _paper_run_prompt_rows(root_path, paper_run)
    return MethodFaithfulCollectionProtocol(
        paper_run_name=paper_run.run_name,
        prompt_set=paper_run.prompt_set,
        prompt_count=paper_run.prompt_count,
        prompt_protocol_digest=canonical_prompt_protocol_digest(prompt_rows),
        target_fpr=paper_run.target_fpr,
        num_inference_steps=paper_run.inference_steps,
        num_inversion_steps=paper_run.inference_steps,
        guidance_scale=paper_run.guidance_scale,
    )


def file_sha256(path: Path) -> str:
    """计算文件字节内容的 SHA-256, 用于校验跨 Colab 包物化结果。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observation_relative_path(baseline_id: str) -> PurePosixPath:
    """构造 baseline observation 相对 collection 根目录的规范路径。"""

    return PurePosixPath(SPLIT_OBSERVATION_DIR_NAME) / f"{baseline_id}_baseline_observations.json"


def transfer_manifest_relative_path(baseline_id: str) -> PurePosixPath:
    """构造 baseline transfer manifest 的规范相对路径。"""

    return PurePosixPath(SPLIT_OBSERVATION_DIR_NAME) / f"{baseline_id}_baseline_transfer_manifest.json"


def _resolve_collection_relative_path(collection_root: Path, relative_path: PurePosixPath) -> Path:
    """解析 collection 内部 POSIX 路径, 并拒绝绝对路径和目录越界。"""

    if relative_path.is_absolute() or any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ValueError(f"method_faithful_collection_path_invalid:{relative_path.as_posix()}")
    resolved_root = collection_root.resolve()
    resolved_path = (resolved_root / Path(*relative_path.parts)).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"method_faithful_collection_path_escaped:{relative_path.as_posix()}") from exc
    return resolved_path


def _read_transfer_manifest(path: Path) -> dict[str, Any]:
    """读取 transfer manifest, 并要求其顶层为 JSON 对象。"""

    if not path.is_file():
        raise FileNotFoundError(f"method_faithful_transfer_manifest_missing:{path.as_posix()}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"method_faithful_transfer_manifest_must_be_object:{path.as_posix()}")
    return dict(payload)


def _require_exact_file_set(split_dir: Path) -> None:
    """要求 collection 中恰好存在三个规范 observation 和三个 transfer manifest。"""

    if not split_dir.is_dir():
        raise FileNotFoundError(f"method_faithful_split_observation_dir_missing:{split_dir.as_posix()}")
    expected_observations = {
        observation_relative_path(baseline_id).name for baseline_id in METHOD_FAITHFUL_BASELINE_IDS
    }
    expected_manifests = {
        transfer_manifest_relative_path(baseline_id).name for baseline_id in METHOD_FAITHFUL_BASELINE_IDS
    }
    actual_observations = {
        path.name for path in split_dir.glob("*_baseline_observations.json") if path.is_file()
    }
    actual_manifests = {
        path.name for path in split_dir.glob("*_baseline_transfer_manifest.json") if path.is_file()
    }
    if actual_observations != expected_observations:
        missing = sorted(expected_observations - actual_observations)
        unexpected = sorted(actual_observations - expected_observations)
        raise ValueError(
            "method_faithful_observation_file_set_mismatch:"
            f"missing={','.join(missing)};unexpected={','.join(unexpected)}"
        )
    if actual_manifests != expected_manifests:
        missing = sorted(expected_manifests - actual_manifests)
        unexpected = sorted(actual_manifests - expected_manifests)
        raise ValueError(
            "method_faithful_transfer_manifest_file_set_mismatch:"
            f"missing={','.join(missing)};unexpected={','.join(unexpected)}"
        )


def _validate_transfer_manifest(
    *,
    baseline_id: str,
    manifest: Mapping[str, Any],
    collection_root: Path,
    protocol: MethodFaithfulCollectionProtocol,
) -> tuple[Path, int, str, Path, Path, Path, Path, str, str]:
    """校验单 baseline manifest 的身份、文件摘要和当前论文协议。"""

    required_fields = (
        "baseline_id",
        "baseline_observations_path",
        "baseline_observation_count",
        "baseline_observations_sha256",
        "prompt_plan_path",
        "prompt_plan_sha256",
        "prompt_protocol_digest",
        "adapter_manifest_path",
        "adapter_manifest_sha256",
        "execution_manifest_path",
        "execution_manifest_sha256",
        "numerical_fidelity_report_path",
        "numerical_fidelity_report_sha256",
        "numerical_fidelity_report_digest",
        "numerical_fidelity_reference_mode",
        "method_faithful_numerical_fidelity_ready",
        "paper_run_name",
        "prompt_set",
        "prompt_count",
        "model_id",
        "model_revision",
        "target_fpr",
        "generation_protocol",
        "detection_protocol",
        "formal_attack_names",
        "threshold",
        "threshold_digest",
        "transfer_ready",
    )
    missing_fields = [field for field in required_fields if field not in manifest]
    if missing_fields:
        raise ValueError(
            f"method_faithful_transfer_manifest_fields_missing:{baseline_id}:{','.join(missing_fields)}"
        )
    manifest_baseline_id = str(manifest["baseline_id"])
    if manifest_baseline_id != baseline_id:
        raise ValueError(
            f"method_faithful_transfer_manifest_baseline_mismatch:{baseline_id}:{manifest_baseline_id}"
        )
    raw_path = manifest["baseline_observations_path"]
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise ValueError(f"method_faithful_transfer_manifest_path_invalid:{baseline_id}")
    manifest_relative_path = PurePosixPath(raw_path)
    expected_relative_path = observation_relative_path(baseline_id)
    if manifest_relative_path != expected_relative_path:
        raise ValueError(
            "method_faithful_transfer_manifest_path_mismatch:"
            f"{baseline_id}:{manifest_relative_path.as_posix()}:{expected_relative_path.as_posix()}"
        )
    observations_path = _resolve_collection_relative_path(collection_root, manifest_relative_path)
    raw_count = manifest["baseline_observation_count"]
    if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count <= 0:
        raise ValueError(f"method_faithful_transfer_manifest_count_invalid:{baseline_id}:{raw_count}")
    expected_sha256 = manifest["baseline_observations_sha256"]
    if not isinstance(expected_sha256, str) or SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError(f"method_faithful_transfer_manifest_sha256_invalid:{baseline_id}")
    if manifest["transfer_ready"] is not True:
        raise ValueError(f"method_faithful_transfer_not_ready:{baseline_id}")
    if str(manifest["paper_run_name"]) != protocol.paper_run_name:
        raise ValueError(f"method_faithful_transfer_paper_run_mismatch:{baseline_id}")
    if str(manifest["prompt_set"]) != protocol.prompt_set:
        raise ValueError(f"method_faithful_transfer_prompt_set_mismatch:{baseline_id}")
    if int(manifest["prompt_count"]) != protocol.prompt_count:
        raise ValueError(f"method_faithful_transfer_prompt_count_mismatch:{baseline_id}")
    if str(manifest["model_id"]) != protocol.model_id:
        raise ValueError(f"method_faithful_transfer_model_id_mismatch:{baseline_id}")
    if str(manifest["model_revision"]) != protocol.model_revision:
        raise ValueError(f"method_faithful_transfer_model_revision_mismatch:{baseline_id}")
    try:
        target_fpr = float(manifest["target_fpr"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"method_faithful_transfer_target_fpr_invalid:{baseline_id}") from exc
    if not math.isclose(target_fpr, protocol.target_fpr, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"method_faithful_transfer_target_fpr_mismatch:{baseline_id}")

    generation = manifest["generation_protocol"]
    detection = manifest["detection_protocol"]
    if not isinstance(generation, Mapping) or not isinstance(detection, Mapping):
        raise ValueError(f"method_faithful_transfer_budget_invalid:{baseline_id}")
    _validate_runtime_budget(baseline_id, generation, detection, protocol, "transfer")

    prompt_plan_path = _validate_manifest_bound_file(
        baseline_id=baseline_id,
        manifest=manifest,
        collection_root=collection_root,
        path_field="prompt_plan_path",
        digest_field="prompt_plan_sha256",
    )
    adapter_manifest_path = _validate_manifest_bound_file(
        baseline_id=baseline_id,
        manifest=manifest,
        collection_root=collection_root,
        path_field="adapter_manifest_path",
        digest_field="adapter_manifest_sha256",
    )
    execution_manifest_path = _validate_manifest_bound_file(
        baseline_id=baseline_id,
        manifest=manifest,
        collection_root=collection_root,
        path_field="execution_manifest_path",
        digest_field="execution_manifest_sha256",
    )
    numerical_fidelity_report_path = _validate_manifest_bound_file(
        baseline_id=baseline_id,
        manifest=manifest,
        collection_root=collection_root,
        path_field="numerical_fidelity_report_path",
        digest_field="numerical_fidelity_report_sha256",
    )
    numerical_fidelity_report = validate_method_faithful_numerical_fidelity_report(
        _read_json_object(
            numerical_fidelity_report_path,
            "numerical_fidelity_report",
        ),
        expected_baseline_id=baseline_id,
    )
    numerical_fidelity_report_digest = str(
        numerical_fidelity_report["numerical_fidelity_report_digest"]
    )
    numerical_fidelity_reference_mode = str(
        numerical_fidelity_report["numerical_fidelity_reference_mode"]
    )
    if (
        manifest["method_faithful_numerical_fidelity_ready"] is not True
        or str(manifest["numerical_fidelity_report_digest"])
        != numerical_fidelity_report_digest
        or str(manifest["numerical_fidelity_reference_mode"])
        != numerical_fidelity_reference_mode
    ):
        raise ValueError(
            f"method_faithful_numerical_fidelity_not_ready:{baseline_id}"
        )
    prompt_rows = _read_json_array(prompt_plan_path, "prompt_plan")
    prompt_digest = canonical_prompt_protocol_digest(prompt_rows)
    if str(manifest["prompt_protocol_digest"]) != prompt_digest:
        raise ValueError(f"method_faithful_transfer_prompt_digest_mismatch:{baseline_id}")
    if prompt_digest != protocol.prompt_protocol_digest:
        raise ValueError(f"method_faithful_transfer_canonical_prompt_mismatch:{baseline_id}")
    if len(prompt_rows) != protocol.prompt_count or any(
        str(row.get("prompt_set", "")) != protocol.prompt_set for row in prompt_rows
    ):
        raise ValueError(f"method_faithful_transfer_prompt_plan_mismatch:{baseline_id}")

    adapter_manifest = _read_json_object(adapter_manifest_path, "adapter_manifest")
    if str(adapter_manifest.get("baseline_id", "")) != baseline_id:
        raise ValueError(f"method_faithful_adapter_manifest_baseline_mismatch:{baseline_id}")
    if str(adapter_manifest.get("adapter_status", "")) != "method_faithful_sd35_adapter_ready":
        raise ValueError(f"method_faithful_adapter_manifest_status_invalid:{baseline_id}")
    if str(adapter_manifest.get("adapter_boundary", "")) != "method_faithful_sd35_adapter_reproduction":
        raise ValueError(f"method_faithful_adapter_manifest_boundary_invalid:{baseline_id}")
    if str(adapter_manifest.get("model_id", "")) != protocol.model_id:
        raise ValueError(f"method_faithful_adapter_manifest_model_id_mismatch:{baseline_id}")
    if str(adapter_manifest.get("model_revision", "")) != protocol.model_revision:
        raise ValueError(f"method_faithful_adapter_manifest_model_revision_mismatch:{baseline_id}")
    if int(adapter_manifest.get("observation_count", -1)) != raw_count:
        raise ValueError(f"method_faithful_adapter_manifest_count_mismatch:{baseline_id}")
    adapter_generation = adapter_manifest.get("generation_protocol")
    adapter_detection = adapter_manifest.get("detection_protocol")
    if not isinstance(adapter_generation, Mapping) or not isinstance(adapter_detection, Mapping):
        raise ValueError(f"method_faithful_adapter_manifest_budget_invalid:{baseline_id}")
    _validate_runtime_budget(
        baseline_id,
        adapter_generation,
        adapter_detection,
        protocol,
        "adapter",
    )

    execution_manifest = _read_json_object(execution_manifest_path, "execution_manifest")
    if str(execution_manifest.get("artifact_name", "")) != "baseline_execution_manifest.json":
        raise ValueError(f"method_faithful_execution_manifest_identity_invalid:{baseline_id}")
    if execution_manifest.get("baseline_ids") != [baseline_id]:
        raise ValueError(f"method_faithful_execution_manifest_baseline_mismatch:{baseline_id}")
    if int(execution_manifest.get("command_count", -1)) != 1:
        raise ValueError(f"method_faithful_execution_manifest_command_count_invalid:{baseline_id}")
    if int(execution_manifest.get("failed_command_count", -1)) != 0:
        raise ValueError(f"method_faithful_execution_manifest_failed:{baseline_id}")
    if int(execution_manifest.get("observation_count", -1)) != raw_count:
        raise ValueError(f"method_faithful_execution_manifest_count_mismatch:{baseline_id}")
    return (
        observations_path,
        raw_count,
        expected_sha256,
        prompt_plan_path,
        adapter_manifest_path,
        execution_manifest_path,
        numerical_fidelity_report_path,
        numerical_fidelity_report_digest,
        numerical_fidelity_reference_mode,
    )


def _read_json_object(path: Path, role: str) -> dict[str, Any]:
    """读取被 transfer manifest 摘要绑定的 JSON 对象。"""

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"method_faithful_{role}_must_be_object:{path.as_posix()}")
    return dict(payload)


def validate_formal_attack_observation_identities(
    rows: Iterable[Mapping[str, Any]],
    *,
    baseline_id: str,
) -> None:
    """校验执行端 observation 绑定唯一正式攻击配置.

    该校验属于项目特定的 producer schema 边界. 它要求攻击身份直接存在于
    每条 attacked observation, 不允许 collection 或结果 writer 根据名称后贴标签.
    """

    attacked_count = 0
    attacked_positive_count = 0
    for row_index, source_row in enumerate(rows):
        row = dict(source_row)
        sample_role = str(row.get("sample_role", ""))
        if sample_role not in {"attacked_negative", "attacked_positive"}:
            continue
        attacked_count += 1
        attacked_positive_count += int(sample_role == "attacked_positive")
        attack_family = str(row.get("attack_family", ""))
        attack_name = str(row.get("attack_name") or row.get("attack_condition") or "")
        try:
            config = resolve_formal_attack_config(
                attack_family=attack_family,
                attack_name=attack_name,
            )
        except ValueError as exc:
            raise ValueError(
                f"method_faithful_observation_attack_unregistered:{baseline_id}:{row_index}"
            ) from exc
        expected = {
            "attack_id": config.attack_id,
            "resource_profile": config.resource_profile,
            "attack_config_digest": attack_config_digest(config),
        }
        for field_name, expected_value in expected.items():
            if str(row.get(field_name, "")) != expected_value:
                raise ValueError(
                    "method_faithful_observation_attack_identity_mismatch:"
                    f"{baseline_id}:{row_index}:{field_name}"
                )
    if attacked_count <= 0 or attacked_positive_count <= 0:
        raise ValueError(
            f"method_faithful_observation_attacked_positive_missing:{baseline_id}"
        )


def _read_json_array(path: Path, role: str) -> list[dict[str, Any]]:
    """读取被 transfer manifest 摘要绑定的 JSON 数组。"""

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise TypeError(f"method_faithful_{role}_must_be_object_array:{path.as_posix()}")
    return [dict(row) for row in payload]


def _validate_manifest_bound_file(
    *,
    baseline_id: str,
    manifest: Mapping[str, Any],
    collection_root: Path,
    path_field: str,
    digest_field: str,
) -> Path:
    """校验 transfer manifest 引用的 collection 内文件及 SHA-256。"""

    raw_path = manifest[path_field]
    expected_digest = manifest[digest_field]
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise ValueError(f"method_faithful_transfer_bound_path_invalid:{baseline_id}:{path_field}")
    if not isinstance(expected_digest, str) or SHA256_PATTERN.fullmatch(expected_digest) is None:
        raise ValueError(f"method_faithful_transfer_bound_digest_invalid:{baseline_id}:{digest_field}")
    resolved_path = _resolve_collection_relative_path(collection_root, PurePosixPath(raw_path))
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"method_faithful_transfer_bound_file_missing:{baseline_id}:{path_field}:{resolved_path.as_posix()}"
        )
    actual_digest = file_sha256(resolved_path)
    if actual_digest != expected_digest:
        raise ValueError(f"method_faithful_transfer_bound_digest_mismatch:{baseline_id}:{digest_field}")
    return resolved_path


def _validate_runtime_budget(
    baseline_id: str,
    generation: Mapping[str, Any],
    detection: Mapping[str, Any],
    protocol: MethodFaithfulCollectionProtocol,
    source_role: str,
) -> None:
    """要求 transfer 与 adapter 均声明统一的 20/20/4.5 公平预算。"""

    try:
        generation_steps = int(generation.get("num_inference_steps", -1))
        inversion_steps = int(detection.get("num_inversion_steps", -1))
        guidance_scale = float(generation.get("guidance_scale", float("nan")))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"method_faithful_{source_role}_budget_invalid:{baseline_id}"
        ) from exc
    ready = (
        str(generation.get("model_id", "")) == protocol.model_id
        and str(generation.get("model_revision", "")) == protocol.model_revision
        and generation_steps == protocol.num_inference_steps
        and inversion_steps == protocol.num_inversion_steps
        and math.isclose(guidance_scale, protocol.guidance_scale, rel_tol=0.0, abs_tol=1e-12)
        and str(detection.get("input_access_mode", "")) == "image_only"
    )
    if not ready:
        raise ValueError(f"method_faithful_{source_role}_budget_mismatch:{baseline_id}")


def load_method_faithful_observation_collection(
    collection_root: str | Path = DEFAULT_METHOD_FAITHFUL_COLLECTION_ROOT,
    *,
    project_root: str | Path = ".",
    protocol: MethodFaithfulCollectionProtocol | None = None,
) -> tuple[MethodFaithfulObservationSource, ...]:
    """读取三个固定 baseline 的 observation, 并执行 exact-set 证据校验。

    这一实现属于项目特定的正式证据入口。它不扫描 aggregate 文件或历史 zip,
    只接受物化后的三个规范 observation 与对应 transfer manifest。这样可以让
    候选记录顺序和来源摘要不依赖 zip 遍历顺序, 也不会静默吞掉重复记录。
    """

    root_path = Path(collection_root).resolve()
    resolved_protocol = protocol or build_method_faithful_collection_protocol(project_root)
    split_dir = root_path / SPLIT_OBSERVATION_DIR_NAME
    _require_exact_file_set(split_dir)
    sources: list[MethodFaithfulObservationSource] = []
    seen_event_keys: set[tuple[str, str]] = set()
    seen_observation_paths: set[Path] = set()
    for baseline_id in METHOD_FAITHFUL_BASELINE_IDS:
        manifest_path = _resolve_collection_relative_path(
            root_path,
            transfer_manifest_relative_path(baseline_id),
        )
        manifest = _read_transfer_manifest(manifest_path)
        (
            observations_path,
            expected_count,
            expected_sha256,
            prompt_plan_path,
            adapter_manifest_path,
            execution_manifest_path,
            numerical_fidelity_report_path,
            numerical_fidelity_report_digest,
            numerical_fidelity_reference_mode,
        ) = _validate_transfer_manifest(
            baseline_id=baseline_id,
            manifest=manifest,
            collection_root=root_path,
            protocol=resolved_protocol,
        )
        if observations_path in seen_observation_paths:
            raise ValueError(f"method_faithful_observation_path_duplicate:{observations_path.as_posix()}")
        seen_observation_paths.add(observations_path)
        if not observations_path.is_file():
            raise FileNotFoundError(f"method_faithful_observations_missing:{observations_path.as_posix()}")
        actual_sha256 = file_sha256(observations_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"method_faithful_observations_sha256_mismatch:{baseline_id}:{expected_sha256}:{actual_sha256}"
            )
        rows = load_baseline_observation_rows(observations_path)
        if len(rows) != expected_count:
            raise ValueError(
                f"method_faithful_observation_count_mismatch:{baseline_id}:{expected_count}:{len(rows)}"
            )
        prompt_rows = _read_json_array(prompt_plan_path, "prompt_plan")
        expected_calibration_count = sum(
            str(row.get("split", "")) == "calibration" for row in prompt_rows
        )
        threshold_audit = audit_fixed_fpr_observation_threshold(
            rows,
            target_fpr=resolved_protocol.target_fpr,
            expected_calibration_source_negative_count=expected_calibration_count,
        )
        if not threshold_audit.fixed_fpr_ready:
            raise ValueError(f"method_faithful_transfer_threshold_audit_failed:{baseline_id}")
        try:
            manifest_threshold = float(manifest["threshold"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"method_faithful_transfer_threshold_invalid:{baseline_id}") from exc
        if threshold_audit.frozen_threshold is None or not math.isclose(
            manifest_threshold,
            threshold_audit.frozen_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"method_faithful_transfer_threshold_mismatch:{baseline_id}")
        if str(manifest["threshold_digest"]) != threshold_audit.threshold_digest:
            raise ValueError(f"method_faithful_transfer_threshold_digest_mismatch:{baseline_id}")
        required_attack_names = {
            attack.attack_name
            for attack in default_attack_configs()
            if attack.enabled and attack.resource_profile in {"full_main", "full_extra"}
        }
        declared_attack_names = manifest["formal_attack_names"]
        if not isinstance(declared_attack_names, list) or set(map(str, declared_attack_names)) != required_attack_names:
            raise ValueError(f"method_faithful_transfer_formal_attack_names_mismatch:{baseline_id}")
        observed_attack_names = {
            str(row.get("attack_name") or row.get("attack_condition") or "")
            for row in rows
            if str(row.get("sample_role", "")).startswith("attacked_")
        }
        if observed_attack_names != required_attack_names:
            raise ValueError(f"method_faithful_observation_formal_attack_names_mismatch:{baseline_id}")
        validate_formal_attack_observation_identities(rows, baseline_id=baseline_id)
        normalized_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            row_baseline_id = str(row.get("baseline_id", ""))
            if row_baseline_id != baseline_id:
                raise ValueError(
                    "method_faithful_observation_baseline_mismatch:"
                    f"{baseline_id}:{row_index}:{row_baseline_id}"
                )
            if (
                str(row.get("generation_model_id", "")) != resolved_protocol.model_id
                or str(row.get("generation_model_revision", ""))
                != resolved_protocol.model_revision
            ):
                raise ValueError(
                    "method_faithful_observation_model_revision_mismatch:"
                    f"{baseline_id}:{row_index}"
                )
            event_id = str(row.get("event_id", "")).strip()
            if not event_id:
                raise ValueError(f"method_faithful_observation_event_id_empty:{baseline_id}:{row_index}")
            event_key = (baseline_id, event_id)
            if event_key in seen_event_keys:
                raise ValueError(f"method_faithful_observation_event_id_duplicate:{baseline_id}:{event_id}")
            seen_event_keys.add(event_key)
            normalized_rows.append(dict(row))
        normalized_rows.sort(key=lambda row: str(row["event_id"]))
        sources.append(
            MethodFaithfulObservationSource(
                baseline_id=baseline_id,
                observations_path=observations_path,
                transfer_manifest_path=manifest_path,
                observations_sha256=actual_sha256,
                prompt_plan_path=prompt_plan_path,
                adapter_manifest_path=adapter_manifest_path,
                execution_manifest_path=execution_manifest_path,
                numerical_fidelity_report_path=numerical_fidelity_report_path,
                numerical_fidelity_report_digest=(
                    numerical_fidelity_report_digest
                ),
                numerical_fidelity_reference_mode=(
                    numerical_fidelity_reference_mode
                ),
                model_id=resolved_protocol.model_id,
                model_revision=resolved_protocol.model_revision,
                rows=tuple(normalized_rows),
                transfer_manifest=manifest,
            )
        )
    return tuple(sources)
