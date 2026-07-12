"""管理 T2SMark 正式复现的逐 Prompt 原子完成单元.

该模块属于项目特定写法. T2SMark 官方入口仍负责真实生成、反演和检测,
本模块只把一个已经完成的 Prompt 绑定到固定协议、源码、运行环境和事实文件.
单元记录自身不支持论文主张, 只有完整单元集合经过外层 fixed-FPR 与导入门禁后
才能进入正式结果包.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from experiments.runtime.scientific_unit_provenance import (
    aggregate_scientific_unit_provenance,
    build_scientific_unit_provenance,
    validate_scientific_unit_provenance,
)
from experiments.protocol.splits import SPLIT_NAMES
from external_baseline.primary.sd35_method_faithful_common import (
    formal_image_attack_identity,
)
from main.core.digest import build_stable_digest


T2SMARK_FORMAL_UNIT_CONTRACT_SCHEMA = "t2smark_formal_unit_contract"
T2SMARK_FORMAL_UNIT_RECORD_SCHEMA = "t2smark_formal_prompt_unit"
T2SMARK_FORMAL_UNIT_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_ATOMIC_TEMPORARY_PATTERN = re.compile(r"^\..+\.[0-9]+\.tmp$")
_STATIC_ENVIRONMENT_FIELDS = (
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
)


def _json_text(payload: Any) -> str:
    """生成禁止 NaN 和 Infinity 的稳定 JSON 文本."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """在目标目录内写临时文件并原子发布 JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(_json_text(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def atomic_save_png(image: Any, path: str | Path) -> Path:
    """把 PIL 兼容图像原子发布为 PNG, 避免中断留下半写文件."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    try:
        image.save(temporary_path, format="PNG")
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def remove_stale_atomic_temporary_files(root: str | Path) -> tuple[str, ...]:
    """移除进程中断遗留的严格命名临时文件, 不触碰任何完成单元."""

    resolved_root = Path(root).resolve()
    if not resolved_root.exists():
        return ()
    removed: list[str] = []
    for path in sorted(resolved_root.rglob("*.tmp")):
        resolved = path.resolve()
        resolved.relative_to(resolved_root)
        if resolved.is_file() and _ATOMIC_TEMPORARY_PATTERN.fullmatch(path.name):
            removed.append(resolved.relative_to(resolved_root).as_posix())
            resolved.unlink()
    return tuple(removed)


def file_sha256(path: str | Path) -> str:
    """计算事实文件 SHA-256."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_t2smark_random_material_digest(*values: Any) -> str:
    """对密钥、消息或基础 latent 计算不可逆长度分隔摘要."""

    digest = hashlib.sha256()
    for value in values:
        if hasattr(value, "detach"):
            encoded = value.detach().cpu().contiguous().numpy().tobytes()
        elif isinstance(value, bytes):
            encoded = value
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


def _required_sha256(value: Any, field_name: str) -> str:
    """解析必需 SHA-256 字段."""

    resolved = str(value or "")
    if not _SHA256_PATTERN.fullmatch(resolved):
        raise ValueError(f"T2SMark {field_name} 不是 SHA-256")
    return resolved


def _required_commit(value: Any, field_name: str) -> str:
    """解析必需40位 Git commit."""

    resolved = str(value or "")
    if not _COMMIT_PATTERN.fullmatch(resolved):
        raise ValueError(f"T2SMark {field_name} 不是40位 Git commit")
    return resolved


def _as_mapping(value: Any, field_name: str) -> dict[str, Any]:
    """把协议对象规范化为字典."""

    if not isinstance(value, Mapping):
        raise TypeError(f"T2SMark {field_name} 必须是对象")
    return dict(value)


def _normalized_prompt_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """验证 Prompt 行的全局索引、split 与摘要身份."""

    normalized = [dict(row) for row in rows]
    if not normalized:
        raise ValueError("T2SMark 正式单元契约缺少 Prompt")
    for index, row in enumerate(normalized):
        if int(row.get("prompt_index", -1)) != index:
            raise ValueError("T2SMark Prompt 索引必须连续且从0开始")
        if not str(row.get("prompt_id", "")):
            raise ValueError("T2SMark Prompt 缺少 prompt_id")
        if str(row.get("split", "")) not in set(SPLIT_NAMES):
            raise ValueError("T2SMark Prompt split 不属于受治理 split 集合")
        _required_sha256(row.get("prompt_digest"), "prompt_digest")
        if not str(row.get("prompt_text", "")):
            raise ValueError("T2SMark Prompt 文本为空")
    return normalized


def build_t2smark_formal_unit_contract(
    *,
    formal_reproduction_config: Mapping[str, Any],
    paper_run_identity: Mapping[str, Any],
    prompt_rows: Sequence[Mapping[str, Any]],
    prompt_plan_digest: str,
    protocol_binding: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    formal_execution_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """构造跨 Colab 会话保持不变的完整科学单元契约."""

    source = dict(source_identity)
    protocol = dict(protocol_binding)
    execution_lock = dict(formal_execution_lock)
    _required_commit(source.get("official_repository_commit"), "官方源码 commit")
    _required_sha256(source.get("protocol_patch_sha256"), "协议补丁摘要")
    _required_sha256(source.get("source_worktree_digest"), "源码工作树摘要")
    if source.get("source_worktree_exact") is not True:
        raise ValueError("T2SMark 源码工作树未通过精确审计")
    _required_sha256(protocol.get("protocol_binding_digest"), "协议绑定摘要")
    _required_commit(execution_lock.get("formal_execution_commit"), "项目代码锁")
    _required_sha256(
        execution_lock.get("formal_execution_lock_digest"),
        "项目代码锁摘要",
    )
    rows = _normalized_prompt_rows(prompt_rows)
    _required_sha256(prompt_plan_digest, "Prompt 计划摘要")
    payload = {
        "record_schema": T2SMARK_FORMAL_UNIT_CONTRACT_SCHEMA,
        "schema_version": T2SMARK_FORMAL_UNIT_SCHEMA_VERSION,
        "formal_reproduction_config": dict(formal_reproduction_config),
        "paper_run_identity": dict(paper_run_identity),
        "prompt_rows": rows,
        "prompt_plan_digest": prompt_plan_digest,
        "protocol_binding": protocol,
        "source_identity": source,
        "formal_execution_lock": execution_lock,
        "supports_paper_claim": False,
    }
    payload["unit_contract_digest"] = build_stable_digest(payload)
    return validate_t2smark_formal_unit_contract(payload)


def validate_t2smark_formal_unit_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """验证科学单元契约及其自摘要."""

    payload = dict(contract)
    if payload.get("record_schema") != T2SMARK_FORMAL_UNIT_CONTRACT_SCHEMA:
        raise ValueError("T2SMark 正式单元契约 schema 不匹配")
    if payload.get("schema_version") != T2SMARK_FORMAL_UNIT_SCHEMA_VERSION:
        raise ValueError("T2SMark 正式单元契约版本不匹配")
    if payload.get("supports_paper_claim") is not False:
        raise ValueError("T2SMark 单元契约自身不得支持论文主张")
    _as_mapping(payload.get("formal_reproduction_config"), "正式复现配置")
    _as_mapping(payload.get("paper_run_identity"), "论文运行身份")
    source = _as_mapping(payload.get("source_identity"), "源码身份")
    protocol = _as_mapping(payload.get("protocol_binding"), "协议绑定")
    execution_lock = _as_mapping(payload.get("formal_execution_lock"), "项目代码锁")
    rows = payload.get("prompt_rows")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("T2SMark 单元契约的 prompt_rows 必须是对象列表")
    _normalized_prompt_rows(rows)
    _required_sha256(payload.get("prompt_plan_digest"), "Prompt 计划摘要")
    if build_stable_digest(rows) != payload["prompt_plan_digest"]:
        raise ValueError("T2SMark Prompt 计划摘要不匹配")
    _required_commit(source.get("official_repository_commit"), "官方源码 commit")
    _required_sha256(source.get("protocol_patch_sha256"), "协议补丁摘要")
    _required_sha256(source.get("source_worktree_digest"), "源码工作树摘要")
    if source.get("source_worktree_exact") is not True:
        raise ValueError("T2SMark 源码工作树身份无效")
    _required_sha256(protocol.get("protocol_binding_digest"), "协议绑定摘要")
    _required_commit(execution_lock.get("formal_execution_commit"), "项目代码锁")
    _required_sha256(
        execution_lock.get("formal_execution_lock_digest"),
        "项目代码锁摘要",
    )
    digest = _required_sha256(payload.get("unit_contract_digest"), "单元契约摘要")
    digest_payload = {key: value for key, value in payload.items() if key != "unit_contract_digest"}
    if build_stable_digest(digest_payload) != digest:
        raise ValueError("T2SMark 正式单元契约自摘要不匹配")
    return payload


def write_or_validate_t2smark_formal_unit_contract(
    path: str | Path,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """首次原子写契约, 已存在时只接受逐字段完全相同的内容."""

    contract = validate_t2smark_formal_unit_contract(expected_contract)
    output_path = Path(path)
    if output_path.is_file():
        existing = validate_t2smark_formal_unit_contract(
            json.loads(output_path.read_text(encoding="utf-8-sig"))
        )
        if existing != contract:
            raise RuntimeError("T2SMark 已有单元契约与当前代码、源码或配置身份漂移")
        return existing
    atomic_write_json(output_path, contract)
    return contract


def _runtime_static_identity(runtime_environment: Mapping[str, Any]) -> dict[str, Any]:
    """提取跨会话必须保持一致的依赖与代码锁身份."""

    environment = dict(runtime_environment)
    isolated_context = _as_mapping(
        environment.get("isolated_scientific_context"),
        "隔离科学解释器身份",
    )
    return {
        "dependency_profile_id": str(environment.get("dependency_profile_id", "")),
        "dependency_profile_digest": str(environment.get("dependency_profile_digest", "")),
        "direct_requirements_digest": str(environment.get("direct_requirements_digest", "")),
        "complete_hash_lock_digest": str(environment.get("complete_hash_lock_digest", "")),
        "formal_execution_commit": str(environment.get("formal_execution_commit", "")),
        "formal_execution_lock_digest": str(environment.get("formal_execution_lock_digest", "")),
        "python_version": str(environment.get("python_version", "")),
        "python_executable_sha256": str(
            isolated_context.get("current_python_executable_sha256", "")
        ),
        "torch_version": str(_as_mapping(environment.get("package_versions"), "依赖版本").get("torch", "")),
        "torch_cuda_version": str(environment.get("cuda_version", "")),
    }


def _finite_number(value: Any, field_name: str) -> float:
    """验证必须落盘的连续检测或准确率数值."""

    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"T2SMark {field_name} 不是有限数值")
    return resolved


def _expected_artifact_paths(
    artifact_root: Path,
    prompt_index: int,
    attack_names: Sequence[str],
) -> dict[str, Path]:
    """构造单个 Prompt 应产生的完整事实图像集合."""

    sample_name = f"{prompt_index:05d}.png"
    paths = {
        "watermarked_image": artifact_root / "images" / sample_name,
        "clean_image": artifact_root / "quality_pairs" / "clean" / sample_name,
    }
    for attack_name in attack_names:
        for sample_role in ("attacked_negative", "attacked_positive"):
            paths[f"{attack_name}:{sample_role}"] = (
                artifact_root
                / "formal_attacks"
                / f"{prompt_index:05d}_{attack_name}_{sample_role}.png"
            )
    return paths


def _repository_root_for_artifact_root(artifact_root: Path) -> Path:
    """从 outputs 祖先解析当前 checkout 根目录.

    正式运行的 artifact_root 必须位于仓库 outputs 下. 测试夹具可以使用不含
    outputs 分量的临时目录, 此时使用 artifact_root 的父目录作为可迁移根.
    """

    resolved_root = artifact_root.resolve()
    for candidate in (resolved_root, *resolved_root.parents):
        if candidate.name == "outputs":
            return candidate.parent
    return resolved_root.parent


def repository_relative_t2smark_path(
    path: str | Path,
    *,
    output_anchor: str | Path,
) -> str:
    """把 outputs 内文件规范化为可跨 checkout 搬迁的相对 POSIX 路径."""

    anchor = Path(output_anchor).resolve()
    repository_root = _repository_root_for_artifact_root(anchor)
    resolved = Path(path).resolve()
    resolved.relative_to((repository_root / "outputs").resolve())
    if resolved.is_symlink():
        raise ValueError("T2SMark 事实文件不得为符号链接")
    return resolved.relative_to(repository_root).as_posix()


def _canonical_artifact_path(
    value: Any,
    *,
    expected_path: Path,
    artifact_root: Path,
    allow_absolute_input: bool,
) -> str:
    """校验事实文件路径并转换为可跨 workspace 搬迁的相对 POSIX 路径."""

    text = str(value or "")
    candidate = Path(text)
    repository_root = _repository_root_for_artifact_root(artifact_root)
    if candidate.is_absolute():
        if not allow_absolute_input:
            raise ValueError("T2SMark 已完成单元不得保存绝对事实文件路径")
        resolved = candidate.resolve()
    else:
        if not text or "\\" in text or any(part in {"", ".", ".."} for part in candidate.parts):
            raise ValueError("T2SMark 单元事实文件路径必须为相对 POSIX 路径")
        resolved = (repository_root / candidate).resolve()
    if resolved != expected_path.resolve():
        raise ValueError("T2SMark 单元事实文件路径与 Prompt 索引不一致")
    resolved.relative_to(artifact_root.resolve())
    if resolved.is_symlink():
        raise ValueError("T2SMark 单元事实文件不得为符号链接")
    return resolved.relative_to(repository_root).as_posix()


def remove_uncommitted_t2smark_unit_artifacts(
    artifact_root: str | Path,
    *,
    prompt_index: int,
) -> tuple[str, ...]:
    """仅为缺少完成记录的 Prompt 清理可重建事实文件."""

    root = Path(artifact_root).resolve()
    sample_name = f"{prompt_index:05d}.png"
    candidates = {
        root / "images" / sample_name,
        root / "quality_pairs" / "clean" / sample_name,
    }
    attack_dir = root / "formal_attacks"
    if attack_dir.is_dir():
        candidates.update(attack_dir.glob(f"{prompt_index:05d}_*.png"))
    removed: list[str] = []
    for path in sorted(candidates):
        resolved = path.resolve()
        resolved.relative_to(root)
        if resolved.is_file():
            removed.append(resolved.relative_to(root).as_posix())
            resolved.unlink()
    return tuple(removed)


def _validated_result_and_artifacts(
    result: Mapping[str, Any],
    *,
    artifact_root: Path,
    prompt_index: int,
    attack_names: Sequence[str],
    allow_absolute_input: bool = False,
) -> tuple[dict[str, Any], dict[str, str]]:
    """验证单元结果结构并复算所有事实图像摘要."""

    payload = dict(result)
    robustness = _as_mapping(payload.get("robustness"), "robustness")
    for field_name in ("norm1_no_w", "norm1_w", "acc_key", "acc_msg"):
        _finite_number(robustness.get(field_name), f"robustness.{field_name}")
    detection = _as_mapping(payload.get("image_only_detection"), "仅图像检测")
    for field_name in ("clean_score", "watermarked_score"):
        _finite_number(detection.get(field_name), f"image_only_detection.{field_name}")
    pair_quality = _as_mapping(payload.get("pair_quality"), "严格成对质量")
    if pair_quality.get("pair_quality_protocol") != "strict_clean_watermarked_pair":
        raise ValueError("T2SMark 单元的严格成对质量协议无效")
    _required_sha256(
        pair_quality.get("clean_base_latent_digest_random"),
        "clean_base_latent_digest_random",
    )
    attacks = _as_mapping(payload.get("formal_attacks"), "正式攻击结果")
    if set(attacks) != set(attack_names):
        raise ValueError("T2SMark 单元未覆盖完整且精确的正式攻击集合")

    expected_paths = _expected_artifact_paths(artifact_root, prompt_index, attack_names)
    artifact_digests: dict[str, str] = {}
    resolved_root = artifact_root.resolve()
    for role, path in expected_paths.items():
        resolved = path.resolve()
        resolved.relative_to(resolved_root)
        if not resolved.is_file() or path.is_symlink():
            raise FileNotFoundError(f"T2SMark 单元事实文件缺失: {resolved}")
        artifact_digests[role] = file_sha256(resolved)
    actual_attack_paths = (
        set((artifact_root / "formal_attacks").glob(f"{prompt_index:05d}_*.png"))
        if (artifact_root / "formal_attacks").is_dir()
        else set()
    )
    expected_attack_paths = {
        path for role, path in expected_paths.items() if ":" in role
    }
    if {path.resolve() for path in actual_attack_paths} != {
        path.resolve() for path in expected_attack_paths
    }:
        raise ValueError("T2SMark 单元包含 split 协议之外的攻击图像")

    pair_quality["clean_image_path"] = _canonical_artifact_path(
        pair_quality.get("clean_image_path"),
        expected_path=expected_paths["clean_image"],
        artifact_root=artifact_root,
        allow_absolute_input=allow_absolute_input,
    )
    pair_quality["watermarked_image_path"] = _canonical_artifact_path(
        pair_quality.get("watermarked_image_path"),
        expected_path=expected_paths["watermarked_image"],
        artifact_root=artifact_root,
        allow_absolute_input=allow_absolute_input,
    )
    payload["pair_quality"] = pair_quality
    if pair_quality.get("clean_image_digest") != artifact_digests["clean_image"]:
        raise ValueError("T2SMark clean pair 文件摘要漂移")
    if pair_quality.get("watermarked_image_digest") != artifact_digests["watermarked_image"]:
        raise ValueError("T2SMark watermarked 文件摘要漂移")

    for attack_name in attack_names:
        attack = _as_mapping(attacks[attack_name], f"正式攻击 {attack_name}")
        identity = formal_image_attack_identity(attack_name)
        if str(attack.get("attack_name", "")) != attack_name:
            raise ValueError("T2SMark 正式攻击名称漂移")
        for field_name, expected_value in identity.items():
            if str(attack.get(field_name, "")) != expected_value:
                raise ValueError("T2SMark 正式攻击身份漂移")
        for sample_role in ("attacked_negative", "attacked_positive"):
            role = _as_mapping(
                attack.get(sample_role),
                f"正式攻击 {attack_name}.{sample_role}",
            )
            for field_name in (
                "attack_id",
                "resource_profile",
                "attack_config_digest",
            ):
                expected_value = identity[field_name]
                if str(role.get(field_name, "")) != expected_value:
                    raise ValueError("T2SMark 正式攻击角色身份漂移")
            _finite_number(
                role.get("detection_score"),
                f"formal_attacks.{attack_name}.{sample_role}.detection_score",
            )
            role_key = f"{attack_name}:{sample_role}"
            role["attacked_image_path"] = _canonical_artifact_path(
                role.get("attacked_image_path"),
                expected_path=expected_paths[role_key],
                artifact_root=artifact_root,
                allow_absolute_input=allow_absolute_input,
            )
            if role.get("attacked_image_digest") != artifact_digests[role_key]:
                raise ValueError("T2SMark 正式攻击图像摘要漂移")
            attack[sample_role] = role
        attacks[attack_name] = attack
    payload["formal_attacks"] = attacks
    _json_text(payload)
    return payload, artifact_digests


def _unit_config_digest(
    contract: Mapping[str, Any],
    prompt_row: Mapping[str, Any],
    prompt_seed: int,
) -> str:
    """绑定完整配置、协议、源码、Prompt 和该 Prompt 的随机种子."""

    return build_stable_digest(
        {
            "formal_reproduction_config": contract["formal_reproduction_config"],
            "paper_run_identity": contract["paper_run_identity"],
            "protocol_binding": contract["protocol_binding"],
            "source_identity": contract["source_identity"],
            "formal_execution_lock": contract["formal_execution_lock"],
            "prompt_plan_digest": contract["prompt_plan_digest"],
            "prompt_row": dict(prompt_row),
            "prompt_seed_random": int(prompt_seed),
        }
    )


def build_t2smark_formal_unit_record(
    *,
    contract: Mapping[str, Any],
    prompt_index: int,
    result: Mapping[str, Any],
    artifact_root: str | Path,
    runtime_environment: Mapping[str, Any],
    torch_module: Any,
    execution_device_name: str,
    random_identity_random: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """在实际 GPU 进程中构造一个已完成 Prompt 的来源记录."""

    resolved_contract = validate_t2smark_formal_unit_contract(contract)
    prompt_rows = resolved_contract["prompt_rows"]
    if prompt_index < 0 or prompt_index >= len(prompt_rows):
        raise IndexError("T2SMark 单元 Prompt 索引超出契约范围")
    prompt_row = dict(prompt_rows[prompt_index])
    config = dict(resolved_contract["formal_reproduction_config"])
    prompt_seed = int(config["seed"]) + prompt_index
    attack_names = (
        tuple(resolved_contract["protocol_binding"]["formal_attack_names"])
        if prompt_row["split"] == "test"
        else ()
    )
    validated_result, artifact_digests = _validated_result_and_artifacts(
        result,
        artifact_root=Path(artifact_root),
        prompt_index=prompt_index,
        attack_names=attack_names,
        allow_absolute_input=True,
    )
    unit_id = f"t2smark:{prompt_row['prompt_id']}"
    config_digest = _unit_config_digest(resolved_contract, prompt_row, prompt_seed)
    resolved_random_identity = {
        "base_seed_random": int(config["seed"]),
        "prompt_seed_random": prompt_seed,
    }
    for field_name, value in dict(random_identity_random or {}).items():
        if field_name in resolved_random_identity:
            raise ValueError("T2SMark 额外随机身份不得覆盖固定 seed 字段")
        resolved_random_identity[field_name] = value
    provenance = build_scientific_unit_provenance(
        scientific_unit_id=unit_id,
        scientific_unit_config_digest=config_digest,
        runtime_environment=runtime_environment,
        execution_device_name=execution_device_name,
        torch_module=torch_module,
        random_identity_random=resolved_random_identity,
    )
    payload = {
        "record_schema": T2SMARK_FORMAL_UNIT_RECORD_SCHEMA,
        "schema_version": T2SMARK_FORMAL_UNIT_SCHEMA_VERSION,
        "scientific_unit_id": unit_id,
        "scientific_unit_config_digest": config_digest,
        "unit_contract_digest": resolved_contract["unit_contract_digest"],
        "formal_reproduction_config": config,
        "protocol_binding_digest": resolved_contract["protocol_binding"]["protocol_binding_digest"],
        "source_identity": resolved_contract["source_identity"],
        "formal_execution_lock": resolved_contract["formal_execution_lock"],
        "prompt_identity": prompt_row,
        "prompt_seed_random": prompt_seed,
        "artifact_sha256": artifact_digests,
        "result": validated_result,
        "scientific_unit_provenance": provenance,
        "supports_paper_claim": False,
    }
    payload["formal_unit_record_digest"] = build_stable_digest(payload)
    return validate_t2smark_formal_unit_record(
        payload,
        contract=resolved_contract,
        artifact_root=artifact_root,
        runtime_environment=runtime_environment,
    )


def validate_t2smark_formal_unit_record(
    record: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    artifact_root: str | Path,
    runtime_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """复验一个已有完成单元的全部身份与事实文件."""

    payload = dict(record)
    resolved_contract = validate_t2smark_formal_unit_contract(contract)
    if payload.get("record_schema") != T2SMARK_FORMAL_UNIT_RECORD_SCHEMA:
        raise ValueError("T2SMark 正式单元记录 schema 不匹配")
    if payload.get("schema_version") != T2SMARK_FORMAL_UNIT_SCHEMA_VERSION:
        raise ValueError("T2SMark 正式单元记录版本不匹配")
    if payload.get("supports_paper_claim") is not False:
        raise ValueError("T2SMark Prompt 单元自身不得支持论文主张")
    prompt = _as_mapping(payload.get("prompt_identity"), "Prompt 身份")
    prompt_index = int(prompt.get("prompt_index", -1))
    rows = resolved_contract["prompt_rows"]
    if prompt_index < 0 or prompt_index >= len(rows) or prompt != rows[prompt_index]:
        raise ValueError("T2SMark 单元 Prompt 身份与契约不一致")
    prompt_seed = int(resolved_contract["formal_reproduction_config"]["seed"]) + prompt_index
    expected_unit_id = f"t2smark:{prompt['prompt_id']}"
    expected_config_digest = _unit_config_digest(
        resolved_contract,
        prompt,
        prompt_seed,
    )
    if payload.get("scientific_unit_id") != expected_unit_id:
        raise ValueError("T2SMark 单元 ID 与 Prompt 不一致")
    if payload.get("scientific_unit_config_digest") != expected_config_digest:
        raise ValueError("T2SMark 单元完整配置摘要不一致")
    if payload.get("prompt_seed_random") != prompt_seed:
        raise ValueError("T2SMark 单元 Prompt seed 不一致")
    for field_name, expected_value in (
        ("unit_contract_digest", resolved_contract["unit_contract_digest"]),
        ("formal_reproduction_config", resolved_contract["formal_reproduction_config"]),
        ("protocol_binding_digest", resolved_contract["protocol_binding"]["protocol_binding_digest"]),
        ("source_identity", resolved_contract["source_identity"]),
        ("formal_execution_lock", resolved_contract["formal_execution_lock"]),
    ):
        if payload.get(field_name) != expected_value:
            raise ValueError(f"T2SMark 单元 {field_name} 与当前契约身份漂移")
    provenance = validate_scientific_unit_provenance(
        _as_mapping(payload.get("scientific_unit_provenance"), "科学运行来源"),
        expected_unit_id=expected_unit_id,
        expected_config_digest=expected_config_digest,
    )
    pair_quality = _as_mapping(
        _as_mapping(payload.get("result"), "单元结果").get("pair_quality"),
        "严格成对质量",
    )
    if (
        provenance["scientific_random_identity_random"].get(
            "clean_base_latent_digest_random"
        )
        != pair_quality.get("clean_base_latent_digest_random")
    ):
        raise ValueError("T2SMark clean 基础 latent 摘要未绑定逐单元随机身份")
    _required_sha256(
        provenance["scientific_random_identity_random"].get(
            "t2smark_secret_material_digest_random"
        ),
        "t2smark_secret_material_digest_random",
    )
    if runtime_environment is not None:
        expected_runtime = _runtime_static_identity(runtime_environment)
        actual_runtime = provenance["scientific_execution_environment"]
        for field_name in _STATIC_ENVIRONMENT_FIELDS:
            if str(actual_runtime.get(field_name, "")) != str(expected_runtime[field_name]):
                raise ValueError(f"T2SMark 单元运行环境的 {field_name} 身份漂移")
    attack_names = (
        tuple(resolved_contract["protocol_binding"]["formal_attack_names"])
        if prompt["split"] == "test"
        else ()
    )
    validated_result, artifact_digests = _validated_result_and_artifacts(
        _as_mapping(payload.get("result"), "单元结果"),
        artifact_root=Path(artifact_root),
        prompt_index=prompt_index,
        attack_names=attack_names,
    )
    if payload.get("artifact_sha256") != artifact_digests:
        raise ValueError("T2SMark 单元事实文件摘要集合漂移")
    payload["result"] = validated_result
    digest = _required_sha256(payload.get("formal_unit_record_digest"), "单元记录摘要")
    digest_payload = {
        key: value for key, value in payload.items() if key != "formal_unit_record_digest"
    }
    if build_stable_digest(digest_payload) != digest:
        raise ValueError("T2SMark 正式单元记录自摘要不匹配")
    return payload


def write_t2smark_formal_unit_record(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    artifact_root: str | Path,
    runtime_environment: Mapping[str, Any],
) -> dict[str, Any]:
    """原子发布完成单元, 已存在时禁止静默覆盖冲突内容."""

    validated = validate_t2smark_formal_unit_record(
        record,
        contract=contract,
        artifact_root=artifact_root,
        runtime_environment=runtime_environment,
    )
    output_path = Path(path)
    if output_path.exists():
        existing = validate_t2smark_formal_unit_record(
            json.loads(output_path.read_text(encoding="utf-8-sig")),
            contract=contract,
            artifact_root=artifact_root,
            runtime_environment=runtime_environment,
        )
        if existing != validated:
            raise RuntimeError("T2SMark 同一 Prompt 已存在冲突完成单元")
        return existing
    atomic_write_json(output_path, validated)
    return validated


def inspect_t2smark_formal_unit_records(
    unit_dir: str | Path,
    *,
    contract: Mapping[str, Any],
    artifact_root: str | Path,
    runtime_environment: Mapping[str, Any] | None = None,
) -> tuple[dict[int, dict[str, Any]], tuple[int, ...]]:
    """验证全部已有单元并返回缺失索引, 任何损坏或额外索引均闭锁."""

    resolved_contract = validate_t2smark_formal_unit_contract(contract)
    directory = Path(unit_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise RuntimeError("T2SMark 单元目录不得为符号链接")
    expected_indices = set(range(len(resolved_contract["prompt_rows"])))
    records: dict[int, dict[str, Any]] = {}
    children = tuple(sorted(directory.iterdir()))
    if any(path.is_symlink() or not path.is_file() for path in children):
        raise RuntimeError("T2SMark 单元目录只能包含普通 JSON 完成记录")
    if any(path.suffix != ".json" for path in children):
        raise RuntimeError("T2SMark 单元目录包含非 JSON 文件")
    for path in children:
        if not path.stem.isdigit():
            raise RuntimeError("T2SMark 单元目录包含非索引 JSON 文件")
        index = int(path.stem)
        if index not in expected_indices or path.name != f"{index:05d}.json":
            raise RuntimeError("T2SMark 单元目录包含契约之外的 Prompt 索引")
        if index in records:
            raise RuntimeError("T2SMark 单元目录包含重复 Prompt 索引")
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"T2SMark 单元文件损坏: {path.name}") from error
        records[index] = validate_t2smark_formal_unit_record(
            _as_mapping(payload, "单元文件"),
            contract=resolved_contract,
            artifact_root=artifact_root,
            runtime_environment=runtime_environment,
        )
    missing = tuple(sorted(expected_indices.difference(records)))
    return records, missing


def aggregate_t2smark_formal_unit_records(
    records: Mapping[int, Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    artifact_root: str | Path,
    runtime_environment: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """在完整单元集合上确定性重建官方结果与跨会话来源聚合."""

    resolved_contract = validate_t2smark_formal_unit_contract(contract)
    expected_indices = tuple(range(len(resolved_contract["prompt_rows"])))
    if tuple(sorted(records)) != expected_indices:
        raise RuntimeError("T2SMark 只有完整 Prompt 单元集合才能生成正式聚合结果")
    validated = {
        index: validate_t2smark_formal_unit_record(
            records[index],
            contract=resolved_contract,
            artifact_root=artifact_root,
            runtime_environment=runtime_environment,
        )
        for index in expected_indices
    }
    artifact_root_path = Path(artifact_root).resolve()
    prompt_rows = resolved_contract["prompt_rows"]
    attack_names = tuple(resolved_contract["protocol_binding"]["formal_attack_names"])
    expected_watermarked = _expected_artifact_paths(
        artifact_root_path,
        0,
        (),
    )["watermarked_image"].parent
    expected_clean = _expected_artifact_paths(
        artifact_root_path,
        0,
        (),
    )["clean_image"].parent
    expected_image_paths = {
        artifact_root_path / "images" / f"{index:05d}.png"
        for index in expected_indices
    }
    expected_clean_paths = {
        artifact_root_path / "quality_pairs" / "clean" / f"{index:05d}.png"
        for index in expected_indices
    }
    expected_attack_paths = {
        artifact_root_path
        / "formal_attacks"
        / f"{index:05d}_{attack_name}_{sample_role}.png"
        for index in expected_indices
        if prompt_rows[index]["split"] == "test"
        for attack_name in attack_names
        for sample_role in ("attacked_negative", "attacked_positive")
    }

    def require_exact_files(directory: Path, expected: set[Path], label: str) -> None:
        """要求事实目录仅包含完整单元集合引用的普通文件."""

        if not directory.is_dir() or directory.is_symlink():
            raise RuntimeError(f"T2SMark {label} 事实目录不存在或不是普通目录")
        children = tuple(directory.iterdir())
        if any(path.is_symlink() or not path.is_file() for path in children):
            raise RuntimeError(f"T2SMark {label} 事实目录包含目录或符号链接")
        actual = {path.resolve() for path in children}
        if actual != {path.resolve() for path in expected}:
            raise RuntimeError(f"T2SMark {label} 事实文件集合不是 Prompt exact set")

    require_exact_files(expected_watermarked, expected_image_paths, "watermarked")
    require_exact_files(expected_clean, expected_clean_paths, "clean pair")
    if expected_attack_paths:
        require_exact_files(
            artifact_root_path / "formal_attacks",
            expected_attack_paths,
            "正式攻击",
        )
    elif (artifact_root_path / "formal_attacks").exists():
        raise RuntimeError("T2SMark 非 test 运行不得残留正式攻击目录")
    provenance_records = [
        validated[index]["scientific_unit_provenance"] for index in expected_indices
    ]
    provenance_aggregate = aggregate_scientific_unit_provenance(
        provenance_records,
        expected_reference_count=len(expected_indices),
    )
    aggregate = {
        "formal_unit_record_count": len(validated),
        "formal_unit_record_digests": [
            validated[index]["formal_unit_record_digest"] for index in expected_indices
        ],
        "formal_unit_records_digest": build_stable_digest(
            [validated[index] for index in expected_indices]
        ),
        "unit_contract_digest": resolved_contract["unit_contract_digest"],
        "prompt_plan_digest": resolved_contract["prompt_plan_digest"],
        **provenance_aggregate,
        "formal_unit_set_complete": True,
        "supports_paper_claim": False,
    }
    aggregate["formal_unit_aggregate_digest"] = build_stable_digest(aggregate)
    results = {str(index): validated[index]["result"] for index in expected_indices}
    return results, aggregate
