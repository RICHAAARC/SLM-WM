"""定义主方法与 baseline 共享的正式随机化和基础 latent 协议。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Iterable, Mapping

from main.core.digest import build_stable_digest, tensor_content_identity
from main.core.keyed_prg import (
    KEYED_PRG_VERSION,
    build_keyed_gaussian_tensor,
    keyed_prg_protocol_record,
)


FORMAL_RANDOMIZATION_PROTOCOL = "crossed_generation_seed_watermark_key_v1"
FORMAL_GENERATION_SEED_OFFSETS = (0, 1_000_003, 2_000_003)
FORMAL_WATERMARK_KEY_INDICES = (0, 1, 2)
DEFAULT_FORMAL_RANDOMIZATION_REPEAT_ID = "seed_00_key_00"
FORMAL_RANDOMIZATION_REPEAT_IDENTITY_FIELDS = (
    "randomization_repeat_id",
    "generation_seed_index",
    "generation_seed_offset",
    "watermark_key_index",
)


@dataclass(frozen=True)
class FormalRandomizationRepeat:
    """描述一个生成种子与水印密钥的交叉重复单元。"""

    randomization_repeat_id: str
    generation_seed_index: int
    generation_seed_offset: int
    watermark_key_index: int

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入配置和 manifest 的稳定记录."""

        return asdict(self)


def formal_randomization_repeats() -> tuple[FormalRandomizationRepeat, ...]:
    """返回3个生成种子与3个密钥组成的9个交叉重复."""

    return tuple(
        FormalRandomizationRepeat(
            randomization_repeat_id=f"seed_{seed_index:02d}_key_{key_index:02d}",
            generation_seed_index=seed_index,
            generation_seed_offset=seed_offset,
            watermark_key_index=key_index,
        )
        for seed_index, seed_offset in enumerate(FORMAL_GENERATION_SEED_OFFSETS)
        for key_index in FORMAL_WATERMARK_KEY_INDICES
    )


def formal_randomization_repeat_ids() -> tuple[str, ...]:
    """返回正式9重复的规范有序 ID 集合."""

    return tuple(
        repeat.randomization_repeat_id
        for repeat in formal_randomization_repeats()
    )


def formal_randomization_repeat_registry_digest() -> str:
    """计算完整 repeat 身份注册表的稳定摘要."""

    return build_stable_digest(
        [repeat.to_dict() for repeat in formal_randomization_repeats()]
    )


def validate_formal_randomization_repeat_records(
    records: Iterable[Mapping[str, Any]],
    *,
    require_exact_registry: bool,
) -> tuple[dict[str, Any], ...]:
    """校验 repeat 记录身份, 并按权威注册表顺序返回规范记录.

    单 repeat component 使用 ``require_exact_registry=False`` 校验一个活动重复;
    最终论文聚合使用 ``True`` 要求9个身份无缺失、无重复且无额外记录. 该函数
    比仅检查 ``count == 9`` 更严格, 因为错误的 seed 或 key 索引也会被拒绝.
    """

    expected_records = tuple(
        repeat.to_dict() for repeat in formal_randomization_repeats()
    )
    expected_by_id = {
        record["randomization_repeat_id"]: record
        for record in expected_records
    }
    materialized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source_record in records:
        if not isinstance(source_record, Mapping):
            raise TypeError("正式随机化 repeat 记录必须是 mapping")
        record = {
            field_name: source_record.get(field_name)
            for field_name in FORMAL_RANDOMIZATION_REPEAT_IDENTITY_FIELDS
        }
        repeat_id = record["randomization_repeat_id"]
        if not isinstance(repeat_id, str) or not repeat_id:
            raise ValueError("正式随机化 repeat 缺少非空 ID")
        if repeat_id in seen_ids:
            raise ValueError(f"正式随机化 repeat ID 重复: {repeat_id}")
        expected = expected_by_id.get(repeat_id)
        if expected is None or record != expected:
            raise ValueError(f"正式随机化 repeat 身份未匹配注册表: {repeat_id}")
        seen_ids.add(repeat_id)
        materialized.append(record)
    expected_ids = set(expected_by_id)
    if require_exact_registry:
        if seen_ids != expected_ids:
            missing = sorted(expected_ids - seen_ids)
            unexpected = sorted(seen_ids - expected_ids)
            raise ValueError(
                "正式随机化 repeat 未精确覆盖注册表: "
                f"missing={missing};unexpected={unexpected}"
            )
    elif len(materialized) != 1:
        raise ValueError("单 repeat component 必须精确声明一个正式 repeat")
    return tuple(
        expected_by_id[repeat_id]
        for repeat_id in formal_randomization_repeat_ids()
        if repeat_id in seen_ids
    )


def build_formal_randomization_repeat_coverage(
    records: Iterable[Mapping[str, Any]],
    *,
    require_exact_registry: bool,
) -> dict[str, Any]:
    """构造可持久化的 repeat 覆盖记录与内容摘要."""

    normalized = validate_formal_randomization_repeat_records(
        records,
        require_exact_registry=require_exact_registry,
    )
    protocol = formal_randomization_protocol_record()
    payload = {
        "formal_randomization_protocol": FORMAL_RANDOMIZATION_PROTOCOL,
        "formal_randomization_protocol_digest": protocol[
            "formal_randomization_protocol_digest"
        ],
        "formal_randomization_repeat_registry_digest": (
            formal_randomization_repeat_registry_digest()
        ),
        "required_repeat_count": len(formal_randomization_repeats()),
        "observed_repeat_count": len(normalized),
        "observed_repeat_ids": [
            record["randomization_repeat_id"] for record in normalized
        ],
        "repeat_records": [dict(record) for record in normalized],
        "exact_repeat_registry_ready": bool(require_exact_registry),
    }
    return {
        **payload,
        "formal_randomization_repeat_coverage_digest": build_stable_digest(
            payload
        ),
        "supports_paper_claim": False,
    }


def formal_randomization_protocol_record() -> dict[str, Any]:
    """返回完整交叉重复注册表及其稳定摘要."""

    repeat_records = [record.to_dict() for record in formal_randomization_repeats()]
    payload = {
        "formal_randomization_protocol": FORMAL_RANDOMIZATION_PROTOCOL,
        "generation_seed_repeat_count": len(FORMAL_GENERATION_SEED_OFFSETS),
        "watermark_key_repeat_count": len(FORMAL_WATERMARK_KEY_INDICES),
        "crossed_repeat_count": len(repeat_records),
        "repeat_records": repeat_records,
        "base_latent_distribution": "standard_normal",
        "base_latent_generation": "device_independent_sha256_box_muller",
        "base_latent_dtype_cast": "cpu_before_device_transfer",
        "base_latent_keyed_prg_version": KEYED_PRG_VERSION,
        "base_latent_keyed_prg_protocol_digest": keyed_prg_protocol_record()[
            "keyed_prg_protocol_digest"
        ],
    }
    return {
        **payload,
        "formal_randomization_protocol_digest": build_stable_digest(payload),
    }


def resolve_formal_randomization_repeat(
    repeat_id: str | None,
) -> FormalRandomizationRepeat:
    """解析一个正式重复身份, 缺失时使用注册表首个重复."""

    resolved_id = str(
        repeat_id or DEFAULT_FORMAL_RANDOMIZATION_REPEAT_ID
    ).strip()
    by_id = {
        repeat.randomization_repeat_id: repeat
        for repeat in formal_randomization_repeats()
    }
    if resolved_id not in by_id:
        raise ValueError(f"未登记的正式随机化重复: {resolved_id}")
    return by_id[resolved_id]


def formal_generation_seed(
    base_seed: int,
    prompt_index: int,
    repeat: FormalRandomizationRepeat,
) -> int:
    """生成跨方法一致且按 Prompt 独立的公开生成种子."""

    if base_seed < 0 or prompt_index < 0:
        raise ValueError("base_seed 和 prompt_index 不得为负数")
    return int(base_seed) + int(repeat.generation_seed_offset) + int(prompt_index)


def formal_watermark_key_seed_random(
    root_key_material: str,
    repeat: FormalRandomizationRepeat,
) -> int:
    """从统一根密钥与重复索引派生跨方法共享的非负整数身份."""

    if not root_key_material:
        raise ValueError("root_key_material 不能为空")
    payload = (
        f"{FORMAL_RANDOMIZATION_PROTOCOL}\0"
        f"{repeat.watermark_key_index}\0{root_key_material}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def formal_watermark_key_material(
    root_key_material: str,
    repeat: FormalRandomizationRepeat,
) -> str:
    """构造主方法可直接消费的重复密钥材料, 不暴露根密钥原文."""

    key_seed = formal_watermark_key_seed_random(root_key_material, repeat)
    return f"slm_wm_formal_key:{repeat.watermark_key_index}:{key_seed:016x}"


def build_formal_randomization_identity(
    *,
    base_seed: int,
    prompt_index: int,
    root_key_material: str,
    repeat: FormalRandomizationRepeat,
) -> dict[str, Any]:
    """构造单个 Prompt 在当前重复中的完整随机身份."""

    protocol = formal_randomization_protocol_record()
    key_material = formal_watermark_key_material(root_key_material, repeat)
    payload = {
        **repeat.to_dict(),
        "generation_seed_random": formal_generation_seed(
            base_seed,
            prompt_index,
            repeat,
        ),
        "watermark_key_seed_random": formal_watermark_key_seed_random(
            root_key_material,
            repeat,
        ),
        "watermark_key_material_digest_random": build_stable_digest(
            {"key_material": key_material}
        ),
        "formal_randomization_protocol_digest": protocol[
            "formal_randomization_protocol_digest"
        ],
    }
    payload["formal_randomization_identity_digest_random"] = build_stable_digest(
        payload
    )
    return payload


def build_canonical_sd35_base_latent(
    *,
    shape: tuple[int, ...],
    generation_seed_random: int,
    model_id: str,
    model_revision: str,
    device: Any,
    dtype: Any,
) -> tuple[Any, dict[str, Any]]:
    """构造所有方法共享的设备无关标准高斯基础 latent.

    该函数先使用版本化 SHA-256 计数器流在 CPU 生成规范 float32 Tensor,
    再统一转换到目标 dtype 和设备. 因此主方法与各 baseline 不依赖各自的
    CPU/CUDA RNG 实现, 可以通过内容摘要直接证明基础 latent 相同.
    """

    normalized_shape = tuple(int(value) for value in shape)
    if not normalized_shape or any(value <= 0 for value in normalized_shape):
        raise ValueError("基础 latent shape 必须全部为正整数")
    if generation_seed_random < 0:
        raise ValueError("generation_seed_random 不得为负数")
    if not model_id or not model_revision:
        raise ValueError("基础 latent 必须绑定完整模型身份")
    protocol = formal_randomization_protocol_record()
    canonical = build_keyed_gaussian_tensor(
        normalized_shape,
        key_material=f"public_generation_seed:{generation_seed_random}",
        domain_fields={
            "operator_role": "formal_sd35_base_latent",
            "formal_randomization_protocol": FORMAL_RANDOMIZATION_PROTOCOL,
            "model_id": model_id,
            "model_revision": model_revision,
            "generation_seed_random": int(generation_seed_random),
        },
    )
    typed_canonical = canonical.to(device="cpu", dtype=dtype)
    latent = typed_canonical.to(device=device)
    tensor_identity = tensor_content_identity(typed_canonical)
    identity = {
        "generation_seed_random": int(generation_seed_random),
        "base_latent_generation_protocol": (
            "device_independent_sha256_box_muller_cpu_dtype_cast_then_device_transfer_v1"
        ),
        "base_latent_keyed_prg_version": KEYED_PRG_VERSION,
        "formal_randomization_protocol_digest": protocol[
            "formal_randomization_protocol_digest"
        ],
        "base_latent_dtype": tensor_identity["tensor_dtype"],
        "base_latent_shape": tensor_identity["tensor_shape"],
        "base_latent_content_digest_random": tensor_identity[
            "tensor_content_sha256"
        ],
    }
    identity["base_latent_identity_digest_random"] = build_stable_digest(
        identity
    )
    return latent, identity


def formal_random_trace_fields(identity: dict[str, Any]) -> dict[str, Any]:
    """提取可写入科学来源随机字段容器的后缀合规字段.

    完整正式随机身份同时包含重复索引、协议版本和随机 Tensor 摘要. 科学来源
    中的随机字段容器只接受 `_random` 或 `_digest_random` 字段, 因此由该函数
    统一提取随机部分, 协议部分继续保存在配置、Prompt 行和事实记录中.
    """

    return {
        field_name: value
        for field_name, value in identity.items()
        if field_name.endswith("_random")
        or field_name.endswith("_digest_random")
    }


__all__ = [
    "DEFAULT_FORMAL_RANDOMIZATION_REPEAT_ID",
    "FORMAL_GENERATION_SEED_OFFSETS",
    "FORMAL_RANDOMIZATION_PROTOCOL",
    "FORMAL_RANDOMIZATION_REPEAT_IDENTITY_FIELDS",
    "FORMAL_WATERMARK_KEY_INDICES",
    "FormalRandomizationRepeat",
    "build_canonical_sd35_base_latent",
    "build_formal_randomization_repeat_coverage",
    "build_formal_randomization_identity",
    "formal_generation_seed",
    "formal_random_trace_fields",
    "formal_randomization_protocol_record",
    "formal_randomization_repeat_ids",
    "formal_randomization_repeat_registry_digest",
    "formal_randomization_repeats",
    "formal_watermark_key_material",
    "formal_watermark_key_seed_random",
    "resolve_formal_randomization_repeat",
    "validate_formal_randomization_repeat_records",
]
