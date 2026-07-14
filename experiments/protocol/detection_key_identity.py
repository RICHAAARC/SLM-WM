"""定义正式仅图像检测的注册密钥与 wrong-key 身份协议."""

from __future__ import annotations

from typing import Any, Mapping

from main.core.digest import build_stable_digest


DETECTION_KEY_PLAN_SCHEMA = "slm_wm_detection_key_plan"
DETECTION_KEY_PLAN_PROTOCOL = (
    "registered_key_and_sha256_domain_separated_wrong_key"
)
REGISTERED_WATERMARK_KEY_ROLE = "registered_watermark_key"
REGISTERED_WRONG_KEY_ROLE = "registered_wrong_key_negative"
DETECTION_KEY_ROLES = (
    REGISTERED_WATERMARK_KEY_ROLE,
    REGISTERED_WRONG_KEY_ROLE,
)


def derive_registered_wrong_key_material(
    registered_key_material: str,
) -> str:
    """从当前注册水印密钥确定性派生唯一 wrong-key 对照材料."""

    if type(registered_key_material) is not str or not registered_key_material:
        raise ValueError("registered_key_material 必须为非空精确 str")
    derivation_digest = build_stable_digest(
        {
            "detection_key_plan_protocol": DETECTION_KEY_PLAN_PROTOCOL,
            "registered_key_material": registered_key_material,
            "derived_role": REGISTERED_WRONG_KEY_ROLE,
        }
    )
    return f"slm-wm-wrong-key:{derivation_digest}"


def build_detection_key_plan_record(
    registered_key_material: str,
) -> dict[str, Any]:
    """返回不暴露密钥原文且可独立重算摘要的检测密钥计划."""

    wrong_key_material = derive_registered_wrong_key_material(
        registered_key_material
    )
    payload = {
        "detection_key_plan_schema": DETECTION_KEY_PLAN_SCHEMA,
        "detection_key_plan_protocol": DETECTION_KEY_PLAN_PROTOCOL,
        "registered_watermark_key_digest_random": build_stable_digest(
            {"key_material": registered_key_material}
        ),
        "registered_wrong_key_negative_digest_random": build_stable_digest(
            {"key_material": wrong_key_material}
        ),
    }
    return {
        **payload,
        "detection_key_plan_digest_random": build_stable_digest(payload),
    }


def validate_detection_key_plan_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """独立复验检测密钥计划字段、角色区分和稳定摘要."""

    expected_fields = {
        "detection_key_plan_schema",
        "detection_key_plan_protocol",
        "registered_watermark_key_digest_random",
        "registered_wrong_key_negative_digest_random",
        "detection_key_plan_digest_random",
    }
    if not isinstance(record, Mapping) or set(record) != expected_fields:
        raise ValueError("检测密钥计划字段集合不完整")
    resolved = dict(record)
    for field_name in (
        "registered_watermark_key_digest_random",
        "registered_wrong_key_negative_digest_random",
        "detection_key_plan_digest_random",
    ):
        value = resolved.get(field_name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value
            )
        ):
            raise ValueError(f"{field_name} 必须为规范 SHA-256")
    payload = {
        field_name: resolved[field_name]
        for field_name in expected_fields
        if field_name != "detection_key_plan_digest_random"
    }
    if (
        resolved["detection_key_plan_schema"] != DETECTION_KEY_PLAN_SCHEMA
        or resolved["detection_key_plan_protocol"]
        != DETECTION_KEY_PLAN_PROTOCOL
        or resolved["registered_watermark_key_digest_random"]
        == resolved["registered_wrong_key_negative_digest_random"]
        or build_stable_digest(payload)
        != resolved["detection_key_plan_digest_random"]
    ):
        raise ValueError("检测密钥计划正文、角色或摘要不一致")
    return resolved


def resolve_detection_key_material_and_identity(
    registered_key_material: str,
    detection_key_role: str,
) -> tuple[str, dict[str, Any]]:
    """返回指定预注册角色的实际密钥材料和可持久化身份."""

    if detection_key_role not in DETECTION_KEY_ROLES:
        raise ValueError("detection_key_role 不属于冻结角色集合")
    plan = build_detection_key_plan_record(registered_key_material)
    if detection_key_role == REGISTERED_WATERMARK_KEY_ROLE:
        key_material = registered_key_material
        material_digest = plan["registered_watermark_key_digest_random"]
    else:
        key_material = derive_registered_wrong_key_material(
            registered_key_material
        )
        material_digest = plan[
            "registered_wrong_key_negative_digest_random"
        ]
    return key_material, {
        "detection_key_role": detection_key_role,
        "detection_key_material_digest_random": material_digest,
        "detection_key_plan_digest_random": plan[
            "detection_key_plan_digest_random"
        ],
    }


def validate_detection_key_identity_record(
    record: Mapping[str, Any],
    detection_key_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """依据顶层计划复验样本的角色、材料摘要与计划引用."""

    role = record.get("detection_key_role")
    if role not in DETECTION_KEY_ROLES:
        raise ValueError("检测记录缺少冻结 detection_key_role")
    plan = validate_detection_key_plan_record(detection_key_plan)
    expected_material_digest = (
        plan["registered_watermark_key_digest_random"]
        if role == REGISTERED_WATERMARK_KEY_ROLE
        else plan["registered_wrong_key_negative_digest_random"]
    )
    if (
        record.get("detection_key_material_digest_random")
        != expected_material_digest
        or record.get("detection_key_plan_digest_random")
        != plan["detection_key_plan_digest_random"]
    ):
        raise ValueError("检测密钥角色、材料摘要或计划摘要不一致")
    return {
        "detection_key_role": role,
        "detection_key_material_digest_random": expected_material_digest,
        "detection_key_plan_digest_random": plan[
            "detection_key_plan_digest_random"
        ],
    }


__all__ = [
    "DETECTION_KEY_PLAN_PROTOCOL",
    "DETECTION_KEY_PLAN_SCHEMA",
    "DETECTION_KEY_ROLES",
    "REGISTERED_WATERMARK_KEY_ROLE",
    "REGISTERED_WRONG_KEY_ROLE",
    "build_detection_key_plan_record",
    "derive_registered_wrong_key_material",
    "resolve_detection_key_material_and_identity",
    "validate_detection_key_identity_record",
    "validate_detection_key_plan_record",
]
