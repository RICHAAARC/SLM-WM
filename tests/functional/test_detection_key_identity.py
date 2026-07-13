"""验证仅图像检测注册密钥与 wrong-key 对照身份协议."""

from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.protocol.detection_key_identity import (
    DETECTION_KEY_PLAN_PROTOCOL,
    REGISTERED_WATERMARK_KEY_ROLE,
    REGISTERED_WRONG_KEY_ROLE,
    build_detection_key_plan_record,
    derive_registered_wrong_key_material,
    resolve_detection_key_material_and_identity,
    validate_detection_key_identity_record,
    validate_detection_key_plan_record,
)


pytestmark = pytest.mark.quick


def test_detection_key_plan_is_deterministic_and_role_separated() -> None:
    """注册密钥和 domain-separated wrong-key 必须确定且互不相同."""

    registered_key = "registered-test-key"
    wrong_key = derive_registered_wrong_key_material(registered_key)
    repeated_wrong_key = derive_registered_wrong_key_material(registered_key)
    plan = build_detection_key_plan_record(registered_key)

    assert wrong_key == repeated_wrong_key
    assert wrong_key != registered_key
    assert plan["detection_key_plan_protocol"] == (
        DETECTION_KEY_PLAN_PROTOCOL
    )
    assert plan["registered_watermark_key_digest_random"] != plan[
        "registered_wrong_key_negative_digest_random"
    ]
    assert validate_detection_key_plan_record(plan) == plan


@pytest.mark.parametrize(
    "role",
    (REGISTERED_WATERMARK_KEY_ROLE, REGISTERED_WRONG_KEY_ROLE),
)
def test_detection_key_identity_binds_material_to_registered_role(
    role: str,
) -> None:
    """每个角色的材料摘要必须能由同一计划正文唯一解释."""

    _key_material, identity = resolve_detection_key_material_and_identity(
        "registered-test-key",
        role,
    )
    plan = build_detection_key_plan_record("registered-test-key")
    assert "detection_key_plan" not in identity
    assert validate_detection_key_identity_record(identity, plan) == identity


@pytest.mark.parametrize(
    "mutation",
    ("plan_digest", "same_role_digest", "record_material", "record_role"),
)
def test_detection_key_identity_rejects_plan_and_role_drift(
    mutation: str,
) -> None:
    """计划、角色或材料摘要任一漂移都必须 fail-closed."""

    _key_material, identity = resolve_detection_key_material_and_identity(
        "registered-test-key",
        REGISTERED_WRONG_KEY_ROLE,
    )
    drifted = deepcopy(identity)
    drifted_plan = build_detection_key_plan_record("registered-test-key")
    if mutation == "plan_digest":
        drifted_plan["detection_key_plan_digest_random"] = "f" * 64
    elif mutation == "same_role_digest":
        drifted_plan[
            "registered_wrong_key_negative_digest_random"
        ] = drifted_plan[
            "registered_watermark_key_digest_random"
        ]
    elif mutation == "record_material":
        drifted["detection_key_material_digest_random"] = "f" * 64
    else:
        drifted["detection_key_role"] = "undeclared_key_role"

    with pytest.raises(ValueError):
        validate_detection_key_identity_record(drifted, drifted_plan)
