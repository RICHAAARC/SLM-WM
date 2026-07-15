"""构造功能测试使用的确定性科学完成单元来源记录."""

from __future__ import annotations

from typing import Any, Mapping

from experiments.runtime.scientific_unit_provenance import (
    SCIENTIFIC_UNIT_PROVENANCE_SCHEMA,
    SCIENTIFIC_UNIT_PROVENANCE_SCHEMA_VERSION,
    validate_scientific_unit_provenance,
)
from main.core.digest import build_stable_digest
from tests.helpers.formal_execution_lock import (
    build_test_formal_execution_lock,
)


def build_test_scientific_unit_provenance(
    scientific_unit_id: str,
    scientific_unit_config_digest: str,
    *,
    seed: int = 0,
    formal_execution_lock: Mapping[str, Any] | None = None,
    dependency_profile_id: str = "sd35_method_runtime_gpu",
    dependency_profile_digest: str = "1" * 64,
    complete_hash_lock_digest: str = "3" * 64,
) -> dict[str, Any]:
    """构造可由生产 validator 完整复算的 CUDA 测试来源记录."""

    execution_lock = dict(
        formal_execution_lock or build_test_formal_execution_lock()
    )
    execution_environment = {
        "dependency_profile_id": dependency_profile_id,
        "dependency_profile_digest": dependency_profile_digest,
        "direct_requirements_digest": "2" * 64,
        "complete_hash_lock_digest": complete_hash_lock_digest,
        "formal_execution_commit": execution_lock["formal_execution_commit"],
        "formal_execution_lock_digest": execution_lock[
            "formal_execution_lock_digest"
        ],
        "dependency_environment_report_digest": "4" * 64,
        "python_version": "3.12.13",
        "python_executable_sha256": "5" * 64,
        "torch_version": "2.11.0+cu128",
        "torch_cuda_version": "12.8",
        "cuda_available": True,
        "visible_cuda_device_count": 1,
        "execution_device_name": "cuda:0",
        "cuda_device_index": 0,
        "cuda_device_name": "NVIDIA T4",
        "cuda_device_capability": [7, 5],
    }
    random_identity_random = {"fixture_seed_random": int(seed)}
    payload = {
        "report_schema": SCIENTIFIC_UNIT_PROVENANCE_SCHEMA,
        "schema_version": SCIENTIFIC_UNIT_PROVENANCE_SCHEMA_VERSION,
        "scientific_unit_id": scientific_unit_id,
        "scientific_unit_config_digest": scientific_unit_config_digest,
        "scientific_execution_environment": execution_environment,
        "scientific_execution_environment_digest": build_stable_digest(
            execution_environment
        ),
        "scientific_random_identity_random": random_identity_random,
        "scientific_random_identity_digest_random": build_stable_digest(
            random_identity_random
        ),
        "supports_paper_claim": False,
    }
    payload["scientific_unit_provenance_digest"] = build_stable_digest(
        payload
    )
    return validate_scientific_unit_provenance(payload)
