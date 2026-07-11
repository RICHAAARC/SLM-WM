"""验证跨 Colab 会话科学完成单元的来源绑定与聚合."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.runtime.scientific_unit_provenance import (
    aggregate_scientific_unit_provenance,
    build_scientific_unit_provenance,
    validate_scientific_unit_provenance,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    build_semantic_watermark_run_id,
    semantic_watermark_runtime_config_digest,
    semantic_watermark_runtime_config_payload,
    validate_semantic_watermark_runtime_result_provenance,
)
from main.core.digest import build_stable_digest


pytestmark = pytest.mark.quick


class _FakeCuda:
    """提供来源构造器需要的最小 CUDA 运行时接口."""

    def __init__(self, name: str) -> None:
        self.name = name

    def is_available(self) -> bool:
        return True

    def current_device(self) -> int:
        return 0

    def device_count(self) -> int:
        return 1

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return self.name

    def get_device_capability(self, index: int) -> tuple[int, int]:
        assert index == 0
        return (8, 0)


class _FakeTorch:
    """模拟固定 PyTorch/CUDA build 和单张 GPU."""

    __version__ = "2.7.1+cu128"
    version = SimpleNamespace(cuda="12.8")

    def __init__(self, gpu_name: str) -> None:
        self.cuda = _FakeCuda(gpu_name)


def _runtime_environment(gpu_name: str) -> dict[str, object]:
    """构造通过完整锁和隔离解释器门禁的环境报告."""

    return {
        "dependency_environment_ready": True,
        "formal_execution_lock_ready": True,
        "isolated_scientific_context_ready": True,
        "dependency_profile_id": "sd35_method_runtime_gpu",
        "dependency_profile_digest": "1" * 64,
        "direct_requirements_digest": "2" * 64,
        "complete_hash_lock_digest": "3" * 64,
        "formal_execution_commit": "4" * 40,
        "formal_execution_lock_digest": "5" * 64,
        "python_version": "3.12.11",
        "package_versions": {"torch": "2.7.1+cu128"},
        "cuda_version": "12.8",
        "device_count": 1,
        "gpu_name": gpu_name,
        "isolated_scientific_context": {
            "dependency_environment_report_actual_digest": "6" * 64,
            "current_python_executable_sha256": "7" * 64,
        },
    }


def _provenance(unit_id: str, seed: int, gpu_name: str) -> dict[str, object]:
    """构造一个可复验完成单元来源记录."""

    return build_scientific_unit_provenance(
        scientific_unit_id=unit_id,
        scientific_unit_config_digest=build_stable_digest(
            {"unit_id": unit_id, "seed": seed}
        ),
        runtime_environment=_runtime_environment(gpu_name),
        execution_device_name="cuda:0",
        torch_module=_FakeTorch(gpu_name),
        random_identity_random={"generation_seed_random": seed},
    )


def test_scientific_unit_provenance_binds_actual_runtime_and_random_identity() -> None:
    """单元记录必须同时绑定设备、代码锁、依赖锁和随机性身份."""

    record = _provenance("prompt_000001", 19, "NVIDIA A100-SXM4-40GB")

    validated = validate_scientific_unit_provenance(record)
    environment = validated["scientific_execution_environment"]
    assert environment["execution_device_name"] == "cuda:0"
    assert environment["cuda_device_name"] == "NVIDIA A100-SXM4-40GB"
    assert environment["torch_version"] == "2.7.1+cu128"
    assert environment["torch_cuda_version"] == "12.8"
    assert environment["complete_hash_lock_digest"] == "3" * 64
    assert environment["formal_execution_commit"] == "4" * 40
    assert validated["scientific_random_identity_random"] == {
        "generation_seed_random": 19
    }


def test_scientific_unit_provenance_aggregate_preserves_prior_session_devices() -> None:
    """最终聚合必须列出每个历史会话设备, 不能只报告最终会话."""

    first = _provenance("prompt_000001", 19, "NVIDIA T4")
    second = _provenance("prompt_000002", 20, "NVIDIA L4")

    summary = aggregate_scientific_unit_provenance(
        (first, second),
        expected_reference_count=2,
    )

    assert summary["scientific_unit_provenance_ready"] is True
    assert summary["scientific_unit_provenance_record_count"] == 2
    assert summary["scientific_cuda_device_names"] == ["NVIDIA L4", "NVIDIA T4"]
    assert summary["scientific_torch_versions"] == ["2.7.1+cu128"]
    assert summary["scientific_formal_execution_commits"] == ["4" * 40]


def test_scientific_unit_provenance_rejects_environment_or_random_tampering() -> None:
    """恢复后改写设备或随机种子必须破坏来源自摘要."""

    record = _provenance("prompt_000001", 19, "NVIDIA T4")
    environment_tampered = deepcopy(record)
    environment_tampered["scientific_execution_environment"][
        "cuda_device_name"
    ] = "forged"
    with pytest.raises(ValueError, match="执行环境摘要不匹配"):
        validate_scientific_unit_provenance(environment_tampered)

    random_tampered = deepcopy(record)
    random_tampered["scientific_random_identity_random"][
        "generation_seed_random"
    ] = 20
    with pytest.raises(ValueError, match="随机性身份摘要不匹配"):
        validate_scientific_unit_provenance(random_tampered)


def test_scientific_unit_provenance_rejects_conflicting_duplicate_unit() -> None:
    """同一完成单元不能由两个会话声明不同随机轨迹."""

    first = _provenance("prompt_000001", 19, "NVIDIA T4")
    conflicting = _provenance("prompt_000001", 20, "NVIDIA T4")

    with pytest.raises(ValueError, match="冲突来源记录"):
        aggregate_scientific_unit_provenance(
            (first, conflicting),
            expected_reference_count=2,
        )


def test_scientific_unit_provenance_rejects_out_of_range_cuda_index() -> None:
    """即使攻击者重算摘要, 越界 CUDA 设备索引也必须失败闭合."""

    record = deepcopy(_provenance("prompt_000001", 19, "NVIDIA T4"))
    record["scientific_execution_environment"]["cuda_device_index"] = 1
    record["scientific_execution_environment_digest"] = build_stable_digest(
        record["scientific_execution_environment"]
    )
    record["scientific_unit_provenance_digest"] = build_stable_digest(
        {
            key: value
            for key, value in record.items()
            if key != "scientific_unit_provenance_digest"
        }
    )

    with pytest.raises(ValueError, match="CUDA 设备索引无效"):
        validate_scientific_unit_provenance(record)


def test_semantic_runtime_result_binds_run_id_to_complete_unit_config() -> None:
    """打包复验必须从结果内完整配置重算 run id 与 provenance 摘要."""

    config = SemanticWatermarkRuntimeConfig(
        prompt="a governed test prompt",
        prompt_id="prompt_000001",
        seed=19,
    )
    run_id = build_semantic_watermark_run_id(config)
    provenance = build_scientific_unit_provenance(
        scientific_unit_id=run_id,
        scientific_unit_config_digest=semantic_watermark_runtime_config_digest(
            config
        ),
        runtime_environment=_runtime_environment("NVIDIA T4"),
        execution_device_name="cuda:0",
        torch_module=_FakeTorch("NVIDIA T4"),
        random_identity_random={"generation_seed_random": 19},
    )
    result = {
        "run_id": run_id,
        "metadata": {
            "scientific_unit_config": semantic_watermark_runtime_config_payload(
                config
            ),
            "scientific_unit_provenance": provenance,
        },
    }

    validate_semantic_watermark_runtime_result_provenance(
        result,
        expected_config=config,
    )
    tampered = deepcopy(result)
    tampered["metadata"]["scientific_unit_config"]["seed"] = 20
    with pytest.raises(ValueError, match="run id 与逐单元配置摘要不一致"):
        validate_semantic_watermark_runtime_result_provenance(tampered)
