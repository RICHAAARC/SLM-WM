"""验证服务器单 Prompt GPU 资格化入口的调用与退出契约."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import struct

import pytest

from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    semantic_watermark_runtime_config_digest,
)
from experiments.protocol import gpu_method_qualification as qualification
from experiments.protocol.detection_key_identity import (
    REGISTERED_WATERMARK_KEY_ROLE,
    REGISTERED_WRONG_KEY_ROLE,
    resolve_detection_key_material_and_identity,
)
from main.core.digest import build_stable_digest
from scripts import run_gpu_method_qualification as entry


pytestmark = pytest.mark.quick


def test_content_smoke_requires_explicit_binary32_references() -> None:
    """Smoke references have no defaults and cannot masquerade as qualified values."""

    parser = entry.build_parser()
    arguments = parser.parse_args(
        [
            "--prompt-id",
            "probe_prompt_0001",
            "--content-runtime-smoke",
            "--reference-gradient",
            "1.0",
            "--reference-response",
            "0.5",
            "--reference-sensitivity",
            str(struct.unpack(">f", bytes.fromhex("3f000001"))[0]),
        ]
    )
    references, identity = entry._explicit_smoke_references(arguments)
    assert references.reference_gradient == 1.0
    assert identity["reference_input_role"] == "explicit_smoke_only_unqualified"
    assert identity["supports_paper_claim"] is False
    arguments.reference_gradient = 1.1
    with pytest.raises(ValueError, match="binary32"):
        entry._explicit_smoke_references(arguments)


def test_content_smoke_parser_does_not_default_reference_values() -> None:
    arguments = entry.build_parser().parse_args(
        ["--prompt-id", "probe_prompt_0001", "--content-runtime-smoke"]
    )
    assert arguments.reference_gradient is None
    assert arguments.reference_response is None
    assert arguments.reference_sensitivity is None


def test_gpu_qualification_revalidates_registered_and_wrong_key_identity() -> None:
    """注册密钥与 wrong-key 必须作用于同一水印图像且材料摘要不同."""

    config = SemanticWatermarkRuntimeConfig(key_material="registered-key")
    rows = []
    for sample_role, key_role in (
        ("positive_source", REGISTERED_WATERMARK_KEY_ROLE),
        ("wrong_key_negative", REGISTERED_WRONG_KEY_ROLE),
    ):
        _material, identity = resolve_detection_key_material_and_identity(
            config.key_material,
            key_role,
        )
        rows.append(
            {
                "sample_role": sample_role,
                "content_score": (
                    0.9 if sample_role == "positive_source" else 0.1
                ),
                "source_image_digest": "a" * 64,
                "evaluated_image_digest": "b" * 64,
                **identity,
            }
        )

    ready, evidence = qualification._registered_and_wrong_key_attribution_ready(
        rows,
        config,
    )

    assert ready is True
    assert evidence["shared_watermarked_image_ready"] is True
    rows[1]["evaluated_image_digest"] = "c" * 64
    assert qualification._registered_and_wrong_key_attribution_ready(
        rows,
        config,
    )[0] is False


def test_gpu_qualification_binding_covers_commit_models_dependency_and_prompt() -> None:
    """资格化身份必须联合绑定 Git、依赖锁、SD3.5、VAE、CLIP 与 Prompt."""

    config = SemanticWatermarkRuntimeConfig(
        prompt="一个受治理测试 Prompt",
        prompt_id="prompt_registered",
    )
    environment = {
        "formal_execution_commit": "b" * 40,
        "dependency_profile_id": "sd35_method_runtime_gpu",
        "dependency_profile_digest": "c" * 64,
        "complete_hash_lock_digest": "d" * 64,
        "torch_version": "2.11.0+cu128",
        "torch_cuda_version": "12.8",
        "execution_device_name": "cuda:0",
    }
    runtime_result = {
        "metadata": {
            "diffusion_model_source": {
                "repository_id": config.model_id,
                "revision": config.model_revision,
            },
            "vision_model_source": {
                "repository_id": config.vision_model_id,
                "revision": config.vision_model_revision,
            },
        }
    }
    compatibility_core = {
        "report_schema": "torch_func_transform_compatibility_v1",
        "schema_version": 1,
        "torch_version": environment["torch_version"],
        "torch_cuda_version": environment["torch_cuda_version"],
        "execution_device_name": environment["execution_device_name"],
        "assert_operator": "torch._assert_async",
        "forward_transform_operator": "torch.func.linearize",
        "reverse_transform_operator": "torch.func.vjp",
        "adjoint_absolute_error": 0.0,
        "operator_compatibility_ready": True,
        "supports_paper_claim": False,
    }
    compatibility = {
        **compatibility_core,
        "compatibility_report_digest": build_stable_digest(compatibility_core),
    }
    core = {
        "code_version": environment["formal_execution_commit"],
        "dependency_profile_id": environment["dependency_profile_id"],
        "dependency_profile_digest": environment["dependency_profile_digest"],
        "complete_hash_lock_digest": environment["complete_hash_lock_digest"],
        "model_revisions": {
            "sd35_model_id": config.model_id,
            "sd35_model_revision": config.model_revision,
            "vae_model_id": config.model_id,
            "vae_model_revision": config.model_revision,
            "vae_class_name": config.vae_class_name,
            "vision_model_id": config.vision_model_id,
            "vision_model_revision": config.vision_model_revision,
        },
        "input_summary": {
            "prompt_id": config.prompt_id,
            "prompt_digest": build_stable_digest({"prompt": config.prompt}),
            "method_runtime_config_digest": (
                semantic_watermark_runtime_config_digest(config)
            ),
        },
        "torch_func_compatibility": compatibility,
    }
    binding = {**core, "qualification_binding_digest": build_stable_digest(core)}

    ready, resolved = qualification._qualification_binding_ready(
        binding,
        runtime_result,
        config,
        environment,
    )

    assert ready is True
    assert resolved["qualification_binding_digest"] == build_stable_digest(core)
    binding["model_revisions"] = {
        **binding["model_revisions"],
        "vision_model_revision": "0" * 40,
    }
    assert qualification._qualification_binding_ready(
        binding,
        runtime_result,
        config,
        environment,
    )[0] is False


@pytest.mark.parametrize(
    ("operator_ready", "expected_exit_code"),
    ((False, 1), (True, 0)),
)
def test_gpu_qualification_entry_uses_real_writer_and_operator_exit_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    operator_ready: bool,
    expected_exit_code: int,
) -> None:
    """入口必须调用真实方法 writer, 且资源门禁不得改变进程状态码."""

    prompt = SimpleNamespace(
        prompt_id="prompt_registered",
        prompt_text="一个受治理测试 Prompt",
        prompt_digest="a" * 64,
        prompt_set="probe_paper",
        split="dev",
    )
    runtime_result = {
        "run_id": "single_prompt_real_method_run",
        "update_record_path": "outputs/update.jsonl",
        "detection_record_path": "outputs/detection.jsonl",
    }

    class FakeResult:
        """提供真实 writer 返回对象所需的最小只读接口."""

        run_id = runtime_result["run_id"]
        update_record_path = runtime_result["update_record_path"]
        detection_record_path = runtime_result["detection_record_path"]

        def to_dict(self) -> dict[str, object]:
            """返回资格化入口后续消费的运行结果映射."""

            return dict(runtime_result)

    calls: list[SemanticWatermarkRuntimeConfig] = []
    monkeypatch.setattr(entry, "_registered_prompt", lambda *_args: prompt)
    monkeypatch.setattr(
        entry.repository_environment,
        "require_published_formal_execution_lock",
        lambda _root: {"formal_execution_commit": "b" * 40},
    )
    monkeypatch.setattr(
        entry.repository_environment,
        "resolve_code_version",
        lambda _root: "b" * 40,
    )
    monkeypatch.setattr(
        entry,
        "require_dependency_profile_ready",
        lambda _profile_id: SimpleNamespace(formal_ready=True),
    )
    monkeypatch.setattr(
        entry,
        "build_method_config",
        lambda _root: SemanticWatermarkRuntimeConfig(
            device_name="cuda",
            diffusion_attacks_enabled=False,
        ),
    )

    def fake_real_writer(config, root):
        """记录入口是否复用正式方法 writer, 不执行 GPU 科学计算."""

        calls.append(config)
        return FakeResult()

    monkeypatch.setattr(
        entry,
        "write_semantic_watermark_runtime_outputs",
        fake_real_writer,
    )
    monkeypatch.setattr(entry, "_read_jsonl", lambda _path: ({},))
    monkeypatch.setattr(
        entry,
        "_qualification_binding",
        lambda **_kwargs: {"qualification_binding_digest": "c" * 64},
    )
    monkeypatch.setattr(
        entry,
        "build_gpu_method_qualification_report",
        lambda **_kwargs: {
            "gpu_operator_preflight_ready": operator_ready,
            "gpu_resource_budget_ready": False,
            "qualification_report_digest": "d" * 64,
            "supports_paper_claim": False,
        },
    )
    monkeypatch.setattr(
        entry,
        "_evaluate_torch_func_compatibility",
        lambda *_args: {
            "report_schema": "torch_func_transform_compatibility_v1",
            "operator_compatibility_ready": True,
            "supports_paper_claim": False,
        },
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 1024,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")

    exit_code = entry.main(
        (
            "--root",
            str(tmp_path),
            "--paper-run-name",
            "probe_paper",
            "--prompt-id",
            prompt.prompt_id,
            "--output-root",
            "outputs/gpu_method_qualification",
        )
    )

    assert exit_code == expected_exit_code
    assert len(calls) == 1
    assert calls[0].prompt_id == prompt.prompt_id
    assert calls[0].standard_attack_profiles == ()
    assert calls[0].diffusion_attacks_enabled is False
    report_path = (
        tmp_path
        / "outputs/gpu_method_qualification"
        / runtime_result["run_id"]
        / "gpu_method_qualification_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["supports_paper_claim"] is False
    assert report["gpu_resource_budget_ready"] is False
    invocation = json.loads(capsys.readouterr().out)
    assert invocation["report_schema"] == (
        "gpu_method_qualification_invocation_result_v1"
    )
    assert invocation["gpu_method_qualification_report_digest"] == "d" * 64
    assert invocation["gpu_operator_preflight_ready"] is operator_ready
    assert invocation["supports_paper_claim"] is False


def test_torch_func_compatibility_executes_linearize_vjp_and_async_assert() -> None:
    """轻量检查必须真实执行项目依赖的 PyTorch 变换组合."""

    import torch

    report = entry._evaluate_torch_func_compatibility(torch, "cpu")

    assert report["report_schema"] == "torch_func_transform_compatibility_v1"
    assert report["execution_device_name"] == "cpu"
    assert report["assert_operator"] == "torch._assert_async"
    assert report["forward_transform_operator"] == "torch.func.linearize"
    assert report["reverse_transform_operator"] == "torch.func.vjp"
    assert report["adjoint_absolute_error"] <= 1e-5
    assert report["operator_compatibility_ready"] is True
    assert report["supports_paper_claim"] is False


def test_torch_func_compatibility_rejects_missing_async_assert() -> None:
    """目标 PyTorch 缺少异步断言时必须在加载大型模型前失败."""

    with pytest.raises(RuntimeError, match="torch._assert_async"):
        entry._evaluate_torch_func_compatibility(SimpleNamespace(), "cpu")
