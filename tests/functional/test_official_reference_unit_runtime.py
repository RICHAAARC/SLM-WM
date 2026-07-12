"""验证三条官方参考共享的原子科学批次与跨会话恢复协议."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
from types import SimpleNamespace
from typing import Any

import pytest

from main.core.digest import build_stable_digest
from paper_experiments.runners import official_reference_unit_runtime as unit_runtime
from paper_experiments.runners import gaussian_shading_official_reference as gaussian_runner
from paper_experiments.runners import shallow_diffuse_official_reference as shallow_runner
from paper_experiments.runners import tree_ring_official_reference as tree_runner


class _FakeCuda:
    """提供来源构造器需要的最小 CUDA 设备接口."""

    def is_available(self) -> bool:
        """声明当前测试科学进程使用 CUDA."""

        return True

    def current_device(self) -> int:
        """返回当前可见设备索引."""

        return 0

    def device_count(self) -> int:
        """返回单设备测试环境."""

        return 1

    def get_device_name(self, _index: int) -> str:
        """返回首个测试会话的设备名称."""

        return "NVIDIA T4"

    def get_device_capability(self, _index: int) -> tuple[int, int]:
        """返回有效 CUDA capability."""

        return (7, 5)


class _FakeTorch:
    """模拟实际子进程中的固定 PyTorch build."""

    __version__ = "2.1.0+cu121"
    version = SimpleNamespace(cuda="12.1")
    cuda = _FakeCuda()


def _file_sha256(path: Path) -> str:
    """计算测试解释器摘要."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def _temporary_environment(values: dict[str, str]) -> Any:
    """只在伪科学子进程调用期间替换指定环境变量."""

    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _formal_lock() -> dict[str, Any]:
    """构造可复算的正式执行锁."""

    payload = {
        "formal_execution_lock_schema": "clean_detached_git_commit_v1",
        "formal_execution_commit": "a" * 40,
        "formal_execution_head_detached": True,
        "formal_execution_worktree_clean": True,
        "formal_execution_lock_ready": True,
    }
    return {**payload, "formal_execution_lock_digest": build_stable_digest(payload)}


def _dependency_report(report_digest: str) -> dict[str, Any]:
    """构造锁相同但报告摘要可跨会话变化的隔离依赖报告."""

    import sys

    executable = Path(sys.executable).resolve()
    return {
        "dependency_environment_ready": True,
        "dependency_profile_id": "tree_ring_official_py39_cu117",
        "dependency_profile_digest": "b" * 64,
        "dependency_lock_digest": "c" * 64,
        "isolated_dependency_environment_report_digest": report_digest,
        "dependency_profile": {
            "direct_requirements_digest": "d" * 64,
            "python_version": platform.python_version(),
        },
        "isolated_dependency_environment_report": {
            "python_executable_sha256_after_preparation": _file_sha256(executable),
        },
    }


def _device_report(gpu_name: str) -> dict[str, Any]:
    """构造隔离解释器的成功 CUDA 检查报告."""

    import sys

    executable = Path(sys.executable).resolve()
    return {
        "decision": "pass",
        "failure_reasons": [],
        "return_code": 0,
        "python_executable": str(executable),
        "python_executable_sha256": _file_sha256(executable),
        "torch_available": True,
        "cuda_available": True,
        "device": "cuda",
        "torch_version": _FakeTorch.__version__,
        "torch_cuda_version": _FakeTorch.version.cuda,
        "device_count": 1,
        "gpu_name": gpu_name,
        "supports_paper_claim": False,
    }


@pytest.mark.quick
def test_official_reference_unit_ranges_use_exact_ten_prompt_batches() -> None:
    """正式三层规模应被10-Prompt原子批次无遗漏地覆盖."""

    assert unit_runtime.build_official_reference_unit_ranges(0, 70) == tuple(
        (start, start + 10) for start in range(0, 70, 10)
    )
    assert unit_runtime.build_official_reference_unit_ranges(30, 700)[-1] == (
        720,
        730,
    )
    with pytest.raises(ValueError, match="整除"):
        unit_runtime.build_official_reference_unit_ranges(0, 34)


PACKAGE_ROOT_CASES = (
    (
        "tree_ring",
        tree_runner.TREE_RING_PACKAGE_ROOT_FILE_WHITELIST,
        tree_runner.TREE_RING_PACKAGE_GENERATED_FILE_NAMES,
        tree_runner.collect_package_entries,
    ),
    (
        "gaussian_shading",
        gaussian_runner.GAUSSIAN_SHADING_PACKAGE_ROOT_FILE_WHITELIST,
        gaussian_runner.GAUSSIAN_SHADING_PACKAGE_GENERATED_FILE_NAMES,
        gaussian_runner.collect_package_entries,
    ),
    (
        "shallow_diffuse",
        shallow_runner.SHALLOW_DIFFUSE_PACKAGE_ROOT_FILE_WHITELIST,
        shallow_runner.SHALLOW_DIFFUSE_PACKAGE_GENERATED_FILE_NAMES,
        shallow_runner.collect_package_entries,
    ),
)


def _write_exact_package_root(
    output_dir: Path,
    allowed_paths: frozenset[str],
    optional_paths: frozenset[str],
) -> None:
    """写出一套包含嵌套官方事实文件的完整结果根夹具."""

    output_dir.mkdir(parents=True)
    for relative_path in sorted(allowed_paths - optional_paths):
        path = output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    unit_dir = output_dir / "scientific_units"
    unit_dir.mkdir()
    (unit_dir / "unit_0000000_0000010.json").write_text(
        "{}\n",
        encoding="utf-8",
    )


@pytest.mark.quick
@pytest.mark.parametrize(
    ("baseline_id", "allowed_paths", "optional_paths", "collector"),
    PACKAGE_ROOT_CASES,
)
@pytest.mark.parametrize("extra_kind", ("file", "directory", "symlink"))
def test_official_package_root_rejects_every_unregistered_entry_type(
    tmp_path: Path,
    baseline_id: str,
    allowed_paths: frozenset[str],
    optional_paths: frozenset[str],
    collector: Any,
    extra_kind: str,
) -> None:
    """三条打包路线均拒绝额外文件、目录和链接, 且不跟随链接."""

    output_dir = tmp_path / baseline_id
    _write_exact_package_root(output_dir, allowed_paths, optional_paths)
    accepted = collector(
        tmp_path,
        output_dir,
        output_dir / "future_archive.zip",
    )
    assert accepted
    if extra_kind == "file":
        (output_dir / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    elif extra_kind == "directory":
        (output_dir / "unexpected_directory").mkdir()
    else:
        link_path = output_dir / "unexpected_link"
        target = next(
            path
            for path in output_dir.iterdir()
            if path.is_file() and not path.is_symlink()
        )
        try:
            link_path.symlink_to(target)
        except OSError:
            pytest.skip("当前平台不允许创建测试符号链接")
    with pytest.raises(RuntimeError, match="白名单外|链接"):
        collector(
            tmp_path,
            output_dir,
            output_dir / "future_archive.zip",
        )


@pytest.mark.quick
@pytest.mark.parametrize(
    ("baseline_id", "allowed_paths", "optional_paths", "collector"),
    PACKAGE_ROOT_CASES,
)
def test_official_package_root_requires_every_runtime_whitelist_file(
    tmp_path: Path,
    baseline_id: str,
    allowed_paths: frozenset[str],
    optional_paths: frozenset[str],
    collector: Any,
) -> None:
    """除本次打包生成文件外, 任一运行时白名单文件缺失都必须闭锁."""

    output_dir = tmp_path / baseline_id
    _write_exact_package_root(output_dir, allowed_paths, optional_paths)
    required_path = output_dir / next(
        path
        for path in sorted(allowed_paths - optional_paths)
        if "reference_schema" in path
    )
    required_path.unlink()
    with pytest.raises(RuntimeError, match="缺少白名单必需文件"):
        collector(
            tmp_path,
            output_dir,
            output_dir / "future_archive.zip",
        )


@pytest.mark.quick
def test_tree_ring_units_resume_across_dependency_report_and_gpu_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同锁运行切换 GPU 或报告实例时应复用完整批次, 但不复用损坏成员."""

    import sys

    executable = Path(sys.executable).resolve()
    source_status = {
        "official_repository_commit": "e" * 40,
        "source_patch_sha256": "f" * 64,
        "source_worktree_digest": "1" * 64,
    }
    scientific_config = {
        "sample_count": 20,
        "start_index": 0,
        "unit_batch_size": 10,
        "official_model_id": "Manojb/stable-diffusion-2-1-base",
        "official_model_revision": "a" * 40,
        "dataset": "Gustavosta/Stable-Diffusion-Prompts",
        "dataset_revision": "e" * 40,
        "reference_model": "ViT-g-14",
        "model_snapshot_content_digest": "b" * 64,
        "openclip_checkpoint_sha256": "c" * 64,
        "openclip_snapshot_content_digest": "d" * 64,
        "gen_seed": 0,
        "w_seed": 999999,
        "w_channel": 3,
        "w_pattern": "ring",
        "with_tracking": True,
    }
    first_model_path = str(tmp_path / "session_a" / "model")
    first_checkpoint_path = str(tmp_path / "session_a" / "openclip.bin")

    def model_report(model_path: str) -> dict[str, Any]:
        """构造内容身份相同但 workspace 路径可变化的模型报告."""

        return {
            "effective_official_model_id": model_path,
            "official_model_id": scientific_config["official_model_id"],
            "official_model_revision": scientific_config["official_model_revision"],
            "model_snapshot_content": {
                "snapshot_content_digest": scientific_config[
                    "model_snapshot_content_digest"
                ]
            },
        }

    def openclip_report(checkpoint_path: str) -> dict[str, Any]:
        """构造内容身份相同但 checkpoint 路径可变化的报告."""

        return {
            "openclip_checkpoint_path": checkpoint_path,
            "openclip_model_name": scientific_config["reference_model"],
            "openclip_checkpoint_sha256": scientific_config[
                "openclip_checkpoint_sha256"
            ],
            "openclip_snapshot_content_digest": scientific_config[
                "openclip_snapshot_content_digest"
            ],
        }

    def fake_subprocess(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout_seconds: int,
        progress: object | None,
        progress_profile: str,
    ) -> SimpleNamespace:
        assert command[0] == str(executable)
        assert cwd == tmp_path
        assert timeout_seconds == 60
        assert "start=" in progress_profile
        context = json.loads(env[unit_runtime.UNIT_CONTEXT_ENV_NAME])
        observations = [
            {
                "prompt_index": prompt_index,
                "prompt_digest": hashlib.sha256(
                    f"prompt-{prompt_index}".encode("utf-8")
                ).hexdigest(),
                "prompt_seed_random": prompt_index,
                "no_w_metric": 2.0 + prompt_index,
                "w_metric": 1.0 + prompt_index,
                "w_no_sim": 0.2,
                "w_sim": 0.21,
            }
            for prompt_index in range(context["unit_start"], context["unit_end"])
        ]
        with _temporary_environment(
            {
                unit_runtime.UNIT_CONTEXT_ENV_NAME: env[
                    unit_runtime.UNIT_CONTEXT_ENV_NAME
                ],
                unit_runtime.UNIT_OUTPUT_ENV_NAME: env[
                    unit_runtime.UNIT_OUTPUT_ENV_NAME
                ],
            }
        ):
            unit_runtime.write_official_reference_source_unit_payload(
                baseline_id="tree_ring",
                observations=observations,
                random_identity_random={
                    "prompt_seed_schedule_digest_random": build_stable_digest(
                        [item["prompt_seed_random"] for item in observations]
                    )
                },
                torch_module=_FakeTorch,
            )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        unit_runtime,
        "run_quiet_subprocess_with_progress",
        fake_subprocess,
    )

    def command_builder(
        unit_start: int,
        unit_end: int,
        _workspace: Path,
    ) -> tuple[list[str], Path, dict[str, str]]:
        return (
            [
                str(executable),
                str(tmp_path / "session_a" / "run_tree_ring_watermark.py"),
                "--run_name",
                "tree_ring_official_reference",
                "--model_id",
                first_model_path,
                "--dataset",
                str(scientific_config["dataset"]),
                "--w_channel",
                "3",
                "--w_pattern",
                "ring",
                "--gen_seed",
                "0",
                "--w_seed",
                "999999",
                "--start",
                str(unit_start),
                "--end",
                str(unit_end),
                "--reference_model",
                "ViT-g-14",
                "--reference_model_pretrain",
                first_checkpoint_path,
                "--with_tracking",
            ],
            tmp_path,
            {},
        )

    first = unit_runtime.run_official_reference_unit_schedule(
        root_path=tmp_path,
        baseline_id="tree_ring",
        start_index=0,
        sample_count=20,
        batch_size=10,
        scientific_config=scientific_config,
        formal_execution_lock=_formal_lock(),
        source_status=source_status,
        dependency_environment_report=_dependency_report("2" * 64),
        device_report=_device_report("NVIDIA T4"),
        model_repository_report=model_report(first_model_path),
        openclip_report=openclip_report(first_checkpoint_path),
        dependency_python_executable=executable,
        unit_dir=tmp_path / "units",
        workspace_root=tmp_path / "workspace",
        timeout_seconds=60,
        command_builder=command_builder,
    )
    assert first["official_unit_executed_count"] == 2
    assert first["official_unit_completed_count"] == 2
    assert first["official_unit_coverage_ready"] is True
    persisted = unit_runtime.validate_persisted_official_reference_units(
        unit_dir=tmp_path / "units",
        baseline_id="tree_ring",
        start_index=0,
        sample_count=20,
        batch_size=10,
    )
    assert persisted["official_unit_records_digest"] == first[
        "official_unit_records_digest"
    ]
    assert persisted["metric_summary"]["sample_count"] == 20
    second_model_path = str(tmp_path / "session_b" / "model")
    second_checkpoint_path = str(tmp_path / "session_b" / "openclip.bin")
    run_summary = {
        "sample_count": 20,
        "start_index": 0,
        "official_unit_batch_size": 10,
        "model_source_repository_id": scientific_config["official_model_id"],
        "model_source_revision": scientific_config["official_model_revision"],
        "prompt_dataset_repository_id": scientific_config["dataset"],
        "prompt_dataset_revision": scientific_config["dataset_revision"],
        "model_snapshot_content_digest": scientific_config[
            "model_snapshot_content_digest"
        ],
        "openclip_checkpoint_sha256": scientific_config[
            "openclip_checkpoint_sha256"
        ],
        "openclip_snapshot_content_digest": scientific_config[
            "openclip_snapshot_content_digest"
        ],
    }
    unit_runtime.validate_official_reference_scientific_config_and_commands(
        baseline_id="tree_ring",
        scientific_config=scientific_config,
        unit_commands=persisted["official_unit_commands"],
        run_summary=run_summary,
        model_repository_report=model_report(second_model_path),
        openclip_report=openclip_report(second_checkpoint_path),
    )

    residual_workspace = tmp_path / "workspace" / "interrupted-after-record"
    residual_workspace.mkdir(parents=True)
    (residual_workspace / "source_unit_payload.json").write_text(
        "stale",
        encoding="utf-8",
    )

    second = unit_runtime.run_official_reference_unit_schedule(
        root_path=tmp_path,
        baseline_id="tree_ring",
        start_index=0,
        sample_count=20,
        batch_size=10,
        scientific_config=scientific_config,
        formal_execution_lock=_formal_lock(),
        source_status=source_status,
        dependency_environment_report=_dependency_report("3" * 64),
        device_report=_device_report("NVIDIA L4"),
        model_repository_report=model_report(second_model_path),
        openclip_report=openclip_report(second_checkpoint_path),
        dependency_python_executable=executable,
        unit_dir=tmp_path / "units",
        workspace_root=tmp_path / "workspace",
        timeout_seconds=60,
        command_builder=command_builder,
    )
    assert second["official_unit_executed_count"] == 0
    assert second["official_unit_resumed_count"] == 2
    assert not (tmp_path / "workspace").exists()
    assert second["official_unit_records_digest"] == first[
        "official_unit_records_digest"
    ]

    extra_directory = tmp_path / "units" / "unexpected"
    extra_directory.mkdir()
    with pytest.raises(RuntimeError, match="exact-set"):
        unit_runtime.validate_persisted_official_reference_units(
            unit_dir=tmp_path / "units",
            baseline_id="tree_ring",
            start_index=0,
            sample_count=20,
            batch_size=10,
        )
    extra_directory.rmdir()

    extra_file = tmp_path / "units" / "unexpected.txt"
    extra_file.write_text("unexpected", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact-set"):
        unit_runtime.validate_persisted_official_reference_units(
            unit_dir=tmp_path / "units",
            baseline_id="tree_ring",
            start_index=0,
            sample_count=20,
            batch_size=10,
        )
    extra_file.unlink()

    link_path = tmp_path / "units" / "unexpected_link.json"
    try:
        link_path.symlink_to(sorted((tmp_path / "units").glob("unit_*.json"))[0])
    except OSError:
        pass
    else:
        with pytest.raises(RuntimeError, match="符号链接"):
            unit_runtime.validate_persisted_official_reference_units(
                unit_dir=tmp_path / "units",
                baseline_id="tree_ring",
                start_index=0,
                sample_count=20,
                batch_size=10,
            )
        link_path.unlink()

    unit_path = sorted((tmp_path / "units").glob("unit_*.json"))[0]
    original_unit_text = unit_path.read_text(encoding="utf-8")
    tampered_config = json.loads(original_unit_text)
    tampered_config["unit_identity"]["scientific_config"]["gen_seed"] = 1
    unit_path.write_text(json.dumps(tampered_config), encoding="utf-8")
    with pytest.raises(ValueError, match="身份摘要"):
        unit_runtime.validate_persisted_official_reference_units(
            unit_dir=tmp_path / "units",
            baseline_id="tree_ring",
            start_index=0,
            sample_count=20,
            batch_size=10,
        )
    unit_path.write_text(original_unit_text, encoding="utf-8")

    damaged = json.loads(unit_path.read_text(encoding="utf-8"))
    damaged["source_payload"]["observations"].pop()
    unit_path.write_text(json.dumps(damaged), encoding="utf-8")
    with pytest.raises(RuntimeError, match="损坏"):
        unit_runtime.run_official_reference_unit_schedule(
            root_path=tmp_path,
            baseline_id="tree_ring",
            start_index=0,
            sample_count=20,
            batch_size=10,
            scientific_config=scientific_config,
            formal_execution_lock=_formal_lock(),
            source_status=source_status,
            dependency_environment_report=_dependency_report("4" * 64),
            device_report=_device_report("NVIDIA L4"),
            model_repository_report=model_report(second_model_path),
            openclip_report=openclip_report(second_checkpoint_path),
            dependency_python_executable=executable,
            unit_dir=tmp_path / "units",
            workspace_root=tmp_path / "workspace",
            timeout_seconds=60,
            command_builder=command_builder,
        )


@pytest.mark.quick
def test_official_scientific_config_rejects_command_parameter_tampering() -> None:
    """批次 argv 的 guidance 等科学参数必须与规范配置完全一致."""

    config = {
        "sample_count": 20,
        "start_index": 0,
        "unit_batch_size": 10,
        "official_model_id": "Manojb/stable-diffusion-2-1-base",
        "official_model_revision": "a" * 40,
        "dataset_revision": "e" * 40,
        "model_snapshot_content_digest": "b" * 64,
        "openclip_checkpoint_sha256": "c" * 64,
        "openclip_snapshot_content_digest": "d" * 64,
        "fpr": 0.000001,
        "channel_copy": 1,
        "hw_copy": 8,
        "user_number": 1000000,
        "gen_seed": 0,
        "image_length": 512,
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
        "num_inversion_steps": 50,
        "dataset_path": "Gustavosta/Stable-Diffusion-Prompts",
        "reference_model": "ViT-g-14",
        "use_chacha": True,
    }
    summary = {
        "sample_count": 20,
        "start_index": 0,
        "official_unit_batch_size": 10,
        "model_source_repository_id": "Manojb/stable-diffusion-2-1-base",
        "model_source_revision": "a" * 40,
        "prompt_dataset_repository_id": "Gustavosta/Stable-Diffusion-Prompts",
        "prompt_dataset_revision": "e" * 40,
        "model_snapshot_content_digest": "b" * 64,
        "openclip_checkpoint_sha256": "c" * 64,
        "openclip_snapshot_content_digest": "d" * 64,
    }
    model_report = {
        "effective_official_model_id": "/models/stable-diffusion-2-1",
        "official_model_id": "Manojb/stable-diffusion-2-1-base",
        "official_model_revision": "a" * 40,
        "model_snapshot_content": {"snapshot_content_digest": "b" * 64},
    }
    openclip_report = {
        "openclip_checkpoint_path": "/models/openclip.bin",
        "openclip_model_name": "ViT-g-14",
        "openclip_checkpoint_sha256": "c" * 64,
        "openclip_snapshot_content_digest": "d" * 64,
    }

    def command(unit_start: int) -> list[str]:
        """构造一个参数完整的 Gaussian Shading 批次命令."""

        return [
            "/python",
            "/repo/run_gaussian_shading.py",
            "--num",
            "10",
            "--fpr",
            "1e-06",
            "--channel_copy",
            "1",
            "--hw_copy",
            "8",
            "--user_number",
            "1000000",
            "--gen_seed",
            "0",
            "--image_length",
            "512",
            "--guidance_scale",
            "7.5",
            "--num_inference_steps",
            "50",
            "--num_inversion_steps",
            "50",
            "--dataset_path",
            "Gustavosta/Stable-Diffusion-Prompts",
            "--model_path",
            "/models/stable-diffusion-2-1",
            "--output_path",
            f"/outputs/scientific_unit_workspace/{unit_start}/official_output/",
            "--chacha",
            "--reference_model",
            "ViT-g-14",
            "--reference_model_pretrain",
            "/models/openclip.bin",
        ]

    commands = [
        {"unit_start": 0, "unit_end": 10, "command": command(0)},
        {"unit_start": 10, "unit_end": 20, "command": command(10)},
    ]
    for unit_command in commands:
        unit_command["canonical_identity"] = (
            unit_runtime.build_official_reference_canonical_command_identity(
                baseline_id="gaussian_shading",
                command=unit_command["command"],
                unit_start=unit_command["unit_start"],
                unit_end=unit_command["unit_end"],
                scientific_config=config,
                runtime_model_locator=model_report[
                    "effective_official_model_id"
                ],
                runtime_openclip_checkpoint_locator=openclip_report[
                    "openclip_checkpoint_path"
                ],
            )
        )
    unit_runtime.validate_official_reference_scientific_config_and_commands(
        baseline_id="gaussian_shading",
        scientific_config=config,
        unit_commands=commands,
        run_summary=summary,
        model_repository_report=model_report,
        openclip_report=openclip_report,
    )

    tampered_commands = json.loads(json.dumps(commands))
    guidance_index = tampered_commands[0]["command"].index("--guidance_scale") + 1
    tampered_commands[0]["command"][guidance_index] = "6.0"
    with pytest.raises(ValueError, match="argv"):
        unit_runtime.validate_official_reference_scientific_config_and_commands(
            baseline_id="gaussian_shading",
            scientific_config=config,
            unit_commands=tampered_commands,
            run_summary=summary,
            model_repository_report=model_report,
            openclip_report=openclip_report,
        )
