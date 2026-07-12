"""验证方法忠实外部基线的真实原子单元恢复与 exact-set 门禁."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from external_baseline.primary import sd35_method_faithful_units as units
from main.core.digest import build_stable_digest
from tests.helpers.formal_execution_lock import build_test_formal_execution_lock


pytestmark = pytest.mark.quick


class FakeCuda:
    """提供科学单元来源构造所需的确定性 CUDA 身份."""

    @staticmethod
    def is_available() -> bool:
        """声明测试 CUDA 可用."""

        return True

    @staticmethod
    def current_device() -> int:
        """返回唯一可见设备索引."""

        return 0

    @staticmethod
    def device_count() -> int:
        """返回唯一可见设备数量."""

        return 1

    @staticmethod
    def get_device_capability(_index: int) -> tuple[int, int]:
        """返回 T4 的计算能力."""

        return (7, 5)

    @staticmethod
    def get_device_name(_index: int) -> str:
        """返回与依赖报告一致的 GPU 名称."""

        return "NVIDIA T4"


class FakeTorch:
    """模拟逐单元来源构造读取的 PyTorch 公开身份."""

    __version__ = "2.11.0+cu128"
    version = SimpleNamespace(cuda="12.8")
    cuda = FakeCuda()


def _write_fake_source_repository(root: Path) -> None:
    """写出源码身份构造所需的最小受治理仓库布局."""

    registry_path = root / "external_baseline/source_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "baseline_sources": [
                    {
                        "baseline_id": "tree_ring",
                        "official_repository_url": "https://github.com/example/tree-ring.git",
                        "official_repository_commit": "a" * 40,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    for relative_path in (
        "external_baseline/primary/tree_ring/adapter/run_slm_eval.py",
        "external_baseline/primary/tree_ring/adapter/method_faithful_sd35.py",
        "external_baseline/primary/sd35_method_faithful_common.py",
        "external_baseline/primary/sd35_method_faithful_units.py",
    ):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative_path}\n", encoding="utf-8")


def _runtime_environment(report_digest: str, *, lock_digest: str | None = None) -> dict[str, object]:
    """构造仅报告摘要可变化的真实环境身份测试夹具."""

    execution_lock = build_test_formal_execution_lock("b" * 40)
    return {
        "dependency_environment_ready": True,
        "isolated_scientific_context_ready": True,
        "formal_execution_lock_ready": True,
        "dependency_profile_id": "sd35_method_runtime_gpu",
        "dependency_profile_digest": "1" * 64,
        "direct_requirements_digest": "2" * 64,
        "complete_hash_lock_digest": lock_digest or "3" * 64,
        "formal_execution_commit": execution_lock["formal_execution_commit"],
        "formal_execution_lock_digest": execution_lock["formal_execution_lock_digest"],
        "python_version": "3.12.13",
        "package_versions": {"torch": "2.11.0+cu128"},
        "cuda_version": "12.8",
        "gpu_name": "NVIDIA T4",
        "device_count": 1,
        "isolated_scientific_context": {
            "dependency_environment_report_actual_digest": report_digest,
            "current_python_executable_sha256": "5" * 64,
        },
    }


def _build_context(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, object],
    *,
    run_config: dict[str, object] | None = None,
) -> units.MethodFaithfulUnitContext:
    """在临时仓库中构造方法忠实单元上下文."""

    monkeypatch.setattr(
        units.repository_environment,
        "require_published_formal_execution_lock",
        lambda _root: build_test_formal_execution_lock("b" * 40),
    )
    monkeypatch.setattr(
        units.repository_environment,
        "build_runtime_environment_report",
        lambda *_args, **_kwargs: dict(environment),
    )
    return units.build_method_faithful_unit_context(
        baseline_id="tree_ring",
        artifact_root=root / "outputs/method_faithful/artifacts",
        run_config=run_config
        or {
            "prompt_count": 1,
            "test_prompt_count": 0,
            "attack_families": [],
        },
        execution_device="cuda:0",
        torch_module=FakeTorch(),
        root=root,
    )


def test_completed_unit_resume_uses_locks_not_report_instance_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """报告实例或 GPU 型号变化不应阻断相同代码锁和完整依赖锁的恢复."""

    _write_fake_source_repository(tmp_path)
    context = _build_context(tmp_path, monkeypatch, _runtime_environment("4" * 64))
    prompt_row = {
        "prompt_id": "prompt_000001",
        "prompt_index": 0,
        "prompt_set": "probe_paper",
        "split": "calibration",
        "prompt_text": "a ceramic fox",
        "prompt_digest": build_stable_digest("a ceramic fox"),
    }
    spec = units.build_method_faithful_unit_spec(
        context,
        unit_kind="source_pair",
        row=prompt_row,
        index=1,
        random_identity_random={"generation_seed_random": 17},
        unit_parameters={"image_id": "image_000001"},
    )
    artifact_path = context.artifact_root / "images/source.png"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"real-image-bytes")
    written = units.write_completed_method_faithful_unit(
        context,
        spec,
        unit_data={
            "measured_score": 0.25,
            "image_path": str(artifact_path),
        },
        artifact_paths=(artifact_path,),
    )

    moved_root = tmp_path.parent / f"{tmp_path.name}_moved"
    shutil.copytree(tmp_path, moved_root)
    abandoned_temp = (
        moved_root
        / "outputs/method_faithful/artifacts"
        / ".method_faithful_run_identity.json.999.tmp"
    )
    abandoned_temp.write_text("partial", encoding="utf-8")
    resumed_environment = _runtime_environment("6" * 64)
    resumed_environment["gpu_name"] = "NVIDIA L4"
    resumed_context = _build_context(
        moved_root,
        monkeypatch,
        resumed_environment,
    )
    resumed_spec = units.build_method_faithful_unit_spec(
        resumed_context,
        unit_kind="source_pair",
        row=prompt_row,
        index=1,
        random_identity_random={"generation_seed_random": 17},
        unit_parameters={"image_id": "image_000001"},
    )
    resumed = units.load_completed_method_faithful_unit(
        resumed_context,
        resumed_spec,
    )

    assert resumed == written
    assert not abandoned_temp.exists()
    assert not Path(str(resumed["unit_data"]["image_path"])).is_absolute()
    assert units.resolve_method_faithful_output_path(
        resumed_context,
        resumed["unit_data"]["image_path"],
    ).is_file()
    assert resumed_context.run_config_digest == context.run_config_digest
    assert (
        resumed["scientific_unit_provenance"]["scientific_execution_environment"]
        ["dependency_environment_report_digest"]
        == "4" * 64
    )


def test_completed_unit_rejects_lock_change_and_artifact_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整依赖锁变化或图像字节损坏都不得复用已有单元."""

    _write_fake_source_repository(tmp_path)
    context = _build_context(tmp_path, monkeypatch, _runtime_environment("4" * 64))
    row = {
        "prompt_id": "prompt_000001",
        "prompt_index": 0,
        "prompt_set": "probe_paper",
        "split": "calibration",
        "prompt_text": "a ceramic fox",
        "prompt_digest": build_stable_digest("a ceramic fox"),
    }
    spec = units.build_method_faithful_unit_spec(
        context,
        unit_kind="source_pair",
        row=row,
        index=1,
        random_identity_random={"generation_seed_random": 17},
        unit_parameters={"image_id": "image_000001"},
    )
    artifact_path = context.artifact_root / "images/source.png"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"real-image-bytes")
    units.write_completed_method_faithful_unit(
        context,
        spec,
        unit_data={"measured_score": 0.25},
        artifact_paths=(artifact_path,),
    )

    with pytest.raises(ValueError, match="原子运行身份"):
        _build_context(
            tmp_path,
            monkeypatch,
            _runtime_environment("4" * 64, lock_digest="9" * 64),
        )

    artifact_path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="产物摘要"):
        units.load_completed_method_faithful_unit(context, spec)


def _source_record(prompt_id: str, split: str) -> dict[str, object]:
    """构造 exact-set validator 使用的源图单元."""

    return {
        "unit_kind": "source_pair",
        "prompt_identity": {"prompt_id": prompt_id, "split": split},
        "unit_data": {
            "observations_without_threshold": [
                {"prompt_id": prompt_id, "sample_role": "clean_negative", "score": 0.1},
                {"prompt_id": prompt_id, "sample_role": "positive_source", "score": 0.9},
            ]
        },
    }


def _attack_record(prompt_id: str, split: str, role: str) -> dict[str, object]:
    """构造 exact-set validator 使用的攻击单元."""

    return {
        "unit_kind": f"formal_attack_jpeg_compression_{role}",
        "prompt_identity": {"prompt_id": prompt_id, "split": split},
        "unit_data": {
            "observation_without_threshold": {
                "prompt_id": prompt_id,
                "sample_role": role,
                "attack_name": "jpeg_compression",
                "score": 0.5,
            }
        },
    }


def test_unit_exact_set_attacks_only_test_prompts_with_two_roles() -> None:
    """攻击单元必须精确覆盖 test Prompt×攻击×阴阳角色, 不得攻击 calibration."""

    run_config = {
        "prompt_count": 2,
        "test_prompt_count": 1,
        "attack_families": ["jpeg_compression"],
    }
    records = [
        _source_record("calibration_prompt", "calibration"),
        _source_record("test_prompt", "test"),
        _attack_record("test_prompt", "test", "attacked_negative"),
        _attack_record("test_prompt", "test", "attacked_positive"),
    ]

    report = units._validate_method_faithful_unit_exact_set(records, run_config)

    assert report["source_prompt_count"] == 2
    assert report["test_prompt_count"] == 1
    assert report["actual_formal_attack_unit_count"] == 2
    records[-1] = _attack_record("calibration_prompt", "calibration", "attacked_positive")
    with pytest.raises(ValueError, match="只能绑定 test Prompt"):
        units._validate_method_faithful_unit_exact_set(records, run_config)


def test_unit_exact_set_rejects_source_role_order_and_missing_attack_role() -> None:
    """源图角色顺序错误或攻击缺少一类角色都必须闭锁."""

    run_config = {
        "prompt_count": 1,
        "test_prompt_count": 1,
        "attack_families": ["jpeg_compression"],
    }
    source = _source_record("test_prompt", "test")
    source["unit_data"]["observations_without_threshold"].reverse()  # type: ignore[index,union-attr]
    with pytest.raises(ValueError, match="角色顺序"):
        units._validate_method_faithful_unit_exact_set(
            [
                source,
                _attack_record("test_prompt", "test", "attacked_negative"),
                _attack_record("test_prompt", "test", "attacked_positive"),
            ],
            run_config,
        )
    with pytest.raises(ValueError, match="攻击 exact set"):
        units._validate_method_faithful_unit_exact_set(
            [
                _source_record("test_prompt", "test"),
                _attack_record("test_prompt", "test", "attacked_positive"),
            ],
            run_config,
        )


def test_runner_recomputes_observations_from_complete_unit_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """runner 必须重读原子单元并确定性复算最终 observation 与来源聚合."""

    _write_fake_source_repository(tmp_path)
    run_config = {
        "prompt_count": 2,
        "test_prompt_count": 1,
        "attack_families": ["jpeg_compression"],
    }
    context = _build_context(
        tmp_path,
        monkeypatch,
        _runtime_environment("4" * 64),
        run_config=run_config,
    )
    records: list[dict[str, object]] = []
    specs: list[units.MethodFaithfulUnitSpec] = []
    for index, (prompt_id, split) in enumerate(
        (("calibration_prompt", "calibration"), ("test_prompt", "test")),
        start=1,
    ):
        prompt_row = {
            "prompt_id": prompt_id,
            "prompt_index": index - 1,
            "prompt_set": "probe_paper",
            "split": split,
            "prompt_text": f"prompt text {index}",
            "prompt_digest": build_stable_digest(f"prompt text {index}"),
        }
        spec = units.build_method_faithful_unit_spec(
            context,
            unit_kind="source_pair",
            row=prompt_row,
            index=index,
            random_identity_random={"generation_seed_random": index},
            unit_parameters={"image_id": f"image_{index}"},
        )
        artifact_path = context.artifact_root / f"images/source_{index}.png"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(f"source-{index}".encode("ascii"))
        records.append(
            units.write_completed_method_faithful_unit(
                context,
                spec,
                unit_data={
                    "image_pair": {"prompt_id": prompt_id},
                    "observations_without_threshold": [
                        {
                            "prompt_id": prompt_id,
                            "sample_role": "clean_negative",
                            "score": 0.1 * index,
                        },
                        {
                            "prompt_id": prompt_id,
                            "sample_role": "positive_source",
                            "score": 0.8,
                        },
                    ],
                },
                artifact_paths=(artifact_path,),
            )
        )
        specs.append(spec)

    test_prompt_row = {
        "prompt_id": "test_prompt",
        "prompt_index": 1,
        "prompt_set": "probe_paper",
        "split": "test",
        "prompt_text": "prompt text 2",
        "prompt_digest": build_stable_digest("prompt text 2"),
    }
    for offset, role in enumerate(("attacked_negative", "attacked_positive"), start=3):
        spec = units.build_method_faithful_unit_spec(
            context,
            unit_kind=f"formal_attack_jpeg_compression_{role}",
            row=test_prompt_row,
            index=2,
            random_identity_random={"attack_seed_random": offset},
            unit_parameters={"sample_role": role},
        )
        artifact_path = context.artifact_root / f"images/attack_{role}.png"
        artifact_path.write_bytes(role.encode("ascii"))
        records.append(
            units.write_completed_method_faithful_unit(
                context,
                spec,
                unit_data={
                    "observation_without_threshold": {
                        "prompt_id": "test_prompt",
                        "sample_role": role,
                        "attack_name": "jpeg_compression",
                        "score": 0.3 if role == "attacked_negative" else 0.7,
                    },
                    "attacked_record": {"sample_role": role},
                },
                artifact_paths=(artifact_path,),
            )
        )
        specs.append(spec)

    aggregate = units.aggregate_method_faithful_unit_records(
        context,
        records,
        expected_specs=specs,
    )
    threshold = 0.5
    observations = units.apply_frozen_threshold(
        [
            row
            for record in records
            for row in (
                record["unit_data"]["observations_without_threshold"]
                if record["unit_kind"] == "source_pair"
                else [record["unit_data"]["observation_without_threshold"]]
            )
        ],
        threshold=threshold,
        threshold_source="calibration_clean_negative_conformal",
    )
    manifest = {
        "baseline_id": "tree_ring",
        "test_prompt_count": 1,
        "expected_formal_attack_unit_count": 2,
        "threshold": threshold,
        "threshold_source": "calibration_clean_negative_conformal",
        "run_config": context.run_config,
        "run_config_digest": context.run_config_digest,
        "stable_scientific_execution_identity": context.stable_execution_identity,
        "stable_scientific_execution_identity_digest": context.stable_execution_identity_digest,
        "method_faithful_source_identity": context.source_identity,
        **aggregate,
    }
    manifest["adapter_digest"] = build_stable_digest(manifest)

    report = units.validate_method_faithful_adapter_unit_evidence(
        manifest=manifest,
        observation_rows=observations,
        root=tmp_path,
    )

    assert report["method_faithful_source_prompt_unit_count"] == 2
    assert report["method_faithful_formal_attack_unit_count"] == 2
    stale_image_path = context.artifact_root / "images/stale_from_other_run.png"
    stale_image_path.write_bytes(b"stale")
    with pytest.raises(ValueError, match="未绑定旧文件"):
        units.aggregate_method_faithful_unit_records(
            context,
            records,
            expected_specs=specs,
        )
    stale_image_path.unlink()
    observations[-1]["score"] = 0.6
    with pytest.raises(ValueError, match="确定性重建"):
        units.validate_method_faithful_adapter_unit_evidence(
            manifest=manifest,
            observation_rows=observations,
            root=tmp_path,
        )
