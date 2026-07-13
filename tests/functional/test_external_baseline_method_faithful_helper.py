"""验证单 baseline method-faithful 正式运行与 transfer 边界。"""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from PIL import Image

from experiments.runtime import repository_environment
from experiments.protocol.attacks import attack_config_digest, resolve_formal_attack_config
from experiments.protocol.fixed_fpr_observation_audit import (
    audit_fixed_fpr_observation_threshold,
)
from experiments.protocol.formal_randomization import (
    formal_randomization_protocol_record,
    resolve_formal_randomization_repeat,
)
from external_baseline.primary.sd35_method_faithful_common import (
    apply_formal_image_attack,
    canonical_attack_family,
    canonical_attack_name,
    supported_formal_image_attack_names,
)
from external_baseline.primary import sd35_method_faithful_units as units
from main.core.digest import build_stable_digest
from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
    inspect_closure_package,
)
from paper_experiments.baselines.method_faithful_observation_collection import (
    canonical_prompt_protocol_digest,
)
from paper_experiments.runners.external_baseline_method_faithful import (
    DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    METHOD_FAITHFUL_BASELINE_IDS,
    ExternalBaselineMethodFaithfulConfig,
    _build_command_plan_command,
    output_paths,
    package_external_baseline_method_faithful_outputs,
    prepare_single_baseline_run_directory,
    resolve_primary_baseline_id,
    validate_formal_run_config,
    write_baseline_transfer_files,
)
from tests.helpers.formal_execution_lock import build_test_formal_execution_lock
from tests.helpers.scientific_execution_binding import (
    write_test_scientific_execution_binding,
)
from tests.helpers.method_faithful_collection import numerical_fidelity_report


PACKAGE_TEST_CODE_VERSION = "b" * 40
FORMAL_EXECUTION_LOCK = build_test_formal_execution_lock(PACKAGE_TEST_CODE_VERSION)


pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def _select_pilot_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    """本模块未显式切换层级的归档夹具固定使用 pilot_paper."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "pilot_paper")
    monkeypatch.setattr(
        repository_environment,
        "require_published_formal_execution_lock",
        lambda _root: dict(FORMAL_EXECUTION_LOCK),
    )


@pytest.fixture
def stub_unit_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """仅为 transfer writer 单元测试隔离原子单元构造成本."""

    monkeypatch.setattr(
        "paper_experiments.runners.external_baseline_method_faithful.validate_method_faithful_adapter_unit_evidence",
        lambda **_kwargs: {
            "method_faithful_scientific_unit_count": 1,
            "method_faithful_scientific_unit_records_digest": "1" * 64,
            "method_faithful_scientific_unit_resume_ready": True,
        },
    )


def write_json(path: Path, payload: object) -> None:
    """写出测试使用的稳定 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def thresholded_row(
    *,
    baseline_id: str,
    event_id: str,
    score: float,
    threshold: float,
    sample_role: str,
    attack_family: str,
    attack_name: str,
    prompt_id: str = "prompt_000001",
    prompt_text: str = "a ceramic fox",
    split: str = "calibration",
) -> dict[str, object]:
    """构造带真实阈值判定关系的最小 observation。"""

    row: dict[str, object] = {
        "event_id": event_id,
        "baseline_id": baseline_id,
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "split": split,
        "sample_role": sample_role,
        "attack_family": attack_family,
        "attack_name": attack_name,
        "score": score,
        "threshold": threshold,
        "threshold_source": "calibration_clean_negative_conformal",
        "detection_decision": score >= threshold,
        "generation_model_id": DEFAULT_MODEL_ID,
        "generation_model_revision": DEFAULT_MODEL_REVISION,
    }
    if sample_role in {"attacked_negative", "attacked_positive"}:
        attack_config = resolve_formal_attack_config(
            attack_family=attack_family,
            attack_name=attack_name,
        )
        row.update(
            {
                "attack_id": attack_config.attack_id,
                "resource_profile": attack_config.resource_profile,
                "attack_config_digest": attack_config_digest(attack_config),
            }
        )
    return row


def prepare_transfer_inputs(
    root: Path,
    config: ExternalBaselineMethodFaithfulConfig,
    *,
    declared_count_delta: int = 0,
) -> dict[str, Path]:
    """写出 transfer writer 所需的最小真实执行边界文件。"""

    paths = output_paths(root, config)
    baseline_id = config.primary_baseline_id
    clean_score = 0.1
    threshold = math.nextafter(clean_score, math.inf)
    rows = [
        thresholded_row(
            baseline_id=baseline_id,
            event_id="clean_negative",
            score=clean_score,
            threshold=threshold,
            sample_role="clean_negative",
            attack_family="clean",
            attack_name="clean_none",
        ),
        thresholded_row(
            baseline_id=baseline_id,
            event_id="positive_source",
            score=0.9,
            threshold=threshold,
            sample_role="positive_source",
            attack_family="clean",
            attack_name="clean_none",
        ),
        thresholded_row(
            baseline_id=baseline_id,
            event_id="attacked_positive",
            score=0.8,
            threshold=threshold,
            sample_role="attacked_positive",
            attack_family="standard_distortion",
            attack_name="jpeg_compression",
        ),
    ]
    declared_count = len(rows) + declared_count_delta
    write_json(paths["baseline_observations"], rows)
    write_json(
        paths["command_results"],
        [{"baseline_id": baseline_id, "return_code": 0, "observation_count": declared_count}],
    )
    write_json(paths["execution_manifest"], {"observation_count": declared_count})
    write_json(
        paths["primary_prompt_plan"],
        [
            {
                "prompt_id": "prompt_000001",
                "prompt_index": 0,
                "prompt_set": config.prompt_set,
                "prompt_text": "a ceramic fox",
                "prompt_digest": "1" * 64,
                "split": "calibration",
            }
        ],
    )
    write_json(
        paths["adapter_output_root"]
        / baseline_id
        / f"{baseline_id}_method_faithful_sd35_adapter_manifest.json",
        {
            "baseline_id": baseline_id,
            "observation_count": declared_count,
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "generation_protocol": {
                "model_id": config.model_id,
                "model_revision": config.model_revision,
                "num_inference_steps": config.num_inference_steps,
                "guidance_scale": config.guidance_scale,
            },
        },
    )
    write_json(
        paths["numerical_fidelity_report"],
        numerical_fidelity_report(baseline_id),
    )
    return paths


def test_formal_image_attack_taxonomy_matches_attack_matrix_names() -> None:
    """baseline 应复用项目统一的17类真实攻击名称。"""

    image = Image.new("RGB", (16, 16), color=(128, 128, 128))
    expected_names = set(DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES.split(","))
    assert set(supported_formal_image_attack_names()) == expected_names
    assert len(expected_names) == 17
    assert canonical_attack_family("flow_matching_inversion_regeneration") == "regeneration_attack"
    assert canonical_attack_name("diffusion_purification") == "diffusion_purification"
    attacked_image, transform_name, execution = apply_formal_image_attack(
        image,
        attack_family="jpeg_compression",
        seed=17,
    )
    assert attacked_image.mode == "RGB"
    assert transform_name
    assert execution["attack_seed_random"] == 17


def test_method_faithful_runner_accepts_exactly_one_registered_baseline() -> None:
    """正式 runner 不得接受集合、通配符或 T2SMark 重复入口。"""

    assert tuple(resolve_primary_baseline_id(item) for item in METHOD_FAITHFUL_BASELINE_IDS) == METHOD_FAITHFUL_BASELINE_IDS
    for invalid in ("", "all", "*", "tree_ring,gaussian_shading", "t2smark"):
        with pytest.raises(ValueError, match="不支持的 method-faithful baseline id"):
            resolve_primary_baseline_id(invalid)


def test_command_plan_uses_fixed_fpr_and_shared_generation_budget(tmp_path: Path) -> None:
    """单 baseline 命令必须显式传入当前 FPR 和与主方法一致的生成预算。"""

    config = ExternalBaselineMethodFaithfulConfig(
        primary_baseline_id="tree_ring",
        target_fpr=0.1,
        num_inference_steps=20,
        num_inversion_steps=20,
        guidance_scale=4.5,
        primary_baseline_max_samples=70,
        tree_ring_attack_families="jpeg_compression",
        require_cuda=True,
    )
    paths = output_paths(tmp_path, config)
    command = _build_command_plan_command(tmp_path, config, paths, paths["primary_prompt_plan"])

    assert command[command.index("--methods") + 1] == "tree_ring"
    assert command[command.index("--target-fpr") + 1] == "0.1"
    assert command[command.index("--model-revision") + 1] == DEFAULT_MODEL_REVISION
    assert command[command.index("--num-inference-steps") + 1] == "20"
    assert command[command.index("--num-inversion-steps") + 1] == "20"
    assert command[command.index("--guidance-scale") + 1] == "4.5"
    assert "--require-cuda" in command


def test_formal_runner_rejects_generation_budget_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式 GPU 入口不得用环境覆盖生成预算后仍产出可导入结果。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    root_path = Path(__file__).resolve().parents[2]
    invalid = ExternalBaselineMethodFaithfulConfig(
        prompt_set="probe_paper",
        prompt_file="configs/paper_main_probe_paper_prompts.txt",
        primary_baseline_id="tree_ring",
        target_fpr=0.1,
        guidance_scale=7.0,
        primary_baseline_max_samples=70,
        require_cuda=True,
    )

    with pytest.raises(ValueError, match="公平预算"):
        validate_formal_run_config(root_path, invalid)


def test_formal_runner_rejects_model_revision_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式 common-backbone baseline 不得切换到未登记的模型 commit。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    with pytest.raises(ValueError, match="未登记"):
        ExternalBaselineMethodFaithfulConfig(
            prompt_set="probe_paper",
            prompt_file="configs/paper_main_probe_paper_prompts.txt",
            primary_baseline_id="tree_ring",
            model_revision="a" * 40,
            target_fpr=0.1,
            primary_baseline_max_samples=70,
            require_cuda=True,
        )

    with pytest.raises(ValueError, match="40位小写十六进制"):
        ExternalBaselineMethodFaithfulConfig(model_revision="main")


def test_transfer_manifest_binds_actual_observations_and_threshold(
    tmp_path: Path,
    stub_unit_evidence: None,
) -> None:
    """transfer manifest 必须绑定真实 observation 数量、摘要和冻结阈值。"""

    config = ExternalBaselineMethodFaithfulConfig(
        primary_baseline_id="tree_ring",
        target_fpr=0.1,
        primary_baseline_max_samples=1,
        tree_ring_attack_families="jpeg_compression",
        require_cuda=False,
    )
    paths = prepare_transfer_inputs(tmp_path, config)

    manifest = write_baseline_transfer_files(tmp_path, config, paths)

    assert manifest["baseline_id"] == "tree_ring"
    assert manifest["baseline_observation_count"] == 3
    assert len(manifest["baseline_observations_sha256"]) == 64
    assert len(manifest["threshold_digest"]) == 64
    assert manifest["generation_protocol"]["num_inference_steps"] == 20
    assert manifest["model_revision"] == DEFAULT_MODEL_REVISION
    assert manifest["detection_protocol"]["input_access_mode"] == "image_only"
    assert paths["split_observations"].is_file()
    assert paths["transfer_manifest"].is_file()


def test_transfer_manifest_rejects_declared_observation_count_mismatch(
    tmp_path: Path,
    stub_unit_evidence: None,
) -> None:
    """命令结果、执行 manifest 或 adapter manifest 不得虚报 observation 数量。"""

    config = ExternalBaselineMethodFaithfulConfig(
        primary_baseline_id="tree_ring",
        target_fpr=0.1,
        primary_baseline_max_samples=1,
        tree_ring_attack_families="jpeg_compression",
        require_cuda=False,
    )
    paths = prepare_transfer_inputs(tmp_path, config, declared_count_delta=1)
    with pytest.raises(ValueError, match="声明计数与实际计数不一致"):
        write_baseline_transfer_files(tmp_path, config, paths)


def test_run_directory_preparation_preserves_atomic_units_only(tmp_path: Path) -> None:
    """重跑应保留 adapter 原子单元, 清除其余派生状态和当前 transfer 文件。"""

    config = ExternalBaselineMethodFaithfulConfig(
        primary_baseline_id="tree_ring",
        target_fpr=0.1,
        primary_baseline_max_samples=1,
        require_cuda=False,
    )
    paths = output_paths(tmp_path, config)
    stale_run_file = paths["run_dir"] / "stale_probe_image.png"
    stale_run_file.parent.mkdir(parents=True, exist_ok=True)
    stale_run_file.write_bytes(b"stale")
    completed_unit = (
        paths["adapter_output_root"]
        / "tree_ring"
        / "artifacts"
        / "completed_scientific_units"
        / "u.json"
    )
    completed_unit.parent.mkdir(parents=True, exist_ok=True)
    write_json(completed_unit, {"unit_complete": True})
    stale_adapter_result = paths["adapter_output_root"] / "tree_ring" / "baseline_observations.json"
    write_json(stale_adapter_result, [{"stale": True}])
    for field_name in ("split_observations", "split_command_results", "transfer_manifest"):
        write_json(paths[field_name], {"baseline_id": "tree_ring", "stale": True})
    other_transfer = paths["split_observation_dir"] / "gaussian_shading_baseline_observations.json"
    write_json(other_transfer, [{"baseline_id": "gaussian_shading"}])

    prepare_single_baseline_run_directory(paths)

    assert paths["run_dir"].is_dir()
    assert not stale_run_file.exists()
    assert completed_unit.is_file()
    assert not stale_adapter_result.exists()
    assert all(
        not paths[field_name].exists()
        for field_name in ("split_observations", "split_command_results", "transfer_manifest")
    )
    assert other_transfer.is_file()


class _PackageFakeCuda:
    """提供原子科学来源夹具所需的确定性 CUDA 身份."""

    @staticmethod
    def is_available() -> bool:
        """声明测试 CUDA 可用."""

        return True

    @staticmethod
    def current_device() -> int:
        """返回唯一设备索引."""

        return 0

    @staticmethod
    def device_count() -> int:
        """返回唯一设备数量."""

        return 1

    @staticmethod
    def get_device_capability(_index: int) -> tuple[int, int]:
        """返回测试设备计算能力."""

        return (7, 5)

    @staticmethod
    def get_device_name(_index: int) -> str:
        """返回测试设备名称."""

        return "NVIDIA T4"


class _PackageFakeTorch:
    """提供科学来源构造读取的 PyTorch 公开身份."""

    __version__ = "2.11.0+cu128"
    version = SimpleNamespace(cuda="12.8")
    cuda = _PackageFakeCuda()


def _package_runtime_environment() -> dict[str, object]:
    """构造满足隔离依赖门禁的最小运行环境报告."""

    return {
        "dependency_environment_ready": True,
        "isolated_scientific_context_ready": True,
        "formal_execution_lock_ready": True,
        "dependency_profile_id": "sd35_method_runtime_gpu",
        "dependency_profile_digest": "1" * 64,
        "direct_requirements_digest": "2" * 64,
        "complete_hash_lock_digest": "3" * 64,
        "formal_execution_commit": PACKAGE_TEST_CODE_VERSION,
        "formal_execution_lock_digest": FORMAL_EXECUTION_LOCK[
            "formal_execution_lock_digest"
        ],
        "python_version": "3.12.13",
        "package_versions": {"torch": "2.11.0+cu128"},
        "cuda_version": "12.8",
        "gpu_name": "NVIDIA T4",
        "device_count": 1,
        "isolated_scientific_context": {
            "dependency_environment_report_actual_digest": "4" * 64,
            "current_python_executable_sha256": "5" * 64,
        },
    }


def _write_package_source_identity_inputs(root: Path) -> None:
    """写出三个 baseline 共用的受治理源码身份夹具."""

    registry_path = root / "external_baseline/source_registry.json"
    write_json(
        registry_path,
        {
            "baseline_sources": [
                {
                    "baseline_id": baseline_id,
                    "official_repository_url": f"https://example.test/{baseline_id}.git",
                    "official_repository_commit": "a" * 40,
                }
                for baseline_id in METHOD_FAITHFUL_BASELINE_IDS
            ]
        },
    )
    relative_paths = {
        "external_baseline/primary/sd35_method_faithful_common.py",
        "external_baseline/primary/sd35_method_faithful_units.py",
        *(
            f"external_baseline/primary/{baseline_id}/adapter/{file_name}"
            for baseline_id in METHOD_FAITHFUL_BASELINE_IDS
            for file_name in ("run_slm_eval.py", "method_faithful_sd35.py")
        ),
    }
    for relative_path in relative_paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative_path}\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    """计算夹具文件的 SHA-256."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_prompt_rows() -> list[dict[str, object]]:
    """构造一个 calibration Prompt 和一个 test Prompt."""

    return [
        {
            "prompt_id": f"prompt_{index:06d}",
            "prompt_index": index - 1,
            "prompt_set": "pilot_paper",
            "prompt_text": f"package prompt {index}",
            "prompt_digest": build_stable_digest(f"package prompt {index}"),
            "split": split,
        }
        for index, split in ((1, "calibration"), (2, "test"))
    ]


def prepare_package_source(
    root: Path,
    baseline_id: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_decision: str = "pass",
) -> dict[str, object]:
    """写出可由真实原子完成单元确定性重建的单 baseline 产物."""

    _write_package_source_identity_inputs(root)
    monkeypatch.setattr(
        units.repository_environment,
        "build_runtime_environment_report",
        lambda *_args, **_kwargs: _package_runtime_environment(),
    )
    output_root = root / "outputs" / "external_baseline_method_faithful" / "pilot_paper"
    run_dir = output_root / "run_records" / baseline_id
    split_dir = output_root / "split_observations"
    adapter_root = run_dir / "adapter_outputs" / baseline_id
    artifact_root = adapter_root / "artifacts"
    prompt_plan_path = run_dir / f"{baseline_id}_prompt_plan.json"
    prompt_rows = _package_prompt_rows()
    write_json(prompt_plan_path, prompt_rows)
    run_config = {
        "prompt_count": len(prompt_rows),
        "test_prompt_count": 1,
        "attack_families": ["jpeg_compression"],
    }
    context = units.build_method_faithful_unit_context(
        baseline_id=baseline_id,
        artifact_root=artifact_root,
        run_config=run_config,
        execution_device="cuda:0",
        torch_module=_PackageFakeTorch(),
        root=root,
    )
    records: list[dict[str, object]] = []
    specs: list[units.MethodFaithfulUnitSpec] = []
    raw_observations: list[dict[str, object]] = []
    image_paths: list[Path] = []
    for index, prompt_row in enumerate(prompt_rows, start=1):
        prompt_id = str(prompt_row["prompt_id"])
        split = str(prompt_row["split"])
        source_rows = [
            units.threshold_independent_observation(
                thresholded_row(
                    baseline_id=baseline_id,
                    event_id=f"{prompt_id}_{role}",
                    score=score,
                    threshold=0.5,
                    sample_role=role,
                    attack_family="clean",
                    attack_name="clean_none",
                    prompt_id=prompt_id,
                    prompt_text=str(prompt_row["prompt_text"]),
                    split=split,
                )
            )
            for role, score in (
                ("clean_negative", 0.1 if split == "calibration" else 0.2),
                ("positive_source", 0.9),
            )
        ]
        spec = units.build_method_faithful_unit_spec(
            context,
            unit_kind="source_pair",
            row=prompt_row,
            index=index,
            random_identity_random={"generation_seed_random": index},
            unit_parameters={"image_id": f"image_{index:06d}"},
        )
        source_images = []
        for role in ("clean", "watermarked"):
            image_path = artifact_root / "images" / f"{prompt_id}_{role}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(f"{baseline_id}-{prompt_id}-{role}".encode("ascii"))
            source_images.append(image_path)
            image_paths.append(image_path)
        records.append(
            units.write_completed_method_faithful_unit(
                context,
                spec,
                unit_data={"observations_without_threshold": source_rows},
                artifact_paths=source_images,
            )
        )
        specs.append(spec)
        raw_observations.extend(source_rows)

    test_prompt = prompt_rows[1]
    for offset, role in enumerate(("attacked_negative", "attacked_positive"), start=11):
        attack_row = units.threshold_independent_observation(
            thresholded_row(
                baseline_id=baseline_id,
                event_id=f"prompt_000002_jpeg_{role}",
                score=0.3 if role == "attacked_negative" else 0.8,
                threshold=0.5,
                sample_role=role,
                attack_family="standard_distortion",
                attack_name="jpeg_compression",
                prompt_id="prompt_000002",
                prompt_text=str(test_prompt["prompt_text"]),
                split="test",
            )
        )
        spec = units.build_method_faithful_unit_spec(
            context,
            unit_kind=f"formal_attack_jpeg_compression_{role}",
            row=test_prompt,
            index=2,
            random_identity_random={"attack_seed_random": offset},
            unit_parameters={"sample_role": role},
        )
        image_path = artifact_root / "images" / f"prompt_000002_jpeg_{role}.png"
        image_path.write_bytes(f"{baseline_id}-{role}".encode("ascii"))
        image_paths.append(image_path)
        records.append(
            units.write_completed_method_faithful_unit(
                context,
                spec,
                unit_data={"observation_without_threshold": attack_row},
                artifact_paths=(image_path,),
            )
        )
        specs.append(spec)
        raw_observations.append(attack_row)

    unit_aggregate = units.aggregate_method_faithful_unit_records(
        context,
        records,
        expected_specs=specs,
    )
    threshold = math.nextafter(0.1, math.inf)
    threshold_source = "calibration_clean_negative_conformal"
    observations = units.apply_frozen_threshold(
        raw_observations,
        threshold=threshold,
        threshold_source=threshold_source,
    )
    threshold_audit = audit_fixed_fpr_observation_threshold(
        observations,
        target_fpr=0.01,
        expected_calibration_negative_count=1,
    )
    assert threshold_audit.fixed_fpr_ready
    image_pairs_path = artifact_root / f"{baseline_id}_image_pairs.json"
    attacked_manifest_path = artifact_root / "attacked_image_manifest.json"
    write_json(image_pairs_path, [{"prompt_id": row["prompt_id"]} for row in prompt_rows])
    write_json(attacked_manifest_path, {"attacked_image_count": 2})
    adapter_observations_path = adapter_root / "baseline_observations.json"
    adapter_manifest_path = (
        adapter_root / f"{baseline_id}_method_faithful_sd35_adapter_manifest.json"
    )
    write_json(adapter_observations_path, observations)
    adapter_manifest = {
        "artifact_name": adapter_manifest_path.name,
        "baseline_id": baseline_id,
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_MODEL_REVISION,
        "prompt_plan_path": prompt_plan_path.relative_to(root).as_posix(),
        "baseline_observations_path": adapter_observations_path.relative_to(root).as_posix(),
        "artifact_root": artifact_root.relative_to(root).as_posix(),
        "image_pairs_path": image_pairs_path.relative_to(root).as_posix(),
        "attacked_image_manifest_path": attacked_manifest_path.relative_to(root).as_posix(),
        "observation_count": len(observations),
        "test_prompt_count": 1,
        "expected_formal_attack_unit_count": 2,
        "generation_protocol": {
            "model_id": DEFAULT_MODEL_ID,
            "model_revision": DEFAULT_MODEL_REVISION,
        },
        "threshold": threshold,
        "threshold_source": threshold_source,
        "run_config": context.run_config,
        "run_config_digest": context.run_config_digest,
        "stable_scientific_execution_identity": context.stable_execution_identity,
        "stable_scientific_execution_identity_digest": (
            context.stable_execution_identity_digest
        ),
        "method_faithful_source_identity": context.source_identity,
        "method_faithful_source_identity_digest": context.source_identity[
            "method_faithful_source_identity_digest"
        ],
        **unit_aggregate,
        "formal_result_claim": False,
        "supports_paper_claim": False,
    }
    adapter_manifest["adapter_digest"] = build_stable_digest(adapter_manifest)
    write_json(adapter_manifest_path, adapter_manifest)
    unit_evidence = units.validate_method_faithful_adapter_unit_evidence(
        manifest=adapter_manifest,
        observation_rows=observations,
        root=root,
    )

    execution_dir = run_dir / "execution"
    command_results = [
        {
            "baseline_id": baseline_id,
            "return_code": 0,
            "observation_count": len(observations),
        }
    ]
    write_json(execution_dir / "baseline_command_results.json", command_results)
    write_json(execution_dir / "baseline_observations.json", observations)
    write_json(
        execution_dir / "baseline_execution_manifest.json",
        {"observation_count": len(observations)},
    )
    write_json(execution_dir / "baseline_command_plan_manifest.json", {"command_count": 1})
    split_observations_path = split_dir / f"{baseline_id}_baseline_observations.json"
    split_command_results_path = split_dir / f"{baseline_id}_baseline_command_results.json"
    transfer_manifest_path = split_dir / f"{baseline_id}_baseline_transfer_manifest.json"
    write_json(split_observations_path, observations)
    write_json(split_command_results_path, command_results)
    fidelity_report = numerical_fidelity_report(baseline_id)
    numerical_fidelity_path = (
        run_dir / f"{baseline_id}_numerical_fidelity_report.json"
    )
    write_json(numerical_fidelity_path, fidelity_report)
    transfer_manifest = {
        "baseline_id": baseline_id,
        "baseline_observations_path": split_observations_path.relative_to(output_root).as_posix(),
        "baseline_observation_count": len(observations),
        "baseline_observations_sha256": _sha256(split_observations_path),
        "baseline_command_results_path": split_command_results_path.relative_to(output_root).as_posix(),
        "baseline_command_results_sha256": _sha256(split_command_results_path),
        "prompt_plan_path": prompt_plan_path.relative_to(output_root).as_posix(),
        "prompt_plan_sha256": _sha256(prompt_plan_path),
        "prompt_protocol_digest": canonical_prompt_protocol_digest(prompt_rows),
        "adapter_manifest_path": adapter_manifest_path.relative_to(output_root).as_posix(),
        "adapter_manifest_sha256": _sha256(adapter_manifest_path),
        "execution_manifest_path": (
            execution_dir / "baseline_execution_manifest.json"
        ).relative_to(output_root).as_posix(),
        "execution_manifest_sha256": _sha256(
            execution_dir / "baseline_execution_manifest.json"
        ),
        "numerical_fidelity_report_path": numerical_fidelity_path.relative_to(
            output_root
        ).as_posix(),
        "numerical_fidelity_report_sha256": _sha256(numerical_fidelity_path),
        "numerical_fidelity_report_digest": fidelity_report[
            "numerical_fidelity_report_digest"
        ],
        "numerical_fidelity_reference_mode": fidelity_report[
            "numerical_fidelity_reference_mode"
        ],
        "method_faithful_numerical_fidelity_ready": True,
        "paper_run_name": "pilot_paper",
        "target_fpr": 0.01,
        "threshold": threshold,
        "threshold_digest": threshold_audit.threshold_digest,
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_MODEL_REVISION,
        "generation_protocol": {
            "model_id": DEFAULT_MODEL_ID,
            "model_revision": DEFAULT_MODEL_REVISION,
        },
        "formal_attack_names": ["jpeg_compression"],
        "code_version": PACKAGE_TEST_CODE_VERSION,
        **unit_evidence,
        "transfer_ready": True,
    }
    write_json(transfer_manifest_path, transfer_manifest)
    summary = {
        "run_decision": run_decision,
        "external_baseline_method_faithful_ready": run_decision == "pass",
        "primary_baseline_adapter_ready": run_decision == "pass",
        "primary_baseline_id": baseline_id,
        "primary_baseline_observation_count": len(observations),
        "paper_run_name": "pilot_paper",
        "target_fpr": 0.01,
        "threshold_digest": transfer_manifest["threshold_digest"],
        "generation_protocol": {
            "model_id": DEFAULT_MODEL_ID,
            "model_revision": DEFAULT_MODEL_REVISION,
        },
        "numerical_fidelity_report_digest": fidelity_report[
            "numerical_fidelity_report_digest"
        ],
        "numerical_fidelity_reference_mode": fidelity_report[
            "numerical_fidelity_reference_mode"
        ],
        "method_faithful_numerical_fidelity_ready": True,
        **unit_evidence,
    }
    write_json(run_dir / f"{baseline_id}_summary.json", summary)
    write_json(
        run_dir / f"{baseline_id}_manifest.local.json",
        {
            "code_version": PACKAGE_TEST_CODE_VERSION,
            "formal_execution_run_lock": FORMAL_EXECUTION_LOCK,
            "config": {
                "prompt_set": "pilot_paper",
                "target_fpr": 0.01,
                "primary_baseline_id": baseline_id,
                "model_id": DEFAULT_MODEL_ID,
                "model_revision": DEFAULT_MODEL_REVISION,
            },
            "metadata": {
                "method_faithful_scientific_unit_resume_ready": True,
                "method_faithful_scientific_unit_records_digest": unit_evidence[
                    "method_faithful_scientific_unit_records_digest"
                ],
                "numerical_fidelity_report_digest": fidelity_report[
                    "numerical_fidelity_report_digest"
                ],
                "numerical_fidelity_reference_mode": fidelity_report[
                    "numerical_fidelity_reference_mode"
                ],
                "method_faithful_numerical_fidelity_ready": True,
            },
        },
    )
    for file_name in (
        f"{baseline_id}_baseline_command_plan.json",
        f"{baseline_id}_command_plan_builder_result.json",
        f"{baseline_id}_command_plan_runner_result.json",
        f"{baseline_id}_evidence_validation_result.json",
        f"{baseline_id}_environment_report.json",
    ):
        write_json(run_dir / file_name, {"baseline_id": baseline_id})
    progress_path = run_dir / f"{baseline_id}_progress_events.jsonl"
    progress_path.write_text("{}\n", encoding="utf-8")
    write_test_scientific_execution_binding(
        repository_root=root,
        artifact_dir=run_dir,
        artifact_role="external_baseline_method_faithful",
        paper_run_name="pilot_paper",
        profile_id="sd35_method_runtime_gpu",
        summary_file_name=f"{baseline_id}_summary.json",
        manifest_file_name=f"{baseline_id}_manifest.local.json",
        formal_execution_lock=FORMAL_EXECUTION_LOCK,
        execution_route="isolated_method_faithful_workflow",
        baseline_id=baseline_id,
    )
    return {
        "output_root": output_root,
        "run_dir": run_dir,
        "transfer_manifest": transfer_manifest_path,
        "split_observations": split_observations_path,
        "unit_records": [spec.record_path for spec in specs],
        "image_paths": image_paths,
    }


def _patch_two_prompt_package_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """把归档层级缩到仍包含 calibration 和 test 的真实两 Prompt 协议."""

    repeat = resolve_formal_randomization_repeat(None)
    monkeypatch.setattr(
        "paper_experiments.runners.external_baseline_method_faithful.build_paper_run_config",
        lambda _root=".": SimpleNamespace(
            run_name="pilot_paper",
            prompt_count=2,
            target_fpr=0.01,
            randomization_repeat_id=repeat.randomization_repeat_id,
            generation_seed_index=repeat.generation_seed_index,
            generation_seed_offset=repeat.generation_seed_offset,
            watermark_key_index=repeat.watermark_key_index,
            formal_randomization_protocol_digest=(
                formal_randomization_protocol_record()[
                    "formal_randomization_protocol_digest"
                ]
            ),
            drive_dir=lambda child_name: f"drive/{child_name}",
        ),
    )


def _short_package_root(tmp_path: Path) -> Path:
    """在 pytest 会话目录内创建适配 Windows 路径长度限制的短根目录."""

    token = hashlib.sha256(tmp_path.name.encode("utf-8")).hexdigest()[:6]
    root = tmp_path.parent / f"m{token}"
    root.mkdir()
    if os.name == "nt":
        return Path("\\\\?\\" + str(root.resolve()))
    return root


def test_packages_are_baseline_isolated_and_failure_is_not_packaged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不同 baseline 包不得包含同名业务产物，失败运行不得打包。"""

    monkeypatch.setattr(
        "paper_experiments.runners.external_baseline_method_faithful.resolve_code_version",
        lambda _root: PACKAGE_TEST_CODE_VERSION,
    )
    _patch_two_prompt_package_run(monkeypatch)
    root = _short_package_root(tmp_path)
    drive_dir = root / "drive"
    archive_entries: list[set[str]] = []
    specs = {
        spec.baseline_id: spec
        for spec in CLOSURE_PACKAGE_FAMILY_SPECS
        if spec.package_family.startswith("method_faithful_")
    }
    for baseline_id in METHOD_FAITHFUL_BASELINE_IDS:
        prepare_package_source(root, baseline_id, monkeypatch)
        archive_name = f"external_baseline_method_faithful_package_{baseline_id}_test.zip"
        record = package_external_baseline_method_faithful_outputs(
            root=root,
            drive_output_dir=str(drive_dir),
            archive_name=archive_name,
            baseline_id=baseline_id,
        )
        with ZipFile(root / record.archive_path) as archive:
            archive_entries.append(set(archive.namelist()))
        run_prefix = (
            "outputs/external_baseline_method_faithful/pilot_paper/"
            f"run_records/{baseline_id}/"
        )
        assert {
            run_prefix + "scientific_execution/scientific_workflow_result_envelope.json",
            run_prefix + "isolated_scientific_execution_report.json",
            run_prefix + "isolated_dependency_environment_report.json",
            run_prefix + "scientific_command_dispatch_report.json",
            run_prefix + "scientific_execution_binding.json",
        }.issubset(archive_entries[-1])
        assert {
            run_prefix + f"package_records/{baseline_id}_package_input_manifest.json",
            run_prefix + f"package_records/{baseline_id}_archive_summary.json",
            run_prefix + f"package_records/{baseline_id}_archive_manifest.local.json",
        }.issubset(archive_entries[-1])
        candidate = inspect_closure_package(
            root / record.archive_path,
            spec=specs[baseline_id],
            paper_run_name="pilot_paper",
            target_fpr=0.01,
        )
        assert candidate.package_family == f"method_faithful_{baseline_id}"
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(archive_entries)
        for right in archive_entries[index + 1 :]
    )

    shallow_run_dir = (
        root
        / "outputs/external_baseline_method_faithful/pilot_paper/run_records/shallow_diffuse"
    )
    shallow_summary_path = shallow_run_dir / "shallow_diffuse_summary.json"
    failed_summary = json.loads(shallow_summary_path.read_text(encoding="utf-8"))
    failed_summary.update(
        {
            "run_decision": "fail",
            "external_baseline_method_faithful_ready": False,
            "primary_baseline_adapter_ready": False,
        }
    )
    write_json(shallow_summary_path, failed_summary)
    write_test_scientific_execution_binding(
        repository_root=root,
        artifact_dir=shallow_run_dir,
        artifact_role="external_baseline_method_faithful",
        paper_run_name="pilot_paper",
        profile_id="sd35_method_runtime_gpu",
        summary_file_name="shallow_diffuse_summary.json",
        manifest_file_name="shallow_diffuse_manifest.local.json",
        formal_execution_lock=FORMAL_EXECUTION_LOCK,
        execution_route="isolated_method_faithful_workflow",
        baseline_id="shallow_diffuse",
    )
    with pytest.raises(RuntimeError, match="不得生成正式结果包"):
        package_external_baseline_method_faithful_outputs(
            root=root,
            drive_output_dir=str(drive_dir),
            archive_name="external_baseline_method_faithful_package_shallow_diffuse_test.zip",
            baseline_id="shallow_diffuse",
        )


def test_package_rejects_transfer_member_digest_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """transfer 成员字节变化但 manifest 摘要未变化时必须闭锁."""

    _patch_two_prompt_package_run(monkeypatch)
    root = _short_package_root(tmp_path)
    paths = prepare_package_source(root, "tree_ring", monkeypatch)
    split_observations = paths["split_observations"]
    assert isinstance(split_observations, Path)
    split_observations.write_bytes(split_observations.read_bytes() + b" ")

    with pytest.raises(RuntimeError, match="字节摘要不一致"):
        package_external_baseline_method_faithful_outputs(
            root=root,
            drive_output_dir=str(tmp_path / "drive"),
            archive_name="external_baseline_method_faithful_package_tree_ring_digest.zip",
            baseline_id="tree_ring",
        )


@pytest.mark.parametrize("tamper_kind", ("unit_record", "image"))
def test_package_rejects_atomic_unit_or_image_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_kind: str,
) -> None:
    """完成单元 JSON 或其绑定图像变化时不得生成正式结果包."""

    _patch_two_prompt_package_run(monkeypatch)
    root = _short_package_root(tmp_path)
    paths = prepare_package_source(root, "tree_ring", monkeypatch)
    if tamper_kind == "unit_record":
        unit_records = paths["unit_records"]
        assert isinstance(unit_records, list)
        unit_path = unit_records[0]
        assert isinstance(unit_path, Path)
        payload = json.loads(unit_path.read_text(encoding="utf-8"))
        payload["unit_data"]["observations_without_threshold"][0]["score"] = 0.25
        write_json(unit_path, payload)
        expected_message = "自摘要不匹配"
    else:
        image_paths = paths["image_paths"]
        assert isinstance(image_paths, list)
        image_path = image_paths[0]
        assert isinstance(image_path, Path)
        image_path.write_bytes(b"tampered-image")
        expected_message = "产物摘要不匹配"

    with pytest.raises(ValueError, match=expected_message):
        package_external_baseline_method_faithful_outputs(
            root=root,
            drive_output_dir=str(root / "drive"),
            archive_name=f"external_baseline_method_faithful_package_tree_ring_{tamper_kind}.zip",
            baseline_id="tree_ring",
        )


@pytest.mark.parametrize("unexpected_kind", ("file", "directory", "symlink"))
def test_package_rejects_unexpected_run_directory_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unexpected_kind: str,
) -> None:
    """运行目录中的额外文件,目录或符号链接均不得进入归档."""

    _patch_two_prompt_package_run(monkeypatch)
    root = _short_package_root(tmp_path)
    paths = prepare_package_source(root, "tree_ring", monkeypatch)
    run_dir = paths["run_dir"]
    assert isinstance(run_dir, Path)
    if unexpected_kind == "file":
        (run_dir / "unexpected.bin").write_bytes(b"stale")
    elif unexpected_kind == "directory":
        (run_dir / "unexpected_directory").mkdir()
    else:
        link_path = run_dir / "unexpected_link.json"
        try:
            link_path.symlink_to(run_dir / "tree_ring_summary.json")
        except OSError:
            pytest.skip("当前 Windows 环境不允许创建测试符号链接")

    with pytest.raises(RuntimeError, match="精确白名单|符号链接"):
        package_external_baseline_method_faithful_outputs(
            root=root,
            drive_output_dir=str(root / "drive"),
            archive_name=(
                "external_baseline_method_faithful_package_tree_ring_"
                f"unexpected_{unexpected_kind}.zip"
            ),
            baseline_id="tree_ring",
        )


def test_notebooks_expose_three_single_baseline_entries() -> None:
    """Colab 层只保留三个 common-backbone 入口，T2SMark 使用专用正式入口。"""

    notebook_root = Path(__file__).resolve().parents[2] / "paper_workflow" / "notebooks"
    for baseline_id in METHOD_FAITHFUL_BASELINE_IDS:
        path = notebook_root / f"external_baseline_{baseline_id}_run.ipynb"
        text = path.read_text(encoding="utf-8")
        assert f'SLM_WM_PRIMARY_BASELINE_ID = \\"{baseline_id}\\"' in text
        assert "SLM_WM_PRIMARY_BASELINE_METHODS" not in text
    assert not (notebook_root / "external_baseline_t2smark_run.ipynb").exists()
    assert (notebook_root / "official_reference_t2smark_run.ipynb").is_file()
