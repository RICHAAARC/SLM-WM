"""验证单 baseline method-faithful 正式运行与 transfer 边界。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image

from external_baseline.primary.sd35_method_faithful_common import (
    apply_formal_image_attack,
    canonical_attack_family,
    canonical_attack_name,
    supported_formal_image_attack_names,
)
from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
    inspect_closure_package,
)
from paper_experiments.runners.external_baseline_method_faithful import (
    DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES,
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


PACKAGE_TEST_CODE_VERSION = "b370425"


pytestmark = pytest.mark.quick


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
) -> dict[str, object]:
    """构造带真实阈值判定关系的最小 observation。"""

    return {
        "event_id": event_id,
        "baseline_id": baseline_id,
        "prompt_id": "prompt_000001",
        "prompt_text": "a ceramic fox",
        "split": "calibration",
        "sample_role": sample_role,
        "attack_family": attack_family,
        "attack_name": attack_name,
        "score": score,
        "threshold": threshold,
        "threshold_source": "calibration_clean_negative_conformal",
        "detection_decision": score >= threshold,
    }


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
            "generation_protocol": {
                "num_inference_steps": config.num_inference_steps,
                "guidance_scale": config.guidance_scale,
            },
        },
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


def test_transfer_manifest_binds_actual_observations_and_threshold(tmp_path: Path) -> None:
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
    assert manifest["detection_protocol"]["input_access_mode"] == "image_only"
    assert paths["split_observations"].is_file()
    assert paths["transfer_manifest"].is_file()


def test_transfer_manifest_rejects_declared_observation_count_mismatch(tmp_path: Path) -> None:
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


def test_run_directory_preparation_removes_only_selected_baseline_state(tmp_path: Path) -> None:
    """重跑必须清除当前 baseline 遗留状态，同时保留其他 baseline 的 transfer 文件。"""

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
    for field_name in ("split_observations", "split_command_results", "transfer_manifest"):
        write_json(paths[field_name], {"baseline_id": "tree_ring", "stale": True})
    other_transfer = paths["split_observation_dir"] / "gaussian_shading_baseline_observations.json"
    write_json(other_transfer, [{"baseline_id": "gaussian_shading"}])

    prepare_single_baseline_run_directory(paths)

    assert paths["run_dir"].is_dir()
    assert not stale_run_file.exists()
    assert all(
        not paths[field_name].exists()
        for field_name in ("split_observations", "split_command_results", "transfer_manifest")
    )
    assert other_transfer.is_file()


def prepare_package_source(root: Path, baseline_id: str, *, run_decision: str = "pass") -> None:
    """写出白名单打包所需的单 baseline 最小产物。"""

    output_root = root / "outputs" / "external_baseline_method_faithful" / "pilot_paper"
    run_dir = output_root / "run_records" / baseline_id
    split_dir = output_root / "split_observations"
    write_json(
        run_dir / f"{baseline_id}_summary.json",
        {
            "run_decision": run_decision,
            "external_baseline_method_faithful_ready": run_decision == "pass",
            "primary_baseline_adapter_ready": run_decision == "pass",
            "primary_baseline_id": baseline_id,
            "paper_run_name": "pilot_paper",
            "target_fpr": 0.01,
        },
    )
    code_version = PACKAGE_TEST_CODE_VERSION
    write_json(
        run_dir / f"{baseline_id}_manifest.local.json",
        {
            "code_version": code_version,
            "config": {
                "prompt_set": "pilot_paper",
                "target_fpr": 0.01,
                "primary_baseline_id": baseline_id,
            },
        },
    )
    write_json(split_dir / f"{baseline_id}_baseline_observations.json", [{"baseline_id": baseline_id}])
    write_json(split_dir / f"{baseline_id}_baseline_command_results.json", [{"baseline_id": baseline_id}])
    write_json(
        split_dir / f"{baseline_id}_baseline_transfer_manifest.json",
        {
            "baseline_id": baseline_id,
            "paper_run_name": "pilot_paper",
            "target_fpr": 0.01,
            "code_version": code_version,
            "transfer_ready": True,
        },
    )


def test_packages_are_baseline_isolated_and_failure_is_not_packaged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不同 baseline 包不得包含同名业务产物，失败运行不得打包。"""

    monkeypatch.setattr(
        "paper_experiments.runners.external_baseline_method_faithful.resolve_code_version",
        lambda _root: PACKAGE_TEST_CODE_VERSION,
    )
    drive_dir = tmp_path / "drive"
    archive_entries: list[set[str]] = []
    specs = {
        spec.baseline_id: spec
        for spec in CLOSURE_PACKAGE_FAMILY_SPECS
        if spec.package_family.startswith("method_faithful_")
    }
    for baseline_id in METHOD_FAITHFUL_BASELINE_IDS:
        prepare_package_source(tmp_path, baseline_id)
        archive_name = f"external_baseline_method_faithful_package_{baseline_id}_test.zip"
        record = package_external_baseline_method_faithful_outputs(
            root=tmp_path,
            drive_output_dir=str(drive_dir),
            archive_name=archive_name,
            baseline_id=baseline_id,
        )
        with ZipFile(tmp_path / record.archive_path) as archive:
            archive_entries.append(set(archive.namelist()))
        candidate = inspect_closure_package(
            tmp_path / record.archive_path,
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

    prepare_package_source(tmp_path, "shallow_diffuse", run_decision="fail")
    with pytest.raises(RuntimeError, match="不得生成正式结果包"):
        package_external_baseline_method_faithful_outputs(
            root=tmp_path,
            drive_output_dir=str(drive_dir),
            archive_name="external_baseline_method_faithful_package_shallow_diffuse_test.zip",
            baseline_id="shallow_diffuse",
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
