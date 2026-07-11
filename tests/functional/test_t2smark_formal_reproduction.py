"""T2SMark SD3.5 formal 复现边界的轻量测试。"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from experiments.protocol.attacks import attack_config_digest
from external_baseline.primary.sd35_method_faithful_common import formal_image_attack_config
import paper_experiments.runners.t2smark_formal_reproduction as t2smark_runtime
from paper_experiments.runners.t2smark_formal_reproduction import (
    DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES,
    T2SMarkFormalReproductionConfig,
    build_t2smark_formal_protocol_binding,
    build_t2smark_formal_run_readiness,
    package_t2smark_formal_reproduction_outputs,
    should_run_official,
    write_t2smark_formal_protocol_binding,
)
from paper_experiments.runners.t2smark_source_runtime import (
    configured_attack_names,
    verify_exact_t2smark_protocol_worktree,
)
from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
    inspect_closure_package,
)


pytestmark = pytest.mark.quick


def _write_results(path: Path, *, sample_count: int, missing_attack_name: str = "") -> None:
    """写出只用于复用门禁的 T2SMark results.json。"""

    attack_names = configured_attack_names(DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES)
    attack_identities = {
        attack_name: {
            "attack_id": config.attack_id,
            "resource_profile": config.resource_profile,
            "attack_config_digest": attack_config_digest(config),
        }
        for attack_name in attack_names
        for config in (formal_image_attack_config(attack_name),)
    }
    payload = {
        str(index): {
            "image_only_detection": {
                "clean_score": 0.1,
                "watermarked_score": 0.9,
            },
            "formal_attacks": {
                attack_name: {
                    **attack_identities[attack_name],
                    "attack_name": attack_name,
                    "attacked_negative": {
                        **attack_identities[attack_name],
                        "detection_score": 0.1,
                    },
                    "attacked_positive": {
                        **attack_identities[attack_name],
                        "detection_score": 0.9,
                    },
                }
                for attack_name in attack_names
                if attack_name != missing_attack_name
            }
        }
        for index in range(sample_count)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _source_report() -> dict[str, object]:
    """构造只用于轻量复用测试的精确源码报告。"""

    return {
        "official_repository_commit": "1" * 40,
        "protocol_patch_sha256": "2" * 64,
        "source_worktree_exact": True,
        "source_worktree_digest": "3" * 64,
    }


def _prompt_report(prompt_count: int, digest: str = "4" * 64) -> dict[str, object]:
    """构造 canonical Prompt 绑定输入。"""

    return {
        "prompt_protocol_name": "paper_probe_paper_prompt_protocol",
        "prompt_protocol_digest": digest,
        "selected_prompt_count": prompt_count,
    }


def _paper_run(prompt_count: int) -> SimpleNamespace:
    """构造协议绑定使用的最小论文运行对象。"""

    return SimpleNamespace(
        run_name="probe_paper",
        protocol_profile="probe_paper_fixed_fpr_0_1",
        prompt_count=prompt_count,
        target_fpr=0.1,
    )


def _write_reuse_binding(
    tmp_path: Path,
    config: T2SMarkFormalReproductionConfig,
    *,
    prompt_digest: str = "4" * 64,
) -> tuple[Path, Path, Path, dict[str, object]]:
    """写出与当前 results/settings 字节绑定的正式复用记录。"""

    results_path = tmp_path / "results.json"
    settings_path = tmp_path / "settings.json"
    binding_path = tmp_path / "slm_formal_protocol_binding.json"
    _write_results(results_path, sample_count=config.prompt_limit)
    settings_path.write_text('{"settings":"formal"}\n', encoding="utf-8")
    expected_binding = build_t2smark_formal_protocol_binding(
        config,
        paper_run=_paper_run(config.prompt_limit),
        prompt_report=_prompt_report(config.prompt_limit, prompt_digest),
        source_report=_source_report(),
    )
    write_t2smark_formal_protocol_binding(
        binding_path,
        expected_binding,
        results_path=results_path,
        settings_path=settings_path,
    )
    return results_path, settings_path, binding_path, expected_binding


def test_t2smark_reuse_requires_every_formal_attack_for_every_prompt(tmp_path: Path) -> None:
    """只有每个 Prompt 都包含全部正式攻击时才允许复用 T2SMark 结果。"""

    config = T2SMarkFormalReproductionConfig(prompt_limit=2, save_clean_pair=False)
    results_path, settings_path, binding_path, expected_binding = _write_reuse_binding(tmp_path, config)

    should_run, reason = should_run_official(
        config,
        results_path,
        protocol_binding_path=binding_path,
        settings_path=settings_path,
        expected_protocol_binding=expected_binding,
    )

    assert should_run is False
    assert reason == "existing_results_found"


def test_t2smark_reuse_rejects_incomplete_formal_attack_matrix(tmp_path: Path) -> None:
    """缺少任一正式攻击时必须重新运行官方生成与检测链路。"""

    results_path = tmp_path / "results.json"
    missing_attack_name = configured_attack_names(DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES)[-1]
    config = T2SMarkFormalReproductionConfig(prompt_limit=2, save_clean_pair=False)
    _write_results(results_path, sample_count=2, missing_attack_name=missing_attack_name)
    settings_path = tmp_path / "settings.json"
    binding_path = tmp_path / "slm_formal_protocol_binding.json"
    settings_path.write_text('{"settings":"formal"}\n', encoding="utf-8")
    expected_binding = build_t2smark_formal_protocol_binding(
        config,
        paper_run=_paper_run(2),
        prompt_report=_prompt_report(2),
        source_report=_source_report(),
    )
    write_t2smark_formal_protocol_binding(
        binding_path,
        expected_binding,
        results_path=results_path,
        settings_path=settings_path,
    )

    should_run, reason = should_run_official(
        config,
        results_path,
        protocol_binding_path=binding_path,
        settings_path=settings_path,
        expected_protocol_binding=expected_binding,
    )

    assert should_run is True
    assert reason == "existing_results_formal_attack_count_mismatch"


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    (
        ("prompt_digest", "9" * 64),
        ("seed", 20260711),
        ("num_inference_steps", 21),
        ("num_inversion_steps", 21),
        ("guidance_scale", 5.0),
    ),
)
def test_t2smark_reuse_rejects_protocol_binding_mismatch(
    tmp_path: Path,
    changed_field: str,
    changed_value: object,
) -> None:
    """Prompt 或公平预算发生变化时不得复用旧官方结果。"""

    config = T2SMarkFormalReproductionConfig(prompt_limit=2, save_clean_pair=False)
    results_path, settings_path, binding_path, _ = _write_reuse_binding(tmp_path, config)
    changed_config = config
    prompt_digest = "4" * 64
    if changed_field == "prompt_digest":
        prompt_digest = str(changed_value)
    else:
        changed_payload = asdict(config)
        changed_payload[changed_field] = changed_value
        changed_config = T2SMarkFormalReproductionConfig(**changed_payload)
    expected_binding = build_t2smark_formal_protocol_binding(
        changed_config,
        paper_run=_paper_run(2),
        prompt_report=_prompt_report(2, prompt_digest),
        source_report=_source_report(),
    )

    should_run, reason = should_run_official(
        changed_config,
        results_path,
        protocol_binding_path=binding_path,
        settings_path=settings_path,
        expected_protocol_binding=expected_binding,
    )

    assert should_run is True
    assert reason == "existing_results_protocol_binding_mismatch"


def test_t2smark_reuse_rejects_source_revision_or_patch_mismatch(tmp_path: Path) -> None:
    """源码 revision 或固定补丁摘要变化时必须重新生成。"""

    config = T2SMarkFormalReproductionConfig(prompt_limit=2, save_clean_pair=False)
    results_path, settings_path, binding_path, _ = _write_reuse_binding(tmp_path, config)
    changed_source = _source_report()
    changed_source["protocol_patch_sha256"] = "8" * 64
    expected_binding = build_t2smark_formal_protocol_binding(
        config,
        paper_run=_paper_run(2),
        prompt_report=_prompt_report(2),
        source_report=changed_source,
    )

    should_run, reason = should_run_official(
        config,
        results_path,
        protocol_binding_path=binding_path,
        settings_path=settings_path,
        expected_protocol_binding=expected_binding,
    )

    assert should_run is True
    assert reason == "existing_results_protocol_binding_mismatch"


def test_t2smark_run_readiness_requires_formal_import_validation() -> None:
    """其他运行条件全部通过时, formal import 失败仍必须使运行失败。"""

    assert build_t2smark_formal_run_readiness(
        official_ready=True,
        adapter_ready=True,
        prompt_ready=True,
        pair_quality_ready=True,
        formal_attack_ready=True,
        formal_import_validation_ready=False,
    ) is False
    assert build_t2smark_formal_run_readiness(
        official_ready=True,
        adapter_ready=True,
        prompt_ready=True,
        pair_quality_ready=True,
        formal_attack_ready=True,
        formal_import_validation_ready=True,
    ) is True


def test_t2smark_source_worktree_must_equal_fixed_patch(tmp_path: Path) -> None:
    """固定补丁之外的 tracked 或 untracked 改动必须被源码门禁拒绝。"""

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    subprocess.run(["git", "init"], cwd=source_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source_dir, check=True)
    (source_dir / "option.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source_dir / "run_sd35.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "option.py", "run_sd35.py"], cwd=source_dir, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=source_dir, check=True, capture_output=True)
    (source_dir / "option.py").write_text("VALUE = 2\n", encoding="utf-8")
    (source_dir / "run_sd35.py").write_text("def run():\n    return 2\n", encoding="utf-8")
    patch_path = tmp_path / "formal_protocol.diff"
    diff_result = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", "option.py", "run_sd35.py"],
        cwd=source_dir,
        check=True,
        capture_output=True,
    )
    patch_path.write_bytes(diff_result.stdout)

    report = verify_exact_t2smark_protocol_worktree(source_dir, patch_path)
    assert report["source_worktree_exact"] is True
    assert len(str(report["source_worktree_digest"])) == 64

    extra_path = source_dir / "extra.txt"
    extra_path.write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="未跟踪文件"):
        verify_exact_t2smark_protocol_worktree(source_dir, patch_path)
    extra_path.unlink()
    (source_dir / "run_sd35.py").write_text("def run():\n    return 3\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="不等于固定 revision"):
        verify_exact_t2smark_protocol_worktree(source_dir, patch_path)


def _write_package_fixture(
    root_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    """写出一个 Prompt 的完整 T2SMark 精确白名单打包 fixture。"""

    code_version = "b370425"
    monkeypatch.setattr(t2smark_runtime, "resolve_code_version", lambda _root: code_version)
    config = T2SMarkFormalReproductionConfig(
        output_dir="outputs/t2smark_formal_reproduction",
        drive_output_dir=str(root_path / "drive"),
        prompt_set="probe_paper",
        prompt_file="configs/paper_main_probe_paper_prompts.txt",
        t2smark_run_name="t",
        prompt_limit=1,
        minimum_prompt_protocol_count=1,
        target_fpr=0.1,
    )
    paper_run = _paper_run(1)
    monkeypatch.setattr(
        t2smark_runtime,
        "validate_t2smark_formal_protocol_config",
        lambda _config, *, root_path: paper_run,
    )
    monkeypatch.setattr(t2smark_runtime, "build_paper_run_config", lambda _root: paper_run)
    paths = t2smark_runtime.output_paths(root_path, config)
    attack_names = configured_attack_names(config.formal_attack_families)
    source_report = _source_report()
    prompt_report = _prompt_report(1)

    for path in (
        paths["official_settings"],
        paths["prompt_dataset"],
        paths["prompt_plan"],
        paths["image_pairs"],
        paths["pair_quality_summary"],
        paths["adapter_observations"],
        paths["adapter_manifest"],
        paths["adapter_artifact_manifest"],
        paths["validation_report"],
        paths["environment_report"],
        paths["source_prepare_result"],
        paths["official_command_result"],
        paths["adapter_command_result"],
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    watermarked_path = paths["official_images"] / "00000.png"
    clean_path = paths["official_run_dir"] / "quality_pairs" / "clean" / "00000.png"
    watermarked_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    watermarked_path.write_bytes(b"watermarked")
    clean_path.write_bytes(b"clean")
    formal_attacks: dict[str, object] = {}
    for attack_name in attack_names:
        attack_config = formal_image_attack_config(attack_name)
        attack_identity = {
            "attack_id": attack_config.attack_id,
            "resource_profile": attack_config.resource_profile,
            "attack_config_digest": attack_config_digest(attack_config),
        }
        role_rows: dict[str, object] = {}
        for sample_role in ("attacked_negative", "attacked_positive"):
            attack_path = paths["official_run_dir"] / "formal_attacks" / (
                f"00000_{attack_name}_{sample_role}.png"
            )
            attack_path.parent.mkdir(parents=True, exist_ok=True)
            attack_path.write_bytes(f"{attack_name}:{sample_role}".encode("utf-8"))
            role_rows[sample_role] = {
                **attack_identity,
                "detection_score": 0.1,
            }
        formal_attacks[attack_name] = {
            **attack_identity,
            "attack_name": attack_name,
            **role_rows,
        }
    official_results = {
        "0": {
            "image_only_detection": {"clean_score": 0.1, "watermarked_score": 0.9},
            "formal_attacks": formal_attacks,
            "pair_quality": {
                "pair_quality_protocol": "strict_clean_watermarked_pair",
                "clean_image_path": str(clean_path),
                "clean_image_digest": "clean",
                "watermarked_image_path": str(watermarked_path),
                "watermarked_image_digest": "watermarked",
            },
        }
    }
    t2smark_runtime.write_json(paths["official_results"], official_results)
    t2smark_runtime.write_json(paths["official_settings"], {"settings": "formal"})
    expected_binding = build_t2smark_formal_protocol_binding(
        config,
        paper_run=paper_run,
        prompt_report=prompt_report,
        source_report=source_report,
    )
    persisted_binding = write_t2smark_formal_protocol_binding(
        paths["official_protocol_binding"],
        expected_binding,
        results_path=paths["official_results"],
        settings_path=paths["official_settings"],
    )
    t2smark_runtime.write_json(paths["prompt_dataset"], {"annotations": [{}]})
    t2smark_runtime.write_json(paths["prompt_plan"], [{"prompt_id": "prompt_000000"}])
    t2smark_runtime.write_json(paths["image_pairs"], [{"image_id": "image_000000"}])
    t2smark_runtime.write_json(
        paths["pair_quality_summary"],
        {"strict_pair_quality_ready": True, "measured_strict_pair_quality_count": 1},
    )
    paths["pair_quality_metrics"].write_text("image_id,ssim\nimage_000000,1.0\n", encoding="utf-8")
    t2smark_runtime.write_json(
        paths["adapter_observations"],
        [{"event_id": f"event_{index}"} for index in range(2 + 2 * len(attack_names))],
    )
    for path in (
        paths["adapter_manifest"],
        paths["adapter_artifact_manifest"],
        paths["environment_report"],
        paths["source_prepare_result"],
        paths["official_command_result"],
        paths["adapter_command_result"],
    ):
        t2smark_runtime.write_json(path, {})
    candidate_count = len(attack_names)
    paths["candidate_records"].write_text(
        "".join(
            json.dumps({"attack_name": name, "baseline_id": "t2smark"}) + "\n"
            for name in attack_names
        ),
        encoding="utf-8",
    )
    validation = {
        "formal_import_validation_ready": True,
        "accepted_formal_import_count": candidate_count,
        "rejected_formal_import_count": 0,
        "formal_import_issue_count": 0,
    }
    t2smark_runtime.write_json(paths["validation_report"], validation)
    summary = {
        "generated_at": "2026-07-11T00:00:00+00:00",
        "baseline_id": "t2smark",
        "paper_claim_scale": "probe_paper",
        "target_fpr": 0.1,
        "run_decision": "pass",
        "t2smark_formal_reproduction_ready": True,
        "t2smark_formal_attack_ready": True,
        "t2smark_strict_pair_quality_ready": True,
        "formal_import_validation_ready": True,
        "selected_prompt_count": 1,
        "formal_import_candidate_record_count": candidate_count,
        "official_protocol_binding_digest": expected_binding["protocol_binding_digest"],
        "metadata": {
            "prompt_report": prompt_report,
            "official_report": {
                "source_report": source_report,
                "official_protocol_binding": persisted_binding,
            },
        },
    }
    t2smark_runtime.write_json(paths["summary"], summary)
    run_manifest = {
        "code_version": code_version,
        "config": asdict(config),
        "metadata": {"run_decision": "pass", "formal_import_validation_ready": True},
    }
    t2smark_runtime.write_json(paths["manifest"], run_manifest)
    return paths["output_dir"], root_path / "drive"


def test_t2smark_package_requires_pass_validation_and_exact_whitelist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pass 结果可打包, 但任何旧文件或 formal validation 失败都必须阻断。"""

    output_dir, drive_dir = _write_package_fixture(tmp_path, monkeypatch)
    record = package_t2smark_formal_reproduction_outputs(
        root=tmp_path,
        output_dir="outputs/t2smark_formal_reproduction",
        drive_output_dir=str(drive_dir),
        archive_name="external_baseline_official_reference_package_t2smark_test.zip",
    )
    assert (tmp_path / record.archive_path).is_file()
    spec = next(
        item
        for item in CLOSURE_PACKAGE_FAMILY_SPECS
        if item.package_family == "official_reference_t2smark"
    )
    candidate = inspect_closure_package(
        tmp_path / record.archive_path,
        spec=spec,
        paper_run_name="probe_paper",
        target_fpr=0.1,
    )
    assert candidate.package_family == "official_reference_t2smark"
    old_archive_path = output_dir / "external_baseline_official_reference_package_t2smark_old.zip"
    old_archive_path.write_bytes(b"old archive is not a package member")
    package_t2smark_formal_reproduction_outputs(
        root=tmp_path,
        output_dir="outputs/t2smark_formal_reproduction",
        drive_output_dir=str(drive_dir),
        archive_name="external_baseline_official_reference_package_t2smark_test.zip",
    )
    stale_path = output_dir / "stale_previous_run.json"
    stale_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="旧运行或非白名单"):
        package_t2smark_formal_reproduction_outputs(
            root=tmp_path,
            output_dir="outputs/t2smark_formal_reproduction",
            drive_output_dir=str(drive_dir),
            archive_name="external_baseline_official_reference_package_t2smark_test.zip",
        )
    stale_path.unlink()
    validation_path = output_dir / "t2smark_formal_import_validation_report.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["formal_import_validation_ready"] = False
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    with pytest.raises(RuntimeError, match="formal import validation"):
        package_t2smark_formal_reproduction_outputs(
            root=tmp_path,
            output_dir="outputs/t2smark_formal_reproduction",
            drive_output_dir=str(drive_dir),
            archive_name="external_baseline_official_reference_package_t2smark_test.zip",
        )
