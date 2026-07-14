"""CPU 论文结果闭合编排的轻量功能测试."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from main.core.digest import build_stable_digest
from scripts import paper_result_closure as closure
from scripts import run_gpu_server_result_closure as server_closure


PAPER_RUN_NAME = "probe_paper"
TARGET_FPR = 0.1
REPOSITORY_COMMIT = "a" * 40


def _sha256(path: Path) -> str:
    """计算测试文件摘要."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate_source(tmp_path: Path) -> SimpleNamespace:
    """构造只用于编排单元测试的已验证来源对象."""

    package_path = tmp_path / "packages" / "aggregate.zip"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    package_path.write_bytes(b"validated-aggregate-bytes")
    return SimpleNamespace(
        package_path=package_path.resolve(),
        package_sha256=_sha256(package_path),
        randomization_aggregate_digest="b" * 64,
        common_code_version=REPOSITORY_COMMIT,
        payload={
            "paper_run_name": PAPER_RUN_NAME,
            "target_fpr": TARGET_FPR,
        },
    )


def _component_summary(
    artifact_id: str,
    *,
    detection_ready: bool,
    paired_ready: bool,
    ablation_ready: bool,
) -> dict[str, object]:
    """构造与四个正式 Writer 判定语义一致的测试摘要."""

    if artifact_id == "randomization_detection_statistics_manifest":
        payload: dict[str, object] = {
            "paper_claim_scale": PAPER_RUN_NAME,
            "target_fpr": TARGET_FPR,
            "randomization_detection_statistics_ready": True,
            "all_methods_clean_fixed_fpr_ready": True,
            "main_method_clean_fixed_fpr_ready": True,
            "main_method_wrong_key_fixed_fpr_ready": True,
            "all_per_attack_fixed_fpr_ready": detection_ready,
            "all_test_negative_populations_fixed_fpr_ready": detection_ready,
            "universal_per_attack_superiority_claim_ready": False,
            "supports_paper_claim": False,
        }
        payload["randomization_detection_statistics_summary_digest"] = (
            build_stable_digest(payload)
        )
        return payload
    if artifact_id == "randomization_paired_superiority_manifest":
        payload = {
            "paper_claim_scale": PAPER_RUN_NAME,
            "target_fpr": TARGET_FPR,
            "randomization_paired_statistics_ready": True,
            "quality_matched_statistics_ready": True,
            "overall_paired_superiority_ready": paired_ready,
            "overall_quality_matched_superiority_ready": paired_ready,
            "conclusion_decision": (
                "supported" if paired_ready else "measured_not_supported"
            ),
            "supports_paper_claim": paired_ready,
        }
        payload["randomization_paired_superiority_summary_digest"] = (
            build_stable_digest(payload)
        )
        return payload
    if artifact_id == "randomization_dataset_quality_manifest":
        payload = {
            "paper_claim_scale": PAPER_RUN_NAME,
            "target_fpr": TARGET_FPR,
            "randomization_dataset_quality_statistics_ready": True,
            "quality_metric_status": "measured",
            "conclusion_decision": "measured_evidence_component",
            "supports_paper_claim": False,
        }
        payload["randomization_dataset_quality_summary_digest"] = (
            build_stable_digest(payload)
        )
        return payload
    if artifact_id == "randomization_ablation_necessity_manifest":
        return {
            "randomization_aggregate_statistics_ready": True,
            "ablation_necessity_statistics_ready": True,
            "all_mechanism_necessity_components_supported": ablation_ready,
            "necessity_component_decision": (
                "measured_supported"
                if ablation_ready
                else "measured_not_supported"
            ),
            "supports_paper_claim": ablation_ready,
        }
    raise AssertionError(f"未知测试统计组件: {artifact_id}")


def _write_rebuilt_artifacts(
    tmp_path: Path,
    source: SimpleNamespace,
    *,
    detection_ready: bool = True,
    paired_ready: bool = True,
    ablation_ready: bool = True,
) -> None:
    """写出四个 Writer 应发布的最小受治理文件集合."""

    for spec in closure._CLOSURE_ARTIFACT_SPECS:
        paths = closure._expected_artifact_paths(
            tmp_path,
            spec,
            PAPER_RUN_NAME,
        )
        for index, path in enumerate(paths[:-1]):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{spec.artifact_id}:{index}\n", encoding="utf-8")
        summary = _component_summary(
            spec.artifact_id,
            detection_ready=detection_ready,
            paired_ready=paired_ready,
            ablation_ready=ablation_ready,
        )
        summary_path = paths[-1].parent / spec.summary_file_name
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        output_sha256 = {
            path.relative_to(tmp_path).as_posix(): _sha256(path)
            for path in paths[:-1]
        }
        config = {
            "paper_run_name": PAPER_RUN_NAME,
            "target_fpr": TARGET_FPR,
            "randomization_aggregate_package_sha256": source.package_sha256,
            "randomization_aggregate_digest": (
                source.randomization_aggregate_digest
            ),
            "common_code_version": source.common_code_version,
        }
        metadata: dict[str, object] = {
            "output_sha256": output_sha256,
            spec.ready_field: True,
            "supports_paper_claim": summary["supports_paper_claim"],
        }
        if spec.artifact_id == "randomization_detection_statistics_manifest":
            digest_field = "randomization_detection_statistics_summary_digest"
            config[digest_field] = summary[digest_field]
            for field_name in (
                "all_test_negative_populations_fixed_fpr_ready",
                "universal_per_attack_superiority_claim_ready",
            ):
                config[field_name] = summary[field_name]
            for field_name in (
                "main_method_clean_fixed_fpr_ready",
                "main_method_wrong_key_fixed_fpr_ready",
                "all_per_attack_fixed_fpr_ready",
                "all_test_negative_populations_fixed_fpr_ready",
                "universal_per_attack_superiority_claim_ready",
            ):
                metadata[field_name] = summary[field_name]
        elif spec.artifact_id == "randomization_paired_superiority_manifest":
            digest_field = "randomization_paired_superiority_summary_digest"
            config[digest_field] = summary[digest_field]
            metadata["conclusion_decision"] = summary["conclusion_decision"]
        elif spec.artifact_id == "randomization_dataset_quality_manifest":
            digest_field = "randomization_dataset_quality_summary_digest"
            config[digest_field] = summary[digest_field]
            metadata["conclusion_decision"] = summary["conclusion_decision"]
        else:
            config["necessity_summary_digest"] = build_stable_digest(summary)
            metadata["necessity_component_decision"] = summary[
                "necessity_component_decision"
            ]
        manifest = {
            "artifact_id": spec.artifact_id,
            "artifact_type": "local_manifest",
            "input_paths": [source.package_path.as_posix()],
            "output_paths": [
                path.relative_to(tmp_path).as_posix() for path in paths
            ],
            "config": config,
            "config_digest": build_stable_digest(config),
            "code_version": source.common_code_version,
            "rebuild_command": "python -m governed_writer",
            "metadata": metadata,
        }
        paths[-1].write_text(
            json.dumps(manifest, ensure_ascii=False),
            encoding="utf-8",
        )


@pytest.mark.quick
def test_closure_command_plan_uses_one_validated_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """四个统计 Writer 必须消费同一个已验证聚合 ZIP."""

    source = _aggregate_source(tmp_path)
    monkeypatch.setattr(
        closure,
        "validate_randomization_aggregate_provenance",
        lambda *_args, **_kwargs: source,
    )

    commands = closure.build_paper_result_closure_commands(
        randomization_aggregate_package_path=source.package_path,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        root=tmp_path,
    )

    assert len(commands) == closure.PAPER_RESULT_CLOSURE_COMMAND_COUNT == 4
    assert all(
        command[command.index("--aggregate-package-path") + 1]
        == str(source.package_path)
        for command in commands
    )


@pytest.mark.quick
def test_cleanup_removes_only_current_derived_results(tmp_path: Path) -> None:
    """清理只删除当前运行的派生统计, 不删除聚合来源和其他层级."""

    current = (
        tmp_path
        / "outputs"
        / "randomization_detection_statistics"
        / PAPER_RUN_NAME
    )
    other_run = (
        tmp_path
        / "outputs"
        / "randomization_detection_statistics"
        / "pilot_paper"
    )
    for directory in (current, other_run):
        directory.mkdir(parents=True)
        (directory / "sentinel.json").write_text("{}", encoding="utf-8")
    source = _aggregate_source(tmp_path)

    removed = closure.clean_paper_result_closure_outputs(
        root=tmp_path,
        paper_run_name=PAPER_RUN_NAME,
        selected_package_paths=(source.package_path,),
    )

    assert not current.exists()
    assert other_run.is_dir()
    assert source.package_path.is_file()
    assert (
        f"outputs/randomization_detection_statistics/{PAPER_RUN_NAME}" in removed
    )


@pytest.mark.quick
def test_cleanup_rejects_locked_package_inside_managed_directory(
    tmp_path: Path,
) -> None:
    """锁定聚合包位于待清理目录时必须拒绝删除."""

    managed_dir = (
        tmp_path
        / "outputs"
        / "randomization_detection_statistics"
        / PAPER_RUN_NAME
    )
    managed_dir.mkdir(parents=True)
    package_path = managed_dir / "aggregate.zip"
    package_path.write_bytes(b"zip")

    with pytest.raises(ValueError, match="锁定输入包位于受管清理目录内"):
        closure.clean_paper_result_closure_outputs(
            root=tmp_path,
            paper_run_name=PAPER_RUN_NAME,
            selected_package_paths=(package_path,),
        )

    assert package_path.is_file()


@pytest.mark.quick
def test_run_rebuilds_gates_and_archives_governed_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整入口必须实际经过统计命令、门禁与自包含归档."""

    source = _aggregate_source(tmp_path)
    _write_rebuilt_artifacts(tmp_path, source)
    executed: list[list[str]] = []
    monkeypatch.setattr(
        closure,
        "_select_randomization_aggregate",
        lambda *_args, **_kwargs: source,
    )
    monkeypatch.setattr(
        closure,
        "clean_paper_result_closure_outputs",
        lambda **_kwargs: (),
    )
    monkeypatch.setattr(
        closure,
        "resolve_code_version",
        lambda _root: REPOSITORY_COMMIT,
    )
    monkeypatch.setattr(
        closure.subprocess,
        "run",
        lambda command, **_kwargs: executed.append(command),
    )

    result = closure.run_paper_result_closure_commands(
        package_search_root=source.package_path,
        complete_drive_output_dir=(
            tmp_path / "outputs" / "complete_result_packages"
        ),
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        root=tmp_path,
        expected_repository_commit=REPOSITORY_COMMIT,
    )

    archive_path = Path(result["archive_path"])
    assert len(executed) == 4
    assert result["paper_result_evidence_ready"] is True
    assert result["conclusion_decision"] == "supported"
    assert result["supports_paper_claim"] is True
    assert archive_path.is_file()
    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        archive_manifest = json.loads(
            archive.read(
                "paper_complete_result_archive_manifest.json"
            ).decode("utf-8")
        )
    assert "inputs/randomization_aggregate_provenance.zip" in names
    assert "paper_complete_result_archive_manifest.json" in names
    assert archive_manifest["paper_result_evidence_ready"] is True
    assert archive_manifest["supports_paper_claim"] is True
    assert (
        f"outputs/paper_result_closure/{PAPER_RUN_NAME}/"
        "paper_result_closure_gate_report.json"
    ) in names


@pytest.mark.quick
@pytest.mark.parametrize(
    ("failed_gate", "expected_role"),
    (
        ("detection", "fixed_fpr_negative_population_gate"),
        ("paired", "paired_quality_matched_superiority_gate"),
        ("ablation", "mechanism_necessity_gate"),
    ),
)
def test_gate_separates_reconstructable_evidence_from_unsupported_claim(
    tmp_path: Path,
    failed_gate: str,
    expected_role: str,
) -> None:
    """任一中心门禁失败时仍保留可重建证据, 但不得支持结论."""

    source = _aggregate_source(tmp_path)
    _write_rebuilt_artifacts(
        tmp_path,
        source,
        detection_ready=failed_gate != "detection",
        paired_ready=failed_gate != "paired",
        ablation_ready=failed_gate != "ablation",
    )

    _report_path, _manifest_path, report = closure._write_closure_gate(
        source,
        root_path=tmp_path,
    )

    assert report["paper_result_evidence_ready"] is True
    assert report["supports_paper_claim"] is False
    assert report["conclusion_decision"] == "measured_not_supported"
    assert report["unsupported_claim_gate_roles"] == [expected_role]


@pytest.mark.quick
def test_dataset_quality_measurement_does_not_vote_against_central_claim(
    tmp_path: Path,
) -> None:
    """FID/KID 的非结论字段只表示测量角色, 不得否决中心结论."""

    source = _aggregate_source(tmp_path)
    _write_rebuilt_artifacts(tmp_path, source)

    _report_path, _manifest_path, report = closure._write_closure_gate(
        source,
        root_path=tmp_path,
    )

    quality_record = next(
        record
        for record in report["artifact_records"]
        if record["component_role"] == "dataset_quality_measurement"
    )
    assert quality_record["supports_paper_claim"] is False
    assert quality_record["contributes_to_central_claim_gate"] is False
    assert report["supports_paper_claim"] is True


@pytest.mark.quick
def test_gate_rejects_tampered_rebuilt_statistic(tmp_path: Path) -> None:
    """Writer 输出在门禁前发生字节漂移时不得进入归档."""

    source = _aggregate_source(tmp_path)
    _write_rebuilt_artifacts(tmp_path, source)
    tampered = (
        tmp_path
        / "outputs"
        / "randomization_detection_statistics"
        / PAPER_RUN_NAME
        / "method_detection_operating_points.csv"
    )
    tampered.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="文件摘要不匹配"):
        closure._write_closure_gate(source, root_path=tmp_path)


@pytest.mark.quick
def test_server_dry_run_validates_commit_and_returns_real_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """服务器预检必须读取聚合来源并给出可执行统计计划."""

    source = _aggregate_source(tmp_path)
    commands = [["python", "-m", f"writer_{index}"] for index in range(4)]
    monkeypatch.setattr(
        server_closure,
        "validate_randomization_aggregate_provenance",
        lambda *_args, **_kwargs: source,
    )
    monkeypatch.setattr(
        server_closure,
        "build_paper_result_closure_commands",
        lambda **_kwargs: commands,
    )
    monkeypatch.setattr(
        server_closure,
        "resolve_code_version",
        lambda _root: REPOSITORY_COMMIT,
    )

    result = server_closure.execute_server_result_closure(
        root=tmp_path,
        paper_run_name=PAPER_RUN_NAME,
        randomization_aggregate_package_path=source.package_path,
        complete_output_dir=tmp_path / "outputs" / "complete",
        repository_commit=REPOSITORY_COMMIT,
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["statistics_commands"] == commands
    assert result["paper_result_evidence_ready"] is False
