"""CPU 论文结果闭合编排的轻量功能测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
)
from paper_experiments.runners import paper_result_closure as closure
from scripts import run_gpu_server_result_closure as server_closure


PAPER_RUN_NAME = "probe_paper"
TARGET_FPR = 0.1


def build_package_records(package_root: Path) -> list[dict[str, object]]:
    """构造按10个受治理 family 排列的绝对包记录。"""

    package_root.mkdir(parents=True, exist_ok=True)
    return [
        {
            "package_family": specification.package_family,
            "package_path": (package_root / f"{specification.package_family}.zip").resolve().as_posix(),
            "package_sha256": "a" * 64,
            "paper_run_name": PAPER_RUN_NAME,
            "target_fpr": TARGET_FPR,
            "code_version": "abc1234",
            "generated_at": "2026-07-11T00:00:00+00:00",
        }
        for specification in CLOSURE_PACKAGE_FAMILY_SPECS
    ]


def argument_value(command: list[str], argument_name: str) -> str:
    """读取单值命令参数。"""

    return command[command.index(argument_name) + 1]


@pytest.mark.quick
def test_closure_command_plan_is_run_scoped_and_binds_exact_packages(tmp_path: Path) -> None:
    """闭合 DAG 应显式绑定10个包及当前 run 的全部输入输出。"""

    records = build_package_records(tmp_path / "packages")
    commands = closure.build_paper_result_closure_commands(
        closure_input_packages=records,
        complete_drive_output_dir=tmp_path / "complete",
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        archive_name="probe_paper_complete_result_package_exact.zip",
        root=tmp_path,
    )

    assert len(commands) == closure.PAPER_RESULT_CLOSURE_COMMAND_COUNT
    assert [Path(command[1]).name for command in commands] == [
        "write_pilot_paper_result_records.py",
        "write_attack_matrix_outputs.py",
        "write_fixed_fpr_threshold_audit_outputs.py",
        "write_primary_baseline_method_faithful_adapter_protocol.py",
        "write_primary_baseline_result_candidates.py",
        "write_primary_baseline_formal_import_protocol.py",
        "write_primary_baseline_evidence_outputs.py",
        "write_external_baseline_comparison_outputs.py",
        "write_pilot_paper_result_records.py",
        "write_pilot_paper_fixed_fpr_common_protocol_outputs.py",
        "write_pilot_paper_result_analysis_outputs.py",
        "write_paper_artifact_evidence_audit_outputs.py",
        "write_submission_readiness_outputs.py",
        "write_evidence_closure_entry_review_outputs.py",
        "write_result_closure_gate_outputs.py",
        "write_pilot_paper_complete_result_package.py",
    ]
    assert all("--package-search-root" not in command for command in commands)
    assert commands[0].count("--package-path") == len(CLOSURE_PACKAGE_FAMILY_SPECS)
    assert commands[-1].count("--package-path") == len(CLOSURE_PACKAGE_FAMILY_SPECS)
    assert "--materialize-only" in commands[0]
    assert "--skip-package-materialization" in commands[-1]
    assert "--require-pass" in commands[2]
    assert "--require-pass" in commands[-2]
    assert argument_value(commands[1], "--output-dir").endswith(
        f"attack_matrix/{PAPER_RUN_NAME}"
    )
    assert argument_value(commands[8], "--baseline-records-path").endswith(
        f"external_baseline_comparison/{PAPER_RUN_NAME}/baseline_result_records.jsonl"
    )
    assert argument_value(commands[-1], "--output-dir").endswith(
        f"pilot_paper_complete_result_package/{PAPER_RUN_NAME}"
    )
    assert argument_value(commands[-2], "--dataset-quality-feature-records-path").endswith(
        f"dataset_level_quality/{PAPER_RUN_NAME}/dataset_quality_formal_feature_records.jsonl"
    )
    assert argument_value(commands[-2], "--dataset-quality-feature-report-path").endswith(
        f"dataset_level_quality/{PAPER_RUN_NAME}/dataset_quality_formal_feature_import_report.json"
    )
    assert argument_value(commands[-2], "--dataset-quality-metrics-path").endswith(
        f"dataset_level_quality/{PAPER_RUN_NAME}/dataset_quality_metrics.csv"
    )
    assert "--t2smark-formal-package-path" not in commands[4]
    assert argument_value(commands[4], "--t2smark-candidate-records-path").endswith(
        f"t2smark_formal_reproduction/{PAPER_RUN_NAME}/t2smark_formal_import_candidate_records.jsonl"
    )


@pytest.mark.quick
def test_cleanup_removes_only_current_run_and_preserves_input_lock(tmp_path: Path) -> None:
    """清理应删除当前 run 残留, 但保留输入锁和其他 run。"""

    current_raw = tmp_path / "outputs" / "image_only_dataset_runtime" / PAPER_RUN_NAME
    current_derived = tmp_path / "outputs" / "attack_matrix" / PAPER_RUN_NAME
    other_run = tmp_path / "outputs" / "attack_matrix" / "pilot_paper"
    lock_dir = tmp_path / "outputs" / "paper_result_closure" / PAPER_RUN_NAME
    for directory in (current_raw, current_derived, other_run, lock_dir):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "sentinel.json").write_text("{}", encoding="utf-8")
    package_path = tmp_path / "packages" / "input.zip"
    package_path.parent.mkdir(parents=True)
    package_path.write_bytes(b"zip")

    removed = closure.clean_paper_result_closure_outputs(
        root=tmp_path,
        paper_run_name=PAPER_RUN_NAME,
        selected_package_paths=(package_path,),
    )

    assert not current_raw.exists()
    assert not current_derived.exists()
    assert other_run.is_dir()
    assert (lock_dir / "sentinel.json").is_file()
    assert f"outputs/attack_matrix/{PAPER_RUN_NAME}" in removed


@pytest.mark.quick
def test_cleanup_rejects_locked_package_inside_managed_directory(tmp_path: Path) -> None:
    """锁定包位于待清理目录时必须 fail closed。"""

    raw_dir = tmp_path / "outputs" / "image_only_dataset_runtime" / PAPER_RUN_NAME
    raw_dir.mkdir(parents=True)
    package_path = raw_dir / "locked.zip"
    package_path.write_bytes(b"zip")

    with pytest.raises(ValueError, match="锁定输入包位于受管清理目录内"):
        closure.clean_paper_result_closure_outputs(
            root=tmp_path,
            paper_run_name=PAPER_RUN_NAME,
            selected_package_paths=(package_path,),
        )

    assert package_path.is_file()


@pytest.mark.quick
def test_run_returns_current_exact_archive_without_latest_glob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """执行器应返回本次命名归档, 不从目录中猜测 latest 文件。"""

    records = build_package_records(tmp_path / "packages")
    for record in records:
        Path(str(record["package_path"])).write_bytes(b"zip")
    lock_path = tmp_path / "outputs" / "paper_result_closure" / PAPER_RUN_NAME / "closure_input_lock.json"
    selection_report = {
        "closure_input_packages": records,
        "selected_package_paths": [record["package_path"] for record in records],
        "closure_input_lock_path": lock_path.resolve().as_posix(),
        "closure_input_lock_digest": "b" * 64,
    }
    selection_calls: list[dict[str, object]] = []

    def fake_selection(*args: object, **kwargs: object) -> dict[str, object]:
        selection_calls.append({"args": args, **kwargs})
        return selection_report

    archive_name = "probe_paper_complete_result_package_current.zip"
    monkeypatch.setattr(closure, "build_closure_input_selection_report", fake_selection)
    monkeypatch.setattr(closure, "_complete_archive_name", lambda *args, **kwargs: archive_name)
    monkeypatch.setattr(
        closure,
        "build_paper_run_config",
        lambda _root: SimpleNamespace(run_name=PAPER_RUN_NAME, target_fpr=TARGET_FPR),
    )
    executed_commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        executed_commands.append(command)
        if Path(command[1]).name == "write_pilot_paper_complete_result_package.py":
            output_dir = tmp_path / argument_value(command, "--output-dir")
            drive_dir = Path(argument_value(command, "--drive-output-dir"))
            output_dir.mkdir(parents=True, exist_ok=True)
            drive_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / archive_name).write_bytes(b"current")
            (drive_dir / archive_name).write_bytes(b"current")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(closure.subprocess, "run", fake_run)
    drive_dir = tmp_path / "drive"
    drive_dir.mkdir()
    (drive_dir / "probe_paper_complete_result_package_zzz.zip").write_bytes(b"history")

    result = closure.run_paper_result_closure_commands(
        package_search_root=tmp_path / "packages",
        complete_drive_output_dir=drive_dir,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        root=tmp_path,
    )

    assert selection_calls[0]["write_lock"] is True
    assert len(executed_commands) == closure.PAPER_RESULT_CLOSURE_COMMAND_COUNT
    assert result["complete_archive_path"] == (drive_dir / archive_name).resolve().as_posix()
    assert result["local_complete_archive_path"].endswith(
        f"pilot_paper_complete_result_package/{PAPER_RUN_NAME}/{archive_name}"
    )
    assert result["complete_archive_name"] == archive_name


@pytest.mark.quick
def test_run_rejects_explicit_protocol_mismatching_current_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """闭合事务开始前必须校验显式 run 与 FPR, 不得先选择或清理输入."""

    selection_called = False

    def fake_selection(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal selection_called
        selection_called = True
        return {}

    monkeypatch.setattr(closure, "build_closure_input_selection_report", fake_selection)
    monkeypatch.setattr(
        closure,
        "build_paper_run_config",
        lambda _root: SimpleNamespace(run_name="pilot_paper", target_fpr=0.01),
    )
    with pytest.raises(ValueError, match="build_paper_run_config"):
        closure.run_paper_result_closure_commands(
            package_search_root=tmp_path / "packages",
            complete_drive_output_dir=tmp_path / "complete",
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            root=tmp_path,
        )
    assert selection_called is False


@pytest.mark.quick
def test_server_dry_run_uses_exact_selection_without_writing_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """服务器 dry-run 应执行包内治理选择, 而不是文件名覆盖统计。"""

    package_root = tmp_path / "packages"
    package_root.mkdir()
    environment_report = {
        "root": tmp_path.as_posix(),
        "paper_run": {"run_name": PAPER_RUN_NAME},
        "package_search_root": package_root.as_posix(),
        "target_fpr": f"{TARGET_FPR:g}",
    }
    monkeypatch.setattr(
        server_closure,
        "configure_closure_environment",
        lambda **kwargs: environment_report,
    )
    selection_calls: list[dict[str, object]] = []

    def fake_selection(*args: object, **kwargs: object) -> dict[str, object]:
        selection_calls.append({"args": args, **kwargs})
        return {
            "closure_input_selection_ready": True,
            "closure_input_package_count": len(CLOSURE_PACKAGE_FAMILY_SPECS),
        }

    monkeypatch.setattr(server_closure, "build_closure_input_selection_report", fake_selection)
    result = server_closure.execute_server_result_closure(
        root=tmp_path,
        paper_run_name=PAPER_RUN_NAME,
        package_search_root=package_root,
        complete_output_dir=tmp_path / "complete",
        dry_run=True,
    )

    assert result["server_result_closure_plan_ready"] is True
    assert result["dry_run"] is True
    assert selection_calls[0]["write_lock"] is False
    assert selection_calls[0]["paper_run_name"] == PAPER_RUN_NAME
