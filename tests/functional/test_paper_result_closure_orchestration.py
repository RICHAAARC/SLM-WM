"""CPU 论文结果闭合编排的轻量功能测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import paper_result_closure as closure
from scripts import run_gpu_server_result_closure as server_closure


PAPER_RUN_NAME = "probe_paper"
TARGET_FPR = 0.1
@pytest.mark.quick
def test_closure_command_plan_rejects_single_repeat_component(
    tmp_path: Path,
) -> None:
    """单 repeat 输入不得继续构造论文闭合 DAG。"""

    with pytest.raises(RuntimeError, match="精确9重复聚合证据"):
        closure.build_paper_result_closure_commands(
            randomization_repeat_components=(
                {
                    "package_family": "randomization_repeat_evidence",
                    "repeat_component_ready": True,
                    "randomization_aggregate_ready": False,
                    "supports_paper_claim": False,
                },
            ),
            complete_drive_output_dir=tmp_path / "complete",
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            archive_name="probe_paper_complete_result_package_exact.zip",
            root=tmp_path,
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
    """聚合闭合执行器必须在任何选择、清理和子命令前失败。"""

    del monkeypatch
    with pytest.raises(RuntimeError, match="精确9重复聚合证据"):
        closure.run_paper_result_closure_commands(
            package_search_root=tmp_path / "packages",
            complete_drive_output_dir=tmp_path / "complete",
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            root=tmp_path,
        )


@pytest.mark.quick
def test_run_rejects_explicit_protocol_mismatching_current_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """聚合证据不存在时, 任意显式协议都必须 fail-closed。"""

    del monkeypatch
    with pytest.raises(RuntimeError, match="精确9重复聚合证据"):
        closure.run_paper_result_closure_commands(
            package_search_root=tmp_path / "packages",
            complete_drive_output_dir=tmp_path / "complete",
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            root=tmp_path,
        )


@pytest.mark.quick
def test_server_closure_requires_cross_repeat_record_recomputation(
    tmp_path: Path,
) -> None:
    """服务器闭合在跨重复原始记录重算接入前必须保持 fail-closed."""

    aggregate_path = tmp_path / "aggregate.zip"
    aggregate_path.write_bytes(b"not-consumed-before-gate")
    repository_commit = "a" * 40
    with pytest.raises(RuntimeError, match="跨重复原始记录"):
        server_closure.execute_server_result_closure(
            root=tmp_path,
            paper_run_name=PAPER_RUN_NAME,
            randomization_aggregate_package_path=aggregate_path,
            complete_output_dir=tmp_path / "complete",
            repository_commit=repository_commit,
            dry_run=True,
        )
