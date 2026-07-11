"""外部续跑检查点的摘要、执行锁与证据边界测试."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

import experiments.runtime.resume_checkpoint as resume_checkpoint
from main.core.digest import build_stable_digest
from tests.helpers.formal_execution_lock import build_test_formal_execution_lock


PAPER_RUN_NAME = "probe_paper"
ARTIFACT_ROLE = "image_only_dataset_runtime"
OUTPUT_PREFIX = f"outputs/{ARTIFACT_ROLE}/{PAPER_RUN_NAME}"
FORMAL_EXECUTION_LOCK = build_test_formal_execution_lock()


@pytest.fixture(autouse=True)
def configure_execution_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """让检查点测试使用确定且可替换的正式执行锁."""

    monkeypatch.delenv(
        resume_checkpoint.CHECKPOINT_ROOT_ENVIRONMENT_KEY,
        raising=False,
    )
    monkeypatch.setattr(
        resume_checkpoint,
        "require_published_formal_execution_lock",
        lambda _root: dict(FORMAL_EXECUTION_LOCK),
    )


def write_completed_unit(root: Path) -> tuple[Path, Path]:
    """写出一个数据文件和最后发布的科学单元 manifest."""

    unit_dir = root / OUTPUT_PREFIX / "prompt_0001"
    data_path = unit_dir / "detection_records.jsonl"
    manifest_path = unit_dir / "manifest.local.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text('{"score": 1.0}\n', encoding="utf-8")
    output_paths = [
        data_path.relative_to(root).as_posix(),
        manifest_path.relative_to(root).as_posix(),
    ]
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_id": "semantic_watermark_runtime_manifest",
                "output_paths": output_paths,
                "supports_paper_claim": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return data_path, manifest_path


def rewrite_checkpoint_manifest(
    manifest_path: Path,
    transform: object,
) -> dict[str, object]:
    """修改检查点声明并重算内容摘要和 manifest 自摘要."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    transform(payload)
    payload["checkpoint_content_digest"] = build_stable_digest(
        payload["entry_records"]
    )
    payload.pop("checkpoint_manifest_digest", None)
    payload["checkpoint_manifest_digest"] = build_stable_digest(payload)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


@pytest.mark.quick
def test_unconfigured_checkpoint_is_a_server_safe_noop(tmp_path: Path) -> None:
    """服务器未配置外部目录时不得创建隐式路径或阻断工作负载."""

    progress_path = tmp_path / OUTPUT_PREFIX / "dataset_runtime_progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("{}\n", encoding="utf-8")

    persisted = resume_checkpoint.persist_progress_checkpoint(
        progress_path,
        repository_root=tmp_path,
        artifact_role=ARTIFACT_ROLE,
        paper_run_name=PAPER_RUN_NAME,
    )
    restored = resume_checkpoint.restore_role_checkpoints(
        repository_root=tmp_path,
        artifact_role=ARTIFACT_ROLE,
        paper_run_name=PAPER_RUN_NAME,
        allowed_output_prefix=OUTPUT_PREFIX,
    )

    assert persisted["checkpoint_persistence_configured"] is False
    assert persisted["checkpoint_persisted"] is False
    assert restored["checkpoint_persistence_configured"] is False
    assert restored["restored_file_count"] == 0


@pytest.mark.quick
def test_empty_role_directory_does_not_clear_local_outputs(tmp_path: Path) -> None:
    """空的持久化角色目录不能触发恢复发布回调."""

    checkpoint_root = tmp_path / "external_checkpoint"
    (
        checkpoint_root / PAPER_RUN_NAME / ARTIFACT_ROLE
    ).mkdir(parents=True)
    callback_calls: list[str] = []

    restored = resume_checkpoint.restore_role_checkpoints(
        repository_root=tmp_path,
        artifact_role=ARTIFACT_ROLE,
        paper_run_name=PAPER_RUN_NAME,
        allowed_output_prefix=OUTPUT_PREFIX,
        checkpoint_root=checkpoint_root,
        before_restore=lambda: callback_calls.append("called"),
    )

    assert restored["restored_manifest_count"] == 0
    assert restored["restored_file_count"] == 0
    assert callback_calls == []


@pytest.mark.quick
def test_completed_unit_restores_atomically_and_manifest_last(tmp_path: Path) -> None:
    """完成单元应按摘要恢复并以 manifest 作为最后可见的完成标记."""

    checkpoint_root = tmp_path / "external_checkpoint"
    data_path, manifest_path = write_completed_unit(tmp_path)
    expected_data = data_path.read_bytes()
    expected_manifest = manifest_path.read_bytes()
    persisted = resume_checkpoint.persist_completed_unit_from_manifest(
        manifest_path,
        repository_root=tmp_path,
        artifact_role=ARTIFACT_ROLE,
        paper_run_name=PAPER_RUN_NAME,
        checkpoint_root=checkpoint_root,
    )
    shutil.rmtree(data_path.parent)
    data_path.parent.mkdir(parents=True)
    data_path.write_bytes(b"corrupted")

    restored = resume_checkpoint.restore_role_checkpoints(
        repository_root=tmp_path,
        artifact_role=ARTIFACT_ROLE,
        paper_run_name=PAPER_RUN_NAME,
        allowed_output_prefix=OUTPUT_PREFIX,
        checkpoint_root=checkpoint_root,
    )

    assert persisted["checkpoint_persisted"] is True
    assert restored["restored_manifest_count"] == 1
    assert restored["restored_file_count"] == 2
    assert data_path.read_bytes() == expected_data
    assert manifest_path.read_bytes() == expected_manifest
    assert not list(tmp_path.rglob("*.partial"))


@pytest.mark.quick
def test_restore_rejects_tampered_payload_digest(tmp_path: Path) -> None:
    """外部 payload 任一字节变化后必须拒绝恢复."""

    checkpoint_root = tmp_path / "external_checkpoint"
    data_path, manifest_path = write_completed_unit(tmp_path)
    persisted = resume_checkpoint.persist_completed_unit_from_manifest(
        manifest_path,
        repository_root=tmp_path,
        artifact_role=ARTIFACT_ROLE,
        paper_run_name=PAPER_RUN_NAME,
        checkpoint_root=checkpoint_root,
    )
    checkpoint_manifest = Path(persisted["checkpoint_manifest_path"])
    payload = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    payload_path = (
        checkpoint_manifest.parent
        / "payload"
        / payload["entry_records"][0]["payload_name_intermediate"]
    )
    payload_path.write_bytes(b"tampered")
    shutil.rmtree(data_path.parent)

    with pytest.raises(RuntimeError, match="payload 摘要不一致"):
        resume_checkpoint.restore_role_checkpoints(
            repository_root=tmp_path,
            artifact_role=ARTIFACT_ROLE,
            paper_run_name=PAPER_RUN_NAME,
            allowed_output_prefix=OUTPUT_PREFIX,
            checkpoint_root=checkpoint_root,
        )


@pytest.mark.quick
def test_restore_rejects_formal_execution_lock_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """当前提交锁变化后不得复用旧提交生成的科学单元."""

    checkpoint_root = tmp_path / "external_checkpoint"
    _, manifest_path = write_completed_unit(tmp_path)
    resume_checkpoint.persist_completed_unit_from_manifest(
        manifest_path,
        repository_root=tmp_path,
        artifact_role=ARTIFACT_ROLE,
        paper_run_name=PAPER_RUN_NAME,
        checkpoint_root=checkpoint_root,
    )
    monkeypatch.setattr(
        resume_checkpoint,
        "require_published_formal_execution_lock",
        lambda _root: build_test_formal_execution_lock("c" * 40),
    )

    with pytest.raises(RuntimeError, match="manifest 未通过治理校验"):
        resume_checkpoint.restore_role_checkpoints(
            repository_root=tmp_path,
            artifact_role=ARTIFACT_ROLE,
            paper_run_name=PAPER_RUN_NAME,
            allowed_output_prefix=OUTPUT_PREFIX,
            checkpoint_root=checkpoint_root,
        )


@pytest.mark.quick
def test_restore_rejects_self_consistent_path_traversal(tmp_path: Path) -> None:
    """即使攻击者重算所有摘要, 包含父目录跳转的路径仍必须被拒绝."""

    checkpoint_root = tmp_path / "external_checkpoint"
    _, manifest_path = write_completed_unit(tmp_path)
    persisted = resume_checkpoint.persist_completed_unit_from_manifest(
        manifest_path,
        repository_root=tmp_path,
        artifact_role=ARTIFACT_ROLE,
        paper_run_name=PAPER_RUN_NAME,
        checkpoint_root=checkpoint_root,
    )
    checkpoint_manifest = Path(persisted["checkpoint_manifest_path"])

    def inject_traversal(payload: dict[str, object]) -> None:
        """把首个成员改成表面仍位于角色前缀下的逃逸路径."""

        payload["entry_records"][0]["path"] = (
            f"{OUTPUT_PREFIX}/../escaped.json"
        )

    rewrite_checkpoint_manifest(checkpoint_manifest, inject_traversal)

    with pytest.raises(RuntimeError, match="角色目录外路径"):
        resume_checkpoint.restore_role_checkpoints(
            repository_root=tmp_path,
            artifact_role=ARTIFACT_ROLE,
            paper_run_name=PAPER_RUN_NAME,
            allowed_output_prefix=OUTPUT_PREFIX,
            checkpoint_root=checkpoint_root,
        )


@pytest.mark.quick
def test_progress_checkpoint_cannot_claim_completed_scientific_unit(
    tmp_path: Path,
) -> None:
    """进度快照恢复后仍不得产生完成单元 manifest 或论文证据资格."""

    checkpoint_root = tmp_path / "external_checkpoint"
    progress_path = tmp_path / OUTPUT_PREFIX / "dataset_runtime_progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps(
            {
                "protocol_decision": "resume_required",
                "evidence_eligibility": "intermediate_state_only",
                "supports_paper_claim": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    persisted = resume_checkpoint.persist_progress_checkpoint(
        progress_path,
        repository_root=tmp_path,
        artifact_role=ARTIFACT_ROLE,
        paper_run_name=PAPER_RUN_NAME,
        checkpoint_root=checkpoint_root,
    )
    progress_path.unlink()

    restored = resume_checkpoint.restore_role_checkpoints(
        repository_root=tmp_path,
        artifact_role=ARTIFACT_ROLE,
        paper_run_name=PAPER_RUN_NAME,
        allowed_output_prefix=OUTPUT_PREFIX,
        checkpoint_root=checkpoint_root,
    )

    checkpoint_manifest = json.loads(
        Path(persisted["checkpoint_manifest_path"]).read_text(encoding="utf-8")
    )
    assert restored["restored_file_count"] == 1
    assert progress_path.is_file()
    assert not list((tmp_path / OUTPUT_PREFIX).rglob("manifest.local.json"))
    assert checkpoint_manifest["evidence_eligibility"] == "intermediate_state_only"
    assert checkpoint_manifest["supports_paper_claim"] is False


@pytest.mark.quick
def test_failed_progress_publication_keeps_previous_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """新进度发布失败时必须保留最近一次完整检查点入口."""

    checkpoint_root = tmp_path / "external_checkpoint"
    progress_path = tmp_path / OUTPUT_PREFIX / "dataset_runtime_progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("{}\n", encoding="utf-8")
    first = resume_checkpoint.persist_progress_checkpoint(
        progress_path,
        repository_root=tmp_path,
        artifact_role=ARTIFACT_ROLE,
        paper_run_name=PAPER_RUN_NAME,
        checkpoint_root=checkpoint_root,
    )
    first_manifest = Path(first["checkpoint_manifest_path"])
    original_persist = resume_checkpoint.persist_checkpoint_files

    def fail_publication(**_kwargs: object) -> dict[str, object]:
        """模拟新快照尚未发布 manifest 前发生中断."""

        raise OSError("simulated interruption")

    monkeypatch.setattr(
        resume_checkpoint,
        "persist_checkpoint_files",
        fail_publication,
    )
    progress_path.write_text('{"completed": 2}\n', encoding="utf-8")
    with pytest.raises(OSError, match="simulated interruption"):
        resume_checkpoint.persist_progress_checkpoint(
            progress_path,
            repository_root=tmp_path,
            artifact_role=ARTIFACT_ROLE,
            paper_run_name=PAPER_RUN_NAME,
            checkpoint_root=checkpoint_root,
        )
    monkeypatch.setattr(
        resume_checkpoint,
        "persist_checkpoint_files",
        original_persist,
    )

    assert first_manifest.is_file()
