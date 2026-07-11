"""验证三套官方参考共用的不可变 OpenCLIP checkpoint。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from paper_experiments.runners import model_snapshot_runtime, openclip_checkpoint_runtime
from paper_experiments.runners.openclip_checkpoint_runtime import (
    OPENCLIP_ALLOW_PATTERNS,
    OPENCLIP_CHECKPOINT_FILENAME,
    OPENCLIP_CHECKPOINT_SHA256,
    OPENCLIP_CHECKPOINT_SIZE_BYTES,
    OPENCLIP_REPOSITORY_ID,
    OPENCLIP_REVISION,
    build_openclip_snapshot_dir,
    prepare_openclip_checkpoint,
    write_openclip_checkpoint_report,
)


pytestmark = pytest.mark.quick


def _write_checkpoint_fixture(path: Path) -> None:
    """创建用于验证物化控制流的轻量 checkpoint 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"openclip checkpoint fixture")


def _patch_snapshot_content(monkeypatch: pytest.MonkeyPatch, digest: str) -> dict[str, Any]:
    """在轻量测试中替代5GB文件哈希, 保留完整证据结构。"""

    snapshot_content = {
        "repository_id": OPENCLIP_REPOSITORY_ID,
        "revision": OPENCLIP_REVISION,
        "allow_patterns": list(OPENCLIP_ALLOW_PATTERNS),
        "file_count": 1,
        "files": [
            {
                "path": OPENCLIP_CHECKPOINT_FILENAME,
                "size_bytes": OPENCLIP_CHECKPOINT_SIZE_BYTES,
                "sha256": digest,
            }
        ],
        "snapshot_content_digest": "f" * 64,
    }
    monkeypatch.setattr(
        model_snapshot_runtime,
        "build_model_snapshot_content",
        lambda *_args, **_kwargs: snapshot_content,
    )
    return snapshot_content


def test_openclip_snapshot_path_binds_exact_repository_and_revision(tmp_path: Path) -> None:
    """三套 runner 应从同一个精确 revision 目录取得 checkpoint。"""

    snapshot_dir = build_openclip_snapshot_dir(tmp_path)

    assert snapshot_dir == (
        tmp_path
        / "models--laion--CLIP-ViT-g-14-laion2B-s12B-b42K"
        / "snapshots"
        / OPENCLIP_REVISION
    )


def test_prepare_openclip_checkpoint_downloads_only_registered_bin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首次物化必须传递精确 revision, 并只下载登记的 bin 文件。"""

    _patch_snapshot_content(monkeypatch, OPENCLIP_CHECKPOINT_SHA256)
    calls: list[dict[str, Any]] = []

    def fake_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        _write_checkpoint_fixture(Path(kwargs["local_dir"]) / OPENCLIP_CHECKPOINT_FILENAME)
        return kwargs["local_dir"]

    report = prepare_openclip_checkpoint(
        report_path=tmp_path / "outputs" / "openclip_prepare.json",
        cache_root=tmp_path / "cache",
        token="token",
        snapshot_download_fn=fake_download,
    )

    assert calls == [
        {
            "repo_id": OPENCLIP_REPOSITORY_ID,
            "revision": OPENCLIP_REVISION,
            "local_dir": str(build_openclip_snapshot_dir(tmp_path / "cache")),
            "local_dir_use_symlinks": False,
            "token": "token",
            "allow_patterns": list(OPENCLIP_ALLOW_PATTERNS),
            "force_download": False,
        }
    ]
    assert report["openclip_checkpoint_ready"] is True
    assert report["openclip_checkpoint_sha256"] == OPENCLIP_CHECKPOINT_SHA256
    assert report["openclip_checkpoint_size_bytes"] == OPENCLIP_CHECKPOINT_SIZE_BYTES
    assert report["model_snapshot_content"]["files"] == [
        {
            "path": OPENCLIP_CHECKPOINT_FILENAME,
            "size_bytes": OPENCLIP_CHECKPOINT_SIZE_BYTES,
            "sha256": OPENCLIP_CHECKPOINT_SHA256,
        }
    ]


def test_prepare_openclip_checkpoint_reuses_matching_written_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """runner 写出的成功报告应允许其他会话逐文件复核后复用缓存。"""

    snapshot_content = _patch_snapshot_content(monkeypatch, OPENCLIP_CHECKPOINT_SHA256)
    cache_root = tmp_path / "cache"
    snapshot_dir = build_openclip_snapshot_dir(cache_root)
    _write_checkpoint_fixture(snapshot_dir / OPENCLIP_CHECKPOINT_FILENAME)
    report_path = tmp_path / "outputs" / "openclip_prepare.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps({"model_snapshot_content": snapshot_content}),
        encoding="utf-8",
    )

    def forbidden_download(**_: Any) -> str:
        raise AssertionError("匹配的共享 checkpoint 不应重复下载")

    report = prepare_openclip_checkpoint(
        report_path=report_path,
        cache_root=cache_root,
        snapshot_download_fn=forbidden_download,
    )

    assert report["openclip_checkpoint_ready"] is True
    assert report["snapshot_materialization"]["download_performed"] is False
    assert report["model_snapshot_content"] == snapshot_content


def test_prepare_openclip_checkpoint_rejects_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """即使下载成功, checkpoint 哈希不匹配也不得返回 ready。"""

    _patch_snapshot_content(monkeypatch, "0" * 64)

    def fake_download(**kwargs: Any) -> str:
        _write_checkpoint_fixture(Path(kwargs["local_dir"]) / OPENCLIP_CHECKPOINT_FILENAME)
        return kwargs["local_dir"]

    with pytest.raises(RuntimeError, match="SHA-256"):
        prepare_openclip_checkpoint(
            report_path=tmp_path / "outputs" / "openclip_prepare.json",
            cache_root=tmp_path / "cache",
            snapshot_download_fn=fake_download,
        )


def test_write_openclip_checkpoint_report_persists_fail_closed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """checkpoint 物化异常必须写出可审计失败报告, 不能静默继续执行。"""

    def fail_prepare(**_: Any) -> dict[str, Any]:
        raise RuntimeError("fixture_failure")

    monkeypatch.setattr(
        openclip_checkpoint_runtime,
        "prepare_openclip_checkpoint",
        fail_prepare,
    )
    report_path = tmp_path / "outputs" / "openclip_prepare.json"

    report = write_openclip_checkpoint_report(report_path)

    assert report["openclip_checkpoint_requested"] is True
    assert report["openclip_checkpoint_ready"] is False
    assert report["failure_reason"] == "openclip_checkpoint_materialization_failed"
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_write_openclip_checkpoint_report_records_not_requested_state(tmp_path: Path) -> None:
    """未请求官方命令时只记录跳过状态, 不触发大型 checkpoint 下载。"""

    report_path = tmp_path / "outputs" / "openclip_prepare.json"

    report = write_openclip_checkpoint_report(report_path, requested=False)

    assert report == {
        "openclip_checkpoint_requested": False,
        "openclip_checkpoint_ready": False,
        "skip_reason": "official_command_not_requested",
    }
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
