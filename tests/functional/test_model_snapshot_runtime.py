"""验证外部模型共享快照的物化与文件级证据。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from paper_experiments.runners.model_snapshot_runtime import (
    DIFFUSERS_PIPELINE_ALLOW_PATTERNS,
    build_model_snapshot_content,
    build_shared_hugging_face_snapshot_dir,
    ensure_hugging_face_snapshot_files,
    validate_reusable_model_snapshot,
)


pytestmark = pytest.mark.quick


def _write_snapshot_report(path: Path, snapshot: dict[str, Any]) -> None:
    """写入测试使用的最小受治理快照报告。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"model_snapshot_content": snapshot}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_shared_snapshot_dir_binds_repository_and_exact_revision(tmp_path: Path) -> None:
    """共享目录应同时隔离仓库和不可变 revision。"""

    revision = "1" * 40
    path = build_shared_hugging_face_snapshot_dir(
        "owner/model",
        revision,
        cache_root=tmp_path,
    )

    assert path == tmp_path / "models--owner--model" / "snapshots" / revision
    with pytest.raises(ValueError, match="40位"):
        build_shared_hugging_face_snapshot_dir("owner/model", "main", cache_root=tmp_path)


def test_model_snapshot_content_binds_patterns_revision_and_every_file(tmp_path: Path) -> None:
    """证据应绑定组件集合并排除下载器缓存。"""

    repository_dir = tmp_path / "model"
    (repository_dir / "unet").mkdir(parents=True)
    (repository_dir / ".cache" / "huggingface").mkdir(parents=True)
    (repository_dir / "model_index.json").write_text("{}\n", encoding="utf-8")
    (repository_dir / "unet" / "weights.bin").write_bytes(b"weights")
    (repository_dir / ".cache" / "huggingface" / "metadata").write_text("cache", encoding="utf-8")

    snapshot = build_model_snapshot_content(
        repository_dir,
        repository_id="owner/model",
        revision="2" * 40,
        allow_patterns=("unet/*", "model_index.json"),
    )

    assert snapshot["repository_id"] == "owner/model"
    assert snapshot["revision"] == "2" * 40
    assert snapshot["allow_patterns"] == ["model_index.json", "unet/*"]
    assert [row["path"] for row in snapshot["files"]] == ["model_index.json", "unet/weights.bin"]
    assert len(snapshot["snapshot_content_digest"]) == 64


def test_model_snapshot_content_rejects_unselected_file(tmp_path: Path) -> None:
    """共享目录中未声明的文件不得进入正式模型输入。"""

    repository_dir = tmp_path / "model"
    repository_dir.mkdir()
    (repository_dir / "model_index.json").write_text("{}", encoding="utf-8")
    (repository_dir / "unregistered.bin").write_bytes(b"unexpected")

    with pytest.raises(RuntimeError, match="allow_patterns 外文件"):
        build_model_snapshot_content(
            repository_dir,
            repository_id="owner/model",
            revision="3" * 40,
            allow_patterns=("model_index.json",),
        )


def test_reusable_model_snapshot_rejects_pattern_or_file_drift(tmp_path: Path) -> None:
    """复用时, 组件选择或任一模型文件变化都必须阻断。"""

    repository_dir = tmp_path / "model"
    repository_dir.mkdir()
    weights_path = repository_dir / "weights.bin"
    weights_path.write_bytes(b"first")
    snapshot = build_model_snapshot_content(
        repository_dir,
        repository_id="owner/model",
        revision="4" * 40,
        allow_patterns=("weights.bin",),
    )
    report_path = tmp_path / "outputs" / "model_repository_prepare_result.json"
    _write_snapshot_report(report_path, snapshot)

    assert validate_reusable_model_snapshot(
        repository_dir,
        report_path=report_path,
        repository_id="owner/model",
        revision="4" * 40,
        allow_patterns=("weights.bin",),
    ) == snapshot

    with pytest.raises(RuntimeError, match="文件级证据不一致"):
        validate_reusable_model_snapshot(
            repository_dir,
            report_path=report_path,
            repository_id="owner/model",
            revision="4" * 40,
            allow_patterns=("weights.*",),
        )

    weights_path.write_bytes(b"second")
    with pytest.raises(RuntimeError, match="文件级证据不一致"):
        validate_reusable_model_snapshot(
            repository_dir,
            report_path=report_path,
            repository_id="owner/model",
            revision="4" * 40,
            allow_patterns=("weights.bin",),
        )


def test_ensure_snapshot_reuses_valid_report_without_download(tmp_path: Path) -> None:
    """文件级证据有效时不应再次访问远端。"""

    repository_dir = tmp_path / "model"
    repository_dir.mkdir()
    (repository_dir / "model_index.json").write_text("{}", encoding="utf-8")
    snapshot = build_model_snapshot_content(
        repository_dir,
        repository_id="owner/model",
        revision="5" * 40,
        allow_patterns=("model_index.json",),
    )
    report_path = tmp_path / "outputs" / "prepare.json"
    _write_snapshot_report(report_path, snapshot)

    def forbidden_download(**_: Any) -> str:
        raise AssertionError("可信快照不应触发下载")

    result = ensure_hugging_face_snapshot_files(
        repository_dir,
        report_path=report_path,
        repository_id="owner/model",
        revision="5" * 40,
        allow_patterns=("model_index.json",),
        token=None,
        snapshot_download_fn=forbidden_download,
    )

    assert result["download_performed"] is False
    assert result["reused_model_snapshot_content"] == snapshot


def test_ensure_snapshot_resumes_when_report_missing_even_with_model_index(tmp_path: Path) -> None:
    """报告缺失时必须按精确来源续传, 不得仅凭 model_index 判定完成。"""

    repository_dir = tmp_path / "model"
    repository_dir.mkdir()
    (repository_dir / "model_index.json").write_text("{}", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def fake_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        (Path(kwargs["local_dir"]) / "unet").mkdir(parents=True, exist_ok=True)
        (Path(kwargs["local_dir"]) / "unet" / "weights.bin").write_bytes(b"weights")
        return kwargs["local_dir"]

    result = ensure_hugging_face_snapshot_files(
        repository_dir,
        report_path=tmp_path / "outputs" / "missing.json",
        repository_id="owner/model",
        revision="6" * 40,
        allow_patterns=("model_index.json", "unet/*"),
        token="token",
        snapshot_download_fn=fake_download,
    )

    assert result["download_performed"] is True
    assert result["repair_requested"] is False
    assert calls[0]["revision"] == "6" * 40
    assert calls[0]["allow_patterns"] == ["model_index.json", "unet/*"]
    assert calls[0]["force_download"] is False
    assert calls[0]["token"] == "token"


def test_ensure_snapshot_repairs_drift_without_overwriting_prior_report(tmp_path: Path) -> None:
    """已知漂移应强制修复, 且物化函数不得覆盖既有证据。"""

    repository_dir = tmp_path / "model"
    repository_dir.mkdir()
    weights_path = repository_dir / "model_index.json"
    weights_path.write_text("original", encoding="utf-8")
    snapshot = build_model_snapshot_content(
        repository_dir,
        repository_id="owner/model",
        revision="7" * 40,
        allow_patterns=("model_index.json",),
    )
    report_path = tmp_path / "outputs" / "prepare.json"
    _write_snapshot_report(report_path, snapshot)
    report_before = report_path.read_bytes()
    weights_path.write_text("drifted", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def fake_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        (Path(kwargs["local_dir"]) / "model_index.json").write_text("original", encoding="utf-8")
        return kwargs["local_dir"]

    result = ensure_hugging_face_snapshot_files(
        repository_dir,
        report_path=report_path,
        repository_id="owner/model",
        revision="7" * 40,
        allow_patterns=("model_index.json",),
        token=None,
        snapshot_download_fn=fake_download,
    )

    assert result["repair_requested"] is True
    assert calls[0]["force_download"] is True
    assert report_path.read_bytes() == report_before
    assert result["downloaded_snapshot_content_before_transform"] == snapshot


def test_diffusers_patterns_cover_declared_pipeline_components() -> None:
    """共享常量应明确列出官方参考 pipeline 读取的组件。"""

    assert DIFFUSERS_PIPELINE_ALLOW_PATTERNS == (
        "feature_extractor/*",
        "model_index.json",
        "safety_checker/*",
        "scheduler/*",
        "text_encoder/*",
        "tokenizer/*",
        "unet/*",
        "vae/*",
    )
