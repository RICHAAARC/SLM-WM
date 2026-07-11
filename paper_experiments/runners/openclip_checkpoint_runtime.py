"""准备三套官方参考共同使用的不可变 OpenCLIP 本地 checkpoint。"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

from experiments.runtime.model_sources import (
    ModelSourceRequiredFile,
    get_model_source,
    require_registered_model_reference,
)
from paper_experiments.runners.model_snapshot_runtime import (
    DEFAULT_SHARED_HUGGING_FACE_SNAPSHOT_ROOT,
    build_shared_hugging_face_snapshot_dir,
    ensure_hugging_face_snapshot_files,
)


OPENCLIP_SOURCE_NAME = "laion_clip_vit_g14"
OPENCLIP_USAGE_ROLE = "official_reference_openclip_encoder"
OPENCLIP_MODEL_NAME = "ViT-g-14"
OPENCLIP_CHECKPOINT_FILENAME = "open_clip_pytorch_model.bin"
OPENCLIP_CHECKPOINT_SHA256 = "6aac683f899159946bc4ca15228bb7016f3cbb1a2c51f365cba0b23923f344da"
OPENCLIP_CHECKPOINT_SIZE_BYTES = 5467006745
OPENCLIP_ALLOW_PATTERNS = (OPENCLIP_CHECKPOINT_FILENAME,)


_OPENCLIP_SOURCE = get_model_source(OPENCLIP_SOURCE_NAME)
_EXPECTED_REQUIRED_FILE = ModelSourceRequiredFile(
    path=OPENCLIP_CHECKPOINT_FILENAME,
    sha256=OPENCLIP_CHECKPOINT_SHA256,
    size_bytes=OPENCLIP_CHECKPOINT_SIZE_BYTES,
)


def _validate_registered_openclip_source() -> None:
    """要求仓库、revision、用途和 checkpoint 文件身份同时受登记表约束。"""

    require_registered_model_reference(
        _OPENCLIP_SOURCE.repository_id,
        _OPENCLIP_SOURCE.revision,
        required_usage_role=OPENCLIP_USAGE_ROLE,
    )
    if _OPENCLIP_SOURCE.required_files != (_EXPECTED_REQUIRED_FILE,):
        raise RuntimeError("OpenCLIP 登记源必须且只能声明正式 ViT-g-14 checkpoint")


_validate_registered_openclip_source()


OPENCLIP_REPOSITORY_ID = _OPENCLIP_SOURCE.repository_id
OPENCLIP_REVISION = _OPENCLIP_SOURCE.revision
DEFAULT_OPENCLIP_SNAPSHOT_DIR = str(
    build_shared_hugging_face_snapshot_dir(
        OPENCLIP_REPOSITORY_ID,
        OPENCLIP_REVISION,
    )
)
DEFAULT_OPENCLIP_CHECKPOINT_PATH = str(
    Path(DEFAULT_OPENCLIP_SNAPSHOT_DIR) / OPENCLIP_CHECKPOINT_FILENAME
)


def build_openclip_snapshot_dir(
    cache_root: str | Path = DEFAULT_SHARED_HUGGING_FACE_SNAPSHOT_ROOT,
) -> Path:
    """返回按登记仓库与精确 revision 命名的共享 OpenCLIP 目录。"""

    return build_shared_hugging_face_snapshot_dir(
        OPENCLIP_REPOSITORY_ID,
        OPENCLIP_REVISION,
        cache_root=cache_root,
    )


def _validate_checkpoint_snapshot_content(snapshot_content: dict[str, Any]) -> None:
    """要求快照证据只包含预期 checkpoint, 且大小与 SHA-256 完全匹配。"""

    expected_files = [
        {
            "path": OPENCLIP_CHECKPOINT_FILENAME,
            "size_bytes": OPENCLIP_CHECKPOINT_SIZE_BYTES,
            "sha256": OPENCLIP_CHECKPOINT_SHA256,
        }
    ]
    if snapshot_content.get("repository_id") != OPENCLIP_REPOSITORY_ID:
        raise RuntimeError("OpenCLIP 快照仓库身份不匹配")
    if snapshot_content.get("revision") != OPENCLIP_REVISION:
        raise RuntimeError("OpenCLIP 快照 revision 不匹配")
    if snapshot_content.get("allow_patterns") != list(OPENCLIP_ALLOW_PATTERNS):
        raise RuntimeError("OpenCLIP 快照 allow_patterns 不匹配")
    if snapshot_content.get("file_count") != 1 or snapshot_content.get("files") != expected_files:
        raise RuntimeError("OpenCLIP checkpoint 文件大小或 SHA-256 与登记证据不一致")
    digest = str(snapshot_content.get("snapshot_content_digest", ""))
    if len(digest) != 64:
        raise RuntimeError("OpenCLIP 快照缺少64位内容摘要")


def prepare_openclip_checkpoint(
    *,
    report_path: str | Path,
    cache_root: str | Path = DEFAULT_SHARED_HUGGING_FACE_SNAPSHOT_ROOT,
    token: str | None = None,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """物化并验证官方参考共用的本地 OpenCLIP checkpoint。

    返回值同时保留通用 ``model_snapshot_content`` 键, runner 可直接将整个
    字典写入各自 outputs 报告。下一次运行会依据该键逐文件复核共享缓存,
    而不会退回可漂移的远程 pretrained tag。
    """

    _validate_registered_openclip_source()
    snapshot_dir = build_openclip_snapshot_dir(cache_root)
    materialization = ensure_hugging_face_snapshot_files(
        snapshot_dir,
        report_path=report_path,
        repository_id=OPENCLIP_REPOSITORY_ID,
        revision=OPENCLIP_REVISION,
        allow_patterns=OPENCLIP_ALLOW_PATTERNS,
        token=token,
        snapshot_download_fn=snapshot_download_fn,
    )
    snapshot_content = materialization.get("reused_model_snapshot_content")
    if not isinstance(snapshot_content, dict):
        snapshot_content = materialization.get("downloaded_snapshot_content_before_transform")
    if not isinstance(snapshot_content, dict):
        raise RuntimeError("OpenCLIP 物化结果缺少文件级快照证据")
    _validate_checkpoint_snapshot_content(snapshot_content)

    checkpoint_path = snapshot_dir / OPENCLIP_CHECKPOINT_FILENAME
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise RuntimeError("OpenCLIP checkpoint 必须是共享快照中的普通文件")

    return {
        "openclip_checkpoint_requested": True,
        "openclip_checkpoint_ready": True,
        "openclip_source_name": OPENCLIP_SOURCE_NAME,
        "openclip_usage_role": OPENCLIP_USAGE_ROLE,
        "openclip_model_name": OPENCLIP_MODEL_NAME,
        "openclip_repository_id": OPENCLIP_REPOSITORY_ID,
        "openclip_revision": OPENCLIP_REVISION,
        "openclip_snapshot_dir": str(snapshot_dir.resolve()),
        "openclip_checkpoint_filename": OPENCLIP_CHECKPOINT_FILENAME,
        "openclip_checkpoint_path": str(checkpoint_path.resolve()),
        "openclip_checkpoint_sha256": OPENCLIP_CHECKPOINT_SHA256,
        "openclip_checkpoint_size_bytes": OPENCLIP_CHECKPOINT_SIZE_BYTES,
        "openclip_snapshot_content_digest": snapshot_content["snapshot_content_digest"],
        "model_source": _OPENCLIP_SOURCE.to_dict(),
        "model_snapshot_content": snapshot_content,
        "snapshot_materialization": materialization,
    }


def write_openclip_checkpoint_report(
    report_path: str | Path,
    *,
    requested: bool = True,
    cache_root: str | Path = DEFAULT_SHARED_HUGGING_FACE_SNAPSHOT_ROOT,
    token: str | None = None,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """物化 checkpoint 并把成功或失败状态写为独立运行证据。"""

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not requested:
        report: dict[str, Any] = {
            "openclip_checkpoint_requested": False,
            "openclip_checkpoint_ready": False,
            "skip_reason": "official_command_not_requested",
        }
    else:
        try:
            report = prepare_openclip_checkpoint(
                report_path=path,
                cache_root=cache_root,
                token=token,
                snapshot_download_fn=snapshot_download_fn,
            )
        except Exception as error:
            report = {
                "openclip_checkpoint_requested": True,
                "openclip_checkpoint_ready": False,
                "failure_reason": "openclip_checkpoint_materialization_failed",
                "error": f"{type(error).__name__}:{error}",
            }
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
