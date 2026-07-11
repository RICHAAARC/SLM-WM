"""构造、物化并校验外部模型本地快照的文件级证据。"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any

from main.core.digest import build_stable_digest


DEFAULT_SHARED_HUGGING_FACE_SNAPSHOT_ROOT = "/content/slm_wm_model_sources"
"""Colab 与独立 GPU 服务器共用的不可变模型快照根目录。"""


DIFFUSERS_PIPELINE_ALLOW_PATTERNS: tuple[str, ...] = (
    "feature_extractor/*",
    "model_index.json",
    "safety_checker/*",
    "scheduler/*",
    "text_encoder/*",
    "tokenizer/*",
    "unet/*",
    "vae/*",
)
"""Stable Diffusion ``DiffusionPipeline`` 实际读取的仓库组件。"""


_IMMUTABLE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


def _validate_repository_id(repository_id: str) -> str:
    """校验 Hugging Face 仓库标识, 避免构造出根目录外路径。"""

    normalized = str(repository_id).strip()
    if not _REPOSITORY_ID_PATTERN.fullmatch(normalized):
        raise ValueError("Hugging Face 仓库标识必须是安全的 repo 或 owner/repo 形式")
    return normalized


def _validate_immutable_revision(revision: str) -> str:
    """要求 revision 为40位小写 Git 提交, 禁止 branch 或 tag 漂移。"""

    normalized = str(revision).strip()
    if not _IMMUTABLE_REVISION_PATTERN.fullmatch(normalized):
        raise ValueError("正式模型 revision 必须是40位小写十六进制 Git 提交")
    return normalized


def normalize_allow_patterns(allow_patterns: Sequence[str]) -> tuple[str, ...]:
    """规范化下载选择器, 使相同组件集合产生同一证据摘要。"""

    normalized: set[str] = set()
    for raw_pattern in allow_patterns:
        pattern = str(raw_pattern).strip().replace("\\", "/")
        parts = PurePosixPath(pattern).parts
        if not pattern or pattern.startswith("/") or ".." in parts:
            raise ValueError("allow_patterns 只能包含仓库内相对匹配模式")
        normalized.add(pattern)
    if not normalized:
        raise ValueError("正式模型快照必须声明非空 allow_patterns")
    return tuple(sorted(normalized))


def build_shared_hugging_face_snapshot_dir(
    repository_id: str,
    revision: str,
    *,
    cache_root: str | Path = DEFAULT_SHARED_HUGGING_FACE_SNAPSHOT_ROOT,
) -> Path:
    """构造由仓库标识和精确 revision 共同命名的共享快照目录。

    该目录结构可由多套官方参考 runner 复用。revision 位于独立目录层级,
    因此更新来源提交不会静默覆盖旧快照, 其他项目也可直接复用这一写法。
    """

    exact_repository_id = _validate_repository_id(repository_id)
    exact_revision = _validate_immutable_revision(revision)
    repository_slug = "models--" + exact_repository_id.replace("/", "--")
    return Path(cache_root).expanduser() / repository_slug / "snapshots" / exact_revision


def validate_frozen_model_source(
    repository_id: str,
    revision: str,
    *,
    expected_repository_id: str,
    expected_revision: str,
) -> None:
    """要求正式参考运行使用登记的仓库与40位不可变 revision。"""

    exact_repository_id = _validate_repository_id(repository_id)
    exact_revision = _validate_immutable_revision(revision)
    if exact_repository_id != _validate_repository_id(expected_repository_id):
        raise ValueError("正式模型仓库必须与登记源一致")
    if exact_revision != _validate_immutable_revision(expected_revision):
        raise ValueError("正式模型 revision 必须与登记提交一致")


def _file_sha256(path: Path) -> str:
    """计算单个快照文件的 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches_allow_patterns(relative_path: str, allow_patterns: tuple[str, ...]) -> bool:
    """按 Hugging Face 的 glob 语义判断文件是否属于声明组件。"""

    return any(fnmatchcase(relative_path, pattern) for pattern in allow_patterns)


def build_model_snapshot_content(
    repository_dir: str | Path,
    *,
    repository_id: str,
    revision: str,
    allow_patterns: Sequence[str],
) -> dict[str, Any]:
    """逐文件计算本地模型内容, 并绑定来源与下载组件集合。

    Hugging Face 的 ``.cache`` 目录只保存下载器元数据, 不属于模型实际输入。
    因此证据覆盖 loader 可见的全部文件, 每个文件都必须落入显式
    ``allow_patterns``。这既避免将整个大型仓库误当成运行依赖, 也能阻断
    共享目录中遗留文件悄然进入正式运行。
    """

    exact_repository_id = _validate_repository_id(repository_id)
    exact_revision = _validate_immutable_revision(revision)
    exact_allow_patterns = normalize_allow_patterns(allow_patterns)
    root = Path(repository_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"模型快照目录不存在: {root}")
    files = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".cache":
            continue
        relative_path = relative.as_posix()
        if not _matches_allow_patterns(relative_path, exact_allow_patterns):
            raise RuntimeError(f"模型快照包含 allow_patterns 外文件: {relative_path}")
        files.append(
            {
                "path": relative_path,
                "size_bytes": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
        )
    if not files:
        raise RuntimeError("模型快照目录不包含可审计文件")
    payload: dict[str, Any] = {
        "repository_id": exact_repository_id,
        "revision": exact_revision,
        "allow_patterns": list(exact_allow_patterns),
        "file_count": len(files),
        "files": files,
    }
    payload["snapshot_content_digest"] = build_stable_digest(payload)
    return payload


def validate_reusable_model_snapshot(
    repository_dir: str | Path,
    *,
    report_path: str | Path,
    repository_id: str,
    revision: str,
    allow_patterns: Sequence[str],
) -> dict[str, Any]:
    """只允许复用与既有运行报告逐文件、逐组件一致的模型快照。"""

    path = Path(report_path)
    if not path.is_file():
        raise RuntimeError("既有模型目录缺少受治理快照报告, 必须按精确 revision 重新物化")
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    expected = report.get("model_snapshot_content") if isinstance(report, dict) else None
    if not isinstance(expected, dict):
        raise RuntimeError("既有模型目录缺少文件级快照证据")
    actual = build_model_snapshot_content(
        repository_dir,
        repository_id=repository_id,
        revision=revision,
        allow_patterns=allow_patterns,
    )
    if expected != actual:
        raise RuntimeError("本地模型快照与登记 revision 或 allow_patterns 的文件级证据不一致")
    return actual


def ensure_hugging_face_snapshot_files(
    repository_dir: str | Path,
    *,
    report_path: str | Path,
    repository_id: str,
    revision: str,
    allow_patterns: Sequence[str],
    token: str | None,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """复用可信快照, 或在同一精确 revision 目录续传并修复文件。

    该函数只负责物化和校验源文件, 不写运行报告。runner 可在返回后完成
    必要的兼容变换, 再调用 ``build_model_snapshot_content`` 生成最终证据。
    因此下载失败不会覆盖上一次成功报告, 三套 runner 也可以共享同一缓存。
    """

    exact_repository_id = _validate_repository_id(repository_id)
    exact_revision = _validate_immutable_revision(revision)
    exact_allow_patterns = normalize_allow_patterns(allow_patterns)
    local_dir = Path(repository_dir).expanduser()
    report = Path(report_path)
    validation_error: str | None = None

    try:
        reusable_content = validate_reusable_model_snapshot(
            local_dir,
            report_path=report,
            repository_id=exact_repository_id,
            revision=exact_revision,
            allow_patterns=exact_allow_patterns,
        )
    except (FileNotFoundError, json.JSONDecodeError, RuntimeError) as error:
        validation_error = f"{type(error).__name__}:{error}"
    else:
        return {
            "download_requested": False,
            "download_performed": False,
            "repair_requested": False,
            "snapshot_path": str(local_dir.resolve()),
            "repository_id": exact_repository_id,
            "revision": exact_revision,
            "allow_patterns": list(exact_allow_patterns),
            "reused_model_snapshot_content": reusable_content,
        }

    if snapshot_download_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_download_fn = snapshot_download

    # 报告存在但校验失败表示已知漂移, 此时强制刷新已选组件以完成修复。
    # 报告缺失时保留下载器的断点续传能力, 但仍会按精确 revision 再次调用。
    force_download = report.is_file()
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download_fn(
        repo_id=exact_repository_id,
        revision=exact_revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token or None,
        allow_patterns=list(exact_allow_patterns),
        force_download=force_download,
    )
    downloaded_content = build_model_snapshot_content(
        local_dir,
        repository_id=exact_repository_id,
        revision=exact_revision,
        allow_patterns=exact_allow_patterns,
    )
    return {
        "download_requested": True,
        "download_performed": True,
        "repair_requested": force_download,
        "force_download": force_download,
        "snapshot_path": str(Path(snapshot_path).resolve()),
        "repository_id": exact_repository_id,
        "revision": exact_revision,
        "allow_patterns": list(exact_allow_patterns),
        "previous_snapshot_validation_error": validation_error,
        "downloaded_snapshot_content_before_transform": downloaded_content,
    }
