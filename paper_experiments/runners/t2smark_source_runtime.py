"""准备固定 revision 的 T2SMark 源码与受审计正式协议补丁。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from paper_experiments.runners.external_source_runtime import (
    load_baseline_registry_item,
    normalize_repository_url,
    run_command_with_progress_status,
)


DEFAULT_T2SMARK_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
DEFAULT_T2SMARK_SOURCE_ENTRY = "external_baseline/primary/t2smark/source/run_sd35.py"
DEFAULT_T2SMARK_PROTOCOL_PATCH = "external_baseline/primary/t2smark/adapter/formal_protocol_git_diff.txt"
T2SMARK_PATCHED_SOURCE_PATHS = ("option.py", "run_sd35.py")


def configured_attack_names(value: str) -> tuple[str, ...]:
    """解析逗号分隔的正式攻击名称。"""

    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def count_t2smark_formal_attack_items(results_path: Path, attack_names: tuple[str, ...]) -> int:
    """统计同时具备 clean/positive 分数与完整攻击对的样本数。"""

    if not attack_names or not results_path.is_file():
        return 0
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(payload, dict):
        return 0
    count = 0
    for key, value in payload.items():
        if not str(key).isdigit() or not isinstance(value, dict):
            continue
        detection = value.get("image_only_detection")
        attacks = value.get("formal_attacks")
        detection_ready = isinstance(detection, dict) and all(
            field_name in detection for field_name in ("clean_score", "watermarked_score")
        )
        attacks_ready = isinstance(attacks, dict) and all(
            isinstance(attacks.get(name), dict)
            and isinstance(attacks[name].get("attacked_negative"), dict)
            and isinstance(attacks[name].get("attacked_positive"), dict)
            and "detection_score" in attacks[name]["attacked_negative"]
            and "detection_score" in attacks[name]["attacked_positive"]
            for name in attack_names
        )
        if detection_ready and attacks_ready:
            count += 1
    return count


def _sha256(path: Path) -> str:
    """计算协议补丁摘要。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    progress: object | None,
    profile: str,
) -> dict[str, Any]:
    """执行受治理 Git 命令并返回诊断。"""

    return run_command_with_progress_status(
        command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        progress=progress,
        progress_profile=profile,
    )


def _read_head_commit(source_dir: Path) -> str:
    """读取源码缓存当前提交。"""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _remove_generated_bytecode(source_dir: Path) -> None:
    """移除 Python 执行产生的缓存, 避免把运行副产物误判为源码改动。

    该清理只处理固定源码缓存内部的 `__pycache__` 与字节码文件。其他未跟踪文件
    仍会被精确工作树审计拒绝, 因而不能借助清理逻辑掩盖额外源码或配置。
    """

    resolved_root = source_dir.resolve()
    cache_dirs = sorted(
        (path for path in source_dir.rglob("__pycache__") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for cache_dir in cache_dirs:
        resolved = cache_dir.resolve()
        resolved.relative_to(resolved_root)
        shutil.rmtree(resolved)
    for bytecode_path in tuple(source_dir.rglob("*.pyc")) + tuple(source_dir.rglob("*.pyo")):
        resolved = bytecode_path.resolve()
        resolved.relative_to(resolved_root)
        if resolved.is_file():
            resolved.unlink()


def _run_git_checked(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """执行只用于源码精确性核验的 Git 命令。"""

    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def verify_exact_t2smark_protocol_worktree(source_dir: Path, patch_path: Path) -> dict[str, Any]:
    """要求源码工作树严格等于固定 revision 加固定协议补丁。

    此处使用临时 Git index 从 `HEAD` 重建“只应用固定补丁”的期望状态, 再让 Git
    比较真实工作树与该期望状态。相比只检查补丁能否反向应用, 这一实现还能拒绝
    补丁文件之外的 tracked 修改、额外未跟踪文件和 Git 暂存区修改。
    """

    resolved_source = source_dir.resolve()
    resolved_patch = patch_path.resolve()
    _remove_generated_bytecode(resolved_source)

    index_delta = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "HEAD", "--"],
        cwd=resolved_source,
        check=False,
    )
    if index_delta.returncode != 0:
        raise RuntimeError("T2SMark 源码 index 含有暂存区修改, 不等于固定协议工作树")

    untracked_result = _run_git_checked(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=resolved_source,
    )
    untracked_paths = tuple(line.strip() for line in untracked_result.stdout.splitlines() if line.strip())
    if untracked_paths:
        raise RuntimeError(
            "T2SMark 源码包含固定补丁之外的未跟踪文件: " + ",".join(untracked_paths)
        )

    with tempfile.TemporaryDirectory(prefix="slm_wm_t2smark_index_") as temporary_dir:
        index_path = Path(temporary_dir) / "expected.index"
        environment = dict(os.environ)
        environment["GIT_INDEX_FILE"] = str(index_path)
        _run_git_checked(["git", "read-tree", "HEAD"], cwd=resolved_source, env=environment)
        _run_git_checked(
            ["git", "apply", "--cached", str(resolved_patch)],
            cwd=resolved_source,
            env=environment,
        )
        comparison = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=resolved_source,
            env=environment,
            check=False,
        )
        if comparison.returncode != 0:
            raise RuntimeError("T2SMark 源码工作树不等于固定 revision 加固定协议补丁")

    source_hashes = {
        relative_path: _sha256(resolved_source / relative_path)
        for relative_path in T2SMARK_PATCHED_SOURCE_PATHS
    }
    worktree_digest = hashlib.sha256(
        json.dumps(
            {
                "protocol_patch_sha256": _sha256(resolved_patch),
                "patched_source_sha256": source_hashes,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "source_worktree_exact": True,
        "source_worktree_digest": worktree_digest,
        "patched_source_sha256": source_hashes,
    }


def _apply_protocol_patch(
    source_dir: Path,
    patch_path: Path,
    *,
    progress: object | None,
) -> dict[str, Any]:
    """应用固定补丁，已应用时通过 reverse-check 精确识别。"""

    check_result = _run_git(
        ["git", "apply", "--check", str(patch_path)],
        cwd=source_dir,
        timeout_seconds=300,
        progress=progress,
        profile="operation=t2smark_protocol_patch_check",
    )
    patch_applied = False
    if int(check_result["return_code"]) == 0:
        apply_result = _run_git(
            ["git", "apply", str(patch_path)],
            cwd=source_dir,
            timeout_seconds=300,
            progress=progress,
            profile="operation=t2smark_protocol_patch_apply",
        )
        if int(apply_result["return_code"]) != 0:
            raise RuntimeError("T2SMark 正式协议补丁应用失败")
        patch_applied = True
    else:
        reverse_result = _run_git(
            ["git", "apply", "--reverse", "--check", str(patch_path)],
            cwd=source_dir,
            timeout_seconds=300,
            progress=progress,
            profile="operation=t2smark_protocol_patch_reverse_check",
        )
        if int(reverse_result["return_code"]) != 0:
            raise RuntimeError("T2SMark 源码既不匹配固定 revision，也不匹配已应用的正式补丁")
    return {
        "protocol_patch_path": patch_path.as_posix(),
        "protocol_patch_sha256": _sha256(patch_path),
        "protocol_patch_applied": patch_applied,
        "protocol_patch_ready": True,
    }


def _verify_formal_source(source_entry: Path) -> None:
    """核验补丁后的入口确实包含仅图像检测、成对质量和完整攻击输出。"""

    option_path = source_entry.with_name("option.py")
    if not source_entry.is_file() or not option_path.is_file():
        raise FileNotFoundError("T2SMark 正式源码入口不完整")
    source_text = source_entry.read_text(encoding="utf-8")
    option_text = option_path.read_text(encoding="utf-8")
    required_source_tokens = (
        "image_only_detection",
        "formal_attacks",
        "attacked_negative",
        "attacked_positive",
        "score_image_with_master_key",
        "strict_clean_watermarked_pair",
        "attack_execution",
    )
    required_option_tokens = (
        "slm_attack_families",
        "slm_attack_image_dir",
        "slm_save_clean_pair",
        "slm_pair_image_dir",
    )
    if any(token not in source_text for token in required_source_tokens):
        raise RuntimeError("T2SMark 正式源码缺少仅图像检测或攻击证据算子")
    if any(token not in option_text for token in required_option_tokens):
        raise RuntimeError("T2SMark 正式源码缺少受治理协议参数")


def ensure_t2smark_source_available(
    root_path: Path,
    paths: dict[str, Path],
    timeout_seconds: int,
    progress: object | None = None,
) -> dict[str, Any]:
    """克隆固定 revision、应用固定补丁并核验正式 T2SMark 源码。"""

    registry_item = load_baseline_registry_item(root_path, "t2smark")
    source_dir = root_path / str(registry_item["source_dir"])
    source_entry = root_path / DEFAULT_T2SMARK_SOURCE_ENTRY
    patch_path = root_path / DEFAULT_T2SMARK_PROTOCOL_PATCH
    if not patch_path.is_file():
        raise FileNotFoundError(f"T2SMark 正式协议补丁不存在: {patch_path}")
    repository_url = normalize_repository_url(str(registry_item["official_repository_url"]))
    expected_commit = str(registry_item["official_repository_commit"])
    clone_result: dict[str, Any] = {
        "command": [],
        "return_code": 0,
        "stdout": "",
        "stderr": "",
    }
    if not source_dir.exists():
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        clone_result = _run_git(
            ["git", "clone", repository_url, str(source_dir)],
            cwd=root_path,
            timeout_seconds=timeout_seconds,
            progress=progress,
            profile="operation=t2smark_source_clone",
        )
        if int(clone_result["return_code"]) != 0:
            raise FileNotFoundError("T2SMark 固定源码克隆失败")
    if not (source_dir / ".git").is_dir():
        raise RuntimeError("T2SMark 源码目录不是可审计 Git 仓库")
    current_commit = _read_head_commit(source_dir)
    if current_commit != expected_commit:
        checkout_result = _run_git(
            ["git", "checkout", expected_commit],
            cwd=source_dir,
            timeout_seconds=300,
            progress=progress,
            profile="operation=t2smark_source_checkout",
        )
        if int(checkout_result["return_code"]) != 0:
            raise RuntimeError("T2SMark 源码无法切换到登记 revision")
        current_commit = _read_head_commit(source_dir)
    if current_commit != expected_commit:
        raise RuntimeError("T2SMark 源码 revision 与登记表不一致")
    patch_report = _apply_protocol_patch(source_dir, patch_path, progress=progress)
    worktree_report = verify_exact_t2smark_protocol_worktree(source_dir, patch_path)
    _verify_formal_source(source_entry)
    source_report = {
        "source_available": True,
        "source_downloaded": bool(clone_result["command"]),
        "source_entry_path": source_entry.relative_to(root_path).as_posix(),
        "source_dir": source_dir.relative_to(root_path).as_posix(),
        "official_repository_url": repository_url,
        "official_repository_commit": expected_commit,
        "source_revision_ready": True,
        **patch_report,
        **worktree_report,
    }
    write_path = paths["output_dir"] / "t2smark_source_prepare_result.json"
    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text(
        json.dumps(
            {"source_report": source_report, "clone_result": clone_result},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return source_report
