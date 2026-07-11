"""为需要 CPU 结果闭合来源的功能测试写出精确输入锁."""

from __future__ import annotations

import json
from pathlib import Path

from main.core.digest import build_stable_digest
from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
    LOCK_FILENAME,
    LOCK_MANIFEST_FILENAME,
    LOCK_OUTPUT_ROOT,
)


def write_test_closure_input_lock(
    root: Path,
    *,
    paper_run_name: str,
    target_fpr: float,
    common_code_version: str = "abc1234",
) -> tuple[Path, Path]:
    """写出不读取大型 ZIP,但满足下游身份复验的10类测试锁."""

    records = [
        {
            "package_family": spec.package_family,
            "package_path": (root / "drive" / f"{spec.package_family}.zip").as_posix(),
            "package_sha256": f"{index + 1:x}" * 64,
            "paper_run_name": paper_run_name,
            "target_fpr": target_fpr,
            "code_version": common_code_version,
            "generated_at": "2026-07-11T00:00:00+00:00",
        }
        for index, spec in enumerate(CLOSURE_PACKAGE_FAMILY_SPECS)
    ]
    lock_payload: dict[str, object] = {
        "paper_run_name": paper_run_name,
        "target_fpr": target_fpr,
        "common_code_version": common_code_version,
        "closure_input_package_count": len(records),
        "closure_input_packages": records,
    }
    lock_payload["closure_input_lock_digest"] = build_stable_digest(lock_payload)
    output_dir = root / LOCK_OUTPUT_ROOT / paper_run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / LOCK_FILENAME
    manifest_path = output_dir / LOCK_MANIFEST_FILENAME
    manifest = {
        "artifact_id": f"{paper_run_name}_closure_input_lock_manifest",
        "artifact_type": "local_manifest",
        "input_paths": [record["package_path"] for record in records],
        "output_paths": [
            f"{LOCK_OUTPUT_ROOT.as_posix()}/{paper_run_name}/{LOCK_FILENAME}",
            f"{LOCK_OUTPUT_ROOT.as_posix()}/{paper_run_name}/{LOCK_MANIFEST_FILENAME}",
        ],
        "config_digest": "a" * 64,
        "code_version": common_code_version,
        "rebuild_command": "test_closure_input_lock",
        "config": {},
        "metadata": {
            "closure_input_lock_ready": True,
            "closure_input_package_count": len(records),
            "closure_input_packages": records,
            "closure_input_lock_digest": lock_payload["closure_input_lock_digest"],
            "paper_run_name": paper_run_name,
            "target_fpr": target_fpr,
            "common_code_version": common_code_version,
        },
    }
    lock_path.write_text(
        json.dumps(lock_payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return lock_path, manifest_path
