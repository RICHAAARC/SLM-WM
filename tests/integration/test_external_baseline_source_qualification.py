"""验证依赖真实外部源码快照的 baseline 资格化边界."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess

import pytest

from main.core.digest import build_stable_digest
from paper_experiments.baselines.method_faithful_numerical_fidelity import (
    METHOD_FAITHFUL_NUMERICAL_FIDELITY_BASELINE_IDS,
    MethodFaithfulNumericalFidelityError,
    build_method_faithful_numerical_fidelity_report,
    validate_method_faithful_numerical_fidelity_report,
)
from paper_experiments.runners.t2smark_source_runtime import (
    _verify_formal_source,
    verify_exact_t2smark_protocol_worktree,
)


pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "baseline_id",
    METHOD_FAITHFUL_NUMERICAL_FIDELITY_BASELINE_IDS,
)
def test_registered_official_operators_match_method_faithful_adapter(
    baseline_id: str,
) -> None:
    """登记 commit 的官方算子必须在同一确定性 Tensor 上匹配适配器."""

    report = build_method_faithful_numerical_fidelity_report(ROOT, baseline_id)
    validated = validate_method_faithful_numerical_fidelity_report(
        report,
        expected_baseline_id=baseline_id,
    )

    assert validated["baseline_id"] == baseline_id
    assert validated["official_source_read_mode"] == "immutable_git_commit_blob"
    assert validated["method_faithful_numerical_fidelity_ready"] is True
    assert validated["operator_record_count"] == len(validated["operator_records"])
    assert all(
        record["numerical_fidelity_ready"] is True
        and record["max_absolute_error"] <= record["absolute_tolerance"]
        for record in validated["operator_records"]
    )


def test_numerical_fidelity_report_rejects_modified_operator_value() -> None:
    """任一比较值或门禁字段被修改后不得继续进入正式证据链."""

    report = build_method_faithful_numerical_fidelity_report(ROOT, "tree_ring")
    modified = deepcopy(report)
    modified["operator_records"][0]["max_absolute_error"] = 1.0

    with pytest.raises(
        MethodFaithfulNumericalFidelityError,
        match="无法通过独立复验",
    ):
        validate_method_faithful_numerical_fidelity_report(
            modified,
            expected_baseline_id="tree_ring",
        )


def test_numerical_fidelity_rebuild_rejects_self_consistent_false_claim() -> None:
    """即使同步重算全部摘要, 超过容差的数值也不得伪装成 ready."""

    report = build_method_faithful_numerical_fidelity_report(ROOT, "tree_ring")
    modified = deepcopy(report)
    record = modified["operator_records"][1]
    record["max_absolute_error"] = 0.5
    record["numerical_fidelity_ready"] = True
    record["comparison_record_digest"] = build_stable_digest(
        {
            key: value
            for key, value in record.items()
            if key != "comparison_record_digest"
        }
    )
    modified["operator_records_digest"] = build_stable_digest(
        modified["operator_records"]
    )
    modified["numerical_fidelity_report_digest"] = build_stable_digest(
        {
            key: value
            for key, value in modified.items()
            if key != "numerical_fidelity_report_digest"
        }
    )

    with pytest.raises(
        MethodFaithfulNumericalFidelityError,
        match="无法通过独立复验",
    ):
        validate_method_faithful_numerical_fidelity_report(
            modified,
            expected_baseline_id="tree_ring",
        )


def test_t2smark_fixed_patch_applies_to_registered_source_snapshot(
    tmp_path: Path,
) -> None:
    """固定补丁必须可应用到登记源码快照并形成精确正式工作树."""

    registered_source = ROOT / "external_baseline/primary/t2smark/source"
    patch_path = (
        ROOT
        / "external_baseline/primary/t2smark/adapter/formal_protocol_git_diff.txt"
    )
    clean_source = tmp_path / "source"
    clean_source.mkdir()
    subprocess.run(["git", "init"], cwd=clean_source, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=clean_source,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=clean_source,
        check=True,
    )
    patched_paths = ("option.py", "run_sd35.py", "src/t2s.py")
    for relative_path in patched_paths:
        content = subprocess.check_output(
            ["git", "show", f"HEAD:{relative_path}"],
            cwd=registered_source,
        )
        target_path = clean_source / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(content)
    subprocess.run(
        ["git", "add", *patched_paths],
        cwd=clean_source,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "base"],
        cwd=clean_source,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "apply", "--unidiff-zero", "--check", str(patch_path)],
        cwd=clean_source,
        check=True,
    )
    subprocess.run(
        ["git", "apply", "--unidiff-zero", str(patch_path)],
        cwd=clean_source,
        check=True,
    )

    report = verify_exact_t2smark_protocol_worktree(clean_source, patch_path)
    _verify_formal_source(clean_source / "run_sd35.py")
    assert report["source_worktree_exact"] is True

    source_entry = clean_source / "run_sd35.py"
    source_text = source_entry.read_text(encoding="utf-8")
    source_entry.write_text(
        source_text.replace(
            '"generation_seed_random": int(',
            '"generation_seed_random_missing": int(',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="严格配对记录缺少正式随机化身份字段"):
        _verify_formal_source(source_entry)
