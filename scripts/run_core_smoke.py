"""运行 stage01 纯算法原语 smoke 并写出本地阶段产物。

该脚本属于外层脚本, 负责把 `main/methods/` 中的 typed object 转换为本地
synthetic records、summary 和 manifest。`main/` 自身不写出 records。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from main.core.records import ExperimentRecord
from main.methods.algorithm_primitives import (
    build_semantic_risk_field,
    compose_latent_update,
    compute_content_score,
    decide_evidence_and_final,
    derive_attention_carrier_stub,
    derive_hf_carrier,
    derive_lf_carrier,
    estimate_safe_basis,
    evaluate_geometry_reliability,
    project_latent_mask,
)


STAGE_NAME = "stage_01_algorithm_primitives"
DEFAULT_OUTPUT_DIR = Path("outputs/local_stage01_core_primitives")


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定、可读的 UTF-8 文本。"""
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def json_line(value: Any) -> str:
    """把 JSON 兼容对象转为单行 JSONL 文本。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def resolve_code_version(root_path: Path) -> str:
    """读取 Git 提交标识, 若工作区有变更则追加 dirty 标记。"""
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--short"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "git_version_unavailable"
    commit_id = commit_result.stdout.strip()
    if not commit_id:
        return "git_version_unavailable"
    return f"{commit_id}-dirty" if status_result.stdout.strip() else commit_id


def ensure_output_dir_under_outputs(root_path: Path, output_dir: Path) -> Path:
    """确保输出目录位于 `outputs/` 下。"""
    resolved_output_dir = (root_path / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved_output_dir.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("stage01 输出目录必须位于 outputs/ 下") from exc
    return resolved_output_dir


def build_synthetic_primitive_bundle() -> dict[str, Any]:
    """构造 stage01 纯算法原语 synthetic 闭环。"""
    latent_values = (0.2, -0.1, 0.4, -0.3, 0.6, -0.2, 0.1, -0.5)
    risk_field = build_semantic_risk_field(
        semantic_values=(0.2, 0.3, 0.4, 0.5, 0.7, 0.6, 0.2, 0.3),
        texture_values=(0.2, 0.3, 0.7, 0.8, 0.9, 0.6, 0.2, 0.7),
        stability_values=(0.8, 0.7, 0.9, 0.6, 0.8, 0.5, 0.7, 0.9),
        saliency_values=(0.2, 0.2, 0.3, 0.3, 0.6, 0.5, 0.2, 0.4),
        attention_stability_values=(0.7, 0.6, 0.8, 0.5, 0.9, 0.5, 0.7, 0.8),
    )
    projection = project_latent_mask(latent_values, mask_values=(1.0, 0.8, 0.9, 0.7))
    safe_basis = estimate_safe_basis(latent_values, projection, risk_field, basis_rank=4)

    event_digest = build_stable_digest({"stage_name": STAGE_NAME, "event_name": "synthetic_unit_event"})
    lf_carrier = derive_lf_carrier(safe_basis, key="correct_key", event_digest=event_digest)
    hf_carrier = derive_hf_carrier(safe_basis, risk_field, key="correct_key", event_digest=event_digest)
    attention_carrier = derive_attention_carrier_stub(safe_basis, key="correct_key", event_digest=event_digest)
    composed_update = compose_latent_update(lf_carrier, hf_carrier, attention_carrier)

    correct_score = compute_content_score(composed_update.combined_update_values, lf_carrier, hf_carrier)
    wrong_lf_carrier = derive_lf_carrier(safe_basis, key="wrong_key", event_digest=event_digest)
    wrong_hf_carrier = derive_hf_carrier(safe_basis, risk_field, key="wrong_key", event_digest=event_digest)
    wrong_score = compute_content_score(composed_update.combined_update_values, wrong_lf_carrier, wrong_hf_carrier)

    full_hf_carrier = derive_hf_carrier(
        safe_basis,
        risk_field,
        key="correct_key",
        event_digest=event_digest,
        tail_fraction=1.0,
    )
    truncated_hf_score = compute_content_score(hf_carrier.update_values, lf_carrier, hf_carrier)
    full_hf_score = compute_content_score(hf_carrier.update_values, lf_carrier, full_hf_carrier)
    hf_tail_truncation_delta = abs(truncated_hf_score.hf_score - full_hf_score.hf_score)

    geometry = evaluate_geometry_reliability(
        registration_confidence=0.9,
        anchor_inlier_ratio=0.8,
        recovered_sync_consistency=0.85,
        alignment_residual=0.1,
    )
    rescued_decision = decide_evidence_and_final(
        raw_content_score=0.48,
        aligned_content_score=0.53,
        content_threshold=0.50,
        geometry=geometry,
        fail_reason="geometry_suspected",
        attestation_pass=True,
        rescue_margin_low=-0.05,
    )
    unattested_decision = decide_evidence_and_final(
        raw_content_score=0.55,
        aligned_content_score=0.55,
        content_threshold=0.50,
        geometry=geometry,
        fail_reason="none",
        attestation_pass=False,
    )

    return {
        "risk_field": risk_field,
        "projection": projection,
        "safe_basis": safe_basis,
        "lf_carrier": lf_carrier,
        "hf_carrier": hf_carrier,
        "attention_carrier": attention_carrier,
        "composed_update": composed_update,
        "correct_score": correct_score,
        "wrong_score": wrong_score,
        "geometry": geometry,
        "rescued_decision": rescued_decision,
        "unattested_decision": unattested_decision,
        "hf_tail_truncation_delta": hf_tail_truncation_delta,
    }


def build_records(bundle: dict[str, Any]) -> list[ExperimentRecord]:
    """把 synthetic 原语闭环结果转换为轻量 records。"""
    run_id = "stage01_synthetic_core_primitives"
    records = [
        ExperimentRecord(
            record_id="stage01_correct_key_content_score",
            run_id=run_id,
            split="synthetic",
            method_name="slm_wm_algorithm_primitives",
            metric_name="correct_key_score",
            metric_value=bundle["correct_score"].content_score,
            metadata={"stage_name": STAGE_NAME},
        ),
        ExperimentRecord(
            record_id="stage01_wrong_key_content_score",
            run_id=run_id,
            split="synthetic",
            method_name="slm_wm_algorithm_primitives",
            metric_name="wrong_key_score",
            metric_value=bundle["wrong_score"].content_score,
            metadata={"stage_name": STAGE_NAME},
        ),
        ExperimentRecord(
            record_id="stage01_hf_tail_truncation_delta",
            run_id=run_id,
            split="synthetic",
            method_name="slm_wm_algorithm_primitives",
            metric_name="hf_tail_truncation_delta",
            metric_value=bundle["hf_tail_truncation_delta"],
            metadata={"stage_name": STAGE_NAME},
        ),
        ExperimentRecord(
            record_id="stage01_rescue_applied",
            run_id=run_id,
            split="synthetic",
            method_name="slm_wm_algorithm_primitives",
            metric_name="rescue_applied",
            metric_value=1.0 if bundle["rescued_decision"].rescue_applied else 0.0,
            metadata={"stage_name": STAGE_NAME},
        ),
        ExperimentRecord(
            record_id="stage01_attestation_layering",
            run_id=run_id,
            split="synthetic",
            method_name="slm_wm_algorithm_primitives",
            metric_name="attestation_layering_pass",
            metric_value=(
                1.0
                if bundle["unattested_decision"].evidence_level and not bundle["unattested_decision"].final_level
                else 0.0
            ),
            metadata={"stage_name": STAGE_NAME},
        ),
    ]
    return records


def build_summary(bundle: dict[str, Any], records: list[ExperimentRecord]) -> dict[str, Any]:
    """构造 stage01 本地 summary。"""
    correct_score = bundle["correct_score"].content_score
    wrong_score = bundle["wrong_score"].content_score
    summary = {
        "stage_name": STAGE_NAME,
        "artifact_id": "stage01_core_primitive_summary",
        "artifact_type": "local_stage_summary",
        "decision": "pass"
        if correct_score > wrong_score
        and bundle["hf_tail_truncation_delta"] > 0.0
        and bundle["geometry"].direct_positive_decision is False
        and bundle["rescued_decision"].rescue_applied
        and bundle["unattested_decision"].evidence_level
        and not bundle["unattested_decision"].final_level
        else "fail",
        "primitive_status": {
            "semantic_risk_field": "implemented",
            "latent_mask_projection": "implemented",
            "safe_basis_estimate": "implemented",
            "lf_carrier": "implemented",
            "hf_carrier": "implemented",
            "attention_carrier_stub": "synthetic_stub",
            "latent_update_composition": "implemented",
            "content_score": "implemented",
            "geometry_reliability": "implemented",
            "evidence_and_final_decision": "implemented",
        },
        "metrics": {
            "correct_key_score": correct_score,
            "wrong_key_score": wrong_score,
            "hf_tail_truncation_delta": bundle["hf_tail_truncation_delta"],
            "rescue_applied": bundle["rescued_decision"].rescue_applied,
            "attestation_layering_pass": (
                bundle["unattested_decision"].evidence_level and not bundle["unattested_decision"].final_level
            ),
        },
        "record_count": len(records),
        "metadata": {
            "attention_runtime": "not_connected",
            "records_are_synthetic": True,
        },
    }
    return summary


def write_stage01_outputs(root: str | Path, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """写出 stage01 summary、synthetic records 和 manifest。"""
    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, Path(output_dir))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_synthetic_primitive_bundle()
    records = build_records(bundle)
    summary = build_summary(bundle, records)

    summary_path = resolved_output_dir / "core_primitive_summary.json"
    records_path = resolved_output_dir / "synthetic_core_records.jsonl"
    manifest_path = resolved_output_dir / "manifest.local.json"
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")
    records_path.write_text("".join(json_line(record.to_dict()) for record in records), encoding="utf-8")

    output_paths = tuple(
        path.relative_to(root_path).as_posix() for path in (summary_path, records_path, manifest_path)
    )
    manifest = build_artifact_manifest(
        artifact_id="stage01_core_primitives_manifest",
        artifact_type="local_stage_manifest",
        input_paths=(
            "outputs/local_stage00_core_boundary/manifest.local.json",
            "docs/builds/phases/stage_01_algorithm_primitives.md",
            "docs/field_registry.md",
            "docs/phase_status.md",
            "main/methods/algorithm_primitives.py",
            "scripts/run_core_smoke.py",
            "tests/functional/test_algorithm_primitives.py",
        ),
        output_paths=output_paths,
        config={
            "stage_name": STAGE_NAME,
            "summary_digest": build_stable_digest(summary),
            "record_count": len(records),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/run_core_smoke.py",
        metadata={
            "stage_name": STAGE_NAME,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "decision": summary["decision"],
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="运行 stage01 纯算法原语 smoke。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="阶段输出目录, 必须位于 outputs/ 下。",
    )
    return parser


def main() -> None:
    """命令行入口。"""
    parser = build_parser()
    args = parser.parse_args()
    manifest = write_stage01_outputs(args.root, args.output_dir)
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
