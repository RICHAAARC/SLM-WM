"""写出语义掩码、风险场和安全子空间产物。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from main.methods.semantic import build_risk_field, build_semantic_route, project_mask_to_latent
from main.methods.subspace import (
    build_safe_basis_plan,
    build_trajectory_features,
    estimate_approximate_jvp,
    project_basis_by_route,
)

CONSTRUCTION_UNIT_NAME = "semantic_subspace"
DEFAULT_OUTPUT_DIR = Path("outputs/semantic_subspace")
PROMPT_RECORDS_PATH = Path("outputs/prompt_event_protocol/prompt_records.jsonl")
PROMPT_MANIFEST_PATH = Path("outputs/prompt_event_protocol/manifest.local.json")
RUNTIME_PROBE_ARCHIVE = Path("outputs/real_sd_runtime_probe_package_20260620t10451781952321z_b2be25c.zip")
INJECTION_ARCHIVE = Path("outputs/minimal_latent_injection_package_20260620t10181781950721z_b2be25c.zip")
VECTOR_WIDTH = 8


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定文本。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: Any) -> str:
    """把 JSON 兼容对象转为单行 JSONL 文本。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def resolve_code_version(root_path: Path) -> str:
    """读取 Git 提交标识, 工作区有变更时附加 dirty 标记。"""
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
    """确保持久化输出目录位于 outputs 下。"""
    resolved_output_dir = (root_path / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved_output_dir.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("语义子空间输出目录必须位于 outputs/ 下。") from exc
    return resolved_output_dir


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """将路径尽量转为相对仓库根目录的字符串。"""
    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def digest_to_unit_values(digest: str, count: int, salt: str) -> tuple[float, ...]:
    """从稳定摘要派生 [0, 1] 数值序列。"""
    values = []
    for index in range(count):
        item_digest = hashlib.sha256(f"{digest}|{salt}|{index}".encode("utf-8")).hexdigest()
        values.append(int(item_digest[:12], 16) / float(0xFFFFFFFFFFFF))
    return tuple(values)


def load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """读取 JSONL 文件。"""
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def load_latent_reference(root_path: Path) -> tuple[float, ...]:
    """从真实运行包摘要中提取轻量 latent 参考向量。"""
    archive_path = root_path / RUNTIME_PROBE_ARCHIVE
    if not archive_path.exists():
        return (0.2, -0.1, 0.4, -0.3, 0.6, -0.2, 0.1, -0.5)
    values: list[float] = []
    with zipfile.ZipFile(archive_path) as archive:
        text = archive.read("sd35_latent_trajectory_records.jsonl").decode("utf-8")
    for line in text.splitlines()[:VECTOR_WIDTH]:
        record = json.loads(line)
        value = float(record["latent_mean"]) + 0.05 * float(record["latent_std"])
        values.append(value)
    return tuple(values) if values else (0.2, -0.1, 0.4, -0.3, 0.6, -0.2, 0.1, -0.5)


def build_prompt_latent_values(prompt_record: dict[str, Any], latent_reference: tuple[float, ...]) -> tuple[float, ...]:
    """为 prompt 派生确定性 latent 摘要向量。"""
    digest = prompt_record["prompt_digest"]
    offsets = digest_to_unit_values(digest, len(latent_reference), "latent_offset")
    return tuple(value + 0.08 * (offset - 0.5) for value, offset in zip(latent_reference, offsets))


def semantic_base_for_tags(tags: tuple[str, ...]) -> float:
    """根据语义标签给出基础风险强度。"""
    weights = {
        "human": 0.85,
        "animal": 0.65,
        "vehicle": 0.60,
        "urban": 0.55,
        "object": 0.45,
        "indoor": 0.40,
        "landscape": 0.35,
        "water": 0.35,
        "nature": 0.30,
        "general": 0.50,
    }
    return max(weights.get(tag, 0.50) for tag in tags)


def build_prompt_feature_inputs(prompt_record: dict[str, Any]) -> dict[str, tuple[float, ...]]:
    """从 prompt 记录派生标准化语义输入向量。"""
    tags = tuple(prompt_record.get("semantic_tags", ("general",)))
    digest = prompt_record["prompt_digest"]
    base = semantic_base_for_tags(tags)
    semantic_noise = digest_to_unit_values(digest, VECTOR_WIDTH, "semantic")
    texture_values = digest_to_unit_values(digest, VECTOR_WIDTH, "texture")
    stability_source = digest_to_unit_values(digest, VECTOR_WIDTH, "stability")
    saliency_source = digest_to_unit_values(digest, VECTOR_WIDTH, "saliency")
    semantic_values = tuple(min(1.0, max(0.0, base * 0.7 + value * 0.3)) for value in semantic_noise)
    stability_values = tuple(min(1.0, max(0.0, 0.35 + value * 0.6)) for value in stability_source)
    saliency_values = tuple(min(1.0, max(0.0, 0.25 + value * 0.7)) for value in saliency_source)
    mask_values = tuple(min(1.0, max(0.0, 1.0 - 0.55 * sem + 0.20 * tex)) for sem, tex in zip(semantic_values, texture_values))
    return {
        "semantic_values": semantic_values,
        "texture_values": texture_values,
        "stability_values": stability_values,
        "saliency_values": saliency_values,
        "mask_values": mask_values,
    }


def build_prompt_subspace_bundle(prompt_record: dict[str, Any], latent_reference: tuple[float, ...]) -> dict[str, Any]:
    """为单条 prompt 构造语义路由与安全子空间计划。"""
    feature_inputs = build_prompt_feature_inputs(prompt_record)
    latent_values = build_prompt_latent_values(prompt_record, latent_reference)
    risk_field = build_risk_field(
        semantic_values=feature_inputs["semantic_values"],
        texture_values=feature_inputs["texture_values"],
        stability_values=feature_inputs["stability_values"],
        saliency_values=feature_inputs["saliency_values"],
    )
    latent_mask = project_mask_to_latent(
        latent_values=latent_values,
        mask_values=feature_inputs["mask_values"],
        mask_source="prompt_semantic_feature_mask",
    )
    route = build_semantic_route(
        prompt_id=prompt_record["prompt_id"],
        risk_profile=prompt_record["risk_profile"],
        risk_field=risk_field,
        latent_mask=latent_mask,
    )
    features = build_trajectory_features(latent_mask)
    jvp_estimate = estimate_approximate_jvp(features)
    semantic_basis = build_safe_basis_plan(features, jvp_estimate, risk_field, route)
    no_mask_basis = build_safe_basis_plan(
        features,
        jvp_estimate,
        risk_field,
        route,
        semantic_mask_enabled=False,
        basis_strategy="no_semantic_mask",
    )
    global_basis = build_safe_basis_plan(features, jvp_estimate, risk_field, route, basis_strategy="global_nullspace")
    diagnostic_basis = build_safe_basis_plan(features, jvp_estimate, risk_field, route, basis_strategy="diagnostic_basis")
    projection = project_basis_by_route(semantic_basis, route)
    return {
        "risk_field": risk_field,
        "latent_mask": latent_mask,
        "route": route,
        "features": features,
        "jvp_estimate": jvp_estimate,
        "semantic_basis": semantic_basis,
        "no_mask_basis": no_mask_basis,
        "global_basis": global_basis,
        "diagnostic_basis": diagnostic_basis,
        "projection": projection,
    }


def route_record(prompt_record: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    """构造语义路由 JSONL 记录。"""
    route = bundle["route"]
    return {
        "route_record_id": build_stable_digest({"prompt_id": prompt_record["prompt_id"], "route_digest": route.route_digest})[:24],
        "prompt_id": prompt_record["prompt_id"],
        "prompt_set": prompt_record["prompt_set"],
        "split": prompt_record["split"],
        "risk_profile": prompt_record["risk_profile"],
        "route_id": route.route_id,
        "route_label": route.route_label,
        "route_digest": route.route_digest,
        "risk_field_digest": bundle["risk_field"].risk_field_digest,
        "mask_source": bundle["latent_mask"].mask_source,
        "mask_source_digest": bundle["latent_mask"].mask_source_digest,
        "latent_mask_digest": bundle["latent_mask"].latent_mask_digest,
        "feature_operator_digest": bundle["features"].feature_operator_digest,
        "trajectory_feature_digest": bundle["features"].trajectory_feature_digest,
        "approximate_jvp_digest": bundle["jvp_estimate"].approximate_jvp_digest,
        "supports_paper_claim": False,
    }


def subspace_record(prompt_record: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    """构造安全子空间 JSONL 记录。"""
    semantic_basis = bundle["semantic_basis"]
    return {
        "subspace_plan_id": build_stable_digest(
            {"prompt_id": prompt_record["prompt_id"], "basis_digest": semantic_basis.basis_digest}
        )[:24],
        "prompt_id": prompt_record["prompt_id"],
        "prompt_set": prompt_record["prompt_set"],
        "split": prompt_record["split"],
        "basis_digest": semantic_basis.basis_digest,
        "basis_strategy": semantic_basis.basis_strategy,
        "semantic_mask_enabled": semantic_basis.semantic_mask_enabled,
        "selected_indices": semantic_basis.selected_indices,
        "basis_digests": {
            "semantic_safe_basis": semantic_basis.basis_digest,
            "no_semantic_mask": bundle["no_mask_basis"].basis_digest,
            "global_nullspace": bundle["global_basis"].basis_digest,
            "diagnostic_basis": bundle["diagnostic_basis"].basis_digest,
        },
        "route_projection_digest": bundle["projection"].route_projection_digest,
        "supports_paper_claim": False,
    }


def mask_report(prompt_record: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    """构造掩码投影报告记录。"""
    latent_mask = bundle["latent_mask"]
    return {
        "prompt_id": prompt_record["prompt_id"],
        "prompt_set": prompt_record["prompt_set"],
        "split": prompt_record["split"],
        "mask_source": latent_mask.mask_source,
        "source_length": latent_mask.source_length,
        "target_length": latent_mask.target_length,
        "mask_source_digest": latent_mask.mask_source_digest,
        "latent_mask_digest": latent_mask.latent_mask_digest,
        "supports_paper_claim": False,
    }


def existing_input_paths(root_path: Path) -> tuple[str, ...]:
    """登记存在的输入路径。"""
    candidates = (
        PROMPT_MANIFEST_PATH,
        PROMPT_RECORDS_PATH,
        RUNTIME_PROBE_ARCHIVE,
        INJECTION_ARCHIVE,
    )
    paths = []
    for candidate in candidates:
        path = root_path / candidate
        if path.exists():
            paths.append(relative_or_absolute(path, root_path))
    return tuple(paths)


def write_semantic_subspace_outputs(
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_records: int | None = None,
) -> dict[str, Any]:
    """写出语义子空间产物。"""
    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, Path(output_dir))
    mask_report_dir = resolved_output_dir / "mask_projection_reports"
    mask_report_dir.mkdir(parents=True, exist_ok=True)
    prompt_records = load_jsonl(root_path / PROMPT_RECORDS_PATH)
    if max_records is not None:
        prompt_records = prompt_records[:max_records]
    latent_reference = load_latent_reference(root_path)

    route_records = []
    subspace_records = []
    mask_reports = []
    basis_digest_sets: dict[str, set[str]] = {
        "semantic_safe_basis": set(),
        "no_semantic_mask": set(),
        "global_nullspace": set(),
        "diagnostic_basis": set(),
    }
    changed_basis_count = 0
    route_digests = set()

    for prompt_record in prompt_records:
        bundle = build_prompt_subspace_bundle(prompt_record, latent_reference)
        route_records.append(route_record(prompt_record, bundle))
        subspace_records.append(subspace_record(prompt_record, bundle))
        mask_reports.append(mask_report(prompt_record, bundle))
        route_digests.add(bundle["route"].route_digest)
        basis_digest_sets["semantic_safe_basis"].add(bundle["semantic_basis"].basis_digest)
        basis_digest_sets["no_semantic_mask"].add(bundle["no_mask_basis"].basis_digest)
        basis_digest_sets["global_nullspace"].add(bundle["global_basis"].basis_digest)
        basis_digest_sets["diagnostic_basis"].add(bundle["diagnostic_basis"].basis_digest)
        changed_basis_count += int(bundle["semantic_basis"].basis_digest != bundle["no_mask_basis"].basis_digest)

    route_records_path = resolved_output_dir / "semantic_route_records.jsonl"
    subspace_records_path = resolved_output_dir / "subspace_plan_records.jsonl"
    mask_reports_path = mask_report_dir / "mask_projection_reports.jsonl"
    basis_digests_path = resolved_output_dir / "basis_digests.json"
    summary_path = resolved_output_dir / "semantic_subspace_summary.json"
    manifest_path = resolved_output_dir / "manifest.local.json"

    route_records_path.write_text("".join(json_line(record) for record in route_records), encoding="utf-8")
    subspace_records_path.write_text("".join(json_line(record) for record in subspace_records), encoding="utf-8")
    mask_reports_path.write_text("".join(json_line(record) for record in mask_reports), encoding="utf-8")

    basis_digests = {
        name: {
            "unique_digest_count": len(digests),
            "sample_digests": sorted(digests)[:20],
        }
        for name, digests in sorted(basis_digest_sets.items())
    }
    summary = {
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "semantic_route_record_count": len(route_records),
        "subspace_plan_record_count": len(subspace_records),
        "mask_projection_report_count": len(mask_reports),
        "unique_route_digest_count": len(route_digests),
        "semantic_mask_changed_basis_count": changed_basis_count,
        "basis_strategies": sorted(basis_digest_sets),
        "protocol_decision": "pass" if route_records and changed_basis_count > 0 else "fail",
        "supports_paper_claim": False,
    }
    basis_digests_path.write_text(stable_json_text(basis_digests), encoding="utf-8")
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")

    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (
            route_records_path,
            subspace_records_path,
            mask_reports_path,
            basis_digests_path,
            summary_path,
            manifest_path,
        )
    )
    manifest = build_artifact_manifest(
        artifact_id="semantic_subspace_manifest",
        artifact_type="local_manifest",
        input_paths=existing_input_paths(root_path),
        output_paths=output_paths,
        config={
            "construction_unit_name": CONSTRUCTION_UNIT_NAME,
            "basis_digests_digest": build_stable_digest(basis_digests),
            "summary_digest": build_stable_digest(summary),
            "semantic_route_record_count": summary["semantic_route_record_count"],
            "subspace_plan_record_count": summary["subspace_plan_record_count"],
            "mask_projection_report_count": summary["mask_projection_report_count"],
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_semantic_subspace_outputs.py",
        metadata={
            "construction_unit_name": CONSTRUCTION_UNIT_NAME,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol_decision": summary["protocol_decision"],
            "supports_paper_claim": False,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="写出语义子空间产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--max-records", type=int, default=None, help="调试时限制处理记录数量。")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    manifest = write_semantic_subspace_outputs(
        root=args.root,
        output_dir=args.output_dir,
        max_records=args.max_records,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
