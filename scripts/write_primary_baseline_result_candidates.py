"""写出主表 external baseline 共同协议候选结果记录。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.baselines import (
    build_method_faithful_baseline_candidate_records,
    build_primary_baseline_formal_import_readiness_rows,
    build_primary_baseline_formal_import_readiness_summary,
    validate_primary_baseline_formal_import_rows,
)
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest

CONSTRUCTION_UNIT_NAME = "primary_baseline_result_candidate_import"
DEFAULT_OUTPUT_DIR = Path("outputs/external_baseline_results")
DEFAULT_ATTACK_MANIFEST_PATH = Path("outputs/attack_matrix/attack_manifest.json")
DEFAULT_GPU_SMOKE_OBSERVATIONS_PATH = Path("outputs/external_baseline_gpu_smoke/execution/baseline_observations.json")
DEFAULT_T2SMARK_CANDIDATE_RECORDS_PATH = Path(
    "outputs/t2smark_full_main_reproduction/t2smark_full_main_formal_import_candidate_records.jsonl"
)
GPU_SMOKE_OBSERVATIONS_ENTRY = "outputs/external_baseline_gpu_smoke/execution/baseline_observations.json"
T2SMARK_CANDIDATE_RECORDS_ENTRY = (
    "outputs/t2smark_full_main_reproduction/t2smark_full_main_formal_import_candidate_records.jsonl"
)
METHOD_FAITHFUL_BASELINE_IDS = ("tree_ring", "gaussian_shading", "shallow_diffuse")


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: dict[str, Any]) -> str:
    """把单条记录写成稳定 JSONL 行。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """写出稳定字段顺序的 CSV 文件。"""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_path(root_path: Path, path: str | Path | None) -> Path | None:
    """把可选路径解析为绝对路径。"""

    if path is None or str(path).strip() == "":
        return None
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()


def ensure_output_dir_under_outputs(root_path: Path, output_dir: str | Path) -> Path:
    """确保候选结果记录输出目录位于 outputs/ 下。"""

    resolved = resolve_path(root_path, output_dir)
    if resolved is None:
        raise ValueError("输出目录不能为空。")
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("主表 baseline 候选结果输出目录必须位于 outputs/ 下。") from exc
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先记录相对仓库根目录的路径, 仓库外路径保留绝对路径。"""

    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_code_version(root_path: Path) -> str:
    """读取当前 Git 短提交标识, 工作区有变更时附加 dirty 标记。"""

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


def build_file_digest(path: Path) -> str:
    """对二进制证据文件生成 SHA-256 摘要。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件, 文件缺失时返回空字典。"""

    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_json_array(path: Path) -> list[dict[str, Any]]:
    """读取 JSON 数组文件, 文件缺失或内容不匹配时返回空列表。"""

    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return [dict(row) for row in payload] if isinstance(payload, list) else []


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 记录, 文件缺失时返回空列表。"""

    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def read_text_from_package(package_path: Path, entry_name: str) -> str:
    """从 zip 结果包中读取文本条目, 条目缺失时返回空字符串。"""

    if not package_path.is_file():
        return ""
    with ZipFile(package_path) as archive:
        if entry_name not in archive.namelist():
            return ""
        with archive.open(entry_name) as handle:
            return handle.read().decode("utf-8-sig")


def read_json_array_from_package(package_path: Path, entry_name: str) -> list[dict[str, Any]]:
    """从 zip 结果包中读取 JSON 数组条目。"""

    text = read_text_from_package(package_path, entry_name)
    if not text:
        return []
    payload = json.loads(text)
    return [dict(row) for row in payload] if isinstance(payload, list) else []


def read_jsonl_rows_from_package(package_path: Path, entry_name: str) -> list[dict[str, Any]]:
    """从 zip 结果包中读取 JSONL 条目。"""

    text = read_text_from_package(package_path, entry_name)
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_gpu_smoke_observations(
    *,
    observations_path: Path,
    package_path: Path | None,
) -> list[dict[str, Any]]:
    """读取方法忠实 SD3.5 adapter observation, 优先使用显式结果包。"""

    rows = read_json_array_from_package(package_path, GPU_SMOKE_OBSERVATIONS_ENTRY) if package_path else []
    if not rows:
        rows = read_json_array(observations_path)
    return rows


def load_t2smark_candidate_rows(
    *,
    candidate_records_path: Path,
    package_path: Path | None,
) -> list[dict[str, Any]]:
    """读取 T2SMark full-main 正式导入候选记录, 优先使用显式结果包。"""

    rows = read_jsonl_rows_from_package(package_path, T2SMARK_CANDIDATE_RECORDS_ENTRY) if package_path else []
    if not rows:
        rows = read_jsonl_rows(candidate_records_path)
    return rows


def build_prompt_protocol_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    """用 observation 中的 prompt id 与文本生成可追溯摘要。"""

    prompts = sorted(
        {
            (
                str(row.get("prompt_id", "")),
                str(row.get("prompt_text", "")),
            )
            for row in rows
            if row.get("prompt_id") or row.get("prompt_text")
        }
    )
    return build_stable_digest(prompts)


def group_method_observations(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """按 baseline id 拆分方法忠实 adapter observation。"""

    grouped = {baseline_id: [] for baseline_id in METHOD_FAITHFUL_BASELINE_IDS}
    for row in rows:
        baseline_id = str(row.get("baseline_id", ""))
        if baseline_id in grouped:
            grouped[baseline_id].append(dict(row))
    return grouped


def evidence_path_for_source(source_path: Path | None, fallback_path: Path, root_path: Path) -> tuple[str, ...]:
    """为候选记录构造可被 validator 解析的证据路径。"""

    if source_path and source_path.is_file():
        return (relative_or_absolute(source_path, root_path),)
    if fallback_path.is_file():
        return (relative_or_absolute(fallback_path, root_path),)
    return ()


def digest_for_source(source_path: Path | None, fallback_rows: Iterable[Mapping[str, Any]]) -> str:
    """为候选记录构造来源摘要, 优先使用实体文件摘要。"""

    if source_path and source_path.is_file():
        return build_file_digest(source_path)
    return build_stable_digest(list(fallback_rows))


def normalize_t2smark_candidate_rows(
    *,
    rows: Iterable[Mapping[str, Any]],
    source_path: Path | None,
    local_candidate_records_path: Path,
    root_path: Path,
) -> list[dict[str, Any]]:
    """把 T2SMark 候选记录绑定到当前可检查的结果包证据。"""

    row_values = [dict(row) for row in rows]
    normalized: list[dict[str, Any]] = []
    source_digest = digest_for_source(source_path, row_values)
    evidence_paths = evidence_path_for_source(source_path, local_candidate_records_path, root_path)
    for row in row_values:
        record = dict(row)
        if evidence_paths:
            record["baseline_result_source"] = evidence_paths[0]
            record["baseline_result_source_digest"] = source_digest
            record["evidence_paths"] = list(evidence_paths)
            record["formal_evidence_paths_ready"] = True
        normalized.append(record)
    return normalized


def build_method_candidate_rows(
    *,
    observations: Iterable[Mapping[str, Any]],
    source_path: Path | None,
    local_observations_path: Path,
    root_path: Path,
    target_fpr: float,
    resource_profile: str,
    full_main_prompt_protocol_ready: bool,
    fixed_fpr_baseline_calibration_ready: bool,
    attack_matrix_baseline_detection_ready: bool,
) -> list[dict[str, Any]]:
    """把方法忠实 SD3.5 adapter observation 聚合为共同协议候选记录。"""

    observation_rows = [dict(row) for row in observations]
    evidence_paths = evidence_path_for_source(source_path, local_observations_path, root_path)
    if source_path and source_path.is_file():
        baseline_result_source = relative_or_absolute(source_path, root_path)
    else:
        baseline_result_source = relative_or_absolute(local_observations_path, root_path)
    source_digest = digest_for_source(source_path if source_path and source_path.is_file() else local_observations_path, observation_rows)
    prompt_protocol_digest = build_prompt_protocol_digest(observation_rows)
    records: list[dict[str, Any]] = []
    for baseline_id, baseline_rows in group_method_observations(observation_rows).items():
        if not baseline_rows:
            continue
        records.extend(
            build_method_faithful_baseline_candidate_records(
                baseline_id=baseline_id,
                observation_rows=baseline_rows,
                target_fpr=target_fpr,
                baseline_result_source=baseline_result_source,
                baseline_result_source_digest=source_digest,
                evidence_paths=evidence_paths,
                prompt_protocol_digest=prompt_protocol_digest,
                full_main_prompt_protocol_ready=full_main_prompt_protocol_ready,
                fixed_fpr_baseline_calibration_ready=fixed_fpr_baseline_calibration_ready,
                attack_matrix_baseline_detection_ready=attack_matrix_baseline_detection_ready,
                resource_profile=resource_profile,
            )
        )
    return records


def write_primary_baseline_result_candidate_outputs(
    *,
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    attack_manifest_path: str | Path = DEFAULT_ATTACK_MANIFEST_PATH,
    gpu_smoke_observations_path: str | Path = DEFAULT_GPU_SMOKE_OBSERVATIONS_PATH,
    t2smark_candidate_records_path: str | Path = DEFAULT_T2SMARK_CANDIDATE_RECORDS_PATH,
    external_gpu_smoke_package_path: str | Path | None = None,
    t2smark_full_main_package_path: str | Path | None = None,
    method_resource_profile: str = "gpu_smoke",
    method_full_main_prompt_protocol_ready: bool = False,
    method_fixed_fpr_baseline_calibration_ready: bool = False,
    method_attack_matrix_baseline_detection_ready: bool = False,
    target_fpr_override: float | None = None,
) -> dict[str, Any]:
    """写出候选记录、候选校验报告、摘要和 manifest。"""

    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, output_dir)
    resolved_attack_manifest_path = resolve_path(root_path, attack_manifest_path)
    resolved_gpu_smoke_observations_path = resolve_path(root_path, gpu_smoke_observations_path)
    resolved_t2smark_candidate_records_path = resolve_path(root_path, t2smark_candidate_records_path)
    resolved_external_gpu_smoke_package_path = resolve_path(root_path, external_gpu_smoke_package_path)
    resolved_t2smark_full_main_package_path = resolve_path(root_path, t2smark_full_main_package_path)
    if resolved_attack_manifest_path is None or resolved_gpu_smoke_observations_path is None:
        raise ValueError("必要输入路径不能为空。")
    if resolved_t2smark_candidate_records_path is None:
        raise ValueError("T2SMark 候选记录路径不能为空。")

    attack_manifest = read_json(resolved_attack_manifest_path)
    target_fpr = (
        float(target_fpr_override)
        if target_fpr_override is not None
        else float(attack_manifest.get("evaluation_boundary", {}).get("target_fpr", 0.05))
    )
    gpu_observations = load_gpu_smoke_observations(
        observations_path=resolved_gpu_smoke_observations_path,
        package_path=resolved_external_gpu_smoke_package_path,
    )
    t2smark_rows = load_t2smark_candidate_rows(
        candidate_records_path=resolved_t2smark_candidate_records_path,
        package_path=resolved_t2smark_full_main_package_path,
    )
    method_candidate_rows = build_method_candidate_rows(
        observations=gpu_observations,
        source_path=resolved_external_gpu_smoke_package_path,
        local_observations_path=resolved_gpu_smoke_observations_path,
        root_path=root_path,
        target_fpr=target_fpr,
        resource_profile=method_resource_profile,
        full_main_prompt_protocol_ready=method_full_main_prompt_protocol_ready,
        fixed_fpr_baseline_calibration_ready=method_fixed_fpr_baseline_calibration_ready,
        attack_matrix_baseline_detection_ready=method_attack_matrix_baseline_detection_ready,
    )
    normalized_t2smark_rows = normalize_t2smark_candidate_rows(
        rows=t2smark_rows,
        source_path=resolved_t2smark_full_main_package_path,
        local_candidate_records_path=resolved_t2smark_candidate_records_path,
        root_path=root_path,
    )
    candidate_rows = method_candidate_rows + normalized_t2smark_rows
    validation_report = validate_primary_baseline_formal_import_rows(
        candidate_rows,
        evidence_root=root_path,
        target_fpr=target_fpr,
        require_existing_evidence=True,
    )
    readiness_rows = build_primary_baseline_formal_import_readiness_rows(candidate_rows, validation_report)
    readiness_summary = build_primary_baseline_formal_import_readiness_summary(readiness_rows)
    summary = {
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_fpr": target_fpr,
        "formal_import_candidate_record_count": len(candidate_rows),
        "accepted_formal_import_count": validation_report["accepted_formal_import_count"],
        "rejected_formal_import_count": validation_report["rejected_formal_import_count"],
        "formal_import_issue_count": validation_report["formal_import_issue_count"],
        "formal_import_validation_ready": validation_report["formal_import_validation_ready"],
        "formal_result_ready_count": readiness_summary["formal_result_ready_count"],
        "blocked_primary_baseline_ids": readiness_summary["blocked_primary_baseline_ids"],
        "primary_baseline_formal_ready": readiness_summary["primary_baseline_formal_ready"],
        "dominant_blocking_reasons": readiness_summary["dominant_blocking_reasons"],
        "supports_paper_claim": False,
    }

    records_path = resolved_output_dir / "baseline_result_records.jsonl"
    validation_path = resolved_output_dir / "baseline_result_candidate_validation_report.json"
    readiness_path = resolved_output_dir / "baseline_formal_import_readiness.csv"
    readiness_summary_path = resolved_output_dir / "baseline_formal_import_readiness_summary.json"
    summary_path = resolved_output_dir / "baseline_result_candidate_summary.json"
    manifest_path = resolved_output_dir / "manifest.local.json"

    records_path.write_text("".join(json_line(row) for row in candidate_rows), encoding="utf-8")
    validation_path.write_text(stable_json_text(validation_report), encoding="utf-8")
    write_csv(
        readiness_path,
        readiness_rows,
        [
            "baseline_id",
            "candidate_record_count",
            "accepted_formal_import_count",
            "rejected_formal_import_count",
            "formal_import_issue_count",
            "formal_result_ready",
            "blocking_reason_count",
            "blocking_reasons",
            "missing_resource_profile_full_main",
            "missing_full_main_prompt_protocol",
            "missing_fixed_fpr_baseline_calibration",
            "missing_attack_matrix_baseline_detection",
            "formal_evidence_paths_ready",
            "supports_paper_claim",
        ],
    )
    readiness_summary_path.write_text(stable_json_text(readiness_summary), encoding="utf-8")
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")

    input_paths = []
    for path in (
        resolved_attack_manifest_path,
        resolved_gpu_smoke_observations_path,
        resolved_t2smark_candidate_records_path,
        resolved_external_gpu_smoke_package_path,
        resolved_t2smark_full_main_package_path,
    ):
        if path and path.exists():
            input_paths.append(relative_or_absolute(path, root_path))
    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (
            records_path,
            validation_path,
            readiness_path,
            readiness_summary_path,
            summary_path,
            manifest_path,
        )
    )
    manifest = build_artifact_manifest(
        artifact_id="primary_baseline_result_candidate_import_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(input_paths),
        output_paths=output_paths,
        config={
            "candidate_record_digest": build_stable_digest(candidate_rows),
            "validation_report_digest": build_stable_digest(validation_report),
            "formal_import_readiness_digest": build_stable_digest(readiness_rows),
            "formal_import_readiness_summary_digest": build_stable_digest(readiness_summary),
            "summary_digest": build_stable_digest(summary),
            "method_resource_profile": method_resource_profile,
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_primary_baseline_result_candidates.py",
        metadata=summary,
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="写出主表 external baseline 共同协议候选结果记录。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--attack-manifest-path", default=str(DEFAULT_ATTACK_MANIFEST_PATH), help="攻击矩阵 manifest 路径。")
    parser.add_argument(
        "--gpu-smoke-observations-path",
        default=str(DEFAULT_GPU_SMOKE_OBSERVATIONS_PATH),
        help="方法忠实 SD3.5 adapter observation JSON 路径。",
    )
    parser.add_argument(
        "--t2smark-candidate-records-path",
        default=str(DEFAULT_T2SMARK_CANDIDATE_RECORDS_PATH),
        help="T2SMark full-main 候选 JSONL 路径。",
    )
    parser.add_argument("--external-gpu-smoke-package-path", default=None, help="可选 external baseline GPU 结果 zip 包。")
    parser.add_argument("--t2smark-full-main-package-path", default=None, help="可选 T2SMark full-main 结果 zip 包。")
    parser.add_argument("--method-resource-profile", default="gpu_smoke", help="方法忠实 adapter 候选记录的资源配置名称。")
    parser.add_argument("--target-fpr-override", type=float, default=None, help="可选 fixed-FPR 目标值覆盖。")
    parser.add_argument(
        "--method-full-main-prompt-protocol-ready",
        action="store_true",
        help="标记方法忠实 adapter 候选已覆盖 full-main prompt 协议。",
    )
    parser.add_argument(
        "--method-fixed-fpr-baseline-calibration-ready",
        action="store_true",
        help="标记方法忠实 adapter 候选已完成 fixed-FPR 校准。",
    )
    parser.add_argument(
        "--method-attack-matrix-baseline-detection-ready",
        action="store_true",
        help="标记方法忠实 adapter 候选已接入共同攻击矩阵检测。",
    )
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    manifest = write_primary_baseline_result_candidate_outputs(
        root=args.root,
        output_dir=args.output_dir,
        attack_manifest_path=args.attack_manifest_path,
        gpu_smoke_observations_path=args.gpu_smoke_observations_path,
        t2smark_candidate_records_path=args.t2smark_candidate_records_path,
        external_gpu_smoke_package_path=args.external_gpu_smoke_package_path,
        t2smark_full_main_package_path=args.t2smark_full_main_package_path,
        method_resource_profile=args.method_resource_profile,
        method_full_main_prompt_protocol_ready=args.method_full_main_prompt_protocol_ready,
        method_fixed_fpr_baseline_calibration_ready=args.method_fixed_fpr_baseline_calibration_ready,
        method_attack_matrix_baseline_detection_ready=args.method_attack_matrix_baseline_detection_ready,
        target_fpr_override=args.target_fpr_override,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
