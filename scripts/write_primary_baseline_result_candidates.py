"""写出主表 external baseline 共同协议候选结果记录。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.baselines import (
    build_method_faithful_baseline_candidate_records,
    build_primary_baseline_formal_import_readiness_rows,
    build_primary_baseline_formal_import_readiness_summary,
    build_primary_baseline_method_threshold_digest_map,
    validate_primary_baseline_formal_import_rows,
)
from paper_experiments.baselines.method_faithful_observation_collection import (
    DEFAULT_METHOD_FAITHFUL_COLLECTION_ROOT as METHOD_FAITHFUL_COLLECTION_ROOT,
    MethodFaithfulObservationSource,
    load_method_faithful_observation_collection,
)
from experiments.protocol.attacks import default_attack_configs
from experiments.protocol.fixed_fpr_observation_audit import audit_fixed_fpr_observation_threshold
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.protocol.prompts import (
    build_prompt_records,
    normalize_prompt_text,
    read_prompt_file,
)
from experiments.protocol.splits import apply_split_assignments, build_group_split_counts
from experiments.runtime.repository_environment import resolve_code_version
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest

CONSTRUCTION_UNIT_NAME = "primary_baseline_result_candidate_import"
DEFAULT_OUTPUT_ROOT = Path("outputs/external_baseline_results")
DEFAULT_ATTACK_MATRIX_ROOT = Path("outputs/attack_matrix")
DEFAULT_T2SMARK_FORMAL_ROOT = Path("outputs/t2smark_formal_reproduction")
T2SMARK_CANDIDATE_RECORDS_NAME = "t2smark_formal_import_candidate_records.jsonl"


def t2smark_candidate_records_entry(paper_run_name: str) -> str:
    """返回当前论文运行在 T2SMark 结果包中的候选记录成员路径。"""

    return (
        DEFAULT_T2SMARK_FORMAL_ROOT / paper_run_name / T2SMARK_CANDIDATE_RECORDS_NAME
    ).as_posix()


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


def load_t2smark_candidate_rows(
    *,
    candidate_records_path: Path,
    package_path: Path | None,
    package_entry_name: str,
) -> list[dict[str, Any]]:
    """从唯一显式来源读取 T2SMark formal 正式导入候选记录。"""

    if package_path is not None:
        if not package_path.is_file():
            raise FileNotFoundError(f"T2SMark 正式结果包不存在: {package_path.as_posix()}")
        rows = read_jsonl_rows_from_package(package_path, package_entry_name)
        source_description = f"{package_path.as_posix()}::{package_entry_name}"
    else:
        if not candidate_records_path.is_file():
            raise FileNotFoundError(f"T2SMark 正式候选记录不存在: {candidate_records_path.as_posix()}")
        rows = read_jsonl_rows(candidate_records_path)
        source_description = candidate_records_path.as_posix()
    if not rows:
        raise ValueError(f"T2SMark 正式候选记录为空: {source_description}")
    return rows


def load_canonical_prompt_protocol(
    root_path: Path,
) -> tuple[dict[str, dict[str, str]], str]:
    """读取当前论文层级的受治理 Prompt 身份、split 和统一摘要。

    该函数属于通用证据绑定写法: 正式 baseline 必须复用仓库冻结的 Prompt，
    不能只提交数量相同但文本或 split 不同的自定义集合。
    """

    paper_run = build_paper_run_config(root_path)
    configured_path = Path(paper_run.prompt_file)
    requested_path = configured_path if configured_path.is_absolute() else (root_path / configured_path).resolve()
    repository_path = configured_path if configured_path.is_absolute() else (ROOT / configured_path).resolve()
    prompt_path = requested_path if requested_path.is_file() else repository_path
    records = apply_split_assignments(
        build_prompt_records(paper_run.prompt_set, read_prompt_file(prompt_path))
    )
    if len(records) != paper_run.prompt_count:
        raise ValueError("受治理 Prompt 记录数量与当前论文运行配置不一致")
    identities = {
        record.prompt_id: {
            "prompt_text": record.prompt_text,
            "prompt_digest": record.prompt_digest,
            "split": record.split,
        }
        for record in records
    }
    protocol_digest = build_stable_digest([record.prompt_digest for record in records])
    return identities, protocol_digest


def build_attack_resource_profile_lookup() -> dict[tuple[str, str], str]:
    """从默认攻击矩阵生成攻击到资源档位的映射, 并优先选择正式主档位。"""

    priority = {"full_main": 0, "full_extra": 1, "probe": 2}
    lookup: dict[tuple[str, str], str] = {("clean", "clean_none"): "full_main"}
    for config in default_attack_configs():
        key = (config.attack_family, config.attack_name)
        current = lookup.get(key)
        if current is None or priority[config.resource_profile] < priority[current]:
            lookup[key] = config.resource_profile
    return lookup


def allowed_resource_profiles_from_attack_lookup(lookup: Mapping[tuple[str, str], str]) -> tuple[str, ...]:
    """返回候选导入校验应接受的资源档位集合。"""

    return tuple(sorted(set(lookup.values())))


def evidence_path_for_source(source_path: Path, root_path: Path) -> tuple[str, ...]:
    """为候选记录构造唯一且可检查的证据路径。"""

    if not source_path.is_file():
        raise FileNotFoundError(f"baseline 证据来源不存在: {source_path.as_posix()}")
    return (relative_or_absolute(source_path, root_path),)


def digest_for_source(source_path: Path) -> str:
    """从唯一证据文件计算候选记录来源摘要。"""

    if not source_path.is_file():
        raise FileNotFoundError(f"baseline 摘要来源不存在: {source_path.as_posix()}")
    return build_file_digest(source_path)


def normalize_t2smark_candidate_rows(
    *,
    rows: Iterable[Mapping[str, Any]],
    source_path: Path,
    root_path: Path,
) -> list[dict[str, Any]]:
    """把 T2SMark 候选记录绑定到当前可检查的结果包证据。"""

    row_values = [dict(row) for row in rows]
    normalized: list[dict[str, Any]] = []
    _, canonical_prompt_protocol_digest = load_canonical_prompt_protocol(root_path)
    source_digest = digest_for_source(source_path)
    evidence_paths = evidence_path_for_source(source_path, root_path)
    for row in row_values:
        record = dict(row)
        if str(record.get("prompt_protocol_digest", "")) != canonical_prompt_protocol_digest:
            raise ValueError("T2SMark 候选记录未绑定当前受治理 Prompt 协议摘要")
        if evidence_paths:
            record["baseline_result_source"] = evidence_paths[0]
            record["baseline_result_source_digest"] = source_digest
            threshold_evidence_path = str(
                record.get("fixed_fpr_observation_evidence_path", "")
            ).strip()
            record["evidence_paths"] = list(
                dict.fromkeys(
                    (
                        *evidence_paths,
                        *([threshold_evidence_path] if threshold_evidence_path else []),
                    )
                )
            )
            record["formal_evidence_paths_ready"] = True
        normalized.append(record)
    return normalized


def _measured_baseline_readiness(
    rows: Iterable[Mapping[str, Any]],
    root_path: Path,
    target_fpr: float,
) -> dict[str, bool]:
    """仅根据 observation 实体记录计算 baseline 三项正式就绪条件。"""

    observations = tuple(dict(row) for row in rows)
    paper_run = build_paper_run_config(root_path)
    expected_splits = build_group_split_counts(paper_run.prompt_count)
    canonical_prompts, _ = load_canonical_prompt_protocol(root_path)
    expected_prompt_ids = set(canonical_prompts)
    clean_rows = tuple(
        row
        for row in observations
        if row.get("sample_role") == "clean_negative" and row.get("attack_family") == "clean"
    )
    positive_rows = tuple(
        row
        for row in observations
        if row.get("sample_role") == "positive_source" and row.get("attack_family") == "clean"
    )
    clean_prompt_ids = [str(row.get("prompt_id", "")) for row in clean_rows]
    positive_prompt_ids = [str(row.get("prompt_id", "")) for row in positive_rows]
    actual_splits = {
        split: sum(row.get("split") == split for row in clean_rows)
        for split in ("dev", "calibration", "test")
    }
    observation_prompt_identity_ready = all(
        str(row.get("prompt_id", "")) in canonical_prompts
        and normalize_prompt_text(str(row.get("prompt_text", "")))
        == canonical_prompts[str(row.get("prompt_id", ""))]["prompt_text"]
        and str(row.get("split", ""))
        == canonical_prompts[str(row.get("prompt_id", ""))]["split"]
        and (
            not str(row.get("prompt_digest", ""))
            or str(row.get("prompt_digest", ""))
            == canonical_prompts[str(row.get("prompt_id", ""))]["prompt_digest"]
        )
        for row in observations
    )
    prompt_protocol_ready = all(
        (
            len(clean_prompt_ids) == paper_run.prompt_count,
            len(positive_prompt_ids) == paper_run.prompt_count,
            set(clean_prompt_ids) == expected_prompt_ids,
            set(positive_prompt_ids) == expected_prompt_ids,
            len(set(clean_prompt_ids)) == len(clean_prompt_ids),
            len(set(positive_prompt_ids)) == len(positive_prompt_ids),
            actual_splits == expected_splits,
            observation_prompt_identity_ready,
        )
    )
    threshold_audit = audit_fixed_fpr_observation_threshold(
        observations,
        target_fpr=target_fpr,
        expected_calibration_source_negative_count=expected_splits["calibration"],
    )
    required_attack_names = {
        config.attack_name
        for config in default_attack_configs()
        if config.enabled and config.resource_profile in {"full_main", "full_extra"}
    }
    actual_attack_names = {
        str(row.get("attack_name") or row.get("attack_condition"))
        for row in observations
        if str(row.get("sample_role", "")).startswith("attacked_")
    }
    attack_matrix_ready = actual_attack_names == required_attack_names
    return {
        "prompt_protocol_ready": prompt_protocol_ready,
        "fixed_fpr_ready": threshold_audit.fixed_fpr_ready,
        "attack_matrix_ready": attack_matrix_ready,
    }


def build_method_candidate_rows(
    *,
    sources: Iterable[MethodFaithfulObservationSource],
    root_path: Path,
    target_fpr: float,
) -> list[dict[str, Any]]:
    """把三个 exact-set observation source 分别聚合为共同协议候选记录。"""

    records: list[dict[str, Any]] = []
    for source in sources:
        baseline_rows = [dict(row) for row in source.rows]
        evidence_paths = (
            relative_or_absolute(source.observations_path, root_path),
            relative_or_absolute(source.transfer_manifest_path, root_path),
            relative_or_absolute(source.prompt_plan_path, root_path),
            relative_or_absolute(source.adapter_manifest_path, root_path),
            relative_or_absolute(source.execution_manifest_path, root_path),
        )
        baseline_result_source = evidence_paths[0]
        source_digest = source.observations_sha256
        _, prompt_protocol_digest = load_canonical_prompt_protocol(root_path)
        readiness = _measured_baseline_readiness(baseline_rows, root_path, target_fpr)
        records.extend(
            build_method_faithful_baseline_candidate_records(
                baseline_id=source.baseline_id,
                observation_rows=baseline_rows,
                target_fpr=target_fpr,
                baseline_result_source=baseline_result_source,
                baseline_result_source_digest=source_digest,
                evidence_paths=evidence_paths,
                prompt_protocol_digest=prompt_protocol_digest,
                paper_run_prompt_protocol_ready=readiness["prompt_protocol_ready"],
                fixed_fpr_baseline_calibration_ready=readiness["fixed_fpr_ready"],
                attack_matrix_baseline_detection_ready=readiness["attack_matrix_ready"],
            )
        )
    return records


def write_primary_baseline_result_candidate_outputs(
    *,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
    attack_manifest_path: str | Path | None = None,
    method_faithful_collection_path: str | Path | None = None,
    t2smark_candidate_records_path: str | Path | None = None,
    t2smark_formal_package_path: str | Path | None = None,
) -> dict[str, Any]:
    """写出候选记录、候选校验报告、摘要和 manifest。"""

    root_path = Path(root).resolve()
    paper_run = build_paper_run_config(root_path)
    resolved_output_dir = ensure_output_dir_under_outputs(
        root_path,
        output_dir or DEFAULT_OUTPUT_ROOT / paper_run.run_name,
    )
    resolved_attack_manifest_path = resolve_path(
        root_path,
        attack_manifest_path
        or DEFAULT_ATTACK_MATRIX_ROOT / paper_run.run_name / "attack_manifest.json",
    )
    resolved_method_faithful_collection_path = resolve_path(
        root_path,
        method_faithful_collection_path
        or METHOD_FAITHFUL_COLLECTION_ROOT / paper_run.run_name,
    )
    resolved_t2smark_candidate_records_path = resolve_path(
        root_path,
        t2smark_candidate_records_path
        or DEFAULT_T2SMARK_FORMAL_ROOT / paper_run.run_name / T2SMARK_CANDIDATE_RECORDS_NAME,
    )
    resolved_t2smark_formal_package_path = resolve_path(root_path, t2smark_formal_package_path)
    if resolved_attack_manifest_path is None or resolved_method_faithful_collection_path is None:
        raise ValueError("必要输入路径不能为空。")
    if resolved_t2smark_candidate_records_path is None:
        raise ValueError("T2SMark 候选记录路径不能为空。")

    if not resolved_attack_manifest_path.is_file():
        raise FileNotFoundError(f"攻击矩阵 manifest 不存在: {resolved_attack_manifest_path.as_posix()}")
    attack_manifest = read_json(resolved_attack_manifest_path)
    target_fpr = paper_run.target_fpr
    manifest_target_fpr = attack_manifest.get("evaluation_boundary", {}).get("target_fpr")
    if manifest_target_fpr is None or not math.isclose(
        float(manifest_target_fpr), target_fpr, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("攻击矩阵 target_fpr 必须与当前论文运行层级一致")
    method_faithful_sources = load_method_faithful_observation_collection(
        resolved_method_faithful_collection_path,
        project_root=root_path,
    )
    t2smark_rows = load_t2smark_candidate_rows(
        candidate_records_path=resolved_t2smark_candidate_records_path,
        package_path=resolved_t2smark_formal_package_path,
        package_entry_name=t2smark_candidate_records_entry(paper_run.run_name),
    )
    t2smark_source_path = (
        resolved_t2smark_formal_package_path
        if resolved_t2smark_formal_package_path is not None
        else resolved_t2smark_candidate_records_path
    )
    method_candidate_rows = build_method_candidate_rows(
        sources=method_faithful_sources,
        root_path=root_path,
        target_fpr=target_fpr,
    )
    normalized_t2smark_rows = normalize_t2smark_candidate_rows(
        rows=t2smark_rows,
        source_path=t2smark_source_path,
        root_path=root_path,
    )
    candidate_rows = method_candidate_rows + normalized_t2smark_rows
    attack_profile_lookup = build_attack_resource_profile_lookup()
    validation_report = validate_primary_baseline_formal_import_rows(
        candidate_rows,
        evidence_root=root_path,
        target_fpr=target_fpr,
        require_existing_evidence=True,
        allowed_resource_profiles=allowed_resource_profiles_from_attack_lookup(attack_profile_lookup),
    )
    readiness_rows = build_primary_baseline_formal_import_readiness_rows(candidate_rows, validation_report)
    readiness_summary = build_primary_baseline_formal_import_readiness_summary(readiness_rows)
    method_threshold_digest_map = build_primary_baseline_method_threshold_digest_map(
        candidate_rows
    )
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
        "method_threshold_digest_map": method_threshold_digest_map,
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
            "missing_formal_attack_resource_profile",
            "missing_paper_run_prompt_protocol",
            "missing_fixed_fpr_baseline_calibration",
            "missing_attack_matrix_baseline_detection",
            "formal_evidence_paths_ready",
            "supports_paper_claim",
        ],
    )
    readiness_summary_path.write_text(stable_json_text(readiness_summary), encoding="utf-8")
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")

    input_paths = []
    method_source_paths = [
        path
        for source in method_faithful_sources
        for path in (source.observations_path, source.transfer_manifest_path)
    ]
    for path in (resolved_attack_manifest_path, *method_source_paths, t2smark_source_path):
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
            "method_threshold_digest_map": method_threshold_digest_map,
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
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录; 默认写入当前论文运行子目录, 且必须位于 outputs/ 下。",
    )
    parser.add_argument(
        "--attack-manifest-path",
        default=None,
        help="攻击矩阵 manifest 路径; 默认读取当前论文运行子目录。",
    )
    parser.add_argument(
        "--method-faithful-collection-path",
        default=None,
        help="三个方法忠实 SD3.5 baseline 的 exact-set 物化根目录; 默认读取当前论文运行子目录。",
    )
    parser.add_argument(
        "--t2smark-candidate-records-path",
        default=None,
        help="T2SMark formal 候选 JSONL 路径; 默认读取当前论文运行子目录。",
    )
    parser.add_argument("--t2smark-formal-package-path", default=None, help="可选 T2SMark formal 结果 zip 包。")
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    manifest = write_primary_baseline_result_candidate_outputs(
        root=args.root,
        output_dir=args.output_dir,
        attack_manifest_path=args.attack_manifest_path,
        method_faithful_collection_path=args.method_faithful_collection_path,
        t2smark_candidate_records_path=args.t2smark_candidate_records_path,
        t2smark_formal_package_path=args.t2smark_formal_package_path,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()

