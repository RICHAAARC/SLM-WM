"""写出当前论文运行层级 fixed-FPR 共同协议结果记录。

该脚本的作用是把方法主流程和外部 baseline 的受治理产物转换为统一
`pilot_paper_result_records.jsonl`。Notebook 只负责在 Colab 中生成上游包,
本脚本负责结果物化、模板覆盖检查、schema 校验和 manifest 写出。
probe_paper、pilot_paper 与 full_paper 都只接受严格正式证据记录。
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol.attacks import (
    attack_config_digest,
    default_attack_configs,
    resolve_formal_attack_config,
)
from experiments.protocol.formal_evidence import contains_nonformal_marker
from experiments.protocol.pilot_paper_fixed_fpr import (
    PILOT_PAPER_METRIC_BOUNDS,
    PilotPaperFixedFprConfig,
    build_attack_matrix_digest,
    build_fixed_fpr_protocol_digest,
    build_paper_fixed_fpr_config,
    build_pilot_paper_attack_matrix_rows,
    build_pilot_paper_method_registry_rows,
    build_pilot_paper_prompt_split_summary,
    build_pilot_paper_result_record_set_digest,
    build_pilot_paper_result_records_manifest_config,
    build_pilot_paper_result_import_schema,
    build_pilot_paper_result_import_template_rows,
    bounded_hoeffding_confidence_interval,
    bounded_metric_value,
    clamp_unit_interval,
    validate_pilot_paper_result_import_rows,
)
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.protocol.paper_run_config import normalize_paper_run_name
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.runtime.image_metrics import measured_score_retention
from main.core.digest import build_stable_digest
from paper_experiments.runners.closure_package_selection import (
    load_validated_closure_input_lock,
)

CONSTRUCTION_UNIT_NAME = "pilot_paper_fixed_fpr_result_records"
DEFAULT_OUTPUT_ROOT = Path("outputs/pilot_paper_fixed_fpr_results")
DEFAULT_BASELINE_RESULTS_ROOT = Path("outputs/external_baseline_results")
DEFAULT_DATASET_QUALITY_ROOT = Path("outputs/dataset_level_quality")
DEFAULT_DATASET_QUALITY_SUMMARY_NAME = "dataset_quality_summary.json"
IMAGE_ONLY_FORMAL_SLM_METRIC_STATUS = "measured_image_only_detection_formal_protocol"
CLAIM_SUPPORTED_METHOD_STATUSES = {
    "measured",
    IMAGE_ONLY_FORMAL_SLM_METRIC_STATUS,
}

def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: dict[str, Any]) -> str:
    """把单条记录转换为 JSONL 文本行。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def resolve_code_version(root_path: Path) -> str:
    """读取 Git 短提交标识, 工作区有变更时附加 dirty 标记。"""

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


def resolve_path(root_path: Path, path: str | Path | None) -> Path | None:
    """把可选路径解析为绝对路径。"""

    if path is None or not str(path).strip():
        return None
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先记录相对仓库根目录的路径, 仓库外路径保留绝对路径。"""

    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def ensure_output_dir_under_outputs(root_path: Path, output_dir: str | Path) -> Path:
    """确保持久化输出目录位于 outputs/ 下。"""

    resolved = resolve_path(root_path, output_dir)
    if resolved is None:
        raise ValueError("pilot_paper 结果记录输出目录不能为空")
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("pilot_paper 结果记录输出目录必须位于 outputs/ 下") from exc
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def file_digest(path: Path) -> str:
    """计算文件 SHA-256 摘要。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WorkProgress:
    """按文件数与字节数输出低噪声工作量进度。

    该工具用于 Colab / Google Drive 的长耗时 I/O 路径。它不改变业务记录语义,
    只让使用者能够判断当前是在扫描、解压还是复制, 便于定位远程文件系统卡顿。
    """

    def __init__(
        self,
        label: str,
        total_count: int,
        *,
        total_bytes: int = 0,
        emit_every_count: int = 50,
        emit_every_seconds: float = 15.0,
    ) -> None:
        self.label = label
        self.total_count = max(0, int(total_count))
        self.total_bytes = max(0, int(total_bytes))
        self.emit_every_count = max(1, int(emit_every_count))
        self.emit_every_seconds = max(0.1, float(emit_every_seconds))
        self.started_at = time.monotonic()
        self.last_emit_at = 0.0
        self.last_emit_count = -1

    def emit(self, count: int, *, copied_bytes: int = 0, profile: str = "", force: bool = False) -> None:
        """输出一行总体进度, 默认按时间和条目数节流。"""

        current_count = max(0, int(count))
        now = time.monotonic()
        if not force and current_count != self.total_count:
            enough_count = current_count - self.last_emit_count >= self.emit_every_count
            enough_time = now - self.last_emit_at >= self.emit_every_seconds
            if not enough_count and not enough_time:
                return
        elapsed = max(0.0, now - self.started_at)
        count_ratio = 1.0 if self.total_count == 0 else min(1.0, current_count / max(1, self.total_count))
        if self.total_count <= 1 and self.total_bytes and current_count < self.total_count:
            ratio = min(1.0, max(0.0, float(copied_bytes) / max(1.0, float(self.total_bytes))))
        else:
            ratio = count_ratio
        eta = 0.0 if ratio <= 0.0 else max(0.0, elapsed * (1.0 - ratio) / ratio)
        bytes_profile = ""
        if self.total_bytes:
            copied_mb = float(copied_bytes) / (1024.0 * 1024.0)
            total_mb = float(self.total_bytes) / (1024.0 * 1024.0)
            bytes_profile = f" copied_mb={copied_mb:.1f}/{total_mb:.1f}"
        print(
            f"工作量进度 | {self.label} | {current_count}/{self.total_count} ({ratio * 100.0:.1f}%) | "
            f"elapsed={elapsed / 60.0:.1f} min | eta={eta / 60.0:.1f} min | "
            f"profile={profile}{bytes_profile}",
            flush=True,
        )
        self.last_emit_at = now
        self.last_emit_count = current_count


def is_safe_output_zip_entry(entry_name: str, root_path: Path, outputs_root: Path) -> tuple[bool, Path | None]:
    """判断 zip 条目是否允许物化到仓库 outputs/ 目录。"""

    if not entry_name.startswith("outputs/"):
        return False, None
    destination = (root_path / entry_name).resolve()
    try:
        destination.relative_to(outputs_root)
    except ValueError:
        return False, None
    return True, destination


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件, 文件缺失时返回空字典。"""

    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 记录, 文件缺失时返回空列表。"""

    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """读取 CSV 表格, 文件缺失时返回空列表。"""

    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """按固定字段顺序写出 CSV 表格。"""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _float_field(row: Mapping[str, Any], field_name: str, default: float = 0.0) -> float:
    """读取浮点字段, 缺失时返回默认值。"""

    return float(row.get(field_name, default) or default)


def _int_field(row: Mapping[str, Any], field_name: str, default: int = 0) -> int:
    """读取整数字段, 缺失时返回默认值。"""

    return int(float(row.get(field_name, default) or default))


def _str_field(row: Mapping[str, Any], field_name: str, default: str = "") -> str:
    """读取字符串字段, 缺失时返回默认值。"""

    return str(row.get(field_name, default) or default)


def _list_field(row: Mapping[str, Any], field_name: str) -> list[str]:
    """读取路径列表字段, 兼容 JSON list 和分号分隔文本。"""

    value = row.get(field_name, ())
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(";") if part.strip()]
    return []


def expand_package_paths(root_path: Path, package_paths: Iterable[str | Path], package_search_roots: Iterable[str | Path]) -> tuple[Path, ...]:
    """解析显式 zip 与镜像目录中的 zip 包路径。"""

    resolved: list[Path] = []
    for raw_path in package_paths:
        path = resolve_path(root_path, raw_path)
        if path is None or not path.is_file() or path.suffix.lower() != ".zip":
            raise FileNotFoundError(f"显式论文结果包不存在或不是 zip: {raw_path}")
        resolved.append(path)
    for raw_root in package_search_roots:
        root = resolve_path(root_path, raw_root)
        if root is None or not root.is_dir():
            raise FileNotFoundError(f"论文结果包搜索目录不存在: {raw_root}")
        resolved.extend(sorted(path.resolve() for path in root.rglob("*.zip") if path.is_file()))
    return tuple(dict.fromkeys(resolved))


def materialize_output_entries(root_path: Path, package_paths: Iterable[Path]) -> dict[str, Any]:
    """从结果包中只解出 outputs/ 条目。

    该函数用于 Colab / Drive 复盘: 前序 Notebook 的 zip 包仍是证据来源, 但共同协议 builder 只消费仓库
    `outputs/` 下的受治理文件副本。路径穿越和非 outputs 条目会被跳过。
    """

    package_path_values = tuple(package_paths)
    outputs_root = (root_path / "outputs").resolve()
    materialized_entries: list[str] = []
    skipped_entries: list[str] = []
    planned_by_destination: dict[Path, tuple[Path, Any, Path]] = {}
    duplicate_identical_entries: list[str] = []
    for package_path in package_path_values:
        with ZipFile(package_path) as archive:
            for entry_info in archive.infolist():
                entry = entry_info.filename
                if entry.endswith("/"):
                    continue
                is_safe, destination = is_safe_output_zip_entry(entry, root_path, outputs_root)
                if not is_safe or destination is None:
                    skipped_entries.append(entry)
                    continue
                existing = planned_by_destination.get(destination)
                if existing is not None:
                    existing_package, existing_info, _ = existing
                    same_payload = (
                        int(existing_info.file_size) == int(entry_info.file_size)
                        and int(existing_info.CRC) == int(entry_info.CRC)
                    )
                    if not same_payload:
                        raise RuntimeError(
                            "结果包包含同路径不同内容, 拒绝跨运行覆盖: "
                            f"{entry} 来自 {existing_package.name} 与 {package_path.name}"
                        )
                    duplicate_identical_entries.append(entry)
                    continue
                planned_by_destination[destination] = (package_path, entry_info, destination)

    planned_items = list(planned_by_destination.values())

    total_bytes = sum(int(entry_info.file_size) for _, entry_info, _ in planned_items)
    progress = WorkProgress(
        "pilot_paper package materialization",
        len(planned_items),
        total_bytes=total_bytes,
        emit_every_count=100,
    )
    progress.emit(0, copied_bytes=0, profile=f"package_count={len(package_path_values)}", force=True)
    copied_bytes = 0
    completed_count = 0
    current_package: Path | None = None
    current_archive: ZipFile | None = None
    try:
        for package_path, entry_info, destination in planned_items:
            if current_package != package_path:
                if current_archive is not None:
                    current_archive.close()
                current_package = package_path
                current_archive = ZipFile(package_path)
            if current_archive is None:
                raise RuntimeError("zip_archive_not_open")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with current_archive.open(entry_info.filename, "r") as source_handle, destination.open("wb") as target_handle:
                for chunk in iter(lambda: source_handle.read(8 * 1024 * 1024), b""):
                    target_handle.write(chunk)
                    copied_bytes += len(chunk)
                    progress.emit(
                        completed_count,
                        copied_bytes=copied_bytes,
                        profile=f"package={package_path.name} extracting={entry_info.filename}",
                    )
            completed_count += 1
            materialized_entries.append(entry_info.filename)
            progress.emit(
                completed_count,
                copied_bytes=copied_bytes,
                profile=f"package={package_path.name} file={entry_info.filename}",
            )
    finally:
        if current_archive is not None:
            current_archive.close()
    progress.emit(
        completed_count,
        copied_bytes=copied_bytes,
        profile=f"package_count={len(package_path_values)} done",
        force=True,
    )
    return {
        "input_package_count": len(package_path_values),
        "input_package_paths": [relative_or_absolute(path, root_path) for path in package_path_values],
        "materialized_output_entry_count": len(materialized_entries),
        "materialized_output_total_bytes": copied_bytes,
        "skipped_output_entry_count": len(skipped_entries),
        "duplicate_identical_entry_count": len(duplicate_identical_entries),
        "materialized_output_entries_digest": build_stable_digest(sorted(materialized_entries)),
        "skipped_output_entries": sorted(skipped_entries),
    }


def build_protocol_context(root_path: Path, config: PilotPaperFixedFprConfig) -> dict[str, Any]:
    """构造 result records 需要引用的共同协议摘要。"""

    prompt_path = resolve_path(root_path, config.prompt_file)
    if prompt_path is None or not prompt_path.is_file():
        raise FileNotFoundError(f"缺少论文运行 prompt 文件: {config.prompt_file}")
    prompt_records = build_prompt_records(config.prompt_set, read_prompt_file(prompt_path))
    prompt_summary = build_pilot_paper_prompt_split_summary(prompt_records, config)
    attack_rows = build_pilot_paper_attack_matrix_rows(default_attack_configs(), config)
    attack_matrix_digest = build_attack_matrix_digest(attack_rows)
    fixed_fpr_protocol_digest = build_fixed_fpr_protocol_digest(config)
    method_rows = build_pilot_paper_method_registry_rows(
        prompt_split_digest=str(prompt_summary["prompt_split_digest"]),
        attack_matrix_digest=attack_matrix_digest,
        fixed_fpr_protocol_digest=fixed_fpr_protocol_digest,
        config=config,
    )
    schema = build_pilot_paper_result_import_schema(
        prompt_split_digest=str(prompt_summary["prompt_split_digest"]),
        attack_matrix_digest=attack_matrix_digest,
        fixed_fpr_protocol_digest=fixed_fpr_protocol_digest,
        config=config,
    )
    template_rows = build_pilot_paper_result_import_template_rows(method_rows, attack_rows, config)
    return {
        "prompt_summary": prompt_summary,
        "attack_rows": attack_rows,
        "method_rows": method_rows,
        "schema": schema,
        "template_rows": template_rows,
    }


def evidence_paths_for_existing(paths: Iterable[Path], root_path: Path) -> list[str]:
    """把存在的证据文件转换为可记录路径。"""

    return [relative_or_absolute(path, root_path) for path in paths if path.is_file()]


def dataset_quality_claim_gate_fields(dataset_quality_metrics_path: Path, root_path: Path) -> dict[str, Any]:
    """读取数据集级正式质量主张门禁字段。

    `dataset_quality_metrics.csv` 只允许包含由正式 Inception 特征计算的
    FID / KID 行。
    """

    dataset_quality_summary_path = dataset_quality_metrics_path.with_name(DEFAULT_DATASET_QUALITY_SUMMARY_NAME)
    summary = read_json(dataset_quality_summary_path)
    metric_rows = read_csv_rows(dataset_quality_metrics_path)
    measured_formal_names = {
        _str_field(row, "quality_metric_name")
        for row in metric_rows
        if _str_field(row, "quality_metric_name") in {"fid", "kid"} and _str_field(row, "metric_status") == "measured"
    }
    metric_names_ready = bool(summary.get("formal_fid_kid_metric_names_ready", measured_formal_names == {"fid", "kid"}))
    canonical_extractor_ready = bool(summary.get("canonical_formal_feature_extractor_ready", False))
    claim_gate_ready = bool(
        summary.get("formal_fid_kid_claim_gate_ready", metric_names_ready)
        and canonical_extractor_ready
    )
    blocker = _str_field(summary, "formal_fid_kid_claim_blocker")
    if not blocker and not claim_gate_ready:
        blocker = "formal_fid_kid_not_measured"
    boundary = _str_field(summary, "dataset_quality_claim_boundary")
    if not boundary:
        boundary = (
            "formal_fid_kid_measured_but_paper_claim_requires_evidence_closure"
            if claim_gate_ready
            else "dataset_quality_formal_fid_kid_not_ready"
        )
    elif contains_nonformal_marker(boundary):
        boundary = "dataset_quality_formal_fid_kid_not_ready"
    return {
        "formal_fid_kid_metric_names_ready": metric_names_ready,
        "formal_fid_kid_claim_gate_ready": claim_gate_ready,
        "canonical_formal_feature_extractor_ready": canonical_extractor_ready,
        "formal_fid_kid_claim_blocker": blocker,
        "dataset_quality_formal_metric_ready": claim_gate_ready,
        "dataset_quality_claim_boundary": boundary,
        "dataset_quality_summary_path": (
            relative_or_absolute(dataset_quality_summary_path, root_path) if dataset_quality_summary_path.is_file() else ""
        ),
    }


def build_common_result_fields(
    *,
    schema: Mapping[str, Any],
    method_id: str,
    attack_id: str,
    attack_family: str,
    attack_name: str,
    resource_profile: str,
    attack_config_digest_value: str,
    baseline_result_source: str,
    baseline_result_source_digest: str,
    method_threshold_digest: str,
    evidence_paths: list[str],
) -> dict[str, Any]:
    """构造所有方法共享的 pilot_paper schema 字段。"""

    return {
        "result_protocol_name": schema["result_protocol_name"],
        "result_scope": schema["result_scope"],
        "result_claim_scope": schema["result_claim_scope"],
        "method_id": method_id,
        "attack_id": attack_id,
        "attack_family": attack_family,
        "attack_name": attack_name,
        "resource_profile": resource_profile,
        "attack_config_digest": attack_config_digest_value,
        "prompt_protocol_name": schema["prompt_protocol_name"],
        "prompt_split_digest": schema["prompt_split_digest"],
        "attack_matrix_digest": schema["attack_matrix_digest"],
        "fixed_fpr_protocol_digest": schema["fixed_fpr_protocol_digest"],
        "method_threshold_digest": method_threshold_digest,
        "target_fpr": schema["target_fpr"],
        "confidence_interval_method": schema["confidence_interval_method"],
        "confidence_level": schema["confidence_level"],
        "baseline_result_source": baseline_result_source,
        "baseline_result_source_digest": baseline_result_source_digest,
        "evidence_paths": evidence_paths,
        "paper_claim_scale": schema.get("paper_claim_scale", "pilot_paper"),
        "paper_run_claim_type": schema.get("paper_run_claim_type", schema["result_claim_scope"]),
        "strict_formal_evidence_required": bool(schema.get("strict_formal_evidence_required", True)),
    }


def attach_metric_fields(
    payload: dict[str, Any],
    *,
    positive_count: int,
    negative_count: int,
    attack_record_count: int,
    supported_record_count: int,
    true_positive_rate: float,
    false_positive_rate: float,
    clean_false_positive_rate: float,
    attacked_false_positive_rate: float,
    quality_score_mean: float,
    score_retention_mean: float,
    confidence_level: float,
) -> dict[str, Any]:
    """补齐指标字段和确定性置信区间字段。"""

    quality_lower, quality_upper = PILOT_PAPER_METRIC_BOUNDS[
        "quality_score_mean"
    ]
    metric_payload = {
        "positive_count": max(0, int(positive_count)),
        "negative_count": max(0, int(negative_count)),
        "attack_record_count": max(0, int(attack_record_count)),
        "supported_record_count": max(0, int(supported_record_count)),
        "true_positive_rate": clamp_unit_interval(true_positive_rate),
        "false_positive_rate": clamp_unit_interval(false_positive_rate),
        "clean_false_positive_rate": clamp_unit_interval(clean_false_positive_rate),
        "attacked_false_positive_rate": clamp_unit_interval(attacked_false_positive_rate),
        "quality_score_mean": bounded_metric_value(
            quality_score_mean,
            lower_bound=quality_lower,
            upper_bound=quality_upper,
        ),
        "score_retention_mean": clamp_unit_interval(score_retention_mean),
    }
    ci_inputs = (
        ("true_positive_rate", metric_payload["positive_count"], "true_positive_rate_ci_low", "true_positive_rate_ci_high", 0.0, 1.0),
        ("false_positive_rate", metric_payload["negative_count"], "false_positive_rate_ci_low", "false_positive_rate_ci_high", 0.0, 1.0),
        (
            "clean_false_positive_rate",
            metric_payload["negative_count"],
            "clean_false_positive_rate_ci_low",
            "clean_false_positive_rate_ci_high",
            0.0,
            1.0,
        ),
        (
            "attacked_false_positive_rate",
            metric_payload["negative_count"],
            "attacked_false_positive_rate_ci_low",
            "attacked_false_positive_rate_ci_high",
            0.0,
            1.0,
        ),
        # 质量与分数保持均值都来自 attacked positive, 因而分母必须是 positive_count.
        ("quality_score_mean", metric_payload["positive_count"], "quality_score_ci_low", "quality_score_ci_high", quality_lower, quality_upper),
        (
            "score_retention_mean",
            metric_payload["positive_count"],
            "score_retention_ci_low",
            "score_retention_ci_high",
            0.0,
            1.0,
        ),
    )
    for (
        metric_name,
        sample_count,
        low_name,
        high_name,
        lower_bound,
        upper_bound,
    ) in ci_inputs:
        low, high = bounded_hoeffding_confidence_interval(
            metric_payload[metric_name],
            sample_count,
            confidence_level,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        metric_payload[low_name] = low
        metric_payload[high_name] = high
    payload.update(metric_payload)
    return payload


def finalize_result_record(
    payload: dict[str, Any],
    *,
    metric_status: str,
    source_kind: str,
    claim_ready: bool = True,
    minimum_claim_count: int = 1,
    paper_run_allows_paper_claim: bool = True,
) -> dict[str, Any]:
    """补齐稳定标识、摘要和 claim 边界字段。"""

    evidence_values = payload.get("evidence_paths", ())
    formal_marker_ready = not contains_nonformal_marker(
        (
            metric_status,
            payload.get("result_source_kind", ""),
            payload.get("baseline_result_source", ""),
            evidence_values,
        )
    )
    supports_claim = (
        claim_ready
        and paper_run_allows_paper_claim
        and formal_marker_ready
        and metric_status in CLAIM_SUPPORTED_METHOD_STATUSES
        and int(payload.get("positive_count", 0)) >= minimum_claim_count
        and int(payload.get("negative_count", 0)) >= minimum_claim_count
        and int(payload.get("supported_record_count", 0)) > 0
        and bool(payload.get("evidence_paths"))
    )
    payload["metric_status"] = metric_status
    payload["result_source_kind"] = source_kind
    payload["strict_formal_result_ready"] = supports_claim
    payload["supports_paper_claim"] = supports_claim
    digest = build_stable_digest(payload)
    payload["pilot_paper_result_record_digest"] = digest
    payload["pilot_paper_result_record_id"] = f"pilot_paper_result_record_{digest[:16]}"
    return payload


def detection_attack_identity_lookup(
    detection_records_path: Path,
) -> dict[tuple[str, str, str], dict[str, str]]:
    """从主方法执行端检测记录提取正式攻击身份."""

    identities: dict[tuple[str, str, str], dict[str, str]] = {}
    for row_index, row in enumerate(read_jsonl_rows(detection_records_path)):
        attack_id = _str_field(row, "attack_id")
        attack_family = _str_field(row, "attack_family")
        attack_name = _str_field(row, "attack_name")
        resource_profile = _str_field(row, "resource_profile")
        attacked_row = (
            attack_family != "clean"
            and attack_name not in {"", "none", "clean_none"}
        )
        if not attacked_row:
            continue
        if not attack_id:
            raise ValueError(f"主方法攻击检测记录缺少 attack_id: {row_index}")
        try:
            config = resolve_formal_attack_config(
                attack_family=attack_family,
                attack_name=attack_name,
                resource_profile=resource_profile,
            )
        except ValueError as exc:
            raise ValueError(
                f"主方法检测记录包含未注册攻击身份: {row_index}"
            ) from exc
        identity = {
            "attack_id": config.attack_id,
            "resource_profile": config.resource_profile,
            "attack_config_digest": attack_config_digest(config),
        }
        for field_name, expected_value in identity.items():
            if _str_field(row, field_name) != expected_value:
                raise ValueError(
                    f"主方法检测记录攻击身份不一致: {row_index}/{field_name}"
                )
        key = (attack_family, attack_name, resource_profile)
        if key in identities and identities[key] != identity:
            raise ValueError(f"主方法检测记录攻击身份不唯一: {key}")
        identities[key] = identity
    return identities


def baseline_accepted_record_keys(
    validation_report: Mapping[str, Any],
) -> set[tuple[str, str, str, str, str, str]]:
    """从主表 baseline 受治理导入报告中提取已接受记录键。"""

    keys: set[tuple[str, str, str, str, str, str]] = set()
    for row in validation_report.get("accepted_records", ()):
        keys.add(
            (
                _str_field(row, "baseline_id"),
                _str_field(row, "attack_id"),
                _str_field(row, "attack_family"),
                _str_field(row, "attack_name"),
                _str_field(row, "resource_profile"),
                _str_field(row, "attack_config_digest"),
            )
        )
    return keys


def build_image_only_slm_wm_result_records(
    *,
    root_path: Path,
    schema: Mapping[str, Any],
    metrics_path: Path,
    summary_path: Path,
    detection_records_path: Path,
    runtime_manifest_path: Path,
    dataset_quality_metrics_path: Path,
) -> list[dict[str, Any]]:
    """从仅图像检测数据集运行结果构造 SLM-WM 正式记录。"""

    rows = read_csv_rows(metrics_path)
    if not rows:
        return []
    attack_identities = detection_attack_identity_lookup(detection_records_path)
    summary = read_json(summary_path)
    row_lookup = {
        (
            _str_field(row, "attack_family"),
            _str_field(row, "attack_name"),
            _str_field(row, "resource_profile"),
            _str_field(row, "sample_role"),
        ): row
        for row in rows
    }
    clean_negative_row = next(
        (
            row
            for row in rows
            if _str_field(row, "attack_name") == "none"
            and _str_field(row, "sample_role") == "clean_negative"
        ),
        None,
    )
    clean_positive_row = next(
        (
            row
            for row in rows
            if _str_field(row, "attack_name") == "none"
            and _str_field(row, "sample_role") == "positive_source"
        ),
        None,
    )
    if clean_negative_row is None or clean_positive_row is None:
        return []
    evidence_paths = evidence_paths_for_existing(
        (
            metrics_path,
            summary_path,
            detection_records_path,
            runtime_manifest_path,
            dataset_quality_metrics_path,
            dataset_quality_metrics_path.with_name(DEFAULT_DATASET_QUALITY_SUMMARY_NAME),
        ),
        root_path,
    )
    dataset_quality_gate_fields = dataset_quality_claim_gate_fields(dataset_quality_metrics_path, root_path)
    minimum_claim_count = int(schema.get("minimum_result_positive_count", 1))
    paper_run_allows_paper_claim = bool(schema.get("paper_run_allows_paper_claim", True))
    source_digest = file_digest(metrics_path)
    clean_false_positive_rate = _float_field(clean_negative_row, "positive_rate")
    clean_positive_score = _float_field(clean_positive_row, "content_score_mean")
    records = []
    for positive_row in rows:
        if _str_field(positive_row, "sample_role") != "positive_source":
            continue
        attack_name = _str_field(positive_row, "attack_name")
        if attack_name == "none":
            continue
        key = (
            _str_field(positive_row, "attack_family"),
            attack_name,
            _str_field(positive_row, "resource_profile"),
            "clean_negative",
        )
        negative_row = row_lookup.get(key)
        if negative_row is None:
            continue
        attack_identity = attack_identities.get(key[:3])
        if attack_identity is None:
            raise ValueError(
                "主方法正式指标缺少执行端攻击身份: "
                f"{key[0]}/{attack_name}/{key[2]}"
            )
        if "source_to_evaluated_ssim_mean" not in positive_row:
            raise ValueError(
                f"主方法攻击记录缺少 source-to-attacked SSIM: {key[0]}/{attack_name}/{key[2]}"
            )
        payload = build_common_result_fields(
            schema=schema,
            method_id="slm_wm_current",
            attack_id=attack_identity["attack_id"],
            attack_family=key[0],
            attack_name=attack_name,
            resource_profile=key[2],
            attack_config_digest_value=attack_identity[
                "attack_config_digest"
            ],
            baseline_result_source=relative_or_absolute(metrics_path, root_path),
            baseline_result_source_digest=source_digest,
            method_threshold_digest=_str_field(summary, "frozen_threshold_digest"),
            evidence_paths=evidence_paths,
        )
        payload.update(
            {
                "detector_input_access_mode": "image_key_public_model_only",
                "blind_image_detector": True,
                "generation_latent_trace_required": False,
                "baseline_fairness_boundary": "all_methods_receive_attacked_image_and_method_key_only",
                "clean_test_fixed_fpr_upper_bound_ready": bool(
                    summary.get("clean_test_fixed_fpr_upper_bound_ready", False)
                ),
                "wrong_key_test_fixed_fpr_upper_bound_ready": bool(
                    summary.get("wrong_key_test_fixed_fpr_upper_bound_ready", False)
                ),
                "scientific_operator_gate_ready": bool(
                    summary.get("scientific_operator_gate_ready", False)
                ),
                "attack_record_coverage_ready": bool(summary.get("attack_record_coverage_ready", False)),
                "attacked_image_evidence_chain_ready": bool(
                    summary.get("attacked_image_evidence_chain_ready", False)
                ),
                "real_gpu_attack_validation_ready": bool(
                    summary.get("real_gpu_attack_validation_ready", False)
                ),
                **dataset_quality_gate_fields,
            }
        )
        attacked_score = _float_field(positive_row, "content_score_mean")
        score_retention = measured_score_retention(clean_positive_score, attacked_score)
        attach_metric_fields(
            payload,
            positive_count=_int_field(positive_row, "record_count"),
            negative_count=_int_field(clean_negative_row, "record_count"),
            attack_record_count=_int_field(positive_row, "record_count") + _int_field(negative_row, "record_count"),
            supported_record_count=_int_field(positive_row, "record_count"),
            true_positive_rate=_float_field(positive_row, "positive_rate"),
            false_positive_rate=clean_false_positive_rate,
            clean_false_positive_rate=clean_false_positive_rate,
            attacked_false_positive_rate=_float_field(negative_row, "positive_rate"),
            quality_score_mean=_float_field(positive_row, "source_to_evaluated_ssim_mean"),
            score_retention_mean=score_retention,
            confidence_level=float(schema["confidence_level"]),
        )
        attacked_negative_count = _int_field(negative_row, "record_count")
        attacked_ci_low, attacked_ci_high = bounded_hoeffding_confidence_interval(
            _float_field(negative_row, "positive_rate"),
            attacked_negative_count,
            float(schema["confidence_level"]),
        )
        payload["attacked_negative_count"] = attacked_negative_count
        payload["attacked_false_positive_rate_ci_low"] = attacked_ci_low
        payload["attacked_false_positive_rate_ci_high"] = attacked_ci_high
        claim_ready = (
            summary.get("protocol_decision") == "pass"
            and bool(summary.get("supports_paper_claim", False))
            and bool(summary.get("clean_test_fixed_fpr_upper_bound_ready", False))
            and bool(summary.get("wrong_key_test_fixed_fpr_upper_bound_ready", False))
            and bool(summary.get("scientific_operator_gate_ready", False))
            and bool(summary.get("attack_record_coverage_ready", False))
            and bool(summary.get("attacked_image_evidence_chain_ready", False))
            and bool(summary.get("real_gpu_attack_validation_ready", False))
            and bool(dataset_quality_gate_fields.get("formal_fid_kid_claim_gate_ready", False))
            and _int_field(positive_row, "record_count") >= minimum_claim_count
            and _int_field(clean_negative_row, "record_count") >= minimum_claim_count
        )
        records.append(
            finalize_result_record(
                payload,
                metric_status=IMAGE_ONLY_FORMAL_SLM_METRIC_STATUS,
                source_kind="slm_wm_image_only_dataset_runtime",
                claim_ready=claim_ready,
                minimum_claim_count=minimum_claim_count,
                paper_run_allows_paper_claim=paper_run_allows_paper_claim,
            )
        )
    return records


def build_baseline_result_records(
    *,
    root_path: Path,
    schema: Mapping[str, Any],
    baseline_records_path: Path,
    baseline_validation_report_path: Path,
) -> list[dict[str, Any]]:
    """从外部 baseline 候选记录构造 pilot_paper 结果记录。"""

    rows = read_jsonl_rows(baseline_records_path)
    if not rows:
        return []
    validation_report = read_json(baseline_validation_report_path)
    accepted_keys = baseline_accepted_record_keys(validation_report)
    source_digest = file_digest(baseline_records_path) if baseline_records_path.is_file() else build_stable_digest(rows)
    minimum_claim_count = int(schema.get("minimum_result_positive_count", 1))
    paper_run_allows_paper_claim = bool(schema.get("paper_run_allows_paper_claim", True))
    records: list[dict[str, Any]] = []
    for row in rows:
        method_id = _str_field(row, "baseline_id")
        if not method_id:
            continue
        record_key = (
            method_id,
            _str_field(row, "attack_id"),
            _str_field(row, "attack_family"),
            _str_field(row, "attack_name"),
            _str_field(row, "resource_profile", "full_main"),
            _str_field(row, "attack_config_digest"),
        )
        evidence_paths = _list_field(row, "evidence_paths")
        if not evidence_paths and baseline_records_path.is_file():
            evidence_paths = [relative_or_absolute(baseline_records_path, root_path)]
        payload = build_common_result_fields(
            schema=schema,
            method_id=method_id,
            attack_id=_str_field(row, "attack_id"),
            attack_family=_str_field(row, "attack_family"),
            attack_name=_str_field(row, "attack_name"),
            resource_profile=_str_field(row, "resource_profile", "full_main"),
            attack_config_digest_value=_str_field(
                row,
                "attack_config_digest",
            ),
            baseline_result_source=_str_field(row, "baseline_result_source", relative_or_absolute(baseline_records_path, root_path)),
            baseline_result_source_digest=_str_field(row, "baseline_result_source_digest", source_digest),
            method_threshold_digest=_str_field(row, "threshold_digest"),
            evidence_paths=evidence_paths,
        )
        payload.update(
            {
                "detector_input_access_mode": _str_field(row, "detector_input_access_mode", "method_native_or_final_image"),
                "blind_image_detector": str(row.get("blind_image_detector", "")).lower() in {"true", "1", "yes"},
                "baseline_fairness_boundary": _str_field(
                    row,
                    "baseline_fairness_boundary",
                    "requires_common_attack_matrix_and_declared_detector_access",
                ),
            }
        )
        attach_metric_fields(
            payload,
            positive_count=_int_field(row, "positive_count"),
            negative_count=_int_field(row, "negative_count"),
            attack_record_count=_int_field(row, "attack_record_count"),
            supported_record_count=_int_field(row, "supported_record_count"),
            true_positive_rate=_float_field(row, "true_positive_rate"),
            false_positive_rate=_float_field(row, "false_positive_rate"),
            clean_false_positive_rate=_float_field(row, "clean_false_positive_rate"),
            attacked_false_positive_rate=_float_field(row, "attacked_false_positive_rate"),
            quality_score_mean=_float_field(row, "quality_score_mean"),
            score_retention_mean=_float_field(row, "score_retention_mean"),
            confidence_level=float(schema["confidence_level"]),
        )
        payload["attacked_negative_count"] = _int_field(row, "attacked_negative_count")
        payload["baseline_formal_import_record_accepted"] = record_key in accepted_keys
        baseline_formal_quality_ready = "quality_score_mean" in row and not contains_nonformal_marker(row.get("quality_score_mean"))
        records.append(
            finalize_result_record(
                payload,
                metric_status=_str_field(row, "metric_status", "measured"),
                source_kind="external_baseline_result",
                claim_ready=record_key in accepted_keys and baseline_formal_quality_ready,
                minimum_claim_count=minimum_claim_count,
                paper_run_allows_paper_claim=paper_run_allows_paper_claim,
            )
        )
    return records


def build_template_coverage_rows(
    *,
    template_rows: Iterable[Mapping[str, Any]],
    result_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """按 method × attack 模板检查结果记录覆盖状态。"""

    record_keys = {
        (
            _str_field(row, "method_id"),
            _str_field(row, "attack_id"),
            _str_field(row, "attack_family"),
            _str_field(row, "attack_name"),
            _str_field(row, "resource_profile"),
            _str_field(row, "attack_config_digest"),
        )
        for row in result_records
    }
    rows: list[dict[str, Any]] = []
    for template in template_rows:
        key = (
            _str_field(template, "method_id"),
            _str_field(template, "attack_id"),
            _str_field(template, "attack_family"),
            _str_field(template, "attack_name"),
            _str_field(template, "resource_profile"),
            _str_field(template, "attack_config_digest"),
        )
        rows.append(
            {
                "method_id": key[0],
                "attack_id": key[1],
                "attack_family": key[2],
                "attack_name": key[3],
                "resource_profile": key[4],
                "attack_config_digest": key[5],
                "template_covered": key in record_keys,
                "supports_paper_claim": False,
            }
        )
    return rows


def template_record_keys(
    template_rows: Iterable[Mapping[str, Any]],
) -> set[tuple[str, str, str, str, str, str]]:
    """提取 pilot_paper 共同协议允许进入 claim 比较的模板键。"""

    return {
        (
            _str_field(row, "method_id"),
            _str_field(row, "attack_id"),
            _str_field(row, "attack_family"),
            _str_field(row, "attack_name"),
            _str_field(row, "resource_profile"),
            _str_field(row, "attack_config_digest"),
        )
        for row in template_rows
    }


def filter_records_to_template(
    records: Iterable[dict[str, Any]],
    template_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """只保留共同协议模板内且通过严格正式门禁的结果记录。

    该函数属于论文 claim 边界治理。调用方会比较过滤前后数量,
    任何模板外或未通过门禁的记录都会使正式物化失败,
    因此不会静默丢弃不合格输入。
    """

    allowed_keys = template_record_keys(template_rows)
    return [
        record
        for record in records
        if (
            _str_field(record, "method_id"),
            _str_field(record, "attack_id"),
            _str_field(record, "attack_family"),
            _str_field(record, "attack_name"),
            _str_field(record, "resource_profile"),
            _str_field(record, "attack_config_digest"),
        )
        in allowed_keys
        and bool(record.get("strict_formal_result_ready"))
    ]


def build_result_summary(
    *,
    records: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    validation_report: Mapping[str, Any],
    materialization_report: Mapping[str, Any],
    schema: Mapping[str, Any],
    result_record_set_digest: str,
    method_threshold_digest_map: Mapping[str, str],
    closure_input_lock_digest: str,
    common_code_version: str,
) -> dict[str, Any]:
    """汇总 pilot_paper 结果记录物化状态。"""

    covered_count = sum(1 for row in coverage_rows if bool(row["template_covered"]))
    method_ids = sorted({str(row.get("method_id", "")) for row in records if row.get("method_id")})
    missing_rows = [row for row in coverage_rows if not bool(row["template_covered"])]
    return {
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pilot_paper_result_record_count": len(records),
        "pilot_paper_result_method_ids": method_ids,
        "pilot_paper_template_record_count": len(coverage_rows),
        "pilot_paper_template_covered_count": covered_count,
        "pilot_paper_template_missing_count": len(missing_rows),
        "pilot_paper_template_coverage_ready": bool(coverage_rows) and covered_count == len(coverage_rows),
        "pilot_paper_result_import_ready": bool(validation_report.get("pilot_paper_result_import_ready", False)),
        "accepted_pilot_paper_import_count": int(validation_report.get("accepted_pilot_paper_import_count", 0)),
        "accepted_pilot_paper_claim_record_count": int(validation_report.get("accepted_pilot_paper_claim_record_count", 0)),
        "pilot_paper_claim_record_ready": bool(validation_report.get("pilot_paper_claim_record_ready", False)),
        "missing_template_examples": missing_rows[:20],
        "materialization_report": dict(materialization_report),
        "paper_claim_scale": str(schema.get("paper_claim_scale", "pilot_paper")),
        "result_record_set_digest": result_record_set_digest,
        "method_threshold_digest_map": dict(sorted(method_threshold_digest_map.items())),
        "closure_input_lock_digest": closure_input_lock_digest,
        "common_code_version": common_code_version,
        "supports_paper_claim": bool(validation_report.get("supports_paper_claim", False)) and not missing_rows,
    }


def write_pilot_paper_result_record_outputs(
    *,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
    baseline_records_path: str | Path | None = None,
    baseline_validation_report_path: str | Path | None = None,
    dataset_quality_metrics_path: str | Path | None = None,
    image_only_runtime_dir: str | Path | None = None,
    package_paths: Iterable[str | Path] = (),
    package_search_roots: Iterable[str | Path] = (),
    require_existing_evidence: bool = False,
    materialize_only: bool = False,
) -> dict[str, Any]:
    """写出 pilot_paper 共同协议结果记录和校验报告。"""

    root_path = Path(root).resolve()
    resolved_run_name = normalize_paper_run_name(os.environ.get("SLM_WM_PAPER_RUN_NAME"))
    output_path = ensure_output_dir_under_outputs(
        root_path,
        output_dir or DEFAULT_OUTPUT_ROOT / resolved_run_name,
    )
    packages = expand_package_paths(root_path, package_paths, package_search_roots)
    materialization_report = materialize_output_entries(root_path, packages) if packages else {
        "input_package_count": 0,
        "materialized_output_entry_count": 0,
        "skipped_output_entry_count": 0,
        "materialized_output_entries_digest": build_stable_digest([]),
        "skipped_output_entries": [],
    }
    if materialize_only:
        report_path = output_path / "pilot_paper_materialization_report.json"
        manifest_path = output_path / "pilot_paper_materialization_manifest.local.json"
        report_path.write_text(stable_json_text(materialization_report), encoding="utf-8")
        manifest = build_artifact_manifest(
            artifact_id="pilot_paper_result_record_materialization_manifest",
            artifact_type="local_manifest",
            input_paths=tuple(relative_or_absolute(path, root_path) for path in packages),
            output_paths=(
                relative_or_absolute(report_path, root_path),
                relative_or_absolute(manifest_path, root_path),
            ),
            config={
                "materialization_report_digest": build_stable_digest(materialization_report),
            },
            code_version=resolve_code_version(root_path),
            rebuild_command="python scripts/write_pilot_paper_result_records.py --materialize-only",
            metadata=materialization_report,
        ).to_dict()
        manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
        return manifest

    config = build_paper_fixed_fpr_config(root_path)
    closure_input_provenance = load_validated_closure_input_lock(
        root_path,
        paper_run_name=config.paper_run_name,
        target_fpr=config.target_fpr,
    )
    context = build_protocol_context(root_path, config)
    schema = context["schema"]
    template_rows = context["template_rows"]
    resolved_image_only_runtime_dir = resolve_path(
        root_path,
        image_only_runtime_dir
        or Path("outputs/image_only_dataset_runtime") / config.paper_run_name,
    )

    resolved_baseline_records_path = resolve_path(
        root_path,
        baseline_records_path
        or DEFAULT_BASELINE_RESULTS_ROOT / config.paper_run_name / "baseline_result_records.jsonl",
    )
    resolved_baseline_validation_report_path = resolve_path(
        root_path,
        baseline_validation_report_path
        or DEFAULT_BASELINE_RESULTS_ROOT
        / config.paper_run_name
        / "baseline_result_candidate_validation_report.json",
    )
    resolved_dataset_quality_metrics_path = resolve_path(
        root_path,
        dataset_quality_metrics_path
        or DEFAULT_DATASET_QUALITY_ROOT / config.paper_run_name / "dataset_quality_metrics.csv",
    )
    image_only_metrics_path = resolved_image_only_runtime_dir / "test_detection_metrics.csv"
    image_only_summary_path = resolved_image_only_runtime_dir / "dataset_runtime_summary.json"
    image_only_detection_path = resolved_image_only_runtime_dir / "image_only_detection_records.jsonl"
    image_only_manifest_path = resolved_image_only_runtime_dir / "manifest.local.json"
    dataset_quality_summary_path = resolved_dataset_quality_metrics_path.with_name(
        DEFAULT_DATASET_QUALITY_SUMMARY_NAME
    )
    required_paths = (
        image_only_metrics_path,
        image_only_summary_path,
        image_only_detection_path,
        image_only_manifest_path,
        resolved_baseline_records_path,
        resolved_baseline_validation_report_path,
        resolved_dataset_quality_metrics_path,
        dataset_quality_summary_path,
    )
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"论文结果记录缺少正式输入: {', '.join(path.as_posix() for path in missing_paths)}")
    if not read_jsonl_rows(image_only_detection_path):
        raise ValueError("仅图像盲检正式检测记录不能为空")
    if not read_json(image_only_manifest_path).get("artifact_id"):
        raise ValueError("仅图像盲检 manifest 缺少 artifact_id")
    if not read_jsonl_rows(resolved_baseline_records_path):  # type: ignore[arg-type]
        raise ValueError("外部 baseline 正式结果记录不能为空")

    slm_wm_records = build_image_only_slm_wm_result_records(
        root_path=root_path,
        schema=schema,
        metrics_path=image_only_metrics_path,
        summary_path=image_only_summary_path,
        detection_records_path=image_only_detection_path,
        runtime_manifest_path=image_only_manifest_path,
        dataset_quality_metrics_path=resolved_dataset_quality_metrics_path,
    )
    if not slm_wm_records:
        raise ValueError("仅图像盲检正式指标未生成任何方法结果记录")
    baseline_records = build_baseline_result_records(
        root_path=root_path,
        schema=schema,
        baseline_records_path=resolved_baseline_records_path,  # type: ignore[arg-type]
        baseline_validation_report_path=resolved_baseline_validation_report_path,  # type: ignore[arg-type]
    )
    candidate_result_records = slm_wm_records + baseline_records
    allowed_result_keys = template_record_keys(template_rows)
    candidate_result_keys = [
        (
            _str_field(record, "method_id"),
            _str_field(record, "attack_id"),
            _str_field(record, "attack_family"),
            _str_field(record, "attack_name"),
            _str_field(record, "resource_profile"),
            _str_field(record, "attack_config_digest"),
        )
        for record in candidate_result_records
    ]
    if any(key not in allowed_result_keys for key in candidate_result_keys):
        raise ValueError("正式结果记录包含不属于当前 method × attack 模板的记录")
    if any(not bool(record.get("strict_formal_result_ready")) for record in candidate_result_records):
        raise ValueError("正式结果记录包含未通过严格正式门禁的记录")
    if len(candidate_result_keys) != len(set(candidate_result_keys)):
        raise ValueError("正式结果记录不得包含重复的 method × attack 模板键")
    result_records = sorted(
        filter_records_to_template(candidate_result_records, template_rows),
        key=lambda row: (
            str(row.get("method_id", "")),
            str(row.get("resource_profile", "")),
            str(row.get("attack_family", "")),
            str(row.get("attack_name", "")),
        ),
    )
    validation_report = validate_pilot_paper_result_import_rows(
        result_records,
        schema,
        evidence_root=root_path,
        require_existing_evidence=require_existing_evidence,
    )
    coverage_rows = build_template_coverage_rows(template_rows=template_rows, result_records=result_records)
    threshold_values_by_method: dict[str, set[str]] = {}
    for record in result_records:
        threshold_values_by_method.setdefault(_str_field(record, "method_id"), set()).add(
            _str_field(record, "method_threshold_digest")
        )
    expected_method_ids = set(str(value) for value in schema["method_ids"])
    if set(threshold_values_by_method) != expected_method_ids or any(
        len(values) != 1 or len(next(iter(values))) != 64
        for values in threshold_values_by_method.values()
    ):
        raise ValueError("正式结果记录必须为每个方法绑定唯一 SHA-256 阈值摘要")
    method_threshold_digest_map = {
        method_id: next(iter(values))
        for method_id, values in sorted(threshold_values_by_method.items())
    }
    result_record_set_digest = build_pilot_paper_result_record_set_digest(result_records)
    summary = build_result_summary(
        records=result_records,
        coverage_rows=coverage_rows,
        validation_report=validation_report,
        materialization_report=materialization_report,
        schema=schema,
        result_record_set_digest=result_record_set_digest,
        method_threshold_digest_map=method_threshold_digest_map,
        closure_input_lock_digest=str(
            closure_input_provenance["closure_input_lock_digest"]
        ),
        common_code_version=str(closure_input_provenance["common_code_version"]),
    )
    summary["require_existing_evidence"] = bool(require_existing_evidence)

    records_path = output_path / "pilot_paper_result_records.jsonl"
    validation_path = output_path / "pilot_paper_result_import_validation_report.json"
    coverage_path = output_path / "pilot_paper_result_template_coverage.csv"
    summary_path = output_path / "pilot_paper_result_record_summary.json"
    manifest_path = output_path / "manifest.local.json"

    records_path.write_text("".join(json_line(row) for row in result_records), encoding="utf-8")
    validation_path.write_text(stable_json_text(validation_report), encoding="utf-8")
    write_csv(
        coverage_path,
        coverage_rows,
        [
            "method_id",
            "attack_id",
            "attack_family",
            "attack_name",
            "resource_profile",
            "attack_config_digest",
            "template_covered",
            "supports_paper_claim",
        ],
    )
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")

    input_paths = [
        relative_or_absolute(path, root_path)
        for path in required_paths
        if path is not None and path.exists()
    ]
    input_paths.extend(relative_or_absolute(path, root_path) for path in packages)
    input_paths.extend(
        relative_or_absolute(closure_input_provenance[field_name], root_path)
        for field_name in (
            "closure_input_lock_path",
            "closure_input_lock_manifest_path",
        )
    )
    flattened_record_evidence_paths = tuple(
        dict.fromkeys(
            path_value
            for record in result_records
            for path_value in (
                _str_field(record, "baseline_result_source"),
                *_list_field(record, "evidence_paths"),
            )
            if path_value
        )
    )
    for path_value in flattened_record_evidence_paths:
        evidence_path = resolve_path(root_path, path_value)
        if evidence_path is None or not evidence_path.is_file():
            raise FileNotFoundError(
                f"正式 result record 证据文件不存在: {path_value}"
            )
        input_paths.append(relative_or_absolute(evidence_path, root_path))
    input_paths = list(dict.fromkeys(input_paths))
    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (records_path, validation_path, coverage_path, summary_path, manifest_path)
    )
    manifest = build_artifact_manifest(
        artifact_id="pilot_paper_fixed_fpr_result_records_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(input_paths),
        output_paths=output_paths,
        config=build_pilot_paper_result_records_manifest_config(
            result_records=result_records,
            method_threshold_digest_map=method_threshold_digest_map,
            closure_input_lock_digest=summary["closure_input_lock_digest"],
            common_code_version=summary["common_code_version"],
            validation_report=validation_report,
            template_coverage_rows=coverage_rows,
            summary=summary,
            require_existing_evidence=require_existing_evidence,
        ),
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_pilot_paper_result_records.py",
        metadata=summary,
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="写出 pilot_paper fixed-FPR 共同协议结果记录。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录; 默认写入当前论文运行子目录, 且必须位于 outputs/ 下。",
    )
    parser.add_argument(
        "--baseline-records-path",
        default=None,
        help="baseline 候选结果路径; 默认读取当前论文运行子目录。",
    )
    parser.add_argument(
        "--baseline-validation-report-path",
        default=None,
        help="baseline 候选校验报告路径; 默认读取当前论文运行子目录。",
    )
    parser.add_argument(
        "--dataset-quality-metrics-path",
        default=None,
        help="数据集质量指标路径; 默认读取当前论文运行子目录。",
    )
    parser.add_argument("--image-only-runtime-dir", default=None)
    parser.add_argument("--package-path", action="append", default=[], help="可重复传入的前序结果 zip 包。")
    parser.add_argument("--package-search-root", action="append", default=[], help="递归查找 zip 包的 Google Drive 镜像根目录。")
    parser.add_argument("--require-existing-evidence", action="store_true", help="校验 result records 中 evidence_paths 指向的文件存在。")
    parser.add_argument("--materialize-only", action="store_true", help="只物化 zip 包中的 outputs/ 条目, 不生成结果记录。")
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    manifest = write_pilot_paper_result_record_outputs(
        root=args.root,
        output_dir=args.output_dir,
        baseline_records_path=args.baseline_records_path,
        baseline_validation_report_path=args.baseline_validation_report_path,
        dataset_quality_metrics_path=args.dataset_quality_metrics_path,
        image_only_runtime_dir=args.image_only_runtime_dir,
        package_paths=args.package_path,
        package_search_roots=args.package_search_root,
        require_existing_evidence=args.require_existing_evidence,
        materialize_only=args.materialize_only,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
