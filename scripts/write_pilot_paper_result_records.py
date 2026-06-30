"""写出 pilot_paper fixed-FPR 共同协议结果记录。

该脚本的作用是把方法主流程和外部 baseline 的受治理产物转换为统一
`pilot_paper_result_records.jsonl`。Notebook 只负责在 Colab 中生成上游包,
本脚本负责结果物化、模板覆盖检查、schema 校验和 manifest 写出。
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol.attacks import default_attack_configs
from experiments.protocol.pilot_paper_fixed_fpr import (
    PILOT_PAPER_MINIMUM_CLEAN_NEGATIVE_COUNT,
    PilotPaperFixedFprConfig,
    build_attack_matrix_digest,
    build_fixed_fpr_protocol_digest,
    build_paper_fixed_fpr_config,
    build_pilot_paper_attack_matrix_rows,
    build_pilot_paper_method_registry_rows,
    build_pilot_paper_prompt_split_summary,
    build_pilot_paper_result_import_schema,
    build_pilot_paper_result_import_template_rows,
    validate_pilot_paper_result_import_rows,
)
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest

CONSTRUCTION_UNIT_NAME = "pilot_paper_fixed_fpr_result_records"
DEFAULT_OUTPUT_DIR = Path("outputs/pilot_paper_fixed_fpr_results")
DEFAULT_ATTACK_FAMILY_METRICS_PATH = Path("outputs/attack_matrix/attack_family_metrics.csv")
DEFAULT_ATTACK_MANIFEST_PATH = Path("outputs/attack_matrix/attack_manifest.json")
DEFAULT_ATTACK_RECORDS_PATH = Path("outputs/attack_matrix/attack_detection_records.jsonl")
DEFAULT_REAL_ATTACK_RECORDS_PATH = Path("outputs/image_attack_evidence/formal_attack_detection_records.jsonl")
DEFAULT_BASELINE_RECORDS_PATH = Path("outputs/external_baseline_results/baseline_result_records.jsonl")
DEFAULT_BASELINE_VALIDATION_REPORT_PATH = Path(
    "outputs/external_baseline_results/baseline_result_candidate_validation_report.json"
)
DEFAULT_DATASET_QUALITY_METRICS_PATH = Path("outputs/dataset_level_quality/dataset_quality_metrics.csv")
DEFAULT_DATASET_QUALITY_SUMMARY_NAME = "dataset_quality_summary.json"
CLAIM_SUPPORTED_METHOD_STATUSES = {
    "measured",
    "measured_from_real_attacked_image_watermark_rescore_formal_protocol",
}
DIAGNOSTIC_ONLY_METHOD_STATUSES = {
    "measured_from_local_proxy",
    "measured_from_real_attacked_image_retention_proxy_formal_protocol",
    "measured_from_real_attacked_image_formal_protocol",
    "measured_from_legacy_real_attacked_image_protocol",
    "measured_from_mixed_real_and_local_proxy",
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


def clamp01(value: float) -> float:
    """把指标值裁剪到 [0, 1] 区间。"""

    return min(1.0, max(0.0, float(value)))


def bounded_confidence_interval(value: float, count: int, confidence_level: float) -> tuple[float, float]:
    """基于聚合计数构造确定性置信区间。

    上游结果当前多为聚合行, 不包含逐样本 bootstrap 原始向量。因此这里使用
    二项比例近似为重跑前治理提供稳定区间; 当后续记录包含逐样本向量时, 可在同一字段上替换为真实 bootstrap 区间。
    """

    normalized = clamp01(value)
    sample_count = max(1, int(count))
    z_value = 1.959963984540054 if confidence_level >= 0.95 else 1.6448536269514722
    margin = z_value * ((normalized * (1.0 - normalized) / sample_count) ** 0.5)
    return clamp01(normalized - margin), clamp01(normalized + margin)


def expand_package_paths(root_path: Path, package_paths: Iterable[str | Path], package_search_roots: Iterable[str | Path]) -> tuple[Path, ...]:
    """解析显式 zip 与镜像目录中的 zip 包路径。"""

    resolved: list[Path] = []
    for raw_path in package_paths:
        path = resolve_path(root_path, raw_path)
        if path and path.is_file() and path.suffix.lower() == ".zip":
            resolved.append(path)
    for raw_root in package_search_roots:
        root = resolve_path(root_path, raw_root)
        if root and root.is_dir():
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
    planned_items: list[tuple[Path, Any, Path]] = []
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
                planned_items.append((package_path, entry_info, destination))

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
    """读取数据集级质量主张门禁字段, 防止 proxy FID / KID 被误解为正式指标。"""

    dataset_quality_summary_path = dataset_quality_metrics_path.with_name(DEFAULT_DATASET_QUALITY_SUMMARY_NAME)
    summary = read_json(dataset_quality_summary_path)
    metric_rows = read_csv_rows(dataset_quality_metrics_path)
    measured_formal_names = {
        _str_field(row, "quality_metric_name")
        for row in metric_rows
        if _str_field(row, "quality_metric_name") in {"fid", "kid"} and _str_field(row, "metric_status") == "measured"
    }
    metric_names_ready = bool(summary.get("formal_fid_kid_metric_names_ready", measured_formal_names == {"fid", "kid"}))
    claim_gate_ready = bool(summary.get("formal_fid_kid_claim_gate_ready", metric_names_ready))
    proxy_ready = bool(
        summary.get(
            "dataset_level_quality_proxy_ready",
            any(_str_field(row, "metric_status") == "measured_small_sample_proxy" for row in metric_rows),
        )
    )
    blocker = _str_field(summary, "formal_fid_kid_claim_blocker")
    if not blocker and not claim_gate_ready:
        blocker = "formal_fid_kid_not_measured"
    boundary = _str_field(summary, "dataset_quality_claim_boundary")
    if not boundary:
        boundary = (
            "formal_fid_kid_measured_but_paper_claim_requires_evidence_closure"
            if claim_gate_ready
            else "dataset_quality_proxy_only_formal_fid_kid_blocked"
        )
    return {
        "formal_fid_kid_metric_names_ready": metric_names_ready,
        "formal_fid_kid_claim_gate_ready": claim_gate_ready,
        "formal_fid_kid_claim_blocker": blocker,
        "dataset_quality_proxy_only": bool(summary.get("dataset_quality_proxy_only", proxy_ready and not claim_gate_ready)),
        "dataset_quality_claim_boundary": boundary,
        "dataset_quality_summary_path": (
            relative_or_absolute(dataset_quality_summary_path, root_path) if dataset_quality_summary_path.is_file() else ""
        ),
    }


def build_common_result_fields(
    *,
    schema: Mapping[str, Any],
    method_id: str,
    attack_family: str,
    attack_name: str,
    resource_profile: str,
    baseline_result_source: str,
    baseline_result_source_digest: str,
    evidence_paths: list[str],
) -> dict[str, Any]:
    """构造所有方法共享的 pilot_paper schema 字段。"""

    return {
        "result_protocol_name": schema["result_protocol_name"],
        "result_scope": schema["result_scope"],
        "result_claim_scope": schema["result_claim_scope"],
        "method_id": method_id,
        "attack_family": attack_family,
        "attack_name": attack_name,
        "resource_profile": resource_profile,
        "prompt_protocol_name": schema["prompt_protocol_name"],
        "prompt_split_digest": schema["prompt_split_digest"],
        "attack_matrix_digest": schema["attack_matrix_digest"],
        "fixed_fpr_protocol_digest": schema["fixed_fpr_protocol_digest"],
        "target_fpr": schema["target_fpr"],
        "bootstrap_iteration_count": schema["bootstrap_iteration_count"],
        "confidence_level": schema["confidence_level"],
        "baseline_result_source": baseline_result_source,
        "baseline_result_source_digest": baseline_result_source_digest,
        "evidence_paths": evidence_paths,
        "paper_claim_scale": schema.get("paper_claim_scale", "pilot_paper"),
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

    metric_payload = {
        "positive_count": max(0, int(positive_count)),
        "negative_count": max(0, int(negative_count)),
        "attack_record_count": max(0, int(attack_record_count)),
        "supported_record_count": max(0, int(supported_record_count)),
        "true_positive_rate": clamp01(true_positive_rate),
        "false_positive_rate": clamp01(false_positive_rate),
        "clean_false_positive_rate": clamp01(clean_false_positive_rate),
        "attacked_false_positive_rate": clamp01(attacked_false_positive_rate),
        "quality_score_mean": clamp01(quality_score_mean),
        "score_retention_mean": clamp01(score_retention_mean),
    }
    ci_inputs = (
        ("true_positive_rate", metric_payload["positive_count"], "true_positive_rate_ci_low", "true_positive_rate_ci_high"),
        ("false_positive_rate", metric_payload["negative_count"], "false_positive_rate_ci_low", "false_positive_rate_ci_high"),
        (
            "clean_false_positive_rate",
            metric_payload["negative_count"],
            "clean_false_positive_rate_ci_low",
            "clean_false_positive_rate_ci_high",
        ),
        (
            "attacked_false_positive_rate",
            metric_payload["negative_count"],
            "attacked_false_positive_rate_ci_low",
            "attacked_false_positive_rate_ci_high",
        ),
        ("quality_score_mean", metric_payload["supported_record_count"], "quality_score_ci_low", "quality_score_ci_high"),
        (
            "score_retention_mean",
            metric_payload["supported_record_count"],
            "score_retention_ci_low",
            "score_retention_ci_high",
        ),
    )
    for metric_name, sample_count, low_name, high_name in ci_inputs:
        low, high = bounded_confidence_interval(metric_payload[metric_name], sample_count, confidence_level)
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
) -> dict[str, Any]:
    """补齐稳定标识、摘要和 claim 边界字段。"""

    supports_claim = (
        claim_ready
        and metric_status in CLAIM_SUPPORTED_METHOD_STATUSES
        and int(payload.get("positive_count", 0)) >= PILOT_PAPER_MINIMUM_CLEAN_NEGATIVE_COUNT
        and int(payload.get("negative_count", 0)) >= PILOT_PAPER_MINIMUM_CLEAN_NEGATIVE_COUNT
        and int(payload.get("supported_record_count", 0)) > 0
        and bool(payload.get("evidence_paths"))
    )
    payload["metric_status"] = metric_status
    payload["result_source_kind"] = source_kind
    payload["supports_paper_claim"] = supports_claim
    digest = build_stable_digest(payload)
    payload["pilot_paper_result_record_digest"] = digest
    payload["pilot_paper_result_record_id"] = f"pilot_paper_result_record_{digest[:16]}"
    return payload


def baseline_accepted_record_keys(validation_report: Mapping[str, Any]) -> set[tuple[str, str, str, str]]:
    """从主表 baseline 受治理导入报告中提取已接受记录键。"""

    keys: set[tuple[str, str, str, str]] = set()
    for row in validation_report.get("accepted_records", ()):
        keys.add(
            (
                _str_field(row, "baseline_id"),
                _str_field(row, "attack_family"),
                _str_field(row, "attack_name"),
                _str_field(row, "resource_profile"),
            )
        )
    return keys


def build_slm_wm_result_records(
    *,
    root_path: Path,
    schema: Mapping[str, Any],
    attack_family_metrics_path: Path,
    attack_manifest_path: Path,
    attack_records_path: Path,
    real_attack_records_path: Path,
    dataset_quality_metrics_path: Path,
) -> list[dict[str, Any]]:
    """从 SLM-WM 攻击矩阵聚合表构造 pilot_paper 结果记录。"""

    rows = read_csv_rows(attack_family_metrics_path)
    if not rows:
        return []
    evidence_paths = evidence_paths_for_existing(
        (
            attack_family_metrics_path,
            attack_manifest_path,
            attack_records_path,
            real_attack_records_path,
            dataset_quality_metrics_path,
            dataset_quality_metrics_path.with_name(DEFAULT_DATASET_QUALITY_SUMMARY_NAME),
        ),
        root_path,
    )
    dataset_quality_gate_fields = dataset_quality_claim_gate_fields(dataset_quality_metrics_path, root_path)
    source_digest = file_digest(attack_family_metrics_path) if attack_family_metrics_path.is_file() else build_stable_digest(rows)
    records: list[dict[str, Any]] = []
    for row in rows:
        metric_status = _str_field(row, "metric_status")
        if metric_status == "unsupported":
            continue
        payload = build_common_result_fields(
            schema=schema,
            method_id="slm_wm_current",
            attack_family=_str_field(row, "attack_family"),
            attack_name=_str_field(row, "attack_name"),
            resource_profile=_str_field(row, "resource_profile"),
            baseline_result_source=relative_or_absolute(attack_family_metrics_path, root_path),
            baseline_result_source_digest=source_digest,
            evidence_paths=evidence_paths,
        )
        payload.update(
            {
                "detector_input_access_mode": "generation_latent_trace_required",
                "blind_image_detector": False,
                "baseline_fairness_boundary": "external_baseline_comparison_requires_matching_detector_access",
                **dataset_quality_gate_fields,
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
            quality_score_mean=_float_field(row, "quality_score_proxy_mean"),
            score_retention_mean=_float_field(row, "score_retention_mean"),
            confidence_level=float(schema["confidence_level"]),
        )
        records.append(finalize_result_record(payload, metric_status=metric_status, source_kind="slm_wm_attack_matrix"))
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
    records: list[dict[str, Any]] = []
    for row in rows:
        method_id = _str_field(row, "baseline_id")
        if not method_id:
            continue
        record_key = (
            method_id,
            _str_field(row, "attack_family"),
            _str_field(row, "attack_name"),
            _str_field(row, "resource_profile", "full_main"),
        )
        evidence_paths = _list_field(row, "evidence_paths")
        if not evidence_paths and baseline_records_path.is_file():
            evidence_paths = [relative_or_absolute(baseline_records_path, root_path)]
        payload = build_common_result_fields(
            schema=schema,
            method_id=method_id,
            attack_family=_str_field(row, "attack_family"),
            attack_name=_str_field(row, "attack_name"),
            resource_profile=_str_field(row, "resource_profile", "full_main"),
            baseline_result_source=_str_field(row, "baseline_result_source", relative_or_absolute(baseline_records_path, root_path)),
            baseline_result_source_digest=_str_field(row, "baseline_result_source_digest", source_digest),
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
            quality_score_mean=_float_field(row, "quality_score_proxy_mean", _float_field(row, "quality_score_mean")),
            score_retention_mean=_float_field(row, "score_retention_mean"),
            confidence_level=float(schema["confidence_level"]),
        )
        payload["baseline_formal_import_record_accepted"] = record_key in accepted_keys
        records.append(
            finalize_result_record(
                payload,
                metric_status=_str_field(row, "metric_status", "measured"),
                source_kind="external_baseline_result",
                claim_ready=record_key in accepted_keys,
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
            _str_field(row, "attack_family"),
            _str_field(row, "attack_name"),
            _str_field(row, "resource_profile"),
        )
        for row in result_records
    }
    rows: list[dict[str, Any]] = []
    for template in template_rows:
        key = (
            _str_field(template, "method_id"),
            _str_field(template, "attack_family"),
            _str_field(template, "attack_name"),
            _str_field(template, "resource_profile"),
        )
        rows.append(
            {
                "method_id": key[0],
                "attack_family": key[1],
                "attack_name": key[2],
                "resource_profile": key[3],
                "template_covered": key in record_keys,
                "supports_paper_claim": False,
            }
        )
    return rows


def build_result_summary(
    *,
    records: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    validation_report: Mapping[str, Any],
    materialization_report: Mapping[str, Any],
    schema: Mapping[str, Any],
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
        "supports_paper_claim": bool(validation_report.get("supports_paper_claim", False)) and not missing_rows,
    }


def write_pilot_paper_result_record_outputs(
    *,
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    attack_family_metrics_path: str | Path = DEFAULT_ATTACK_FAMILY_METRICS_PATH,
    attack_manifest_path: str | Path = DEFAULT_ATTACK_MANIFEST_PATH,
    attack_records_path: str | Path = DEFAULT_ATTACK_RECORDS_PATH,
    real_attack_records_path: str | Path = DEFAULT_REAL_ATTACK_RECORDS_PATH,
    baseline_records_path: str | Path = DEFAULT_BASELINE_RECORDS_PATH,
    baseline_validation_report_path: str | Path = DEFAULT_BASELINE_VALIDATION_REPORT_PATH,
    dataset_quality_metrics_path: str | Path = DEFAULT_DATASET_QUALITY_METRICS_PATH,
    package_paths: Iterable[str | Path] = (),
    package_search_roots: Iterable[str | Path] = (),
    require_existing_evidence: bool = False,
    materialize_only: bool = False,
) -> dict[str, Any]:
    """写出 pilot_paper 共同协议结果记录和校验报告。"""

    root_path = Path(root).resolve()
    output_path = ensure_output_dir_under_outputs(root_path, output_dir)
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
    context = build_protocol_context(root_path, config)
    schema = context["schema"]
    template_rows = context["template_rows"]

    resolved_attack_family_metrics_path = resolve_path(root_path, attack_family_metrics_path)
    resolved_attack_manifest_path = resolve_path(root_path, attack_manifest_path)
    resolved_attack_records_path = resolve_path(root_path, attack_records_path)
    resolved_real_attack_records_path = resolve_path(root_path, real_attack_records_path)
    resolved_baseline_records_path = resolve_path(root_path, baseline_records_path)
    resolved_baseline_validation_report_path = resolve_path(root_path, baseline_validation_report_path)
    resolved_dataset_quality_metrics_path = resolve_path(root_path, dataset_quality_metrics_path)
    required_paths = (
        resolved_attack_family_metrics_path,
        resolved_attack_manifest_path,
        resolved_attack_records_path,
        resolved_real_attack_records_path,
        resolved_baseline_records_path,
        resolved_baseline_validation_report_path,
        resolved_dataset_quality_metrics_path,
    )
    if any(path is None for path in required_paths):
        raise ValueError("pilot_paper 结果记录输入路径不能为空")

    slm_wm_records = build_slm_wm_result_records(
        root_path=root_path,
        schema=schema,
        attack_family_metrics_path=resolved_attack_family_metrics_path,  # type: ignore[arg-type]
        attack_manifest_path=resolved_attack_manifest_path,  # type: ignore[arg-type]
        attack_records_path=resolved_attack_records_path,  # type: ignore[arg-type]
        real_attack_records_path=resolved_real_attack_records_path,  # type: ignore[arg-type]
        dataset_quality_metrics_path=resolved_dataset_quality_metrics_path,  # type: ignore[arg-type]
    )
    baseline_records = build_baseline_result_records(
        root_path=root_path,
        schema=schema,
        baseline_records_path=resolved_baseline_records_path,  # type: ignore[arg-type]
        baseline_validation_report_path=resolved_baseline_validation_report_path,  # type: ignore[arg-type]
    )
    result_records = sorted(
        slm_wm_records + baseline_records,
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
    summary = build_result_summary(
        records=result_records,
        coverage_rows=coverage_rows,
        validation_report=validation_report,
        materialization_report=materialization_report,
        schema=schema,
    )

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
        ["method_id", "attack_family", "attack_name", "resource_profile", "template_covered", "supports_paper_claim"],
    )
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")

    input_paths = [
        relative_or_absolute(path, root_path)
        for path in required_paths
        if path is not None and path.exists()
    ]
    input_paths.extend(relative_or_absolute(path, root_path) for path in packages)
    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (records_path, validation_path, coverage_path, summary_path, manifest_path)
    )
    manifest = build_artifact_manifest(
        artifact_id="pilot_paper_fixed_fpr_result_records_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(input_paths),
        output_paths=output_paths,
        config={
            "result_record_digest": build_stable_digest(result_records),
            "validation_report_digest": build_stable_digest(validation_report),
            "template_coverage_digest": build_stable_digest(coverage_rows),
            "summary_digest": build_stable_digest(summary),
            "require_existing_evidence": require_existing_evidence,
        },
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
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--attack-family-metrics-path", default=str(DEFAULT_ATTACK_FAMILY_METRICS_PATH))
    parser.add_argument("--attack-manifest-path", default=str(DEFAULT_ATTACK_MANIFEST_PATH))
    parser.add_argument("--attack-records-path", default=str(DEFAULT_ATTACK_RECORDS_PATH))
    parser.add_argument("--real-attack-records-path", default=str(DEFAULT_REAL_ATTACK_RECORDS_PATH))
    parser.add_argument("--baseline-records-path", default=str(DEFAULT_BASELINE_RECORDS_PATH))
    parser.add_argument("--baseline-validation-report-path", default=str(DEFAULT_BASELINE_VALIDATION_REPORT_PATH))
    parser.add_argument("--dataset-quality-metrics-path", default=str(DEFAULT_DATASET_QUALITY_METRICS_PATH))
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
        attack_family_metrics_path=args.attack_family_metrics_path,
        attack_manifest_path=args.attack_manifest_path,
        attack_records_path=args.attack_records_path,
        real_attack_records_path=args.real_attack_records_path,
        baseline_records_path=args.baseline_records_path,
        baseline_validation_report_path=args.baseline_validation_report_path,
        dataset_quality_metrics_path=args.dataset_quality_metrics_path,
        package_paths=args.package_path,
        package_search_roots=args.package_search_root,
        require_existing_evidence=args.require_existing_evidence,
        materialize_only=args.materialize_only,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
