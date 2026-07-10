"""通过完整重新生成、攻击前检测和图像盲检执行机制消融。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZIP_STORED, ZipFile

from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    load_completed_semantic_watermark_runtime_result,
    load_semantic_watermark_runtime_context,
    write_semantic_watermark_runtime_outputs,
)
from experiments.runtime.repository_environment import resolve_code_version
from experiments.runtime.archive_naming import utc_archive_token
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取真实运行检测记录。"""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean(values: Iterable[float]) -> float:
    """计算非空数值集合均值。"""

    resolved = tuple(values)
    return sum(resolved) / len(resolved) if resolved else 0.0


@dataclass(frozen=True)
class RuntimeRerunAblationSpec:
    """描述需要重新执行生成和检测的单个机制配置。"""

    ablation_id: str
    semantic_routing_enabled: bool = True
    null_space_enabled: bool = True
    lf_enabled: bool = True
    tail_robust_enabled: bool = True
    tail_truncation_enabled: bool = True
    attention_geometry_enabled: bool = True
    image_alignment_enabled: bool = True

    def apply(self, config: SemanticWatermarkRuntimeConfig, output_root: str) -> SemanticWatermarkRuntimeConfig:
        """把消融开关写入真实运行配置。"""

        return replace(
            config,
            semantic_routing_enabled=self.semantic_routing_enabled,
            null_space_enabled=self.null_space_enabled,
            lf_enabled=self.lf_enabled,
            tail_robust_enabled=self.tail_robust_enabled,
            tail_truncation_enabled=self.tail_truncation_enabled,
            attention_geometry_enabled=self.attention_geometry_enabled,
            image_alignment_enabled=self.image_alignment_enabled,
            output_dir=f"{output_root}/runs/{self.ablation_id}",
        )


def default_runtime_rerun_ablation_specs() -> tuple[RuntimeRerunAblationSpec, ...]:
    """返回论文机制贡献所需的真实重运行消融集合。"""

    return (
        RuntimeRerunAblationSpec("complete_method"),
        RuntimeRerunAblationSpec("shared_unrouted_budget", semantic_routing_enabled=False),
        RuntimeRerunAblationSpec("without_jacobian_null_space", null_space_enabled=False),
        RuntimeRerunAblationSpec("lf_content_only", tail_robust_enabled=False),
        RuntimeRerunAblationSpec("tail_robust_only", lf_enabled=False),
        RuntimeRerunAblationSpec("without_tail_truncation", tail_truncation_enabled=False),
        RuntimeRerunAblationSpec("without_attention_geometry", attention_geometry_enabled=False),
        RuntimeRerunAblationSpec("without_image_alignment", image_alignment_enabled=False),
    )


def run_runtime_rerun_ablations(
    base_configs: Iterable[SemanticWatermarkRuntimeConfig],
    root: str | Path = ".",
    specs: tuple[RuntimeRerunAblationSpec, ...] | None = None,
    output_dir: str = "outputs/formal_mechanism_ablation",
    minimum_prompt_count: int = 100,
    max_new_runs_per_session: int = 0,
) -> dict[str, Any]:
    """对每个 Prompt 和机制配置执行完整真实运行。

    该函数不读取既有检测分数, 不应用倍率或偏置。每个表格单元都对应一次新的
    clean/watermarked 生成、真实嵌入和最终图像检测。
    """

    root_path = Path(root).resolve()
    resolved_output = (root_path / output_dir).resolve()
    outputs_root = (root_path / "outputs").resolve()
    if resolved_output != outputs_root and outputs_root not in resolved_output.parents:
        raise ValueError("消融输出必须位于 outputs 目录")
    resolved_output.mkdir(parents=True, exist_ok=True)
    progress_path = resolved_output / "runtime_rerun_progress.json"
    resolved_specs = specs or default_runtime_rerun_ablation_specs()
    resolved_base_configs = tuple(base_configs)
    if not resolved_base_configs:
        raise ValueError("真实重运行消融至少需要一个 Prompt 配置")
    if max_new_runs_per_session < 0:
        raise ValueError("max_new_runs_per_session 不得为负")
    shared_context = None
    records = []
    resumed_run_count = 0
    new_run_count = 0
    expected_run_count = len(resolved_base_configs) * len(resolved_specs)
    for prompt_index, base_config in enumerate(resolved_base_configs):
        for spec in resolved_specs:
            run_config = spec.apply(base_config, output_dir)
            result = load_completed_semantic_watermark_runtime_result(run_config, root=root_path)
            if result is not None:
                resumed_run_count += 1
            elif max_new_runs_per_session and new_run_count >= max_new_runs_per_session:
                continue
            else:
                if shared_context is None:
                    shared_context = load_semantic_watermark_runtime_context(resolved_base_configs[0])
                result = write_semantic_watermark_runtime_outputs(
                    run_config,
                    root=root_path,
                    runtime_context=shared_context,
                )
                new_run_count += 1
            detection_records = _read_jsonl(root_path / result.detection_record_path)
            clean_negative = next(
                record
                for record in detection_records
                if record.get("sample_role") == "clean_negative" and not record.get("attack_id")
            )
            positive_source = next(
                record
                for record in detection_records
                if record.get("sample_role") == "positive_source" and not record.get("attack_id")
            )
            attacked_positive = tuple(
                record
                for record in detection_records
                if record.get("sample_role") == "positive_source" and record.get("attack_id")
            )
            attacked_negative = tuple(
                record
                for record in detection_records
                if record.get("sample_role") == "clean_negative" and record.get("attack_id")
            )
            records.append(
                {
                    "prompt_index": prompt_index,
                    "prompt_digest": build_stable_digest({"prompt": base_config.prompt}),
                    "ablation_id": spec.ablation_id,
                    "runtime_config": asdict(spec),
                    "runtime_result": result.to_dict(),
                    "generation_rerun": True,
                    "standard_attack_and_detection_rerun": True,
                    "diffusion_attack_and_detection_rerun": bool(run_config.diffusion_attacks_enabled),
                    "counterfactual_score_transform_used": False,
                    "frozen_content_threshold": run_config.content_threshold,
                    "clean_negative_positive": bool(clean_negative["evidence_positive"]),
                    "positive_source_positive": bool(positive_source["evidence_positive"]),
                    "clean_negative_content_score": float(clean_negative["content_score"]),
                    "positive_source_content_score": float(positive_source["content_score"]),
                    "attacked_positive_count": len(attacked_positive),
                    "attacked_positive_rate": _mean(
                        float(bool(record["evidence_positive"])) for record in attacked_positive
                    ),
                    "attacked_negative_count": len(attacked_negative),
                    "attacked_negative_rate": _mean(
                        float(bool(record["evidence_positive"])) for record in attacked_negative
                    ),
                    "paired_ssim": float(result.metadata.get("paired_quality", {}).get("ssim", 0.0)),
                    "supports_paper_claim": False,
                }
            )
    if len(records) != expected_run_count:
        progress = {
            "expected_run_count": expected_run_count,
            "completed_run_count": len(records),
            "remaining_run_count": expected_run_count - len(records),
            "resumed_run_count": resumed_run_count,
            "new_run_count": new_run_count,
            "max_new_runs_per_session": max_new_runs_per_session,
            "protocol_decision": "resume_required",
            "supports_paper_claim": False,
        }
        progress_path.write_text(
            json.dumps(progress, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return progress
    # 完整消融到达后清理旧的续跑状态, 使外部调度器只观察当前运行事实。
    progress_path.unlink(missing_ok=True)
    records_path = resolved_output / "runtime_rerun_records.jsonl"
    metrics_path = resolved_output / "mechanism_ablation_metrics.csv"
    delta_path = resolved_output / "mechanism_pairwise_delta.csv"
    summary_path = resolved_output / "ablation_claim_summary.json"
    manifest_path = resolved_output / "manifest.local.json"
    records_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    grouped_records: dict[str, list[dict[str, Any]]] = {}
    for spec in resolved_specs:
        grouped_records[spec.ablation_id] = [
            record for record in records if record["ablation_id"] == spec.ablation_id
        ]
    metric_rows = []
    for spec in resolved_specs:
        group = grouped_records[spec.ablation_id]
        metric_rows.append(
            {
                "ablation_id": spec.ablation_id,
                "prompt_count": len(group),
                "clean_false_positive_rate": _mean(
                    float(record["clean_negative_positive"]) for record in group
                ),
                "clean_true_positive_rate": _mean(
                    float(record["positive_source_positive"]) for record in group
                ),
                "attacked_true_positive_rate": _mean(
                    float(record["attacked_positive_rate"]) for record in group
                ),
                "attacked_false_positive_rate": _mean(
                    float(record["attacked_negative_rate"]) for record in group
                ),
                "positive_content_score_mean": _mean(
                    float(record["positive_source_content_score"]) for record in group
                ),
                "paired_ssim_mean": _mean(float(record["paired_ssim"]) for record in group),
                "generation_rerun_ready": all(record["generation_rerun"] for record in group),
                "counterfactual_score_transform_count": sum(
                    bool(record["counterfactual_score_transform_used"]) for record in group
                ),
                "metric_status": "measured_runtime_rerun",
                "supports_paper_claim": False,
            }
        )
    complete_row = next(row for row in metric_rows if row["ablation_id"] == "complete_method")
    delta_rows = [
        {
            "ablation_id": row["ablation_id"],
            "clean_true_positive_rate_delta": row["clean_true_positive_rate"] - complete_row["clean_true_positive_rate"],
            "attacked_true_positive_rate_delta": row["attacked_true_positive_rate"] - complete_row["attacked_true_positive_rate"],
            "paired_ssim_delta": row["paired_ssim_mean"] - complete_row["paired_ssim_mean"],
            "metric_status": "measured_runtime_rerun",
            "supports_paper_claim": False,
        }
        for row in metric_rows
        if row["ablation_id"] != "complete_method"
    ]
    import csv

    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    with delta_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(delta_rows[0]))
        writer.writeheader()
        writer.writerows(delta_rows)
    ablation_claim_gate_ready = (
        bool(records)
        and all(record["runtime_result"]["run_decision"] == "pass" for record in records)
        and all(record["generation_rerun"] for record in records)
        and not any(record["counterfactual_score_transform_used"] for record in records)
        and len({record["frozen_content_threshold"] for record in records}) == 1
        and all(len(grouped_records[spec.ablation_id]) >= minimum_prompt_count for spec in resolved_specs)
    )
    summary = {
        "record_count": len(records),
        "resumed_run_count": resumed_run_count,
        "new_run_count": new_run_count,
        "prompt_count": len(resolved_base_configs),
        "ablation_count": len(resolved_specs),
        "generation_rerun_count": sum(bool(record["generation_rerun"]) for record in records),
        "standard_attack_rerun_count": sum(bool(record["standard_attack_and_detection_rerun"]) for record in records),
        "diffusion_attack_rerun_count": sum(
            bool(record["diffusion_attack_and_detection_rerun"]) for record in records
        ),
        "counterfactual_score_transform_count": 0,
        "ablation_claim_gate_ready": ablation_claim_gate_ready,
        "frozen_content_threshold": records[0]["frozen_content_threshold"] if records else None,
        "minimum_prompt_count": minimum_prompt_count,
        "protocol_decision": "pass" if records else "fail",
        "supports_paper_claim": ablation_claim_gate_ready,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest = build_artifact_manifest(
        artifact_id="formal_mechanism_ablation_manifest",
        artifact_type="local_manifest",
        input_paths=(),
        output_paths=(
            records_path.relative_to(root_path).as_posix(),
            metrics_path.relative_to(root_path).as_posix(),
            delta_path.relative_to(root_path).as_posix(),
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ),
        config={
            "specs": [asdict(spec) for spec in resolved_specs],
            "record_digest": build_stable_digest(records),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 experiments.ablations.runtime_rerun.run_runtime_rerun_ablations",
        metadata={
            "protocol_decision": summary["protocol_decision"],
            "generation_rerun_required": True,
            "supports_paper_claim": summary["supports_paper_claim"],
        },
    ).to_dict()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def package_runtime_rerun_ablations(
    root: str | Path = ".",
    output_dir: str = "outputs/formal_mechanism_ablation",
) -> Path:
    """打包真实重运行消融记录和逐配置运行证据。"""

    root_path = Path(root).resolve()
    source_dir = (root_path / output_dir).resolve()
    outputs_root = (root_path / "outputs").resolve()
    if source_dir != outputs_root and outputs_root not in source_dir.parents:
        raise ValueError("消融打包目录必须位于 outputs 目录")
    if not source_dir.is_dir():
        raise FileNotFoundError("缺少真实重运行消融输出目录")
    code_version = resolve_code_version(root_path).replace("-dirty", "")
    archive_path = source_dir / f"runtime_rerun_ablation_package_{utc_archive_token()}_{code_version}.zip"
    entries = tuple(
        path
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() != ".zip"
    )
    with ZipFile(archive_path, "w", compression=ZIP_STORED, allowZip64=True) as archive:
        for path in entries:
            archive.write(path, path.relative_to(root_path).as_posix())
    return archive_path
