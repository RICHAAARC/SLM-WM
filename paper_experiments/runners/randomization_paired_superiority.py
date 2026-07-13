"""从精确随机化聚合包重建主方法与4个 baseline 的总体优势.

该入口先重算45个 method-repeat fixed-FPR 阈值, 再在每个 repeat 内从同一
原始 observation 成员构造配对 outcome. 跨重复统计只以 Prompt 为独立单位,
不会读取或平均任何单重复配对表、summary 或报告.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.protocol.attacks import attack_config_digest, default_attack_configs
from experiments.protocol.formal_randomization import formal_randomization_repeat_ids
from experiments.protocol.paper_run_config import (
    RUN_EXPECTED_PROMPT_COUNTS,
    normalize_paper_run_name,
)
from experiments.runtime.repository_environment import resolve_code_version
from main.core.digest import build_stable_digest
from paper_experiments.analysis.paired_superiority import (
    PRIMARY_BASELINE_IDS,
    build_paired_outcomes,
    canonical_attack_registry_rows,
)
from paper_experiments.analysis.randomization_paired_superiority import (
    RANDOMIZATION_PAIRED_BOOTSTRAP_RESAMPLE_COUNT,
    RANDOMIZATION_PAIRED_SUPERIORITY_FIELDNAMES,
    build_randomization_aggregate_paired_superiority_statistics,
)
from paper_experiments.runners.randomization_aggregate_provenance import (
    RandomizationAggregateProvenance,
    validate_randomization_aggregate_provenance,
)
from paper_experiments.runners.randomization_method_repeat_thresholds import (
    RandomizationMethodRepeatReconstruction,
    rebuild_randomization_method_repeat_observation_sources,
)


RANDOMIZATION_PAIRED_SUPERIORITY_OUTPUT_ROOT = (
    "outputs/randomization_paired_superiority"
)
RANDOMIZATION_PAIRED_SUPERIORITY_REPORT_SCHEMA = (
    "randomization_paired_superiority_reconstruction_report"
)


class RandomizationPairedSuperiorityRunnerError(ValueError):
    """表示聚合来源不能形成公平且可重建的跨重复总体优势结果."""


@dataclass(frozen=True)
class RandomizationPairedSuperiorityResult:
    """保存45阈值、配对 outcome、统计行、摘要与来源报告."""

    threshold_records: tuple[Mapping[str, Any], ...]
    paired_outcomes: tuple[Mapping[str, Any], ...]
    superiority_rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    report: Mapping[str, Any]


def _require_provenance(source: RandomizationAggregateProvenance) -> None:
    """拒绝路径、字典或失去 validator 冻结身份的聚合来源."""

    if not isinstance(source, RandomizationAggregateProvenance):
        raise TypeError(
            "跨重复配对重建只接受 RandomizationAggregateProvenance"
        )
    payload = source.payload
    if not all(
        (
            payload.get("randomization_aggregate_ready") is True,
            payload.get("supports_paper_claim") is False,
            str(payload.get("randomization_aggregate_digest", ""))
            == source.randomization_aggregate_digest,
            str(payload.get("common_code_version", ""))
            == source.common_code_version,
            tuple(payload.get("randomization_repeat_ids", ()))
            == formal_randomization_repeat_ids(),
        )
    ):
        raise RandomizationPairedSuperiorityRunnerError(
            "聚合来源对象未保持 validator 冻结身份"
        )


def _formal_attack_registry() -> tuple[dict[str, str], ...]:
    """从唯一攻击配置来源构造完整论文攻击 registry."""

    return canonical_attack_registry_rows(
        {
            "attack_id": config.attack_id,
            "attack_family": config.attack_family,
            "attack_name": config.attack_name,
            "resource_profile": config.resource_profile,
            "attack_config_digest": attack_config_digest(config),
        }
        for config in default_attack_configs()
        if config.enabled
        and config.resource_profile in {"full_main", "full_extra"}
    )


def _validate_threshold_source_binding(
    rebuilt: RandomizationMethodRepeatReconstruction,
) -> tuple[
    dict[tuple[str, str], Any],
    dict[tuple[str, str], dict[str, Any]],
]:
    """按 method-repeat 连接原始来源与45条重算阈值, 不依赖输入顺序."""

    source_by_key = {
        (source.randomization_repeat_id, source.method_id): source
        for source in rebuilt.method_sources
    }
    threshold_by_key = {
        (
            str(record.get("randomization_repeat_id", "")),
            str(record.get("method_id", "")),
        ): dict(record)
        for record in rebuilt.threshold_records
    }
    expected_keys = {
        (repeat_id, method_id)
        for repeat_id in formal_randomization_repeat_ids()
        for method_id in ("slm_wm", *PRIMARY_BASELINE_IDS)
    }
    if set(source_by_key) != expected_keys or set(threshold_by_key) != expected_keys:
        raise RandomizationPairedSuperiorityRunnerError(
            "原始来源或阈值未精确覆盖9重复与5方法"
        )
    lineage_pairs = (
        ("observation_source_sha256", "observation_source_sha256"),
        ("observation_archive_member", "observation_archive_member"),
        ("leaf_package_sha256", "leaf_package_sha256"),
        (
            "randomization_repeat_component_sha256",
            "randomization_repeat_component_sha256",
        ),
        (
            "randomization_aggregate_package_sha256",
            "randomization_aggregate_package_sha256",
        ),
        ("randomization_aggregate_digest", "randomization_aggregate_digest"),
        ("common_code_version", "common_code_version"),
    )
    for key in expected_keys:
        source = source_by_key[key]
        threshold = threshold_by_key[key]
        if threshold.get("fixed_fpr_threshold_ready") is not True:
            raise RandomizationPairedSuperiorityRunnerError(
                "method-repeat fixed-FPR 阈值未通过"
            )
        declared_digest = str(
            threshold.get("method_repeat_threshold_record_digest", "")
        )
        digest_payload = {
            field_name: field_value
            for field_name, field_value in threshold.items()
            if field_name != "method_repeat_threshold_record_digest"
        }
        if declared_digest != build_stable_digest(digest_payload):
            raise RandomizationPairedSuperiorityRunnerError(
                "method-repeat 阈值记录摘要无法重建"
            )
        if any(
            str(getattr(source, source_field))
            != str(threshold.get(threshold_field, ""))
            for source_field, threshold_field in lineage_pairs
        ):
            raise RandomizationPairedSuperiorityRunnerError(
                "paired observation 与 method-repeat 阈值来源不一致"
            )
    return source_by_key, threshold_by_key


def _rebuild_randomization_paired_superiority(
    source: RandomizationAggregateProvenance,
) -> RandomizationPairedSuperiorityResult:
    """从同一聚合 provenance 重建45阈值、真实配对与总体统计."""

    _require_provenance(source)
    paper_run_name = normalize_paper_run_name(
        str(source.payload.get("paper_run_name", ""))
    )
    target_fpr = float(source.payload.get("target_fpr", float("nan")))
    rebuilt = rebuild_randomization_method_repeat_observation_sources(source)
    source_by_key, threshold_by_key = _validate_threshold_source_binding(rebuilt)
    attack_registry = _formal_attack_registry()
    protocol_payload = {
        "protocol_schema": "randomization_method_repeat_paired_superiority_v1",
        "method_repeat_fixed_fpr_report_digest": rebuilt.report[
            "method_repeat_fixed_fpr_report_digest"
        ],
        "method_repeat_reconstruction_report_digest": rebuilt.reconstruction_report[
            "reconstruction_report_digest"
        ],
        "reconstruction_report_digest": rebuilt.reconstruction_report[
            "reconstruction_report_digest"
        ],
        "formal_attack_registry": list(attack_registry),
        "randomization_repeat_ids": list(formal_randomization_repeat_ids()),
    }
    protocol_digest = build_stable_digest(protocol_payload)

    paired_outcomes: list[dict[str, Any]] = []
    for repeat_id in formal_randomization_repeat_ids():
        proposed_source = source_by_key[(repeat_id, "slm_wm")]
        proposed_threshold = threshold_by_key[(repeat_id, "slm_wm")]
        for baseline_id in PRIMARY_BASELINE_IDS:
            baseline_source = source_by_key[(repeat_id, baseline_id)]
            baseline_threshold = threshold_by_key[(repeat_id, baseline_id)]
            paired_outcomes.extend(
                build_paired_outcomes(
                    proposed_source.observation_rows,
                    baseline_source.observation_rows,
                    baseline_id=baseline_id,
                    proposed_method_threshold_digest=str(
                        proposed_threshold["threshold_digest"]
                    ),
                    baseline_method_threshold_digest=str(
                        baseline_threshold["threshold_digest"]
                    ),
                    attack_registry_rows=attack_registry,
                    require_image_only_evidence=True,
                )
            )

    superiority_rows, summary = (
        build_randomization_aggregate_paired_superiority_statistics(
            paired_outcomes,
            paper_run_name=paper_run_name,
            target_fpr=target_fpr,
            protocol_digest=protocol_digest,
            attack_registry_rows=attack_registry,
        )
    )
    threshold_records = tuple(dict(row) for row in rebuilt.threshold_records)
    paired_outcome_records = tuple(dict(row) for row in paired_outcomes)
    superiority_records = tuple(dict(row) for row in superiority_rows)
    report = {
        "report_schema": RANDOMIZATION_PAIRED_SUPERIORITY_REPORT_SCHEMA,
        "paper_run_name": paper_run_name,
        "target_fpr": target_fpr,
        "randomization_aggregate_package_sha256": source.package_sha256,
        "randomization_aggregate_digest": source.randomization_aggregate_digest,
        "common_code_version": source.common_code_version,
        "method_repeat_fixed_fpr_report_digest": rebuilt.report[
            "method_repeat_fixed_fpr_report_digest"
        ],
        "method_repeat_threshold_record_count": len(threshold_records),
        "method_repeat_threshold_records_digest": build_stable_digest(
            threshold_records
        ),
        "fairness_record_count": len(rebuilt.fairness_records),
        "fairness_records_digest": rebuilt.report["fairness_records_digest"],
        "prompt_protocol_digest": rebuilt.report["prompt_protocol_digest"],
        "randomization_repeat_ids": list(formal_randomization_repeat_ids()),
        "randomization_repeat_count": len(formal_randomization_repeat_ids()),
        "formal_attack_registry": list(attack_registry),
        "formal_attack_registry_digest": build_stable_digest(list(attack_registry)),
        "paired_outcome_count": len(paired_outcome_records),
        "paired_outcome_set_digest": summary["paired_outcome_set_digest"],
        "paired_superiority_rows_digest": summary[
            "paired_superiority_rows_digest"
        ],
        "randomization_paired_superiority_summary_digest": summary[
            "randomization_paired_superiority_summary_digest"
        ],
        "protocol_digest": protocol_digest,
        "bootstrap_resample_count": RANDOMIZATION_PAIRED_BOOTSTRAP_RESAMPLE_COUNT,
        "randomization_paired_statistics_ready": True,
        "conclusion_decision": summary["conclusion_decision"],
        "supports_paper_claim": summary["supports_paper_claim"],
    }
    report["randomization_paired_superiority_report_digest"] = (
        build_stable_digest(report)
    )
    return RandomizationPairedSuperiorityResult(
        threshold_records=threshold_records,
        paired_outcomes=paired_outcome_records,
        superiority_rows=superiority_records,
        summary=summary,
        report=report,
    )


def rebuild_randomization_paired_superiority(
    source: RandomizationAggregateProvenance,
    *,
    root: str | Path = ".",
) -> RandomizationPairedSuperiorityResult:
    """在与聚合来源相同的 clean Git 提交上执行冻结正式重建."""

    _require_provenance(source)
    repository_root = Path(root).resolve()
    if resolve_code_version(repository_root) != source.common_code_version:
        raise RandomizationPairedSuperiorityRunnerError(
            "跨重复配对分析必须使用与聚合来源相同的 clean Git 提交"
        )
    return _rebuild_randomization_paired_superiority(source)


def _resolve_output_directory(
    root: Path,
    output_dir: str | Path | None,
    *,
    paper_run_name: str,
) -> Path:
    """把持久结果限制在仓库 outputs 目录内."""

    requested = (
        root / RANDOMIZATION_PAIRED_SUPERIORITY_OUTPUT_ROOT / paper_run_name
        if output_dir is None
        else Path(output_dir).expanduser()
    )
    if not requested.is_absolute():
        requested = root / requested
    resolved = requested.resolve()
    try:
        resolved.relative_to((root / "outputs").resolve())
    except ValueError as exc:
        raise RandomizationPairedSuperiorityRunnerError(
            "跨重复配对输出目录必须位于 outputs 下"
        ) from exc
    return resolved


def _file_sha256(path: Path) -> str:
    """计算持久结果文件的字节摘要."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: tuple[Mapping[str, Any], ...]) -> None:
    """以稳定字段顺序写出样本级或阈值级 JSONL."""

    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def write_randomization_paired_superiority_outputs(
    source: RandomizationAggregateProvenance,
    *,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
) -> Path:
    """全部重建完成后事务写出最小跨重复总体优势证据."""

    result = rebuild_randomization_paired_superiority(source, root=root)
    repository_root = Path(root).resolve()
    paper_run_name = str(result.report["paper_run_name"])
    destination = _resolve_output_directory(
        repository_root,
        output_dir,
        paper_run_name=paper_run_name,
    )
    if destination.exists():
        raise RandomizationPairedSuperiorityRunnerError(
            "跨重复配对正式输出目录已存在, 不得覆盖或混选运行"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}_publish_",
            dir=destination.parent,
        )
    )
    try:
        thresholds_path = temporary_directory / "method_repeat_threshold_records.jsonl"
        outcomes_path = temporary_directory / "paired_outcomes.jsonl"
        table_path = temporary_directory / "paired_superiority_table.csv"
        summary_path = temporary_directory / "paired_superiority_summary.json"
        report_path = (
            temporary_directory / "randomization_paired_superiority_report.json"
        )
        manifest_path = temporary_directory / "manifest.local.json"

        _write_jsonl(thresholds_path, result.threshold_records)
        _write_jsonl(outcomes_path, result.paired_outcomes)
        with table_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=RANDOMIZATION_PAIRED_SUPERIORITY_FIELDNAMES,
            )
            writer.writeheader()
            writer.writerows(result.superiority_rows)
        summary_path.write_text(
            json.dumps(
                dict(result.summary),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        report_path.write_text(
            json.dumps(
                dict(result.report),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        data_paths = (
            thresholds_path,
            outcomes_path,
            table_path,
            summary_path,
            report_path,
        )
        published_paths = tuple(destination / path.name for path in data_paths)
        output_sha256 = {
            published.relative_to(repository_root).as_posix(): _file_sha256(path)
            for path, published in zip(data_paths, published_paths, strict=True)
        }
        published_manifest_path = destination / manifest_path.name
        manifest = build_artifact_manifest(
            artifact_id="randomization_paired_superiority_manifest",
            artifact_type="local_manifest",
            input_paths=(source.package_path.as_posix(),),
            output_paths=tuple(output_sha256)
            + (published_manifest_path.relative_to(repository_root).as_posix(),),
            config={
                "paper_run_name": paper_run_name,
                "target_fpr": result.report["target_fpr"],
                "randomization_aggregate_package_sha256": source.package_sha256,
                "randomization_aggregate_digest": (
                    source.randomization_aggregate_digest
                ),
                "common_code_version": source.common_code_version,
                "randomization_repeat_ids": list(
                    formal_randomization_repeat_ids()
                ),
                "method_repeat_threshold_records_digest": result.report[
                    "method_repeat_threshold_records_digest"
                ],
                "paired_outcome_set_digest": result.summary[
                    "paired_outcome_set_digest"
                ],
                "paired_superiority_rows_digest": result.summary[
                    "paired_superiority_rows_digest"
                ],
                "randomization_paired_superiority_summary_digest": (
                    result.summary[
                        "randomization_paired_superiority_summary_digest"
                    ]
                ),
                "randomization_paired_superiority_report_digest": result.report[
                    "randomization_paired_superiority_report_digest"
                ],
                "bootstrap_resample_count": (
                    RANDOMIZATION_PAIRED_BOOTSTRAP_RESAMPLE_COUNT
                ),
            },
            code_version=source.common_code_version,
            rebuild_command=(
                "python -m paper_experiments.runners."
                "randomization_paired_superiority "
                f"--paper-run-name {paper_run_name} "
                f"--target-fpr {result.report['target_fpr']} "
                "--aggregate-package-path {aggregate_package_path}"
            ),
            metadata={
                "output_sha256": output_sha256,
                "randomization_paired_statistics_ready": True,
                "conclusion_decision": result.summary["conclusion_decision"],
                "supports_paper_claim": result.summary["supports_paper_claim"],
            },
        ).to_dict()
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary_directory.rename(destination)
        return destination / manifest_path.name
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    """构造可脱离 Notebook 运行的跨重复总体优势入口."""

    parser = argparse.ArgumentParser(
        description="从已验证的精确随机化聚合包重建正式总体优势统计。"
    )
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument(
        "--paper-run-name",
        required=True,
        choices=tuple(RUN_EXPECTED_PROMPT_COUNTS),
        help="论文运行层级。",
    )
    parser.add_argument(
        "--target-fpr",
        required=True,
        type=float,
        help="与聚合包冻结协议一致的目标 FPR。",
    )
    parser.add_argument(
        "--aggregate-package-path",
        required=True,
        help="精确随机化聚合来源 ZIP。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="可选输出目录, 必须位于仓库 outputs/ 下。",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """验证聚合 provenance 后重建并写出跨重复总体优势结果."""

    arguments = build_parser().parse_args(argv)
    source = validate_randomization_aggregate_provenance(
        arguments.aggregate_package_path,
        paper_run_name=arguments.paper_run_name,
        target_fpr=arguments.target_fpr,
    )
    manifest_path = write_randomization_paired_superiority_outputs(
        source,
        root=arguments.root,
        output_dir=arguments.output_dir,
    )
    print(manifest_path.as_posix())


if __name__ == "__main__":
    main()


__all__ = [
    "RANDOMIZATION_PAIRED_SUPERIORITY_OUTPUT_ROOT",
    "RandomizationPairedSuperiorityResult",
    "RandomizationPairedSuperiorityRunnerError",
    "rebuild_randomization_paired_superiority",
    "write_randomization_paired_superiority_outputs",
]
