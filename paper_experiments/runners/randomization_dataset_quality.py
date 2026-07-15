"""从精确随机化聚合包重建9重复正式数据集质量结果.

该入口只接受生产 validator 返回的聚合来源. 它从包内逐字节 Prompt 契约
确定精确样本集合, 对每个 repeat 读取原始质量图像记录与 Inception 特征,
验证真实 CUDA 科学完成单元来源, 最后联合重算一次 FID/KID 三行指标.
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
from experiments.artifacts.dataset_level_quality_outputs import (
    validate_inception_feature_provenance_groups,
)
from experiments.protocol.formal_randomization import (
    formal_randomization_repeat_ids,
)
from experiments.protocol.paper_run_config import (
    RUN_EXPECTED_PROMPT_COUNTS,
    normalize_paper_run_name,
)
from experiments.runtime.repository_environment import resolve_code_version
from experiments.runtime.scientific_unit_provenance import (
    aggregate_scientific_unit_provenance,
)
from main.core.digest import build_stable_digest
from paper_experiments.analysis.formal_record_statistics import (
    DATASET_QUALITY_METRIC_FIELDNAMES,
    FORMAL_FEATURE_DEPENDENCY_PROFILE_ID,
    validate_dataset_quality_image_records,
)
from paper_experiments.analysis.randomization_dataset_quality import (
    RandomizationDatasetQualityStatistics,
    rebuild_randomization_dataset_quality_statistics,
)
from paper_experiments.runners.randomization_aggregate_provenance import (
    RandomizationAggregateProvenance,
    validate_randomization_aggregate_provenance,
)
from paper_experiments.runners.randomization_aggregate_record_workspace import (
    RandomizationAggregateRecordSource,
    open_randomization_aggregate_record_workspace,
)
from paper_experiments.runners.randomization_prompt_source_contract import (
    rebuild_randomization_prompt_source_contract,
)


RANDOMIZATION_DATASET_QUALITY_OUTPUT_ROOT = (
    "outputs/randomization_dataset_quality"
)
RANDOMIZATION_DATASET_QUALITY_REPORT_SCHEMA = (
    "randomization_dataset_quality_reconstruction_report"
)


class RandomizationDatasetQualityRunnerError(ValueError):
    """表示聚合来源不能形成真实且精确配对的数据集质量统计."""


@dataclass(frozen=True)
class RandomizationDatasetQualityResult:
    """保存跨重复成员、Prompt 分布统计、正式指标、摘要和来源报告。"""

    membership_records: tuple[Mapping[str, Any], ...]
    prompt_distribution_records: tuple[Mapping[str, Any], ...]
    metric_rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    report: Mapping[str, Any]


def _materialize_json(value: Any) -> Any:
    """把工作区冻结视图转换为科学重建器可消费的 JSON 值."""

    if isinstance(value, Mapping):
        return {
            str(field_name): _materialize_json(field_value)
            for field_name, field_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_materialize_json(item) for item in value]
    return value


def _source_record(source: RandomizationAggregateRecordSource) -> dict[str, str]:
    """保留每个 repeat 两份原始质量成员的最小来源身份."""

    return {
        "record_role": source.record_role,
        "record_member": source.record_member,
        "record_sha256": source.record_sha256,
        "leaf_package_sha256": source.leaf_package_sha256,
        "randomization_repeat_component_sha256": (
            source.randomization_repeat_component_sha256
        ),
    }


def _require_provenance(source: RandomizationAggregateProvenance) -> None:
    """拒绝路径, 字典或失去 validator 冻结身份的聚合来源."""

    if not isinstance(source, RandomizationAggregateProvenance):
        raise TypeError(
            "跨重复数据集质量重建只接受 RandomizationAggregateProvenance"
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
        raise RandomizationDatasetQualityRunnerError(
            "聚合来源对象未保持 validator 冻结身份"
        )


def _validated_scientific_provenance(
    feature_records: tuple[dict[str, Any], ...],
    *,
    expected_code_version: str,
) -> dict[str, Any]:
    """验证特征由正式 CUDA Inception 完成单元真实产生."""

    try:
        references = validate_inception_feature_provenance_groups(
            list(feature_records)
        )
        summary = aggregate_scientific_unit_provenance(
            references,
            expected_reference_count=len(feature_records),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RandomizationDatasetQualityRunnerError(
            "Inception feature 科学完成单元来源无效"
        ) from exc
    if not all(
        (
            summary.get("scientific_unit_provenance_ready") is True,
            summary.get("scientific_dependency_profile_ids")
            == [FORMAL_FEATURE_DEPENDENCY_PROFILE_ID],
            summary.get("scientific_formal_execution_commits")
            == [expected_code_version],
            bool(summary.get("scientific_torch_versions")),
            bool(summary.get("scientific_torch_cuda_versions")),
            bool(summary.get("scientific_execution_device_names")),
            bool(summary.get("scientific_cuda_device_names")),
        )
    ):
        raise RandomizationDatasetQualityRunnerError(
            "Inception feature 未绑定冻结 GPU profile, Git 提交或真实 CUDA 身份"
        )
    return summary


def _rebuild_randomization_dataset_quality(
    source: RandomizationAggregateProvenance,
) -> RandomizationDatasetQualityResult:
    """从同一聚合 provenance 重建 Prompt 成员关系与 FID/KID."""

    _require_provenance(source)
    paper_run_name = normalize_paper_run_name(
        str(source.payload.get("paper_run_name", ""))
    )
    target_fpr = float(source.payload.get("target_fpr", float("nan")))
    expected_prompt_count = RUN_EXPECTED_PROMPT_COUNTS[paper_run_name]
    membership_records: list[dict[str, Any]] = []
    feature_records: list[dict[str, Any]] = []
    repeat_source_records: list[dict[str, Any]] = []

    with open_randomization_aggregate_record_workspace(source) as workspace:
        prompt_contract = rebuild_randomization_prompt_source_contract(
            workspace,
            source,
            paper_run_name=paper_run_name,
        )
        prompt_rows = tuple(
            _materialize_json(record)
            for record in prompt_contract["prompt_rows"]
        )
        if len(prompt_rows) != expected_prompt_count:
            raise RandomizationDatasetQualityRunnerError(
                "内嵌 Prompt 来源数量未匹配论文运行层级"
            )
        expected_prompt_ids = tuple(
            str(record["prompt_id"]) for record in prompt_rows
        )
        expected_prompt_id_digest = build_stable_digest(
            sorted(expected_prompt_ids)
        )

        for repeat_id in formal_randomization_repeat_ids():
            pairs = tuple(workspace.iter_quality_feature_pairs(repeat_id))
            if len(pairs) != expected_prompt_count:
                raise RandomizationDatasetQualityRunnerError(
                    f"质量特征对数量未匹配完整 Prompt 集合: {repeat_id}"
                )
            image_records = tuple(
                _materialize_json(pair.image_record) for pair in pairs
            )
            validated_images = validate_dataset_quality_image_records(
                image_records,
                expected_pair_count=expected_prompt_count,
                expected_prompt_id_digest=expected_prompt_id_digest,
            )
            pair_by_record_id = {
                pair.dataset_quality_record_id: pair for pair in pairs
            }
            if len(pair_by_record_id) != expected_prompt_count:
                raise RandomizationDatasetQualityRunnerError(
                    f"质量特征对记录身份重复: {repeat_id}"
                )
            first_pair = pairs[0]
            if any(
                pair.image_record_source != first_pair.image_record_source
                or pair.feature_record_source
                != first_pair.feature_record_source
                for pair in pairs[1:]
            ):
                raise RandomizationDatasetQualityRunnerError(
                    f"同一 repeat 的质量记录来自多个成员: {repeat_id}"
                )
            repeat_source_records.append(
                {
                    "randomization_repeat_id": repeat_id,
                    "quality_image_source": _source_record(
                        first_pair.image_record_source
                    ),
                    "quality_feature_source": _source_record(
                        first_pair.feature_record_source
                    ),
                }
            )

            for image_record in validated_images:
                record_id = str(image_record["dataset_quality_record_id"])
                pair = pair_by_record_id.get(record_id)
                if pair is None or pair.randomization_repeat_id != repeat_id:
                    raise RandomizationDatasetQualityRunnerError(
                        "质量图像记录无法连接到同一 repeat 的 feature 对"
                    )
                source_feature = _materialize_json(
                    pair.source_feature_record
                )
                comparison_feature = _materialize_json(
                    pair.comparison_feature_record
                )
                if not all(
                    (
                        source_feature.get("dataset_quality_image_role")
                        == "source",
                        comparison_feature.get("dataset_quality_image_role")
                        == "comparison",
                        source_feature.get("image_digest")
                        == image_record["source_image_digest"],
                        comparison_feature.get("image_digest")
                        == image_record["comparison_image_digest"],
                    )
                ):
                    raise RandomizationDatasetQualityRunnerError(
                        "质量 feature 对未绑定图像记录的 source/comparison 摘要"
                    )
                membership_records.append(
                    {
                        "randomization_repeat_id": repeat_id,
                        "prompt_id": str(image_record["prompt_id"]),
                        "dataset_quality_record_id": record_id,
                        "dataset_quality_record_digest": str(
                            image_record["dataset_quality_record_digest"]
                        ),
                        "source_image_digest": str(
                            image_record["source_image_digest"]
                        ),
                        "comparison_image_digest": str(
                            image_record["comparison_image_digest"]
                        ),
                    }
                )
                feature_records.extend(
                    (source_feature, comparison_feature)
                )

    materialized_features = tuple(feature_records)
    scientific_provenance = _validated_scientific_provenance(
        materialized_features,
        expected_code_version=source.common_code_version,
    )
    statistics: RandomizationDatasetQualityStatistics = (
        rebuild_randomization_dataset_quality_statistics(
            materialized_features,
            membership_records,
            paper_run_name=paper_run_name,
            target_fpr=target_fpr,
            expected_prompt_ids=expected_prompt_ids,
        )
    )
    prompt_report = _materialize_json(prompt_contract["report"])
    report: dict[str, Any] = {
        "report_schema": RANDOMIZATION_DATASET_QUALITY_REPORT_SCHEMA,
        "paper_run_name": paper_run_name,
        "target_fpr": target_fpr,
        "randomization_aggregate_package_sha256": source.package_sha256,
        "randomization_aggregate_digest": (
            source.randomization_aggregate_digest
        ),
        "common_code_version": source.common_code_version,
        "prompt_source_contract_digest": prompt_report[
            "prompt_source_contract_digest"
        ],
        "prompt_rows_digest": prompt_report["prompt_rows_digest"],
        "randomization_repeat_ids": list(
            formal_randomization_repeat_ids()
        ),
        "repeat_source_records": repeat_source_records,
        "repeat_source_records_digest": build_stable_digest(
            repeat_source_records
        ),
        "scientific_unit_provenance": scientific_provenance,
        "quality_feature_membership_digest": statistics.summary[
            "quality_feature_membership_digest"
        ],
        "quality_feature_records_digest": statistics.summary[
            "quality_feature_records_digest"
        ],
        "randomization_dataset_quality_metric_protocol_digest": (
            statistics.summary[
                "randomization_dataset_quality_metric_protocol_digest"
            ]
        ),
        "fid_kid_metric_rows_digest": statistics.summary[
            "fid_kid_metric_rows_digest"
        ],
        "prompt_distribution_records_digest": statistics.summary[
            "prompt_distribution_records_digest"
        ],
        "randomization_dataset_quality_summary_digest": statistics.summary[
            "randomization_dataset_quality_summary_digest"
        ],
        "randomization_dataset_quality_statistics_ready": True,
        "conclusion_decision": statistics.summary["conclusion_decision"],
        "supports_paper_claim": statistics.summary["supports_paper_claim"],
    }
    report["randomization_dataset_quality_report_digest"] = (
        build_stable_digest(report)
    )
    return RandomizationDatasetQualityResult(
        membership_records=statistics.membership_records,
        prompt_distribution_records=(
            statistics.prompt_distribution_records
        ),
        metric_rows=statistics.metric_rows,
        summary=statistics.summary,
        report=report,
    )


def rebuild_randomization_dataset_quality(
    source: RandomizationAggregateProvenance,
    *,
    root: str | Path = ".",
) -> RandomizationDatasetQualityResult:
    """在与聚合来源相同的 clean Git 提交上执行正式质量重建."""

    _require_provenance(source)
    repository_root = Path(root).resolve()
    if resolve_code_version(repository_root) != source.common_code_version:
        raise RandomizationDatasetQualityRunnerError(
            "跨重复数据集质量分析必须使用与聚合来源相同的 clean Git 提交"
        )
    return _rebuild_randomization_dataset_quality(source)


def _resolve_output_directory(
    root: Path,
    output_dir: str | Path | None,
    *,
    paper_run_name: str,
) -> Path:
    """把持久结果限制在仓库 outputs 目录内."""

    requested = (
        root / RANDOMIZATION_DATASET_QUALITY_OUTPUT_ROOT / paper_run_name
        if output_dir is None
        else Path(output_dir).expanduser()
    )
    if not requested.is_absolute():
        requested = root / requested
    resolved = requested.resolve()
    try:
        resolved.relative_to((root / "outputs").resolve())
    except ValueError as exc:
        raise RandomizationDatasetQualityRunnerError(
            "跨重复数据集质量输出目录必须位于 outputs 下"
        ) from exc
    return resolved


def _file_sha256(path: Path) -> str:
    """计算持久结果文件的字节摘要."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_randomization_dataset_quality_outputs(
    source: RandomizationAggregateProvenance,
    *,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
) -> Path:
    """完成全部重建后事务写出最小正式质量证据."""

    result = rebuild_randomization_dataset_quality(source, root=root)
    repository_root = Path(root).resolve()
    paper_run_name = str(result.report["paper_run_name"])
    destination = _resolve_output_directory(
        repository_root,
        output_dir,
        paper_run_name=paper_run_name,
    )
    if destination.exists():
        raise RandomizationDatasetQualityRunnerError(
            "跨重复数据集质量正式输出目录已存在, 不得覆盖或混选运行"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}_publish_",
            dir=destination.parent,
        )
    )
    try:
        metrics_path = temporary_directory / "fid_kid_metrics.csv"
        membership_path = (
            temporary_directory / "quality_feature_membership.jsonl"
        )
        prompt_distribution_path = (
            temporary_directory / "prompt_distributional_quality_records.jsonl"
        )
        summary_path = (
            temporary_directory / "randomization_dataset_quality_summary.json"
        )
        report_path = (
            temporary_directory / "randomization_dataset_quality_report.json"
        )
        manifest_path = temporary_directory / "manifest.local.json"

        with metrics_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=DATASET_QUALITY_METRIC_FIELDNAMES,
            )
            writer.writeheader()
            writer.writerows(result.metric_rows)
        membership_path.write_text(
            "".join(
                json.dumps(dict(record), ensure_ascii=False, sort_keys=True)
                + "\n"
                for record in result.membership_records
            ),
            encoding="utf-8",
        )
        prompt_distribution_path.write_text(
            "".join(
                json.dumps(dict(record), ensure_ascii=False, sort_keys=True)
                + "\n"
                for record in result.prompt_distribution_records
            ),
            encoding="utf-8",
        )
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
            metrics_path,
            membership_path,
            prompt_distribution_path,
            summary_path,
            report_path,
        )
        published_paths = tuple(
            destination / path.name for path in data_paths
        )
        output_sha256 = {
            published.relative_to(repository_root).as_posix(): (
                _file_sha256(path)
            )
            for path, published in zip(
                data_paths,
                published_paths,
                strict=True,
            )
        }
        published_manifest_path = destination / manifest_path.name
        manifest = build_artifact_manifest(
            artifact_id="randomization_dataset_quality_manifest",
            artifact_type="local_manifest",
            input_paths=(source.package_path.as_posix(),),
            output_paths=tuple(output_sha256)
            + (
                published_manifest_path.relative_to(
                    repository_root
                ).as_posix(),
            ),
            config={
                "paper_run_name": paper_run_name,
                "target_fpr": result.report["target_fpr"],
                "randomization_aggregate_package_sha256": (
                    source.package_sha256
                ),
                "randomization_aggregate_digest": (
                    source.randomization_aggregate_digest
                ),
                "common_code_version": source.common_code_version,
                "prompt_source_contract_digest": result.report[
                    "prompt_source_contract_digest"
                ],
                "randomization_repeat_ids": list(
                    formal_randomization_repeat_ids()
                ),
                "aggregate_quality_pair_count": result.summary[
                    "aggregate_quality_pair_count"
                ],
                "quality_feature_membership_digest": result.summary[
                    "quality_feature_membership_digest"
                ],
                "quality_feature_records_digest": result.summary[
                    "quality_feature_records_digest"
                ],
                "randomization_dataset_quality_metric_protocol_digest": (
                    result.summary[
                        "randomization_dataset_quality_metric_protocol_digest"
                    ]
                ),
                "fid_kid_metric_rows_digest": result.summary[
                    "fid_kid_metric_rows_digest"
                ],
                "prompt_distribution_records_digest": result.summary[
                    "prompt_distribution_records_digest"
                ],
                "paper_quality_claim_protocol_digest": result.summary[
                    "paper_quality_claim_protocol"
                ]["paper_quality_claim_protocol_digest"],
                "randomization_dataset_quality_summary_digest": (
                    result.summary[
                        "randomization_dataset_quality_summary_digest"
                    ]
                ),
                "randomization_dataset_quality_report_digest": result.report[
                    "randomization_dataset_quality_report_digest"
                ],
            },
            code_version=source.common_code_version,
            rebuild_command=(
                "python -m paper_experiments.runners."
                "randomization_dataset_quality "
                f"--paper-run-name {paper_run_name} "
                f"--target-fpr {result.report['target_fpr']} "
                "--aggregate-package-path {aggregate_package_path}"
            ),
            metadata={
                "output_sha256": output_sha256,
                "randomization_dataset_quality_statistics_ready": True,
                "conclusion_decision": result.summary[
                    "conclusion_decision"
                ],
                "quality_subclaim_decisions": result.summary[
                    "quality_subclaim_decisions"
                ],
                "per_attack_quality_decisions": result.summary[
                    "per_attack_quality_decisions"
                ],
                "cross_attack_quality_decision": result.summary[
                    "cross_attack_quality_decision"
                ],
                "quality_preservation_claim_decision": result.summary[
                    "quality_preservation_claim_decision"
                ],
                "supports_paper_claim": result.summary[
                    "supports_paper_claim"
                ],
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
    """构造可脱离 Notebook 运行的跨重复数据集质量入口."""

    parser = argparse.ArgumentParser(
        description="从已验证的精确随机化聚合包重建正式 FID/KID。"
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
    """验证聚合 provenance 后重建并写出正式质量结果."""

    arguments = build_parser().parse_args(argv)
    source = validate_randomization_aggregate_provenance(
        arguments.aggregate_package_path,
        paper_run_name=arguments.paper_run_name,
        target_fpr=arguments.target_fpr,
    )
    manifest_path = write_randomization_dataset_quality_outputs(
        source,
        root=arguments.root,
        output_dir=arguments.output_dir,
    )
    print(manifest_path.as_posix())


if __name__ == "__main__":
    main()


__all__ = [
    "RANDOMIZATION_DATASET_QUALITY_OUTPUT_ROOT",
    "RANDOMIZATION_DATASET_QUALITY_REPORT_SCHEMA",
    "RandomizationDatasetQualityResult",
    "RandomizationDatasetQualityRunnerError",
    "rebuild_randomization_dataset_quality",
    "write_randomization_dataset_quality_outputs",
]
