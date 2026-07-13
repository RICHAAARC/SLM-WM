"""从已验证的精确9重复聚合包重建正式机制必要性结果。

该模块是跨重复消融证据的唯一生产入口。它只接受已经通过聚合来源
validator 的 ``RandomizationAggregateProvenance``，逐重复读取真实重运行
记录、检测原子、冻结阈值协议和运行 manifest，并先执行单重复原子重建。
随后以 Prompt 为统计聚类单位汇总9个注册 repeat，不读取或平均任何单重复
必要性 CSV、summary 或其他派生统计。
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
from typing import Any, Iterable, Mapping

from experiments.ablations.necessity_statistics import (
    ABLATION_NECESSITY_BOOTSTRAP_RESAMPLE_COUNT,
    ABLATION_NECESSITY_FIELDNAMES,
    build_randomization_aggregate_ablation_necessity_statistics,
)
from experiments.ablations.runtime_rerun import (
    FORMAL_RUNTIME_RERUN_ABLATION_IDS,
    FORMAL_RUNTIME_RERUN_ABLATION_SPECS,
)
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.protocol.formal_randomization import (
    formal_randomization_repeat_ids,
)
from experiments.protocol.paper_run_config import (
    RUN_EXPECTED_PROMPT_COUNTS,
    normalize_paper_run_name,
)
from experiments.runtime.repository_environment import resolve_code_version
from main.core.digest import build_stable_digest
from paper_experiments.analysis.formal_record_statistics import (
    rebuild_and_validate_ablation_runtime_aggregates,
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


RANDOMIZATION_ABLATION_NECESSITY_REPORT_SCHEMA = (
    "randomization_ablation_necessity_report"
)
RANDOMIZATION_ABLATION_NECESSITY_OUTPUT_ROOT = (
    "outputs/randomization_ablation_necessity"
)


class RandomizationAblationNecessityError(ValueError):
    """表示精确9重复不能形成唯一且公平的正式消融统计。"""


@dataclass(frozen=True)
class RandomizationAblationNecessityResult:
    """保存由原始跨重复记录重建的统计行、摘要和来源报告。"""

    rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    report: Mapping[str, Any]


def _materialize_json(value: Any) -> Any:
    """把工作区冻结视图转换为可传给科学重建器的 JSON 值。"""

    if isinstance(value, Mapping):
        return {
            str(field_name): _materialize_json(field_value)
            for field_name, field_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_materialize_json(item) for item in value]
    return value


def _source_record(source: RandomizationAggregateRecordSource) -> dict[str, str]:
    """保留统计重建所需的最小来源成员身份。"""

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
    """拒绝绕过生产 validator 构造的任意来源对象。"""

    if not isinstance(source, RandomizationAggregateProvenance):
        raise TypeError(
            "跨重复消融重建只接受 RandomizationAggregateProvenance"
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
        raise RandomizationAblationNecessityError(
            "聚合来源对象未保持 validator 冻结身份"
        )


def _rebuild_randomization_ablation_necessity(
    source: RandomizationAggregateProvenance,
    *,
    bootstrap_resample_count: int = (
        ABLATION_NECESSITY_BOOTSTRAP_RESAMPLE_COUNT
    ),
) -> RandomizationAblationNecessityResult:
    """从9份原始消融组件重建 Prompt 聚类必要性统计。"""

    _require_provenance(source)
    paper_run_name = normalize_paper_run_name(
        str(source.payload.get("paper_run_name", ""))
    )
    target_fpr = float(source.payload.get("target_fpr", float("nan")))
    expected_prompt_count = RUN_EXPECTED_PROMPT_COUNTS[paper_run_name]
    expected_ablation_specs = [
        spec.to_dict() for spec in FORMAL_RUNTIME_RERUN_ABLATION_SPECS
    ]
    expected_runtime_configs = {
        spec.ablation_id: spec.to_dict()
        for spec in FORMAL_RUNTIME_RERUN_ABLATION_SPECS
    }
    variant_ids = tuple(
        ablation_id
        for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS
        if ablation_id != "complete_method"
    )

    aggregate_runtime_records: list[dict[str, Any]] = []
    repeat_rebuild_records: list[dict[str, Any]] = []
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
            raise RandomizationAblationNecessityError(
                "内嵌 Prompt 来源数量未匹配论文运行层级"
            )
        prompt_split_by_id = {
            str(record["prompt_id"]): str(record["split"])
            for record in prompt_rows
        }
        prompt_digest_by_id = {
            str(record["prompt_id"]): str(record["prompt_digest"])
            for record in prompt_rows
        }
        prompt_index_by_id = {
            str(record["prompt_id"]): int(record["prompt_index"])
            for record in prompt_rows
        }
        expected_test_count = sum(
            split == "test" for split in prompt_split_by_id.values()
        )
        if expected_test_count <= 0:
            raise RandomizationAblationNecessityError(
                "内嵌 Prompt 来源没有正式 test 集合"
            )

        for repeat_id in formal_randomization_repeat_ids():
            runtime_source = workspace.find_source(
                randomization_repeat_id=repeat_id,
                package_family="runtime_rerun_ablation",
                record_role="ablation_runtime_record",
            )
            detection_source = workspace.find_source(
                randomization_repeat_id=repeat_id,
                package_family="runtime_rerun_ablation",
                record_role="ablation_detection_record",
            )
            protocol_source = workspace.find_source(
                randomization_repeat_id=repeat_id,
                package_family="runtime_rerun_ablation",
                record_role="ablation_frozen_protocol",
            )
            manifest_source = workspace.find_source(
                randomization_repeat_id=repeat_id,
                package_family="runtime_rerun_ablation",
                record_role="ablation_run_manifest",
            )
            runtime_records = tuple(
                _materialize_json(record)
                for record in workspace.iter_records(runtime_source)
            )
            detection_records = tuple(
                _materialize_json(record)
                for record in workspace.iter_records(detection_source)
            )
            frozen_protocols = _materialize_json(
                workspace.read_object(protocol_source)
            )
            manifest = _materialize_json(
                workspace.read_object(manifest_source)
            )
            manifest_config = manifest.get("config")
            if not isinstance(manifest_config, dict) or not all(
                (
                    manifest.get("code_version") == source.common_code_version,
                    manifest_config.get("specs") == expected_ablation_specs,
                    manifest_config.get("target_fpr") == target_fpr,
                    manifest_config.get("randomization_repeat_id") == repeat_id,
                )
            ):
                raise RandomizationAblationNecessityError(
                    f"消融 manifest 未匹配冻结方法、FPR 或 repeat: {repeat_id}"
                )
            rebuilt = rebuild_and_validate_ablation_runtime_aggregates(
                runtime_records,
                detection_records,
                frozen_protocols,
                scientific_unit_identity_records=manifest_config.get(
                    "scientific_unit_identity_records",
                    (),
                ),
                expected_ablation_ids=FORMAL_RUNTIME_RERUN_ABLATION_IDS,
                expected_prompt_split_by_id=prompt_split_by_id,
                expected_prompt_digest_by_id=prompt_digest_by_id,
                expected_prompt_index_by_id=prompt_index_by_id,
                expected_runtime_config_by_ablation_id=(
                    expected_runtime_configs
                ),
                expected_runtime_output_root=(
                    "outputs/formal_mechanism_ablation/"
                    f"{paper_run_name}/runs"
                ),
                expected_target_fpr=target_fpr,
                formal_randomization_plan=manifest_config.get(
                    "formal_randomization_plan",
                    {},
                ),
                randomization_repeat_identity=manifest_config.get(
                    "randomization_repeat_identity",
                    {},
                ),
            )
            if rebuilt.get("ablation_runtime_aggregate_rebuild_ready") is not True:
                raise RandomizationAblationNecessityError(
                    f"消融原子重建未通过: {repeat_id}"
                )
            for record in runtime_records:
                existing_repeat_id = record.get("randomization_repeat_id")
                if existing_repeat_id not in (None, "", repeat_id):
                    raise RandomizationAblationNecessityError(
                        "消融运行记录包含与来源成员冲突的 repeat 身份"
                    )
                aggregate_runtime_records.append(
                    {
                        **record,
                        "randomization_repeat_id": repeat_id,
                    }
                )
            repeat_rebuild_records.append(
                {
                    "randomization_repeat_id": repeat_id,
                    "runtime_source": _source_record(runtime_source),
                    "detection_source": _source_record(detection_source),
                    "protocol_source": _source_record(protocol_source),
                    "manifest_source": _source_record(manifest_source),
                    **rebuilt,
                }
            )

    rows, summary = (
        build_randomization_aggregate_ablation_necessity_statistics(
            aggregate_runtime_records,
            expected_ablation_ids=variant_ids,
            expected_paired_prompt_count=expected_test_count,
            bootstrap_resample_count=bootstrap_resample_count,
        )
    )
    prompt_report = _materialize_json(prompt_contract["report"])
    repeat_rebuild_digest = build_stable_digest(repeat_rebuild_records)
    report = {
        "report_schema": RANDOMIZATION_ABLATION_NECESSITY_REPORT_SCHEMA,
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
        "randomization_repeat_ids": list(formal_randomization_repeat_ids()),
        "randomization_repeat_count": len(formal_randomization_repeat_ids()),
        "repeat_rebuild_records": repeat_rebuild_records,
        "repeat_rebuild_records_digest": repeat_rebuild_digest,
        "necessity_statistic_rows_digest": summary[
            "necessity_statistic_rows_digest"
        ],
        "necessity_summary_digest": build_stable_digest(summary),
        "necessity_component_decision": summary[
            "necessity_component_decision"
        ],
        "all_mechanism_necessity_components_supported": summary[
            "all_mechanism_necessity_components_supported"
        ],
        "randomization_aggregate_statistics_ready": True,
        "supports_paper_claim": summary["supports_paper_claim"],
    }
    report["randomization_ablation_necessity_report_digest"] = (
        build_stable_digest(report)
    )
    return RandomizationAblationNecessityResult(
        rows=tuple(dict(row) for row in rows),
        summary=dict(summary),
        report=report,
    )


def rebuild_randomization_ablation_necessity(
    source: RandomizationAggregateProvenance,
    *,
    root: str | Path = ".",
) -> RandomizationAblationNecessityResult:
    """在同一 clean Git 提交上执行冻结100000次正式重建。"""

    repository_root = Path(root).resolve()
    current_code_version = resolve_code_version(repository_root)
    if current_code_version != source.common_code_version:
        raise RandomizationAblationNecessityError(
            "跨重复消融分析必须使用与聚合来源相同的 clean Git 提交"
        )
    return _rebuild_randomization_ablation_necessity(
        source,
        bootstrap_resample_count=(
            ABLATION_NECESSITY_BOOTSTRAP_RESAMPLE_COUNT
        ),
    )


def _resolve_output_directory(
    root: Path,
    output_dir: str | Path | None,
    *,
    paper_run_name: str,
) -> Path:
    """把持久结果限制在仓库 ``outputs/`` 目录内。"""

    requested = (
        root
        / RANDOMIZATION_ABLATION_NECESSITY_OUTPUT_ROOT
        / paper_run_name
        if output_dir is None
        else Path(output_dir).expanduser()
    )
    if not requested.is_absolute():
        requested = root / requested
    resolved = requested.resolve()
    try:
        resolved.relative_to((root / "outputs").resolve())
    except ValueError as exc:
        raise RandomizationAblationNecessityError(
            "跨重复消融输出目录必须位于 outputs 下"
        ) from exc
    return resolved


def _file_sha256(path: Path) -> str:
    """计算一个持久结果文件的字节摘要。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_randomization_ablation_necessity_outputs(
    source: RandomizationAggregateProvenance,
    *,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
) -> Path:
    """完成全部重建后写出最小正式统计、报告和 manifest。"""

    result = rebuild_randomization_ablation_necessity(
        source,
        root=root,
    )
    repository_root = Path(root).resolve()
    paper_run_name = str(result.report["paper_run_name"])
    destination = _resolve_output_directory(
        repository_root,
        output_dir,
        paper_run_name=paper_run_name,
    )
    if destination.exists():
        raise RandomizationAblationNecessityError(
            "跨重复消融正式输出目录已存在, 不得覆盖或混选运行"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}_publish_",
            dir=destination.parent,
        )
    )
    try:
        statistics_path = (
            temporary_directory / "mechanism_necessity_statistics.csv"
        )
        summary_path = temporary_directory / "mechanism_necessity_summary.json"
        report_path = (
            temporary_directory / "randomization_ablation_necessity_report.json"
        )
        manifest_path = temporary_directory / "manifest.local.json"

        with statistics_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=ABLATION_NECESSITY_FIELDNAMES,
            )
            writer.writeheader()
            writer.writerows(result.rows)
        summary_path.write_text(
            json.dumps(
                result.summary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        report_path.write_text(
            json.dumps(
                result.report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary_paths = (statistics_path, summary_path, report_path)
        published_paths = tuple(
            destination / path.name for path in temporary_paths
        )
        output_sha256 = {
            published_path.relative_to(repository_root).as_posix(): (
                _file_sha256(temporary_path)
            )
            for temporary_path, published_path in zip(
                temporary_paths,
                published_paths,
                strict=True,
            )
        }
        published_manifest_path = destination / manifest_path.name
        output_paths = tuple(output_sha256) + (
            published_manifest_path.relative_to(repository_root).as_posix(),
        )
        manifest = build_artifact_manifest(
            artifact_id="randomization_ablation_necessity_manifest",
            artifact_type="local_manifest",
            input_paths=(source.package_path.as_posix(),),
            output_paths=output_paths,
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
                "necessity_statistic_rows_digest": result.summary[
                    "necessity_statistic_rows_digest"
                ],
                "necessity_summary_digest": build_stable_digest(
                    result.summary
                ),
                "randomization_ablation_necessity_report_digest": (
                    result.report[
                        "randomization_ablation_necessity_report_digest"
                    ]
                ),
                "bootstrap_resample_count": (
                    ABLATION_NECESSITY_BOOTSTRAP_RESAMPLE_COUNT
                ),
            },
            code_version=source.common_code_version,
            rebuild_command=(
                "python -m paper_experiments.runners."
                "randomization_ablation_necessity "
                f"--paper-run-name {paper_run_name} "
                f"--target-fpr {result.report['target_fpr']} "
                "--aggregate-package-path {aggregate_package_path}"
            ),
            metadata={
                "output_sha256": output_sha256,
                "randomization_aggregate_statistics_ready": True,
                "necessity_component_decision": result.summary[
                    "necessity_component_decision"
                ],
                "supports_paper_claim": result.summary[
                    "supports_paper_claim"
                ],
            },
        ).to_dict()
        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary_directory.rename(destination)
        return destination / manifest_path.name
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    """构造可脱离 Notebook 运行的精确9重复消融统计入口。"""

    parser = argparse.ArgumentParser(
        description="从已验证的精确9重复聚合包重建正式机制必要性统计。"
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
        help="精确9+3聚合来源 ZIP。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="可选输出目录, 必须位于仓库 outputs/ 下。",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """先验证聚合来源, 再执行重建并写出最小论文证据。"""

    arguments = build_parser().parse_args(argv)
    source = validate_randomization_aggregate_provenance(
        arguments.aggregate_package_path,
        paper_run_name=arguments.paper_run_name,
        target_fpr=arguments.target_fpr,
    )
    manifest_path = write_randomization_ablation_necessity_outputs(
        source,
        root=arguments.root,
        output_dir=arguments.output_dir,
    )
    print(manifest_path.as_posix())


if __name__ == "__main__":
    main()


__all__ = [
    "RANDOMIZATION_ABLATION_NECESSITY_OUTPUT_ROOT",
    "RANDOMIZATION_ABLATION_NECESSITY_REPORT_SCHEMA",
    "RandomizationAblationNecessityError",
    "RandomizationAblationNecessityResult",
    "build_parser",
    "rebuild_randomization_ablation_necessity",
    "write_randomization_ablation_necessity_outputs",
]
