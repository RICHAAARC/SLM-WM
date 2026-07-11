"""写出主表 external baseline 证据边界审计产物。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.protocol.attacks import default_attack_configs
from experiments.protocol.fixed_fpr_observation_audit import audit_fixed_fpr_observation_threshold
from experiments.protocol.paper_run_config import PaperRunConfig, build_paper_run_config
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.protocol.splits import apply_split_assignments
from experiments.runtime.repository_environment import resolve_code_version
from main.core.digest import build_stable_digest
from paper_experiments.baselines.method_faithful_observation_collection import (
    DEFAULT_METHOD_FAITHFUL_COLLECTION_ROOT,
    MethodFaithfulObservationSource,
    canonical_prompt_protocol_digest,
    file_sha256,
    load_method_faithful_observation_collection,
)
from paper_experiments.baselines.observation_io import load_baseline_observation_rows
from paper_experiments.baselines.primary_evidence import (
    build_primary_baseline_evidence_records,
    build_primary_baseline_evidence_summary,
    load_optional_json,
)


DEFAULT_OUTPUT_ROOT = Path("outputs/primary_baseline_evidence")
DEFAULT_SOURCE_REGISTRY_PATH = Path("external_baseline/source_registry.json")
DEFAULT_COLLECTION_ROOT = DEFAULT_METHOD_FAITHFUL_COLLECTION_ROOT
DEFAULT_T2SMARK_FORMAL_OUTPUT_ROOT = Path("outputs/t2smark_formal_reproduction")


@dataclass(frozen=True)
class T2SMarkFormalEvidenceSource:
    """保存独立 T2SMark formal runner 的已校验证据。"""

    observations: tuple[dict[str, Any], ...]
    command_result: dict[str, Any]
    evidence_paths: tuple[Path, ...]
    prompt_plan_path: Path
    evidence_digest: str


def stable_json_text(value: Any) -> str:
    """以稳定顺序序列化 JSON。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: dict[str, Any]) -> str:
    """把单条记录写成稳定 JSONL 行。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def resolve_path(root_path: Path, path: str | Path) -> Path:
    """把相对路径解析到仓库根目录。"""

    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()


def ensure_output_dir_under_outputs(root_path: Path, output_dir: str | Path) -> Path:
    """确保输出目录位于 outputs/ 下。"""

    resolved = resolve_path(root_path, output_dir)
    try:
        resolved.relative_to((root_path / "outputs").resolve())
    except ValueError as exc:
        raise ValueError(f"主表 baseline 证据输出目录必须位于 outputs/ 下: {resolved}") from exc
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先记录相对仓库根目录的路径。"""

    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_json_object(path: Path, role: str) -> dict[str, Any]:
    """读取正式证据 JSON 对象，缺失或类型错误时立即停止。"""

    if not path.is_file():
        raise FileNotFoundError(f"{role} 不存在: {path.as_posix()}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"{role} 必须是 JSON 对象: {path.as_posix()}")
    return dict(payload)


def read_json_array(path: Path, role: str) -> list[dict[str, Any]]:
    """读取正式证据 JSON 对象数组。"""

    if not path.is_file():
        raise FileNotFoundError(f"{role} 不存在: {path.as_posix()}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise TypeError(f"{role} 必须是 JSON 对象数组: {path.as_posix()}")
    return [dict(row) for row in payload]


def _command_option(command: list[str], option: str) -> str:
    """从显式 argv 中读取必需选项值。"""

    if option not in command:
        raise ValueError(f"T2SMark formal adapter 命令缺少 {option}")
    option_index = command.index(option)
    if option_index + 1 >= len(command):
        raise ValueError(f"T2SMark formal adapter 命令的 {option} 缺少值")
    return str(command[option_index + 1])


def _required_attack_names() -> set[str]:
    """返回主表正式攻击矩阵要求的全部攻击名称。"""

    return {
        attack.attack_name
        for attack in default_attack_configs()
        if attack.enabled and attack.resource_profile in {"full_main", "full_extra"}
    }


def _observation_protocol_readiness(
    *,
    rows: tuple[dict[str, Any], ...],
    prompt_rows: list[dict[str, Any]],
    paper_run: PaperRunConfig,
    formal_evidence_paths_ready: bool,
) -> dict[str, bool]:
    """从 observation 实体重算 Prompt、fixed-FPR 和完整攻击覆盖。"""

    prompt_by_id = {str(row.get("prompt_id", "")): dict(row) for row in prompt_rows}
    expected_prompt_ids = set(prompt_by_id)
    expected_test_prompt_ids = {
        prompt_id
        for prompt_id, row in prompt_by_id.items()
        if str(row.get("split", "")) == "test"
    }
    expected_calibration_count = sum(
        str(row.get("split", "")) == "calibration" for row in prompt_rows
    )
    clean_by_role: dict[str, list[dict[str, Any]]] = {
        "clean_negative": [],
        "positive_source": [],
    }
    for row in rows:
        sample_role = str(row.get("sample_role", ""))
        if str(row.get("attack_family", "")) == "clean" and sample_role in clean_by_role:
            clean_by_role[sample_role].append(row)
    prompt_protocol_ready = bool(prompt_rows) and len(prompt_rows) == paper_run.prompt_count
    for role_rows in clean_by_role.values():
        role_prompt_ids = [str(row.get("prompt_id", "")) for row in role_rows]
        prompt_protocol_ready = bool(
            prompt_protocol_ready
            and len(role_prompt_ids) == len(expected_prompt_ids)
            and set(role_prompt_ids) == expected_prompt_ids
            and len(role_prompt_ids) == len(set(role_prompt_ids))
            and all(
                str(row.get("split", ""))
                == str(prompt_by_id[str(row.get("prompt_id", ""))].get("split", ""))
                for row in role_rows
                if str(row.get("prompt_id", "")) in prompt_by_id
            )
        )
    threshold_audit = audit_fixed_fpr_observation_threshold(
        rows,
        target_fpr=paper_run.target_fpr,
        expected_calibration_negative_count=expected_calibration_count,
    )
    required_attacks = _required_attack_names()
    attacked_keys = [
        (
            str(row.get("prompt_id", "")),
            str(row.get("attack_name") or row.get("attack_condition") or ""),
            str(row.get("sample_role", "")),
        )
        for row in rows
        if str(row.get("sample_role", "")).startswith("attacked_")
    ]
    expected_attacked_keys = {
        (prompt_id, attack_name, sample_role)
        for prompt_id in expected_test_prompt_ids
        for attack_name in required_attacks
        for sample_role in ("attacked_negative", "attacked_positive")
    }
    attack_matrix_ready = bool(expected_attacked_keys) and set(attacked_keys) == expected_attacked_keys and len(
        attacked_keys
    ) == len(expected_attacked_keys)
    return {
        "paper_run_prompt_protocol_ready": prompt_protocol_ready,
        "fixed_fpr_baseline_calibration_ready": threshold_audit.fixed_fpr_ready,
        "attack_matrix_baseline_detection_ready": attack_matrix_ready,
        "formal_evidence_paths_ready": formal_evidence_paths_ready,
    }


def _validate_t2smark_budget(
    manifest: dict[str, Any],
    command_result: dict[str, Any],
    paper_run: PaperRunConfig,
) -> None:
    """核验 T2SMark 独立 formal runner 和 adapter 的 20/20/4.5 预算。"""

    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("T2SMark formal manifest 缺少 config")
    config_ready = (
        str(config.get("prompt_set", "")) == paper_run.prompt_set
        and str(config.get("model_id", "")) == "stabilityai/stable-diffusion-3.5-medium"
        and int(config.get("num_inference_steps", -1)) == paper_run.inference_steps == 20
        and int(config.get("num_inversion_steps", -1)) == paper_run.inference_steps == 20
        and math.isclose(float(config.get("guidance_scale", float("nan"))), 4.5, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(config.get("target_fpr", float("nan"))), paper_run.target_fpr, rel_tol=0.0, abs_tol=1e-12)
    )
    if not config_ready:
        raise ValueError("T2SMark formal manifest 与当前论文预算不一致")
    raw_command = command_result.get("command")
    if not isinstance(raw_command, list):
        raise ValueError("T2SMark formal adapter command result 缺少显式 argv")
    command = [str(value) for value in raw_command]
    command_ready = (
        int(_command_option(command, "--num-inference-steps")) == 20
        and int(_command_option(command, "--num-inversion-steps")) == 20
        and math.isclose(float(_command_option(command, "--guidance-scale")), 4.5, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(
            float(_command_option(command, "--target-fpr")),
            paper_run.target_fpr,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    if not command_ready:
        raise ValueError("T2SMark formal adapter 命令与当前论文预算不一致")


def load_t2smark_formal_evidence(
    root_path: Path,
    output_dir: Path,
    paper_run: PaperRunConfig,
) -> T2SMarkFormalEvidenceSource:
    """读取并独立核验 T2SMark 唯一 formal runner 的正式证据。"""

    paths = {
        "summary": output_dir / "t2smark_formal_reproduction_summary.json",
        "manifest": output_dir / "t2smark_formal_reproduction_manifest.local.json",
        "prompt_plan": output_dir / "t2smark_formal_prompt_plan.json",
        "image_pairs": output_dir / "t2smark_formal_image_pairs.json",
        "official_results": output_dir / "t2smark_official" / f"t2smark_sd35_medium_{paper_run.run_name}" / "results.json",
        "adapter_observations": output_dir / "t2smark_adapter" / "baseline_observations.json",
        "adapter_manifest": output_dir / "t2smark_adapter" / "t2smark_slm_adapter_manifest.json",
        "adapter_command_result": output_dir / "t2smark_formal_adapter_command_result.json",
        "validation_report": output_dir / "t2smark_formal_import_validation_report.json",
        "pair_quality_summary": output_dir / "t2smark_formal_strict_pair_quality_summary.json",
    }
    missing_paths = [path for path in paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "T2SMark formal evidence 缺少文件: " + ", ".join(path.as_posix() for path in missing_paths)
        )
    summary = read_json_object(paths["summary"], "T2SMark formal summary")
    manifest = read_json_object(paths["manifest"], "T2SMark formal manifest")
    prompt_rows = read_json_array(paths["prompt_plan"], "T2SMark formal prompt plan")
    adapter_manifest = read_json_object(paths["adapter_manifest"], "T2SMark adapter manifest")
    command_result = read_json_object(paths["adapter_command_result"], "T2SMark adapter command result")
    validation_report = read_json_object(paths["validation_report"], "T2SMark formal import validation")
    pair_quality_summary = read_json_object(paths["pair_quality_summary"], "T2SMark pair quality summary")
    observations = tuple(load_baseline_observation_rows(paths["adapter_observations"]))

    summary_ready = (
        summary.get("run_decision") == "pass"
        and summary.get("t2smark_formal_reproduction_ready") is True
        and summary.get("paper_run_prompt_protocol_ready") is True
        and summary.get("t2smark_formal_attack_ready") is True
        and summary.get("t2smark_strict_pair_quality_ready") is True
        and summary.get("formal_import_validation_ready") is True
        and str(summary.get("paper_claim_scale", "")) == paper_run.run_name
        and int(summary.get("selected_prompt_count", -1)) == paper_run.prompt_count
        and math.isclose(float(summary.get("target_fpr", float("nan"))), paper_run.target_fpr, rel_tol=0.0, abs_tol=1e-12)
    )
    if not summary_ready:
        raise ValueError("T2SMark formal summary 未通过当前论文证据门禁")
    manifest_ready = (
        manifest.get("artifact_id") == "t2smark_formal_reproduction_manifest"
        and manifest.get("metadata", {}).get("run_decision") == "pass"
        and manifest.get("metadata", {}).get("t2smark_formal_reproduction_ready") is True
        and validation_report.get("formal_import_validation_ready") is True
        and pair_quality_summary.get("strict_pair_quality_ready") is True
    )
    if not manifest_ready:
        raise ValueError("T2SMark formal manifest 或受绑定报告未通过")
    expected_records = apply_split_assignments(
        build_prompt_records(
            paper_run.prompt_set,
            read_prompt_file(root_path / paper_run.prompt_file),
        )
    )
    expected_prompt_digest = canonical_prompt_protocol_digest(
        [
            {
                "prompt_id": record.prompt_id,
                "prompt_index": record.prompt_index,
                "prompt_set": record.prompt_set,
                "split": record.split,
                "prompt_text": record.prompt_text,
                "prompt_digest": record.prompt_digest,
            }
            for record in expected_records
        ]
    )
    actual_prompt_digest = canonical_prompt_protocol_digest(prompt_rows)
    prompt_report = summary.get("metadata", {}).get("prompt_report", {})
    if (
        actual_prompt_digest != expected_prompt_digest
        or str(prompt_report.get("prompt_protocol_digest", "")) != actual_prompt_digest
    ):
        raise ValueError("T2SMark formal Prompt 摘要与当前规范 Prompt 不一致")
    adapter_ready = (
        adapter_manifest.get("baseline_id") == "t2smark"
        and adapter_manifest.get("adapter_status") == "sd35_native_result_adapter_ready"
        and int(adapter_manifest.get("observation_count", -1)) == len(observations)
        and adapter_manifest.get("strict_pair_quality_ready") is True
        and not adapter_manifest.get("missing_result_indices")
        and set(adapter_manifest.get("formal_attack_names", [])) == _required_attack_names()
        and str(adapter_manifest.get("threshold_source", "")) == "calibration_clean_negative_conformal"
        and int(command_result.get("return_code", -1)) == 0
        and all(str(row.get("baseline_id", "")) == "t2smark" for row in observations)
    )
    if not adapter_ready:
        raise ValueError("T2SMark formal adapter evidence 未通过")
    _validate_t2smark_budget(manifest, command_result, paper_run)
    evidence_paths = tuple(paths.values())
    evidence_digest = build_stable_digest(
        [(relative_or_absolute(path, root_path), file_sha256(path)) for path in evidence_paths]
    )
    normalized_command_result = {
        **command_result,
        "baseline_id": "t2smark",
        "observation_count": len(observations),
        "output_path": relative_or_absolute(paths["adapter_observations"], root_path),
    }
    return T2SMarkFormalEvidenceSource(
        observations=tuple(dict(row) for row in observations),
        command_result=normalized_command_result,
        evidence_paths=evidence_paths,
        prompt_plan_path=paths["prompt_plan"],
        evidence_digest=evidence_digest,
    )


def _resolve_collection_path(collection_root: Path, value: Any, field_name: str) -> Path:
    """解析 transfer manifest 内路径并拒绝目录越界。"""

    text = str(value or "").strip()
    relative_path = PurePosixPath(text)
    if not text or relative_path.is_absolute() or any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ValueError(f"transfer manifest 的 {field_name} 不是规范 collection 相对路径")
    resolved = (collection_root / Path(*relative_path.parts)).resolve()
    try:
        resolved.relative_to(collection_root.resolve())
    except ValueError as exc:
        raise ValueError(f"transfer manifest 的 {field_name} 越出 collection") from exc
    return resolved


def load_command_results_from_sources(
    sources: tuple[MethodFaithfulObservationSource, ...],
    collection_root: Path,
) -> tuple[list[dict[str, Any]], tuple[Path, ...]]:
    """读取并核验三个 transfer manifest 绑定的 command results。"""

    all_rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    for source in sources:
        manifest = source.transfer_manifest
        command_path = _resolve_collection_path(
            collection_root,
            manifest.get("baseline_command_results_path"),
            "baseline_command_results_path",
        )
        if not command_path.is_file():
            raise FileNotFoundError(f"baseline command result 不存在: {command_path}")
        expected_digest = str(manifest.get("baseline_command_results_sha256", ""))
        if len(expected_digest) != 64 or file_sha256(command_path) != expected_digest:
            raise ValueError(f"{source.baseline_id} command result 摘要与 transfer manifest 不一致")
        payload = json.loads(command_path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, list) or len(payload) != 1:
            raise ValueError(f"{source.baseline_id} 必须且只能包含一条 command result")
        row = dict(payload[0])
        if str(row.get("baseline_id", "")) != source.baseline_id:
            raise ValueError(f"{source.baseline_id} command result 方法身份不一致")
        if int(row.get("return_code", 1)) != 0:
            raise RuntimeError(f"{source.baseline_id} command result 未成功")
        if int(row.get("observation_count", -1)) != len(source.rows):
            raise ValueError(f"{source.baseline_id} command result observation 数量不一致")
        all_rows.append(row)
        paths.append(command_path)
    return all_rows, tuple(paths)


def write_primary_baseline_evidence_outputs(
    *,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
    source_registry_path: str | Path = DEFAULT_SOURCE_REGISTRY_PATH,
    collection_root: str | Path | None = None,
    t2smark_formal_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """从三个 exact-set source 和独立 T2SMark formal source 写出证据记录。"""

    root_path = Path(root).resolve()
    paper_run = build_paper_run_config(root_path)
    resolved_output_dir = ensure_output_dir_under_outputs(
        root_path,
        output_dir or DEFAULT_OUTPUT_ROOT / paper_run.run_name,
    )
    resolved_source_registry_path = resolve_path(root_path, source_registry_path)
    resolved_collection_root = resolve_path(
        root_path,
        collection_root or DEFAULT_COLLECTION_ROOT / paper_run.run_name,
    )
    resolved_t2smark_output_dir = resolve_path(
        root_path,
        t2smark_formal_output_dir or DEFAULT_T2SMARK_FORMAL_OUTPUT_ROOT / paper_run.run_name,
    )
    source_registry = load_optional_json(resolved_source_registry_path) or {}
    sources = load_method_faithful_observation_collection(
        resolved_collection_root,
        project_root=root_path,
    )
    t2smark_source = load_t2smark_formal_evidence(
        root_path,
        resolved_t2smark_output_dir,
        paper_run,
    )
    observation_rows = [dict(row) for source in sources for row in source.rows]
    observation_rows.extend(dict(row) for row in t2smark_source.observations)
    command_results, command_result_paths = load_command_results_from_sources(
        sources,
        resolved_collection_root,
    )
    command_results.append(t2smark_source.command_result)
    formal_evidence_paths_by_baseline = {
        source.baseline_id: [
            relative_or_absolute(path, root_path)
            for path in (
                source.observations_path,
                source.transfer_manifest_path,
                source.prompt_plan_path,
                source.adapter_manifest_path,
                source.execution_manifest_path,
                command_result_paths[index],
            )
        ]
        for index, source in enumerate(sources)
    }
    formal_evidence_paths_by_baseline["t2smark"] = [
        relative_or_absolute(path, root_path) for path in t2smark_source.evidence_paths
    ]
    readiness_by_baseline: dict[str, dict[str, bool]] = {}
    for source in sources:
        prompt_rows = read_json_array(source.prompt_plan_path, f"{source.baseline_id} prompt plan")
        readiness_by_baseline[source.baseline_id] = _observation_protocol_readiness(
            rows=source.rows,
            prompt_rows=prompt_rows,
            paper_run=paper_run,
            formal_evidence_paths_ready=True,
        )
    t2smark_prompt_rows = read_json_array(t2smark_source.prompt_plan_path, "T2SMark prompt plan")
    readiness_by_baseline["t2smark"] = _observation_protocol_readiness(
        rows=t2smark_source.observations,
        prompt_rows=t2smark_prompt_rows,
        paper_run=paper_run,
        formal_evidence_paths_ready=True,
    )
    if not all(readiness_by_baseline["t2smark"].values()):
        raise ValueError("T2SMark formal observations 未通过独立 Prompt、fixed-FPR 或攻击覆盖审计")
    records = build_primary_baseline_evidence_records(
        source_registry=source_registry,
        command_results=command_results,
        observation_rows=observation_rows,
        protocol_readiness_by_baseline=readiness_by_baseline,
        formal_evidence_paths_by_baseline=formal_evidence_paths_by_baseline,
    )
    summary = build_primary_baseline_evidence_summary(records)
    summary.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "paper_claim_scale": paper_run.run_name,
            "target_fpr": paper_run.target_fpr,
            "source_registry_path": relative_or_absolute(resolved_source_registry_path, root_path),
            "collection_root": relative_or_absolute(resolved_collection_root, root_path),
            "input_baseline_ids": [source.baseline_id for source in sources] + ["t2smark"],
            "input_observation_count": len(observation_rows),
            "input_command_result_count": len(command_results),
            "t2smark_formal_output_dir": relative_or_absolute(resolved_t2smark_output_dir, root_path),
            "t2smark_formal_evidence_digest": t2smark_source.evidence_digest,
        }
    )

    records_path = resolved_output_dir / "primary_baseline_evidence_records.jsonl"
    summary_path = resolved_output_dir / "primary_baseline_evidence_summary.json"
    manifest_path = resolved_output_dir / "manifest.local.json"
    records_path.write_text("".join(json_line(row) for row in records), encoding="utf-8")
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")
    input_paths = [relative_or_absolute(resolved_source_registry_path, root_path)]
    input_paths.extend(
        relative_or_absolute(path, root_path)
        for source in sources
        for path in (
            source.observations_path,
            source.transfer_manifest_path,
            source.prompt_plan_path,
            source.adapter_manifest_path,
            source.execution_manifest_path,
        )
    )
    input_paths.extend(relative_or_absolute(path, root_path) for path in command_result_paths)
    input_paths.extend(relative_or_absolute(path, root_path) for path in t2smark_source.evidence_paths)
    output_paths = (
        relative_or_absolute(records_path, root_path),
        relative_or_absolute(summary_path, root_path),
        relative_or_absolute(manifest_path, root_path),
    )
    manifest = build_artifact_manifest(
        artifact_id="primary_baseline_evidence_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(input_paths),
        output_paths=output_paths,
        config={
            "records_digest": summary["primary_baseline_evidence_records_digest"],
            "primary_baseline_evidence_records_digest": summary[
                "primary_baseline_evidence_records_digest"
            ],
            "summary_digest": build_stable_digest(summary),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_primary_baseline_evidence_outputs.py",
        metadata=summary,
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="写出主表 external baseline 证据边界审计产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录; 默认写入当前论文运行子目录, 且必须位于 outputs/ 下。",
    )
    parser.add_argument("--source-registry-path", default=str(DEFAULT_SOURCE_REGISTRY_PATH))
    parser.add_argument(
        "--collection-root",
        default=None,
        help="方法忠实 baseline 物化根目录; 默认读取当前论文运行子目录。",
    )
    parser.add_argument(
        "--t2smark-formal-output-dir",
        default=None,
        help="T2SMark formal 输出目录; 默认读取当前论文运行子目录。",
    )
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    manifest = write_primary_baseline_evidence_outputs(
        root=args.root,
        output_dir=args.output_dir,
        source_registry_path=args.source_registry_path,
        collection_root=args.collection_root,
        t2smark_formal_output_dir=args.t2smark_formal_output_dir,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
