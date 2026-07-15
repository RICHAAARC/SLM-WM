"""写出三个论文运行规模的协议同构与流程迁移报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.runtime.repository_environment import resolve_code_version
from paper_experiments.analysis.paper_profile_protocol_isomorphism import (
    build_paper_profile_protocol_isomorphism_report,
    build_paper_profile_protocol_records,
    validate_paper_profile_protocol_isomorphism_report,
)


DEFAULT_OUTPUT_DIR = Path("outputs/paper_profile_protocol_isomorphism")
REPORT_FILE_NAME = "paper_profile_protocol_isomorphism_report.json"
MANIFEST_FILE_NAME = "manifest.local.json"


def _read_json_object(path: Path) -> dict[str, Any]:
    """读取 probe 闭合报告并拒绝非对象 JSON。"""

    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("probe 结果闭合报告无法读取") from exc
    if not isinstance(payload, dict):
        raise ValueError("probe 结果闭合报告根节点必须是对象")
    return payload


def _resolve_under_outputs(
    root_path: Path,
    output_dir: str | Path,
) -> Path:
    """解析持久化目录, 并强制其位于当前根目录的 outputs/ 下。"""

    candidate = Path(output_dir)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (root_path / candidate).resolve()
    )
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("profile 同构报告必须写入 outputs/ 下") from exc
    return resolved


def _relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先返回相对仓库根目录的稳定 POSIX 路径。"""

    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_paper_profile_protocol_isomorphism_report(
    *,
    probe_result_closure_report_path: str | Path,
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """从真实 probe 闭合报告写出可审计同构报告与 provenance manifest。"""

    root_path = Path(root).resolve()
    source_path = Path(probe_result_closure_report_path)
    if not source_path.is_absolute():
        source_path = (root_path / source_path).resolve()
    source_report = _read_json_object(source_path)
    code_version = resolve_code_version(root_path)
    source_code_version = str(source_report.get("common_code_version", ""))
    if code_version != source_code_version:
        raise RuntimeError("当前 clean Git 提交必须与 probe 闭合报告代码身份一致")

    resolved_output_dir = _resolve_under_outputs(root_path, output_dir)
    if resolved_output_dir.exists():
        raise FileExistsError("profile 同构报告目录已存在, 不得混入旧产物")
    profile_records = build_paper_profile_protocol_records(root_path)
    report = build_paper_profile_protocol_isomorphism_report(
        source_report,
        root=root_path,
        profile_records=profile_records,
    )
    validate_paper_profile_protocol_isomorphism_report(
        report,
        root=root_path,
    )

    resolved_output_dir.mkdir(parents=True)
    report_path = resolved_output_dir / REPORT_FILE_NAME
    manifest_path = resolved_output_dir / MANIFEST_FILE_NAME
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rebuild_command = (
        "python -m scripts.write_paper_profile_protocol_isomorphism_report "
        f"--probe-result-closure-report-path {_relative_or_absolute(source_path, root_path)} "
        f"--output-dir {_relative_or_absolute(resolved_output_dir, root_path)}"
    )
    manifest = build_artifact_manifest(
        artifact_id="paper_profile_protocol_isomorphism_report",
        artifact_type="paper_protocol_governance_report",
        input_paths=(_relative_or_absolute(source_path, root_path),),
        output_paths=(_relative_or_absolute(report_path, root_path),),
        config={
            "paper_profile_protocol_registry_digest": report[
                "paper_profile_protocol_registry_digest"
            ],
            "profile_record_digests": report["profile_record_digests"],
            "probe_result_closure_binding": report[
                "probe_result_closure_binding"
            ],
        },
        code_version=code_version,
        rebuild_command=rebuild_command,
        metadata={
            "probe_workflow_closed": report["probe_workflow_closed"],
            "protocol_isomorphism_ready": report[
                "protocol_isomorphism_ready"
            ],
            "artifact_contract_isomorphic": report[
                "artifact_contract_isomorphic"
            ],
            "workflow_transfer_ready": report["workflow_transfer_ready"],
            "paper_profile_protocol_isomorphism_report_digest": report[
                "paper_profile_protocol_isomorphism_report_digest"
            ],
        },
    ).to_dict()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造服务器与 CPU 汇总环境共用的命令行参数。"""

    parser = argparse.ArgumentParser(
        description="写出 probe、pilot 与 full 的协议同构和流程迁移报告。"
    )
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--probe-result-closure-report-path",
        required=True,
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main() -> None:
    """解析参数并写出受治理报告。"""

    args = build_parser().parse_args()
    manifest = write_paper_profile_protocol_isomorphism_report(
        root=args.root,
        output_dir=args.output_dir,
        probe_result_closure_report_path=(
            args.probe_result_closure_report_path
        ),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
