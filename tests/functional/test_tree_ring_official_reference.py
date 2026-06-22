"""验证 Tree-Ring 官方原始环境补充表 governed import 协议。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.baselines import (
    TREE_RING_OFFICIAL_REFERENCE_PROTOCOL_NAME,
    build_tree_ring_official_reference_record,
    build_tree_ring_official_reference_schema,
    validate_tree_ring_official_reference_records,
)
from paper_workflow.colab_utils.tree_ring_official_reference import (
    TreeRingOfficialReferenceConfig,
    build_official_command,
    ensure_tree_ring_source_available,
    output_paths,
    parse_metric_text,
    write_tree_ring_official_reference_outputs,
)


@pytest.mark.quick
def test_tree_ring_official_reference_record_validates_when_all_boundaries_ready() -> None:
    """官方 legacy 复现记录满足证据边界时应通过补充表导入校验。"""

    record = build_tree_ring_official_reference_record(
        official_entrypoint="external_baseline/primary/tree_ring/source/run_tree_ring_watermark.py",
        official_repository_commit="3015283d9cf82e90b628f02ad2121bd37408ca9a",
        official_environment_profile="python3.8_diffusers0.11.1_legacy_ddim",
        baseline_result_source="outputs/tree_ring_official_reference/summary.json",
        baseline_result_source_digest="digest",
        evidence_paths=["outputs/tree_ring_official_reference/summary.json"],
        metric_values={
            "sample_count": 10,
            "positive_count": 10,
            "negative_count": 10,
            "auc": 0.9,
            "accuracy": 0.8,
            "true_positive_rate_at_one_percent_fpr": 0.7,
            "clip_score_mean": 0.3,
            "watermarked_clip_score_mean": 0.29,
        },
        ready_flags={
            "official_source_ready": True,
            "official_environment_report_ready": True,
            "official_result_summary_ready": True,
            "governed_import_ready": True,
        },
    )

    report = validate_tree_ring_official_reference_records([record])
    schema = build_tree_ring_official_reference_schema()

    assert schema["reference_protocol_name"] == TREE_RING_OFFICIAL_REFERENCE_PROTOCOL_NAME
    assert record["supplemental_table_role"] == "supplemental_method_fidelity_reference"
    assert record["main_table_eligible"] is False
    assert report["reference_import_ready"] is True
    assert report["accepted_reference_record_count"] == 1


@pytest.mark.quick
def test_tree_ring_official_reference_rejects_main_table_eligibility() -> None:
    """官方 legacy 参考记录不得伪装为主表同协议结果。"""

    record = build_tree_ring_official_reference_record(
        official_entrypoint="external_baseline/primary/tree_ring/source/run_tree_ring_watermark.py",
        official_repository_commit="3015283d9cf82e90b628f02ad2121bd37408ca9a",
        official_environment_profile="python3.8_diffusers0.11.1_legacy_ddim",
        baseline_result_source="outputs/tree_ring_official_reference/summary.json",
        baseline_result_source_digest="digest",
        evidence_paths=["outputs/tree_ring_official_reference/summary.json"],
        metric_values={
            "sample_count": 10,
            "positive_count": 10,
            "negative_count": 10,
            "auc": 0.9,
            "accuracy": 0.8,
            "true_positive_rate_at_one_percent_fpr": 0.7,
            "clip_score_mean": 0.3,
            "watermarked_clip_score_mean": 0.29,
        },
        ready_flags={
            "official_source_ready": True,
            "official_environment_report_ready": True,
            "official_result_summary_ready": True,
            "governed_import_ready": True,
        },
    )
    record["main_table_eligible"] = True

    report = validate_tree_ring_official_reference_records([record])
    reasons = {issue["reason"] for issue in report["issues"]}

    assert report["reference_import_ready"] is False
    assert "legacy_reference_must_not_enter_main_table" in reasons


@pytest.mark.quick
def test_tree_ring_official_reference_helper_imports_governed_summary(tmp_path: Path) -> None:
    """专用 helper 应能把外部官方复现 summary 转换为 governed import 记录。"""

    source_dir = tmp_path / "external_baseline" / "primary" / "tree_ring" / "source"
    source_dir.mkdir(parents=True)
    (source_dir / "run_tree_ring_watermark.py").write_text("print('tree-ring official entry')\n", encoding="utf-8")
    (source_dir / "requirements.txt").write_text("diffusers==0.11.1\ntransformers==4.23.1\n", encoding="utf-8")
    imported_summary = tmp_path / "outputs" / "tree_ring_official_reference" / "imported_summary.json"
    imported_summary.parent.mkdir(parents=True)
    imported_summary.write_text(
        json.dumps(
            {
                "sample_count": 5,
                "positive_count": 5,
                "negative_count": 5,
                "auc": 0.91,
                "accuracy": 0.82,
                "true_positive_rate_at_one_percent_fpr": 0.73,
                "clip_score_mean": 0.31,
                "watermarked_clip_score_mean": 0.3,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = TreeRingOfficialReferenceConfig(
        output_dir="outputs/tree_ring_official_reference",
        drive_output_dir=str(tmp_path / "drive"),
        source_dir="external_baseline/primary/tree_ring/source",
        sample_count=5,
        run_official_command=False,
        summary_import_path=str(imported_summary),
        require_cuda=False,
    )

    summary = write_tree_ring_official_reference_outputs(config, root=tmp_path)
    records_path = tmp_path / summary["reference_records_path"]
    validation_path = tmp_path / summary["reference_validation_path"]

    assert summary["run_decision"] == "pass"
    assert summary["sample_count"] == 5
    assert summary["governed_reference_record_count"] == 1
    assert records_path.read_text(encoding="utf-8").strip()
    assert json.loads(validation_path.read_text(encoding="utf-8"))["reference_import_ready"] is True


@pytest.mark.quick
def test_tree_ring_official_reference_cold_start_clones_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Colab 冷启动缺少官方源码时, helper 应按登记表补齐 source 缓存。"""

    registry_path = tmp_path / "external_baseline" / "source_registry.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(
            {
                "baseline_sources": [
                    {
                        "baseline_id": "tree_ring",
                        "source_dir": "external_baseline/primary/tree_ring/source",
                        "official_repository_url": "git@github.com:YuxinWenRick/tree-ring-watermark.git",
                        "official_repository_commit": "3015283d9cf82e90b628f02ad2121bd37408ca9a",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = TreeRingOfficialReferenceConfig(
        output_dir="outputs/tree_ring_official_reference",
        source_dir="external_baseline/primary/tree_ring/source",
        require_cuda=False,
    )
    paths = output_paths(tmp_path, config)

    def fake_run_command(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict[str, object]:
        if command[:2] == ["git", "clone"]:
            source_dir = Path(command[-1])
            source_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "run_tree_ring_watermark.py").write_text("print('official source')\n", encoding="utf-8")
            (source_dir / "requirements.txt").write_text("diffusers==0.11.1\n", encoding="utf-8")
        return {"command": command, "return_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("paper_workflow.colab_utils.tree_ring_official_reference.run_command", fake_run_command)

    report = ensure_tree_ring_source_available(tmp_path, config, paths)

    assert report["source_available"] is True
    assert report["source_downloaded"] is True
    assert report["official_entrypoint_ready"] is True
    assert report["official_repository_url"] == "https://github.com/YuxinWenRick/tree-ring-watermark.git"
    assert paths["source_prepare_result"].is_file()


@pytest.mark.quick
def test_tree_ring_official_reference_parses_metric_text_and_custom_python(tmp_path: Path) -> None:
    """官方日志解析与 legacy Python 可执行文件配置应保持可审计。"""

    config = TreeRingOfficialReferenceConfig(
        source_dir="external_baseline/primary/tree_ring/source",
        official_python_executable="/opt/tree-ring-legacy/bin/python",
        sample_count=5,
    )

    metrics = parse_metric_text(
        "clip_score_mean: 0.33\nw_clip_score_mean: 0.32\nauc: 0.95\nacc: 0.84\nTPR@1%FPR: 0.72\n",
        sample_count=5,
    )
    command = build_official_command(tmp_path, config)

    assert metrics["sample_count"] == 5
    assert metrics["auc"] == 0.95
    assert command[0] == "/opt/tree-ring-legacy/bin/python"
    assert "--start" in command
    assert command[command.index("--end") + 1] == "5"
