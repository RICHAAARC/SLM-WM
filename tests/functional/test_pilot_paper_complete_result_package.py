"""pilot_paper 完整结果包打包脚本的轻量功能测试。"""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.write_pilot_paper_complete_result_package import (
    REQUIRED_OUTPUT_DIRS,
    write_pilot_paper_complete_result_package_outputs,
)


@pytest.mark.quick
def test_complete_result_package_collects_required_pilot_outputs(tmp_path: Path) -> None:
    """完整结果包应收集全部 pilot_paper 受治理目录并镜像到指定目录。"""

    for index, relative_dir in enumerate(REQUIRED_OUTPUT_DIRS):
        output_dir = tmp_path / relative_dir
        output_dir.mkdir(parents=True)
        (output_dir / f"sample_{index}.json").write_text(
            json.dumps({"paper_claim_scale": "pilot_paper", "index": index}, ensure_ascii=False),
            encoding="utf-8",
        )
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "paper_main_pilot_paper_prompts.txt").write_text("a pilot prompt\n", encoding="utf-8")
    workflow_dir = tmp_path / "paper_workflow"
    workflow_dir.mkdir()
    (workflow_dir / "README.md").write_text("# Paper Workflow\n", encoding="utf-8")

    drive_dir = tmp_path / "drive" / "SLM" / "pilot_paper_results" / "complete_result_package"
    manifest = write_pilot_paper_complete_result_package_outputs(
        root=tmp_path,
        drive_output_dir=str(drive_dir),
        package_search_roots=(),
    )
    output_dir = tmp_path / "outputs" / "pilot_paper_complete_result_package"
    summary = json.loads((output_dir / "pilot_paper_complete_package_summary.json").read_text(encoding="utf-8"))
    archive_path = tmp_path / summary["archive_path"]

    assert manifest["artifact_id"] == "pilot_paper_complete_result_package_manifest"
    assert summary["metadata"]["pilot_paper_complete_result_package_ready"] is True
    assert archive_path.is_file()
    assert (drive_dir / "pilot_paper_complete_result_package.zip").is_file()
    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert "outputs/pilot_paper_fixed_fpr_results/sample_17.json" in names
    assert "outputs/pilot_paper_fixed_fpr_common_protocol/sample_18.json" in names
    assert "configs/paper_main_pilot_paper_prompts.txt" in names
