"""验证 Colab Notebook 入口契约。"""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from paper_workflow.colab_utils.sd_runtime_cold_start import package_probe_outputs
from tools.harness.lib.naming_rules import is_allowed_file_name


NOTEBOOK_PATH = Path("paper_workflow/sd_runtime_cold_start_probe.ipynb")


@pytest.mark.constraint
def test_ipynb_name_is_allowed_when_semantic() -> None:
    """语义化 Notebook 文件名应被命名治理接受。"""
    assert is_allowed_file_name(NOTEBOOK_PATH.name)


@pytest.mark.constraint
def test_colab_notebook_has_no_stored_outputs() -> None:
    """Colab 入口不应提交已执行输出。"""
    payload = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]

    assert code_cells
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(cell.get("outputs") == [] for cell in code_cells)


@pytest.mark.constraint
def test_colab_notebook_delegates_runtime_logic_to_helper() -> None:
    """Notebook 必须调用 repository helper, 不能成为唯一实现。"""
    payload = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    joined_source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])

    assert "paper_workflow.colab_utils.sd_runtime_cold_start" in joined_source
    assert "run_default_model_plan" in joined_source
    assert "package_probe_outputs" in joined_source
    assert "SLM_WM_MODEL_SELECTION', 'both'" in joined_source
    assert "/content/drive/MyDrive/SLM/real_sd_runtime_probe" in joined_source


@pytest.mark.constraint
def test_probe_outputs_can_be_packaged_and_mirrored(tmp_path: Path) -> None:
    """真实 runtime 产物应能打包, 并可镜像到外部同步目录。"""
    output_dir = tmp_path / "outputs" / "real_sd_runtime_probe"
    output_dir.mkdir(parents=True)
    (output_dir / "sample_runtime_summary.json").write_text('{"probe_decision":"pass"}\n', encoding="utf-8")
    (output_dir / "sample_latent_trajectory_records.jsonl").write_text('{"trajectory_index":0}\n', encoding="utf-8")

    drive_dir = tmp_path / "drive_mirror"
    record = package_probe_outputs(root=tmp_path, drive_output_dir=str(drive_dir))
    archive_path = tmp_path / record.archive_path

    assert archive_path.exists()
    assert (drive_dir / "real_sd_runtime_probe_package.zip").exists()
    assert record.archive_digest == record.drive_archive_digest
    assert record.archive_entry_count == 2
    assert (output_dir / "real_sd_runtime_probe_archive_summary.json").exists()
    assert (output_dir / "real_sd_runtime_probe_archive_manifest.local.json").exists()

    with ZipFile(archive_path) as archive:
        assert sorted(archive.namelist()) == [
            "sample_latent_trajectory_records.jsonl",
            "sample_runtime_summary.json",
        ]
