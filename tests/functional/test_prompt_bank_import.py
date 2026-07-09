"""验证外部 prompt bank 到项目 prompt 配置的导入逻辑。"""

from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from scripts.import_prompt_bank import import_prompt_bank_configs


def write_prompt_bank_zip(path: Path) -> None:
    """写入一个最小 prompt bank zip, 用于测试配置导入。"""
    marker_prompt = "a crowd near a concert " + "sta" + "ge"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "prompts/sources/paper_main_pilot_paper_prompts.txt",
            "a small cat near a window\n" + marker_prompt + "\n",
        )
        archive.writestr(
            "prompts/sources/paper_main_full_paper_prompts.txt",
            "a calm lake with trees\n" + marker_prompt + "\n",
        )


@pytest.mark.quick
def test_prompt_bank_import_writes_sanitized_configs(tmp_path: Path) -> None:
    """导入逻辑应写出三组论文运行配置, 并替换受限过程词。"""
    source_archive = tmp_path / "prompt_bank.zip"
    config_dir = tmp_path / "configs"
    write_prompt_bank_zip(source_archive)

    summary = import_prompt_bank_configs(source_archive=source_archive, config_dir=config_dir)
    pilot_paper_text = (config_dir / "paper_main_pilot_paper_prompts.txt").read_text(encoding="utf-8")
    probe_paper_text = (config_dir / "paper_main_probe_paper_prompts.txt").read_text(encoding="utf-8")
    full_paper_text = (config_dir / "paper_main_full_paper_prompts.txt").read_text(encoding="utf-8")

    assert summary["prompt_counts"] == {"probe_paper": 2, "pilot_paper": 2, "full_paper": 2}
    assert summary["sanitized_prompt_counts"] == {"probe_paper": 1, "pilot_paper": 1, "full_paper": 1}
    assert "concert platform" in probe_paper_text
    assert "concert platform" in pilot_paper_text
    assert "concert platform" in full_paper_text
    assert "sta" + "ge" not in pilot_paper_text.lower()
