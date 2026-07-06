"""验证 prompt 与事件协议的轻量行为。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.protocol.calibration import binomial_rate_upper_confidence_bound
from experiments.protocol.events import build_event_records
from experiments.protocol.prompts import PROMPT_FILES, build_prompt_record, build_prompt_records, read_prompt_file
from experiments.protocol.splits import (
    SAMPLE_ROLES,
    assert_disjoint_calibration_and_test,
    build_group_split_counts,
    group_prompt_ids_by_split,
)
from scripts.write_prompt_event_protocol import write_prompt_event_protocol_outputs


@pytest.mark.quick
def test_prompt_and_event_ids_are_stable_and_unique() -> None:
    """prompt_id 与 event_id 应稳定且唯一。"""
    prompt_texts = (
        "a city street cafe at dusk with warm lamps",
        "a small sailboat crossing a calm blue lake",
        "a ceramic teapot beside citrus fruit on a wooden table",
    )
    prompt_records = build_prompt_records("probe", prompt_texts)
    rebuilt_prompt = build_prompt_record("probe", 0, prompt_texts[0])
    event_records = build_event_records(prompt_records)

    assert prompt_records[0].prompt_id == rebuilt_prompt.prompt_id
    assert len({record.prompt_id for record in prompt_records}) == len(prompt_records)
    assert len({record.event_id for record in event_records}) == len(event_records)
    assert len(event_records) == len(prompt_records) * len(SAMPLE_ROLES)
    assert {record.supports_paper_claim for record in event_records} == {False}


@pytest.mark.quick
def test_calibration_and_test_prompt_ids_are_disjoint() -> None:
    """calibration 与 test 的 prompt_id 不能交叉。"""
    prompt_records = build_prompt_records(
        "pilot_paper",
        tuple(f"a controlled prompt variant {index}" for index in range(9)),
    )
    split_groups = group_prompt_ids_by_split(prompt_records)

    assert assert_disjoint_calibration_and_test(split_groups)
    assert set(split_groups["calibration"]).isdisjoint(split_groups["test"])


@pytest.mark.quick
def test_paper_prompt_split_uses_shared_calibration_heavy_ratio() -> None:
    """三类论文运行层级应共享 dev 5%、calibration 55%、test 40% 的目标比例。"""

    assert build_group_split_counts(60) == {"dev": 3, "calibration": 33, "test": 24}
    assert build_group_split_counts(600) == {"dev": 30, "calibration": 330, "test": 240}
    assert build_group_split_counts(6000) == {"dev": 300, "calibration": 3300, "test": 2400}


@pytest.mark.quick
def test_full_paper_calibration_split_supports_low_fpr_confidence_boundary() -> None:
    """full_paper calibration split 应为 FPR=0.001 保留足够 clean negative 样本。"""
    prompt_records = build_prompt_records(
        "full_paper",
        tuple(read_prompt_file(PROMPT_FILES["full_paper"])),
    )
    split_groups = group_prompt_ids_by_split(prompt_records)

    assert binomial_rate_upper_confidence_bound(0, len(split_groups["calibration"]), 0.95) <= 0.001
    assert len(split_groups["test"]) >= 0.40 * len(prompt_records) - 1


def write_prompt_config(repo_root: Path, prompt_set: str, lines: tuple[str, ...]) -> None:
    """写入测试用 prompt 配置。"""
    config_dir = repo_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / f"paper_main_{prompt_set}_prompts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.quick
def test_protocol_writer_creates_governed_outputs(tmp_path: Path) -> None:
    """协议写出脚本应只在 outputs 下生成受治理产物。"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_prompt_config(repo_root, "probe", ("a city street with lamps", "a calm lake at sunrise"))
    write_prompt_config(repo_root, "probe_paper", ("a small server validation prompt",))
    write_prompt_config(repo_root, "pilot_paper", ("a ceramic teapot on a table",))
    write_prompt_config(repo_root, "full_paper", ("a quiet library reading room",))

    manifest = write_prompt_event_protocol_outputs(root=repo_root)
    output_dir = repo_root / "outputs" / "prompt_event_protocol"
    prompt_manifest = json.loads((output_dir / "prompt_manifest.json").read_text(encoding="utf-8"))
    event_manifest = json.loads((output_dir / "event_protocol_manifest.json").read_text(encoding="utf-8"))
    statistics = json.loads((output_dir / "prompt_statistics.json").read_text(encoding="utf-8"))

    assert manifest["metadata"]["protocol_decision"] == "pass"
    assert manifest["metadata"]["supports_paper_claim"] is False
    assert set(manifest["output_paths"]) == {
        "outputs/prompt_event_protocol/prompt_records.jsonl",
        "outputs/prompt_event_protocol/event_records.jsonl",
        "outputs/prompt_event_protocol/prompt_manifest.json",
        "outputs/prompt_event_protocol/split_manifest.json",
        "outputs/prompt_event_protocol/event_protocol_manifest.json",
        "outputs/prompt_event_protocol/prompt_statistics.json",
        "outputs/prompt_event_protocol/manifest.local.json",
    }
    assert prompt_manifest["protocol_decision"] == "pass"
    assert event_manifest["protocol_decision"] == "pass"
    assert {record["split"] for record in prompt_manifest["prompt_records"]} <= {"dev", "calibration", "test"}
    assert statistics["prompt_count"] == 5
    assert statistics["event_count"] == 5 * len(SAMPLE_ROLES)
    assert statistics["calibration_test_disjoint"] is True


@pytest.mark.quick
def test_protocol_writer_rejects_output_outside_outputs(tmp_path: Path) -> None:
    """协议写出脚本应拒绝 outputs 之外的持久化目录。"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    with pytest.raises(ValueError):
        write_prompt_event_protocol_outputs(root=repo_root, output_dir=repo_root / "outside")
