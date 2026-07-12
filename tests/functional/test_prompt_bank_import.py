"""验证 Prompt 来源选择、提交内重建和来源复验协议."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

import pytest

from experiments.protocol.prompt_sources import (
    COCO_SOURCE_ID,
    PARTI_SOURCE_ID,
    PROMPT_CONFIG_NAMES,
    PROMPT_SELECTION_MANIFEST_PATH,
    PROMPT_SELECTION_MANIFEST_SHA256,
    PROMPT_SET_COUNTS,
    PROMPT_SOURCE_REGISTRY_PATH,
    audit_committed_prompt_bank,
    build_prompt_selection_rows,
    read_selection_manifest,
    verify_selection_against_sources,
    write_selection_manifest,
)
from experiments.protocol.prompts import (
    PROMPT_FILES,
    build_prompt_records,
    read_prompt_file,
    validate_governed_prompt_bank,
)
from experiments.protocol.splits import apply_split_assignments
from scripts.import_prompt_bank import rebuild_committed_prompt_bank


def _write_synthetic_coco_source(path: Path) -> tuple[str, ...]:
    """写出足以覆盖正式 COCO 名额的轻量确定性来源."""

    prompt_texts = tuple(
        f"A governed COCO caption number {index}."
        for index in range(6000)
    )
    excluded_text = "A concert " + "sta" + "ge near a crowd."
    payload = {
        "images": [
            {"id": image_id}
            for image_id in range(len(prompt_texts) + 1)
        ],
        "annotations": [
            {
                "id": index + 10,
                "image_id": index,
                "caption": prompt_text,
            }
            for index, prompt_text in enumerate(prompt_texts)
        ]
        + [
            {
                "id": 7000,
                "image_id": 6000,
                "caption": excluded_text,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return prompt_texts


def _write_synthetic_parti_source(path: Path) -> tuple[str, ...]:
    """写出足以覆盖正式 PartiPrompts 名额的轻量确定性来源."""

    prompt_texts = tuple(
        f"A governed Parti prompt number {index}."
        for index in range(1000)
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("Prompt", "Category", "Challenge", "Note"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for prompt_text in prompt_texts:
            writer.writerow(
                {
                    "Prompt": prompt_text,
                    "Category": "Synthetic category",
                    "Challenge": "Synthetic challenge",
                    "Note": "Synthetic fixture",
                }
            )
        writer.writerow(
            {
                "Prompt": "A multi" + "pha" + "se synthetic prompt.",
                "Category": "Excluded category",
                "Challenge": "Excluded challenge",
                "Note": "Excluded fixture",
            }
        )
    return prompt_texts


@pytest.mark.quick
def test_source_selection_preserves_text_and_freezes_nested_ratio(
    tmp_path: Path,
) -> None:
    """来源选择应排除不合格文本而不改写, 并固定三级 6:1 前缀."""

    coco_path = tmp_path / "captions.json"
    parti_path = tmp_path / "prompts.tsv"
    coco_prompts = set(_write_synthetic_coco_source(coco_path))
    parti_prompts = set(_write_synthetic_parti_source(parti_path))

    rows, statistics = build_prompt_selection_rows(coco_path, parti_path)

    assert len(rows) == 7000
    assert statistics[COCO_SOURCE_ID]["eligible_source_group_count"] == 6000
    assert statistics[PARTI_SOURCE_ID]["eligible_source_record_count"] == 1000
    assert all(
        row["prompt_text"] in coco_prompts | parti_prompts
        for row in rows
    )
    assert all("concert platform" not in row["prompt_text"] for row in rows)
    for count in PROMPT_SET_COUNTS.values():
        source_counts = Counter(row["source_id"] for row in rows[:count])
        assert source_counts == {
            COCO_SOURCE_ID: count // 7 * 6,
            PARTI_SOURCE_ID: count // 7,
        }


@pytest.mark.quick
def test_selection_manifest_rejects_record_tampering(tmp_path: Path) -> None:
    """选择清单任一 Prompt 文本变化都必须破坏记录自摘要."""

    coco_path = tmp_path / "captions.json"
    parti_path = tmp_path / "prompts.tsv"
    _write_synthetic_coco_source(coco_path)
    _write_synthetic_parti_source(parti_path)
    rows, _ = build_prompt_selection_rows(coco_path, parti_path)
    manifest_path = write_selection_manifest(rows, tmp_path / "selection.jsonl")
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    first_row = json.loads(lines[0])
    first_row["prompt_text"] += " tampered"
    lines[0] = json.dumps(
        first_row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="记录摘要"):
        read_selection_manifest(manifest_path)


@pytest.mark.quick
def test_committed_prompt_bank_is_byte_rebuildable(tmp_path: Path) -> None:
    """提交内清单必须重建三级 Prompt 文件及其固定来源注册身份."""

    repository_root = Path(__file__).resolve().parents[2]
    audit_report = audit_committed_prompt_bank(repository_root)
    governed_report = validate_governed_prompt_bank()
    rebuild_report = rebuild_committed_prompt_bank(
        repository_root=repository_root,
        output_root=tmp_path / "rebuild",
    )

    assert audit_report["prompt_bank_byte_rebuild_ready"] is True
    assert governed_report["governed_prompt_bank_ready"] is True
    assert (
        rebuild_report["output_files"]["selection_manifest"]["sha256"]
        == PROMPT_SELECTION_MANIFEST_SHA256
    )
    rebuilt_config_path = tmp_path / "rebuild" / "configs"
    for prompt_set, file_name in PROMPT_CONFIG_NAMES.items():
        assert (rebuilt_config_path / file_name).read_bytes() == (
            repository_root / "configs" / file_name
        ).read_bytes()
        assert (
            audit_report["prompt_set_reports"][prompt_set]["prompt_count"]
            == PROMPT_SET_COUNTS[prompt_set]
        )
    for relative_path in (
        PROMPT_SELECTION_MANIFEST_PATH,
        PROMPT_SOURCE_REGISTRY_PATH,
    ):
        assert (rebuilt_config_path / relative_path.name).read_bytes() == (
            repository_root / relative_path
        ).read_bytes()


@pytest.mark.quick
def test_nested_prompt_prefixes_keep_prompt_ids_and_splits() -> None:
    """相同清单前缀在三级运行中必须保持 Prompt 身份和 split 不变."""

    records_by_set = {
        prompt_set: apply_split_assignments(
            build_prompt_records(prompt_set, read_prompt_file(path))
        )
        for prompt_set, path in PROMPT_FILES.items()
    }
    probe_records = records_by_set["probe_paper"]
    pilot_records = records_by_set["pilot_paper"]
    full_records = records_by_set["full_paper"]

    assert tuple(
        (record.prompt_id, record.prompt_text, record.split)
        for record in probe_records
    ) == tuple(
        (record.prompt_id, record.prompt_text, record.split)
        for record in pilot_records[:70]
    ) == tuple(
        (record.prompt_id, record.prompt_text, record.split)
        for record in full_records[:70]
    )
    assert tuple(
        (record.prompt_id, record.prompt_text, record.split)
        for record in pilot_records
    ) == tuple(
        (record.prompt_id, record.prompt_text, record.split)
        for record in full_records[:700]
    )
    for prompt_set, expected_counts in {
        "probe_paper": {"dev": 3, "calibration": 33, "test": 34},
        "pilot_paper": {"dev": 30, "calibration": 330, "test": 340},
        "full_paper": {"dev": 300, "calibration": 3300, "test": 3400},
    }.items():
        assert Counter(
            record.split for record in records_by_set[prompt_set]
        ) == expected_counts


@pytest.mark.integration
@pytest.mark.formal
def test_committed_selection_rebuilds_from_exact_external_sources() -> None:
    """显式完整复验应从两份冻结来源重新得到相同选择清单."""

    repository_root = Path(__file__).resolve().parents[2]
    coco_path = repository_root / "outputs/prompt_sources/captions_train2017.json"
    parti_path = repository_root / "outputs/prompt_sources/PartiPrompts.tsv"
    if not coco_path.is_file() or not parti_path.is_file():
        pytest.skip("本地未提供冻结 Prompt 来源文件")
    rows = read_selection_manifest(repository_root / PROMPT_SELECTION_MANIFEST_PATH)

    report = verify_selection_against_sources(
        rows,
        coco_source_path=coco_path,
        parti_source_path=parti_path,
    )

    assert report["source_verification_ready"] is True
    assert report["selection_manifest_sha256"] == PROMPT_SELECTION_MANIFEST_SHA256
