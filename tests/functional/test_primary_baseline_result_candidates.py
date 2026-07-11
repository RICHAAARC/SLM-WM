"""主表 baseline 正式候选记录来源边界测试。"""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.write_primary_baseline_result_candidates import (
    METHOD_FAITHFUL_OBSERVATIONS_ENTRY,
    T2SMARK_CANDIDATE_RECORDS_ENTRY,
    evidence_path_for_source,
    load_method_faithful_observations,
    load_t2smark_candidate_rows,
    normalize_t2smark_candidate_rows,
)


pytestmark = pytest.mark.quick


def test_method_observation_package_must_contain_canonical_entry(tmp_path: Path) -> None:
    """显式指定结果包后只能读取包内规范条目, 不得改读本地记录。"""

    local_path = tmp_path / "outputs" / "observations.json"
    local_path.parent.mkdir(parents=True)
    local_path.write_text('[{"baseline_id":"tree_ring"}]\n', encoding="utf-8")
    package_path = tmp_path / "method_results.zip"
    with ZipFile(package_path, "w") as archive:
        archive.writestr("outputs/unrelated.json", "[]\n")

    with pytest.raises(ValueError, match="observation 为空"):
        load_method_faithful_observations(observations_path=local_path, package_path=package_path)


def test_t2smark_package_must_contain_canonical_entry(tmp_path: Path) -> None:
    """显式 T2SMark 结果包缺少规范条目时必须停止导入。"""

    local_path = tmp_path / "outputs" / "t2smark_records.jsonl"
    local_path.parent.mkdir(parents=True)
    local_path.write_text('{"baseline_id":"t2smark"}\n', encoding="utf-8")
    package_path = tmp_path / "t2smark_results.zip"
    with ZipFile(package_path, "w") as archive:
        archive.writestr("outputs/unrelated.jsonl", "{}\n")

    with pytest.raises(ValueError, match="T2SMark 正式候选记录为空"):
        load_t2smark_candidate_rows(candidate_records_path=local_path, package_path=package_path)


def test_explicit_packages_supply_canonical_baseline_records(tmp_path: Path) -> None:
    """规范包条目应成为候选记录的唯一数据来源。"""

    method_package = tmp_path / "method_results.zip"
    with ZipFile(method_package, "w") as archive:
        archive.writestr(
            METHOD_FAITHFUL_OBSERVATIONS_ENTRY,
            json.dumps([{"baseline_id": "tree_ring", "prompt_id": "prompt_1"}]),
        )
    t2smark_package = tmp_path / "t2smark_results.zip"
    with ZipFile(t2smark_package, "w") as archive:
        archive.writestr(
            T2SMARK_CANDIDATE_RECORDS_ENTRY,
            json.dumps({"baseline_id": "t2smark", "attack_name": "jpeg_compression"}) + "\n",
        )

    method_rows = load_method_faithful_observations(
        observations_path=tmp_path / "missing_method.json",
        package_path=method_package,
    )
    t2smark_rows = load_t2smark_candidate_rows(
        candidate_records_path=tmp_path / "missing_t2smark.jsonl",
        package_path=t2smark_package,
    )

    assert method_rows == [{"baseline_id": "tree_ring", "prompt_id": "prompt_1"}]
    assert t2smark_rows == [{"baseline_id": "t2smark", "attack_name": "jpeg_compression"}]


def test_evidence_source_must_exist(tmp_path: Path) -> None:
    """不存在的证据文件不得形成 baseline 候选记录来源。"""

    with pytest.raises(FileNotFoundError, match="baseline 证据来源不存在"):
        evidence_path_for_source(tmp_path / "missing.zip", tmp_path)


def test_t2smark_normalization_does_not_promote_protocol_readiness(tmp_path: Path) -> None:
    """来源归一化不得根据计数自行提升 Prompt 协议门禁。"""

    source_path = tmp_path / "t2smark_results.jsonl"
    source_path.write_text('{"baseline_id":"t2smark"}\n', encoding="utf-8")
    rows = normalize_t2smark_candidate_rows(
        rows=[
            {
                "baseline_id": "t2smark",
                "positive_count": 340,
                "negative_count": 340,
                "paper_run_prompt_protocol_ready": False,
            }
        ],
        source_path=source_path,
        root_path=tmp_path,
    )

    assert rows[0]["paper_run_prompt_protocol_ready"] is False
    assert rows[0]["formal_evidence_paths_ready"] is True
