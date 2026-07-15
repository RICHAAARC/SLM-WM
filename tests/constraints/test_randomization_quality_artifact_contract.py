"""约束质量 writer、结果闭合和三档 profile 使用同一产物清单."""

from __future__ import annotations

from pathlib import Path

import pytest

from paper_experiments.analysis.paper_profile_protocol_isomorphism import (
    build_paper_profile_protocol_records,
    registered_artifact_contract,
)
from paper_experiments.runners import randomization_dataset_quality as runner
from scripts.paper_result_closure import _CLOSURE_ARTIFACT_SPECS
from tests.functional.test_randomization_dataset_quality_runner import (
    _provenance,
    _result,
)


pytestmark = pytest.mark.constraint


def _closure_quality_file_names() -> set[str]:
    """返回结果闭合对质量统计登记的精确文件集合."""

    matches = [
        specification
        for specification in _CLOSURE_ARTIFACT_SPECS
        if specification.artifact_id
        == "randomization_dataset_quality_manifest"
    ]
    assert len(matches) == 1
    return set(matches[0].file_names)


def test_quality_writer_closure_and_profile_registry_are_exactly_isomorphic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实 writer 输出、闭合规格和三档登记不得新增、遗漏或额外文件."""

    source = _provenance(tmp_path)
    monkeypatch.setattr(
        runner,
        "rebuild_randomization_dataset_quality",
        lambda _source, root=".": _result(),
    )
    manifest_path = runner.write_randomization_dataset_quality_outputs(
        source,
        root=tmp_path,
    )
    writer_file_names = {
        path.name for path in manifest_path.parent.iterdir() if path.is_file()
    }
    registered = registered_artifact_contract(
        "randomization_dataset_quality_manifest"
    )
    registered_file_names = set(registered["file_names"])
    profile_records = build_paper_profile_protocol_records(Path.cwd())
    profile_file_names = {
        profile_id: set(
            next(
                record["file_names"]
                for record in profile["artifact_contract"]
                if record["artifact_id"]
                == "randomization_dataset_quality_manifest"
            )
        )
        for profile_id, profile in profile_records.items()
    }

    assert writer_file_names == registered_file_names
    assert _closure_quality_file_names() == registered_file_names
    assert all(
        file_names == registered_file_names
        for file_names in profile_file_names.values()
    )
    assert "attack_prompt_distributional_quality_records.jsonl" in (
        registered_file_names
    )


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_quality_writer_rejects_any_registered_file_set_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """登记清单缺文件或多文件时, 真实 writer 必须在发布前失败."""

    source = _provenance(tmp_path)
    monkeypatch.setattr(
        runner,
        "rebuild_randomization_dataset_quality",
        lambda _source, root=".": _result(),
    )
    contract = registered_artifact_contract(
        "randomization_dataset_quality_manifest"
    )
    file_names = list(contract["file_names"])
    if mutation == "missing":
        file_names.remove(
            "attack_prompt_distributional_quality_records.jsonl"
        )
    else:
        file_names.append("undeclared_extra_quality_file.json")
    mutated_contract = {**contract, "file_names": file_names}
    monkeypatch.setattr(
        runner,
        "registered_artifact_contract",
        lambda _artifact_id: mutated_contract,
    )

    with pytest.raises(
        runner.RandomizationDatasetQualityRunnerError,
        match="未精确匹配唯一正式产物清单",
    ):
        runner.write_randomization_dataset_quality_outputs(
            source,
            root=tmp_path,
        )
