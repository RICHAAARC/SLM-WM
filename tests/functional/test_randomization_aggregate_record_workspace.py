"""验证精确9+3聚合原始记录临时工作区的不可绕过边界."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_STORED, ZipFile

import pytest

from experiments.protocol.formal_randomization import (
    formal_randomization_repeat_ids,
    resolve_formal_randomization_repeat,
)
from paper_experiments.runners.randomization_aggregate_provenance import (
    RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES,
    RandomizationAggregateProvenance,
)
from paper_experiments.runners import (
    randomization_aggregate_record_workspace as workspace_module,
)
from paper_experiments.runners.randomization_aggregate_record_workspace import (
    RandomizationAggregateRecordWorkspaceError,
    open_randomization_aggregate_record_workspace,
)
from paper_experiments.runners.randomization_repeat_evidence import (
    RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES,
)
from paper_experiments.runners.randomization_prompt_source_contract import (
    rebuild_randomization_prompt_source_contract,
)


pytestmark = pytest.mark.quick

PAPER_RUN_NAME = "probe_paper"
TARGET_FPR = 0.1
CODE_VERSION = "a" * 40


def _digest(value: str | bytes) -> str:
    """为测试来源构造稳定 SHA-256."""

    payload = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    """写出确定性的测试 JSON 字节."""

    return (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")


def _record_members_for_family(
    package_family: str,
    *,
    repeat_id: str,
) -> dict[str, bytes]:
    """按生产成员模板为一个测试 leaf 构造可迭代原始记录."""

    members: dict[str, bytes] = {}
    for specification in workspace_module._RECORD_MEMBER_SPECS:
        if (
            specification.package_family != package_family
            or specification.randomization_scope
            != "active_repeat_component"
        ):
            continue
        member_name = specification.member_template.format(
            paper_run=PAPER_RUN_NAME
        )
        row = {
            "randomization_repeat_id": repeat_id,
            "package_family": package_family,
            "record_role": specification.record_role,
        }
        if specification.record_role == "quality_image_record":
            row["dataset_quality_record_id"] = f"quality:{repeat_id}"
            payload = _json_bytes(row)
        elif specification.record_role == "quality_feature_record":
            feature_rows = [
                {
                    **row,
                    "dataset_quality_record_id": f"quality:{repeat_id}",
                    "dataset_quality_image_role": image_role,
                }
                for image_role in ("source", "comparison")
            ]
            payload = b"".join(_json_bytes(item) for item in feature_rows)
        elif specification.record_format == workspace_module.RECORD_FORMAT_JSONL:
            payload = _json_bytes(row)
        elif (
            specification.record_format
            == workspace_module.RECORD_FORMAT_JSON_ARRAY
        ):
            payload = _json_bytes([row])
        elif (
            specification.record_format
            == workspace_module.RECORD_FORMAT_RAW_BYTES
        ):
            repository_root = Path(__file__).resolve().parents[2]
            source_name_by_role = {
                "governed_prompt_file_bytes": (
                    f"paper_main_{PAPER_RUN_NAME}_prompts.txt"
                ),
                "governed_prompt_selection_manifest_bytes": (
                    "prompt_selection_manifest.jsonl"
                ),
                "governed_prompt_source_registry_bytes": (
                    "prompt_source_registry.json"
                ),
            }
            payload = (
                repository_root
                / "configs"
                / source_name_by_role[specification.record_role]
            ).read_bytes()
        else:
            payload = _json_bytes({"record": row})
        members[member_name] = payload
    return members


def _write_leaf_package(
    path: Path,
    *,
    package_family: str,
    repeat_id: str,
) -> None:
    """写出包含登记记录成员的活动 leaf ZIP."""

    path.parent.mkdir(parents=True, exist_ok=True)
    members = _record_members_for_family(
        package_family,
        repeat_id=repeat_id,
    )
    members["identity.json"] = _json_bytes(
        {"package_family": package_family, "repeat_id": repeat_id}
    )
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        for member_name, payload in members.items():
            archive.writestr(member_name, payload)


def _write_repeat_component(path: Path, repeat_id: str) -> None:
    """写出含7个真实嵌套 leaf ZIP 字节的测试重复组件."""

    path.parent.mkdir(parents=True, exist_ok=True)
    leaf_records = []
    leaf_paths: list[Path] = []
    for package_family in RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES:
        leaf_path = path.parent / f"{repeat_id}_{package_family}.zip"
        _write_leaf_package(
            leaf_path,
            package_family=package_family,
            repeat_id=repeat_id,
        )
        archive_member = (
            f"randomization_repeat_evidence/{repeat_id}/leaf_packages/"
            f"{package_family}.zip"
        )
        leaf_records.append(
            {
                "package_family": package_family,
                "archive_member": archive_member,
                "package_sha256": _digest(leaf_path.read_bytes()),
                "code_version": CODE_VERSION,
                "formal_execution_run_lock_digest": _digest(
                    f"run:{repeat_id}:{package_family}"
                ),
                "formal_execution_package_lock_digest": _digest(
                    f"package:{repeat_id}:{package_family}"
                ),
            }
        )
        leaf_paths.append(leaf_path)
    manifest_member = (
        f"randomization_repeat_evidence/{repeat_id}/"
        "randomization_repeat_evidence_manifest.json"
    )
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        for leaf_path, record in zip(leaf_paths, leaf_records, strict=True):
            archive.write(leaf_path, record["archive_member"])
        archive.writestr(
            manifest_member,
            _json_bytes({"leaf_packages": leaf_records}),
        )
    for leaf_path in leaf_paths:
        leaf_path.unlink()


def _write_invariant_package(path: Path, package_family: str) -> None:
    """写出带官方参考记录成员的不变 leaf ZIP."""

    path.parent.mkdir(parents=True, exist_ok=True)
    specification = next(
        item
        for item in workspace_module._RECORD_MEMBER_SPECS
        if item.package_family == package_family
        and item.randomization_scope == "cross_repeat_invariant"
    )
    member_name = specification.member_template.format(
        paper_run=PAPER_RUN_NAME
    )
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        archive.writestr(
            member_name,
            _json_bytes(
                {
                    "package_family": package_family,
                    "record_role": specification.record_role,
                }
            ),
        )
        archive.writestr(
            "identity.json",
            _json_bytes({"package_family": package_family}),
        )


@pytest.fixture
def aggregate_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[RandomizationAggregateProvenance, dict[str, list[object]]]:
    """构造9个重复组件、3个不变包和生产复验调用记录."""

    repeat_component_paths: dict[str, Path] = {}
    repeat_records: list[dict[str, object]] = []
    for repeat_id in formal_randomization_repeat_ids():
        component_path = tmp_path / "components" / f"{repeat_id}.zip"
        _write_repeat_component(component_path, repeat_id)
        repeat_component_paths[repeat_id] = component_path
        repeat_records.append(
            {
                "randomization_repeat_id": repeat_id,
                "archive_member": f"repeat_components/{repeat_id}.zip",
                "package_sha256": _digest(component_path.read_bytes()),
                "randomization_repeat_evidence_manifest_digest": _digest(
                    f"manifest:{repeat_id}"
                ),
                "component_content_digest": _digest(
                    f"component:{repeat_id}"
                ),
            }
        )

    invariant_paths: dict[str, Path] = {}
    invariant_records: list[dict[str, object]] = []
    for package_family in RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES:
        package_path = tmp_path / "invariants" / f"{package_family}.zip"
        _write_invariant_package(package_path, package_family)
        invariant_paths[package_family] = package_path
        invariant_records.append(
            {
                "package_family": package_family,
                "archive_member": f"invariant_packages/{package_family}.zip",
                "package_sha256": _digest(package_path.read_bytes()),
                "formal_execution_run_lock_digest": _digest(
                    f"run:{package_family}"
                ),
                "formal_execution_package_lock_digest": _digest(
                    f"package:{package_family}"
                ),
            }
        )

    aggregate_path = tmp_path / "aggregate.zip"
    with ZipFile(aggregate_path, "w", compression=ZIP_STORED) as archive:
        for repeat_id, component_path in repeat_component_paths.items():
            archive.write(component_path, f"repeat_components/{repeat_id}.zip")
        for package_family, package_path in invariant_paths.items():
            archive.write(
                package_path,
                f"invariant_packages/{package_family}.zip",
            )

    aggregate_digest = _digest("aggregate")
    payload = {
        "paper_run_name": PAPER_RUN_NAME,
        "target_fpr": TARGET_FPR,
        "randomization_aggregate_digest": aggregate_digest,
    }
    source = RandomizationAggregateProvenance(
        package_path=aggregate_path.resolve(),
        package_sha256=_digest(aggregate_path.read_bytes()),
        payload_path="aggregate.zip!/payload.json",
        payload_sha256=_digest("payload"),
        manifest_path="aggregate.zip!/manifest.json",
        manifest_sha256=_digest("manifest"),
        payload=payload,
        manifest={"artifact_id": "aggregate"},
        randomization_repeat_components=tuple(repeat_records),
        invariant_packages=tuple(invariant_records),
        common_code_version=CODE_VERSION,
        randomization_aggregate_digest=aggregate_digest,
    )
    calls: dict[str, list[object]] = {
        "aggregate": [],
        "repeat": [],
        "leaf": [],
    }

    def validate_aggregate(package_path, *, paper_run_name, target_fpr):
        calls["aggregate"].append(
            (Path(package_path).resolve(), paper_run_name, target_fpr)
        )
        return source

    def validate_repeat(
        package_path,
        *,
        paper_run_name,
        target_fpr,
        randomization_repeat_id,
    ):
        calls["repeat"].append(randomization_repeat_id)
        repeat_record = next(
            record
            for record in repeat_records
            if record["randomization_repeat_id"] == randomization_repeat_id
        )
        return {
            "archive_sha256": _digest(Path(package_path).read_bytes()),
            "randomization_repeat_id": randomization_repeat_id,
            "code_version": CODE_VERSION,
            "repeat_component_ready": True,
            "randomization_aggregate_ready": False,
            "supports_paper_claim": False,
            "expected_archive_sha256": repeat_record["package_sha256"],
        }

    def inspect_leaf(
        package_path,
        *,
        spec,
        paper_run_name,
        target_fpr,
        randomization_repeat_id,
    ):
        path = Path(package_path)
        calls["leaf"].append((spec.package_family, randomization_repeat_id))
        if randomization_repeat_id is None:
            record = next(
                item
                for item in invariant_records
                if item["package_family"] == spec.package_family
            )
            return SimpleNamespace(
                package_family=spec.package_family,
                package_sha256=_digest(path.read_bytes()),
                code_version=CODE_VERSION,
                formal_execution_run_lock_digest=record[
                    "formal_execution_run_lock_digest"
                ],
                formal_execution_package_lock_digest=record[
                    "formal_execution_package_lock_digest"
                ],
                randomization_scope="cross_repeat_invariant",
                randomization_repeat_id="",
                generation_seed_index=-1,
                generation_seed_offset=-1,
                watermark_key_index=-1,
            )
        manifest_path = repeat_component_paths[randomization_repeat_id]
        manifest_member = (
            f"randomization_repeat_evidence/{randomization_repeat_id}/"
            "randomization_repeat_evidence_manifest.json"
        )
        with ZipFile(manifest_path) as archive:
            manifest = json.loads(archive.read(manifest_member))
        leaf_record = next(
            item
            for item in manifest["leaf_packages"]
            if item["package_family"] == spec.package_family
        )
        repeat = resolve_formal_randomization_repeat(
            randomization_repeat_id
        )
        return SimpleNamespace(
            package_family=spec.package_family,
            package_sha256=_digest(path.read_bytes()),
            code_version=CODE_VERSION,
            formal_execution_run_lock_digest=leaf_record[
                "formal_execution_run_lock_digest"
            ],
            formal_execution_package_lock_digest=leaf_record[
                "formal_execution_package_lock_digest"
            ],
            randomization_scope="active_repeat_component",
            randomization_repeat_id=randomization_repeat_id,
            generation_seed_index=repeat.generation_seed_index,
            generation_seed_offset=repeat.generation_seed_offset,
            watermark_key_index=repeat.watermark_key_index,
        )

    monkeypatch.setattr(
        workspace_module,
        "validate_randomization_aggregate_provenance",
        validate_aggregate,
    )
    monkeypatch.setattr(
        workspace_module,
        "validate_randomization_repeat_evidence_package",
        validate_repeat,
    )
    monkeypatch.setattr(
        workspace_module,
        "inspect_closure_package",
        inspect_leaf,
    )
    return source, calls


def test_workspace_revalidates_and_inspects_every_leaf_before_exposing_records(
    aggregate_source,
) -> None:
    """工作区必须复验完整来源, 且按职责精确分组原始记录."""

    source, calls = aggregate_source
    workspace = open_randomization_aggregate_record_workspace(source)
    with workspace:
        assert len(workspace.observation_sources) == 45
        assert len(workspace.threshold_binding_sources) == 45
        assert len(workspace.prompt_runtime_sources) == 9
        assert len(workspace.prompt_source_sources) == 27
        assert len(workspace.ablation_sources) == 18
        assert len(workspace.quality_sources) == 27
        assert len(workspace.quality_feature_sources) == 9
        assert len(workspace.reference_sources) == 3
        assert len(workspace.record_sources) == 174
        first_prompt_source = workspace.prompt_source_sources[0]
        assert workspace.read_bytes(first_prompt_source)
        assert all(
            item.randomization_scope == "active_repeat_component"
            for item in workspace.observation_sources
        )
        assert all(
            item.randomization_repeat_component_sha256
            and item.randomization_repeat_evidence_manifest_digest
            and item.component_content_digest
            and item.randomization_aggregate_package_sha256
            == source.package_sha256
            for item in workspace.observation_sources
        )
        assert all(
            item.randomization_scope == "cross_repeat_invariant"
            for item in workspace.reference_sources
        )
        assert all(
            not item.randomization_repeat_component_sha256
            and not item.randomization_repeat_evidence_manifest_digest
            and not item.component_content_digest
            and item.randomization_aggregate_package_sha256
            == source.package_sha256
            for item in workspace.reference_sources
        )
        repeat_id = formal_randomization_repeat_ids()[0]
        observation_source = workspace.find_source(
            randomization_repeat_id=repeat_id,
            package_family="image_only_dataset_runtime",
            record_role="semantic_watermark_detection_observation",
        )
        rows = tuple(workspace.iter_records(observation_source))
        assert rows[0]["randomization_repeat_id"] == repeat_id
        with pytest.raises(TypeError):
            rows[0]["package_family"] = "forged"
        protocol_source = workspace.find_source(
            randomization_repeat_id=repeat_id,
            package_family="image_only_dataset_runtime",
            record_role="semantic_watermark_frozen_evidence_protocol",
        )
        protocol = workspace.read_object(protocol_source)
        assert protocol["record"]["randomization_repeat_id"] == repeat_id
        quality_pairs = tuple(
            workspace.iter_quality_feature_pairs(repeat_id)
        )
        assert len(quality_pairs) == 1
        assert quality_pairs[0].dataset_quality_record_id == (
            f"quality:{repeat_id}"
        )
        assert quality_pairs[0].source_feature_record[
            "dataset_quality_image_role"
        ] == "source"
        assert quality_pairs[0].comparison_feature_record[
            "dataset_quality_image_role"
        ] == "comparison"
        temporary_root = Path(workspace._temporary_directory.name)
        assert temporary_root.is_dir()

    assert not temporary_root.exists()
    assert len(calls["aggregate"]) == 1
    assert len(calls["repeat"]) == 9
    assert len(calls["leaf"]) == 66
    with pytest.raises(
        RandomizationAggregateRecordWorkspaceError,
        match="with context",
    ):
        _ = workspace.record_sources


def test_workspace_rejects_forged_descriptor_and_wrong_reader(
    aggregate_source,
) -> None:
    """调用方不得注入成员路径或用错误解析器读取来源."""

    source, _calls = aggregate_source
    with open_randomization_aggregate_record_workspace(source) as workspace:
        descriptor = workspace.observation_sources[0]
        forged = replace(descriptor, record_member="../outside.json")
        with pytest.raises(
            RandomizationAggregateRecordWorkspaceError,
            match="不是当前工作区登记对象",
        ):
            tuple(workspace.iter_records(forged))
        with pytest.raises(
            RandomizationAggregateRecordWorkspaceError,
            match="iter_records",
        ):
            workspace.read_object(descriptor)


def test_workspace_rebuilds_embedded_governed_prompt_source_contract(
    aggregate_source,
) -> None:
    """三份内嵌来源字节必须重建70条受治理 probe Prompt."""

    source, _calls = aggregate_source
    with open_randomization_aggregate_record_workspace(source) as workspace:
        result = rebuild_randomization_prompt_source_contract(
            workspace,
            source,
            paper_run_name=PAPER_RUN_NAME,
        )

    assert len(result["prompt_rows"]) == 70
    assert result["report"]["prompt_source_contract_ready"] is True
    assert result["report"]["supports_paper_claim"] is False


def test_workspace_rejects_revalidated_provenance_field_drift_before_tempdir(
    aggregate_source,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任一 provenance 字段漂移都必须在临时目录创建前阻断."""

    source, _calls = aggregate_source
    revalidated = replace(
        source,
        randomization_aggregate_digest=_digest("other aggregate"),
    )
    monkeypatch.setattr(
        workspace_module,
        "validate_randomization_aggregate_provenance",
        lambda *_args, **_kwargs: revalidated,
    )
    workspace = open_randomization_aggregate_record_workspace(source)
    with pytest.raises(
        RandomizationAggregateRecordWorkspaceError,
        match="randomization_aggregate_digest",
    ):
        with workspace:
            raise AssertionError("不应进入工作区")
    assert workspace._temporary_directory is None


def test_workspace_rejects_leaf_replaced_after_production_inspector(
    aggregate_source,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspector 返回前替换 leaf 路径时仍须绑定外层 leaf 字节摘要."""

    source, _calls = aggregate_source
    production_inspector = workspace_module.inspect_closure_package
    target_repeat_id = formal_randomization_repeat_ids()[0]
    replaced = False

    def inspect_then_replace(
        package_path,
        *,
        spec,
        paper_run_name,
        target_fpr,
        randomization_repeat_id,
    ):
        nonlocal replaced
        candidate = production_inspector(
            package_path,
            spec=spec,
            paper_run_name=paper_run_name,
            target_fpr=target_fpr,
            randomization_repeat_id=randomization_repeat_id,
        )
        if (
            not replaced
            and randomization_repeat_id == target_repeat_id
            and spec.package_family == "image_only_dataset_runtime"
        ):
            path = Path(package_path)
            with ZipFile(path) as archive:
                members = {
                    info.filename: archive.read(info.filename)
                    for info in archive.infolist()
                }
            members["identity.json"] = _json_bytes(
                {
                    "package_family": spec.package_family,
                    "repeat_id": target_repeat_id,
                    "tampered_after_inspector": True,
                }
            )
            replacement_path = path.with_suffix(".replacement")
            with ZipFile(
                replacement_path,
                "w",
                compression=ZIP_STORED,
            ) as replacement:
                for member_name, payload in members.items():
                    replacement.writestr(member_name, payload)
            replacement_path.replace(path)
            replaced = True
        return candidate

    monkeypatch.setattr(
        workspace_module,
        "inspect_closure_package",
        inspect_then_replace,
    )
    with pytest.raises(
        RandomizationAggregateRecordWorkspaceError,
        match="活动 leaf .*与外层登记摘要不一致",
    ):
        with open_randomization_aggregate_record_workspace(source):
            raise AssertionError("被替换的 leaf 不得生成 descriptor")
    assert replaced is True


def test_workspace_rejects_leaf_replaced_before_record_read(
    aggregate_source,
) -> None:
    """Descriptor 建立后替换 leaf 时记录读取仍须复验完整 ZIP 摘要."""

    source, _calls = aggregate_source
    with open_randomization_aggregate_record_workspace(source) as workspace:
        descriptor = workspace.observation_sources[0]
        package_key = (
            descriptor.randomization_scope,
            descriptor.randomization_repeat_id,
            descriptor.package_family,
        )
        package_path = workspace._package_paths[package_key]
        with ZipFile(package_path) as archive:
            members = {
                info.filename: archive.read(info.filename)
                for info in archive.infolist()
            }
        members["identity.json"] = _json_bytes(
            {"tampered_before_record_read": True}
        )
        replacement_path = package_path.with_suffix(".replacement")
        with ZipFile(
            replacement_path,
            "w",
            compression=ZIP_STORED,
        ) as replacement:
            for member_name, payload in members.items():
                replacement.writestr(member_name, payload)
        package_path.write_bytes(replacement_path.read_bytes())
        replacement_path.unlink()
        with pytest.raises(
            RandomizationAggregateRecordWorkspaceError,
            match="临时 leaf 包 与外层登记摘要不一致",
        ):
            tuple(workspace.iter_records(descriptor))


def test_workspace_public_entry_requires_provenance_object(tmp_path: Path) -> None:
    """公共入口不得接受 aggregate 路径或任意 mapping."""

    with pytest.raises(TypeError, match="只接受"):
        open_randomization_aggregate_record_workspace(tmp_path / "aggregate.zip")
    with pytest.raises(TypeError, match="只接受"):
        open_randomization_aggregate_record_workspace({"package_path": "x.zip"})
