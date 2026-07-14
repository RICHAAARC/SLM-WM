"""验证精确9+3随机化 aggregate 来源包的轻量治理契约."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_STORED, ZipFile

import pytest

from experiments.protocol.formal_randomization import (
    formal_randomization_protocol_record,
    formal_randomization_repeat_ids,
    resolve_formal_randomization_repeat,
)
from experiments.protocol.paper_run_config import RUN_DEFAULTS
from paper_experiments.runners import randomization_aggregate_provenance as aggregate
from paper_experiments.runners.randomization_aggregate_provenance import (
    RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES,
    RandomizationAggregateProvenanceError,
    validate_randomization_aggregate_provenance,
    write_randomization_aggregate_provenance_package,
)
from paper_experiments.runners.randomization_aggregate_provenance import (
    build_parser,
    parse_randomization_aggregate_input_paths,
)


pytestmark = pytest.mark.quick

PAPER_RUN_NAME = "probe_paper"
TARGET_FPR = float(RUN_DEFAULTS[PAPER_RUN_NAME]["target_fpr"])
CODE_VERSION = "a" * 40


def test_aggregate_cli_requires_explicit_ordered_identity_paths() -> None:
    """CLI 不得从目录扫描推断 repeat 或不变 package 身份."""

    repeat_ids = formal_randomization_repeat_ids()
    parsed = parse_randomization_aggregate_input_paths(
        (f"{repeat_id}=/inputs/{repeat_id}.zip" for repeat_id in repeat_ids),
        expected_keys=repeat_ids,
        role="repeat component",
    )
    assert tuple(parsed) == repeat_ids
    with pytest.raises(ValueError, match="规范顺序"):
        parse_randomization_aggregate_input_paths(
            (
                f"{repeat_ids[1]}=/inputs/{repeat_ids[1]}.zip",
                f"{repeat_ids[0]}=/inputs/{repeat_ids[0]}.zip",
            ),
            expected_keys=repeat_ids,
            role="repeat component",
        )

    arguments = build_parser().parse_args(
        [
            "--paper-run-name",
            "probe_paper",
            "--target-fpr",
            "0.1",
        ]
    )
    assert arguments.repeat_component == []
    assert arguments.invariant_package == []


def _file_sha256(path: Path) -> str:
    """计算测试 ZIP 的文件摘要."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: str) -> str:
    """从测试身份构造稳定 SHA-256."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_repeat_component(path: Path, repeat_id: str) -> None:
    """写出可供轻量生产 validator 替身识别的单重复 ZIP."""

    path.parent.mkdir(parents=True, exist_ok=True)
    leaf_packages = [
        {
            "package_family": "test_leaf",
            "package_sha256": _digest(f"leaf:{repeat_id}"),
        }
    ]
    manifest = {
        "randomization_repeat_id": repeat_id,
        "leaf_packages": leaf_packages,
        "leaf_package_set_digest": aggregate._stable_digest(leaf_packages),
    }
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        archive.writestr(
            aggregate._repeat_manifest_member_name(repeat_id),
            json.dumps(manifest, sort_keys=True) + "\n",
        )
        archive.writestr("opaque_repeat_bytes.bin", repeat_id.encode("utf-8"))


def _write_invariant_package(path: Path, package_family: str) -> None:
    """写出可供轻量生产 inspector 替身识别的不变 ZIP."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        archive.writestr("opaque_invariant.json", json.dumps({"family": package_family}))


@pytest.fixture
def aggregate_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """构造12个唯一 ZIP, 并保留生产校验调用边界."""

    repeat_paths: dict[str, Path] = {}
    invariant_paths: dict[str, Path] = {}
    code_versions: dict[str, str] = {}
    validation_calls: list[tuple[str, str]] = []
    protocol_digest = formal_randomization_protocol_record()[
        "formal_randomization_protocol_digest"
    ]
    for repeat_id in formal_randomization_repeat_ids():
        path = tmp_path / "inputs" / "repeat" / f"{repeat_id}.zip"
        _write_repeat_component(path, repeat_id)
        repeat_paths[repeat_id] = path
        code_versions[repeat_id] = CODE_VERSION
    for package_family in RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES:
        path = tmp_path / "inputs" / "invariant" / f"{package_family}.zip"
        _write_invariant_package(path, package_family)
        invariant_paths[package_family] = path
        code_versions[package_family] = CODE_VERSION

    def validate_repeat(
        package_path: Path,
        *,
        randomization_repeat_id: str,
        paper_run_name: str,
        target_fpr: float,
    ):
        assert paper_run_name == PAPER_RUN_NAME
        assert target_fpr == TARGET_FPR
        path = Path(package_path)
        repeat = resolve_formal_randomization_repeat(randomization_repeat_id)
        validation_calls.append(("repeat", randomization_repeat_id))
        return {
            "archive_sha256": _file_sha256(path),
            **repeat.to_dict(),
            "formal_randomization_protocol_digest": protocol_digest,
            "code_version": code_versions[randomization_repeat_id],
            "randomization_repeat_evidence_manifest_digest": _digest(
                f"manifest:{randomization_repeat_id}"
            ),
            "component_content_digest": _digest(
                f"component:{randomization_repeat_id}"
            ),
            "repeat_component_ready": True,
            "randomization_aggregate_ready": False,
            "supports_paper_claim": False,
        }

    def inspect_invariant(
        package_path: Path,
        *,
        spec,
        paper_run_name: str,
        target_fpr: float,
        randomization_repeat_id,
    ):
        assert paper_run_name == PAPER_RUN_NAME
        assert target_fpr == TARGET_FPR
        assert randomization_repeat_id is None
        package_family = spec.package_family
        validation_calls.append(("invariant", package_family))
        return SimpleNamespace(
            package_family=package_family,
            package_sha256=_file_sha256(Path(package_path)),
            code_version=code_versions[package_family],
            formal_execution_run_lock_digest=_digest(f"run:{package_family}"),
            formal_execution_package_lock_digest=_digest(
                f"package:{package_family}"
            ),
            randomization_scope="cross_repeat_invariant",
            randomization_repeat_id="",
            generation_seed_index=-1,
            generation_seed_offset=-1,
            watermark_key_index=-1,
        )

    monkeypatch.setattr(
        aggregate,
        "validate_randomization_repeat_evidence_package",
        validate_repeat,
    )
    monkeypatch.setattr(aggregate, "inspect_closure_package", inspect_invariant)
    monkeypatch.setattr(
        aggregate,
        "validate_closure_candidate_repository_profile",
        lambda candidate, **_kwargs: validation_calls.append(
            ("profile", candidate.package_family)
        ),
    )
    monkeypatch.setattr(
        aggregate,
        "resolve_code_version",
        lambda _root: CODE_VERSION,
    )
    return (
        repeat_paths,
        invariant_paths,
        code_versions,
        validation_calls,
    )


def _write_aggregate(tmp_path: Path, aggregate_inputs):
    """通过公开写包器生成一个有效 aggregate 来源包."""

    repeat_paths, invariant_paths, _versions, _calls = aggregate_inputs
    return write_randomization_aggregate_provenance_package(
        repeat_paths,
        invariant_paths,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        root=tmp_path,
    )


def _resign_aggregate_with_invariant_payload(
    package_path: Path,
    *,
    package_family: str,
    structured_payload: dict[str, object],
) -> None:
    """替换一个不变包并重签全部外层摘要, 只保留包内语义攻击."""

    payload_member = aggregate._payload_member_name(PAPER_RUN_NAME)
    manifest_member = aggregate._manifest_member_name(PAPER_RUN_NAME)
    target_member = aggregate._invariant_package_member_name(
        PAPER_RUN_NAME,
        package_family,
    )
    with ZipFile(package_path) as archive:
        members = {
            info.filename: archive.read(info.filename)
            for info in archive.infolist()
        }

    nested_buffer = io.BytesIO()
    with ZipFile(nested_buffer, "w", compression=ZIP_STORED) as nested_archive:
        nested_archive.writestr(
            "opaque_invariant.json",
            json.dumps(
                {"family": package_family, **structured_payload},
                sort_keys=True,
            )
            + "\n",
        )
    nested_payload = nested_buffer.getvalue()
    nested_digest = hashlib.sha256(nested_payload).hexdigest()
    members[target_member] = nested_payload

    payload = json.loads(members[payload_member].decode("utf-8"))
    manifest = json.loads(members[manifest_member].decode("utf-8"))
    payload_record = next(
        record
        for record in payload["invariant_packages"]
        if record["package_family"] == package_family
    )
    manifest_record = next(
        record
        for record in manifest["invariant_packages"]
        if record["package_family"] == package_family
    )
    payload_record["package_sha256"] = nested_digest
    manifest_record["package_sha256"] = nested_digest

    aggregate_core = dict(payload)
    aggregate_core.pop("report_schema")
    aggregate_core.pop("generated_at")
    aggregate_core.pop("randomization_aggregate_digest")
    payload["randomization_aggregate_digest"] = aggregate._stable_digest(
        aggregate_core
    )
    payload_bytes = aggregate._json_bytes(payload)
    manifest["randomization_aggregate_digest"] = payload[
        "randomization_aggregate_digest"
    ]
    manifest["metadata"]["randomization_aggregate_digest"] = payload[
        "randomization_aggregate_digest"
    ]
    manifest["payload_sha256"] = hashlib.sha256(payload_bytes).hexdigest()
    manifest["entry_sha256"][payload_member] = manifest["payload_sha256"]
    manifest["entry_sha256"][target_member] = nested_digest
    manifest.pop("manifest_digest")
    manifest["manifest_digest"] = aggregate._stable_digest(manifest)
    members[payload_member] = payload_bytes
    members[manifest_member] = aggregate._json_bytes(manifest)

    with ZipFile(package_path, "w", compression=ZIP_STORED) as archive:
        for member_name, member_payload in members.items():
            archive.writestr(member_name, member_payload)
    for sidecar_path in aggregate._sidecar_paths(package_path):
        sidecar_path.unlink()


def test_aggregate_package_preserves_exact_inputs_and_returns_immutable_source(
    tmp_path: Path,
    aggregate_inputs,
) -> None:
    """成功路径必须保存12个原始 ZIP 并返回深度不可变来源对象."""

    repeat_paths, invariant_paths, _versions, calls = aggregate_inputs
    source = _write_aggregate(tmp_path, aggregate_inputs)

    assert source.package_path.is_file()
    assert source.payload["randomization_aggregate_schema_version"] == 1
    assert source.payload["randomization_repeat_ids"] == tuple(
        formal_randomization_repeat_ids()
    )
    assert source.payload["randomization_aggregate_ready"] is True
    assert source.payload["supports_paper_claim"] is False
    assert source.common_code_version == CODE_VERSION
    assert len(source.randomization_repeat_components) == 9
    assert len(source.invariant_packages) == 3
    for repeat_id, record in zip(
        formal_randomization_repeat_ids(),
        source.randomization_repeat_components,
        strict=True,
    ):
        repeat = resolve_formal_randomization_repeat(repeat_id)
        assert record["generation_seed_index"] == repeat.generation_seed_index
        assert record["generation_seed_offset"] == repeat.generation_seed_offset
        assert record["watermark_key_index"] == repeat.watermark_key_index
    assert calls.count(("repeat", formal_randomization_repeat_ids()[0])) == 2
    assert calls.count(
        ("invariant", RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES[0])
    ) == 2
    assert calls.count(
        ("profile", RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES[0])
    ) == 1
    with pytest.raises(TypeError):
        source.payload["supports_paper_claim"] = True

    with ZipFile(source.package_path) as archive:
        assert len(archive.namelist()) == 14
        for repeat_id, path in repeat_paths.items():
            assert archive.read(f"repeat_components/{repeat_id}.zip") == path.read_bytes()
        for package_family, path in invariant_paths.items():
            assert (
                archive.read(f"invariant_packages/{package_family}.zip")
                == path.read_bytes()
            )
    payload_sidecar, manifest_sidecar = aggregate._sidecar_paths(source.package_path)
    assert payload_sidecar.is_file()
    assert manifest_sidecar.is_file()
    assert Path(source.payload_path) == payload_sidecar
    assert Path(source.manifest_path) == manifest_sidecar
    assert source.payload_sha256 == _file_sha256(payload_sidecar)
    assert source.manifest_sha256 == _file_sha256(manifest_sidecar)
    payload = json.loads(payload_sidecar.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_sidecar.read_text(encoding="utf-8"))
    aggregate_core = dict(payload)
    aggregate_core.pop("report_schema")
    aggregate_core.pop("generated_at")
    declared_digest = aggregate_core.pop("randomization_aggregate_digest")
    assert declared_digest == aggregate._stable_digest(aggregate_core)
    assert all(
        set(record) == aggregate._REPEAT_COMPONENT_RECORD_FIELDS
        for record in payload["randomization_repeat_components"]
    )
    assert all(
        set(record) == aggregate._INVARIANT_PACKAGE_RECORD_FIELDS
        for record in payload["invariant_packages"]
    )
    expected_rebuild_arguments = [
        "python",
        "-m",
        "paper_experiments.runners.randomization_aggregate_provenance",
        "--paper-run-name",
        PAPER_RUN_NAME,
        "--target-fpr",
        str(TARGET_FPR),
        "--rebuild-source-aggregate-package-path",
        "{aggregate_package_path}",
    ]
    assert manifest["rebuild_command"].split() == expected_rebuild_arguments
    assert manifest["config"]["rebuild_input_mode"] == (
        "self_contained_aggregate_zip"
    )
    assert manifest["config"]["rebuild_working_directory"] == (
        "repository_root"
    )
    assert manifest["config"]["rebuild_source_argument"] == (
        "{aggregate_package_path}"
    )
    assert "repeat_components/" not in manifest["rebuild_command"]
    assert "invariant_packages/" not in manifest["rebuild_command"]


def test_aggregate_rejects_missing_repeat_and_duplicate_input_path(
    tmp_path: Path,
    aggregate_inputs,
) -> None:
    """权威 repeat 缺失和同一路径复用都必须在写包前拒绝."""

    repeat_paths, invariant_paths, _versions, _calls = aggregate_inputs
    missing = dict(repeat_paths)
    missing.pop(formal_randomization_repeat_ids()[-1])
    with pytest.raises(RandomizationAggregateProvenanceError, match="精确唯一覆盖"):
        write_randomization_aggregate_provenance_package(
            missing,
            invariant_paths,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            root=tmp_path,
        )

    duplicated = dict(repeat_paths)
    duplicated[formal_randomization_repeat_ids()[-1]] = duplicated[
        formal_randomization_repeat_ids()[0]
    ]
    with pytest.raises(RandomizationAggregateProvenanceError, match="同一输入路径"):
        write_randomization_aggregate_provenance_package(
            duplicated,
            invariant_paths,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            root=tmp_path,
        )


@pytest.mark.parametrize(
    "package_family",
    RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES,
)
@pytest.mark.parametrize(
    "positive_claim_payload",
    (
        {"paper_ready": True},
        {"claim_decision": "paper_supported"},
    ),
    ids=("paper_ready", "claim_decision"),
)
def test_aggregate_writer_rejects_positive_claim_in_every_invariant_package(
    tmp_path: Path,
    aggregate_inputs,
    package_family: str,
    positive_claim_payload: dict[str, object],
) -> None:
    """3个不变包中的任一正向结论字段都不得进入聚合来源."""

    repeat_paths, invariant_paths, _versions, _calls = aggregate_inputs
    _write_invariant_package(invariant_paths[package_family], package_family)
    with ZipFile(
        invariant_paths[package_family],
        "a",
        compression=ZIP_STORED,
    ) as archive:
        archive.writestr(
            "paper_claim.json",
            json.dumps(positive_claim_payload, sort_keys=True) + "\n",
        )

    with pytest.raises(
        RandomizationAggregateProvenanceError,
        match="包含正向论文结论字段",
    ):
        write_randomization_aggregate_provenance_package(
            repeat_paths,
            invariant_paths,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            root=tmp_path,
        )


def test_aggregate_writer_accepts_registered_negative_claim_states(
    tmp_path: Path,
    aggregate_inputs,
) -> None:
    """显式负结论状态可保留, 避免把字段存在本身误判为正向结论."""

    _repeat_paths, invariant_paths, _versions, _calls = aggregate_inputs
    for package_family, package_path in invariant_paths.items():
        with ZipFile(package_path, "w", compression=ZIP_STORED) as archive:
            archive.writestr(
                "opaque_invariant.json",
                json.dumps(
                    {
                        "family": package_family,
                        "paper_ready": False,
                        "claim_decision": "unsupported",
                        "supports_paper_claim": False,
                    },
                    sort_keys=True,
                )
                + "\n",
            )

    source = _write_aggregate(tmp_path, aggregate_inputs)
    assert source.payload["randomization_aggregate_ready"] is True
    assert source.payload["supports_paper_claim"] is False


def test_aggregate_rejects_cross_input_code_version_drift(
    tmp_path: Path,
    aggregate_inputs,
) -> None:
    """任一组件代码版本漂移都不能形成 aggregate 来源."""

    repeat_paths, invariant_paths, versions, _calls = aggregate_inputs
    versions[formal_randomization_repeat_ids()[-1]] = "b" * 40
    with pytest.raises(RandomizationAggregateProvenanceError, match="code version"):
        write_randomization_aggregate_provenance_package(
            repeat_paths,
            invariant_paths,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            root=tmp_path,
        )


def test_aggregate_writer_rejects_another_clean_checkout_version(
    tmp_path: Path,
    aggregate_inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """历史输入的共同版本不得被当前另一 clean checkout 重新归因."""

    repeat_paths, invariant_paths, _versions, _calls = aggregate_inputs
    historical_source = _write_aggregate(tmp_path, aggregate_inputs)
    monkeypatch.setattr(
        aggregate,
        "resolve_code_version",
        lambda _root: "b" * 40,
    )
    historical_validation = validate_randomization_aggregate_provenance(
        historical_source.package_path,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
    )
    assert historical_validation.common_code_version == CODE_VERSION
    with pytest.raises(
        RandomizationAggregateProvenanceError,
        match="匹配当前 clean Git checkout",
    ):
        write_randomization_aggregate_provenance_package(
            repeat_paths,
            invariant_paths,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            root=tmp_path,
        )


def test_aggregate_writer_rejects_invariant_repository_profile_drift(
    tmp_path: Path,
    aggregate_inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不变官方参考包必须匹配当前仓库的正式依赖 profile registry."""

    repeat_paths, invariant_paths, _versions, _calls = aggregate_inputs

    def reject_profile(candidate, **_kwargs):
        if candidate.package_family == (
            RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES[0]
        ):
            raise ValueError("profile drift")

    monkeypatch.setattr(
        aggregate,
        "validate_closure_candidate_repository_profile",
        reject_profile,
    )
    with pytest.raises(
        RandomizationAggregateProvenanceError,
        match="未匹配当前仓库依赖 profile",
    ):
        write_randomization_aggregate_provenance_package(
            repeat_paths,
            invariant_paths,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            root=tmp_path,
        )


def test_aggregate_validation_rejects_nested_input_tampering(
    tmp_path: Path,
    aggregate_inputs,
) -> None:
    """聚合 ZIP 内任一原始输入字节被替换后必须由摘要门禁拒绝."""

    source = _write_aggregate(tmp_path, aggregate_inputs)
    with ZipFile(source.package_path) as archive:
        members = [
            (info.filename, archive.read(info.filename))
            for info in archive.infolist()
        ]
    target_member = f"repeat_components/{formal_randomization_repeat_ids()[0]}.zip"
    with ZipFile(source.package_path, "w", compression=ZIP_STORED) as archive:
        for member_name, payload in members:
            archive.writestr(
                member_name,
                b"tampered" if member_name == target_member else payload,
            )

    with pytest.raises(RandomizationAggregateProvenanceError, match="成员字节摘要"):
        validate_randomization_aggregate_provenance(
            source.package_path,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


@pytest.mark.parametrize(
    "package_family",
    RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES,
)
@pytest.mark.parametrize(
    "positive_claim_payload",
    (
        {"paper_ready": True},
        {"claim_decision": "paper_supported"},
    ),
    ids=("paper_ready", "claim_decision"),
)
def test_aggregate_validator_rejects_resigned_invariant_positive_claim(
    tmp_path: Path,
    aggregate_inputs,
    package_family: str,
    positive_claim_payload: dict[str, object],
) -> None:
    """外层摘要全部重签后, validator 仍须复验不变包内部结论语义."""

    source = _write_aggregate(tmp_path, aggregate_inputs)
    _resign_aggregate_with_invariant_payload(
        source.package_path,
        package_family=package_family,
        structured_payload=positive_claim_payload,
    )

    with pytest.raises(
        RandomizationAggregateProvenanceError,
        match="包含正向论文结论字段",
    ):
        validate_randomization_aggregate_provenance(
            source.package_path,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_aggregate_rejects_resigned_boolean_schema_version(
    tmp_path: Path,
    aggregate_inputs,
) -> None:
    """布尔值不得利用 ``True == 1`` 冒充整数 schema version."""

    source = _write_aggregate(tmp_path, aggregate_inputs)
    payload_member = aggregate._payload_member_name(PAPER_RUN_NAME)
    manifest_member = aggregate._manifest_member_name(PAPER_RUN_NAME)
    with ZipFile(source.package_path) as archive:
        members = {
            info.filename: archive.read(info.filename)
            for info in archive.infolist()
        }
    payload = json.loads(members[payload_member].decode("utf-8"))
    manifest = json.loads(members[manifest_member].decode("utf-8"))
    payload["randomization_aggregate_schema_version"] = True
    aggregate_core = dict(payload)
    aggregate_core.pop("report_schema")
    aggregate_core.pop("generated_at")
    aggregate_core.pop("randomization_aggregate_digest")
    payload["randomization_aggregate_digest"] = aggregate._stable_digest(
        aggregate_core
    )
    payload_bytes = aggregate._json_bytes(payload)
    manifest["randomization_aggregate_schema_version"] = True
    manifest["randomization_aggregate_digest"] = payload[
        "randomization_aggregate_digest"
    ]
    manifest["metadata"]["randomization_aggregate_schema_version"] = True
    manifest["metadata"]["randomization_aggregate_digest"] = payload[
        "randomization_aggregate_digest"
    ]
    manifest["payload_sha256"] = hashlib.sha256(payload_bytes).hexdigest()
    manifest["entry_sha256"][payload_member] = manifest["payload_sha256"]
    manifest.pop("manifest_digest")
    manifest["manifest_digest"] = aggregate._stable_digest(manifest)
    members[payload_member] = payload_bytes
    members[manifest_member] = aggregate._json_bytes(manifest)
    with ZipFile(source.package_path, "w", compression=ZIP_STORED) as archive:
        for member_name, member_payload in members.items():
            archive.writestr(member_name, member_payload)
    payload_sidecar, manifest_sidecar = aggregate._sidecar_paths(source.package_path)
    payload_sidecar.unlink()
    manifest_sidecar.unlink()

    with pytest.raises(
        RandomizationAggregateProvenanceError,
        match="运行身份、协议或结论边界",
    ):
        validate_randomization_aggregate_provenance(
            source.package_path,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


@pytest.mark.parametrize(
    "field_name,forged_value",
    (
        ("generation_seed_index", False),
        ("generation_seed_index", 0.0),
        ("generation_seed_offset", 0.0),
        ("watermark_key_index", False),
    ),
)
def test_aggregate_rejects_resigned_non_integer_repeat_identity(
    tmp_path: Path,
    aggregate_inputs,
    field_name: str,
    forged_value: object,
) -> None:
    """重签 payload 与 manifest 后, 数值相等对象仍不得冒充整数身份."""

    source = _write_aggregate(tmp_path, aggregate_inputs)
    payload_member = aggregate._payload_member_name(PAPER_RUN_NAME)
    manifest_member = aggregate._manifest_member_name(PAPER_RUN_NAME)
    with ZipFile(source.package_path) as archive:
        members = {
            info.filename: archive.read(info.filename)
            for info in archive.infolist()
        }
    payload = json.loads(members[payload_member].decode("utf-8"))
    manifest = json.loads(members[manifest_member].decode("utf-8"))
    payload_record = payload["randomization_repeat_components"][0]
    manifest_record = manifest["randomization_repeat_components"][0]
    assert type(payload_record[field_name]) is int
    assert payload_record[field_name] == forged_value
    payload_record[field_name] = forged_value
    manifest_record[field_name] = forged_value
    aggregate_core = dict(payload)
    aggregate_core.pop("report_schema")
    aggregate_core.pop("generated_at")
    aggregate_core.pop("randomization_aggregate_digest")
    payload["randomization_aggregate_digest"] = aggregate._stable_digest(
        aggregate_core
    )
    payload_bytes = aggregate._json_bytes(payload)
    manifest["randomization_aggregate_digest"] = payload[
        "randomization_aggregate_digest"
    ]
    manifest["metadata"]["randomization_aggregate_digest"] = payload[
        "randomization_aggregate_digest"
    ]
    manifest["payload_sha256"] = hashlib.sha256(payload_bytes).hexdigest()
    manifest["entry_sha256"][payload_member] = manifest["payload_sha256"]
    manifest.pop("manifest_digest")
    manifest["manifest_digest"] = aggregate._stable_digest(manifest)
    members[payload_member] = payload_bytes
    members[manifest_member] = aggregate._json_bytes(manifest)
    with ZipFile(source.package_path, "w", compression=ZIP_STORED) as archive:
        for member_name, member_payload in members.items():
            archive.writestr(member_name, member_payload)
    payload_sidecar, manifest_sidecar = aggregate._sidecar_paths(
        source.package_path
    )
    payload_sidecar.unlink()
    manifest_sidecar.unlink()

    with pytest.raises(
        RandomizationAggregateProvenanceError,
        match="repeat 身份未精确匹配权威注册表",
    ):
        validate_randomization_aggregate_provenance(
            source.package_path,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_aggregate_validation_rejects_path_traversal_member(
    tmp_path: Path,
    aggregate_inputs,
) -> None:
    """额外路径穿越成员不得进入自包含 aggregate ZIP."""

    source = _write_aggregate(tmp_path, aggregate_inputs)
    with ZipFile(source.package_path, "a", compression=ZIP_STORED) as archive:
        archive.writestr("../forged.zip", b"forged")

    with pytest.raises(RandomizationAggregateProvenanceError, match="不安全成员"):
        validate_randomization_aggregate_provenance(
            source.package_path,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_aggregate_rejects_output_escape_and_input_symlink(
    tmp_path: Path,
    aggregate_inputs,
) -> None:
    """输出穿越和输入符号链接都不能绕过文件系统边界."""

    repeat_paths, invariant_paths, _versions, _calls = aggregate_inputs
    with pytest.raises(RandomizationAggregateProvenanceError, match="outputs 下"):
        write_randomization_aggregate_provenance_package(
            repeat_paths,
            invariant_paths,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            root=tmp_path,
            output_dir=tmp_path / "outside",
        )

    first_repeat = formal_randomization_repeat_ids()[0]
    link_path = tmp_path / "linked_repeat.zip"
    try:
        link_path.symlink_to(repeat_paths[first_repeat])
    except OSError as exc:
        pytest.skip(f"当前平台不能创建文件符号链接: {exc}")
    linked_inputs = dict(repeat_paths)
    linked_inputs[first_repeat] = link_path
    with pytest.raises(RandomizationAggregateProvenanceError, match="符号链接"):
        write_randomization_aggregate_provenance_package(
            linked_inputs,
            invariant_paths,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            root=tmp_path,
        )

def test_aggregate_validator_rejects_final_package_symlink(
    tmp_path: Path,
) -> None:
    """独立 validator 必须在解析目标前拒绝最终 aggregate 符号链接."""

    target_path = tmp_path / "aggregate_target.zip"
    with ZipFile(target_path, "w", compression=ZIP_STORED) as archive:
        archive.writestr("opaque.bin", b"target")
    link_path = tmp_path / "aggregate_link.zip"
    try:
        link_path.symlink_to(target_path)
    except OSError as exc:
        pytest.skip(f"当前平台不能创建文件符号链接: {exc}")

    with pytest.raises(
        RandomizationAggregateProvenanceError,
        match="不得是符号链接",
    ):
        validate_randomization_aggregate_provenance(
            link_path,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )

def test_aggregate_writer_rejects_run_fpr_drift_before_reading_inputs(
    tmp_path: Path,
) -> None:
    """Aggregate Python API 不得绕过运行层级冻结的统计工作点."""

    with pytest.raises(ValueError, match="必须使用冻结值"):
        write_randomization_aggregate_provenance_package(
            {},
                {},
                paper_run_name="pilot_paper",
                target_fpr=TARGET_FPR,
            root=tmp_path,
        )

    assert not (tmp_path / "outputs").exists()

def test_aggregate_rebuild_command_uses_safe_self_contained_extraction(
    tmp_path: Path,
    aggregate_inputs,
) -> None:
    """Manifest 登记的重建入口必须从来源 ZIP 安全提取并重新复验12个输入."""

    source = _write_aggregate(tmp_path, aggregate_inputs)
    rebuilt = aggregate.rebuild_randomization_aggregate_provenance_package(
        source.package_path,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        root=tmp_path,
        output_dir="outputs/rebuilt_randomization_aggregate",
    )

    assert rebuilt.package_path.is_file()
    assert rebuilt.package_path != source.package_path
    assert rebuilt.randomization_aggregate_digest == (
        source.randomization_aggregate_digest
    )
    assert rebuilt.common_code_version == source.common_code_version
    assert rebuilt.payload["randomization_aggregate_ready"] is True
    assert rebuilt.payload["supports_paper_claim"] is False
