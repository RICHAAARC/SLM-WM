"""验证单重复7类 leaf ZIP 的自包含证据包契约."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_STORED, ZipFile

import pytest

from experiments.protocol.formal_randomization import (
    formal_randomization_protocol_record,
    resolve_formal_randomization_repeat,
)
from paper_experiments.analysis.paper_claim_field_policy import (
    find_component_paper_claim_violation,
    find_zip_paper_claim_violation,
)
from paper_experiments.runners import randomization_repeat_evidence as evidence_module
from paper_experiments.runners.randomization_repeat_evidence import (
    RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES,
    RandomizationRepeatEvidenceError,
    validate_randomization_repeat_evidence_package,
    write_randomization_repeat_evidence_package,
)


pytestmark = pytest.mark.quick

PAPER_RUN_NAME = "probe_paper"
TARGET_FPR = 0.1
REPEAT_ID = "seed_01_key_02"
CODE_VERSION = "a" * 40


def _file_sha256(path: Path) -> str:
    """计算测试 leaf 包摘要."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inspected_candidate(package_path: Path, *, package_family: str):
    """构造与 manifest 身份一致的轻量 selector 复验结果."""

    index = RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES.index(package_family)
    return SimpleNamespace(
        package_family=package_family,
        package_sha256=_file_sha256(Path(package_path)),
        code_version=CODE_VERSION,
        formal_execution_run_lock_digest=f"{index + 1:x}" * 64,
        formal_execution_package_lock_digest=f"{index + 2:x}" * 64,
    )


@pytest.fixture(autouse=True)
def _inspect_nested_leaf_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """用轻量候选模拟已由独立 selector 覆盖的嵌套包内复验."""

    def inspect(package_path: Path, *, spec, **_kwargs):
        return _inspected_candidate(
            Path(package_path),
            package_family=spec.package_family,
        )

    monkeypatch.setattr(evidence_module, "inspect_closure_package", inspect)


def _candidates(tmp_path: Path, *, changed_repeat: str | None = None):
    """构造按稳定 family 顺序排列的7个选择器候选."""

    repeat = resolve_formal_randomization_repeat(changed_repeat or REPEAT_ID)
    protocol_digest = formal_randomization_protocol_record()[
        "formal_randomization_protocol_digest"
    ]
    candidates = []
    for index, family in enumerate(RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES):
        path = tmp_path / "leaf" / f"{family}.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(path, "w", compression=ZIP_STORED) as archive:
            archive.writestr("evidence.json", f'{{"family":"{family}"}}\n')
        candidates.append(
            SimpleNamespace(
                package_family=family,
                package_path=path,
                package_sha256=_file_sha256(path),
                paper_run_name=PAPER_RUN_NAME,
                target_fpr=TARGET_FPR,
                code_version=CODE_VERSION,
                formal_execution_run_lock_digest=f"{index + 1:x}" * 64,
                formal_execution_package_lock_digest=f"{index + 2:x}" * 64,
                randomization_repeat_id=repeat.randomization_repeat_id,
                generation_seed_index=repeat.generation_seed_index,
                generation_seed_offset=repeat.generation_seed_offset,
                watermark_key_index=repeat.watermark_key_index,
                formal_randomization_protocol_digest=protocol_digest,
                randomization_scope="active_repeat_component",
            )
        )
    return tuple(candidates)


def test_repeat_evidence_package_nests_exact_leaf_zips_without_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """写出结果必须包含7个原样 leaf ZIP 和一个受治理 manifest."""

    candidates = _candidates(tmp_path)
    monkeypatch.setattr(
        evidence_module,
        "select_randomization_repeat_package_candidates",
        lambda *args, **kwargs: candidates,
    )

    report = write_randomization_repeat_evidence_package(
        tmp_path / "search",
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        randomization_repeat_id=REPEAT_ID,
        root=tmp_path,
    )

    archive_path = Path(report["archive_path"])
    assert archive_path.is_file()
    assert report["archive_entry_count"] == 8
    assert report["repeat_component_ready"] is True
    assert report["randomization_aggregate_ready"] is False
    assert report["supports_paper_claim"] is False
    assert report["randomization_repeat_id"] == REPEAT_ID
    assert report["code_version"] == CODE_VERSION
    assert set(report["leaf_package_sha256_map"]) == set(
        RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES
    )
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        leaf_names = [name for name in names if name.endswith(".zip")]
        assert set(leaf_names) == {
            evidence_module._leaf_member_name(
                randomization_repeat_id=REPEAT_ID,
                package_family=family,
            )
            for family in RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES
        }
        assert len(names) == 8
        for candidate in candidates:
            member = next(
                name
                for name in leaf_names
                if name.endswith(f"/{candidate.package_family}.zip")
            )
            assert archive.read(member) == Path(candidate.package_path).read_bytes()


def test_repeat_evidence_rejects_leaf_from_another_repeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任一 leaf 身份漂移时不得生成单重复证据包."""

    candidates = list(_candidates(tmp_path))
    foreign = _candidates(tmp_path / "foreign", changed_repeat="seed_02_key_02")[0]
    candidates[0] = foreign
    monkeypatch.setattr(
        evidence_module,
        "select_randomization_repeat_package_candidates",
        lambda *args, **kwargs: tuple(candidates),
    )

    with pytest.raises(RandomizationRepeatEvidenceError, match="随机化重复身份"):
        write_randomization_repeat_evidence_package(
            tmp_path / "search",
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            randomization_repeat_id=REPEAT_ID,
            root=tmp_path,
        )


def test_repeat_evidence_write_after_validation_rejects_leaf_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """外层包内任一 leaf 字节被替换后必须由写后校验发现."""

    candidates = _candidates(tmp_path)
    monkeypatch.setattr(
        evidence_module,
        "select_randomization_repeat_package_candidates",
        lambda *args, **kwargs: candidates,
    )
    report = write_randomization_repeat_evidence_package(
        tmp_path / "search",
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        randomization_repeat_id=REPEAT_ID,
        root=tmp_path,
    )
    archive_path = Path(report["archive_path"])
    with ZipFile(archive_path) as source:
        members = [(info.filename, source.read(info.filename)) for info in source.infolist()]
    leaf_name = next(name for name, _payload in members if name.endswith(".zip"))
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as target:
        for name, payload in members:
            target.writestr(name, b"tampered" if name == leaf_name else payload)

    with pytest.raises(RandomizationRepeatEvidenceError, match="leaf 摘要"):
        validate_randomization_repeat_evidence_package(
            archive_path,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            randomization_repeat_id=REPEAT_ID,
        )


def test_repeat_evidence_rejects_nested_leaf_contract_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """外层摘要正确也不能替代嵌套 leaf 的包内契约复验."""

    candidates = _candidates(tmp_path)
    monkeypatch.setattr(
        evidence_module,
        "select_randomization_repeat_package_candidates",
        lambda *args, **kwargs: candidates,
    )
    monkeypatch.setattr(
        evidence_module,
        "inspect_closure_package",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("嵌套包身份失败")
        ),
    )

    with pytest.raises(
        RandomizationRepeatEvidenceError,
        match="嵌套 leaf ZIP 未通过独立包内契约复验",
    ):
        write_randomization_repeat_evidence_package(
            tmp_path / "search",
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            randomization_repeat_id=REPEAT_ID,
            root=tmp_path,
        )


@pytest.mark.parametrize(
    "positive_claim_payload",
    (
        {"supports_paper_claim": True},
        {"claim_decision": "paper_supported"},
        {"entry_review_decision": "ready_for_evidence_closure"},
        {"paper_ready": True},
    ),
)
def test_repeat_evidence_rejects_positive_claim_inside_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    positive_claim_payload: dict[str, object],
) -> None:
    """leaf 内布尔或枚举形式的正向论文结论都不能进入单重复组件."""

    candidates = list(_candidates(tmp_path))
    forged_path = Path(candidates[0].package_path)
    with ZipFile(forged_path, "w", compression=ZIP_STORED) as archive:
        archive.writestr(
            "summary.json",
            json.dumps(
                {
                    "repeat_component_ready": True,
                    "randomization_aggregate_ready": False,
                    "supports_paper_claim": False,
                    **positive_claim_payload,
                }
            ),
        )
    candidates[0].package_sha256 = _file_sha256(forged_path)
    monkeypatch.setattr(
        evidence_module,
        "select_randomization_repeat_package_candidates",
        lambda *args, **kwargs: tuple(candidates),
    )

    with pytest.raises(RandomizationRepeatEvidenceError, match="正向论文结论字段"):
        write_randomization_repeat_evidence_package(
            tmp_path / "search",
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            randomization_repeat_id=REPEAT_ID,
            root=tmp_path,
        )


@pytest.mark.parametrize(
    "field_name,field_value",
    (
        ("formal_result_claim", True),
        ("paper_claim_support", True),
        ("paper_run_supports_superiority_claim", True),
        ("ablation_standalone_claim_ready", True),
        ("strong_ablation_standalone_claim_ready", True),
        ("supports_main_table_superiority_claim", True),
        ("supports_quality_matched_paper_claim", "yes"),
        ("supports_paper_claim", 1),
        ("supports_paper_claim", "1"),
    ),
)
def test_positive_claim_field_classifier_covers_governed_name_forms(
    field_name: str,
    field_value: object,
) -> None:
    """显式登记的布尔结论字段必须拒绝所有正向表示."""

    violation = find_component_paper_claim_violation(
        {field_name: field_value}
    )
    assert violation is not None
    assert violation.path == f"$.{field_name}"


@pytest.mark.parametrize(
    "member_name,member_payload,expected_path",
    (
        (
            "summary.json",
            '{"paper_ready":true}\n',
            "summary.json[0].paper_ready",
        ),
        (
            "records.jsonl",
            '{"supports_paper_claim":false}\n'
            '{"claim_decision":"paper_supported"}\n',
            "records.jsonl[1].claim_decision",
        ),
        (
            "table.csv",
            "paper_ready,claim_decision\nfalse,unsupported\ntrue,unsupported\n",
            "table.csv[1].paper_ready",
        ),
    ),
)
def test_zip_claim_scanner_covers_all_structured_member_formats(
    tmp_path: Path,
    member_name: str,
    member_payload: str,
    expected_path: str,
) -> None:
    """共享扫描器必须统一解释 JSON、JSONL 与 CSV 成员."""

    archive_path = tmp_path / "structured_claim.zip"
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        archive.writestr(member_name, member_payload)

    violation = find_zip_paper_claim_violation(archive_path)
    assert violation is not None
    assert violation.path == expected_path


@pytest.mark.parametrize(
    "component_payload",
    (
        {"supports_paper_claim": False},
        {"supports_paper_claim": "false"},
        {"claim_decision": "unsupported"},
        {"claim_decision": "preview_only"},
        {"claim_decision": "engineering_supported_not_paper_final"},
        {"entry_review_decision": "blocked_before_evidence_closure"},
        {"readiness_decision": "blocked"},
        {"closure_decision": "blocked"},
        {"paper_ready": False},
        {"paper_claim_support": False},
        {"ablation_standalone_claim_ready": False},
        {"superiority_claim_ready_count": 0},
        {
            "claim_id": "claim_internal_mechanism_necessity",
            "claim_text": "该字段只保存主张身份和文本, 不保存支持结论",
        },
        {"necessity_component_supported": True},
    ),
)
def test_component_claim_policy_accepts_only_registered_negative_states(
    component_payload: dict[str, object],
) -> None:
    """登记负值与普通组件状态必须通过, 不得依赖字段名子串猜测."""

    assert find_component_paper_claim_violation(component_payload) is None


@pytest.mark.parametrize(
    "component_payload,expected_path",
    (
        ({"claim_decision": "paper_supported"}, "$.claim_decision"),
        (
            {"entry_review_decision": "ready_for_evidence_closure"},
            "$.entry_review_decision",
        ),
        ({"readiness_decision": "ready"}, "$.readiness_decision"),
        ({"closure_decision": "pass"}, "$.closure_decision"),
        ({"paper_ready": "True"}, "$.paper_ready"),
        ({"supports_paper_claim": None}, "$.supports_paper_claim"),
        ({"claim_decision": ""}, "$.claim_decision"),
        (
            {"nested": [{"superiority_claim_ready_count": 1}]},
            "$.nested[0].superiority_claim_ready_count",
        ),
    ),
)
def test_component_claim_policy_rejects_values_outside_registered_negative_set(
    component_payload: dict[str, object],
    expected_path: str,
) -> None:
    """正向或含糊值不得冒充登记负值, 且必须返回精确嵌套路径."""

    violation = find_component_paper_claim_violation(component_payload)
    assert violation is not None
    assert violation.path == expected_path


@pytest.mark.parametrize(
    "drift_field,drift_value",
    (
        ("code_version", "b" * 40),
        ("formal_execution_run_lock_digest", "c" * 64),
        ("formal_execution_package_lock_digest", "d" * 64),
    ),
)
def test_repeat_evidence_rejects_nested_code_or_lock_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift_field: str,
    drift_value: str,
) -> None:
    """嵌套复验得到的代码版本或执行锁漂移时必须拒绝外层包."""

    candidates = _candidates(tmp_path)
    monkeypatch.setattr(
        evidence_module,
        "select_randomization_repeat_package_candidates",
        lambda *args, **kwargs: candidates,
    )
    report = write_randomization_repeat_evidence_package(
        tmp_path / "search",
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        randomization_repeat_id=REPEAT_ID,
        root=tmp_path,
    )

    def inspect_with_drift(package_path: Path, *, spec, **_kwargs):
        candidate = _inspected_candidate(
            Path(package_path),
            package_family=spec.package_family,
        )
        if spec.package_family == RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES[0]:
            setattr(candidate, drift_field, drift_value)
        return candidate

    monkeypatch.setattr(
        evidence_module,
        "inspect_closure_package",
        inspect_with_drift,
    )
    with pytest.raises(RandomizationRepeatEvidenceError, match="复验身份不一致"):
        validate_randomization_repeat_evidence_package(
            report["archive_path"],
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            randomization_repeat_id=REPEAT_ID,
        )


def test_repeat_evidence_rejects_resigned_noncanonical_leaf_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """攻击者即使重签 manifest, 也不能改变 family 的规范包内路径."""

    candidates = _candidates(tmp_path)
    monkeypatch.setattr(
        evidence_module,
        "select_randomization_repeat_package_candidates",
        lambda *args, **kwargs: candidates,
    )
    report = write_randomization_repeat_evidence_package(
        tmp_path / "search",
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        randomization_repeat_id=REPEAT_ID,
        root=tmp_path,
    )
    archive_path = Path(report["archive_path"])
    manifest_name = evidence_module._manifest_member_name(REPEAT_ID)
    with ZipFile(archive_path) as source:
        payloads = {
            info.filename: source.read(info.filename)
            for info in source.infolist()
        }
    manifest = json.loads(payloads.pop(manifest_name).decode("utf-8"))
    record = manifest["leaf_packages"][0]
    canonical_member = record["archive_member"]
    renamed_member = canonical_member.replace("/leaf_packages/", "/renamed/")
    payloads[renamed_member] = payloads.pop(canonical_member)
    record["archive_member"] = renamed_member
    manifest["leaf_package_set_digest"] = evidence_module._stable_digest(
        manifest["leaf_packages"]
    )
    content_payload = {
        "paper_run_name": manifest["paper_run_name"],
        "target_fpr": manifest["target_fpr"],
        **evidence_module._repeat_identity(REPEAT_ID),
        "formal_randomization_repeat_registry_digest": manifest[
            "formal_randomization_repeat_registry_digest"
        ],
        "code_version": manifest["code_version"],
        "leaf_packages": manifest["leaf_packages"],
        "repeat_component_ready": True,
        "randomization_aggregate_ready": False,
        "supports_paper_claim": False,
    }
    manifest["component_content_digest"] = evidence_module._stable_digest(
        content_payload
    )
    manifest.pop("randomization_repeat_evidence_manifest_digest")
    manifest["randomization_repeat_evidence_manifest_digest"] = (
        evidence_module._stable_digest(manifest)
    )
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as target:
        for member_name, payload in payloads.items():
            target.writestr(member_name, payload)
        target.writestr(
            manifest_name,
            json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        )

    with pytest.raises(RandomizationRepeatEvidenceError, match="结论边界未通过"):
        validate_randomization_repeat_evidence_package(
            archive_path,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            randomization_repeat_id=REPEAT_ID,
        )


@pytest.mark.parametrize(
    "field_name,forged_value",
    (
        ("generation_seed_index", True),
        ("generation_seed_index", 1.0),
        ("generation_seed_offset", 1_000_003.0),
        ("watermark_key_index", 2.0),
    ),
)
def test_repeat_evidence_rejects_resigned_non_integer_repeat_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    forged_value: object,
) -> None:
    """重签 manifest 后, bool 或 float 仍不得冒充 repeat 整数身份字段."""

    candidates = _candidates(tmp_path)
    monkeypatch.setattr(
        evidence_module,
        "select_randomization_repeat_package_candidates",
        lambda *args, **kwargs: candidates,
    )
    report = write_randomization_repeat_evidence_package(
        tmp_path / "search",
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        randomization_repeat_id=REPEAT_ID,
        root=tmp_path,
    )
    archive_path = Path(report["archive_path"])
    manifest_name = evidence_module._manifest_member_name(REPEAT_ID)
    with ZipFile(archive_path) as source:
        payloads = {
            info.filename: source.read(info.filename)
            for info in source.infolist()
        }
    manifest = json.loads(payloads[manifest_name].decode("utf-8"))
    assert type(manifest[field_name]) is int
    assert manifest[field_name] == forged_value
    manifest[field_name] = forged_value
    manifest.pop("randomization_repeat_evidence_manifest_digest")
    manifest["randomization_repeat_evidence_manifest_digest"] = (
        evidence_module._stable_digest(manifest)
    )
    payloads[manifest_name] = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as target:
        for member_name, payload in payloads.items():
            target.writestr(member_name, payload)

    with pytest.raises(RandomizationRepeatEvidenceError, match="正式随机化身份无效"):
        validate_randomization_repeat_evidence_package(
            archive_path,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            randomization_repeat_id=REPEAT_ID,
        )


@pytest.mark.parametrize(
    "output_dir_factory",
    (
        lambda root: root / "outputs" / ".." / "outside",
        lambda root: root.parent / "absolute_outside",
    ),
)
def test_repeat_evidence_rejects_output_directory_outside_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_dir_factory,
) -> None:
    """相对穿越和绝对外部目录都不能绕过 outputs 写入边界."""

    candidates = _candidates(tmp_path)
    monkeypatch.setattr(
        evidence_module,
        "select_randomization_repeat_package_candidates",
        lambda *args, **kwargs: candidates,
    )
    with pytest.raises(RandomizationRepeatEvidenceError, match="必须位于 outputs 下"):
        write_randomization_repeat_evidence_package(
            tmp_path / "search",
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            randomization_repeat_id=REPEAT_ID,
            root=tmp_path,
            output_dir=output_dir_factory(tmp_path),
        )


def test_repeat_evidence_rejects_output_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """outputs 内指向外部目录的符号链接不能扩大持久化边界."""

    candidates = _candidates(tmp_path)
    monkeypatch.setattr(
        evidence_module,
        "select_randomization_repeat_package_candidates",
        lambda *args, **kwargs: candidates,
    )
    outputs_root = tmp_path / "outputs"
    outside_root = tmp_path / "outside"
    outputs_root.mkdir()
    outside_root.mkdir()
    link = outputs_root / "escaped"
    try:
        link.symlink_to(outside_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"当前平台不能创建目录符号链接: {exc}")
    with pytest.raises(RandomizationRepeatEvidenceError, match="必须位于 outputs 下"):
        write_randomization_repeat_evidence_package(
            tmp_path / "search",
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            randomization_repeat_id=REPEAT_ID,
            root=tmp_path,
            output_dir=link / "nested",
        )


def test_repeat_evidence_rejects_leaf_and_outer_archive_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """leaf 输入和待验证外层归档都必须是普通文件而非符号链接."""

    candidates = list(_candidates(tmp_path))
    original_leaf = Path(candidates[0].package_path)
    leaf_link = original_leaf.with_name("linked_leaf.zip")
    try:
        leaf_link.symlink_to(original_leaf)
    except OSError as exc:
        pytest.skip(f"当前平台不能创建文件符号链接: {exc}")
    candidates[0].package_path = leaf_link
    candidates[0].package_sha256 = _file_sha256(original_leaf)
    monkeypatch.setattr(
        evidence_module,
        "select_randomization_repeat_package_candidates",
        lambda *args, **kwargs: tuple(candidates),
    )
    with pytest.raises(RandomizationRepeatEvidenceError, match="不得是符号链接"):
        write_randomization_repeat_evidence_package(
            tmp_path / "search",
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            randomization_repeat_id=REPEAT_ID,
            root=tmp_path,
        )

    valid_candidates = _candidates(tmp_path / "valid")
    monkeypatch.setattr(
        evidence_module,
        "select_randomization_repeat_package_candidates",
        lambda *args, **kwargs: valid_candidates,
    )
    report = write_randomization_repeat_evidence_package(
        tmp_path / "search",
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        randomization_repeat_id=REPEAT_ID,
        root=tmp_path,
    )
    archive_path = Path(report["archive_path"])
    archive_link = archive_path.with_name("linked_repeat_evidence.zip")
    archive_link.symlink_to(archive_path)
    with pytest.raises(RandomizationRepeatEvidenceError, match="不得是符号链接"):
        validate_randomization_repeat_evidence_package(
            archive_link,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            randomization_repeat_id=REPEAT_ID,
        )

def test_repeat_writer_rejects_run_fpr_drift_before_reading_inputs(
    tmp_path: Path,
) -> None:
    """单 repeat Python API 不得绕过运行层级冻结的统计工作点."""

    with pytest.raises(ValueError, match="必须使用冻结值"):
        write_randomization_repeat_evidence_package(
            tmp_path / "missing_inputs",
            paper_run_name="pilot_paper",
            target_fpr=0.01,
            randomization_repeat_id=REPEAT_ID,
            root=tmp_path,
        )

    assert not (tmp_path / "outputs").exists()
