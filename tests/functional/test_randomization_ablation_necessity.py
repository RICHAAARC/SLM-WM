"""验证精确9重复消融原始记录到必要性结果的生产边界。"""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.ablations.runtime_rerun import (
    FORMAL_RUNTIME_RERUN_ABLATION_IDS,
    FORMAL_RUNTIME_RERUN_ABLATION_SPECS,
)
from experiments.protocol.formal_randomization import (
    formal_randomization_repeat_ids,
)
from main.core.digest import build_stable_digest
from paper_experiments.runners.randomization_ablation_necessity import (
    RandomizationAblationNecessityError,
    _rebuild_randomization_ablation_necessity,
    rebuild_randomization_ablation_necessity,
    write_randomization_ablation_necessity_outputs,
)
from paper_experiments.runners.randomization_aggregate_provenance import (
    RandomizationAggregateProvenance,
)
from paper_experiments.runners.randomization_aggregate_record_workspace import (
    RandomizationAggregateRecordSource,
)


CODE_VERSION = "a" * 40
PACKAGE_SHA256 = "b" * 64
AGGREGATE_DIGEST = "c" * 64


def _provenance() -> RandomizationAggregateProvenance:
    """构造已通过类型边界的最小不可变来源对象。"""

    repeat_ids = formal_randomization_repeat_ids()
    payload = {
        "paper_run_name": "probe_paper",
        "target_fpr": 0.1,
        "randomization_aggregate_ready": True,
        "supports_paper_claim": False,
        "randomization_aggregate_digest": AGGREGATE_DIGEST,
        "common_code_version": CODE_VERSION,
        "randomization_repeat_ids": list(repeat_ids),
    }
    return RandomizationAggregateProvenance(
        package_path=Path("aggregate.zip"),
        package_sha256=PACKAGE_SHA256,
        payload_path="aggregate.zip!/payload.json",
        payload_sha256="d" * 64,
        manifest_path="aggregate.zip!/manifest.json",
        manifest_sha256="e" * 64,
        payload=payload,
        manifest={},
        randomization_repeat_components=tuple(
            {
                "randomization_repeat_id": repeat_id,
                "package_sha256": f"{repeat_index + 1:064x}",
            }
            for repeat_index, repeat_id in enumerate(repeat_ids)
        ),
        invariant_packages=(),
        common_code_version=CODE_VERSION,
        randomization_aggregate_digest=AGGREGATE_DIGEST,
    )


def _prompt_rows() -> tuple[dict[str, Any], ...]:
    """构造70个规范 Prompt, 其中34个属于正式 test split。"""

    return tuple(
        {
            "prompt_id": f"prompt_{prompt_index:03d}",
            "prompt_index": prompt_index,
            "prompt_text": f"提示词 {prompt_index}",
            "prompt_digest": f"{prompt_index + 1:064x}",
            "split": "test" if prompt_index < 34 else "calibration",
        }
        for prompt_index in range(70)
    )


def _runtime_records() -> tuple[dict[str, Any], ...]:
    """构造每个变体共享34个 test Prompt 的逐 Prompt 运行记录。"""

    return tuple(
        {
            "ablation_id": ablation_id,
            "prompt_id": prompt["prompt_id"],
            "prompt_index": prompt["prompt_index"],
            "prompt_digest": prompt["prompt_digest"],
            "split": "test",
            "formal_attack_coverage_ready": True,
            "attacked_positive_rate": (
                1.0 if ablation_id == "complete_method" else 0.0
            ),
            "positive_source_positive": ablation_id == "complete_method",
            "paired_ssim": 0.95,
        }
        for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS
        for prompt in _prompt_rows()
        if prompt["split"] == "test"
    )


def _source(repeat_id: str, role: str) -> RandomizationAggregateRecordSource:
    """构造绑定同一聚合来源的规范成员描述符。"""

    return RandomizationAggregateRecordSource(
        randomization_scope="active_repeat_component",
        randomization_repeat_id=repeat_id,
        package_family="runtime_rerun_ablation",
        record_group="ablation",
        record_role=role,
        record_format="json_object",
        record_member=f"records/{repeat_id}/{role}.json",
        record_sha256=build_stable_digest({"repeat": repeat_id, "role": role}),
        leaf_package_sha256="1" * 64,
        randomization_repeat_component_sha256="2" * 64,
        randomization_repeat_evidence_manifest_digest="3" * 64,
        component_content_digest="4" * 64,
        randomization_aggregate_package_sha256=PACKAGE_SHA256,
        common_code_version=CODE_VERSION,
        randomization_aggregate_digest=AGGREGATE_DIGEST,
    )


class _AggregateWorkspace:
    """提供适配器测试所需的四类已登记消融来源。"""

    def __init__(self, *, manifest_repeat_override: str | None = None) -> None:
        self.manifest_repeat_override = manifest_repeat_override

    def find_source(
        self,
        *,
        randomization_repeat_id: str,
        package_family: str,
        record_role: str,
    ) -> RandomizationAggregateRecordSource:
        assert package_family == "runtime_rerun_ablation"
        return _source(randomization_repeat_id, record_role)

    def iter_records(
        self,
        source: RandomizationAggregateRecordSource,
    ) -> tuple[dict[str, Any], ...]:
        if source.record_role == "ablation_runtime_record":
            return _runtime_records()
        return ({"detection": source.randomization_repeat_id},)

    def read_object(
        self,
        source: RandomizationAggregateRecordSource,
    ) -> dict[str, Any]:
        if source.record_role == "ablation_frozen_protocol":
            return {
                ablation_id: {"protocol": ablation_id}
                for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS
            }
        repeat_id = (
            self.manifest_repeat_override
            or source.randomization_repeat_id
        )
        return {
            "code_version": CODE_VERSION,
            "config": {
                "specs": [
                    spec.to_dict()
                    for spec in FORMAL_RUNTIME_RERUN_ABLATION_SPECS
                ],
                "target_fpr": 0.1,
                "randomization_repeat_id": repeat_id,
                "scientific_unit_identity_records": [],
                "formal_randomization_plan": {},
                "randomization_repeat_identity": {},
            },
        }


def _install_sources(
    monkeypatch: pytest.MonkeyPatch,
    workspace: _AggregateWorkspace,
) -> list[str]:
    """替换临时工作区 I/O, 保留适配器本身的生产控制流。"""

    import paper_experiments.runners.randomization_ablation_necessity as module

    atomic_repeat_ids: list[str] = []
    monkeypatch.setattr(
        module,
        "open_randomization_aggregate_record_workspace",
        lambda _source: nullcontext(workspace),
    )
    monkeypatch.setattr(
        module,
        "rebuild_randomization_prompt_source_contract",
        lambda *_args, **_kwargs: {
            "prompt_rows": _prompt_rows(),
            "report": {
                "prompt_source_contract_digest": "5" * 64,
                "prompt_rows_digest": "6" * 64,
            },
        },
    )

    def _atomic_rebuild(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        repeat_id = kwargs["randomization_repeat_identity"].get(
            "randomization_repeat_id",
            "",
        )
        atomic_repeat_ids.append(repeat_id)
        return {
            "ablation_runtime_record_count": len(_runtime_records()),
            "ablation_detection_record_count": 1,
            "ablation_frozen_protocol_count": len(
                FORMAL_RUNTIME_RERUN_ABLATION_IDS
            ),
            "ablation_runtime_records_digest": "7" * 64,
            "ablation_detection_records_digest": "8" * 64,
            "ablation_frozen_protocols_digest": "9" * 64,
            "ablation_rebuilt_runtime_aggregates_digest": "a" * 64,
            "ablation_expected_runtime_configs_digest": "b" * 64,
            "ablation_image_only_measurement_config_digests": {},
            "ablation_runtime_aggregate_rebuild_ready": True,
        }

    monkeypatch.setattr(
        module,
        "rebuild_and_validate_ablation_runtime_aggregates",
        _atomic_rebuild,
    )
    return atomic_repeat_ids


@pytest.mark.quick
def test_public_rebuild_locks_clean_commit_and_formal_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式入口不得用不同提交或缩减 bootstrap 生成 claim 结果。"""

    import paper_experiments.runners.randomization_ablation_necessity as module

    marker = object()
    observed_resample_counts: list[int] = []
    monkeypatch.setattr(
        module,
        "resolve_code_version",
        lambda _root: CODE_VERSION,
    )

    def _rebuild(_source: Any, *, bootstrap_resample_count: int) -> Any:
        observed_resample_counts.append(bootstrap_resample_count)
        return marker

    monkeypatch.setattr(
        module,
        "_rebuild_randomization_ablation_necessity",
        _rebuild,
    )
    assert rebuild_randomization_ablation_necessity(
        _provenance(),
        root=tmp_path,
    ) is marker
    assert observed_resample_counts == [100_000]

    monkeypatch.setattr(
        module,
        "resolve_code_version",
        lambda _root: "f" * 40,
    )
    with pytest.raises(RandomizationAblationNecessityError, match="clean Git"):
        rebuild_randomization_ablation_necessity(
            _provenance(),
            root=tmp_path,
        )


@pytest.mark.quick
def test_adapter_rebuilds_all_repeats_before_claim_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """生产适配器必须读取9份原始消融而不是单重复派生统计。"""

    repeat_ids = formal_randomization_repeat_ids()
    workspace = _AggregateWorkspace()
    atomic_repeat_ids = _install_sources(monkeypatch, workspace)

    # 原子重建器收到的身份来自 manifest; 为本测试补入每次来源的正式 ID。
    original_read_object = workspace.read_object

    def _read_object(source: RandomizationAggregateRecordSource) -> dict[str, Any]:
        value = original_read_object(source)
        if source.record_role == "ablation_run_manifest":
            value["config"]["randomization_repeat_identity"] = {
                "randomization_repeat_id": source.randomization_repeat_id
            }
        return value

    workspace.read_object = _read_object  # type: ignore[method-assign]
    result = _rebuild_randomization_ablation_necessity(
        _provenance(),
        bootstrap_resample_count=20,
    )

    assert tuple(atomic_repeat_ids) == repeat_ids
    assert len(result.rows) == len(FORMAL_RUNTIME_RERUN_ABLATION_IDS) - 1
    assert result.summary["paired_prompt_count"] == 34
    assert result.summary["paired_observation_count"] == 9 * 34
    assert result.report["randomization_repeat_ids"] == list(repeat_ids)
    assert len(result.report["repeat_rebuild_records"]) == 9
    assert result.report["randomization_aggregate_statistics_ready"] is True
    assert result.report["supports_paper_claim"] is True


@pytest.mark.quick
def test_adapter_rejects_manifest_repeat_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """来源成员与消融 manifest 的 repeat 身份冲突必须在统计前拒绝。"""

    workspace = _AggregateWorkspace(manifest_repeat_override="seed_99_key_99")
    atomic_repeat_ids = _install_sources(monkeypatch, workspace)

    with pytest.raises(RandomizationAblationNecessityError, match="manifest"):
        _rebuild_randomization_ablation_necessity(
            _provenance(),
            bootstrap_resample_count=20,
        )
    assert not atomic_repeat_ids


@pytest.mark.quick
def test_writer_persists_only_rebuilt_aggregate_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writer 必须在完整重建成功后才创建 outputs 内的最小结果集。"""

    import paper_experiments.runners.randomization_ablation_necessity as module

    workspace = _AggregateWorkspace()
    _install_sources(monkeypatch, workspace)
    original_read_object = workspace.read_object

    def _read_object(source: RandomizationAggregateRecordSource) -> dict[str, Any]:
        value = original_read_object(source)
        if source.record_role == "ablation_run_manifest":
            value["config"]["randomization_repeat_identity"] = {
                "randomization_repeat_id": source.randomization_repeat_id
            }
        return value

    workspace.read_object = _read_object  # type: ignore[method-assign]
    source = _provenance()
    result = _rebuild_randomization_ablation_necessity(
        source,
        bootstrap_resample_count=20,
    )
    monkeypatch.setattr(
        module,
        "rebuild_randomization_ablation_necessity",
        lambda *_args, **_kwargs: result,
    )

    manifest_path = write_randomization_ablation_necessity_outputs(
        source,
        root=tmp_path,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.is_relative_to(tmp_path / "outputs")
    assert manifest["metadata"]["randomization_aggregate_statistics_ready"] is True
    assert manifest["config"]["randomization_aggregate_package_sha256"] == (
        PACKAGE_SHA256
    )
    assert len(manifest["output_paths"]) == 4
    manifest_bytes = manifest_path.read_bytes()
    with pytest.raises(
        RandomizationAblationNecessityError,
        match="不得覆盖或混选",
    ):
        write_randomization_ablation_necessity_outputs(
            source,
            root=tmp_path,
        )
    assert manifest_path.read_bytes() == manifest_bytes


@pytest.mark.quick
def test_writer_removes_unpublished_files_after_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任一写出异常都不得留下部分结果或可混选的旧 manifest。"""

    import paper_experiments.runners.randomization_ablation_necessity as module

    workspace = _AggregateWorkspace()
    _install_sources(monkeypatch, workspace)
    original_read_object = workspace.read_object

    def _read_object(source: RandomizationAggregateRecordSource) -> dict[str, Any]:
        value = original_read_object(source)
        if source.record_role == "ablation_run_manifest":
            value["config"]["randomization_repeat_identity"] = {
                "randomization_repeat_id": source.randomization_repeat_id
            }
        return value

    workspace.read_object = _read_object  # type: ignore[method-assign]
    source = _provenance()
    result = _rebuild_randomization_ablation_necessity(
        source,
        bootstrap_resample_count=20,
    )
    monkeypatch.setattr(
        module,
        "rebuild_randomization_ablation_necessity",
        lambda *_args, **_kwargs: result,
    )

    def _reject_manifest(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("manifest 写出失败")

    monkeypatch.setattr(module, "build_artifact_manifest", _reject_manifest)
    with pytest.raises(RuntimeError, match="manifest 写出失败"):
        write_randomization_ablation_necessity_outputs(
            source,
            root=tmp_path,
        )
    destination = (
        tmp_path
        / "outputs"
        / "randomization_ablation_necessity"
        / "probe_paper"
    )
    assert not destination.exists()
    assert not tuple(destination.parent.glob(".*_publish_*"))


@pytest.mark.quick
def test_writer_leaves_no_output_when_rebuild_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """来源或重建失败时不得留下可被误读为正式结果的目录。"""

    import paper_experiments.runners.randomization_ablation_necessity as module

    def _reject(*_args: Any, **_kwargs: Any) -> Any:
        raise RandomizationAblationNecessityError("重建失败")

    monkeypatch.setattr(
        module,
        "rebuild_randomization_ablation_necessity",
        _reject,
    )
    with pytest.raises(RandomizationAblationNecessityError, match="重建失败"):
        write_randomization_ablation_necessity_outputs(
            _provenance(),
            root=tmp_path,
        )
    assert not (tmp_path / "outputs").exists()
