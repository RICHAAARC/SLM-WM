"""主表 method-faithful observation exact-set 集合的轻量功能测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_experiments.baselines.method_faithful_observation_collection import (
    FORMAL_MODEL_REVISION,
    METHOD_FAITHFUL_BASELINE_IDS,
    MethodFaithfulCollectionProtocol,
    file_sha256,
    load_method_faithful_observation_collection,
    observation_relative_path,
    transfer_manifest_relative_path,
)
from tests.helpers.method_faithful_collection import (
    collection_protocol,
    formal_observation_rows,
    prompt_rows,
    write_collection_source,
    write_complete_collection,
)


pytestmark = pytest.mark.quick


def complete_inputs(collection_root: Path) -> tuple[list[dict[str, object]], object]:
    """写出带 Prompt、adapter 和 execution 摘要绑定的完整集合。"""

    prompts = prompt_rows("probe_paper", ("calibration", "test"))
    protocol = collection_protocol(prompts)
    observations = {
        baseline_id: formal_observation_rows(baseline_id, prompts, protocol)
        for baseline_id in METHOD_FAITHFUL_BASELINE_IDS
    }
    write_complete_collection(collection_root, observations, prompts, protocol)
    return prompts, protocol


def test_collection_loads_exact_baselines_in_canonical_order(tmp_path: Path) -> None:
    """文件创建顺序不得影响 baseline 和 event 的正式读取顺序。"""

    collection_root = tmp_path / "outputs" / "external_baseline_method_faithful"
    _, protocol = complete_inputs(collection_root)

    sources = load_method_faithful_observation_collection(collection_root, protocol=protocol)

    assert tuple(source.baseline_id for source in sources) == METHOD_FAITHFUL_BASELINE_IDS
    assert all(
        tuple(row["event_id"] for row in source.rows)
        == tuple(sorted(str(row["event_id"]) for row in source.rows))
        for source in sources
    )
    assert all(source.observations_sha256 == file_sha256(source.observations_path) for source in sources)
    assert all(source.prompt_plan_path.is_file() for source in sources)
    assert all(source.adapter_manifest_path.is_file() for source in sources)
    assert all(source.execution_manifest_path.is_file() for source in sources)
    assert all(source.model_revision == FORMAL_MODEL_REVISION for source in sources)
    assert all(
        row["generation_model_revision"] == FORMAL_MODEL_REVISION
        for source in sources
        for row in source.rows
    )


def test_collection_rejects_missing_or_unexpected_observation_file(tmp_path: Path) -> None:
    """正式 collection 必须恰好包含三个已注册 baseline observation。"""

    collection_root = tmp_path / "collection"
    prompts, protocol = complete_inputs(collection_root)
    missing_path = collection_root / Path(*observation_relative_path("tree_ring").parts)
    missing_path.unlink()

    with pytest.raises(ValueError, match="method_faithful_observation_file_set_mismatch"):
        load_method_faithful_observation_collection(collection_root, protocol=protocol)

    write_collection_source(
        collection_root,
        "tree_ring",
        formal_observation_rows("tree_ring", prompts, protocol),
        prompts,
        protocol,
    )
    unexpected_path = collection_root / "split_observations" / "unknown_baseline_observations.json"
    unexpected_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="method_faithful_observation_file_set_mismatch"):
        load_method_faithful_observation_collection(collection_root, protocol=protocol)


def test_collection_rejects_manifest_baseline_or_path_mismatch(tmp_path: Path) -> None:
    """manifest 不得把当前 baseline 指向其他方法或 collection 外部路径。"""

    collection_root = tmp_path / "collection"
    prompts, protocol = complete_inputs(collection_root)
    _, manifest_path = write_collection_source(
        collection_root,
        "tree_ring",
        formal_observation_rows("tree_ring", prompts, protocol),
        prompts,
        protocol,
        manifest_overrides={"baseline_id": "gaussian_shading"},
    )

    with pytest.raises(ValueError, match="method_faithful_transfer_manifest_baseline_mismatch"):
        load_method_faithful_observation_collection(collection_root, protocol=protocol)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["baseline_id"] = "tree_ring"
    manifest["baseline_observations_path"] = "../outside.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="method_faithful_transfer_manifest_path_mismatch"):
        load_method_faithful_observation_collection(collection_root, protocol=protocol)


def test_collection_rejects_row_baseline_mismatch_and_duplicate_event_id(tmp_path: Path) -> None:
    """每个文件内的 baseline identity 和 event identity 必须唯一且一致。"""

    collection_root = tmp_path / "collection"
    prompts, protocol = complete_inputs(collection_root)
    mismatched_rows = formal_observation_rows("tree_ring", prompts, protocol)
    mismatched_rows[0]["baseline_id"] = "gaussian_shading"
    write_collection_source(
        collection_root,
        "tree_ring",
        mismatched_rows,
        prompts,
        protocol,
    )

    with pytest.raises(ValueError, match="method_faithful_observation_baseline_mismatch"):
        load_method_faithful_observation_collection(collection_root, protocol=protocol)

    duplicate_rows = formal_observation_rows("tree_ring", prompts, protocol)
    duplicate_rows[1]["event_id"] = duplicate_rows[0]["event_id"]
    write_collection_source(
        collection_root,
        "tree_ring",
        duplicate_rows,
        prompts,
        protocol,
    )

    with pytest.raises(ValueError, match="method_faithful_observation_event_id_duplicate"):
        load_method_faithful_observation_collection(collection_root, protocol=protocol)


def test_collection_rejects_count_and_sha256_mismatch(tmp_path: Path) -> None:
    """manifest 的 observation 数量和 SHA-256 必须与实际文件完全一致。"""

    collection_root = tmp_path / "collection"
    prompts, protocol = complete_inputs(collection_root)
    tree_rows = formal_observation_rows("tree_ring", prompts, protocol)
    _, manifest_path = write_collection_source(
        collection_root,
        "tree_ring",
        tree_rows,
        prompts,
        protocol,
        manifest_overrides={"baseline_observation_count": len(tree_rows) + 1},
    )

    with pytest.raises(ValueError, match="method_faithful_adapter_manifest_count_mismatch"):
        load_method_faithful_observation_collection(collection_root, protocol=protocol)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["baseline_observation_count"] = len(tree_rows)
    manifest["baseline_observations_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="method_faithful_observations_sha256_mismatch"):
        load_method_faithful_observation_collection(collection_root, protocol=protocol)


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    (
        ("transfer_ready", False, "method_faithful_transfer_not_ready"),
        ("paper_run_name", "pilot_paper", "method_faithful_transfer_paper_run_mismatch"),
        ("target_fpr", 0.01, "method_faithful_transfer_target_fpr_mismatch"),
    ),
)
def test_collection_rejects_transfer_protocol_mismatch(
    tmp_path: Path,
    field: str,
    value: object,
    expected_error: str,
) -> None:
    """transfer ready、论文层级和 target FPR 均必须由当前协议约束。"""

    collection_root = tmp_path / "collection"
    prompts, protocol = complete_inputs(collection_root)
    write_collection_source(
        collection_root,
        "tree_ring",
        formal_observation_rows("tree_ring", prompts, protocol),
        prompts,
        protocol,
        manifest_overrides={field: value},
    )

    with pytest.raises(ValueError, match=expected_error):
        load_method_faithful_observation_collection(collection_root, protocol=protocol)


def test_collection_rejects_budget_and_bound_digest_mismatch(tmp_path: Path) -> None:
    """20/20/4.5 预算以及 Prompt、adapter、execution 摘要不得被篡改。"""

    collection_root = tmp_path / "collection"
    _, protocol = complete_inputs(collection_root)
    manifest_path = collection_root / Path(*transfer_manifest_relative_path("tree_ring").parts)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generation_protocol"]["num_inference_steps"] = 19
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="method_faithful_transfer_budget_mismatch"):
        load_method_faithful_observation_collection(collection_root, protocol=protocol)

    complete_inputs(collection_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prompt_plan_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="method_faithful_transfer_bound_digest_mismatch"):
        load_method_faithful_observation_collection(collection_root, protocol=protocol)


def test_collection_rejects_model_revision_drift(tmp_path: Path) -> None:
    """transfer、adapter 和 observation 任一层漂移模型 commit 都必须阻断。"""

    collection_root = tmp_path / "collection"
    prompts, protocol = complete_inputs(collection_root)
    manifest_path = collection_root / Path(*transfer_manifest_relative_path("tree_ring").parts)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_revision"] = "a" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="method_faithful_transfer_model_revision_mismatch"):
        load_method_faithful_observation_collection(collection_root, protocol=protocol)

    complete_inputs(collection_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adapter_path = collection_root / Path(manifest["adapter_manifest_path"])
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    adapter["model_revision"] = "a" * 40
    adapter_path.write_text(json.dumps(adapter), encoding="utf-8")
    manifest["adapter_manifest_sha256"] = file_sha256(adapter_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="method_faithful_adapter_manifest_model_revision_mismatch"):
        load_method_faithful_observation_collection(collection_root, protocol=protocol)

    rows = formal_observation_rows("tree_ring", prompts, protocol)
    rows[0]["generation_model_revision"] = "a" * 40
    write_collection_source(collection_root, "tree_ring", rows, prompts, protocol)

    with pytest.raises(ValueError, match="method_faithful_observation_model_revision_mismatch"):
        load_method_faithful_observation_collection(collection_root, protocol=protocol)

    with pytest.raises(ValueError, match="method_faithful_collection_model_source_mismatch"):
        MethodFaithfulCollectionProtocol(
            paper_run_name="probe_paper",
            prompt_set="probe_paper",
            prompt_count=2,
            prompt_protocol_digest="0" * 64,
            target_fpr=0.1,
            model_revision="a" * 40,
        )


def test_collection_rejects_noncanonical_prompt_digest(tmp_path: Path) -> None:
    """三个 baseline 即使各自文件摘要有效，也必须共享当前规范 Prompt 摘要。"""

    collection_root = tmp_path / "collection"
    prompts, protocol = complete_inputs(collection_root)
    alternate_prompts = [dict(row) for row in prompts]
    alternate_prompts[0]["prompt_digest"] = "f" * 64
    write_collection_source(
        collection_root,
        "tree_ring",
        formal_observation_rows("tree_ring", alternate_prompts, protocol),
        alternate_prompts,
        protocol,
    )

    with pytest.raises(ValueError, match="method_faithful_transfer_canonical_prompt_mismatch"):
        load_method_faithful_observation_collection(collection_root, protocol=protocol)


def test_collection_rejects_threshold_digest_and_formal_attack_mismatch(tmp_path: Path) -> None:
    """阈值 provenance 和17类正式攻击集合必须由 observation 实体重算。"""

    collection_root = tmp_path / "collection"
    _, protocol = complete_inputs(collection_root)
    manifest_path = collection_root / Path(*transfer_manifest_relative_path("tree_ring").parts)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["threshold_digest"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="method_faithful_transfer_threshold_digest_mismatch"):
        load_method_faithful_observation_collection(collection_root, protocol=protocol)

    complete_inputs(collection_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["formal_attack_names"] = manifest["formal_attack_names"][:-1]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="method_faithful_transfer_formal_attack_names_mismatch"):
        load_method_faithful_observation_collection(collection_root, protocol=protocol)


def test_collection_rejects_forged_attacked_positive_identity(tmp_path: Path) -> None:
    """attacked positive 必须直接绑定执行时使用的正式 AttackConfig."""

    collection_root = tmp_path / "collection"
    prompts, protocol = complete_inputs(collection_root)
    rows = formal_observation_rows("tree_ring", prompts, protocol)
    attacked_positive = next(
        row for row in rows if row["sample_role"] == "attacked_positive"
    )
    attacked_positive["attack_config_digest"] = "0" * 64
    write_collection_source(
        collection_root,
        "tree_ring",
        rows,
        prompts,
        protocol,
    )

    with pytest.raises(
        ValueError,
        match="method_faithful_observation_attack_identity_mismatch",
    ):
        load_method_faithful_observation_collection(
            collection_root,
            protocol=protocol,
        )
