"""验证独立语义评估器的冻结身份、真实图像边界和特征契约."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.artifacts.independent_semantic_quality_outputs import (
    extract_independent_semantic_feature_rows,
    validate_independent_semantic_feature_rows,
)
from experiments.protocol.independent_semantic_quality import (
    INDEPENDENT_SEMANTIC_FEATURE_BACKEND,
    INDEPENDENT_SEMANTIC_FEATURE_DIMENSION,
    load_independent_semantic_quality_evaluator,
)
from main.core.digest import build_stable_digest
from paper_experiments.analysis.paper_profile_protocol_isomorphism import (
    build_paper_profile_protocol_records,
)
from paper_experiments.analysis.paper_quality_decisions import (
    load_paper_quality_claim_protocol,
)
from tests.helpers.scientific_unit_provenance import (
    build_test_scientific_unit_provenance,
)


pytestmark = pytest.mark.quick


def test_independent_evaluator_freezes_model_preprocessing_and_dependency_lock() -> None:
    """DINOv2 身份必须与方法 CLIP 不同源并绑定完整依赖锁."""

    protocol = load_independent_semantic_quality_evaluator()

    assert protocol["model_contract"] == {
        "model_class": "Dinov2Model",
        "model_family": "dinov2_self_supervised_vision_transformer",
        "model_id": "facebook/dinov2-base",
        "model_revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
    }
    assert protocol["feature_contract"]["feature_layer"] == (
        "last_hidden_state_cls_token"
    )
    assert protocol["feature_contract"]["normalization"] == "l2"
    assert protocol["preprocessing_contract"]["resize_shortest_edge"] == 256
    assert protocol["preprocessing_contract"]["center_crop_height"] == 224
    assert protocol["dependency_contract"]["dependency_profile_id"] == (
        "sd35_method_runtime_gpu"
    )
    assert len(protocol["dependency_profile_digest"]) == 64
    assert len(protocol["complete_hash_lock_digest"]) == 64
    assert protocol["independence_contract"]["quality_evaluator_family"] != (
        protocol["independence_contract"]["method_condition_encoder_family"]
    )


def test_independent_evaluator_rejects_missing_persisted_image_before_model_load(
    tmp_path: Path,
) -> None:
    """正式评估器不得为缺失图像构造随机、预制或替代特征."""

    record = SimpleNamespace(
        dataset_quality_record_id="dataset_quality_record_missing",
        source_image_path="outputs/missing_source.png",
        source_image_digest="a" * 64,
        comparison_image_path="outputs/missing_comparison.png",
        comparison_image_digest="b" * 64,
    )

    with pytest.raises(FileNotFoundError, match="真实持久化图像"):
        extract_independent_semantic_feature_rows(
            records=(record,),
            root_path=tmp_path,
            image_search_roots=(),
            output_path=tmp_path / "outputs/features.jsonl",
        )


def test_independent_feature_validator_requires_exact_model_and_l2_vectors() -> None:
    """原始记录必须绑定精确 evaluator、提交、锁摘要及 L2 特征."""

    protocol = load_independent_semantic_quality_evaluator()
    record_id = "dataset_quality_record_0000000000000000"
    vector = [1.0] + [0.0] * (INDEPENDENT_SEMANTIC_FEATURE_DIMENSION - 1)
    provenance = build_test_scientific_unit_provenance(
        "independent_semantic_validation_batch",
        build_stable_digest({"batch": 1}),
        dependency_profile_digest=protocol["dependency_profile_digest"],
        complete_hash_lock_digest=protocol["complete_hash_lock_digest"],
    )
    rows = [
        {
            "dataset_quality_record_id": record_id,
            "dataset_quality_image_role": role,
            "image_path": f"outputs/{role}.png",
            "image_digest": build_stable_digest({"role": role}),
            "feature_backend": INDEPENDENT_SEMANTIC_FEATURE_BACKEND,
            "feature_extractor_id": (
                "facebook/dinov2-base@"
                "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
            ),
            "feature_dimension": INDEPENDENT_SEMANTIC_FEATURE_DIMENSION,
            "feature_layer": "last_hidden_state_cls_token",
            "feature_normalization": "l2",
            "feature_vector": vector,
            "feature_vector_digest": build_stable_digest(vector),
            "independent_semantic_quality_protocol_digest": protocol[
                "independent_semantic_quality_protocol_digest"
            ],
            "scientific_unit_provenance": provenance,
            "supports_paper_claim": False,
        }
        for role in ("source", "comparison")
    ]

    assert len(
        validate_independent_semantic_feature_rows(
            rows,
            expected_record_ids=(record_id,),
            expected_code_version="b" * 40,
        )
    ) == 2
    rows[0]["feature_extractor_id"] = "openai/clip-vit-base-patch32@invalid"
    with pytest.raises(ValueError, match="协议身份无效"):
        validate_independent_semantic_feature_rows(
            rows,
            expected_record_ids=(record_id,),
            expected_code_version="b" * 40,
        )


def test_three_paper_profiles_share_one_independent_semantic_decision_rule() -> None:
    """三档 profile 只能改变规模, 不得切换语义评估器或决策指标."""

    records = build_paper_profile_protocol_records(Path.cwd())
    protocol_digests = {
        profile["protocol_contract"]["metric_semantics"]
        ["paper_quality_claim_protocol"]
        ["independent_semantic_quality_evaluator"]
        ["independent_semantic_quality_protocol_digest"]
        for profile in records.values()
    }
    quality_protocol = load_paper_quality_claim_protocol()

    assert len(protocol_digests) == 1
    assert quality_protocol["semantic_alignment_noninferiority"][
        "metric_name"
    ] == "independent_semantic_cosine"
