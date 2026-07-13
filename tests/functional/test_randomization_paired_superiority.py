"""验证注册9重复配对总体优势公式与伪重复反例."""

from __future__ import annotations

import math

import pytest

from experiments.protocol.formal_randomization import formal_randomization_repeat_ids
from main.core.digest import build_stable_digest
from paper_experiments.analysis import randomization_paired_superiority as analysis
from paper_experiments.analysis.paired_superiority import PRIMARY_BASELINE_IDS


pytestmark = pytest.mark.quick

PAPER_RUN_NAME = "probe_paper"
TARGET_FPR = 0.1
PROTOCOL_DIGEST = build_stable_digest({"protocol": "paired"})
ATTACK_REGISTRY = (
    {
        "attack_id": "attack_main",
        "attack_family": "standard_distortion",
        "attack_name": "attack",
        "resource_profile": "full_main",
        "attack_config_digest": build_stable_digest({"attack": "main"}),
    },
)


def _digest(label: str, *values: object) -> str:
    """为测试记录构造稳定且互不混淆的 SHA-256 身份."""

    return build_stable_digest({"label": label, "values": list(values)})


def _paired_outcomes(
    *,
    proposed_positive_prompt_count: int = 17,
) -> list[dict[str, object]]:
    """构造34个 Prompt、9重复、4 baseline 和1攻击的完整集合."""

    rows: list[dict[str, object]] = []
    for prompt_index in range(34):
        prompt_id = f"prompt_{prompt_index:03d}"
        proposed_decision = prompt_index < proposed_positive_prompt_count
        for repeat_index, repeat_id in enumerate(formal_randomization_repeat_ids()):
            randomization = {
                "randomization_repeat_id": repeat_id,
                "generation_seed_index": repeat_index // 3,
                "generation_seed_offset": repeat_index // 3,
                "generation_seed_random": 1703 + prompt_index + repeat_index // 3,
                "watermark_key_index": repeat_index % 3,
                "watermark_key_seed_random": 9000 + repeat_index % 3,
                "watermark_key_material_digest_random": _digest(
                    "key", repeat_index % 3
                ),
                "formal_randomization_protocol_digest": _digest(
                    "randomization_protocol"
                ),
                "formal_randomization_identity_digest_random": _digest(
                    "randomization_identity", repeat_id, prompt_id
                ),
                "base_latent_content_digest_random": _digest(
                    "latent_content", repeat_id, prompt_id
                ),
                "base_latent_identity_digest_random": _digest(
                    "latent_identity", repeat_id, prompt_id
                ),
            }
            for baseline_id in PRIMARY_BASELINE_IDS:
                payload = {
                    "baseline_id": baseline_id,
                    "prompt_id": prompt_id,
                    **randomization,
                    "attack_id": ATTACK_REGISTRY[0]["attack_id"],
                    "attack_family": ATTACK_REGISTRY[0]["attack_family"],
                    "attack_name": ATTACK_REGISTRY[0]["attack_name"],
                    "resource_profile": ATTACK_REGISTRY[0]["resource_profile"],
                    "attack_config_digest": ATTACK_REGISTRY[0][
                        "attack_config_digest"
                    ],
                    "attack_seed_random": 8000 + prompt_index + repeat_index,
                    "formal_attack_seed_protocol_digest": _digest(
                        "attack_seed_protocol"
                    ),
                    "proposed_method_threshold_digest": _digest(
                        "threshold", repeat_id, "slm_wm"
                    ),
                    "baseline_method_threshold_digest": _digest(
                        "threshold", repeat_id, baseline_id
                    ),
                    "proposed_decision": proposed_decision,
                    "baseline_decision": False,
                    "paired_difference": int(proposed_decision),
                    "proposed_detector_digest": _digest(
                        "detector", repeat_id, prompt_id
                    ),
                    "proposed_attacked_image_digest": _digest(
                        "proposed_image", repeat_id, prompt_id
                    ),
                    "baseline_attacked_image_digest": _digest(
                        "baseline_image", repeat_id, prompt_id, baseline_id
                    ),
                }
                payload["paired_outcome_digest"] = build_stable_digest(payload)
                rows.append(payload)
    return rows


def _statistics(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """以较小 bootstrap 次数执行不对外公开的公式测试入口."""

    return analysis._build_randomization_aggregate_paired_superiority_statistics(
        rows,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        protocol_digest=PROTOCOL_DIGEST,
        attack_registry_rows=ATTACK_REGISTRY,
        confidence_level=0.95,
        bootstrap_resample_count=256,
    )


def _replace_row(
    rows: list[dict[str, object]],
    row_index: int,
    **updates: object,
) -> None:
    """修改一条反例记录并同步其逐行摘要."""

    payload = {
        key: value
        for key, value in rows[row_index].items()
        if key != "paired_outcome_digest"
    }
    payload.update(updates)
    payload["paired_outcome_digest"] = build_stable_digest(payload)
    rows[row_index] = payload


def test_registered_repeat_formula_does_not_treat_nine_repeats_as_independent() -> None:
    """用34个 Prompt 的临界反例阻止9倍伪重复制造显著性."""

    rows, summary = _statistics(_paired_outcomes())

    expected_p_value = math.exp(-34 * 0.5 * 0.5 / 2.0)
    incorrect_p_value = math.exp(-(34 * 9) * 0.5 * 0.5 / 2.0)
    assert len(rows) == 4
    assert set(rows[0]) == set(analysis.RANDOMIZATION_PAIRED_SUPERIORITY_FIELDNAMES)
    for row in rows:
        assert row["paired_prompt_count"] == 34
        assert row["randomization_repeat_count"] == 9
        assert row["paired_observation_count"] == 34 * 9
        assert row["statistical_unit"] == "prompt_cluster"
        assert row["mean_paired_true_positive_rate_difference"] == 0.5
        assert row["one_sided_bounded_hoeffding_mean_p_value"] == pytest.approx(
            expected_p_value
        )
        assert row["holm_adjusted_p_value"] == pytest.approx(
            4.0 * expected_p_value
        )
        assert row["holm_adjusted_p_value"] > 0.05
        assert row["paired_superiority_ready"] is False
        assert incorrect_p_value < 1e-16
    assert summary["paired_test_prompt_count"] == 34
    assert summary["paired_observation_count"] == 4 * 34 * 9
    assert summary["randomization_paired_statistics_ready"] is True
    assert summary["conclusion_decision"] == "measured_not_supported"
    assert summary["supports_paper_claim"] is False


def test_all_sample_advantage_cannot_publish_claim_without_quality_matching() -> None:
    """全样本优势通过时仍须等待独立质量匹配比较后才能支持主张."""

    rows, summary = _statistics(
        _paired_outcomes(proposed_positive_prompt_count=34)
    )

    assert all(row["paired_superiority_ready"] is True for row in rows)
    assert all(row["supports_paper_claim"] is False for row in rows)
    assert summary["overall_paired_superiority_ready"] is True
    assert summary["conclusion_decision"] == "all_sample_superiority_ready"
    assert summary["supports_paper_claim"] is False


def test_registered_repeat_statistics_are_input_order_invariant() -> None:
    """相同 outcome 集合的输入顺序不得改变摘要、seed 或统计结果."""

    outcomes = _paired_outcomes()
    forward_rows, forward_summary = _statistics(outcomes)
    reverse_rows, reverse_summary = _statistics(list(reversed(outcomes)))

    assert reverse_rows == forward_rows
    assert reverse_summary == forward_summary


def test_registered_repeat_statistics_reject_missing_repeat_cell() -> None:
    """缺少一个 repeat-Prompt-attack 单元时不得给予不等权重."""

    outcomes = _paired_outcomes()
    outcomes.pop()

    with pytest.raises(
        analysis.RandomizationPairedSuperiorityError,
        match="test Prompt 集合|完整攻击 registry",
    ):
        _statistics(outcomes)


def test_registered_repeat_statistics_allow_repeat_specific_thresholds() -> None:
    """不同 repeat 可独立校准阈值, 但同一 method-repeat 必须唯一."""

    rows, summary = _statistics(_paired_outcomes())

    assert rows
    threshold_map = summary["method_repeat_threshold_digest_map"]
    assert len(
        {threshold_map[repeat_id]["slm_wm"] for repeat_id in threshold_map}
    ) == 9


def test_registered_repeat_statistics_reject_within_repeat_threshold_drift() -> None:
    """同一 repeat 内混入第二套 baseline 阈值必须失败."""

    outcomes = _paired_outcomes()
    _replace_row(
        outcomes,
        1,
        baseline_method_threshold_digest=_digest("forged_threshold"),
    )

    with pytest.raises(
        analysis.RandomizationPairedSuperiorityError,
        match="多套 fixed-FPR 阈值",
    ):
        _statistics(outcomes)


def test_registered_repeat_statistics_reject_cross_baseline_main_drift() -> None:
    """四个比较必须引用同一主方法判定、检测器和攻击图像."""

    outcomes = _paired_outcomes()
    _replace_row(
        outcomes,
        1,
        proposed_detector_digest=_digest("forged_detector"),
    )

    with pytest.raises(
        analysis.RandomizationPairedSuperiorityError,
        match="同一主方法判定",
    ):
        _statistics(outcomes)


def test_public_statistics_lock_one_hundred_thousand_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """公开正式算子不得暴露或静默降低 bootstrap 次数."""

    observed_counts: list[int] = []

    def fake_interval(
        prompt_values,
        *,
        confidence_level: float,
        resample_count: int,
        seed: int,
    ) -> tuple[float, float]:
        del prompt_values, confidence_level, seed
        observed_counts.append(resample_count)
        return 0.1, 0.9

    monkeypatch.setattr(analysis, "_cluster_bootstrap_interval", fake_interval)
    analysis.build_randomization_aggregate_paired_superiority_statistics(
        _paired_outcomes(),
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        protocol_digest=PROTOCOL_DIGEST,
        attack_registry_rows=ATTACK_REGISTRY,
    )

    assert observed_counts == [100_000] * len(PRIMARY_BASELINE_IDS)
