"""CPU integration contracts for the four-prompt formal terminal-HF screen."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from experiments.protocol.content_survival_observation import (
    CONTENT_SURVIVAL_PROMPT_IDS,
)
from experiments.runners import formal_terminal_hf_runtime as runtime
from experiments.runners.semantic_watermark_runtime import (
    FinalImageEvidenceGateFailure,
    SemanticWatermarkRuntimeConfig,
    SemanticWatermarkRuntimeResult,
)


pytestmark = pytest.mark.quick


def _configs() -> dict[str, SemanticWatermarkRuntimeConfig]:
    return {
        prompt_id: SemanticWatermarkRuntimeConfig(
            prompt=f"formal terminal HF prompt {prompt_id}",
            prompt_id=prompt_id,
            key_material="registered-formal-terminal-hf-key",
            standard_attack_profiles=(),
            diffusion_attacks_enabled=False,
        )
        for prompt_id in CONTENT_SURVIVAL_PROMPT_IDS
    }


@pytest.mark.parametrize(
    ("prompt_count", "scientific_failure_reason", "carrier_screen_pass"),
    [
        (1, None, True),
        (4, None, True),
        (1, "final_image_attention_observability", True),
        (4, "final_image_attention_observability", True),
        (1, None, False),
        (1, "late_qk_geometry_not_ready", True),
    ],
)
def test_formal_screen_runs_writer_loader_and_32_wrong_key_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_count: int,
    scientific_failure_reason: str | None,
    carrier_screen_pass: bool,
) -> None:
    (tmp_path / "outputs").mkdir()
    completed: dict[str, SimpleNamespace] = {}
    writer_calls: list[str] = []
    wrong_keys = tuple(
        {
            "wrong_key_index": index,
            "wrong_key_material": f"wrong-{index}",
            "wrong_key_material_digest_random": f"{index + 1:064x}",
        }
        for index in range(32)
    )
    monkeypatch.setattr(runtime, "validate_formal_execution_lock_record", lambda value: value)
    monkeypatch.setattr(runtime, "load_content_survival_observation_protocol", lambda root: object())
    monkeypatch.setattr(
        runtime,
        "build_content_survival_observation_roster",
        lambda key, protocol: {"wrong_keys": list(wrong_keys)},
    )
    monkeypatch.setattr(
        runtime,
        "load_semantic_watermark_runtime_context",
        lambda *args, **kwargs: SimpleNamespace(pipeline=object()),
    )

    def load(config: SemanticWatermarkRuntimeConfig, root: Path):
        return completed.get(config.prompt_id)

    def write(
        config: SemanticWatermarkRuntimeConfig,
        root: Path,
        **kwargs: object,
    ) -> None:
        writer_calls.append(config.prompt_id)
        run_id = f"formal_{config.prompt_id}"
        run_root = root / config.output_dir / run_id
        run_root.mkdir(parents=True)
        image_path = run_root / "watermarked.png"
        carrier_image_path = run_root / "carrier_only.png"
        Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image_path)
        Image.new("RGB", (8, 8), color=(9, 19, 29)).save(
            carrier_image_path
        )
        detection_path = run_root / "image_only_detection_records.jsonl"
        rows = [
            ("clean_negative", 0.0),
            ("positive_source", 0.1),
            ("wrong_key_negative", 0.0),
        ]
        detection_records = [
            {
                "sample_role": role,
                "content_score": score,
                "metadata": {"method_role": "hf_tail_only_content"},
            }
            for role, score in rows
        ]
        detection_path.write_text(
            "".join(
                json.dumps(record, sort_keys=True)
                + "\n"
                for record in detection_records
            ),
            encoding="utf-8",
        )
        metadata = {
            "method_runtime": "formal_terminal_hf_content_dual_chain",
            "formal_attribution_carrier": "terminal_pre_vae_hf_tail",
            "formal_attribution_strength_multiplier": 8.0,
            "paired_quality": {"ssim": 0.996, "psnr": 45.0},
            "final_image_preservation": {
                "final_image_preservation_gate_ready": True,
            },
            "carrier_only_final_image_preservation": {
                "carrier_only_counterfactual_three_way_preservation_gate_ready": True,
            },
            "final_image_attention_observability": {
                "final_image_attention_observability_gate_ready": (
                    scientific_failure_reason
                    != "final_image_attention_observability"
                ),
                "final_image_attention_blind_attribution_gain": -0.0002,
                "final_image_attention_carrier_paired_attribution_gain": -0.0001,
            },
            "carrier_only_image_path": carrier_image_path.relative_to(
                root
            ).as_posix(),
        }
        if scientific_failure_reason is not None:
            failed_runtime = SemanticWatermarkRuntimeResult(
                run_id=run_id,
                run_decision="fail",
                clean_image_path="",
                watermarked_image_path="",
                update_record_path="",
                detection_record_path="",
                manifest_path="",
                update_count=1,
                elapsed_seconds=1.0,
                metadata={
                    **metadata,
                    "final_image_evidence_gate_failures": [
                        scientific_failure_reason
                    ],
                },
            )
            raise FinalImageEvidenceGateFailure(
                failure_reasons=(scientific_failure_reason,),
                runtime_outputs=(
                    failed_runtime,
                    (),
                    (),
                    tuple(detection_records),
                    Image.new("RGB", (8, 8)),
                    Image.new("RGB", (8, 8), color=(10, 20, 30)),
                    Image.new("RGB", (8, 8), color=(9, 19, 29)),
                    {},
                ),
            )
        result_path = run_root / "runtime_result.json"
        result_path.write_text("{}\n", encoding="utf-8")
        manifest_path = run_root / "manifest.local.json"
        manifest_path.write_text("{}\n", encoding="utf-8")
        completed[config.prompt_id] = SimpleNamespace(
            prompt_id=config.prompt_id,
            run_id=run_id,
            detection_record_path=detection_path.relative_to(root).as_posix(),
            watermarked_image_path=image_path.relative_to(root).as_posix(),
            manifest_path=manifest_path.relative_to(root).as_posix(),
            metadata=metadata,
        )

    monkeypatch.setattr(runtime, "load_completed_semantic_watermark_runtime_result", load)
    monkeypatch.setattr(runtime, "write_semantic_watermark_runtime_outputs", write)
    encoded_image_colors: list[tuple[int, int, int]] = []

    def encode_image(_pipeline: object, image: Image.Image) -> torch.Tensor:
        encoded_image_colors.append(image.getpixel((0, 0)))
        return torch.zeros((1, 4, 8, 8))

    monkeypatch.setattr(runtime, "_encode_image_latent", encode_image)
    score_calls: list[tuple[object, ...]] = []

    def score_roster(*args: object, **kwargs: object) -> dict[str, object]:
        call_index = len(score_calls)
        score_calls.append(tuple(kwargs["wrong_keys"]))
        carrier_roster_call = call_index % 3 == 1
        wrong_score = (
            0.2
            if carrier_roster_call and not carrier_screen_pass
            else 0.0
        )
        wrong_count = len(tuple(kwargs["wrong_keys"]))
        return {
            "latent_content_sha256": "a" * 64,
            "key_score_records": [
                {"key_role": "registered", "blind_content_score": 0.1},
                *(
                    {
                        "key_role": "wrong",
                        "blind_content_score": wrong_score,
                    }
                    for _ in range(wrong_count)
                ),
            ],
            "rank_record": {
                "registered_rank": 2 if wrong_score > 0.1 else 1,
                "registered_minus_max_wrong_margin": 0.1 - wrong_score,
            },
        }

    monkeypatch.setattr(runtime, "_score_key_roster", score_roster)
    kwargs = {
        "references": object(),
        "verified_formal_execution_lock": {"lock": "synthetic"},
        "verified_execution_environment_identity": {"identity": "synthetic"},
        "repository_root": tmp_path,
        "output_dir": "outputs/formal_terminal_hf",
    }

    selected_prompt_ids = CONTENT_SURVIVAL_PROMPT_IDS[:prompt_count]
    configs = {
        prompt_id: config
        for prompt_id, config in _configs().items()
        if prompt_id in selected_prompt_ids
    }
    summary = runtime.run_formal_terminal_hf_screen(configs, **kwargs)

    assert summary["decision"] == "pass"
    assert summary["method_screening_decision"] == (
        "pass" if carrier_screen_pass else "fail"
    )
    assert summary["prompt_ids"] == list(selected_prompt_ids)
    assert summary["prompt_count"] == prompt_count
    assert summary["diffusion_chain_count"] == prompt_count * 7
    assert summary["key_score_count"] == prompt_count * 66
    assert summary["registered_rank_one_count"] == prompt_count
    assert summary["carrier_only_registered_rank_one_count"] == (
        prompt_count if carrier_screen_pass else 0
    )
    assert summary["carrier_only_fixed_wrong_key_pass_count"] == prompt_count
    assert summary["final_image_evidence_pass_count"] == (
        0 if scientific_failure_reason is not None else prompt_count
    )
    assert summary["scientific_gate_failure_count"] == (
        prompt_count if scientific_failure_reason is not None else 0
    )
    if scientific_failure_reason is not None:
        first = summary["prompt_results"][0]
        assert first["formal_runtime_result_path"] == ""
        assert first["scientific_gate_failure_reasons"] == [
            scientific_failure_reason
        ]
        assert first["registered_rank_one"] is True
        assert first["carrier_only_registered_rank_one"] is True
        assert first["carrier_only_fixed_wrong_key_margin"] == pytest.approx(0.1)
        assert first["hf_attribution_views"] == {
            "full_combined": "multi_key_scores",
            "carrier_only": "carrier_only_multi_key_scores",
            "qk_geometry_gain_source": "final_image_attention_observability",
        }
        assert first["formal_fixed_wrong_key_margin"] == pytest.approx(0.1)
        assert first["final_image_attention_observability"][
            "final_image_attention_blind_attribution_gain"
        ] == -0.0002
        assert issubclass(FinalImageEvidenceGateFailure, RuntimeError)
    assert writer_calls == list(selected_prompt_ids)
    assert [len(call) for call in score_calls] == [32, 32, 1] * prompt_count
    assert encoded_image_colors == [(10, 20, 30), (9, 19, 29)] * prompt_count
    assert len(list((tmp_path / "outputs").rglob("cell_manifest.json"))) == prompt_count

    resumed = runtime.run_formal_terminal_hf_screen(configs, **kwargs)
    assert resumed["registered_rank_one_count"] == prompt_count
    assert resumed["method_screening_decision"] == (
        "pass" if carrier_screen_pass else "fail"
    )
    assert resumed["scientific_gate_failure_count"] == (
        prompt_count if scientific_failure_reason is not None else 0
    )
    assert writer_calls == list(selected_prompt_ids)
