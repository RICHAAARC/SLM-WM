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
    SemanticWatermarkRuntimeConfig,
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


def test_formal_screen_runs_writer_loader_and_32_wrong_key_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
        Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image_path)
        detection_path = run_root / "image_only_detection_records.jsonl"
        rows = [
            ("clean_negative", 0.0),
            ("positive_source", 0.1),
            ("wrong_key_negative", 0.0),
        ]
        detection_path.write_text(
            "".join(
                json.dumps(
                    {
                        "sample_role": role,
                        "content_score": score,
                        "metadata": {"method_role": "hf_tail_only_content"},
                    },
                    sort_keys=True,
                )
                + "\n"
                for role, score in rows
            ),
            encoding="utf-8",
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
            metadata={
                "method_runtime": "formal_terminal_hf_content_dual_chain",
                "formal_attribution_carrier": "terminal_pre_vae_hf_tail",
                "formal_attribution_strength_multiplier": 8.0,
                "paired_quality": {"ssim": 0.996, "psnr": 45.0},
            },
        )

    monkeypatch.setattr(runtime, "load_completed_semantic_watermark_runtime_result", load)
    monkeypatch.setattr(runtime, "write_semantic_watermark_runtime_outputs", write)
    monkeypatch.setattr(
        runtime,
        "_encode_image_latent",
        lambda pipeline, image: torch.zeros((1, 4, 8, 8)),
    )
    monkeypatch.setattr(
        runtime,
        "_score_key_roster",
        lambda *args, **kwargs: {
            "latent_content_sha256": "a" * 64,
            "key_score_records": [
                {"key_role": "registered", "blind_content_score": 0.1},
                *(
                    {"key_role": "wrong", "blind_content_score": 0.0}
                    for _ in range(32)
                ),
            ],
            "rank_record": {"registered_rank": 1},
        },
    )
    kwargs = {
        "references": object(),
        "verified_formal_execution_lock": {"lock": "synthetic"},
        "verified_execution_environment_identity": {"identity": "synthetic"},
        "repository_root": tmp_path,
        "output_dir": "outputs/formal_terminal_hf",
    }

    summary = runtime.run_formal_terminal_hf_screen(_configs(), **kwargs)

    assert summary["decision"] == "pass"
    assert summary["method_screening_decision"] == "pass"
    assert summary["prompt_count"] == 4
    assert summary["diffusion_chain_count"] == 28
    assert summary["key_score_count"] == 132
    assert summary["registered_rank_one_count"] == 4
    assert writer_calls == list(CONTENT_SURVIVAL_PROMPT_IDS)
    assert len(list((tmp_path / "outputs").rglob("cell_manifest.json"))) == 4

    resumed = runtime.run_formal_terminal_hf_screen(_configs(), **kwargs)
    assert resumed["registered_rank_one_count"] == 4
    assert writer_calls == list(CONTENT_SURVIVAL_PROMPT_IDS)
