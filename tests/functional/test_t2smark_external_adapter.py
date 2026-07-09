"""验证 T2SMark 外部 baseline 结果适配器。"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from external_baseline.primary.t2smark.adapter.run_slm_eval import build_t2smark_observations


@pytest.mark.quick
def test_t2smark_adapter_builds_positive_and_negative_observations() -> None:
    """T2SMark adapter 应把每个样本转为 clean negative 与 positive source 观测。"""

    image_pairs = [{"image_id": "img_001", "prompt_id": "prompt_001", "split": "test"}]
    results = {"0": {"robustness": {"norm1_no_w": 0.1, "norm1_w": 0.9, "acc_msg": 1.0, "acc_key": 1.0}}}

    observations, manifest = build_t2smark_observations(image_pairs=image_pairs, t2smark_results=results)

    assert len(observations) == 2
    assert {row["sample_role"] for row in observations} == {"clean_negative", "positive_source"}
    assert {row["baseline_id"] for row in observations} == {"t2smark"}
    assert manifest["adapter_status"] == "sd35_native_result_adapter_ready"
    assert manifest["observation_count"] == 2
    assert manifest["formal_result_claim"] is False


@pytest.mark.quick
def test_t2smark_adapter_maps_formal_attack_results_to_common_observations() -> None:
    """T2SMark adapter 应把官方运行中的正式攻击结果映射为共同攻击簇观测。"""

    image_pairs = [
        {
            "image_id": "img_001",
            "prompt_id": "prompt_001",
            "prompt_text": "a ceramic fox",
            "generated_image_path": "outputs/t2smark/images/00000.png",
            "generated_image_digest": "source_digest",
            "clean_image_path": "outputs/t2smark/quality_pairs/clean/00000.png",
            "clean_image_digest": "clean_digest",
            "watermarked_image_path": "outputs/t2smark/images/00000.png",
            "watermarked_image_digest": "source_digest",
            "strict_pair_quality_ready": True,
        }
    ]
    results = {
        "0": {
            "robustness": {"norm1_no_w": 0.1, "norm1_w": 0.9, "acc_msg": 1.0, "acc_key": 1.0},
            "formal_attacks": {
                "img2img_regeneration": {
                    "norm1_no_w": 0.2,
                    "norm1_w": 0.7,
                    "acc_msg": 0.8,
                    "acc_key": 0.9,
                    "attack_family": "regeneration_attack",
                    "attack_name": "img2img_regeneration",
                    "attack_condition": "img2img_regeneration",
                    "attacked_image_path": "outputs/t2smark/formal_attacks/00000_img2img_regeneration.png",
                    "attacked_image_digest": "attacked_digest",
                }
            },
        }
    }

    observations, manifest = build_t2smark_observations(image_pairs=image_pairs, t2smark_results=results)

    attacked_rows = [row for row in observations if row["attack_condition"] == "img2img_regeneration"]
    assert len(observations) == 4
    assert {row["sample_role"] for row in attacked_rows} == {"attacked_negative", "attacked_positive"}
    assert all(row["attack_family"] == "regeneration_attack" for row in attacked_rows)
    assert manifest["formal_attack_names"] == ["img2img_regeneration"]
    assert manifest["formal_attack_observation_count"] == 2
    assert manifest["strict_pair_quality_ready"] is True
    assert all(row["clean_image_digest"] == "clean_digest" for row in observations)


@pytest.mark.quick
def test_t2smark_adapter_cli_writes_observation_file(tmp_path: Path) -> None:
    """T2SMark adapter CLI 应写出 observation 和 manifest 文件。"""

    image_pairs_path = tmp_path / "image_pairs.json"
    results_path = tmp_path / "results.json"
    output_path = tmp_path / "baseline_observations.json"
    image_pairs_path.write_text(json.dumps([{"image_id": "img_001"}], ensure_ascii=False), encoding="utf-8")
    results_path.write_text(
        json.dumps({"0": {"robustness": {"norm1_no_w": 0.2, "norm1_w": 0.8}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "external_baseline/primary/t2smark/adapter/run_slm_eval.py",
            "--image-pairs",
            str(image_pairs_path),
            "--t2smark-results",
            str(results_path),
            "--out",
            str(output_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    rows = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "t2smark_slm_adapter_manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert manifest["baseline_observations_path"] == str(output_path)


@pytest.mark.quick
def test_t2smark_adapter_contract_only_cli_writes_empty_observation(tmp_path: Path) -> None:
    """T2SMark contract-only CLI 应写出空 observation 与诊断 manifest。"""

    output_path = tmp_path / "baseline_observations.json"
    completed = subprocess.run(
        [
            sys.executable,
            "external_baseline/primary/t2smark/adapter/run_slm_eval.py",
            "--contract-only",
            "--out",
            str(output_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output_path.read_text(encoding="utf-8")) == []
    manifest = json.loads((tmp_path / "t2smark_slm_adapter_manifest.json").read_text(encoding="utf-8"))
    assert manifest["adapter_status"] == "sd35_native_result_adapter_ready"
    assert manifest["formal_result_claim"] is False
