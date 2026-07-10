"""验证正式 baseline 只使用负样本校准、实测质量和真实 SD3.5 模式。"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
from PIL import Image

from external_baseline.primary.sd35_method_faithful_common import (
    derive_threshold,
    measured_image_ssim,
)
from external_baseline.primary.t2smark.adapter.run_slm_eval import _auto_threshold
from scripts.build_external_baseline_command_plan import build_parser, build_plan


@pytest.mark.quick
def test_method_faithful_threshold_uses_calibration_negatives_only() -> None:
    """改变阳性分数不得改变 fixed-FPR 阈值。"""

    negatives = [
        {"split": "calibration", "sample_role": "clean_negative", "score": value}
        for value in (0.1, 0.2, 0.3, 0.4, 0.5)
    ]
    first, source = derive_threshold(
        (*negatives, {"split": "calibration", "sample_role": "positive_source", "score": 0.6}),
        None,
        0.1,
    )
    second, _ = derive_threshold(
        (*negatives, {"split": "calibration", "sample_role": "positive_source", "score": 100.0}),
        None,
        0.1,
    )

    assert first == second
    assert source == "calibration_clean_negative_conformal"
    assert sum(row["score"] >= first for row in negatives) == 0


@pytest.mark.quick
def test_t2smark_threshold_uses_only_calibration_clean_scores() -> None:
    """T2SMark 适配阈值必须遵循同一负样本校准协议。"""

    pairs = [
        {"split": "calibration"},
        {"split": "calibration"},
        {"split": "test"},
    ]
    results = {
        0: {"robustness": {"norm1_no_w": 0.1, "norm1_w": 0.9}},
        1: {"robustness": {"norm1_no_w": 0.2, "norm1_w": -100.0}},
        2: {"robustness": {"norm1_no_w": 999.0, "norm1_w": 999.0}},
    }

    threshold, source = _auto_threshold(results, pairs, 0.1)

    assert threshold > 0.2
    assert source == "calibration_clean_negative_conformal"


@pytest.mark.quick
def test_measured_ssim_reads_image_content() -> None:
    """质量分数必须随实际像素变化, 不能固定为 1。"""

    white = Image.new("RGB", (32, 32), (255, 255, 255))
    black = Image.new("RGB", (32, 32), (0, 0, 0))

    assert measured_image_ssim(white, white) == pytest.approx(1.0, abs=1e-6)
    assert measured_image_ssim(white, black) < 0.1


@pytest.mark.quick
def test_external_baseline_plan_requires_target_fpr_and_real_mode(tmp_path: Path) -> None:
    """命令计划必须显式传递论文级 FPR 并选择唯一真实模式。"""

    adapter = tmp_path / "external_baseline/primary/tree_ring/adapter/run_slm_eval.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("print('adapter')\n", encoding="utf-8")
    prompt_plan = tmp_path / "prompt_plan.json"
    prompt_plan.write_text('[{"prompt_text":"a fox","split":"calibration"}]\n', encoding="utf-8")
    base_args = [
        "--root", str(tmp_path),
        "--methods", "tree_ring",
        "--output-root", str(tmp_path / "outputs/baselines"),
        "--prompt-plan", str(prompt_plan),
    ]
    parser = build_parser()
    with pytest.raises(ValueError, match="target-fpr"):
        build_plan(parser.parse_args(base_args))

    plan = build_plan(parser.parse_args([*base_args, "--target-fpr", "0.1"]))
    command = plan[0]["command"]
    assert command[command.index("--adapter-mode") + 1] == "method_faithful_sd35"
    assert command[command.index("--target-fpr") + 1] == "0.1"
