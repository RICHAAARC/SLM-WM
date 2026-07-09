"""验证 T2SMark 严格 pair-level 质量证据构造。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from paper_experiments.baselines.t2smark_pair_quality import write_t2smark_strict_pair_quality_outputs


@pytest.mark.quick
def test_t2smark_strict_pair_quality_outputs_measure_pixel_metrics(tmp_path: Path) -> None:
    """严格 clean/watermarked 图像对存在时应输出可审计的 pair-level 质量指标。"""

    output_dir = tmp_path / "outputs" / "external_baseline_method_faithful"
    clean_path = output_dir / "quality_pairs" / "clean" / "00000.png"
    watermarked_path = output_dir / "t2smark_official" / "run" / "images" / "00000.png"
    clean_path.parent.mkdir(parents=True)
    watermarked_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(clean_path)
    Image.new("RGB", (8, 8), color=(129, 128, 128)).save(watermarked_path)
    image_pairs_path = output_dir / "t2smark_image_pairs.json"
    image_pairs_path.write_text(
        json.dumps(
            [
                {
                    "image_id": "t2smark_00000",
                    "prompt_id": "prompt_00000",
                    "prompt_index": 0,
                    "prompt_set": "pilot_paper",
                    "split": "test",
                    "clean_image_path": clean_path.relative_to(tmp_path).as_posix(),
                    "clean_image_digest": "clean_digest",
                    "watermarked_image_path": watermarked_path.relative_to(tmp_path).as_posix(),
                    "watermarked_image_digest": "watermarked_digest",
                    "strict_pair_quality_ready": True,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = write_t2smark_strict_pair_quality_outputs(
        root_path=tmp_path,
        image_pairs_path=image_pairs_path,
        metrics_path=output_dir / "t2smark_strict_pair_quality_metrics.csv",
        summary_path=output_dir / "t2smark_strict_pair_quality_summary.json",
    )

    metrics_text = (output_dir / "t2smark_strict_pair_quality_metrics.csv").read_text(encoding="utf-8")
    assert summary["strict_pair_quality_ready"] is True
    assert summary["measured_strict_pair_quality_count"] == 1
    assert summary["lpips_status_values"] == ["disabled"]
    assert summary["clip_score_status_values"] == ["disabled"]
    assert "strict_clean_watermarked_pair" in metrics_text
    assert "measured" in metrics_text
