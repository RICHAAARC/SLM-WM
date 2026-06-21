"""验证外部 baseline GPU smoke helper 的冷启动兼容能力。"""

from __future__ import annotations

from pathlib import Path

import pytest

from paper_workflow.colab_utils.external_baseline_gpu_smoke import (
    DEFAULT_T2SMARK_INVERSION_ENTRY,
    DEFAULT_T2SMARK_SOURCE_ENTRY,
    T2SMARK_INVERSION_COMPAT_MARKER,
    ExternalBaselineGpuSmokeConfig,
    output_paths,
    patch_t2smark_inversion_compatibility,
    run_t2smark_official_if_needed,
)


@pytest.mark.quick
def test_t2smark_inversion_import_patch_is_idempotent(tmp_path: Path) -> None:
    """T2SMark 官方 inversion 入口缺少 typing 导入时应被 helper 自动补齐。"""

    inversion_path = tmp_path / DEFAULT_T2SMARK_INVERSION_ENTRY
    inversion_path.parent.mkdir(parents=True)
    inversion_path.write_text(
        "from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import *\n\n"
        "class InversionDiffusion3Pipeline(StableDiffusion3Pipeline):\n"
        "    def sample(self, prompt: Union[str, List[str]] = None) -> Optional[str]:\n"
        "        return prompt\n",
        encoding="utf-8",
    )
    paths = {"output_dir": tmp_path / "outputs" / "external_baseline_gpu_smoke"}

    first_report = patch_t2smark_inversion_compatibility(tmp_path, paths)
    second_report = patch_t2smark_inversion_compatibility(tmp_path, paths)
    patched_text = inversion_path.read_text(encoding="utf-8")

    assert first_report["source_patch_applied"] is True
    assert second_report["source_patch_applied"] is False
    assert patched_text.count(T2SMARK_INVERSION_COMPAT_MARKER) == 1
    assert "from typing import Any, Callable, Dict, List, Optional, Union" in patched_text
    assert "from diffusers.image_processor import PipelineImageInput" in patched_text
    compile(patched_text, str(inversion_path), "exec")


@pytest.mark.quick
def test_t2smark_result_reuse_does_not_require_source_cache(tmp_path: Path) -> None:
    """已有官方结果可复用时, helper 不应要求重新下载或修补 T2SMark 源码。"""

    config = ExternalBaselineGpuSmokeConfig(require_cuda=False, reuse_existing=True, force_generate=False)
    paths = output_paths(tmp_path, config)
    paths["official_results"].parent.mkdir(parents=True)
    paths["official_results"].write_text('{"0":{"robustness":{"norm1_no_w":0.1,"norm1_w":0.9}}}\n', encoding="utf-8")

    report = run_t2smark_official_if_needed(tmp_path, config, paths)

    assert report["official_result_reused"] is True
    assert report["official_return_code"] == 0
    assert report["source_report"]["source_prepare_skipped"] is True
    assert not (tmp_path / DEFAULT_T2SMARK_SOURCE_ENTRY).exists()
