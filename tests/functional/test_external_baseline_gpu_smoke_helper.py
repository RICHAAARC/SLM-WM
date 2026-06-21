"""验证外部 baseline GPU smoke helper 的冷启动兼容能力。"""

from __future__ import annotations

from pathlib import Path

import pytest

from paper_workflow.colab_utils.external_baseline_gpu_smoke import (
    DEFAULT_T2SMARK_INVERSION_ENTRY,
    DEFAULT_T2SMARK_SOURCE_ENTRY,
    PRIMARY_BASELINE_METHODS,
    T2SMARK_INVERSION_COMPAT_MARKER,
    ExternalBaselineGpuSmokeConfig,
    build_and_run_primary_baseline_adapters,
    build_t2smark_image_pairs,
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


@pytest.mark.quick
def test_t2smark_image_pairs_refreshes_stale_image_provenance(tmp_path: Path) -> None:
    """已有 image_pairs 缺少图像路径与 digest 时, helper 应按当前图像目录刷新。"""

    config = ExternalBaselineGpuSmokeConfig(require_cuda=False, reuse_existing=True, force_generate=False)
    paths = output_paths(tmp_path, config)
    paths["official_images"].mkdir(parents=True)
    image_path = paths["official_images"] / "00000.png"
    image_path.write_bytes(b"fake_png_bytes_for_t2smark_smoke")
    paths["image_pairs"].parent.mkdir(parents=True, exist_ok=True)
    paths["image_pairs"].write_text(
        '[{"image_id":"t2smark_00000","generated_image_path":"","generated_image_digest":""}]\n',
        encoding="utf-8",
    )

    rows = build_t2smark_image_pairs(tmp_path, config, paths)

    assert rows[0]["generated_image_path"] == "outputs/external_baseline_gpu_smoke/t2smark_official/t2smark_sd35_medium_gpu_smoke/images/00000.png"
    assert rows[0]["generated_image_digest"]
    assert '"generated_image_digest": ""' not in paths["image_pairs"].read_text(encoding="utf-8")


@pytest.mark.quick
def test_primary_baseline_adapter_plan_includes_four_methods(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GPU smoke helper 应把四个主表 baseline 并入同一个命令计划。"""

    config = ExternalBaselineGpuSmokeConfig(
        output_dir="outputs/external_baseline_gpu_smoke",
        require_cuda=True,
        primary_baseline_max_samples=1,
    )
    paths = output_paths(tmp_path, config)
    paths["output_dir"].mkdir(parents=True)
    captured_commands: list[list[str]] = []

    def fake_run_command(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict[str, object]:
        captured_commands.append(command)
        command_text = " ".join(command)
        if "run_external_baseline_command_plan.py" in command_text:
            paths["execution_manifest"].parent.mkdir(parents=True, exist_ok=True)
            paths["execution_manifest"].write_text('{"observation_count":8}\n', encoding="utf-8")
            paths["baseline_observations"].write_text("[]\n", encoding="utf-8")
            paths["command_results"].write_text(
                "["
                + ",".join(
                    f'{{"baseline_id":"{baseline_id}","return_code":0,"observation_count":2}}'
                    for baseline_id in PRIMARY_BASELINE_METHODS
                )
                + "]\n",
                encoding="utf-8",
            )
        return {"command": command, "return_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("paper_workflow.colab_utils.external_baseline_gpu_smoke.run_command", fake_run_command)

    report = build_and_run_primary_baseline_adapters(tmp_path, config, paths)

    build_command = captured_commands[0]
    assert "--methods" in build_command
    assert build_command[build_command.index("--methods") + 1] == ",".join(PRIMARY_BASELINE_METHODS)
    assert "--prompt-plan" in build_command
    assert "--require-cuda" in build_command
    assert paths["primary_prompt_plan"].is_file()
    assert report["primary_baseline_adapter_ready"] is True
    assert report["primary_baseline_adapter_count"] == 4
    assert report["primary_baseline_observation_count"] == 8
