"""验证外部 baseline GPU smoke helper 的冷启动兼容能力。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from external_baseline.primary.sd35_method_faithful_common import (
    apply_image_attack,
    canonical_attack_family,
    canonical_attack_name,
    regeneration_formal_image_attack_names,
    standard_geometric_formal_image_attack_names,
    supported_formal_image_attack_names,
)
from paper_workflow.colab_utils.external_baseline_gpu_smoke import (
    DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES,
    DEFAULT_T2SMARK_INVERSION_ENTRY,
    DEFAULT_T2SMARK_SOURCE_ENTRY,
    T2SMARK_FORMAL_ATTACK_COMPAT_MARKER,
    PRIMARY_BASELINE_METHODS,
    T2SMARK_INVERSION_COMPAT_MARKER,
    ExternalBaselineGpuSmokeConfig,
    build_and_run_primary_baseline_adapters,
    build_t2smark_image_pairs,
    count_t2smark_result_items,
    output_paths,
    patch_t2smark_formal_attack_compatibility,
    patch_t2smark_inversion_compatibility,
    run_t2smark_official_if_needed,
    should_run_t2smark_official,
    write_primary_baseline_prompt_plan,
    write_t2smark_prompt_input,
)


@pytest.mark.quick
def test_formal_image_attack_taxonomy_matches_attack_matrix_names() -> None:
    """method-faithful adapter 应把图像攻击名称映射到攻击矩阵共同协议。"""

    image = Image.new("RGB", (16, 16), color=(128, 128, 128))
    expected_names = set(DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES.split(","))

    assert set(supported_formal_image_attack_names()) == expected_names
    assert canonical_attack_family("jpeg_compression") == "standard_distortion"
    assert canonical_attack_name("rotate") == "rotation"
    assert canonical_attack_family("crop_resize") == "geometric_transform"
    assert canonical_attack_family("ddim_inversion") == "regeneration_attack"
    assert canonical_attack_name("purification") == "diffusion_purification"
    assert set(regeneration_formal_image_attack_names()).issubset(expected_names)
    for attack_name in standard_geometric_formal_image_attack_names():
        attacked_image, transform_name = apply_image_attack(image, attack_family=attack_name, seed=17)
        assert attacked_image.mode == "RGB"
        assert transform_name


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
def test_t2smark_formal_attack_patch_adds_common_attack_outputs(tmp_path: Path) -> None:
    """T2SMark 官方入口应在冷启动时被补齐共同攻击簇输出参数与逻辑。"""

    source_path = tmp_path / DEFAULT_T2SMARK_SOURCE_ENTRY
    option_path = source_path.with_name("option.py")
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "import os\n"
        "import json\n"
        "import torch\n"
        "from option import args\n\n"
        "def decode(post_reversed_latents, master_key, key, fake_key, msg):\n"
        "    return {'norm1_no_w': 0.0, 'norm1_w': 1.0}\n\n"
        "pipe = InversionDiffusion3Pipeline.from_pretrained(args.model_key, torch_dtype=torch.float16).to(device)\n"
        "results = {}\n"
        "            results[prompt_id][\"robustness\"] = decode_result\n",
        encoding="utf-8",
    )
    option_path.write_text(
        "import argparse\n\n"
        "def parse_args():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument(\"--SDv35M\", action=\"store_true\", default=False)\n"
        "    return parser.parse_args()\n",
        encoding="utf-8",
    )
    paths = {"output_dir": tmp_path / "outputs" / "external_baseline_gpu_smoke"}

    first_report = patch_t2smark_formal_attack_compatibility(tmp_path, paths)
    second_report = patch_t2smark_formal_attack_compatibility(tmp_path, paths)
    patched_source = source_path.read_text(encoding="utf-8")
    patched_option = option_path.read_text(encoding="utf-8")

    assert first_report["formal_attack_patch_applied"] is True
    assert second_report["formal_attack_patch_applied"] is False
    assert T2SMARK_FORMAL_ATTACK_COMPAT_MARKER in patched_source
    assert "slm_attack_families" in patched_option
    assert "formal_attacks" in patched_source
    assert "apply_formal_image_attack" in patched_source
    assert "from PIL import Image" in patched_source
    assert "prepare_t2smark_decode_image" in patched_source
    assert "resize((512, 512), Image.Resampling.BICUBIC)" in patched_source


@pytest.mark.quick
def test_t2smark_result_reuse_does_not_require_source_cache(tmp_path: Path) -> None:
    """已有官方结果可复用时, helper 不应要求重新下载或修补 T2SMark 源码。"""

    config = ExternalBaselineGpuSmokeConfig(
        require_cuda=False,
        reuse_existing=True,
        force_generate=False,
        robust_test_num=1,
        t2smark_formal_attack_families="",
    )
    paths = output_paths(tmp_path, config)
    paths["official_results"].parent.mkdir(parents=True)
    paths["official_results"].write_text('{"0":{"robustness":{"norm1_no_w":0.1,"norm1_w":0.9}}}\n', encoding="utf-8")

    report = run_t2smark_official_if_needed(tmp_path, config, paths)

    assert report["official_result_reused"] is True
    assert report["official_return_code"] == 0
    assert report["source_report"]["source_prepare_skipped"] is True
    assert not (tmp_path / DEFAULT_T2SMARK_SOURCE_ENTRY).exists()


@pytest.mark.quick
def test_t2smark_result_reuse_requires_configured_sample_count(tmp_path: Path) -> None:
    """历史 T2SMark 结果样本数不足时, helper 应重新运行而不是复用旧包。"""

    config = ExternalBaselineGpuSmokeConfig(
        require_cuda=False,
        reuse_existing=True,
        robust_test_num=5,
        t2smark_formal_attack_families="",
    )
    results_path = tmp_path / "outputs" / "external_baseline_gpu_smoke" / "results.json"
    results_path.parent.mkdir(parents=True)
    results_path.write_text('{"0":{"robustness":{}},"metadata":{"note":"old smoke"}}\n', encoding="utf-8")

    should_run, reason = should_run_t2smark_official(config, results_path)

    assert count_t2smark_result_items(results_path) == 1
    assert should_run is True
    assert reason == "existing_results_sample_count_insufficient"


@pytest.mark.quick
def test_shared_prompt_inputs_default_to_pilot_paper_samples(tmp_path: Path) -> None:
    """T2SMark 与主表 adapter 应共享 pilot_paper 规模 prompt 计划。"""

    config = ExternalBaselineGpuSmokeConfig(require_cuda=False)
    paths = output_paths(tmp_path, config)

    t2smark_prompt_path = write_t2smark_prompt_input(tmp_path, paths, config)
    primary_prompt_path = write_primary_baseline_prompt_plan(tmp_path, paths, config)

    t2smark_payload = json.loads(t2smark_prompt_path.read_text(encoding="utf-8"))
    primary_rows = json.loads(primary_prompt_path.read_text(encoding="utf-8"))
    assert len(t2smark_payload["annotations"]) == 600
    assert len(primary_rows) == 600
    assert primary_rows[0]["prompt_text"] == t2smark_payload["annotations"][0]["caption"]
    assert primary_rows[0]["prompt_set"] == "pilot_paper"


@pytest.mark.quick
def test_t2smark_image_pairs_refreshes_stale_image_provenance(tmp_path: Path) -> None:
    """已有 image_pairs 缺少图像路径与 digest 时, helper 应按当前图像目录刷新。"""

    config = ExternalBaselineGpuSmokeConfig(
        require_cuda=False,
        reuse_existing=True,
        force_generate=False,
        robust_test_num=5,
        t2smark_formal_attack_families="",
    )
    paths = output_paths(tmp_path, config)
    paths["official_images"].mkdir(parents=True)
    for index in range(5):
        image_path = paths["official_images"] / f"{index:05d}.png"
        image_path.write_bytes(f"fake_png_bytes_for_t2smark_smoke_{index}".encode("utf-8"))
    paths["image_pairs"].parent.mkdir(parents=True, exist_ok=True)
    paths["image_pairs"].write_text(
        '[{"image_id":"t2smark_00000","generated_image_path":"","generated_image_digest":""}]\n',
        encoding="utf-8",
    )

    rows = build_t2smark_image_pairs(tmp_path, config, paths)

    assert len(rows) == 5
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
        tree_ring_attack_families="jpeg_compression,rotation",
        gaussian_shading_attack_families="jpeg_compression,rotation",
        shallow_diffuse_attack_families="jpeg_compression,rotation",
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
            for baseline_id in ("tree_ring", "gaussian_shading", "shallow_diffuse"):
                manifest_path = paths["adapter_output_root"] / baseline_id / f"{baseline_id}_method_faithful_sd35_adapter_manifest.json"
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(
                    json.dumps({"baseline_id": baseline_id, "attacked_image_count": 4}, ensure_ascii=False),
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
    assert build_command[build_command.index("--tree-ring-adapter-mode") + 1] == "method_faithful_sd35"
    assert build_command[build_command.index("--gaussian-shading-adapter-mode") + 1] == "method_faithful_sd35"
    assert build_command[build_command.index("--shallow-diffuse-adapter-mode") + 1] == "method_faithful_sd35"
    assert build_command[build_command.index("--tree-ring-attack-families") + 1] == "jpeg_compression,rotation"
    assert build_command[build_command.index("--gaussian-shading-attack-families") + 1] == "jpeg_compression,rotation"
    assert build_command[build_command.index("--shallow-diffuse-attack-families") + 1] == "jpeg_compression,rotation"
    assert paths["primary_prompt_plan"].is_file()
    assert report["primary_baseline_adapter_ready"] is True
    assert report["primary_baseline_adapter_count"] == 4
    assert report["primary_baseline_observation_count"] == 8
    assert report["primary_baseline_attacked_image_count"] == 12


@pytest.mark.quick
def test_primary_baseline_adapter_report_keeps_partial_counts_on_runner_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """命令计划中单个 baseline 失败时, helper 仍应汇总已完成 baseline 的诊断计数。"""

    config = ExternalBaselineGpuSmokeConfig(
        output_dir="outputs/external_baseline_gpu_smoke",
        require_cuda=True,
        primary_baseline_max_samples=1,
    )
    paths = output_paths(tmp_path, config)
    paths["output_dir"].mkdir(parents=True)

    def fake_run_command(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict[str, object]:
        command_text = " ".join(command)
        if "run_external_baseline_command_plan.py" in command_text:
            paths["execution_manifest"].parent.mkdir(parents=True, exist_ok=True)
            paths["execution_manifest"].write_text('{"observation_count":6}\n', encoding="utf-8")
            paths["baseline_observations"].write_text("[]\n", encoding="utf-8")
            paths["command_results"].write_text(
                "["
                '{"baseline_id":"tree_ring","return_code":0,"observation_count":2},'
                '{"baseline_id":"gaussian_shading","return_code":0,"observation_count":2},'
                '{"baseline_id":"shallow_diffuse","return_code":0,"observation_count":2},'
                '{"baseline_id":"t2smark","return_code":1,"observation_count":0}'
                "]\n",
                encoding="utf-8",
            )
            for baseline_id in ("tree_ring", "gaussian_shading", "shallow_diffuse"):
                manifest_path = paths["adapter_output_root"] / baseline_id / f"{baseline_id}_method_faithful_sd35_adapter_manifest.json"
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(
                    json.dumps({"baseline_id": baseline_id, "attacked_image_count": 4}, ensure_ascii=False),
                    encoding="utf-8",
                )
            return {"command": command, "return_code": 1, "stdout": "", "stderr": ""}
        return {"command": command, "return_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("paper_workflow.colab_utils.external_baseline_gpu_smoke.run_command", fake_run_command)

    report = build_and_run_primary_baseline_adapters(tmp_path, config, paths)

    assert report["adapter_execution_ready"] is False
    assert report["adapter_unsupported_reason"] == "command_plan_runner_failed"
    assert report["adapter_observation_count"] == 6
    assert report["primary_baseline_adapter_ready"] is False
    assert report["primary_baseline_observation_count"] == 6
    assert report["ready_primary_baseline_ids"] == ["tree_ring", "gaussian_shading", "shallow_diffuse"]
    assert report["attacked_image_count_by_baseline"]["t2smark"] == 0
