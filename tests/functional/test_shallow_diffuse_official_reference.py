"""验证 Shallow Diffuse 官方原始环境补充表 governed import 协议。"""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.baselines import (
    SHALLOW_DIFFUSE_OFFICIAL_REFERENCE_PROTOCOL_NAME,
    build_shallow_diffuse_official_reference_record,
    build_shallow_diffuse_official_reference_schema,
    validate_shallow_diffuse_official_reference_records,
)
from paper_workflow.colab_utils.shallow_diffuse_official_reference import (
    ShallowDiffuseOfficialReferenceConfig,
    build_default_config,
    build_official_command,
    ensure_shallow_diffuse_source_available,
    output_paths,
    package_shallow_diffuse_official_reference_outputs,
    parse_metric_text,
    patch_shallow_diffuse_model_repository_layout,
    prepare_shallow_diffuse_legacy_environment,
    prepare_shallow_diffuse_model_repository,
    write_shallow_diffuse_official_reference_outputs,
)


@pytest.mark.quick
def test_shallow_diffuse_official_reference_record_validates_when_all_boundaries_ready() -> None:
    """官方 legacy 复现记录满足证据边界时应通过补充表导入校验。"""

    record = build_shallow_diffuse_official_reference_record(
        official_entrypoint="external_baseline/primary/shallow_diffuse/source/run_shallow_diffuse_t2i.py",
        official_repository_commit="c80c553fdf66fda8db735d77a9d56538b7a0ade8",
        official_environment_profile="python3.9_diffusers0.11.1_shallow_diffuse",
        baseline_result_source="outputs/shallow_diffuse_official_reference/summary.json",
        baseline_result_source_digest="digest",
        evidence_paths=["outputs/shallow_diffuse_official_reference/summary.json"],
        metric_values={
            "sample_count": 5,
            "positive_count": 5,
            "negative_count": 5,
            "auc": 0.9,
            "accuracy": 0.8,
            "true_positive_rate_at_one_percent_fpr": 0.7,
            "clip_score_mean": 0.0,
            "watermarked_clip_score_mean": 0.0,
        },
        ready_flags={
            "official_source_ready": True,
            "official_environment_report_ready": True,
            "official_result_summary_ready": True,
            "governed_import_ready": True,
        },
    )

    report = validate_shallow_diffuse_official_reference_records([record])
    schema = build_shallow_diffuse_official_reference_schema()

    assert schema["reference_protocol_name"] == SHALLOW_DIFFUSE_OFFICIAL_REFERENCE_PROTOCOL_NAME
    assert record["supplemental_table_role"] == "supplemental_method_fidelity_reference"
    assert record["main_table_eligible"] is False
    assert report["reference_import_ready"] is True
    assert report["accepted_reference_record_count"] == 1


@pytest.mark.quick
def test_shallow_diffuse_official_reference_rejects_main_table_eligibility() -> None:
    """官方 legacy 参考记录不得伪装为主表同协议结果。"""

    record = build_shallow_diffuse_official_reference_record(
        official_entrypoint="external_baseline/primary/shallow_diffuse/source/run_shallow_diffuse_t2i.py",
        official_repository_commit="c80c553fdf66fda8db735d77a9d56538b7a0ade8",
        official_environment_profile="python3.9_diffusers0.11.1_shallow_diffuse",
        baseline_result_source="outputs/shallow_diffuse_official_reference/summary.json",
        baseline_result_source_digest="digest",
        evidence_paths=["outputs/shallow_diffuse_official_reference/summary.json"],
        metric_values={
            "sample_count": 5,
            "positive_count": 5,
            "negative_count": 5,
            "auc": 0.9,
            "accuracy": 0.8,
            "true_positive_rate_at_one_percent_fpr": 0.7,
            "clip_score_mean": 0.0,
            "watermarked_clip_score_mean": 0.0,
        },
        ready_flags={
            "official_source_ready": True,
            "official_environment_report_ready": True,
            "official_result_summary_ready": True,
            "governed_import_ready": True,
        },
    )
    record["main_table_eligible"] = True

    report = validate_shallow_diffuse_official_reference_records([record])
    reasons = {issue["reason"] for issue in report["issues"]}

    assert report["reference_import_ready"] is False
    assert "legacy_reference_must_not_enter_main_table" in reasons


@pytest.mark.quick
def test_shallow_diffuse_official_reference_patches_source_runtime_boundaries(tmp_path: Path) -> None:
    """helper 应把 Shallow Diffuse 官方源码运行补丁记录为可审计产物。"""

    source_dir = tmp_path / "external_baseline" / "primary" / "shallow_diffuse" / "source"
    source_dir.mkdir(parents=True)
    entrypoint = source_dir / "run_shallow_diffuse_t2i.py"
    entrypoint.write_text(
        "pipe = InversableStableDiffusionPipeline.from_pretrained(\n"
        "        args.model_id,\n"
        "        scheduler=scheduler,\n"
        "        torch_dtype=torch.float16,\n"
        "        revision='fp16',\n"
        "        )\n"
        "    suffixes = ['none', 'jpeg', 'gaussianblur', 'gaussianstd', 'colorjitter','randomdrop', 'saltandpepper', 'resizerestore','vaebmshj', 'vaecheng', 'diff']\n"
        "    attackers = {\n"
        "        'none': image_distortion_none,\n"
        "        'diffpure': image_distortion_diffpure,\n"
        "    }\n",
        encoding="utf-8",
    )
    attackers = source_dir / "attackers.py"
    attackers.write_text(
        "import os\n"
        "from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor\n"
        "def initialize_attackers(args, device):\n"
        "    global vae_attacker1, vae_attacker2, diff_attacker\n"
        "    if args.vae_attack_model_name1 is not None and args.vae_attack_model_name2 is not None:\n"
        "        vae_attacker1 = VAEWMAttacker(args.vae_attack_model_name1, quality=3, metric='mse', device=device)\n"
        "    att_pipe = ReSDPipeline.from_pretrained(\"stabilityai/stable-diffusion-2-1\", torch_dtype=torch.float16, revision=\"fp16\")\n"
        "    diff_attacker = DiffWMAttacker(att_pipe, batch_size=1, noise_step=60, captions={})\n"
        "\n"
        "def image_distortion_none(imgs, seed, args):\n"
        "    return\n",
        encoding="utf-8",
    )
    config = ShallowDiffuseOfficialReferenceConfig(
        output_dir="outputs/shallow_diffuse_official_reference",
        source_dir="external_baseline/primary/shallow_diffuse/source",
        patch_model_repository_layout=True,
        require_cuda=False,
    )
    paths = output_paths(tmp_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    report = patch_shallow_diffuse_model_repository_layout(tmp_path, config, paths)
    patched_entrypoint = entrypoint.read_text(encoding="utf-8")
    patched_attackers = attackers.read_text(encoding="utf-8")

    assert report["patch_applied"] is True
    assert "remove_fp16_revision_branch" in report["patch_items"]
    assert "environment_controlled_attacker_suffixes" in report["patch_items"]
    assert "lazy_heavy_attacker_initialization" in report["patch_items"]
    assert "revision='fp16'" not in patched_entrypoint
    assert "SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_ATTACKER_NAMES" in patched_entrypoint
    assert "compressai_required_for_vae_attackers" in patched_attackers


@pytest.mark.quick
def test_shallow_diffuse_official_reference_prepares_local_model_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """本地模型目录应补齐 legacy transformers 需要的 model_index 兼容项。"""

    local_model_dir = tmp_path / "runtime_model" / "stable_diffusion_2_1_base"
    config = ShallowDiffuseOfficialReferenceConfig(
        output_dir="outputs/shallow_diffuse_official_reference",
        source_dir="external_baseline/primary/shallow_diffuse/source",
        official_model_id="Manojb/stable-diffusion-2-1-base",
        local_model_repository_dir=str(local_model_dir),
        prepare_local_model_repository=True,
        patch_model_index_for_legacy_transformers=True,
        require_cuda=False,
    )
    paths = output_paths(tmp_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    def fake_download_hf_snapshot(repo_id: str, *, local_dir: Path, token: str | None) -> str:
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "StableDiffusionPipeline",
                    "feature_extractor": ["transformers", "CLIPImageProcessor"],
                    "scheduler": ["diffusers", "PNDMScheduler"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return str(local_dir)

    monkeypatch.setattr(
        "paper_workflow.colab_utils.shallow_diffuse_official_reference.download_hf_snapshot",
        fake_download_hf_snapshot,
    )

    report = prepare_shallow_diffuse_model_repository(tmp_path, config, paths)
    patched_index = json.loads((local_model_dir / "model_index.json").read_text(encoding="utf-8"))

    assert report["local_model_repository_ready"] is True
    assert report["model_index_patch_applied"] is True
    assert report["effective_official_model_id"] == str(local_model_dir)
    assert patched_index["feature_extractor"] == ["transformers", "CLIPFeatureExtractor"]


@pytest.mark.quick
def test_shallow_diffuse_official_reference_prepares_isolated_legacy_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Colab 独立会话应能把官方 legacy 依赖准备过程收敛为可审计报告。"""

    config = ShallowDiffuseOfficialReferenceConfig(
        output_dir="outputs/shallow_diffuse_official_reference",
        source_dir="external_baseline/primary/shallow_diffuse/source",
        require_cuda=False,
        prepare_legacy_environment=True,
        legacy_environment_prefix=str(tmp_path / "legacy_env"),
        micromamba_path=str(tmp_path / "bin" / "micromamba"),
        legacy_torch_specs="torch==1.13.0+cu117 torchvision==0.14.0+cu117",
        legacy_package_specs="transformers==4.23.1 diffusers==0.11.1 datasets==2.6.1",
    )
    paths = output_paths(tmp_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    def fake_run_shell_command(command: str, *, cwd: Path, timeout_seconds: int) -> dict[str, object]:
        micromamba_path = Path(config.micromamba_path)
        micromamba_path.parent.mkdir(parents=True, exist_ok=True)
        micromamba_path.write_text("#!/bin/sh\n", encoding="utf-8")
        return {"command": command, "return_code": 0, "stdout": "micromamba ready", "stderr": ""}

    def fake_run_command(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict[str, object]:
        if command[:2] == [config.micromamba_path, "create"]:
            legacy_python = Path(config.legacy_environment_prefix) / "bin" / "python"
            legacy_python.parent.mkdir(parents=True, exist_ok=True)
            legacy_python.write_text("#!/bin/sh\n", encoding="utf-8")
        return {"command": command, "return_code": 0, "stdout": "{}", "stderr": ""}

    monkeypatch.setattr("paper_workflow.colab_utils.shallow_diffuse_official_reference.run_shell_command", fake_run_shell_command)
    monkeypatch.setattr("paper_workflow.colab_utils.shallow_diffuse_official_reference.run_command", fake_run_command)

    report = prepare_shallow_diffuse_legacy_environment(tmp_path, config, paths)

    assert report["legacy_environment_requested"] is True
    assert report["legacy_environment_ready"] is True
    assert report["legacy_python_executable"].replace("\\", "/").endswith("legacy_env/bin/python")
    assert len(report["command_results"]) >= 4


@pytest.mark.quick
def test_shallow_diffuse_official_reference_parses_metric_text_and_custom_python(tmp_path: Path) -> None:
    """官方日志解析与 legacy Python 可执行文件配置应保持可审计。"""

    config = ShallowDiffuseOfficialReferenceConfig(
        source_dir="external_baseline/primary/shallow_diffuse/source",
        official_python_executable="/opt/shallow-diffuse-legacy/bin/python",
        sample_count=5,
        edit_time_list="0.3",
        num_inference_steps=50,
        attacker_names="none",
    )

    metrics = parse_metric_text(
        "clip_score_mean: 0.0\navg_clip_score_mean: 0.0\nauc: 0.95, acc: 0.84, TPR@1%FPR: 0.72\n",
        sample_count=5,
    )
    command = build_official_command(tmp_path, config)

    assert metrics["sample_count"] == 5
    assert metrics["auc"] == 0.95
    assert command[0] == "/opt/shallow-diffuse-legacy/bin/python"
    assert "--edit_time_list" in command
    assert command[command.index("--reference_model") + 1] == "ViT-g-14"
    assert command[command.index("--reference_model_pretrain") + 1] == "laion2b_s12b_b42k"
    assert command[command.index("--end") + 1] == "5"


@pytest.mark.quick
def test_shallow_diffuse_official_reference_helper_imports_governed_summary(tmp_path: Path) -> None:
    """专用 helper 应能把外部官方复现 summary 转换为 governed import 记录。"""

    source_dir = tmp_path / "external_baseline" / "primary" / "shallow_diffuse" / "source"
    source_dir.mkdir(parents=True)
    (source_dir / "run_shallow_diffuse_t2i.py").write_text("print('shallow diffuse official entry')\n", encoding="utf-8")
    (source_dir / "attackers.py").write_text("print('attackers')\n", encoding="utf-8")
    imported_summary = tmp_path / "outputs" / "shallow_diffuse_official_reference" / "imported_summary.json"
    imported_summary.parent.mkdir(parents=True)
    imported_summary.write_text(
        json.dumps(
            {
                "sample_count": 5,
                "positive_count": 5,
                "negative_count": 5,
                "auc": 0.91,
                "accuracy": 0.82,
                "true_positive_rate_at_one_percent_fpr": 0.73,
                "clip_score_mean": 0.0,
                "watermarked_clip_score_mean": 0.0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = ShallowDiffuseOfficialReferenceConfig(
        output_dir="outputs/shallow_diffuse_official_reference",
        drive_output_dir=str(tmp_path / "drive"),
        source_dir="external_baseline/primary/shallow_diffuse/source",
        sample_count=5,
        run_official_command=False,
        summary_import_path=str(imported_summary),
        require_cuda=False,
    )

    summary = write_shallow_diffuse_official_reference_outputs(config, root=tmp_path)
    records_path = tmp_path / summary["reference_records_path"]
    validation_path = tmp_path / summary["reference_validation_path"]

    assert summary["run_decision"] == "pass"
    assert summary["sample_count"] == 5
    assert summary["governed_reference_record_count"] == 1
    assert records_path.read_text(encoding="utf-8").strip()
    assert json.loads(validation_path.read_text(encoding="utf-8"))["reference_import_ready"] is True


@pytest.mark.quick
def test_shallow_diffuse_official_reference_package_embeds_archive_self_description(tmp_path: Path) -> None:
    """打包结果应包含归档摘要、归档 manifest 和输入清单。"""

    output_dir = tmp_path / "outputs" / "shallow_diffuse_official_reference"
    output_dir.mkdir(parents=True)
    (output_dir / "shallow_diffuse_official_reference_summary.json").write_text(
        json.dumps({"run_decision": "pass"}, ensure_ascii=False),
        encoding="utf-8",
    )

    record = package_shallow_diffuse_official_reference_outputs(
        root=tmp_path,
        output_dir="outputs/shallow_diffuse_official_reference",
        drive_output_dir=str(tmp_path / "drive" / "SLM" / "shallow_diffuse_official_reference"),
        archive_name="shallow_diffuse_official_reference_package.zip",
    )

    archive_path = tmp_path / record.archive_path
    expected_entries = {
        "outputs/shallow_diffuse_official_reference/shallow_diffuse_official_reference_summary.json",
        "outputs/shallow_diffuse_official_reference/shallow_diffuse_official_reference_package_input_manifest.json",
        "outputs/shallow_diffuse_official_reference/shallow_diffuse_official_reference_archive_summary.json",
        "outputs/shallow_diffuse_official_reference/shallow_diffuse_official_reference_archive_manifest.local.json",
    }
    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        package_manifest = json.loads(
            archive.read(
                "outputs/shallow_diffuse_official_reference/shallow_diffuse_official_reference_package_input_manifest.json"
            ).decode("utf-8")
        )

    assert expected_entries <= names
    assert package_manifest["entry_count"] == len(names)
    assert record.archive_digest
    assert record.drive_archive_digest


@pytest.mark.quick
def test_shallow_diffuse_official_reference_cold_start_clones_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Colab 冷启动缺少官方源码时, helper 应按登记表补齐 source 缓存。"""

    registry_path = tmp_path / "external_baseline" / "source_registry.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(
            {
                "baseline_sources": [
                    {
                        "baseline_id": "shallow_diffuse",
                        "source_dir": "external_baseline/primary/shallow_diffuse/source",
                        "official_repository_url": "git@github.com:liwd190019/Shallow-Diffuse.git",
                        "official_repository_commit": "c80c553fdf66fda8db735d77a9d56538b7a0ade8",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = ShallowDiffuseOfficialReferenceConfig(
        output_dir="outputs/shallow_diffuse_official_reference",
        source_dir="external_baseline/primary/shallow_diffuse/source",
        require_cuda=False,
    )
    paths = output_paths(tmp_path, config)

    def fake_run_command(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict[str, object]:
        if command[:2] == ["git", "clone"]:
            source_dir = Path(command[-1])
            source_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "run_shallow_diffuse_t2i.py").write_text("print('official source')\n", encoding="utf-8")
            (source_dir / "README.md").write_text("# Shallow Diffuse\n", encoding="utf-8")
        return {"command": command, "return_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("paper_workflow.colab_utils.shallow_diffuse_official_reference.run_command", fake_run_command)

    report = ensure_shallow_diffuse_source_available(tmp_path, config, paths)

    assert report["source_available"] is True
    assert report["source_downloaded"] is True
    assert report["official_entrypoint_ready"] is True
    assert report["official_repository_url"] == "https://github.com/liwd190019/Shallow-Diffuse.git"
    assert paths["source_prepare_result"].is_file()


@pytest.mark.quick
def test_shallow_diffuse_official_reference_default_config_reads_runtime_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Notebook 参数层应能显式传递 Shallow Diffuse 官方运行参数。"""

    monkeypatch.setenv("SLM_WM_SHALLOW_DIFFUSE_PREPARE_LEGACY_ENV", "1")
    monkeypatch.setenv("SLM_WM_SHALLOW_DIFFUSE_LEGACY_ENV_PREFIX", "/content/shallow_diffuse_legacy_env")
    monkeypatch.setenv("SLM_WM_SHALLOW_DIFFUSE_EDIT_TIME_LIST", "0.3")
    monkeypatch.setenv("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_ATTACKER_NAMES", "none")
    monkeypatch.setenv("SLM_WM_SHALLOW_DIFFUSE_W_PATTERN", "complex2_ring")
    monkeypatch.setenv("SLM_WM_SHALLOW_DIFFUSE_W_MEASUREMENT", "l1_complex2")
    monkeypatch.setenv("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_MODEL_ID", "Manojb/stable-diffusion-2-1-base")
    monkeypatch.setenv("SLM_WM_SHALLOW_DIFFUSE_REFERENCE_MODEL", "ViT-g-14")
    monkeypatch.setenv("SLM_WM_SHALLOW_DIFFUSE_REFERENCE_MODEL_PRETRAIN", "laion2b_s12b_b42k")

    config = build_default_config()

    assert config.prepare_legacy_environment is True
    assert config.legacy_environment_prefix == "/content/shallow_diffuse_legacy_env"
    assert config.edit_time_list == "0.3"
    assert config.attacker_names == "none"
    assert config.w_pattern == "complex2_ring"
    assert config.w_measurement == "l1_complex2"
    assert config.official_model_id == "Manojb/stable-diffusion-2-1-base"
    assert config.reference_model == "ViT-g-14"
    assert config.reference_model_pretrain == "laion2b_s12b_b42k"
