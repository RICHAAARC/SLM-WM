"""验证 Gaussian Shading 官方原始环境补充表 governed import 协议。"""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.baselines import (
    GAUSSIAN_SHADING_OFFICIAL_REFERENCE_PROTOCOL_NAME,
    build_gaussian_shading_official_reference_record,
    build_gaussian_shading_official_reference_schema,
    validate_gaussian_shading_official_reference_records,
)
from paper_workflow.colab_utils.gaussian_shading_official_reference import (
    GaussianShadingOfficialReferenceConfig,
    build_default_config,
    build_official_command,
    ensure_gaussian_shading_source_available,
    output_paths,
    package_gaussian_shading_official_reference_outputs,
    parse_metric_text,
    patch_gaussian_shading_model_repository_layout,
    prepare_gaussian_shading_legacy_environment,
    prepare_gaussian_shading_model_repository,
    write_gaussian_shading_official_reference_outputs,
)


@pytest.mark.quick
def test_gaussian_shading_official_reference_record_validates_when_all_boundaries_ready() -> None:
    """官方 legacy 复现记录满足证据边界时应通过补充表导入校验。"""

    record = build_gaussian_shading_official_reference_record(
        official_entrypoint="external_baseline/primary/gaussian_shading/source/run_gaussian_shading.py",
        official_repository_commit="09c678fadc7545acf7be12647ddf2a5e66f6a9dc",
        official_environment_profile="python3.9_diffusers0.11.1_legacy_gaussian_shading",
        baseline_result_source="outputs/gaussian_shading_official_reference/summary.json",
        baseline_result_source_digest="digest",
        evidence_paths=["outputs/gaussian_shading_official_reference/summary.json"],
        metric_values={
            "sample_count": 5,
            "positive_count": 5,
            "detection_true_positive_rate": 0.8,
            "traceability_true_positive_rate": 0.6,
            "mean_bit_accuracy": 0.9,
            "std_bit_accuracy": 0.1,
            "mean_clip_score": 0.0,
            "std_clip_score": 0.0,
        },
        ready_flags={
            "official_source_ready": True,
            "official_environment_report_ready": True,
            "official_result_summary_ready": True,
            "governed_import_ready": True,
        },
    )

    report = validate_gaussian_shading_official_reference_records([record])
    schema = build_gaussian_shading_official_reference_schema()

    assert schema["reference_protocol_name"] == GAUSSIAN_SHADING_OFFICIAL_REFERENCE_PROTOCOL_NAME
    assert record["supplemental_table_role"] == "supplemental_method_fidelity_reference"
    assert record["main_table_eligible"] is False
    assert report["reference_import_ready"] is True
    assert report["accepted_reference_record_count"] == 1


@pytest.mark.quick
def test_gaussian_shading_official_reference_rejects_main_table_eligibility() -> None:
    """官方 legacy 参考记录不得伪装为主表同协议结果。"""

    record = build_gaussian_shading_official_reference_record(
        official_entrypoint="external_baseline/primary/gaussian_shading/source/run_gaussian_shading.py",
        official_repository_commit="09c678fadc7545acf7be12647ddf2a5e66f6a9dc",
        official_environment_profile="python3.9_diffusers0.11.1_legacy_gaussian_shading",
        baseline_result_source="outputs/gaussian_shading_official_reference/summary.json",
        baseline_result_source_digest="digest",
        evidence_paths=["outputs/gaussian_shading_official_reference/summary.json"],
        metric_values={
            "sample_count": 5,
            "positive_count": 5,
            "detection_true_positive_rate": 0.8,
            "traceability_true_positive_rate": 0.6,
            "mean_bit_accuracy": 0.9,
            "std_bit_accuracy": 0.1,
            "mean_clip_score": 0.0,
            "std_clip_score": 0.0,
        },
        ready_flags={
            "official_source_ready": True,
            "official_environment_report_ready": True,
            "official_result_summary_ready": True,
            "governed_import_ready": True,
        },
    )
    record["main_table_eligible"] = True

    report = validate_gaussian_shading_official_reference_records([record])
    reasons = {issue["reason"] for issue in report["issues"]}

    assert report["reference_import_ready"] is False
    assert "legacy_reference_must_not_enter_main_table" in reasons


@pytest.mark.quick
def test_gaussian_shading_official_reference_patches_model_repository_layout(tmp_path: Path) -> None:
    """公开镜像缺少 fp16 分支时, helper 应把官方入口补丁记录为可审计产物。"""

    source_dir = tmp_path / "external_baseline" / "primary" / "gaussian_shading" / "source"
    source_dir.mkdir(parents=True)
    entrypoint = source_dir / "run_gaussian_shading.py"
    entrypoint.write_text(
        "pipe = InversableStableDiffusionPipeline.from_pretrained(\n"
        "            args.model_path,\n"
        "            scheduler=scheduler,\n"
        "            torch_dtype=torch.float16,\n"
        "            revision='fp16',\n"
        ")\n",
        encoding="utf-8",
    )
    config = GaussianShadingOfficialReferenceConfig(
        output_dir="outputs/gaussian_shading_official_reference",
        source_dir="external_baseline/primary/gaussian_shading/source",
        official_model_id="Manojb/stable-diffusion-2-1-base",
        patch_model_repository_layout=True,
        require_cuda=False,
    )
    paths = output_paths(tmp_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    report = patch_gaussian_shading_model_repository_layout(tmp_path, config, paths)
    patched_text = entrypoint.read_text(encoding="utf-8")
    saved_report = json.loads(paths["source_patch_result"].read_text(encoding="utf-8"))

    assert report["patch_applied"] is True
    assert "revision='fp16'" not in patched_text
    assert "公开镜像没有 fp16 分支" in patched_text
    assert saved_report["official_model_id"] == "Manojb/stable-diffusion-2-1-base"
    assert saved_report["upstream_official_model_id"] == "stabilityai/stable-diffusion-2-1-base"


@pytest.mark.quick
def test_gaussian_shading_official_reference_prepares_local_model_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """本地模型目录应补齐 legacy diffusers 需要的 model_index 兼容项。"""

    local_model_dir = tmp_path / "runtime_model" / "stable_diffusion_2_1_base"
    config = GaussianShadingOfficialReferenceConfig(
        output_dir="outputs/gaussian_shading_official_reference",
        source_dir="external_baseline/primary/gaussian_shading/source",
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
        "paper_workflow.colab_utils.gaussian_shading_official_reference.download_hf_snapshot",
        fake_download_hf_snapshot,
    )

    report = prepare_gaussian_shading_model_repository(tmp_path, config, paths)
    patched_index = json.loads((local_model_dir / "model_index.json").read_text(encoding="utf-8"))
    saved_report = json.loads(paths["model_repository_prepare_result"].read_text(encoding="utf-8"))

    assert report["local_model_repository_ready"] is True
    assert report["model_index_patch_applied"] is True
    assert report["effective_official_model_id"] == str(local_model_dir)
    assert patched_index["feature_extractor"] == ["transformers", "CLIPFeatureExtractor"]
    assert saved_report["model_index_feature_extractor"] == ["transformers", "CLIPFeatureExtractor"]


@pytest.mark.quick
def test_gaussian_shading_official_reference_prepares_isolated_legacy_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Colab 独立会话应能把官方 legacy 依赖准备过程收敛为可审计报告。"""

    config = GaussianShadingOfficialReferenceConfig(
        output_dir="outputs/gaussian_shading_official_reference",
        source_dir="external_baseline/primary/gaussian_shading/source",
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
            environment_prefix = Path(command[command.index("-p") + 1])
            legacy_python = environment_prefix / "bin" / "python"
            legacy_python.parent.mkdir(parents=True, exist_ok=True)
            legacy_python.write_text("#!/bin/sh\n", encoding="utf-8")
        return {"command": command, "return_code": 0, "stdout": "{}", "stderr": ""}

    monkeypatch.setattr("paper_workflow.colab_utils.gaussian_shading_official_reference.run_shell_command", fake_run_shell_command)
    monkeypatch.setattr("paper_workflow.colab_utils.gaussian_shading_official_reference.run_command", fake_run_command)

    report = prepare_gaussian_shading_legacy_environment(tmp_path, config, paths)
    saved_report = json.loads(paths["legacy_environment_prepare_result"].read_text(encoding="utf-8"))

    assert report["legacy_environment_requested"] is True
    assert report["legacy_environment_ready"] is True
    assert saved_report["legacy_environment_profile"] == "colab_compatible_fallback"
    assert saved_report["strict_official_environment_ready"] is False
    assert saved_report["compatible_environment_fallback_ready"] is True
    assert saved_report["legacy_python_executable"].replace("\\", "/").endswith(
        "legacy_env/colab_compatible_fallback/bin/python"
    )
    assert len(saved_report["command_results"]) >= 4


@pytest.mark.quick
def test_gaussian_shading_official_reference_prefers_strict_official_requirements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """官方 requirements 可安装时, helper 应优先使用严格官方环境。"""

    source_dir = tmp_path / "external_baseline" / "primary" / "gaussian_shading" / "source"
    source_dir.mkdir(parents=True)
    (source_dir / "requirements.txt").write_text("diffusers==0.11.1\n", encoding="utf-8")
    config = GaussianShadingOfficialReferenceConfig(
        output_dir="outputs/gaussian_shading_official_reference",
        source_dir="external_baseline/primary/gaussian_shading/source",
        require_cuda=False,
        prepare_legacy_environment=True,
        legacy_environment_prefix=str(tmp_path / "legacy_env"),
        micromamba_path=str(tmp_path / "bin" / "micromamba"),
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
            environment_prefix = Path(command[command.index("-p") + 1])
            legacy_python = environment_prefix / "bin" / "python"
            legacy_python.parent.mkdir(parents=True, exist_ok=True)
            legacy_python.write_text("#!/bin/sh\n", encoding="utf-8")
        return {"command": command, "return_code": 0, "stdout": "{}", "stderr": ""}

    monkeypatch.setattr("paper_workflow.colab_utils.gaussian_shading_official_reference.run_shell_command", fake_run_shell_command)
    monkeypatch.setattr("paper_workflow.colab_utils.gaussian_shading_official_reference.run_command", fake_run_command)

    report = prepare_gaussian_shading_legacy_environment(tmp_path, config, paths)

    assert report["legacy_environment_ready"] is True
    assert report["legacy_environment_profile"] == "official_requirements_strict"
    assert report["strict_official_environment_ready"] is True
    assert report["compatible_environment_fallback_ready"] is False
    assert report["legacy_python_executable"].replace("\\", "/").endswith(
        "legacy_env/official_requirements_strict/bin/python"
    )


@pytest.mark.quick
def test_gaussian_shading_official_reference_falls_back_after_strict_dependency_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """官方 requirements 依赖冲突时, helper 应切换到受治理兼容环境。"""

    source_dir = tmp_path / "external_baseline" / "primary" / "gaussian_shading" / "source"
    source_dir.mkdir(parents=True)
    (source_dir / "requirements.txt").write_text("transformers==4.34.0\ndiffusers==0.11.1\n", encoding="utf-8")
    config = GaussianShadingOfficialReferenceConfig(
        output_dir="outputs/gaussian_shading_official_reference",
        source_dir="external_baseline/primary/gaussian_shading/source",
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
            environment_prefix = Path(command[command.index("-p") + 1])
            legacy_python = environment_prefix / "bin" / "python"
            legacy_python.parent.mkdir(parents=True, exist_ok=True)
            legacy_python.write_text("#!/bin/sh\n", encoding="utf-8")
        is_strict_pip_install = "-r" in command and "official_requirements_strict" in str(command[0])
        if is_strict_pip_install:
            return {"command": command, "return_code": 1, "stdout": "", "stderr": "ResolutionImpossible"}
        return {"command": command, "return_code": 0, "stdout": "{}", "stderr": ""}

    monkeypatch.setattr("paper_workflow.colab_utils.gaussian_shading_official_reference.run_shell_command", fake_run_shell_command)
    monkeypatch.setattr("paper_workflow.colab_utils.gaussian_shading_official_reference.run_command", fake_run_command)

    report = prepare_gaussian_shading_legacy_environment(tmp_path, config, paths)

    assert report["legacy_environment_ready"] is True
    assert report["legacy_environment_profile"] == "colab_compatible_fallback"
    assert report["strict_official_environment_ready"] is False
    assert report["compatible_environment_fallback_ready"] is True
    assert any(
        item["environment_profile"] == "official_requirements_strict" and not item["environment_ready"]
        for item in report["environment_profile_reports"]
    )


@pytest.mark.quick
def test_gaussian_shading_official_reference_helper_imports_governed_summary(tmp_path: Path) -> None:
    """专用 helper 应能把外部官方复现 summary 转换为 governed import 记录。"""

    source_dir = tmp_path / "external_baseline" / "primary" / "gaussian_shading" / "source"
    source_dir.mkdir(parents=True)
    (source_dir / "run_gaussian_shading.py").write_text("print('gaussian shading official entry')\n", encoding="utf-8")
    (source_dir / "requirements.txt").write_text("diffusers==0.11.1\ntransformers==4.34.0\n", encoding="utf-8")
    imported_summary = tmp_path / "outputs" / "gaussian_shading_official_reference" / "imported_summary.json"
    imported_summary.parent.mkdir(parents=True)
    imported_summary.write_text(
        json.dumps(
            {
                "sample_count": 5,
                "positive_count": 5,
                "detection_true_positive_rate": 0.8,
                "traceability_true_positive_rate": 0.6,
                "mean_bit_accuracy": 0.9,
                "std_bit_accuracy": 0.1,
                "mean_clip_score": 0.0,
                "std_clip_score": 0.0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = GaussianShadingOfficialReferenceConfig(
        output_dir="outputs/gaussian_shading_official_reference",
        drive_output_dir=str(tmp_path / "drive"),
        source_dir="external_baseline/primary/gaussian_shading/source",
        sample_count=5,
        run_official_command=False,
        summary_import_path=str(imported_summary),
        require_cuda=False,
    )

    summary = write_gaussian_shading_official_reference_outputs(config, root=tmp_path)
    records_path = tmp_path / summary["reference_records_path"]
    validation_path = tmp_path / summary["reference_validation_path"]

    assert summary["run_decision"] == "pass"
    assert summary["sample_count"] == 5
    assert summary["governed_reference_record_count"] == 1
    assert records_path.read_text(encoding="utf-8").strip()
    assert json.loads(validation_path.read_text(encoding="utf-8"))["reference_import_ready"] is True


@pytest.mark.quick
def test_gaussian_shading_official_reference_package_embeds_archive_self_description(tmp_path: Path) -> None:
    """打包结果应包含归档摘要、归档 manifest 和输入清单。"""

    output_dir = tmp_path / "outputs" / "gaussian_shading_official_reference"
    output_dir.mkdir(parents=True)
    (output_dir / "gaussian_shading_official_reference_summary.json").write_text(
        json.dumps({"run_decision": "pass"}, ensure_ascii=False),
        encoding="utf-8",
    )

    record = package_gaussian_shading_official_reference_outputs(
        root=tmp_path,
        output_dir="outputs/gaussian_shading_official_reference",
        drive_output_dir=str(tmp_path / "drive" / "SLM" / "external_baseline_official_reference"),
        archive_name="external_baseline_official_reference_package_gaussian_shading.zip",
    )

    archive_path = tmp_path / record.archive_path
    expected_entries = {
        "outputs/gaussian_shading_official_reference/gaussian_shading_official_reference_summary.json",
        "outputs/gaussian_shading_official_reference/gaussian_shading_official_reference_package_input_manifest.json",
        "outputs/gaussian_shading_official_reference/gaussian_shading_official_reference_archive_summary.json",
        "outputs/gaussian_shading_official_reference/gaussian_shading_official_reference_archive_manifest.local.json",
    }
    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        package_manifest = json.loads(
            archive.read(
                "outputs/gaussian_shading_official_reference/gaussian_shading_official_reference_package_input_manifest.json"
            ).decode("utf-8")
        )
        embedded_summary = json.loads(
            archive.read(
                "outputs/gaussian_shading_official_reference/gaussian_shading_official_reference_archive_summary.json"
            ).decode("utf-8")
        )
        embedded_manifest = json.loads(
            archive.read(
                "outputs/gaussian_shading_official_reference/gaussian_shading_official_reference_archive_manifest.local.json"
            ).decode("utf-8")
        )

    local_summary = json.loads(
        (output_dir / "gaussian_shading_official_reference_archive_summary.json").read_text(encoding="utf-8")
    )
    local_manifest = json.loads(
        (output_dir / "gaussian_shading_official_reference_archive_manifest.local.json").read_text(encoding="utf-8")
    )

    assert expected_entries <= names
    assert package_manifest["entry_count"] == len(names)
    assert package_manifest["embedded_digest_scope"] == "external_summary_records_final_archive_digest"
    assert embedded_summary["metadata"]["embedded_digest_scope"] == "external_summary_records_final_archive_digest"
    assert embedded_manifest["metadata"]["embedded_digest_scope"] == "external_summary_records_final_archive_digest"
    assert local_summary["archive_digest"] == record.archive_digest
    assert local_summary["drive_archive_digest"] == record.drive_archive_digest
    assert local_manifest["metadata"]["archive_digest"] == record.archive_digest
    assert local_manifest["metadata"]["drive_archive_digest"] == record.drive_archive_digest


@pytest.mark.quick
def test_gaussian_shading_official_reference_cold_start_clones_source(
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
                        "baseline_id": "gaussian_shading",
                        "source_dir": "external_baseline/primary/gaussian_shading/source",
                        "official_repository_url": "git@github.com:bsmhmmlf/Gaussian-Shading.git",
                        "official_repository_commit": "09c678fadc7545acf7be12647ddf2a5e66f6a9dc",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = GaussianShadingOfficialReferenceConfig(
        output_dir="outputs/gaussian_shading_official_reference",
        source_dir="external_baseline/primary/gaussian_shading/source",
        require_cuda=False,
    )
    paths = output_paths(tmp_path, config)

    def fake_run_command(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict[str, object]:
        if command[:2] == ["git", "clone"]:
            source_dir = Path(command[-1])
            source_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "run_gaussian_shading.py").write_text("print('official source')\n", encoding="utf-8")
            (source_dir / "requirements.txt").write_text("diffusers==0.11.1\n", encoding="utf-8")
        return {"command": command, "return_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("paper_workflow.colab_utils.gaussian_shading_official_reference.run_command", fake_run_command)

    report = ensure_gaussian_shading_source_available(tmp_path, config, paths)

    assert report["source_available"] is True
    assert report["source_downloaded"] is True
    assert report["official_entrypoint_ready"] is True
    assert report["official_repository_url"] == "https://github.com/bsmhmmlf/Gaussian-Shading.git"
    assert paths["source_prepare_result"].is_file()


@pytest.mark.quick
def test_gaussian_shading_official_reference_parses_metric_text_and_custom_python(tmp_path: Path) -> None:
    """官方日志解析与 legacy Python 可执行文件配置应保持可审计。"""

    config = GaussianShadingOfficialReferenceConfig(
        source_dir="external_baseline/primary/gaussian_shading/source",
        official_python_executable="/opt/gaussian-shading-legacy/bin/python",
        sample_count=5,
        official_model_id="/content/model",
    )
    paths = output_paths(tmp_path, config)

    metrics = parse_metric_text(
        "tpr_detection:0.8      tpr_traceability:0.6      mean_acc:0.9      std_acc:0.1\n",
        sample_count=5,
    )
    command = build_official_command(tmp_path, config, paths)

    assert metrics["sample_count"] == 5
    assert metrics["detection_true_positive_rate"] == 0.8
    assert metrics["mean_bit_accuracy"] == 0.9
    assert command[0] == "/opt/gaussian-shading-legacy/bin/python"
    assert "--num" in command
    assert command[command.index("--num") + 1] == "5"


@pytest.mark.quick
def test_gaussian_shading_official_reference_default_config_reads_legacy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Notebook 参数层应能显式开启 Gaussian Shading 官方 legacy 环境准备。"""

    monkeypatch.setenv("SLM_WM_GAUSSIAN_SHADING_PREPARE_LEGACY_ENV", "1")
    monkeypatch.setenv("SLM_WM_GAUSSIAN_SHADING_LEGACY_ENV_PREFIX", "/content/gaussian_shading_legacy_env")
    monkeypatch.setenv("SLM_WM_GAUSSIAN_SHADING_LEGACY_PYTHON_VERSION", "3.8")
    monkeypatch.setenv("SLM_WM_GAUSSIAN_SHADING_STRICT_OFFICIAL_ENV", "1")
    monkeypatch.setenv("SLM_WM_GAUSSIAN_SHADING_ALLOW_COMPATIBLE_ENV_FALLBACK", "1")
    monkeypatch.setenv("SLM_WM_GAUSSIAN_SHADING_LEGACY_TORCH_SPECS", "torch==1.13.0+cu117")
    monkeypatch.setenv("SLM_WM_GAUSSIAN_SHADING_LEGACY_PACKAGE_SPECS", "transformers==4.23.1 diffusers==0.11.1")
    monkeypatch.setenv("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_MODEL_ID", "Manojb/stable-diffusion-2-1-base")
    monkeypatch.setenv("SLM_WM_GAUSSIAN_SHADING_PATCH_MODEL_REPOSITORY_LAYOUT", "1")
    monkeypatch.setenv("SLM_WM_GAUSSIAN_SHADING_PREPARE_LOCAL_MODEL_REPOSITORY", "1")
    monkeypatch.setenv(
        "SLM_WM_GAUSSIAN_SHADING_LOCAL_MODEL_REPOSITORY_DIR",
        "/content/gaussian_shading_model_repository/stable_diffusion_2_1_base",
    )
    monkeypatch.setenv("SLM_WM_GAUSSIAN_SHADING_PATCH_MODEL_INDEX_FOR_LEGACY_TRANSFORMERS", "1")

    config = build_default_config()

    assert config.prepare_legacy_environment is True
    assert config.legacy_environment_prefix == "/content/gaussian_shading_legacy_env"
    assert config.legacy_python_version == "3.8"
    assert config.strict_official_environment is True
    assert config.allow_compatible_environment_fallback is True
    assert config.legacy_torch_specs == "torch==1.13.0+cu117"
    assert config.legacy_package_specs == "transformers==4.23.1 diffusers==0.11.1"
    assert config.official_model_id == "Manojb/stable-diffusion-2-1-base"
    assert config.upstream_official_model_id == "stabilityai/stable-diffusion-2-1-base"
    assert config.patch_model_repository_layout is True
    assert config.prepare_local_model_repository is True
    assert config.local_model_repository_dir == "/content/gaussian_shading_model_repository/stable_diffusion_2_1_base"
    assert config.patch_model_index_for_legacy_transformers is True
