"""T2SMark full-main 真实复现 helper 的轻量功能测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_experiments.runners.t2smark_full_main_reproduction import (
    T2SMarkFullMainReproductionConfig,
    build_t2smark_full_main_image_pairs,
    output_paths,
    synchronize_environment_report_with_device_report,
    write_full_main_prompt_inputs,
    write_t2smark_full_main_reproduction_outputs,
)


@pytest.mark.quick
def test_full_main_prompt_inputs_use_pilot_paper_prompt_file(tmp_path: Path) -> None:
    """helper 应把 pilot_paper prompt 文件转换为官方 dataset_key 和 adapter prompt 计划。"""

    prompt_file = tmp_path / "configs" / "paper_main_pilot_paper_prompts.txt"
    prompt_file.parent.mkdir(parents=True)
    prompt_file.write_text(
        "\n".join(f"a pilot_paper prompt {index}" for index in range(120)) + "\n",
        encoding="utf-8",
    )
    config = T2SMarkFullMainReproductionConfig(prompt_file="configs/paper_main_pilot_paper_prompts.txt", require_cuda=False)
    paths = output_paths(tmp_path, config)

    report = write_full_main_prompt_inputs(tmp_path, config, paths)
    dataset = json.loads(paths["prompt_dataset"].read_text(encoding="utf-8"))
    prompt_plan = json.loads(paths["prompt_plan"].read_text(encoding="utf-8"))

    assert report["full_main_prompt_count"] == 120
    assert report["selected_prompt_count"] == 120
    assert report["full_main_prompt_protocol_ready"] is True
    assert dataset["annotations"][0]["caption"] == "a pilot_paper prompt 0"
    assert prompt_plan[0]["prompt_set"] == "pilot_paper"


@pytest.mark.quick
def test_probe_paper_prompt_inputs_use_probe_gate_count(tmp_path: Path) -> None:
    """probe_paper 的 T2SMark 官方参考入口应使用较小门禁验证 Colab 拆分链路。"""

    prompt_file = tmp_path / "configs" / "paper_main_probe_paper_prompts.txt"
    prompt_file.parent.mkdir(parents=True)
    prompt_file.write_text(
        "\n".join(f"a probe_paper prompt {index}" for index in range(60)) + "\n",
        encoding="utf-8",
    )
    config = T2SMarkFullMainReproductionConfig(
        prompt_set="probe_paper",
        prompt_file="configs/paper_main_probe_paper_prompts.txt",
        prompt_limit=60,
        minimum_prompt_protocol_count=10,
        require_cuda=False,
    )
    paths = output_paths(tmp_path, config)

    report = write_full_main_prompt_inputs(tmp_path, config, paths)

    assert report["selected_prompt_count"] == 60
    assert report["minimum_prompt_protocol_count"] == 10
    assert report["full_main_prompt_protocol_ready"] is True
    assert report["paper_claim_scale"] == "probe_paper"


@pytest.mark.quick
def test_full_main_image_pairs_record_image_digest(tmp_path: Path) -> None:
    """image_pairs 应记录 full-main 生成图像路径和 digest。"""

    config = T2SMarkFullMainReproductionConfig(require_cuda=False)
    paths = output_paths(tmp_path, config)
    paths["official_images"].mkdir(parents=True)
    (paths["official_images"] / "00000.png").write_bytes(b"fake_image")
    prompt_rows = [{"prompt_id": "prompt_alpha", "prompt_index": 0, "split": "test"}]

    rows = build_t2smark_full_main_image_pairs(tmp_path, paths, prompt_rows)

    assert rows[0]["generated_image_path"].endswith("images/00000.png")
    assert rows[0]["generated_image_digest"]
    assert json.loads(paths["image_pairs"].read_text(encoding="utf-8"))[0]["prompt_id"] == "prompt_alpha"


@pytest.mark.quick
def test_environment_report_promotes_explicit_gpu_device_report() -> None:
    """环境报告顶层 GPU 字段应与真实 GPU 检查结果保持一致。"""

    environment_report = {
        "cuda_available": None,
        "cuda_version": None,
        "device_count": 0,
        "gpu_name": "",
        "package_versions": {"torch": "2.11.0+cu128"},
    }
    device_report = {
        "cuda_available": True,
        "device_count": 1,
        "device_name": "NVIDIA L4",
    }

    merged_report = synchronize_environment_report_with_device_report(environment_report, device_report)

    assert merged_report["cuda_available"] is True
    assert merged_report["device_count"] == 1
    assert merged_report["gpu_name"] == "NVIDIA L4"
    assert merged_report["t2smark_full_main_device_report"] == device_report


@pytest.mark.quick
def test_full_main_reproduction_reuses_existing_results_and_writes_candidate_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """已有 T2SMark 官方结果可复用时, helper 应跑通 adapter、候选记录和校验报告落盘链路。"""

    prompt_file = tmp_path / "configs" / "paper_main_pilot_paper_prompts.txt"
    prompt_file.parent.mkdir(parents=True)
    prompt_file.write_text(
        "\n".join(f"a pilot_paper prompt {index}" for index in range(120)) + "\n",
        encoding="utf-8",
    )
    config = T2SMarkFullMainReproductionConfig(
        output_dir="outputs/t2smark_full_main_reproduction",
        prompt_file="configs/paper_main_pilot_paper_prompts.txt",
        require_cuda=False,
        reuse_existing=True,
        force_generate=False,
    )
    paths = output_paths(tmp_path, config)
    paths["official_results"].parent.mkdir(parents=True)
    paths["official_results"].write_text('{"0":{"robustness":{"norm1_no_w":0.1,"norm1_w":0.9}}}\n', encoding="utf-8")
    paths["official_images"].mkdir(parents=True)
    (paths["official_images"] / "00000.png").write_bytes(b"fake_png")

    def fake_run_command(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict[str, object]:
        output_path = Path(command[command.index("--out") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                [
                    {"baseline_id": "t2smark", "attack_family": "clean", "attack_condition": "clean_none", "sample_role": "clean_negative", "detection_decision": False},
                    {"baseline_id": "t2smark", "attack_family": "clean", "attack_condition": "clean_none", "sample_role": "positive_source", "detection_decision": True},
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {"command": command, "return_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("paper_experiments.runners.t2smark_full_main_reproduction.run_command", fake_run_command)

    summary = write_t2smark_full_main_reproduction_outputs(config=config, root=tmp_path)
    validation_report = json.loads(paths["validation_report"].read_text(encoding="utf-8"))

    assert summary["run_decision"] == "pass"
    assert summary["t2smark_full_main_reproduction_ready"] is True
    assert summary["full_main_prompt_protocol_ready"] is True
    assert summary["formal_import_candidate_record_count"] == 1
    assert validation_report["accepted_formal_import_count"] == 0
    assert paths["candidate_records"].is_file()
    assert paths["manifest"].is_file()

