"""验证 stage02 核心方法 synthetic smoke 闭环。"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from main.methods.synthetic_smoke import build_core_method_smoke_bundle


@pytest.mark.quick
def test_core_method_smoke_separates_correct_and_wrong_key() -> None:
    """错误 key 不应稳定过阈值, 正确 key 应形成明显分数间隔。"""
    bundle = build_core_method_smoke_bundle()
    scenarios = {scenario.scenario_id: scenario for scenario in bundle.scenarios}

    assert scenarios["watermarked_synthetic_latent"].positive_by_content is True
    assert scenarios["wrong_key_negative"].positive_by_content is False
    assert bundle.metrics["key_separation_margin"] > 0
    assert bundle.metrics["wrong_key_over_threshold"] is False


@pytest.mark.quick
def test_core_method_smoke_rescue_requires_boundary_and_geometry() -> None:
    """rescue 只能在边界窗口且几何可靠时触发。"""
    bundle = build_core_method_smoke_bundle()
    scenarios = {scenario.scenario_id: scenario for scenario in bundle.scenarios}

    assert scenarios["geometric_shifted_latent"].rescue_applied is False
    assert scenarios["aligned_recovered_latent"].rescue_eligible is True
    assert scenarios["aligned_recovered_latent"].rescue_applied is True
    assert scenarios["unreliable_geometry_shifted_latent"].rescue_eligible is False
    assert scenarios["unreliable_geometry_shifted_latent"].rescue_applied is False


@pytest.mark.quick
def test_core_method_smoke_attestation_only_changes_final_decision() -> None:
    """attestation 不改变 evidence decision, 只改变 final decision。"""
    bundle = build_core_method_smoke_bundle()
    scenarios = {scenario.scenario_id: scenario for scenario in bundle.scenarios}

    assert scenarios["unattested_positive"].evidence_decision is True
    assert scenarios["unattested_positive"].final_decision is False
    assert scenarios["final_positive"].evidence_decision is True
    assert scenarios["final_positive"].final_decision is True
    assert bundle.metrics["attestation_layering_pass"] is True


@pytest.mark.quick
def test_minimal_method_smoke_script_runs() -> None:
    """minimal method smoke 脚本必须可运行并返回 pass。"""
    completed = subprocess.run(
        [sys.executable, "scripts/run_minimal_method_smoke.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["decision"] == "pass"
    assert report["metadata"]["minimal_method_dependency"] == "main.methods"
