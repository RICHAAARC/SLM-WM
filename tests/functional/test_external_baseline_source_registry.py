"""验证外部 baseline 源码快照边界与协议清单一致。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.baselines import PRIMARY_BASELINE_ADAPTERS, default_baseline_specs
from tools.harness.lib.file_scanner import should_skip_path


@pytest.mark.quick
def test_external_baseline_source_cache_is_ignored_without_hiding_adapters() -> None:
    """官方源码快照应被忽略, 项目维护 adapter 应继续接受 harness 审计。"""

    root_ignored_lines = {line.strip() for line in Path(".gitignore").read_text(encoding="utf-8").splitlines()}
    local_ignored_lines = {
        line.strip() for line in Path("external_baseline/.gitignore").read_text(encoding="utf-8").splitlines()
    }

    assert "external_baseline/" not in root_ignored_lines
    assert "**/source/**" in local_ignored_lines
    assert should_skip_path(Path("external_baseline/primary/tree_ring/source/run.py"))
    assert should_skip_path(Path("external_baseline/primary/tree_ring/artifacts/report.json"))
    assert not should_skip_path(Path("external_baseline/primary/tree_ring/adapter/run_slm_eval.py"))


@pytest.mark.quick
def test_external_baseline_source_registry_matches_adapter_specs_when_present() -> None:
    """本地来源登记应与 baseline adapter 的主表和补充表分组一致。"""

    registry_path = Path("external_baseline/source_registry.json")
    if not registry_path.exists():
        return

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    sources = registry["baseline_sources"]
    expected_specs = {spec.baseline_id: spec for spec in default_baseline_specs()}
    assert set(item["baseline_id"] for item in sources) == set(expected_specs)

    primary_ids = {item["baseline_id"] for item in sources if item["comparison_group"] == "primary"}
    supplemental_ids = {item["baseline_id"] for item in sources if item["comparison_group"] == "supplemental"}
    assert primary_ids == {"tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark"}
    assert supplemental_ids == {"stable_signature", "rivagan", "trustmark", "watermark_anything"}

    for item in sources:
        spec = expected_specs[item["baseline_id"]]
        assert item["baseline_name"] == spec.baseline_name
        assert item["baseline_family"] == spec.baseline_family
        assert item["comparison_group"] == spec.comparison_group
        source_dir = Path(item["source_dir"])
        assert item["source_status"] in {"not_downloaded", "downloaded"}
        if source_dir.exists():
            assert (source_dir / "README.md").exists() or (source_dir / ".git").exists()
        assert item["paper_claim_support"] is False
        assert item["official_source_tracked"] is False
        if item["comparison_group"] == "primary":
            assert item["adapter_path"] == PRIMARY_BASELINE_ADAPTERS[item["baseline_id"]]
            assert Path(item["adapter_path"]).is_file()
            assert item["adapter_status"] in {
                "adapter_contract_ready",
                "sd35_latent_smoke_adapter_ready",
                "method_faithful_sd35_adapter_available",
                "sd35_native_result_adapter_ready",
            }
        else:
            assert item["adapter_status"] == "source_registered_adapter_not_planned"
