"""验证外部 baseline 源码缓存边界与协议清单一致。"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.baselines import default_baseline_specs


def test_external_baseline_source_cache_is_ignored_by_repository() -> None:
    """外部源码缓存目录应由根 .gitignore 排除, 防止第三方源码进入提交。"""
    ignored_lines = {line.strip() for line in Path(".gitignore").read_text(encoding="utf-8").splitlines()}
    assert "external_baseline/" in ignored_lines


def test_external_baseline_source_registry_matches_adapter_specs_when_present() -> None:
    """若本地存在来源登记, 其内容应与 baseline adapter 的主表和补充表分组一致。"""
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
