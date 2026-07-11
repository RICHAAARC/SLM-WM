"""T2SMark SD3.5 formal 复现边界的轻量测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_experiments.runners.external_baseline_method_faithful import (
    DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES,
    configured_attack_names,
)
from paper_experiments.runners.t2smark_formal_reproduction import (
    T2SMarkFormalReproductionConfig,
    should_run_official,
)


pytestmark = pytest.mark.quick


def _write_results(path: Path, *, sample_count: int, missing_attack_name: str = "") -> None:
    """写出只用于复用门禁的 T2SMark results.json。"""

    attack_names = configured_attack_names(DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES)
    payload = {
        str(index): {
            "formal_attacks": {
                attack_name: {"attack_name": attack_name}
                for attack_name in attack_names
                if attack_name != missing_attack_name
            }
        }
        for index in range(sample_count)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_t2smark_reuse_requires_every_formal_attack_for_every_prompt(tmp_path: Path) -> None:
    """只有每个 Prompt 都包含全部正式攻击时才允许复用 T2SMark 结果。"""

    results_path = tmp_path / "results.json"
    config = T2SMarkFormalReproductionConfig(prompt_limit=2, save_clean_pair=False)
    _write_results(results_path, sample_count=2)

    should_run, reason = should_run_official(config, results_path)

    assert should_run is False
    assert reason == "existing_results_found"


def test_t2smark_reuse_rejects_incomplete_formal_attack_matrix(tmp_path: Path) -> None:
    """缺少任一正式攻击时必须重新运行官方生成与检测链路。"""

    results_path = tmp_path / "results.json"
    missing_attack_name = configured_attack_names(DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES)[-1]
    config = T2SMarkFormalReproductionConfig(prompt_limit=2, save_clean_pair=False)
    _write_results(results_path, sample_count=2, missing_attack_name=missing_attack_name)

    should_run, reason = should_run_official(config, results_path)

    assert should_run is True
    assert reason == "existing_results_formal_attack_count_insufficient"
