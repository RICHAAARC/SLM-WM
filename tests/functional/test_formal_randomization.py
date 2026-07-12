"""验证主方法与 baseline 共享的正式随机化协议."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.protocol.formal_randomization import (
    build_canonical_sd35_base_latent,
    build_formal_randomization_identity,
    formal_randomization_protocol_record,
    formal_randomization_repeats,
    resolve_formal_randomization_repeat,
)
from experiments.runners.image_only_dataset_workload import build_method_config
from main.core.digest import build_stable_digest
from paper_experiments.runners.external_baseline_method_faithful import (
    build_default_config as build_baseline_config,
)
from paper_experiments.runners.t2smark_formal_reproduction import (
    build_default_config as build_t2smark_config,
)


pytestmark = pytest.mark.quick
ROOT = Path(__file__).resolve().parents[2]


def test_formal_randomization_registry_is_exact_three_by_three_cross() -> None:
    """正式注册表必须精确覆盖3个生成种子与3个密钥的笛卡尔积."""

    repeats = formal_randomization_repeats()
    protocol = formal_randomization_protocol_record()

    assert len(repeats) == 9
    assert len({repeat.randomization_repeat_id for repeat in repeats}) == 9
    assert {
        (repeat.generation_seed_index, repeat.watermark_key_index)
        for repeat in repeats
    } == {(seed_index, key_index) for seed_index in range(3) for key_index in range(3)}
    assert protocol["generation_seed_repeat_count"] == 3
    assert protocol["watermark_key_repeat_count"] == 3
    assert protocol["crossed_repeat_count"] == 9
    assert protocol["formal_randomization_protocol_digest"] == (
        "bac52313e8c4ed3e4339b65c3da897013c060d153c3c0b98de8e2e1d5bd679a0"
    )


def test_canonical_base_latent_is_byte_stable_and_seed_sensitive() -> None:
    """相同生成身份必须得到相同 Tensor 字节, 不同 seed 必须改变内容."""

    torch = pytest.importorskip("torch")
    common = {
        "shape": (1, 2, 3, 4),
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "model_revision": "a" * 40,
        "device": "cpu",
        "dtype": torch.float16,
    }
    first, first_identity = build_canonical_sd35_base_latent(
        generation_seed_random=1703,
        **common,
    )
    repeated, repeated_identity = build_canonical_sd35_base_latent(
        generation_seed_random=1703,
        **common,
    )
    different, different_identity = build_canonical_sd35_base_latent(
        generation_seed_random=1704,
        **common,
    )

    assert torch.equal(first, repeated)
    assert first_identity == repeated_identity
    assert not torch.equal(first, different)
    assert (
        first_identity["base_latent_content_digest_random"]
        == "6dced2163ba3a204de2653e49121eb647fba3cf8c76afa7418b47a4d18a7117c"
    )
    assert (
        first_identity["base_latent_content_digest_random"]
        != different_identity["base_latent_content_digest_random"]
    )


def test_main_and_all_formal_baselines_share_active_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """主方法、通用 baseline 与 T2SMark 必须消费同一活动重复."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    monkeypatch.setenv("SLM_WM_RANDOMIZATION_REPEAT_ID", "seed_02_key_01")
    monkeypatch.setenv("SLM_WM_PRIMARY_BASELINE_ID", "tree_ring")
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)
    monkeypatch.delenv("SLM_WM_EXTERNAL_BASELINE_SEED", raising=False)
    monkeypatch.delenv("SLM_WM_T2SMARK_FORMAL_SEED", raising=False)
    monkeypatch.delenv("SLM_WM_KEY_MATERIAL", raising=False)

    method = build_method_config(ROOT)
    baseline = build_baseline_config()
    t2smark = build_t2smark_config()
    repeat = resolve_formal_randomization_repeat("seed_02_key_01")
    prompt_identity = build_formal_randomization_identity(
        base_seed=1703,
        prompt_index=7,
        root_key_material="slm_wm_paper_key",
        repeat=repeat,
    )

    assert method.seed == baseline.seed == t2smark.seed == 2_001_706
    assert (
        method.randomization_repeat_id
        == baseline.randomization_repeat_id
        == t2smark.randomization_repeat_id
        == "seed_02_key_01"
    )
    assert (
        method.generation_seed_offset
        == baseline.generation_seed_offset
        == t2smark.generation_seed_offset
        == 2_000_003
    )
    assert (
        method.watermark_key_seed_random
        == baseline.watermark_key_seed_random
        == t2smark.watermark_key_seed_random
        == prompt_identity["watermark_key_seed_random"]
    )
    assert prompt_identity["generation_seed_random"] == method.seed + 7
    assert prompt_identity["watermark_key_material_digest_random"] == (
        build_stable_digest({"key_material": method.key_material})
    )


def test_sd35_adapters_construct_the_shared_canonical_base_latent() -> None:
    """所有 SD3.5 正式适配器都必须调用唯一基础 latent 构造器."""

    adapter_paths = (
        ROOT
        / "external_baseline/primary/tree_ring/adapter/method_faithful_sd35.py",
        ROOT
        / "external_baseline/primary/gaussian_shading/adapter/method_faithful_sd35.py",
        ROOT
        / "external_baseline/primary/shallow_diffuse/adapter/method_faithful_sd35.py",
        ROOT / "external_baseline/primary/t2smark/source/run_sd35.py",
    )

    for adapter_path in adapter_paths:
        source = adapter_path.read_text(encoding="utf-8")
        assert "build_canonical_sd35_base_latent" in source

    protocol_source = (
        ROOT / "experiments/protocol/formal_randomization.py"
    ).read_text(encoding="utf-8")
    assert "base_latent_content_digest_random" in protocol_source
