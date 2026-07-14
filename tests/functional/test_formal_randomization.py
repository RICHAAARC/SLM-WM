"""验证主方法与 baseline 共享的正式随机化协议."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from experiments.protocol.formal_randomization import (
    build_canonical_sd35_base_latent,
    build_formal_randomization_identity,
    build_formal_randomization_repeat_coverage,
    formal_randomization_sample_reference,
    formal_randomization_protocol_record,
    formal_randomization_repeat_ids,
    formal_randomization_repeat_registry_digest,
    formal_randomization_repeats,
    formal_watermark_key_material,
    formal_watermark_key_material_from_seed,
    formal_watermark_key_plan_record,
    formal_watermark_key_seed_random,
    formal_runtime_randomization_plan_record,
    require_formal_watermark_key_plan,
    resolve_formal_randomization_repeat,
    validate_formal_randomization_repeat_records,
    validate_formal_prompt_randomization_identity,
)
from experiments.runners.image_only_dataset_workload import build_method_config
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.runners.image_only_dataset_runtime import (
    validate_formal_dataset_randomization_identity,
)
from main.core.digest import build_stable_digest
from paper_experiments.runners.external_baseline_method_faithful import (
    baseline_run_identity_manifest_config,
    build_default_config as build_baseline_config,
)
from paper_experiments.runners.t2smark_formal_reproduction import (
    build_default_config as build_t2smark_config,
    t2smark_run_identity_manifest_config,
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
        "d03d6ef0d58f2fd02f54aa85e3cd695dbab391586d48001524a95ef62ec97630"
    )
    assert formal_randomization_repeat_ids() == tuple(
        f"seed_{seed_index:02d}_key_{key_index:02d}"
        for seed_index in range(3)
        for key_index in range(3)
    )
    assert len(formal_randomization_repeat_registry_digest()) == 64


def test_formal_randomization_exact_coverage_rejects_count_only_substitution() -> None:
    """最终覆盖必须逐身份匹配注册表, 不能只依赖9条记录计数."""

    records = [repeat.to_dict() for repeat in formal_randomization_repeats()]
    coverage = build_formal_randomization_repeat_coverage(
        reversed(records),
        require_exact_registry=True,
    )

    assert coverage["observed_repeat_ids"] == list(
        formal_randomization_repeat_ids()
    )
    assert coverage["observed_repeat_count"] == 9
    assert coverage["exact_repeat_registry_ready"] is True
    assert coverage["supports_paper_claim"] is False

    wrong_identity = [dict(record) for record in records]
    wrong_identity[-1]["watermark_key_index"] = 1
    with pytest.raises(ValueError, match="身份未匹配注册表"):
        validate_formal_randomization_repeat_records(
            wrong_identity,
            require_exact_registry=True,
        )

    duplicated = [dict(record) for record in records[:-1]]
    duplicated.append(dict(records[0]))
    with pytest.raises(ValueError, match="ID 重复"):
        validate_formal_randomization_repeat_records(
            duplicated,
            require_exact_registry=True,
        )


@pytest.mark.parametrize(
    "field_name,forged_value",
    (
        ("generation_seed_index", False),
        ("generation_seed_index", 0.0),
        ("generation_seed_offset", False),
        ("generation_seed_offset", 0.0),
        ("watermark_key_index", False),
        ("watermark_key_index", 0.0),
    ),
)
def test_formal_randomization_repeat_identity_requires_exact_integer_types(
    field_name: str,
    forged_value: object,
) -> None:
    """整数身份字段必须拒绝与0数值相等的 bool 或 float."""

    record = resolve_formal_randomization_repeat("seed_00_key_00").to_dict()
    record[field_name] = forged_value

    with pytest.raises(ValueError, match="整数身份字段类型无效"):
        validate_formal_randomization_repeat_records(
            [record],
            require_exact_registry=False,
        )


def test_single_repeat_component_coverage_never_supports_paper_claim() -> None:
    """单个 GPU repeat 只能形成 component, 不能冒充最终论文聚合."""

    repeat = resolve_formal_randomization_repeat("seed_01_key_02")
    coverage = build_formal_randomization_repeat_coverage(
        [repeat.to_dict()],
        require_exact_registry=False,
    )

    assert coverage["observed_repeat_ids"] == ["seed_01_key_02"]
    assert coverage["exact_repeat_registry_ready"] is False
    assert coverage["supports_paper_claim"] is False

    with pytest.raises(ValueError, match="精确声明一个"):
        build_formal_randomization_repeat_coverage(
            [
                resolve_formal_randomization_repeat("seed_00_key_00").to_dict(),
                repeat.to_dict(),
            ],
            require_exact_registry=False,
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
        == "e1ccefcb4a091a109ecf095e822ca73bed97fc13e788f0601e12927e0d0fea40"
    )
    assert (
        first_identity["base_latent_content_digest_random"]
        != different_identity["base_latent_content_digest_random"]
    )


def test_formal_sd35_shape_base_latent_has_frozen_cpu_golden_identity() -> None:
    """正式 SD3.5 shape 的 CPU float16 字节身份必须完整冻结."""

    torch = pytest.importorskip("torch")
    latent, identity = build_canonical_sd35_base_latent(
        shape=(1, 16, 64, 64),
        generation_seed_random=1703,
        model_id="stabilityai/stable-diffusion-3.5-medium",
        model_revision="b940f670f0eda2d07fbb75229e779da1ad11eb80",
        device="cpu",
        dtype=torch.float16,
    )

    assert latent.shape == (1, 16, 64, 64)
    assert latent.dtype == torch.float16
    assert identity["base_latent_generation_protocol"] == (
        "device_independent_sha256_normal_icdf_table20_"
        "cpu_dtype_cast_then_device_transfer"
    )
    assert identity["base_latent_keyed_prg_protocol_digest"] == (
        "e1f97fd7457893cf4d92c0ffa383b44219cf6b1034055e43dcadf1d535ab1595"
    )
    assert identity["base_latent_content_digest_random"] == (
        "b665be1585924e50b868c98fbf7a6bfd9170745b8298d1203f979b36926633b8"
    )
    assert identity["base_latent_identity_digest_random"] == (
        "b8d6cdb11e814f3c098f6a567aefd39a2b2b29da79b85a50f2894889473dfa58"
    )


def test_formal_key_material_can_be_rebuilt_from_governed_key_seed() -> None:
    """审计端必须能在不读取根密钥时重建正式 key material 摘要."""

    repeat = resolve_formal_randomization_repeat("seed_02_key_01")
    root_key_material = "slm_wm_paper_key"
    key_seed = formal_watermark_key_seed_random(root_key_material, repeat)

    rebuilt = formal_watermark_key_material_from_seed(key_seed, repeat)

    assert rebuilt == formal_watermark_key_material(
        root_key_material,
        repeat,
    )
    assert build_stable_digest({"key_material": rebuilt}) == (
        build_formal_randomization_identity(
            base_seed=1703,
            prompt_index=7,
            root_key_material=root_key_material,
            repeat=repeat,
        )["watermark_key_material_digest_random"]
    )


def test_formal_prompt_identity_is_rebuilt_from_frozen_formula() -> None:
    """正式入口必须从 base seed、Prompt 索引和注册 key 重建实际身份."""

    repeat = resolve_formal_randomization_repeat("seed_01_key_02")
    identity = build_formal_randomization_identity(
        base_seed=1703,
        prompt_index=11,
        root_key_material="slm_wm_paper_key",
        repeat=repeat,
    )
    key_material = formal_watermark_key_material(
        "slm_wm_paper_key",
        repeat,
    )

    rebuilt = validate_formal_prompt_randomization_identity(
        base_generation_seed_random=1703,
        prompt_index=11,
        randomization_repeat_id=repeat.randomization_repeat_id,
        generation_seed_index=repeat.generation_seed_index,
        generation_seed_offset=repeat.generation_seed_offset,
        watermark_key_index=repeat.watermark_key_index,
        generation_seed_random=identity["generation_seed_random"],
        watermark_key_seed_random=identity["watermark_key_seed_random"],
        key_material=key_material,
        formal_randomization_protocol_digest=identity[
            "formal_randomization_protocol_digest"
        ],
    )

    assert rebuilt == identity


@pytest.mark.parametrize(
    "drifted_field",
    (
        "generation_seed_random",
        "watermark_key_seed_random",
        "key_material",
        "formal_randomization_protocol_digest",
    ),
)
def test_formal_prompt_identity_rejects_declared_runtime_drift(
    drifted_field: str,
) -> None:
    """实际 seed、key 或协议与顶层声明分叉时必须失败关闭."""

    repeat = resolve_formal_randomization_repeat("seed_02_key_01")
    identity = build_formal_randomization_identity(
        base_seed=1703,
        prompt_index=5,
        root_key_material="slm_wm_paper_key",
        repeat=repeat,
    )
    arguments = {
        "base_generation_seed_random": 1703,
        "prompt_index": 5,
        "randomization_repeat_id": repeat.randomization_repeat_id,
        "generation_seed_index": repeat.generation_seed_index,
        "generation_seed_offset": repeat.generation_seed_offset,
        "watermark_key_index": repeat.watermark_key_index,
        "generation_seed_random": identity["generation_seed_random"],
        "watermark_key_seed_random": identity["watermark_key_seed_random"],
        "key_material": formal_watermark_key_material(
            "slm_wm_paper_key",
            repeat,
        ),
        "formal_randomization_protocol_digest": identity[
            "formal_randomization_protocol_digest"
        ],
    }
    if drifted_field in {"generation_seed_random", "watermark_key_seed_random"}:
        arguments[drifted_field] = int(arguments[drifted_field]) + 1
    else:
        arguments[drifted_field] = "drifted"

    with pytest.raises(ValueError):
        validate_formal_prompt_randomization_identity(**arguments)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("seed", 999_001),
        ("watermark_key_seed_random", 999_002),
        ("key_material", "unregistered-key-material"),
        ("randomization_repeat_id", "seed_00_key_01"),
    ),
)
def test_formal_dataset_entry_rejects_actual_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    replacement: object,
) -> None:
    """正式数据集入口必须核对实际运行配置, 不能只信任顶层声明."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    monkeypatch.setenv(
        "SLM_WM_RANDOMIZATION_REPEAT_ID",
        "seed_01_key_02",
    )
    monkeypatch.delenv("SLM_WM_KEY_MATERIAL", raising=False)
    config = build_method_config(ROOT)
    paper_run = build_paper_run_config(ROOT)

    identity = validate_formal_dataset_randomization_identity(
        config,
        paper_run,
        prompt_index=0,
    )
    assert identity["generation_seed_random"] == config.seed

    with pytest.raises(ValueError):
        drifted = replace(config, **{field_name: replacement})
        validate_formal_dataset_randomization_identity(
            drifted,
            paper_run,
            prompt_index=0,
        )


def test_runtime_plan_keeps_full_body_while_sample_reference_is_compact() -> None:
    """顶层 manifest 保存完整计划, 样本只保留公平配对所需引用."""

    repeat = resolve_formal_randomization_repeat("seed_00_key_02")
    identity = build_formal_randomization_identity(
        base_seed=1703,
        prompt_index=3,
        root_key_material="slm_wm_paper_key",
        repeat=repeat,
    )
    plan = formal_runtime_randomization_plan_record(1703)
    reference = formal_randomization_sample_reference(identity)

    assert len(plan["repeat_records"]) == 9
    assert len(plan["watermark_key_records"]) == 3
    assert plan["base_generation_seed_random"] == 1703
    assert "repeat_records" not in reference
    assert reference["formal_randomization_protocol_digest"] == identity[
        "formal_randomization_protocol_digest"
    ]
    assert reference["formal_randomization_identity_digest_random"] == identity[
        "formal_randomization_identity_digest_random"
    ]


def test_formal_watermark_key_plan_is_preregistered_and_root_bound() -> None:
    """正式3-key 计划必须在运行前冻结并拒绝结果后选择根密钥."""

    plan = require_formal_watermark_key_plan("slm_wm_paper_key")

    assert plan == formal_watermark_key_plan_record()
    assert len(plan["watermark_key_records"]) == 3
    assert len(plan["formal_watermark_key_plan_digest"]) == 64
    with pytest.raises(ValueError, match="预注册正式 key plan"):
        require_formal_watermark_key_plan("post_selected_alternate_key")
    with pytest.raises(ValueError, match="预注册正式 key plan"):
        formal_watermark_key_material(
            "post_selected_alternate_key",
            resolve_formal_randomization_repeat("seed_00_key_00"),
        )


@pytest.mark.parametrize(
    "invalid_seed",
    (False, -1, 1 << 63),
)
def test_formal_key_material_rebuild_rejects_noncanonical_key_seed(
    invalid_seed: object,
) -> None:
    """重建入口必须拒绝 bool、负数和超出正式63位域的 seed."""

    repeat = resolve_formal_randomization_repeat("seed_00_key_00")

    with pytest.raises(ValueError, match="非负63位整数"):
        formal_watermark_key_material_from_seed(invalid_seed, repeat)


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
    expected_plan = formal_runtime_randomization_plan_record(1703)
    assert baseline_run_identity_manifest_config(baseline)[
        "formal_randomization_plan"
    ] == expected_plan
    assert t2smark_run_identity_manifest_config(t2smark)[
        "formal_randomization_plan"
    ] == expected_plan


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
