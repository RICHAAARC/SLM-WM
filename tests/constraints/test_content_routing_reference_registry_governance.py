from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import pytest


pytestmark = pytest.mark.constraint

ROOT = Path(__file__).resolve().parents[2]
METHOD_DOCUMENT = ROOT / "docs/builds/method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md"
ALGORITHM_DOCUMENT = ROOT / "docs/builds/algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md"
STATE_DOCUMENT = ROOT / "docs/builds/project_construction_state.md"
FIELD_REGISTRY = ROOT / "docs/field_registry.md"
FIELD_LIFECYCLE_REGISTRY = ROOT / "configs/field_lifecycle_registry.json"
REFERENCE_REGISTRY = ROOT / "configs/content_routing_reference_registry.json"
CONTRACT_NAME = "CONTENT_ROUTING_REFERENCE_REGISTRY_MACHINE_CONTRACT"
EXPECTED_CONTRACT_SHA256 = (
    "fd678f75405e1fb093e4f4ada90e329e5843e24d11bdfe06073ce5bc0745efe5"
)


def _machine_contract() -> dict[str, Any]:
    text = METHOD_DOCUMENT.read_text(encoding="utf-8-sig")
    values: list[dict[str, Any]] = []
    for code_block in re.findall(r"```python\n(.*?)\n```", text, flags=re.DOTALL):
        syntax_tree = ast.parse(code_block)
        for statement in syntax_tree.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id == CONTRACT_NAME:
                value = ast.literal_eval(statement.value)
                assert type(value) is dict
                values.append(value)
    assert len(values) == 1, "content-routing reference机器契约必须只有一个权威AST literal"
    return values[0]


def _stable_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _target_field_rows() -> dict[str, dict[str, Any]]:
    text = FIELD_REGISTRY.read_text(encoding="utf-8-sig")
    section = text.split("## 冻结目标方法新增字段", maxsplit=1)[1].split(
        "## 字段登记表",
        maxsplit=1,
    )[0]
    rows: dict[str, dict[str, Any]] = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) != 7 or cells[0] in {"field_name", "---"}:
            continue
        field_name = cells[0]
        assert field_name not in rows
        rows[field_name] = {
            "category": cells[1],
            "required_suffix": cells[2],
            "allowed_in_records": cells[3] == "true",
            "allowed_in_claims": cells[4] == "true",
            "replacement_required": cells[5] == "true",
            "description": cells[6],
        }
    return rows


def _assert_rule_predicates_are_governed(
    rules: dict[str, dict[str, Any]],
    predicates: set[str],
) -> None:
    assert rules
    for rule in rules.values():
        assert type(rule) is dict
        assert rule["predicate"] in predicates


def test_content_routing_reference_registry_machine_contract() -> None:
    contract = _machine_contract()
    assert _stable_digest(contract) == EXPECTED_CONTRACT_SHA256

    predicates = set(contract["type_predicates"])
    top_rules = contract["registry_top_level_field_rules"]
    population_rules = contract["population_field_rules"]
    member_rules = contract["member_record_field_rules"]
    _assert_rule_predicates_are_governed(top_rules, predicates)
    _assert_rule_predicates_are_governed(population_rules, predicates)
    _assert_rule_predicates_are_governed(member_rules, predicates)

    prompt_projection = contract["prompt_projection_contract"]
    seed_projection = contract["seed_projection_contract"]
    assert prompt_projection["container_predicate"] in predicates
    assert prompt_projection["entry_predicate"] in predicates
    assert seed_projection["container_predicate"] in predicates
    assert seed_projection["element_predicate"] in predicates
    _assert_rule_predicates_are_governed(
        prompt_projection["entry_field_rules"],
        predicates,
    )
    invariants = set(contract["cross_field_invariants"])
    for projection in (prompt_projection, seed_projection):
        assert projection["order_source"]
        assert projection["order_rule"] in invariants
        assert projection["length_rule"] in invariants
        assert projection["digest_rule"].startswith("build_stable_digest(")
    assert prompt_projection["order_source"] == seed_projection["order_source"]

    assert "model_identity_digest" not in top_rules
    assert top_rules["method_parameter_prompt_list_digest"][
        "digest_contract"
    ] == "prompt_projection_contract"
    assert top_rules["method_parameter_seed_list_digest_random"][
        "digest_contract"
    ] == "seed_projection_contract"
    assert top_rules["runtime_component_identity_digest"][
        "digest_contract"
    ] == "runtime_component_identity_payload_contract"
    assert population_rules["reference_observation_member_records_digest"][
        "digest_rule"
    ].startswith("build_stable_digest(")
    assert top_rules["registry_schema"]["exact_value"] == contract[
        "registry_schema_token"
    ]
    assert top_rules["content_routing_reference_quantile_algorithm"][
        "exact_value"
    ] == contract["quantile_algorithm_token"]
    assert top_rules["content_routing_reference_quantile_rank_rule"][
        "exact_value"
    ] == contract["quantile_rank_rule_token"]
    assert top_rules["content_routing_reference_quantile_index_rule"][
        "exact_value"
    ] == contract["quantile_index_rule_token"]

    population_order = tuple(contract["population_order"])
    binding = contract["population_scalar_binding"]
    assert tuple(binding) == population_order
    scalar_fields = [record["scalar_field"] for record in binding.values()]
    binary32_fields = [
        record["binary32_hex_field"] for record in binding.values()
    ]
    assert len(scalar_fields) == len(set(scalar_fields)) == len(population_order)
    assert len(binary32_fields) == len(set(binary32_fields)) == len(
        population_order
    )
    assert set(scalar_fields) | set(binary32_fields) <= set(top_rules)

    runtime_contract = contract["runtime_component_identity_payload_contract"]
    assert runtime_contract["digest_rule"].startswith("build_stable_digest(")
    runtime_rules = runtime_contract["field_rules"]
    _assert_rule_predicates_are_governed(runtime_rules, predicates)
    probe_contract = runtime_contract["public_probe_identity_contract"]
    _assert_rule_predicates_are_governed(
        probe_contract["field_rules"],
        predicates,
    )
    _assert_rule_predicates_are_governed(
        probe_contract["domain_field_rules"],
        predicates,
    )
    assert set(runtime_contract["forbidden_fields"]).isdisjoint(runtime_rules)
    assert runtime_rules["model_id"]["exact_value"] == top_rules["model_id"][
        "exact_value"
    ]
    assert runtime_rules["model_revision"]["exact_value"] == top_rules[
        "model_revision"
    ]["exact_value"]

    rows = contract["field_registry_row_contract"]
    exceptions = tuple(contract["schema_freeze_exception_fields"])
    assert len(exceptions) == len(set(exceptions)) == 21
    assert set(rows) == set(exceptions)
    assert all(record["allowed_in_claims"] is False for record in rows.values())
    assert all(
        record["replacement_required"] is False for record in rows.values()
    )
    assert sum(record["allowed_in_records"] is True for record in rows.values()) == 1
    assert len(
        {record["description_semantics_token"] for record in rows.values()}
    ) == len(rows)


def test_content_routing_reference_registry_identity_matches_governed_sd35() -> None:
    contract = _machine_contract()
    top_rules = contract["registry_top_level_field_rules"]
    runtime_rules = contract["runtime_component_identity_payload_contract"][
        "field_rules"
    ]
    config_values = {
        key.strip(): value.strip()
        for line in (ROOT / "configs/model_sd35.yaml").read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line and not line.lstrip().startswith("#") and ": " in line
        for key, value in [line.split(": ", maxsplit=1)]
    }
    for field_name in (
        "model_id",
        "model_revision",
        "pipeline_class_name",
        "vae_class_name",
        "transformer_class_name",
        "scheduler_class_name",
        "latent_torch_dtype",
    ):
        assert runtime_rules[field_name]["exact_value"] == config_values[field_name]
    assert top_rules["model_id"]["exact_value"] == config_values["model_id"]
    assert top_rules["model_revision"]["exact_value"] == config_values[
        "model_revision"
    ]
    assert runtime_rules["vae_scaling_factor"]["exact_value"] == float(
        config_values["vae_scaling_factor"]
    )
    assert runtime_rules["vae_shift_factor"]["exact_value"] == float(
        config_values["vae_shift_factor"]
    )

    source_registry = json.loads(
        (ROOT / "configs/model_source_registry.json").read_text(
            encoding="utf-8-sig"
        )
    )
    source_record = source_registry["sources"][
        "stabilityai_stable_diffusion_3_5_medium"
    ]
    assert source_record["repository_id"] == top_rules["model_id"]["exact_value"]
    assert source_record["revision"] == top_rules["model_revision"]["exact_value"]

    algorithm_text = ALGORITHM_DOCUMENT.read_text(encoding="utf-8-sig")
    frozen_pair = (
        f"{top_rules['model_id']['exact_value']}@"
        f"{top_rules['model_revision']['exact_value']}"
    )
    assert frozen_pair in algorithm_text


def test_content_routing_reference_registry_fields_match_one_narrow_exception() -> None:
    contract = _machine_contract()
    row_contract = contract["field_registry_row_contract"]
    exceptions = set(contract["schema_freeze_exception_fields"])
    registry_rows = _target_field_rows()
    lifecycle = json.loads(FIELD_LIFECYCLE_REGISTRY.read_text(encoding="utf-8-sig"))

    assert set(row_contract) == exceptions
    assert exceptions <= set(lifecycle["target_required_exact_fields"])
    assert exceptions.isdisjoint(lifecycle["legacy_only_exact_fields"])
    assert set(lifecycle) == {
        "active_shared_default",
        "target_required_exact_fields",
        "legacy_only_contains",
        "legacy_only_exact_fields",
        "legacy_only_prefixes",
        "registry_schema",
        "resolution_precedence",
        "target_required_sources",
    }
    assert not any("exception" in key for key in lifecycle)

    for field_name, expected in row_contract.items():
        actual = registry_rows[field_name]
        assert actual["category"] == expected["category"]
        assert actual["required_suffix"] == expected["required_suffix"]
        assert actual["allowed_in_records"] is expected["allowed_in_records"]
        assert actual["allowed_in_claims"] is expected["allowed_in_claims"]
        assert actual["replacement_required"] is expected[
            "replacement_required"
        ]
        assert f"`{expected['description_semantics_token']}`" in actual[
            "description"
        ]

    field_registry_text = FIELD_REGISTRY.read_text(encoding="utf-8-sig")
    assert (
        'CONTENT_ROUTING_REFERENCE_REGISTRY_MACHINE_CONTRACT["schema_freeze_exception_fields"]'
        in field_registry_text
    )
    assert "不得从 `governance`、`schema`、字段类别、命名或自然语言推导其他预登记例外" in field_registry_text
    assert "除该精确21项外，runtime/record 字段与 writer、consumer、aggregator 同一原子迁移的门禁原样有效" in field_registry_text

    registered_names = re.findall(
        r"^\| ([A-Za-z0-9_]+) \|",
        field_registry_text,
        flags=re.MULTILINE,
    )
    registered_names = [
        name for name in registered_names if name not in {"field", "field_name"}
    ]
    assert len(registered_names) == len(set(registered_names))


def test_content_routing_reference_registry_remains_unmaterialized() -> None:
    assert not REFERENCE_REGISTRY.exists()

    method_text = METHOD_DOCUMENT.read_text(encoding="utf-8-sig")
    assert "candidate 始终 `supports_paper_claim=false`" in method_text
    assert "本原子不生成该文件" in method_text
    assert "不实现 validator、loader、producer、writer、qualification、promotion 或真实 registry" in method_text

    state_text = STATE_DOCUMENT.read_text(encoding="utf-8-sig")
    assert "| 路由 reference registry | `not_materialized` |" in state_text
    assert "不得提升为 `document_ecosystem_synchronized`" in state_text
    assert "机器随机化、攻击职责 writer" in state_text

    implementation_pattern = re.compile(
        r"^def (?:load|validate|build|write)_content_routing_reference_registry\(",
        flags=re.MULTILINE,
    )
    implementation_paths = []
    for root_name in ("main", "experiments", "paper_experiments", "scripts"):
        for path in (ROOT / root_name).rglob("*.py"):
            if implementation_pattern.search(path.read_text(encoding="utf-8-sig")):
                implementation_paths.append(path.relative_to(ROOT).as_posix())
    assert implementation_paths == []


def test_content_routing_reference_registry_digest_and_promotion_are_one_way() -> None:
    contract = _machine_contract()
    top_rules = contract["registry_top_level_field_rules"]
    assert "content_routing_reference_registry_file_sha256" not in top_rules
    assert "method_definition_digest" not in top_rules
    assert "runtime_config_digest" not in top_rules
    assert "model_identity_digest" not in top_rules

    method_text = METHOD_DOCUMENT.read_text(encoding="utf-8-sig")
    assert contract["semantic_digest_rule"] in method_text
    assert contract["file_sha256_rule"] in method_text
    assert contract["binary32_hex_rule"] in method_text
    assert "完全相同字节复制到固定路径 `configs/content_routing_reference_registry.json`" in method_text
    assert "loader 不接受路径、默认值或 fallback" in method_text
    assert "先比较 exact file SHA" in method_text
    assert "复算 semantic digest 并先对 embedded、再对 expected" in method_text
