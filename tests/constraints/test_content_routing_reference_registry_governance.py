from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
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
    "8b301ae238146fecb27903a2d9bb88e6a6cd48492bcc84b9d7fd2350a3de621b"
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


def _iter_rule_maps(value: Any) -> list[dict[str, dict[str, Any]]]:
    rule_maps: list[dict[str, dict[str, Any]]] = []
    if type(value) is dict:
        for key, nested in value.items():
            if key == "field_rules" or key.endswith("_field_rules"):
                assert type(nested) is dict
                rule_maps.append(nested)
            else:
                rule_maps.extend(_iter_rule_maps(nested))
    elif type(value) in {list, tuple}:
        for nested in value:
            rule_maps.extend(_iter_rule_maps(nested))
    return rule_maps


def _assert_materialization_nested_contracts_are_closed(
    materialization: dict[str, Any],
) -> None:
    for contract_name, nested_contract in materialization.items():
        if type(nested_contract) is not dict or "field_rules" not in nested_contract:
            continue
        field_rules = nested_contract["field_rules"]
        container_fields = {
            field_name
            for field_name, rule in field_rules.items()
            if rule["predicate"] in {"exact_object", "exact_list"}
        }
        nested_bindings = nested_contract.get("nested_contracts", {})
        assert type(nested_bindings) is dict, contract_name
        assert set(nested_bindings) == container_fields, contract_name
        for field_name, target_name in nested_bindings.items():
            assert type(target_name) is str and target_name
            if target_name in materialization:
                target = materialization[target_name]
                assert type(target) is dict
                assert "field_rules" in target
            else:
                assert target_name in nested_contract, (contract_name, field_name)
                target = nested_contract[target_name]
                assert type(target) is dict and target


def _assert_exact_flag_tokens(
    tokens: Any,
    expected_names: tuple[str, ...],
) -> None:
    assert type(tokens) is tuple
    assert len(tokens) == len(set(tokens))
    assert set(tokens) == set(expected_names)
    allowed_flags = {
        name: getattr(os, name)
        for name in (
            "O_RDONLY",
            "O_WRONLY",
            "O_CREAT",
            "O_EXCL",
            "O_DIRECTORY",
            "O_CLOEXEC",
            "O_NOFOLLOW",
            "O_NONBLOCK",
        )
    }
    assert set(tokens) <= set(allowed_flags)
    captured_flags = 0
    expected_flags = 0
    for name in tokens:
        captured_flags |= allowed_flags[name]
    for name in expected_names:
        expected_flags |= allowed_flags[name]
    assert captured_flags == expected_flags
    if "O_RDONLY" in expected_names:
        assert captured_flags & os.O_ACCMODE == os.O_RDONLY


def _assert_ordered_fragments(values: Any, fragments: tuple[str, ...]) -> None:
    assert type(values) is tuple
    assert len(values) == len(fragments)
    assert len(values) == len(set(values))
    positions: list[int] = []
    for fragment in fragments:
        matches = [index for index, value in enumerate(values) if fragment in value]
        assert len(matches) == 1, fragment
        positions.append(matches[0])
    assert positions == list(range(len(values)))


def _assert_writer_contract_is_closed(materialization: dict[str, Any]) -> None:
    writer = materialization["writer_file_dag_contract"]
    api = writer["public_api_contract"]
    assert tuple(api["keyword_only_parameters"]) == tuple(api["input_contracts"])
    assert set(api["keyword_only_parameters"]).isdisjoint(
        api["forbidden_parameters"]
    )
    assert "qualification_report" not in " ".join(api["keyword_only_parameters"])
    assert "only_writer_symbol" in api["public_export_rule"]

    pre_mutation = writer["pre_filesystem_mutation_contract"]
    _assert_ordered_fragments(
        pre_mutation["ordered_steps"],
        (
            "machine_contract",
            "input_containers",
            "manifest_bytes",
            "candidate_directory_identity",
            "raw_member_dict",
            "raw_member_bytes",
            "candidate_registry",
            "manifest_registry",
        ),
    )
    assert "before_repository_root_open_or_mutation" in pre_mutation[
        "first_filesystem_mutation_rule"
    ]
    assert "zero_candidate_filesystem_mutation" in pre_mutation["failure_rule"]

    root = writer["repository_root_contract"]
    future_writer_path = root["future_writer_module_path"]
    assert future_writer_path.endswith(".py")
    assert api["module"] == future_writer_path.removesuffix(".py").replace("/", ".")
    assert api["symbol"] == (
        "write_content_routing_reference_materialization_candidate"
    )
    assert root["future_writer_module_path"] == (
        "experiments/protocol/content_routing_reference_candidate_writer.py"
    )
    assert root["derivation_expression"] == "Path(__file__).resolve().parents[2]"
    assert root["resolve_allowed_scope"] == "trusted_loaded_module_file_only"
    assert root["resolve_for_candidate_paths_is_forbidden"] is True
    assert root["candidate_path_operations_are_single_component_dirfd_relative"] is True
    assert root["path_precheck_and_follow_are_forbidden"] is True
    _assert_exact_flag_tokens(
        root["root_open_flags"],
        ("O_RDONLY", "O_DIRECTORY", "O_CLOEXEC", "O_NOFOLLOW"),
    )
    _assert_ordered_fragments(root["root_open_sequence"], ("open", "fstat", "require"))

    directories = writer["directory_contract"]
    assert tuple(directories["shared_parent_components"]) == tuple(
        writer["persistent_root"].split("/")
    )
    assert directories["shared_parent_missing_policy"].startswith("create")
    assert directories["created_directory_mode"] == 0o700
    _assert_ordered_fragments(
        directories["created_directory_sequence"],
        ("mkdir", "open", "fchmod", "fstat", "fsync_new", "fsync_parent"),
    )
    _assert_ordered_fragments(
        directories["existing_shared_parent_sequence"],
        ("open", "fstat", "do_not_chmod"),
    )
    assert "without_ownership_or_chmod" in directories["mkdir_eexist_rule"]
    assert directories["process_umask_read_or_mutation_is_forbidden"] is True
    assert directories["raw_directory_component"] == "raw"
    assert directories["candidate_subdirectory_sources"][0] == (
        "directory_contract.raw_directory_component"
    )
    _assert_exact_flag_tokens(
        directories["open_flags"],
        ("O_RDONLY", "O_DIRECTORY", "O_CLOEXEC", "O_NOFOLLOW"),
    )

    existing = writer["existing_candidate_read_contract"]
    _assert_exact_flag_tokens(
        existing["file_open_flags"],
        ("O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK"),
    )
    _assert_ordered_fragments(
        existing["file_open_sequence"],
        ("openat", "fstat", "require_regular", "read_all", "close"),
    )
    assert existing["directory_enumeration_rule"] == (
        "os.listdir(opened_directory_fd)_only"
    )
    assert existing[
        "path_stat_exists_is_file_resolve_second_open_glob_and_path_walk_are_forbidden"
    ] is True
    assert type(existing["read_chunk_bytes"]) is int
    assert existing["read_chunk_bytes"] > 0
    read_contract = existing["read_all_contract"]
    read_rules = tuple(read_contract.values())
    assert len(read_rules) == len(set(read_rules))
    read_contract_tokens = tuple(
        f"{key} {value}" for key, value in read_contract.items()
    )
    for fragment in (
        "retry_os_read",
        "short",
        "eof",
        "fail_closed",
        "zero_reads",
        "single_close_no_retry",
        "exception_group_primary_then_close",
    ):
        assert any(fragment in token for token in read_contract_tokens)
    assert existing["writer_owned_file_mode"] == 0o600

    registry_filename = materialization["candidate_registry_file_record_contract"][
        "field_rules"
    ]["path"]["exact_value"]
    manifest_filename = materialization["materialization_manifest_contract"][
        "filename"
    ]
    report_filename = materialization["qualification_report_contract"]["filename"]
    assert tuple(existing["candidate_root_exact_required_entry_sources"]) == (
        "directory_contract.raw_directory_component",
        "candidate_registry_file_record_contract.field_rules.path.exact_value",
        "materialization_manifest_contract.filename",
    )
    assert existing["candidate_root_optional_entry_source"] == (
        "qualification_report_contract.filename"
    )

    temp = writer["temp_file_contract"]
    assert temp["name_template"] == ".{final_component}.tmp"
    for final_component in (
        registry_filename,
        manifest_filename,
        report_filename,
    ):
        assert temp["name_template"].format(
            final_component=final_component
        ) == f".{final_component}.tmp"
    assert temp["created_file_mode"] == 0o600
    _assert_ordered_fragments(
        temp["open_sequence"],
        ("openat", "fchmod", "fstat", "allow_first_write"),
    )
    assert set(temp["random_name_sources_are_forbidden"]) >= {
        "nonce",
        "pid",
        "time",
        "random",
        "caller_supplied_temp_name",
    }
    assert temp["glob_or_prefix_cleanup_is_forbidden"] is True
    _assert_exact_flag_tokens(
        temp["open_flags"],
        ("O_WRONLY", "O_CREAT", "O_EXCL", "O_CLOEXEC", "O_NOFOLLOW"),
    )

    write_rules = tuple(writer["write_all_contract"].values())
    for fragment in ("retry_os_write", "positive_written_count", "fail_closed"):
        assert any(fragment in rule for rule in write_rules)
    publish = writer["durable_publish_contract"]
    _assert_ordered_fragments(
        publish["per_file_sequence"],
        ("write_all", "fsync_temp", "close", "os_replace", "fsync_containing"),
    )
    assert tuple(publish["publication_order"]) == (
        "ordered_raw_member_files",
        "candidate_registry",
        "materialization_manifest",
    )
    assert publish["qualification_report_writer_is_excluded"] is True
    assert "not_single_atomic_directory_replace" in publish["atomicity_claim"]

    states = writer["candidate_state_contract"]
    _assert_ordered_fragments(
        states["exclusive_states"],
        (
            "new_candidate",
            "complete_byte_identical",
            "partial_candidate",
            "collision",
            "nonregular_or_unknown",
        ),
    )
    assert states["mkdir_winner_is_only_writer"] is True
    assert states["existing_candidate_branch_is_read_only"] is True
    assert "bytes_equal_inputs" in states["idempotent_success_rule"]
    assert "without_overwrite" in states["collision_rule"]
    assert "without_wait_resume_takeover_or_cleanup" in states["partial_rule"]
    assert "partial_candidate" in existing["unknown_or_temp_entry_rule"]
    assert states["overwrite_and_fallback_are_forbidden"] is True

    cleanup = writer["owned_cleanup_contract"]
    assert len(cleanup["owned_entry_sources"]) == len(
        set(cleanup["owned_entry_sources"])
    )
    assert "reverse_exact_creation_order" in cleanup["cleanup_order"]
    assert "only_exact_owned_names" in cleanup["file_cleanup_rule"]
    assert "only_owned_and_current_empty_directories" in cleanup[
        "directory_cleanup_rule"
    ]
    assert cleanup["preexisting_unknown_and_other_candidate_deletion_is_forbidden"] is True
    assert cleanup["recursive_glob_prefix_and_age_based_cleanup_are_forbidden"] is True
    assert cleanup["operator_reclamation_requires_separate_governed_protocol"] is True

    ordered_steps = tuple(writer["ordered_steps"])
    assert "qualification_report_last" in ordered_steps[-1]
    assert writer["manifest_binds_only_qualification_report_path"] is True
    assert writer["qualification_report_is_last_candidate_file"] is True
    assert writer["partial_candidate_supports_paper_claim"] is False
    assert writer["configs_write_is_forbidden"] is True


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

    materialization = contract["materialization_contract"]
    extension_predicates = set(materialization["type_predicate_extensions"])
    assert extension_predicates
    assert extension_predicates <= predicates
    for rule_map in _iter_rule_maps(materialization):
        _assert_rule_predicates_are_governed(rule_map, predicates)
    _assert_materialization_nested_contracts_are_closed(materialization)

    generation_contract = materialization[
        "generation_input_identity_payload_contract"
    ]
    generation_rules = generation_contract["field_rules"]
    assert generation_rules
    assert generation_contract["prompt_text_digest_rule"].startswith(
        "build_stable_digest("
    )
    assert generation_contract["digest_rule"].startswith("build_stable_digest(")

    base_latent = materialization["base_latent_identity_reconstruction_contract"]
    assert type(base_latent["argument_sources"]) is dict
    assert base_latent["argument_sources"]
    assert len(base_latent["returned_identity_fields"]) == len(
        set(base_latent["returned_identity_fields"])
    )
    base_invariants = set(base_latent["cross_field_invariants"])
    assert base_invariants

    generation_records = materialization[
        "ordered_generation_input_record_contract"
    ]
    assert generation_records["container_predicate"] == "exact_list"
    assert generation_records["field_rules"]

    raw_record = materialization["raw_member_file_record_contract"]
    assert "big_endian" in raw_record["encoding_token"]
    assert "times_four" in raw_record["file_length_rule"]
    assert raw_record["tensor_digest_rule"].startswith("tensor_content_sha256(")
    assert tuple(raw_record["path_templates"]) == tuple(contract["population_order"])
    assert all(
        path.endswith("/{sequence_index:08d}.f32be")
        for path in raw_record["path_templates"].values()
    )

    raw_populations = materialization["raw_member_population_contract"]
    assert raw_populations["container_predicate"] == "exact_list"
    assert raw_populations["exact_length"] == len(contract["population_order"])
    assert tuple(raw_populations["registry_member_projection_fields"]) == tuple(
        member_rules
    )

    registry_file = materialization["candidate_registry_file_record_contract"]
    assert registry_file["field_rules"]["path"]["exact_value"] == (
        "content_routing_reference_registry.json"
    )

    manifest = materialization["materialization_manifest_contract"]
    manifest_rules = manifest["field_rules"]
    assert manifest["filename"] == "materialization_manifest.json"
    assert manifest_rules["qualification_report_path"]["exact_value"] == (
        "qualification_report.json"
    )
    assert set(manifest["forbidden_fields"]).isdisjoint(manifest_rules)
    assert "qualification_report" not in manifest["semantic_digest_rule"]
    assert manifest["file_sha256_location"] == (
        "qualification_report_and_external_promotion_input_only"
    )

    qualification = materialization["qualification_report_contract"]
    qualification_rules = qualification["field_rules"]
    assert qualification["filename"] == "qualification_report.json"
    check_order = tuple(qualification["check_order"])
    assert len(check_order) == len(set(check_order))
    assert qualification_rules["qualification_check_count"]["exact_value"] == len(
        check_order
    )
    assert qualification_rules["qualification_checks"]["exact_length"] == len(
        check_order
    )
    assert "cuda_execution_environment" in check_order
    assert qualification["check_entry_field_rules"]["check_status"][
        "allowed_tokens"
    ] == ("pass", "blocked")
    assert qualification["cuda_execution_environment_contract"][
        "actual_pipeline_execution_device_type"
    ] == "cuda"
    assert qualification["cuda_execution_environment_contract"][
        "missing_cuda_failure_code"
    ] == "cuda_execution_environment_unavailable"
    assert qualification["ready_rule"]

    writer = materialization["writer_file_dag_contract"]
    assert writer
    _assert_writer_contract_is_closed(materialization)

    promotion = materialization["promotion_contract"]
    assert promotion["destination_path"] == (
        "configs/content_routing_reference_registry.json"
    )
    assert promotion["copy_rule"] == (
        "byte_identical_copy_without_recomputation_or_reserialization"
    )

    rows = contract["field_registry_row_contract"]
    exceptions = tuple(contract["schema_freeze_exception_fields"])
    assert len(exceptions) == len(set(exceptions))
    assert set(rows) == set(exceptions)
    assert all(record["allowed_in_claims"] is False for record in rows.values())
    assert all(
        record["replacement_required"] is False for record in rows.values()
    )
    assert len(
        {record["description_semantics_token"] for record in rows.values()}
    ) == len(rows)


def test_materialization_nested_contract_mutations_fail_closed() -> None:
    materialization = _machine_contract()["materialization_contract"]
    _assert_materialization_nested_contracts_are_closed(materialization)

    for contract_name, nested_contract in materialization.items():
        if type(nested_contract) is not dict or not nested_contract.get(
            "nested_contracts"
        ):
            continue
        for field_name in nested_contract["nested_contracts"]:
            missing_binding = copy.deepcopy(materialization)
            del missing_binding[contract_name]["nested_contracts"][field_name]
            with pytest.raises(AssertionError):
                _assert_materialization_nested_contracts_are_closed(missing_binding)

            unknown_target = copy.deepcopy(materialization)
            unknown_target[contract_name]["nested_contracts"][field_name] = (
                "missing_nested_contract"
            )
            with pytest.raises(AssertionError):
                _assert_materialization_nested_contracts_are_closed(unknown_target)


def test_candidate_writer_contract_mutations_fail_closed() -> None:
    materialization = _machine_contract()["materialization_contract"]
    _assert_writer_contract_is_closed(materialization)

    mutations: tuple[tuple[tuple[str, ...], Any], ...] = (
        (
            ("repository_root_contract", "root_open_sequence"),
            (
                "open_and_fstat_same_fd",
                "irrelevant_filler",
                "require_directory_before_child_operation",
            ),
        ),
        (
            ("public_api_contract", "module"),
            "experiments.protocol.other_writer",
        ),
        (
            ("public_api_contract", "symbol"),
            "unsafe_writer",
        ),
        (
            ("candidate_state_contract", "exclusive_states"),
            (
                "new_candidate_directory_mkdir_winner",
                "existing_complete_byte_identical_candidate",
                "existing_partial_candidate",
                "existing_same_digest_different_bytes_collision",
                "unclassified_junk_state",
            ),
        ),
        (
            ("existing_candidate_read_contract", "file_open_flags"),
            ("O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"),
        ),
        (
            ("existing_candidate_read_contract", "file_open_sequence"),
            (
                "path_exists_precheck",
                "openat_exact_final_component_once",
                "fstat_same_fd",
                "require_regular_and_exact_mode_0600_before_first_read",
                "read_all_from_same_fd",
                "second_open",
                "close_same_fd_exactly_once",
            ),
        ),
        (
            ("existing_candidate_read_contract", "directory_enumeration_rule"),
            "pathlib_path_rglob",
        ),
        (
            ("temp_file_contract", "name_template"),
            ".{final_component}.{pid}.tmp",
        ),
        (
            ("temp_file_contract", "open_sequence"),
            (
                "openat_deterministic_temp_component_with_exact_flags_and_mode_0600",
                "fstat_same_fd_require_regular",
                "allow_first_write",
            ),
        ),
        (
            ("directory_contract", "created_directory_sequence"),
            (
                "mkdirat_with_mode_0700",
                "open_same_component_with_exact_directory_flags",
                "fstat_same_fd_require_directory",
                "fsync_parent_directory_fd",
            ),
        ),
        (
            ("directory_contract", "process_umask_read_or_mutation_is_forbidden"),
            False,
        ),
        (
            ("repository_root_contract", "derivation_expression"),
            "Path.cwd()",
        ),
        (
            (
                "repository_root_contract",
                "resolve_for_candidate_paths_is_forbidden",
            ),
            False,
        ),
        (
            ("directory_contract", "shared_parent_components"),
            ("outputs", "alternate_materialization_root"),
        ),
        (
            ("candidate_state_contract", "existing_candidate_branch_is_read_only"),
            False,
        ),
        (
            (
                "owned_cleanup_contract",
                "recursive_glob_prefix_and_age_based_cleanup_are_forbidden",
            ),
            False,
        ),
        (
            (
                "owned_cleanup_contract",
                "preexisting_unknown_and_other_candidate_deletion_is_forbidden",
            ),
            False,
        ),
        (
            ("durable_publish_contract", "publication_order"),
            (
                "materialization_manifest",
                "ordered_raw_member_files",
                "candidate_registry",
            ),
        ),
    )
    for path, replacement in mutations:
        changed = copy.deepcopy(materialization)
        target = changed["writer_file_dag_contract"]
        for component in path[:-1]:
            target = target[component]
        target[path[-1]] = replacement
        with pytest.raises(AssertionError):
            _assert_writer_contract_is_closed(changed)


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

    materialization = contract["materialization_contract"]
    generation_rules = materialization[
        "generation_input_identity_payload_contract"
    ]["field_rules"]
    for field_name in ("model_id", "model_revision", "negative_prompt"):
        assert generation_rules[field_name]["exact_value"] == config_values[
            field_name
        ]
    for field_name in ("width", "height", "inference_steps"):
        assert generation_rules[field_name]["exact_value"] == int(
            config_values[field_name]
        )
    assert generation_rules["guidance_scale"]["exact_value"] == float(
        config_values["guidance_scale"]
    )

    base_latent = materialization["base_latent_identity_reconstruction_contract"]
    assert base_latent["builder_symbol"] in (
        ROOT / "experiments/protocol/formal_randomization.py"
    ).read_text(encoding="utf-8-sig")
    assert tuple(base_latent["shape"]) == (1, 16, 64, 64)
    assert base_latent["dtype"] == f"torch.{config_values['latent_torch_dtype']}"
    assert base_latent["argument_sources"]["generation_seed_random"] == (
        "generation_input_identity_payload.generation_seed_random"
    )
    assert base_latent["argument_sources"]["model_id"] == (
        "generation_input_identity_payload.model_id"
    )
    assert base_latent["argument_sources"]["model_revision"] == (
        "generation_input_identity_payload.model_revision"
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
    assert (
        f"除该精确{len(exceptions)}项外，runtime/record 字段与 writer、consumer、aggregator 同一原子迁移的门禁原样有效"
        in field_registry_text
    )
    assert (
        "| check_status | governance | none | true | false | false | "
        "单项结果闭合语义检查的 pass 或 blocked 状态。 |"
        in field_registry_text
    )

    materialization = contract["materialization_contract"]
    status_tokens = materialization["qualification_report_contract"][
        "check_entry_field_rules"
    ]["check_status"]["allowed_tokens"]
    assert tuple(status_tokens) == ("pass", "blocked")
    assert set(status_tokens).isdisjoint({"passed", "failed", "skipped"})

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
        r"^def (load|validate|build|write)_content_routing_reference_registry\(",
        flags=re.MULTILINE,
    )
    implementations = []
    for root_name in ("main", "experiments", "paper_experiments", "scripts"):
        for path in (ROOT / root_name).rglob("*.py"):
            for match in implementation_pattern.finditer(
                path.read_text(encoding="utf-8-sig")
            ):
                implementations.append(
                    (match.group(1), path.relative_to(ROOT).as_posix())
                )
    assert implementations == [
        (
            "load",
            "experiments/protocol/content_routing_reference_registry.py",
        )
    ]


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

    materialization = contract["materialization_contract"]
    manifest = materialization["materialization_manifest_contract"]
    qualification = materialization["qualification_report_contract"]
    promotion = materialization["promotion_contract"]
    manifest_fields = set(manifest["field_rules"])
    qualification_fields = set(qualification["field_rules"])

    assert {
        "content_routing_reference_qualification_report_digest",
        "content_routing_reference_qualification_report_file_sha256",
        "content_routing_reference_qualification_ready",
        "qualification_checks",
        "qualification_errors",
    }.isdisjoint(manifest_fields)
    assert {
        "content_routing_reference_materialization_manifest_digest",
        "content_routing_reference_materialization_manifest_file_sha256",
        "content_routing_reference_registry_digest",
        "content_routing_reference_registry_file_sha256",
    } <= qualification_fields
    assert "content_routing_reference_qualification_report_file_sha256" not in (
        qualification_fields
    )
    assert qualification["file_sha256_location"] == (
        "external_promotion_input_only"
    )

    comparison_order = tuple(promotion["comparison_order"])
    assert comparison_order.index(
        "compare_report_manifest_identity_to_expected_and_actual_manifest"
    ) < comparison_order.index(
        "copy_exact_already_read_candidate_registry_bytes_to_fixed_config_path"
    )
    assert comparison_order.index(
        "compare_report_registry_identity_to_expected_and_actual_registry"
    ) < comparison_order.index(
        "copy_exact_already_read_candidate_registry_bytes_to_fixed_config_path"
    )
    assert promotion["manifest_and_report_mutation_is_forbidden"] is True
    assert (
        "manifest 只保存固定 qualification report 相对路径，不得保存 report digest、file SHA、ready、checks 或 errors"
        in method_text
    )
    assert (
        "manifest 不得反向绑定 report" in method_text
    )
