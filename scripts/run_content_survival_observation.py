"""Launch the explicitly requested GPU content-survival observation."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol.content_routing_reference_registry import (
    load_content_routing_reference_registry,
)
from experiments.protocol.content_survival_observation import (
    CONTENT_SURVIVAL_PROMPT_IDS,
    build_content_survival_execution_environment_identity,
    load_content_survival_observation_protocol,
)
from experiments.protocol.formal_randomization import (
    formal_generation_seed,
    resolve_formal_randomization_repeat,
    validate_formal_prompt_randomization_identity,
)
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.protocol.splits import apply_split_assignments
from experiments.runners.terminal_content_carrier_runtime import (
    run_terminal_content_carrier_observation,
)
from experiments.runners.image_only_dataset_workload import build_method_config
from experiments.runtime.dependency_profiles import (
    WORKFLOW_ORCHESTRATOR_PROFILE_ID,
    require_dependency_profile_ready,
)
from experiments.runtime.repository_environment import (
    build_runtime_environment_report,
    require_published_formal_execution_lock,
)
from main.core.digest import build_stable_digest


_ORCHESTRATOR_INSPECTION_FIELDS = frozenset(
    {
        "profile_name",
        "profile_digest",
        "complete_hash_lock_digest",
        "profile_formal_ready",
        "expected_environment",
        "observed_environment",
        "environment_match",
        "mismatches",
        "readiness_blockers",
        "decision",
        "inspection_digest",
    }
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the compact terminal content-carrier observation; this output "
            "cannot support paper claims or candidate promotion."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _prompt_records(root: Path) -> dict[str, object]:
    paper_run = build_paper_run_config(root)
    records = apply_split_assignments(
        build_prompt_records(
            paper_run.prompt_set,
            read_prompt_file(root / paper_run.prompt_file),
        )
    )
    by_id = {record.prompt_id: record for record in records}
    if any(prompt_id not in by_id for prompt_id in CONTENT_SURVIVAL_PROMPT_IDS):
        raise RuntimeError("the governed prompt file does not contain the frozen roster")
    return {prompt_id: by_id[prompt_id] for prompt_id in CONTENT_SURVIVAL_PROMPT_IDS}


def _prompt_configs(root: Path) -> dict[str, object]:
    if not os.environ.get("SLM_WM_KEY_MATERIAL"):
        raise RuntimeError("SLM_WM_KEY_MATERIAL must be explicitly provided")
    base = build_method_config(root)
    repeat = resolve_formal_randomization_repeat(base.randomization_repeat_id)
    base_generation_seed = int(base.seed) - int(repeat.generation_seed_offset)
    configs: dict[str, object] = {}
    for prompt_id, record in _prompt_records(root).items():
        seed = formal_generation_seed(
            base_generation_seed,
            int(record.prompt_index),
            repeat,
        )
        config = replace(
            base,
            prompt=record.prompt_text,
            prompt_id=record.prompt_id,
            split=record.split,
            seed=seed,
            standard_attack_profiles=(),
            diffusion_attacks_enabled=False,
        )
        validate_formal_prompt_randomization_identity(
            base_generation_seed_random=base_generation_seed,
            prompt_index=int(record.prompt_index),
            randomization_repeat_id=config.randomization_repeat_id,
            generation_seed_index=config.generation_seed_index,
            generation_seed_offset=config.generation_seed_offset,
            watermark_key_index=config.watermark_key_index,
            generation_seed_random=config.seed,
            watermark_key_seed_random=config.watermark_key_seed_random,
            key_material=config.key_material,
            formal_randomization_protocol_digest=(
                config.formal_randomization_protocol_digest
            ),
        )
        configs[prompt_id] = config
    return configs


def _prepare_prompt_saliency_model_cache(
    configs: Mapping[str, object],
    *,
    snapshot_downloader: Callable[..., str] | None = None,
) -> Path:
    """Download the fixed CLIP snapshot before its local-only formal load."""

    identities = {
        (
            str(getattr(config, "vision_model_id")),
            str(getattr(config, "vision_model_revision")),
        )
        for config in configs.values()
    }
    if len(identities) != 1:
        raise RuntimeError("prompt saliency model identity is not unique")
    model_id, model_revision = identities.pop()
    if snapshot_downloader is None:
        from huggingface_hub import snapshot_download

        snapshot_downloader = snapshot_download
    download_arguments: dict[str, object] = {
        "repo_id": model_id,
        "revision": model_revision,
        "token": os.environ.get("HF_TOKEN") or None,
    }
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        download_arguments["cache_dir"] = str(Path(hf_home) / "hub")
    snapshot_path = Path(snapshot_downloader(**download_arguments)).resolve()
    if not snapshot_path.is_dir():
        raise RuntimeError("prompt saliency model snapshot was not materialized")
    return snapshot_path


def _dependency_report_object_digest(value: Mapping[str, object]) -> str:
    """Rebuild the byte digest used by the formal dependency report writer."""

    report_bytes = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(report_bytes).hexdigest()


def _validated_orchestrator_report_identity(
    root: Path,
    dependency_report: Mapping[str, object],
    formal_execution_lock: Mapping[str, object],
    scientific_runtime_report: Mapping[str, object],
) -> dict[str, str]:
    """Read the verified orchestrator identity from the nested provision report."""

    scientific_expected = {
        "profile_id": scientific_runtime_report.get("dependency_profile_id"),
        "profile_digest": scientific_runtime_report.get("dependency_profile_digest"),
        "direct_requirements_digest": scientific_runtime_report.get(
            "direct_requirements_digest"
        ),
        "complete_hash_lock_digest": scientific_runtime_report.get(
            "complete_hash_lock_digest"
        ),
        "complete_hash_lock_dependency_count": scientific_runtime_report.get(
            "complete_hash_lock_dependency_count"
        ),
    }
    if any(
        dependency_report.get(field) != expected
        for field, expected in scientific_expected.items()
    ):
        raise RuntimeError("scientific dependency report identity drifted")
    if any(
        (
            dependency_report.get("formal_execution_lock")
            != dict(formal_execution_lock),
            dependency_report.get("formal_execution_commit")
            != formal_execution_lock.get("formal_execution_commit"),
            dependency_report.get("formal_execution_lock_digest")
            != formal_execution_lock.get("formal_execution_lock_digest"),
            dependency_report.get("formal_execution_lock_ready") is not True,
        )
    ):
        raise RuntimeError("scientific dependency report formal lock drifted")

    provision_report = dependency_report.get("provision_report")
    if type(provision_report) is not dict:
        raise RuntimeError("scientific dependency provision identity is absent")
    provision_digest = dependency_report.get("provision_report_digest")
    if (
        type(provision_digest) is not str
        or provision_digest != _dependency_report_object_digest(provision_report)
    ):
        raise RuntimeError("scientific dependency provision digest drifted")
    provision_expected = {
        "report_schema": "isolated_dependency_python_provision_report",
        "schema_version": 1,
        "operation_kind": "isolated_python_provision",
        "profile_id": dependency_report.get("profile_id"),
        "profile_digest": dependency_report.get("profile_digest"),
        "formal_execution_lock": dict(formal_execution_lock),
        "formal_execution_commit": formal_execution_lock.get(
            "formal_execution_commit"
        ),
        "formal_execution_lock_digest": formal_execution_lock.get(
            "formal_execution_lock_digest"
        ),
        "formal_execution_lock_ready": True,
        "provisioned": True,
        "formal_ready": False,
        "decision": "provisioned",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    if any(
        provision_report.get(field) != expected
        for field, expected in provision_expected.items()
    ):
        raise RuntimeError("scientific dependency provision identity drifted")

    orchestrator = require_dependency_profile_ready(
        WORKFLOW_ORCHESTRATOR_PROFILE_ID,
        root / "configs" / "dependency_profile_registry.json",
    )
    if any(
        (
            provision_report.get("orchestrator_profile_digest")
            != orchestrator.profile_digest,
            provision_report.get("orchestrator_complete_hash_lock_digest")
            != orchestrator.complete_hash_lock_digest,
        )
    ):
        raise RuntimeError("orchestrator provision identity drifted")
    orchestrator_inspection = provision_report.get("orchestrator_inspection")
    if type(orchestrator_inspection) is not dict:
        raise RuntimeError("orchestrator inspection identity is absent")
    if set(orchestrator_inspection) != _ORCHESTRATOR_INSPECTION_FIELDS:
        raise RuntimeError("orchestrator inspection schema drifted")
    inspection_digest = orchestrator_inspection.get("inspection_digest")
    inspection_payload = {
        field: value
        for field, value in orchestrator_inspection.items()
        if field != "inspection_digest"
    }
    if (
        type(inspection_digest) is not str
        or inspection_digest != build_stable_digest(inspection_payload)
    ):
        raise RuntimeError("orchestrator inspection digest drifted")
    if any(
        (
            orchestrator_inspection.get("profile_name")
            != WORKFLOW_ORCHESTRATOR_PROFILE_ID,
            orchestrator_inspection.get("profile_digest")
            != orchestrator.profile_digest,
            orchestrator_inspection.get("complete_hash_lock_digest")
            != orchestrator.complete_hash_lock_digest,
            orchestrator_inspection.get("profile_formal_ready") is not True,
            orchestrator_inspection.get("environment_match") is not True,
            orchestrator_inspection.get("mismatches") != [],
            orchestrator_inspection.get("readiness_blockers") != [],
            orchestrator_inspection.get("decision") != "pass",
        )
    ):
        raise RuntimeError("orchestrator inspection readiness drifted")
    return {
        "orchestrator_profile_digest": orchestrator.profile_digest,
        "orchestrator_complete_hash_lock_digest": str(
            orchestrator.complete_hash_lock_digest
        ),
        "orchestrator_inspection_digest": inspection_digest,
    }


def _execution_environment_identity(
    root: Path,
    formal_execution_lock: dict[str, object],
) -> dict[str, object]:
    report = build_runtime_environment_report(
        "sd35_method_runtime_gpu",
        verified_formal_execution_lock=formal_execution_lock,
        repository_root=root,
    )
    if (
        report.get("dependency_environment_ready") is not True
        or report.get("isolated_scientific_context_ready") is not True
    ):
        raise RuntimeError("scientific execution environment is not formally ready")
    context = report["isolated_scientific_context"]
    dependency_path = Path(context["dependency_environment_report_path"])
    dependency_bytes = dependency_path.read_bytes()
    dependency_digest = hashlib.sha256(dependency_bytes).hexdigest()
    if dependency_digest != context["dependency_environment_report_actual_digest"]:
        raise RuntimeError("scientific dependency report digest drifted")
    dependency_report = json.loads(dependency_bytes.decode("utf-8"))
    if type(dependency_report) is not dict:
        raise RuntimeError("scientific dependency report is not an object")
    orchestrator_identity = _validated_orchestrator_report_identity(
        root,
        dependency_report,
        formal_execution_lock,
        report,
    )
    return build_content_survival_execution_environment_identity(
        {
            **orchestrator_identity,
            "scientific_profile_id": report["dependency_profile_id"],
            "scientific_profile_digest": report["dependency_profile_digest"],
            "scientific_direct_requirements_digest": report[
                "direct_requirements_digest"
            ],
            "scientific_complete_hash_lock_digest": report[
                "complete_hash_lock_digest"
            ],
            "scientific_dependency_environment_report_digest": (
                dependency_digest
            ),
            "scientific_python_executable_sha256": context[
                "reported_python_executable_sha256"
            ],
        }
    )


def main() -> int:
    arguments = _arguments()
    root = arguments.repository_root.resolve()
    output_dir = arguments.output_dir
    resolved_output = (root / output_dir).resolve()
    resolved_output.relative_to((root / "outputs").resolve())
    lock = require_published_formal_execution_lock(root)
    protocol = load_content_survival_observation_protocol(root)
    routing_identity = protocol.payload["routing_reference_registry"]
    references = load_content_routing_reference_registry(
        expected_registry_digest=routing_identity["semantic_digest"],
        expected_file_sha256=routing_identity["file_sha256"],
    )
    execution_environment_identity = _execution_environment_identity(root, lock)
    prompt_configs = _prompt_configs(root)
    _prepare_prompt_saliency_model_cache(prompt_configs)
    summary = run_terminal_content_carrier_observation(
        prompt_configs,
        references=references,
        verified_formal_execution_lock=lock,
        verified_execution_environment_identity=execution_environment_identity,
        repository_root=root,
        output_dir=output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
