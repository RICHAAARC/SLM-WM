"""Launch the explicitly requested GPU content-survival observation."""

from __future__ import annotations

import argparse
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
from experiments.runners.content_survival_observation_runtime import (
    run_content_survival_observation,
)
from experiments.runners.image_only_dataset_workload import build_method_config
from experiments.runtime.repository_environment import (
    build_runtime_environment_report,
    require_published_formal_execution_lock,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed 148-chain content-survival observation; this output "
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
    orchestrator_inspection = dependency_report.get("orchestrator_inspection")
    if type(orchestrator_inspection) is not dict:
        raise RuntimeError("orchestrator inspection identity is absent")
    return build_content_survival_execution_environment_identity(
        {
            "orchestrator_profile_digest": dependency_report[
                "orchestrator_profile_digest"
            ],
            "orchestrator_complete_hash_lock_digest": dependency_report[
                "orchestrator_complete_hash_lock_digest"
            ],
            "orchestrator_inspection_digest": orchestrator_inspection[
                "inspection_digest"
            ],
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
    summary = run_content_survival_observation(
        _prompt_configs(root),
        references=references,
        verified_formal_execution_lock=lock,
        verified_execution_environment_identity=(
            _execution_environment_identity(root, lock)
        ),
        repository_root=root,
        output_dir=output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
