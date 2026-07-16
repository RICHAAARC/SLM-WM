"""profile 同构登记表与真实结果闭合产物契约的一致性测试。"""

from __future__ import annotations

import pytest

from paper_experiments.analysis.paper_profile_protocol_isomorphism import (
    load_paper_profile_protocol_registry,
)
from scripts.paper_result_closure import (
    _CLOSURE_ARTIFACT_SPECS,
    _OPTIONAL_DIAGNOSTIC_ARTIFACT_SPECS,
)


pytestmark = pytest.mark.constraint


def test_registered_artifact_contract_matches_result_closure_writers() -> None:
    """同构报告不得维护一套与真实结果闭合 writer 不同的产物 schema。"""

    registry = load_paper_profile_protocol_registry()
    registered = [
        *registry["artifact_contract"],
        *registry["diagnostic_artifact_contract"],
    ]
    actual = [
        {
            "artifact_id": spec.artifact_id,
            "writer_module": spec.module_name,
            "ready_field": spec.ready_field,
            "file_names": list(spec.file_names),
        }
        for spec in (
            *_CLOSURE_ARTIFACT_SPECS,
            *_OPTIONAL_DIAGNOSTIC_ARTIFACT_SPECS,
        )
    ]

    assert registered == actual
