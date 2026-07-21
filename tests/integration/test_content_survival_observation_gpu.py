"""Explicit real-GPU entry for the fixed content-survival observation."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_content_survival_observation_real_gpu() -> None:
    if os.environ.get("SLM_WM_RUN_CONTENT_SURVIVAL_OBSERVATION_GPU") != "1":
        pytest.skip("explicit GPU observation authorization is absent")
    if not torch.cuda.is_available():
        pytest.fail("explicit observation requires a real CUDA device")
    output_dir = os.environ.get("SLM_WM_CONTENT_SURVIVAL_OBSERVATION_OUTPUT_DIR")
    if not output_dir:
        pytest.fail("an explicit persistent outputs/ path is required")
    root = Path(__file__).resolve().parents[2]
    resolved_output = (root / output_dir).resolve()
    resolved_output.relative_to((root / "outputs").resolve())
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_content_survival_observation.py",
            "--repository-root",
            str(root),
            "--output-dir",
            str(resolved_output.relative_to(root)),
        ],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )
    assert completed.returncode == 0, completed.stderr
    assert '"chain_count": 148' in completed.stdout
    assert '"evaluation_count": 29304' in completed.stdout
