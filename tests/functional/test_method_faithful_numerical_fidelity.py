"""验证 method-faithful baseline 的默认源码缺失边界."""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from paper_experiments.baselines.method_faithful_numerical_fidelity import (
    MethodFaithfulNumericalFidelityError,
    build_method_faithful_numerical_fidelity_report,
)


pytestmark = pytest.mark.quick
ROOT = Path(__file__).resolve().parents[2]


def test_numerical_fidelity_report_rejects_missing_registered_source(
    tmp_path: Path,
) -> None:
    """默认 quick 边界必须确认正式构造器对缺失源码失败关闭."""

    registry_dir = tmp_path / "external_baseline"
    registry_dir.mkdir()
    shutil.copyfile(
        ROOT / "external_baseline/source_registry.json",
        registry_dir / "source_registry.json",
    )

    with pytest.raises(
        MethodFaithfulNumericalFidelityError,
        match="无法读取登记的官方 Git 源码",
    ):
        build_method_faithful_numerical_fidelity_report(
            tmp_path,
            "tree_ring",
        )
