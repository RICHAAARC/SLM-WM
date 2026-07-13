"""精确9重复聚合后的 CPU 论文结果闭合入口.

单个 seed-key repeat 只允许写出随机化证据包. 在聚合器精确复验权威9个
repeat 并重算阈值与统计前, 本模块始终 fail-closed, 不保留可误调用的单重复
论文结果 DAG.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
from typing import Any, Callable

from experiments.protocol.paper_run_config import normalize_paper_run_name


CommandHook = Callable[[list[str]], None]
ProgressHook = Callable[[int, int, str], None]
PAPER_RESULT_CLOSURE_COMMAND_COUNT = 0

CLOSURE_RAW_OUTPUT_DIR_TEMPLATES: tuple[str, ...] = (
    "outputs/image_only_dataset_runtime/{paper_run_name}",
    "outputs/formal_mechanism_ablation/{paper_run_name}",
    "outputs/dataset_level_quality/{paper_run_name}",
    "outputs/external_baseline_method_faithful/{paper_run_name}",
    "outputs/tree_ring_official_reference/{paper_run_name}",
    "outputs/gaussian_shading_official_reference/{paper_run_name}",
    "outputs/shallow_diffuse_official_reference/{paper_run_name}",
    "outputs/t2smark_formal_reproduction/{paper_run_name}",
)

CLOSURE_DERIVED_OUTPUT_DIR_TEMPLATES: tuple[str, ...] = (
    "outputs/official_reference_fidelity_evidence/{paper_run_name}",
    "outputs/attack_matrix/{paper_run_name}",
    "outputs/paired_superiority_analysis/{paper_run_name}",
    "outputs/primary_baseline_method_faithful_adapter_protocol/{paper_run_name}",
    "outputs/external_baseline_results/{paper_run_name}",
    "outputs/primary_baseline_formal_import/{paper_run_name}",
    "outputs/fixed_fpr_threshold_audit/{paper_run_name}",
    "outputs/primary_baseline_evidence/{paper_run_name}",
    "outputs/external_baseline_comparison/{paper_run_name}",
    "outputs/pilot_paper_fixed_fpr_results/{paper_run_name}",
    "outputs/pilot_paper_fixed_fpr_common_protocol/{paper_run_name}",
    "outputs/pilot_paper_result_analysis/{paper_run_name}",
    "outputs/paper_artifact_evidence_audit/{paper_run_name}",
    "outputs/submission_readiness/{paper_run_name}",
    "outputs/evidence_closure_entry_review/{paper_run_name}",
    "outputs/result_closure_gate/{paper_run_name}",
    "outputs/pilot_paper_complete_result_package/{paper_run_name}",
)


def clean_paper_result_closure_outputs(
    *,
    root: str | Path,
    paper_run_name: str,
    selected_package_paths: Sequence[str | Path],
) -> tuple[str, ...]:
    """清理当前 run 的受管物化目录, 且绝不删除锁定输入包."""

    root_path = Path(root).resolve()
    outputs_root = (root_path / "outputs").resolve()
    run_name = normalize_paper_run_name(paper_run_name)
    selected_paths = tuple(
        Path(path).expanduser().resolve() for path in selected_package_paths
    )
    managed_paths: list[Path] = []
    for template in (
        CLOSURE_RAW_OUTPUT_DIR_TEMPLATES
        + CLOSURE_DERIVED_OUTPUT_DIR_TEMPLATES
    ):
        managed_path = (
            root_path / template.format(paper_run_name=run_name)
        ).resolve()
        try:
            managed_path.relative_to(outputs_root)
        except ValueError as exc:
            raise ValueError("结果闭合清理路径必须位于 outputs/ 下") from exc
        if any(
            package_path == managed_path or managed_path in package_path.parents
            for package_path in selected_paths
        ):
            raise ValueError(
                f"锁定输入包位于受管清理目录内, 拒绝删除: {managed_path.as_posix()}"
            )
        managed_paths.append(managed_path)

    removed_paths: list[str] = []
    for managed_path in managed_paths:
        if managed_path.is_dir():
            shutil.rmtree(managed_path)
            removed_paths.append(
                managed_path.relative_to(root_path).as_posix()
            )
        elif managed_path.exists():
            managed_path.unlink()
            removed_paths.append(
                managed_path.relative_to(root_path).as_posix()
            )
    return tuple(removed_paths)


def build_paper_result_closure_commands(
    *,
    randomization_repeat_components: Sequence[Mapping[str, Any]],
    complete_drive_output_dir: str | Path,
    paper_run_name: str,
    target_fpr: float,
    archive_name: str,
    root: str | Path = ".",
) -> list[list[str]]:
    """拒绝从单 repeat 输入构造正式论文闭合命令."""

    del (
        randomization_repeat_components,
        complete_drive_output_dir,
        paper_run_name,
        target_fpr,
        archive_name,
        root,
    )
    raise RuntimeError(
        "论文结果闭合只接受精确9重复聚合证据;单 repeat 输入不得构造论文闭合 DAG"
    )


def run_paper_result_closure_commands(
    *,
    package_search_root: str | Path,
    complete_drive_output_dir: str | Path,
    paper_run_name: str,
    target_fpr: float,
    root: str | Path = ".",
    before_command: CommandHook | None = None,
    progress_hook: ProgressHook | None = None,
) -> dict[str, Any]:
    """拒绝在缺少精确9重复聚合证据时执行论文闭合."""

    del (
        package_search_root,
        complete_drive_output_dir,
        paper_run_name,
        target_fpr,
        root,
        before_command,
        progress_hook,
    )
    raise RuntimeError(
        "论文结果闭合只接受精确9重复聚合证据;请先生成全部 repeat 证据包"
    )
