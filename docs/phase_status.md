# 阶段状态

## 文档定位

本文档记录当前分阶段构建推进状态。它只描述阶段门禁、输入、输出和阻断项,
不承载正式论文实验结论。

## stage_00_core_package_boundary_freeze

| item | value |
| --- | --- |
| stage_name | `stage_00_core_package_boundary_freeze` |
| phase_status | `completed` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/audit_reports/harness_audit_summary.json` |
| expected_output_manifest | `outputs/local_stage00_core_boundary/manifest.local.json` |
| expected_outputs | `outputs/local_stage00_core_boundary/core_boundary_report.json`; `outputs/local_stage00_core_boundary/core_import_report.json`; `outputs/local_stage00_core_boundary/core_package_layout.txt`; `outputs/local_stage00_core_boundary/manifest.local.json` |
| blocking_items | 无。 |
| fallback_path | 若核心包边界检查失败, 停止推进并修复 `main/` 反向依赖。 |
| invariants | `main/` 不依赖 Colab、Drive、experiments、scripts、tests、tools/harness、paper_workflow 或外部 baseline。 |
| next_stage_entry | stage00 验证通过后, 才能进入 `stage_01_algorithm_primitives`。 |

### stage00 已冻结内容

1. `main/` 最小包结构包括 `main/core/`、`main/methods/`、`main/protocol/`、`main/analysis/` 和 `main/cli/`。
2. `main/core/method_objects.py` 定义语义条件、潜空间子空间、水印载体、注意力锚点、检测证据和融合决策的最小 typed object。
3. `tests/constraints/test_main_boundary_contract.py` 对核心包导入边界进行轻量约束测试。
4. `scripts/write_stage00_core_boundary_outputs.py` 只向 `outputs/local_stage00_core_boundary/` 写入本地阶段报告。

### stage00 验证结果

| command | result |
| --- | --- |
| `python -c "import main"` | pass |
| `python scripts/write_stage00_core_boundary_outputs.py` | pass |
| `pytest tests/constraints -q` | pass, 9 passed |
| `pytest -q` | pass, 12 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |
