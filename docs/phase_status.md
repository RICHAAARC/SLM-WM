# 阶段状态

## 文档定位

本文档记录当前分阶段构建推进状态。它只描述阶段门禁、输入、输出和阻断项,
不承载正式论文实验结论。

## stage_00_core_package_boundary_freeze

| item | value |
| --- | --- |
| construction_unit_name | `stage_00_core_package_boundary_freeze` |
| phase_status | `completed` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/audit_reports/harness_audit_summary.json` |
| expected_output_manifest | `outputs/core_package_boundary_freeze/manifest.local.json` |
| expected_outputs | `outputs/core_package_boundary_freeze/core_boundary_report.json`; `outputs/core_package_boundary_freeze/core_import_report.json`; `outputs/core_package_boundary_freeze/core_package_layout.txt`; `outputs/core_package_boundary_freeze/manifest.local.json` |
| blocking_items | 无。 |
| fallback_path | 若核心包边界检查失败, 停止推进并修复 `main/` 反向依赖。 |
| invariants | `main/` 不依赖 Colab、Drive、experiments、scripts、tests、tools/harness、paper_workflow 或外部 baseline。 |
| next_stage_entry | stage00 验证通过后, 才能进入 `stage_01_algorithm_primitives`。 |

### stage00 已冻结内容

1. `main/` 最小包结构包括 `main/core/`、`main/methods/`、`main/protocol/`、`main/analysis/` 和 `main/cli/`。
2. `main/core/method_objects.py` 定义语义条件、潜空间子空间、水印载体、注意力锚点、检测证据和融合决策的最小 typed object。
3. `tests/constraints/test_main_boundary_contract.py` 对核心包导入边界进行轻量约束测试。
4. `scripts/write_core_package_boundary_outputs.py` 只向 `outputs/core_package_boundary_freeze/` 写入本地阶段报告。

### stage00 验证结果

| command | result |
| --- | --- |
| `python -c "import main"` | pass |
| `python scripts/write_core_package_boundary_outputs.py` | pass |
| `pytest tests/constraints -q` | pass, 9 passed |
| `pytest -q` | pass, 12 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_01_algorithm_primitives

| item | value |
| --- | --- |
| construction_unit_name | `stage_01_algorithm_primitives` |
| phase_status | `completed` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/core_package_boundary_freeze/manifest.local.json` |
| expected_output_manifest | `outputs/algorithm_primitives/manifest.local.json` |
| expected_outputs | `outputs/algorithm_primitives/core_primitive_summary.json`; `outputs/algorithm_primitives/synthetic_core_records.jsonl`; `outputs/algorithm_primitives/manifest.local.json` |
| blocking_items | 无。 |
| fallback_path | 若纯算法原语不能在无 SD3、无 Colab、无 Drive 环境下通过测试, 停止推进并修复 `main/methods/` 原语实现。 |
| invariants | 不引入 diffusers、transformers、SD 权重、Colab、Drive 或 Notebook; `main/` 不写出 records; attention carrier 仅为 synthetic stub。 |
| next_stage_entry | stage01 验证通过后, 才能进入 `stage_02_core_method_smoke_test`。 |

### stage01 已完成内容

1. `main/methods/algorithm_primitives.py` 实现纯算法原语闭环, 包括语义风险场、latent mask 投影、安全基底估计、LF/HF carrier、attention synthetic stub、latent update 合成、内容分数、几何可靠性和 evidence/final 判定。
2. `scripts/run_core_smoke.py` 根据 typed objects 生成 stage01 本地 summary、synthetic records 和 manifest, 且所有输出均写入 `outputs/algorithm_primitives/`。
3. `tests/functional/test_algorithm_primitives.py` 覆盖正确 key、错误 key、HF tail truncation、rescue 边界和 attestation 分层。
4. `docs/field_registry.md` 已登记 stage01 新增字段。

### stage01 验证结果

| command | result |
| --- | --- |
| `python scripts/run_core_smoke.py` | pass |
| `pytest tests/functional -q` | pass, 7 passed |
| `pytest tests/constraints -q` | pass, 9 passed |
| `pytest -q` | pass, 16 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_02_core_method_smoke_test

| item | value |
| --- | --- |
| construction_unit_name | `stage_02_core_method_smoke_test` |
| phase_status | `completed` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/core_package_boundary_freeze/manifest.local.json`; `outputs/algorithm_primitives/manifest.local.json` |
| expected_output_manifest | `outputs/core_method_synthetic_smoke/manifest.local.json` |
| expected_outputs | `outputs/core_method_synthetic_smoke/synthetic_event_records.jsonl`; `outputs/core_method_synthetic_smoke/core_smoke_metrics.json`; `outputs/core_method_synthetic_smoke/core_smoke_summary.md`; `outputs/core_method_synthetic_smoke/manifest.local.json` |
| blocking_items | 无。 |
| fallback_path | 若 synthetic latent smoke 不能复现 key 区分、rescue 边界或 attestation 分层, 停止推进并修复 `main/methods/synthetic_smoke.py`。 |
| invariants | 不接入真实 SD3/SD3.5、Colab、Drive 或 Notebook; 不把 smoke 结果写成论文 supported claims; attention carrier 仍为 synthetic stub。 |
| next_stage_entry | stage02 验证通过后, 才能进入 `stage_03_sd3_runtime_adapter`。 |

### stage02 已完成内容

1. `main/methods/synthetic_smoke.py` 构造 clean、watermarked、wrong-key negative、geometric shifted、aligned recovered、unattested positive 和 final positive 等 synthetic latent 场景。
2. `scripts/run_core_smoke.py --unit core_method_smoke` 写出 stage02 synthetic records、metrics、summary 和 manifest。
3. `scripts/run_minimal_method_smoke.py` 提供 minimal method package 可复用的 stdout smoke。
4. `tests/functional/test_core_method_smoke.py` 覆盖错误 key、rescue 边界、几何可靠性不足阻断 rescue 和 attestation 分层。

### stage02 验证结果

| command | result |
| --- | --- |
| `python scripts/run_minimal_method_smoke.py` | pass |
| `python scripts/run_core_smoke.py --unit core_method_smoke` | pass |
| `pytest tests/functional -q` | pass, 11 passed |
| `pytest tests/constraints -q` | pass, 9 passed |
| `pytest -q` | pass, 20 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_03_sd3_runtime_adapter

| item | value |
| --- | --- |
| construction_unit_name | `stage_03_sd3_runtime_adapter` |
| phase_status | `completed` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/core_method_synthetic_smoke/manifest.local.json` |
| expected_output_manifest | `outputs/sd_runtime_adapter/manifest.local.json` |
| expected_outputs | `outputs/sd_runtime_adapter/sd_generation_records.jsonl`; `outputs/sd_runtime_adapter/latent_trace_records.jsonl`; `outputs/sd_runtime_adapter/attention_capture_records.jsonl`; `outputs/sd_runtime_adapter/generation_quality_summary.json`; `outputs/sd_runtime_adapter/manifest.local.json` |
| blocking_items | 无。 |
| fallback_path | 本地没有真实 SD3 / SD3.5 权重、GPU 或模型访问权限时, 使用 synthetic fallback 生成工程 records, 并在 records 中写入 `unsupported_reason`; fallback records 不支持正式论文 claim。 |
| invariants | `main/` 不依赖 diffusers、transformers、模型权重、experiments runtime 或脚本; runtime 层只能调用 core, core 不反向依赖 runtime。 |
| next_stage_entry | stage03 验证通过后, 才能进入 `stage_04_minimal_diffusion_latent_injection`。 |

### stage03 已完成内容

1. `experiments/runtime/diffusion/` 提供 SD3 / SD3.5 runtime adapter、synthetic fallback、sampler hook、latent trace、attention capture 和 latent estimator。
2. `configs/model_sd3.yaml` 与 `configs/model_sd35.yaml` 提供轻量 runtime probe 配置。
3. `scripts/run_diffusion_runtime_probe.py` 写出 generation records、latent trace records、attention capture records、quality summary 和 manifest, 且所有输出均写入 `outputs/sd_runtime_adapter/`。
4. `tests/functional/test_diffusion_runtime_adapter.py` 覆盖 fallback 原因、相同 prompt / seed 复现和输出目录约束。
5. `docs/field_registry.md` 已登记 runtime adapter 新增字段。

### stage03 验证结果

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/run_diffusion_runtime_probe.py` | pass |
| `pytest tests/functional/test_diffusion_runtime_adapter.py -q` | pass, 3 passed |
| `pytest -q` | pass, 24 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |
