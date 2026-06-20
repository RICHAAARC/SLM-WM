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

## stage_04_minimal_diffusion_latent_injection

| item | value |
| --- | --- |
| construction_unit_name | `stage_04_minimal_diffusion_latent_injection` |
| phase_status | `completed` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/sd_runtime_adapter/manifest.local.json`; `outputs/core_method_synthetic_smoke/manifest.local.json` |
| expected_output_manifest | `outputs/minimal_diffusion_latent_injection/manifest.local.json` |
| expected_outputs | `paper_workflow/sd_runtime_cold_start_probe.ipynb`; `paper_workflow/colab_utils/sd_runtime_cold_start.py`; `paper_workflow/minimal_latent_injection_run.ipynb`; `paper_workflow/colab_utils/minimal_latent_injection.py`; `outputs/real_sd_runtime_probe/*_manifest.local.json`; `outputs/real_sd_runtime_probe/*_environment_report.json`; `outputs/real_sd_runtime_probe_package_<utc>_<short_commit>.zip`; `GoogleDrive/SLM/real_sd_runtime_probe/real_sd_runtime_probe_package_<utc>_<short_commit>.zip`; `outputs/minimal_diffusion_latent_injection/*_injection_result.json`; `outputs/minimal_diffusion_latent_injection/*_latent_update_records.jsonl`; `outputs/minimal_diffusion_latent_injection/*_paired_quality_metrics.csv`; `outputs/minimal_diffusion_latent_injection/*_environment_report.json`; `outputs/minimal_diffusion_latent_injection/*_manifest.local.json`; `outputs/minimal_latent_injection_package_<utc>_<short_commit>.zip`; `GoogleDrive/SLM/minimal_diffusion_latent_injection/minimal_latent_injection_package_<utc>_<short_commit>.zip` |
| blocking_items | 无。 |
| fallback_path | SD3.5 Medium 是主线; 若主模型在 Colab 不可用, 运行 SD3 Medium 兼容 fallback 并写出 `unsupported_reason`; fallback 产物不得支持正式论文 claim。 |
| invariants | Notebook 只作为入口; runtime 逻辑位于 repository helper; `main/` 不依赖 Colab、Drive、diffusers、transformers 或模型权重。 |
| next_stage_entry | Colab 真实推理、真实 latent trajectory、paired images、latent update records 和质量指标均已通过本地审计; 可进入 `stage_05_colab_drive_workflow`。 |

### stage04 已完成内容

1. `paper_workflow/sd_runtime_cold_start_probe.ipynb` 提供 Colab 冷启动入口, 支持拉取代码、安装依赖、登录 Hugging Face、挂载 Google Drive, 并可运行 SD3.5 Medium 主模型与 SD3 Medium 兼容 fallback。
2. `paper_workflow/colab_utils/sd_runtime_cold_start.py` 承载真实 SD runtime 调用、latent callback 捕获、图像摘要、trajectory records、environment report、summary、manifest、zip 打包和 Google Drive 镜像逻辑。
3. 已审计 `outputs/real_sd_runtime_probe_package_20260620t10451781952321z_b2be25c.zip`; 该包对应提交 `b2be25c`, ZIP 完整性通过, SHA-256 为 `be6e4373edf81311209e0eb220ac189fd43e046128e2ba05815a0775dd9fceb7`。
4. runtime probe 结果中, SD3.5 Medium 主模型 `stabilityai/stable-diffusion-3.5-medium` 与 SD3 Medium fallback `stabilityai/stable-diffusion-3-medium-diffusers` 均完成真实推理, 均捕获 28 条真实 latent trajectory records, latent shape 均为 `[1, 16, 64, 64]`。
5. runtime probe 环境快照已记录 Colab L4、CUDA 12.8、Python 3.12.13、torch 2.11.0+cu128、diffusers 0.38.0、transformers 5.12.1、accelerate 1.14.0 和 huggingface_hub 1.20.1。
6. `paper_workflow/minimal_latent_injection_run.ipynb` 提供最小 latent injection 的 Colab 冷启动入口, 首个代码单元先挂载 Google Drive, 默认 `SLM_WM_MODEL_SELECTION=auto`, 当前以 SD3.5 Medium 主模型为最小真实注入验证对象, 并将 zip 镜像到 `SLM/minimal_diffusion_latent_injection/`。
7. `paper_workflow/colab_utils/minimal_latent_injection.py` 承载 clean / watermarked paired image 生成、latent callback 注入、latent update records、paired quality metrics、environment report、manifest、zip 打包和 Google Drive 镜像逻辑。
8. 已审计 `outputs/minimal_latent_injection_package_20260620t10181781950721z_b2be25c.zip`; 该包对应提交 `b2be25c`, ZIP 完整性通过, SHA-256 为 `bff5f14c7e57e669dc6e9e371bb999fa663581bf4033ba771ab6595ff5d0ec0c`。
9. minimal latent injection 结果中, SD3.5 Medium 生成 clean / watermarked paired images、3 条 latent update records、paired quality metrics、manifest 和 environment report; 质量指标记录包括 PSNR `37.86754851645436`、SSIM `0.9987065282916542`、MSE `0.00016339740250259638` 和 mean_abs_error `0.00732430862262845`。
10. `configs/colab_sd35_runtime_constraints.txt` 记录本次已验证的 SD3.5 Medium Colab 依赖组合, 仅作为远程 Notebook 复现参考, 不属于本地默认安装依赖。
11. `tests/constraints/test_notebook_entrypoint_contract.py` 验证 Notebook 文件名、未保存执行输出、Notebook 调用 repository helper、probe / injection 产物可被打包和镜像, 以及 Colab 运行环境约束记录不强制安装平台提供的 torch。
12. `tests/functional/test_minimal_latent_injection_helpers.py` 验证最小 injection 配置、稳定摘要、轻量质量指标、默认模型选择、运行环境版本快照和 environment report 写出。
13. `docs/field_registry.md` 已登记真实 runtime probe、archive 和最小 latent injection 新增字段。

### stage04 完成边界

1. 本阶段完成的是真实 SD3.5 / SD3 推理链路、真实 latent trajectory 捕获和 SD3.5 Medium 最小 latent injection 工程验证。
2. 当前 `supports_paper_claim=false` 的边界保持不变; 这些结果不得直接作为论文中的 watermark detection、robustness 或 fixed-FPR 结论。
3. 当前阶段不要求真实 attention capture; Q/K attention 或可审计 attention map 应在后续 attention capture 专门构建单元中接入。
4. SD3.5 Medium 是后续主线模型; SD3 Medium 仅保留为兼容性 fallback 与对照证据。

### stage04 当前验证结果

| command | result |
| --- | --- |
| `pytest tests/constraints/test_notebook_entrypoint_contract.py -q` | pass, 7 passed |
| `pytest tests/functional/test_minimal_latent_injection_helpers.py -q` | pass, 7 passed |
| `pytest -q` | pass, 38 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_05_colab_drive_workflow

| item | value |
| --- | --- |
| construction_unit_name | `stage_05_colab_drive_workflow` |
| phase_status | `in_progress` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/core_package_boundary_freeze/manifest.local.json`; `outputs/algorithm_primitives/manifest.local.json`; `outputs/core_method_synthetic_smoke/manifest.local.json`; `outputs/sd_runtime_adapter/manifest.local.json`; `outputs/real_sd_runtime_probe_package_20260620t10451781952321z_b2be25c.zip`; `outputs/minimal_latent_injection_package_20260620t10181781950721z_b2be25c.zip` |
| expected_output_manifest | `outputs/colab_drive_workflow/manifest.local.json`; `GoogleDrive/SLM/colab_drive_workflow/manifest.json` |
| expected_outputs | `paper_workflow/colab_drive_cold_start_smoke.ipynb`; `paper_workflow/drive_manifest_reload_smoke.ipynb`; `paper_workflow/colab_utils/drive_paths.py`; `paper_workflow/colab_utils/dependency_check.py`; `paper_workflow/colab_utils/mount_drive.py`; `paper_workflow/colab_utils/runtime_setup.py`; `paper_workflow/colab_utils/manifest_io.py`; `paper_workflow/colab_utils/drive_workflow.py`; `scripts/colab_drive_entry.py`; `scripts/sync_local_outputs_to_drive.py`; `scripts/write_workflow_manifest.py`; `scripts/verify_drive_artifacts.py`; `outputs/colab_drive_workflow/colab_env_report.json`; `outputs/colab_drive_workflow/drive_mount_report.json`; `outputs/colab_drive_workflow/cold_start_smoke_record.jsonl`; `outputs/colab_drive_workflow/reload_smoke_record.jsonl`; `outputs/colab_drive_workflow/local_output_sync_report.json`; `outputs/colab_drive_workflow/manifest.local.json` |
| blocking_items | 本地环境不能真实挂载 Google Drive; Drive 侧 manifest reload 仍需要在 Colab 中执行 Notebook 或等价 CLI 后审计。 |
| fallback_path | 若 Colab 无法挂载 Drive, 本地命令仅写入 `outputs/colab_drive_workflow/drive_mirror/` 镜像目录并记录 `unsupported_reason`; 该镜像不得支持正式论文 claim。 |
| invariants | Notebook 只作为入口; Drive manifest、镜像与重载校验逻辑位于 repository helper 和 scripts; `main/` 不依赖 Colab、Drive 或 Notebook。 |
| next_stage_entry | 只有 Drive manifest 在 Colab 中生成、镜像和 reload 校验均通过后, 才能进入 `stage_06_prompt_split_records_protocol`。 |

### stage05 已完成内容

1. 新增 Colab Drive workflow helper, 将路径解析、依赖快照、Drive 挂载报告、manifest 读写、本地 outputs 镜像和 reload 校验分离到 `paper_workflow/colab_utils/` 下的语义化模块。
2. 新增 `scripts/colab_drive_entry.py`、`scripts/sync_local_outputs_to_drive.py`、`scripts/write_workflow_manifest.py` 和 `scripts/verify_drive_artifacts.py`, 作为 Notebook 可调用的仓库入口。
3. 新增 `paper_workflow/colab_drive_cold_start_smoke.ipynb` 与 `paper_workflow/drive_manifest_reload_smoke.ipynb`, 两个 Notebook 均不保存执行输出, 且只调用 repository helper。
4. 新增轻量测试覆盖本地 outputs 镜像、manifest 写入、reload 校验、本地输出目录约束、Drive 挂载跳过报告和依赖快照非 claim 边界。
5. 本地执行 `python scripts/colab_drive_entry.py` 已在 `outputs/colab_drive_workflow/` 生成可审计 smoke 产物, 并验证本地镜像 reload 通过。
6. 已修正 Colab 冷启动输入边界: 若 clone 后本地 `outputs/` 为空, workflow 会登记 Google Drive 中已有的 `SLM/real_sd_runtime_probe/` 与 `SLM/minimal_diffusion_latent_injection/` 真实运行产物, 而不是把空 manifest 误判为有效证据。
7. `docs/field_registry.md` 已登记 Colab Drive workflow 新增字段。

### stage05 尚未完成内容

1. 已审计 `outputs/colab_drive_workflow-20260620T112309Z-3-001.zip`; 该包结构完整, 但 `local_manifest_count=0`、`mirrored_file_count=0`, 只能证明空 manifest reload, 不能作为本阶段完成证据。
2. 尚未使用修正后的 Drive 输入边界在 Colab 中重新运行 `paper_workflow/colab_drive_cold_start_smoke.ipynb` 生成包含前序 Drive 产物的真实 `manifest.json`。
3. 尚未在 Colab 中重新运行 `paper_workflow/drive_manifest_reload_smoke.ipynb` 验证非空 Drive manifest reload。
4. 尚未审计 Google Drive 同步目录中的非空 `colab_env_report.json`、`drive_mount_report.json`、`local_output_sync_report.json`、`manifest.json` 和 `reload_smoke_record.jsonl`。

### stage05 当前验证结果

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/colab_drive_entry.py` | pass, local_manifest_count=7, mirrored_file_count=18, reload_decision=pass |
| `pytest tests/constraints/test_notebook_entrypoint_contract.py -q` | pass, 8 passed |
| `pytest tests/functional/test_colab_drive_workflow_helpers.py -q` | pass, 6 passed |
| `pytest -q` | pass, 43 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |
