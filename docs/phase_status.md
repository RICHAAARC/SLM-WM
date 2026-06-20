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
| phase_status | `completed` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/core_package_boundary_freeze/manifest.local.json`; `outputs/algorithm_primitives/manifest.local.json`; `outputs/core_method_synthetic_smoke/manifest.local.json`; `outputs/sd_runtime_adapter/manifest.local.json`; `outputs/real_sd_runtime_probe_package_20260620t10451781952321z_b2be25c.zip`; `outputs/minimal_latent_injection_package_20260620t10181781950721z_b2be25c.zip` |
| expected_output_manifest | `outputs/colab_drive_workflow/manifest.local.json`; `GoogleDrive/SLM/colab_drive_workflow/manifest.json`; `outputs/colab_drive_workflow-20260620T114217Z-3-001.zip` |
| expected_outputs | `paper_workflow/colab_drive_cold_start_smoke.ipynb`; `paper_workflow/drive_manifest_reload_smoke.ipynb`; `paper_workflow/colab_utils/drive_paths.py`; `paper_workflow/colab_utils/dependency_check.py`; `paper_workflow/colab_utils/mount_drive.py`; `paper_workflow/colab_utils/runtime_setup.py`; `paper_workflow/colab_utils/manifest_io.py`; `paper_workflow/colab_utils/drive_workflow.py`; `scripts/colab_drive_entry.py`; `scripts/sync_local_outputs_to_drive.py`; `scripts/write_workflow_manifest.py`; `scripts/verify_drive_artifacts.py`; `outputs/colab_drive_workflow/colab_env_report.json`; `outputs/colab_drive_workflow/drive_mount_report.json`; `outputs/colab_drive_workflow/cold_start_smoke_record.jsonl`; `outputs/colab_drive_workflow/reload_smoke_record.jsonl`; `outputs/colab_drive_workflow/local_output_sync_report.json`; `outputs/colab_drive_workflow/manifest.local.json` |
| blocking_items | 无。 |
| fallback_path | 若 Colab 无法挂载 Drive, 本地命令仅写入 `outputs/colab_drive_workflow/drive_mirror/` 镜像目录并记录 `unsupported_reason`; 该镜像不得支持正式论文 claim。 |
| invariants | Notebook 只作为入口; Drive manifest、镜像与重载校验逻辑位于 repository helper 和 scripts; `main/` 不依赖 Colab、Drive 或 Notebook。 |
| next_stage_entry | Drive manifest 在 Colab 中生成, 且非空输入登记、镜像和 reload 校验均通过; 可进入 `stage_06_prompt_split_records_protocol`。 |

### stage05 已完成内容

1. 新增 Colab Drive workflow helper, 将路径解析、依赖快照、Drive 挂载报告、manifest 读写、本地 outputs 镜像和 reload 校验分离到 `paper_workflow/colab_utils/` 下的语义化模块。
2. 新增 `scripts/colab_drive_entry.py`、`scripts/sync_local_outputs_to_drive.py`、`scripts/write_workflow_manifest.py` 和 `scripts/verify_drive_artifacts.py`, 作为 Notebook 可调用的仓库入口。
3. 新增 `paper_workflow/colab_drive_cold_start_smoke.ipynb` 与 `paper_workflow/drive_manifest_reload_smoke.ipynb`, 两个 Notebook 均不保存执行输出, 且只调用 repository helper。
4. 新增轻量测试覆盖本地 outputs 镜像、manifest 写入、reload 校验、本地输出目录约束、Drive 挂载跳过报告和依赖快照非 claim 边界。
5. 本地执行 `python scripts/colab_drive_entry.py` 已在 `outputs/colab_drive_workflow/` 生成可审计 smoke 产物, 并验证本地镜像 reload 通过。
6. 已修正 Colab 冷启动输入边界: 若 clone 后本地 `outputs/` 为空, workflow 会登记 Google Drive 中已有的 `SLM/real_sd_runtime_probe/` 与 `SLM/minimal_diffusion_latent_injection/` 真实运行产物, 而不是把空 manifest 误判为有效证据。
7. 已审计 `outputs/colab_drive_workflow-20260620T114217Z-3-001.zip`; ZIP 完整性通过, SHA-256 为 `427f01ed221c26cc1ee319c6a45ffdd9ab35caccf96541b741af872dab0fcb98`。
8. 该结果包中 `metadata.workflow_decision=pass`, `reload_decision=pass`, `verified_file_count=2`, `missing_input_count=0`, `digest_mismatch_count=0`。
9. 该结果包登记了 Google Drive 中已有的前序真实产物: `SLM/minimal_diffusion_latent_injection/minimal_latent_injection_package_20260620t10181781950721z_b2be25c.zip` 和 `SLM/real_sd_runtime_probe/real_sd_runtime_probe_package_20260620t10451781952321z_b2be25c.zip`。
10. `docs/field_registry.md` 已登记 Colab Drive workflow 新增字段。

### stage05 完成边界

1. 本阶段完成的是 Colab 与 Google Drive 之间的非空前序产物登记、镜像和重载校验。
2. 当前 `supports_paper_claim=false` 的边界保持不变; 这些结果只作为 workflow provenance, 不直接作为论文中的 detection、robustness 或 fixed-FPR 结论。
3. `drive_mount_report.json` 中 `mount_decision=skipped`、`mounted=true`、`unsupported_reason=mount_not_requested` 表示 Notebook 已预先挂载 Drive, helper 未重复执行挂载动作, 不构成阻断项。

### stage05 验证结果

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/colab_drive_entry.py` | pass, local_manifest_count=7, mirrored_file_count=18, reload_decision=pass |
| `pytest tests/constraints/test_notebook_entrypoint_contract.py -q` | pass, 8 passed |
| `pytest tests/functional/test_colab_drive_workflow_helpers.py -q` | pass, 6 passed |
| `pytest -q` | pass, 43 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |
| `outputs/colab_drive_workflow-20260620T114217Z-3-001.zip` | pass, SHA-256 `427f01ed221c26cc1ee319c6a45ffdd9ab35caccf96541b741af872dab0fcb98` |

## stage_06_prompt_split_records_protocol

| item | value |
| --- | --- |
| construction_unit_name | `stage_06_prompt_split_records_protocol` |
| phase_status | `completed` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/colab_drive_workflow-20260620T114217Z-3-001.zip`; `outputs/prompts.zip`; `configs/paper_main_probe_prompts.txt`; `configs/paper_main_pilot_prompts.txt`; `configs/paper_main_full_prompts.txt` |
| expected_output_manifest | `outputs/prompt_event_protocol/manifest.local.json` |
| expected_outputs | `configs/paper_main_probe_prompts.txt`; `configs/paper_main_pilot_prompts.txt`; `configs/paper_main_full_prompts.txt`; `outputs/prompt_event_protocol/prompt_records.jsonl`; `outputs/prompt_event_protocol/event_records.jsonl`; `outputs/prompt_event_protocol/prompt_manifest.json`; `outputs/prompt_event_protocol/split_manifest.json`; `outputs/prompt_event_protocol/event_protocol_manifest.json`; `outputs/prompt_event_protocol/prompt_statistics.json`; `outputs/prompt_event_protocol/manifest.local.json` |
| blocking_items | 无。 |
| fallback_path | 若 prompt bank、prompt 配置、前序 Drive workflow 证据或字段登记缺失, 停止推进并修复协议输入; 不得手工改写 prompt_id 或 event_id。 |
| invariants | records 只能由 `experiments/` 或 `scripts/` 写出; `main/` 不写 records; calibration 与 test 不共享 prompt_id; 当前协议产物不支持正式论文 claim。 |
| next_stage_entry | prompt、split、sample role 和 event manifest 均可复现, 可进入 `stage_07_semantic_mask_risk_field_safe_subspace`。 |

### stage06 已完成内容

1. 使用 `outputs/prompts.zip` 重新生成项目 prompt 配置; 输入 zip 的 SHA-256 为 `197cb1c40d2ff131e761c70b56f41164c4e7ad168f35a63cb1c2bbe5c46e1eee`。
2. 新增 `scripts/import_prompt_bank.py`, 从外部 prompt bank 读取 probe、pilot 和 full 三组 prompt, 统一规范化空白, 并替换仓库治理不允许写入配置正文的过程标记词。
3. `configs/paper_main_probe_prompts.txt`、`configs/paper_main_pilot_prompts.txt` 和 `configs/paper_main_full_prompts.txt` 当前分别包含 10、600 和 6000 条 prompt。
4. prompt bank 导入过程中, pilot 与 full 各有 1 条 prompt 因命名治理约束被语义等价替换为 `concert platform` 表达, probe 无需替换。
5. `experiments/protocol/prompts.py` 负责 prompt 文本规范化、语义标签派生、风险配置派生、稳定 `prompt_id` 生成, 并在 prompt record 中保留 split 字段。
6. `experiments/protocol/splits.py` 固定 `dev`、`calibration`、`test` 三个 split, 并按 prompt set 与 risk profile 分层后进行稳定划分, 避免 calibration/test 在 prompt_id 层面交叉。
7. `experiments/protocol/events.py` 由 prompt 与 sample role 构造 `positive_source`、`clean_negative` 和 `attacked_negative` 三类事件, 并生成稳定 `event_id`。
8. `experiments/protocol/records.py` 与 `experiments/protocol/calibration.py` 负责 JSONL 写出、轻量唯一性校验和协议统计摘要。
9. `scripts/write_prompt_event_protocol.py` 将 prompt records、event records、prompt manifest、split manifest、event protocol manifest、prompt statistics 和本地 manifest 写入 `outputs/prompt_event_protocol/`, 并在 manifest 输入中登记 `outputs/prompts.zip`。
10. 当前协议输出 `prompt_count=6610`, `event_count=19830`, `split_counts` 为 `dev=659`、`calibration=2970`、`test=2981`, 三个 sample role 各 6610 条事件。
11. 当前协议输出 `calibration_test_disjoint=true`, `protocol_decision=pass`, `supports_paper_claim=false`。
12. `docs/field_registry.md` 已登记 prompt、split、event、sample role、protocol manifest、prompt bank 导入摘要和统计摘要相关字段。
13. 新增 `tests/functional/test_prompt_bank_import.py` 与 `tests/functional/test_prompt_event_protocol.py`, 覆盖 prompt bank 导入、稳定 ID、split 无交叉、受治理输出目录和 manifest 写出边界。

### stage06 完成边界

1. 本阶段完成的是论文实验协议索引, 不是正式检测指标、鲁棒性指标或 fixed-FPR 结论。
2. `prompt_records.jsonl` 和 `event_records.jsonl` 可以作为后续实验 runner 的输入索引, 但不得直接作为论文 claim 支撑证据。
3. `calibration` 与 `test` 的统计边界在 prompt_id 层面保持无交叉; 后续阈值校准必须继续沿用这一边界。
4. `dev` split 仅用于开发和链路检查, 不得用于冻结 fixed-FPR 阈值或 rescue gate。
5. `outputs/prompts.zip` 是本次 prompt bank 导入来源, 不属于应提交到 Git 的仓库内容。

### stage06 验证结果

| command | result |
| --- | --- |
| `python scripts/import_prompt_bank.py` | pass, probe=10, pilot=600, full=6000, sanitized counts probe=0、pilot=1、full=1 |
| `python scripts/write_prompt_event_protocol.py` | pass, prompt_count=6610, event_count=19830, calibration_test_disjoint=true |
| `pytest tests/functional/test_prompt_bank_import.py tests/functional/test_prompt_event_protocol.py -q` | pass, 5 passed |
| `pytest -q` | pass, 50 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |
