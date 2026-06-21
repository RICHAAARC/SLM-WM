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


## stage_07_semantic_mask_risk_field_safe_subspace

| item | value |
| --- | --- |
| construction_unit_name | `stage_07_semantic_mask_risk_field_safe_subspace` |
| phase_status | `completed` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/prompt_event_protocol/manifest.local.json`; `outputs/prompt_event_protocol/prompt_records.jsonl`; `outputs/real_sd_runtime_probe_package_20260620t10451781952321z_b2be25c.zip`; `outputs/minimal_latent_injection_package_20260620t10181781950721z_b2be25c.zip` |
| expected_output_manifest | `outputs/semantic_subspace/manifest.local.json` |
| expected_outputs | `outputs/semantic_subspace/semantic_route_records.jsonl`; `outputs/semantic_subspace/subspace_plan_records.jsonl`; `outputs/semantic_subspace/mask_projection_reports/mask_projection_reports.jsonl`; `outputs/semantic_subspace/basis_digests.json`; `outputs/semantic_subspace/semantic_subspace_summary.json`; `outputs/semantic_subspace/manifest.local.json` |
| blocking_items | 无。 |
| fallback_path | 若真实 latent trace 摘要包不可用, 使用确定性 lightweight latent reference 继续验证语义掩码影响 feature operator 与 basis, 且保持 `supports_paper_claim=false`。 |
| invariants | saliency、segmentation 和 SD attention capture 不进入 `main/`; `main/` 只接收标准化 mask、latent mask 和 feature tensor; 无语义掩码路径只作为消融或诊断路径。 |
| next_stage_entry | semantic route、mask projection、approximate JVP 和 safe basis 均有 digest, 且语义掩码会改变 basis; 可进入 `stage_08_lf_hf_content_carriers`。 |

### stage07 已完成内容

1. 新增 `main/methods/semantic/risk_field.py`, 实现标准化语义、纹理、稳定性和显著性向量到风险场与承载预算的映射。
2. 新增 `main/methods/semantic/latent_mask.py`, 实现 `M_z = Pi_{x->z}(M_x)` 的轻量投影和 `M_z * z_t` 掩码作用。
3. 新增 `main/methods/semantic/routing.py`, 根据风险场与 latent mask 生成 LF、HF 和 attention 候选轴路由。
4. 新增 `main/methods/subspace/trajectory_features.py`, 实现 `P^T vec(Norm(M_z * z_t))` 的轻量 feature operator。
5. 新增 `main/methods/subspace/jvp_estimator.py`, 用相邻差分实现可审计 approximate JVP 摘要。
6. 新增 `main/methods/subspace/safe_basis.py` 和 `main/methods/subspace/route_projection.py`, 实现 semantic safe basis、no semantic mask、global nullspace 和 diagnostic basis 四种可运行基底策略, 并生成 route projection digest。
7. 新增 `scripts/write_semantic_subspace_outputs.py`, 从 prompt protocol records 与真实 SD3.5 latent trace 摘要包中构造 semantic route records、subspace plan records、mask projection reports、basis digests、summary 和 manifest。
8. 当前 `outputs/semantic_subspace/semantic_subspace_summary.json` 显示 `semantic_route_record_count=6610`, `subspace_plan_record_count=6610`, `mask_projection_report_count=6610`, `unique_route_digest_count=6610`, `semantic_mask_changed_basis_count=6610`, `protocol_decision=pass`。
9. 当前 `supports_paper_claim=false` 边界保持不变; 本阶段产物证明机制链路可审计, 不直接作为 detection 或 fixed-FPR 论文结论。
10. 新增 `tests/functional/test_semantic_subspace.py`, 覆盖不同语义掩码产生不同 route、关闭语义掩码改变 basis、消融基底可运行、脚本输出 manifest 和输出目录约束。
11. `docs/field_registry.md` 已登记本阶段新增 route、mask、feature operator、approximate JVP、basis strategy、basis digest 和 summary 字段。

### stage07 完成边界

1. 本阶段完成的是核心方法层的标准化 semantic mask、risk field、feature operator、approximate JVP 和 semantic safe basis, 不是正式 SD attention capture 或论文主实验统计。
2. runtime 层仍负责 saliency、segmentation、predicted x0 与 attention capture; core 方法层不加载模型权重。
3. `no_semantic_mask`、`global_nullspace` 和 `diagnostic_basis` 仅作为消融或诊断路径, 不得伪装成 SLM-WM 主方法。
4. 后续 LF/HF carrier 构建应读取 `subspace_plan_records.jsonl` 与 `basis_digests.json`, 并保留 calibration/test split 边界。

### stage07 验证结果

| command | result |
| --- | --- |
| `python scripts/write_semantic_subspace_outputs.py` | pass, semantic_route_record_count=6610, subspace_plan_record_count=6610, semantic_mask_changed_basis_count=6610 |
| `pytest tests/functional/test_semantic_subspace.py -q` | pass, 4 passed |
| `pytest -q` | pass, 54 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_08_lf_hf_content_carriers

| item | value |
| --- | --- |
| construction_unit_name | `stage_08_lf_hf_content_carriers` |
| phase_status | `completed` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/semantic_subspace/manifest.local.json`; `outputs/semantic_subspace/subspace_plan_records.jsonl`; `outputs/semantic_subspace/semantic_route_records.jsonl`; `outputs/minimal_latent_injection_package_20260620t10181781950721z_b2be25c.zip` |
| expected_output_manifest | `outputs/content_carriers/manifest.local.json` |
| expected_outputs | `outputs/content_carriers/content_detection_records.jsonl`; `outputs/content_carriers/lf_hf_score_table.csv`; `outputs/content_carriers/paired_quality_metrics.csv`; `outputs/content_carriers/content_score_distribution.csv`; `outputs/content_carriers/content_carrier_summary.json`; `outputs/content_carriers/manifest.local.json` |
| blocking_items | 无。 |
| fallback_path | 若语义子空间 records 或真实最小注入质量包不可用, 停止推进并修复前序输入; 不允许用手工阈值投票或未登记文件替代内容分数链路。 |
| invariants | LF 为内容主证据, HF 仅为补充证据; 不为 LF/HF 分别设置独立正判阈值后投票; 当前产物保持 `supports_paper_claim=false`, 不能直接作为论文 fixed-FPR 或 robustness 结论。 |
| next_stage_entry | 内容载体 records、统一内容分数、机制开关摘要与 manifest 均可重建, 可进入 `stage_09_self_attention_graph_geometry`。 |

### stage08 已完成内容

1. 新增 `main/methods/carrier/lf.py`, 实现稳定 LF 内容模板、低频平滑和 latent update 派生。
2. 新增 `main/methods/carrier/hf.py`, 实现稳定 HF 内容模板、tail truncation 和关闭 tail truncation 的机制路径。
3. 新增 `main/methods/carrier/compose.py`, 统一组合 `full_content_chain`、`lf_only`、`hf_only`、`no_hf`、`no_tail_truncation` 和 `no_lf` 六类内容机制开关。
4. 新增 `main/methods/detection/scores.py` 和 `main/methods/detection/fusion.py`, 实现 `s_c = lambda_LF s_LF + lambda_HF s_HF`, 且 `lambda_LF > lambda_HF`, `used_independent_branch_vote=false`。
5. 新增 `scripts/write_content_carrier_outputs.py`, 从语义子空间 records 与最小 latent injection 质量包重建内容检测 records、LF/HF score table、paired quality metrics、score distribution、summary 和 manifest。
6. 当前 `outputs/content_carriers/content_carrier_summary.json` 显示 `content_detection_record_count=19830`, `score_count=19830`, `fixed_fpr_ready=true`, `used_independent_branch_vote=false`, `protocol_decision=pass`, `supports_paper_claim=false`。
7. 新增 `tests/functional/test_content_carriers.py`, 覆盖 LF/HF 载体摘要稳定性、机制开关真实改变 update、统一内容分数 fixed-FPR 边界、写出脚本 manifest 和 outputs 目录约束。
8. `docs/field_registry.md` 已登记内容载体、内容分数、机制开关、score distribution 和 summary 相关字段。

### stage08 完成边界

1. 本阶段完成的是核心方法层 LF/HF 内容载体和统一内容分数机制, 不是最终论文阈值校准、attack matrix 或正式固定 FPR 实验结论。
2. `fixed_fpr_ready=true` 仅表示内容分数记录保留了可进入后续 fixed-FPR calibration 的统计形态; 真实阈值冻结必须继续使用 calibration split, 并且不能与 test split 混用。
3. `rescue` 不在本阶段触发正判, 后续几何 rescue 必须在同一 fixed-FPR 统计边界内审计, 不能新增独立阳性通道。
4. LF-only、HF-only、No-HF、No-tail-truncation 和 No-LF 均作为机制诊断或消融路径, 不得伪装为 SLM-WM 主方法。

### stage08 验证结果

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/write_content_carrier_outputs.py` | pass, content_detection_record_count=19830, score_count=19830, protocol_decision=pass |
| `pytest tests/functional/test_content_carriers.py -q` | pass, 5 passed |
| `pytest -q` | pass, 59 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |


## stage_09_self_attention_graph_geometry

| item | value |
| --- | --- |
| construction_unit_name | `stage_09_self_attention_graph_geometry` |
| phase_status | `real_capture_workflow_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/content_carriers/manifest.local.json`; `outputs/content_carriers/content_carrier_summary.json`; `outputs/sd_runtime_adapter/manifest.local.json`; `outputs/sd_runtime_adapter/attention_capture_records.jsonl`; Colab 运行后可替换为 `outputs/real_attention_geometry/real_attention_capture_records.jsonl` |
| expected_output_manifest | `outputs/attention_geometry/manifest.local.json`; `outputs/real_attention_geometry/real_attention_manifest.local.json`; `outputs/real_attention_geometry/attention_geometry_archive_manifest.local.json` |
| expected_outputs | `paper_workflow/attention_geometry_capture_run.ipynb`; `paper_workflow/colab_utils/attention_geometry_capture.py`; `outputs/real_attention_geometry/real_attention_capture_records.jsonl`; `outputs/real_attention_geometry/real_attention_capture_summary.json`; `outputs/real_attention_geometry/real_attention_environment_report.json`; `outputs/attention_geometry/attention_graph_records.jsonl`; `outputs/attention_geometry/geometry_evidence_records.jsonl`; `outputs/attention_geometry/attention_relation_consistency.csv`; `outputs/attention_geometry/geometry_evidence_summary.json`; `outputs/attention_geometry/manifest.local.json`; `outputs/real_attention_geometry/attention_geometry_package_<utc>_<short_commit>.zip`; `GoogleDrive/SLM/attention_geometry/attention_geometry_package_<utc>_<short_commit>.zip` |
| blocking_items | 本地环境无 GPU 和真实 SD3.5 Medium 权重, 因此本地默认产物仍来自前序 synthetic attention capture; 真实 `attention_geometry_ready=true` 需要运行 Colab Notebook 并回传结果包审计。 |
| fallback_path | 若真实 attention hook 不可用, Notebook 会让 `attention_geometry_ready` 断言失败, 并保留失败 summary; 不允许把 synthetic attention capture 改写为真实 capture。 |
| invariants | 几何证据只记录可靠性统计; `direct_positive_decision=false`; 只有所有 capture records 均为真实可审计记录、`real_attention_capture_count>0` 且 `unsupported_capture_count=0` 时, `attention_geometry_ready` 才能为 true。 |
| next_stage_entry | 运行并审计 `attention_geometry_package_<utc>_<short_commit>.zip` 后, 若 summary 显示 `attention_geometry_ready=true`, 才允许把真实 attention-relative latent update 作为后续方法实现输入。 |

### stage09 已完成内容

1. 新增 `main/methods/geometry/attention_graph_types.py`, 定义 attention graph record 与 geometry evidence record 的 typed object。
2. 新增 `main/methods/geometry/recovery.py`, 实现 `softmax(QK^T / sqrt(d))`、稳定 token 集选择、相对关系抽取、anchor graph digest 和几何恢复统计。
3. 更新 `experiments/runtime/diffusion/attention_capture.py`, 增加从 Q/K 向量构造可审计 attention capture record 的纯函数入口, 保持真实 runtime hook 与核心方法层解耦。
4. 更新 `scripts/write_attention_geometry_outputs.py`, 支持通过 `--attention-records-path` 指向真实 Colab capture records; 只有 records 全部无 `unsupported_reason`、`metadata.capture_is_synthetic=false`、包含有界 `attention_matrix_preview`, 且 `real_attention_capture_count>0`, summary 中 `attention_geometry_ready` 才能为 true。
5. 新增 `paper_workflow/colab_utils/attention_geometry_capture.py`, 在真实 SD3.5 Medium pipeline 的 transformer attention 模块上注册 hook, 从真实 hidden states 构造有界可审计 attention map, 写出真实 capture records, 并调用几何重建脚本刷新 `outputs/attention_geometry/`。
6. 新增 `paper_workflow/attention_geometry_capture_run.ipynb`, 支持 Colab 冷启动: 挂载 Google Drive、拉取代码、安装依赖、读取 `HF_TOKEN`、加载 SD3.5 Medium、执行真实 attention capture、断言 `attention_geometry_ready=true`, 并打包镜像到 `GoogleDrive/SLM/attention_geometry/`。
7. 打包逻辑会把真实 capture records、真实 capture summary、运行环境报告、attention geometry records、summary、manifest、输入核对 manifest 等文件纳入 zip, 避免只上传单一 summary。
8. 新增和更新测试覆盖 Q/K 注意力公式、真实 preview 矩阵 ready gate、Notebook 入口契约、打包镜像契约和 outputs 目录约束。
9. `docs/field_registry.md` 已登记真实 attention map preview、attention records path、捕获 tensor 形状、几何 manifest / summary 路径和压缩包输入 manifest 相关字段。

### stage09 当前完成边界

1. 本地默认 `outputs/attention_geometry/geometry_evidence_summary.json` 仍使用前序 synthetic attention capture, 因此 `real_attention_capture_count=0`, `unsupported_capture_count=4`, `attention_geometry_ready=false`。
2. 新 Notebook 的完成判定是强断言: 若真实 SD3.5 Medium 推理没有生成无 unsupported reason 的 capture records, Notebook 会失败, 不会伪造 ready 状态。
3. `attention_geometry_ready=true` 的唯一有效路径是: Colab GPU 运行真实 SD3.5 Medium -> 写出 `outputs/real_attention_geometry/real_attention_capture_records.jsonl` -> 用该 records 重建 `outputs/attention_geometry/` -> summary 满足所有 records 均为真实可审计记录、`real_attention_capture_count>0` 且 `unsupported_capture_count=0`。
4. 几何证据仍不得直接给出 positive 判定; 后续真实 attention-relative latent update 必须读取已经 ready 的 attention geometry 产物。

### stage09 验证结果

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/write_attention_geometry_outputs.py` | pass, 默认 synthetic 输入下 `attention_geometry_ready=false` |
| `pytest tests/functional/test_attention_geometry.py -q` | pass, 5 passed |
| `pytest tests/constraints/test_notebook_entrypoint_contract.py -q` | pass, 10 passed |
| `pytest -q` | pass, 66 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |


## stage_10_attention_relative_latent_update

| item | value |
| --- | --- |
| construction_unit_name | `stage_10_attention_relative_latent_update` |
| phase_status | `real_injection_audited` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/attention_geometry_package_20260620t13511781963497z_b237bb3.zip`; `outputs/semantic_subspace/manifest.local.json`; `outputs/content_carriers/manifest.local.json` |
| expected_output_manifest | `outputs/attention_latent_update/manifest.local.json`; Colab 运行后为 `outputs/attention_latent_injection/attention_latent_injection_manifest.local.json` |
| expected_outputs | `outputs/attention_latent_update/attention_carrier_records.jsonl`; `outputs/attention_latent_update/attention_update_stability.csv`; `outputs/attention_latent_update/attention_update_quality_metrics.csv`; `outputs/attention_latent_update/attention_update_summary.json`; `outputs/attention_latent_update/manifest.local.json`; `paper_workflow/attention_latent_injection_run.ipynb`; `paper_workflow/colab_utils/attention_latent_injection.py`; Colab 运行后为 `outputs/attention_latent_injection/attention_latent_injection_result.json`; `outputs/attention_latent_injection/attention_latent_update_records.jsonl`; `outputs/attention_latent_injection/attention_paired_quality_metrics.csv`; `outputs/attention_latent_injection/attention_injection_environment_report.json`; `outputs/attention_latent_injection/attention_latent_injection_manifest.local.json`; `GoogleDrive/SLM/attention_latent_injection/attention_latent_injection_package_<utc>_<short_commit>.zip` |
| blocking_items | 真实 Colab GPU 结果包已回传并完成本地审计, `image_quality_metrics_ready=true`; 但 `full_method_claim_ready=false`, 因为 fixed-FPR 与 rescue 统计边界尚未冻结。 |
| fallback_path | 若几何证据不可靠或 update 稳定性边界不满足, carrier 自动降级为 `evidence_only`, 只保留几何证据, 不写入 Full 方法主张。 |
| invariants | 几何链不直接 positive; attention update 只在 `attention_geometry_ready=true` 且几何证据可靠时 active; 本地质量仅为 proxy, 不替代真实 paired image 质量指标。 |
| next_stage_entry | 已允许把真实 attention latent injection 包作为 same-threshold geometric rescue 的输入; `full_method_claim_ready` 仍需后续 fixed-FPR 与 rescue 链路共同确认。 |

### stage10 已完成内容

1. 新增 `main/methods/carrier/attention.py`, 定义 `AttentionRelativeCarrier`, 关系损失、关系梯度投影、active update 与 `evidence_only` 降级边界。
2. 更新 `main/methods/carrier/__init__.py`, 导出 attention-relative carrier 方法入口。
3. 新增 `scripts/write_attention_latent_update_outputs.py`, 可从 ready attention geometry zip 或本地 ready 目录读取图与几何证据, 结合 semantic safe subspace records 生成 attention carrier records、强度稳定性表、质量代理表、summary 和 manifest。
4. 新增 `tests/functional/test_attention_latent_update.py`, 覆盖可靠几何证据触发 active update、不可靠几何证据降级为 `evidence_only`, 以及脚本从 ready geometry 包重建受治理产物。
5. `docs/field_registry.md` 已登记 attention-relative carrier、关系损失、强度稳定性、质量代理和 Full 方法 claim 边界相关字段。
6. 新增 `paper_workflow/colab_utils/attention_latent_injection.py`, 支持从 Google Drive 读取最新 ready attention geometry 包, 重建 prompt / semantic / content / attention update 输入链, 选择 active carrier, 并在真实 SD3.5 latent callback 中执行 attention-relative update。
7. 新增 `paper_workflow/attention_latent_injection_run.ipynb`, 支持 Colab 冷启动、挂载 Drive、读取 `HF_TOKEN`、检查 GPU、执行真实 attention latent injection、强断言真实 latent update 与质量指标存在, 并打包镜像到 `GoogleDrive/SLM/attention_latent_injection/`。
8. 更新 `tests/constraints/test_notebook_entrypoint_contract.py`, 覆盖新 Notebook 入口委托、无执行输出和真实 injection 产物打包镜像。

### stage10 当前产物摘要

1. 当前输入使用真实 SD3.5 Medium attention geometry 包 `outputs/attention_geometry_package_20260620t13511781963497z_b237bb3.zip`, 其中 `attention_geometry_ready=true`。
2. `outputs/attention_latent_update/attention_update_summary.json` 显示 `attention_carrier_record_count=64`, `active_update_count=16`, `evidence_only_count=48`, `attention_update_stable_count=16`, `protocol_decision=pass`。
3. 已审计真实结果包 `outputs/attention_latent_injection_package_20260620t14471781966861z_8199dbc.zip`, SHA256 为 `c34577f71e549b6cf0dda43ed3dc8a582a45073f36b269d40cf454d598402b48`。
4. 真实结果包显示 `run_decision=pass`, `latent_update_count=3`, 注入步为 `6, 10, 14`, `image_quality_metrics_ready=true`, PSNR 为 `35.18531747817406`, SSIM 为 `0.9976578187804996`。
5. `full_method_claim_ready=false` 仍保持不变, 表示尚不能声称 fixed-FPR 完整方法主张已经完成。

### stage10 验证结果

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/write_attention_latent_update_outputs.py --attention-geometry-package-path outputs/attention_geometry_package_20260620t13511781963497z_b237bb3.zip` | pass, `active_update_count=16`, `evidence_only_count=48` |
| `pytest tests/functional/test_attention_latent_update.py -q` | pass, 3 passed |
| `pytest tests/constraints/test_notebook_entrypoint_contract.py -q` | pass, 11 passed |
| `pytest -q` | pass, 70 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_11_same_threshold_geometric_rescue

| item | value |
| --- | --- |
| construction_unit_name | `stage_11_same_threshold_geometric_rescue` |
| phase_status | `local_rescue_protocol_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/content_carriers/manifest.local.json`; `outputs/attention_latent_injection_package_20260620t14471781966861z_8199dbc.zip` |
| expected_output_manifest | `outputs/geometric_rescue/manifest.local.json` |
| expected_outputs | `outputs/geometric_rescue/aligned_detection_records.jsonl`; `outputs/geometric_rescue/rescue_metrics_summary.csv`; `outputs/geometric_rescue/content_failed_subset_summary.csv`; `outputs/geometric_rescue/geometry_rescue_audit.json`; `outputs/geometric_rescue/manifest.local.json` |
| blocking_items | 当前为本地受治理机制记录, aligned content score 仍是由几何可靠性、边界距离和样本角色派生的轻量代理; 后续若要形成正式论文主张, 需要在真实 aligned latent 上重新运行内容检测, 并在 calibration split 中冻结完整 evidence-level 协议。 |
| fallback_path | 若几何不可靠、内容分数不在边界失败窗口, 或 fail reason 不属于 `geometry_suspected` / `low_confidence`, 则不触发 rescue; `geo_direct_positive_audit` 只作为反例审计, 不进入正式方法。 |
| invariants | 几何链不得直接 positive; rescue 后仍复用同一个 `content_threshold=0.75`; 当前 `supports_paper_claim=false` 且 `full_method_claim_ready=false`。 |
| next_stage_entry | 可以进入 fixed-FPR calibration 与指标冻结构建; 下一步必须同时审计 raw content clean FPR、rescue 后 clean negative FPR 和 rescue 后 attacked negative FPR。 |

### stage11 已完成内容

1. 更新 `main/methods/detection/fusion.py`, 新增 `SameThresholdRescueConfig`、`GeometricRescueDecisionRecord`、`decide_same_threshold_geometric_rescue` 与消融模式下的几何可靠性选择逻辑。
2. 更新 `main/methods/geometry/recovery.py`, 新增 `estimate_aligned_content_score` 轻量代理入口, 用于本地受治理记录; 后续真实 aligned latent 内容检测可替换该入口。
3. 新增 `scripts/write_geometric_rescue_outputs.py`, 从真实 attention latent injection 包和内容检测 records 重建 aligned detection records、rescue metrics、内容失败子集摘要、geometry rescue audit 和 manifest。
4. 新增 `tests/functional/test_geometric_rescue.py`, 覆盖同阈值 rescue、no-rescue 阻断、geo-direct-positive 反例审计以及受治理产物重建。
5. `docs/field_registry.md` 已登记 aligned detection、rescue 消融、rescue gain、clean / attacked FPR 与 geo-direct-positive audit 字段。

### stage11 当前产物摘要

1. `outputs/geometric_rescue/geometry_rescue_audit.json` 显示 `protocol_decision=pass`, `attention_geometry_ready=true`, `image_quality_metrics_ready=true`, `latent_update_count=3`。
2. 当前本地采样 `max_content_records=96`, 生成 `aligned_detection_record_count=576`, 其中 full-rescue 模式记录数为 `96`, `full_rescue_applied_count=1`。
3. full-rescue 模式下 `raw_content_clean_fpr=0.0`, `evidence_clean_fpr=0.0`, `evidence_attacked_fpr=0.03125`; 这些统计只作为后续 fixed-FPR 构建输入, 不能替代正式 calibration。
4. `geo_direct_positive_audit_rate=0.5625` 显示几何直接判正对 clean negative 具有明显 FPR 风险, 因此该分支继续保持为反例审计, 不进入正式方法。
5. 所有新增产物保持 `supports_paper_claim=false`, `full_method_claim_ready=false`。

### stage11 验证结果

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/write_geometric_rescue_outputs.py --attention-injection-package-path outputs/attention_latent_injection_package_20260620t14471781966861z_8199dbc.zip` | pass, `full_rescue_applied_count=1`, `evidence_clean_fpr=0.0`, `evidence_attacked_fpr=0.03125` |
| `pytest tests/functional/test_geometric_rescue.py -q` | pass, 2 passed |
| `pytest -q` | pass, 72 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_12_threshold_calibration_metrics

| item | value |
| --- | --- |
| construction_unit_name | `stage_12_threshold_calibration_metrics` |
| phase_status | `local_calibration_protocol_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/geometric_rescue/manifest.local.json`; `outputs/geometric_rescue/geometry_rescue_audit.json`; `outputs/geometric_rescue/aligned_detection_records.jsonl` |
| expected_output_manifest | `outputs/threshold_calibration/manifest.local.json` |
| expected_outputs | `outputs/threshold_calibration/calibration_thresholds.json`; `outputs/threshold_calibration/fixed_fpr_operating_points.csv`; `outputs/threshold_calibration/standard_watermark_metrics.csv`; `outputs/threshold_calibration/quality_metrics_summary.csv`; `outputs/threshold_calibration/roc_curve_points.csv`; `outputs/threshold_calibration/det_curve_points.csv`; `outputs/threshold_calibration/score_distribution_table.csv`; `outputs/threshold_calibration/threshold_degeneracy_report.json`; `outputs/threshold_calibration/rescue_fpr_audit.csv`; `outputs/threshold_calibration/manifest.local.json` |
| blocking_items | 当前 fixed-FPR 框架可由 governed records 重建, 但 `aligned_content_score` 仍来自本地代理; 最新真实 aligned rescoring 包已向下游传播 PSNR、SSIM、MSE、MAE、LPIPS 与 CLIP score, FID / KID 仍是 dataset-level 未计算指标; 因此 `full_method_claim_ready=false`。 |
| fallback_path | 若 rescue 后 evidence-level FPR 超过目标 operating point, 只允许保留 raw content claim 或将完整系统 fixed-FPR 主张标记为 unsupported; 不允许只报告 raw content FPR。 |
| invariants | 内容阈值只由 calibration clean negative 冻结; test split 不参与调阈值; clean negative 与 attacked negative 分开审计; rescue window 与 fail reason gate 保持冻结。 |
| next_stage_entry | 可以进入攻击矩阵与再扩散攻击记录构建; 若要支撑论文级 fixed-FPR 主张, 仍需把真实 aligned latent 重判扩展到完整 calibration / test 规模, 并补齐 dataset-level FID / KID 与真实图像攻击闭环。 |

### stage12 已完成内容

1. 更新 `experiments/protocol/calibration.py`, 新增 `FixedFprCalibrationConfig`、`FixedFprThreshold`、fixed-FPR 阈值冻结、校准后判定、AUC、ROC / DET 与 score distribution 计算函数。
2. 更新 `experiments/protocol/__init__.py`, 导出 fixed-FPR 校准核心对象和函数。
3. 新增 `scripts/write_threshold_calibration_outputs.py`, 从 `outputs/geometric_rescue/` 记录重建 calibration thresholds、operating point、standard metrics、quality metrics、ROC / DET、score distribution、threshold degeneracy 和 rescue FPR audit。
4. 新增 `tests/functional/test_threshold_calibration.py`, 覆盖阈值只来自 calibration clean negative、clean negative 与 attacked negative 分开审计、unsupported 质量指标不伪装为论文证据。
5. `docs/field_registry.md` 已登记 fixed-FPR、operating point、AUC、ROC / DET、FPR audit、metric status 和阈值退化相关字段。

### stage12 当前产物摘要

1. `outputs/threshold_calibration/calibration_thresholds.json` 显示 `target_fpr=0.05`, `calibration_negative_count=14`, `observed_fpr=0.0`, `threshold_degenerate=false`, `threshold_value=0.5174190728458973`。
2. `outputs/threshold_calibration/fixed_fpr_operating_points.csv` 显示 `true_positive_rate=0.84375`, `raw_content_clean_fpr=0.03125`, `evidence_clean_fpr=0.03125`, `evidence_attacked_fpr=0.15625`。
3. `outputs/threshold_calibration/rescue_fpr_audit.csv` 显示 attacked negative 的 evidence-level FPR 超过 `target_fpr=0.05`, 因此完整系统 fixed-FPR 主张必须保持 unsupported。
4. `outputs/threshold_calibration/quality_metrics_summary.csv` 已由最新真实 aligned rescoring 包刷新: PSNR=`28.774532071397005`, SSIM=`0.9903153991736182`, MSE=`0.0013260099804028869`, MAE=`0.02013162337243557`, LPIPS=`0.03199240565299988`, `clip_score=0.3809072971343994`; FID / KID 仍保留 `dataset_level_metric_not_computed_in_pair_run`。
5. `outputs/threshold_calibration/threshold_degeneracy_report.json` 中 `raw_content_claim_ready=true`, 但 `full_method_claim_ready=false`, `unsupported_reason=aligned_content_score_local_proxy`。

### stage12 验证结果

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/write_threshold_calibration_outputs.py --aligned-rescoring-package-path outputs/aligned_rescoring_package_20260620t17281781976491z_b37b14f.zip` | pass, `aligned_rescoring_quality_metrics_ready=true`, `real_aligned_rescore_count=3`, `evidence_attacked_fpr=0.15625` |
| `pytest tests/functional/test_threshold_calibration.py -q` | pass, 3 passed |
| `pytest -q` | pass, 86 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_13_attack_matrix_regeneration

| item | value |
| --- | --- |
| construction_unit_name | `stage_13_attack_matrix_regeneration` |
| phase_status | `local_attack_matrix_protocol_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/geometric_rescue/manifest.local.json`; `outputs/threshold_calibration/manifest.local.json` |
| expected_output_manifest | `outputs/attack_matrix/manifest.local.json` |
| expected_outputs | `outputs/attack_matrix/attacked_images/`; `outputs/attack_matrix/attack_manifest.json`; `outputs/attack_matrix/attacked_image_registry.jsonl`; `outputs/attack_matrix/attack_detection_records.jsonl`; `outputs/attack_matrix/attack_family_metrics.csv`; `outputs/attack_matrix/attack_strength_curve.csv`; `outputs/attack_matrix/score_retention_by_attack.csv`; `outputs/attack_matrix/rescue_by_attack.csv`; `outputs/attack_matrix/manifest.local.json` |
| blocking_items | 当前常规攻击为 record-level proxy, 未生成真实 attacked image 文件; 再扩散攻击需要真实 GPU 图像重生成产物, 当前统一标记为 `unsupported`。 |
| fallback_path | 常规攻击只作为本地可重建协议与表格链路; 再扩散攻击保留配置、digest 和 unsupported reason, 不进入论文主张。 |
| invariants | 攻击后检测复用 `stage_12` 冻结的 fixed-FPR 阈值、rescue window 和 fail reason gate; clean negative 与 attacked negative 分开统计; `full_method_claim_ready=false`; `supports_paper_claim=false`。 |
| next_stage_entry | 可进入外部 baseline 对比与内部消融证据构建; 若要形成 robustness 主张, 需要用真实 attacked image 文件和真实再扩散攻击产物替换本地代理记录。 |

### stage13 已完成内容

1. 新增 `experiments/protocol/attacks.py`, 定义 `AttackConfig`、`AttackEvaluationBoundary`、`AttackDetectionRecord`、默认攻击矩阵配置、攻击配置摘要、record-level 攻击代理、attack family metrics、strength curve、score retention 和 rescue-by-attack 聚合函数。
2. 更新 `experiments/protocol/__init__.py`, 导出攻击矩阵协议对象和聚合函数。
3. 新增 `scripts/write_attack_matrix_outputs.py`, 从 `outputs/geometric_rescue/aligned_detection_records.jsonl`、`outputs/geometric_rescue/manifest.local.json`、`outputs/threshold_calibration/calibration_thresholds.json`、`outputs/threshold_calibration/threshold_degeneracy_report.json` 和 `outputs/threshold_calibration/manifest.local.json` 重建攻击矩阵产物。
4. 当前默认攻击矩阵覆盖 JPEG compression、Gaussian noise、Gaussian blur、resize、crop、rotation、crop-resize、composite geometric attacks, 同时登记 img2img regeneration、DDIM inversion + regeneration、SDEdit regeneration 和 diffusion purification。
5. 常规攻击配置写入 `probe`、`pilot` 和 `full_main` 资源档位; 再扩散攻击写入 `full_extra` 资源档位并保留 `real_gpu_attack_required` unsupported reason。
6. 新增 `tests/functional/test_attack_matrix.py`, 覆盖攻击配置摘要稳定性、常规攻击分数保持率下降、再扩散攻击 unsupported 边界、脚本产物可重建和 outputs 目录约束。
7. `docs/field_registry.md` 已登记攻击配置、攻击记录、source / attacked digest、score retention、quality proxy、attention consistency、攻击统计和 manifest 相关字段。

### stage13 当前产物摘要

1. `outputs/attack_matrix/attack_manifest.json` 显示 `attack_config_count=14`, `attack_family_count=3`, `attack_record_count=1344`, `performed_attack_record_count=960`, `gpu_attack_unsupported_count=384`, `attack_metrics_ready=true`。
2. `outputs/attack_matrix/attack_family_metrics.csv` 已包含常规攻击的 `true_positive_rate`、`false_positive_rate`、`clean_false_positive_rate`、`attacked_false_positive_rate`、`quality_score_proxy_mean`、`score_retention_mean`、`lf_score_retention_mean`、`hf_score_retention_mean`、`attention_consistency_proxy_mean`、`geometry_reliable_rate` 和 `rescue_rate`。
3. 再扩散攻击行保留配置与 digest, 但 `metric_status=unsupported`, `supported_record_count=0`, 不支持论文 robustness 主张。
4. `outputs/attack_matrix/attacked_images/` 当前为空目录, 表示本地未生成真实 attacked image 文件; `attacked_image_registry.jsonl` 只登记受治理代理摘要。
5. `outputs/attack_matrix/attack_manifest.json` 与 `outputs/attack_matrix/manifest.local.json` 已继承最新真实 aligned rescoring 包路径、SHA256 摘要、`aligned_rescoring_quality_metrics_ready=true`、`perceptual_metrics_ready=true` 与 `real_aligned_rescore_count=3`。

### stage13 完成边界

1. 本阶段完成的是攻击矩阵协议、表格重建链路和常规攻击本地代理统计, 不是正式 robustness 实验结论。
2. record-level proxy 只能用于验证字段、统计边界、表格形态和 artifact rebuild, 不能替代真实图像攻击。
3. 再扩散攻击必须在真实 GPU 环境中生成 attacked image、source image digest、attack config digest 和检测记录后, 才能从 `unsupported` 进入可统计状态。
4. fixed-FPR 边界仍沿用 `stage_12` 的结论: raw content claim 可以局部 ready, 完整方法 `full_method_claim_ready=false`。

### stage13 验证结果

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/write_attack_matrix_outputs.py` | pass, `attack_record_count=1344`, `performed_attack_record_count=960`, `gpu_attack_unsupported_count=384`, `attack_metrics_ready=true` |
| `pytest tests/functional/test_attack_matrix.py -q` | pass, 4 passed |
| `pytest -q` | pass, 86 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_14_external_baseline_comparison

| item | value |
| --- | --- |
| construction_unit_name | `external_baseline_comparison` |
| phase_status | `baseline_protocol_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-21` |
| input_manifest | `outputs/attack_matrix/manifest.local.json`; `outputs/attack_matrix/attack_manifest.json`; `outputs/threshold_calibration/threshold_degeneracy_report.json` |
| expected_output_manifest | `outputs/external_baseline_comparison/manifest.local.json` |
| expected_outputs | `outputs/external_baseline_comparison/baseline_observations.jsonl`; `outputs/external_baseline_comparison/baseline_metrics.csv`; `outputs/external_baseline_comparison/baseline_comparison_table.csv`; `outputs/external_baseline_comparison/baseline_runtime_report.json`; `outputs/external_baseline_comparison/manifest.local.json` |
| blocking_items | 当前只完成外部 baseline 的协议 adapter 与公平对比表格链路; 尚未接入官方代码、复现实验结果或受治理导入结果, 因此所有外部 baseline 指标保持 `unsupported`。 |
| fallback_path | 外部 baseline 无结果时只登记 `external_baseline_result_missing`, 不手工填写指标; 当前方法行也只保留 attack matrix local proxy, 不作为论文级 superiority 或 robustness 主张。 |
| invariants | baseline 与 SLM-WM 必须共享 prompt 协议、攻击矩阵协议和 fixed-FPR operating point; unsupported baseline 不进入主结论; 所有新产物保持 `supports_paper_claim=false`。 |
| next_stage_entry | 可进入内部消融证据构建; 若要形成论文级外部对比, 需后续接入 baseline 官方代码或受治理导入结果, 并在相同协议下重建表格。 |

### stage14 已完成内容

1. 新增 `experiments/baselines/adapters.py`, 定义 `BaselineSpec`、`BaselineObservation`、默认外部 baseline 清单、baseline 观测记录构造、baseline 指标聚合和同协议对比表构造。
2. 新增 `scripts/write_external_baseline_comparison_outputs.py`, 从 `outputs/attack_matrix/` 与 `outputs/threshold_calibration/` 读取受治理输入, 重建外部 baseline observations、metrics、comparison table、runtime report 和 manifest。
3. 默认登记 8 个外部 baseline: Tree-Ring、Gaussian Shading、Shallow Diffuse、T2SMark、Stable Signature、RivaGAN、TrustMark 和 Watermark Anything。
4. 新增根目录 `external_baseline/`, 用于本地保存外部 baseline 官方源码或复现镜像; 主表源码槽位包括 Tree-Ring、Gaussian Shading、Shallow Diffuse、T2SMark, 补充表源码槽位包括 Stable Signature、RivaGAN、TrustMark、Watermark Anything。
5. 当前 baseline adapter 只冻结公平协议边界, 不运行或伪造外部方法结果; 所有外部 baseline 行均写入 `metric_status=unsupported` 与 `unsupported_reason=external_baseline_result_missing`。
6. 新增 `tests/functional/test_external_baseline_comparison.py`, 覆盖默认 baseline 清单、产物重建、claim 安全边界和 outputs 目录约束。
7. `docs/field_registry.md` 已登记 baseline observation、baseline readiness、共同协议、comparison table、runtime report 和外部源码来源登记相关字段。

### stage14 当前产物摘要

1. `outputs/external_baseline_comparison/baseline_runtime_report.json` 显示 `baseline_count=8`, `baseline_observation_count=112`, `comparable_baseline_count=8`, `baseline_result_ready_count=0`, `comparison_protocol_ready=true`, `baseline_results_ready=false`。
2. `outputs/external_baseline_comparison/baseline_metrics.csv` 中 8 个 baseline 均为 `baseline_adapter_ready=True`, 但 `baseline_official_code_ready=False`, `baseline_reproduced_result_ready=False`, `baseline_imported_result_ready=False`。
3. `outputs/external_baseline_comparison/baseline_comparison_table.csv` 包含 `slm_wm_current` 本地代理行和 8 个外部 baseline unsupported 行; 所有行均为 `supports_paper_claim=False`。
4. `outputs/external_baseline_comparison/manifest.local.json` 记录输入、输出、baseline spec 摘要、summary 摘要、代码版本和重建命令 `python scripts/write_external_baseline_comparison_outputs.py`。

### stage14 完成边界

1. 本阶段完成的是外部 baseline 公平对比协议、schema adapter、受治理表格链路和缺失结果边界, 不是外部 baseline 实测性能结论。
2. 当前外部 baseline 指标不得用于论文主表结论, 只能说明对比协议已经可审计、可复现并等待真实 baseline 结果接入。
3. 后续若接入官方代码或导入结果, 必须由 adapter 或受治理导入文件生成 records, 不得手工补表。

### stage14 验证结果

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/write_external_baseline_comparison_outputs.py` | pass, `baseline_count=8`, `baseline_observation_count=112`, `baseline_result_ready_count=0` |
| `pytest tests/functional/test_external_baseline_comparison.py -q` | pass, 2 passed |
| `pytest -q` | pass, 88 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_15_internal_ablation_evidence

| item | value |
| --- | --- |
| construction_unit_name | `internal_ablation_evidence` |
| phase_status | `ablation_protocol_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-21` |
| input_manifest | `outputs/attack_matrix/manifest.local.json`; `outputs/attack_matrix/attack_manifest.json`; `outputs/threshold_calibration/threshold_degeneracy_report.json`; `outputs/external_baseline_comparison/manifest.local.json` |
| expected_output_manifest | `outputs/internal_ablation_evidence/manifest.local.json` |
| expected_outputs | `outputs/internal_ablation_evidence/ablation_records.jsonl`; `outputs/internal_ablation_evidence/mechanism_ablation_table.csv`; `outputs/internal_ablation_evidence/method_pairwise_delta_table.csv`; `outputs/internal_ablation_evidence/ablation_by_attack_family.csv`; `outputs/internal_ablation_evidence/ablation_claim_summary.json`; `outputs/internal_ablation_evidence/manifest.local.json` |
| blocking_items | 当前内部消融复用 attack matrix 的 record-level proxy, 真实 attacked image 与再扩散攻击 GPU 闭环仍未补齐; 外部 baseline 真实结果仍为 `baseline_results_ready=false`, 因此外部 superiority 与完整 robustness 主张仍不能成立。 |
| fallback_path | 内部消融只用于冻结机制必要性协议、退化链和表格重建链路; 不把本地代理消融结果写成论文 supported claim。 |
| invariants | 每个消融必须真实改变机制字段或判定边界; `full_slm_wm` 为参考行; `geo_direct_positive_audit` 只能作为审计反例; 所有产物保持 `supports_paper_claim=false`。 |
| next_stage_entry | 可进入论文产物证据审计, 但审计结论必须保留 local proxy 边界, 并把真实图像攻击、外部 baseline 实测与 full-main 规模统计列为后续补证任务。 |

### stage15 已完成内容

1. 新增 `experiments/ablations/mechanisms.py`, 定义 `AblationSpec`、默认内部消融清单、消融 records 构造、机制表聚合、按攻击族聚合、pairwise delta 和 claim summary 构造。
2. 新增 `scripts/write_internal_ablation_outputs.py`, 从攻击矩阵、阈值校准和外部 baseline 对比 manifest 读取受治理输入, 重建内部消融 records、机制表、pairwise delta、attack-family 表、claim summary 和 manifest。
3. 默认登记 17 个内部消融: Full SLM-WM、Global Null Space、No Semantic Mask、No Semantic JVP、No Risk Weight、Random Basis、LF-only、HF-only、No-HF、No-LF、No Tail Truncation、FFT-sync-only、Image-registration-only、No Attention Anchor、No Rescue、No Attestation 和 Geo-direct-positive audit。
4. `full_slm_wm` 保持上游攻击记录的完整方法判定; 其他消融通过 LF/HF retention、aligned gain、attention consistency、geometry reliability、rescue gate、attestation gate 或 content gate 反例路径产生实际字段变化。
5. 新增 `tests/functional/test_internal_ablation_evidence.py`, 覆盖消融清单完整性、关键机制实际变化、输出目录约束、claim 安全边界和表格可重建性。
6. `docs/field_registry.md` 已登记内部消融 records、机制表、pairwise delta、claim summary 和 manifest 相关字段。

### stage15 当前产物摘要

1. `outputs/internal_ablation_evidence/ablation_claim_summary.json` 显示 `ablation_count=17`, `ablation_record_count=22848`, `mechanism_group_count=7`, `ablation_protocol_ready=true`, `mechanism_coverage_ready=true`, `attack_metrics_ready=true`, `external_baseline_result_ready=false`。
2. `outputs/internal_ablation_evidence/mechanism_ablation_table.csv` 包含 17 个消融行, 每行均记录 TPR、FPR、score retention、quality proxy、attention consistency、geometry reliability、rescue rate、attestation availability 和相对完整方法的 delta。
3. `outputs/internal_ablation_evidence/method_pairwise_delta_table.csv` 包含 96 条相对 `full_slm_wm` 的指标差异记录。
4. `outputs/internal_ablation_evidence/ablation_by_attack_family.csv` 包含 51 条按消融和攻击族聚合的退化记录。
5. 所有内部消融产物均保持 `supports_paper_claim=false`, `full_method_claim_ready=false`。

### stage15 完成边界

1. 本阶段完成的是内部消融协议、机制退化链、表格重建链路和 claim 安全边界, 不是论文级最终消融结论。
2. 当前消融结果复用 record-level attack proxy, 不能替代真实图像攻击和 full-main 规模统计。
3. `geo_direct_positive_audit` 明确是 content gate 反例审计, 不得作为正式方法或主表方法行。
4. `no_attestation` 会让检测证据不能进入可审计方法主张, 用于证明 attestation gate 的必要性。

### stage15 验证结果

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/write_internal_ablation_outputs.py` | pass, `ablation_count=17`, `ablation_record_count=22848`, `mechanism_coverage_ready=true` |
| `pytest tests/functional/test_internal_ablation_evidence.py -q` | pass, 3 passed |
| `pytest -q` | pass, 91 passed, 2 deselected |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## real_gpu_aligned_rescoring_workflow

| item | value |
| --- | --- |
| construction_unit_name | `aligned_rescoring` |
| phase_status | `colab_pair_metric_workflow_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-21` |
| input_manifest | `outputs/attention_geometry/manifest.local.json`; `outputs/content_carriers/manifest.local.json`; `outputs/attention_latent_update/manifest.local.json` |
| expected_output_manifest | `outputs/aligned_rescoring/aligned_rescoring_manifest.local.json` |
| expected_outputs | `paper_workflow/aligned_rescoring_run.ipynb`; `paper_workflow/colab_utils/aligned_rescoring.py`; `outputs/aligned_rescoring/aligned_rescoring_records.jsonl`; `outputs/aligned_rescoring/aligned_rescoring_result.json`; `outputs/aligned_rescoring/aligned_rescoring_quality_metrics.csv`; `outputs/aligned_rescoring/aligned_rescoring_environment_report.json`; `outputs/aligned_rescoring/aligned_rescoring_manifest.local.json`; `GoogleDrive/SLM/aligned_rescoring/aligned_rescoring_package_<utc>_<short_commit>.zip` |
| blocking_items | 本地环境无 GPU 和真实 SD3.5 Medium 权重, 因此本次完成 Colab workflow 与 repository helper 的 LPIPS / CLIP pair-level 指标接入; LPIPS / CLIP 默认在 CPU 上计算以避开 SD3.5 pipeline 占用 GPU 后的显存压力, 新真实产物需要在 Colab GPU 中重新运行 notebook 后回传审计。 |
| fallback_path | 若没有 ready attention geometry 包、HF_TOKEN、GPU runtime、真实 latent callback 或 required pair-level perceptual metrics, helper 会写出 fail result 和 unsupported reason, 不会伪造 real aligned score 或感知指标。 |
| invariants | Notebook 只作为入口; 正式逻辑位于 `paper_workflow/colab_utils/aligned_rescoring.py`; 输出仍保持 `supports_paper_claim=false` 和 `full_method_claim_ready=false`, 直到重新运行 geometric rescue 与 threshold calibration 并审计 FPR。 |
| next_stage_entry | Colab 生成并回传 aligned rescoring 包后, 本地应先审计包内 records、quality metrics、manifest 和 environment report, 再决定是否重跑 geometric rescue、threshold calibration 与 attack matrix。 |

### aligned rescoring workflow 已完成内容

1. 新增 `paper_workflow/colab_utils/aligned_rescoring.py`, 支持读取 ready attention geometry 包、重建 prompt / semantic / content / attention update 输入链, 选择 active attention carrier, 在真实 SD3.5 Medium latent callback 中获取对齐前后 latent 投影并重新计算 LF/HF 内容分数。
2. 新增并更新 `paper_workflow/aligned_rescoring_run.ipynb`, 支持 Colab 冷启动: 挂载 Google Drive、安装当前 Colab 可运行依赖组合和 LPIPS 可选依赖、拉取仓库、读取 `HF_TOKEN`、检查 GPU、执行真实 aligned rescoring, 计算 LPIPS 与 CLIP pair-level 指标, 并将结果包保存到 `GoogleDrive/SLM/aligned_rescoring/`。
3. 新增打包函数 `package_aligned_rescoring_outputs`, 会把 aligned rescoring records、result、quality metrics、environment report、manifest、attention update 方法文件和 package input manifest 纳入 zip。
4. 更新 `tests/constraints/test_notebook_entrypoint_contract.py`, 覆盖新 Notebook 入口委托、无执行输出、Drive 镜像路径和打包产物核对。
5. 更新 `docs/field_registry.md`, 登记真实 aligned rescoring、latent projection、LPIPS / FID / KID / CLIP 状态、clean / aligned CLIP score、CLIP delta 和质量指标相关字段。
6. 新增轻量测试 `tests/functional/test_aligned_rescoring_metrics.py`, 验证 LPIPS / CLIP pair-level ready 边界、默认配置和质量指标表字段。
7. 更新感知指标诊断: 若 LPIPS 或 CLIP 未 measured, `unsupported_reason` 会写入 `lpips_status` 与 `clip_score_status`, 质量表会记录对应 error type 和压缩错误信息; Notebook 在断言失败前会打印质量表便于定位。
8. 更新 CLIP 计算兼容路径: 优先使用 `get_image_features` / `get_text_features`, 若当前 transformers 版本缺少该 API, 则退回到 `CLIPModel` forward 输出中的 `image_embeds` / `text_embeds` 或 `logits_per_image`。

### aligned rescoring workflow 当前边界

1. 当前 workflow 默认只运行少量 active attention carrier, 用于验证真实 GPU latent 投影重打分链路, 不是 full-main 规模统计。
2. `aligned_rescoring_quality_metrics.csv` 默认记录 PSNR、SSIM、MSE、MAE、LPIPS、`clip_score_clean`、`clip_score_aligned` 和 `clip_score_delta`; 若 LPIPS 或 CLIP 计算失败且 `require_pair_perceptual_metrics=true`, 本运行的 `run_decision` 应为 `fail`, 并在 `unsupported_reason` 与质量表中保留诊断状态。
3. FID / KID 仍是 dataset-level metric, 当前 pair-level Colab workflow 不计算 FID / KID, 继续写入明确的 unsupported status。
4. 新真实 aligned rescoring 包回传后, 必须重新审计 `real_aligned_rescore_count > 0`、`image_quality_metrics_ready=true`、`perceptual_metrics_ready=true`、环境依赖版本和所有输入 manifest, 之后才能重跑 fixed-FPR 相关产物。

### aligned rescoring result 下游传播记录

1. 已将 `outputs/aligned_rescoring_package_20260620t17281781976491z_b37b14f.zip` 作为阈值校准的显式输入, 并在 `outputs/threshold_calibration/manifest.local.json` 中记录输入路径与 SHA256 摘要 `ac1c8578f611de53aaae68ab22ecc667746090272bcb5c95d2e7844b6913964e`。
2. `outputs/threshold_calibration/quality_metrics_summary.csv` 已改为优先使用 aligned rescoring 包中的真实 pair-level 质量指标; PSNR、SSIM、MSE、MAE、LPIPS 与 CLIP score 均为 measured, FID / KID 保持 dataset-level unsupported。
3. `outputs/threshold_calibration/threshold_degeneracy_report.json`、`outputs/threshold_calibration/manifest.local.json`、`outputs/attack_matrix/attack_manifest.json` 与 `outputs/attack_matrix/manifest.local.json` 均已写入 `aligned_rescoring_quality_metrics_ready=true`、`perceptual_metrics_ready=true`、`aligned_rescoring_record_count=3` 与 `real_aligned_rescore_count=3`。
4. 该传播只解决真实 aligned rescoring 包的质量指标与 provenance 进入下游产物的问题; fixed-FPR 统计仍沿用 governed geometric rescue records, 因此 `evidence_attacked_fpr=0.15625` 与 `full_method_claim_ready=false` 不因本次传播而改变。

### aligned rescoring workflow 验证结果

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `pytest tests/functional/test_aligned_rescoring_metrics.py tests/constraints/test_notebook_entrypoint_contract.py -q` | pass, 18 passed |
| `pytest -q` | pass, 86 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |
