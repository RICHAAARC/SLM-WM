# 阶段3：SD3 / SD3.5 运行适配层

## 一、阶段定位

在不改变 `main/` 的前提下，建立 SD3 / SD3.5 runtime adapter，并提供 toy / synthetic fallback 以保证工程测试可持续。

本阶段属于 SLM-WM 项目分阶段构建流程的一部分。整体流程遵循 `core-first -> runtime-second -> workflow-third -> paper-artifact-final`。本阶段不得跳过前序 artifact 审计，不得绕过 harness，不得将临时结果伪装为正式论文证据。

## 二、阶段开始前的强制读取与审计

本阶段开始前，执行者必须先完成以下检查，且不得以任何理由绕过 harness、项目契约或阶段门禁。

1. 读取 `AGENTS.md`，确认仓库协作约束，包括修改前读取项目契约、不得绕过 harness、不得在默认测试路径加入重型测试、placeholder 与 random 字段命名规则、正式 claim 必须绑定 governed artifacts 等规则。
2. 读取 `.codex/project_contract.md`，确认当前 `project_stage`、`target_construction_phase`、目录边界、Notebook 边界、paper artifact governance、naming governance、placeholder/random governance 和 test governance。
3. 读取本阶段相关 `.codex/skills/*.skill.md`。至少应读取 `repository_intake.skill.md`、`stage_progression_guard.skill.md`、`naming_governance.skill.md` 和 `test_case_governance.skill.md`；若本阶段涉及 Notebook、artifact、claim、release 或 placeholder/random 字段，还必须读取对应 skill。
4. 执行或检查仓库 intake：

```bash
python tools/harness/inspect_repository.py .
```

5. 审计已有 artifacts。stage00 至 stage04 检查 `outputs/local_stageXX_*/manifest.local.json`；stage05 及之后检查 Google Drive 对应阶段的 `manifest.json`、`input_manifest.json`、`output_manifest.json`、`artifact_manifest.json` 和 `stage_summary.json`。禁止使用未登记到 manifest 的文件作为本阶段输入。
6. 检查 `outputs/audit_reports/harness_audit_summary.json`。若上一阶段存在 harness fail、unsupported reason 未处理、placeholder claim 或字段未登记问题，本阶段不得进入正式实现。
7. 本阶段实施前应在 `docs/phase_status.md` 或等价阶段状态文件中记录：阶段名、输入 manifest、预期输出、负责人、执行日期、阻断项与本阶段不变量。

## 三、本阶段必须读取的 skill

除通用项目契约外，本阶段必须重点读取以下 skill：

1. `.codex/skills/repository_intake.skill.md`
2. `.codex/skills/stage_progression_guard.skill.md`
3. `.codex/skills/test_case_governance.skill.md`
4. `.codex/skills/naming_governance.skill.md`

## 四、阶段输入与前置条件

本阶段开始时应确认以下输入存在、可读取、digest 可校验，并且已经登记到本地或 Drive manifest：

1. stage02 的 core smoke manifest。
2. 当前环境模型访问条件、GPU 能力与 `configs/model_sd3.yaml`、`configs/model_sd35.yaml`。

若任一输入缺失，应先写入阻断报告，不得通过手工补文件或临时路径继续执行。

## 五、本阶段实现范围

本阶段需要实现或更新以下功能：

1. 创建 `experiments/runtime/diffusion/`，实现 `model_adapter.py`、`sd3_adapter.py`、`sd35_adapter.py`、`sampler_hook.py`、`attention_capture.py`、`latent_trace.py`、`latent_estimator.py`。
2. 支持 SD3 / SD3.5 模型加载、采样 step callback、VAE encode/decode、latent trace 记录、attention maps 捕获和 generation metadata。
3. 实现 fallback：SD3 reduced profile、toy diffusion adapter、synthetic latent adapter。fallback 只能用于工程测试，不支持正式论文主张。
4. 所有 runtime 输出均转换为 core 可消费的 tensor、ndarray 或 typed object；core 不得反向 import runtime。
5. 记录 `unsupported_reason`，例如权重不可访问、显存不足、API 不兼容。

## 六、禁止事项与边界

本阶段不得执行以下操作：

1. 不得把 `diffusers`、`transformers` 或模型权重依赖引入 `main/`。
2. 不得以 toy adapter 结果支持正式论文结论。
3. 不得保存未登记的 latent trace 或 attention artifact。

## 七、产物与项目信息更新

本阶段应产出或更新以下内容：

1. `outputs/local_stage03_sd_adapter/sd_generation_records.jsonl`
2. `outputs/local_stage03_sd_adapter/latent_trace_records.jsonl`
3. `outputs/local_stage03_sd_adapter/attention_capture_records.jsonl`
4. `outputs/local_stage03_sd_adapter/generation_quality_summary.json`
5. `outputs/local_stage03_sd_adapter/manifest.local.json`

同时必须更新以下项目信息：

1. `docs/phase_status.md`：记录本阶段输入 manifest、输出 manifest、完成情况、阻断项、降级路径和下一阶段入口。
2. `docs/field_registry.md`：登记本阶段新增正式字段；若为 placeholder 字段，必须以 `_placeholder` 结尾，且不得支持 supported claim。
3. `.codex/project_contract.md`：仅在阶段语义、长期目标或 target construction phase 发生变化时更新；不得随意弱化既有约束。
4. `outputs/audit_reports/`：由 harness 自动生成或更新，不得手工编辑审计通过状态。

## 八、验证内容与通过标准

本阶段至少应完成以下验证：

1. 相同 prompt、seed、model config 下生成过程可复现或近似可解释。
2. latent trace 和 attention maps 有 digest 或压缩 artifact。
3. runtime adapter 只调用 core，core 不反向依赖 runtime。
4. 无法加载 SD3 / SD3.5 时记录 `unsupported_reason` 并使用 fallback 完成工程测试。

本阶段通过标准为：

runtime 层可产生 clean generation、latent trace 与 attention capture 记录；核心包未被污染。

## 九、统一门禁与完成命令

每个阶段完成后，默认必须运行：

```bash
pytest -q
python tools/harness/run_all_audits.py
```

若本阶段包含集成、smoke、slow 或 formal 测试，应额外显式运行对应命令，但不得把重型测试加入默认 `pytest -q` 路径。所有测试输出必须写入 `tmp_path`、本地阶段目录或 Drive 阶段目录，不得写入被版本控制的正式输出目录。

## 十、阶段交接要求

本阶段结束时必须形成可审计交接：

1. 阶段 summary 明确列出完成项、未完成项、unsupported reason、降级路径和下一阶段输入。
2. 所有新增字段已经登记到 `docs/field_registry.md` 或在阶段 summary 中说明尚为 `_placeholder` 字段。
3. 所有正式路径、配置键、JSON key 和 Python 模块名均符合 `snake_case` 与命名治理规则。
4. 不得存在“仅在 Notebook cell 中成立”的结果；Notebook 只能调用仓库模块。
5. 若本阶段无法满足通过标准，应停止推进并写出阻断报告，不得通过改名、跳过测试或手工补表绕过阶段门禁。
