# 阶段1：纯算法原语实现

## 一、阶段定位

在不接入 SD3 / SD3.5、不接入 Colab / Drive 的条件下，实现 SLM-WM 的最小算法原语。

本阶段属于 SLM-WM 项目分阶段构建流程的一部分。整体流程遵循 `core-first -> runtime-second -> workflow-third -> paper-artifact-final`。本阶段不得跳过前序 artifact 审计，不得绕过 harness，不得将临时结果伪装为正式论文证据。

## 二、阶段开始前的强制读取与审计

本阶段开始前，执行者必须先完成以下检查，且不得以任何理由绕过 harness、项目契约或阶段门禁。

1. 读取 `AGENTS.md`，确认仓库协作约束，包括修改前读取项目契约、不得绕过 harness、不得在默认测试路径加入重型测试、placeholder 与 random 字段命名规则、正式 claim 必须绑定 governed artifacts 等规则。
2. 读取 `.codex/project_contract.md`，确认当前 `project_unit`、`target_construction_unit`、目录边界、Notebook 边界、paper artifact governance、naming governance、placeholder/random governance 和 test governance。
3. 读取本阶段相关 `.codex/skills/*.skill.md`。至少应读取 `repository_intake.skill.md`、`progression_guard.skill.md`、`naming_governance.skill.md` 和 `test_case_governance.skill.md`；若本阶段涉及 Notebook、artifact、claim、release 或 placeholder/random 字段，还必须读取对应 skill。
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
2. `.codex/skills/progression_guard.skill.md`
3. `.codex/skills/test_case_governance.skill.md`
4. `.codex/skills/placeholder_random_field_governance.skill.md`

## 四、阶段输入与前置条件

本阶段开始时应确认以下输入存在、可读取、digest 可校验，并且已经登记到本地或 Drive manifest：

1. stage00 的本地 manifest 与边界报告。
2. `main/` 最小包结构。
3. `docs/field_registry.md` 中已有字段登记。

若任一输入缺失，应先写入阻断报告，不得通过手工补文件或临时路径继续执行。

## 五、本阶段实现范围

本阶段需要实现或更新以下功能：

1. 实现 `build_semantic_risk_field(...)`、`project_latent_mask(...)`、`estimate_safe_basis(...)` 的 synthetic / tensor 版本。
2. 实现 `derive_lf_carrier(...)`、`derive_hf_carrier(...)` 与 `derive_attention_carrier_stub(...)`。注意 attention 仅允许 synthetic stub，不接入真实 SD attention。
3. 实现 `compose_latent_update(...)`，返回 `Delta z_t = Delta z_t^LF + Delta z_t^HF + Delta z_t^A` 的 typed object。
4. 实现 `compute_content_score(...)`，内容分数固定为 `s_c = lambda_lf * s_lf + lambda_hf * s_hf`，不得使用 LF/HF 独立阈值投票作为主判。
5. 实现 `evaluate_geometry_reliability(...)` 与 `decide_evidence_and_final(...)`，确保几何链不直接判 positive，attestation 不改变 evidence-level。
6. 新增核心单元测试，覆盖正确 key、错误 key、HF tail truncation、rescue 边界、attestation 分层。

## 六、禁止事项与边界

本阶段不得执行以下操作：

1. 不得引入 `diffusers`、`transformers`、SD3 权重、Colab、Drive 或 Notebook。
2. 不得让 `main/` 写任何 JSONL records；`synthetic_core_records.jsonl` 只能由 `scripts/run_core_smoke.py` 或测试 harness 根据 typed objects 生成。
3. 不得把 attention stub 表述为真实 Self-Attention carrier。

## 七、产物与项目信息更新

本阶段应产出或更新以下内容：

1. `outputs/algorithm_primitives/core_primitive_summary.json`
2. `outputs/algorithm_primitives/synthetic_core_records.jsonl`，由外层脚本生成
3. `outputs/algorithm_primitives/manifest.local.json`

同时必须更新以下项目信息：

1. `docs/phase_status.md`：记录本阶段输入 manifest、输出 manifest、完成情况、阻断项、降级路径和下一阶段入口。
2. `docs/field_registry.md`：登记本阶段新增正式字段；若为 placeholder 字段，必须以 `_placeholder` 结尾，且不得支持 supported claim。
3. `.codex/project_contract.md`：仅在阶段语义、长期目标或 target construction phase 发生变化时更新；不得随意弱化既有约束。
4. `outputs/audit_reports/`：由 harness 自动生成或更新，不得手工编辑审计通过状态。

## 八、验证内容与通过标准

本阶段至少应完成以下验证：

1. 正确 key 的 LF / HF score 高于错误 key。
2. HF tail truncation 改变 HF score 分布。
3. 几何链不直接判 positive。
4. Attestation 不改变 evidence-level。
5. `pytest tests/functional -q`、`pytest tests/constraints -q`、`pytest -q`、harness 全部通过。

本阶段通过标准为：

纯算法原语具备最小闭环；所有 outputs 由外层脚本写出；核心包仍保持零 workflow 依赖。

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
