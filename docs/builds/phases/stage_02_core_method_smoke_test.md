# 阶段2：核心方法最小闭环 smoke test

## 一、阶段定位

使用 synthetic latent tensor 验证 SLM-WM 核心链路闭环，避免真实模型生成不确定性干扰方法逻辑。

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
4. `.codex/skills/placeholder_random_field_governance.skill.md`

## 四、阶段输入与前置条件

本阶段开始时应确认以下输入存在、可读取、digest 可校验，并且已经登记到本地或 Drive manifest：

1. stage00 与 stage01 的本地 manifest。
2. `main/` 的 typed objects 与核心函数。

若任一输入缺失，应先写入阻断报告，不得通过手工补文件或临时路径继续执行。

## 五、本阶段实现范围

本阶段需要实现或更新以下功能：

1. 实现 `scripts/run_core_smoke.py`，由脚本调用 core 并写出 synthetic records。
2. 实现 `scripts/run_minimal_method_smoke.py`，用于 minimal_method_package 后续独立验证。
3. 构造 synthetic latent 场景：clean synthetic latent、watermarked synthetic latent、wrong-key negative、geometric shifted latent、aligned recovered latent、unattested positive、final positive。
4. 生成 `core_smoke_metrics.json`，统计 key 区分、score margin、rescue trigger rate、attestation 分层状态。
5. 检查 `positive_by_content`、`rescue_eligible`、`rescue_applied`、`evidence_decision`、`final_decision` 是否符合设计。

## 六、禁止事项与边界

本阶段不得执行以下操作：

1. 不得接入真实 SD3 / SD3.5。
2. 不得实现 Colab / Drive。
3. 不得把 smoke 结果写成论文 supported claims。
4. 不得将 synthetic attention stub 宣称为真实 attention geometry。

## 七、产物与项目信息更新

本阶段应产出或更新以下内容：

1. `outputs/local_stage02_core_synthetic_smoke/synthetic_event_records.jsonl`
2. `outputs/local_stage02_core_synthetic_smoke/core_smoke_metrics.json`
3. `outputs/local_stage02_core_synthetic_smoke/core_smoke_summary.md`
4. `outputs/local_stage02_core_synthetic_smoke/manifest.local.json`

同时必须更新以下项目信息：

1. `docs/phase_status.md`：记录本阶段输入 manifest、输出 manifest、完成情况、阻断项、降级路径和下一阶段入口。
2. `docs/field_registry.md`：登记本阶段新增正式字段；若为 placeholder 字段，必须以 `_placeholder` 结尾，且不得支持 supported claim。
3. `.codex/project_contract.md`：仅在阶段语义、长期目标或 target construction phase 发生变化时更新；不得随意弱化既有约束。
4. `outputs/audit_reports/`：由 harness 自动生成或更新，不得手工编辑审计通过状态。

## 八、验证内容与通过标准

本阶段至少应完成以下验证：

1. 错误 key 不应稳定过阈值。
2. rescue 只在边界窗口触发。
3. 几何可靠性不足时不得触发 rescue。
4. `python scripts/run_minimal_method_smoke.py` 可运行。
5. `pytest tests/functional -q` 与 `pytest tests/constraints -q` 通过。

本阶段通过标准为：

核心链路在 synthetic latent 上可复现；smoke records 由外层脚本生成；`main/` 核心包可独立运行 smoke demo。

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
