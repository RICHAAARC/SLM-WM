# 阶段5：Colab + Google Drive 运行层

## 一、阶段定位

在 core 和 runtime 最小闭环成立后，引入 Colab 冷启动与 Google Drive 持久化。Colab 冷启动 clone 后的本地 `outputs/` 可能为空, 因此前序真实运行产物应优先从 Google Drive 的既有 `SLM/real_sd_runtime_probe/` 与 `SLM/minimal_diffusion_latent_injection/` 目录登记, 本地 `outputs/` 仅作为补充输入。

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
2. `.codex/skills/notebook_entrypoint.skill.md`
3. `.codex/skills/artifact_rebuild.skill.md`
4. `.codex/skills/progression_guard.skill.md`

## 四、阶段输入与前置条件

本阶段开始时应确认以下输入存在、可读取、digest 可校验，并且已经登记到本地或 Drive manifest：

1. Google Drive 目标根目录, 其中应包含前序真实运行产物目录, 例如 `SLM/real_sd_runtime_probe/` 与 `SLM/minimal_diffusion_latent_injection/`。
2. stage00–stage04 本地 manifest; 在 Colab 冷启动 clone 后本地 `outputs/` 为空时, 该项不得作为唯一输入来源。
3. 现有 `paper_workflow/` 与 `scripts/` 边界。

若任一输入缺失，应先写入阻断报告，不得通过手工补文件或临时路径继续执行。

## 五、本阶段实现范围

本阶段需要实现或更新以下功能：

1. 在 `paper_workflow/colab_utils/` 中实现 `mount_drive.py`、`runtime_setup.py`、`drive_paths.py`、`manifest_io.py`、`dependency_check.py`。
2. 在 `scripts/` 中实现 `colab_stage_entry.py`、`write_stage_manifest.py`、`verify_drive_artifacts.py`、`sync_local_stages_to_drive.py`。
3. 创建单一 Notebook：`paper_workflow/colab_drive_cold_start_smoke.ipynb`, 同时覆盖 Colab 冷启动、Drive 镜像、工作流清单写入与 reload 校验。
4. 登记 Google Drive 中已有的前序真实运行产物, 同时在本地存在 outputs 产物时将其同步到 Google Drive, 并生成 `local_output_sync_report.json`。
5. 实现 Drive manifest 校验，确保后续阶段只能读取 manifest 登记文件, 且空 manifest 不得被误判为有效 reload 证据。

## 六、禁止事项与边界

本阶段不得执行以下操作：

1. Notebook 不得直接实现算法逻辑、阈值计算、正式 records、tables、figures 或 reports。
2. 不得让 `main/` import Drive / Colab helper。
3. 不得使用未登记到 manifest 的 `/content/` 临时文件。

## 七、产物与项目信息更新

本阶段应产出或更新以下内容：

1. `GoogleDrive/SLM-WM/runs/stage05_colab_drive_runner/colab_env_report.json`
2. `drive_mount_report.json`
3. `cold_start_smoke_record.jsonl`
4. `reload_smoke_record.jsonl`
5. `local_stage_sync_report.json`
6. `manifest.json`

同时必须更新以下项目信息：

1. `docs/phase_status.md`：记录本阶段输入 manifest、输出 manifest、完成情况、阻断项、降级路径和下一阶段入口。
2. `docs/field_registry.md`：登记本阶段新增正式字段；若为 placeholder 字段，必须以 `_placeholder` 结尾，且不得支持 supported claim。
3. `.codex/project_contract.md`：仅在阶段语义、长期目标或 target construction phase 发生变化时更新；不得随意弱化既有约束。
4. `outputs/audit_reports/`：由 harness 自动生成或更新，不得手工编辑审计通过状态。

## 八、验证内容与通过标准

本阶段至少应完成以下验证：

1. 重启 Colab runtime 后，能只凭 Drive manifest 重载前序结果。
2. Google Drive 中已有的前序真实运行产物已登记到 Drive manifest; 若本地存在 outputs 产物, 也应补登记到 Drive。
3. Notebook 只调用 CLI 或 repository function。
4. `main/` 仍不依赖 Colab / Drive。

本阶段通过标准为：

Colab 冷启动、Drive 持久化和 manifest reload 成立；workflow 未侵入 core。

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
