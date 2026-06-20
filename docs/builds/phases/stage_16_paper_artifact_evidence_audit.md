# 阶段16：论文图表、报告与 evidence audit

## 一、阶段定位

本阶段负责实现并验证论文产物重建层，即从 governed records 与 manifests 自动重建 tables、figures、reports 和 evidence audit 的能力。本阶段不负责冻结 full-main 最终实验结果，也不负责提交前最终发布包冻结。

本阶段属于 `paper-artifact-final` 路径中的 artifact rebuild gate。其核心职责是证明：只要 stage17 提供最终 records 与 manifests，仓库就能自动重建论文所需产物，并为每个 supported claim 给出 evidence path。

本阶段不得跳过前序 artifact 审计，不得绕过 harness，不得将预览结果、probe 结果或占位结果伪装为正式论文证据。

## 二、阶段开始前的强制读取与审计

本阶段开始前，执行者必须先完成以下检查：

1. 读取 `AGENTS.md`，确认仓库协作约束。
2. 读取 `.codex/project_contract.md`，确认 `main/`、`experiments/`、`paper_workflow/`、`scripts/`、`tests/` 和 `tools/harness/` 的边界。
3. 读取本阶段相关 skill：`artifact_rebuild.skill.md`、`claim_audit.skill.md`、`notebook_entrypoint.skill.md`、`progression_guard.skill.md`、`minimal_release.skill.md`。
4. 执行或检查仓库 intake：

```bash
python tools/harness/inspect_repository.py .
```

5. 检查上一阶段 records 与 manifests 的可用性。若只有 probe / pilot records，本阶段只能产出 builder readiness 或预览产物，不得冻结最终论文结果。
6. 检查 `outputs/audit_reports/harness_audit_summary.json`。若上一阶段存在 harness fail、unsupported reason 未处理、placeholder claim 或字段未登记问题，本阶段不得进入正式 artifact rebuild。

## 三、本阶段必须读取的 skill

1. `.codex/skills/artifact_rebuild.skill.md`
2. `.codex/skills/claim_audit.skill.md`
3. `.codex/skills/notebook_entrypoint.skill.md`
4. `.codex/skills/progression_guard.skill.md`
5. `.codex/skills/minimal_release.skill.md`

## 四、阶段输入与前置条件

本阶段开始时应确认以下输入存在、可读取、digest 可校验，并且已经登记到本地或外部 manifest：

1. stage12 至 stage15 的 records、metrics、tables 或 manifests。
2. `docs/field_registry.md`。
3. supported claims 草案或 claim mapping 草案。
4. artifact rebuild 配置草案。

若缺少 final full records，本阶段仍可实现和验证 builders，但必须在 summary 中明确“最终结果冻结留待 stage17”。

## 五、本阶段实现范围

本阶段需要实现或更新以下功能：

1. 在 `main/analysis/` 中实现或完善 records 聚合、表格构建、图数据构建、报告构建和 evidence audit 逻辑。
2. 在 `scripts/` 中实现或完善 `build_paper_outputs.py`、`export_paper_results_package.py`、`verify_paper_package.py` 等入口。
3. 定义正式论文产物的 manifest schema，确保每个产物记录输入、输出、配置摘要、代码版本和重建命令。
4. 定义 claim 到 evidence path 的映射规则，禁止 placeholder 字段支撑 supported claim。
5. 支持从已有 governed records 重建 formal main table、baseline comparison table、mechanism ablation table、attack metrics、quality metrics、fixed-FPR operating points、score distribution figures、ROC / DET figures、attack robustness figures、ablation delta figures 和 case study figures。
6. 生成 artifact builder readiness report，说明哪些产物已可由 records 重建，哪些仍等待 stage17 final records。

## 六、禁止事项与边界

本阶段不得执行以下操作：

1. 不得手工填写正式表格或图数据。
2. Notebook 不得直接写正式 records、tables、figures、reports 或 thresholds。
3. 无 evidence path 的 claim 不得进入 reports。
4. 不得将 probe、pilot、synthetic 或 fallback 结果伪装为 full-main 最终结果。
5. 不得把 stage16 预览产物作为 submission-ready final package。

## 七、产物与项目信息更新

本阶段应产出或更新以下内容：

1. `GoogleDrive/SLM-WM/runs/stage16_paper_artifacts/tables/`
2. `GoogleDrive/SLM-WM/runs/stage16_paper_artifacts/figures/`
3. `GoogleDrive/SLM-WM/runs/stage16_paper_artifacts/reports/`
4. `GoogleDrive/SLM-WM/runs/stage16_paper_artifacts/latex_tables/`
5. `GoogleDrive/SLM-WM/runs/stage16_paper_artifacts/artifact_builder_readiness_report.json`
6. `GoogleDrive/SLM-WM/runs/stage16_paper_artifacts/evidence_audit_dry_run.json`
7. `GoogleDrive/SLM-WM/runs/stage16_paper_artifacts/manifest.json`

同时必须更新以下项目信息：

1. `docs/phase_status.md`：记录本阶段输入 manifest、输出 manifest、完成情况、阻断项、降级路径和 stage17 入口。
2. `docs/field_registry.md`：登记本阶段新增正式字段；若为 placeholder 字段，必须以 `_placeholder` 结尾，且不得支持 supported claim。
3. `.codex/project_contract.md`：仅在阶段语义、长期目标或 target construction phase 发生变化时更新；不得随意弱化既有约束。
4. `outputs/audit_reports/`：由 harness 自动生成或更新，不得手工编辑审计通过状态。

## 八、验证内容与通过标准

本阶段至少应完成以下验证：

1. 所有已声明的论文表格均可由 records 与 manifests 重建，或明确标记等待 stage17 final records。
2. 所有已声明的 claim 均有 evidence path，或被标记为 unsupported / pending。
3. 不存在 placeholder 支撑 claim。
4. artifact manifest 记录输入、输出、配置摘要、代码版本和重建命令。
5. paper package builder 可在冷启动环境中用登记的 manifest 进行 dry-run 或验证。

本阶段通过标准为：

artifact rebuild 层可运行且可审计；evidence audit 规则已建立；最终 full-main 结果冻结明确交由 stage17 完成。

## 九、统一门禁与完成命令

每个阶段完成后，默认必须运行：

```bash
pytest -q
python tools/harness/run_all_audits.py
```

若本阶段包含 integration、smoke、slow 或 formal 测试，应额外显式运行对应命令，但不得把重型测试加入默认 `pytest -q` 路径。所有测试输出必须写入 `tmp_path`、本地阶段目录或外部阶段目录，不得写入被版本控制的正式输出目录。

## 十、阶段交接要求

本阶段结束时必须形成可审计交接：

1. 阶段 summary 明确列出完成项、未完成项、unsupported reason、降级路径和 stage17 所需输入。
2. 所有新增字段已经登记到 `docs/field_registry.md` 或在阶段 summary 中说明尚为 `_placeholder` 字段。
3. 所有正式路径、配置键、JSON key 和 Python 模块名均符合 `snake_case` 与命名治理规则。
4. 不得存在“仅在 Notebook cell 中成立”的结果；Notebook 只能调用仓库模块。
5. 若本阶段无法满足通过标准，应停止推进并写出阻断报告，不得通过改名、跳过测试或手工补表绕过阶段门禁。
