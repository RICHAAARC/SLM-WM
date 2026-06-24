# 阶段17：PilotPaper、Full 与提交前冻结

## 一、阶段定位

本阶段负责完成 probe、pilot_paper、full-main 与必要 full-extra，冻结最终 records、最终论文产物、最终 evidence audit 和 release profiles。本阶段必须调用 stage16 已验证的 artifact builders 重新生成最终论文表格、图、报告和 evidence audit。

本阶段属于 submission readiness gate 与 minimal release extraction 的衔接阶段。其职责不是重新设计 artifact rebuild 规则，而是在配置冻结后运行最终实验、重建最终产物并验证发布包。

本阶段不得跳过前序 artifact 审计，不得绕过 harness，不得将未通过 evidence audit 的 claim 写入 final report。

## 二、阶段开始前的强制读取与审计

本阶段开始前，执行者必须先完成以下检查：

1. 读取 `AGENTS.md`，确认仓库协作约束。
2. 读取 `.codex/project_contract.md`，确认 release boundary、paper artifact governance、Notebook 边界和测试治理。
3. 读取本阶段相关 skill：`minimal_release.skill.md`、`artifact_rebuild.skill.md`、`claim_audit.skill.md`、`notebook_entrypoint.skill.md`、`progression_guard.skill.md`。
4. 执行或检查仓库 intake：

```bash
python tools/harness/inspect_repository.py .
```

5. 检查 stage16 artifact builder readiness report，确认最终 records 到 tables、figures、reports 和 evidence audit 的重建链路可用。
6. 检查 `outputs/audit_reports/harness_audit_summary.json`。若上一阶段存在 harness fail、unsupported reason 未处理、placeholder claim 或字段未登记问题，本阶段不得进入提交前冻结。

## 三、本阶段必须读取的 skill

1. `.codex/skills/minimal_release.skill.md`
2. `.codex/skills/artifact_rebuild.skill.md`
3. `.codex/skills/claim_audit.skill.md`
4. `.codex/skills/notebook_entrypoint.skill.md`
5. `.codex/skills/progression_guard.skill.md`

## 四、阶段输入与前置条件

本阶段开始时应确认以下输入存在、可读取、digest 可校验，并且已经登记到本地或外部 manifest：

1. stage16 artifact builder readiness manifest。
2. stage06 至 stage15 的协议、阈值、攻击、baseline、ablation 和 evidence manifests。
3. probe、pilot_paper、full-main 和 full-extra 配置。
4. calibration 冻结阈值与 fixed-FPR 统计边界说明。
5. release profile 文档：`docs/extraction_profiles.md` 与 `docs/release_boundary.md`。

若 stage16 builder 未通过 dry-run 或 evidence audit schema 未建立，本阶段不得开始 final freeze。

## 五、本阶段实现范围

本阶段需要实现或更新以下功能：

1. 跑通 probe，确认 runner、manifest、records 和 artifact builder 调用路径有效。
2. 跑通 pilot_paper，用于发现失败模式、确认攻击强度和固定 full-main 配置。
3. 在 pilot_paper 后冻结 full-main 配置、阈值和 rescue 协议；full-main 不得继续调参。
4. 执行 full-main，覆盖常规攻击、主表 baseline、主要消融和质量指标。
5. 执行必要 full-extra，覆盖再扩散攻击或高成本实验代表子集，并明确其规模和主张边界。
6. 调用 stage16 builders，从 final records 自动重建最终 tables、figures、reports、latex tables、paper results package 和 evidence audit。
7. 生成 submission readiness report，明确 claims、evidence paths、unsupported reasons、limitations 和 release scope。
8. 使用当前 release profile 导出并验证 `minimal_method_package` 和 `paper_artifact_rebuild_package`。
9. 在干净环境或隔离目录验证最小方法包至少可导入 `main`，并可运行轻量功能测试或最小 smoke。

## 六、禁止事项与边界

本阶段不得执行以下操作：

1. full-main 不得调参。
2. test split 不得回调阈值、rescue window 或 baseline 配置。
3. release package 不得包含大型模型权重、未授权第三方 baseline 源码、本地审计输出、未登记输出或私有路径。
4. `minimal_method_package` 不得包含 Notebook、Drive utils、full experiment outputs、external baseline runner 或 paper workflow。
5. 未通过 evidence audit 的 claim 不得进入 final report。
6. 不得手工补表、手工改图数据或手工标记 harness 通过。

## 七、产物与项目信息更新

本阶段应产出或更新以下内容：

1. `GoogleDrive/SLM-WM/runs/stage17_full_paper_run/final_event_records.jsonl`
2. `GoogleDrive/SLM-WM/runs/stage17_full_paper_run/final_artifacts/`
3. `GoogleDrive/SLM-WM/runs/stage17_full_paper_run/final_paper_results_package/`
4. `GoogleDrive/SLM-WM/runs/stage17_full_paper_run/final_evidence_audit_report.json`
5. `GoogleDrive/SLM-WM/runs/stage17_full_paper_run/submission_readiness_report.json`
6. `GoogleDrive/SLM-WM/runs/stage17_full_paper_run/final_manifest.json`
7. `GoogleDrive/SLM-WM/release_packages/minimal_method_package/`
8. `GoogleDrive/SLM-WM/release_packages/paper_artifact_rebuild_package/`

同时必须更新以下项目信息：

1. `docs/phase_status.md`：记录 final freeze 输入 manifest、输出 manifest、完成情况、阻断项、limitations 和 release scope。
2. `docs/field_registry.md`：登记本阶段新增正式字段；若为 placeholder 字段，必须以 `_placeholder` 结尾，且不得支持 supported claim。
3. `.codex/project_contract.md`：仅在阶段语义、长期目标或 target construction phase 发生变化时更新；不得随意弱化既有约束。
4. `outputs/audit_reports/`：由 harness 自动生成或更新，不得手工编辑审计通过状态。

## 八、验证内容与通过标准

本阶段至少应完成以下验证：

1. full-main 仅使用 calibration 冻结阈值和冻结 rescue 协议。
2. raw content FPR、rescue 后 clean negative FPR 和 rescue 后 attacked negative FPR 均已报告。
3. 主表、消融、baseline、攻击、质量指标、图表、reports 和 evidence audit 均由 final records 自动重建。
4. 所有 supported claims 均绑定 governed artifacts，不存在 placeholder 支撑 claim。
5. `minimal_method_package` 可独立导入并通过轻量验证。
6. `paper_artifact_rebuild_package` 可通过 manifest 重建论文结果包或执行 dry-run 验证。
7. `pytest -q` 与 `python tools/harness/run_all_audits.py` 通过。

本阶段通过标准为：

最终论文结果包、evidence audit、submission readiness report 和两个 release profiles 均已冻结并通过验证；项目达到投稿前复核状态。

## 九、统一门禁与完成命令

每个阶段完成后，默认必须运行：

```bash
pytest -q
python tools/harness/run_all_audits.py
```

若本阶段包含 integration、smoke、slow 或 formal 测试，应额外显式运行对应命令，但不得把重型测试加入默认 `pytest -q` 路径。所有测试输出必须写入 `tmp_path`、本地阶段目录或外部阶段目录，不得写入被版本控制的正式输出目录。

## 十、阶段交接要求

本阶段结束时必须形成可审计交接：

1. 阶段 summary 明确列出 final records、final artifacts、release packages、unsupported reason、limitations 和复现命令。
2. 所有新增字段已经登记到 `docs/field_registry.md` 或在阶段 summary 中说明尚为 `_placeholder` 字段。
3. 所有正式路径、配置键、JSON key 和 Python 模块名均符合 `snake_case` 与命名治理规则。
4. 不得存在“仅在 Notebook cell 中成立”的结果；Notebook 只能调用仓库模块。
5. 若本阶段无法满足通过标准，应停止推进并写出阻断报告，不得通过改名、跳过测试或手工补表绕过阶段门禁。
