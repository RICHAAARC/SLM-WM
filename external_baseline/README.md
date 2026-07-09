# 外部 baseline 接入边界

本目录用于保存外部 watermark baseline 的来源登记、项目维护 adapter 和本地官方源码快照。`source/` 子目录由 `external_baseline/.gitignore` 排除, 不进入本项目提交; 项目维护的 `adapter/` 代码由本仓库跟踪并接受 harness 审计。

## 并入方法

本项目采用 adapter-command-observation-evidence 流程接入外部 baseline:

1. 官方源码快照保存在各方法的 `source/` 子目录。
2. 项目维护 adapter 保存在各方法的 `adapter/` 子目录。
3. `scripts/build_external_baseline_command_plan.py` 生成显式 argv 命令计划。
4. `scripts/run_external_baseline_command_plan.py` 执行命令计划, 汇总 `baseline_observations.json` 和 `baseline_execution_manifest.json`。
5. `scripts/validate_external_baseline_evidence.py` 校验证据边界, 防止把无证据接口检查误当作论文级结果。
6. 下游表格和报告只能读取受治理 observation、manifest 或结果记录, 不手工拼接正式结论。

## 主表 baseline

- `primary/tree_ring/`: Tree-Ring, 需要 SD3.5 latent 形状和 inversion 路径适配。
- `primary/gaussian_shading/`: Gaussian Shading, 需要 SD3.5 noise message 与 latent channel 适配。
- `primary/shallow_diffuse/`: Shallow Diffuse, 需要 SD3.5 shallow latent update 适配。
- `primary/t2smark/`: T2SMark, 官方源码包含 SD3.5 入口, 当前 adapter 负责结果转写和共同协议落盘。

当前主表 baseline 由 `paper_workflow/notebooks/external_baseline_tree_ring_run.ipynb`、`paper_workflow/notebooks/external_baseline_gaussian_shading_run.ipynb`、`paper_workflow/notebooks/external_baseline_shallow_diffuse_run.ipynb` 和 `paper_workflow/notebooks/external_baseline_t2smark_run.ipynb` 分别运行。每个入口只调度一个 baseline, 并写出可合并的 `split_observations` 产物, 避免单次 Colab 会话串行运行四个 baseline 导致超时。

运行产物统一写入 `outputs/external_baseline_method_faithful/`, 并镜像到当前论文运行层级的 Google Drive 子目录 `external_baseline_method_faithful/`。该命名表示主表外部 baseline 的 SD3.5 method-faithful 适配证据。

## official reference

official reference 用于记录外部方法在其官方或 legacy 环境下的参考结果, 再通过 governed import 协议进入补充表和方法忠实度审计。该链路不替代 SD3.5 method-faithful adapter, 也不能单独支持主表 common-backbone 结论。

四个入口分别为:

- `paper_workflow/notebooks/official_reference_tree_ring_run.ipynb`
- `paper_workflow/notebooks/official_reference_gaussian_shading_run.ipynb`
- `paper_workflow/notebooks/official_reference_shallow_diffuse_run.ipynb`
- `paper_workflow/notebooks/official_reference_t2smark_run.ipynb`

四个官方参考复现入口统一把压缩包镜像到当前论文运行层级的 Google Drive 子目录 `external_baseline_official_reference/`。本地 `outputs/` 子目录仍保留各方法原有语义名称, 用于结果闭合脚本读取和产物来源审计。

## 证据门禁

`probe_paper`、`pilot_paper` 和 `full_paper` 的正式共同协议记录只接受受治理 baseline 证据。proxy、placeholder、fallback、synthetic 和 formal-null 记录只能作为诊断或失败原因进入 manifest, 不能进入 claim-ready 统计。

## 补充表 baseline

`supplemental/` 下的方法先登记来源与证据边界, 不默认进入主表对比命令计划。若后续纳入统一对比, 必须补充 adapter、命令计划、正式 observation、manifest 和 governed import validator。
