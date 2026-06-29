# 外部 baseline 接入边界

本目录用于保存外部 watermark baseline 的来源登记、项目维护 adapter 和本地官方源码快照。

## 并入方法

本项目采用 adapter-command-observation-evidence 流程接入外部 baseline:

1. 官方源码快照保存在各方法的 `source/` 子目录, 由本目录 `.gitignore` 排除, 不进入本项目提交。
2. 项目维护的 adapter 保存在各方法的 `adapter/` 子目录, 由本仓库跟踪并接受 harness 审计。
3. `scripts/build_external_baseline_command_plan.py` 生成显式 argv 命令计划。
4. `scripts/run_external_baseline_command_plan.py` 执行命令计划, 汇总 `baseline_observations.json` 和 `baseline_execution_manifest.json`。
5. `scripts/validate_external_baseline_evidence.py` 校验证据边界, 防止把无证据接口检查误当作论文级结果。
6. 下游表格和报告只能读取受治理 observation、manifest 或结果记录, 不手工拼接正式结论。

## 主表 baseline

- `primary/tree_ring/`: Tree-Ring, 需要 SD3.5 latent 形状和 inversion 路径适配。
- `primary/gaussian_shading/`: Gaussian Shading, 需要 SD3.5 noise message 与 latent channel 适配。
- `primary/shallow_diffuse/`: Shallow Diffuse, 需要 SD3.5 shallow latent update 适配。
- `primary/t2smark/`: T2SMark, 官方源码包含 SD3.5 入口, 当前 adapter 负责结果转写。

当前主表 baseline 由 `paper_workflow/external_baseline_tree_ring_run.ipynb`、`paper_workflow/external_baseline_gaussian_shading_run.ipynb`、`paper_workflow/external_baseline_shallow_diffuse_run.ipynb` 和 `paper_workflow/external_baseline_t2smark_run.ipynb` 分别运行。每个入口只调度一个 baseline, 并写出可合并的 `split_observations` 产物, 避免单次 Colab 会话串行运行四个 baseline 导致超时。

运行产物统一落盘到 `outputs/external_baseline_method_faithful/`, 并镜像到当前论文运行层级的 Google Drive 子目录 `external_baseline_method_faithful/`。该命名表示主表外部 baseline 的 SD3.5 method-faithful 适配证据, 不再使用早期链路诊断命名。

## 补充表 baseline

补充表 baseline 先登记来源与证据边界, 不默认进入主表对比命令计划。

官方原始环境参考复现由 `paper_workflow/official_reference_*_run.ipynb` 分别调度。四个入口的 Google Drive 镜像目录统一为当前论文运行层级下的 `external_baseline_official_reference/`, 便于结果闭合时一次性扫描官方参考包; 本地 `outputs/` 仍按方法保留原始语义目录。
