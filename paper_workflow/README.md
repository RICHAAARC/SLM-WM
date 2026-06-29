# Paper Workflow

此目录保存论文相关 Notebook workflow 入口和执行环境包装。Notebook 只能调度 repository modules, 不得成为唯一正式实现。

## 入口分组

### 诊断入口

- `colab_drive_cold_start_smoke.ipynb`: 验证 Colab 挂载 Google Drive、拉取仓库、镜像本地 `outputs/`、写入工作流清单并重载校验。该入口已经合并原独立 reload 功能。
- `runtime_method_precheck_run.ipynb`: 合并运行时诊断与最小机制预检。该入口验证 Colab 依赖组合、Hugging Face 登录、SD3.5 Medium 加载、真实 latent trajectory 捕获和最小 latent injection 闭环。该入口不参与 pilot_paper 或 full_paper 的正式统计产出。

### 方法主流程入口

以下入口统一由 `SLM_WM_PAPER_RUN_NAME` 派生协议、prompt 文件、样本数和 Google Drive 结果根目录。默认值为 `pilot_paper`; 切换为 `full_paper` 时, Notebook 会重新写入 `SLM_WM_PROTOCOL_PROFILE`、`SLM_WM_PROMPT_SET`、`SLM_WM_PROMPT_FILE` 和各子流程 Drive 目录, 避免沿用旧运行层级的环境变量。

- `attention_geometry_capture_run.ipynb`: 捕获真实 attention 载体或可审计 attention map, 生成 attention geometry 产物。
- `attention_latent_injection_run.ipynb`: 基于 attention geometry 执行 attention-relative latent update, 生成注入记录和配对质量记录。
- `aligned_rescoring_run.ipynb`: 对注入结果进行 aligned rescoring, 生成检测重打分记录和配对感知质量指标。`pilot_paper` 默认使用该层级全部 prompt, 当前配置为600个; `full_paper` 当前配置为6000个。
- `threshold_calibration_run.ipynb`: 在 fixed-FPR=0.01 协议下校准阈值, 并记录 geometric rescue 边界。
- `real_attack_evaluation_run.ipynb`: 生成或导入真实 attacked image, 执行正式攻击后检测, 记录 source / attacked image digest 和再扩散攻击覆盖状态。处理数量由当前论文运行层级的样本数派生。
- `conventional_geometric_attack_evaluation_run.ipynb`: 使用 CPU 图像算子生成常规失真与几何变换 attacked image, 写出 source / attacked image digest、formal detection records 和总体进度条。该入口不运行 diffusion 模型, 负责补齐 `standard_distortion` 与 `geometric_transform` 的真实图像级攻击闭环。
- `dataset_level_quality_run.ipynb`: 从受治理图像集合导入 dataset-level 质量特征, 计算或登记 FID / KID 等集合级指标。pilot_paper 默认使用100作为 dataset-level 正式特征最小样本阈值, 其结果只允许支撑 pilot_paper 样本规模内的主张, 不得提升为 full_paper 主张。

### 外部 baseline 入口

- `external_baseline_tree_ring_run.ipynb`: 只运行 Tree-Ring 的 SD3.5 method-faithful adapter, 覆盖共同攻击矩阵中的常规失真、几何变换与再生成攻击, 并写出 `split_observations/tree_ring_baseline_observations.json`。
- `external_baseline_gaussian_shading_run.ipynb`: 只运行 Gaussian Shading 的 SD3.5 method-faithful adapter, 覆盖共同攻击矩阵中的常规失真、几何变换与再生成攻击, 并写出 `split_observations/gaussian_shading_baseline_observations.json`。
- `external_baseline_shallow_diffuse_run.ipynb`: 只运行 Shallow Diffuse 的 SD3.5 method-faithful adapter, 覆盖共同攻击矩阵中的常规失真、几何变换与再生成攻击, 并写出 `split_observations/shallow_diffuse_baseline_observations.json`。
- `external_baseline_t2smark_run.ipynb`: 只运行 T2SMark 官方 SD3.5 路径和 adapter, 冷启动时会补丁为同一攻击簇输出正式攻击分数, 并写出 `split_observations/t2smark_baseline_observations.json`。
- `official_reference_t2smark_run.ipynb`: 运行 T2SMark 官方 SD3.5 路径并生成 governed import 候选记录。默认使用当前论文运行层级的 prompt 文件、样本数和 fixed-FPR=0.01。
- `official_reference_tree_ring_run.ipynb`: 运行 Tree-Ring 官方原始环境参考复现, 样本数和 Drive 目录由当前论文运行层级派生。
- `official_reference_gaussian_shading_run.ipynb`: 运行 Gaussian Shading 官方原始环境参考复现, 样本数和 Drive 目录由当前论文运行层级派生。
- `official_reference_shallow_diffuse_run.ipynb`: 运行 Shallow Diffuse 官方原始环境参考复现, 样本数和 Drive 目录由当前论文运行层级派生。

四个官方参考复现入口统一把压缩包镜像到当前论文运行层级的 Google Drive 子目录 `external_baseline_official_reference/`。本地 `outputs/` 子目录仍保留各方法原有语义名称, 用于结果闭合脚本读取和产物来源审计。

### 结果闭合入口

- `pilot_paper_result_closure_run.ipynb`: 在前序结果包已经写入 Google Drive 后, 从当前论文运行层级的 Drive 根目录物化上游包, 依次重建 attack matrix、external baseline formal import、internal ablation、fixed-FPR 共同协议记录和完整结果包。该入口不需要 GPU, 只调度 `scripts/` 中的 repository commands, 不直接手写正式 records、tables、figures 或 reports。

## pilot_paper 与 full_paper 运行原则

pilot_paper 与 full_paper 应共享同一批方法主流程和 baseline 入口, 只通过配置切换 prompt split、样本量、随机种子、Drive 输出根目录、攻击覆盖范围和 bootstrap 次数。不得为 pilot_paper 和 full_paper 维护两套互相分叉的 Notebook 逻辑。pilot_paper 是受样本规模约束的论文主张层级, 不是仅用于链路调试的临时层级。

诊断入口不参与 pilot_paper 或 full_paper 的正式统计产出。它们只用于在 Colab 环境、模型依赖或 Drive 持久化出问题时进行预检。

## Colab pilot_paper 重跑顺序

建议在清理 `/content/drive/MyDrive/SLM/pilot_paper_results/` 后按以下顺序重跑。该顺序只依赖 Google Drive 结果包, 不要求本地 `outputs/` 中存在历史文件。

1. 可选诊断: `colab_drive_cold_start_smoke.ipynb`。
2. 可选诊断: `runtime_method_precheck_run.ipynb`。
3. 方法主流程: `attention_geometry_capture_run.ipynb`。
4. 方法主流程: `attention_latent_injection_run.ipynb`。
5. 方法主流程: `aligned_rescoring_run.ipynb`。
6. 阈值与 rescue 边界: `threshold_calibration_run.ipynb`。
7. 再扩散真实攻击闭环: `real_attack_evaluation_run.ipynb`。
8. 常规失真与几何变换真实攻击闭环: `conventional_geometric_attack_evaluation_run.ipynb`。
9. 数据集级质量: `dataset_level_quality_run.ipynb`。
10. 主表 baseline: `external_baseline_tree_ring_run.ipynb`。
11. 主表 baseline: `external_baseline_gaussian_shading_run.ipynb`。
12. 主表 baseline: `external_baseline_shallow_diffuse_run.ipynb`。
13. 主表 baseline: `external_baseline_t2smark_run.ipynb`。
14. 官方复现: `official_reference_t2smark_run.ipynb`。
15. 官方复现: `official_reference_tree_ring_run.ipynb`。
16. 官方复现: `official_reference_gaussian_shading_run.ipynb`。
17. 官方复现: `official_reference_shallow_diffuse_run.ipynb`。
18. 结果闭合: `pilot_paper_result_closure_run.ipynb`。

结果闭合 Notebook 已经封装以下收尾命令。若不使用该 Notebook, 也可以在同一个 Colab 仓库工作区或本地仓库中手动执行以下等价命令。若在本地执行, `--package-search-root` 应指向已经同步或挂载的 Google Drive `SLM/pilot_paper_results` 目录。

```bash
python scripts/write_pilot_paper_result_records.py \
  --package-search-root /content/drive/MyDrive/SLM/pilot_paper_results \
  --materialize-only

python scripts/write_attack_matrix_outputs.py

python scripts/write_primary_baseline_result_candidates.py \
  --target-fpr-override 0.01

python scripts/write_primary_baseline_formal_import_protocol.py

python scripts/write_external_baseline_comparison_outputs.py

python scripts/write_internal_ablation_outputs.py

python scripts/write_pilot_paper_result_records.py \
  --require-existing-evidence

python scripts/write_pilot_paper_fixed_fpr_common_protocol_outputs.py \
  --candidate-records-path outputs/pilot_paper_fixed_fpr_results/pilot_paper_result_records.jsonl \
  --require-existing-evidence

python scripts/write_pilot_paper_complete_result_package.py \
  --package-search-root /content/drive/MyDrive/SLM/pilot_paper_results \
  --drive-output-dir /content/drive/MyDrive/SLM/pilot_paper_results/complete_result_package \
  --archive-name pilot_paper_complete_result_package_<utc>_<short_commit>.zip \
  --skip-package-materialization \
  --zip-compression stored
```

第一条 `write_pilot_paper_result_records.py --materialize-only` 命令用于把 Google Drive 结果包中的 `outputs/` 条目物化到当前仓库工作区, 供后续重建脚本读取。后续结果记录命令不应再次传入 `--package-search-root`, 否则会重复解压大包。完整结果包命令默认使用 `--skip-package-materialization` 复用已物化的本地 `outputs/`, 并用 `--zip-compression stored` 避免对 PNG 和已有压缩内容重复压缩。若模板覆盖仍不完整, 产物会保留 `pilot_paper_template_coverage_ready=false`, 这表示只能继续补齐缺失方法或攻击项, 不能提升为 full_paper 主张。
