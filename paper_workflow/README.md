# Paper Workflow

此目录保存论文相关 Notebook workflow 入口和执行环境包装。Notebook 只能调度 repository modules, 不得成为唯一正式实现。

## 入口分组

### 诊断入口

- `colab_drive_cold_start_smoke.ipynb`: 验证 Colab 挂载 Google Drive、拉取仓库、镜像本地 `outputs/`、写入工作流清单并重载校验。该入口已经合并原独立 reload 功能。
- `runtime_method_precheck_run.ipynb`: 合并运行时诊断与最小机制预检。该入口验证 Colab 依赖组合、Hugging Face 登录、SD3.5 Medium 加载、真实 latent trajectory 捕获和最小 latent injection 闭环。该入口不参与 pilot_paper 或 full_paper 的正式统计产出。

### 方法主流程入口

以下入口当前默认使用 `SLM_WM_PROTOCOL_PROFILE=pilot_paper_fixed_fpr_0_01`, `SLM_WM_PROMPT_SET=pilot_paper`, `SLM_WM_PROMPT_FILE=configs/paper_main_pilot_paper_prompts.txt`, 并把 Google Drive 结果写入 `/content/drive/MyDrive/SLM/pilot_paper_results/` 下的对应子目录。

- `attention_geometry_capture_run.ipynb`: 捕获真实 attention 载体或可审计 attention map, 生成 attention geometry 产物。
- `attention_latent_injection_run.ipynb`: 基于 attention geometry 执行 attention-relative latent update, 生成注入记录和配对质量记录。
- `aligned_rescoring_run.ipynb`: 对注入结果进行 aligned rescoring, 生成检测重打分记录和配对感知质量指标。pilot_paper 默认最多重打分120个 carrier, 用于满足 fixed-FPR=0.01 下至少100个 clean negative 的统计分辨率边界。
- `threshold_calibration_run.ipynb`: 在 fixed-FPR=0.01 协议下校准阈值, 并记录 geometric rescue 边界。
- `real_attack_evaluation_run.ipynb`: 生成或导入真实 attacked image, 执行正式攻击后检测, 记录 source / attacked image digest 和再扩散攻击覆盖状态。pilot_paper 默认最多处理120张 source image, 以便攻击矩阵重建和下游 dataset-level 指标具有 pilot_paper 级样本支撑。
- `dataset_level_quality_run.ipynb`: 从受治理图像集合导入 dataset-level 质量特征, 计算或登记 FID / KID 等集合级指标。pilot_paper 默认使用100作为 dataset-level 正式特征最小样本阈值, 其结果只允许支撑 pilot_paper 样本规模内的主张, 不得提升为 full_paper 主张。

### 外部 baseline 入口

- `external_baseline_gpu_smoke_run.ipynb`: 运行主表 baseline 的 SD3.5 method-faithful adapter 链路, 当前覆盖 Tree-Ring、Gaussian Shading、Shallow Diffuse 和 T2SMark。当前默认写入 `/content/drive/MyDrive/SLM/pilot_paper_results/external_baseline_gpu_smoke`, 共享样本数为120。
- `t2smark_full_main_reproduction_run.ipynb`: 运行 T2SMark 官方 SD3.5 路径并生成 governed import 候选记录。当前默认使用 pilot_paper prompt 文件、120个 prompt、fixed-FPR=0.01, 并写入 `/content/drive/MyDrive/SLM/pilot_paper_results/t2smark_full_main_reproduction`。
- `tree_ring_official_reference_run.ipynb`: 运行 Tree-Ring 官方原始环境参考复现, 默认样本数120, 结果写入 `/content/drive/MyDrive/SLM/pilot_paper_results/tree_ring_official_reference`。
- `gaussian_shading_official_reference_run.ipynb`: 运行 Gaussian Shading 官方原始环境参考复现, 默认样本数120, 结果写入 `/content/drive/MyDrive/SLM/pilot_paper_results/gaussian_shading_official_reference`。
- `shallow_diffuse_official_reference_run.ipynb`: 运行 Shallow Diffuse 官方原始环境参考复现, 默认样本数120, 结果写入 `/content/drive/MyDrive/SLM/pilot_paper_results/shallow_diffuse_official_reference`。

### 结果闭合入口

- `pilot_paper_result_closure_run.ipynb`: 在前序结果包已经写入 Google Drive 后, 从 `/content/drive/MyDrive/SLM/pilot_paper_results` 物化上游包, 依次重建 attack matrix、external baseline formal import、internal ablation、pilot_paper fixed-FPR 共同协议记录和完整结果包。该入口不需要 GPU, 只调度 `scripts/` 中的 repository commands, 不直接手写正式 records、tables、figures 或 reports。

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
7. 真实攻击闭环: `real_attack_evaluation_run.ipynb`。
8. 数据集级质量: `dataset_level_quality_run.ipynb`。
9. 主表 baseline: `external_baseline_gpu_smoke_run.ipynb`。
10. 官方复现: `t2smark_full_main_reproduction_run.ipynb`。
11. 官方复现: `tree_ring_official_reference_run.ipynb`。
12. 官方复现: `gaussian_shading_official_reference_run.ipynb`。
13. 官方复现: `shallow_diffuse_official_reference_run.ipynb`。
14. 结果闭合: `pilot_paper_result_closure_run.ipynb`。

第14个 Notebook 已经封装以下收尾命令。若不使用该 Notebook, 也可以在同一个 Colab 仓库工作区或本地仓库中手动执行以下等价命令。若在本地执行, `--package-search-root` 应指向已经同步或挂载的 Google Drive `SLM/pilot_paper_results` 目录。

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
  --package-search-root /content/drive/MyDrive/SLM/pilot_paper_results \
  --require-existing-evidence

python scripts/write_pilot_paper_fixed_fpr_common_protocol_outputs.py \
  --candidate-records-path outputs/pilot_paper_fixed_fpr_results/pilot_paper_result_records.jsonl \
  --require-existing-evidence

python scripts/write_pilot_paper_complete_result_package.py \
  --package-search-root /content/drive/MyDrive/SLM/pilot_paper_results \
  --drive-output-dir /content/drive/MyDrive/SLM/pilot_paper_results/complete_result_package
```

第一条 `write_pilot_paper_result_records.py --materialize-only` 命令用于把 Google Drive 结果包中的 `outputs/` 条目物化到当前仓库工作区, 供后续重建脚本读取。最后两条命令才生成正式的 pilot_paper result records 和共同协议导入报告。若模板覆盖仍不完整, 产物会保留 `pilot_paper_template_coverage_ready=false`, 这表示只能继续补齐缺失方法或攻击项, 不能提升为 full_paper 主张。
