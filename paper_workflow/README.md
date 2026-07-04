# Paper Workflow

此目录保存论文相关 Notebook 入口和执行环境包装。Notebook 只能调度 repository modules, 不得成为唯一正式实现。

## 统一运行环境配置

所有论文入口统一使用 `paper_workflow/colab_utils/paper_run_environment.py` 派生运行环境。该 helper 根据 `SLM_WM_PAPER_RUN_NAME` 写入 prompt 文件、样本数、目标 FPR、协议 profile、Google Drive 根目录和各子流程输出目录。

Notebook 顶部只应保留容易人工确认的入口变量:

```python
SLM_WM_PAPER_RUN_NAME = "pilot_paper"
```

四个 method-faithful 外部 baseline 入口还会保留单方法选择变量:

```python
SLM_WM_PRIMARY_BASELINE_METHODS = "tree_ring"
```

该变量只用于防止单个 baseline Notebook 误跑其他方法, 不表示只支持 Tree-Ring。其他运行配置不应继续散落在 Notebook cell 中。若需要修复配置、命令拼接、压缩包命名或 Google Drive 子目录, 优先修改 `paper_workflow/colab_utils/` 或 `scripts/`, 不应把正式逻辑写入 Notebook。

## 入口分组

### 诊断入口

- `colab_drive_cold_start_smoke.ipynb`: 验证 Colab 挂载 Google Drive、拉取仓库、镜像本地 `outputs/`、写入工作流清单并重载校验。该入口已经合并原独立 reload 功能。
- `runtime_method_precheck_run.ipynb`: 合并运行时诊断与最小机制预检。该入口验证 Colab 依赖组合、Hugging Face 登录、SD3.5 Medium 加载、真实 latent trajectory 捕获和最小 latent injection 闭环。该入口不参与 `pilot_paper` 或 `full_paper` 的正式统计产出。

### 方法主流程入口

以下入口统一由 `SLM_WM_PAPER_RUN_NAME` 派生协议、prompt 文件、样本数、目标 FPR 和 Google Drive 结果根目录。默认值为 `pilot_paper`; 切换为 `full_paper` 时, helper 会重新写入 `SLM_WM_PROTOCOL_PROFILE`、`SLM_WM_PROMPT_SET`、`SLM_WM_PROMPT_FILE`、`SLM_WM_PAPER_RUN_TARGET_FPR` 和各子流程 Drive 目录, 避免沿用旧运行层级的环境变量。

- `attention_geometry_capture_run.ipynb`: 捕获真实 attention 载体或可审计 attention map, 生成 attention geometry 产物。
- `attention_latent_injection_run.ipynb`: 基于 attention geometry 执行 attention-relative latent update, 生成注入记录和配对质量记录。
- `aligned_rescoring_run.ipynb`: 对注入结果进行 aligned rescoring, 生成检测重打分记录和配对感知质量指标。`pilot_paper` 默认使用该层级全部 prompt, 当前配置为600个; `full_paper` 当前配置为6000个。
- `threshold_calibration_run.ipynb`: 按当前运行层级的 fixed-FPR 协议校准阈值, 并记录 geometric rescue 边界。`pilot_paper` 使用 FPR=0.01, `full_paper` 使用 FPR=0.001。
- `real_attack_evaluation_run.ipynb`: 生成或导入真实 attacked image, 执行正式攻击后检测, 记录 source / attacked image digest 和真实 GPU 攻击覆盖状态。该入口负责再扩散、全局编辑、局部编辑、视觉改写和自适应去水印类攻击。处理数量由当前论文运行层级的样本数派生。
- `conventional_geometric_attack_evaluation_run.ipynb`: 使用 CPU / PIL 图像算子生成常规失真、几何变换与 photometric attacked image, 写出 source / attacked image digest、formal detection records 和总体进度条。正式检测口径需要对 attacked image 执行 SD3.5 detector / VAE latent rescore, 因此论文运行应使用 Colab GPU。该入口负责补齐 `standard_distortion`、`geometric_transform` 与 `photometric_distortion_attack` 的真实图像级攻击闭环。
- `dataset_level_quality_run.ipynb`: 从受治理图像集合导入 dataset-level 质量特征, 计算或登记 FID / KID 等集合级指标。`pilot_paper` 默认使用100作为 dataset-level 正式特征最小样本阈值, 其结果只允许支撑 `pilot_paper` 样本规模内的主张, 不得提升为 `full_paper` 主张。

### 外部 baseline 入口

method-faithful 入口把外部方法适配到 SD3.5 共同协议, 便于与主方法共享 prompt split、攻击矩阵和 fixed-FPR 校准边界。

- `external_baseline_tree_ring_run.ipynb`: 只运行 Tree-Ring 的 SD3.5 method-faithful adapter, 覆盖共同攻击矩阵中的常规失真、几何变换、photometric、再生成和高级编辑攻击, 并写出 `split_observations/tree_ring_baseline_observations.json`。
- `external_baseline_gaussian_shading_run.ipynb`: 只运行 Gaussian Shading 的 SD3.5 method-faithful adapter, 覆盖共同攻击矩阵中的常规失真、几何变换与再生成攻击, 并写出 `split_observations/gaussian_shading_baseline_observations.json`。
- `external_baseline_shallow_diffuse_run.ipynb`: 只运行 Shallow Diffuse 的 SD3.5 method-faithful adapter, 覆盖共同攻击矩阵中的常规失真、几何变换与再生成攻击, 并写出 `split_observations/shallow_diffuse_baseline_observations.json`。
- `external_baseline_t2smark_run.ipynb`: 只运行 T2SMark 官方 SD3.5 路径和 adapter, 冷启动时会补丁为同一攻击簇输出正式攻击分数, 并写出 `split_observations/t2smark_baseline_observations.json`。

method-faithful 入口统一把压缩包镜像到当前论文运行层级的 Google Drive 子目录 `external_baseline_method_faithful/`。

### 官方参考复现入口

官方参考复现入口用于记录外部方法在其官方或 legacy 环境下的参考结果, 再通过 governed import 协议进入对比表。该链路不替代 SD3.5 method-faithful adapter, 而是用于说明 baseline 复现来源和方法忠实性边界。

- `official_reference_t2smark_run.ipynb`: 运行 T2SMark 官方 SD3.5 路径并生成 governed import 候选记录。prompt 文件、样本数和目标 FPR 由当前论文运行层级派生。
- `official_reference_tree_ring_run.ipynb`: 运行 Tree-Ring 官方原始环境参考复现, 样本数和 Drive 目录由当前论文运行层级派生。
- `official_reference_gaussian_shading_run.ipynb`: 运行 Gaussian Shading 官方原始环境参考复现, 样本数和 Drive 目录由当前论文运行层级派生。
- `official_reference_shallow_diffuse_run.ipynb`: 运行 Shallow Diffuse 官方原始环境参考复现, 样本数和 Drive 目录由当前论文运行层级派生。

四个官方参考复现入口统一把压缩包镜像到当前论文运行层级的 Google Drive 子目录 `external_baseline_official_reference/`。本地 `outputs/` 子目录仍保留各方法原有语义名称, 用于结果闭合脚本读取和产物来源审计。

### 结果闭合入口

- `pilot_paper_result_closure_run.ipynb`: 在前序结果包已经写入 Google Drive 后, 从当前论文运行层级的 Drive 根目录物化上游包, 依次重建 attack matrix、external baseline formal import、internal ablation、fixed-FPR 共同协议记录和完整结果包。该入口不需要 GPU, 只调度 `scripts/` 和 `paper_workflow/colab_utils/paper_result_closure.py`, 不直接手写正式 records、tables、figures 或 reports。

## `pilot_paper` 与 `full_paper` 运行原则

`pilot_paper` 与 `full_paper` 应共享同一批方法主流程和 baseline 入口, 只通过配置切换 prompt split、样本量、随机种子、Drive 输出根目录、攻击覆盖范围、目标 FPR 和 bootstrap 次数。不得为 `pilot_paper` 和 `full_paper` 维护两套互相分叉的 Notebook 逻辑。`pilot_paper` 是受样本规模约束的论文主张层级, 不是仅用于链路调试的临时层级。

除诊断入口外, 正式运行 Notebook 的第一个代码行固定为:

```python
SLM_WM_PAPER_RUN_NAME = "pilot_paper"
```

切换到完整论文运行层级时只修改这一行。依赖诊断由 `paper_workflow.colab_utils.dependency_check.build_notebook_dependency_report` 统一执行, archive 文件名和 Drive 镜像打包由 `paper_workflow.colab_utils.notebook_entrypoint.package_workflow_outputs` 统一执行。Notebook 不再维护单独的短提交读取、UTC 后缀拼接或具体 `package_*_outputs` 调用, 以避免项目代码调试时同步修改多个 Notebook。

统一 archive 打包入口会在对应 `outputs/<workflow>/notebook_runtime_report.json` 写入运行时间报告, 并把该报告纳入当前结果包。报告字段包括 `notebook_runtime_started_at`、`notebook_runtime_finished_at`、`notebook_runtime_elapsed_seconds`、`notebook_runtime_start_source` 和 `notebook_runtime_timing_boundary`。该报告只用于 Colab 运行观测和耗时审计, 不作为方法有效性或论文结论的证据来源。

诊断入口不参与 `pilot_paper` 或 `full_paper` 的正式统计产出。它们只用于在 Colab 环境、模型依赖或 Drive 持久化出问题时进行预检。

## Notebook 运行依赖与并行关系

下表只描述 Notebook 入口之间的运行依赖。具体配置、路径、打包命名和依赖诊断仍由 `paper_workflow/colab_utils/` 与 `scripts/` 统一实现, Notebook 不承载正式算法或统计逻辑。

| Notebook 入口 | GPU 要求 | 必须依赖的前序结果包 | 并行关系 | 说明 |
|---|---|---|---|---|
| `colab_drive_cold_start_smoke.ipynb` | 不需要 | 无 | 可独立运行 | 仅检查 Drive 挂载、仓库拉取、结果包镜像和清单重载。 |
| `runtime_method_precheck_run.ipynb` | 需要 | 无 | 可独立运行 | 检查 SD3.5 Medium 加载、真实 latent trajectory 和最小 latent injection 闭环, 不进入正式统计。 |
| `attention_geometry_capture_run.ipynb` | 需要 | 无 | 主方法链路起点 | 生成后续注入与重打分需要的 attention geometry 包。 |
| `attention_latent_injection_run.ipynb` | 需要 | `attention_geometry` | 不能早于 `attention_geometry_capture_run.ipynb` | 读取 attention geometry, 执行 attention-relative latent update。 |
| `aligned_rescoring_run.ipynb` | 需要 | `attention_geometry`, `attention_latent_injection` | 不能早于主方法注入结果 | 生成正式 fixed-FPR 校准与攻击检测需要的 real aligned scores 和图像。 |
| `threshold_calibration_run.ipynb` | 通常不需要; 可复用 GPU 会话 | `attention_latent_injection`, `aligned_rescoring` | 不能早于 aligned rescoring | 校准 fixed-FPR 阈值, 记录 rescue 边界和统计口径。 |
| `real_attack_evaluation_run.ipynb` | 需要 | `aligned_rescoring`, `threshold_calibration` | 可与 `conventional_geometric_attack_evaluation_run.ipynb` 并行 | 覆盖再扩散、再生成、全局编辑、局部编辑、视觉改写和自适应去水印类真实攻击闭环。 |
| `conventional_geometric_attack_evaluation_run.ipynb` | 需要 | `aligned_rescoring`, `threshold_calibration` | 可与 `real_attack_evaluation_run.ipynb` 并行 | 攻击图像生成主要是 CPU / PIL 算子, 但正式 attacked image latent rescore 需要 SD3.5 detector / VAE pipeline, 因此论文运行使用 GPU。 |
| `dataset_level_quality_run.ipynb` | 建议使用 GPU | `aligned_rescoring`, `real_attack_evaluation`, `conventional_geometric_attack_evaluation` | 必须等待攻击产物齐备 | 计算或导入 dataset-level FID / KID 特征, 不替代攻击检测。 |
| `external_baseline_tree_ring_run.ipynb` | 需要 | 无; 共享当前 prompt 与配置 | 可与其他 external baseline 入口并行 | 生成 Tree-Ring method-faithful SD3.5 观测记录。 |
| `external_baseline_gaussian_shading_run.ipynb` | 需要 | 无; 共享当前 prompt 与配置 | 可与其他 external baseline 入口并行 | 生成 Gaussian Shading method-faithful SD3.5 观测记录。 |
| `external_baseline_shallow_diffuse_run.ipynb` | 需要 | 无; 共享当前 prompt 与配置 | 可与其他 external baseline 入口并行 | 生成 Shallow Diffuse method-faithful SD3.5 观测记录。 |
| `external_baseline_t2smark_run.ipynb` | 需要 | 无; 共享当前 prompt 与配置 | 可与其他 external baseline 入口并行 | 生成 T2SMark method-faithful SD3.5 观测记录。 |
| `official_reference_t2smark_run.ipynb` | 需要 | 无; 共享当前 prompt 与配置 | 可与其他 official reference 入口并行 | 生成 T2SMark 官方路径 governed import 候选记录。 |
| `official_reference_tree_ring_run.ipynb` | 建议使用 GPU | 无; 共享当前 prompt 与配置 | 可与其他 official reference 入口并行 | 生成 Tree-Ring 官方原始环境参考记录。 |
| `official_reference_gaussian_shading_run.ipynb` | 建议使用 GPU | 无; 共享当前 prompt 与配置 | 可与其他 official reference 入口并行 | 生成 Gaussian Shading 官方原始环境参考记录。 |
| `official_reference_shallow_diffuse_run.ipynb` | 建议使用 GPU | 无; 共享当前 prompt 与配置 | 可与其他 official reference 入口并行 | 生成 Shallow Diffuse 官方原始环境参考记录。 |
| `pilot_paper_result_closure_run.ipynb` | 不需要 | 所有需要纳入完整结果包的前序结果包 | 必须最后运行 | 物化 Drive 结果包, 重建 attack matrix、baseline import、ablation、fixed-FPR common protocol 和完整结果包。 |


## Colab `pilot_paper` 重跑顺序

建议在清理 `/content/drive/MyDrive/SLM/pilot_paper_results/` 后按以下顺序重跑。该顺序只依赖 Google Drive 结果包, 不要求本地 `outputs/` 中存在历史文件。

1. 可选诊断: `colab_drive_cold_start_smoke.ipynb`。
2. 可选诊断: `runtime_method_precheck_run.ipynb`。
3. 主方法串行链路: `attention_geometry_capture_run.ipynb`。
4. 主方法串行链路: `attention_latent_injection_run.ipynb`。
5. 主方法串行链路: `aligned_rescoring_run.ipynb`。
6. 阈值与 rescue 边界: `threshold_calibration_run.ipynb`。
7. 攻击闭环并行批次: `real_attack_evaluation_run.ipynb` 与 `conventional_geometric_attack_evaluation_run.ipynb` 可在两个 Colab 会话中并行运行。
8. 数据集级质量: `dataset_level_quality_run.ipynb`, 必须等待两个攻击闭环入口都完成。
9. method-faithful external baseline 并行批次: `external_baseline_tree_ring_run.ipynb`、`external_baseline_gaussian_shading_run.ipynb`、`external_baseline_shallow_diffuse_run.ipynb`、`external_baseline_t2smark_run.ipynb` 可分别在独立 Colab 会话中并行运行。
10. official reference 并行批次: `official_reference_t2smark_run.ipynb`、`official_reference_tree_ring_run.ipynb`、`official_reference_gaussian_shading_run.ipynb`、`official_reference_shallow_diffuse_run.ipynb` 可分别在独立 Colab 会话中并行运行。
11. 结果闭合: `pilot_paper_result_closure_run.ipynb`, 必须等待本次要纳入完整结果包的所有前序结果包写入 Drive 后再运行。

## 结果闭合等价命令

`paper_workflow/colab_utils/paper_result_closure.py` 会生成并执行以下等价命令。若不使用结果闭合 Notebook, 也可以在同一个 Colab 仓库工作区或本地仓库中手动执行。若在本地执行, `--package-search-root` 应指向已经同步或挂载的 Google Drive `SLM/pilot_paper_results` 目录。

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

第一条 `write_pilot_paper_result_records.py --materialize-only` 命令用于把 Google Drive 结果包中的 `outputs/` 条目物化到当前仓库工作区, 供后续重建脚本读取。后续结果记录命令不应再次传入 `--package-search-root`, 否则会重复解压大包。完整结果包命令默认使用 `--skip-package-materialization` 复用已物化的本地 `outputs/`, 并用 `--zip-compression stored` 避免对 PNG 和已有压缩内容重复压缩。若模板覆盖仍不完整, 产物会保留 `pilot_paper_template_coverage_ready=false`, 这表示只能继续补齐缺失方法或攻击项, 不能提升为 `full_paper` 主张。
