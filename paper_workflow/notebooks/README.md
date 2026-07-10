# Notebook 入口目录

该目录只保存 Colab / Notebook 入口文件。Notebook 负责选择运行层级、挂载远程存储、调用 repository modules 和触发打包落盘, 不承载正式算法、统计协议或论文结果生成的唯一实现。

正式逻辑应位于 `main/`、`experiments/`、`paper_experiments/` 或 `scripts/`。

## 运行层级切换

每个正式 Notebook 顶部应保留独立可见的运行层级变量:

```python
SLM_WM_PAPER_RUN_NAME = "pilot_paper"
```

可选值为 `probe_paper`、`pilot_paper` 和 `full_paper`。三者都要求正式证据包; 区别只在 prompt 数量和 fixed-FPR 标准:

| 运行层级 | prompt 数量 | 目标 FPR | 支持主张 |
| --- | ---: | ---: | --- |
| `probe_paper` | 60 | 0.1 | `probe_claim` |
| `pilot_paper` | 600 | 0.01 | `pilot_claim` |
| `full_paper` | 6000 | 0.001 | `full_claim` |

## Notebook 前后依赖、并行要求与资源需求

| 流程节点 | Notebook | 资源需求 | 前置结果包 | 并行要求 | 主要输出 |
| --- | --- | --- | --- | --- | --- |
| Drive 冷启动诊断 | `colab_drive_cold_start_smoke.ipynb` | CPU 即可 | 无 | 可独立运行 | Drive 挂载、仓库拉取、输出镜像和 reload 清单。 |
| 运行时与机制预检 | `runtime_method_precheck_run.ipynb` | 需要 GPU | 无 | 可独立运行 | SD3.5 加载、latent trajectory 和最小 latent injection 诊断包; 不进入正式 claim-ready 统计。 |
| attention geometry 捕获 | `attention_geometry_capture_run.ipynb` | 需要 GPU | 无 | 主方法串行起点 | `attention_geometry` 结果包, 包含真实 attention 载体或可审计 attention map。 |
| latent 写入 | `attention_latent_injection_run.ipynb` | 需要 GPU | `attention_geometry` | 必须等待 geometry 完成 | `attention_latent_injection` 结果包, 包含注入记录、图像 digest 和配对质量记录。 |
| aligned rescoring | `aligned_rescoring_run.ipynb` | 需要 GPU | `attention_geometry`, `attention_latent_injection` | 必须等待 latent 写入完成 | `aligned_rescoring` 结果包, 包含真实重打分、分数组件审计和配对质量指标。 |
| fixed-FPR 校准 | `threshold_calibration_run.ipynb` | CPU 通常即可; GPU 非必需 | `attention_latent_injection`, `aligned_rescoring` | 必须等待 aligned rescoring 完成 | `threshold_calibration` 与 `geometric_rescue` 结果包, 包含阈值、clean negative 边界和 rescue 边界。 |
| 再扩散与语义编辑攻击 | `real_attack_evaluation_run.ipynb` | 需要 GPU | `aligned_rescoring`, `threshold_calibration` | 可与常规失真和几何变换攻击并行 | `real_attack_evaluation` 与 `image_attack_evidence` 结果包, 覆盖再扩散、再生成、语义编辑和自适应去水印攻击。 |
| 常规失真和几何变换攻击 | `conventional_geometric_attack_evaluation_run.ipynb` | 需要 GPU | `aligned_rescoring`, `threshold_calibration` | 可与再扩散与语义编辑攻击并行 | `conventional_geometric_attack_evaluation` 与 `image_attack_evidence` 结果包; 图像算子多为 CPU/PIL, 但正式检测重打分需要 SD3.5 detector / VAE 路径。 |
| dataset-level 质量 | `dataset_level_quality_run.ipynb` | 建议 GPU; CPU 可运行但较慢 | `aligned_rescoring`, `real_attack_evaluation`, `conventional_geometric_attack_evaluation` | 必须等待攻击结果齐备 | `dataset_level_quality` 结果包, `dataset_quality_metrics.csv` 只包含正式 FID / KID, pixel proxy 仅进入诊断表。 |
| Tree-Ring method-faithful | `external_baseline_tree_ring_run.ipynb` | 需要 GPU | 无; 共享当前 prompt 配置 | 可与其他 baseline 入口并行 | `external_baseline_method_faithful` 结果包中的 Tree-Ring SD3.5 观测记录。 |
| Gaussian Shading method-faithful | `external_baseline_gaussian_shading_run.ipynb` | 需要 GPU | 无; 共享当前 prompt 配置 | 可与其他 baseline 入口并行 | `external_baseline_method_faithful` 结果包中的 Gaussian Shading SD3.5 观测记录。 |
| Shallow Diffuse method-faithful | `external_baseline_shallow_diffuse_run.ipynb` | 需要 GPU | 无; 共享当前 prompt 配置 | 可与其他 baseline 入口并行 | `external_baseline_method_faithful` 结果包中的 Shallow Diffuse SD3.5 观测记录。 |
| T2SMark method-faithful | `external_baseline_t2smark_run.ipynb` | 需要 GPU | 无; 共享当前 prompt 配置 | 可与其他 baseline 入口并行 | `external_baseline_method_faithful` 结果包中的 T2SMark SD3.5 观测记录。 |
| Tree-Ring official reference | `official_reference_tree_ring_run.ipynb` | 建议 GPU; legacy 环境准备耗时 | 无; 共享当前 prompt 配置 | 可与其他 official reference 入口并行 | `external_baseline_official_reference` 结果包中的 Tree-Ring 官方参考记录。 |
| Gaussian Shading official reference | `official_reference_gaussian_shading_run.ipynb` | 建议 GPU; legacy 环境准备耗时 | 无; 共享当前 prompt 配置 | 可与其他 official reference 入口并行 | `external_baseline_official_reference` 结果包中的 Gaussian Shading 官方参考记录。 |
| Shallow Diffuse official reference | `official_reference_shallow_diffuse_run.ipynb` | 建议 GPU; legacy 环境准备耗时 | 无; 共享当前 prompt 配置 | 可与其他 official reference 入口并行 | `external_baseline_official_reference` 结果包中的 Shallow Diffuse 官方参考记录。 |
| T2SMark official reference | `official_reference_t2smark_run.ipynb` | 需要 GPU | 无; 共享当前 prompt 配置 | 可与其他 official reference 入口并行 | `external_baseline_official_reference` 结果包中的 T2SMark 官方 SD3.5 参考记录。 |
| 结果闭合 | `paper_result_closure_run.ipynb` | CPU 即可; 大包解压和打包依赖磁盘 I/O | 所有需要纳入完整结果包的前序结果包 | 必须最后运行 | common protocol、attack matrix、baseline import、ablation、result analysis 和完整结果包。 |

## 推荐运行顺序

1. 可选诊断: `colab_drive_cold_start_smoke.ipynb`。
2. 可选预检: `runtime_method_precheck_run.ipynb`。
3. 主方法串行链路: `attention_geometry_capture_run.ipynb` -> `attention_latent_injection_run.ipynb` -> `aligned_rescoring_run.ipynb` -> `threshold_calibration_run.ipynb`。
4. 攻击闭环并行批次: `real_attack_evaluation_run.ipynb` 与 `conventional_geometric_attack_evaluation_run.ipynb` 可在两个 Colab 会话中并行运行。
5. 数据集级质量: `dataset_level_quality_run.ipynb`, 必须等待两个攻击闭环入口都完成。
6. method-faithful baseline 并行批次: 四个 `external_baseline_*_run.ipynb` 可分别在独立 Colab 会话中并行运行。
7. official reference 并行批次: 四个 `official_reference_*_run.ipynb` 可分别在独立 Colab 会话中并行运行。
8. 结果闭合: `paper_result_closure_run.ipynb`, 必须等待本次要纳入完整结果包的所有结果包写入 Drive 后再运行。

## 落盘约定

Colab 入口统一把结果镜像到当前运行层级的 Google Drive 根目录:

```text
/content/drive/MyDrive/SLM/<run_name>_results/
```

其中 `<run_name>` 为 `probe_paper`、`pilot_paper` 或 `full_paper`。本地 `outputs/` 只用于当前会话工作区和审计副本, 不应作为 Colab 上游输入路径。
