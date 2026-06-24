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
- `aligned_rescoring_run.ipynb`: 对注入结果进行 aligned rescoring, 生成检测重打分记录和配对感知质量指标。pilot_paper 默认最多重打分5个 carrier。
- `threshold_calibration_run.ipynb`: 在 fixed-FPR=0.01 协议下校准阈值, 并记录 geometric rescue 边界。
- `real_attack_evaluation_run.ipynb`: 生成或导入真实 attacked image, 执行正式攻击后检测, 记录 source / attacked image digest 和再扩散攻击覆盖状态。pilot_paper 默认最多处理5张 source image。
- `dataset_level_quality_run.ipynb`: 从受治理图像集合导入 dataset-level 质量特征, 计算或登记 FID / KID 等集合级指标。pilot_paper 默认使用5作为小样本论文配置阈值, 其结果只允许支撑 pilot_paper 样本规模内的主张, 不得提升为 full_paper 主张。

### 外部 baseline 入口

- `external_baseline_gpu_smoke_run.ipynb`: 运行主表 baseline 的 SD3.5 method-faithful adapter 链路, 当前覆盖 Tree-Ring、Gaussian Shading、Shallow Diffuse 和 T2SMark。当前默认写入 `/content/drive/MyDrive/SLM/pilot_paper_results/external_baseline_gpu_smoke`, 共享样本数为5。
- `t2smark_full_main_reproduction_run.ipynb`: 运行 T2SMark 官方路径并生成 governed import 候选记录。该入口属于专项复现入口, 不应替代共同协议入口。
- `tree_ring_official_reference_run.ipynb`: 运行 Tree-Ring 官方原始环境参考复现, 生成补充表 governed import 记录和 Google Drive 压缩包。
- `gaussian_shading_official_reference_run.ipynb`: 运行 Gaussian Shading 官方原始环境参考复现, 生成补充表 governed import 记录和 Google Drive 压缩包。
- `shallow_diffuse_official_reference_run.ipynb`: 运行 Shallow Diffuse 官方原始环境参考复现, 生成补充表 governed import 记录和 Google Drive 压缩包。

## pilot_paper 与 full_paper 运行原则

pilot_paper 与 full_paper 应共享同一批方法主流程和 baseline 入口, 只通过配置切换 prompt split、样本量、随机种子、Drive 输出根目录、攻击覆盖范围和 bootstrap 次数。不得为 pilot_paper 和 full_paper 维护两套互相分叉的 Notebook 逻辑。pilot_paper 是受样本规模约束的论文主张层级, 不是仅用于链路调试的临时层级。

诊断入口不参与 pilot_paper 或 full_paper 的正式统计产出。它们只用于在 Colab 环境、模型依赖或 Drive 持久化出问题时进行预检。
