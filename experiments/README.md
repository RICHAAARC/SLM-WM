# Experiments

`experiments/` 是核心方法复现层的一部分。该目录只保存 SLM-WM 自身的实验协议、运行时配置、正式机制消融、主实验攻击和主方法 runner。

外部 baseline 的 method-faithful 适配、官方参考复现编排、受治理结果导入和公平对比不得放在这里, 应放入 `paper_experiments/`。

## 子目录职责

- `protocol/`: 保存 Prompt split、fixed-FPR 校准、攻击矩阵、质量指标、正式证据门禁和论文运行配置。`formal_evidence.py` 负责拒绝 proxy、placeholder、fallback、synthetic 和 formal-null 证据进入正式结果记录。
- `artifacts/`: 保存 SLM-WM records、tables、manifests 和诊断报告的构建模块, 包括 Prompt protocol、semantic subspace、dataset-level quality 与 fixed-FPR 结果构建。`scripts/` 可以包装这些模块形成命令行入口, 但核心实验 runner 不应反向导入 `scripts/`。
- `runners/`: 保存主方法、攻击和数据集协议的服务器可复用 runner。该目录可以被 Colab 包装层调用, 但自身不得导入 `paper_workflow/` 或外部 baseline 适配工程。
- `ablations/`: 保存消融配置和真实重跑逻辑。正式消融必须重新生成、重新攻击和重新检测, 不能修改历史分数模拟机制变化。
- `runtime/`: 保存不依赖 Notebook 的模型适配、仓库环境和进度工具。

## 当前正式 runner

- `experiments/runners/semantic_watermark_runtime.py`: 在真实 SD3/SD3.5 采样回调中执行分支风险、精确 JVP/SVD、空间 LF、高斯幅值尾部截断和真实 Q/K 几何更新。
- `experiments/runners/image_only_dataset_runtime.py`: 运行完整 Prompt split, 只从最终图像重建检测输入, 在 calibration split 冻结完整 evidence 协议, 并在 test split 执行攻击评估。
- `experiments/ablations/runtime_rerun.py`: 对每个机制配置重新生成图像、重新攻击和重新检测, 形成正式机制消融。
- `experiments/runners/real_attack_evaluation.py`: 执行再扩散、再生成、语义编辑和自适应去水印类真实攻击。
- `experiments/runners/conventional_geometric_attack_evaluation.py`: 执行常规失真、几何变换与 photometric 攻击。
- `experiments/artifacts/dataset_level_quality_outputs.py`: 通过 torch-fidelity 0.4.0 的 `inception-v3-compat` 2048 维特征构建正式 FID/KID 证据; pixel proxy 只能进入诊断表。

## 独立诊断 runner

`attention_geometry_capture.py`、`attention_latent_injection.py`、`aligned_rescoring.py` 和 `threshold_calibration.py` 用于分量级诊断。它们允许访问生成轨迹或样本级中间状态, 因而不能替代仅图像盲检正式链路。`sd_runtime_cold_start.py` 与 `minimal_latent_injection.py` 只用于运行环境和最小写入预检。

## 载体术语边界

当前正式分支名称为 `lf_content`、`tail_robust` 和 `attention_geometry`。`tail_robust` 算子按高斯元素绝对幅值执行尾部截断, 没有空间频率变换或频带定义。

## 输出边界

本层持久化输出必须位于 `outputs/`。当由 Colab 或服务器包装层发布结果包时, runner 仍应先写入本地 `outputs/` 语义子目录, 再由包装层复制或打包到对应结果根目录。
