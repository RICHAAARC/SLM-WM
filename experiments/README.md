# Experiments

`experiments/` 是核心方法复现层的一部分。该目录只保存 SLM 项目自身的实验协议、运行时配置、内部消融、主实验攻击和主方法 runner。

外部 baseline 的 method-faithful 适配、官方参考复现编排、受治理结果导入和公平对比不得放在这里, 应放入 `paper_experiments/`。

## 子目录职责

- `protocol/`: 保存 prompt split、fixed-FPR 校准、攻击矩阵、质量指标、正式证据门禁和论文运行层级配置。`formal_evidence.py` 负责拒绝 proxy、placeholder、fallback、synthetic 和 formal-null 证据进入三类正式结果记录。
- `artifacts/`: 保存 SLM 主方法实验 records、tables、manifests 和诊断报告的正式构建模块, 包括 prompt event protocol、semantic subspace、content carrier、attention latent update、attention geometry、geometric rescue 与 threshold calibration 产物构建。`scripts/` 可以包装这些模块形成命令行入口, 但核心实验 runner 不应反向导入 `scripts/`。
- `runners/`: 保存 SLM 主方法和主实验的服务器可复用 runner。该目录可以被 Colab 包装层调用, 但自身不得导入 `paper_workflow/` 或外部 baseline 适配工程。
- `runtime/`: 保存不依赖 Notebook 的运行时工具和模型适配代码。`runtime/repository_environment.py` 负责 Git 版本、依赖版本、文件摘要与运行环境报告, 可被服务器 runner 和 Colab 包装层共同复用。`runtime/progress.py` 负责总体工作量进度显示。

## 主要 runner

- `experiments/runners/attention_geometry_capture.py`: 真实 SD3.5 attention 载体或可审计 attention map 捕获。
- `experiments/runners/attention_latent_injection.py`: attention-relative latent update 写入和注入记录生成。
- `experiments/runners/aligned_rescoring.py`: aligned rescoring、配对质量指标和检测分数组件审计。
- `experiments/runners/threshold_calibration.py`: fixed-FPR 阈值校准、clean negative 边界和 geometric rescue 边界记录。
- `experiments/runners/real_attack_evaluation.py`: 再扩散、再生成、语义编辑和自适应去水印类真实攻击闭环。
- `experiments/runners/conventional_geometric_attack_evaluation.py`: 常规失真、几何变换与 photometric 攻击闭环。
- `experiments/runners/dataset_level_quality.py`: dataset-level FID / KID 质量证据导入、特征提取与打包。
- `experiments/runners/sd_runtime_cold_start.py`: 真实 SD runtime 诊断, 包括模型加载、latent trajectory 和环境报告。
- `experiments/runners/minimal_latent_injection.py`: 最小 diffusion latent injection 机制预检。

## 输出边界

本层持久化输出必须位于 `outputs/`。当由 Colab 或服务器包装层发布结果包时, runner 仍应先写入本地 `outputs/` 语义子目录, 再由包装层复制或打包到对应结果根目录。
