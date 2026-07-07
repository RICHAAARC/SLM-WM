# Experiments

`experiments/` 是核心方法复现层的一部分。该目录只保存 SLM 项目自身的实验协议、运行时配置、内部消融和主方法 runner。

外部 baseline 的 method-faithful 适配、官方参考复现编排、受治理结果导入和公平对比不得放在这里, 应放入 `paper_experiments/`。

## 子目录职责

- `protocol/`: 保存 prompt split、fixed-FPR 校准、攻击矩阵、质量指标和论文运行层级配置。
- `artifacts/`: 保存 SLM 主方法实验 records、tables、manifests 和诊断报告的正式构建模块, 包括 prompt event protocol、semantic subspace、content carrier、attention latent update、attention geometry、geometric rescue 与 threshold calibration 产物构建。`scripts/` 可以包装这些模块形成命令行入口, 但核心实验 runner 不应反向导入 `scripts/`。
- `runners/`: 保存 SLM 主方法和主实验的服务器可复用 runner。该目录可以被 Colab 包装层调用, 但自身不得导入 `paper_workflow/` 或外部 baseline 适配工程。
- `runtime/`: 保存不依赖 Notebook 的运行时工具和模型适配代码。`runtime/repository_environment.py` 负责 Git 版本、依赖版本、文件摘要与运行环境报告, 可被服务器 runner 和 Colab 包装层共同复用。`runtime/progress.py` 负责总体工作量进度显示, 使长耗时服务器命令和 Colab 包装层使用同一套进度语义。

- `experiments/runners/dataset_level_quality.py`: 数据集级 FID / KID 质量证据导入、特征提取与打包 runner。

- `experiments/runners/real_attack_evaluation.py`: 真实再扩散与语义编辑攻击闭环 runner, 负责 attacked image、formal detection records 与攻击 manifest。

- `experiments/runners/conventional_geometric_attack_evaluation.py`: 常规失真、几何变换与 photometric 攻击闭环 runner。

- `experiments/runners/sd_runtime_cold_start.py`: 真实 SD runtime 诊断 probe runner, 负责模型加载、latent trajectory 记录和 probe 打包。

- `experiments/runners/minimal_latent_injection.py`: 最小 diffusion latent injection 机制预检 runner, 负责 paired image、latent update records 和质量摘要打包。

- `experiments/runtime/archive_naming.py`: 服务器与 Notebook 共用的 workflow 归档命名工具。
