# 三层结构与发布边界

## 文档定位

本文档说明 SLM-WM 仓库的三层结构。该结构用于同时满足三个目标: 核心方法可最小发布、完整论文实验可在服务器复现、Colab 只作为远程运行入口。

## 结构总览

```text
paper_workflow/ -> paper_experiments/ -> main/ 与 experiments/
```

该依赖方向表示: Colab 入口可以调用完整论文实验层, 完整论文实验层可以调用核心方法复现层, 但反向依赖被禁止。

## 核心方法复现层

核心方法复现层包含:

```text
main/
experiments/
configs/
```

该层的职责是实现 SLM-WM 方法机制、主方法攻击协议、阈值协议、内部消融和服务器运行能力。该层不包含 Notebook, 不包含外部 baseline 源码缓存, 也不包含 baseline method-faithful adapter。

`experiments/runtime/repository_environment.py` 属于核心方法复现层的共享运行环境工具。它集中提供 Git 版本、依赖版本、文件摘要和运行环境报告, 使正式实验 runner 不需要为了记录复现信息而依赖 Colab helper。

`experiments/runtime/progress.py` 属于核心方法复现层的共享进度工具。它只负责总体工作量显示和长命令心跳, 不写正式 records、tables、figures、reports 或 manifests, 因而可以被服务器 runner、完整论文实验 runner 和 Colab 包装层共同复用。

`experiments/artifacts/` 保存 SLM 主方法实验产物构建模块。该目录承载核心实验 records、tables、manifests 与诊断报告的正式生成逻辑; `scripts/` 只作为命令行兼容入口包装这些模块, 避免核心方法复现层反向依赖命令层。

`experiments/runners/attention_geometry_capture.py` 保存真实 attention 几何捕获正式调度逻辑。该 runner 负责加载 SD3.5 运行时、捕获可审计 attention map 并调用核心产物构建模块写出 attention graph 与 geometry evidence。

`experiments/runners/attention_latent_injection.py` 保存 attention-relative latent update 正式调度逻辑。该 runner 读取 attention geometry 包、生成内容载体与语义子空间产物、执行真实 SD3.5 latent injection, 并复用 `experiments/runtime/diffusion/sd3_pipeline_runtime.py` 加载模型和计算轻量 paired image 质量指标。

`experiments/runners/aligned_rescoring.py` 保存 aligned rescoring 正式调度逻辑。该 runner 复用 attention latent injection 的内容载体和水印张量构造能力, 在真实 SD3.5 latent 空间中生成 raw / aligned 成对重打分记录、component audit 和 paired quality metrics。

`experiments/runners/threshold_calibration.py` 保存 SLM 主方法 fixed-FPR 阈值校准与 geometric rescue 边界记录的正式调度逻辑。该 runner 属于核心方法复现层, 可以被服务器入口和 Colab 包装层复用, 但不得导入 Colab 层或外部 baseline 适配工程。

## 完整论文实验层

完整论文实验层包含:

```text
paper_experiments/
paper_experiments/runners/
```

该层的职责是实现论文所需的完整实验证据链, 包括外部 baseline 适配、官方参考复现编排、受治理结果导入、共同攻击协议对齐、对比表构造和结果闭合。该层可以依赖核心方法复现层, 但不得依赖 Colab 运行层。

`paper_experiments/runners/external_baseline_method_faithful.py` 保存外部 baseline method-faithful 正式调度逻辑。Colab 层只允许通过兼容包装调用该实现, 不得重新维护独立的 baseline 调度分支。

`paper_experiments/runners/tree_ring_official_reference.py` 保存 Tree-Ring 官方参考复现正式调度逻辑。该 runner 可以调用完整论文实验层中的 baseline schema 与核心运行环境工具, 但不得导入 Colab 层。

`paper_experiments/runners/gaussian_shading_official_reference.py` 保存 Gaussian Shading 官方参考复现正式调度逻辑。该 runner 记录 strict / compatible legacy 环境边界、官方命令结果和 governed import 记录, 不得导入 Colab 层。

`paper_experiments/runners/shallow_diffuse_official_reference.py` 保存 Shallow Diffuse 官方参考复现正式调度逻辑。该 runner 记录源码运行边界修补、legacy 环境、官方命令结果和 governed import 记录, 不得导入 Colab 层。

`paper_experiments/runners/t2smark_full_main_reproduction.py` 保存 T2SMark 官方 SD3.5 路径复现正式调度逻辑。该 runner 生成 fixed-FPR 候选记录与 governed import 校验结果, 不得导入 Colab 层。

## Colab 运行层

Colab 运行层包含:

```text
paper_workflow/
paper_workflow/notebooks/
```

该层的职责是提供 Notebook 入口、Google Drive 落盘、Colab session helper 和远程运行包装。Notebook 不得成为正式 records、tables、figures、reports 或 manifests 的唯一实现。

## 发布边界

最小方法发布包默认只包含核心方法复现层。完整论文实验发布包可以包含完整论文实验层, 但仍应排除 Colab 运行层和未受治理的第三方源码缓存。

## 迁移要求

1. baseline 适配、官方参考复现和受治理导入应放入 `paper_experiments/`。
2. `experiments/` 只保留 SLM 项目自身实验协议、运行配置、内部消融和主方法 runner。
3. 新增服务器运行入口应优先调用核心方法复现层或完整论文实验层, 不应依赖 Notebook helper。
4. 若保留旧路径兼容层, 必须说明其兼容目的, 并由测试覆盖迁移后的正式路径。

- `experiments/runners/dataset_level_quality.py`: 数据集级质量证据 runner, 供服务器命令与 Colab 入口共同调用。
- `experiments/artifacts/dataset_level_quality_outputs.py`: 数据集级质量 records、metrics、summary 与 manifest 构建实现。

- `experiments/runners/real_attack_evaluation.py`: 真实再扩散与语义编辑攻击闭环 runner, 供服务器命令与 Colab 入口共同调用。

- `experiments/runners/conventional_geometric_attack_evaluation.py`: 常规失真、几何变换与 photometric 攻击闭环 runner, 供服务器命令与 Colab 入口共同调用。

- `experiments/runners/sd_runtime_cold_start.py`: 真实 SD runtime 诊断 probe runner, 供服务器命令与 Colab 入口共同调用。

- `experiments/runners/minimal_latent_injection.py`: 最小 diffusion latent injection 机制预检 runner, 供服务器命令与 Colab 入口共同调用。

- `experiments/runtime/archive_naming.py`: 服务器与 Notebook 共用的归档命名工具, 避免服务器 runner 依赖 Notebook helper。
