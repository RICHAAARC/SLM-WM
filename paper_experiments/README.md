# paper_experiments

`paper_experiments/` 是完整论文实验层。该目录用于保存外部 baseline 适配、官方参考复现编排、受治理结果导入、共同协议对比和论文证据闭合代码。

该目录可以依赖 `main/` 与 `experiments/`, 但不得依赖 `paper_workflow/`。Notebook 入口只能调用这里的正式实现, 不能把正式论文结果逻辑写在 cell 中。

## 子目录

```text
baselines/   外部 baseline 适配、官方参考复现和受治理导入协议
runners/     完整论文实验的服务器可复用 runner
```

`runners/paper_result_closure.py` 保存论文结果闭合的正式命令计划、输入包预检和服务器可复用执行逻辑。Colab 入口只应通过 `paper_workflow/colab_utils/paper_result_closure.py` 添加进度显示和 Notebook runtime 报告。

`runners/external_baseline_method_faithful.py` 保存四个主表外部 baseline 的 method-faithful 适配调度、共同攻击簇输出、governed observation 汇总和结果打包逻辑。Colab 入口通过 `paper_workflow/colab_utils/external_baseline_method_faithful.py` 兼容旧导入路径。

`runners/tree_ring_official_reference.py` 保存 Tree-Ring 官方参考复现、legacy 环境准备、模型仓库布局修补、governed import 记录和结果打包逻辑。Colab 入口通过 `paper_workflow/colab_utils/tree_ring_official_reference.py` 兼容旧导入路径。

`runners/gaussian_shading_official_reference.py` 保存 Gaussian Shading 官方参考复现、legacy 环境准备、严格官方依赖优先策略、governed import 记录和结果打包逻辑。Colab 入口通过 `paper_workflow/colab_utils/gaussian_shading_official_reference.py` 兼容旧导入路径。

`runners/shallow_diffuse_official_reference.py` 保存 Shallow Diffuse 官方参考复现、legacy 环境准备、源码运行边界修补、governed import 记录和结果打包逻辑。Colab 入口通过 `paper_workflow/colab_utils/shallow_diffuse_official_reference.py` 兼容旧导入路径。

`runners/t2smark_full_main_reproduction.py` 保存 T2SMark 官方 SD3.5 路径复现、prompt split 导出、固定 FPR 候选记录生成、governed import 校验和结果打包逻辑。Colab 入口通过 `paper_workflow/colab_utils/t2smark_full_main_reproduction.py` 兼容旧导入路径。
