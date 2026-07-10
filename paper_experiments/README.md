# paper_experiments

`paper_experiments/` 是完整论文实验层。该目录保存外部 baseline 适配、官方参考复现编排、受治理结果导入、共同协议对比和论文证据闭合代码。

该目录可以依赖 `main/` 与 `experiments/`, 但不得依赖 `paper_workflow/`。Notebook 入口只能调用这里的正式实现, 不能把正式论文结果逻辑写在 cell 中。

## 子目录职责

```text
baselines/   外部 baseline 适配、官方参考复现和受治理导入协议
runners/     完整论文实验的服务器可复用 runner
```

## runner 职责

- `runners/paper_result_closure.py`: 论文结果闭合的正式命令计划、输入包预检和服务器可复用执行逻辑。
- 主方法正式输入包来自 `image_only_dataset_runtime`、`runtime_rerun_ablation` 和 `dataset_level_quality`; attention capture、latent injection 或 aligned rescoring 诊断包不能替代这些输入。
- `runners/external_baseline_method_faithful.py`: 四个主表外部 baseline 的 SD3.5 method-faithful 适配调度、共同攻击簇输出、governed observation 汇总和结果打包逻辑。
- `runners/t2smark_full_main_reproduction.py`: T2SMark 官方 SD3.5 路径复现、prompt split 导出、固定 FPR 候选记录生成、governed import 校验和结果打包逻辑。
- `runners/tree_ring_official_reference.py`: Tree-Ring 官方参考复现、legacy 环境准备、模型仓库布局修补、governed import 记录和结果打包逻辑。
- `runners/gaussian_shading_official_reference.py`: Gaussian Shading 官方参考复现、legacy 环境准备、严格官方依赖优先策略、governed import 记录和结果打包逻辑。
- `runners/shallow_diffuse_official_reference.py`: Shallow Diffuse 官方参考复现、legacy 环境准备、源码运行边界修补、governed import 记录和结果打包逻辑。

## 结果口径

- SLM-WM 主方法表使用“空间低通 LF”“高斯幅值尾部截断”和“Q/K attention geometry”三个机制名称。正式论文消融使用 Tail-only、No-Tail 和 No-Tail-Truncation。
- method-faithful 结果用于主表 common-backbone 对比: 同一 prompt split、同一 SD3.5 主线、同一攻击簇、同一 fixed-FPR 协议。
- official reference 结果用于补充表和方法忠实度审计: 记录官方或 legacy 环境下的参考输出, 并通过 governed import 协议进入候选结果。
- `probe_paper`、`pilot_paper` 和 `full_paper` 都是正式结果包, 分别支持 `probe_claim`、`pilot_claim` 和 `full_claim`。三者只允许样本数量和 fixed-FPR 标准不同。
- proxy、placeholder、fallback、synthetic 和 formal-null 证据不得进入共同协议 claim-ready 统计。

Colab 入口通过 `paper_workflow/colab_utils/` 增加进度显示、Drive 路径和 Notebook runtime 报告; 正式协议逻辑仍以本目录与核心方法复现层为准。
