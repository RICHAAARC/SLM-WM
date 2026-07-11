# paper_experiments

`paper_experiments/` 是完整论文实验层。该目录保存外部 baseline 适配、官方参考复现编排、受治理结果导入、共同协议对比和论文证据闭合代码。

该目录可以依赖 `main/` 与 `experiments/`, 但不得依赖 `paper_workflow/`。Notebook 入口只能调用这里的正式实现, 不能把正式论文结果逻辑写在 cell 中。

## 子目录职责

```text
baselines/   外部 baseline 适配、官方参考复现和受治理导入协议
runners/     完整论文实验的服务器可复用 runner
analysis/    论文表图实际数据验证、证据审计、闭合门禁与投稿就绪分析
```

## runner 职责

- `runners/paper_result_closure.py`: 对精确输入锁执行 current-run 清理、显式物化、run-scoped 证据 DAG、语义闭合门禁和最终打包, 并返回本次唯一归档路径。
- `runners/closure_package_selection.py`: 按10类包内身份、当前论文运行层级、fixed-FPR、baseline、ready 状态和 `outputs/` 白名单选择精确输入, 并写出 run-scoped 输入锁与独立 manifest。
- 主方法正式输入包来自 `image_only_dataset_runtime`、`runtime_rerun_ablation` 和 `dataset_level_quality`; attention capture、latent injection 或 aligned rescoring 诊断包不能替代这些输入。
- `runners/external_baseline_method_faithful.py`: 单个 Tree-Ring、Gaussian Shading 或 Shallow Diffuse 的 SD3.5 common-backbone 适配调度、fixed-FPR 审计、transfer manifest 和独占结果打包逻辑。
- `runners/t2smark_source_runtime.py`: 按来源登记 commit 准备 T2SMark 源码并应用摘要固定的正式协议 Git diff。
- `runners/t2smark_formal_reproduction.py`: T2SMark 官方 SD3.5 路径复现、prompt split 导出、固定 FPR 候选记录生成、governed import 校验和结果打包逻辑。
- `runners/tree_ring_official_reference.py`: Tree-Ring 官方参考复现、官方依赖环境准备、模型仓库布局核验、governed import 记录和结果打包逻辑。
- `runners/gaussian_shading_official_reference.py`: Gaussian Shading 官方参考复现、官方依赖环境准备、严格官方依赖优先策略、governed import 记录和结果打包逻辑。
- `runners/shallow_diffuse_official_reference.py`: Shallow Diffuse 官方参考复现、官方依赖环境准备、源码运行边界核验、governed import 记录和结果打包逻辑。
- `analysis/paired_superiority.py`: 对 SLM-WM 与4个主表 baseline 的完整 test Prompt x attack 观测执行精确配对, 以 Prompt 为聚类单位计算总体优势统计。
- `analysis/paper_artifact_data_validation.py`: 实际读取11类正式表图源文件, 以仅图像盲检原始 JSONL 和冻结协议独立重建分数分布、ROC 与 DET, 再精确核验表格列序、行序、单元格、曲线端点和跨摘要 readiness 一致性。

## official-reference 忠实度证据

- `baselines/official_reference_fidelity_evidence.py`: 只读取当前论文运行下已物化的 Tree-Ring、Gaussian Shading 和 Shallow Diffuse 三个 official-reference family, 独立复算 package input 的精确文件集合与逐文件摘要, 并核验运行决策、governed import、validation、归档治理语义和共同 clean `code_version`。
- 输出记录固定为精确三个方法身份, 只支持补充方法忠实度证据。该证据明确设置 `main_table_eligible=false` 和 `supports_main_table_superiority_claim=false`, 不替代 common-backbone 主表比较, 也不扩展到其他 supplemental 方法。
- 18步 CPU 结果闭合 DAG 在精确10包物化后、攻击矩阵与主表比较构建前执行该审计。语义门禁核验 `official_reference_exact_set_ready`、`common_code_version_ready`、`official_reference_fidelity_ready_count` 和 `official_reference_fidelity_evidence_ready`。

六个 common-backbone / official-reference family 与 T2SMark 均把正式运行写入 `outputs/<artifact>/<paper_run_name>/...`。归档名称包含方法身份和唯一时间后缀, 包内 summary、运行 manifest、package input manifest 与 archive manifest 必须绑定同一 `paper_run_name`、`target_fpr`、`baseline_id` 和 `code_version`。归档白名单只允许当前 family 的 `outputs/` 成员, 不把 runner 源码、README、source registry 或 Notebook runtime 报告复制进正式输入包。

## 主表总体优势统计

- 主表比较只使用 SLM-WM 与 Tree-Ring、Gaussian Shading、Shallow Diffuse、T2SMark 四个正式 baseline。每个 baseline 必须与 SLM-WM 在完全相同的 test Prompt 和攻击条件上形成一一配对, 重复、缺失或额外配对键都会阻断统计。
- 单个 Prompt 的效应值是该 Prompt 跨完整攻击集合的平均二元检测差值。正式统计使用100000次 Prompt-clustered bootstrap 的95% percentile CI, 对取值位于 [-1,1] 的 Prompt 聚类差执行单侧 bounded Hoeffding 均值检验, 再对4个主表 claim p 值执行 Holm 校正。整数动态规划 exact sign-flip 只披露 sharp-null 诊断, 不参与均值优势 claim 门禁。
- 配对 outcome 逐条绑定规范 test Prompt 身份、正式攻击 ID、资源档位、攻击配置摘要、两方法冻结阈值摘要和原始 observation 文件字节摘要。结果闭合门禁会读取五方法原始 observation 独立重建全部 outcome、样本计数、TPR、clean/attacked FPR、质量均值、分数保持诊断和 Hoeffding 区间, 再精确核验正式 result records、来源文件 SHA-256、模板覆盖、schema validation 与 manifest 配置摘要。
- 单个 baseline 只有在总体平均差值大于0、bootstrap CI 下界大于0且 Holm 校正后 `p < 0.05` 时才满足 `paired_superiority_ready`。4个 baseline 必须全部满足该条件, `overall_paired_superiority_ready` 才能支持主表总体 superiority claim。
- 上述统计结论严格限定于当前受治理 Prompt benchmark。将结论外推到未采样自然 Prompt 总体还需要 Prompt 聚类独立或可交换假设及相应数据来源证明, 当前门禁不据此声明无条件总体优势。
- 逐攻击比较表负责完整披露每个攻击条件的效应与保守 CI, 不要求每个攻击均显著胜出。`universal_per_attack_superiority_claim_ready` 只控制“所有攻击均显著胜出”的更强主张, 不替代 Prompt 聚类配对的总体优势门禁。
- 三个 official-reference family 只承担补充方法忠实度证据, 明确设置 `main_table_eligible=false` 与 `supports_main_table_superiority_claim=false`, 不进入上述主表配对统计。

## CPU 结果闭合 DAG

CPU 闭合固定执行18个 run-scoped 步骤: 精确10包物化、official-reference 忠实度审计、攻击矩阵、统一 fixed-FPR 阈值审计、Prompt 聚类配对优势、主表 adapter 协议、主表候选、formal import、主表证据、baseline 比较、正式结果 records、共同协议、结果分析、论文表图实际数据审计、投稿就绪审计、证据闭合入口审计、结果闭合语义门禁和完整结果包打包。每一步只消费本次输入锁与前序受治理产物, 任一步失败都会阻断后续闭合。

## 结果口径

- SLM-WM 主方法表使用“空间低通 LF”“高斯幅值尾部截断”和“Q/K attention geometry”三个机制名称。正式论文消融使用 Tail-only、No-Tail 和 No-Tail-Truncation。
- method-faithful 结果用于主表 common-backbone 对比: 同一 prompt split、同一 SD3.5 主线、同一攻击簇、同一 fixed-FPR 协议。
- Tree-Ring、Gaussian Shading 和 Shallow Diffuse 的 official reference 结果用于补充表和方法忠实度审计。T2SMark 原生 SD3.5 formal 路径直接生成主表候选, 但同样必须通过完整攻击模板、fixed-FPR 与 governed import 门禁。
- `probe_paper`、`pilot_paper` 和 `full_paper` 都是正式结果包, 分别支持 `probe_claim`、`pilot_claim` 和 `full_claim`。三者只允许样本数量和 fixed-FPR 标准不同。
- proxy、placeholder、fallback、synthetic 和 formal-null 证据不得进入共同协议 claim-ready 统计。
- 论文表图审计实际读取仅图像盲检原始记录、冻结协议、test 指标、记录级连续分数分布、完整 ROC / DET sweep、攻击指标、主表 baseline、正式消融和 FID / KID 表。分数分布与曲线必须由原始记录精确重建并逐单元格一致, readiness flag、路径字符串或仅满足单调性的自造曲线不能替代该核验。
- `analysis/evidence_closure_entry_review.py`: 联合投稿就绪、待补证据、claim 阻断、主表 baseline 和数据集质量等受治理输入逐项执行确定性入口判定。只有全部检查项通过时才输出 `entry_review_decision=ready_for_evidence_closure` 与 `evidence_closure_allowed=true`; 任一缺口都会自动阻断, 不使用人工批准状态。

Colab 入口通过 `paper_workflow/colab_utils/` 增加进度显示、Drive 路径和 Notebook runtime 报告; 正式协议逻辑仍以本目录与核心方法复现层为准。
