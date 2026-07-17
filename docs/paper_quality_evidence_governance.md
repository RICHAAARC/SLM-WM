# 论文质量证据与 GPU 资格化治理

## 1. 适用边界

本协议只治理实验数据如何形成论文结论，不重新定义 `main/` 核心算法。目标算法以 `builds/algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md` 为唯一权威来源。核心/补充攻击集合和 baseline 集合只能从各自冻结 registry 读取；本协议不得自行增删或改变证据职责。

`probe_paper`、`pilot_paper` 和 `full_paper` 必须使用同一结论集合、同一 estimand、同一指标语义、同一7项核心攻击与4 baseline 协议、同一 fixed-FPR 校准程序和同一闭合门禁。三者只允许改变登记的样本规模、目标 FPR `0.1/0.01/0.001` 和统计强度。`probe_paper` 负责流程与初步可行性，`pilot_paper` 是主投稿证据，`full_paper` 是可选扩展。`probe_paper` 不能使用减少核心结论项的快速替代协议；pilot 与 full 结论均只能由自身核心证据支持。full 未运行不阻断完整 pilot 的投稿作用域，但正文不得据此报告 full 规模或 FPR=0.001 结论。10项补充攻击不进入任一 profile 的核心闭合门禁，实际运行时仍使用同一 estimand 和冻结检测器。

## 2. 四图配对质量 estimand

核心或已运行补充攻击的规范质量观察单元都由同一 Prompt, 同一 `randomization_repeat_id`, 同一攻击配置摘要和同一 `attack_seed_random` 下的4张真实图像组成:

1. `clean`.
2. `watermarked`.
3. `attacked_clean = attack(clean)`.
4. `attacked_watermarked = attack(watermarked)`.

基础水印质量比较为 `clean` 对 `watermarked`. 逐攻击水印质量比较为 `attacked_clean` 对 `attacked_watermarked`. 后者估计相同攻击条件下水印带来的增量质量变化, 不能用 `watermarked` 对 `attacked_watermarked` 替代. `clean` 对 `attacked_clean` 和 `watermarked` 对 `attacked_watermarked` 只描述攻击强度, 不直接支持水印质量保持主张.

冻结配置位于 `configs/attack_conditioned_quality_estimand.json`. 正式生产链为:

```text
真实攻击图像
-> 四图身份记录
-> 攻击后 Inception 特征
-> base 与逐攻击同源 CLIP 诊断 embedding
-> base 与逐攻击独立 DINOv2 CLS embedding
-> 配对 SSIM、诊断 CLIP cosine 和独立视觉内容 cosine
-> 5-repeat 原始记录聚合
-> Prompt 聚类 bootstrap
-> 逐攻击质量决策
-> 跨攻击质量决策
```

只有精确7项核心攻击的逐攻击决策进入上述跨攻击质量决策和 `quality_preservation`。补充攻击只写入独立描述性质量报告；其特征、区间和失败状态不能与核心集合合并。

所有记录必须绑定 Prompt, repeat, 样本角色, 攻击身份, 攻击随机种子, 图像文件 SHA-256, 图像像素 SHA-256, 分辨率, 代码提交和科学依赖 profile. `openai/clip-vit-base-patch32` 与方法语义条件编码器同源, 因而其 cosine 只保留为 `mechanism_consistency_diagnostic`. 正式独立视觉内容表示保持主张使用不参与优化或检测的 `facebook/dinov2-base` 冻结 CLS 特征。`independent_semantic_cosine` 是现有兼容字段名，其估计对象严格是配对图像之间的视觉内容保持，不是 Prompt 或图文对齐。模型 ID、精确 revision、预处理、特征层、L2 归一化和完整依赖锁由 `configs/independent_semantic_quality_evaluator.json` 冻结. 聚合器从两套原始向量分别复算 cosine, 但只有独立视觉内容 cosine 进入质量决策; Prompt 条件 KID 仍从 Inception 特征复算. 调用方不得直接注入最终质量决策.

同一图像字节的 VAE latent、Inception、DINOv2 和诊断 CLIP 特征可以只计算一次，并在6个方法角色、registered/wrong-key 测量以及嵌套 profile 中复用。缓存键必须包含图像 SHA-256、模型 ID 与精确 revision、预处理、特征层、dtype、代码提交、科学依赖摘要和 evaluator schema；消费者必须复算身份摘要。图像对、质量 estimand、分母、Prompt 聚类、bootstrap 和逐 profile 决策仍分别构造，特征缓存不能直接携带 `quality_preservation` 结论。

## 3. baseline 两层字段

baseline 来源注册层与正式结果记录层承担不同职责, 不得合并字段:

- `external_baseline/source_registry.json` 中的 `result_status` 表示来源注册层的结果可用状态，`paper_claim_support` 表示该来源注册项是否可支持论文主张。
- 正式 baseline 结果或观测记录中的 `baseline_result_source` 表示具体结果证据来自哪个受治理文件或结果包。

交叉一致性是条件不变量：当来源注册层声明 `result_status=not_available` 时，`paper_claim_support` 必须为 false，正式记录若存在则 `baseline_result_source` 必须为 `not_available`，相应 reproduced/imported ready 字段必须为 false。来源可用时，来源注册状态和正式记录来源必须一起更新并通过各自 validator。不能删除 `result_status`，也不能用它替换 `baseline_result_source`。实际 baseline 可用状态只从机器 registry 和项目构建状态规范读取，本协议不复制实时值。

## 4. GPU 方法与资源双门禁

单 Prompt 资格化必须分别写出：

- `gpu_operator_preflight_ready`：表示两份无状态核心规范登记的方法算子、身份、数值、最终图像检测与救回事实是否在真实 CUDA 模型图上完整通过。
- `gpu_resource_budget_ready`：表示显存、单 Prompt 耗时和估算总时长是否满足显式登记的资源预算。

两类结论相互独立。资源超限只能阻断当前设备或调度方案，不能篡改方法真实性；方法真实性失败必须逐层返回非零状态，不能由资源充足掩盖。具体方法事实、冻结参数和失败条件只引用算法原语与方法机制设计文档，本文件不复制第二套资格化清单。

资格化报告必须绑定精确提交、依赖 profile、模型 revision、输入摘要和报告内容摘要，且固定 `supports_paper_claim=false`。Notebook 只能调用服务器入口；服务器入口、报告位置和实际迁移状态只由执行层文档与项目构建状态登记。
## 5. 论文结论边界

单 Prompt GPU 资格化, 单攻击闭环和单 repeat 闭环均不得支持论文主张. 只有精确5-repeat 聚合同时完成 fixed-FPR, 质量保持, baseline 对比, 机制必要性和结果包闭合后, 才能评估对应 profile 的完整注册结论集合。主投稿证据以 `pilot_paper` 为准；`full_paper` 只在实际完成独立闭合时增加扩展结论。

代码路径通过测试只证明生产者和消费者契约可闭合，不证明 SLM-WM 的实证主张已经成立。真实 GPU 证据和三档闭合状态只能从 `builds/project_construction_state.md` 及对应受治理结果包判断。
