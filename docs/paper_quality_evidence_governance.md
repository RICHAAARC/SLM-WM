# 论文质量证据与 GPU 资格化治理

## 1. 适用边界

本协议只治理实验数据如何形成论文结论. 它不修改 `main/` 核心算法, Null Space 数学机制, Q/K 方法目标, 风险场公式, 检测器核心规则或已冻结阈值. 当前也不扩充攻击集合和 baseline 集合.

`probe_paper`, `pilot_paper` 和 `full_paper` 必须使用同一结论集合, 同一 estimand, 同一指标语义, 同一攻击与 baseline 协议, 同一 fixed-FPR 规则和同一闭合门禁. 三者只允许改变样本数量与统计强度. `probe_paper` 只有在完整论文结论集合闭合后才可通过, 不能使用减少结论项的快速替代协议.

## 2. 四图配对质量 estimand

逐攻击质量的规范观察单元由同一 Prompt, 同一 `randomization_repeat_id`, 同一攻击配置摘要和同一 `attack_seed_random` 下的4张真实图像组成:

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
-> base 与逐攻击 CLIP 图像 embedding
-> 配对 SSIM 和 CLIP cosine
-> 9-repeat 原始记录聚合
-> Prompt 聚类 bootstrap
-> 逐攻击质量决策
-> 跨攻击质量决策
```

所有记录必须绑定 Prompt, repeat, 样本角色, 攻击身份, 攻击随机种子, 图像文件 SHA-256, 图像像素 SHA-256, 分辨率, 代码提交和科学依赖 profile. 聚合器从原始向量复算 CLIP cosine, 从 Inception 特征复算 Prompt 条件 KID, 不接受调用方直接注入最终质量决策.

## 3. baseline 两层字段

baseline 来源注册层与正式结果记录层承担不同职责, 不得合并字段:

- `external_baseline/source_registry.json` 中的 `result_status` 表示该 baseline 来源注册项当前是否已有可用结果, `paper_claim_support` 表示该来源注册项当前是否支持论文主张.
- 正式 baseline 结果或观测记录中的 `baseline_result_source` 表示具体结果证据来自哪个受治理文件或结果包. 没有结果时使用 `not_available`.

当前注册状态下的交叉一致关系为:

```text
result_status=not_available
paper_claim_support=false
baseline_result_source=not_available
baseline_reproduced_result_ready=false
baseline_imported_result_ready=false
```

将来结果可用时, 来源注册状态和正式记录来源必须一起更新并通过各自 validator. 不能删除 `result_status`, 也不能用它替换 `baseline_result_source`.

## 4. GPU 方法与资源双门禁

单 Prompt GPU 资格化报告必须同时写出两个相互独立的结论:

### 4.1 `gpu_operator_preflight_ready`

该门禁验证方法机制和数值事实:

- 冻结 SD3.5 Q/K 层在真实 CUDA 执行中存在并产生真实 Tensor.
- 每个活动分支使用完整716维特征的精确 JVP/VJP.
- 无阻尼 PSD-CG 在登记的64次上限内收敛并通过逐列残差门禁.
- 三个注入时刻均产生非零实际 dtype latent 写回, 共同缩放不超过24次.
- clean, carrier-only 和 watermarked 最终成图通过完整特征保持门禁.
- 最终成图 Q/K 双归因增益和仅图像检测 Q/K 原子通过.
- 当前平台重建 `configs/keyed_prg_cross_platform_known_answer.json` 中的 uniform 和 Gaussian 固定向量, Tensor 内容摘要与 Windows 冻结值逐字节相同.

该报告固定 `supports_paper_claim=false`. 它只表示单 Prompt 方法算子资格化通过, 不是 `probe_paper` 结论.

### 4.2 `gpu_resource_budget_ready`

该门禁只比较以下观测与显式登记的资源上限:

- `peak_gpu_memory_bytes`.
- `single_prompt_wall_time_seconds`.
- `estimated_probe_total_gpu_hours`.

缺少观测或上限时状态为 `not_evaluated_missing_observation_or_registered_limit`. 超出预算时该门禁失败, 但不得改变 `gpu_operator_preflight_ready`. 资源失败只说明当前设备, 会话长度或调度方案不足, 不说明方法机制不成立.

报告入口为:

```text
python scripts/write_gpu_method_qualification_report.py \
  --runtime-result <单 Prompt runtime_result.json> \
  --update-records <更新记录 JSONL> \
  --detection-records <检测记录 JSONL> \
  --resource-observation <可选资源观测 JSON> \
  --registered-budget <可选登记预算 JSON>
```

输出固定写入 `outputs/gpu_method_qualification/<run_id>/`. 进程状态码只跟随方法算子门禁, 资源预算结论在报告中独立保存.

## 5. 论文结论边界

单 Prompt GPU 资格化, 单攻击闭环和单 repeat 闭环均不得支持论文主张. 只有精确9-repeat 聚合同时完成 fixed-FPR, 质量保持, baseline 对比, 机制必要性和结果包闭合后, 才能评估 `probe_paper` 的完整注册结论集合.

真实 GPU 结果尚未产生前, 代码路径通过测试只证明生产者和消费者契约可闭合, 不证明 SLM-WM 的实证主张已经成立.
