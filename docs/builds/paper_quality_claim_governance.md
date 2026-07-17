# 论文质量结论治理

## 文档职责

本文档是质量主张估计对象、统计单位、区间方向和非劣效判据的唯一规范。真实四图配对、逐攻击图像、独立视觉内容特征和 Inception 特征如何生产，由 `../paper_quality_evidence_governance.md` 约束；最终 `quality_preservation` 如何进入论文主张集合，由 `paper_claim_decision_governance.md` 约束。三份文档不得分别维护不同指标、阈值或文件名。

三档使用相同质量 estimand 和判据，但分别形成自身决策。`pilot_paper` 的质量结论属于主投稿证据，`full_paper` 的质量结论属于可选扩展；full 缺失不阻断完整 pilot 质量主张。相同图像的受治理特征缓存可以减少重复计算，但不得跨 profile 复制区间、三态决策或 `quality_preservation` 结论。正式质量合取只覆盖7项 `core_claim_required` 攻击；10项 `supplementary_descriptive` 攻击若运行则复用同一 estimand 单独披露，不进入质量主张投票。

## 一、结论对象

质量保持不再由一个 `formal_fid_kid_ready` 布尔值代替，而是拆分为三个独立质量子主张，也就是唯一正式 claim `quality_preservation` 的三个决策分量：

1. `paired_perceptual_quality_noninferiority`：同一 Prompt、同一随机化身份下 clean 与 watermarked 图像的配对感知质量非劣效；
2. `independent_visual_content_preservation_noninferiority`：watermarked 图像相对同 Prompt、同随机化身份的配对 clean 图像，其独立视觉内容表示保持非劣效；DINOv2 不编码 Prompt 文本，该子主张不得解释为图文或 Prompt 语义对齐；
3. `distributional_preservation_noninferiority`：watermarked 图像相对 clean generated images 的分布位移不超过冻结边界。

三个质量子主张都必须输出 `supported`、`measured_not_supported` 或 `evidence_incomplete`，并共同派生唯一正式 claim `quality_preservation`，不得扩张为额外正式 claim。完成某个测量不等价于支持子主张，缺少原子记录或统计协议也不等价于测得不支持。

## 二、估计对象与统计单位

Prompt 是主要总体采样单位。每个 Prompt 下的5个 seed-key repeat 是嵌套随机化观测, 不按350个独立样本解释 `probe_paper`, 也不把5个 repeat 当作唯一的 `n=5`。

分布保持主分析先在每个 Prompt 内用5个 clean Inception 特征和5个 watermarked Inception 特征计算三阶多项式核的完整无偏 KID U-statistic, 得到一个 `prompt_conditional_kid`。随后只对 Prompt 级 KID 值执行冻结的 PCG64 bootstrap, 形成均值和95%区间。该估计对象是“对 Prompt 总体取期望的条件分布位移”, 其中 repeat 只刻画同一 Prompt 下的随机化变化。

联合350/3500/35000行特征重建的 FID/KID 三行继续保留, 用于描述 clean 与 watermarked 图像总体分布位移和复验既有数学算子；这些行不再充当独立样本推断, 也不直接投票支持质量主张。

## 三、解释边界

本协议两侧图像均由同一生成模型产生, 一侧是 clean generated images, 另一侧是 watermarked generated images。因此 FID/KID 只能解释为分布保持, 不能解释为相对真实图像参考分布的生成质量。

若后续登记共同真实参考分布, 必须分别计算：

```text
delta_fid = fid(watermarked, reference) - fid(clean, reference)
delta_kid = kid(watermarked, reference) - kid(clean, reference)
```

在共同参考产物实际存在前, 不得物化或推断上述差值。

## 四、冻结非劣效边界

`configs/paper_quality_claim_protocol.json` 在新的正式实验前冻结以下边界：

- 配对 SSIM 的 Prompt 聚类区间下界不得低于0.99, 对应相对恒等配对的最大0.01损失；该0.01必须与机制必要性分析登记的配对 SSIM 非劣效口径一致；
- 独立视觉内容评估器 cosine 的 Prompt 聚类区间下界不得低于0.995；该评估器不参与 SLM-WM 优化约束且与机制 CLIP 属于不同模型族。同源 CLIP cosine `>=0.995` 只作为核心方法内部机制一致性门禁，两者数值边界相同不表示证据来源相同；
- Prompt 条件 KID 均值的区间上界不得超过0.001, 使用未缩放的正式 KID 数值；FID 只作描述性披露。

这些值属于预登记判据，不得从 `probe_paper` 或任何后续结果反向选择、放宽或调整。

## 五、逐攻击边界

每个核心证据攻击都必须具有独立质量决策，且单项攻击必须同时覆盖配对感知、独立视觉内容保持和分布保持三类 Prompt 级推断；核心跨攻击结论只能由精确7项核心逐攻击集合派生。任一核心攻击的感知、独立视觉内容或分布质量原子缺失时，对应攻击和核心跨攻击质量结论必须为 `evidence_incomplete`，总体 `quality_preservation` 也必须保持 `evidence_incomplete`，即使 clean 条件下的分布保持子主张已经测得支持或不支持。

补充攻击不进入上述合取。已执行的补充攻击仍应使用相同三类 Prompt 级推断并输出逐攻击三态状态，但该状态固定为非主张描述角色；未执行的补充攻击记录 `not_run` 或等价受治理状态，不得伪装为 `supported`。补充攻击之间不得只汇总成功子集形成“跨补充攻击支持”结论。真实生产链的满足状态只由 `project_construction_state.md` 登记。

该处理是 fail-closed 的证据边界, 不是代理实现。后续只有真实逐攻击图像质量观测、Prompt 级统计和共同来源摘要进入同一聚合包后, 才能改变这些三态决策。

## 六、实施入口与验收

质量链修改必须按以下顺序进行：

1. 在 `configs/paper_quality_claim_protocol.json` 冻结 estimand、统计方向、区间和非劣效边界。
2. 由 `experiments/` 从真实持久化图像生产配对 SSIM、独立视觉内容特征和 Inception 特征，不接受机制内部 CLIP 诊断值替代独立视觉内容质量。
3. 由 `paper_experiments/analysis/randomization_dataset_quality.py` 按 Prompt 和5重复结构重建 clean 条件、7项核心逐攻击统计以及实际存在的补充描述性统计。
4. 由 `paper_experiments/analysis/paper_quality_decisions.py::build_quality_preservation_decisions` 只从 clean 与7项核心攻击形成三个质量子主张、核心逐攻击决策和核心跨攻击决策；补充状态由独立非主张消费者保存。
5. 由 `scripts/paper_result_closure.py` 消费完整质量决策，禁止跳过原始记录直接写入最终状态。

正向集成测试必须从真实 writer 格式的图像质量原子开始，经过单 repeat 包、5-repeat 聚合和 Prompt 聚类区间，最终得到三态质量决策。缺少记录应得到 `evidence_incomplete`；完整但未达到边界应得到 `measured_not_supported`；只有完整且达到边界才能得到 `supported`。
