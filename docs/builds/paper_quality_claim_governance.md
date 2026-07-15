# 论文质量结论治理

## 一、结论对象

质量保持不再由一个 `formal_fid_kid_ready` 布尔值代替, 而是拆分为三个独立主张：

1. `paired_perceptual_quality_noninferiority`：同一 Prompt、同一随机化身份下 clean 与 watermarked 图像的配对感知质量非劣效；
2. `semantic_alignment_noninferiority`：watermarked 图像相对生成语义条件的对齐非劣效；
3. `distributional_preservation_noninferiority`：watermarked 图像相对 clean generated images 的分布位移不超过冻结边界。

三个主张都必须输出 `supported`、`measured_not_supported` 或 `evidence_incomplete`。完成某个测量不等价于支持主张, 缺少原子记录或统计协议也不等价于测得不支持。

## 二、估计对象与统计单位

Prompt 是主要总体采样单位。每个 Prompt 下的9个 seed-key repeat 是嵌套随机化观测, 不按630个独立样本解释 `probe_paper`, 也不把9个 repeat 当作唯一的 `n=9`。

分布保持主分析先在每个 Prompt 内用9个 clean Inception 特征和9个 watermarked Inception 特征计算三阶多项式核的完整无偏 KID U-statistic, 得到一个 `prompt_conditional_kid`。随后只对 Prompt 级 KID 值执行冻结的 PCG64 bootstrap, 形成均值和95%区间。该估计对象是“对 Prompt 总体取期望的条件分布位移”, 其中 repeat 只刻画同一 Prompt 下的随机化变化。

联合630/6300/63000行特征重建的 FID/KID 三行继续保留, 用于描述 clean 与 watermarked 图像总体分布位移和复验既有数学算子；这些行不再充当独立样本推断, 也不直接投票支持质量主张。

## 三、解释边界

当前两侧图像均由同一生成模型产生, 一侧是 clean generated images, 另一侧是 watermarked generated images。因此 FID/KID 只能解释为分布保持, 不能解释为相对真实图像参考分布的生成质量。

若后续登记共同真实参考分布, 必须分别计算：

```text
delta_fid = fid(watermarked, reference) - fid(clean, reference)
delta_kid = kid(watermarked, reference) - kid(clean, reference)
```

在共同参考产物实际存在前, 不得物化或推断上述差值。

## 四、冻结非劣效边界

`configs/paper_quality_claim_protocol.json` 在新的正式实验前冻结以下边界：

- 配对 SSIM 的 Prompt 聚类区间下界不得低于0.99, 对应相对恒等配对的最大0.01损失；该0.01与现有机制必要性分析的配对 SSIM 非劣效口径一致；
- CLIP cosine 的 Prompt 聚类区间下界不得低于0.995, 与核心方法已冻结的最终图像语义保持下界一致；
- Prompt 条件 KID 均值的区间上界不得超过0.001, 使用未缩放的正式 KID 数值；FID 只作描述性披露。

这些值属于预登记判据, 不是从 `probe_paper` 结果选择的通过阈值。本次重构不运行正式结果, 因而没有以当前实验值反向调整边界。

## 五、逐攻击边界

每个正式攻击都必须具有独立质量决策, 且单项攻击必须同时覆盖配对感知、语义对齐和分布保持三类 Prompt 级推断；跨攻击结论只能由完整逐攻击集合派生。当前随机化质量包只包含 clean-to-watermarked 特征, 尚未绑定逐攻击感知与语义质量原子；因此每项攻击和跨攻击质量结论均为 `evidence_incomplete`。这会使总体 `quality_preservation` 保持 `evidence_incomplete`, 即使 clean 条件下的分布保持子主张已经测得支持或不支持。

该处理是 fail-closed 的证据边界, 不是代理实现。后续只有真实逐攻击图像质量观测、Prompt 级统计和共同来源摘要进入同一聚合包后, 才能改变这些三态决策。
