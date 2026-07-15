# 单模型分支风险参数敏感性协议

## 1. 证据问题

手工分支风险函数的数值参数来自预注册方法配置, 但固定配置本身不能证明结论不依赖狭窄取值。正式补证采用单模型内部、一次只改变一个参数的敏感性实验。该实验只回答当前登记 SD3.5 模型内部的参数稳定性, 不声明跨模型泛化。

## 2. 固定身份

18项设置共享以下身份:

- 精确模型与 revision;
- 完整 prompt split 与 prompt digest;
- 当前 `randomization_repeat_id` 对应的生成 seed 和水印密钥;
- 既有正式攻击矩阵;
- 图像检测机制与 fixed-FPR 规则;
- 其余全部方法参数。

每个设置必须重新执行生成、嵌入、攻击和检测, 并只用自己的 calibration clean negatives 冻结阈值。禁止复用 `formal_reference` 的阈值或图像结果。

## 3. 一次一参数设置

参考设置为 `formal_reference`。以下数值参数对三个分支中的同名字段同步变化:

- 五个风险权重分别乘以0.5和1.5;
- `eligibility_threshold` 分别取0.45和0.65;
- `budget_floor` 分别取0.025和0.10;
- `budget_ceiling` 取0.75, 正式参考值1.0已经覆盖上边界;
- `budget_gain` 分别取0.35和1.05。

总计18项设置。`texture_preference` 是三个分支的离散机制角色, 不作为数值参数敏感性处理; 改变它属于机制定义变化, 应进入独立消融而不是混入当前分析。

## 4. 统计规则

单重复输出逐 prompt records、检测原子、逐设置冻结协议、点估计和有界 Hoeffding 区间。跨重复聚合不得读取单重复派生 CSV, 必须从精确9重复聚合包内重新核验原始 records、检测原子、冻结协议和 manifest。

正式跨重复区间以9个注册 seed-key repeat 的设置均值作为统计单位。每个非参考设置同时报告绝对指标及相对 `formal_reference` 的同 repeat 配对差值。输出至少包括 clean FPR、wrong-key FPR、clean TPR、attacked TPR、attacked FPR 和配对 SSIM。

## 5. 主张边界

`parameter_sensitivity_aggregate_ready=true` 只支持“登记主模型内部、上述18项范围内的参数敏感性已经完整测量”。该字段不支持跨模型稳健性、全局最优参数、任意参数范围稳定性或通用感知质量结论。最终论文结果闭合必须把该证据作为必要测量组件, 但不得把它误作主方法优越性的额外投票项。
