# SD3.5 Medium 外部 baseline 正式协议

## 共同主表边界

主表固定使用 `stabilityai/stable-diffusion-3.5-medium`、当前论文运行层级的完整 Prompt split、20步生成、`guidance_scale=4.5`、仅图像检测、calibration clean negative 冻结阈值和统一17类攻击矩阵。跨方法正式比较只使用 fixed-FPR 下的 TPR、FPR、置信区间和 source-to-attacked 图像质量。检测器内部的分数稳定性只作为诊断指标, 不进入跨方法优越性结论。

Tree-Ring、Gaussian Shading 和 Shallow Diffuse 每次只运行一个方法。runner 将 execution、adapter 图像与 manifest 放入 `run_records/<baseline_id>/`, 将跨结果包交换面写入 `split_observations/`。transfer manifest 绑定 observation、command result、Prompt 计划、adapter manifest、execution manifest、阈值摘要、生成预算、检测预算和代码版本。三个方法缺一、文件重复、方法身份不一致、计数不一致或摘要不一致时, 正式导入立即失败。

## Tree-Ring

Tree-Ring 在 SD3.5 的16通道初始 latent 傅里叶域中心环形区域写入密钥模板。检测器只读取图像、方法密钥和公开模型配置, 通过 VAE 编码与 SD3 scheduler 的流匹配反向 Euler 积分恢复初始噪声, 再计算环形区域密钥距离。阈值仅由 calibration clean negative 分数冻结。

## Gaussian Shading

Gaussian Shading 使用二值 message 控制16通道 latent 的正负截断 Gaussian 采样, 并通过可配置的通道与空间重复因子构造 message 映射。检测器对输入图像执行 VAE 编码和流匹配反演, 恢复 noise sign 后进行 key 解码与 block voting, 输出连续 bit-accuracy 分数。阈值仅由 calibration clean negative 分数冻结。

## Shallow Diffuse

Shallow Diffuse 在 SD3.5 denoising 的固定浅层位置通过 `callback_on_step_end` 将局部 watermark patch 写入受控 mask。callback 必须在指定步实际执行, 否则运行立即终止。检测器对输入图像执行 VAE 编码和流匹配反演, 再计算 masked patch 距离。阈值仅由 calibration clean negative 分数冻结。

## T2SMark

T2SMark 使用独立的 `t2smark_formal_reproduction` 权威路径。该路径按 `external_baseline/source_registry.json` 的固定 commit 克隆源码, 应用 `adapter/formal_protocol_git_diff.txt` 中摘要固定的协议改动, 生成同 Prompt clean/watermarked 图像对, 使用同一正式密钥分别计算仅图像连续分数, 并对 clean negative 与 watermarked positive 同时执行统一攻击矩阵。T2SMark 不进入前三个方法的批量命令计划。

## 官方环境复现

Tree-Ring、Gaussian Shading 和 Shallow Diffuse 的官方环境 runner 记录来源 commit、依赖环境、运行命令、日志与指标, 用于补充方法忠实度审计。主表 common-backbone 结果与官方环境结果分别保存和报告, 不在同一统计行混合。

## 正式证据条件

外部 baseline 记录进入论文主表前必须同时满足:

1. 当前论文层级完整 Prompt 的真实 GPU 执行已完成。
2. calibration clean negative 重算阈值与全部 observation 携带的冻结阈值完全一致。
3. 每条 `detection_decision` 与 `score >= threshold` 一致。
4. test clean negative、test positive 和全部统一攻击的 positive/negative 图像对完整。
5. observation 实际数量与 command result、execution manifest 和 adapter manifest 一致。
6. transfer manifest 的路径、SHA-256、方法身份与代码版本可核验。
7. formal import validator、共同模板覆盖和论文证据审计全部通过。
