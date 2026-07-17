# SD3.5 Medium 外部 baseline 正式协议

## 共同主表边界

主表固定使用 `stabilityai/stable-diffusion-3.5-medium`、当前论文运行层级的完整 Prompt split、20步生成、`guidance_scale=4.5`、仅图像检测、三组 calibration 负观测共同约束的独立阈值和7项核心攻击 exact set。跨方法正式比较只使用核心攻击 fixed-FPR 下的 TPR、FPR、置信区间和 source-to-attacked 图像质量。10项补充攻击只由完整主方法在冻结决策器后可选执行描述性评测，不要求四个主表 baseline 覆盖。检测器内部的分数稳定性只作为诊断指标, 不进入跨方法优越性结论。

Tree-Ring、Gaussian Shading 和 Shallow Diffuse 每次只运行一个方法。runner 将 execution、adapter 图像与 manifest 放入 `run_records/<baseline_id>/`, 将跨结果包交换面写入 `split_observations/`。transfer manifest 绑定 observation、command result、Prompt 计划、adapter manifest、execution manifest、阈值摘要、生成预算、检测预算和代码版本。三个方法缺一、文件重复、方法身份不一致、计数不一致或摘要不一致时, 正式导入立即失败。

## Tree-Ring

Tree-Ring 在 Prompt 循环外由固定 `watermark_seed` 构造一次全局 Fourier ring key 与 mask, 整次运行的所有 Prompt 和恢复会话均复用该载体。每个 Prompt 的 clean / watermarked 图像共享基础 latent, watermarked 路线只在初始 latent 傅里叶域中心环形区域写入全局 key。检测器只读取图像、方法密钥和公开模型配置, 通过 VAE 编码与 SD3 scheduler 的流匹配反向 Euler 积分恢复初始噪声, 再计算环形区域密钥距离。目标共同协议使用该 baseline 自身连续分数，并由 `clean_negative_registered`、`attacked_negative_registered` 和 `watermarked_wrong_key` 三组 calibration 负观测共同约束后独立冻结阈值；其中 attacked negatives 只覆盖核心登记攻击。

## Gaussian Shading

Gaussian Shading 使用官方 `tensor.repeat` 调度把 watermark bit 扩展到16通道 latent, 再以256-bit key 与96-bit nonce 执行 ChaCha20 加密, 加密 bit 控制逐坐标正负截断 Gaussian 条件。每个 Prompt 只采样一次 clean Gaussian latent, watermarked latent 精确复用其逐坐标绝对幅值并仅替换符号, 因而 clean / watermarked 质量测量是同一基础 latent 的严格配对。检测器对输入图像执行 VAE 编码和流匹配反演, 恢复 noise sign 后使用同一 ChaCha20 key / nonce 解密并执行 block voting, 输出连续 bit-accuracy 分数。原始 key、nonce、watermark 和 message 不进入完成记录, 单元只绑定 seed 与不可逆摘要。目标共同协议使用该 baseline 自身连续分数，并由 `clean_negative_registered`、`attacked_negative_registered` 和 `watermarked_wrong_key` 三组 calibration 负观测共同约束后独立冻结阈值；其中 attacked negatives 只覆盖核心登记攻击。

## Shallow Diffuse

Shallow Diffuse 在 Prompt 循环外由固定 `watermark_seed` 构造一次全局 mask 与 watermark patch, 所有 Prompt 和恢复会话复用同一载体。clean 与 watermarked 路线从同一基础 latent 出发, 先以配置的 guidance 去噪到 `edit_timestep`, 再在 watermarked 分支按 mask 写入 patch。分支后的 clean 与 watermarked 路线都固定使用 `guidance_scale=1.0` 完成剩余去噪, 最终 latent 只用 watermarked 分支替换指定水印通道, 其余通道严格保留 clean 分支。检测器只接收图像, 经 VAE 编码后使用空 Prompt 和 `guidance_scale=1.0` 反演到同一 `edit_timestep`, 在该停止位置计算 masked patch 距离, 不反演到初始 noise。目标共同协议使用该 baseline 自身连续分数，并由 `clean_negative_registered`、`attacked_negative_registered` 和 `watermarked_wrong_key` 三组 calibration 负观测共同约束后独立冻结阈值；其中 attacked negatives 只覆盖核心登记攻击。

## T2SMark

T2SMark 使用独立的 `t2smark_formal_reproduction` 权威路径。该路径按 `external_baseline/source_registry.json` 的固定 commit 克隆源码, 应用 `adapter/formal_protocol_git_diff.txt` 中摘要固定的协议改动。每个 Prompt 具有独立的 `seed + prompt_index` 随机流; clean 图像精确重放编码器为 watermarked latent 实际采样的水印前基础 Gaussian latent, 二者共享 Prompt、采样器和生成预算。检测器使用同一正式密钥分别计算仅图像连续分数。calibration 对 clean negative 执行7项核心登记攻击以物化 `attacked_negative_registered`，并与 `clean_negative_registered`、`watermarked_wrong_key` 共同冻结 T2SMark 自己的决策；test 再对核心登记正样本和负样本执行 exact-set 攻击评测。T2SMark 不进入前三个方法的批量命令计划，也不承担补充攻击评测。

## 原子科学恢复

三个 common-backbone adapter 为每个 Prompt 发布一个 `source_pair` 单元, 并为每个 test Prompt、核心攻击和阴阳角色发布一个攻击单元。T2SMark 为每个全局 `prompt_index` 发布一个单元。单元绑定方法配置、Prompt、随机性、代码锁、依赖锁、实际 CUDA 来源与事实图像 SHA-256, 使用仓库相对 POSIX 路径支持跨 workspace 搬迁。fixed-FPR 阈值只在全部 source 单元齐备后冻结, 所有 observation 与 manifest 均由完整核心单元集合确定性重建。

## 官方环境复现

Tree-Ring、Gaussian Shading 和 Shallow Diffuse 的官方环境 runner 以固定10-Prompt 批次发布原子科学记录, 每批绑定规范科学配置、来源 commit、依赖环境、实际命令、GPU、随机性与逐 Prompt 原始观测。完整覆盖后由原始观测重新计算官方指标、受治理 record 与 validation report, 用于补充方法忠实度审计。主表 common-backbone 结果与官方环境结果分别保存和报告, 不在同一统计行混合。

## 正式证据条件

外部 baseline 记录进入论文主表前必须同时满足:

1. 当前论文层级完整 Prompt 的真实 GPU 执行已完成。
2. 三组 calibration 负观测按共同预算算法重算的阈值与全部 observation 携带的冻结阈值完全一致。
3. 每条 `detection_decision` 与 `score >= threshold` 一致。
4. test clean negative、test positive 和7项核心统一攻击的 positive/negative 图像对完整。
5. observation 实际数量与 command result、execution manifest 和 adapter manifest 一致。
6. transfer manifest 的路径、SHA-256、方法身份与代码版本可核验。
7. formal import validator、共同模板覆盖和论文证据审计全部通过。
8. 全部原子单元、派生 observation、受治理 record、validation report 与归档白名单均可从事实记录重新验证, 且不存在额外文件、目录或符号链接。
