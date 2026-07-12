# 真实科学算子实现与论文证据闭合

## 一、文件级职责

| 机制 | 正式实现 | 作用 |
| --- | --- | --- |
| 分支风险场 | `main/methods/semantic/branch_risk.py` | 分别构造 LF、尾部截断和注意力几何风险与承载预算 |
| 真实 Jacobian Null Space | `main/methods/subspace/jacobian_nullspace.py` | 通过 autograd JVP、联合响应矩阵和 SVD 求解 latent 低响应基底 |
| LF 与尾部载体 | `main/methods/carrier/keyed_tensor.py` | 构造检测端可重建的固定模板, 并在嵌入端投影到安全子空间 |
| 真实注意力梯度 | `main/methods/geometry/differentiable_attention.py` | 从 Transformer `to_q`/`to_k` 得到真实 attention, 对 latent 求梯度 |
| 几何恢复 | `main/methods/geometry/attention_alignment.py` | 执行密钥关系匹配、三点 RANSAC 和仿射估计 |
| 仅图像检测 | `main/methods/detection/image_only.py` | 只接收图像、密钥和公开模型配置, 完成内容主判与同阈值救回 |
| 真实模型运行 | `experiments/runners/semantic_watermark_runtime.py` | 在 SD3.5 Medium 采样过程中执行全部真实嵌入算子 |
| 共同攻击算子 | `experiments/runtime/diffusion/regeneration_attacks.py` | 为主方法与全部 baseline 统一执行 SD3.5 img2img、flow-matching 反向 Euler 积分、inpainting 和检测器引导搜索 |
| 科学会话 | `experiments/runtime/semantic_watermark_scientific_session.py` | 在同一受验证主方法子解释器中调度主运行、质量评估、正式消融与绑定打包 |
| 主方法工作负载 | `experiments/runners/image_only_dataset_workload.py` | 构造当前论文规模的正式配置并执行数据集协议与质量评估 |
| 数据集协议 | `experiments/runners/image_only_dataset_runtime.py` | 运行 Prompt 数据集、冻结完整 evidence 协议并生成 test 记录 |
| 消融工作负载 | `experiments/ablations/mechanism_ablation_workload.py` | 构造完整 Prompt 消融配置并调用真实重运行协议 |
| 正式消融 | `experiments/ablations/runtime_rerun.py` | 对每个机制配置重新生成、重新攻击和重新检测 |
| 正式 FID/KID | `experiments/artifacts/dataset_level_quality_outputs.py` | 使用 torch-fidelity 0.4.0 的 TensorFlow 兼容 Inception v3 2048 维特征生成可审计质量记录 |
| Tree-Ring common-backbone | `external_baseline/primary/tree_ring/adapter/method_faithful_sd35.py` | 在 SD3.5 初始 latent 傅里叶域写入全局固定 ring key, 并通过图像反演计算 key 区域距离 |
| Gaussian Shading common-backbone | `external_baseline/primary/gaussian_shading/adapter/method_faithful_sd35.py` | 执行 ChaCha20 message、同幅值截断 Gaussian 条件采样、图像反演解密与 block voting |
| Shallow Diffuse common-backbone | `external_baseline/primary/shallow_diffuse/adapter/method_faithful_sd35.py` | 从同一基础 latent 去噪到固定 `edit_timestep` 后注入全局固定 patch, 以 `guidance_scale=1.0` 完成双分支去噪并仅融合指定水印通道, 检测时仅从图像反演到同一停止位置计算 masked distance |
| T2SMark 正式复现 | `paper_experiments/runners/t2smark_formal_reproduction.py` | 调用固定官方源码补丁, 执行逐 Prompt 严格同基础 latent 配对、仅图像检测与正式攻击 |
| 官方参考原子批次 | `paper_experiments/runners/official_reference_unit_runtime.py` | 以10-Prompt 批次运行登记官方算子, 保存逐 Prompt 观测并确定性重建官方指标 |
| Colab 续跑 | `paper_workflow/notebooks/semantic_watermark_image_only_run.ipynb` | 在 Drive 持久化工作区分批运行主方法、质量评估与正式消融 |

高斯幅值尾部截断分支的正式运行标识为 `tail_robust`。`build_tail_robust_template(...)` 对标准高斯模板按元素绝对幅值分位点截断, 不执行 FFT、DCT、带通滤波或空间频带 mask, 因而不具有空间频带定义。内容模板与安全投影实现位于 `main/methods/carrier/keyed_tensor.py`。

四个主表外部 baseline 保留各自关键科学算子, 共同协议只统一 backbone、Prompt、攻击、仅图像访问和 fixed-FPR 统计边界。Tree-Ring 与 Shallow Diffuse 分别按官方调度在 Prompt 循环外构造一次全局载体。Gaussian Shading 使用 ChaCha20 key / nonce 加密 message, 并以同一 clean Gaussian latent 的逐坐标幅值构造严格配对的符号条件样本。T2SMark clean 图像重放编码器实际使用的水印前基础 Gaussian latent。simple XOR、独立 clean / watermarked latent 采样或逐 Prompt 替换官方全局载体均不属于正式主表实现。

## 二、不可变模型输入

正式科学算子不把模型仓库名当作完整输入。`configs/model_source_registry.json` 同时固定仓库标识和40位不可变 revision。当前主方法与 common-backbone baseline 共用 `stabilityai/stable-diffusion-3.5-medium@b940f670f0eda2d07fbb75229e779da1ad11eb80`, 语义条件与成对质量评估使用 `openai/clip-vit-base-patch32@3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`。

Tree-Ring、Gaussian Shading 和 Shallow Diffuse 的 official-reference 模型使用 `Manojb/stable-diffusion-2-1-base@0094d483a120f3f33dafbd187ea4aa60d10de75c` 公开镜像。运行报告明确区分该有效镜像与当前不可访问的上游仓库, 并对本地快照的每个 loader 可见文件记录 SHA-256。既有目录只有在仓库、revision 和逐文件摘要全部一致时才可复用。

三套 official-reference 的语义质量指标统一由 `ViT-g-14` 和 `laion/CLIP-ViT-g-14-laion2B-s12B-b42K@4b0305adc6802b2632e11cbe6606a9bdd43d35c9` 中的 `open_clip_pytorch_model.bin` 计算。checkpoint 文件固定为5467006745字节, SHA-256 为 `6aac683f899159946bc4ca15228bb7016f3cbb1a2c51f365cba0b23923f344da`。runner 只向官方命令传入通过本地逐文件核验的 checkpoint 路径, 不使用可漂移的远程 pretrained tag。

## 三、检测访问边界

正式检测接口为：

```python
detect_image_only_watermark(
    image,
    key_material,
    config,
    image_latent_encoder,
    image_attention_extractor,
    image_aligner,
)
```

该接口没有原始 latent、生成轨迹、原始图像、Prompt 或样本级安全基底参数。数据集运行器只允许通过该接口生成正式检测记录。

## 四、固定模板与安全投影的盲检闭合

检测模板由密钥、公开模型标识和 latent 形状确定。嵌入端求得安全基底 $B$ 后执行：

$$
\bar\nu=BB^\top\nu.
$$

检测端与原始固定模板 $\nu$ 计算相关性, 不需要恢复 $B$。该设计的主要考虑在于：

1. $B$ 只控制嵌入方向对语义和视觉特征的响应；
2. 固定模板提供图像盲检所需的可重建参考；
3. $\nu^\top BB^\top\nu\ge0$, 投影保留的模板能量可以作为运行记录审计；
4. 投影能量过低时运行必须失败, 不能退回 one-hot 或周期平铺代理。

## 五、完整实验链

1. `scripts/semantic_watermark_scientific_workflow.py` 创建一个 `sd35_method_runtime_gpu` 隔离执行, 子解释器入口为 `experiments.runtime.semantic_watermark_scientific_session`；
2. 科学会话通过 `experiments.runners.image_only_dataset_workload` 读取当前 `paper_run` Prompt 文件并构造唯一正式方法配置；
3. 复用一次加载的 SD3.5 Medium、VAE 和 CLIP 运行时；
4. 对每个 Prompt 生成 clean 与 watermarked 图像；
5. 对选定 test Prompt 执行9类标准图像攻击和8类真实 GPU 扩散攻击；
6. 所有样本只从最终图像重新编码并检测；
7. calibration clean negative 冻结包含 rescue 的完整判定协议；
8. test split 只应用冻结协议并报告置信上界；
9. 按需通过 `experiments.ablations.mechanism_ablation_workload` 重新运行全部机制消融；
10. 外层 workflow 写入科学执行绑定并复用同一子解释器重新打包；
11. 结果 records 和 manifest 进入论文共同协议 builder；
12. 生成轨迹检测、proxy 分数和 counterfactual 分数变换不能支持论文主张。

8类 GPU 扩散攻击共享同一个受治理实现。`img2img_regeneration`、`sdedit_regeneration`、`diffusion_purification`、`global_editing_attack` 和 `visual_paraphrase_attack` 通过 `StableDiffusion3Img2ImgPipeline` 的 `image + strength` 路径执行；`flow_matching_inversion_regeneration` 通过 SD3 scheduler 的反向 Euler 积分恢复高噪声 latent 后重建；`local_editing_attack` 通过 inpainting pipeline 和面积受控 mask 执行，并在输出端严格保留 mask 外源像素；`adversarial_removal_attack` 在冻结候选查询预算内逐一调用对应方法的真实仅图像连续检测分数并选择最低分候选。攻击参数、随机种子、mask 摘要和查询轨迹写入 `attack_execution`，外部 baseline 不维护第二套 latent 混合攻击。

## 六、主张边界

实现存在不等于论文结果成立。下列条件全部满足后, 结果记录才允许进入主张门禁：

1. 运行记录的 `jvp_mode` 为 `torch_func_linearize_exact_jvp`、`torch_autograd_exact_jvp` 或 `torch_autograd_exact_jvp_compatibility`, 且不得使用有限差分；
2. 基底记录包含响应残差、归一化相对响应残差和正交误差, 且相对响应不超过 0.0001；
3. attention 来源为真实 Q/K 投影和 autograd；
4. 检测访问模式为 `image_key_public_model_only`；
5. test clean negative 的 95% 误报率上界不超过目标 FPR；
6. FID/KID 使用 torch-fidelity `inception-v3-compat` 的 2048 维特征, 配对质量指标来自真实图像集合；
7. 消融记录明确 `generation_rerun=true` 且未使用 counterfactual 分数变换；
8. 外部 baseline 使用相同 Prompt、攻击和固定 FPR 统计边界。

正式 FID/KID 的样本门禁分别为 70/700/7000 对 clean/watermarked 图像, 与三个运行层级的 Prompt 总数一致。该门禁属于项目特定的证据治理要求; 通用做法是明确记录特征提取器版本、输入图像摘要、特征维度和实际样本数。

质量后端固定为 [torch-fidelity v0.4.0](https://github.com/toshas/torch-fidelity/tree/v0.4.0), 提取器为 `inception-v3-compat`, 特征层为 `2048`。运行记录必须保存 `feature_extractor_id=torch_fidelity_0_4_0_inception_v3_compat_2048`; 只有 `canonical_formal_feature_extractor_ready=true` 的质量摘要才能通过论文记录门禁。普通 torchvision ImageNet 分类权重或像素直方图不能冒充该后端。

为降低 Colab 上精确 JVP 遇到 fused attention 不支持 forward AD 的风险, 正式运行固定 CLIP 视觉编码器使用 eager attention, VAE 使用 Diffusers `AttnProcessor`。该调整只改变等价注意力算子的运行实现, 不更改模型权重或方法目标; 实际配置写入 `scientific_autograd_compatibility` 环境记录。显存不足、形状错误或模型实现错误仍直接失败, 不能被兼容路径吞掉。
