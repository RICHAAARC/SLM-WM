# 真实科学算子实现与论文证据闭合

## 一、文件级职责

| 机制 | 正式实现 | 作用 |
| --- | --- | --- |
| 分支风险场 | `main/methods/semantic/branch_risk.py` | 分别构造 LF、尾部截断和注意力几何风险与承载预算 |
| 密钥随机原语 | `main/core/keyed_prg.py` | 通过版本化 SHA-256 计数器流为内容模板、Jacobian 候选方向和注意力关系符号生成规范 CPU float32 Tensor |
| Tensor 内容身份 | `main/core/digest.py` | 通过版本化 dtype、shape 与连续原始字节 SHA-256 绑定风险、基底、分支更新和 Q/K 原子 |
| 真实 Jacobian Null Space | `main/methods/subspace/jacobian_nullspace.py` | 通过完整特征 JVP/VJP、显式风险算子和无阻尼 PSD-CG 求解 rank-4 latent Null Space |
| 语义与手工结构统计 | `experiments/runtime/diffusion/semantic_features.py` | 以512维完整归一化 CLIP embedding 和204维 RGB 统计/梯度/8x8池化向量定义716维 Jacobian，并提供有限更新与最终成图复验 |
| LF 与尾部载体 | `main/methods/carrier/keyed_tensor.py` | 通过版本化、设备无关的 SHA-256 计数器高斯 PRG 构造检测端可重建模板, 并在嵌入端投影到安全子空间 |
| 真实注意力梯度 | `main/methods/geometry/differentiable_attention.py` | 从 Transformer `to_q`/`to_k` 得到真实 attention, 构造有身份摘要的稳定 token pair 权重并对 latent 求梯度 |
| 几何恢复 | `main/methods/geometry/attention_alignment.py` | 使用同一 pair 权重联合规范拉回 $W A_{\mathrm{obs}} W^\top$、观测前推 $V S_K V^\top$、双向覆盖惩罚和攻击无关的分层局部搜索恢复图像参考系 |
| 仅图像检测 | `main/methods/detection/image_only.py` | 只接收图像、密钥和公开模型配置, 传递注册前后的同一 pair 权重并完成内容主判与同阈值救回 |
| 真实模型运行 | `experiments/runners/semantic_watermark_runtime.py` | 在 SD3.5 Medium 中执行完整方法与同种子 carrier-only 总机制效应反事实, 持久化无 attention 更新原子, 并以三边最终特征保持和真实 Q/K 双归因增益门禁验证 attention 可观测性 |
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

高斯幅值尾部截断分支的正式运行标识为 `tail_robust`。`build_tail_robust_template(...)` 对标准高斯模板按元素绝对幅值稳定排序，精确保留 `ceil(element_count * tail_fraction)` 个元素，并以展平索引处理同幅值排序；该算子不执行 FFT、DCT、带通滤波或空间频带 mask, 因而不具有空间频带定义。内容模板与安全投影实现位于 `main/methods/carrier/keyed_tensor.py`。

注意力分支风险必须接收由真实跨层 Q/K 关系计算的独立稳定度，核心接口不接受缺失值。正式层集合精确固定为 `transformer_blocks.0.attn` 与 `transformer_blocks.23.attn`, 运行时直接按名称解析公开 `to_q`、`to_k` 与 `heads` 协议。token 坐标采用 `normalized_xy_token_centers_corner_endpoints_v1`, 角点中心分别落在 -1 与 1；关系稳定图插值和图像仿射重采样统一使用 `align_corners=True`。最终图像 Q/K 提取在冻结检测日程上调用 scheduler 的 `scale_noise`；缺少该方法或方法不可调用时运行失败，当前协议不定义线性 latent/noise 混合作为替代算子。

分支风险中的 `local_contrast_risk` 定义为解码灰度图相对反射填充5x5局部均值的绝对偏离。`adjacent_step_stability` 直接来自当前与紧邻上一 scheduler 步 latent 的解码 RGB 差异；注入回调在每个 post-step 时刻更新参考 latent，并把参考索引和 Tensor 内容 SHA-256 写入更新原子。204维 `handcrafted_structure_feature` 由 RGB 通道均值/标准差、水平/垂直绝对梯度均值和8x8 RGB 平均池化组成；一般感知质量结论必须独立依赖正式 FID、KID 与配对图像质量指标。

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

全部密钥化随机原语由密钥、算子 domain、输出 shape 和 `keyed_prg_version=sha256_counter_box_muller_float32_v1` 确定。PRG 使用 SHA-256 大端计数器流和53位开区间均匀映射；内容模板与 Jacobian 候选方向继续执行 Box-Muller 高斯变换，注意力关系符号直接使用均匀数阈值。所有结果先物化为 CPU float32 规范 Tensor，CPU/CUDA 设备 RNG 均不参与方法身份。检测模板的 domain 额外绑定公开模型标识和分支。嵌入端求得安全基底 $B$ 后执行：

$$
\bar\nu=BB^\top\nu.
$$

检测端与原始固定模板 $\nu$ 计算相关性, 不需要恢复 $B$。该设计的主要考虑在于：

1. $B$ 只控制嵌入方向对 CLIP 语义和声明的手工结构统计坐标的响应；
2. 固定模板提供图像盲检所需的可重建参考；
3. $\nu^\top BB^\top\nu\ge0$, 投影保留的模板能量可以作为运行记录审计；
4. 投影能量过低时运行必须失败, 不能退回 one-hot 或周期平铺代理。

## 五、完整实验链

1. `scripts/semantic_watermark_scientific_workflow.py` 创建一个 `sd35_method_runtime_gpu` 隔离执行, 子解释器入口为 `experiments.runtime.semantic_watermark_scientific_session`；
2. 科学会话通过 `experiments.runners.image_only_dataset_workload` 读取当前 `paper_run` Prompt 文件并构造唯一正式方法配置；
3. 复用一次加载的 SD3.5 Medium、VAE 和 CLIP 运行时；
4. 对每个 Prompt 生成 clean 与 watermarked 图像；
5. 对每次实际写回和最终 clean/watermarked 成图执行完整 CLIP/手工结构统计保持门禁；
6. 对选定 test Prompt 执行9类标准图像攻击和8类真实 GPU 扩散攻击；
7. 所有样本只从最终图像重新编码并检测；
8. calibration clean negative 冻结包含 rescue 的完整判定协议；
9. test split 只应用冻结协议并报告置信上界；
10. 按需通过 `experiments.ablations.mechanism_ablation_workload` 重新运行全部机制消融；
11. 外层 workflow 写入科学执行绑定并复用同一子解释器重新打包；
12. 结果 records 和 manifest 进入论文共同协议 builder；
13. 生成轨迹检测、proxy 分数和 counterfactual 分数变换不能支持论文主张。

8类 GPU 扩散攻击共享同一个受治理实现。`img2img_regeneration`、`sdedit_regeneration`、`diffusion_purification`、`global_editing_attack` 和 `visual_paraphrase_attack` 通过 `StableDiffusion3Img2ImgPipeline` 的 `image + strength` 路径执行；`flow_matching_inversion_regeneration` 通过 SD3 scheduler 的反向 Euler 积分恢复高噪声 latent 后重建；`local_editing_attack` 通过 inpainting pipeline 和面积受控 mask 执行，并在输出端严格保留 mask 外源像素；`adversarial_removal_attack` 在冻结候选查询预算内逐一调用对应方法的真实仅图像连续检测分数并选择最低分候选。攻击参数、随机种子、mask 摘要和查询轨迹写入 `attack_execution`，外部 baseline 不维护第二套 latent 混合攻击。

## 六、主张边界

实现存在不等于论文结果成立。下列条件全部满足后, 结果记录才允许进入主张门禁：

1. 运行记录的 `jvp_mode` 为 `torch_func_exact_jvp_vjp` 或 `torch_autograd_exact_jvp_vjp_compatibility`，且 `feature_compression_applied=false`；
2. 求解器为 `matrix_free_full_jacobian_psd_cg`、`cg_damping=0`，全部方向 CG 收敛且相对残差不超过 $10^{-6}$；
3. QR 后每个基底列的完整 Jacobian 相对响应不超过0.0001，投影能量不低于0.01，正交误差不超过 $10^{-5}$；
4. 三分支合成更新按真实 latent dtype 写回后，以实际 `written_latent - latent` 增量重新执行完整特征精确 JVP，其响应范数相对当前完整特征范数的比例不超过0.0001；
5. 每次实际写回以及最终 clean/完整方法、clean/carrier-only、carrier-only/完整方法三条成图边均通过 CLIP cosine 和视觉漂移门禁；
6. attention 来源为真实 Q/K 投影和 autograd；中心化 logit、可微 rank、抽样图像 token 关系概率和概率偏离与距离偏离的双中心交互四分量分别完成逐行加权归一化后等权组合, 嵌入与盲检共享该关系算子及 pair 构造规则；
7. carrier-only 与完整方法首个注入前 latent 字节级相同, 更新数、顺序和 scheduler 轨迹一致；carrier-only 更新原子无 attention 来源、分数、更新、关系、pair 身份或 attention Null Space, 且其 JSONL 路径、文件 SHA-256 和内容摘要绑定结果与 manifest；
8. 最终 clean、carrier-only 与完整方法成图在 CUDA 上重新构造直接 Q/K 四分量关系, 完整方法相对同种子 carrier-only 的自身盲选择归因增益和冻结 carrier-only pair 权重归因增益都严格超过0.0001, 且反事实保持记录、四分量归因、关系图身份、Q/K 记录与 manifest 绑定同一身份和图像 SHA-256；
9. `slm_wm_tensor_content_v1` 完整绑定三个风险场、三个 Null Space 的候选/预算/响应/基底、三分支更新与实际写回增量；Q/K 原子完整覆盖注入四角色、最终成图三角色及盲检两角色；
10. 检测访问模式为 `image_key_public_model_only`；
11. test clean negative 的95%误报率上界不超过目标 FPR；
12. FID/KID 使用 torch-fidelity `inception-v3-compat` 的2048维特征，配对质量指标来自真实图像集合；
13. 消融记录明确 `generation_rerun=true` 且未使用 counterfactual 分数变换；
14. 外部 baseline 使用相同 Prompt、攻击和固定 FPR 统计边界。

正式 FID/KID 的样本门禁分别为 70/700/7000 对 clean/watermarked 图像, 与三个运行层级的 Prompt 总数一致。该门禁属于项目特定的证据治理要求; 通用做法是明确记录特征提取器版本、输入图像摘要、特征维度和实际样本数。

质量后端固定为 [torch-fidelity v0.4.0](https://github.com/toshas/torch-fidelity/tree/v0.4.0), 提取器为 `inception-v3-compat`, 特征层为 `2048`。运行记录必须保存 `feature_extractor_id=torch_fidelity_0_4_0_inception_v3_compat_2048`; 只有 `canonical_formal_feature_extractor_ready=true` 的质量摘要才能通过论文记录门禁。普通 torchvision ImageNet 分类权重或像素直方图不能冒充该后端。

为降低 Colab 上完整特征 JVP/VJP 遇到 fused attention 不支持自动微分的风险，正式运行固定 CLIP 视觉编码器使用 eager attention，VAE 使用 Diffusers `AttnProcessor`。该调整只改变等价注意力算子的运行实现，不更改模型权重或方法目标；实际配置写入 `scientific_autograd_compatibility` 环境记录。显存不足、形状错误、CG 不收敛或模型实现错误仍直接失败，不能被兼容路径吞掉。

## 七、原子证据与派生结论绑定

论文结论只接受能够从原子记录独立重建并通过即时文件摘要核验的派生产物：

主方法科学原子先执行 Tensor 内容自校验。风险值、预算、资格 mask、Jacobian 候选矩阵、风险预算、响应矩阵、最终基底和三个分支更新均使用同一版本化协议；真实 Q/K 逐层绑定抽样 Q、K、中心化 logit、关系概率和二维 token 索引。算子身份摘要只描述冻结协议, 内容摘要描述一次实际评价, 两者不能互相替代。

1. 正式消融的每个 `ablation_id` 必须逐字段等于登记的机制开关配置, 每个运行结果必须通过科学完成单元来源校验, 且输出目录必须属于该论文层级和消融身份；
2. `formal_detection_records.jsonl` 与 `per_ablation_frozen_protocols.json` 同时保存字节级 SHA-256 和解析内容稳定摘要, 两类身份均绑定到 summary、manifest config 与 manifest metadata；
3. 消融表中的检测判定、攻击覆盖率和 `paired_ssim` 只能由冻结协议、逐检测原子和实际 runtime 结果重建, 不接受由 `ablation_id` 推断的分数变换；
4. 正式 FID/KID 只接受 `watermark_embedding` 的 `clean_to_watermarked` 图像对, 每条记录绑定唯一 `run_id`, 原始 feature 行不得直接声明论文支持；
5. 每对 source/comparison 必须指向不同实际文件, source 路径与 comparison 路径各自全局唯一且跨角色不相交, $N$ 个 Prompt 必须对应精确 $2N$ 条图像解析记录；
6. 每个 source/comparison 路径都必须有唯一图像解析记录, 解析记录的 SHA-256 必须等于闭合阶段即时读取实际图像文件所得摘要, feature record 只能引用该解析后的实际路径, 固定提取器和2048维有限向量；
7. 逐攻击保守优势以所有 baseline 的 TPR 置信区间上界最大者作为比较对象, 只有主方法置信区间下界严格高于该上界时才允许声明该攻击上的优势；
8. 失败案例记录精确等于全部假阴性按冻结上限12截取的结果, 上限不能设为0以跳过语义核验；
9. 主表、攻击表、FID/KID 表、置信区间表、逐攻击表、失败 JSONL 与失败 SVG 的联合语义摘要必须由闭合侧独立重建, 并与结果分析 summary 和 manifest metadata 同时一致。

这一证据绑定方式属于通用的可复现实验治理写法。项目特定部分是把真实消融重运行、仅图像检测协议、实际图像 SHA 和七类论文 payload 纳入同一个 fail-closed 闭合门禁。
