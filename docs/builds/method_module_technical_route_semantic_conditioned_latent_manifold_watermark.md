# 详细技术路线：SLM-WM 方法模块设计

## 一、文档定位

本文档描述语义条件潜流形水印（SLM-WM）的模块职责、数据流和可审计边界。可证伪方法语义以 `method_semantic_invariants.md` 为权威来源, 算法展开以 `algorithm_primitives_semantic_conditioned_latent_manifold_watermark.md` 为准，论文叙事以 `method_section_semantic_conditioned_latent_manifold_watermark.md` 为准，代码映射以 `real_scientific_operator_implementation.md` 为准。

“潜流形”在本项目中严格限定为当前 latent 点处隐式完整特征水平集的局部安全切空间解释。正式代码求解分支风险支持的 Jacobian Null Space 数值基底，并以逐列残差、投影能量、风险支持清理后逐分支 JVP、风险硬包络、实际 dtype 联合写回 JVP 和有限特征变化进行门控；它不构造全局非线性流形，不验证常秩定理条件，也不执行坐标图、测地线或回缩求解。三分支更新由模板投影、风险约束缩放、真实 Q/K 梯度投影与单调回溯顺序构造，不是联合标量优化器。

### 方法机制解释边界

完整716维特征函数由512维归一化 CLIP `image_embeds` 和204维显式手工结构描述符组成。204维分量不是通用感知质量模型或学习式语义表示, 但它也不是代理、占位或诊断旁路；它按冻结坐标顺序真实参与风险计算、完整 Jacobian、写回后 JVP 和有限特征保持门禁。感知质量结论必须由外层实验指标独立给出。

局部特征水平集切空间就是本项目冻结的“潜流形”方法对象。项目不以全局流形学习为当前机制目标, 因而没有全局参数化、坐标图、测地线、曲率或回缩不表示实现缺失。后续审计必须分别检查局部切空间公式是否真实执行和论文是否越界声称全局流形；不得把“没有全局流形学习”本身登记为算法缺口。

本设计固定采用三个正式分支：

1. `lf_content`：空间低通 LF 主证据；
2. `tail_robust`：高斯幅值尾部截断鲁棒补充证据；
3. `attention_geometry`：真实 Q/K Self-Attention 相对关系几何锚点。

`tail_robust` 按高斯模板元素绝对幅值和展平索引稳定排序, 精确保留冻结比例的分布尾部, 不具有空间频带定义。

---

## 二、统一方法数据流

SLM-WM 的正式方法链为

$$
\text{branch-specific semantic risk fields}
\rightarrow
\text{full-feature JVP/VJP constrained Null Spaces}
\rightarrow
\text{LF/tail/attention latent update}
\rightarrow
\text{image-only content detection}
\rightarrow
\text{same-threshold geometric rescue}
\rightarrow
\text{complete fixed-FPR decision}.
$$

数据流含义如下：

1. 当前样本的语义、纹理、稳定性、显著性和注意力稳定性决定三个独立风险场；
2. 分支资格集合与承载预算形成显式风险对角算子 $B_b$；
3. 完整特征 JVP/VJP 与无阻尼 PSD-CG 执行风险支持 Jacobian 约束投影；
4. LF、尾部截断和注意力几何更新分别投影到对应安全子空间, 再沿原方向执行风险硬包络允许的最大可行标量缩放；
5. 检测端只从待检图像重建固定模板和注意力关系，不读取生成侧私有状态；
6. calibration split 冻结包含几何救回的完整判定协议，test split 只报告结果。

这不是三个独立水印器的串联。分支风险、完整特征 Null Space、载体投影和完整 fixed-FPR 判定共同构成一个方法闭环。

该闭环属于项目特定的构造式实现。通用工程写法包括精确 JVP/VJP、矩阵自由 PSD-CG、QR 正交化、数值残差门禁和实际写回复验；项目特定写法包括三类分支风险、LF/tail 模板角色、直接 Q/K 四分量目标以及注意力更新的内容基底单调回溯。

### （一）冻结科学算子身份

正式运行只读取 `configs/model_sd35.yaml` 的唯一解析结果。算子身份字段固定为 `model_id=stabilityai/stable-diffusion-3.5-medium`、`model_revision=b940f670f0eda2d07fbb75229e779da1ad11eb80`、`vision_model_id=openai/clip-vit-base-patch32`、`vision_model_revision=3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`、`pipeline_class_name=diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline`、`vae_class_name=diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL`、`transformer_class_name=diffusers.models.transformers.transformer_sd3.SD3Transformer2DModel`、`scheduler_class_name=diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler`、`vae_scaling_factor=1.5305`、`vae_shift_factor=0.0609`、`latent_torch_dtype=float16`、`vision_torch_dtype=float32` 与 `public_detection_schedule_index=7`。

完整解析 dataclass 的 `asdict` 值与 `formal_method_config_schema=slm_wm_formal_method_runtime_config` 形成规范 payload；使用 UTF-8、`ensure_ascii=false`、键名升序、无空白分隔符 `(',', ':')` 和 SHA-256 得到 `formal_method_config_digest`。运行时类名、VAE scale/shift、dtype、检测索引、冻结层集合和配置摘要必须逐项一致, 不能由环境变量或运行时默认值改变。

---

## 三、模块与仓库职责映射

| 方法模块 | 正式实现 | 输入 | 输出 | 禁止行为 |
| --- | --- | --- | --- | --- |
| 分支风险 | `main/methods/semantic/branch_risk.py` | 语义、纹理、相邻步稳定度、局部对比度风险和跨层 Q/K 稳定度 | 三个风险场、预算和资格集合 | 用单一共享风险标量替代三个分支 |
| Jacobian Null Space | `main/methods/subspace/jacobian_nullspace.py` | latent、716维完整特征函数、分支预算、实际载体方向和密钥 | 三个 rank-4 Null Space 基底及 CG/逐列残差记录 | 低维草图制造代数零空间；阻尼求解后仍声明 Null Space |
| 固定内容模板 | `main/methods/carrier/keyed_tensor.py` | 密钥、公开模型标识和 latent 形状 | LF 模板与尾部截断模板 | 模板依赖 Prompt、生成轨迹或样本级基底 |
| 固定载体安全投影 | `main/methods/carrier/keyed_tensor.py` | 固定模板、分支基底和最小能量保留率 | 投影方向和能量保留率 | 对近零投影继续运行；使用未投影模板写回 |
| 风险硬包络与唯一合成 | `main/methods/update_composition.py` | float32 投影方向、同一 NCHW 有效预算、名义强度和 original latent | 风险清理单位方向、float32 分支更新、逐位置包络、共同缩放候选、实际 dtype 增量和可重算合成摘要 | 恢复固定 L2 强度；混用方向 epsilon 与数值 epsilon；按样本预算最大值重标定；分别量化分支；在 runner 或 attention 模块实现第二套加法顺序 |
| 科学内容身份 | `main/core/digest.py` | 风险、预算、基底、更新和 Q/K Tensor | 绑定 dtype、shape 与原始字节的版本化 SHA-256 | 只保存均值、形状或可替换标签作为科学内容证据 |
| 注意力几何 | `main/methods/geometry/differentiable_attention.py` | 真实 Transformer Q/K 与 latent | 目标梯度、分数增益和回溯记录 | 使用合成 attention map 支持正式主张 |
| 图像盲检 | `main/methods/detection/image_only.py` | 待检图像、密钥和公开模型配置 | 内容分数、几何统计和 evidence 判定 | 读取 Prompt、源 latent、轨迹或样本级 Null Space |
| 真实模型运行 | `experiments/runners/semantic_watermark_runtime.py` | SD3.5 Medium pipeline 与方法配置 | clean/watermarked 图像和科学算子记录 | 绕过科学算子门禁写出正式结果 |
| 数据集协议 | `experiments/runners/image_only_dataset_runtime.py` | Prompt split、攻击配置和目标 FPR | 冻结协议、test records 和质量 registry | 使用 test split 调阈值 |
| 正式消融 | `experiments/ablations/runtime_rerun.py` | 改变后的机制配置与冻结检测协议 | 重新生成、攻击和检测的消融记录 | 修改完整方法已有分数模拟消融 |

`paper_workflow/` 只负责 Colab 调度与 Drive 续跑，`scripts/` 只负责命令包装，`paper_experiments/` 负责 baseline 和论文结果闭合。正式算法不能只存在于 Notebook cell。

---

## 四、分支风险模块

### （一）输入信号

运行时提供统一形状的信号图：

$$
\phi(u)=
[
\phi_{\mathrm{sem}}(u),
\phi_{\mathrm{tex}}(u),
\phi_{\mathrm{adj}}(u),
\phi_{\mathrm{lcr}}(u),
\phi_{\mathrm{attn\_stab}}(u)
].
$$

$\phi_{\mathrm{sem}}$ 把 CLIP patch-to-CLS cosine 经 $(c+1)/2$ 映射后双线性插值；$\phi_{\mathrm{tex}}$ 把解码灰度水平/垂直前向绝对差之和除以2后插值；$\phi_{\mathrm{lcr}}$ 固定为解码灰度相对反射填充5x5局部均值的绝对偏离；前三者只截断到 $[0,1]$, 不执行逐样本 min-max, 且精确使用 `risk_image_signal_interpolation_mode=bilinear` 与 `risk_image_signal_align_corners=false`。$\phi_{\mathrm{adj}}$ 固定为当前与紧邻上一 scheduler 步解码 RGB 的逐位置稳定度。运行时在全部 post-step 回调上维护上一 latent，缺失时直接失败。$\phi_{\mathrm{attn\_stab}}$ 独立来自不少于两个冻结层的直接 Q/K 关系, 并精确使用 `risk_attention_signal_interpolation_mode=bilinear` 与 `risk_attention_signal_align_corners=true`。

### （二）分支输出

对分支 $b\in\{\mathrm{LF},\mathrm{tail},\mathrm A\}$ 分别计算

$$
\rho_b(u)
=
\frac{1}{Z_b}
\left[
\eta_c^b\phi_{\mathrm{lcr}}(u)
+\eta_m^b\phi_{\mathrm{sem}}(u)
+\eta_t^b\psi_b(\phi_{\mathrm{tex}}(u))
+\eta_d^b(1-\phi_{\mathrm{adj}}(u))
+\eta_A^b(1-\phi_{\mathrm{attn\_stab}}(u))
\right],
\qquad
Z_b=\eta_c^b+\eta_m^b+\eta_t^b+\eta_d^b+\eta_A^b.
$$

其中 $Z_b$ 精确等于非负权重和, $\psi_{LF}(q)=q$、$\psi_{tail}(q)=1-q$, $\psi_A(q)$ 由 `risk_neutral_texture_value=0.5` 定义。系统生成承载预算 $b_b(u)$ 与严格资格集合 $\Omega_b=\{u:\rho_b(u)<\tau_b\}$；等于阈值的位置不合格。空间有效预算在 channel 维复制成 NCHW Tensor, 再按连续 NCHW 顺序进入 $B_b$。同一 Tensor 身份同时进入子空间求解、最终逐元素硬包络和运行记录。零预算位置的安全方向泄漏超过 `risk_bounded_scale_direction_epsilon=1e-12` 时失败；风险投影后不得无条件恢复固定二范数强度。三分支配置精确来自 `lf_content_risk_config.*`、`tail_robust_risk_config.*` 与 `attention_geometry_risk_config.*`, 例如 LF 局部对比权重读取 `lf_content_risk_config.local_contrast_risk_weight`, 不读取 dataclass 隐式默认值。

---

## 五、精确 Jacobian Null Space 模块

### （一）种子方向与完整特征

每个分支构造

$$
D_b=[d_1^b,\ldots,d_m^b],\qquad D_b^\top D_b=I.
$$

LF 与尾部截断分支把对应固定模板作为首个方向；注意力分支把真实 Q/K 目标梯度作为首个方向；其余列由密钥化方向补齐。完整特征函数先按 VAE 的 `latent / scaling_factor + shift_factor` 解码约定得到 $[0,1]$ RGB, 再将224x224 bicubic、`align_corners=false`、`antialias=true` 和冻结 CLIP mean/std 作为唯一可微预处理。语义坐标必须读取512维 `image_embeds` 并进行非零能量 L2 归一化；结构坐标依次连接 RGB 均值、总体标准差、水平绝对梯度、垂直绝对梯度和按 CHW 展平的8x8平均池化。最终按512维语义在前、204维结构在后的顺序形成716维输出, 且不执行压缩。该204维向量不声明覆盖一般感知质量；非有限值、零语义能量、多样本输入或宽度漂移均直接失败。

### （二）矩阵自由风险支持约束投影

正式实现通过 `torch.func.linearize` 与 `torch.func.vjp` 计算精确 JVP/VJP。对分支风险对角算子 $B_b$ 和种子 $d_i^b$，无阻尼 PSD-CG 求解

$$
(JB_b^2J^\top)y_i=JB_bd_i^b,
$$

并构造

$$
u_i^b=B_bd_i^b-B_b^2J^\top y_i.
$$

收集4个独立合格方向后得到

$$
N_b=\operatorname{qr}([u_1^b,\ldots,u_4^b]).
$$

设接受的路由候选和投影方向为 $V_b$ 与 $U_b$, float32 reduced QR 满足 $U_b=N_bR_b$。基底列使用 `null_space_numerical_epsilon=1e-12` 规范符号；$R_b$ 对角近零或条件数超过 `maximum_qr_condition_number=1000000.0` 时失败。独立参考严格采用 `qr_reference_solve_protocol=right_upper_triangular_solve_without_explicit_inverse`, 通过 $\widetilde V_bR_b=V_b$ 求解。每列相对残差固定为 $\|JN_b[:,j]\|_2/\max(\|J\widetilde V_b[:,j]\|_2,\texttt{null\_space\_numerical\_epsilon})$, 不能使用共享 RMS。CG 最大64次、阻尼为0；只有右端项精确为零时允许零步返回, 其余候选必须通过最终解重新执行算子得到的真实相对残差 $10^{-6}$ 门禁。绝对残差或递推 residual 不能替代该定义。平方能量保留率不得低于0.01, 每列残差不得超过 $10^{-4}$, 正交误差不得超过 `maximum_orthogonality_error=0.00001`。

角色限定内容摘要必须分别绑定 $D_b$、$B_b$、$J(B_bD_b)$、$U_b$、$JU_b$、$N_b$、$JN_b$ 和 $J\widetilde V_b$；每类响应只能使用其对应角色字段并绑定该数学对象。

三个分支在 float32 中按 LF、tail、attention 固定顺序累加, original latent 也先转换为 float32, 随后只执行一次 cast。若实际写回违反联合包络, 按 `quantized_budget_envelope_backtracking_factor=0.5` 对三个分支共同缩放并从头重算单次 cast, 最多执行 `quantized_budget_envelope_backtracking_maximum_steps=24` 次减半。每个候选都必须通过非零写回、严格联合包络、完整 JVP、有限变化及实际 Q/K 相对原 latent/同倍率内容基底的单调性；全部失败时停止。完整扩散结束后还必须比较最终 clean 与 watermarked 成图。两级保持门禁均要求 CLIP cosine 不低于0.995且手工结构统计特征相对漂移不高于0.02。

---

## 六、内容载体模块

### （一）空间低通 LF

LF 模板由密钥高斯模板在 latent 二维空间轴上执行平均池化：

$$
\nu_{\mathrm{LF}}
=
\operatorname{Norm}
\left(\operatorname{AvgPool}_{k\times k}
(\operatorname{PRG}_{\mathcal N}(K_{\mathrm{LF}},M,shape))\right).
$$

高斯 PRG 固定为 `sha256_counter_normal_icdf_table20_float32`。domain payload 精确包含 `keyed_prg_version`、`key_material`、`domain_fields` 和 `shape`, 通过 UTF-8、`ensure_ascii=false`、键名升序和无空白分隔符 `(',', ':')` 形成 stable JSON, 再执行 SHA-256 得到32字节 domain digest。counter 从0开始并编码为16字节无符号大端整数；`SHA256(domain_digest || counter_uint128_be)` 的连续32字节块组成 MSB-first 大端比特流。高斯路径跨块连续提取20位索引 $i$, 查询 $q_i=\operatorname{round}_{\mathrm{binary32}}\!\left(\Phi^{-1}((i+0.5)/2^{20})\right)$ 的冻结1048576项表。表的完整大端字节 SHA-256 为 `70abf440a7f3670147965ffa52f5aaa639dab97f6282b68f3a9a1b1ce5e6cf5a`。该有限离散 Q20 量化标准正态不是连续精确的 $\mathcal N(0,1)$；其理想中点 KS 距离为 $2^{-21}$, 含 float32 舍入的登记上界为 `4.912236096776823e-7`。规范 float32 生成和目标 dtype 转换都在 CPU 完成, 随后才搬运到执行设备。独立 uniform 路径把每个块按 offset 0、8、16、24 分成8字节无符号大端 word, 取高53位并按 $u=(m+1)/(2^{53}+2)$ 映射到 $(0,1)$；该路径只用于按0.5阈值生成 attention 关系符号。设备 RNG 不参与方法身份。MPFR 复验是外层参考证据, 不进入 PRG 算法摘要。当前固定向量只在 Windows CPU 实测, Linux/Colab 逐字节一致性留给 GPU 运行前门禁。

该分支只在 height/width 二维轴执行平均池化, 对每个 batch/channel 独立使用 kernel 5、stride 1、padding 2、二维零填充、`ceil_mode=false`、`count_include_pad=true` 和 `divisor_override=null`, 不跨 batch 或 channel 轴传播数值；池化后去均值与 L2 归一化各自在整个模板 Tensor 上计算一个全局标量, 因而具有明确的离散空间低通定义。

### （二）高斯幅值尾部截断

尾部模板为

$$
\widetilde\nu_{\mathrm{tail},i}
=
\nu_{\mathrm{tail},i}
\mathbb I(i\in I_\gamma),
\qquad
I_\gamma=\operatorname{TopK}_{\lceil n\gamma\rceil}
\left(\left\{(|\nu_i|,-i)\right\}_{i=1}^{n}\right).
$$

$\gamma$ 是 Q20 中点逆 CDF 量化标准正态元素绝对幅值的尾部保留比例，同幅值元素由展平索引升序确定选择顺序。非入选元素保持精确0, 模板只执行整体二范数归一化, 不执行去均值。该过程不执行 FFT、DCT、空间波数排序或频带 mask，因此只定义幅值域筛选, 不定义空间频带。其攻击鲁棒性必须由真实实验验证。

尾部载体协议绑定 $\gamma$、稳定排序规则与 PRG 版本。仅图像检测原子保存模板 shape、元素总数、精确选中数、阈值、实际保留比例与模板内容摘要；raw/aligned 路径必须复用同一 LF 与尾部固定模板身份。

### （三）安全投影与内容分数

两个固定模板分别投影到对应安全子空间, 令 $v_b=\operatorname{Norm}(N_bN_b^\top\nu_b)$。最终更新由风险硬包络算子构造：

$$
\Delta z_t^b
=
\operatorname{RiskBoundedScale}
\left(v_b,b_b^{\mathrm{eff}},\lambda_b\|z_t\|_2\right),
\qquad b\in\{\mathrm{LF},\mathrm{tail}\}.
$$

投影结果始终保持 float32。风险算子先在单位方向上检查零预算支持, 清理允许的数值残差并重新单位化；方向活动阈值固定为 `risk_bounded_scale_direction_epsilon`, 最终步长退化阈值固定为 `null_space_numerical_epsilon`。清理后的 LF、tail 与 attention 单位方向分别相对其投影前模板或真实 Q/K 梯度重新执行完整716维精确 JVP, 相对响应残差不大于 `maximum_relative_response_residual` 后才允许合成。该逐分支门禁不能由最终联合 JVP 替代, 因为联合响应可能抵消。

检测端不恢复 $N_b$，只计算待检图像编码与固定模板的相关性。观测量和模板分别要求有限、非零中心化能量；实现先分别归一化方向再求内积, 不得把零方差或非有限输入写成合法0分：

$$
s_c=0.70s_{\mathrm{LF}}+0.30s_{\mathrm{tail}}.
$$

两个分支不能分别设置独立正判阈值后投票。

---

## 七、Q/K 注意力几何模块

正式关系算子精确绑定 `transformer_blocks.0.attn` 与 `transformer_blocks.23.attn`, 在冻结空文本条件和二维图像 token 抽样集合上直接调用模型 `to_q` 与 `to_k`；层解析不依赖模块枚举顺序。若 source grid 边长为 $s$, 抽样边长为 $m=\min(s,8)$, 每轴使用 $\operatorname{round}(k(s-1)/(m-1))$ 得到至多64个二维 token, 不执行一维序号等距抽样。
对每个注意力头先计算 $\ell^{(h)}=Q^{(h)}K^{(h)\top}/\sqrt{d_h}$, 再构造

$$
R_{ij}=[L_{ij},\rho_{ij},P_{ij},G_{ij}]\in\mathbb R^4.
$$

$L$ 是各头中心化 logits 的平均；$\rho$ 是基于 $L$、温度0.25并按
$1/n$ 缩放的可微降序行内 rank；$P$ 是各头在抽样图像 token 集合上的
row-softmax 概率平均；第4分量为
$G_{ij}=(P_{ij}-\overline P_i)(D_{ij}-\overline D_i)$, 其中 $D$ 是由公开二维
`token_indices` 得到并按最大网格距离归一化的相对距离。角点 token 中心固定映射到 -1 与 1, 坐标身份为 `normalized_xy_token_centers_corner_endpoints`, token 稳定图插值和图像仿射重采样统一使用 `align_corners=True`。概率与距离双行中心化
移除了距离行均值对第3分量的一阶重复, 且均匀 $P$ 使 $G$ 严格为0。$P$ 是项目定义的抽样图像 token 关系概率, 不表示 SD3.5
包含文本 token 与未抽样图像 token 的完整 joint-attention 权重。正式证据要求
同时直接保存真实 Q/K 的 $L$ 与 $P$。多头概率采用逐头 softmax 后平均, 不以
平均 logits 的 softmax 替代。层名、模块类、头数、head width、scale、Q/K
归一化和源/抽样网格必须进入元数据, 且公开 module scale 与
$1/\sqrt{head\_width}$ 不一致时立即失败。

算子元数据身份与图像数据内容身份分开记录。每层内容原子包含抽样后的 Q、K、中心化 logits、关系概率和二维 token 索引 SHA-256；单次注入保存 `latent_before`、`optimization_content_base_latent`、`accepted_attention_candidate`、`actual_written_content_base_latent` 与 `actual_written_combined_latent` 5个角色, 每个角色同时绑定实际 float32 求值 latent 摘要和 Q/K 分数。最终成图保存 clean、carrier-only、完整方法三个角色, 仅图像盲检保存原图和对齐图两个角色。优化基底角色证明 attention 内部回溯，实际写回基底角色证明共同缩放后的最终单调性；两者不得合并。任何角色或冻结层缺失均使科学门禁失败。

对 batch $b$、层 $\ell$ 和 token $i$, 中心化概率行、跨层一致性、incoming centrality 与排序分数为

$$
\widetilde P_{bi:}^{\ell}=P_{bi:}^{\ell}-\frac1n\mathbf1,
\qquad
\overline P_{bi:}^{\ell}=\frac{\widetilde P_{bi:}^{\ell}}
{\max(\|\widetilde P_{bi:}^{\ell}\|_2,\epsilon_{\mathrm{num}})},
$$

$$
\operatorname{Stab}_i=\operatorname{clip}_{[0,1]}\left(
\operatorname{Mean}_{b,\ell<r}
\frac{1+\langle\overline P_{bi:}^{\ell},\overline P_{bi:}^{r}\rangle}{2}
\right),
\qquad
c_i=\operatorname{Mean}_{\ell,b,j}P_{bji}^{\ell},
$$

$$
\operatorname{Cent}_i=\frac{c_i}{\max(\sum_r c_r,\epsilon_{\mathrm{num}})},
\qquad
q_i=\operatorname{Stab}_i\operatorname{Cent}_i.
$$

选择数固定为 $k=\min(n,\max(4,\lceil0.5n\rceil))$；按 $(-q_i,\operatorname{token\_index}_i)$ 升序稳定排序选择前 $k$ 个, 再把选中位置升序记录。单点与 pair 权重为

$$
a_i=\begin{cases}1,&i\in\mathcal V_A,\\0.25,&i\notin\mathcal V_A,\end{cases}
\qquad
w_{ij}=a_ia_j\mathbf1[i\ne j].
$$

选择摘要、原始二维索引、权重参数和外积规则共同形成 `stable_pair_weight_identity_digest`。一次注入的梯度、内容基底复算、回溯和最终写回复验只消费这一 $(\mathcal V_A,a,w)$ 对象。

每层由独立 uniform Tensor $U^\ell$ 构造密钥对称符号图：

$$
S_{K,ij}^{\ell}=\begin{cases}
0,&i=j,\\
1,&i<j\ \land\ U_{ij}^{\ell}\ge0.5,\\
-1,&i<j\ \land\ U_{ij}^{\ell}<0.5,\\
S_{K,ji}^{\ell},&i>j,
\end{cases}
\qquad
T_{ijc}^{\ell}=\pi_cS_{K,ij}^{\ell},
\quad
\pi=(1,-1,1,1).
$$

令 $W_i=\sum_jw_{ij}$。对 batch、层、row 和分量分别计算

$$
\mu_{R,bic}^{\ell}=\frac{\sum_jw_{ij}R_{bijc}^{\ell}}{W_i},
\qquad
\mu_{T,ic}^{\ell}=\frac{\sum_jw_{ij}T_{ijc}^{\ell}}{W_i},
$$

$$
\operatorname{RowCorr}_{bic}^{\ell}=
\frac{\sum_jw_{ij}(R_{bijc}^{\ell}-\mu_{R,bic}^{\ell})
(T_{ijc}^{\ell}-\mu_{T,ic}^{\ell})}
{\sqrt{\sum_jw_{ij}(R_{bijc}^{\ell}-\mu_{R,bic}^{\ell})^2
\sum_jw_{ij}(T_{ijc}^{\ell}-\mu_{T,ic}^{\ell})^2}}.
$$

$W_i=0$ 或任一中心化能量不大于 $\epsilon_{\mathrm{num}}^2$ 时该 row 无效。每个 batch 先对有效 row 取均值, 无有效 row 时记0；随后按 batch、层等权聚合：

$$
C_c^{\ell}=\frac1B\sum_b
\operatorname{Mean}_{i\in\mathcal I_{b\ell c}^{valid}}
\operatorname{RowCorr}_{bic}^{\ell},
\qquad
C_c=\frac1{|\mathcal L|}\sum_{\ell\in\mathcal L}C_c^{\ell},
$$

$$
s_A(z;K)=\sum_{c=1}^{4}\omega_cC_c(z;K).
$$

完整方法使用 $\omega=(0.25,0.25,0.25,0.25)$；正式留一变体把一个分量置零并令其余三个分量各取 $1/3$。$G$ 通过中心化 $P$ 对 latent 保留非零梯度, 因而四个通道都依赖真实 Q/K。

注意力梯度在 $z_t^{base}=z_t+\Delta z_t^{LF}+\Delta z_t^{tail}$ 上重算, 再投影到 $N_A$。回溯使用 `attention_backtracking_factor=0.5` 和 `attention_backtracking_maximum_steps=8`, 以风险包络最大步长为起点检查初始候选及最多8次减半；只有完整候选分数严格高于原 latent 和优化内容基底时才接受。随后对三分支执行共同缩放时，实际写回组合 latent 还必须严格优于同一共同缩放系数下的实际写回内容基底。安全投影为零、九个候选均失败或5个 Q/K 原子角色 `latent_before`、`optimization_content_base_latent`、`accepted_attention_candidate`、`actual_written_content_base_latent`、`actual_written_combined_latent` 不完整时阻断。优化内容基底和实际写回内容基底分别对应 attention 内部回溯与三分支共同缩放，不得合并。

检测端从待检图像、密钥和公开模型构造 $R_{\mathrm{obs}}\in\mathbb R^{n\times n\times4}$, 只执行一次稳定 token 选择。观测参考系使用该选择产生的 pair 权重, 规范拉回参考系使用同一个 $W_T$ 将单点权重传递后再做外积, 两者共享同一个权重身份摘要。对每个有界相似仿射与方形二面体候选, 四个关系通道分别执行 $\widehat R_{T,c}=W_TR_{\mathrm{obs},c}W_T^\top$；四通道密钥投影分别执行 $\widetilde S_{T,c}=V_T(\pi_cS_K)V_T^\top$。两个方向都先逐通道计算相关, 再使用与嵌入端相同的冻结分量权重组合；注册目标以0.10和0.90组合规范拉回与观测前推总分，再显式扣除规范侧与观测侧的覆盖率、唯一采样率损失。搜索协议只由旋转、log-scale 和归一化位移的公开连续定义域及三分层级分辨率生成, 每轮组合后严格过滤残余旋转、均匀尺度与平移, 保证相对方形二面体基元分别位于 $[-32,32]$、$[1/\sqrt2,\sqrt2]$ 和 $[-0.28,0.28]^2$；协议不读取任何攻击角度、裁剪比例或位移参数。确定性随机 held-out 验证使用远离正式攻击取值的连续变换。identity 变换 $I$ 必须作为每层候选集合中的精确候选实际评分, 目标增益唯一地定义为 $\Delta J(\ell)=J(\widehat T_\ell,\ell)-J(I,\ell)$。输出必须记录四通道分数、相对 identity 候选的观测关系增益、双向关系分数、覆盖惩罚、$J(\widehat T_\ell,\ell)$、$J(I,\ell)$、$\Delta J(\ell)$、恢复变换和权重身份；缺少 identity、三项公式不一致或使用次优候选作为基准时结构注册失败。结构注册预注册 `attention_anchor_count=12`、`attention_residual_threshold=0.20` 和 `attention_minimum_inlier_ratio=0.50`。锚点在抽样 token 索引中确定性均匀选择；实际 token 数少于12时失败。残差是 `normalized_xy_token_centers_corner_endpoints` 坐标中的欧氏距离, 关系采样和图像重采样统一使用 `align_corners=True`；内点率只以具有有效双线性覆盖的锚点为分母, 并要求唯一观测匹配。完整四通道观测与双向分数必须为正、$\Delta J(\ell)>0$、两个方向覆盖率均不低于0.45、内点率不低于0.50且平均内点残差不超过0.20。活动 Q/K、关系图、密钥投影、pair weight 和组合分数必须完整有限, 数值异常不得替换为0或负分。均匀 attention 使 $G=0$ 且其他分量在逐行中心化后也不产生密钥相关, 因而公开坐标不能单独通过门禁。恢复图像参考系后必须重新提取全部冻结层的真实 Q/K, 并使用注册传递后的同一 pair 权重计算同步分数, 不重新选择稳定 token。只允许在注意力几何开启时启用图像对齐, alignment 与 `aligned_content_score` 也必须双向同时存在。几何救回要求 `raw_attention_geometry_score`、`attention_geometry_score`、`registration_confidence`、`attention_sync_score` 与 `aligned_content_score` 五个连续原子同时有限。任一原子缺失或异常时测量失败, 不得改用仅原始内容判定。阈值无关 measurement 配置只保存在顶层运行 manifest；对齐与 measurement 记录保存三项结构常量、连续测量原子及配置摘要, calibration/test 必须绑定同一摘要。measurement 不得包含阈值、窗口或判定。calibration 或 test 不得调节结构常量；registered-key clean negatives 通过互斥 window-fit 与 threshold-freeze 分区派生全部判定参数。几何统计只能决定是否允许重对齐，不能独立给出 positive。冻结层有序集合精确为 `transformer_blocks.0.attn`、`transformer_blocks.23.attn`。每层先独立完成层内搜索；跨层候选依次按注册目标、观测关系分、注册置信度执行字典序最大化, 三者完全同分时选择冻结顺序中更靠前的层。恢复图像固定使用 bilinear、`padding_mode=border`、`align_corners=True`, 连续结果按 `floor(clamp(x, 0, 1) * 255)` 转回 RGB uint8 后才重新编码。

最终图像 attention 归因必须额外执行一次同 seed、同 scheduler、同 LF/tail 配置与算子且只关闭 attention geometry 的 carrier-only 生成。首个注入前 latent 必须具有相同 dtype、shape 与原始字节 SHA-256；两侧更新原子精确覆盖同一注入序列和 scheduler 轨迹。carrier-only 原子的 attention 分数、更新、关系、pair 身份和 attention Null Space 必须为空, `attention_source` 必须为 `disabled_attention_geometry`。该干预测量 attention 开关包含后续 LF、tail 与轨迹交互的总机制效应, 不冻结干预后的 realized carrier, 也不解释为纯直接效应。clean、carrier-only 与完整方法成图都重新执行 VAE 编码、公开固定噪声加噪和直接 Q/K 四分量关系构造。clean 到完整方法、clean 到 carrier-only 及 carrier-only 到完整方法三条 CLIP/手工结构统计边全部通过后, 门禁才同时要求完整方法相对 carrier-only 的自身盲选择分数增益及冻结 carrier-only pair 权重后的配对分数增益严格大于 `minimum_final_image_attention_score_gain=0.0001`。四个 Q/K 依赖分量的配对增益同时写入记录。正式保持记录和 Q/K 记录必须共享直接来源、四分量身份、密钥投影身份、反事实身份、原子 JSONL 路径、文件 SHA-256、内容摘要、持久化图像路径与图像 SHA-256, 并在缓存复用时从实际文件重建。clean 分数只作为总体水印对照；该累计归因门禁不向仅图像检测器提供 clean、carrier-only 图像或生成轨迹。

---

## 八、仅图像检测与完整 fixed-FPR

正式检测严格拆分为

$$
E=\operatorname{Measure}(x',K,M),\qquad
P=\operatorname{Calibrate}(D_{cal}^{-},\alpha),\qquad
\operatorname{Detect}=\operatorname{Apply}(P,E).
$$

`Measure` 只产生阈值无关连续证据, 不保存 calibration 参数、失败原因或判定。

检测图像使用冻结 VAE posterior mode 编码为 $\hat z=(\operatorname{mode}(q_{VAE}(x'))-\texttt{vae\_shift\_factor})\cdot\texttt{vae\_scaling\_factor}$。固定模板只依赖密钥、精确模型标识和 shape, 内容分数固定为 $0.70s_{LF}+0.30s_{tail}$。

完整注入、carrier-only 和全部 measurement 记录对每个活动内容分支必须共享唯一模板内容摘要及载体协议摘要。LF/tail 协议、内容权重和尾部比例进入 measurement 配置摘要；冻结正文引用该摘要并绑定嵌套分区、派生窗口、几何门和最终阈值。test 应用前必须分别重算 measurement 摘要、`threshold_digest` 与 threshold-freeze 假阳性率。

检测密钥计划分为注册水印密钥和预注册 wrong-key 两个角色。wrong-key 使用版本化 SHA-256 domain separation 派生；注入密钥摘要、注册密钥检测模板、wrong-key 独立模板及计划摘要必须在总科学内容记录中交叉一致。

仅图像 Q/K 的公开噪声输出 `shape` 精确等于 $\hat z$ 的 NCHW shape。调用使用 `public_detection_noise_prg_protocol=sha256_counter_normal_icdf_table20_float32`, 且 `key_material=public_detection_noise_domain=public_image_only_qk_detection_noise`；`domain_fields` 精确包含同值 `operator`、冻结 `model_id`、40位 `model_revision`、`width=512`、`height=512`、`inference_steps=20`、`public_detection_schedule_index=7` 和 `latent_shape=shape`。domain 不含水印密钥、Prompt、生成 seed 或轨迹, 并完整执行本节的 SHA-256 大端计数器比特流、连续20位索引提取与 Q20 中点逆 CDF float32 表查询；实际 dtype 转换在 CPU 完成, 随后才搬运到 $\hat z$ 的设备。令 $t_{det}=\operatorname{scheduler.timesteps}[7]$, 检测 latent 只能由 `scheduler.scale_noise(hat_z,t_det,epsilon_det)` 得到；缺失 `scale_noise` 时失败。Transformer 条件固定为 `public_detection_conditioning_protocol=sd3_empty_text_triplet_without_cfg` 与 `public_detection_condition_text=""`。raw 与 aligned 图像分别重新编码和提取 Q/K, 原子角色固定为 `raw_detection_image` 和 `aligned_detection_image`。

calibration source 只接受 registered-key、未攻击的 clean negatives。按版本化 Prompt SHA-256 排序后, 前 $\lfloor n/3\rfloor$ 条为 window-fit, 其余为 threshold-freeze；允许假阳性数统一为 $K(m,\alpha)=\max(0,\lfloor\alpha(m+1)\rfloor-1)$。window-fit 冻结三类几何门、临时 raw 阈值和最宽可行 rescue window。threshold-freeze 以 $e=\max(s^{raw},\min(s^{aligned},s^{raw}-\delta_{low}))$ 在几何可靠时计算判定等价分数, 否则取 raw, 并只由 $e$ 冻结最终阈值。主方法和四个 baseline 共享 threshold-freeze Prompt 身份。rescue 复用同一个内容阈值；几何分数不能独立产生 positive。test 与 detector-guided attack 只消费冻结协议, 不参与参数选择。

当 `attention_geometry_enabled` 或 `image_alignment_enabled` 为 false 时, `geometry_rescue_enabled` 必须同时为 false。该路径以 raw 内容分数直接执行 fixed-FPR 判定, rescue 与几何参数均为 `None`, 相关计数为0, 不生成被关闭机制的替代原子。

三级运行配置分别使用70/700/7000个 Prompt, test 数量为34/340/3400, 目标 FPR 依次为0.1、0.01和0.001。test split 只应用对应层级的冻结协议并报告同一种置信上界。

---

## 九、正式消融与证据需求

正式消融精确包含完整方法和14个真实重运行对照：共享全局风险、完全移除风险路由、移除 Jacobian Null Space、LF-only、Tail-only、移除 LF、移除尾部载体、移除幅值尾部截断、中心化 Q/K logit/可微行内 rank/抽样关系概率/距离调制中心化概率四个逐项留一变体、移除完整 Q/K attention geometry 和移除图像对齐。每个配置都必须重新生成、重新攻击、重新检测并独立冻结 calibration 协议, 不能修改已有分数模拟机制差异。

论文结果还必须包含：

1. clean negative 与 attacked negative 下的完整 fixed-FPR 审计；
2. LF 与尾部截断分数分布、攻击保持率和配对质量；
3. 几何可靠性、对齐增益和 rescue 后整体 FPR；
4. torch-fidelity 0.4.0 `inception-v3-compat` 2048 维特征的 FID/KID；
5. 相同 Prompt、攻击和统计边界下的外部 baseline；
6. 可由 records 和 manifests 重建的 tables、figures、reports 与主张映射。

---

## 十、非核心边界

事件签名、payload probe、证据 manifest 和结果审计属于来源治理或诊断能力，不进入仅图像水印证据判定。SLM-WM 科学方法终止于完整 fixed-FPR 下的 image-only evidence decision。
