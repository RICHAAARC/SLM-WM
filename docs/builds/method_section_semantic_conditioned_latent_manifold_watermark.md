# Method：语义条件潜流形水印方法

本章的公式、参数和失败边界必须逐项服从 `method_semantic_invariants.md`。该权威定义与代码不一致时, 只能修正实现, 不能根据现有行为弱化本章方法。

## 3.1 问题定义

本文研究文本到图像扩散模型中的鲁棒图像水印检测问题。给定文本提示 $p$、扩散生成模型 $G_\theta$、采样更新算子 $S_\theta$、随机种子与初始噪声 $\xi$，无水印生成过程表示为

$$
z_{t-1}=S_\theta(z_t,p,t),\qquad x=D(z_0),
$$

其中，$z_t$ 为第 $t$ 步潜变量，$D$ 表示 VAE 解码器。SLM-WM 在扩散采样过程中施加受约束的潜变量更新：

$$
\widetilde z_t=z_t+\Delta z_t,\qquad
z_{t-1}=S_\theta(\widetilde z_t,p,t).
$$

嵌入目标是在不显著改变语义内容和视觉质量的前提下，提高最终图像在常规失真、几何变换和再扩散攻击后的可检测性。正式检测器遵循仅图像盲检制度：

$$
\widehat y=\operatorname{Detect}(x',K,M),
$$

其中 $x'$ 为待检图像，$K$ 为水印密钥，$M$ 为公开模型配置。检测器不得读取生成 Prompt、初始噪声、源 latent、生成轨迹、未攻击原图或样本级安全子空间。

---

## 3.2 方法总览与术语边界

SLM-WM 先从当前样本构造分支风险场，再通过完整特征 Jacobian 的真实 JVP/VJP 与无阻尼矩阵自由约束投影求解 Null Space，最后把同一潜变量更新分解为三个互补分量：

$$
\Delta z_t
=
\Delta z_t^{\mathrm{LF}}
+
\Delta z_t^{\mathrm{tail}}
+
\Delta z_t^{\mathrm{A}}.
$$

$\Delta z_t^{\mathrm{LF}}$ 是具有明确空间低通构造的内容主证据；$\Delta z_t^{\mathrm{tail}}$ 是高斯幅值尾部截断鲁棒补充证据；$\Delta z_t^{\mathrm{A}}$ 是基于真实 Q/K Self-Attention 相对关系的几何锚点。

高斯幅值尾部截断分支只按高斯模板元素的绝对幅值和展平索引执行稳定排序，并精确保留冻结比例，不执行 FFT、DCT、带通滤波或空间频带选择。该分支不具有空间频率语义。正式分支标识为 `tail_robust`，论文公式使用下标 $\mathrm{tail}$。

“潜流形”在本文中严格指当前 latent 点附近的局部隐式特征水平集切空间解释。令 $\mathcal U_t$ 为 $z_t$ 的局部邻域，并定义

$$
\mathcal L_F(z_t;\mathcal U_t)
=
\{z\in\mathcal U_t:F(z)=F(z_t)\}.
$$

当 $F$ 在该邻域可微且满足局部常秩条件时，水平集在 $z_t$ 处的切空间为 $\ker J(z_t)$。正式实现不验证常秩定理条件，也不构造全局非线性流形、坐标图、测地线或回缩算子；实际计算对象是经分支风险算子约束并通过数值残差门禁的局部安全切空间近似：

$$
\widehat{\mathcal T}_{b,t}
=
\operatorname{span}(N_b),
\qquad
\|J(z_t)N_b[:,j]\|_2\leq\varepsilon_J.
$$

因此，方法名称中的“潜流形”不表示已经求得全局流形。它表示使用当前样本的完整特征局部线性化，把合格更新限制到隐式特征水平集的数值切空间候选。

正式方法采用构造式实现，不求解一个联合标量 `argmax`。对 $b\in\{\mathrm{LF},\mathrm{tail}\}$，先将固定密钥模板投影到各自的局部安全切空间，再执行分支风险硬包络允许的最大全局标量缩放：

$$
\Delta z_t^b
=
\operatorname{RiskBoundedScale}\left(
\operatorname{Norm}\!\left(N_bN_b^\top\nu_b\right),
b_b^{\mathrm{eff}},
\lambda_b\|z_t\|_2
\right).
$$

令 $z_t^{\mathrm{base}}=z_t+\Delta z_t^{\mathrm{LF}}+\Delta z_t^{\mathrm{tail}}$，注意力分支在该内容基底上计算真实 Q/K 目标梯度

$$
g_A
=
\left.\nabla_z\mathcal S_A(z)\right|_{z=z_t^{\mathrm{base}}},
\qquad
\Delta z_t^{\mathrm A}
=
\operatorname{MonotonicBacktrack}\left(
\operatorname{RiskBoundedScale}\left(
\operatorname{Norm}\!\left(N_AN_A^\top g_A\right),
b_A^{\mathrm{eff}},
\lambda_A\|z_t\|_2
\right)
\right).
$$

注意力分支严格使用 `attention_backtracking_factor=0.5` 和 `attention_backtracking_maximum_steps=8`, 从风险允许的最大步长开始检查初始候选及最多8次减半；只有真实 Q/K 分数同时高于原始 latent 和 LF/tail 内容基底时才接受。最终更新为三个分量之和，随后按扩散 latent 的实际 dtype 写回，并重新执行联合风险包络、完整特征 JVP 与有限变化门禁。该分支构造与验收顺序属于项目特定设计；JVP/VJP、无阻尼半正定求解、逐列残差和固定 FPR 校准属于可迁移的通用方法。

### 3.2.1 冻结科学算子身份

正式生成与检测共用 `configs/model_sd35.yaml` 的唯一解析结果。科学算子身份必须逐项包含：

- `model_id=stabilityai/stable-diffusion-3.5-medium` 与 `model_revision=b940f670f0eda2d07fbb75229e779da1ad11eb80`；
- `vision_model_id=openai/clip-vit-base-patch32` 与 `vision_model_revision=3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`；
- `pipeline_class_name=diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline`；
- `vae_class_name=diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL`；
- `transformer_class_name=diffusers.models.transformers.transformer_sd3.SD3Transformer2DModel`；
- `scheduler_class_name=diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler`；
- `vae_scaling_factor=1.5305`、`vae_shift_factor=0.0609`、`latent_torch_dtype=float16`、`vision_torch_dtype=float32`；
- `public_detection_schedule_index=7`、冻结 attention 层名与统一坐标约定。

将解析 dataclass 的完整 `asdict` 结果与 `formal_method_config_schema=slm_wm_formal_method_runtime_config_v2` 组成规范 payload, 使用 UTF-8、`ensure_ascii=false`、键名升序和无空白分隔符 `(',', ':')` 序列化后执行 SHA-256, 得到 `formal_method_config_digest`。该摘要不依赖 YAML 排版或仓库绝对路径。运行原子必须同时记录上述配置值、运行时实际类名、VAE 实际 scale/shift、实际 dtype、检测索引和 `formal_method_config_digest`；任一不一致都使科学单元失败。

---

## 3.3 分支语义风险场

### 3.3.1 内容属性建模

对潜空间位置 $u$ 定义内容属性向量：

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

$\phi_{\mathrm{sem}}$ 是冻结 CLIP patch token 与 CLS token 的余弦一致性经 $(c+1)/2$ 映射，$\phi_{\mathrm{tex}}$ 是解码灰度图水平与垂直绝对梯度和除以2，$\phi_{\mathrm{lcr}}$ 是灰度相对反射填充5x5局部均值的绝对偏离。三者经双线性插值映射到 latent 网格后只执行公开 $[0,1]$ 截断, 不在单个样本内重新定标。$\phi_{\mathrm{adj}}$ 直接比较当前 latent 与紧邻上一 scheduler 步 latent 的解码 RGB 图像：

$$
\phi_{\mathrm{adj}}(u)
=
1-\frac{1}{3}\sum_{c=1}^{3}
\left|x_t(c,u)-x_{t-1}(c,u)\right|.
$$

注入回调在每个非注入步保存真实 post-step latent，并在注入后保存实际写回 latent，因而 $x_{t-1}$ 总是紧邻上一 scheduler 步状态。首个注入必须晚于第0步；缺失上一时刻时运行失败。$\phi_{\mathrm{attn\_stab}}$ 必须由不少于两个冻结层的真实 Q/K 关系独立计算，不能以 $\phi_{\mathrm{adj}}$、局部对比度、纹理图或常数替代。

### 3.3.2 分支风险与硬资格集合

三个分支对纹理和注意力稳定性的偏好不同，因而不能复用一个风险标量。对分支 $b\in\{\mathrm{LF},\mathrm{tail},\mathrm{A}\}$ 定义

$$
\rho_b(u)=
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

LF 使用 $\psi_{\mathrm{LF}}(q)=q$，从而降低高纹理位置的优先级；尾部截断分支使用 $\psi_{\mathrm{tail}}(q)=1-q$，从而偏好稳定纹理位置；注意力几何分支使用 `risk_neutral_texture_value=0.5` 的中性纹理基线, 不随纹理升降改变风险方向并主要受注意力稳定性控制。三分支参数分别读取 `lf_content_risk_config.*`、`tail_robust_risk_config.*` 与 `attention_geometry_risk_config.*`, 例如 LF 局部对比权重精确读取 `lf_content_risk_config.local_contrast_risk_weight`, 不读取 dataclass 隐式默认值。承载预算为

$$
b_b(u)=\operatorname{clip}
\left(b_{\min}+\kappa_b(1-\rho_b(u)),b_{\min},b_{\max}\right).
$$

每个分支通过独立阈值得到资格集合

$$
\Omega_b=\{u:\rho_b(u)<\tau_b\}.
$$

当 $u\notin\Omega_b$ 时，正式候选矩阵中的对应预算被置为 0；只有资格集合内部保留连续预算。该硬门控确保风险场会真实改变后续子空间，而不是仅作为日志字段。完整方法仅对当前活动且启用语义风险路由的分支执行空资格集合 fail-closed 门禁；移除风险路由的正式消融不使用风险阈值决定样本是否继续运行。每次注入保存三个分支完整风险值、连续预算和资格 mask 的 Tensor 内容 SHA-256, 联合摘要必须与三个逐分支记录一致。

风险输入采用冻结解析范围而不是逐样本 min-max：CLIP patch cosine 通过 $(c+1)/2$ 映射，灰度双向梯度和除以2，5x5局部对比与相邻步 RGB 差异使用 `risk_image_signal_interpolation_mode=bilinear`、`risk_image_signal_align_corners=false`, 直接 Q/K stability 使用 `risk_attention_signal_interpolation_mode=bilinear`、`risk_attention_signal_align_corners=true`。有效预算 $b_b^{\mathrm{eff}}=b_b\mathbf1[u\in\Omega_b]$ 除以冻结的 $b_{\max}^b$ 得到绝对预算比例, 不能除以当前样本的预算最大值。对单位安全方向 $v_b$ 和名义强度 $a_b=\lambda_b\|z_t\|_2$, 逐位置硬包络为

$$
E_b(u)=a_b\|v_b\|_\infty
\frac{b_b^{\mathrm{eff}}(u)}{b_{\max}^b}.
$$

实际分支更新只允许沿 $v_b$ 采用满足 $|\Delta z_t^b(u)|\le E_b(u)$ 的最大可行全局标量。零预算方向泄漏门禁精确使用 `risk_bounded_scale_direction_epsilon=1e-12`。该缩放保持 Null Space 方向, 并保证预算整体收缩不会增加实际写回强度。

---

## 3.4 语义条件 Jacobian Null Space

### 3.4.1 完整特征 Null Space 定义

令完整特征函数为

$$
F(z_t)=\left[F_{\mathrm{sem}}(z_t);F_{\mathrm{struct}}(z_t)\right],
\qquad J(z_t)=\frac{\partial F}{\partial z_t}.
$$

SLM-WM 将安全方向定义为完整特征的一阶零响应方向：

$$
\mathcal{N}_{\mathrm{sem}}(z_t)
=
\left\{v:J(z_t)v=0\right\}.
$$

$F_{\mathrm{sem}}$ 对应 VAE 可微解码与冻结 CLIP 图像编码器输出的512维归一化完整 embedding。$F_{\mathrm{struct}}$ 是范围明确的204维手工结构统计向量：3维通道均值、3维通道标准差、3维水平绝对梯度均值、3维垂直绝对梯度均值和192维8x8 RGB 平均池化。正式输出宽度为716，不执行降维投影。$F_{\mathrm{struct}}$ 只约束这些显式坐标，不单独代表一般感知质量；正式 FID/KID 与配对图像质量指标提供独立结果证据。

### 3.4.2 精确 JVP/VJP 与种子方向

对 latent 方向 $v$ 和特征余切 $y$ 计算

$$
\operatorname{JVP}(v)=Jv,
\qquad
\operatorname{VJP}(y)=J^\top y.
$$

正式实现优先通过 `torch.func.linearize` 与 `torch.func.vjp` 在同一 latent 点复用精确线性算子。若底层等价算子不支持该接口，则使用 `torch.autograd.functional.jvp/vjp` 重算。两条路径都属于精确自动微分；有限差分或轨迹线性近似只能用于诊断。

分支种子矩阵为

$$
D_b=[d_1^b,\ldots,d_m^b]\in\mathbb{R}^{n\times m},
\qquad D_b^\top D_b=I.
$$

LF 与尾部截断分支分别把固定盲检模板作为首个方向；注意力分支把真实 Q/K 目标梯度作为首个方向；其余列由设备无关密钥 PRG 方向补齐。候选方向的 domain 绑定算子角色、分支名称、latent shape 和矩阵 shape, 并使用与内容模板相同的版本化 SHA-256 计数器原语。分支资格 mask 与连续风险预算构成非负对角算子 $B_b$，并显式进入约束系统。

### 3.4.3 无阻尼约束投影与门禁

对每个种子方向，无阻尼 PSD-CG 求解

$$
(JB_b^2J^\top)y_i=JB_bd_i^b.
$$

得到风险支持的约束投影方向

$$
u_i^b=B_bd_i^b-B_b^2J^\top y_i,
\qquad Ju_i^b=0.
$$

收集4个独立合格方向后执行

$$
N_b=\operatorname{qr}([u_1^b,\ldots,u_4^b]).
$$

正式记录必须保存完整特征 schema、每列 CG 迭代数和收敛残差、约束投影平方能量、QR 后逐列完整 Jacobian 响应与正交误差。设接受的路由候选矩阵为 $V_b$, 且投影方向矩阵满足 float32 reduced QR $U_b=N_bR_b$。基底列使用 `null_space_numerical_epsilon=1e-12` 规范符号；$R_b$ 对角近零或二范数条件数超过 `maximum_qr_condition_number=1000000.0` 时失败。独立参考严格使用 `qr_reference_solve_protocol=right_upper_triangular_solve_without_explicit_inverse_v1`, 通过 $\widetilde V_bR_b=V_b$ 求解；第 $j$ 列相对残差分母使用 `null_space_numerical_epsilon`, 不能使用跨列共享 RMS。CG 阻尼固定为0，最多64次迭代且相对收敛阈值为 $10^{-6}$；QR 后每列相对响应不得超过 $10^{-4}$，能量保留比例不得低于0.01，正交误差不得超过 `maximum_orthogonality_error=0.00001`。任一条件失败时直接停止运行。

三个分支在 float32 中按 LF、tail、attention 固定顺序求和。不得逐分支 cast。对共同缩放 $q_k=2^{-k}$、$k=0,\ldots,24$, original latent 的 float32 表示只与共同缩放后的三分支和相加一次, 随后单次 cast 到扩散 latent 的真实存储 dtype：

$$
\Delta z_t^{\mathrm{written}}
=
\operatorname{cast}_{\operatorname{dtype}(z_t)}
\left(z_t^{(32)}+q_k\Delta z_t^{(32)}\right)-z_t.
$$

运行时必须对该实际量化增量重新执行完整特征精确 JVP，并计算

$$
r_{\mathrm{written}}
=
\frac{\left\|J_t\Delta z_t^{\mathrm{written}}\right\|_2}
{\max\!\left(\left\|F(z_t)\right\|_2,\epsilon\right)}.
$$

当前冻结阈值要求 $r_{\mathrm{written}}\leq10^{-4}$。实际写回还必须严格满足共同缩放后的逐元素联合包络；零包络坐标要求写回严格为0, 正包络坐标的最大比值不得超过1, 不添加 ULP 容差。attention 启用时, 写回完整候选必须高于原 latent 和相同 $q_k$ 下单次 cast 的 LF/tail 内容基底 Q/K 分数。共同回溯精确使用 `quantized_budget_envelope_backtracking_factor=0.5` 与 `quantized_budget_envelope_backtracking_maximum_steps=24`。选择首个同时通过包络、非零写回、Q/K 单调性、JVP 和有限变化的候选；初始候选及最多24次共同减半均失败时停止运行。每个分支分别绑定 $D_b$、$B_b$、$J(B_bD_b)$、投影方向 $U_b$、投影响应 $JU_b$、QR 基底 $N_b$、基底响应 $JN_b$ 和独立参考响应 $J\widetilde V_b$ 的角色限定内容 SHA-256。两级有限变化门禁均要求 CLIP cosine 不低于0.995且手工结构统计特征相对漂移不高于0.02。

---

## 3.5 LF 与高斯幅值尾部截断内容载体

### 3.5.1 空间低通 LF 主证据

给定密钥 $K_{\mathrm{LF}}$、公开模型标识和 latent 形状，生成固定 Q20 中点逆 CDF 量化标准正态模板并执行空间平均池化：

$$
\nu_{\mathrm{LF}}
=
\operatorname{Norm}
\left(
\operatorname{AvgPool}_{k\times k}
(\operatorname{PRG}_{\mathcal N}(K_{\mathrm{LF}},M,shape))
\right).
$$

其中，$\operatorname{PRG}_{\mathcal N}$ 固定为 `sha256_counter_normal_icdf_table20_float32_v2`。其 domain payload 精确包含 `keyed_prg_version`、`key_material`、`domain_fields` 和 `shape`；payload 通过 UTF-8、`ensure_ascii=false`、键名升序及无空白分隔符 `(',', ':')` 形成 stable JSON, 再执行 SHA-256 得到32字节 domain digest。计数器从0开始, 使用16字节无符号大端编码并拼接在 domain digest 后再执行 SHA-256。连续32字节输出块形成 MSB-first 大端比特流, 高斯路径跨块连续提取20位索引 $i$ 并查询

$$
q_i=\operatorname{round}_{\mathrm{binary32}}\!\left(\Phi^{-1}\!\left(\frac{i+0.5}{2^{20}}\right)\right),
\qquad 0\le i<2^{20}.
$$

冻结表全部1048576个 binary32 位模式的完整大端字节 SHA-256 为 `70abf440a7f3670147965ffa52f5aaa639dab97f6282b68f3a9a1b1ce5e6cf5a`。该输出是有限离散的 Q20 中点逆 CDF 量化标准正态, 不是连续精确的 $\mathcal N(0,1)$；理想中点 KS 距离为 $2^{-21}$, 计入 float32 舍入误差后的登记上界为 `4.912236096776823e-7`。规范 float32 生成与目标 dtype 转换均在 CPU 完成, 随后才搬运到执行设备。独立 uniform domain 仍把每个 SHA-256 块按 offset 0、8、16、24 划分为8字节无符号大端 word, 取高53位 $m=w\gg11$ 并按 $u=(m+1)/(2^{53}+2)$ 映射到严格开区间 $(0,1)$；该路径只用于按0.5阈值生成 attention 关系符号。CPU 或 CUDA 设备 RNG 均不参与方法身份。MPFR 逐项舍入复验属于外层审计证据, 不进入 PRG 算法摘要。当前固定向量只在 Windows CPU 实测, Linux/Colab 逐字节一致性必须通过 GPU 运行前门禁。

LF 平均池化只作用于 height/width 二维轴, 对每个 batch/channel 独立使用 kernel 5、stride 1、padding 2、二维零填充和 `count_include_pad=true`, 不跨 batch 或 channel 轴传播数值；池化后去均值与 L2 归一化各自在整个模板 Tensor 上计算一个全局标量。不允许由运行库默认 padding 改写边界条件。

平均池化在 latent 的二维空间轴上抑制快速空间变化，因此 LF 分支具有明确的空间低通定义。嵌入端把固定模板投影到分支安全子空间：

$$
\overline\nu_{\mathrm{LF}}
=N_{\mathrm{LF}}N_{\mathrm{LF}}^\top\nu_{\mathrm{LF}},
$$

令单位安全方向为

$$
v_{\mathrm{LF}}
=
\frac{\overline\nu_{\mathrm{LF}}}{\|\overline\nu_{\mathrm{LF}}\|_2}.
$$

LF 更新由3.3.2节的逐位置硬包络确定最大可行全局标量：

$$
\Delta z_t^{\mathrm{LF}}
=
\operatorname{RiskBoundedScale}\left(
v_{\mathrm{LF}},
b_{\mathrm{LF}}^{\mathrm{eff}},
\lambda_{\mathrm{LF}}\|z_t\|_2
\right).
$$

### 3.5.2 高斯幅值尾部截断补充证据

给定密钥 $K_{\mathrm{tail}}$、公开模型标识和 latent 形状，生成固定 Q20 中点逆 CDF 量化标准正态模板：

$$
\nu_{\mathrm{tail}}
=
\operatorname{PRG}_{\mathcal N}(K_{\mathrm{tail}},M,shape).
$$

设模板包含 $n$ 个元素。对冻结的尾部比例 $\gamma$，按绝对幅值降序、展平索引升序进行稳定排序，并取

$$
I_\gamma
=
\operatorname{TopK}_{\lceil n\gamma\rceil}
\left(\left\{(|\nu_{\mathrm{tail},i}|,-i)\right\}_{i=1}^{n}\right),
$$

$$
\widetilde\nu_{\mathrm{tail},i}
=
\nu_{\mathrm{tail},i}
\mathbb{I}\left(i\in I_\gamma\right).
$$

因此保留元素数精确等于 $\lceil n\gamma\rceil$；同幅值元素由公开的展平索引规则消除歧义。非入选元素保持精确0, 截断模板只除以整体二范数, 不执行会使非入选位置重新非零的去均值。记录中的 `tail_threshold` 是 $I_\gamma$ 内最小绝对幅值，不是由设备算子重新估计的随机分位点。

嵌入更新为

$$
\Delta z_t^{\mathrm{tail}}
=
\operatorname{RiskBoundedScale}\left(
\operatorname{Norm}\left(
N_{\mathrm{tail}}N_{\mathrm{tail}}^\top
\widetilde\nu_{\mathrm{tail}}
\right),
b_{\mathrm{tail}}^{\mathrm{eff}},
\lambda_{\mathrm{tail}}\|z_t\|_2
\right).
$$

该算子定义的是**概率分布幅值域的稀疏尾部选择**。元素索引没有按空间频率排序，保留集合也不是 Fourier 或余弦基上的频带。截断后的模板可能包含宽频谱成分，因此不能把“幅值大”解释为“空间频率高”。其鲁棒性是需要通过压缩、噪声、重采样和再扩散实验检验的假设，不能由“高频”名称先验推出。

### 3.5.3 仅图像内容分数

检测端从待检图像通过冻结 VAE posterior mode 得到 $\widehat z=(\operatorname{mode}(q_{\mathrm{VAE}}(x'))-\texttt{vae\_shift\_factor})\cdot\texttt{vae\_scaling\_factor}$，并使用密钥、精确模型标识和公开 shape 重建两个固定模板：

$$
s_{\mathrm{LF}}=\operatorname{Corr}(\widehat z,\nu_{\mathrm{LF}}),
\qquad
s_{\mathrm{tail}}=\operatorname{Corr}(\widehat z,\widetilde\nu_{\mathrm{tail}}).
$$

统一内容分数为

$$
s_c
=
0.70s_{\mathrm{LF}}+0.30s_{\mathrm{tail}}.
$$

两个分支不分别设置独立正判阈值后投票。安全子空间只在嵌入端控制失真；检测端不恢复样本级 $N_{\mathrm{LF}}$ 或 $N_{\mathrm{tail}}$。

---

## 3.6 Self-Attention 相对关系几何锚点

### 3.6.1 真实 Q/K 关系图

正式层集合精确固定为 `transformer_blocks.0.attn` 与 `transformer_blocks.23.attn`。对冻结二维图像 token 抽样集合 $\mathcal I$，正式算子直接调用每个注意力头的 `to_q` 与 `to_k` 投影, 不按模块枚举位置动态选择层。令 $\ell_{ij}^{(h)}=q_i^{(h)\top}k_j^{(h)}/\sqrt{d_h}$,
四分量关系边为

$$
r_{ij}=[L_{ij},\rho_{ij},P_{ij},G_{ij}],
$$

其中 $L$ 是各头 logits 移除逐行均值后的平均，$P$ 是各头在 $\mathcal I$ 上
row-softmax 概率的平均。公开坐标采用 `normalized_xy_token_centers_corner_endpoints_v1`, 角点 token 中心分别落在 -1 与 1, token 插值与图像仿射重采样统一使用 `align_corners=True`。$D_{ij}=\|p_i-p_j\|_2/(2\sqrt2)$ 来自该公开二维索引，
第4分量为

$$
G_{ij}=\left(P_{ij}-\frac1{|\mathcal I|}\sum_kP_{ik}\right)
\left(D_{ij}-\frac1{|\mathcal I|}\sum_kD_{ik}\right),
$$

可微降序 rank 为

$$
\rho_{ij}=\frac1{|\mathcal I|}\left[1+\sum_{k\ne j}
\sigma\left(\frac{L_{ik}-L_{ij}}{0.25}\right)\right].
$$

$P$ 是项目定义的抽样图像 token 关系概率；SD3.5 joint attention 还包含文本
token 与未抽样图像 token, 因此该量不是模块完整 attention 权重。正式来源
同时保存直接 Q/K 计算的 $L$ 与 $P$。多头 $P$ 先逐头 softmax 再平均, 一般
不等于平均 logits 的 softmax。每层必须记录模块名、头数、head width、scale、
Q/K 归一化和抽样网格, 且模块 scale 必须等于 $1/\sqrt{d_h}$。

对 batch 样本 $b$、冻结层 $\ell$ 和 token $i$, 先定义中心化单位概率行

$$
\widetilde P_{bi:}^{\ell}=P_{bi:}^{\ell}-\frac1n\mathbf1,
\qquad
\overline P_{bi:}^{\ell}=\frac{\widetilde P_{bi:}^{\ell}}
{\max(\|\widetilde P_{bi:}^{\ell}\|_2,\epsilon_{\mathrm{num}})}.
$$

跨层一致性、incoming centrality 和排序分数精确为

$$
\operatorname{Stab}_i=
\operatorname{clip}_{[0,1]}\left(
\operatorname{Mean}_{b,\ell<r}
\frac{1+\langle\overline P_{bi:}^{\ell},\overline P_{bi:}^{r}\rangle}{2}
\right),
$$

$$
c_i=\operatorname{Mean}_{\ell,b,j}P_{bji}^{\ell},
\qquad
\operatorname{Cent}_i=\frac{c_i}{\max(\sum_r c_r,\epsilon_{\mathrm{num}})},
\qquad
q_i=\operatorname{Stab}_i\operatorname{Cent}_i.
$$

选择数量为 $k=\min(n,\max(4,\lceil0.5n\rceil))$。按 $(-q_i,\operatorname{token\_index}_i)$ 升序稳定排序选择前 $k$ 个, 再把选中位置升序写入记录。单点与非对角 pair 权重为

$$
a_i=\begin{cases}1,&i\in\mathcal V_A,\\0.25,&i\notin\mathcal V_A,\end{cases}
\qquad
w_{ij}=a_i a_j\mathbf1[i\ne j].
$$

选择摘要、原始二维索引、权重值和外积规则形成 `stable_pair_weight_identity_digest`。一次注入的梯度、内容基底复算、回溯和最终写回复验冻结同一个 $(\mathcal V_A,a,w)$ 对象。

对每个冻结层 $\ell$, PRG 使用 `operator=attention_relation_signs`、精确层名和 token 数构造独立 uniform Tensor $U^\ell\in(0,1)^{n\times n}$。密钥对称符号图为

$$
S_{K,ij}^{\ell}=\begin{cases}
0,&i=j,\\
1,&i<j\ \land\ U_{ij}^{\ell}\ge0.5,\\
-1,&i<j\ \land\ U_{ij}^{\ell}<0.5,\\
S_{K,ji}^{\ell},&i>j.
\end{cases}
$$

四分量 polarity 固定为 $\pi=(1,-1,1,1)$, 目标图为 $T_{ijc}^{\ell}=\pi_cS_{K,ij}^{\ell}$。对 batch $b$、层 $\ell$、row $i$ 和分量 $c$, 令 $W_i=\sum_jw_{ij}$, 并定义

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

$W_i=0$ 或任一加权中心化能量不大于 $\epsilon_{\mathrm{num}}^2$ 时该 row 无效。每个 batch 先对有效 row 取均值；没有有效 row 时该 batch 分量分数为0。随后对 batch 和冻结层等权聚合：

$$
C_c^{\ell}=\frac1B\sum_b
\operatorname{Mean}_{i\in\mathcal I_{b\ell c}^{valid}}
\operatorname{RowCorr}_{bic}^{\ell},
\qquad
C_c=\frac1{|\mathcal L|}\sum_{\ell\in\mathcal L}C_c^{\ell},
$$

$$
s_A(z;K)=\sum_{c=1}^{4}\omega_cC_c(z;K),
\qquad
\omega=(1/4,1/4,1/4,1/4),
\qquad
g_A=\nabla_{z_t}s_A.
$$

$G$ 通过中心化 $P$ 对 latent 保留非零梯度, 且在均匀 attention 下严格为0。概率和距离都逐行中心化, 从而移除距离行均值乘概率偏离所产生的第3通道一阶重复。四个分量均依赖真实 Q/K, 公开距离只作为 $G$ 的冻结调制因子。

每个冻结层同时记录抽样后真实 $Q$、$K$、中心化 $L$、逐头 softmax 后平均的 $P$ 与原始二维 token 索引的版本化 Tensor 内容 SHA-256。一次注入按原 latent、内容基底、接受候选和实际量化写回 latent 四个角色持久化原子；最终成图按 clean、carrier-only 和完整方法三个角色持久化原子；仅图像盲检按原图和对齐图两个角色持久化原子。角色顺序、层顺序、逐层自摘要和联合摘要均须一致。

完整方法使用 $\omega=(1/4,1/4,1/4,1/4)$ 组合四个分量。正式留一变体对每个分量分别设置一次 $\omega_c=0$, 其余三个权重均为 $1/3$。该留一操作只移除目标关系通道对组合目标的直接贡献, 用于估计其在其他通道存在时的边际必要性。同一权重协议用于生成端目标、回溯、实际写回、最终图像归因、检测端原分数、注册目标与恢复后同步分数, 并由活动分量集合和协议摘要证明生成与检测没有使用不同组合。

### 3.6.2 安全投影与单调回溯

注意力种子方向进入其独立分支的 Jacobian Null Space 求解器，得到 $N_{\mathrm A}$。投影梯度为

$$
\overline g_A=N_{\mathrm A}N_{\mathrm A}^\top g_A.
$$

令 $v_A=\operatorname{Norm}(\overline g_A)$。注意力分支从 $\operatorname{RiskBoundedScale}(v_A,b_A^{\mathrm{eff}},\lambda_A\|z_t\|_2)$ 给出的最大风险可行步长开始回溯, 后续候选只能继续减小强度。仅接受同时满足注意力分支包络、三分支实际 dtype 联合包络且使四分量目标分数相对原 latent 和冻结内容基底都严格上升的步长：

$$
s_A(z_t^{\mathrm{base}}+\Delta z_t^{\mathrm A})>
\max\{s_A(z_t),s_A(z_t^{\mathrm{base}})\}.
$$

若所有候选步长都不能提高分数，运行必须报告失败，不能写入未验证的 attention 更新。

### 3.6.3 几何恢复

检测端只从待检图像、密钥和公开模型构造
$R_{\mathrm{obs}}\in\mathbb R^{n\times n\times4}$。对冻结的有界相似仿射与
方形二面体候选 $T$，构造双线性矩阵 $W_T$ 并对四通道分别执行规范拉回：

$$
\widehat R_{T,c}=W_T R_{\mathrm{obs},c}W_T^\top,
\qquad c=1,\ldots,4.
$$

同时使用 $T^{-1}$ 构造矩阵 $V_T$, 将四通道密钥投影前推到观测参考系：

$$
\widetilde S_{T,c}=V_T(\pi_cS_K)V_T^\top.
$$

规范拉回与观测前推都先计算四个逐行加权相关分数, 再使用与嵌入端相同的冻结分量权重组合。候选目标为

$$
J(T)=0.10s_{\mathrm{can}}+0.90s_{\mathrm{obs}}
-0.01\sum_{q\in\{c_{\mathrm{can}},u_{\mathrm{can}},c_{\mathrm{obs}},u_{\mathrm{obs}}\}}(1-q).
$$

$c$ 和 $u$ 分别表示有效坐标覆盖率与唯一采样率。观测前推项使用完整观测关系解释候选变换，防止只保留中心子图的尺度候选通过较小有效区域获得更高目标。检测端只在原始观测 Q/K 上执行一次稳定 token 选择。观测项使用 $w_{\mathrm{obs}}$, 规范项使用 $a_{\mathrm{can},T}=W_Ta_{\mathrm{obs}}$ 后重新外积得到的 $w_{\mathrm{can},T}$, 两者共享权重身份摘要。检测器从与攻击注册表无关的旋转、log-scale 和归一化位移连续定义域生成粗锚点，并按固定三分比例执行三层局部细化。每轮局部组合都相对方形二面体基元执行严格过滤, 保证残余旋转不超过32°、尺度始终位于 $[1/\sqrt2,\sqrt2]$、两个平移分量绝对值不超过0.28；搜索器不读取攻击角度、裁剪比例或位移参数，验证集使用远离正式攻击取值的确定性随机连续变换。输出包括四通道分数、相对 identity 候选的观测关系增益、双向关系分数、覆盖惩罚、目标间隔、注册置信度、有效锚点内点比例、对齐残差和权重身份。结构门禁预注册锚点数12、归一化 xy 欧氏残差上界0.20和最小内点率0.50。锚点按抽样 token 索引确定性均匀选择, token 数少于12时失败；内点率以具有有效双线性覆盖的锚点为分母, 并要求唯一观测匹配。坐标固定为 `normalized_xy_token_centers_corner_endpoints_v1`, token 采样与图像重采样统一使用 `align_corners=True`。结构注册要求完整四通道观测与双向分数为正、目标间隔为正、两个方向覆盖率均不低于0.45，并通过上述固定内点率和残差门禁。三项常量必须进入方法配置、对齐、检测、冻结阈值与结果摘要, calibration 和 test 不得调节。由于均匀 attention 使 $G=0$ 且其余通道在逐行中心化后也为0, 公开坐标不能单独形成可靠注册。得到 $\widehat T$ 后，检测器重采样待检图像并重新提取全部冻结层的真实 Q/K；恢复后同步分数使用传递后的 $w_{\mathrm{can},\widehat T}$, 不重新选择稳定 token。权重身份一致且同步分数通过 calibration split 冻结阈值后, 才允许进入同阈值内容救回。几何链只负责参考系恢复和救回资格门禁，不独立产生 positive 判定。

### 3.6.4 最终成图注意力可观测性

最终图像 attention 归因使用三路同 seed 生成：clean、完整方法, 以及保持同一 scheduler、LF/tail 配置与算子且只关闭 attention geometry 的 carrier-only 反事实。完整方法与 carrier-only 首个注入前 latent 必须以 dtype、shape 和全部连续原始字节 SHA-256 证明相同；两侧更新数、顺序和完整 scheduler 轨迹必须一致。carrier-only 每条更新原子必须明确 `attention_source=disabled_attention_geometry`, attention 分数与更新为空, 关系和 pair 身份为空, 直接 Q/K 来源为 false, 且 Null Space 记录不含 attention 分支。原子 JSONL 的路径、实际文件 SHA-256 和解析内容摘要绑定到结果、manifest 与缓存复验。该干预估计 attention 开关经后续 LF、tail 和状态轨迹交互传播的总机制效应；不要求干预后两侧 realized LF/tail 更新相等, 也不声称纯直接效应。三张成图都必须通过 VAE 编码、公开固定噪声加噪和同一冻结 Transformer 前向重新构造直接 Q/K 四分量关系图。公开固定噪声只能由冻结 scheduler 的 `scale_noise` 在正式检测 timestep 上施加；缺少该算子的 scheduler 不属于当前方法协议，运行必须失败。归因计算前, clean 到完整方法、clean 到 carrier-only 及 carrier-only 到完整方法三条边的完整 CLIP 语义余弦相似度和手工结构统计特征相对漂移必须同时通过冻结阈值。运行随后计算两类归因增益：完整方法与 carrier-only 分别执行自身稳定 token 盲选后的分数差, 以及冻结 carrier-only pair 权重后同时评价两张图的分数差。两类总分增益均须严格大于 `minimum_final_image_attention_score_gain`, 当前冻结值为0.0001；四个 Q/K 依赖分量的配对增益同时写入记录。正式保持记录与 Q/K 记录必须绑定直接 Q/K 来源、四分量身份、密钥投影身份、反事实配置、更新原子、scheduler 轨迹、持久化图像路径、图像 SHA-256 及 CUDA 执行证据。反事实缺失、原子残留 attention、首 latent 或调度漂移、三边保持失败、产物身份不一致、分数非有限或任一总分增益未通过都会终止当前科学单元, 中间 latent attention 分数不能替代该门禁。clean 分数保留为总体水印对照, 不作为 attention 因果归因基线, 也不进入仅图像检测函数。

---

## 3.7 完整 fixed-FPR 判定

仅图像 Q/K 提取先对待检图像执行 VAE posterior mode 编码, 得到 $\hat z$ 与精确 NCHW `shape=tuple(int(v) for v in hat_z.shape)`。公开噪声调用使用 `public_detection_noise_prg_protocol=sha256_counter_normal_icdf_table20_float32_v2`, 并固定 `key_material=public_detection_noise_domain=public_image_only_qk_detection_noise_v1`。`domain_fields` 精确包含同值 `operator`、冻结 `model_id`、40位 `model_revision`、`width=512`、`height=512`、`inference_steps=20`、`public_detection_schedule_index=7` 和 `latent_shape=shape`；它不含水印密钥、Prompt、生成 seed 或轨迹。该 payload 完整执行 SHA-256、从0开始的16字节大端计数器、MSB-first 连续20位索引提取和 Q20 中点逆 CDF float32 表查询, 在 CPU 完成实际 dtype 转换后再搬运到 $\hat z$ 的设备。令 $t_{det}=\operatorname{scheduler.timesteps}[7]$, 冻结 scheduler 必须通过 `scale_noise(hat_z,t_det,epsilon_det)` 加噪, 不允许线性混合替代。Transformer 条件固定为 `public_detection_conditioning_protocol=sd3_empty_text_triplet_without_cfg_v1` 和 `public_detection_condition_text=""`。raw 待检图和 aligned 图分别重新执行 VAE posterior mode 编码与 Q/K 提取, 对应原子角色为 `raw_detection_image` 和 `aligned_detection_image`；aligned 路径不能复制 raw 摘要。

### 3.7.1 内容主判

给定 calibration split 中的 clean negative 分数分布和目标误报率 $\alpha$，内容阈值为

$$
\tau_c=Q_{1-\alpha}(s_c\mid\mathcal D_0^{\mathrm{cal}}).
$$

原始内容余量与正判为

$$
m_c^{\mathrm{raw}}=s_c^{\mathrm{raw}}-\tau_c,
\qquad
positive\_by\_content=\mathbb I(m_c^{\mathrm{raw}}\ge0).
$$

### 3.7.2 同阈值几何救回

只有原始内容分数位于冻结的边界失败窗口，且失败原因和几何可靠性满足冻结条件时，才允许对齐：

$$
rescue\_eligible
=
(\delta_{\mathrm{low}}\le m_c^{\mathrm{raw}}<0)
\land geometry\_reliable
\land fail\_reason\in\{geometry\_suspected,low\_confidence\}.
$$

对齐后重新从图像计算 $s_c^{\mathrm{align}}$，并复用同一个 $\tau_c$：

$$
rescue\_applied
=
rescue\_eligible
\land
\mathbb I(s_c^{\mathrm{align}}-\tau_c\ge0).
$$

最终水印证据判定为

$$
y_{\mathrm{evidence}}
=
positive\_by\_content
\lor
rescue\_applied.
$$

### 3.7.3 完整协议冻结

fixed-FPR 约束的是包含 rescue 的完整 evidence 判定，而不是只冻结内容阈值后任意增加第二条判定路径。calibration clean negatives 只冻结内容分数阈值、几何关系分阈值、注册置信度阈值、恢复后同步分阈值和 rescue window；锚点数12、归一化 xy 残差上界0.20、最小内点率0.50和失败原因映射属于预注册确定性协议。test split 只能应用冻结协议并报告 clean negative 的二项分布置信上界，不参与任何参数选择。

当前三类运行配置为：

| Prompt 数量 | dev | calibration | test | 目标 FPR |
| ---: | ---: | ---: | ---: | ---: |
| 70 | 3 | 33 | 34 | 0.1 |
| 700 | 30 | 330 | 340 | 0.01 |
| 7000 | 300 | 3300 | 3400 | 0.001 |

---

## 3.8 正式机制消融

正式消融必须对改变后的机制配置重新生成图像、重新执行攻击并重新运行仅图像检测。禁止通过修改完整方法已有分数或保留率模拟机制被移除后的结果。正式集合精确包含完整方法和以下14个变体：

1. 共享全局风险路由；
2. 完全移除分支风险路由；
3. 移除 Jacobian Null Space；
4. LF-only；
5. Tail-only；
6. 移除 LF 内容载体；
7. 移除尾部鲁棒载体；
8. 移除高斯幅值尾部截断；
9. 移除中心化 Q/K logit 分量；
10. 移除可微行内 rank 分量；
11. 移除抽样图像 token 关系概率分量；
12. 移除距离调制中心化关系概率分量；
13. 移除完整 Q/K attention geometry；
14. 移除同阈值图像对齐。

四个分量变体均将目标分量权重置零并将其余三个权重归一化为 $1/3$, 且在嵌入、注册和检测全链使用同一协议摘要。

---

## 3.9 嵌入与检测流程

嵌入端执行以下步骤：

1. 在选定采样步提取 latent、可微 CLIP 语义、204维手工结构统计和真实 Q/K attention；
2. 构造 `lf_content`、`tail_robust` 和 `attention_geometry` 三个分支风险场；
3. 为每个分支构造包含优先载体方向的候选矩阵；
4. 通过完整特征 JVP/VJP、无阻尼 PSD-CG 和逐列门禁求解三个 rank-4 Null Space；
5. 构造空间 LF、高斯幅值尾部截断和 attention 几何更新；
6. 对 attention 更新执行单调回溯，对内容投影执行能量门禁；
7. 生成 clean、carrier-only 与完整方法最终图像，执行三边累计完整特征门禁和最终图像真实 Q/K 双归因增益门禁，并记录反事实原子及文件摘要、图像摘要、算子成本、残差、投影能量、pair 权重身份和环境信息。

检测端执行以下步骤：

1. 只读取待检图像、密钥和公开模型配置；
2. 通过 VAE 编码重建图像 latent，并重建固定 LF 与尾部截断模板；
3. 计算原始统一内容分数；
4. 必要时从图像 attention 关系估计几何变换并重新计算对齐分数；
5. 应用 calibration split 冻结的完整 evidence 协议；
6. 输出水印证据判定和可审计统计量。

---

## 3.10 方法边界

SLM-WM 的科学方法终止于仅图像 $y_{\mathrm{evidence}}$ 判定。事件签名、payload probe、证据 manifest 和结果审计可以用于数据来源治理或失败诊断，但不能作为图像水印检测能力的一部分。

### 实验随机化与方法间配对

正式比较采用3个生成种子与3个水印密钥组成的9个交叉重复。对固定 Prompt 索引, SLM-WM 与全部主表 baseline 共享同一模型 revision、实际生成 seed 和基础 latent Tensor。基础 latent 由 `sha256_counter_normal_icdf_table20_float32_v2` 在 CPU float32 中查询冻结 Q20 中点逆 CDF 表规范生成, 在 CPU 转换到目标 dtype 后才搬运到执行设备；目标 Tensor 的 shape、dtype、内容 SHA-256 和联合身份摘要随 observation 持久化。水印密钥重复共享根密钥派生索引与整数身份, 各方法仍使用自身载体和检测机制。配对统计在比较检测判定之前先拒绝任何 seed、密钥重复或基础 latent 身份不一致的样本。

单次 GPU 运行只物化一个登记重复, 最终论文统计必须覆盖全部9个重复。三个论文运行层级使用同一个随机化注册表、方法参数、攻击与 baseline 协议, 只改变 Prompt 数量、划分规模和目标 FPR。因此 probe 结果只能支持 probe 的统计强度, 但其方法与公平评测定义不能比 pilot 或 full 更弱。

方法实现存在不等于论文结论成立。空间 LF 的有效性、高斯幅值尾部截断的攻击鲁棒性、Q/K 几何恢复的增益和完整 fixed-FPR 均必须由真实 GPU 生成、clean negative、真实攻击、正式机制重跑消融、外部 baseline 和受治理结果包共同支撑。
