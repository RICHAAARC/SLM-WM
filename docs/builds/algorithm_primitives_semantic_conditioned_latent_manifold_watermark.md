# 算法原语：语义条件潜流形水印的统一设计

## 一、文档定位

本文档面向一个从零开始构建的文本到图像扩散模型水印项目，给出算法原语（algorithmic primitives）层面的统一设计。本文档不以工程模块清单为中心，而以“image processing theory + latent manifold watermark + robust detection statistics + semantic subspace optimization”为主线，定义水印扰动如何在扩散潜空间中被选择、分解、注入、检测与归因。

本文档采用语义条件潜流形水印（Semantic-conditioned Latent Manifold Watermarking，SLM-WM）作为方法名称。SLM-WM 的核心思想是：水印不是多个外部技巧的并列组合，而是在当前样本的分支语义风险场约束下，通过完整特征 Jacobian 的真实 JVP/VJP 与无阻尼约束投影求解 Null Space。低频（Low-Frequency，LF）内容证据、尾部截断鲁棒证据和 Self-Attention 几何锚点均由真实安全子空间导出。正式检测只接收待检图像、密钥和公开模型配置, 不读取生成 latent 或生成轨迹。

核心公式的数值常量、NCHW 广播、风险写回、QR 列参考、PRG 分流、实际 dtype 共同回溯和证据角色以 `method_semantic_invariants.md` 为权威定义。本文档中的叙述不得覆盖或弱化其中任一 fail-closed 条件。

---

## 二、统一问题建模

给定文本提示 $p$、扩散生成模型 $G_\theta$、采样更新算子 $S_\theta$、随机种子和初始噪声 $\xi$，扩散采样过程可写为

$$
z_{t-1}=S_\theta(z_t,p,t),\qquad x=D(z_0),
$$

其中，$z_t$ 表示第 $t$ 步潜变量，$D$ 表示变分自编码器（Variational Autoencoder，VAE）decoder 或最终图像解码算子。SLM-WM 不在生成后图像域叠加水印，而是在采样轨迹中执行受约束 latent update：

$$
\tilde{z}_t=z_t+\Delta z_t,\qquad z_{t-1}=S_\theta(\tilde{z}_t,p,t).
$$

因此，算法的基本对象是 $\Delta z_t$。该扰动应满足以下条件：

1. 对完整 CLIP 语义与声明的204维手工结构统计具有通过数值门禁的一阶零响应；
2. 对攻击后检测统计量具有稳定响应；
3. 在几何失配后能够通过参考系恢复回到同一内容判定标准；
4. 所有主证据均可在固定误报率（False Positive Rate，FPR）约束下校准。

SLM-WM 使用构造式协议选择水印方向，不声明求解一个联合标量优化问题。首先定义当前 latent 点附近的局部隐式特征水平集

$$
\mathcal L_F(z_t;\mathcal U_t)
=
\{z\in\mathcal U_t:F(z)=F(z_t)\}.
$$

当 $F$ 在局部可微并满足常秩条件时，其在 $z_t$ 处的切空间等于 $\ker J(z_t)$。正式算子不验证常秩定理条件，也不构造全局非线性流形、坐标图、测地线或回缩；“潜流形”严格限定为这一隐式水平集的局部切空间解释。实际数值对象是每个分支经风险支持约束后得到的残差门控子空间

$$
\widehat{\mathcal T}_{b,t}
=
\operatorname{span}(N_b),
\qquad
\|J(z_t)N_b[:,j]\|_2\leq\varepsilon_J.
$$

对内容分支 $b\in\{\mathrm{LF},\mathrm{tail}\}$，固定密钥模板先投影到对应子空间，再由该分支的有效风险预算约束实际幅值：

$$
v_b=\operatorname{Norm}\!\left(N_bN_b^\top\nu_b\right),
\qquad
\Delta z_t^b
=
\operatorname{RiskBoundedScale}
\left(v_b,b_b^{\mathrm{eff}},\lambda_b\|z_t\|_2\right).
$$

令 $z_t^{\mathrm{base}}=z_t+\Delta z_t^{\mathrm{LF}}+\Delta z_t^{\mathrm{tail}}$。注意力分支在该内容基底上重算直接 Q/K 目标梯度，经自身子空间投影和单调回溯得到

$$
g_A
=
\left.\nabla_z\mathcal S_A(z)\right|_{z=z_t^{\mathrm{base}}},
\qquad
\Delta z_t^{\mathrm A}
=
\alpha^*\operatorname{Norm}\!\left(N_AN_A^\top g_A\right).
$$

$\alpha^*$ 只有在候选的真实 Q/K 分数同时高于原始 latent 与内容基底时才被接受。最终构造为

$$
\Delta z_t
=
\Delta z_t^{\mathrm{LF}}
+
\Delta z_t^{\mathrm{tail}}
+
\Delta z_t^{\mathrm{A}}.
$$

风险零支持清理和重新单位化后，算子先对每个分支的实际 float32 单位方向独立重新执行完整特征精确 JVP；实际 dtype 写回后，再对真正联合写入的 Tensor 重新执行完整特征 JVP，并通过有限特征变化门禁。逐分支复验防止不同分支响应抵消或共同缩放掩盖非安全方向，联合写回复验捕获 dtype 量化误差。三个分支分别使用对应风险预算求解局部 Jacobian Null Space，而不是三个独立水印器的简单叠加，也不是未执行的联合目标求解器。

---

## 三、原语 1：分支风险与局部安全几何

### （一）设计目标

该原语解决“不同图像是否应使用同一嵌入位置、同一扰动方向与同一扰动强度”的问题。固定噪声模式或固定统计子空间忽略图像内容差异，容易在平滑或语义显著区域产生可见扰动，也难以充分利用纹理区域的冗余。SLM-WM 通过语义风险场为每个样本构造分支风险算子，使局部安全切空间候选与水印载体随内容而变化。

### （二）内容属性场

在采样过程中，对潜空间位置 $u$ 定义内容属性向量：

$$
\phi(u)=
[
\phi_{\mathrm{sem}}(u),
\phi_{\mathrm{tex}}(u),
\phi_{\mathrm{adj}}(u),
\phi_{\mathrm{lcr}}(u),
\phi_{\mathrm{attn\_stab}}(u)
],
$$

其中，$\phi_{\mathrm{sem}}$ 固定为 CLIP patch-to-CLS 余弦一致性经 $(c+1)/2$ 映射，$\phi_{\mathrm{tex}}$ 固定为解码灰度水平与垂直绝对梯度和除以2，$\phi_{\mathrm{lcr}}$ 固定为灰度相对反射填充5x5局部均值的绝对偏离；三者双线性映射到 latent 网格后只执行公开 $[0,1]$ 截断, 不逐样本重新定标。$\phi_{\mathrm{adj}}$ 使用当前与紧邻上一 scheduler 步的真实解码 RGB 差异：

$$
\phi_{\mathrm{adj}}(u)
=
1-\frac{1}{3}\sum_{c=1}^{3}
\left|x_t(c,u)-x_{t-1}(c,u)\right|.
$$

运行时在每个 callback-on-step-end 时刻维护真实相邻 latent；第0步和缺失上一状态的注入不属于当前协议。$\phi_{\mathrm{attn\_stab}}$ 由不少于两个冻结层的直接 Q/K 关系一致性独立计算。

### （三）风险场与承载预算

LF、尾部截断鲁棒载体和注意力几何对纹理与稳定性的偏好不同, 因此正式方法为三个分支分别定义风险场。对分支 $b$ 定义：

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

其中 LF 使用 $\psi_{\mathrm{LF}}(q)=q$, 从而回避高纹理区域；尾部截断鲁棒分支使用 $\psi_{\mathrm{tail}}(q)=1-q$, 从而偏好稳定纹理区域；注意力分支使用 `risk_neutral_texture_value=0.5` 的常数中性项, 因而不随纹理升降改变风险方向并主要由 attention stability 控制。三分支权重、阈值和预算参数分别读取 `lf_content_risk_config.*`、`tail_robust_risk_config.*` 和 `attention_geometry_risk_config.*`；例如 LF 局部对比权重的精确路径是 `lf_content_risk_config.local_contrast_risk_weight`。风险越低，区域越适合对应分支承载水印。承载预算定义为

$$
b_b(u)=\operatorname{clip}\left(b_{\min}+\kappa_b(1-\rho_b(u)),b_{\min},b_{\max}\right).
$$

据此得到三个候选区域：

$$
\Omega_{\mathrm{LF}}=\{u:\rho_{\mathrm{LF}}(u)<\tau_{\mathrm{LF}}\},
$$

$$
\Omega_{\mathrm{tail}}=\{u:\rho_{\mathrm{tail}}(u)<\tau_{\mathrm{tail}}\},
$$

$$
\Omega_{\mathrm{A}}=\{u:\rho_{\mathrm{A}}(u)<\tau_A\}.
$$

$\Omega_{\mathrm{A}}$ 使用独立 attention stability 风险, 不复用 LF 或尾部截断分支的区域集合。该输入必须由不少于两个冻结层的真实 Q/K 关系计算；核心风险接口把它定义为必需参数，缺失时直接失败，普通跨步稳定度不能成为替代值。正式候选矩阵构造时, $u\notin\Omega_b$ 的预算被置为 0, 资格集合内部才保留连续风险预算; 因而 `eligible_indices` 不是仅供日志展示的字段。完整方法只对当前实际参与嵌入的活动分支执行资格集合 fail-closed 门禁；移除风险路由的正式消融不执行该门禁, 避免被移除机制继续筛选实验样本。每次注入同时保存三个分支完整风险值、连续预算和资格 mask 的 Tensor 内容 SHA-256, 并以 `branch_risk_content_digest` 联合绑定三组内容。

连续预算还必须约束最终实际分支更新。空间预算在 channel 维复制到 NCHW latent 后按连续 NCHW 顺序形成 $B_b$；风险求解、写回包络和证据摘要必须消费同一有效预算 Tensor。安全投影保持 float32；先单位化投影方向，以 `risk_bounded_scale_direction_epsilon=1e-12` 检查零预算坐标泄漏，将允许的零预算数值残差置为精确0并重新单位化。对最终单位安全方向 $v_b$、名义强度 $a_b=\lambda_b\|z_t\|_2$ 和冻结常量 $b_{\max}^b$, 定义 $\widehat b_b=b_b^{\mathrm{eff}}/b_{\max}^b$ 以及逐元素硬包络 $E_b(i)=a_b\|v_b\|_\infty\widehat b_b(i)$。`RiskBoundedScale` 只在 $|v_b(i)|>\epsilon_{dir}$ 的活动集合上求满足 $|\Delta z_t^b(i)|\le E_b(i)$ 的最大可行全局标量；活动集为空或最终步长不大于 `null_space_numerical_epsilon=1e-12` 时失败。方向 epsilon 与数值 epsilon 角色不同, 不能共用隐式常量。风险修正后必须以投影前模板或真实 Q/K 梯度响应为分母重新执行逐分支精确 JVP, 相对残差不大于 $10^{-4}$ 才能继续。禁止用当前样本的预算最大值重新归一化, 也禁止在风险投影后无条件恢复固定二范数强度。图像风险输入使用 `risk_image_signal_interpolation_mode=bilinear` 与 `risk_image_signal_align_corners=false`, Q/K stability 使用 `risk_attention_signal_interpolation_mode=bilinear` 与 `risk_attention_signal_align_corners=true`；两者均不执行逐样本 min-max。中性纹理值只由 `risk_neutral_texture_value=0.5` 定义。

---

## 四、原语 2：语义条件安全子空间优化

### （一）语义条件 Null Space

SLM-WM 将固定统计 Null Space 推广为样本条件化的完整特征 Null Space。定义

$$
F(z_t)=
\begin{bmatrix}
F_{\mathrm{sem}}(z_t)\\
F_{\mathrm{struct}}(z_t)
\end{bmatrix},
\qquad J(z_t)=\frac{\partial F}{\partial z_t}.
$$

正式安全空间为

$$
\mathcal{N}_{\mathrm{sem}}(z_t,p)
=
\{v:J(z_t)v=0\}.
$$

其中，$F_{\mathrm{sem}}$ 是冻结 CLIP 输出的512维归一化完整 embedding；$F_{\mathrm{struct}}$ 是由3维通道均值、3维通道标准差、3维水平绝对梯度均值、3维垂直绝对梯度均值和192维8x8 RGB 平均池化组成的204维手工结构统计向量。正式 Jacobian 输出宽度为716，不执行坐标选择、随机投影或特征草图。该向量只声明对这些显式结构统计坐标的保持，不等同于一般感知质量模型；FID/KID 与配对质量指标在实验层独立评价最终图像。

### （二）真实 JVP 与候选方向

为避免显式构造 $716\times n$ Jacobian，系统同时使用 Jacobian-Vector Product（JVP）与 Vector-Jacobian Product（VJP）。对方向 $v$ 和特征余切 $y$：

$$
\operatorname{JVP}(v)=Jv,\qquad
\operatorname{VJP}(y)=J^\top y.
$$

正式方法调用精确自动微分。运行时优先使用 `torch.func.linearize` 与 `torch.func.vjp` 在同一 latent 点复用线性算子；若底层等价算子明确不支持该接口，才使用 `torch.autograd.functional.jvp/vjp` 重算。两条路径都不使用有限差分。候选池由实际载体方向和密钥方向构成：

$$
D_b=[d_1^b,\ldots,d_m^b]\in\mathbb{R}^{n\times m},\qquad {D_b}^{\top}D_b=I.
$$

LF 与尾部截断分支分别把固定盲检模板作为首个方向；注意力分支把真实 Q/K 目标梯度作为首个方向；其余列由设备无关密钥 PRG 方向补齐。候选方向以算子角色、分支、latent shape 和候选矩阵 shape 形成独立 domain, 并使用与内容模板相同的版本化 SHA-256 计数器流。分支风险预算以非负对角算子 $B_b=\operatorname{diag}(b_b)$ 显式进入约束投影，而不是预乘后再用低维响应矩阵选择方向。

在正式算法中, $F_{\mathrm{sem}}$ 不只由 CLIP 架构名称定义, 还由 `openai/clip-vit-base-patch32@3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` 的精确权重函数定义。同理, 潜空间变量、VAE 解码器和扩散 Transformer 来自 `stabilityai/stable-diffusion-3.5-medium@b940f670f0eda2d07fbb75229e779da1ad11eb80`。科学算子还精确冻结以下 YAML 字段：`pipeline_class_name=diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline`、`vae_class_name=diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL`、`transformer_class_name=diffusers.models.transformers.transformer_sd3.SD3Transformer2DModel`、`scheduler_class_name=diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler`、`vae_scaling_factor=1.5305`、`vae_shift_factor=0.0609`、`latent_torch_dtype=float16`、`vision_torch_dtype=float32` 和 `public_detection_schedule_index=7`。

完整解析配置以 `formal_method_config_schema=slm_wm_formal_method_runtime_config_v4` 和 dataclass 的完整 `asdict` 值形成 payload, 通过 UTF-8、`ensure_ascii=false`、键名升序、无空白分隔符 `(',', ':')` 和 SHA-256 得到 `formal_method_config_digest`。该摘要不含 YAML 排版与绝对路径。运行时实际类名、VAE scale/shift、dtype、检测索引、冻结 attention 层和配置摘要必须全部一致；任一漂移都会改变 Jacobian、Null Space 或检测算子并使科学单元失败。

### （三）风险支持矩阵自由约束投影

对每个种子方向 $d_i^b$，先计算完整响应 $r_i=J B_b d_i^b$，再通过无阻尼半正定共轭梯度求解一致系统

$$
(J B_b^2 J^\top)y_i=r_i.
$$

约束投影方向为

$$
u_i^b=B_b d_i^b-B_b^2J^\top y_i.
$$

因此

$$
Ju_i^b=J B_b d_i^b-JB_b^2J^\top y_i=0.
$$

从冻结的20个种子池依次求解，得到4个独立合格方向后执行 QR：

$$
N_b=\operatorname{qr}([u_1^b,\ldots,u_4^b]).
$$

三个分支分别得到：

$$
N_{\mathrm{LF}},\qquad N_{\mathrm{tail}},\qquad N_{\mathrm{A}}.
$$

正式实现逐方向记录 PSD-CG 迭代数与残差、$\|J u_i^b\|_2/\|J B_b d_i^b\|_2$ 和平方二范数投影能量保留比例。设被接受的路由候选矩阵为 $V_b$、投影方向矩阵为 $U_b=N_bR_b$。float32 reduced QR 使用 `null_space_numerical_epsilon=1e-12` 规范列符号；任一对角项不大于该 epsilon 或条件数超过 `maximum_qr_condition_number=1000000.0` 时失败。独立参考严格采用 `qr_reference_solve_protocol=right_upper_triangular_solve_without_explicit_inverse_v1`, 通过 $\widetilde V_bR_b=V_b$ 求解；第 $j$ 列残差使用 $J\widetilde V_b[:,j]$, 不能使用跨列共享 RMS。PSD-CG 不添加阻尼，最大64次迭代，相对收敛阈值为 $10^{-6}$；QR 后每列完整 Jacobian 相对残差不得超过 $10^{-4}$，能量保留比例不得低于0.01，且正交误差不得超过 `maximum_orthogonality_error=0.00001`。任一门禁失败时运行阻断。每个分支分别保存 $D_b$、$B_b$、$J(B_bD_b)$、$U_b$、$JU_b$、$N_b$、$JN_b$ 和 $J\widetilde V_b$ 的角色限定 Tensor 内容 SHA-256；每类响应只能使用其对应角色字段并绑定该数学对象。`solver_digest` 由候选与响应形状、秩、接受列索引、全部逐列数值和上述8类 Tensor 内容摘要通过共享纯记录函数重建；结果门禁必须重算, 不接受 producer 自报布尔值。

三个分支先在 float32 中按 `lf_content -> tail_robust -> attention_geometry` 固定顺序累加。不得逐分支 cast。attention 单调回溯与最终写回共同调用唯一的 `compose_ordered_float32_update_once`；不得分别实现 `(z+base)+attention` 和 `z+(LF+tail+attention)` 两种非结合加法路径。对共同缩放 $q_k=2^{-k}$、$k=0,\ldots,24$, 只把 original latent 的 float32 表示与 $q_k\Delta z_t$ 相加, 随后单次 cast 到真实 latent dtype：

$$
\Delta z_t^{\mathrm{written}}
=
\operatorname{cast}_{\operatorname{dtype}(z_t)}
\left(z_t^{(32)}+q_k\Delta z_t^{(32)}\right)-z_t.
$$

联合包络同时按 $q_k$ 缩放。零包络位置要求实际增量严格为0, 正包络位置要求实际增量与包络之比不超过1；该比较不增加 ULP 容差。随后对该 Tensor 重新执行完整特征精确 JVP：

$$
r_{\mathrm{written}}
=
\frac{\left\|J_t\Delta z_t^{\mathrm{written}}\right\|_2}
{\max\!\left(\left\|F(z_t)\right\|_2,\epsilon\right)}
\leq10^{-4}.
$$

这一复验用于捕获 float32 Null Space 方向转换为扩散 latent dtype 后产生的响应，验证对象是实际进入 scheduler 的增量。attention 启用时, 实际写回完整候选还必须高于原 latent 与相同 $q_k$ 下单次 cast 的 LF/tail 内容基底 Q/K 分数。共同回溯只使用 `quantized_budget_envelope_backtracking_factor=0.5` 与 `quantized_budget_envelope_backtracking_maximum_steps=24`。实现选择首个同时通过联合包络、非零写回、Q/K 单调性、完整 JVP 和有限变化的候选；初始候选及最多24次共同减半全部失败时阻断。局部 Jacobian 零响应仍不能单独证明有限写回更新保持完整语义，因此实现还对每次真正写回的 latent 重新提取完整特征；全部扩散步骤结束后，再直接比较最终 clean 与 watermarked 成图。两级有限变化门禁均要求 CLIP cosine 不低于0.995且手工结构统计特征相对漂移不高于0.02。

全部关键 Tensor 使用 `slm_wm_tensor_content_v1` 形成内容身份。该协议依次绑定协议版本、PyTorch dtype、有序 shape 和连续原始字节。单次注入分别保存 $\Delta z_t^{\mathrm{LF}}$、$\Delta z_t^{\mathrm{tail}}$ 与 $\Delta z_t^{\mathrm A}$ 的原生计算 Tensor SHA-256, 再保存三者联合摘要、合成更新摘要和实际量化写回增量摘要。`quantized_composition_evidence_digest` 还按固定角色绑定 original latent、candidate latent、活动分支 update 与 envelope、联合 update 与 envelope、实际写回 dtype/shape、共同缩放因子和回溯步数；结果消费端必须独立重算。不同 dtype、shape、角色、轨迹或任一字节变化都会产生不同身份。

---

## 五、原语 3：LF 精确内容载体

### （一）角色定义

LF 分支负责 clean 条件下的高精度主证据，目标是低误报、低感知损伤和轻度失真下的稳定检测。LF 的“低频”严格指 latent 的 height/width 二维空间轴, 不使用一维向量索引平滑代替空间低通。

### （二）固定检测模板与安全投影

给定 LF 密钥 $K_{\mathrm{LF}}$、模型标识和 latent 形状, 生成与 latent 同形状的高斯模板, 并只在二维空间轴执行低通：

$$
\nu_{\mathrm{LF}}
=
\operatorname{Norm}\left(
\operatorname{AvgPool}_{k\times k}
(\operatorname{PRG}_{\mathcal{N}}(K_{\mathrm{LF}},model,shape))
\right).
$$

平均池化只作用于 height/width 二维轴, 对每个 batch/channel 独立使用 kernel 5、stride 1、padding 2、二维零填充、`ceil_mode=false`、`count_include_pad=true` 和 `divisor_override=null`, 不跨 batch 或 channel 轴传播数值；池化后去均值与 L2 归一化各自在整个模板 Tensor 上计算一个全局标量。七个离散池化字段均属于方法配置, 不能使用运行库默认值替代。

正式 PRG 以 `sha256_counter_normal_icdf_table20_float32_v2` 登记完整逐字节协议。规范 domain payload 包含 `keyed_prg_version`、`key_material`、`domain_fields` 与 `shape`, 使用 UTF-8、`ensure_ascii=false`、键名升序和无空白分隔符 `(',', ':')` 形成 stable JSON, 再执行 SHA-256 得到32字节 domain digest。counter 从0开始, 以16字节无符号大端编码拼接到 domain digest 后再执行 SHA-256。Gaussian 路径把连续 SHA-256 块解释为 MSB-first 大端比特流, 跨计数器块连续提取20位索引 $i$, 并查询

$$
q_i=\operatorname{round}_{\mathrm{binary32}}\!\left(\Phi^{-1}\!\left(\frac{i+0.5}{2^{20}}\right)\right),
\qquad 0\le i<2^{20}.
$$

冻结表全部1048576个 binary32 位模式的完整大端字节 SHA-256 为 `70abf440a7f3670147965ffa52f5aaa639dab97f6282b68f3a9a1b1ce5e6cf5a`。该分布是有限离散的 Q20 中点逆 CDF 量化标准正态, 不是连续精确的 $\mathcal N(0,1)$；理想中点 KS 距离为 $2^{-21}=4.76837158203125\times10^{-7}$, 计入登记的 float32 CDF 舍入误差后总上界为 $4.912236096776823\times10^{-7}$。规范 float32 Tensor 和目标 dtype 转换都在 CPU 完成, 随后才搬运到执行设备。独立 uniform 路径把每个32字节块按 offset 0、8、16、24 切成4个连续8字节无符号大端 word, 取 $m=w\gg11$, 按 $u=(m+1)/(2^{53}+2)$ 得到严格位于 $(0,1)$ 的53位值；该路径只用于把 attention 关系符号按0.5阈值映射为 $\{-1,+1\}$。所有 domain 同时绑定算子角色、分支、精确模型标识和 shape；CPU/CUDA 设备 RNG 不参与方法身份。MPFR 逐项舍入复验是 `tools/harness/verify_normal_quantile_reference.py` 生成的外层证据, 不进入 PRG 算法摘要或 `keyed_prg_protocol_digest`。当前逐字节固定向量只在 Windows CPU 实测, Linux/Colab 一致性必须由 GPU 运行前门禁复验。

固定模板投影到真实安全子空间：

$$
\bar{\nu}_{\mathrm{LF}}
=
N_{\mathrm{LF}}N_{\mathrm{LF}}^\top\nu_{\mathrm{LF}}.
$$

令

$$
v_{\mathrm{LF}}=
\frac{\bar{\nu}_{\mathrm{LF}}}{\|\bar{\nu}_{\mathrm{LF}}\|_2}.
$$

LF latent update 使用 `branch_risk_bounds_written_update` 定义的最大可行全局标量, 满足逐位置硬包络：

$$
\Delta z_t^{\mathrm{LF}}
=
\operatorname{RiskBoundedScale}
\left(v_{\mathrm{LF}},b_{\mathrm{LF}}^{\mathrm{eff}},
\lambda_{\mathrm{LF}}\|z_t\|_2\right).
$$

检测模板 $\nu_{\mathrm{LF}}$ 只依赖密钥、公开模型标识和 latent 形状, 不依赖生成轨迹或样本级安全基底。

### （三）检测统计量

检测端只对待检图像执行 VAE 编码得到 $\hat z$, 并计算去均值归一化相关统计量：

$$
s_{\mathrm{LF}}=\operatorname{Corr}(\hat z,\nu_{\mathrm{LF}}).
$$

---

## 六、原语 4：尾部截断鲁棒补充载体

### （一）角色定义

尾部截断分支负责纹理区域和困难攻击条件下的鲁棒补充，尤其面向压缩、噪声、重采样、裁剪重缩放和再扩散后仍可能保留的残余证据。该分支不单独承担主判定。

该分支的正式标识为 `tail_robust`。“尾部”指 Q20 中点逆 CDF 量化标准正态随机变量绝对幅值分布的尾部, 与二维空间频谱无关。

### （二）高斯幅值尾部截断

给定尾部载体密钥 $K_{\mathrm{tail}}$，生成与 latent 同形状的 Q20 中点逆 CDF 量化标准正态候选模板：

$$
\nu_{\mathrm{tail}}
=
\operatorname{PRG}_{\mathcal{N}}(K_{\mathrm{tail}},model,shape).
$$

令元素总数为 $n$。根据冻结的尾部比例 $\gamma$，按 $(|\nu_i|,-i)$ 降序稳定选择 $\lceil n\gamma\rceil$ 个元素：

$$
I_\gamma
=
\operatorname{TopK}_{\lceil n\gamma\rceil}
\left(\left\{(|\nu_{\mathrm{tail},i}|,-i)\right\}_{i=1}^{n}\right),
$$

$$
\widetilde{\nu}_{\mathrm{tail},i}
=
\nu_{\mathrm{tail},i}
\mathbb{I}(i\in I_\gamma).
$$

该选择发生在元素幅值域，同幅值时以公开展平索引消除歧义；记录中的阈值是入选集合内最小绝对幅值。非入选元素保持精确0, 截断模板只除以整体二范数, 不执行会使非入选位置重新非零的去均值。该过程不对元素执行 Fourier 变换、余弦变换、带通滤波或按空间波数排序。截断后的模板具有由随机样本决定的宽频谱。`tail_fraction` 是概率分布尾部保留比例，不是频率截止值，也不定义空间频带。

尾部载体的版本化协议正文绑定 `tail_fraction`、稳定排序规则和 PRG 版本。检测原子保存模板 shape、元素总数、`ceil(n * tail_fraction)` 选中数、阈值、实际保留比例和模板内容摘要；raw/aligned 两路的 latent shape、LF/tail 模板摘要及尾部统计必须完全相同。

尾部载体同样先投影到分支安全子空间并执行风险硬包络缩放：

$$
\Delta z_t^{\mathrm{tail}}
=
\operatorname{RiskBoundedScale}\left(
\operatorname{Norm}\left(
N_{\mathrm{tail}}N_{\mathrm{tail}}^\top
\widetilde{\nu}_{\mathrm{tail}}
\right),
b_{\mathrm{tail}}^{\mathrm{eff}},
\lambda_{\mathrm{tail}}\|z_t\|_2
\right).
$$

### （三）检测统计量

$$
s_{\mathrm{tail}}
=
\operatorname{Corr}(\hat z,\widetilde{\nu}_{\mathrm{tail}}).
$$

内容分数由 LF 与尾部截断分支统一融合：

$$
s_c=
\lambda_{\mathrm{LF}}s_{\mathrm{LF}}
+
\lambda_{\mathrm{tail}}s_{\mathrm{tail}},
$$

其中

$$
\lambda_{\mathrm{LF}}>\lambda_{\mathrm{tail}},\qquad
\lambda_{\mathrm{LF}}+\lambda_{\mathrm{tail}}=1.
$$

两个分支不分别设置独立正判阈值后投票。固定 FPR 只约束统一内容分数和后续完整 evidence 判定。

---

## 七、原语 5：Self-Attention 相对关系几何锚点

### （一）设计目标

几何链用于参考系恢复，不直接提供 positive 判定。SLM-WM 将几何锚点绑定在扩散模型 Self-Attention 的相对关系上，使参考系依赖语义结构关系，而不是依赖绝对像素坐标。

### （二）Attention graph 构造

正式层集合精确固定为 `transformer_blocks.0.attn` 与 `transformer_blocks.23.attn`。对第 $t$ 步、第 $\ell$ 层 Transformer block, 在冻结二维图像 token 抽样集合
$\mathcal I$ 上直接读取每个注意力头的 `to_q` 与 `to_k` 投影；层缺失或不公开 `to_q`、`to_k`、`heads` 时运行失败。设

$$
\ell_{ij}^{(h)}=\frac{q_i^{(h)\top}k_j^{(h)}}{\sqrt{d_h}},\qquad
L_{ij}=\frac1H\sum_{h=1}^{H}
\left(\ell_{ij}^{(h)}-\frac1{|\mathcal I|}\sum_{k\in\mathcal I}\ell_{ik}^{(h)}\right),
$$

$$
P_{ij}=\frac1H\sum_{h=1}^{H}
\frac{\exp \ell_{ij}^{(h)}}{\sum_{k\in\mathcal I}\exp \ell_{ik}^{(h)}}.
$$

$P$ 是 SLM-WM 在抽样图像 token 集合上定义的关系概率。SD3.5 的完整 joint
attention 前向还包含未抽样图像 token 与文本 token, 因此 $P$ 不表示模块完整
attention 权重。科学算子的真实来源是模型实际 Q/K 投影及其对 latent 的
autograd 路径。多头概率必须先逐头执行 softmax 再平均, 一般不等于平均 logits
的 softmax。每层记录模块名、头数、head width、实际 scale、Q/K 归一化类、
源网格与抽样索引；模块公开 scale 必须与 $1/\sqrt{d_h}$ 一致, 否则运行失败。

对 batch $b$、冻结层 $\ell$ 和 token $i$, 先定义中心化单位概率行

$$
\overline P_{bi:}^{\ell}=
\frac{P_{bi:}^{\ell}-\frac1n\mathbf1}
{\max(\|P_{bi:}^{\ell}-\frac1n\mathbf1\|_2,\epsilon_{\mathrm{num}})}.
$$

跨层一致性、incoming centrality 和排序分数为

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

选择数为 $\min(n,\max(4,\lceil0.5n\rceil))$。按 $(-q_i,\operatorname{token\_index}_i)$ 升序稳定排序选择对应 token, 再把选中位置升序记录为 $\mathcal{V}_A$。定义单点权重和非对角 pair 权重

$$
a_i=
\begin{cases}
1,&i\in\mathcal{V}_A,\\
\omega,&i\notin\mathcal{V}_A,
\end{cases}
\qquad
w_{ij}=a_i a_j\mathbf{1}[i\ne j],
\qquad \omega=0.25.
$$

一次注入中的梯度、内容基底复算、单调回溯和最终 combined latent 复验必须
冻结同一个 $(a,w)$。选择摘要、原始二维 token 索引、$\omega$ 和外积规则共同
形成该次注入的 pair 权重身份摘要。仅图像盲检从自身可见的最终图像 Q/K 重新
盲选, 并在该次检测的 raw、registration 与 aligned 路径内部冻结另一个身份。
嵌入端与检测端共享构造规则, 但由于选择依赖各自 Q/K 数据, 不比较两个私有
身份摘要。完整规则网格的非零支撑用于连续二维注册。

以冻结温度 $\tau_r=0.25$ 个 logit 单位定义降序可微行内 rank

$$
\rho_{ij}=\frac1{|\mathcal I|}
\left[1+\sum_{k\ne j}\sigma\left(\frac{L_{ik}-L_{ij}}{\tau_r}\right)\right].
$$

公开二维坐标 $p_i\in[-1,1]^2$ 由 `token_indices` 唯一恢复。角点 token 中心分别落在 -1 与 1, 坐标身份固定为 `normalized_xy_token_centers_corner_endpoints_v1`；token 稳定图插值和图像仿射重采样统一使用 `align_corners=True`。距离因子为
$D_{ij}=\|p_i-p_j\|_2/(2\sqrt2)$。第4分量使用中心化关系概率调制距离：

$$
G_{ij}=\left(P_{ij}-\frac1{|\mathcal I|}\sum_{k\in\mathcal I}P_{ik}\right)
\left(D_{ij}-\frac1{|\mathcal I|}\sum_{k\in\mathcal I}D_{ik}\right).
$$

相对关系定义为

$$
r_{ij}=
[
L_{ij},
\rho_{ij},
P_{ij},
G_{ij}
].
$$

四个分量分别表达线性 Q/K 强度、序关系、非线性概率和概率偏离与距离偏离的
双中心交互, 不重复登记同一个 row-softmax 概率。距离行均值对应的
$\overline D_i(P_{ij}-1/n)$ 项被显式移除, 避免第4分量保留对第3分量的一阶
重复。均匀关系概率使 $G=0$, 因而
公开坐标不能脱离 Q/K 内容形成密钥相关。正式来源必须同时保存直接计算的
$L$ 与 $P$；只有概率矩阵的调用不能形成正式证据。每层 Q/K 原子还必须保存
完成模块 Q/K 归一化和二维抽样后的 $Q$、$K$、中心化 $L$、逐头 softmax 后
平均的 $P$ 以及原始二维 token 索引的内容 SHA-256。逐层原子自摘要和有序层
联合摘要用于同时约束数值内容与冻结层顺序。

由此得到 attention-relative graph：

$$
\mathcal{G}_A=(\mathcal{V}_A,\mathcal{E}_A,\{r_{ij}\}).
$$

### （三）几何水印嵌入

正式实现直接调用 Transformer attention 模块的 `to_q` 和 `to_k` 投影。为使检测端不需要原始 prompt, 嵌入与检测都使用冻结空文本条件和公开检测时刻。每层使用 `operator=attention_relation_signs`、精确层名和 token 数构造独立 uniform Tensor $U^\ell\in(0,1)^{n\times n}$。密钥对称符号图和四分量目标精确定义为

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

对 batch $b$、层 $\ell$、row $i$ 和分量 $c$, 令 $W_i=\sum_jw_{ij}$, 并分别计算

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

$W_i=0$ 或任一加权中心化能量不大于 $\epsilon_{\mathrm{num}}^2$ 时该 row 无效。每个 batch 先对有效 row 取均值；没有有效 row 时该 batch 分量分数为0。随后依次对 batch 和冻结层等权平均：

$$
C_c^{\ell}=\frac1B\sum_b
\operatorname{Mean}_{i\in\mathcal I_{b\ell c}^{valid}}
\operatorname{RowCorr}_{bic}^{\ell},
\qquad
C_c=\frac1{|\mathcal L|}\sum_{\ell\in\mathcal L}C_c^{\ell},
$$

$$
s_A(z_t;K)=\sum_{c=1}^{4}\omega_cC_c(z_t;K),
\quad \omega_c\ge0,
\quad \sum_{c=1}^{4}\omega_c=1.
$$

$G$ 真实进入评分与注册, 并通过中心化 $P$ 对 latent 保留非零梯度。四个分量
都依赖真实 Q/K；公开距离只作为第4分量的冻结调制因子。

完整方法的组合权重固定为 $\omega=(1/4,1/4,1/4,1/4)$。四个分量的正式留一消融分别把 $\omega_c$ 置为0, 并把其余三个权重固定为 $1/3$。留一变体只移除对应关系通道对组合目标的直接贡献, 其余通道仍按同一真实 Q/K 构造；该定义用于估计显式评分通道在其他通道存在时的边际必要性。该权重协议不是汇总层参数；它同时进入 latent 梯度目标、单调回溯、实际写回复验、最终成图归因、盲检原分数、双边仿射注册和对齐后同步分数。分量名称、活动集合、权重和归一化非负求和规则共同形成 `attention_relation_component_protocol_digest`。

通过 autograd 计算 $\nabla_{z_t}s_A$, 再投影到注意力分支的真实 Jacobian Null Space 并以风险硬包络允许的最大步长开始单调回溯：

$$
\Delta z_t^{\mathrm{A}}
=
\operatorname{MonotonicBacktrack}\left(
\operatorname{RiskBoundedScale}\left(
\operatorname{Norm}(N_A N_A^\top\nabla_{z_t}s_A),
b_A^{\mathrm{eff}},
\lambda_A\|z_t\|_2
\right)
\right).
$$

实现必须在 $z_t^{\mathrm{base}}=z_t+\Delta z_t^{\mathrm{LF}}+\Delta z_t^{\mathrm{tail}}$ 上重算梯度和候选分数。优化器直接接收 attention 分支已物化的 `RiskBoundedUpdate`, 每次减半都沿该对象的同一单位方向重新物化 update；禁止仅传入标量强度后在优化器内重建第二个方向。接受候选的 update Tensor 摘要必须与随后进入三分支合成的 attention update Tensor 摘要相同。回溯严格使用 `attention_backtracking_factor=0.5` 与 `attention_backtracking_maximum_steps=8`, 即以风险允许步长为起点最多执行8次减半, 只有

$$
s_A(z_t^{\mathrm{base}}+\Delta z_t^{\mathrm A})>
\max\{s_A(z_t),s_A(z_t^{\mathrm{base}})\}
$$

时才接受。记录必须包含原 latent 分数、优化内容基底分数、attention 接受候选分数、实际写回内容基底分数、实际写回组合分数、实际更新强度、回溯次数、原始梯度范数和投影后梯度范数。单次注入的 Q/K 原子角色固定为 `latent_before`、`optimization_content_base_latent`、`accepted_attention_candidate`、`actual_written_content_base_latent` 与 `actual_written_combined_latent`；5个角色必须逐层完整, 并分别绑定实际 float32 求值 latent 内容 SHA-256 与该次 Q/K 分数后形成联合摘要。角色分数还必须与顶层五个单调门禁分数字段精确一致。前3个角色验证 attention 分支自身的单调性，后2个角色验证共同缩放和实际 dtype 量化后的最终单调性。若安全投影为零、九个候选均不能提升真实 Q/K 目标或任一 Q/K 原子角色缺失, 运行必须失败。关系梯度 proxy 不得进入正式结果。

最终成图还必须通过 attention 因果可观测性门禁。除 clean 与完整方法图像外,
运行端以同一生成 seed、同一 scheduler 轨迹、相同 LF/tail 配置与算子重新生成
carrier-only 反事实 $x_{\mathrm C}$, 配置中只关闭 attention geometry。两侧首个
注入前 latent 必须具有相同的 dtype、shape 和连续原始字节 SHA-256。carrier-only
逐注入原子必须精确覆盖冻结注入步, 与完整方法逐项共享 scheduler 轨迹, 且每条
原子都没有 attention 分数、attention update、关系身份、pair 身份或 attention
Null Space 记录；该 JSONL 的路径、实际文件 SHA-256 和解析内容摘要共同进入结果
与 manifest。三张图像分别经 VAE 编码，并在 `public_detection_schedule_index=7` 对应的 `scheduler.timesteps[7]` 调用冻结 scheduler
的 `scale_noise` 施加公开固定噪声，再由同一冻结 Transformer 前向得到真实 Q/K。
当前协议不定义线性 latent/noise 混合作为等价加噪算子；scheduler 缺少可调用的
`scale_noise` 时科学单元直接失败。

三张最终图像的 Q/K 原子角色固定为 `final_clean_image`、
`final_carrier_only_image` 和 `final_watermarked_image`。仅图像检测对每个观测
另行保存 `raw_detection_image` 与 `aligned_detection_image` 两个角色；对齐后
必须重新提取真实 Q/K, 不能复制原图摘要。最终成图或盲检链缺少任一角色、冻结
层或逐层内容摘要时, 对应科学完成单元不能进入正式统计。

该干预估计的是 attention geometry 开关的总机制效应。首个 attention 干预发生
后, 完整方法与 carrier-only 的 latent 会不同, 因而后续 LF、tail 更新及生成轨迹
可以通过状态依赖算子产生交互差异；方法不假设两侧 realized carrier update 完全
相等, 也不把最终差异解释为排除全部中介路径后的纯直接效应。
在计算 attention 归因前, $x_{\mathrm C}$ 必须先通过相对 clean 成图 $x_0$ 的
完整特征保持门禁：

$$
\cos\!\left(\phi_{\mathrm{CLIP}}(x_0),\phi_{\mathrm{CLIP}}(x_{\mathrm C})\right)
\ge \tau_{\mathrm{sem}},\qquad
\frac{\|\phi_{\mathrm{struct}}(x_{\mathrm C})-\phi_{\mathrm{struct}}(x_0)\|_2}
{\max(\|\phi_{\mathrm{struct}}(x_0)\|_2,\epsilon)}
\le \tau_{\mathrm{struct}}.
$$

相同阈值还必须分别约束 $x_0$ 与完整方法成图 $x_{\mathrm F}$、以及
$x_{\mathrm C}$ 与 $x_{\mathrm F}$。三条最终图像边全部通过后才能计算归因, 从而
排除任意一侧内容漂移伪造 attention 差异的混杂路径。
盲检式归因增益允许 carrier-only 与完整方法分别执行自身稳定 token 盲选；配对
归因增益冻结 carrier-only 图像的 $w_{\mathrm C}$：

$$
g_{\mathrm{blind}}^{\mathrm{attr}}=s_A(x_{\mathrm F};w_{\mathrm F})-s_A(x_{\mathrm C};w_{\mathrm C}),
$$

$$
g_{\mathrm{paired}}^{\mathrm{attr}}=s_A(x_{\mathrm F};w_{\mathrm C})-s_A(x_{\mathrm C};w_{\mathrm C}).
$$

正式运行要求两者都严格大于冻结正下界 $\gamma_A$，当前
`minimum_final_image_attention_score_gain=0.0001`。该门禁必须由 CUDA 上最终
图像重编码的真实 Q/K 支撑, 并记录反事实配置、注入记录和 scheduler 轨迹摘要。
保持记录与 Q/K 记录必须绑定同一个反事实身份、更新原子路径及双摘要、持久化图像
路径和 SHA-256, 这些身份同时写入 manifest 并在缓存复用时从实际文件重建。
反事实缺失、首 latent 不同、仅 attention 开关以外的配置漂移、scheduler 不同、
carrier 原子残留 attention、三边保持失败、产物身份不一致或分数非有限时均失败。
clean 分数继续记录为总体水印对照, 但不替代 carrier-only 归因比较, 也不进入仅图像
检测接口。

### （四）几何可靠性统计量

检测端只从待检图像执行 VAE 编码, 在公开固定噪声时刻运行空文本 Transformer
前向并由真实 Q/K 构造四分量关系张量 $R_{\mathrm{obs}}^{(l)}\in
\mathbb R^{n\times n\times4}$。对冻结的有界相似仿射与方形二面体候选 $T$,
根据规范 token 坐标和观测 token 坐标构造双线性采样矩阵 $W_T$, 对四个
分量使用同一个空间变换：

$$
\widehat R_{T,c}^{(l)}=W_TR_{\mathrm{obs},c}^{(l)}W_T^\top,
\qquad c=1,\ldots,4.
$$

同时使用逆变换构造观测坐标到规范坐标的采样矩阵 $V_T$, 将四个冻结密钥
分量投影 $S_{K,c}=\pi_cS_K$ 前推到观测参考系：

$$
\widetilde S_{T,c}^{(l)}=V_TS_{K,c}^{(l)}V_T^\top.
$$

观测参考系直接使用一次盲选得到的 $a_{\mathrm{obs}}$ 与
$w_{\mathrm{obs}}$。规范参考系不重新选择 token, 而是使用与关系图相同的采样
矩阵传递单点权重：

$$
a_{\mathrm{can},T}=W_Ta_{\mathrm{obs}},
\qquad
w_{\mathrm{can},T,ij}=a_{\mathrm{can},T,i}a_{\mathrm{can},T,j}\mathbf{1}[i\ne j].
$$

无覆盖位置的单点权重置零, 并由双向覆盖门禁处理。这里先计算
$a_{\mathrm{can},T}=W_Ta_{\mathrm{obs}}$ 再做外积, 不使用
$W_Tw_{\mathrm{obs}}W_T^\top$。两个坐标实现共享该次盲检内部的同一个 pair 权重
身份摘要, 仅数值实现摘要不同。

两个方向的非对角归一化相关分数分别为

$$
s_{\mathrm{can}}(T,l)=
\frac14\sum_{c=1}^{4}
\operatorname{RowCorr}_{w_{\mathrm{can},T}}
\left(\widehat R_{T,c}^{(l)},S_{K,c}^{(l)}\right),
$$

$$
s_{\mathrm{obs}}(T,l)=
\frac14\sum_{c=1}^{4}
\operatorname{RowCorr}_{w_{\mathrm{obs}}}
\left(R_{\mathrm{obs},c}^{(l)},\widetilde S_{T,c}^{(l)}\right).
$$

注册同时记录四个分量的独立分数。均匀 attention 下 $L$、中心化后的
$\rho$、$P$ 与 $G$ 均不产生密钥相关, 注册总分为0并无法通过可靠性门禁。

设 $c_{\mathrm{can}}$、$u_{\mathrm{can}}$、$c_{\mathrm{obs}}$ 和 $u_{\mathrm{obs}}$ 分别为规范拉回与观测前推的有效坐标覆盖率和唯一采样率。覆盖惩罚、双向关系分数和候选目标定义为

$$
p_{\mathrm{coverage}}(T)=0.01
\left[
(1-c_{\mathrm{can}})+(1-u_{\mathrm{can}})
+(1-c_{\mathrm{obs}})+(1-u_{\mathrm{obs}})
\right],
$$

$$
s_{\mathrm{bi}}(T,l)=0.10s_{\mathrm{can}}(T,l)+0.90s_{\mathrm{obs}}(T,l),
$$

$$
J(T,l)=s_{\mathrm{bi}}(T,l)-p_{\mathrm{coverage}}(T).
$$

冻结层有序集合精确为 $\mathcal L=(\texttt{transformer\_blocks.0.attn},\texttt{transformer\_blocks.23.attn})$。每层先独立得到层内最优候选 $\widehat T_l$；检测器再按

$$
(\widehat l,\widehat T)=
\operatorname*{lexargmax}_{l\in\mathcal L}
\left(J(\widehat T_l,l),s_{\mathrm{obs}}(\widehat T_l,l),r_{\mathrm{reg}}(\widehat T_l,l),-\operatorname{rank}_{\mathcal L}(l)\right)
$$

执行唯一跨层裁决。前三项依次比较注册目标、观测关系分与注册置信度；完全同分时选择冻结顺序中更靠前的层。

检测器先在与攻击注册表无关的连续相似变换定义域上构造粗网格, 再执行三层三分层级局部优化。旋转定义域为 $[-32,32]$ 度并等分为4个区间；尺度在 log-scale 上以 $[-\log\sqrt{2},0,\log\sqrt{2}]$ 为对称锚点；两个方向的归一化位移锚点为 $\{-0.28,0,0.28\}$，另加入方形二面体变换。第一层局部步长取粗网格单元半宽, 后续层均除以3。因此旋转步长为 $8,8/3,8/9$ 度，log-scale 步长为 $\log\sqrt{2}/2,\log\sqrt{2}/6,\log\sqrt{2}/18$，位移步长为 $0.14,0.14/3,0.14/9$；尺度局部候选使用 $\exp(-\delta_s),1,\exp(\delta_s)$。每轮组合后立即相对全部方形二面体基元分解候选, 只保留残余旋转位于 $[-32,32]$、均匀尺度位于 $[1/\sqrt2,\sqrt2]$ 且两个平移分量位于 $[-0.28,0.28]$ 的候选；局部细化不能越过公开定义域。定义域、网格分辨率、边界过滤和缩小比例不读取攻击角度、裁剪比例或攻击实现参数，粗锚点及全部层级组合也不精确复述正式攻击尺度或其逆尺度。相对 identity 候选的可审计对齐增益为

$$
g_{\mathrm{align}}=
s_{\mathrm{obs}}(\widehat T,l)-s_{\mathrm{obs}}(I,l).
$$

注册关系分数记为 $s_{\mathrm{reg}}=s_{\mathrm{can}}(\widehat T,l)$。令规则抽样网格包含 $n$ 个 token, 正式方法预注册 $A=12$ 个锚点, 并按 $h_j=\operatorname{round}(j(n-1)/(A-1))$、$j=0,\ldots,11$ 在抽样索引范围内确定性均匀选择；$n<12$ 时直接失败, 不允许自动缩减锚点数。残差使用 `normalized_xy_token_centers_corner_endpoints_v1` 中的归一化 xy 欧氏距离, token 关系采样与图像重采样统一采用 `align_corners=true`。仅具有有效双线性覆盖的锚点组成分母集合 $\mathcal V$；匹配到唯一观测 token 且残差 $d_j\le0.20$ 的锚点记为内点：

$$
r_{\mathrm{inlier}}
=\frac{\sum_{j\in\mathcal V}\mathbf1[\operatorname{unique}(j)\land d_j\le0.20]}
{|\mathcal V|}.
$$

无有效覆盖锚点时 $r_{\mathrm{inlier}}=0$。内点平均残差记为 $e_{\mathrm{affine}}$；没有内点时该值为非有限失败值。注册置信度为

$$
r_{\mathrm{reg}}=
\max(0,s_{\mathrm{bi}})
\cdot r_{\mathrm{inlier}}
\cdot
\exp(-e_{\mathrm{affine}})
\cdot\min(c_{\mathrm{can}},c_{\mathrm{obs}}).
$$

结构注册门禁为

$$
registration\_geometry\_reliable=
(s_{\mathrm{obs}}>0)
\land(s_{\mathrm{bi}}>0)
\land(\Delta J>0)
\land(c_{\mathrm{can}}\ge0.45)
\land(c_{\mathrm{obs}}\ge0.45)
\land(r_{\mathrm{inlier}}\ge0.50)
\land(e_{\mathrm{affine}}\le0.20).
$$

`attention_anchor_count=12`、`attention_residual_threshold=0.20` 与 `attention_minimum_inlier_ratio=0.50` 是预注册结构常量, 不是 calibration 或 test 选择的统计阈值。完整检测器配置只保存在顶层运行 manifest；alignment 和样本记录保存决策所需字段及检测器配置摘要, calibration/test 必须绑定同一摘要。决策字段缺失、数值漂移或配置摘要不一致均使当前闭合失败。

使用 $\hat T$ 重采样待检图像时, 输入先解码为 RGB uint8 Tensor 并归一化到 $[0,1]$；仿射网格和采样固定采用 bilinear、`padding_mode=border` 与 `align_corners=true`。连续结果按 $\operatorname{floor}(255\cdot\operatorname{clip}(x,0,1))$ 转回 RGB uint8。检测端随后必须重新运行同一个仅图像 Q/K 提取器, 并使用 $a_{\mathrm{can},\hat T}$ 对全部冻结层计算恢复后同步分数 $s_{\mathrm{sync}}$。恢复后不得重新选择稳定 token, 且观测、注册和同步记录的 pair 权重身份必须一致。最终几何可靠性定义为：

$$
geometry\_reliable=
(registration\_geometry\_reliable)
\land
(pair\_weight\_identity\_ready)
\land
(s_{\mathrm{reg}}\ge\tau_{\mathrm{reg}})
\land
(r_{\mathrm{reg}}\ge\tau_{\mathrm{confidence}})
\land
(s_{\mathrm{sync}}\ge\tau_{\mathrm{sync}}).
$$

calibration split 的未删失 clean negatives 只冻结内容分数阈值、几何关系分阈值、注册置信度阈值、恢复后同步分阈值和 rescue window。失败原因由冻结余量与几何可靠性规则确定, 不能作为额外调参入口。任何统计阈值缺失或不满足时, 几何路径都不得执行内容救回。

---

## 八、原语 6：鲁棒检测统计与固定 FPR 判定

### （零）仅图像检测输入边界

正式检测函数只允许输入待检图像 $x'$, 密钥 $K$ 和公开模型配置 $M$：

$$
\hat y=\operatorname{Detect}(x',K,M).
$$

不得输入原始生成 latent、采样轨迹、原始图像、原始 prompt、生成 seed、生成端 Q/K 原子或样本级安全基底。检测端使用冻结 VAE posterior mode 重建

$$
\hat z=(\operatorname{mode}(q_{\mathrm{VAE}}(x'))-\texttt{vae\_shift\_factor})\cdot\texttt{vae\_scaling\_factor},
$$

并通过密钥、精确模型标识和公开 latent 形状重建 LF 与尾部截断模板。内容分数固定为 $0.70s_{\mathrm{LF}}+0.30s_{\mathrm{tail}}$, 不执行分支投票。

完整注入、carrier-only 反事实与全部检测记录必须对每个活动内容分支共享唯一模板内容摘要及协议摘要。calibration 冻结协议同时绑定 LF/tail 协议摘要、0.70/0.30权重和尾部比例；应用协议前从全部冻结字段重算 `threshold_digest`, 并从假阳性计数重算派生比率。

检测密钥计划包含注册水印密钥与 SHA-256 domain-separated wrong-key 两个角色。注入密钥摘要必须等于计划中的注册密钥摘要；注册密钥检测模板与嵌入模板相同, wrong-key 模板必须唯一且不同。完整计划正文只在顶层运行身份中保存一次；样本级检测记录只保存角色、实际材料摘要和计划摘要引用, 总科学内容证据通过该计划摘要绑定顶层计划。

检测 Q/K 的公开噪声输出 `shape` 精确等于 $\hat z$ 的 NCHW shape。调用固定 `public_detection_noise_prg_protocol=sha256_counter_normal_icdf_table20_float32_v2`, 且 `key_material=public_detection_noise_domain=public_image_only_qk_detection_noise_v1`。`domain_fields` 精确包含同值 `operator`、冻结 `model_id`、40位 `model_revision`、`width=512`、`height=512`、`inference_steps=20`、`public_detection_schedule_index=7` 与 `latent_shape=shape`；不含水印密钥、Prompt、生成 seed 或轨迹。该 payload 执行本节冻结的 SHA-256 大端计数器比特流、连续20位索引提取和 Q20 中点逆 CDF float32 表查询, 在 CPU 完成实际 dtype 转换后再搬运到 $\hat z$ 的设备。令 $t_{det}=\operatorname{scheduler.timesteps}[7]$, 检测 latent 只能由 `scheduler.scale_noise(hat_z,t_det,epsilon_det)` 得到；scheduler 缺少真实 `scale_noise` 时直接失败。Transformer 条件固定为 `public_detection_conditioning_protocol=sd3_empty_text_triplet_without_cfg_v1` 与 `public_detection_condition_text=""`。安全基底只在嵌入端控制失真, 不成为检测端 side information。raw 与 aligned 图像必须分别重新编码和提取 Q/K, 原子角色固定为 `raw_detection_image` 与 `aligned_detection_image`。

### （一）Neyman-Pearson 判定

给定 clean negative 分布 $\mathcal{D}_0$ 和目标误报率 $\alpha$，内容阈值由 calibration split 决定：

$$
\tau_c=Q_{1-\alpha}(s_c\mid\mathcal{D}_0).
$$

### （二）内容主判

原始内容边界余量为

$$
m_c^{\mathrm{raw}}=s_c^{\mathrm{raw}}-\tau_c.
$$

原始内容正判为

$$
positive\_by\_content=\mathbb{I}(m_c^{\mathrm{raw}}\ge0).
$$

### （三）几何救回

仅当原始内容证据处于边界失败窗口、失败原因可解释为几何失配或低置信度，并且几何恢复可信时，才允许 rescue：

$$
rescue\_eligible
=(\delta_{\mathrm{low}}\le m_c^{\mathrm{raw}}<0)
\land geometry\_reliable
\land fail\_reason\in\{geometry\_suspected,low\_confidence\}.
$$

恢复后内容边界余量为

$$
m_c^{\mathrm{align}}=s_c^{\mathrm{align}}-\tau_c.
$$

几何救回成立为

$$
rescue\_applied=rescue\_eligible\land\mathbb{I}(m_c^{\mathrm{align}}\ge0).
$$

Evidence-level 判定为

$$
y_{\mathrm{evidence}}=positive\_by\_content\lor rescue\_applied.
$$

### （四）fixed-FPR 与 rescue 的统计边界

同阈值 rescue 不等价于无条件保持目标 FPR。正式 fixed-FPR 口径必须约束完整 evidence-level 判定协议，而不仅是内容阈值 $\tau_c$。因此，以下规则属于本原语的统计边界：

1. $\tau_c$ 只能由 calibration split 中的 clean negative 分布冻结；
2. 几何关系分阈值、注册置信度阈值、恢复后同步分阈值和 rescue window 必须由 calibration clean negatives 冻结；
3. 锚点数量12、归一化 xy 欧氏残差上界0.20和相对有效覆盖锚点的最小内点比例0.50必须由方法配置预注册, calibration 与 test 均不得调节；
4. test split 不得用于调阈值、调 rescue window、调失败原因规则或调几何可靠性条件；
5. rescue 只能作用于 $\delta_{\mathrm{low}}\le m_c^{\mathrm{raw}}<0$ 的边界失败样本，不能对远离阈值的 negative 样本开放；
6. 报告 fixed-FPR 时必须同时报告 raw content FPR、rescue 后 clean negative FPR 和 rescue 后 attacked negative FPR；
7. 若 rescue 后整体 evidence-level FPR 超过目标 operating point，则论文不得声称完整系统仍满足该 fixed-FPR 目标，除非重新在 calibration split 中冻结包含 rescue 的完整决策协议。

### 正式随机化与基础 latent 公平控制

正式比较使用生成种子索引 $i\in\{0,1,2\}$ 与水印密钥索引 $j\in\{0,1,2\}$ 的笛卡尔积, 共9个交叉重复。生成 seed 偏移固定为

$$
\Delta=(0,1000003,2000003).
$$

对 Prompt 全局索引 $q$ 和冻结基础 seed $s_0=1703$, 当前重复的实际生成 seed 为

$$
s_{i,q}=s_0+\Delta_i+q.
$$

基础 latent 不调用各适配器自己的 CPU/CUDA RNG。协议使用 `sha256_counter_normal_icdf_table20_float32_v2`, 把模型 ID、40位 revision、$s_{i,q}$、Tensor shape 和协议名称共同写入 domain, 从 SHA-256 大端计数器的连续 MSB-first 比特流提取20位索引并查询冻结 Q20 中点逆 CDF float32 表。规范 float32 生成和目标 dtype 转换均在 CPU 完成, 随后才搬运到目标设备：

$$
z^{(0)}_{i,q}=\operatorname{Cast}_{d}\left(
\operatorname{BoxMuller}\left(
\operatorname{SHA256Counter}(\mathrm{domain}_{i,q})
\right)\right).
$$

SLM-WM、Tree-Ring、Gaussian Shading、Shallow Diffuse 和 T2SMark 必须消费 $z^{(0)}_{i,q}$ 的 clone, 并在方法写入前保存实际目标 dtype Tensor 的 shape、dtype、内容 SHA-256 和联合身份摘要。水印密钥整数身份由统一根密钥和 $j$ 经 SHA-256 派生；各方法可以把该身份映射到自身 ring、message、patch 或 T2SMark 编码, 但不得改变重复索引。配对统计逐字段要求两侧的重复 ID、seed/key 索引、实际生成 seed、密钥材料摘要、基础 latent 内容摘要和基础 latent 身份摘要完全相同。

单次执行只选择一个登记重复, 用于支持中断恢复和 GPU 会话隔离。正式论文汇总必须覆盖全部9个重复并把 Prompt、生成 seed 与密钥重复纳入统计组织；任一单重复结果都不能替代跨种子、跨密钥证据。`probe_paper`、`pilot_paper` 与 `full_paper` 使用同一个9重复注册表, 只改变下表中的 Prompt 数量、划分数量和目标 FPR。

三级正式规模固定为：

| Prompt 数量 | dev | calibration | test | 目标 FPR |
| ---: | ---: | ---: | ---: | ---: |
| 70 | 3 | 33 | 34 | 0.1 |
| 700 | 30 | 330 | 340 | 0.01 |
| 7000 | 300 | 3300 | 3400 | 0.001 |

阈值搜索必须直接调用 `content OR same-threshold rescue` 的最终布尔判定。`fail_reason` 是内容阈值的函数, 因此应用冻结阈值时必须重新计算, 不能沿用数据生成时临时阈值得到的分类。test split 只报告冻结协议结果及二项分布置信上界, 不参与任何阈值或 rescue 参数选择。

正式 FID/KID 使用 torch-fidelity 0.4.0 的 `inception-v3-compat` 2048 维特征。三个运行层级必须分别覆盖 70/700/7000 对 clean/watermarked 图像, 不允许以小规模像素直方图或前100对样本替代完整运行层级质量统计。

长耗时运行允许跨多个 Colab 会话完成, 但科学身份以完成单元而不是汇总会话为
边界.每个 Prompt、每个正式消融运行和每个 Inception feature batch 都保存实际
GPU、PyTorch/CUDA build、完整依赖锁、代码锁、精确配置摘要与随机性身份.最终
manifest 聚合全部完成单元的来源集合, 不得用最后一次会话的环境字段代表其他
完成样本.Inception batch 还必须把来源配置摘要绑定到组内实际图像路径与摘要.

正式机制消融集合由完整方法和14个变体组成：共享全局风险、移除分支风险路由、移除 Jacobian Null Space、LF-only、Tail-only、移除 LF、移除尾部载体、移除幅值尾部截断、四个 Q/K 分量逐项留一、移除完整 attention geometry 和移除图像对齐。每个变体都重新执行生成、真实攻击、仅图像检测与独立 calibration 阈值冻结。

该边界保证几何链仍是参考系恢复机制，而不是绕过内容阈值的第二条 positive 判定路径。

---

## 九、非核心扩展边界

事件签名和 payload probe 可以作为数据来源审计或失效模式诊断工具, 但不属于
当前 SLM-WM 科学方法算子, 不进入仅图像检测判定, 也不作为正式机制消融的
必要组件。正式水印判定终止于 $y_{\mathrm{evidence}}$。该边界避免把需要外部
事件记录的完整性校验误写成图像水印检测能力。

---

## 十、算法原语之间的非组合关系

SLM-WM 必须满足以下不变量：

1. 所有水印方向均由分支风险支持的局部 Jacobian Null Space 约束；
2. LF 与尾部截断载体是分支安全子空间中的互补内容证据；
3. Self-Attention 几何锚点是同一 latent update 的几何分量；
4. 几何链只恢复参考系，不直接判定 positive；
5. 恢复后内容重判复用同一内容阈值 $\tau_c$；
6. 外部事件签名和 payload probe 不进入正式水印判定；
7. 所有正式主张必须能够由固定 FPR、clean negative、attack family metrics 和机制消融支撑。

---

## 十一、算法原语清单

| 原语 | 输入 | 输出 | 论文角色 | 实现角色 |
|---|---|---|---|---|
| 分支语义风险场 | 解码图像、CLIP patch token、纹理和稳定性图 | $\rho_{\mathrm{LF/tail/A}}$、$b_{\mathrm{LF/tail/A}}$ | 内容自适应理论基础 | branch risk builder |
| 语义安全子空间 | 完整特征 JVP/VJP、分支风险预算、密钥种子方向 | $N_{\mathrm{LF}}$、$N_{\mathrm{tail}}$、$N_{\mathrm{A}}$ | semantic Null Space optimization | matrix-free Jacobian Null Space solver |
| LF 载体 | $N_{\mathrm{LF}}$、$K_{\mathrm{LF}}$ | $\Delta z_t^{\mathrm{LF}}$、$s_{\mathrm{LF}}$ | clean precision 主证据 | LF coder / detector |
| 尾部截断载体 | $N_{\mathrm{tail}}$、$K_{\mathrm{tail}}$、tail fraction | $\Delta z_t^{\mathrm{tail}}$、$s_{\mathrm{tail}}$ | 困难攻击条件下的补充证据 | tail robust embedder / detector |
| Attention 几何锚点 | 直接 Q/K 关系图、稳定 token | $\Delta z_t^{\mathrm{A}}$、$r_{\mathrm{sync}}$ | 几何同步创新 | attention anchor module |
| 仅图像鲁棒检测 | 待检图像、密钥、公开模型、$s_{\mathrm{LF}}$、$s_{\mathrm{tail}}$、geometry stats | $y_{\mathrm{evidence}}$ | fixed-FPR 主判 | image-only decision module |

---

## 十二、最终建议

实现与论文写作应始终围绕以下链条展开：

$$
\text{semantic risk field}
\rightarrow
\text{semantic-conditioned safe null-space}
\rightarrow
\text{LF/tail/attention carrier decomposition}
\rightarrow
\text{fixed-FPR robust detection}
\rightarrow
\text{same-threshold geometric rescue}.
$$

该链条在公式、代码接口、消融实验和论文叙事中必须保持一致。只有这样，SLM-WM 才能被清晰地定义为一个统一的 image processing theory 驱动方法，而不是工程组件堆叠。
