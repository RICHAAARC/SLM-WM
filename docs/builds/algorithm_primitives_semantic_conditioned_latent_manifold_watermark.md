# 算法原语：语义条件潜流形水印的统一设计

## 一、文档定位

本文档面向一个从零开始构建的文本到图像扩散模型水印项目，给出算法原语（algorithmic primitives）层面的统一设计。本文档不以工程模块清单为中心，而以“image processing theory + latent manifold watermark + robust detection statistics + semantic subspace optimization”为主线，定义水印扰动如何在扩散潜空间中被选择、分解、注入、检测与归因。

本文档采用语义条件潜流形水印（Semantic-conditioned Latent Manifold Watermarking，SLM-WM）作为方法名称。SLM-WM 的核心思想是：水印不是多个外部技巧的并列组合，而是在当前样本的分支语义风险场约束下，通过完整特征 Jacobian 的真实 JVP/VJP 与无阻尼约束投影求解 Null Space。低频（Low-Frequency，LF）内容证据、尾部截断鲁棒证据和 Self-Attention 几何锚点均由真实安全子空间导出。正式检测只接收待检图像、密钥和公开模型配置, 不读取生成 latent 或生成轨迹。

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

1. 对完整语义与视觉特征具有通过数值门禁的一阶零响应；
2. 对攻击后检测统计量具有稳定响应；
3. 在几何失配后能够通过参考系恢复回到同一内容判定标准；
4. 所有主证据均可在固定误报率（False Positive Rate，FPR）约束下校准。

SLM-WM 将水印方向选择统一为如下受约束优化问题：

$$
\Delta z_t^\star
=
\arg\max_{\Delta z_t}
\mathcal{S}_{\mathrm{wm}}(z_t+\Delta z_t)
+
\beta_g\mathcal{S}_{\mathrm{geo}}(z_t+\Delta z_t)
-
\beta_s\mathcal{D}_{\mathrm{sem}}(z_t,z_t+\Delta z_t)
-
\beta_v\mathcal{D}_{\mathrm{vis}}(z_t,z_t+\Delta z_t),
$$

约束为

$$
\Delta z_t\in
\mathcal{N}_{\mathrm{sem}}(z_t)
\cap
\mathcal{M}_{\mathrm{route}}(z_t,p)
\cap
\mathcal{B}_{\epsilon},
$$

其中，$\mathcal{S}_{\mathrm{wm}}$ 表示内容水印可检测性，$\mathcal{S}_{\mathrm{geo}}$ 表示几何同步可恢复性，$\mathcal{D}_{\mathrm{sem}}$ 表示语义偏移代价，$\mathcal{D}_{\mathrm{vis}}$ 表示视觉失真代价，$\mathcal{N}_{\mathrm{sem}}$ 表示语义条件安全子空间，$\mathcal{M}_{\mathrm{route}}$ 表示内容路由诱导的潜流形，$\mathcal{B}_{\epsilon}$ 表示强度预算约束。

在实现层面，$\Delta z_t^\star$ 被分解为三个互补方向：

$$
\Delta z_t^\star=
\Delta z_t^{\mathrm{LF}}
+
\Delta z_t^{\mathrm{tail}}
+
\Delta z_t^{\mathrm{A}},
$$

其中，$\Delta z_t^{\mathrm{LF}}$ 为低频稳定主证据，$\Delta z_t^{\mathrm{tail}}$ 为高斯幅值尾部截断鲁棒补充证据，$\Delta z_t^{\mathrm{A}}$ 为 Self-Attention 相对关系几何锚点证据。三个分支分别使用对应风险预算求解真实完整特征 Null Space，而不是三个独立水印器的简单叠加。

---

## 三、原语 1：语义条件潜流形构造

### （一）设计目标

该原语解决“不同图像是否应使用同一嵌入位置、同一扰动方向与同一扰动强度”的问题。固定噪声模式或固定统计子空间忽略图像内容差异，容易在平滑或语义显著区域产生可见扰动，也难以充分利用纹理区域的冗余。SLM-WM 通过语义风险场为每个样本构造自适应潜流形，使水印载体随内容而变化。

### （二）内容属性场

在采样过程中，对潜空间位置 $u$ 定义内容属性向量：

$$
\phi(u)=
[
\phi_{\mathrm{sem}}(u),
\phi_{\mathrm{tex}}(u),
\phi_{\mathrm{stab}}(u),
\phi_{\mathrm{sal}}(u)
],
$$

其中，$\phi_{\mathrm{sem}}$ 表示语义显著性，$\phi_{\mathrm{tex}}$ 表示纹理复杂度，$\phi_{\mathrm{stab}}$ 表示跨采样步稳定性，$\phi_{\mathrm{sal}}$ 表示视觉显著性风险。语义显著性可由预测干净图像 $\hat{x}_0^t$、最终图像的 saliency model、分割模型或 Self-Attention saliency 估计。若语义掩码首先在图像域得到，则应映射到 latent 分辨率：

$$
M_z=\Pi_{x\rightarrow z}(M_x).
$$

该映射必须进入后续分支风险场与子空间规划，不能仅作为可视化或日志字段。

### （三）风险场与承载预算

LF、尾部截断鲁棒载体和注意力几何对纹理与稳定性的偏好不同, 因此正式方法为三个分支分别定义风险场。对分支 $b$ 定义：

$$
\rho_b(u)=
\frac{1}{Z_b}
\left[
\eta_s^b\phi_{\mathrm{sal}}(u)
+\eta_m^b\phi_{\mathrm{sem}}(u)
+\eta_t^b\psi_b(\phi_{\mathrm{tex}}(u))
+\eta_i^b(1-\phi_{\mathrm{stab}}(u))
+\eta_A^b(1-\phi_{\mathrm{attn\_stab}}(u))
\right].
$$

其中 LF 使用 $\psi_{\mathrm{LF}}(q)=q$, 从而回避高纹理区域；尾部截断鲁棒分支使用 $\psi_{\mathrm{tail}}(q)=1-q$, 从而偏好稳定纹理区域；注意力分支主要由 attention stability 控制。风险越低，区域越适合对应分支承载水印。承载预算定义为

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

$\Omega_{\mathrm{A}}$ 使用独立 attention stability 风险, 不复用 LF 或尾部截断分支的区域集合。该输入必须由不少于两个冻结层的真实 Q/K 关系计算；核心风险接口把它定义为必需参数，缺失时直接失败，普通跨步稳定度不能成为替代值。正式候选矩阵构造时, $u\notin\Omega_b$ 的预算被置为 0, 资格集合内部才保留连续风险预算; 因而 `eligible_indices` 不是仅供日志展示的字段。完整方法只对当前实际参与嵌入的活动分支执行资格集合 fail-closed 门禁；移除风险路由的正式消融不执行该门禁, 避免被移除机制继续筛选实验样本。每次注入同时保存三个分支的风险摘要、资格位置数和风险场摘要值。

---

## 四、原语 2：语义条件安全子空间优化

### （一）语义条件 Null Space

SLM-WM 将固定统计 Null Space 推广为样本条件化的完整特征 Null Space。定义

$$
F(z_t)=
\begin{bmatrix}
F_{\mathrm{sem}}(z_t)\\
F_{\mathrm{vis}}(z_t)
\end{bmatrix},
\qquad J(z_t)=\frac{\partial F}{\partial z_t}.
$$

正式安全空间为

$$
\mathcal{N}_{\mathrm{sem}}(z_t,p)
=
\{v:J(z_t)v=0\}.
$$

其中，$F_{\mathrm{sem}}$ 是冻结 CLIP 输出的512维归一化完整 embedding；$F_{\mathrm{vis}}$ 是由3维通道均值、3维通道标准差、3维水平梯度、3维垂直梯度和192维8x8 RGB 空间池化组成的204维完整视觉向量。正式 Jacobian 输出宽度为716，不执行坐标选择、随机投影或特征草图。

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

在正式算法中, $F_{\mathrm{sem}}$ 不只由 CLIP 架构名称定义, 还由 `openai/clip-vit-base-patch32@3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` 的精确权重函数定义。同理, 潜空间变量、VAE 解码器和扩散 Transformer 来自 `stabilityai/stable-diffusion-3.5-medium@b940f670f0eda2d07fbb75229e779da1ad11eb80`。仓库分支漂移会改变 Jacobian 与 Null Space, 因此精确 revision 属于科学算子定义, 而不是可选的工程元数据。

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

正式实现逐方向记录 PSD-CG 迭代数与残差、$\|J u_i^b\|_2/\|J B_b d_i^b\|_2$、投影能量保留比例，并在 QR 后重新计算每列 $\|J N_b[:,j]\|_2$。PSD-CG 不添加阻尼，最大64次迭代，相对收敛阈值为 $10^{-6}$；QR 后每列完整 Jacobian 相对残差不得超过 $10^{-4}$，能量保留比例不得低于0.01，且基底正交误差不得超过 $10^{-5}$。任一门禁失败时运行阻断。只有这些条件全部成立的记录才称为 Null Space。

三个分支合成后，先按真实 latent dtype 执行写回，并显式恢复实际量化增量

$$
\Delta z_t^{\mathrm{written}}
=
\operatorname{cast}_{\operatorname{dtype}(z_t)}
\left(z_t+\Delta z_t\right)-z_t.
$$

随后对该 Tensor 重新执行完整特征精确 JVP：

$$
r_{\mathrm{written}}
=
\frac{\left\|J_t\Delta z_t^{\mathrm{written}}\right\|_2}
{\max\!\left(\left\|F(z_t)\right\|_2,\epsilon\right)}
\leq10^{-4}.
$$

这一复验用于捕获 float32 Null Space 方向转换为扩散 latent dtype 后产生的响应，验证对象是实际进入 scheduler 的增量。局部 Jacobian 零响应仍不能单独证明有限写回更新保持完整语义，因此实现还对每次真正写回的 latent 重新提取完整特征；全部扩散步骤结束后，再直接比较最终 clean 与 watermarked 成图。两级有限变化门禁均要求 CLIP cosine 不低于0.995且完整视觉特征相对漂移不高于0.02。

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

正式 $\operatorname{PRG}_{\mathcal N}$ 版本为 `sha256_counter_box_muller_float32_v1`。其 domain 同时绑定密钥、精确模型标识、分支名称和 latent shape；SHA-256 大端计数器流经53位开区间均匀映射与 Box-Muller 变换后，先生成 CPU float32 规范 Tensor，再搬运到目标设备。该定义不调用 CPU/CUDA 的设备 RNG，因而同一协议输入在两种设备上重建相同模板字节。

固定模板投影到真实安全子空间：

$$
\bar{\nu}_{\mathrm{LF}}
=
N_{\mathrm{LF}}N_{\mathrm{LF}}^\top\nu_{\mathrm{LF}}.
$$

LF latent update 为

$$
\Delta z_t^{\mathrm{LF}}
=
\alpha_t^{\mathrm{LF}}
\frac{\bar{\nu}_{\mathrm{LF}}}{\|\bar{\nu}_{\mathrm{LF}}\|_2}.
$$

$\alpha_t^{\mathrm{LF}}$ 按当前 latent 范数定义相对注入强度。检测模板 $\nu_{\mathrm{LF}}$ 只依赖密钥、公开模型标识和 latent 形状, 不依赖生成轨迹或样本级安全基底。

### （三）检测统计量

检测端只对待检图像执行 VAE 编码得到 $\hat z$, 并计算去均值归一化相关统计量：

$$
s_{\mathrm{LF}}=\operatorname{Corr}(\hat z,\nu_{\mathrm{LF}}).
$$

---

## 六、原语 4：尾部截断鲁棒补充载体

### （一）角色定义

尾部截断分支负责纹理区域和困难攻击条件下的鲁棒补充，尤其面向压缩、噪声、重采样、裁剪重缩放和再扩散后仍可能保留的残余证据。该分支不单独承担主判定。

该分支的正式标识为 `tail_robust`。“尾部”指标准高斯随机变量绝对幅值分布的尾部, 与二维空间频谱无关。

### （二）高斯幅值尾部截断

给定尾部载体密钥 $K_{\mathrm{tail}}$，生成与 latent 同形状的标准高斯候选模板：

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

该选择发生在元素幅值域，同幅值时以公开展平索引消除歧义；记录中的阈值是入选集合内最小绝对幅值。该过程不对元素执行 Fourier 变换、余弦变换、带通滤波或按空间波数排序。截断后的模板具有由随机样本决定的宽频谱。`tail_fraction` 是概率分布尾部保留比例，不是频率截止值，也不定义空间频带。

尾部载体同样先投影到分支安全子空间：

$$
\Delta z_t^{\mathrm{tail}}
=
\alpha_t^{\mathrm{tail}}
\operatorname{Norm}\left(
N_{\mathrm{tail}}N_{\mathrm{tail}}^\top
\widetilde{\nu}_{\mathrm{tail}}
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

对第 $t$ 步、第 $\ell$ 层 Transformer block, 在冻结二维图像 token 抽样集合
$\mathcal I$ 上直接读取每个注意力头的 `to_q` 与 `to_k` 投影。设

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

对规则二维 Q/K 网格中的第 $i$ 个 token, 跨层关系稳定度定义为其中心化
attention 行在全部冻结层对之间的平均余弦一致性, 显著度定义为该 token 在
全部冻结层中接收的平均 attention 质量。冻结选择分数为

$$
q_i=\operatorname{Stab}(i)\operatorname{Sal}(i).
$$

按 $q_i$ 降序并以原始二维 token 索引作为并列规则, 选择固定前50%且不少于4个
token 组成 $\mathcal{V}_A$。定义单点权重和非对角 pair 权重

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

公开二维坐标 $p_i\in[-1,1]^2$ 由 `token_indices` 唯一恢复, 距离因子为
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
$L$ 与 $P$；只有概率矩阵的调用不能形成正式证据。

由此得到 attention-relative graph：

$$
\mathcal{G}_A=(\mathcal{V}_A,\mathcal{E}_A,\{r_{ij}\}).
$$

### （三）几何水印嵌入

正式实现直接调用 Transformer attention 模块的 `to_q` 和 `to_k` 投影。为使检测端不需要原始 prompt, 嵌入与检测都使用冻结的空文本条件和公开检测时刻。密钥通过同一版本化 SHA-256 计数器 PRG 的均匀数流产生零对角关系符号矩阵 $S_K$；domain 绑定层名和 token 数, 规范 CPU float32 符号搬运到目标设备后再形成对称矩阵。四个分量采用冻结极性
$\pi=(1,-1,1,1)$；rank 的反极性使较高 logit 对应较小降序 rank。对每个分量
分别执行逐行 pair 加权中心化和归一化相关, 再严格等权组合：

$$
c_c=\operatorname{RowCorr}_{w}(r^{(c)},\pi_cS_K),\qquad
s_A(z_t;K)=\frac14\sum_{c=1}^{4}c_c.
$$

$G$ 真实进入评分与注册, 并通过中心化 $P$ 对 latent 保留非零梯度。四个分量
都依赖真实 Q/K；公开距离只作为第4分量的冻结调制因子。

通过 autograd 计算 $\nabla_{z_t}s_A$, 再投影到注意力分支的真实 Jacobian Null Space：

$$
\Delta z_t^{\mathrm{A}}
=
\alpha_t^{\mathrm{A}}
N_A N_A^\top
\nabla_{z_t}s_A.
$$

实现必须实际复算候选 latent 的 attention score, 通过单调回溯搜索保证接受的更新满足 `score_after > score_before`, 并记录 `score_before`、`score_after`、实际更新强度、回溯次数、原始梯度范数和投影后梯度范数。若安全投影为零或回溯后仍不能提升真实 Q/K 目标, 运行必须失败。关系梯度 proxy 不得进入正式结果。

最终成图还必须通过 attention 因果可观测性门禁。除 clean 与完整方法图像外,
运行端以同一生成 seed、同一 scheduler 轨迹、相同 LF/tail 配置与算子重新生成
carrier-only 反事实 $x_{\mathrm C}$, 配置中只关闭 attention geometry。两侧首个
注入前 latent 必须具有相同的 dtype、shape 和连续原始字节 SHA-256。carrier-only
逐注入原子必须精确覆盖冻结注入步, 与完整方法逐项共享 scheduler 轨迹, 且每条
原子都没有 attention 分数、attention update、关系身份、pair 身份或 attention
Null Space 记录；该 JSONL 的路径、实际文件 SHA-256 和解析内容摘要共同进入结果
与 manifest。三张图像分别经 VAE 编码，并在正式检测 timestep 调用冻结 scheduler
的 `scale_noise` 施加公开固定噪声，再由同一冻结 Transformer 前向得到真实 Q/K。
当前协议不定义线性 latent/noise 混合作为等价加噪算子；scheduler 缺少可调用的
`scale_noise` 时科学单元直接失败。

该干预估计的是 attention geometry 开关的总机制效应。首个 attention 干预发生
后, 完整方法与 carrier-only 的 latent 会不同, 因而后续 LF、tail 更新及生成轨迹
可以通过状态依赖算子产生交互差异；方法不假设两侧 realized carrier update 完全
相等, 也不把最终差异解释为排除全部中介路径后的纯直接效应。
在计算 attention 归因前, $x_{\mathrm C}$ 必须先通过相对 clean 成图 $x_0$ 的
完整特征保持门禁：

$$
\cos\!\left(\phi_{\mathrm{CLIP}}(x_0),\phi_{\mathrm{CLIP}}(x_{\mathrm C})\right)
\ge \tau_{\mathrm{sem}},\qquad
\frac{\|\phi_{\mathrm{vis}}(x_{\mathrm C})-\phi_{\mathrm{vis}}(x_0)\|_2}
{\max(\|\phi_{\mathrm{vis}}(x_0)\|_2,\epsilon)}
\le \tau_{\mathrm{vis}}.
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

检测器先在与攻击注册表无关的连续相似变换定义域上构造粗网格, 再执行三层三分层级局部优化。旋转定义域为 $[-32,32]$ 度并等分为4个区间；尺度在 log-scale 上以 $[-\log\sqrt{2},0,\log\sqrt{2}]$ 为对称锚点；两个方向的归一化位移锚点为 $\{-0.28,0,0.28\}$，另加入方形二面体变换。第一层局部步长取粗网格单元半宽, 后续层均除以3。因此旋转步长为 $8,8/3,8/9$ 度，log-scale 步长为 $\log\sqrt{2}/2,\log\sqrt{2}/6,\log\sqrt{2}/18$，位移步长为 $0.14,0.14/3,0.14/9$；尺度局部候选使用 $\exp(-\delta_s),1,\exp(\delta_s)$。每轮组合后立即相对全部方形二面体基元分解候选, 只保留残余旋转位于 $[-32,32]$、均匀尺度位于 $[1/\sqrt2,\sqrt2]$ 且两个平移分量位于 $[-0.28,0.28]$ 的候选；局部细化不能越过公开定义域。定义域、网格分辨率、边界过滤和缩小比例不读取攻击角度、裁剪比例或攻击实现参数，粗锚点及全部层级组合也不精确复述正式攻击尺度或其逆尺度。相对 identity 候选的可审计对齐增益为

$$
g_{\mathrm{align}}=
s_{\mathrm{obs}}(\widehat T,l)-s_{\mathrm{obs}}(I,l).
$$

注册关系分数记为 $s_{\mathrm{reg}}=s_{\mathrm{can}}(\widehat T,l)$。锚点内点比例 $r_{\mathrm{inlier}}$ 只以具有有效双线性覆盖的锚点数量为分母, 无覆盖区域由 $c_{\mathrm{can}}$ 与 $c_{\mathrm{obs}}$ 单独门禁；内点平均残差记为 $e_{\mathrm{affine}}$。注册置信度为

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
\land(r_{\mathrm{inlier}}\ge\tau_{\mathrm{inlier}})
\land(e_{\mathrm{affine}}\le\tau_{\mathrm{residual}}).
$$

使用 $\hat T$ 重采样待检图像后, 检测端必须重新运行同一个仅图像 Q/K 提取器, 并使用 $a_{\mathrm{can},\hat T}$ 对全部冻结层计算恢复后同步分数 $s_{\mathrm{sync}}$。恢复后不得重新选择稳定 token, 且观测、注册和同步记录的 pair 权重身份必须一致。最终几何可靠性定义为：

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

关系分数、注册置信度和恢复后同步阈值都由 calibration split 的未删失 clean negatives 冻结。任何一项缺失或不满足阈值时, 几何路径都不得执行内容救回。

---

## 八、原语 6：鲁棒检测统计与固定 FPR 判定

### （零）仅图像检测输入边界

正式检测函数只允许输入待检图像 $x'$, 密钥 $K$ 和公开模型配置 $M$：

$$
\hat y=\operatorname{Detect}(x',K,M).
$$

不得输入原始生成 latent、采样轨迹、原始图像、原始 prompt 或样本级安全基底。检测端通过 VAE 编码重建 $\hat z$, 通过密钥和公开 latent 形状重建 LF 与尾部截断模板。安全基底只在嵌入端控制失真, 不成为检测端 side information。

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
2. $\tau_{\mathrm{reg}}$、$\tau_{\mathrm{inlier}}$、$\tau_{\mathrm{sync}}$、rescue window 和 fail reason gate 也必须在 calibration 或预注册协议中冻结；
3. test split 不得用于调阈值、调 rescue window 或调几何可靠性条件；
4. rescue 只能作用于 $\delta_{\mathrm{low}}\le m_c^{\mathrm{raw}}<0$ 的边界失败样本，不能对远离阈值的 negative 样本开放；
5. 报告 fixed-FPR 时必须同时报告 raw content FPR、rescue 后 clean negative FPR 和 rescue 后 attacked negative FPR；
6. 若 rescue 后整体 evidence-level FPR 超过目标 operating point，则论文不得声称完整系统仍满足该 fixed-FPR 目标，除非重新在 calibration split 中冻结包含 rescue 的完整决策协议。

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
manifest 聚合全部完成单元的来源集合, 不得用最后一次会话的环境字段代表此前已
完成样本.Inception batch 还必须把来源配置摘要绑定到组内实际图像路径与摘要.

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

1. 所有水印方向均由 $\mathcal{N}_{\mathrm{sem}}\cap\mathcal{M}_{\mathrm{route}}$ 约束；
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
| Attention 几何锚点 | Self-Attention maps、稳定 token | $\Delta z_t^{\mathrm{A}}$、$r_{\mathrm{sync}}$ | 几何同步创新 | attention anchor module |
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
