# 算法原语：语义条件潜流形水印的统一设计

## 一、文档定位

本文档面向一个从零开始构建的文本到图像扩散模型水印项目，给出算法原语（algorithmic primitives）层面的统一设计。本文档不以工程模块清单为中心，而以“image processing theory + latent manifold watermark + robust detection statistics + semantic subspace optimization”为主线，定义水印扰动如何在扩散潜空间中被选择、分解、注入、检测与归因。

本文档采用语义条件潜流形水印（Semantic-conditioned Latent Manifold Watermarking，SLM-WM）作为方法名称。SLM-WM 的核心思想是：水印不是多个外部技巧的并列组合，而是在当前样本的分支语义风险场约束下，通过真实 Jacobian-Vector Product（JVP）和奇异值分解求解低响应潜空间方向。低频（Low-Frequency，LF）内容证据、尾部截断鲁棒证据和 Self-Attention 几何锚点均由真实安全子空间导出。正式检测只接收待检图像、密钥和公开模型配置, 不读取生成 latent 或生成轨迹。

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

1. 对语义内容与视觉质量具有低响应；
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

其中，$\Delta z_t^{\mathrm{LF}}$ 为低频稳定主证据，$\Delta z_t^{\mathrm{tail}}$ 为高斯幅值尾部截断鲁棒补充证据，$\Delta z_t^{\mathrm{A}}$ 为 Self-Attention 相对关系几何锚点证据。三个分支分别使用对应风险预算求解真实低响应子空间，而不是三个独立水印器的简单叠加。

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

LF、尾部截断鲁棒载体和注意力几何对纹理与稳定性的偏好不同, 因此正式方法不再共享单个风险标量。对分支 $b$ 定义：

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

$\Omega_{\mathrm{A}}$ 使用独立 attention stability 风险, 不再隐式复用 LF 或尾部截断分支的区域集合。正式候选矩阵构造时, $u\notin\Omega_b$ 的预算被置为 0, 资格集合内部才保留连续风险预算; 因而 `eligible_indices` 不是仅供日志展示的字段。每次注入同时保存三个分支的风险摘要、资格位置数和风险场摘要值。

---

## 四、原语 2：语义条件安全子空间优化

### （一）语义条件 Null Space

SLM-WM 将固定统计 Null Space 推广为样本条件化的语义安全子空间：

$$
\mathcal{N}_{\mathrm{sem}}(z_t,p)
=
\{v:
\|W_{\mathrm{sem}}J_{\mathrm{sem}}(z_t)v\|_2\le\epsilon_s,\quad
\|W_{\mathrm{vis}}J_{\mathrm{vis}}(z_t)v\|_2\le\epsilon_v
\}.
$$

其中，$J_{\mathrm{sem}}$ 表示语义特征映射的 Jacobian，$J_{\mathrm{vis}}$ 表示视觉质量相关映射的 Jacobian。当前正式实现令 $W_{\mathrm{sem}}=I$、$W_{\mathrm{vis}}=I$, 分支风险预算不伪装成特征维权重, 而是在候选矩阵 $C_b$ 构造时直接作用于 latent 空间坐标。该定义要求候选方向在语义特征与视觉质量空间中均为低响应方向。

### （二）真实 JVP 与候选方向

为避免显式构造完整 Jacobian，系统通过 Jacobian-Vector Product（JVP）估计局部响应。对候选方向 $v_i$：

$$
\psi_i^{\mathrm{sem}}=J_{\mathrm{sem}}(z_t)v_i,\qquad
\psi_i^{\mathrm{vis}}=J_{\mathrm{vis}}(z_t)v_i.
$$

正式方法调用精确 JVP。运行时优先使用 `torch.func.linearize` 在同一 latent 点复用精确线性算子; 若某个算子明确不支持 forward AD, 才使用 `torch.autograd.functional.jvp` 逐方向计算。两条路径都不是有限差分。trajectory-linearized approximation 只允许作为诊断对照, 不得生成论文主方法记录。候选方向先由密钥和分支预算构造：

$$
C_b=[v_1^b,\ldots,v_m^b]\in\mathbb{R}^{n\times m},\qquad {C_b}^{\top}C_b=I.
$$

其中每个 $v_i^b$ 在生成候选时已经乘入分支预算 $b_b(u)$, 因而路由约束发生在 SVD 之前。LF 与尾部截断分支分别把固定盲检模板作为首个优先候选; 注意力分支先计算真实 Q/K 目标梯度, 再把该梯度作为优先候选。其余列由密钥化随机方向补齐并统一 QR 正交化。该设计避免在 16384 维 latent 中使用极低秩纯随机候选时几乎完全丢失真实载体方向。

$$
\psi_i^{\mathrm{sem}}=\operatorname{JVP}(F_{\mathrm{sem}},z_t,v_i^b),\qquad
\psi_i^{\mathrm{vis}}=\operatorname{JVP}(F_{\mathrm{vis}},z_t,v_i^b).
$$

$F_{\mathrm{sem}}$ 由 VAE 可微解码与冻结 CLIP 图像编码器组成；$F_{\mathrm{vis}}$ 包含亮度、对比度、边缘和多尺度结构特征。模型参数冻结, 但从输出到 $z_t$ 的输入梯度保持可用。

### （三）安全基底求解

构造响应矩阵：

$$
R_b=
\begin{bmatrix}
W_{\mathrm{sem}}J_{\mathrm{sem}}C_b\\
\sqrt{\lambda_v}W_{\mathrm{vis}}J_{\mathrm{vis}}C_b
\end{bmatrix}
\in\mathbb{R}^{q\times m}.
$$

对 $R_b$ 执行奇异值分解（Singular Value Decomposition，SVD）：

$$
R_b=U\Sigma Q^\top.
$$

设 $Q_0$ 为最小奇异值对应的右奇异向量。$Q_0$ 位于候选系数空间, 必须乘回候选矩阵后才能得到 latent 基底：

$$
B_b=\operatorname{orth}(C_bQ_0).
$$

三个分支分别得到：

$$
B_{\mathrm{LF}},\qquad B_{\mathrm{tail}},\qquad B_{\mathrm{A}}.
$$

正式实现还必须记录 $\|R_bQ_0\|_F$、相对平均响应残差和 $\|B_b^\top B_b-I\|_F$, 分别验证绝对响应、相对低响应程度与基底正交性。选中方向的平均响应不得超过全部候选平均响应的 0.75。固定盲检模板投影后的能量保留比例还必须不低于 0.01; 低于门禁时运行失败, 不能对近零投影重新归一化后伪装为可检测载体。不得在 SVD 后直接施加未验证的坐标 mask, 因为后置 mask 可能破坏 Null Space 性质。

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

固定模板投影到真实安全子空间：

$$
\bar{\nu}_{\mathrm{LF}}
=
B_{\mathrm{LF}}B_{\mathrm{LF}}^\top\nu_{\mathrm{LF}}.
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

根据冻结的尾部比例 $\gamma$ 计算幅值分位点：

$$
\widetilde{\nu}_{\mathrm{tail},i}
=
\nu_{\mathrm{tail},i}
\mathbb{I}(|\nu_{\mathrm{tail},i}|\ge q_{1-\gamma}).
$$

该选择发生在元素幅值域，不对元素执行 Fourier 变换、余弦变换、带通滤波或按空间波数排序。截断后的模板具有由随机样本决定的宽频谱。`tail_fraction` 是概率分布尾部保留比例，不是频率截止值，也不定义空间频带。

尾部载体同样先投影到分支安全子空间：

$$
\Delta z_t^{\mathrm{tail}}
=
\alpha_t^{\mathrm{tail}}
\operatorname{Norm}\left(
B_{\mathrm{tail}}B_{\mathrm{tail}}^\top
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

对第 $t$ 步、第 $\ell$ 层 transformer block，Self-Attention 矩阵为

$$
A_t^{(\ell)}=\operatorname{softmax}\left(\frac{Q_t^{(\ell)}{K_t^{(\ell)\top}}}{\sqrt{d}}\right).
$$

选择稳定 token 集合：

$$
\mathcal{V}_A=
\{i:\operatorname{Stab}(A_{\cdot,i}^{(\ell)})\ge\tau_A,\ \operatorname{Sal}(i)\ge\tau_s\}.
$$

构造相对关系：

$$
r_{ij}=
[
A_{ij},
\operatorname{rank}_j(A_{ij}),
A_{ij}/\sum_k A_{ik},
\operatorname{dist}_{\mathrm{rel}}(i,j)
].
$$

由此得到 attention-relative graph：

$$
\mathcal{G}_A=(\mathcal{V}_A,\mathcal{E}_A,\{r_{ij}\}).
$$

### （三）几何水印嵌入

正式实现直接调用 Transformer attention 模块的 `to_q` 和 `to_k` 投影, 不使用 hidden-state cosine proxy。为使检测端不需要原始 prompt, 嵌入与检测都使用冻结的空文本条件和公开检测时刻。密钥产生零对角关系符号矩阵 $S_K$：

$$
s_A(z_t;K)
=
\operatorname{Corr}
\left(
A_t^{(\ell)}-\operatorname{RowMean}(A_t^{(\ell)}),
S_K
\right).
$$

通过 autograd 计算 $\nabla_{z_t}s_A$, 再投影到注意力分支的真实 Jacobian Null Space：

$$
\Delta z_t^{\mathrm{A}}
=
\alpha_t^{\mathrm{A}}
B_A B_A^\top
\nabla_{z_t}s_A.
$$

实现必须实际复算候选 latent 的 attention score, 通过单调回溯搜索保证接受的更新满足 `score_after > score_before`, 并记录 `score_before`、`score_after`、实际更新强度、回溯次数、原始梯度范数和投影后梯度范数。若安全投影为零或回溯后仍不能提升真实 Q/K 目标, 运行必须失败。关系梯度 proxy 不得进入正式结果。

### （四）几何可靠性统计量

检测端只从待检图像执行 VAE 编码, 在公开固定噪声时刻运行一次空文本 Transformer 前向并提取真实 Q/K attention。密钥关系行与观测 attention 行执行余弦一对一匹配, 再用三点确定性 RANSAC 拟合二维仿射变换 $\hat T$。几何可靠性定义为：

$$
geometry\_reliable=
(s_A\ge\tau_A)
\land
(r_{\mathrm{inlier}}\ge\tau_{\mathrm{inlier}})
\land
(e_{\mathrm{affine}}\le\tau_{\mathrm{residual}})
\land
(r_{\mathrm{sync}}\ge\tau_{\mathrm{sync}}).
$$

其中，$r_{\mathrm{reg}}$ 表示注册置信度，$r_{\mathrm{inlier}}$ 表示 attention anchor 内点比例，$r_{\mathrm{sync}}$ 表示恢复后相对关系一致性。

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
| 语义安全子空间 | 真实 JVP、分支风险预算、密钥候选方向 | $B_{\mathrm{LF}}$、$B_{\mathrm{tail}}$、$B_{\mathrm{A}}$ | semantic subspace optimization | Jacobian Null Space solver |
| LF 载体 | $B_{\mathrm{LF}}$、$K_{\mathrm{LF}}$ | $\Delta z_t^{\mathrm{LF}}$、$s_{\mathrm{LF}}$ | clean precision 主证据 | LF coder / detector |
| 尾部截断载体 | $B_{\mathrm{tail}}$、$K_{\mathrm{tail}}$、tail fraction | $\Delta z_t^{\mathrm{tail}}$、$s_{\mathrm{tail}}$ | 困难攻击条件下的补充证据 | tail robust embedder / detector |
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
