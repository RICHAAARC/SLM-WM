# Method：语义条件潜流形水印方法

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

SLM-WM 先从当前样本构造分支风险场，再通过真实 Jacobian-Vector Product（JVP）和奇异值分解（SVD）求解语义条件低响应子空间，最后把同一潜变量更新分解为三个互补分量：

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

高斯幅值尾部截断分支只比较高斯模板元素的绝对幅值与分位点，不执行 FFT、DCT、带通滤波或空间频带选择。该分支不具有空间频率语义。正式分支标识为 `tail_robust`，论文公式使用下标 $\mathrm{tail}$。

该设计由以下统一目标刻画：

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
\mathcal{B}_{\epsilon}.
$$

这里的三个分量共享同一个受约束优化目标，但使用各自的风险预算和候选方向。该实现属于项目特定设计；JVP、SVD、投影残差和固定 FPR 校准属于可迁移的通用方法。

---

## 3.3 分支语义风险场

### 3.3.1 内容属性建模

对潜空间位置 $u$ 定义内容属性向量：

$$
\phi(u)=
[
\phi_{\mathrm{sem}}(u),
\phi_{\mathrm{tex}}(u),
\phi_{\mathrm{stab}}(u),
\phi_{\mathrm{sal}}(u),
\phi_{\mathrm{attn\_stab}}(u)
].
$$

$\phi_{\mathrm{sem}}$ 表示语义显著性，$\phi_{\mathrm{tex}}$ 表示纹理复杂度，$\phi_{\mathrm{stab}}$ 表示跨采样步稳定性，$\phi_{\mathrm{sal}}$ 表示视觉显著性风险，$\phi_{\mathrm{attn\_stab}}$ 表示注意力关系稳定性。图像域语义掩码必须映射到 latent 分辨率并实际参与候选方向构造：

$$
M_z=\Pi_{x\rightarrow z}(M_x).
$$

### 3.3.2 分支风险与硬资格集合

三个分支对纹理和注意力稳定性的偏好不同，因而不能复用一个风险标量。对分支 $b\in\{\mathrm{LF},\mathrm{tail},\mathrm{A}\}$ 定义

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

LF 使用 $\psi_{\mathrm{LF}}(q)=q$，从而降低高纹理位置的优先级；尾部截断分支使用 $\psi_{\mathrm{tail}}(q)=1-q$，从而偏好稳定纹理位置；注意力几何分支主要受注意力稳定性控制。承载预算为

$$
b_b(u)=\operatorname{clip}
\left(b_{\min}+\kappa_b(1-\rho_b(u)),b_{\min},b_{\max}\right).
$$

每个分支通过独立阈值得到资格集合

$$
\Omega_b=\{u:\rho_b(u)<\tau_b\}.
$$

当 $u\notin\Omega_b$ 时，正式候选矩阵中的对应预算被置为 0；只有资格集合内部保留连续预算。该硬门控确保风险场会真实改变后续子空间，而不是仅作为日志字段。

---

## 3.4 语义条件 Jacobian Null Space

### 3.4.1 低响应子空间定义

SLM-WM 将安全方向定义为同时对语义特征和视觉特征低响应的局部方向：

$$
\mathcal{N}_{\mathrm{sem}}(z_t)
=
\left\{
v:
\|J_{\mathrm{sem}}(z_t)v\|_2\le\epsilon_s,
\|J_{\mathrm{vis}}(z_t)v\|_2\le\epsilon_v
\right\}.
$$

$J_{\mathrm{sem}}$ 对应 VAE 可微解码与冻结 CLIP 图像编码器组成的特征映射；$J_{\mathrm{vis}}$ 对应亮度、对比度、边缘和多尺度结构特征映射。模型参数冻结，但输出到 $z_t$ 的梯度保持可用。

### 3.4.2 精确 JVP 与候选方向

对候选方向 $v_i^b$ 计算

$$
\psi_{i,b}^{\mathrm{sem}}=J_{\mathrm{sem}}(z_t)v_i^b,
\qquad
\psi_{i,b}^{\mathrm{vis}}=J_{\mathrm{vis}}(z_t)v_i^b.
$$

正式实现优先通过 `torch.func.linearize` 在同一 latent 点复用精确线性算子。若底层等价算子不支持 forward AD，则使用 `torch.autograd.functional.jvp` 逐方向计算。两条路径都属于精确自动微分 JVP；有限差分或轨迹线性近似只能用于诊断。

分支候选矩阵为

$$
C_b=[v_1^b,\ldots,v_m^b]\in\mathbb{R}^{n\times m},
\qquad C_b^\top C_b=I.
$$

每个候选方向在正交化前乘入分支资格 mask 和风险预算。LF 与尾部截断分支分别把固定盲检模板作为优先候选；注意力分支把真实 Q/K 目标梯度作为优先候选；其余列由密钥化随机方向补齐。该设计避免低秩纯随机候选完全错过待嵌入方向。

### 3.4.3 SVD 求解与门禁

构造联合响应矩阵

$$
R_b=
\begin{bmatrix}
J_{\mathrm{sem}}C_b\\
\sqrt{\lambda_v}J_{\mathrm{vis}}C_b
\end{bmatrix}
\in\mathbb{R}^{q\times m}.
$$

对其执行 SVD：

$$
R_b=U\Sigma Q^\top.
$$

选择最小奇异值对应的右奇异向量 $Q_0$，再映射回 latent 空间：

$$
B_b=\operatorname{orth}(C_bQ_0).
$$

$Q_0$ 只位于候选系数空间，不能直接当作 latent 基底。正式记录必须保存绝对响应残差 $\|R_bQ_0\|_F$、相对平均响应残差和正交误差 $\|B_b^\top B_b-I\|_F$。当前协议要求选中方向的平均响应不超过全部候选平均响应的 0.75，固定模板投影后的能量保留比例不低于 0.01。门禁失败时直接停止运行，不能使用 one-hot、周期平铺或近零投影归一化作为替代。

---

## 3.5 LF 与高斯幅值尾部截断内容载体

### 3.5.1 空间低通 LF 主证据

给定密钥 $K_{\mathrm{LF}}$、公开模型标识和 latent 形状，生成固定标准高斯模板并执行空间平均池化：

$$
\nu_{\mathrm{LF}}
=
\operatorname{Norm}
\left(
\operatorname{AvgPool}_{k\times k}
(\operatorname{PRG}_{\mathcal N}(K_{\mathrm{LF}},M,shape))
\right).
$$

平均池化在 latent 的二维空间轴上抑制快速空间变化，因此 LF 分支具有明确的空间低通定义。嵌入端把固定模板投影到分支安全子空间：

$$
\overline\nu_{\mathrm{LF}}
=B_{\mathrm{LF}}B_{\mathrm{LF}}^\top\nu_{\mathrm{LF}},
$$

并构造

$$
\Delta z_t^{\mathrm{LF}}
=
\alpha_t^{\mathrm{LF}}
\frac{\overline\nu_{\mathrm{LF}}}{\|\overline\nu_{\mathrm{LF}}\|_2}.
$$

### 3.5.2 高斯幅值尾部截断补充证据

给定密钥 $K_{\mathrm{tail}}$、公开模型标识和 latent 形状，生成固定标准高斯模板：

$$
\nu_{\mathrm{tail}}
=
\operatorname{PRG}_{\mathcal N}(K_{\mathrm{tail}},M,shape).
$$

对冻结的尾部比例 $\gamma$，计算绝对幅值的 $1-\gamma$ 分位点，并仅保留大幅值元素：

$$
q_{1-\gamma}
=
\operatorname{Quantile}_{1-\gamma}(|\nu_{\mathrm{tail}}|),
$$

$$
\widetilde\nu_{\mathrm{tail},i}
=
\nu_{\mathrm{tail},i}
\mathbb{I}
\left(|\nu_{\mathrm{tail},i}|\ge q_{1-\gamma}\right).
$$

嵌入更新为

$$
\Delta z_t^{\mathrm{tail}}
=
\alpha_t^{\mathrm{tail}}
\operatorname{Norm}
\left(
B_{\mathrm{tail}}B_{\mathrm{tail}}^\top
\widetilde\nu_{\mathrm{tail}}
\right).
$$

该算子定义的是**概率分布幅值域的稀疏尾部选择**。元素索引没有按空间频率排序，保留集合也不是 Fourier 或余弦基上的频带。截断后的模板可能包含宽频谱成分，因此不能把“幅值大”解释为“空间频率高”。其鲁棒性是需要通过压缩、噪声、重采样和再扩散实验检验的假设，不能由“高频”名称先验推出。

### 3.5.3 仅图像内容分数

检测端从待检图像通过 VAE 编码得到 $\widehat z$，并使用密钥和公开配置重建两个固定模板：

$$
s_{\mathrm{LF}}=\operatorname{Corr}(\widehat z,\nu_{\mathrm{LF}}),
\qquad
s_{\mathrm{tail}}=\operatorname{Corr}(\widehat z,\widetilde\nu_{\mathrm{tail}}).
$$

统一内容分数为

$$
s_c
=
\lambda_{\mathrm{LF}}s_{\mathrm{LF}}
+
\lambda_{\mathrm{tail}}s_{\mathrm{tail}},
\qquad
\lambda_{\mathrm{LF}}>\lambda_{\mathrm{tail}},
\qquad
\lambda_{\mathrm{LF}}+\lambda_{\mathrm{tail}}=1.
$$

两个分支不分别设置独立正判阈值后投票。安全子空间只在嵌入端控制失真；检测端不恢复样本级 $B_{\mathrm{LF}}$ 或 $B_{\mathrm{tail}}$。

---

## 3.6 Self-Attention 相对关系几何锚点

### 3.6.1 真实 Q/K 关系图

对第 $t$ 步、第 $\ell$ 层 Transformer block，Self-Attention 为

$$
A_t^{(\ell)}
=
\operatorname{softmax}
\left(
\frac{Q_t^{(\ell)}K_t^{(\ell)\top}}{\sqrt d}
\right).
$$

正式实现直接调用模型的 `to_q` 与 `to_k` 投影，不使用合成 attention map。密钥确定 token 对与目标关系符号，形成注意力目标损失 $\mathcal L_A$。通过 autograd 得到

$$
g_A=\nabla_{z_t}\mathcal L_A.
$$

### 3.6.2 安全投影与单调回溯

注意力候选方向进入其独立分支的 Jacobian Null Space 求解器，得到 $B_{\mathrm A}$。投影梯度为

$$
\overline g_A=B_{\mathrm A}B_{\mathrm A}^\top g_A.
$$

以 $-\overline g_A$ 为更新方向执行回溯搜索，仅接受同时满足强度预算且使目标损失单调下降的步长：

$$
\mathcal L_A(z_t+\Delta z_t^{\mathrm A})
<
\mathcal L_A(z_t).
$$

若所有候选步长都不能降低损失，运行必须报告失败，不能写入未验证的 attention 更新。

### 3.6.3 几何恢复

检测端只从待检图像提取公开视觉模型的 token 关系和密钥关系签名，通过匹配、三点 RANSAC 与仿射估计得到参考系变换 $\widehat T$。几何链输出注册置信度、锚点内点比例、同步一致性和对齐残差。它只负责参考系恢复，不直接产生 positive 判定。

---

## 3.7 完整 fixed-FPR 判定

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

fixed-FPR 约束的是包含 rescue 的完整 evidence 判定，而不是只冻结内容阈值后任意增加第二条判定路径。内容阈值、几何可靠性阈值、rescue window 和失败原因 gate 均在 calibration 或预注册协议中冻结。test split 只能应用冻结协议并报告 clean negative 的二项分布置信上界，不参与任何参数选择。

当前三类运行配置为：

| Prompt 数量 | dev | calibration | test | 目标 FPR |
| ---: | ---: | ---: | ---: | ---: |
| 70 | 3 | 33 | 34 | 0.1 |
| 700 | 30 | 330 | 340 | 0.01 |
| 7000 | 300 | 3300 | 3400 | 0.001 |

---

## 3.8 正式机制消融

正式消融必须对改变后的机制配置重新生成图像、重新执行攻击并重新运行仅图像检测。禁止通过修改历史分数或保留率模拟机制被移除后的结果。与本文核心算子直接对应的消融至少包括：

1. 共享全局风险对比分支风险；
2. 移除语义 JVP 或以随机基底替代 Jacobian Null Space；
3. LF-only；
4. Tail-only；
5. No-Tail；
6. No-Tail-Truncation；
7. 移除 Q/K attention anchor；
8. 移除同阈值几何救回。

---

## 3.9 嵌入与检测流程

嵌入端执行以下步骤：

1. 在选定采样步提取 latent、可微语义/视觉特征和真实 Q/K attention；
2. 构造 `lf_content`、`tail_robust` 和 `attention_geometry` 三个分支风险场；
3. 为每个分支构造包含优先载体方向的候选矩阵；
4. 通过精确 JVP、联合响应矩阵和 SVD 求解三个低响应子空间；
5. 构造空间 LF、高斯幅值尾部截断和 attention 几何更新；
6. 对 attention 更新执行单调回溯，对内容投影执行能量门禁；
7. 生成最终图像并记录算子摘要、残差、投影能量和环境信息。

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

方法实现存在不等于论文结论成立。空间 LF 的有效性、高斯幅值尾部截断的攻击鲁棒性、Q/K 几何恢复的增益和完整 fixed-FPR 均必须由真实 GPU 生成、clean negative、真实攻击、正式机制重跑消融、外部 baseline 和受治理结果包共同支撑。
