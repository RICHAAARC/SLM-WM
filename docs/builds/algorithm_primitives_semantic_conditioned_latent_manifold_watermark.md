# 算法原语：语义条件潜流形水印的统一设计

## 一、文档定位

本文档面向一个从零开始构建的文本到图像扩散模型水印项目，给出算法原语（algorithmic primitives）层面的统一设计。本文档不以工程模块清单为中心，而以“image processing theory + latent manifold watermark + robust detection statistics + semantic subspace optimization”为主线，定义水印扰动如何在扩散潜空间中被选择、分解、注入、检测与归因。

本文档采用语义条件潜流形水印（Semantic-conditioned Latent Manifold Watermarking，SLM-WM）作为方法名称。SLM-WM 的核心思想是：水印不是多个外部技巧的并列组合，而是在当前样本的语义风险场与扩散轨迹稳定性约束下，求解一个受限潜空间水印方向。低频（Low-Frequency，LF）内容证据、高频（High-Frequency，HF）补充证据和 Self-Attention 几何锚点均由同一个语义条件安全子空间导出。

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
\mathcal{N}_{\mathrm{sem}}(z_t,p)
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
\Delta z_t^{\mathrm{HF}}
+
\Delta z_t^{\mathrm{A}},
$$

其中，$\Delta z_t^{\mathrm{LF}}$ 为低频稳定主证据，$\Delta z_t^{\mathrm{HF}}$ 为高频鲁棒补充证据，$\Delta z_t^{\mathrm{A}}$ 为 Self-Attention 相对关系几何锚点证据。该分解来自同一安全子空间，而不是三个独立水印器的简单叠加。

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

该映射必须进入后续轨迹特征算子与子空间规划，不能仅作为可视化或日志字段。

### （三）风险场与承载预算

定义语义风险场：

$$
\rho(u)=
\eta_s\phi_{\mathrm{sal}}(u)
+
\eta_m\phi_{\mathrm{sem}}(u)
-
\eta_t\phi_{\mathrm{tex}}(u)
-
\eta_r\phi_{\mathrm{stab}}(u).
$$

风险越低，区域越适合承载水印。承载预算定义为

$$
b(u)=\operatorname{clip}\left(b_{\min}+\kappa(1-\rho(u)),b_{\min},b_{\max}\right).
$$

据此得到三个候选区域：

$$
\Omega_{\mathrm{LF}}=\{u:\phi_{\mathrm{tex}}(u)<\tau_{\mathrm{tex}},\ \rho(u)<\tau_{\rho}\},
$$

$$
\Omega_{\mathrm{HF}}=\{u:\phi_{\mathrm{tex}}(u)\ge\tau_{\mathrm{tex}},\ \phi_{\mathrm{stab}}(u)\ge\tau_{\mathrm{stab}}\},
$$

$$
\Omega_{\mathrm{A}}=\{u:\operatorname{AttnStable}(u)\ge\tau_A,\ \phi_{\mathrm{stab}}(u)\ge\tau_{\mathrm{stab}}\}.
$$

$\Omega_{\mathrm{A}}$ 可被设置为 $\Omega_{\mathrm{LF}}\cup\Omega_{\mathrm{HF}}$ 的子集，也可作为独立同步候选区；实现与论文中必须显式给出两者关系。

---

## 四、原语 2：语义条件安全子空间优化

### （一）语义条件 Null Space

SLM-WM 将固定统计 Null Space 推广为样本条件化的语义安全子空间：

$$
\mathcal{N}_{\mathrm{sem}}(z_t,p)
=
\{v:
\|W_{\mathrm{sem}}J_{\mathrm{sem}}(z_t,p)v\|_2\le\epsilon_s,\ 
\|W_{\mathrm{vis}}J_{\mathrm{vis}}(z_t,p)v\|_2\le\epsilon_v
\}.
$$

其中，$J_{\mathrm{sem}}$ 表示语义特征映射的 Jacobian，$J_{\mathrm{vis}}$ 表示视觉质量相关映射的 Jacobian，$W_{\mathrm{sem}}$ 和 $W_{\mathrm{vis}}$ 由风险场 $\rho(u)$ 与承载预算 $b(u)$ 构造。该定义要求候选方向在语义特征与视觉质量空间中均为低响应方向。

### （二）JVP 与轨迹线性化

为避免显式构造完整 Jacobian，系统通过 Jacobian-Vector Product（JVP）估计局部响应。对候选方向 $v_i$：

$$
\psi_i^{\mathrm{sem}}=J_{\mathrm{sem}}(z_t,p)v_i,\qquad
\psi_i^{\mathrm{vis}}=J_{\mathrm{vis}}(z_t,p)v_i.
$$

若完整 autograd 代价过高，可采用 trajectory-linearized JVP approximation。首先定义轨迹特征：

$$
\varphi_t=P^\top\operatorname{vec}(\operatorname{Norm}(M_z\odot z_t)),
$$

随后拟合局部转移算子：

$$
\varphi_{t-1}\approx A_t\varphi_t.
$$

此时 $A_t v$ 可作为采样轨迹传播方向的近似响应。论文中必须明确该近似不是完整 denoiser Jacobian，而是面向可实现性的轨迹线性化估计。

### （三）安全基底求解

构造响应矩阵：

$$
D=
\begin{bmatrix}
W_{\mathrm{sem}}J_{\mathrm{sem}}v_1 & \cdots & W_{\mathrm{sem}}J_{\mathrm{sem}}v_m\\
W_{\mathrm{vis}}J_{\mathrm{vis}}v_1 & \cdots & W_{\mathrm{vis}}J_{\mathrm{vis}}v_m
\end{bmatrix}^{\top}.
$$

对 $D$ 执行奇异值分解（Singular Value Decomposition，SVD）：

$$
D=U\Sigma V^\top.
$$

取小奇异值对应方向构成安全基底：

$$
B_{\mathrm{safe}}=V_{[:,r+1:m]}.
$$

再由路由区域获得子基底：

$$
B_{\mathrm{LF}}=\operatorname{Proj}_{\Omega_{\mathrm{LF}}}(B_{\mathrm{safe}}),\qquad
B_{\mathrm{HF}}=\operatorname{Proj}_{\Omega_{\mathrm{HF}}}(B_{\mathrm{safe}}),\qquad
B_{\mathrm{A}}=\operatorname{Proj}_{\Omega_{\mathrm{A}}}(B_{\mathrm{safe}}).
$$

该原语是方法统一性的核心：LF、HF 和 attention geometry 均从 $B_{\mathrm{safe}}$ 或其投影中产生。

---

## 五、原语 3：LF 主证据载体

### （一）角色定义

LF 分支负责 clean 条件下的高精度主证据，目标是低误报、低感知损伤和轻度失真下的稳定检测。它采用密钥化伪随机模板，但模板被限制在语义条件安全子空间中。

### （二）模板生成与嵌入

给定 LF 密钥 $K_{\mathrm{LF}}$、事件摘要 $d_e$、安全基底摘要 $d_B$ 和路由摘要 $d_R$，生成伪随机模板：

$$
\nu_{\mathrm{LF}}=\operatorname{PRG}(K_{\mathrm{LF}},d_e,d_B,d_R).
$$

经编码函数 $C_{\mathrm{LF}}$ 得到低频编码模板：

$$
\nu_{\mathrm{LF}}^c=C_{\mathrm{LF}}(\nu_{\mathrm{LF}}).
$$

LF latent update 为

$$
\Delta z_t^{\mathrm{LF}}
=
\alpha_t^{\mathrm{LF}}B_{\mathrm{LF}}\nu_{\mathrm{LF}}^c.
$$

其中，$\alpha_t^{\mathrm{LF}}$ 为采样步强度调度。

### （三）检测统计量

检测端估计待检样本的 LF 系数 $\hat{c}_{\mathrm{LF}}$，并计算归一化相关统计量：

$$
s_{\mathrm{LF}}
=
\frac{\langle \hat{c}_{\mathrm{LF}},\nu_{\mathrm{LF}}^c\rangle}
{\|\hat{c}_{\mathrm{LF}}\|_2\|\nu_{\mathrm{LF}}^c\|_2}.
$$

若使用纠错码或软解码，其 bit-level agreement 和 codeword consistency 仅作为诊断统计量，不替代正式内容分数。

---

## 六、原语 4：HF 鲁棒补充载体

### （一）角色定义

HF 分支负责纹理区域和 harder regime 下的鲁棒补充，尤其面向压缩、噪声、重采样、裁剪重缩放和再扩散后仍可能保留的残余证据。HF 不单独承担主判定。

### （二）尾部截断 carrier

给定 HF 密钥 $K_{\mathrm{HF}}$，生成候选模板：

$$
\nu_{\mathrm{HF}}=\operatorname{PRG}(K_{\mathrm{HF}},d_e,d_B,d_R).
$$

根据纹理权重、稳定性权重和攻击敏感性权重构造 $w_i$，并执行尾部截断：

$$
\widetilde{\nu}_{\mathrm{HF},i}
=
\nu_{\mathrm{HF},i}\cdot
\mathbb{I}(|\nu_{\mathrm{HF},i}|w_i\ge q_{\gamma}).
$$

HF latent update 为

$$
\Delta z_t^{\mathrm{HF}}
=
\alpha_t^{\mathrm{HF}}B_{\mathrm{HF}}\widetilde{\nu}_{\mathrm{HF}}.
$$

### （三）检测统计量

HF 分数定义为截断模板相关性与稳定性修正项的乘积：

$$
s_{\mathrm{HF}}
=
\operatorname{Corr}(\hat{c}_{\mathrm{HF}},\widetilde{\nu}_{\mathrm{HF}})
\cdot
\operatorname{Stab}(\hat{c}_{\mathrm{HF}};\Omega_{\mathrm{HF}}).
$$

内容分数由 LF 与 HF 统一融合：

$$
s_c=
\lambda_{\mathrm{LF}}s_{\mathrm{LF}}+
\lambda_{\mathrm{HF}}s_{\mathrm{HF}},
\qquad
\lambda_{\mathrm{LF}}>\lambda_{\mathrm{HF}},
\qquad
\lambda_{\mathrm{LF}}+\lambda_{\mathrm{HF}}=1.
$$

LF 和 HF 不应分别设置独立正判阈值后投票，否则会破坏 fixed-FPR 判定的统计清晰性。

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

给定目标关系扰动 $r_{ij}^{\star}$，定义几何关系损失：

$$
\mathcal{L}_A
=
\sum_{(i,j)\in\mathcal{E}_A}
\|r_{ij}(z_t+\Delta z_t)-r_{ij}^{\star}\|_2^2.
$$

几何更新方向为

$$
\Delta z_t^{\mathrm{A}}
=
-\alpha_t^{\mathrm{A}}
\operatorname{Proj}_{\mathcal{N}_{\mathrm{sem}}}
(\nabla_{z_t}\mathcal{L}_A).
$$

该投影保证几何锚点仍属于同一语义安全子空间。

### （四）几何可靠性统计量

检测端从待检图像的反演 latent、重采样轨迹或辅助估计中提取 attention relation graph，估计参考系恢复参数 $\hat{T}$，并计算：

$$
geometry\_reliable=
(r_{\mathrm{reg}}\ge\tau_{\mathrm{reg}})
\land
(r_{\mathrm{inlier}}\ge\tau_{\mathrm{inlier}})
\land
(r_{\mathrm{sync}}\ge\tau_{\mathrm{sync}}).
$$

其中，$r_{\mathrm{reg}}$ 表示注册置信度，$r_{\mathrm{inlier}}$ 表示 attention anchor 内点比例，$r_{\mathrm{sync}}$ 表示恢复后相对关系一致性。

---

## 八、原语 6：鲁棒检测统计与固定 FPR 判定

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

该边界保证几何链仍是参考系恢复机制，而不是绕过内容阈值的第二条 positive 判定路径。

---

## 九、原语 7：事件 attestation 与 payload probe 边界

事件 attestation 用于 final-level 归因一致性，不替代 watermark 主证据。设事件声明为 $e$，事件摘要为 $d_e$，签名为 $\sigma(e)$，则

$$
attestation\_pass=\mathbb{I}[\operatorname{Verify}_K(d_e,\sigma(e))=1].
$$

Final-level 判定为

$$
y_{\mathrm{final}}=y_{\mathrm{evidence}}\land attestation\_pass.
$$

Payload probe 仅用于 bit-level agreement、codeword consistency 和失效模式诊断，不进入 $y_{\mathrm{evidence}}$ 或 $y_{\mathrm{final}}$。

---

## 十、算法原语之间的非组合关系

SLM-WM 必须满足以下不变量：

1. 所有水印方向均由 $\mathcal{N}_{\mathrm{sem}}\cap\mathcal{M}_{\mathrm{route}}$ 约束；
2. LF 与 HF 是同一安全流形中的互补证据分解；
3. Self-Attention 几何锚点是同一 latent update 的几何分量；
4. 几何链只恢复参考系，不直接判定 positive；
5. 恢复后内容重判复用同一内容阈值 $\tau_c$；
6. Attestation 只进入 final-level；
7. Payload probe 只作为诊断统计量；
8. 所有正式主张必须能够由固定 FPR、clean negative、attack family metrics 和机制消融支撑。

---

## 十一、算法原语清单

| 原语 | 输入 | 输出 | 论文角色 | 实现角色 |
|---|---|---|---|---|
| 语义条件潜流形 | $z_t$、$p$、$M_x$、attention maps | $M_z$、$\rho(u)$、$b(u)$、$\Omega_{\mathrm{LF/HF/A}}$ | 内容自适应理论基础 | semantic router |
| 语义安全子空间 | 轨迹特征、JVP、风险权重 | $B_{\mathrm{safe}}$、$B_{\mathrm{LF}}$、$B_{\mathrm{HF}}$、$B_{\mathrm{A}}$ | semantic subspace optimization | subspace planner |
| LF 载体 | $B_{\mathrm{LF}}$、$K_{\mathrm{LF}}$、$d_e$ | $\Delta z_t^{\mathrm{LF}}$、$s_{\mathrm{LF}}$ | clean precision 主证据 | LF coder / detector |
| HF 载体 | $B_{\mathrm{HF}}$、$K_{\mathrm{HF}}$、tail truncation | $\Delta z_t^{\mathrm{HF}}$、$s_{\mathrm{HF}}$ | harder regime 补充证据 | HF embedder / detector |
| Attention 几何锚点 | Self-Attention maps、稳定 token | $\Delta z_t^{\mathrm{A}}$、$r_{\mathrm{sync}}$ | 几何同步创新 | attention anchor module |
| 鲁棒检测统计 | $s_{\mathrm{LF}}$、$s_{\mathrm{HF}}$、geometry stats | $y_{\mathrm{evidence}}$ | fixed-FPR 主判 | decision module |
| Attestation | 事件摘要、签名、evidence digest | $y_{\mathrm{final}}$ | final-level 归因 | attestation module |

---

## 十二、最终建议

实现与论文写作应始终围绕以下链条展开：

$$
\text{semantic risk field}
\rightarrow
\text{semantic-conditioned safe null-space}
\rightarrow
\text{LF/HF/attention carrier decomposition}
\rightarrow
\text{fixed-FPR robust detection}
\rightarrow
\text{same-threshold geometric rescue}
\rightarrow
\text{attested final attribution}.
$$

该链条在公式、代码接口、消融实验和论文叙事中必须保持一致。只有这样，SLM-WM 才能被清晰地定义为一个统一的 image processing theory 驱动方法，而不是工程组件堆叠。
