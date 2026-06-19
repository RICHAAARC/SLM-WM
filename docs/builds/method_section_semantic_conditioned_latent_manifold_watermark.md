# Method：语义条件潜流形水印方法

## 3.1 问题定义

本文研究文本到图像扩散模型中的鲁棒水印归因问题。给定文本提示 $p$、扩散生成模型 $G_\theta$、采样更新算子 $S_\theta$、随机种子与初始噪声 $\xi$，无水印生成过程表示为

$$
z_{t-1}=S_\theta(z_t,p,t),\qquad x=D(z_0),
$$

其中，$z_t$ 为第 $t$ 步潜变量，$D$ 表示 VAE decoder 或最终图像解码算子。本文目标是在扩散采样过程中嵌入不可见且鲁棒的水印证据，使生成图像在常规失真、几何攻击和再扩散类攻击后仍可被可靠归因。

与生成后图像域水印不同，本文将水印嵌入定义为采样轨迹中的受约束 latent update：

$$
\tilde{z}_t=z_t+\Delta z_t,\qquad
z_{t-1}=S_\theta(\tilde{z}_t,p,t).
$$

因此，核心问题不是如何在像素域叠加扰动，而是如何在不破坏语义内容与视觉质量的前提下，选择当前样本最稳定、最隐蔽、最可恢复的潜空间水印方向。为此，本文提出语义条件潜流形水印方法（Semantic-conditioned Latent Manifold Watermarking，SLM-WM）。该方法将内容自适应区域划分、语义条件子空间优化、LF/HF 互补载体、Self-Attention 相对关系几何锚点和鲁棒统计判定统一到同一个潜流形水印框架中。

本文输出三类判定结果：evidence-level 判定用于回答 watermark 主证据是否成立；attestation-level 判定用于回答主证据成立后事件级归因是否一致；final-level 判定用于回答 watermark 主证据与事件约束是否同时成立。Payload probe 仅用于 bit-level agreement 与 codeword consistency 诊断，不进入正式主判定。

---

## 3.2 方法总览

SLM-WM 的基本思想是：首先根据当前样本的语义显著性、纹理复杂度和扩散轨迹稳定性构造语义条件潜流形；随后在该流形内估计语义安全 Null Space；最后将同一潜变量扰动分解为 LF 主证据、HF 鲁棒补充和 Self-Attention 几何锚点三个互补分量：

$$
\Delta z_t
=
\Delta z_t^{\mathrm{LF}}
+
\Delta z_t^{\mathrm{HF}}
+
\Delta z_t^{\mathrm{A}}.
$$

其中，$\Delta z_t^{\mathrm{LF}}$ 负责低频稳定主证据，$\Delta z_t^{\mathrm{HF}}$ 负责纹理区域和 harder regime 下的鲁棒补充，$\Delta z_t^{\mathrm{A}}$ 负责基于 Self-Attention 相对关系的几何同步。

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
\mathcal{N}_{\mathrm{sem}}(z_t,p)
\cap
\mathcal{M}_{\mathrm{route}}(z_t,p)
\cap
\mathcal{B}_{\epsilon}.
$$

其中，$\mathcal{N}_{\mathrm{sem}}$ 表示语义条件安全子空间，$\mathcal{M}_{\mathrm{route}}$ 表示样本条件路由流形，$\mathcal{B}_{\epsilon}$ 表示扰动强度预算。该公式说明 LF、HF 与几何锚点并非独立拼接模块，而是同一个受约束水印优化问题在不同证据方向上的分解。

---

## 3.3 语义条件潜流形构造

### 3.3.1 内容属性建模

对采样过程中潜空间位置 $u$，定义内容属性向量：

$$
\phi(u)=
[
\phi_{\mathrm{sem}}(u),
\phi_{\mathrm{tex}}(u),
\phi_{\mathrm{stab}}(u),
\phi_{\mathrm{sal}}(u)
],
$$

其中，$\phi_{\mathrm{sem}}$ 表示语义显著性，$\phi_{\mathrm{tex}}$ 表示纹理复杂度，$\phi_{\mathrm{stab}}$ 表示跨采样步稳定性，$\phi_{\mathrm{sal}}$ 表示视觉显著性风险。若语义掩码由图像域 saliency model、预测干净图像 $\hat{x}_0^t$ 或分割模型得到，则通过映射 $\Pi_{x\rightarrow z}$ 得到 latent mask：

$$
M_z=\Pi_{x\rightarrow z}(M_x).
$$

$M_z$ 用于约束轨迹特征提取和子空间规划，而不是仅作为记录字段。

### 3.3.2 语义风险场

定义嵌入风险场：

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

风险越低，区域越适合承载水印。承载预算为

$$
b(u)=\operatorname{clip}\left(b_{\min}+\kappa(1-\rho(u)),b_{\min},b_{\max}\right).
$$

基于风险场与承载预算，构造三类候选区域：

$$
\Omega_{\mathrm{LF}}=\{u:\phi_{\mathrm{tex}}(u)<\tau_{\mathrm{tex}},\ \rho(u)<\tau_{\rho}\},
$$

$$
\Omega_{\mathrm{HF}}=\{u:\phi_{\mathrm{tex}}(u)\ge\tau_{\mathrm{tex}},\ \phi_{\mathrm{stab}}(u)\ge\tau_{\mathrm{stab}}\},
$$

$$
\Omega_{\mathrm{A}}=\{u:\operatorname{AttnStable}(u)\ge\tau_A,\ \phi_{\mathrm{stab}}(u)\ge\tau_{\mathrm{stab}}\}.
$$

$\Omega_{\mathrm{LF}}$ 偏向平滑稳定区域，适合承载低感知主证据；$\Omega_{\mathrm{HF}}$ 偏向纹理复杂区域，适合承载高频鲁棒补充；$\Omega_{\mathrm{A}}$ 由 attention 稳定性和局部结构稳定性共同决定，用于几何同步锚点。

---

## 3.4 语义条件安全子空间优化

### 3.4.1 语义条件 Null Space

固定统计子空间难以区分不同内容区域的视觉风险。本文将其推广为语义条件安全子空间：

$$
\mathcal{N}_{\mathrm{sem}}(z_t,p)
=
\{v:
\|W_{\mathrm{sem}}J_{\mathrm{sem}}(z_t,p)v\|_2\le\epsilon_s,
\|W_{\mathrm{vis}}J_{\mathrm{vis}}(z_t,p)v\|_2\le\epsilon_v
\}.
$$

其中，$J_{\mathrm{sem}}$ 表示语义特征映射的 Jacobian，$J_{\mathrm{vis}}$ 表示视觉质量相关映射的 Jacobian，$W_{\mathrm{sem}}$ 和 $W_{\mathrm{vis}}$ 由语义风险场 $\rho(u)$ 与承载预算 $b(u)$ 构造。

### 3.4.2 JVP 估计与轨迹特征

对候选方向 $v_i$，通过 JVP 估计局部响应：

$$
\psi_i^{\mathrm{sem}}=J_{\mathrm{sem}}(z_t,p)v_i,\qquad
\psi_i^{\mathrm{vis}}=J_{\mathrm{vis}}(z_t,p)v_i.
$$

同时，从采样轨迹中提取 latent feature：

$$
\varphi_t=P^\top\operatorname{vec}\left(\operatorname{Norm}(M_z\odot z_t)\right).
$$

其中，$P$ 表示随机投影或低维特征投影矩阵，$M_z$ 为 latent semantic mask。该定义保证子空间规划受语义显著性显式约束。

### 3.4.3 安全基底求解

构造加权响应矩阵：

$$
D=
\begin{bmatrix}
W_{\mathrm{sem}}J_{\mathrm{sem}}v_1 & \cdots & W_{\mathrm{sem}}J_{\mathrm{sem}}v_m\\
W_{\mathrm{vis}}J_{\mathrm{vis}}v_1 & \cdots & W_{\mathrm{vis}}J_{\mathrm{vis}}v_m
\end{bmatrix}^{\top}.
$$

对 $D$ 执行 SVD：

$$
D=U\Sigma V^\top.
$$

取小奇异值方向构成安全基底：

$$
B_{\mathrm{safe}}=V_{[:,r+1:m]}.
$$

再根据语义路由得到

$$
B_{\mathrm{LF}}=\operatorname{Proj}_{\Omega_{\mathrm{LF}}}(B_{\mathrm{safe}}),
$$

$$
B_{\mathrm{HF}}=\operatorname{Proj}_{\Omega_{\mathrm{HF}}}(B_{\mathrm{safe}}),
$$

$$
B_{\mathrm{A}}=\operatorname{Proj}_{\Omega_{\mathrm{A}}}(B_{\mathrm{safe}}).
$$

---

## 3.5 LF/HF 互补内容载体

### 3.5.1 LF 主证据分支

给定 LF 密钥 $K_{\mathrm{LF}}$、事件摘要 $d_e$、子空间摘要 $d_B$ 和路由摘要 $d_R$，生成密钥化低频模板：

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

LF 分支借鉴伪随机编码水印的隐蔽性思想，但其 carrier 被限制在语义条件安全子空间中，因此不等同于独立的图像域编码水印。

### 3.5.2 HF 鲁棒补充分支

给定 HF 密钥 $K_{\mathrm{HF}}$，生成候选模板：

$$
\nu_{\mathrm{HF}}=\operatorname{PRG}(K_{\mathrm{HF}},d_e,d_B,d_R).
$$

对其执行 tail truncation：

$$
\widetilde{\nu}_{\mathrm{HF},i}
=
\nu_{\mathrm{HF},i}\cdot
\mathbb{I}\left(|\nu_{\mathrm{HF},i}|w_i\ge q_{\gamma}\right),
$$

其中，$w_i$ 由纹理复杂度、局部稳定性和攻击敏感性共同决定，$q_\gamma$ 为分位数阈值。HF latent update 为

$$
\Delta z_t^{\mathrm{HF}}
=
\alpha_t^{\mathrm{HF}}B_{\mathrm{HF}}\widetilde{\nu}_{\mathrm{HF}}.
$$

### 3.5.3 内容分数

检测端分别计算 LF 与 HF 分数：

$$
s_{\mathrm{LF}}
=
\frac{\langle \hat{c}_{\mathrm{LF}},\nu_{\mathrm{LF}}^c\rangle}
{\|\hat{c}_{\mathrm{LF}}\|_2\|\nu_{\mathrm{LF}}^c\|_2},
$$

$$
s_{\mathrm{HF}}
=
\operatorname{Corr}(\hat{c}_{\mathrm{HF}},\widetilde{\nu}_{\mathrm{HF}})
\cdot
\operatorname{Stab}(\hat{c}_{\mathrm{HF}};\Omega_{\mathrm{HF}}).
$$

最终内容分数为

$$
s_c
=
\lambda_{\mathrm{LF}}s_{\mathrm{LF}}
+
\lambda_{\mathrm{HF}}s_{\mathrm{HF}},
\qquad
\lambda_{\mathrm{LF}}>\lambda_{\mathrm{HF}},
\qquad
\lambda_{\mathrm{LF}}+\lambda_{\mathrm{HF}}=1.
$$

---

## 3.6 Self-Attention 相对关系几何锚点

### 3.6.1 Attention graph

对第 $t$ 步、第 $\ell$ 层 transformer block，Self-Attention 为

$$
A_t^{(\ell)}
=
\operatorname{softmax}\left(
\frac{Q_t^{(\ell)}{K_t^{(\ell)\top}}}{\sqrt{d}}
\right).
$$

选择稳定 attention token：

$$
\mathcal{V}_A=
\{i:\operatorname{Stab}(A_{\cdot,i}^{(\ell)})\ge\tau_A,
\operatorname{Sal}(i)\ge\tau_s\}.
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

得到 attention-relative graph：

$$
\mathcal{G}_A=(\mathcal{V}_A,\mathcal{E}_A,\{r_{ij}\}).
$$

### 3.6.2 几何锚点嵌入

将几何锚点绑定在 attention 相对关系上。给定目标关系扰动 $r_{ij}^{\star}$，定义

$$
\mathcal{L}_A
=
\sum_{(i,j)\in\mathcal{E}_A}
\|r_{ij}(z_t+\Delta z_t)-r_{ij}^{\star}\|_2^2.
$$

几何 latent update 为

$$
\Delta z_t^{\mathrm{A}}
=
-\alpha_t^{\mathrm{A}}
\operatorname{Proj}_{\mathcal{N}_{\mathrm{sem}}}
(\nabla_{z_t}\mathcal{L}_A).
$$

检测端通过 attention relation graph 估计参考系恢复参数 $\hat{T}$，并输出 `registration_confidence`、`anchor_inlier_ratio`、`recovered_sync_consistency` 和 `alignment_residual` 等统计量。

---

## 3.7 鲁棒检测与几何救回

### 3.7.1 Fixed-FPR 内容判定

给定 calibration split 中的 clean negative 分数分布，内容阈值由固定 FPR 原则确定：

$$
\tau_c=Q_{1-\alpha}(s_c\mid\mathcal{D}_0^{\mathrm{cal}}).
$$

对待检图像 $y$，原始内容分数为

$$
s_c^{\mathrm{raw}}
=
\lambda_{\mathrm{LF}}s_{\mathrm{LF}}^{\mathrm{raw}}
+
\lambda_{\mathrm{HF}}s_{\mathrm{HF}}^{\mathrm{raw}}.
$$

原始内容边界余量为

$$
m_c^{\mathrm{raw}}=s_c^{\mathrm{raw}}-\tau_c.
$$

原始内容正判为

$$
positive\_by\_content=\mathbb{I}(m_c^{\mathrm{raw}}\ge0).
$$

### 3.7.2 几何可靠性

几何恢复可信条件为

$$
geometry\_reliable
=
(r_{\mathrm{reg}}\ge\tau_{\mathrm{reg}})
\land
(r_{\mathrm{inlier}}\ge\tau_{\mathrm{inlier}})
\land
(r_{\mathrm{sync}}\ge\tau_{\mathrm{sync}}).
$$

### 3.7.3 Rescue 规则

只对边界失败样本开放 rescue：

$$
rescue\_eligible
=
(\delta_{\mathrm{low}}\le m_c^{\mathrm{raw}}<0)
\land geometry\_reliable
\land fail\_reason\in\{geometry\_suspected,low\_confidence\}.
$$

对恢复后表示 $y^{\mathrm{align}}$ 重新提取内容分数：

$$
s_c^{\mathrm{align}}
=
\lambda_{\mathrm{LF}}s_{\mathrm{LF}}^{\mathrm{align}}
+
\lambda_{\mathrm{HF}}s_{\mathrm{HF}}^{\mathrm{align}}.
$$

恢复后内容边界余量为

$$
m_c^{\mathrm{align}}=s_c^{\mathrm{align}}-\tau_c.
$$

几何救回成立为

$$
rescue\_applied
=
rescue\_eligible
\land
\mathbb{I}(m_c^{\mathrm{align}}\ge0).
$$

Evidence-level 判定为

$$
y_{\mathrm{evidence}}
=
positive\_by\_content
\lor
rescue\_applied.
$$

几何链只负责参考系恢复，最终仍由内容链在相同阈值下重判。

### 3.7.4 fixed-FPR 与 rescue 的完整统计边界

本文中的 fixed-FPR 不是只固定 $\tau_c$ 后再任意追加 rescue 分支。正式 operating point 必须对应一个在 calibration split 上冻结的完整 evidence-level 判定协议。具体要求如下：

1. 内容阈值 $\tau_c$ 由 clean negative calibration 分布确定；
2. 几何可靠性阈值、rescue window 与 fail reason gate 在 calibration 或预注册协议中冻结；
3. test split 只用于报告，不用于选择 $\tau_c$、$\delta_{\mathrm{low}}$ 或几何 gate；
4. rescue 后的整体 evidence-level FPR 必须在 clean negative 和 attacked negative 上分别报告；
5. 若 raw content decision 满足目标 FPR，但 rescue 后整体 FPR 超过目标 operating point，则只能声称 raw content 分支满足该 FPR，不能声称完整 SLM-WM evidence decision 满足该 FPR；
6. `Geo-direct-positive` 只能作为反例审计，用于展示几何直接判正的 FPR 风险，不能进入正式方法。

该边界使几何 rescue 成为“同阈值内容重判”的辅助恢复机制，而不是独立第二检测器。

---

## 3.8 事件 attestation 与最终归因

仅凭 watermark evidence 成立，并不足以证明图像来自某次可核验生成事件。本文进一步引入事件级 attestation。设事件声明为 $e$，事件摘要为 $d_e$，签名为 $\sigma(e)$，则

$$
attestation\_pass
=
\mathbb{I}[\operatorname{Verify}_K(d_e,\sigma(e))=1].
$$

Final-level 判定为

$$
y_{\mathrm{final}}
=
y_{\mathrm{evidence}}
\land
attestation\_pass.
$$

最终输出包括三类状态：

1. `evidence_negative`：watermark 主证据未成立；
2. `evidence_positive_but_unattested`：watermark 主证据成立，但事件级归因未确认；
3. `final_positive`：watermark 主证据与事件级归因同时成立。

Attestation 不改变内容阈值，不替代几何恢复，也不进入 evidence-level 判定。Payload probe 仅作为诊断字段记录，不参与正式决策。

---

## 3.9 嵌入与检测流程

嵌入端执行以下步骤：

1. 采集中间 latent、预测图像和 Self-Attention maps；
2. 构造 semantic latent mask 与内容风险场；
3. 估计语义条件安全子空间 $B_{\mathrm{safe}}$；
4. 得到 $B_{\mathrm{LF}}$、$B_{\mathrm{HF}}$、$B_{\mathrm{A}}$ 和 attention anchor graph；
5. 生成 LF、HF 与 attention geometry carrier；
6. 在选定采样步执行 latent update；
7. 输出 watermarked 图像、event statement 和证据摘要。

检测端执行以下步骤：

1. 从待检图像的 VAE 编码、反演 latent 或重采样轨迹中估计 LF / HF 内容证据；
2. 提取 attention relation graph 并估计几何恢复参数；
3. 计算 $s_c^{\mathrm{raw}}$ 和 $s_c^{\mathrm{align}}$；
4. 根据 fixed-FPR 阈值执行 evidence-level 判定；
5. 验证 attestation 并输出 final-level 判定。

---

## 3.10 方法边界

本文方法的主贡献是扩散采样内部的语义条件潜流形水印，不以 payload 消息恢复为主要目标。Payload probe、attestation、evidence manifest 和 result audit 服务于诊断与归因治理，不应被表述为水印载体创新。所有主张必须由 fixed-FPR calibration、clean negative、attacked negative、几何 rescue 消融、再扩散攻击和外部 baseline 对比共同支撑。
