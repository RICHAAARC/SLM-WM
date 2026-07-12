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

这里的三个分量共享同一个受约束优化目标，但使用各自的风险预算和种子方向。该实现属于项目特定设计；JVP/VJP、无阻尼半正定求解、逐列残差和固定 FPR 校准属于可迁移的通用方法。

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

$\phi_{\mathrm{sem}}$ 是冻结 CLIP patch token 与 CLS token 的余弦一致性图，$\phi_{\mathrm{tex}}$ 是解码灰度图水平与垂直绝对梯度和，$\phi_{\mathrm{lcr}}$ 是灰度相对反射填充5x5局部均值的绝对偏离。三者经双线性插值映射到 latent 网格，并在单个样本内归一化到 $[0,1]$。$\phi_{\mathrm{adj}}$ 直接比较当前 latent 与紧邻上一 scheduler 步 latent 的解码 RGB 图像：

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

当 $u\notin\Omega_b$ 时，正式候选矩阵中的对应预算被置为 0；只有资格集合内部保留连续预算。该硬门控确保风险场会真实改变后续子空间，而不是仅作为日志字段。完整方法仅对当前活动且启用语义风险路由的分支执行空资格集合 fail-closed 门禁；移除风险路由的正式消融不使用风险阈值决定样本是否继续运行。每次注入保存三个分支完整风险值、连续预算和资格 mask 的 Tensor 内容 SHA-256, 联合摘要必须与三个逐分支记录一致。

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

正式记录必须保存完整特征 schema、每列 CG 迭代数和收敛残差、约束投影能量、QR 后逐列完整 Jacobian 响应与正交误差。CG 阻尼固定为0，最多64次迭代且相对收敛阈值为 $10^{-6}$；QR 后每列相对响应不得超过 $10^{-4}$，能量保留比例不得低于0.01，正交误差不得超过 $10^{-5}$。任一条件失败时直接停止运行。

三个分支合成后，先按扩散 latent 的真实存储 dtype 执行加法，再从实际写回值恢复量化增量

$$
\Delta z_t^{\mathrm{written}}
=
\operatorname{cast}_{\operatorname{dtype}(z_t)}
\left(z_t+\Delta z_t\right)-z_t.
$$

运行时必须对该实际量化增量重新执行完整特征精确 JVP，并计算

$$
r_{\mathrm{written}}
=
\frac{\left\|J_t\Delta z_t^{\mathrm{written}}\right\|_2}
{\max\!\left(\left\|F(z_t)\right\|_2,\epsilon\right)}.
$$

当前冻结阈值要求 $r_{\mathrm{written}}\leq10^{-4}$。该门禁直接约束真正进入 scheduler 的低精度 Tensor，不能由量化前方向的 Null Space 残差替代。每个分支记录进一步绑定实际候选矩阵、风险预算、响应矩阵和最终基底的精确内容 SHA-256；单次注入还分别绑定 LF、尾部截断和注意力几何更新以及三分支联合摘要。随后，运行时还对每次实际写回执行完整特征有限更新复验；全部扩散步骤结束后，再直接比较最终 clean 与 watermarked 成图。两级有限变化门禁均要求 CLIP cosine 不低于0.995且手工结构统计特征相对漂移不高于0.02。

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

其中，$\operatorname{PRG}_{\mathcal N}$ 固定为 `sha256_counter_box_muller_float32_v1`：密钥、精确模型标识、分支名称和 shape 先形成 SHA-256 domain；大端计数器流提供53位开区间均匀数，Box-Muller 在 CPU float64 中完成高斯变换，再统一取整为 CPU float32 规范 Tensor。CPU 或 CUDA 只接收该规范 Tensor 的副本，PyTorch 设备 RNG 不参与模板定义。

平均池化在 latent 的二维空间轴上抑制快速空间变化，因此 LF 分支具有明确的空间低通定义。嵌入端把固定模板投影到分支安全子空间：

$$
\overline\nu_{\mathrm{LF}}
=N_{\mathrm{LF}}N_{\mathrm{LF}}^\top\nu_{\mathrm{LF}},
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

因此保留元素数精确等于 $\lceil n\gamma\rceil$；同幅值元素由公开的展平索引规则消除歧义。记录中的 `tail_threshold` 是 $I_\gamma$ 内最小绝对幅值，不是由设备算子重新估计的随机分位点。

嵌入更新为

$$
\Delta z_t^{\mathrm{tail}}
=
\alpha_t^{\mathrm{tail}}
\operatorname{Norm}
\left(
N_{\mathrm{tail}}N_{\mathrm{tail}}^\top
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

每个 token 的选择分数由跨冻结层 Q/K 关系行稳定度与接收 attention 显著度相乘得到, 固定选择前50%且不少于4个 token。稳定 token 的单点权重 $a_i$ 为1, 其余 token 为0.25；非对角 pair 权重为 $w_{ij}=a_i a_j\mathbf1[i\ne j]$。选择摘要、原始二维索引、权重值和外积规则形成 `stable_pair_weight_identity_digest`。一次注入的梯度、回溯和最终写回复验冻结同一个 pair 权重对象。密钥产生零对角符号图 $S_K$, 四通道投影为 $\Pi S_K$, $\Pi=\operatorname{diag}(1,-1,1,1)$。四个分量各自逐行加权中心化与归一化后严格等权：

$$
s_A=\frac14\sum_{c=1}^{4}\operatorname{RowCorr}_{w}
\left(r^{(c)},\pi_cS_K\right),\qquad
g_A=\nabla_{z_t}s_A.
$$

$G$ 通过中心化 $P$ 对 latent 保留非零梯度, 且在均匀 attention 下严格为0。
概率和距离都逐行中心化, 从而移除距离行均值乘概率偏离所产生的第3通道一阶
重复。四个分量均依赖真实 Q/K, 公开距离只作为 $G$ 的冻结调制因子。

每个冻结层同时记录抽样后真实 $Q$、$K$、中心化 $L$、逐头 softmax 后平均的 $P$ 与原始二维 token 索引的版本化 Tensor 内容 SHA-256。一次注入按原 latent、内容基底、接受候选和实际量化写回 latent 四个角色持久化原子；最终成图按 clean、carrier-only 和完整方法三个角色持久化原子；仅图像盲检按原图和对齐图两个角色持久化原子。角色顺序、层顺序、逐层自摘要和联合摘要均须一致。

### 3.6.2 安全投影与单调回溯

注意力种子方向进入其独立分支的 Jacobian Null Space 求解器，得到 $N_{\mathrm A}$。投影梯度为

$$
\overline g_A=N_{\mathrm A}N_{\mathrm A}^\top g_A.
$$

以 $\overline g_A$ 为更新方向执行回溯搜索，仅接受同时满足强度预算且使四分量
目标分数单调上升的步长：

$$
s_A(z_t+\Delta z_t^{\mathrm A})>s_A(z_t).
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

规范拉回与观测前推都先计算四个逐行加权相关分数再等权组合。候选目标为

$$
J(T)=0.10s_{\mathrm{can}}+0.90s_{\mathrm{obs}}
-0.01\sum_{q\in\{c_{\mathrm{can}},u_{\mathrm{can}},c_{\mathrm{obs}},u_{\mathrm{obs}}\}}(1-q).
$$

$c$ 和 $u$ 分别表示有效坐标覆盖率与唯一采样率。观测前推项使用完整观测关系解释候选变换，防止只保留中心子图的尺度候选通过较小有效区域获得更高目标。检测端只在原始观测 Q/K 上执行一次稳定 token 选择。观测项使用 $w_{\mathrm{obs}}$, 规范项使用 $a_{\mathrm{can},T}=W_Ta_{\mathrm{obs}}$ 后重新外积得到的 $w_{\mathrm{can},T}$, 两者共享权重身份摘要。检测器从与攻击注册表无关的旋转、log-scale 和归一化位移连续定义域生成粗锚点，并按固定三分比例执行三层局部细化。每轮局部组合都相对方形二面体基元执行严格过滤, 保证残余旋转不超过32°、尺度始终位于 $[1/\sqrt2,\sqrt2]$、两个平移分量绝对值不超过0.28；搜索器不读取攻击角度、裁剪比例或位移参数，验证集使用远离正式攻击取值的确定性随机连续变换。输出包括四通道分数、相对 identity 候选的观测关系增益、双向关系分数、覆盖惩罚、目标间隔、注册置信度、有效锚点内点比例、对齐残差和权重身份。结构注册要求完整四通道观测与双向分数为正、目标间隔为正、两个方向覆盖率均不低于0.45，并继续执行内点比例和残差门禁。由于均匀 attention 使 $G=0$ 且其余通道在逐行中心化后也为0, 公开坐标不能单独形成可靠注册。得到 $\widehat T$ 后，检测器重采样待检图像并重新提取全部冻结层的真实 Q/K；恢复后同步分数使用传递后的 $w_{\mathrm{can},\widehat T}$, 不重新选择稳定 token。权重身份一致且同步分数通过 calibration split 冻结阈值后, 才允许进入同阈值内容救回。几何链只负责参考系恢复和救回资格门禁，不独立产生 positive 判定。

### 3.6.4 最终成图注意力可观测性

最终图像 attention 归因使用三路同 seed 生成：clean、完整方法, 以及保持同一 scheduler、LF/tail 配置与算子且只关闭 attention geometry 的 carrier-only 反事实。完整方法与 carrier-only 首个注入前 latent 必须以 dtype、shape 和全部连续原始字节 SHA-256 证明相同；两侧更新数、顺序和完整 scheduler 轨迹必须一致。carrier-only 每条更新原子必须明确 `attention_source=disabled_attention_geometry`, attention 分数与更新为空, 关系和 pair 身份为空, 直接 Q/K 来源为 false, 且 Null Space 记录不含 attention 分支。原子 JSONL 的路径、实际文件 SHA-256 和解析内容摘要绑定到结果、manifest 与缓存复验。该干预估计 attention 开关经后续 LF、tail 和状态轨迹交互传播的总机制效应；不要求干预后两侧 realized LF/tail 更新相等, 也不声称纯直接效应。三张成图都必须通过 VAE 编码、公开固定噪声加噪和同一冻结 Transformer 前向重新构造直接 Q/K 四分量关系图。公开固定噪声只能由冻结 scheduler 的 `scale_noise` 在正式检测 timestep 上施加；缺少该算子的 scheduler 不属于当前方法协议，运行必须失败。归因计算前, clean 到完整方法、clean 到 carrier-only 及 carrier-only 到完整方法三条边的完整 CLIP 语义余弦相似度和手工结构统计特征相对漂移必须同时通过冻结阈值。运行随后计算两类归因增益：完整方法与 carrier-only 分别执行自身稳定 token 盲选后的分数差, 以及冻结 carrier-only pair 权重后同时评价两张图的分数差。两类总分增益均须严格大于 `minimum_final_image_attention_score_gain`, 当前冻结值为0.0001；四个 Q/K 依赖分量的配对增益同时写入记录。正式保持记录与 Q/K 记录必须绑定直接 Q/K 来源、四分量身份、密钥投影身份、反事实配置、更新原子、scheduler 轨迹、持久化图像路径、图像 SHA-256 及 CUDA 执行证据。反事实缺失、原子残留 attention、首 latent 或调度漂移、三边保持失败、产物身份不一致、分数非有限或任一总分增益未通过都会终止当前科学单元, 中间 latent attention 分数不能替代该门禁。clean 分数保留为总体水印对照, 不作为 attention 因果归因基线, 也不进入仅图像检测函数。

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

方法实现存在不等于论文结论成立。空间 LF 的有效性、高斯幅值尾部截断的攻击鲁棒性、Q/K 几何恢复的增益和完整 fixed-FPR 均必须由真实 GPU 生成、clean negative、真实攻击、正式机制重跑消融、外部 baseline 和受治理结果包共同支撑。
