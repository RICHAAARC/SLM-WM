# 详细技术路线：SLM-WM 方法模块设计

## 一、文档定位

本文档描述语义条件潜流形水印（SLM-WM）的模块职责、数据流和可审计边界。公式细节以 `algorithm_primitives_semantic_conditioned_latent_manifold_watermark.md` 为准，论文叙事以 `method_section_semantic_conditioned_latent_manifold_watermark.md` 为准，代码映射以 `real_scientific_operator_implementation.md` 为准。

本设计固定采用三个正式分支：

1. `lf_content`：空间低通 LF 主证据；
2. `tail_robust`：高斯幅值尾部截断鲁棒补充证据；
3. `attention_geometry`：真实 Q/K Self-Attention 相对关系几何锚点。

`tail_robust` 按高斯模板元素绝对幅值的分位点选择分布尾部, 不具有空间频带定义。

---

## 二、统一方法数据流

SLM-WM 的正式方法链为

$$
\text{branch-specific semantic risk fields}
\rightarrow
\text{exact JVP and SVD low-response subspaces}
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
2. 分支资格集合与承载预算在 JVP 之前进入候选方向构造；
3. 精确 JVP 形成语义与视觉联合响应矩阵，SVD 选择相对低响应方向；
4. LF、尾部截断和注意力几何更新分别投影到对应安全子空间；
5. 检测端只从待检图像重建固定模板和注意力关系，不读取生成侧私有状态；
6. calibration split 冻结包含几何救回的完整判定协议，test split 只报告结果。

这不是三个独立水印器的串联。分支风险、低响应子空间、载体投影和完整 fixed-FPR 判定共同构成一个方法闭环。

---

## 三、模块与仓库职责映射

| 方法模块 | 正式实现 | 输入 | 输出 | 禁止行为 |
| --- | --- | --- | --- | --- |
| 分支风险 | `main/methods/semantic/branch_risk.py` | 语义、纹理、稳定性、显著性和 attention 信号图 | 三个风险场、预算和资格集合 | 用单一共享风险标量替代三个分支 |
| Jacobian Null Space | `main/methods/subspace/jacobian_nullspace.py` | latent、特征函数、分支预算、优先方向和密钥 | 三个低响应基底及残差记录 | 有限差分冒充精确 JVP；SVD 系数直接冒充 latent 基底 |
| 固定内容模板 | `main/methods/carrier/keyed_tensor.py` | 密钥、公开模型标识和 latent 形状 | LF 模板与尾部截断模板 | 模板依赖 Prompt、生成轨迹或样本级基底 |
| 安全投影 | `main/methods/carrier/keyed_tensor.py` | 固定模板、分支基底和相对强度 | 投影更新和能量保留率 | 对近零投影强制归一化后继续运行 |
| 注意力几何 | `main/methods/geometry/differentiable_attention.py` | 真实 Transformer Q/K 与 latent | 目标梯度、分数增益和回溯记录 | 使用合成 attention map 支持正式主张 |
| 图像盲检 | `main/methods/detection/image_only.py` | 待检图像、密钥和公开模型配置 | 内容分数、几何统计和 evidence 判定 | 读取 Prompt、源 latent、轨迹或样本级 Null Space |
| 真实模型运行 | `experiments/runners/semantic_watermark_runtime.py` | SD3/SD3.5 pipeline 与方法配置 | clean/watermarked 图像和科学算子记录 | 以 proxy 或 fallback 记录支持正式结果 |
| 数据集协议 | `experiments/runners/image_only_dataset_runtime.py` | Prompt split、攻击配置和目标 FPR | 冻结协议、test records 和质量 registry | 使用 test split 调阈值 |
| 正式消融 | `experiments/ablations/runtime_rerun.py` | 改变后的机制配置与冻结检测协议 | 重新生成、攻击和检测的消融记录 | 修改历史分数模拟消融 |

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
\phi_{\mathrm{stab}}(u),
\phi_{\mathrm{sal}}(u),
\phi_{\mathrm{attn\_stab}}(u)
].
$$

若 saliency 或 segmentation 首先在图像域计算，必须通过 $\Pi_{x\rightarrow z}$ 映射到 latent 分辨率。该 mask 需要参与风险和候选方向构造，不能只用于可视化。

### （二）分支输出

对分支 $b\in\{\mathrm{LF},\mathrm{tail},\mathrm A\}$ 分别计算

$$
\rho_b(u)
=
\frac{1}{Z_b}
\left[
\eta_s^b\phi_{\mathrm{sal}}(u)
+\eta_m^b\phi_{\mathrm{sem}}(u)
+\eta_t^b\psi_b(\phi_{\mathrm{tex}}(u))
+\eta_i^b(1-\phi_{\mathrm{stab}}(u))
+\eta_A^b(1-\phi_{\mathrm{attn\_stab}}(u))
\right],
$$

并生成承载预算 $b_b(u)$ 与资格集合 $\Omega_b$。资格集合外的预算必须置为 0，确保路由会真实改变子空间求解。

---

## 五、精确 Jacobian Null Space 模块

### （一）候选方向

每个分支构造

$$
C_b=[v_1^b,\ldots,v_m^b],\qquad C_b^\top C_b=I.
$$

LF 与尾部截断分支把对应固定模板作为优先方向；注意力分支把真实 Q/K 目标梯度作为优先方向；其余列由密钥化随机方向补齐。分支预算在 QR 正交化之前作用于候选方向。

### （二）联合响应与 SVD

正式实现通过 `torch.func.linearize` 或 `torch.autograd.functional.jvp` 计算精确 JVP，构造

$$
R_b=
\begin{bmatrix}
J_{\mathrm{sem}}C_b\\
\sqrt{\lambda_v}J_{\mathrm{vis}}C_b
\end{bmatrix}.
$$

若

$$
R_b=U\Sigma Q^\top,
$$

则最小奇异值对应的 $Q_0$ 必须通过

$$
B_b=\operatorname{orth}(C_bQ_0)
$$

映射回 latent 空间。正式记录同时保存响应残差、相对响应残差、正交误差和固定模板投影能量。当前门禁要求相对响应残差不超过 0.75，内容模板能量保留比例不低于 0.01。

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

该分支具有明确的空间低通定义。

### （二）高斯幅值尾部截断

尾部模板为

$$
\widetilde\nu_{\mathrm{tail},i}
=
\nu_{\mathrm{tail},i}
\mathbb I(|\nu_{\mathrm{tail},i}|\ge q_{1-\gamma}).
$$

$\gamma$ 是标准高斯元素绝对幅值的尾部保留比例。该过程不执行 FFT、DCT、空间波数排序或频带 mask，因此只定义幅值域筛选, 不定义空间频带。其攻击鲁棒性必须由真实实验验证。

### （三）安全投影与内容分数

两个固定模板分别投影到对应安全子空间：

$$
\Delta z_t^b
=
\alpha_t^b\operatorname{Norm}(B_bB_b^\top\nu_b),
\qquad b\in\{\mathrm{LF},\mathrm{tail}\}.
$$

检测端不恢复 $B_b$，只计算待检图像编码与固定模板的相关性：

$$
s_c
=
\lambda_{\mathrm{LF}}s_{\mathrm{LF}}
+
\lambda_{\mathrm{tail}}s_{\mathrm{tail}}.
$$

两个分支不能分别设置独立正判阈值后投票。

---

## 七、Q/K 注意力几何模块

正式 attention map 由模型的 Q/K 投影计算：

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}\right).
$$

密钥确定关系目标，autograd 对 latent 求目标梯度。该梯度作为注意力分支优先方向进入 Null Space 求解，再在安全基底中投影。嵌入使用单调回溯，只接受使注意力目标分数改善且满足强度预算的更新。

检测端从待检图像的公开视觉模型 token 关系估计锚点匹配，通过三点 RANSAC 与仿射估计恢复参考系。几何统计只能决定是否允许重对齐，不能独立给出 positive。

---

## 八、仅图像检测与完整 fixed-FPR

正式检测接口只允许

$$
\operatorname{Detect}(x',K,M).
$$

calibration split 同时冻结内容阈值、几何可靠性阈值、rescue window 和失败原因 gate。rescue 只对内容阈值附近的边界失败样本开放，对齐后复用同一个内容阈值。最终 fixed-FPR 对应 `content OR same-threshold rescue` 的完整布尔协议。

三级运行配置分别使用 70/700/7000 个 Prompt，test 数量为 34/340/3400，目标 FPR 为 0.1/0.01/0.001。test split 只应用冻结协议并报告置信上界。

---

## 九、正式消融与证据需求

正式消融至少覆盖分支风险、Jacobian Null Space、LF、尾部截断、Q/K attention geometry 和图像对齐。每个消融必须重新生成、重新攻击和重新检测。正式配置使用 `tail_robust_only`、`without_tail_truncation` 和无尾部分支配置。

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
