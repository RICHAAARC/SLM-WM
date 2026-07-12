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
4. LF、尾部截断和注意力几何更新分别投影到对应安全子空间；
5. 检测端只从待检图像重建固定模板和注意力关系，不读取生成侧私有状态；
6. calibration split 冻结包含几何救回的完整判定协议，test split 只报告结果。

这不是三个独立水印器的串联。分支风险、完整特征 Null Space、载体投影和完整 fixed-FPR 判定共同构成一个方法闭环。

---

## 三、模块与仓库职责映射

| 方法模块 | 正式实现 | 输入 | 输出 | 禁止行为 |
| --- | --- | --- | --- | --- |
| 分支风险 | `main/methods/semantic/branch_risk.py` | 语义、纹理、稳定性、显著性和 attention 信号图 | 三个风险场、预算和资格集合 | 用单一共享风险标量替代三个分支 |
| Jacobian Null Space | `main/methods/subspace/jacobian_nullspace.py` | latent、716维完整特征函数、分支预算、实际载体方向和密钥 | 三个 rank-4 Null Space 基底及 CG/逐列残差记录 | 低维草图制造代数零空间；阻尼求解后仍声明 Null Space |
| 固定内容模板 | `main/methods/carrier/keyed_tensor.py` | 密钥、公开模型标识和 latent 形状 | LF 模板与尾部截断模板 | 模板依赖 Prompt、生成轨迹或样本级基底 |
| 安全投影 | `main/methods/carrier/keyed_tensor.py` | 固定模板、分支基底和相对强度 | 投影更新和能量保留率 | 对近零投影强制归一化后继续运行 |
| 注意力几何 | `main/methods/geometry/differentiable_attention.py` | 真实 Transformer Q/K 与 latent | 目标梯度、分数增益和回溯记录 | 使用合成 attention map 支持正式主张 |
| 图像盲检 | `main/methods/detection/image_only.py` | 待检图像、密钥和公开模型配置 | 内容分数、几何统计和 evidence 判定 | 读取 Prompt、源 latent、轨迹或样本级 Null Space |
| 真实模型运行 | `experiments/runners/semantic_watermark_runtime.py` | SD3.5 Medium pipeline 与方法配置 | clean/watermarked 图像和科学算子记录 | 绕过科学算子门禁写出正式结果 |
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

### （一）种子方向与完整特征

每个分支构造

$$
D_b=[d_1^b,\ldots,d_m^b],\qquad D_b^\top D_b=I.
$$

LF 与尾部截断分支把对应固定模板作为首个方向；注意力分支把真实 Q/K 目标梯度作为首个方向；其余列由密钥化方向补齐。完整特征函数连接512维归一化 CLIP embedding 和204维视觉向量，正式输出宽度为716且不执行压缩。

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

正式记录保存每列 CG 迭代数、CG 相对残差、投影能量、QR 后完整 Jacobian 响应与正交误差。CG 最大64次、相对收敛阈值为 $10^{-6}$ 且阻尼固定为0；QR 后每列相对响应不得超过 $10^{-4}$，投影能量不得低于0.01，正交误差不得超过 $10^{-5}$。三个分支合成后必须复验实际写回 latent；完整扩散结束后还必须比较最终 clean 与 watermarked 成图。两级保持门禁均要求 CLIP cosine 不低于0.995且视觉特征相对漂移不高于0.02。

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
\alpha_t^b\operatorname{Norm}(N_bN_b^\top\nu_b),
\qquad b\in\{\mathrm{LF},\mathrm{tail}\}.
$$

检测端不恢复 $N_b$，只计算待检图像编码与固定模板的相关性：

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

跨冻结层 Q/K 关系行稳定度与接收 attention 显著度共同确定前50%的稳定 token 集合。一次注入的梯度、内容基底复算、回溯和最终写回复验冻结同一集合；集合内关系权重为1, 其余规则网格关系保留0.25权重。密钥确定关系目标，autograd 对 latent 求目标梯度。该梯度作为注意力分支优先方向进入 Null Space 求解，再在安全基底中投影。嵌入使用单调回溯，只接受使注意力目标分数改善且满足强度预算的更新。

检测端从待检图像、密钥和公开模型提取真实 Q/K 关系图。对冻结的有界相似仿射与方形二面体候选构造双线性矩阵 $W_T$ 与逆向矩阵 $V_T$，分别计算规范拉回 $W_TA_{\mathrm{obs}}W_T^\top$ 和观测前推 $V_TS_KV_T^\top$。注册目标以0.10和0.90组合两个方向的关系相关分数，再显式扣除规范侧与观测侧的覆盖率、唯一采样率损失。输出必须记录相对 identity 候选的对齐增益、双向关系分数、覆盖惩罚、目标间隔和恢复变换。结构注册只接受正双向关系、正目标间隔以及两个方向覆盖率均不低于0.45的候选。恢复图像参考系后必须重新提取全部冻结层的真实 Q/K，并由 calibration split 冻结注册分数、注册置信度和恢复后同步阈值。几何统计只能决定是否允许重对齐，不能独立给出 positive。

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

正式消融精确包含完整方法和10个真实重运行对照：共享全局风险、完全移除风险路由、移除 Jacobian Null Space、LF-only、Tail-only、移除 LF、移除尾部载体、移除幅值尾部截断、移除 Q/K attention geometry、移除图像对齐。每个配置都必须重新生成、重新攻击和重新检测, 不能修改已有分数模拟机制差异。

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
