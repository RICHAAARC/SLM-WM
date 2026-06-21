# 详细技术路线：SLM-WM 方法模块设计

## 一、文档定位

本文档只描述“语义条件化的潜空间流形水印（Semantic-conditioned Latent Manifold Watermarking, SLM-WM）”的方法模块设计、数据流和统计边界。本文档不作为项目构建操作指导，不定义阶段推进命令，不要求创建具体目录，也不覆盖 `.codex/project_contract.md` 中的项目框架契约。

若本文档中的方法模块需要落地到仓库，必须遵循当前项目框架：

1. `main/` 保存论文方法、核心协议、核心评估、表格重建和 CLI 复现能力。
2. `experiments/` 保存阶段性实验 runner、baseline、ablation 和 paper protocol。
3. `paper_workflow/` 保存 Notebook / Colab workflow 入口和 session helper。
4. `scripts/` 保存数据准备、结果检查、结果打包和 release 辅助命令。
5. `tools/harness/` 只作为外层治理审计，不得被 `main/` 反向依赖。

因此，本文档中的“模块”是方法职责划分，而不是目录创建清单。项目实施顺序、阶段门禁、测试分层和发布包边界应以项目契约和总阶段构建指引为准。

---

## 二、方法主线

SLM-WM 的统一方法链为：

$$
\text{semantic risk field}
\rightarrow
\text{semantic-conditioned safe null-space}
\rightarrow
\text{latent LF/HF/attention carrier decomposition}
\rightarrow
\text{fixed-FPR robust detection}
\rightarrow
\text{same-threshold geometric rescue}
\rightarrow
\text{attested final attribution}.
$$

该链条表达的是方法机制的连续推导关系：

1. 图像内容和提示语决定语义风险场。
2. 语义风险场约束潜空间中的安全子空间。
3. 同一安全子空间分解出 LF 主证据、HF 鲁棒补充和 Self-Attention 几何锚点。
4. 内容证据在固定 FPR 口径下校准阈值。
5. 几何链只恢复参考系，恢复后仍使用同一内容阈值重判。
6. attestation 只在 final-level 约束事件归因，不替代 watermark evidence。

该方法不应被表述为“semantic mask + LF + HF + geometry + attestation”的组件堆叠。所有论文主张都应回到上述链条中的某个机制位置，并由 records、tables、figures、reports 或 manifests 支撑。

---

## 三、方法模块与项目框架的职责映射

下表说明方法模块在当前项目框架中的推荐归属。该表用于保持职责边界，不要求一次性创建所有文件。

| 方法模块 | 推荐仓库归属 | 核心职责 | 不应承担的职责 |
|---|---|---|---|
| 核心数据结构与摘要 | `main/core/` | 定义可复用 typed objects、稳定摘要、最小 records / manifests 基础结构 | 运行 SD 模型、访问 Drive、写论文最终图表 |
| 语义路由与风险场 | `main/methods/` | 根据语义显著性、纹理复杂度、稳定性和显著风险形成路由约束 | 直接加载大型 segmentation / saliency 模型权重 |
| 安全子空间规划 | `main/methods/` | 基于 JVP 或可审计近似求解 semantic-conditioned safe basis | 直接写实验 records 或 claim audit |
| LF / HF 内容载体 | `main/methods/` | 生成同一安全子空间内的互补 carrier，并计算内容证据 | 为 LF 和 HF 分别设置独立主判阈值 |
| Self-Attention 几何锚点 | `main/methods/` | 定义 attention-relative graph、几何锚点和参考系恢复统计 | 直接输出 positive 判定 |
| 检测与决策协议 | `main/protocol/` 或 `main/methods/` | 定义内容分数、阈值、rescue 条件、final-level 决策 | 使用 test split 调阈值 |
| 实验协议与运行适配 | `experiments/` | 组织 prompt、split、SD 适配、attack、baseline、ablation runner | 污染 `main/` 的核心方法依赖方向 |
| 论文产物重建 | `main/analysis/`、`scripts/` | 从 governed records 与 manifests 重建 tables、figures、reports 和 evidence audit | 手工拼接正式论文结果 |
| Notebook / Colab 工作流 | `paper_workflow/` | 调度仓库模块、挂载外部存储、展示阶段摘要 | 在 Notebook cell 中实现唯一正式算法逻辑 |

通用工程写法是将运行适配、实验调度和产物重建分层。项目特定写法是以 `main/` 作为核心包边界，并要求核心包不反向依赖 `experiments/`、`paper_workflow/`、`scripts/`、`tools/harness/` 或本地输出目录。

---

## 四、语义条件潜流形模块

### （一）输入与输出

语义条件潜流形模块接收提示语、扩散轨迹中的 latent、中间预测图像、attention 统计或外层 runtime 提供的语义掩码。模块输出语义风险场、latent mask、承载预算和 LF / HF / attention 的候选区域。

该模块的核心公式为：

$$
\phi(u)=
[
\phi_{\mathrm{sem}}(u),
\phi_{\mathrm{tex}}(u),
\phi_{\mathrm{stab}}(u),
\phi_{\mathrm{sal}}(u)
],
$$

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

风险越低，区域越适合承载水印。若语义掩码来自图像域，必须映射到 latent 分辨率：

$$
M_z=\Pi_{x\rightarrow z}(M_x).
$$

### （二）设计约束

1. `M_z` 必须进入轨迹特征和子空间规划，不能只作为可视化产物。
2. 默认全 latent mask 只能作为 fallback 或 `No-semantic-mask` 消融，不能作为正式主方法。
3. saliency、segmentation 或 SD attention 捕获属于运行适配层；方法核心只接收标准化后的 mask 或 feature tensor。

---

## 五、语义条件安全子空间模块

### （一）目标

该模块将固定统计 Null Space 推广为样本条件化安全子空间，目标是在语义特征和视觉质量特征上寻找低响应扰动方向。

定义：

$$
\mathcal{N}_{\mathrm{sem}}(z_t,p)
=
\{v:
\|W_{\mathrm{sem}}J_{\mathrm{sem}}(z_t,p)v\|_2\le\epsilon_s,
\|W_{\mathrm{vis}}J_{\mathrm{vis}}(z_t,p)v\|_2\le\epsilon_v
\}.
$$

轨迹特征可写为：

$$
\varphi_t=P^\top\operatorname{vec}\left(\operatorname{Norm}(M_z\odot z_t)\right).
$$

通过 JVP、有限差分或可审计的轨迹线性化近似构造响应矩阵，并通过 SVD 得到安全基底：

$$
D=U\Sigma V^\top,
\qquad
B_{\mathrm{safe}}=V_{[:,r+1:m]}.
$$

随后根据路由区域获得：

$$
B_{\mathrm{LF}}=\operatorname{Proj}_{\Omega_{\mathrm{LF}}}(B_{\mathrm{safe}}),
\quad
B_{\mathrm{HF}}=\operatorname{Proj}_{\Omega_{\mathrm{HF}}}(B_{\mathrm{safe}}),
\quad
B_{\mathrm{A}}=\operatorname{Proj}_{\Omega_{\mathrm{A}}}(B_{\mathrm{safe}}).
$$

### （二）可审计要求

1. 若使用近似 JVP，必须在论文和 records 中说明近似对象，而不能声称构造了完整 denoiser Jacobian。
2. 每个 basis、route 和 JVP 近似应具有稳定摘要，便于实验记录和消融复核。
3. `Global-nullspace`、`No-semantic-mask`、`No-risk-weight` 和 `Random-basis` 应作为机制消融，而不是 fallback 后伪装成主方法。

---

## 六、LF / HF 内容载体模块

### （一）LF 主证据

LF 分支负责 clean 条件和轻度失真下的主内容证据。其 latent update 为：

$$
\Delta z_t^{\mathrm{LF}}
=
\alpha_t^{\mathrm{LF}}B_{\mathrm{LF}}
C_{\mathrm{LF}}(\operatorname{PRG}(K_{\mathrm{LF}},d_e,d_B,d_R)).
$$

LF 检测统计量为归一化相关分数：

$$
s_{\mathrm{LF}}
=
\frac{\langle \hat{c}_{\mathrm{LF}},\nu_{\mathrm{LF}}^c\rangle}
{\|\hat{c}_{\mathrm{LF}}\|_2\|\nu_{\mathrm{LF}}^c\|_2}.
$$

### （二）HF 鲁棒补充

HF 分支负责纹理区域和 harder regime 下的补充证据。其模板需要经过 tail truncation：

$$
\widetilde{\nu}_{\mathrm{HF},i}
=
\nu_{\mathrm{HF},i}\cdot
\mathbb{I}(|\nu_{\mathrm{HF},i}|w_i\ge q_{\gamma}).
$$

HF latent update 为：

$$
\Delta z_t^{\mathrm{HF}}
=
\alpha_t^{\mathrm{HF}}B_{\mathrm{HF}}\widetilde{\nu}_{\mathrm{HF}}.
$$

### （三）统一内容分数

LF 与 HF 的融合分数为：

$$
s_c=\lambda_{\mathrm{LF}}s_{\mathrm{LF}}+
\lambda_{\mathrm{HF}}s_{\mathrm{HF}},
\qquad
\lambda_{\mathrm{LF}}>\lambda_{\mathrm{HF}},
\qquad
\lambda_{\mathrm{LF}}+\lambda_{\mathrm{HF}}=1.
$$

该模块的统计边界是：LF 和 HF 不分别设置独立正判阈值后投票；payload probe、bit agreement 和 codeword consistency 只能作为诊断统计，不替代正式内容分数。

---

## 七、Self-Attention 几何锚点模块

### （一）几何锚点角色

Self-Attention 几何锚点的作用是帮助参考系恢复，而不是直接提供 positive 判定。该模块应构造 attention-relative graph，而不是只依赖绝对像素坐标。

attention map 定义为：

$$
A_t^{(\ell)}=
\operatorname{softmax}\left(
\frac{Q_t^{(\ell)}K_t^{(\ell)\top}}{\sqrt d}
\right).
$$

稳定 token 集合为：

$$
\mathcal{V}_A=
\{i:\operatorname{Stab}(A_{\cdot,i}^{(\ell)})\ge\tau_A,
\operatorname{Sal}(i)\ge\tau_s\}.
$$

相对关系可表示为：

$$
r_{ij}=[A_{ij},\operatorname{rank}_j(A_{ij}),A_{ij}/\sum_k A_{ik},\operatorname{dist}_{\mathrm{rel}}(i,j)].
$$

### （二）attention-relative latent update

当 attention graph 稳定时，可定义几何关系损失：

$$
\mathcal{L}_A=
\sum_{(i,j)\in\mathcal{E}_A}
\|r_{ij}(z_t+\Delta z_t)-r_{ij}^{\star}\|_2^2.
$$

几何 update 为：

$$
\Delta z_t^{\mathrm{A}}
=
-\alpha_t^{\mathrm{A}}
\operatorname{Proj}_{\mathcal{N}_{\mathrm{sem}}}
(\nabla_{z_t}\mathcal{L}_A).
$$

若真实 update 不稳定，该模块只能降级为几何证据或诊断机制，不能在 Full 方法中声称 attention-relative latent carrier 已成立。

---

## 八、fixed-FPR 与 rescue 的统计边界

### （一）内容阈值边界

正式内容阈值只能由 calibration split 中的 clean negative 分布确定：

$$
\tau_c=Q_{1-\alpha}(s_c\mid\mathcal{D}_0^{\mathrm{cal}}).
$$

该阈值冻结后，不得使用 test split、攻击后正样本或论文主结果反向调参。

### （二）rescue 边界

原始内容边界余量为：

$$
m_c^{\mathrm{raw}}=s_c^{\mathrm{raw}}-\tau_c.
$$

几何救回只允许作用于边界失败窗口：

$$
rescue\_eligible=
(\delta_{\mathrm{low}}\le m_c^{\mathrm{raw}}<0)
\land geometry\_reliable
\land fail\_reason\in\{geometry\_suspected,low\_confidence\}.
$$

恢复后必须复用同一内容阈值：

$$
m_c^{\mathrm{align}}=s_c^{\mathrm{align}}-\tau_c,
\qquad
rescue\_applied=rescue\_eligible\land\mathbb{I}(m_c^{\mathrm{align}}\ge0).
$$

### （三）整体误报审计

同阈值 rescue 并不自动保证整体 evidence-level FPR 等于目标 FPR。正式报告必须同时审计：

1. raw content decision 在 clean negative 上的 FPR；
2. rescue 后 evidence-level decision 在 clean negative 上的整体 FPR；
3. rescue 后 evidence-level decision 在 attacked negative 上的整体 FPR；
4. geometry-reliable gate、rescue window 和 fail-reason gate 在 negative 样本上的触发率；
5. 若整体 evidence-level FPR 超过目标 operating point，应降调 fixed-FPR 主张，或重新在 calibration split 中冻结包含 rescue 的完整决策协议。

因此，论文中“fixed-FPR”指的是经过 calibration 冻结的完整判定协议，而不是只固定内容阈值后任意叠加 rescue 分支。

---

## 九、attestation 与 payload probe 边界

attestation 只用于 final-level 事件归因，不改变 evidence-level 判定：

$$
y_{\mathrm{final}}=y_{\mathrm{evidence}}\land attestation\_pass.
$$

payload probe 只用于 bit-level agreement、codeword consistency 和失败模式诊断，不进入 `positive_by_content`、`rescue_applied`、`y_evidence` 或 `y_final` 的主判定链。

---

## 十、论文证据需求

SLM-WM 若要支撑投稿级论文主张，至少需要以下证据类型：

1. clean negative 与 attacked negative 下的固定 FPR 审计；
2. LF / HF score retention 与内容分数分布；
3. 几何恢复可靠性、rescue gain 和 rescue 后整体 FPR；
4. 语义路由、安全子空间、LF/HF、attention geometry 和 attestation 的机制消融；
5. 同协议外部 baseline 对比；
6. 由 records 与 manifests 自动重建的 tables、figures、reports 和 claim evidence audit。

这些证据属于实验与论文产物层，不属于本文档定义的方法模块本身。实现时必须通过项目契约规定的 records、manifests 和 artifact rebuild 流程支撑，不能以 Notebook 输出或手工表格替代。
