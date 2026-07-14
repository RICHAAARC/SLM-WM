# SLM-WM 方法语义不变量

## 文档职责

本文档是“语义条件潜流形水印”数学语义的权威来源。`main/methods/method_definition.py` 只能镜像本文档已经冻结的定义, 不能根据当前代码行为反向改写公式。`configs/method_semantic_registry.json` 只保存公式、实现、CPU 性质测试和 GPU 原子证据之间的追踪关系, 不允许自行声明任何不变量已经通过。

方法完成度严格区分以下层级：

- `specified`: 本文档中的公式、参数、适用边界和失败条件已经完整冻结。
- `cpu_verified`: 独立 CPU 性质测试同时接受正确实现并拒绝构造反例。
- `gpu_verified`: 同一正式实现已经由真实 SD3.5 CUDA 运行产生完整原子证据。
- `evidence_closed`: 运行原子、records、统计和结果包已经形成不可绕过的论文证据链。

方法定义摘要、字段存在、源码路径存在或单个局部测试通过都不能单独证明方法完成。

## 冻结模型、符号与数值约定

正式生成模型固定为 `stabilityai/stable-diffusion-3.5-medium@b940f670f0eda2d07fbb75229e779da1ad11eb80`, 语义模型固定为 `openai/clip-vit-base-patch32@3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`。模型 ID、40位 revision、VAE 缩放参数、scheduler 类、Transformer 模块类和实际 latent dtype 都属于科学算子身份。

将当前扩散 latent 按 NCHW 连续顺序展平为 $z_t\in\mathbb R^n$。完整特征函数为

$$
F:\mathbb R^n\rightarrow\mathbb R^{716},
\qquad
F(z)=[F_{\mathrm{CLIP}}(z);F_{\mathrm{struct}}(z)],
$$

局部 Jacobian 为

$$
J_t=\frac{\partial F(z_t)}{\partial z_t}.
$$

三个正式分支固定为 `lf_content`、`tail_robust` 和 `attention_geometry`, 公式中分别记为 $\mathrm{LF}$、$\mathrm{tail}$ 和 $\mathrm A$。

下列方法配置必须进入唯一版本化配置摘要, 不允许由环境变量或单次运行数据改写：

| 配置语义 | 冻结值 |
| --- | --- |
| `null_space_numerical_epsilon` | $10^{-12}$ |
| `risk_bounded_scale_direction_epsilon` | $10^{-12}$ |
| `risk_image_signal_interpolation_mode` / `risk_image_signal_align_corners` | `bilinear` / `false` |
| `risk_attention_signal_interpolation_mode` / `risk_attention_signal_align_corners` | `bilinear` / `true` |
| `risk_neutral_texture_value` | 0.5 |
| `attention_grid_align_corners` | `true` |
| `attention_anchor_count` | 12 |
| `attention_residual_threshold` | 0.20 |
| `attention_minimum_inlier_ratio` | 0.50 |
| `lf_kernel_size` / `lf_stride` / `lf_padding` / `lf_boundary_mode` | 5 / 1 / 2 / `zero_padding` |
| `lf_ceil_mode` / `lf_count_include_pad` / `lf_divisor_override` | `false` / `true` / `null` |
| `lf_detection_score_weight` / `tail_robust_detection_score_weight` | 0.70 / 0.30 |
| `tail_fraction` | 0.20 |
| `attention_backtracking_factor` / `attention_backtracking_maximum_steps` | 0.5 / 8 |
| `quantized_budget_envelope_backtracking_factor` / `quantized_budget_envelope_backtracking_maximum_steps` | 0.5 / 24 |
| `maximum_qr_condition_number` | $10^6$ |
| `maximum_orthogonality_error` | $10^{-5}$ |
| `qr_reference_solve_protocol` | `right_upper_triangular_solve_without_explicit_inverse_v1` |

风险权重顺序固定为 `(local_contrast, semantic, texture, adjacent_instability, attention_instability)`：

| 分支 | 权重 | 纹理解释 | $\tau_b$ | $b_{\min}^b$ | $b_{\max}^b$ | $\kappa_b$ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `lf_content` | (0.30, 0.30, 0.20, 0.20, 0.00) | `avoid` | 0.55 | 0.05 | 1.00 | 0.70 |
| `tail_robust` | (0.25, 0.25, 0.30, 0.20, 0.00) | `prefer` | 0.55 | 0.05 | 1.00 | 0.70 |
| `attention_geometry` | (0.20, 0.25, 0.05, 0.20, 0.30) | `neutral` | 0.55 | 0.05 | 1.00 | 0.70 |

上表每个数值必须通过真实 nested path 进入方法配置摘要。以 LF 为例, 精确字段为 `lf_content_risk_config.local_contrast_risk_weight`、`lf_content_risk_config.semantic_weight`、`lf_content_risk_config.texture_weight`、`lf_content_risk_config.adjacent_step_instability_weight`、`lf_content_risk_config.attention_instability_weight`、`lf_content_risk_config.texture_preference`、`lf_content_risk_config.eligibility_threshold`、`lf_content_risk_config.budget_floor`、`lf_content_risk_config.budget_ceiling` 和 `lf_content_risk_config.budget_gain`。另两个分支使用完全同构的 `tail_robust_risk_config.*` 与 `attention_geometry_risk_config.*`。实现中的 dataclass 默认值不能替代这些配置字段；`attention_geometry_risk_config.texture_preference=neutral` 必须结合 `risk_neutral_texture_value=0.5` 解释。

后文的 $\epsilon_{\mathrm{num}}$ 精确表示 `null_space_numerical_epsilon`, $\epsilon_{\mathrm{dir}}$ 精确表示 `risk_bounded_scale_direction_epsilon`。这两个 epsilon 角色不同, 不得共用隐式代码常量。

## `constructive_local_tangent_scope`

“潜流形”只表示当前 $z_t$ 处完整特征水平集的一阶局部切空间解释：

$$
T_{z_t}=\ker J_t.
$$

正式方法逐分支构造数值低响应方向并合成更新, 不声明求解联合标量 $\arg\max$。该术语不表示全局非线性流形、常秩定理条件、坐标图、测地线或回缩已经构造或验证。CPU 性质测试必须拒绝 `joint_argmax_solved=true`、全局流形已经构造以及把数值基底秩写成全局流形维数的解释。

## `frozen_model_operator_identity`

VAE decoder、VAE encoder、冻结 CLIP、扩散 Transformer、FlowMatch scheduler、公开检测 timestep、冻结 attention 层集合和完整方法配置共同定义科学算子。正式字段和值精确为：

| 字段 | 冻结值 |
| --- | --- |
| `model_id` / `model_revision` | `stabilityai/stable-diffusion-3.5-medium` / `b940f670f0eda2d07fbb75229e779da1ad11eb80` |
| `vision_model_id` / `vision_model_revision` | `openai/clip-vit-base-patch32` / `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| `pipeline_class_name` | `diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline` |
| `vae_class_name` | `diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL` |
| `transformer_class_name` | `diffusers.models.transformers.transformer_sd3.SD3Transformer2DModel` |
| `scheduler_class_name` | `diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler` |
| `vae_scaling_factor` / `vae_shift_factor` | 1.5305 / 0.0609 |
| `latent_torch_dtype` / `vision_torch_dtype` | `float16` / `float32` |
| `public_detection_schedule_index` | 7 |

解析后的 `configs/model_sd35.yaml` dataclass 记为 $C_{mathrm{method}}$。规范配置 payload 和摘要精确定义为

```text
{
  "formal_method_config_schema": "slm_wm_formal_method_runtime_config_v4",
  "formal_method_config": asdict(C_method)
}
```

以及

$$
	exttt{formal\_method\_config\_digest}
=operatorname{SHA256}
\left(operatorname{UTF8}(\operatorname{StableJSON}(payload))\right).
$$

这里 `StableJSON` 使用 `ensure_ascii=false`、键名升序和无空白分隔符 `(',', ':')`；摘要不含 YAML 排版、配置文件绝对路径或派生摘要本身。运行时实际 pipeline、VAE、Transformer、scheduler 的完全限定类名，VAE `scaling_factor`/`shift_factor`、latent/vision dtype、冻结 attention 层名和 `public_detection_schedule_index=7` 对应的 scheduler 索引都必须逐项等于上述配置, 并与 `formal_method_config_digest` 一起进入原子证据。VAE 编码和解码只能直接读取实际 `vae.config.scaling_factor` 与 `vae.config.shift_factor`; 任一字段缺失时必须失败, 不允许用1或0补齐。任一 revision 缺失、snapshot 漂移、类身份改变、VAE 参数改变或缺失、层名缺失、dtype 漂移、检测索引漂移或配置摘要不一致时, 当前科学单元必须失败。模型族名称或 `main` 分支名不能替代精确 revision。

## `branch_signal_origin`

设 VAE 解码图像 $x_t\in[0,1]^{N\times3\times H_x\times W_x}$, 灰度图为 $g_t=\frac13\sum_c x_{t,c}$。风险输入使用解析范围, 不执行逐样本 min-max。

1. CLIP patch-to-CLS 一致性：

$$
\phi_{\mathrm{sem}}=\operatorname{clip}_{[0,1]}
\left(\operatorname{Interp}_{\mathrm{bilinear},false}
\left(\frac{\cos(e_{\mathrm{patch}},e_{\mathrm{CLS}})+1}{2}\right)\right).
$$

2. 纹理图使用水平和垂直前向绝对差, 最后一列与最后一行补0：

$$
\phi_{\mathrm{tex}}=
\operatorname{clip}_{[0,1]}
\left(\operatorname{Interp}_{\mathrm{bilinear},false}
\left(\frac{|\nabla_xg_t|+|\nabla_yg_t|}{2}\right)\right).
$$

3. 局部对比风险使用反射填充2像素的5x5均值：

$$
\phi_{\mathrm{lcr}}=
\operatorname{clip}_{[0,1]}
\left(\operatorname{Interp}_{\mathrm{bilinear},false}
\left(|g_t-\operatorname{AvgPool}_{5\times5}(\operatorname{ReflectPad}_2(g_t))|\right)\right).
$$

4. 相邻步稳定度只比较紧邻上一 scheduler 步的真实解码 RGB：

$$
\phi_{\mathrm{adj}}=
1-\operatorname{clip}_{[0,1]}
\left(\operatorname{Interp}_{\mathrm{bilinear},false}
\left(\frac13\sum_c|x_{t,c}-x_{t-1,c}|\right)\right).
$$

5. $\phi_{\mathrm{attn\_stab}}$ 必须由不少于两个冻结层的直接 Q/K 关系行两两余弦一致性得到, 经 $(c+1)/2$ 映射并使用 `align_corners=true` 插值到 latent 网格。

缺失紧邻前一步 latent、真实 CLIP patch token、任一冻结 Q/K 层或使用隐藏状态稳定度替代 Q/K stability 时直接失败。每个 GPU 注入原子必须绑定五个输入图、当前/前一步 latent、当前/前一步解码 RGB、CLIP patch token、CLIP CLS token 和冻结 Q/K 层身份。

## `branch_risk_bounds_written_update`

### 分支风险与 NCHW 有效预算

对分支 $b$ 定义

$$
\rho_b(u)=\frac{1}{Z_b}\left[
\eta_c^b\phi_{\mathrm{lcr}}(u)
+\eta_m^b\phi_{\mathrm{sem}}(u)
+\eta_t^b\psi_b(\phi_{\mathrm{tex}}(u))
+\eta_d^b(1-\phi_{\mathrm{adj}}(u))
+\eta_A^b(1-\phi_{\mathrm{attn\_stab}}(u))
\right],
\qquad
Z_b=\eta_c^b+\eta_m^b+\eta_t^b+\eta_d^b+\eta_A^b.
$$

纹理项分别使用 $\psi_{\mathrm{LF}}(q)=q$、$\psi_{\mathrm{tail}}(q)=1-q$ 和 $\psi_{\mathrm A}(q)=0.5$。常数0.5表示注意力分支不随纹理升降改变风险方向, 但保留冻结的中性基线。连续预算、严格资格集合和有效预算为

$$
b_b(u)=\operatorname{clip}
\left(b_{\min}^b+\kappa_b(1-\rho_b(u)),b_{\min}^b,b_{\max}^b\right),
$$

$$
\Omega_b=\{u:\rho_b(u)<\tau_b\},
\qquad
b_b^{\mathrm{eff}}(u)=b_b(u)\mathbf1[u\in\Omega_b].
$$

等于阈值的位置不合格。空间预算 $[N,H,W]$ 在 channel 维复制为

$$
b_{b,nchw}^{\mathrm{eff}}[n,c,h,w]=b_b^{\mathrm{eff}}[n,h,w],
$$

再按 NCHW 连续顺序展平形成 $B_b=\operatorname{diag}(b_{b,nchw}^{\mathrm{eff}})$。风险求解、写回包络和记录摘要必须消费同一 Tensor 字节身份。

### 风险约束分支更新

设 $v_b\in\ker J_t$ 是单位安全方向。先执行零支持核验：若 $b_{b,nchw}^{\mathrm{eff}}(i)=0$ 且 $|v_b(i)|>\epsilon_{\mathrm{dir}}$, 则失败；不超过 $\epsilon_{\mathrm{dir}}$ 的非资格坐标置为精确0, 重新单位化并重新执行 JVP 门禁。

名义强度和绝对预算比例为

$$
a_b=\lambda_b\|z_t\|_2,
\qquad
\widehat b_b(i)=\frac{b_{b,nchw}^{\mathrm{eff}}(i)}{b_{\max}^b}.
$$

逐元素硬包络定义为

$$
E_b(i)=a_b\|v_b\|_\infty\widehat b_b(i).
$$

令 $\mathcal A_b=\{i:|v_b(i)|>\epsilon_{\mathrm{dir}}\}$。沿固定方向允许的最大步长为

$$
\alpha_b=\min\left(a_b,
\min_{i\in\mathcal A_b}\frac{E_b(i)}{|v_b(i)|}\right),
\qquad
\Delta z_t^b=\alpha_bv_b.
$$

$\mathcal A_b$ 为空、$\alpha_b\le\epsilon_{\mathrm{num}}$、任一非有限值或逐元素复验 $|\Delta z_t^b(i)|\le E_b(i)$ 失败时直接停止。禁止用当前样本的 `max(budget)` 重新归一化, 也禁止在风险投影后无条件恢复固定二范数强度。该全局标量缩放保持安全方向, 并保证固定方向下预算整体收缩不会增加实际更新强度。

## `complete_716_feature_jacobian`

$F_{\mathrm{CLIP}}$ 是冻结 CLIP 的512维投影后 `image_embeds` 的完整归一化 image embedding。冻结视觉模型必须显式返回 `image_embeds`; 缺失时直接失败, 不允许用投影前 `pooler_output` 或任一隐藏状态代替。204维结构向量严格为：3维通道均值、3维总体标准差、3维水平绝对梯度均值、3维垂直绝对梯度均值和192维8x8 RGB 自适应平均池化。正式输出宽度固定为716, 不允许坐标选择、随机投影、低维草图或把204维向量宣称为一般感知质量模型。

CPU 测试必须对解析图像逐坐标验证204维值, 并对小型可微特征函数验证完整宽度。GPU 原子必须记录两个冻结模型 revision、feature schema、各分量宽度和实际特征内容摘要。

## `exact_jacobian_low_response_subspace`

正式线性算子只能来自精确自动微分 JVP/VJP。底层算子不支持 `torch.func.linearize/vjp` 时可以重新执行精确 `torch.autograd.functional.jvp/vjp`；两条正式路径都不能使用有限差分、轨迹拟合或静默常量替代。CPU 测试还必须验证伴随恒等式

$$
\langle Jv,y\rangle=\langle v,J^\top y\rangle.
$$

对候选方向 $d_i^b$, 无阻尼 PSD-CG 求解

$$
(JB_b^2J^\top)y_i=JB_bd_i^b,
\qquad
u_i^b=B_bd_i^b-B_b^2J^\top y_i.
$$

投影能量使用平方二范数比例

$$
e_i=\frac{\|u_i^b\|_2^2}{\|B_bd_i^b\|_2^2}.
$$

设接受的路由候选和投影方向矩阵为

$$
V_b=[B_bd_{i_1}^b,\ldots,B_bd_{i_r}^b],
\qquad
U_b=[u_{i_1}^b,\ldots,u_{i_r}^b].
$$

正式 QR 使用 float32 reduced QR：

$$
U_b=N_bR_b.
$$

每列以首个绝对值大于 $\epsilon_{\mathrm{num}}$ 的元素为符号锚点；若该元素为负, 同时翻转 $N_b$ 对应列和 $R_b$ 对应行。若任一 $|R_{jj}|\le\epsilon_{\mathrm{num}}$ 或 $\operatorname{cond}_2(R_b)>10^6$, 当前基底失败。

QR 会混合候选列。独立参考方向定义为满足

$$
\widetilde V_bR_b=V_b
$$

的 $\widetilde V_b$。实现必须使用上三角求解得到 $\widetilde V_b$, 不显式构造 $R_b^{-1}$。逐列残差为

$$
r_{b,j}^{\mathrm{QR}}=
\frac{\|JN_b[:,j]\|_2}
{\max(\|J\widetilde V_b[:,j]\|_2,\epsilon_{\mathrm{num}})}.
$$

不得使用跨列共享 RMS。PSD-CG 最大64次、相对残差不高于 $10^{-6}$、阻尼为0；$e_i\ge0.01$、$r_{b,j}^{\mathrm{QR}}\le10^{-4}$、$\|N_b^\top N_b-I\|_F\le10^{-5}$。

每个分支必须使用角色限定字段绑定：

- `candidate_matrix_content_sha256`: $D_b$；
- `risk_budget_content_sha256`: 实际广播后的 $B_b$ 对角项；
- `routed_candidate_response_matrix_content_sha256`: $J(B_bD_b)$；
- `projected_direction_matrix_content_sha256`: $U_b$；
- `projected_direction_response_matrix_content_sha256`: $JU_b$；
- `latent_basis_content_sha256`: $N_b$；
- `basis_response_matrix_content_sha256`: $JN_b$；
- `basis_reference_response_matrix_content_sha256`: $J\widetilde V_b$。

每类响应只能使用上述角色限定字段, 并与对应数学对象的 dtype、shape 和连续原始字节内容绑定。

## `versioned_key_prg_reconstruction`

PRG 的 domain payload 精确为

```text
{
  "keyed_prg_version": keyed_prg_version,
  "key_material": key_material,
  "domain_fields": domain_fields,
  "shape": shape
}
```

`keyed_prg_version` 必须等于 `sha256_counter_normal_icdf_table20_float32_v2`。上述对象通过 UTF-8、`ensure_ascii=false`、键名升序和无空白分隔符 `(',', ':')` 形成 stable JSON, 再执行 SHA-256 得到32字节 domain digest。计数器从0开始, 以16字节无符号大端编码拼接在 domain digest 后；每个计数器输入再执行 SHA-256。连续32字节输出块组成 MSB-first 大端比特流。Gaussian 路径跨块连续提取20位索引 $i$, 不按字节边界丢弃剩余比特, 并查询

$$
q_i=\operatorname{round}_{\mathrm{binary32}}\!\left(
\Phi^{-1}\!\left(\frac{i+0.5}{2^{20}}\right)
\right),
\qquad 0\le i<2^{20}.
$$

冻结表包含1048576个 binary32 位模式, 完整表按大端顺序连接后的 SHA-256 必须为 `70abf440a7f3670147965ffa52f5aaa639dab97f6282b68f3a9a1b1ce5e6cf5a`。运行时只解码冻结位模式, 不调用平台 `log`、`sqrt`、`sin`、`cos` 或逆 CDF。该概率律是有限离散的 Q20 中点逆 CDF 量化标准正态, 不是连续精确的 $\mathcal N(0,1)$。理想中点离散化的 KS 距离为

$$
D_{KS}^{\mathrm{midpoint}}=2^{-21}
=4.76837158203125\times10^{-7},
$$

计入已登记的最大 float32 CDF 舍入误差 `1.4386451474557305e-08` 后, 总 KS 距离上界为 `4.912236096776823e-7`。LF/tail 模板、Jacobian 密钥候选、公开检测噪声和论文实验共享基础 latent 都使用该 Q20 路径。输出先在 CPU 物化为规范 float32 Tensor；需要其他 dtype 时必须在 CPU 完成转换, 随后才搬运到执行设备。论文实验共享基础 latent 可复用同一通用原语, 但不属于核心水印方法密钥定义。

attention 关系符号使用独立 uniform 路径。每个 SHA-256 块按 offset 0、8、16、24 划分为4个连续8字节无符号大端 word, 对每个 word 取高53位

$$
m=w\gg11,
\qquad
u=\frac{m+1}{2^{53}+2},
$$

得到严格位于 $(0,1)$ 的 float64 值, 再物化为规范 CPU float32。该路径只把 $u<0.5$ 映射为 $-1$, 否则映射为 $+1$；不得用于高斯 Tensor。

公开检测噪声也使用同一 Gaussian 路径。令待检图像 VAE posterior mode 重编码后的 NCHW Tensor 为 $\hat z$, 则输出 `shape` 精确为按轴顺序序列化的 `tuple(int(v) for v in hat_z.shape)`。公开调用不使用方法密钥, 而固定使用

```text
key_material = public_detection_noise_domain
             = "public_image_only_qk_detection_noise_v1"
domain_fields = {
  "operator": "public_image_only_qk_detection_noise_v1",
  "model_id": "stabilityai/stable-diffusion-3.5-medium",
  "model_revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
  "width": 512,
  "height": 512,
  "inference_steps": 20,
  "public_detection_schedule_index": 7,
  "latent_shape": shape
}
```

`public_detection_noise_prg_protocol` 必须等于 `sha256_counter_normal_icdf_table20_float32_v2`。上述 `key_material`、`domain_fields` 和 `shape` 进入本节定义的完整 stable JSON domain payload, 经 SHA-256、从0开始的16字节大端计数器、MSB-first 连续20位索引提取和冻结 Q20 表查询得到 CPU float32 Tensor；实际 latent dtype 转换在 CPU 完成, 随后才搬运到执行设备。该公开 domain 不含水印密钥、Prompt、生成 seed 或生成轨迹。

CPU/CUDA 设备 RNG 不参与方法身份。PRG 算法摘要固定为 `a6266dc1fb4a59f8038062dcd120f145582153138b8176baae12013d5a22687b`, 核心方法定义摘要固定为 `8875c24ad29344b2ea8a6fc6ae62d8bccaafd0fa426e8072e2380bb1abca7d43`, 正式随机化协议摘要固定为 `d09928b763c17d2c68fa2bc3921b59b76c3df2fc264fb364ae33bcc8bdeba0d5`。`tools/harness/verify_normal_quantile_reference.py` 以 `normal_quantile_reference_verification_protocol=mpfr_192bit_erf_midpoint_bracket_and_newton_v2` 使用192位 MPFR, 对正半轴全部524288个表项验证相邻 binary32 中点的 CDF 严格夹逼, 并以 Newton 根作为独立交叉检查；报告同时登记最小中点概率余量。该复验是外层参考证据, 不进入 PRG 算法摘要或采样身份。固定 test vector 必须分别覆盖20位索引跨 SHA-256 块边界、Q20 值、uniform 值、relation signs、公开检测噪声和正式 shape 基础 latent。当前固定向量只在 Windows CPU 实测；Linux/Colab 的逐字节 KAT 仍是 GPU 运行前阻断门禁, 不得表述为已经完成跨平台实测。

## `spatial_low_pass_and_amplitude_tail_carriers`

LF 对每个 batch/channel 独立执行 kernel 5、stride 1、padding 2、零填充、`ceil_mode=false`、`count_include_pad=true` 和 `divisor_override=null` 的二维平均池化。低通卷积只作用于 height/width, 不跨 batch 或 channel 传播样本值；池化完成后, 去均值和 L2 归一化明确在整个模板 Tensor 上各计算一个全局标量。嵌入、raw 检测和 aligned 检测必须消费同一版本化 LF 协议；完整检测分数固定使用 LF 权重0.70与尾部权重0.30。

`tail_robust` 按高斯模板元素的 $(|\nu_i|,-i)$ 降序稳定排序, 精确保留 $\lceil n\gamma\rceil$ 个元素, 其余位置保持精确0, 随后只除以整体二范数, 不执行会使非入选位置重新非零的去均值操作。该分支不执行 FFT、DCT、带通滤波或空间频率 mask, 不具有空间高频语义。

尾部载体协议固定为 `slm_wm_tail_robust_carrier_protocol_v1`, 正文同时绑定 `tail_fraction`、绝对幅值降序且展平索引升序的选择规则和密钥 PRG 版本。仅图像检测记录必须保存模板 NCHW 形状、元素总数、$\lceil n\gamma\rceil$ 选中数、阈值、实际保留比例及模板内容摘要；raw 与 aligned 路径必须具有相同 latent 形状、LF 模板摘要、尾部模板摘要、尾部阈值和保留比例。

两个模板分别投影到对应 $N_b$, 投影平方能量保留率低于0.01或投影方向近零时直接失败；最终更新必须再经过分支风险硬包络。

载体构造时, 每个投影身份必须由模板 shape、规范模板内容摘要、投影方向内容摘要、Null Space 摘要、最小能量保留率、实际平方能量保留率、Tensor 摘要版本、PRG 摘要和载体协议摘要共同计算。正式样本记录不复制该内部摘要正文, 只保存模板摘要、模板内容摘要、模板 shape、协议摘要和投影能量比例。完整注入、carrier-only 反事实及 `registered_watermark_key` 检测必须对每个活动内容分支形成唯一相同的模板 shape、规范模板内容摘要和协议摘要；`registered_wrong_key_negative` 在自身 raw/aligned 路径中保持唯一模板身份, 使用相同载体协议, 且模板摘要必须与注册密钥不同。

## `direct_qk_four_component_relation`

正式层集合精确固定为 `transformer_blocks.0.attn` 与 `transformer_blocks.23.attn`。每个模块必须显式公开非 bool 的正整数 `heads`, 缺失、非整数或不大于0时直接失败, 不允许静默按单头解释。冻结层 recorder 的每次 hook 调用必须取得可核验的 `hidden_states` Tensor; 没有 Tensor 输入时必须立即失败, 不允许跳过当前层并继续形成不完整 Q/K 记录。仅使用冻结空文本条件, 直接调用模块 `to_q`、`to_k` 和公开 Q/K normalization。设 source grid 边长为 $s$, 抽样边长为 $m=\min(s,\lfloor\sqrt{64}\rfloor)$ 且要求 $m\ge2$, 抽样 token 总数为 $n=m^2$, 每轴索引为

$$
i_k=\operatorname{round}\frac{k(s-1)}{m-1},\qquad k=0,\ldots,m-1.
$$

对每个头先计算

$$
\ell_{ij}^{(h)}=q_i^{(h)\top}k_j^{(h)}/\sqrt{d_h},
$$

再定义

$$
L_{ij}=\frac1H\sum_h\left(\ell_{ij}^{(h)}-\frac1n\sum_k\ell_{ik}^{(h)}\right),
$$

$$
P_{ij}=\frac1H\sum_h\operatorname{softmax}_j(\ell_{ij}^{(h)}),
$$

$$
\rho_{ij}=\frac1n\left[1+\sum_{k\ne j}\sigma\left(\frac{L_{ik}-L_{ij}}{0.25}\right)\right],
$$

$$
G_{ij}=(P_{ij}-\overline P_i)(D_{ij}-\overline D_i),
\qquad
D_{ij}=\|p_i-p_j\|_2/(2\sqrt2).
$$

这里 $p_i$ 使用角点 token 中心分别为 -1 和1的公开二维坐标, token stability 插值和图像仿射重采样统一使用 `align_corners=true`。$P$ 是抽样图像 token 子图概率, 不声明为完整 joint-attention 权重。任何读取 $P$ 的核心算子, 包括四分量描述、关系图身份、跨层 stability、稳定 token 选择、pair weight 和几何分数, 都必须接收 `QKAttentionRelation`, 要求 `relation_source=direct_qk_centered_logits_and_probabilities`, 并从当前 Tensor 重新核验完整 Q/K operator metadata 与 Q/K atom 内容身份。每条外层 `(layer_name, relation, token_indices)` 记录还必须满足 `layer_name=relation.metadata.record_layer_name` 和 `token_indices=relation.metadata.sampled_token_indices`; 多层记录的层名必须唯一。裸概率 Tensor、错误 `relation_source`、缺失算子元数据、缺失 Q/K 原子内容、外层改名、更换外层 token 索引或复制同一内部层冒充多层时必须立即失败, 不允许继续计算 stability、saliency、token 选择、pair weight、关系图身份或几何分数。

四分量为 $R=[L,\rho,P,G]$。对每个冻结层 $\ell$, PRG 使用 `operator=attention_relation_signs`、精确层名和 token 数构造独立 $n\times n$ uniform Tensor $U^\ell$。密钥对称符号图精确定义为

$$
S_{K,ij}^{\ell}=
\begin{cases}
0,&i=j,\\
1,&i<j\ \land\ U_{ij}^{\ell}\ge0.5,\\
-1,&i<j\ \land\ U_{ij}^{\ell}<0.5,\\
S_{K,ji}^{\ell},&i>j.
\end{cases}
$$

四分量 polarity 固定为 $\pi=(1,-1,1,1)$, 因而第 $c$ 个目标图为

$$
T_{ijc}^{\ell}=\pi_cS_{K,ij}^{\ell}.
$$

### 稳定 token 与 pair 权重

对 batch 样本 $b$、层 $\ell$ 和 token $i$, 先把概率行中心化并单位化：

$$
\widetilde P_{bi:}^{\ell}=P_{bi:}^{\ell}-\frac1n\mathbf1,
\qquad
\overline P_{bi:}^{\ell}=
\frac{\widetilde P_{bi:}^{\ell}}
{\max(\|\widetilde P_{bi:}^{\ell}\|_2,\epsilon_{\mathrm{num}})}.
$$

跨层中心化概率行一致性为

$$
\operatorname{Stab}_i=
\operatorname{clip}_{[0,1]}\left(
\operatorname{Mean}_{b,\ell<r}
\left[
\frac{1+\langle\overline P_{bi:}^{\ell},
\overline P_{bi:}^{r}\rangle}{2}
\right]\right).
$$

incoming attention centrality 先对 source row、batch 和层求均值, 再在 token 轴归一化：

$$
c_i=\operatorname{Mean}_{\ell,b,j}P_{bji}^{\ell},
\qquad
\operatorname{Cent}_i=
\frac{c_i}{\max(\sum_r c_r,\epsilon_{\mathrm{num}})}.
$$

排序分数为

$$
q_i=\operatorname{Stab}_i\operatorname{Cent}_i.
$$

选择数量固定为 $k=\min(n,\max(4,\lceil0.5n\rceil))$。按 $(-q_i,\operatorname{token\_index}_i)$ 升序稳定排序选择前 $k$ 个, 随后仅为规范记录把选中位置升序排列。单点和 pair 权重为

$$
a_i=\begin{cases}1,&i\in\mathcal V_A,\\0.25,&i\notin\mathcal V_A,\end{cases}
\qquad
w_{ij}=a_ia_j\mathbf1[i\ne j].
$$

一次注入内的梯度、内容基底、回溯和实际写回必须消费同一个 $(\mathcal V_A,a,w)$ 身份。

### 逐行加权相关与完整分数

对 batch $b$、层 $\ell$、row $i$ 和分量 $c$, 令 $W_i=\sum_jw_{ij}$, 并分别计算

$$
\mu_{R,bic}^{\ell}=\frac{\sum_jw_{ij}R_{bijc}^{\ell}}{W_i},
\qquad
\mu_{T,ic}^{\ell}=\frac{\sum_jw_{ij}T_{ijc}^{\ell}}{W_i}.
$$

逐行加权中心化相关为

$$
\operatorname{RowCorr}_{bic}^{\ell}=
\frac{
\sum_jw_{ij}(R_{bijc}^{\ell}-\mu_{R,bic}^{\ell})
(T_{ijc}^{\ell}-\mu_{T,ic}^{\ell})
}{
\sqrt{
\sum_jw_{ij}(R_{bijc}^{\ell}-\mu_{R,bic}^{\ell})^2
\sum_jw_{ij}(T_{ijc}^{\ell}-\mu_{T,ic}^{\ell})^2
}
}.
$$

$W_i=0$ 或任一加权中心化能量不大于 $\epsilon_{\mathrm{num}}^2$ 时该行无效。每个 batch 先对有效 row 取均值；没有有效 row 时该 batch 分量分数为0。随后依次对 batch 和冻结层等权平均：

$$
C_c^{\ell}=
\frac1B\sum_b
\operatorname{Mean}_{i\in\mathcal I_{b\ell c}^{valid}}
\operatorname{RowCorr}_{bic}^{\ell},
\qquad
C_c=\frac1{|\mathcal L|}\sum_{\ell\in\mathcal L}C_c^{\ell}.
$$

完整方法分量权重固定为 $\omega=(1/4,1/4,1/4,1/4)$, 最终可微注意力目标精确为

$$
s_A(z;K)=\sum_{c=1}^{4}\omega_cC_c(z;K).
$$

均匀 $P$ 必须使 $G=0$。任何把平均 logits 的 softmax、绝对坐标分数、跨分量先合并再相关或不同 batch/layer 非等权聚合写入正式路径的实现都不满足该不变量。

每层原子必须绑定 Q、K、中心化 $L$、逐头 softmax 后平均的 $P$、原二维索引、层名、模块类、head count、head width、scale、Q/K normalization 和 source/sample grid。缺任一内容时失败。

## `direct_qk_monotonic_attention_update`

在原 latent 计算并冻结稳定 token 选择, 在内容基底

$$
z_t^{\mathrm{base}}=z_t+\Delta z_t^{\mathrm{LF}}+\Delta z_t^{\mathrm{tail}}
$$

重算真实四分量梯度 $g_A=\nabla_zs_A(z)|_{z=z_t^{\mathrm{base}}}$。令

$$
v_A=\operatorname{Norm}(N_AN_A^\top g_A),
$$

attention 单调回溯必须直接消费已经物化的 `RiskBoundedUpdate`，不得只复制其标量强度后重新计算或替换方向。以该对象的最大步长 $\alpha_A$ 为起点, 每个候选都通过同一对象缩小并重新物化, 依次检查 $\alpha_A2^{-k}$, $k=0,\ldots,8$。接受候选的 update 内容摘要必须与进入三分支合成的 attention 分支 update 内容摘要完全相同。候选必须满足

$$
s_A(z_t^{\mathrm{base}}+\alpha_A2^{-k}v_A)
>
\max\{s_A(z_t),s_A(z_t^{\mathrm{base}})\}.
$$

安全投影为零、九个候选均失败、分数非有限或 Q/K 原子不完整时直接失败。单次注入 Q/K 角色精确为 `latent_before`、`optimization_content_base_latent`、`accepted_attention_candidate`、`actual_written_content_base_latent` 和 `actual_written_combined_latent`。每个角色必须共同绑定实际 float32 求值 latent 内容 SHA-256、该次 Q/K 分数和逐层 Q/K 原子；角色分数必须与顶层两阶段单调门禁字段精确交叉复验。其中前3个角色证明 attention 分支在未执行共同缩放前满足单调回溯，后2个角色证明实际 dtype 写回候选在共同缩放后仍严格优于同尺度内容基底；优化基底与实际写回基底不得合并为同一角色。

## `three_branch_update_composition`

核心方法只执行一次

$$
\Delta z_t=
\Delta z_t^{\mathrm{LF}}+
\Delta z_t^{\mathrm{tail}}+
\Delta z_t^{\mathrm A}.
$$

纯组合与共同缩放规则属于 `main/`；attention 内部回溯候选与最终实际写回都必须调用同一个 `compose_ordered_float32_update_once` 原语，先按冻结分支顺序形成 float32 联合更新，再与 original latent 的 float32 表示相加并只执行一次 dtype cast。`experiments/` 只绑定真实 SD3.5 callback、scheduler 和记录, 不得实现第二套加法结合顺序或简化组合。

## `actual_dtype_write_revalidation`

三分支首先在 float32 中构造, 并严格按 `lf_content -> tail_robust -> attention_geometry` 顺序在 float32 累加。不得逐分支 cast, 也不得改变求和顺序。为处理实际 dtype 舍入, 定义共同缩放候选 $q_k=2^{-k}$, $k=0,\ldots,24$：

$$
z_{C,k}=\operatorname{cast}_{d}
\left(z_t^{(32)}+q_k(\Delta z_t^{\mathrm{LF}}+\Delta z_t^{\mathrm{tail}})^{(32)}\right),
$$

$$
z_{F,k}=\operatorname{cast}_{d}
\left(z_t^{(32)}+q_k\Delta z_t^{(32)}\right),
\qquad
\Delta z_{t,k}^{\mathrm{written}}=z_{F,k}-z_t.
$$

合成包络为

$$
E_{\mathrm{sum},k}(i)=q_k\sum_bE_b(i).
$$

包络核验不使用 epsilon 放宽边界：当 $E_{\mathrm{sum},k}(i)=0$ 时要求 $\Delta z_{t,k}^{\mathrm{written}}(i)=0$；其余位置要求

$$
\max_{i:E_{\mathrm{sum},k}(i)>0}
\frac{|\Delta z_{t,k}^{\mathrm{written}}(i)|}{E_{\mathrm{sum},k}(i)}\le1.
$$

接受候选还必须非零、通过实际 Tensor 完整特征 JVP 和有限变化门禁。attention 启用时还必须重新提取 $z_{F,k}$ 与 $z_{C,k}$ 的真实 Q/K, 满足

$$
s_A(z_{F,k})>\max\{s_A(z_t),s_A(z_{C,k})\}.
$$

选择第一个同时满足全部条件的 $k$。初始候选及最多24次共同减半全部失败时停止运行, 不允许增加 ULP 容差、改用量化前更新或继续写入零更新。记录必须保存固定求和顺序、共同缩放因子、候选次数、联合包络、实际写回和全部复验结果。

## `finite_feature_preservation`

每次接受的实际写回都必须满足

$$
\frac{\|J_t\Delta z_t^{\mathrm{written}}\|_2}
{\max(\|F(z_t)\|_2,\epsilon_{\mathrm{num}})}\le10^{-4}.
$$

局部有限更新及最终成图还必须满足 CLIP cosine 不低于0.995且204维结构相对漂移不高于0.02。该门禁只说明冻结特征坐标的保持, 不替代 FID/KID、配对感知质量或人工评价。

## `final_image_attention_attribution`

最终成图 attention 归因必须包含 clean 图像 $x_0$、仅关闭 attention geometry 的 carrier-only 图像 $x_C$ 和完整方法图像 $x_F$。carrier-only 与完整方法必须共享生成 seed、基础 latent 字节、scheduler、注入步、LF/tail 配置和所有非 attention 算子；carrier-only 的逐注入原子中不得出现 attention 分数、更新、关系、pair 身份或 attention Null Space。

$x_0\leftrightarrow x_C$、$x_0\leftrightarrow x_F$ 和 $x_C\leftrightarrow x_F$ 三条边都必须通过有限特征保持。随后要求自身盲选增益和冻结 carrier-only pair 权重增益均严格大于0.0001。最终图像 Q/K 角色精确为 `final_clean_image`、`final_carrier_only_image` 和 `final_watermarked_image`。该反事实估计 attention 开关经后续轨迹交互传播的总机制效应, 不声明纯直接效应。

## `image_only_detection_boundary`

正式接口只允许

$$
\operatorname{Detect}(x',K,M).
$$

检测图像经公开预处理后使用冻结 VAE posterior mode 编码：

$$
\hat z=(\operatorname{mode}(q_{\mathrm{VAE}}(x'))-\texttt{vae\_shift\_factor})\cdot\texttt{vae\_scaling\_factor}.
$$

LF/tail 模板只由 $K$、精确模型标识和 $\hat z$ shape 重建。内容分数固定为

$$
s_c=0.70\operatorname{Corr}(\hat z,\nu_{\mathrm{LF}})
+0.30\operatorname{Corr}(\hat z,\widetilde\nu_{\mathrm{tail}}).
$$

完整方法的0.70/0.30权重、LF 协议摘要、尾部协议摘要和 `tail_fraction` 随 calibration clean negatives 一起进入唯一 `threshold_digest`。应用冻结阈值前必须从全部协议字段重新计算该摘要, 并由计数重算 calibration 假阳性率；test、攻击和消融记录不得替换任一载体协议或权重。

仅图像检测密钥计划固定包含 `registered_watermark_key` 与 `registered_wrong_key_negative` 两个角色。wrong-key 由 `registered_key_and_sha256_domain_separated_wrong_key_v1` 从当前注册水印密钥确定性派生；记录只保存两个材料摘要、版本化计划正文和计划摘要, 不保存密钥原文。完整注入与 carrier-only 更新中的 `watermark_key_material_digest_random` 必须等于计划中的注册密钥摘要；注册密钥检测模板必须与嵌入模板相同, wrong-key 模板必须在每个分支内部唯一且与注册密钥模板不同。只修改 `sample_role` 不能建立 wrong-key 证据。

检测 Q/K 的公开 schedule 索引和实际 timestep 精确为

$$
k_{det}=\texttt{public\_detection\_schedule\_index}=7,
\qquad
t_{det}=\operatorname{scheduler.timesteps}[k_{det}].
$$

令 $shape_{det}=\operatorname{shape}(\hat z)$, 则公开噪声固定为

$$
\epsilon_{det}=\operatorname{PRG}_{\mathcal N}
\left(
shape_{det},
\texttt{key\_material}=\texttt{public\_detection\_noise\_domain},
\texttt{domain\_fields}=D_{det}
\right),
$$

其中 `public_detection_noise_domain=public_image_only_qk_detection_noise_v1`, $D_{det}$ 是 `versioned_key_prg_reconstruction` 中逐字段冻结的公开 domain。该调用完整执行 SHA-256 大端计数器、MSB-first 连续20位索引提取和冻结 Q20 中点逆 CDF float32 表查询, 在 CPU 完成实际 dtype 转换后再搬运到 $\hat z$ 的设备。加噪 latent 必须由冻结 scheduler 的真实

$$
z_{det}=\operatorname{scheduler.scale\_noise}(\hat z,t_{det},\epsilon_{det})
$$

得到；缺失 `scale_noise` 时失败, 不允许线性 latent/noise 混合替代。Transformer 条件固定为 `public_detection_conditioning_protocol=sd3_empty_text_triplet_without_cfg_v1` 和 `public_detection_condition_text=""`。该噪声与水印密钥、Prompt、生成 seed 和生成轨迹无关。

检测器不得读取 Prompt、源 latent、采样轨迹、样本级 Null Space、生成端 Q/K 原子或生成 seed。raw 图与对齐图必须分别重新编码并重新提取 Q/K, 原子角色精确为 `raw_detection_image` 和 `aligned_detection_image`。

## `same_threshold_geometry_rescue`

检测端只在 raw 图像 Q/K 上盲选一次稳定 token。对公开有界相似仿射与方形二面体候选 $T$, 关系拉回和密钥前推为

$$
\widehat R_{T,c}^{(l)}=W_TR_{\mathrm{obs},c}^{(l)}W_T^\top,
\qquad
\widetilde S_{T,c}^{(l)}=V_T(\pi_cS_K^{(l)})V_T^\top.
$$

观测权重为 $w_{\mathrm{obs}}$；规范权重先传递单点权重 $a_{\mathrm{can},T}=W_Ta_{\mathrm{obs}}$, 再执行非对角外积, 不能用 $W_Tw_{\mathrm{obs}}W_T^\top$ 代替。双向分数和注册目标为

$$
s_{\mathrm{can}}(T,l)=\frac14\sum_c
\operatorname{RowCorr}_{w_{\mathrm{can},T}}(\widehat R_{T,c}^{(l)},S_{K,c}^{(l)}),
$$

$$
s_{\mathrm{obs}}(T,l)=\frac14\sum_c
\operatorname{RowCorr}_{w_{\mathrm{obs}}}(R_{\mathrm{obs},c}^{(l)},\widetilde S_{T,c}^{(l)}),
$$

$$
J(T,l)=0.10s_{\mathrm{can}}(T,l)+0.90s_{\mathrm{obs}}(T,l)
-0.01\sum_{q\in\{c_{\mathrm{can}},u_{\mathrm{can}},c_{\mathrm{obs}},u_{\mathrm{obs}}\}}(1-q).
$$

identity 变换 $I$ 必须作为候选集合中的精确候选实际执行。每层结构目标增益定义为

$$
\Delta J(l)=J(\widehat T_l,l)-J(I,l).
$$

检测记录必须同时保存 $J(\widehat T_l,l)$、$J(I,l)$ 与 $\Delta J(l)$, 并能由前两者独立重算后者。候选集合缺少精确 identity、三者公式不一致或 $\Delta J(l)\le0$ 时, 该层结构注册失败；不得用次优候选、近邻候选或截断后的目标差替代。

冻结层有序集合精确为 $\mathcal L=(\texttt{transformer\_blocks.0.attn},\texttt{transformer\_blocks.23.attn})$。每层先独立完成层内搜索并得到 $\widehat T_l$；跨层结果按

$$
(\widehat l,\widehat T)=
\operatorname*{lexargmax}_{l\in\mathcal L}
\left(J(\widehat T_l,l),s_{\mathrm{obs}}(\widehat T_l,l),r_{\mathrm{reg}}(\widehat T_l,l),-\operatorname{rank}_{\mathcal L}(l)\right)
$$

唯一选择。比较优先级依次是注册目标、观测关系分和注册置信度；三者完全相同时选择冻结层顺序中更靠前的层。检测器不得依赖回调或容器的偶然遍历顺序改变该裁决。

搜索定义域固定为残余旋转 $[-32,32]$ 度、均匀尺度 $[1/\sqrt2,\sqrt2]$ 和两个归一化平移分量 $[-0.28,0.28]$, 并使用三层三分局部细化；搜索器不得读取攻击参数。结构可靠性要求观测和双向关系分数为正、$\Delta J(l)>0$、两个方向覆盖率均不低于0.45, 并通过下述预注册结构门禁。aligned 图像重新提取真实 Q/K 后, 使用传递的同一 pair 身份计算 sync, 不重新选择稳定 token。

令当前规则二维抽样网格包含 $n$ 个 token。正式锚点数量固定为 $A=12$；若 $n<12$, 当前检测单元必须失败, 不允许缩减锚点数。锚点位置按

$$
h_j=\operatorname{round}\left(\frac{j(n-1)}{A-1}\right),
\qquad j=0,\ldots,A-1
$$

在抽样 token 索引范围内确定性均匀选择。所有锚点和残差都使用 `normalized_xy_token_centers_corner_endpoints_v1` 坐标, 角点 token 中心分别为 -1 与1；关系图采样和图像仿射重采样统一使用 `align_corners=true`。对候选恢复变换, 仅具有有效双线性覆盖的锚点进入分母集合 $\mathcal V$。锚点同时满足观测匹配唯一且归一化 xy 欧氏残差 $d_j\le0.20$ 时记为内点, 因而

$$
r_{\mathrm{inlier}}
=\frac{\sum_{j\in\mathcal V}\mathbf1[\operatorname{unique}(j)\land d_j\le0.20]}
{|\mathcal V|},
\qquad
r_{\mathrm{inlier}}\ge0.50.
$$

无有效覆盖锚点时 $r_{\mathrm{inlier}}=0$；无内点时平均内点残差为非有限失败值。`attention_anchor_count=12`、`attention_residual_threshold=0.20` 和 `attention_minimum_inlier_ratio=0.50` 是正式方法的预注册结构常量。完整检测器配置只保存在顶层运行 manifest；alignment 和样本记录保存决策所需字段及检测器配置摘要, calibration/test 必须绑定同一摘要。calibration 或 test 数据不得选择、放宽或替换这三项常量；任一决策字段缺失、数值漂移或配置摘要不一致都使证据闭合失败。

令内点平均残差为 $e_{\mathrm{affine}}$, 则跨层裁决与 calibration 共同使用的注册置信度精确定义为

$$
r_{\mathrm{reg}}=
\max(0,0.10s_{\mathrm{can}}+0.90s_{\mathrm{obs}})
\cdot r_{\mathrm{inlier}}
\cdot\exp(-e_{\mathrm{affine}})
\cdot\min(c_{\mathrm{can}},c_{\mathrm{obs}}).
$$

使用 $\widehat T$ 恢复待检图像时, 输入先解码为 RGB uint8 Tensor 并归一化到 $[0,1]$。仿射网格和图像采样固定采用 bilinear、`padding_mode=border` 与 `align_corners=true`。连续结果执行 $\operatorname{floor}(255\cdot\operatorname{clip}(x,0,1))$, 转回 RGB uint8 后才进入 aligned VAE 编码和 Q/K 重提取。

calibration clean negatives 只冻结内容阈值 $\tau_c$、几何关系分阈值、注册置信度阈值、恢复后同步分阈值和 rescue window。失败原因由冻结内容余量与几何可靠性布尔规则确定, 不作为可从 calibration 或 test 调参的独立机制。原图主判为

$$
positive_{content}=\mathbf1[s_c^{raw}-\tau_c\ge0].
$$

只有

$$
\delta_{low}\le s_c^{raw}-\tau_c<0,
$$

失败原因为 `geometry_suspected` 或 `low_confidence`, 且双向注册、覆盖、唯一采样、inlier、残差、pair 身份和恢复后真实 Q/K sync 全部通过时才允许对齐。恢复图像不得重新选择稳定 token, 并必须复用同一个 $\tau_c$：

$$
rescue=eligible\land\mathbf1[s_c^{aligned}-\tau_c\ge0],
$$

$$
y_{evidence}=positive_{content}\lor rescue.
$$

几何分数不能独立产生 positive。test split 应用冻结阈值时必须重新计算 threshold-dependent `fail_reason`, 不得沿用生成记录中的临时分类。

## `scientific_content_binding`

`slm_wm_tensor_content_v1` 按协议版本、dtype、shape 和连续原始字节绑定 Tensor 身份。最终图像不能只绑定 PNG 文件字节, 因为相同像素可以使用不同压缩参数编码。`slm_wm_image_rgb_uint8_content_v1` 必须先把持久化图像解码为 RGB, 再按以下固定顺序计算规范 RGB uint8 像素摘要:

$$
h_{rgb}=\operatorname{SHA256}(
\texttt{slm\_wm\_image\_rgb\_uint8\_content\_v1}
\Vert \operatorname{uint64be}(W)
\Vert \operatorname{uint64be}(H)
\Vert \operatorname{RGBBytes}_{uint8}).
$$

一次方法运行的写入端必须先持久化更新 JSONL、检测 JSONL 和图像文件, 再从磁盘重新读取更新 JSONL、检测 JSONL 和最终图像构造 `scientific_content_binding_record`。该记录按冻结顺序联合绑定:

1. 当前/相邻解码 RGB、CLIP patch/CLS 和五类风险信号。
2. 三个分支的风险值、预算、资格 mask、有效预算、单位方向、硬包络和实际分支写回。
3. 每个活动分支 Null Space 的候选矩阵、风险预算、路由响应、投影方向、投影响应、最终基底、基底响应和逐列参考响应八类内容身份。
4. 量化写回前后 latent、三个分支更新、联合包络、实际 dtype 写回、完整716维参考特征 Tensor 与实际写回 JVP Tensor, 并重算唯一量化合成证据摘要。
5. 注意力注入的 `latent_before`、`optimization_content_base_latent`、`accepted_attention_candidate`、`actual_written_content_base_latent`、`actual_written_combined_latent` 五角色真实 Q/K 原子及精确算子元数据。
6. clean、carrier-only 和 watermarked 三张最终图像的文件摘要与规范 RGB uint8 像素摘要, 以及三角色最终图像 Q/K 原子、公开噪声证据和图像到 Q/K/噪声的逐角色绑定。
7. 每次仅图像检测的 source/evaluated 图像文件与像素身份、raw/aligned 检测 Q/K、公开检测噪声 Tensor、版本化 PRG 身份和逐评价噪声证据。
8. carrier-only 更新链、最终图像保持记录和 attention 开关反事实身份。

公开检测噪声由同一个 detector extractor 在一次方法运行内共享。最终 clean、carrier-only、watermarked 三图评价固定消费全局索引0、1、2, 因而首条 detection 的 raw 评价必须从索引3开始；后续 raw/aligned 评价沿实际调用顺序形成 `range(3, 3+n)`。禁止每进入一条 detection 就归零, 也禁止接受任意非3起点。公开噪声内容摘要、PRG 身份、Q/K 原子和图像到 Q/K 绑定必须逐评价引用同一个全局索引。执行对齐时, `alignment_digest` 必须是完整 alignment 记录的规范小写 SHA-256；未执行对齐时该字段必须为空字符串, 对象字符串化或任意文本不能充当对齐身份。

最终三图 Q/K 不能只记录 Q/K Tensor。每个角色必须持久化公开噪声证据记录, 并逐项复验证据索引分别为0、1、2、噪声 Tensor 内容摘要相同、版本化 PRG 身份相同且证据摘要可由叶子重建。`final_image_qk_image_content_bindings` 的每一项必须同时包含图像像素摘要、Q/K 原子摘要、公开噪声内容摘要、PRG 身份和评价索引。后续检测的首个噪声身份必须与最终三图完全相同, 只能从索引3继续；为最终三图与检测分别构造两套公开噪声、只记录索引而不记录 Tensor/PRG 身份或让像素-Q/K 绑定遗漏噪声身份均必须拒绝。

结果读取端不得信任已保存的顶层摘要。`load_completed_semantic_watermark_runtime_result` 必须重新读取上述 JSONL 和图像文件, 重算叶子内容、子记录摘要、有序角色摘要及顶层摘要, 并要求重建记录与已保存记录完全相等。路径存在、文件 SHA-256 相同或顶层摘要格式合法均不能替代该重建。

数据集汇总不得直接信任每个样本已保存的 `scientific_content_binding_digest`。汇总端必须先重算每个内嵌 `scientific_content_binding_record`, 要求其 schema、run id、自摘要与结果 metadata 一致, 再按 Prompt 运行顺序收集摘要；摘要数量必须等于当前层级完整 Prompt 数量, 随后才构造数据集级有序摘要。

每个科学单元 manifest 的 `output_paths` 必须是非空且无重复的相对路径集合, 并必须包含该单元 manifest 自身。所有单元路径集合必须两两互斥。数据集 manifest 的 `output_paths` 自身也不得重复, 且必须是全部单元路径并集的超集, 从而纳入每个单元声明的全部 JSONL、图像、结果记录和单元 manifest 叶子。省略部分单元叶子、单元内重复、跨单元重复或只声明数据集汇总文件均必须拒绝。

`package_image_only_dataset_runtime` 在打包前必须从数据集顶层 manifest 的 `scientific_unit_identity_records` 读取每个科学单元的完整脱敏配置, 再读取 `runtime_results.jsonl` 中的配置摘要和紧凑随机化引用、该单元 manifest 及其声明的叶子文件。打包器以 `run_id` 连接顶层配置与样本结果, 独立重算配置摘要、运行身份和来源绑定, 随后才调用与完成结果加载器相同的叶子重建路径。缺少完整 `scientific_content_binding_record`、缺失顶层单元配置或 manifest、配置摘要或随机化引用漂移、叶子内容损坏、数据集 manifest 漏报单元叶子时都必须拒绝打包。

attention geometry 开启时, 写出单元产物后和数据集打包前都必须调用 `_carrier_only_counterfactual_artifact_binding_ready`。同一个 validator 必须分别接受进程内完整 `SemanticWatermarkRuntimeConfig` 与由数据集顶层 manifest 按 `run_id` 取得的完整脱敏单元配置。它从 carrier-only 更新 JSONL、图像、保持记录、Q/K 记录与单元 manifest 重建反事实身份；任一 carrier-only 原子残留 attention 分数、更新、关系、pair 身份、Q/K 内容或 attention Null Space 字段时必须拒绝。仅比较 carrier-only 顶层摘要或只在写出时检查一次均不满足该门禁。

该门禁必须同时进入科学算子门禁、summary、manifest、打包前逐单元重建和完成包核验, 不能用部分样本摘要或只含数据集汇总文件的 manifest 替代完整覆盖。

这些摘要只证明同一结果包内部的内容一致性、角色顺序和持久化前后没有不可见漂移。SHA-256 本身不能重建 Tensor, 也不能证明外部数据来源真实、公开模型权重真实、GPU/CUDA 算子执行正确、方法效果成立或论文结论成立。真实外部来源、CUDA 运行和科学有效性仍分别由来源锁、GPU 原子证据、统计协议及论文结果包负责。

## GPU 观察边界

CPU 性质测试可以证明公式、反例拒绝、字段一致性和分层边界。真实 SD3.5 权重、CUDA 混合精度、Q/K 梯度、最终成图和图像盲检只能由 Colab GPU 运行验证。在全部不变量达到 `cpu_verified` 前不得开始正式论文结果生产；单样本真实 GPU 科学预检也不得支持论文 claim。
