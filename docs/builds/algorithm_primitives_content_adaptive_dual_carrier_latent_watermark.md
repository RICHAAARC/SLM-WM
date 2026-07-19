# 算法原语：语义显著性自适应内容-几何双链潜空间水印

## 1. 文档权威性

本文档是正式方法的唯一算法原语权威来源。实现、配置、测试、GPU 资格化、实验记录、论文表述和发布包均不得定义与本文档不同的第二套方法。

项目标识 `SLM-WM` 只表示仓库与实验协议身份，不定义方法数学语义，也不承担全局流形、716维特征水平集、局部切空间或 Jacobian Null Space 的数学主张。

正式方法中文名称固定为：

> 语义显著性自适应内容-几何双链潜空间水印。

该方法不是单一内容载体方法，而是由两个职责严格分离的核心链组成：

1. **内容链**：依据当前生成内容的 Prompt 条件语义显著性、纹理复杂度、潜空间响应特征和局部扰动敏感性四类内容事实，自适应路由 LF 与 HF-tail 载体，并产生唯一的水印主检测证据。
2. **几何链**：在生成阶段嵌入带密钥的 Q/K 注意力关系同步模板，在检测阶段恢复几何参考系，并将恢复图像交回内容链执行同阈值重判。

---

## 2. 方法身份与范围

目标方法必须同时包含以下原语：

1. 从当前注入 latent 解码得到真实 RGB 观测。
2. 从真实 RGB 与当前 Prompt 构造空间级 Prompt 条件语义显著图。
3. 从真实 RGB 构造纹理复杂度图。
4. 从相邻真实 scheduler latent 构造潜空间响应特征图。
5. 使用一个公开、确定、与水印密钥无关的扰动方向构造局部扰动敏感性图。
6. 依据四类内容事实构造 LF 与 HF-tail 空间路由及局部强度预算。
7. 使用独立密钥域构造二维 LF 载体和 HF-tail 载体。
8. 从冻结 SD3.5 注意力层的真实 Q/K 构造带密钥几何同步模板和几何链更新。
9. 在一个登记注入时刻，将两个内容分支和一个几何分支合成为一次实际 latent 写回。
10. 从待检图像独立提取原始内容分数和真实 Q/K 几何关系。
11. 对近阈值失败样本执行几何恢复、图像回正、重新编码和同阈值内容重判。
12. 使用完整决策器在独立 calibration split 上校准 fixed-FPR。

以下内容不属于目标方法：

- 716维特征水平集及其一阶局部切空间解释。
- Jacobian Null Space、完整 JVP/VJP 和 PSD-CG。
- 多注入时刻重复写回。
- 几何分数独立投票为水印阳性。
- 攻击特定内容阈值或恢复后降阈值。

---

## 3. 冻结模型与基础约定

### 3.1 冻结模型身份

正式实现必须绑定以下模型和精确 revision：

- 生成模型：`stabilityai/stable-diffusion-3.5-medium@b940f670f0eda2d07fbb75229e779da1ad11eb80`。
- 内容显著性模型：`openai/clip-vit-base-patch32@3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`。
- CLIP 图像预处理固定为 RGB、按冻结 `CLIPImageProcessor` 配置使用 bicubic 将短边缩放到224、中心裁剪为224×224、乘 `1/255`，再使用均值 `(0.48145466,0.4578275,0.40821073)` 和标准差 `(0.26862954,0.26130258,0.27577711)` 归一化；实现必须消费上述精确 revision 的 `preprocessor_config` 与科学依赖锁，不得换用另一套 resize 舍入规则。
- CLIP 文本预处理固定使用同 revision tokenizer、`max_length=77`、`padding=max_length` 和 `truncation=true`。
- VAE、Transformer 和 Scheduler 类必须来自受治理模型配置；VAE 标量还必须逐值验证为 `scaling_factor=1.5305`、`shift_factor=0.0609`，不得只接受任意可解析值。
- 几何链使用的冻结注意力层固定为 `transformer_blocks.0.attn` 与 `transformer_blocks.23.attn`；后续改变层集合属于方法原语变更。

依赖锁、预处理、模型 revision 或注意力层身份缺失时必须失败关闭，不得回退到其他模型、随机特征或预制分数。

### 3.2 张量和坐标约定

- latent 使用 NCHW 排列。
- 正式生成、共享方法参数物化、仅图像检测和 calibration 单样本测量统一固定为 `B=1`。外层 runner 可以顺序处理多个样本，但不得把多个样本拼成一个方法批次；PRG、图像摘要和检测身份均不得依赖批内位置或外层批大小。
- 所有内容空间图先在自身原始分辨率计算，再使用 `torch.nn.functional.interpolate` 的 `mode=bilinear`、`align_corners=false`、`antialias=false` 一次映射到 latent 的 H×W；不得使用 nearest、bicubic、`align_corners=true` 或后端默认值替代。该规则只用于内容图，不覆盖第11节几何图像恢复的独立 `align_corners=true` 规则。
- LF、HF-tail 和几何链分别使用独立 PRG domain。
- 所有候选更新先在 `float32` 中构造和组合，只允许在最终写回前转换一次实际 latent dtype。
- 正式运行只允许一个登记注入索引。回调固定使用 Diffusers `callback_on_step_end`，索引从0开始；`z_i` 精确表示 scheduler step `i` 完成后、下一 step 开始前交给该回调的 latent。目标注入发生在20步采样的 callback 索引10，并以一次合成后的 latent 替换该回调返回值中的 latent；索引9只保存 `z_9`，不得写回。

### 3.3 冻结标量和版本边界

以下数值是目标方法的一部分，不是运行时推荐值：

| 类别 | 冻结值 |
|---|---|
| 注入索引 | 20步采样的 callback 索引10 |
| LF 相对强度 | `0.0025` |
| HF-tail 相对强度 | `0.0015` |
| geometry 相对强度 | `0.0010` |
| HF tail 比例 | `0.20` |
| LF/HF 检测权重 | `0.70 / 0.30` |
| 局部敏感性探针相对步长 | `1e-3` |
| Q/K 最大抽样 token 数 | `64` |
| 几何嵌入 latent 来源 | callback 索引10的 content-only 基底 |
| 几何嵌入 Q/K operator schedule 索引 | `7` |
| 仅图像检测 Q/K 公开 schedule 索引 | `7` |
| Q/K 稳定 token 比例 | `0.50`，且至少选择4个 token |
| 非稳定 token 权重 | `0.25` |
| Q/K 四分量权重 | `(0.25, 0.25, 0.25, 0.25)` |
| Q/K 四分量极性 | `(+1, -1, +1, +1)` |
| soft-rank 温度与分块 | `0.25 / 32` |
| 数值 epsilon | `1e-12` |
| 几何嵌入回溯 | 因子 `0.5`，最多8次缩小，即最多检查9个比例 |
| 合成写回回溯 | 因子 `0.5`，最多24次缩小，即最多检查25个比例 |
| GPU 机制一致性诊断 | 同源 CLIP cosine 不低于 `0.995`；只影响算子资格化，不参与正式阳性和逐样本筛除 |

`g_ref`、`r_ref` 和 `q_ref` 是方法参数标量。它们必须从与 calibration/test 均隔离的共享方法参数划分计算，并写入唯一登记文件 `configs/content_routing_reference_registry.json`。每个参数划分成员均按 `B=1` 独立运行，三个观测总体精确冻结为：

- `g_ref`：拼接全部成员在 RGB 原始分辨率、映射到 latent 之前产生的所有有限且严格为正的 `G` 标量，不先计算逐图像均值、最大值或分位数。
- `r_ref`：拼接全部成员在 latent 分辨率产生的所有有限且严格为正的 `d_R` 标量，不执行逐图像汇总。
- `q_ref`：拼接全部成员在 RGB 原始分辨率、映射到 latent 之前产生的所有有限且严格为正的 `d_Q` 标量，不执行逐图像汇总。

三者分别在自己的拼接总体上采用 nearest-rank 95分位数：对 `n` 个有限正标量升序排列，取索引 `(19*n+19)//20-1`。任一总体为空时参数物化失败关闭；不得以0、epsilon、逐图像统计或预制默认值回退。

该登记文件必须绑定 `registry_schema`、方法参数划分 ID、Prompt 清单摘要、seed 清单摘要、样本数量、模型与预处理身份摘要、原始观测成员摘要、分位数算法、分位数值以及三个精确标量。登记文件未物化或任一摘要不一致时，正式 GPU 运行必须失败关闭。三个标量不得在单张图像、论文 profile、calibration split 或 test split 上重算。

真实 reference 候选只允许由与 calibration/test 隔离的 `probe_paper` dev 参数划分产生：同一正式生成身份必须独立重放两次，逐成员复算原始 `G/d_R/d_Q`，并要求候选规范字节、总体身份和三个标量逐字一致。隔离 scientific child 只能把候选及复验报告写入 `outputs/`；它不得改写或读取固定登记路径。只有外层独立复验通过后，才允许把候选原始字节逐字晋升到固定登记文件，禁止 JSON 重序列化、默认值或单图现场重算。

正式 calibration 与 qualification 必须通过固定登记文件的 semantic digest 和 exact file SHA-256 双身份取得三个标量；单样本 smoke 的显式未资格化标量仍是独立入口，不能进入正式 qualification。`probe_paper` 的 calibration-only 执行在33个 calibration Prompt 完成后立即持久化并重建 `FrozenEvidenceProtocol`，随后返回 `calibration_complete` 中间状态；该路径不得执行 test Prompt、攻击、FID/KID，也不得把 repeat 或 paper 标记为完成。

同源 CLIP cosine 只用于单 Prompt GPU 算子资格化，不能承担论文的独立视觉内容保持主张，也不能作为正式运行中删除低质量样本的依据。手工结构描述符可以保留为 `experiments/` 层诊断，但不属于算法原语、核心门禁或水印阳性公式。

---

## 4. 原语一：真实内容观测

### 4.1 当前 latent 解码

在登记注入时刻取得实际进入 scheduler 的 latent `z_t*`。冻结 SD3.5 VAE 标量为 `scaling_factor=1.5305`、`shift_factor=0.0609`；先在 `float32` 中执行逆 latent 变换，再只为 VAE 前向转换到 VAE 参数 dtype：

\[
z_{\mathrm{VAE}}
=
\operatorname{Cast}_{\mathrm{VAE\ dtype}}
\left(z_{t^*}/1.5305+0.0609\right),
\]

\[
d_{t^*}=\operatorname{VAE.decode}(z_{\mathrm{VAE}}),
\qquad
x_{t^*}=\operatorname{clip}
\left(\operatorname{float32}(d_{t^*})/2+0.5,0,1\right).
\]

`VAE.decode` 精确取 `return_dict=false` 返回 tuple 的第0个 Tensor；不得调用随机 posterior sample。内容观测必须来自该真实 `[0,1]` RGB Tensor。Prompt digest、样本 ID、密钥、随机种子或预制统计量不能替代真实图像内容。

### 4.2 Prompt 条件语义显著图

本文所称“语义显著性”固定表示：当前图像各空间区域与当前生成 Prompt 的局部语义相关程度。它不是人类注视真值，也不宣称是显著目标分割真值。

对冻结 CLIP 视觉编码器最后一层 patch token 应用该模型的冻结 `post_layernorm`，得到 `h_i`；随后应用冻结视觉投影 `W_v`。文本特征固定使用同一 CLIP 文本编码器在 EOS token 位置产生的标准 pooled output，再应用冻结 `text_projection` 得到 `e_p`；不得改用 token 均值、首 token 或自定义池化：

\[
v_i=\operatorname{L2Norm}(W_v h_i),
\qquad
q=\operatorname{L2Norm}(e_p).
\]

局部 Prompt 条件语义相关分数为：

\[
r_i=\langle v_i,q\rangle.
\]

使用冻结映射将其变换到 `[0,1]`：

\[
S_i=\operatorname{clip}\left(\frac{r_i+1}{2},0,1\right).
\]

移除 CLIP 类别 token 后，将 patch 网格按第3.2节内容图插值规则映射到 latent H×W，得到：

\[
S\in[0,1]^{B\times1\times H\times W}.
\]

正式实现必须满足：

1. `S` 同时依赖真实图像和当前 Prompt。
2. 改变 Prompt 且保持图像不变时，`S` 必须能够发生变化。
3. 不得使用全图单一 CLIP cosine 广播成空间图。
4. 不得使用 Prompt digest、随机图、边缘图或手工类别标签冒充语义显著图。
5. 嵌入使用的 CLIP 可作为机制模型；论文独立视觉内容质量评估器仍必须位于 `experiments/`，不得与本原语混为独立质量证据。

### 4.3 纹理复杂度图

将 `[0,1]` 真实 RGB 按 `Y=0.299R+0.587G+0.114B` 转换为亮度图。先使用 replicate padding 扩展1像素，再以 stride 1 应用固定 Sobel 核：

\[
K_x=
\begin{bmatrix}
-1&0&1\\
-2&0&2\\
-1&0&1
\end{bmatrix},
\qquad
K_y=K_x^\top.
\]

得到梯度幅值：

\[
G=\sqrt{G_x^2+G_y^2}.
\]

使用共享方法参数划分冻结的正标量 `g_ref` 将其截断到 `[0,1]`：

\[
T=\operatorname{clip}(G/g_{ref},0,1).
\]

再将 `T` 按第3.2节内容图插值规则映射到 latent H×W。`g_ref` 不得在 calibration split、test split 或逐攻击结果上回调，也不得按单张图像重新取最大值。

### 4.4 潜空间响应特征图

在 callback 索引9保留实际 scheduler latent `z_9`，在注入索引10取得实际 scheduler latent `z_10`。对每个空间位置 `u` 计算跨通道 RMS 相对变化：

\[
d_R(u)=
\frac{
\operatorname{RMS}_c\left(z_{10}(u)-z_9(u)\right)
}{
\operatorname{RMS}_c\left(z_{10}(u)\right)
+\operatorname{RMS}_c\left(z_9(u)\right)+10^{-12}
}.
\]

使用冻结 `r_ref` 得到潜空间响应不稳定度：

\[
R(u)=\operatorname{clip}\left(d_R(u)/r_{ref},0,1\right).
\]

`R` 越大表示该位置在相邻真实采样状态间变化越大，局部嵌入预算应越低。该量不增加 Transformer 前向，也不表示完整生成轨迹的全局稳定性。

### 4.5 局部扰动敏感性图

使用公开 PRG domain `local_sensitivity_public_probe` 构造与 latent 同形状的确定性高斯方向 `P`，移除样本均值后将其单位 RMS 化。PRG 版本固定为 `sha256_counter_normal_icdf_table20_float32`，公开 `key_material` 固定为 `semantic_saliency_dual_chain_public_probe_v1`，domain fields 固定包含 `purpose=local_sensitivity_public_probe`、精确生成模型 revision 和 `probe_version=v1`；规范 PRG 还必须自动绑定 latent 形状。不得加入水印密钥、Prompt、样本 ID、生成 seed 或攻击标签。

探针步长固定为：

\[
\epsilon_p(z_{10})
=10^{-3}\max\left(\operatorname{RMS}(z_{10}),10^{-12}\right).
\]

复用已经解码的 `x_10=Decode(z_10)`，只增加一次 VAE 解码：

\[
x_p=\operatorname{Decode}_{\mathrm{VAE}}(z_{10}+\epsilon_pP).
\]

对每个 RGB 空间位置计算单方向有限差分敏感度：

\[
d_Q(u)=
\frac{
\operatorname{RMS}_{rgb}\left(x_p(u)-x_{10}(u)\right)
}{\epsilon_p},
\qquad
Q(u)=\operatorname{clip}\left(d_Q(u)/q_{ref},0,1\right).
\]

`Q` 先在 RGB 分辨率计算，再按第3.2节内容图插值规则映射到 latent H×W。它只表示一个公开方向上的局部有限差分敏感性，不等于完整 Jacobian 范数、最坏方向敏感性或通用感知质量。

为避免与注意力 Query 混淆，持久化字段和代码接口必须使用 `local_sensitivity_map`；本节公式中的单变量 `Q(u)` 只表示局部敏感性图，后文成对出现的 `Q/K` 专指 attention Query/Key。

---

## 5. 原语二：内容自适应路由与局部强度

定义四类内容事实共同形成的可写容量：

\[
A=
\sqrt[3]{
(1-S)\odot(1-R)\odot(1-Q)
}.
\]

两个内容分支的空间路由为：

\[
M_{\mathrm{LF}}=A\odot(1-T),
\]

\[
M_{\mathrm{HF-tail}}=A\odot T.
\]

其中：

- `S` 保护 Prompt 语义显著区域，`R` 降低高 latent 响应区域的预算，`Q` 降低公开探针下高局部敏感区域的预算。
- LF 分支优先使用低语义显著、低响应、低敏感且较平滑的位置。
- HF-tail 分支优先使用低语义显著、低响应、低敏感且纹理丰富的位置。
- 数学掩码 $M_{\mathrm{LF}}$ 与 $M_{\mathrm{HF-tail}}$ 分别映射到代码字段 `lf_mask` 与 `hf_tail_mask`，同时决定空间位置和局部实际幅值。
- 掩码作用后不得重新单位化整个分支，否则会抵消内容自适应强度。

几何平均避免四个独立阈值和额外可调权重；任一保护事实接近1时都会收缩容量，同时不会把多个小于1的因子直接相乘造成过快衰减。该设计属于项目特定原语。通用工程写法只包括张量广播、插值和数值检查；由 `S`、`T`、`R`、`Q` 形成两个互补载体路由是本项目的方法定义。

---

## 6. 原语三：二维 LF 密钥载体

使用 LF 独立 PRG domain 生成标准高斯张量：

\[
U_{\mathrm{LF}}=
\operatorname{PRGNormal}(K,\texttt{lf_content},B,C,H,W).
\]

只在 latent 的 H、W 轴应用冻结二维低通算子：

\[
L(U)=\operatorname{AvgPool2D}_{5\times5,\,stride=1,\,padding=2}(U),
\]

其中边界精确使用常数0 padding，`ceil_mode=false`、`count_include_pad=true`、`divisor_override=null`。因此边界窗口仍固定除以25；不得改为 reflect、replicate、排除 padding 的平均或其他后端默认。

去除每个样本的张量均值并执行 L2 归一化，得到标准 LF 载体 `C_LF`。内容路由后的实际方向为：

\[
D_{\mathrm{LF}}
=
\operatorname{BroadcastChannel}(M_{\mathrm{LF}})
\odot C_{\mathrm{LF}}.
\]

“低频”只表示 latent 二维空间轴上的明确低通结果，不表示短向量周期平铺或名称标签。

LF 是内容链的**主证据分支**。它承担 clean、轻度失真和大多数登记攻击下的主要检测稳定性，正式检测权重固定为 `0.70`。如果 HF-tail 被消融，LF 仍应形成可校准的连续内容分数；因此不得把 LF 降级为只服务图像质量的辅助载体。

---

## 7. 原语四：HF-tail 密钥载体

使用与 LF 隔离的 PRG domain 生成：

\[
U_{\mathrm{HF-tail}}
=
\operatorname{PRGNormal}(K,\texttt{hf_tail_robust},B,C,H,W).
\]

先使用与 LF 配对的高通残差：

\[
H(U)=U-L(U).
\]

再在每个样本内部展平 C×H×W，按“绝对值降序、展平索引升序”稳定排序。保留数量精确为 `k=max(1,ceil(0.20*C*H*W))`，其余坐标精确置零，最后执行 L2 归一化；不得跨 batch 展平，也不得使用 floor、round 或后端相关的 `topk` 并列顺序：

\[
C_{\mathrm{HF-tail}}
=
\operatorname{L2Norm}
\left(
\operatorname{TopAbs}_{0.20}(H(U_{\mathrm{HF-tail}}))
\right).
\]

内容路由后的实际方向为：

\[
D_{\mathrm{HF-tail}}
=
\operatorname{BroadcastChannel}(M_{\mathrm{HF-tail}})
\odot C_{\mathrm{HF-tail}}.
\]

LF 与 HF-tail 的命名只描述空间路由掩码作用前的密钥载体 $C_{\mathrm{LF}}$ 与 $C_{\mathrm{HF-tail}}$ 的频率来源。掩码作用后的实际写入 $D_{\mathrm{LF}}$ 与 $D_{\mathrm{HF-tail}}$ 分别称为 LF-origin 与 HF-tail-origin 空间调制更新；本方法不主张二者仍严格带限、频谱互不重叠或彼此正交。

该原语同时包含明确的二维高通和幅值尾部截断，因而允许称为 HF-tail。仅对原始高斯幅值做 TopAbs 不构成 HF-tail；任何只对原始高斯幅值执行 TopAbs、但未先执行高通的载体都不符合该原语。

HF-tail 是内容链的**困难攻击补充分支**。其职责是在强压缩、噪声、重采样、几何失配和登记生成式攻击等 LF 分数明显衰减的条件下提供额外鲁棒性余量，正式检测权重固定为 `0.30`。HF-tail 不是第二个独立判定器，不允许单独使用不同阈值产生阳性；完整方法只对 `0.70 s_LF + 0.30 s_HF-tail` 校准一个内容阈值。

“困难攻击补充”是待实验验证的机制职责，不预先保证 HF-tail 在每项攻击上优于 LF。正式消融必须分别报告 LF-only、HF-tail-only 和完整内容链，以验证该职责是否成立。LF-only 的内容检测权重固定为 `1.0/0.0`，HF-tail-only 固定为 `0.0/1.0`；两个消融均保持保留分支的名义嵌入强度，不转移被关闭分支预算，并使用同一 fixed-FPR 算法独立校准完整决策参数。

---

## 8. 原语五：带密钥 Q/K 几何同步模板

### 8.1 真实 Q/K 与四分量关系公式

几何链使用冻结层真实 `to_q`、`to_k` 投影得到的图像 token Q/K，不使用 hidden state cosine 代理。注意力模块的 `hidden_states` 必须只包含图像 token，并具有 `[1,N,D]` 形状；若 `N` 不是某个不小于2的整数 `s_source` 的平方则失败关闭，不得混入文本 token 或裁剪成近似方形。最大 token 数固定为64，令 `s_sample=min(s_source,floor(sqrt(64)))`。对 `i=0,...,s_sample-1`，单轴抽样索引精确冻结为 `j_i=floor(i*(s_source-1)/(s_sample-1)+0.5)`；该定义对非负坐标执行 half-up 舍入，不得替换为语言或后端相关的 `round()`。最终 token 索引按 row-major 的“行索引外层、列索引内层”枚举。两个冻结层必须得到完全相同的 token 索引。对第 `h` 个 attention head：

\[
L^{(h)}=Q^{(h)}K^{(h)\top}/\sqrt d,
\qquad
P^{(h)}=\operatorname{softmax}_{row}(L^{(h)}).
\]

先逐 head 计算，再跨 head 求均值；严禁先平均 logit 后做 softmax：

\[
L=\operatorname{mean}_h\left(L^{(h)}-\operatorname{mean}_{row}(L^{(h)})\right),
\qquad
P=\operatorname{mean}_h P^{(h)}.
\]

令 token 数为 `N`，公开二维归一化坐标为 `c_i`。四个关系分量固定为：

\[
F^{(1)}_{ij}=L_{ij},
\]

\[
F^{(2)}_{ij}=\frac{1+\sum_{k\ne j}
\sigma\left((L_{ik}-L_{ij})/0.25\right)}{N},
\]

\[
F^{(3)}_{ij}=\frac{P_{ij}}{\sum_k P_{ik}+10^{-12}},
\]

\[
d_{ij}=\frac{\lVert c_i-c_j\rVert_2}{2\sqrt2},
\qquad
F^{(4)}_{ij}=
\left(F^{(3)}_{ij}-\operatorname{mean}_jF^{(3)}_{ij}\right)
\left(d_{ij}-\operatorname{mean}_jd_{ij}\right).
\]

soft-rank 的计算分块固定为32；分块只改变峰值显存，不改变上式。坐标固定为 `normalized_xy_token_centers_corner_endpoints`，网格变换固定 `align_corners=true`。

### 8.2 稳定 token、密钥投影与几何分数

稳定 token 只由第8.1节的 attention probability `P` 决定，不使用四分量 `F^(m)`、密钥模板或水印分数。两冻结层必须具有相同的 head 数 `H`、相同的抽样 token 数 `N` 和相同 token 索引，否则失败关闭。对层 `ell`、head `h` 和 query token `i`，先在 key 轴中心化并执行 L2 归一化：

\[
r^{(\ell,h)}_i
=P^{(\ell,h)}_{i,:}
-\operatorname{mean}_jP^{(\ell,h)}_{ij},
\qquad
\bar r^{(\ell,h)}_i
=\frac{r^{(\ell,h)}_i}{\max(\lVert r^{(\ell,h)}_i\rVert_2,10^{-12})}.
\]

任一输入或归一化结果不有限时失败关闭；零中心化行按上述 epsilon 规则得到零行，不另行删除 token。层顺序固定为 `transformer_blocks.0.attn`、`transformer_blocks.23.attn`，两层只有一个规范层对；跨层一致性和 incoming centrality 固定为：

\[
c_i=\operatorname{clip}\left(
\frac{1+H^{-1}\sum_h
\langle\bar r^{(0,h)}_i,\bar r^{(23,h)}_i\rangle}{2},0,1
\right),
\]

\[
a_i=\frac{1}{2H}\sum_{\ell\in\{0,23\}}\sum_h
\left(\frac1N\sum_qP^{(\ell,h)}_{qi}\right),
\qquad
\bar a_i=\frac{a_i}{\max(\sum_ja_j,10^{-12})},
\qquad
u_i=c_i\bar a_i.
\]

分母和选择分数必须有限。按 `u_i` 降序、`token_indices[i]` 表示的原始图像 token 索引升序选择 `max(4,ceil(0.50N))` 个稳定 token；选择完成后，为了形成唯一持久化身份，将选中 token position 重新按升序保存，并据此保存对应原始 token 索引。稳定 token 权重为1，其余 token 权重为0.25；非对角 pair 权重为两个端点权重的外积，对角权重精确为0。

几何链的 PRG domain family 为 `attention_geometry`。其中 pair sign 子域精确使用 domain fields `operator=attention_relation_signs`、`layer_name=<冻结层全名>`、`token_count=N`，不得增加、删除或改名字段。对每层使用登记密钥与规范 PRG 生成 `N×N` 的 `[0,1)` uniform 矩阵 `U`；只读取严格上三角 `i<j`，固定映射为 `U_ij>=0.5 -> +1`、否则 `-1`，再镜像到 `B^K_{ji}=B^K_{ij}`，对角 `B^K_{ii}` 精确置0。模型 revision 和 token 索引由外层模板身份另行绑定，不重复进入该 PRG domain fields。四个关系分量共享同一对称 pair sign，并乘固定极性：

\[
p=(+1,-1,+1,+1),
\qquad
G^{(m)}_{K,ij}=p_mB^K_{ij}.
\]

对每层、每个分量、每一行，以非对角有效 pair 权重 `omega_ij` 计算唯一的加权中心化 Pearson。令 `Z_i=sum_j omega_ij`，只在 `Z_i>0` 时计算：

\[
\mu_{F,i}^{(m)}=Z_i^{-1}\sum_j\omega_{ij}F_{ij}^{(m)},
\qquad
\mu_{G,i}^{(m)}=Z_i^{-1}\sum_j\omega_{ij}G_{K,ij}^{(m)},
\]

\[
c_i^{(m)}=
\frac{\sum_j\omega_{ij}(F_{ij}^{(m)}-\mu_{F,i}^{(m)})(G_{K,ij}^{(m)}-\mu_{G,i}^{(m)})}
{\sqrt{\sum_j\omega_{ij}(F_{ij}^{(m)}-\mu_{F,i}^{(m)})^2}\sqrt{\sum_j\omega_{ij}(G_{K,ij}^{(m)}-\mu_{G,i}^{(m)})^2}}.
\]

任一加权方差不大于 `1e-24` 时跳过该行，不添加 epsilon。每层分量分数是其有效行的算术平均；任一必要层/分量没有有效行时失败关闭。随后按冻结层顺序对两层分量分数做算术平均，最后固定组合：

\[
s_{\mathrm{geo}}
=0.25s_1+0.25s_2+0.25s_3+0.25s_4.
\]

模型 revision、层名和层顺序、Q/K 头数与头宽、scale 来源、Q/K normalization 类、token 索引、稳定 token、pair 权重、PRG 版本和四分量身份都必须进入模板摘要。任一身份变化均是方法变化。

生成端几何嵌入使用 callback 索引10的 content-only latent 基底、`scheduler.timesteps[7]` 和真实当前 Prompt 条件求值。检测端从待检图像构造公开仅图像条件，并同样固定在 `scheduler.timesteps[7]` 提取同层 Q/K。二者的模型 revision、scheduler、层、token、四分量和密钥投影身份必须一致，但不得要求检测器访问生成 latent 或生成 Prompt。

公开仅图像 Q/K 条件精确冻结为：

1. 使用受治理 SD3 图像处理器预处理待检 RGB，并取 VAE posterior mode；latent 变换固定为 `(mode-shift_factor)*scaling_factor`。
2. 每次测量先将同一 scheduler 重置为20步正式日程，读取 `scheduler.timesteps[7]`。
3. 使用 PRG 版本 `sha256_counter_normal_icdf_table20_float32` 构造公开高斯噪声；`key_material` 与 `operator` 均固定为 `public_image_only_qk_detection_noise`，domain fields 还必须绑定模型 ID、精确 revision、宽、高、20步、schedule 索引7和 latent 形状。
4. 只允许调用 scheduler 的 `scale_noise(encoded_latent,timestep,public_noise)` 构造检测 latent；缺少该算子时失败关闭。
5. 文本条件固定为 `sd3_empty_text_triplet_without_cfg`，condition text 精确为空字符串，不执行 classifier-free guidance，也不读取原始 Prompt。
6. 在 `no_grad` 下执行一次冻结 Transformer 前向并捕获指定层直接 Q/K；任一层无真实 Tensor 时失败关闭。

公开噪声只定义可重现的 Q/K 观测条件，不携带水印密钥；登记密钥只进入第8.2节的关系模板和分数。

### 8.3 几何链嵌入和预算

先按 `method_role` 解析两个内容分支，再在 `float32` 中使用尚未经过共同缩放的角色登记更新构造唯一内容基底：

\[
z_{\mathrm{content}}
=
\operatorname{float32}(z_{10})
+\Delta z_{\mathrm{LF}}
+\Delta z_{\mathrm{HF-tail}}.
\]

其中 `Delta z_LF` 与 `Delta z_HF-tail` 分别使用第9节的 `0.0025||z_10||_2` 与 `0.0015||z_10||_2` 名义更新；若角色关闭某一内容分支，该项精确为零且预算不得转移。`z_content` 始终写成上述两个角色解析项之和，位于几何梯度和几何独立回溯之前、共同 `gamma` 回溯之前。不得遗漏本角色仍处于活动状态的内容分支，也不得使用共同缩放后的基底或实际 dtype 已量化基底。仅对启用几何同步嵌入的角色，随后在该同一 `float32` latent 点对冻结的稳定 token 和 pair 权重计算一次真实 autograd 梯度：

\[
g_{\mathrm{geo}}
=\nabla_z s_{\mathrm{geo}}(z_{\mathrm{content}},G_K).
\]

应用内容容量但不在掩码后重新归一化：

\[
D_{\mathrm{geo}}
=\operatorname{BroadcastChannel}(A)
\odot\frac{g_{\mathrm{geo}}}{\lVert g_{\mathrm{geo}}\rVert_2+10^{-12}}.
\]

几何名义预算固定为：

\[
b_{\mathrm{geo}}=0.0010\lVert z_{10}\rVert_2.
\]

按 `k=0,1,...,8` 依次检查 `b_geo*2^{-k}D_geo`，接受第一个在实际 latent dtype 下有限、非零、未超预算，并满足下式的候选：

\[
s_{\mathrm{geo}}
\left(\operatorname{Cast}_{dtype}
(z_{\mathrm{content}}+b_{\mathrm{geo}}2^{-k}D_{\mathrm{geo}})
\right)
>
s_{\mathrm{geo}}
\left(\operatorname{Cast}_{dtype}(z_{\mathrm{content}})\right).
\]

搜索期间稳定 token、pair 权重和梯度保持冻结，不得重新选择或重新求梯度。9个候选均失败时，该样本失败关闭；不得调用 JVP/VJP、不得投影到 Jacobian Null Space，也不得以随机向量、hidden state cosine 或预制分数替代。

---

## 9. 原语六：单时刻三分支合成与一次写回

以 `n_z=||z_10||_2` 为样本级唯一强度尺度，内容分支固定为：

\[
\Delta z_{\mathrm{LF}}=0.0025n_zD_{\mathrm{LF}},
\qquad
\Delta z_{\mathrm{HF-tail}}=0.0015n_zD_{\mathrm{HF-tail}}.
\]

启用几何同步嵌入的角色使用第8.3节接受的几何分支：

\[
\Delta z_{\mathrm{geo}}
=0.0010\,2^{-k}n_zD_{\mathrm{geo}},
\qquad k\in\{0,\ldots,8\}.
\]

`content_chain_only` 与 `geometry_recovery_without_embedded_sync` 不调用几何梯度或独立回溯，精确令 `Delta z_geo=0`；后者只保留检测阶段的几何恢复与重判。其他4个角色保持几何同步写入活动。

完整方法的三个标准方向在内容掩码后均不重新单位化，分别满足 `||D_b||_2<=1`；消融角色的禁用方向按零更新处理。因此所有角色都满足同一个硬 L2 上界：

\[
\lVert\Delta z\rVert_2
\le(0.0025+0.0015+0.0010)n_z
=0.0050n_z.
\]

内容链与几何链在同一个登记时刻合成：

\[
\Delta z
=
\Delta z_{\mathrm{LF}}
+\Delta z_{\mathrm{HF-tail}}
+\Delta z_{\mathrm{geo}}.
\]

在 `float32` 中按冻结顺序 `LF -> HF-tail -> geometry` 组合。共同缩放候选固定为 `gamma_j=2^{-j}, j=0,...,24`，按 `j` 从小到大选择第一个同时满足下列条件的候选：

1. 组合更新全部有限且不超过 `0.0050n_z`。
2. 每个活动分支在实际 latent dtype 中分别产生非零差分。
3. 按 `method_role` 登记的全部活动分支组合在实际 latent dtype 中产生非零差分；禁用分支的 effective L2 精确记录为0，不作为失败。

然后只执行一次实际 dtype 转换和一次 latent 写回：

\[
z'_{t^*}=\operatorname{Cast}_{dtype}
\left(z_{t^*}+\gamma\Delta z\right).
\]

写回后必须使用同一冻结 Q/K 身份复验 geometry 关系仍高于同一 `gamma` 下的 content-only 基底。该写回门禁失败时样本失败关闭，不允许继续缩小强度直至迎合结果，也不得对不同论文 profile 使用不同强度。

最终图像生成后必须记录同源 CLIP cosine。单 Prompt GPU 资格化要求该值不低于 `0.995`；正式批量运行不得据此删除样本或重试生成，未达到诊断阈值的样本仍保留在质量统计和检测分母中。

内容链和几何链是方法上的双链，不等于只有两个张量分支。实际写回包含 LF、HF-tail 和 geometry 三个分支，但只有内容链产生主检测分数。

禁止行为包括：

- 在多个 scheduler 索引重复注入。
- 分支分别量化后再相加。
- 用三次实际写回替代一次组合写回。
- 对 LF/HF-tail 调用 Jacobian Null Space。
- 对几何分支执行716维 JVP/VJP 或 PSD-CG。

---

## 10. 原语七：仅图像内容链检测

### 10.1 输入边界

检测器只允许访问：

- 待检图像。
- 检测密钥。
- 公开模型与精确 revision。
- 冻结载体、几何模板和检测配置。

检测器不得访问原始 Prompt、生成 latent、原始图像、生成轨迹、嵌入掩码、样本级风险图或攻击标签。

### 10.2 原始内容分数

待检输入必须是实际 RGB 图像。若输入是受治理 `uint8` 持久化图像，先精确转换为 `x_y=float32(y_uint8)/255`；若输入已经是内存 RGB Tensor，则必须逐元素位于 `[0,1]`。图像宽高必须满足冻结 VAE 处理器的尺寸约束，正式检测不得为编码而执行未登记的裁剪或几何缩放。VAE 像素输入和 posterior mode 固定为：

\[
p_y=2x_y-1,
\qquad
m_y=\operatorname{VAE.encode}
\left(\operatorname{Cast}_{\mathrm{VAE\ dtype}}(p_y)\right)
.\operatorname{latent\_dist.mode}(),
\]

\[
\widehat z(y)=(\operatorname{float32}(m_y)-0.0609)\times1.5305.
\]

该公式与冻结 SD3 `VaeImageProcessor.preprocess` 的 RGB `[0,1] -> [-1,1]` 归一化和 VAE posterior mode 语义一致；不得使用 posterior sample、生成 latent、攻击标签或另一套 scaling/shift 公式。随后按同一密钥和 `z_hat(y)` 的公开 NCHW 形状重建未加内容掩码的标准 `C_LF` 与 `C_HF-tail`：

\[
s_{\mathrm{LF}}=\operatorname{corr}(\widehat z(y),C_{\mathrm{LF}}),
\]

\[
s_{\mathrm{HF-tail}}
=
\operatorname{corr}(\widehat z(y),C_{\mathrm{HF-tail}}).
\]

`corr` 精确表示将两个 Tensor 转为 `float32`、展平、分别去均值后的归一化内积：

\[
\operatorname{corr}(a,b)=
\frac{\langle a-\bar a,b-\bar b\rangle}
{\lVert a-\bar a\rVert_2\lVert b-\bar b\rVert_2}.
\]

任一输入非有限、元素数量不一致或中心化 L2 能量为零时必须失败关闭，不得添加样本相关 epsilon 改写统计量。

冻结内容分数为：

\[
s_{\mathrm{raw}}(y,K)
=
0.70s_{\mathrm{LF}}
+
0.30s_{\mathrm{HF-tail}}.
\]

内容权重作为方法参数预先冻结为 `0.70/0.30`。calibration split 只能校准完整决策器的阈值和救回窗口，不得重新拟合内容权重；任何攻击、样本、profile 或结果均不得触发权重重选。

该统计量是**掩码不可观测的盲相关统计量**，不是已知嵌入掩码下的匹配滤波器。生成端必须记录：

\[
e_{\mathrm{LF}}=\lVert D_{\mathrm{LF}}\rVert_2,
\qquad
e_{\mathrm{HF-tail}}=\lVert D_{\mathrm{HF-tail}}\rVert_2,
\]

以及两个掩码的均值、非零比例和摘要。两个分支的实际 dtype 写回都必须有限且非零；若任一分支为零，该样本是嵌入失败，不能伪装成成功样本。低容量造成的检测衰减由完整 calibration/test 分布真实反映，不能用重新生成、逐样本阈值或删除低分样本进行补偿。

calibration、clean test、攻击 test 和三档 profile 必须使用完全相同的未掩码载体重建与相关统计量。检测器不得通过 side channel 读取生成端的 `S/T/R/Q` 或嵌入掩码。

### 10.3 最终图像 Q/K 归因诊断

对最终图像 `y_full` 使用公开仅图像 Q/K 条件，分别计算登记密钥和 wrong-key 的几何分数：

\[
s_{\mathrm{geo}}^{\mathrm{registered}}
=s_{\mathrm{geo}}(y_{\mathrm{full}},K),
\qquad
s_{\mathrm{geo}}^{\mathrm{wrong}}
=s_{\mathrm{geo}}(y_{\mathrm{full}},K_{\mathrm{wrong}}).
\]

每个正式样本必须持久化两个分数以及精确定义的差值：

\[
\texttt{registered\_wrong\_key\_geometry\_score\_margin}
=s_{\mathrm{geo}}^{\mathrm{registered}}-s_{\mathrm{geo}}^{\mathrm{wrong}}.
\]

三个字段都只用于最终图像 Q/K 归因诊断，不得进入内容阈值、几何搜索资格、几何可靠性、救回资格或水印阳性公式。单 Prompt GPU 资格化还必须使用相同 seed、Prompt 和内容分支生成 matched content-only 图像 `y_content`，并同时满足：

\[
s_{\mathrm{geo}}(y_{\mathrm{full}},K)
>
s_{\mathrm{geo}}(y_{\mathrm{full}},K_{\mathrm{wrong}}),
\]

\[
s_{\mathrm{geo}}(y_{\mathrm{full}},K)
>
s_{\mathrm{geo}}(y_{\mathrm{content}},K).
\]

这两个单 Prompt 条件只验证最终图像上的几何分支归因，报告必须保持 `supports_paper_claim=false`。正式论文中的几何同步有效性仍由完整样本上的 registered/wrong-key 分布和正式消融决定，不能由单个 Prompt 外推。

---

## 11. 原语八：几何恢复与同阈值救回重判

### 11.1 几何关系测量

本节定义几何搜索算子，不定义它的调用时机。正式 application 的调用顺序由第11.4节冻结：必须先计算原始内容分数和近阈值窗口，只有近阈值失败样本才执行本节搜索。检测器在公开、冻结的仅图像 Q/K 条件下提取相同层的真实 Q/K 关系图，并使用相同密钥模板搜索有界相似变换：

\[
\widehat T=[\widehat A\mid\widehat t]\in\mathbb R^{2\times3},
\qquad
candidate\_identity=(\widehat D,\widehat\theta,\widehat\rho,\widehat t_x,\widehat t_y).
\]

`candidate_identity` 不是从最终矩阵事后任意选择的等价分解。135个相似粗候选的继承身份固定为 `D0`，追加的纯二面体候选分别继承 `D1,...,D7`；局部候选始终继承其父 best 的 `D`。对继承的正交矩阵 `D`，令 `S=A D^T`，只允许把 `S` 唯一解析为正尺度 `rho` 与 `R(theta)`：`rho=sqrt(S00^2+S10^2)`、`theta=atan2(S10,S00)`，并要求 `max_abs(S-rho*R(theta))<=1e-6`。平移直接取矩阵第3列。尺度非正、残差超限或相对继承 `D` 超出捕获域时删除候选；不得改试另一个 `D` 以保留候选。

连续捕获域固定为：

\[
\theta\in[-32^\circ,32^\circ],
\quad
\rho\in[1/\sqrt2,\sqrt2],
\quad
t_x,t_y\in[-0.28,0.28].
\]

`t_x,t_y` 使用 `[-1,1]` 归一化图像坐标。粗搜索网格固定为：

- 旋转 `{-32,-16,0,16,32}` 度。
- 尺度 `{1/sqrt(2),1,sqrt(2)}`。
- 两轴平移分别为 `{-0.28,0,0.28}`。
- 相似变换矩阵精确为 `M(theta,rho,tx,ty)=((rho*cos(theta),-rho*sin(theta),tx),(rho*sin(theta),rho*cos(theta),ty))`。
- 方形二面体线性部分按下列规范顺序冻结：`D0=((1,0),(0,1))`、`D1=((-1,0),(0,1))`、`D2=((1,0),(0,-1))`、`D3=((-1,0),(0,-1))`、`D4=((0,-1),(1,0))`、`D5=((0,1),(-1,0))`、`D6=((0,1),(1,0))`、`D7=((0,-1),(-1,0))`。
- 粗候选先按 `theta -> rho -> tx -> ty` 的外层到内层顺序枚举135个相似变换，再按 `D1,...,D7` 追加7个纯二面体候选；`D0` 对应的精确 identity 已存在于相似变换网格，只保留一次。粗搜索不构造“8个 D4 × 135个相似变换”的笛卡尔积。

对粗搜索最优候选执行3轮局部细化。第1轮的 `(rotation, log-scale, translation)` 半宽分别为 `(8度, 0.5*log(sqrt(2)), 0.14)`，第2、3轮分别除以3和9。每轮按 `delta_theta -> delta_log_rho -> delta_tx -> delta_ty` 的外层到内层顺序枚举 `{-delta,0,+delta}`；线性部分固定为 `M(delta_theta,exp(delta_log_rho),0,0) @ A_best[:,:2]`，平移固定为 `A_best[:,2]+(delta_tx,delta_ty)`。候选只相对父 best 继承的同一个 `D` 执行上述唯一分解和捕获域检查；失败时删除，不得裁剪参数或改试另一个 `D` 后继续使用。

本节所有2×3矩阵都采用 PyTorch `affine_grid/grid_sample` 的 output-to-input 约定：矩阵把 canonical 输出坐标映射到待检 observation 输入坐标。正 `theta` 的矩阵符号严格由上述 `M` 定义，不再使用“顺时针/逆时针”的图像语言解释符号。

对每个冻结层 `ell`，令 `R_obs^(ell,m)` 是待检图像真实 Q/K 产生的第8.1节四分量关系，`G_K^(ell,m)` 是同层登记密钥关系模板，`p_i` 是抽样 token 的规范二维坐标。对 output-to-input 候选 `T`，先计算 `q_i=T[p_i;1]`。规范双线性取样矩阵 `W(T)` 按以下算法唯一构造：

1. 从 `p_i` 提取升序唯一 x/y 轴，并要求其笛卡尔积精确覆盖全部 `N` 个 token；否则失败关闭。
2. 当 `q_i` 的任一坐标超出对应轴端点 `1e-6` 以上时，令 `valid_i=false` 且 `W_i,:=0`；否则令 `valid_i=true`。
3. 对有效坐标分别以 `searchsorted(...,right=true)` 找上邻点，索引截断到 `[1,axis_size-1]`，下邻点为上邻点减1；线性比例按真实轴坐标计算并截断到 `[0,1]`。
4. `W_ij` 只在二维四个邻点上取标准双线性权重；若邻点索引重合则相加。`nearest_residual_i` 是 `q_i` 到这四个邻点规范坐标的最小二维 L2 距离。

令 `T^{-1}` 表示把2×3矩阵补成齐次3×3后求逆再取前两行。使用完全相同的算法，以 `T^{-1}[p_i;1]` 构造观测前推矩阵 `V(T)`、`observation_valid_i` 和相应残差；矩阵不可逆时候选无效。双边关系重采样固定为：

\[
R_{canonical}^{(\ell,m)}(T)
=W(T)R_{obs}^{(\ell,m)}W(T)^\top,
\]

\[
G_{observation}^{(\ell,m)}(T)
=V(T)G_K^{(\ell,m)}V(T)^\top.
\]

令第8.2节稳定 token 权重为 $w$。canonical token 权重精确为 $w_{can}=(Ww)\odot valid$，canonical pair 权重为 $w_{can}w_{can}^{\top}$ 且对角置0；observation 方向继续使用原始 $ww^{\top}$ 且对角置0。记 `RelScore` 为第8.2节同一个四分量算子：每个分量、每一行只在行列均有效的非对角 pair 上计算 pair-weighted 加权中心化 Pearson correlation，关系或模板加权方差不大于 `1e-24` 的行跳过；先平均有效行，再按固定 `(0.25,0.25,0.25,0.25)` 合并分量。没有有效行时该候选方向无效。两个方向的分数唯一为：

\[
s_{canonical}^{(\ell)}(T)
=\operatorname{RelScore}(R_{canonical}^{(\ell)}(T),G_K^{(\ell)},valid,w_{can}w_{can}^\top),
\]

\[
s_{observation}^{(\ell)}(T)
=\operatorname{RelScore}(R_{obs}^{(\ell)},G_{observation}^{(\ell)}(T),observation\_valid,ww^\top).
\]

coverage 和 unique ratio 不允许由实现自行解释。它们精确定义为：

\[
coverage_{canonical}=N^{-1}\sum_i valid_i,
\qquad
coverage_{observation}=N^{-1}\sum_i observation\_valid_i.
\]

对 `W` 的每个有效行取 `argmax_j W_ij`，严格并列取最小 `j`；canonical unique ratio 等于这些 argmax 中不同索引数量除以有效行数量，有效行为空时取0。observation unique ratio 对 `V` 与 `observation_valid` 使用同一规则。

每层候选的双向关系目标固定为：

\[
J^{(\ell)}(T)=0.10s_{canonical}^{(\ell)}(T)+0.90s_{observation}^{(\ell)}(T)
-0.01\sum_{v\in\mathcal C}(1-v),
\]

其中 $\mathcal C$ 精确包含 canonical coverage、observation coverage、canonical unique ratio 和 observation unique ratio 四项。候选搜索不得读取攻击类型、攻击角度、裁剪比例或 test 标签。

12个锚点直接作用于抽样 token position，而不是原始未抽样 token ID。要求 `N>=12`，并按 `r=0,...,11` 冻结：

\[
a_r=\operatorname{floor}\left(r\frac{N-1}{11}+0.5\right).
\]

这同样是非负 half-up 舍入。对候选 `T`，观测锚点 `o_r=argmax_j W(T)_{a_rj}`，严格并列取最小 `j`；12个 `o_r` 均参与唯一性计数。锚点 residual 使用前述 `nearest_residual_(a_r)`。单个锚点是 inlier 当且仅当：其 canonical `valid` 为真、其 `o_r` 在12个锚点中只出现一次、且 residual 不高于 `0.20`。令 `n_valid` 为12个锚点中 `valid` 为真的数量，则 `inlier_ratio=n_inlier/n_valid`；`n_valid=0` 时固定为0。`mean_inlier_residual` 只对 inlier residual 求均值；没有 inlier 时，运行时比较值固定为正无穷、正式可序列化字段记为 `null`，因此 residual 门禁必然失败，禁止写入 NaN、0或 placeholder。

每个冻结层独立执行同一候选序列。层内选择只以有限 `J^(ell)(T)` 最大化为排序依据；非有限目标按无效候选处理，目标严格相等时取当前序列最早候选。粗搜索先得到 seed；每轮局部细化只在该轮候选中按同一规则取 best 并作为下一轮父候选。最后按“全部粗候选 -> 第1轮全部候选 -> 第2轮全部候选 -> 第3轮全部候选”的连接顺序再次执行同一选择，得到该层唯一 best；不得以 observation score、inlier 或 residual 作为层内第二排序键。

`candidate_sequence_index` 是上述每层最终连接序列中的从0开始索引。每一段先按其规范枚举顺序删除捕获域外或数值无效候选，再保持剩余顺序连接；索引范围精确为 `0,...,L_ell-1`，其中 `L_ell` 是该层最终有效连接序列长度。跨层选择后，正式记录保存选中层 best 在该层序列中的索引，不得改用粗网格索引、轮内索引或两个层合并后的索引。该索引也是层内目标严格并列时“最早候选”的唯一判据。

两个冻结层都必须完成上述层内搜索。对每层 best 定义：

\[
b^{(\ell)}=0.10s_{canonical}^{(\ell)}+0.90s_{observation}^{(\ell)},
\]

\[
confidence^{(\ell)}=
\max(0,b^{(\ell)})\cdot inlier\_ratio^{(\ell)}
\cdot\exp(-mean\_inlier\_residual^{(\ell)})
\cdot\min(coverage_{canonical}^{(\ell)},coverage_{observation}^{(\ell)}),
\]

没有 inlier 时 `confidence=0`。跨层唯一 transform 按冻结层顺序 `transformer_blocks.0.attn`、`transformer_blocks.23.attn`，对每层 best 的四元组 `(J^(ell), s_observation^(ell), confidence^(ell), -layer_index)` 执行字典序最大化；完全并列时选择第一个冻结层。该字典序只负责在两个已完成层内搜索的 best 之间选择唯一 transform，不改变层内“只按 J”规则。最终 `T_hat`、`candidate_identity`、coverage、unique、anchor 和 residual 事实全部来自该选中层；两层 Q/K 与稳定 pair 身份仍必须共同完整。

对选中层，`observation_relation_score=s_observation(T_hat)`、`bidirectional_relation_score=b(T_hat)`、`registration_objective_margin=J(T_hat)-J(I)`，并精确记录：

\[
registration\_alignment\_gain
=s_{observation}(\widehat T)-s_{observation}(I).
\]

两项都使用最终选中层、同一密钥关系、同一双边重采样算子和该层精确 identity 候选；不得使用另一层、raw geometry score、bidirectional score 或回正后图像分数替代。`I` 必须是同层粗候选序列中的精确 `(D0,theta=0,rho=1,tx=0,ty=0)`。结构门禁固定为平均 inlier residual 不高于 `0.20`、inlier ratio 不低于 `0.50`，且 canonical/observation coverage 均不低于 `0.45`。最终候选还必须满足 `observation_relation_score>0`、`bidirectional_relation_score>0`、`registration_objective_margin>0`，并保持直接 Q/K 来源、两层 Q/K 身份、四分量身份和稳定 pair 权重身份一致。canonical/observation unique ratio 只进入目标惩罚和证据记录，不另设硬阈值。

几何可靠性记为：

\[
geometry_{\mathrm{reliable}}
=
gate_{\mathrm{objective}}
\land gate_{\mathrm{inlier}}
\land gate_{\mathrm{residual}}
\land gate_{\mathrm{coverage}}
\land gate_{\mathrm{identity}}.
\]

其中 `gate_objective` 精确要求选中层候选目标、observation relation score 和 bidirectional relation score 均有限，后两者严格大于0，且 `registration_objective_margin>0`；`gate_identity` 精确要求两冻结层关系来源均为直接 Q/K，并且两层 Q/K operator metadata、原子内容身份、token schema 和稳定 pair 身份共同完整。任何必需 Q/K、层身份、pair 身份、有限数值或几何门禁缺失时，必须令 `geometry_reliable=false`；若缺失导致无法完成受治理测量，则还必须按第12.2节形成失败记录，不能伪造数值。

### 11.2 crop 与 crop-rescale 捕获边界

几何链不显式估计裁剪窗口，也不重建已经丢失的内容。`crop` 或 `crop-rescale` 只有在剩余可见 Q/K token 能被第11.1节的“二面体 + 有界相似变换”表示，并同时满足 coverage、inlier、residual 和正 registration objective margin 门禁时，才属于可救回范围。unique ratio 只进入冻结目标惩罚与证据记录，不构成独立硬门禁。

下列任一情况均必须失败关闭：

- 裁剪使 canonical 或 observation coverage 低于 `0.45`。
- 有效锚点不足以使12锚点协议达到 `inlier_ratio>=0.50`。
- crop-rescale 相对任一 `D0,...,D7` 的相似残差尺度都超出 `[1/sqrt(2),sqrt(2)]`。
- 相对任一 `D0,...,D7` 的相似残差旋转都超出 `[-32度,32度]`，或任一归一化平移绝对值超过 `0.28`。
- 攻击包含无法由该变换族表达的明显非均匀缩放、透视、剪切或局部形变。

因此，项目可以正式评测登记的全部 crop/crop-rescale 强度，但只能把满足捕获域和可靠性门禁的样本计为几何救回。域外失败必须进入攻击后检出率，不得从分母删除，也不得表述为“任意裁剪鲁棒”。

### 11.3 真实图像恢复

当样本先满足近阈值搜索条件，且第11.1节搜索得到 `geometry_reliable=true` 时，对待检图像执行参考系恢复。由于 `T_hat` 已按 output-to-input 约定把 canonical 输出坐标映射到 observation 输入坐标，`grid_sample` 必须直接消费 `T_hat`，不得再次求逆：

\[
y_{\mathrm{align}}=\mathcal W_{\mathrm{output\rightarrow input}}(y;\widehat T).
\]

必须对真实持久化或实际内存图像执行重采样，再重新编码 `y_align`。仅变换分数、token 坐标或缓存 latent，不能替代真实图像恢复和重新编码。

图像回正固定使用 bilinear 重采样、border padding 和 `align_corners=true`。重采样结果先截断到 `[0,1]`，再乘255、取 floor 并编码为 `uint8 RGB`；后续 VAE 和 Q/K 复验必须消费这一实际量化图像，而不是量化前浮点张量。回正后的 Q/K 复验用于记录实际重采样后的关系事实，不新增第二个阳性阈值；最终救回仍只由同一内容阈值决定。

### 11.4 同阈值救回规则

设内容阈值为 `tau`：

\[
m_{\mathrm{raw}}=s_{\mathrm{raw}}-\tau.
\]

直接内容通过为：

\[
positive_{\mathrm{by\_content}}
=
\mathbb I(m_{\mathrm{raw}}\ge0).
\]

设 calibration split 冻结的救回窗口下界为 `delta_low<0`。首先只根据原始内容分数确定是否需要执行几何搜索：

\[
geometry_{\mathrm{search\_required}}
=
\mathbb I(\delta_{\mathrm{low}}\le m_{\mathrm{raw}}<0).
\]

当 `geometry_search_required=false` 时，不得执行参考系搜索、图像回正或重新编码，并记录 `geometry_search_attempted=false`。当其为真时，才执行第11.1节的 Q/K 搜索并计算 `geometry_reliable`。救回资格随后定义为：

\[
rescue_{\mathrm{eligible}}
=
geometry_{\mathrm{search\_required}}
\land geometry_{\mathrm{reliable}}.
\]

对 `y_align` 使用相同内容检测器得到 `s_align`。救回成立条件为：

\[
rescue_{\mathrm{applied}}
=
rescue_{\mathrm{eligible}}
\land
\mathbb I(s_{\mathrm{align}}\ge\tau).
\]

最终水印证据为：

\[
evidence_{\mathrm{positive}}
=
positive_{\mathrm{by\_content}}
\lor rescue_{\mathrm{applied}}.
\]

因此，`geometry_reliable` 是几何搜索的输出，不能作为启动几何搜索的前置输入。几何链不得独立产生阳性。恢复后不得降低 `tau`，不得使用攻击特定阈值，也不得使用人工 `fail_reason` 代替数值救回窗口和几何可靠性门禁。

---

## 12. fixed-FPR 完整决策器校准

### 12.1 三档 fixed-FPR 结构

对启用几何救回的方法角色，内容阈值 `tau` 和救回窗口 `delta_low` 必须在该角色的独立 calibration 记录上校准。第11.1节的全部几何可靠性门禁是预先冻结的方法参数，calibration 不得重新拟合，但校准过程必须实际执行包含这些门禁、真实回正和同阈值重判的完整决策器。不得只校准原始内容分数后再无约束增加救回机会。`content_chain_only` 只按相同三组负观测和预算规则校准 raw content threshold，不执行窗口拟合或几何测量，并登记 `rescue_margin_low=null`。

校准负观测精确分为三个互斥统计组：

- `clean_negative_registered`：`sample_role=clean_negative`、`attack_id=clean`、`key_relation=registered_key`。
- `attacked_negative_registered`：`sample_role=attacked_negative`、`key_relation=registered_key`，并覆盖 registry 中 `attack_evidence_role=core_claim_required` 的全部核心证据攻击。
- `watermarked_wrong_key`：`sample_role=watermarked_positive`、`key_relation=wrong_key`，使用与 registered-key 正样本相同的 clean/核心攻击图像，不重新生成图像。

任何观测只能进入一个统计组；三组都必须非空，且 calibration 至少包含3个唯一 Prompt。按唯一 Prompt ID 执行确定性嵌套划分：使用 `sha256("dual_chain_nested_calibration_v1\0" + prompt_id)` 升序排序，前 `floor(N_prompt/3)` 个 Prompt 进入 `rescue_window_fit`，其余进入 `threshold_freeze`；同一 Prompt 的全部角色、核心攻击和密钥观测必须留在同一子集。两子集不得与 test Prompt 重叠，补充攻击不得进入任一 calibration 子集。

对任一统计组的 `n` 条观测，经验假阳性预算固定为：

\[
B(n,\alpha)=\max\left(0,\left\lfloor\alpha(n+1)\right\rfloor-1\right).
\]

`rescue_window_fit` 先仅使用 raw content score，选择满足三个统计组各自预算的最低 `nextafter(score,+inf)` 作为 provisional content threshold。救回窗口候选集合固定为 `nextafter(0,-inf)`，以及全部负 raw margin 和这些 margin 的 `nextafter(margin,0)`；仅保留有限负值并升序排列。calibration 期间允许在不读取 test 数据的前提下，对 calibration 负观测各执行一次真实 Q/K 搜索、冻结可靠性判断、图像回正和 aligned 内容测量，以便拟合尚未知的窗口；不同窗口候选只能复用这些真实测量并重算布尔决策，不得重跑搜索迎合候选。选择数值最小、即最宽且同时满足三组假阳性预算的 `delta_low`。这一 calibration-only 测量例外不改变 test/application 阶段“先判近阈值、再按需搜索”的规则。

在 `threshold_freeze` 上固定该 `delta_low`。对每条观测定义与完整决策等价的连续分数：

\[
e=
\begin{cases}
\max\left(s_{raw},\min(s_{align},s_{raw}-\delta_{low})\right),
& geometry_{reliable}=true,\\
s_{raw},& geometry_{reliable}=false.
\end{cases}
\]

从全部 `nextafter(e,+inf)` 候选中选择数值最低、且使三个统计组分别不超过 `B(n_g,alpha)` 的 `tau`。该规则唯一冻结 `tau` 和 `delta_low`；不得从正样本检测率选择候选，不得重新拟合几何门禁，不得读取 test 数据。若任一统计组、Prompt 子集、有限分数或可行候选缺失，校准失败关闭。

上述经验预算定义 operating point。论文结论闭合还必须按统一统计协议报告三个统计组及逐攻击 FPR 的分子、分母和登记单侧置信上界；样本量不足以支持目标 FPR 时，不得仅凭经验0次误报宣称 fixed-FPR 已成立。

test positive、clean negative、wrong-key negative 和登记攻击样本必须使用同一个冻结决策协议。任何 test split 回调、逐攻击回调或恢复后降阈值均构成证据泄漏。

三档目标 FPR 精确冻结为：

| profile | 目标 FPR |
|---|---:|
| `probe_paper` | `0.1` |
| `pilot_paper` | `0.01` |
| `full_paper` | `0.001` |

每档分别在该档独立 calibration split 上校准完整决策器，但使用相同校准算法、候选阈值规则、负样本角色、几何门禁规则、字段和产物 schema。低层 profile 的阈值不得直接外推为高层 profile 的阈值。

三档承担不同证据职责：`probe_paper` 只负责同构流程和初步可行性；`pilot_paper` 是主投稿证据 profile，完整闭合时允许在 FPR=0.01 与其登记样本规模内支撑论文主张；`full_paper` 是可选扩展，只用于在资源允许时检验 FPR=0.001 与更大样本规模。`full_paper` 未运行或证据不完整不阻断 `pilot_paper` 的结果闭合、投稿就绪或作用域内主张，但不得把 `pilot_paper` 结论外推为 `full_paper` 结论。

`probe_paper`、`pilot_paper`、`full_paper` 必须使用完全相同的：

- 方法原语。
- 模型与 revision。
- 载体定义。
- 几何模板定义。
- 决策字段和产物 schema。
- fixed-FPR 校准程序。
- 几何救回规则。
- 第12.2节最小6角色方法/消融集合。

三档的 seed-key 交叉重复集合与顺序固定相同，`registered_repeat_count` 必须精确为5，不属于允许变化的规模字段。5个重复必须是权威随机化登记表中预先冻结、身份互异且顺序唯一的5个 `(generation_seed_offset, watermark_key_index)` 配对；它们不表示5个 seed 与5个 key 的笛卡尔积，也不得在观察结果后选择。三档只允许在登记的 Prompt / 样本规模、上述目标 FPR 和由样本规模与 FPR 派生的统计强度字段上变化；操作输出路径只能由 `profile_id` 派生，不能作为科学协议差异。三档的 `full_dual_chain` 不得通过关闭几何链、修改嵌入强度、改变搜索范围或替换内容分数形成不同方法。第12.2节登记的消融必须作为同一角色集合在三档同步存在，不能替代主方法。实际执行的每档都必须按同一结论集合形成自身闭合结果；`probe_paper` 通过只说明同构全流程在 FPR=0.1 和较小样本规模下闭合，`pilot_paper` 只支持自身作用域，二者均不自动证明 FPR=0.001 的统计结论已经成立。

### 12.1.1 等价执行复用边界

以下优化只改变执行计划，不改变样本、方法或统计协议，因此在身份绑定和等价性检查通过时允许使用：

- 同一 `generation_input_identity_digest` 的 clean 图像可跨6个方法角色复用；每个角色仍独立形成角色记录、攻击观测、阈值和决策。
- 同一图像字节及完全一致的模型 revision、预处理、dtype 和科学依赖下，VAE latent、公开 Q/K 原子、S/T/R/Q 内容观测以及 Inception、DINO、CLIP 等角色无关质量特征只计算一次。密钥依赖载体、稳定 token、几何极性、角色分数和决策不得包含在共享包中。
- 同一 evaluated image 的 registered-key 与 wrong-key 观测可共享图像持久化、公开原子和密钥无关测量；切换密钥后必须重新构造全部密钥模板、内容分数、几何目标、搜索结果和最终决策。
- 普通攻击结果可按 source image SHA-256、完整攻击配置、攻击随机种子、代码版本和科学依赖摘要缓存。字节不同的 clean 与 watermarked 图像不得命中同一攻击缓存。
- application/test 阶段只对落入冻结近阈值窗口的失败样本惰性执行几何搜索。calibration 为拟合救回窗口允许对每条负观测真实搜索一次，并在候选阈值之间只重算布尔决策。
- Prompt-repeat 是最小持久化与恢复原子；核心完成判据必须验证全部6角色、样本角色、7项核心攻击、密钥关系和失败记录。补充攻击若运行则按攻击 ID 另记描述性完成状态。恢复不得按得分选择、补跑到成功或丢弃失败样本。
- 样本可跨 GPU 并行，但每个正式样本仍保持 `batch_size=1`、确定性身份、唯一写入和同一 GPU 类别及软件环境约束；设备分配只作为 provenance，不得成为样本选择变量。
- 三档可复用共同 Prompt 上与 profile 无关的图像、攻击和公开特征，但必须各自执行 calibration、冻结阈值、决策重建和统计聚合。

若实现共享6角色在 callback 索引10以前的生成前缀，必须冻结并持久化相同的 Prompt 条件、scheduler 状态、`z_9`、`z_10`、模型/依赖身份和随机状态，再从同一前缀分叉角色后缀。正式启用前必须以逐角色独立运行作对照，证明图像摘要、latent、原子测量和最终记录在声明的数值容差内等价；未通过时必须回退到独立执行。缓存命中同样必须由完整身份摘要验证，缓存文件本身不支持论文主张。

### 12.1.2 核心与补充攻击证据职责

攻击的计算资源档位与论文证据职责必须分离。`resource_profile` 只描述执行成本和设备需求；唯一决定是否进入四项 required claims 的字段是 `attack_evidence_role`：

- `core_claim_required`：进入三组 calibration 负观测、完整6角色、4个主表 baseline、逐攻击检测/误报/质量统计和四项 required-claim conjunction。
- `supplementary_descriptive`：不进入 calibration、required quality gate、baseline superiority 或 mechanism necessity；只在核心决策冻结后以同一阈值和同一检测器形成补充描述性检测、误报与质量结果。

三档必须共享完全相同的核心攻击身份、顺序、参数、随机化和证据职责。补充攻击也必须保持配置身份稳定，但可以整体不运行；部分补充结果只能逐攻击报告完成状态，不能形成“完整补充集合已验证”的结论。补充攻击失败必须保留在该攻击自身的描述性分母，但不回写核心 calibration、阈值、核心跨攻击统计或 required claims。

核心攻击精确为7项，补充攻击精确为10项；其唯一 ID 集合由三档协议同构规范和机器攻击 registry 共同绑定。任何把补充攻击静默纳入 required gate、把核心攻击降级为补充、按观测结果改变职责，或用 `full_main/full_extra` 推断证据职责的实现都不符合本原语。

### 12.2 正式单样本 schema

三档正式方法角色固定为相同的最小6角色集合：`full_dual_chain`、`uniform_content_routing`、`lf_only_content`、`hf_tail_only_content`、`content_chain_only`、`geometry_recovery_without_embedded_sync`。其中 `full_dual_chain` 是正式主方法，其余5个是从同一实现关闭登记分支得到的必要消融；不得在不同 profile 增删角色。每个角色使用同一 fixed-FPR 算法独立校准自己的内容阈值。冻结决策身份必须同时绑定 `method_role`、目标 FPR、内容阈值、可空救回窗口、geometry/rescue enabled、几何门禁和 calibration population；不得跨角色复用决策。只有 `content_chain_only` 允许 `rescue_margin_low=null` 并禁用 geometry/rescue；其成功记录的 geometry/rescue 布尔字段固定为 false。其余5个角色必须记录有限负 `rescue_margin_low` 并启用同一救回协议。

正式记录的 `sample_role` 只允许以下三个互斥值：

| `sample_role` | 来源图像 | 评测职责 |
|---|---|---|
| `watermarked_positive` | 使用登记密钥嵌入的图像 | 测量 clean 与攻击后的检出能力。 |
| `clean_negative` | 未嵌入水印的真实生成图像 | 测量自然误报和 calibration 负分布。 |
| `attacked_negative` | 对未嵌入水印的真实生成图像执行与正样本相同的登记攻击 | 测量攻击、恢复、重采样或重生成是否诱发伪阳性。 |

`key_relation` 与 `sample_role` 正交，只允许 `registered_key` 和 `wrong_key`；`attack_id` 使用 `clean` 或唯一 attack registry 中的精确攻击 ID。正式 success/failure 身份必须显式绑定 `attack_evidence_role`：当 `attack_id=clean` 时其值精确为 `null`；当 `attack_id` 为登记攻击时，其值只允许 `core_claim_required` 或 `supplementary_descriptive`，并且必须与唯一 attack registry 逐项一致。不得从 `resource_profile`、攻击成本、输出路径或观测结果推断该字段。非主张 probe 不得进入正式论文 success/failure 总体，也不得借用 `null`、核心或补充职责形成正式记录。`key_relation` 必须选择该条观测用于 LF、HF-tail、原始内容分数、几何模板与搜索、aligned 内容分数和最终布尔决策的唯一评分密钥，并记录不泄露密钥材料的 `scoring_key_identity_digest`。当 `key_relation=wrong_key` 时，不得继续使用 registered key 计算内容分数或救回，仅把同一图像上的 registered/wrong-key 几何分数保留为并列诊断。每条检测记录至少绑定：

- `method_role`、`prompt_id`、`randomization_repeat_id`、`sample_role`、`key_relation`、`attack_id` 和可空 `attack_evidence_role`。
- `prompt_text_digest`、`generation_input_identity_digest` 和 `generation_seed_random`。
- `scoring_key_identity_digest`，以及 source/evaluated 图像 SHA-256、宽高和持久化成员路径。
- 攻击配置摘要、`attack_seed_random`、代码提交和科学依赖 profile 摘要。
- `model_id`、`model_revision` 和 `runtime_component_identity_digest`。
- `prg_version`、`prg_identity_digest`、`method_definition_digest`、`runtime_config_digest` 和 `content_routing_reference_registry_digest`。
- 原始 LF/HF/内容分数、aligned LF/HF/内容分数、`content_threshold`、`rescue_margin_low` 和目标 FPR。
- Q/K 原子、几何候选、捕获域、coverage、inlier、residual、identity 和 rescue 字段。
- 最终图像 `registered_key_geometry_score`、`wrong_key_geometry_score` 和精确差值 `registered_wrong_key_geometry_score_margin`。
- `positive_by_content`、`geometry_search_required`、`geometry_search_attempted`、`geometry_reliable`、`rescue_eligible`、`rescue_applied`、`evidence_positive`。
- `decision_identity_digest` 和 `measurement_identity_digest`。

正式单样本记录是 success/failure 联合 schema。成功记录必须具有 `measurement_status=success`，全部必需身份、图像和数值字段均非空且有限；其中 `registered_wrong_key_geometry_score_margin` 必须由同一记录的两个最终图像几何分数相减得到并保持有限。几何搜索未启动时 `geometry_measurement=null`；几何搜索完成时必须嵌套可序列化几何对象，至少显式包含两个冻结层各自 best 的候选摘要、选中层、`candidate_identity`、规范候选索引、2×3 transform、12锚点索引与 inlier mask、canonical/observation/bidirectional relation score、目标、identity 目标及 margin、alignment gain、confidence、coverage、unique、inlier、可空 mean residual、两层 Q/K 身份摘要、回正图像 SHA-256/成员路径/尺寸和 `geometry_measurement_digest`。不得把含 Tensor 的运行时 `aligned_image` 直接序列化为正式记录。

当 generation、attack、image persistence、VAE encode、Q/K extract、geometry search、alignment resample、aligned VAE encode 或 schema materialization 失败时，必须形成 `measurement_status=failure` 的失败记录。失败记录继续绑定相同的 Prompt/repeat/role/key/attack/attack-evidence/code/dependency/model/PRG/method/runtime/reference/decision 身份，显式记录受治理的 `failure_boundary` 与稳定 `failure_code`；能取得的图像身份照实记录，不可得图像和测量字段为 `null`，`evidence_positive=false`。若两个最终图像几何分数尚未完整取得，`registered_wrong_key_geometry_score_margin=null`；若二者均已取得，该差值必须同步取得并等于两者之差。不得用 NaN、0、随机向量、synthetic 值或 placeholder 冒充不可得测量。失败样本按 `sample_role/key_relation/attack_id` 保留在对应 detection/FPR 分母，分子贡献0；质量缺失另报失败数量，不得插补质量分数。

`generation_seed_random` 与 `attack_seed_random` 是随机轨迹字段，必须保留 `_random` 后缀。成功与失败记录的 `measurement_identity_digest` 都必须使用规范编码绑定该记录除自身外的全部身份、显式 null、可得图像身份和测量字段，包括 `attack_evidence_role` 和 `registered_wrong_key_geometry_score_margin`；任一记录只要已经完成几何测量，就必须继续绑定嵌套 `geometry_measurement_digest`。不得只绑定分数或最终决策。

`attacked_negative` 不得由正样本更换错误密钥冒充；它必须以未嵌入源图像真实执行同一攻击。缺失角色或来源身份的记录不得进入正式聚合；具有完整来源身份的受治理失败记录必须进入分母，不能按“缺失测量”删除。

### 12.3 正式检测与质量指标

正式结果至少同时报告：

1. `watermarked_positive` 的 clean detection rate。
2. 每项核心攻击、核心攻击族和核心集合跨攻击聚合的 attacked detection rate，并给出 Prompt 聚类置信区间；补充攻击只单独报告描述性结果。
3. `clean_negative` 的经验 FPR。
4. `attacked_negative` 的核心逐攻击 FPR 和核心集合跨攻击 FPR；补充攻击 FPR 不进入 required gate。
5. wrong-key FPR。
6. `positive_by_content`、`rescue_eligible`、可靠恢复数、`rescue_applied` 和净 `rescue_gain`。
7. clean/watermarked 及 source/attacked 的配对 SSIM。
8. 与优化模型族隔离的冻结独立视觉内容 cosine；其唯一身份来源为 `configs/independent_semantic_quality_evaluator.json`，其中 `independent_semantic` 是兼容标识符，不表示 Prompt 语义或图文对齐；三档必须消费同一配置摘要。同源 CLIP cosine 只作为 mechanism consistency 诊断。
9. clean/watermarked 的 FID、KID、Prompt 条件 KID，以及7项核心攻击要求的逐攻击分布质量推断；已运行补充攻击的同类质量只作描述性披露。

图像质量指标只评价方法成本和攻击后保持性，不得进入水印阳性公式。所有点估计、置信区间、分母、失败样本和缺失样本必须按同一 schema 持久化；域外几何失败和生成式攻击失败不得从 attacked detection rate 分母剔除。

### 12.4 生成式攻击的正式评测职责

对 registry 中已经登记且实际执行的重生成、扩散净化或其他生成式攻击，流程必须对 `watermarked_positive` 与 `attacked_negative` 执行相同攻击实现、模型 revision、参数和随机化协议，并报告攻击后检出率、受攻击负样本 FPR 与图像质量。核心生成式攻击同时进入 required claims；补充生成式攻击只使用已经冻结的核心决策器形成描述性结果，不参与阈值拟合或核心结论投票。

生成式攻击主要检验内容链证据是否在内容重构后保留，以及攻击过程是否制造伪阳性。几何链只在其真实 Q/K 关系满足同一捕获域和可靠性门禁时尝试回正；不得把几何链表述为生成式攻击的专用恢复器，也不得因生成式攻击无法表示为有界相似变换而跳过该样本。攻击集合由实验协议登记；算法原语冻结核心/补充证据职责、对称评测和禁止结果后选择的边界。

### 12.5 单模型内部敏感性

正式方法始终使用第3.3节的名义参数。参数依据由独立的单模型内部敏感性实验补充，不允许为不同论文 profile、攻击或样本选择不同参数。该实验只使用冻结 SD3.5 模型、一个登记的小规模 Prompt 子集和一个固定 repeat，并采用单因素变化；非名义 reference 分位数必须写入敏感性专用临时身份，不得覆盖正式 `configs/content_routing_reference_registry.json`：

| 敏感性轴 | 候选值 | 名义值 |
|---|---|---|
| `g_ref/r_ref/q_ref` 共同分位数 | `0.90, 0.95, 0.99` | `0.95` |
| 局部敏感性探针相对步长 | `5e-4, 1e-3, 2e-3` | `1e-3` |
| LF/HF 内容强度共同倍率 | `0.75, 1.00, 1.25` | `1.00` |
| geometry 强度倍率 | `0.75, 1.00, 1.25` | `1.00` |

内容共同倍率必须同时乘到 `0.0025/0.0015`，保持 LF/HF 比例不变；geometry 倍率只作用于 `0.0010`。Q/K 关系公式、捕获域、搜索网格、正式内容权重和 fixed-FPR 决策规则在敏感性实验中不得改变。

敏感性实验必须报告 clean/登记代表攻击下的检测率、配对质量、分支有效能量和几何归因诊断。它只用于说明名义参数附近是否稳定，不替代三档正式结果，也不得把表现最好的候选反馈到 test split 后重新冻结方法。

---

## 13. 正式主张边界

### 13.1 允许主张

只有在真实 GPU 证据和正式统计支持后，项目才允许主张：

1. 嵌入区域与局部强度由当前图像和 Prompt 的空间语义显著性、纹理复杂度、相邻潜空间响应和单方向局部扰动敏感性共同决定。
2. LF 承担主内容证据，HF-tail 提供困难攻击条件下的补充鲁棒性余量，二者只形成一个加权内容分数。
3. 带密钥 Q/K 关系模板在生成阶段形成真实几何同步信号。
4. 几何链能够在登记几何攻击下恢复参考系，并通过同阈值内容重判产生可测量的救回增益。
5. 完整双链决策器在7项预登记核心攻击和登记 fixed-FPR 下保持受控误报。
6. 正样本、干净负样本和受攻击负样本的对称评测能够区分检出能力、自然误报与攻击诱发伪阳性。

### 13.2 明确禁止主张

以下主张不属于本文档：

- 学习或构建了全局潜流形。
- 716维特征水平集表示通用感知质量流形。
- CLIP 局部相关图等同于人类注视真值或语义分割真值。
- 相邻 latent 响应等同于全轨迹稳定性，或单方向有限差分等同于完整 Jacobian 敏感性。
- HF-tail 忠实复现 T2SMark 或 PRC-Watermark。
- 几何链自身能够证明水印归属。
- 几何救回对所有旋转、缩放、裁剪或任意重生成攻击都有效。
- 单个 Prompt、单个 seed-key 或 CPU fixture 能支持论文主张。
- 7项核心攻击结果能够外推为对未进入核心集合的高级编辑、自适应去除或任意攻击普遍鲁棒。

---

## 14. 禁止代理与静默回退

正式路径禁止：

- 使用随机图、固定图、Prompt digest 或样本 ID 代替语义显著图。
- 使用全图 CLIP 分数广播代替 patch 级 Prompt 条件相关图。
- 使用随机图、样本密钥相关探针、完整 Jacobian 代理或预制风险值代替真实 `R`、`Q`。
- 使用原始高斯幅值 TopAbs 冒充 HF-tail，而不执行高通。
- 使用 hidden state cosine、随机矩阵或预制关系分数代替真实 Q/K。
- 检测不到冻结 Q/K 层时关闭几何链并继续标记正式结果。
- 不执行真实图像回正和重新编码，仅重算缓存分数。
- 几何分数直接绕过内容阈值。
- 用 synthetic、proxy、placeholder 或 compatibility path 支持正式主张。

---

## 15. 最小可证伪验收

目标方法至少必须通过以下互相独立的验收：

1. 图像变化或 Prompt 变化会产生可解释的语义显著图变化。
2. 相邻真实 scheduler latent 产生非代理的 `R`，公开确定探针和一次额外 VAE 解码产生非代理的 `Q`。
3. LF 和 HF-tail 使用不同 PRG domain，掩码前载体具有不同的空间频率和稀疏性事实；掩码后的空间调制更新不被错误主张为严格带限、频谱正交或不重叠。
4. HF-tail 在执行高通后才做20%幅值截断。
5. `configs/content_routing_reference_registry.json` 能由隔离参数划分逐字段重建 `g_ref/r_ref/q_ref`。
6. 内容路由后不重新归一化，`S/T/R/Q` 确实改变实际写入幅值；两个分支的有效能量和实际 dtype 写回均非零。
7. 未掩码载体盲相关在 calibration/test 中使用同一统计量，低容量样本没有被删除或重新生成。
8. 几何分支来自真实冻结层 Q/K，对带密钥关系目标产生正增益；仅图像 Q/K 复验使用冻结 VAE、公开噪声、schedule 索引7和 empty-text 无 CFG 条件。
9. 最终图像上的 registered-key 几何分数同时高于 wrong-key 和 matched content-only 反事实；该单 Prompt 归因不直接支持论文主张。
10. 三个分支只形成一次实际 latent 写回，并满足固定的分支强度和总 L2 预算。
11. 关闭 Jacobian、JVP/VJP 和 PSD-CG 后，目标核心路径仍完整运行。
12. 只有原始内容分数位于近阈值窗口时才启动几何搜索；`geometry_reliable` 由搜索产生，不能作为搜索前置条件。
13. 捕获域内的旋转、缩放、平移和 crop-rescale 能够产生真实几何恢复记录；域外样本明确失败关闭且保留在分母。
14. 恢复后仍使用同一内容阈值，identity 对齐、wrong-key 或不可靠几何不能产生伪救回。
15. 三类 `sample_role` 均能从真实图像形成完整记录，受攻击负样本不能由 wrong-key 正样本替代。
16. 三档完整决策器分别在独立 calibration split 达到 `0.1/0.01/0.001` 的登记 FPR。
17. 7项核心攻击完整覆盖6角色、4个主表 baseline 和对称正负样本职责；已运行补充生成式攻击同样对正样本与受攻击负样本执行对称评测，但不进入 required claims。
18. 单模型内部敏感性使用冻结小规模子集且不反馈修改正式 test 参数。
19. 单 Prompt GPU 资格化只证明方法算子可运行，并保持 `supports_paper_claim=false`。

---

## 16. 变更治理

以下任一变化都属于方法原语变化，必须先修改本文档，再同步方法定义、配置、测试、GPU 资格化和实验协议：

- 替换语义显著性模型、revision、特征层、视觉投影、文本编码或归一化规则。
- 修改纹理算子或 `g_ref` 参数物化语义。
- 修改潜空间响应公式、局部敏感性探针、探针步长或 `r_ref/q_ref` 参数物化语义。
- 修改 LF 滤波、HF 高通、tail 比例或 PRG domain。
- 修改冻结 Q/K 层、关系分量、token 规则或几何候选搜索。
- 修改注入索引、写回次数、分支顺序、三分支强度或任一幅值预算。
- 修改内容权重、救回窗口、几何可靠性门禁或最终判定公式。
- 修改三档目标 FPR、三类样本角色、核心/补充攻击 ID 集合、`attack_evidence_role` 或生成式攻击的正负样本职责。
- 允许 Prompt、原始 latent、生成轨迹或攻击标签进入检测器。

任何实现便利、显存优化或运行时回退均不得静默改变上述方法身份。
