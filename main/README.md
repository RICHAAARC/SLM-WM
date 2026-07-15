# Main

`main/` 是 SLM-WM 的最小论文方法实现包。该目录只保存方法定义和方法运行所需的最小数学工具, 不保存实验编排、结果治理、baseline、命令行入口或 Notebook 逻辑。

## 目录职责

- `methods/semantic/`: 使用冻结解析范围的 CLIP patch-to-CLS 一致性、解码纹理、5x5局部对比度、紧邻 scheduler 步 RGB 稳定度和独立跨层 Q/K 稳定度, 构造三个分支的风险场、严格资格集合和连续承载预算。`DifferentiableSemanticFeatureRuntime` 接收外部注入的 VAE 与视觉编码器, 真实计算512维 CLIP 特征和204维手工结构描述符, 不依赖实验层模型注册表。
- `methods/subspace/`: 使用512维归一化 CLIP embedding 与204维明确限定的手工结构描述符组成的716维特征 JVP/VJP、显式风险算子、无阻尼 PSD-CG 和逐列残差门禁求解 Jacobian Null Space。`solve_semantic_branch_subspace` 是单 latent、单分支的公开核心方法入口, 把密钥候选生成、局部 Jacobian 低响应投影和残差门禁组合为可复用算子。
- `methods/carrier/`: 构造空间低通 LF 模板、高斯幅值尾部截断模板及其安全子空间投影。
- `methods/update_composition.py`: 在单位方向上分离处理方向活动 epsilon 与数值退化 epsilon, 清理零预算支持后执行逐位置风险硬包络缩放；attention 候选和最终写回共同调用唯一固定顺序 float32 合成原语, 对 original latent 只执行一次实际 dtype 转换并产出可重算共同回溯证据。
- `methods/geometry/`: 从真实 Transformer Q/K 直接构造中心化 logit、可微 rank、抽样图像 token 关系概率和距离调制中心化概率四分量图, 计算目标梯度, 构造可核对身份的稳定 token pair 权重, 并通过攻击配置无关的分层搜索恢复二维参考系。归一化 token 坐标把角点中心映射到 -1 与 1, 与图像仿射重采样的 `align_corners=True` 完全一致。
- `methods/detection/`: 实现只读取待检图像、密钥和公开检测配置的盲检接口, 注册前后使用同一 pair 权重身份且不在对齐后重新选择 token。
- `core/digest.py`: 提供 JSON 稳定摘要及绑定 dtype、shape 与连续原始字节的版本化 Tensor 内容摘要, 供风险、Null Space、分支更新和 Q/K 科学原子共同复用。
- `core/keyed_prg.py`: 实现 `sha256_counter_normal_icdf_table20_float32`。高斯用途从 SHA-256 大端计数器块组成的 MSB-first 连续比特流提取20位索引, 查询冻结 Q20 中点逆 CDF 表；注意力关系符号使用独立的53位开区间 uniform 路径, 使设备 RNG 不进入方法身份。
- `core/normal_quantile_table.py`: 保存 $q_i=\operatorname{round}_{\mathrm{binary32}}\!\left(\Phi^{-1}((i+0.5)/2^{20})\right)$ 的1048576项冻结表。完整大端字节 SHA-256 为 `70abf440a7f3670147965ffa52f5aaa639dab97f6282b68f3a9a1b1ce5e6cf5a`；该有限离散量化分布的理想中点 KS 距离为 $2^{-21}$, 加上已登记的 float32 舍入误差后上界为 `4.912236096776823e-7`。

## 分层边界

- 数据划分、模型加载、fixed-FPR、攻击、正式消融、单模型参数敏感性和数据集运行位于 `experiments/`。
- baseline、公平对比、论文证据审计和投稿就绪分析位于 `paper_experiments/`。
- 独立服务器命令位于 `scripts/`。
- Colab 包装和 Notebook 位于 `paper_workflow/`。

`main/` 不得导入上述任何外层目录。

核心包提供方法数学原语、可微特征运行时、局部子空间求解、载体、注意力几何、更新合成和图像检测。具体 SD3.5 模型下载、设备放置、prompt 循环和结果持久化属于可替换的外层模型执行适配, 不得反向进入核心包。该划分属于通用依赖倒置写法, 不是用外层代理替代核心算法。

## 方法语义

方法名称中的“潜流形”只表示当前样本716维完整特征隐式水平集的局部安全切空间解释。核心数值对象是分支风险支持且通过逐列 Jacobian 残差门禁的 Null Space 基底；`main/` 不构造全局非线性流形，不验证常秩定理条件，也不实现坐标图、测地线或回缩算子。正式更新采用构造式协议：LF 与尾部模板分别投影并由对应风险硬包络缩放，注意力分支直接消费同一风险有界单位方向并从最大允许步长执行真实 Q/K 单调回溯, 三者按固定顺序在 float32 中唯一合成并只执行一次实际 dtype 转换。外层运行时对风险清理后的每个实际方向重新执行独立精确 JVP, 随后对实际联合写回 Tensor 复验预算、JVP、有限变化和 Q/K。该协议不等价于联合标量 `argmax` 求解。

`methods/method_definition.py` 提供上述语义边界的版本化可机读记录与稳定摘要。运行配置身份和结果 metadata 同时绑定该摘要，使方法术语或构造协议发生变化时必须形成新的科学单元身份。

内容分支由空间低通 LF 与高斯幅值尾部截断 `tail_robust` 组成。LF 在二维低通后执行全 Tensor 去均值和 L2 归一化；`tail_robust` 按高斯模板元素绝对幅值选择分布尾部, 截断后只执行 L2 归一化, 使非入选位置保持精确0。尾部分支不使用 FFT、DCT、带通滤波或空间频带掩码。

正式 Q/K 算子只消费配置中精确冻结的 `transformer_blocks.0.attn` 与 `transformer_blocks.23.attn`, 不按模型枚举顺序动态挑选层。正式检测不得接收 Prompt、生成种子、初始噪声、生成轨迹、源 latent 或样本级 Null Space。检测端可使用的全部公开随机量必须由密钥、模型标识和冻结检测协议确定。密钥化随机原语固定为 `sha256_counter_normal_icdf_table20_float32`：高斯用途查询冻结 Q20 中点逆 CDF float32 表, 53位开区间 uniform 路径只用于注意力关系符号。Q20 输出是有限离散的量化标准正态, 不是连续精确的 $\mathcal N(0,1)$。规范 float32 生成与目标 dtype 转换均在 CPU 完成, 随后才搬运到目标设备；CPU 与 CUDA 的 PyTorch RNG 均不参与方法身份。MPFR 逐项舍入复验属于 `tools/harness/` 的外层证据, 不进入 PRG 算法摘要或采样身份。当前逐字节固定向量只在 Windows CPU 实测, Linux/Colab 一致性必须在 GPU 运行前门禁中复验。

核心方法对象为三个分支的风险值、连续预算和资格 mask, Jacobian 候选矩阵、风险路由候选及其响应、投影方向及其响应、QR 基底及其响应、逐列 QR 参考响应, 三个分支更新以及真实 Q/K 输入输出提供角色限定的精确内容摘要。Q/K 原子按冻结层记录抽样后的 Q、K、中心化 logit、关系概率和原始二维 token 索引；算子身份与数据内容身份分开保存, 避免把不同图像的数值内容误写为同一个算子协议。

四分量评分使用显式非负归一化权重协议。完整方法为 `(0.25,0.25,0.25,0.25)`, 正式留一消融将一个分量置零并令其余三个分量各取 $1/3$。同一权重对象贯穿嵌入梯度、回溯验收、最终成图归因、仅图像盲检、仿射注册和恢复后同步评分；活动分量集合与协议摘要进入方法记录。
