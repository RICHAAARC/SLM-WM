# Configs

此目录保存论文运行配置、prompt 配置和运行环境约束记录。

`paper_claim_registry.json` 冻结5项论文主张、必要主张集合、可选主张集合和唯一三态决策枚举。当前 `parameter_robustness` 是可选主张, 不得无条件否决固定参数下的必要结论；旧 `supports_paper_claim` 只能由必要主张集合的重算结果派生。

`paper_quality_claim_protocol.json` 冻结感知质量、语义对齐和分布保持三类非劣效语义。Prompt 是唯一总体重采样单位, 9个注册 repeat 嵌套在每个 Prompt 内；clean-watermarked FID 仅作为描述性分布位移, Prompt 条件 KID 及其 Prompt 聚类 bootstrap 区间是主要分布证据。配置中的界限在新的正式运行前提交, 不允许由结果回灌修改。

`core_method_dependency_identity.json` 只定义最小核心方法包的可安装边界: Python `>=3.11`、`torch>=2.11,<2.12`、`setuptools.build_meta` 构建后端以及唯一 `main` 包根。PyTorch 范围与正式 SD3.5 GPU 锁中的2.11系列一致, 具体 CPU 或 CUDA wheel 由安装环境负责选择。该身份不替代、引用或放宽论文实验层的六个完整 wheel 哈希锁。

## 论文 prompt 配置

| 配置文件 | prompt 数量 | 目标 FPR | 支持主张 | 用途 |
| --- | ---: | ---: | --- | --- |
| `paper_main_probe_paper_prompts.txt` | 70 | 0.1 | `probe_claim` | 小规模正式结果包, test 包含34个独立 Prompt。 |
| `paper_main_pilot_paper_prompts.txt` | 700 | 0.01 | `pilot_claim` | 中等规模正式结果包, test 包含340个独立 Prompt。 |
| `paper_main_full_paper_prompts.txt` | 7000 | 0.001 | `full_claim` | 全规模正式结果包, test 包含3400个独立 Prompt。 |

正式 Prompt bank 由 Microsoft COCO 2017 train captions 与 Google Research PartiPrompts 的固定字节版本构造。`configs/prompt_selection_manifest.jsonl` 保存6000条 COCO caption 与1000条 PartiPrompt 的来源记录身份、原始文本和选择摘要；三级 Prompt 文件分别是该清单的前70、前700和前7000条, 并保持 6:1 来源比例。Prompt ID 由统一清单索引和原始文本构造, split 在每个连续70条块内固定分配3/33/34, 因而相同前缀在三级运行中保持相同 Prompt ID 与 split。`configs/prompt_source_registry.json` 固定来源 revision、文件大小、SHA-256、选择清单摘要和三个 Prompt 文件摘要。运行入口会从清单前缀逐字节重建当前 Prompt 文件, 不接受未登记文本、来源改写或独立重新抽样。

三组配置使用同构 fixed-FPR 统计协议, 目标 FPR 分别为0.1、0.01和0.001；方法参数、攻击协议、baseline 入口、Wilson 单侧 FPR 上界算法、Hoeffding 结果区间、随机种子和结果闭合逻辑必须保持一致。共同协议结果记录必须拒绝 proxy、placeholder、fallback、synthetic 和 formal-null 证据。

生成随机性由 `crossed_generation_seed_watermark_key` 协议统一治理。协议登记3个生成 seed 偏移与3个密钥索引形成9个交叉重复, 三类论文运行层级使用完全相同的重复注册表。单次执行通过 `SLM_WM_RANDOMIZATION_REPEAT_ID` 选择一个重复；主方法和4个主表 baseline 对同一 Prompt 使用相同生成 seed、相同模型 revision、相同 latent shape 以及由 `sha256_counter_normal_icdf_table20_float32` 生成的同一个基础 latent。高斯用途把 SHA-256 大端计数器块解释为 MSB-first 连续比特流, 跨块连续提取20位索引并查询 $q_i=\operatorname{round}_{\mathrm{binary32}}\!\left(\Phi^{-1}((i+0.5)/2^{20})\right)$。规范 float32 生成和目标 dtype 转换都在 CPU 完成, 随后才搬运到执行设备。实际目标 dtype Tensor 的内容摘要与身份摘要必须进入 observation, 配对优势门禁逐字段核验一致性。生成 seed 和密钥重复可以改变运行实例, 但不得改变冻结方法参数、攻击协议或 fixed-FPR 统计定义。

当前统一方法参数位于 `configs/model_sd35.yaml`。正式数据集入口通过 `experiments/protocol/method_runtime_config.py` 直接解析该文件, 目标根目录缺少该文件时直接失败, 不从 Python 包目录回退。完整配置值产生稳定的 `formal_method_config_digest`, 并进入三级论文配置和每个科学运行配置摘要。该 YAML 同时精确冻结 SD3 pipeline、VAE、Transformer 和 FlowMatch scheduler 的完整类名, VAE `scaling_factor=1.5305`、`shift_factor=0.0609`、latent `float16` 与视觉编码 `float32`；模型加载后必须复验实际对象和参数 dtype, 不能只验证传给 `from_pretrained` 的参数。环境变量只用于设备、数据访问凭据、输出位置和单次 session 调度等运行控制, 已登记变量若声明不同的模型或算子身份会在下载前失败。

仅图像检测固定使用 `public_detection_schedule_index=7`, 该位置严格等于首次注入索引6之后的下一 scheduler 索引。检测条件由 SD3 三路空文本且关闭 classifier-free guidance 的协议构造。公开检测噪声使用 `public_detection_noise_prg_protocol=sha256_counter_normal_icdf_table20_float32` 和固定公开 domain, 在 CPU 上查询冻结 Q20 中点逆 CDF 表生成规范 float32 Tensor, 在 CPU 转换到 latent dtype 后才搬运到目标设备；生成 Prompt、生成 seed、样本序号和设备 RNG 均不参与该噪声身份。

正式 Jacobian 直接使用512维归一化 CLIP embedding 与204维手工结构统计向量。204维向量固定为 RGB 通道均值/标准差、水平/垂直绝对梯度均值和8x8 RGB 平均池化, 不单独表示一般感知质量。分支风险中的局部对比度固定为灰度相对反射填充5x5局部均值的绝对偏离；相邻步稳定度直接比较当前与紧邻上一 scheduler 步的解码 RGB, 因而注入时刻必须同时保留真实前后调度时刻。`jacobian_candidate_count=20` 定义可用于形成基底的密钥种子池, `null_space_rank=4` 定义必须得到的独立基底秩。每个种子通过无阻尼 PSD-CG 执行风险支持完整 Jacobian 约束投影; `null_space_cg_max_iterations=64`、`null_space_cg_relative_tolerance=0.000001`、`maximum_relative_response_residual=0.0001` 和 `minimum_projection_energy_retention=0.01` 共同构成 fail-closed 门禁。只有 CG 收敛、QR 后每列完整 Jacobian 残差、正交性和能量全部通过时, 运行记录才能声明 Null Space。三个分支先在 float32 中按冻结顺序合成并与 original latent 的 float32 值相加, 最后只 cast 一次到真实 latent dtype；运行时从该实际写回值恢复 `written_latent - latent`, 并重新执行完整特征精确 JVP。`maximum_quantized_write_relative_jacobian_response=0.0001` 约束该响应范数相对当前完整特征范数的比例。`keyed_prg_version=sha256_counter_normal_icdf_table20_float32` 冻结共享的 SHA-256 大端计数器协议。内容模板、Jacobian 候选方向和公开检测噪声从连续计数器比特流提取20位索引并查询冻结 Q20 中点逆 CDF float32 表；表的完整大端字节 SHA-256 为 `70abf440a7f3670147965ffa52f5aaa639dab97f6282b68f3a9a1b1ce5e6cf5a`。该输出是有限离散的 Q20 量化标准正态, 不是连续精确的 $\mathcal N(0,1)$；理想中点 KS 距离为 $2^{-21}$, 含已登记 float32 舍入误差的上界为 `4.912236096776823e-7`。独立的53位开区间 uniform 路径只用于注意力关系符号, 两类输出角色不可互换。规范 float32 生成和目标 dtype 转换均在 CPU 完成, 随后才搬运到目标设备；CPU/CUDA 设备 RNG 不参与方法身份。MPFR 逐项复验是外层审计证据, 不进入 `keyed_prg_protocol_digest`。当前逐字节固定向量只在 Windows CPU 实测, Linux/Colab 一致性由 GPU 运行前门禁复验。尾部模板按绝对幅值和展平索引稳定排序，精确保留 `ceil(element_count * tail_fraction)` 个元素。注意力分支以 `attention_stable_token_fraction=0.50` 和 `attention_unstable_pair_weight=0.25` 共享 pair 权重构造规则。`attention_module_names` 精确固定为 `transformer_blocks.0.attn` 与 `transformer_blocks.23.attn`, `max_attention_tokens=64`；运行时按名称直接解析并拒绝缺失层, 不按模块枚举位置选择。`attention_coordinate_convention=normalized_xy_token_centers_corner_endpoints` 把角点 token 中心映射到 -1 与 1, `attention_grid_align_corners=true` 使 token 插值、关系稳定图和图像仿射重采样使用同一端点语义。检测端对两个冻结层先分别完成层内配准, 再依次按注册目标、观测关系分、注册置信度执行字典序最大化；完全同分时选择冻结顺序中更靠前的层。aligned 图像固定使用 bilinear、`padding_mode=border`、`align_corners=true`, 并按 `floor(clamp(x, 0, 1) * 255)` 转回 RGB uint8。数据依赖的选择身份不跨生成端与检测端比较：一次注入内部冻结一个身份, 一次仅图像盲检在 raw、registration 与 aligned 路径内部冻结另一个身份。`minimum_semantic_preservation_cosine` 与 `maximum_handcrafted_structure_feature_relative_drift` 同时约束 clean 到完整方法、clean 到 carrier-only 及 carrier-only 到完整方法的最终成图保持性, 防止内容漂移伪造 attention 差异。`minimum_final_image_attention_score_gain=0.0001` 要求完整方法相对同 seed、同 scheduler、同 LF/tail 配置与算子且只关闭 attention geometry 的 carrier-only 反事实, 在自身盲选择分数和冻结 carrier-only pair 权重分数上都取得严格超过0.0001的真实 Q/K 增益。该对照包含 attention 开关对后续 LF、tail 与轨迹的交互效应, 不冻结两侧后续 realized update。缺失首 latent 字节身份、完整调度、无 attention 原子 JSONL、CUDA 证据、三边最终内容保持证据或产物身份一致性时正式运行失败。

仅图像注意力配准使用 `formal_method_config_schema=slm_wm_formal_method_runtime_config` 中的唯一结构门禁: `attention_anchor_count=12`、`attention_residual_threshold=0.20`、`attention_minimum_inlier_ratio=0.50`。锚点按抽样 token 索引确定性均匀选择；token 数少于12时失败。残差采用上述归一化 xy 坐标中的欧氏距离, 内点率只以具有有效双线性覆盖的锚点为分母并要求唯一观测匹配。三项值为预注册方法常量, calibration/test 不得调整。仅图像测量配置固定为 `slm_wm_image_only_measurement_config`, 并通过 `slm_wm_image_only_extraction_profile` 绑定模型 revision、VAE 编码、图像预处理、公开噪声与条件、Q/K 层和 token 坐标协议。完整配置正文只保存在顶层运行 manifest, 样本记录保存测量配置摘要及决策必需原子。calibration clean negatives 按版本化 Prompt 散列确定性拆分为互斥的1/3 `window_fit` 与2/3 `threshold_freeze`: 前者只拟合几何门和 rescue 窗口, 后者只使用判定等价连续分数冻结最终 fixed-FPR 阈值。calibration 与 test 必须绑定同一测量配置摘要和冻结协议摘要。

三分支风险常量由 `lf_content_risk_config`、`tail_robust_risk_config` 和 `attention_geometry_risk_config` 唯一冻结。三个分支共同使用严格小于0.55的资格边界、0.05/1.0预算下界与上界和0.70预算增益；LF 权重依次为局部对比0.30、语义0.30、纹理0.20、相邻步不稳定0.20并回避纹理，尾部载体对应0.25、0.25、0.30、0.20并偏好纹理，注意力分支对应0.20、0.25、0.05、0.20、Q/K 不稳定0.30，neutral texture 风险项固定为0.5。风险图采用解析范围和 bilinear 插值：图像信号使用 `align_corners=false`，Q/K stability 使用 `align_corners=true`。每个样本的 HxW 预算只沿自身通道维重复，不跨 batch 混合；零预算坐标上的安全方向必须精确为0，否则运行失败。`RiskBoundedScale` 使用冻结预算上界、方向峰值和 `risk_bounded_scale_direction_epsilon=1e-12` 构造逐坐标硬包络，不允许以单样本预算最大值重新定标。

Null Space 数值协议固定 `null_space_numerical_epsilon=1e-12`、`maximum_qr_condition_number=1e6` 和 `maximum_orthogonality_error=1e-5`。QR 后逐列参考通过右侧上三角求解实现 $VR^{-1}$ 的作用，不显式构造 `R` 的逆矩阵。LF 载体固定使用5x5二维平均核、stride 1、padding 2、零边界、`ceil_mode=false`、`count_include_pad=true` 和 `divisor_override=null`, 随后执行全 Tensor 去均值和 L2 归一化；完整检测分数固定使用 LF 权重0.70与尾部权重0.30。尾部模板完成幅值截断后只执行 L2 归一化, 非入选位置必须保持精确0。注意力回溯和量化包络回溯的缩减因子均为0.5；注意力最多缩减8次，量化联合包络最多缩减24次且绝对超限容差为0。三分支先按 `lf_content`、`tail_robust`、`attention_geometry` 顺序在 float32 中求和，再与 original latent 的 float32 值相加，最后只执行一次到 latent dtype 的 cast；联合包络精确等于当前活动分支包络之和。

内容载体证据同时绑定 LF 与尾部协议摘要、raw/aligned 模板摘要、尾部 shape 与精确保留计数以及 fixed-FPR 阈值摘要。检测密钥计划分为 `registered_watermark_key` 和 `registered_wrong_key_negative`；wrong-key 由版本化 SHA-256 domain separation 从当前注册密钥确定性派生。总科学内容记录要求注入密钥摘要等于检测计划中的注册密钥摘要, 注册密钥模板与嵌入模板相同, wrong-key 模板唯一且不同。

上述字段冻结方法定义, 不表示相应运行算子已经通过 CPU 或 GPU 验证。实现状态只能由独立性质测试与受治理构建状态报告给出。

`configs/method_semantic_registry.json` 登记核心方法不变量到实现符号、运行证据字段和验证职责的精确映射。该文件不得保存自行赋值的通过结论；`method_definition_digest` 只绑定当前可执行定义身份, 不证明实现已经符合权威公式。

Tensor 内容摘要协议固定为 `slm_wm_tensor_content`, 由核心方法代码定义且不提供运行时改写配置。该协议同时散列版本、PyTorch dtype、有序 shape 和连续原始字节；正式记录必须用它绑定三个风险场、三个 Null Space 求解过程、三个分支更新及全部真实 Q/K 原子。仅修改摘要字符串或汇总数值不能通过联合内容摘要复验。

`attention_relation_component_weights=[0.25,0.25,0.25,0.25]` 是完整方法唯一配置。正式四分量留一消融不改写该 YAML, 而由受治理消融规范分别使用 `(0,1/3,1/3,1/3)`、`(1/3,0,1/3,1/3)`、`(1/3,1/3,0,1/3)` 和 `(1/3,1/3,1/3,0)`。嵌入目标、单调回溯、最终成图归因、盲检原分数、仿射注册和对齐后同步分数必须共同消费同一权重协议摘要。

模型与外部数据资源统一登记在 `configs/model_source_registry.json`。每条 Hugging Face 资源记录同时给出仓库标识、资源类型和40位小写十六进制提交 revision。主方法的 SD3.5 Medium 与 CLIP 图像编码器必须使用 `configs/model_sd35.yaml` 中的精确 revision, 不允许使用 `main`、分支名或短提交。该约束用于固定实际下载内容, 与固定本项目 Git 代码版本共同构成可重建运行输入。

对于单一权重文件即可确定运行输入的资源, `required_files` 进一步登记相对路径、字节大小和 SHA-256。official-reference 的 `ViT-g-14` OpenCLIP checkpoint 采用该约束; 共享缓存中的额外文件、符号链接、大小漂移或摘要漂移都会阻断正式命令。

## 隔离依赖 profile

正式运行路径使用 `dependency_profile_registry.json` 登记一个通用 CPU 父编排环境与五个相互隔离的 CUDA 科学执行环境。每条记录固定 CPython patch 版本和 Linux 机器架构; 科学 profile 还固定 CUDA、PyTorch wheel index 以及匹配的 `torch` / `torchvision` local version。一次正式 session 必须先资格化父编排环境, 再准备当前 workflow 唯一选择的一个科学子环境。

| profile | Python | accelerator | torch / torchvision | 运行职责 |
| --- | --- | --- | --- | --- |
| `workflow_orchestrator` | `3.12.13` | `cpu` | 不适用 | 父进程编排、固定模型快照下载、NumPy 配对统计、CPU 结果闭合和隔离子环境调度。 |
| `sd35_method_runtime_gpu` | `3.12.13` | `12.8` | `2.11.0+cu128` / `0.26.0+cu128` | SLM-WM SD3.5 主方法、攻击与正式质量指标。 |
| `t2smark_sd35_gpu` | `3.12.13` | `12.4` | `2.5.0+cu124` / `0.20.0+cu124` | T2SMark SD3.5 官方入口。 |
| `tree_ring_official_py39_cu117` | `3.9.19` | `11.7` | `1.13.0+cu117` / `0.14.0+cu117` | Tree-Ring 官方原始机制参考。 |
| `gaussian_shading_official_py38_cu117` | `3.8.20` | `11.7` | `1.13.0+cu117` / `0.14.0+cu117` | Gaussian Shading 官方原始机制参考。 |
| `shallow_diffuse_official_py39_cu117` | `3.9.19` | `11.7` | `1.13.0+cu117` / `0.14.0+cu117` | Shallow Diffuse 官方原始机制参考。 |

`dependency_profiles/*_direct.txt` 是已经提交的精确直接依赖输入。当前六个文件共111个条目; 每个有效条目只能采用 `name==version` 形式, 不允许版本范围、可漂移标签、安装升级选项、远程 URL 或未版本化包。五个 CUDA profile 的 `torch` 和 `torchvision` 必须与登记的 CUDA local version 和 PyTorch index 完全一致。父编排 profile 不登记 `torch`、`torchvision`、CUDA 或 PyTorch index, 但固定 `uv==0.11.28`、`huggingface_hub==1.20.1` 与 `numpy==2.0.2`; 前两者用于科学子解释器创建和固定 revision 模型快照下载, NumPy 用于 CPU 侧 Prompt 聚类 bootstrap、质量匹配统计和结果语义重建。五个科学执行 profile 均固定 `pip==24.3.1`、`setuptools==75.3.0` 和 `wheel==0.45.1`。

直接输入不等于完整 Python wheel 闭包。每个 profile 必须在登记的 Linux x86_64 与完整 CPython patch 中解析并生成 `dependency_profiles/<profile>_lock.txt`; 五个科学 profile 还必须向各自登记的 PyTorch wheel index 解析固定 `torch` / `torchvision` identity。候选生成使用 `pip --dry-run` 且不导入 torch 或执行 CUDA, 因而可以在无 GPU Linux x86_64 host 完成。锁中的直接依赖和传递依赖都必须固定版本并携带实际候选 wheel 的 SHA-256。完整锁需要从实际 resolver report 重建、人工审查、通过回传接收器复验并提交, 不允许人工编造 wheel 摘要。只有锁文件存在、格式有效、逐项带 SHA-256 且覆盖全部直接输入时, 对应 profile 才能达到 `formal_ready=True`。

当前六个 profile 均登记了完整哈希锁并通过仓库静态锁门禁。该状态不表示目标 Linux wheel 已在本机安装, 也不表示任何 CUDA 科学运行已经完成。本地 Windows / CPU 环境只复验锁的结构、摘要和直接依赖覆盖; 正式环境资格仍由后续 Colab 会话中的真实安装、`pip check`、解释器身份、torch/CUDA identity 与对应科学 smoke 决定。

`dependency_profiles/dependency_qualification_uv_linux_x86_64_lock.txt` 只服务于 fresh host 的资格化工具引导。该文件同时固定 `uv==0.11.28` 的 PyPI Linux manylinux x86_64 wheel URL、平台文件名和 SHA-256, 不属于六个运行 profile 的完整依赖锁, 也不支持论文 claim。host launcher 固定由 `python -I` 启动, 仅使用宿主 Python 标准库下载、复验和提取该 wheel, 不调用宿主 `venv`、`pip` 或 `ensurepip`; 随后才由固定 `uv` 创建精确 orchestrator CPython。正式运行仍只能消费六个 registry profile 各自已提交的完整哈希锁。

`workflow_orchestrator` inspection 只核验 CPU 平台与完整锁包集合, 不导入 torch, 也不执行 CUDA 门禁。`experiments.runtime.dependency_preparation` 对五个科学 profile 额外执行 `sys.executable -m pip check`; 父编排 profile 仍核验全部锁包, 但将该兼容性命令标记为不适用。`experiments.runtime.isolated_dependency_environment` 使用固定 `uv==0.11.28` 内置的冻结可下载 Python distribution 列表和完整 CPython patch 创建科学子环境, 并要求实际 `uv` executable 同时通过当前解释器 distribution `RECORD` 的路径与 SHA-256 核验, 防止 PATH 中同版本伪造文件。详细证据契约见 `docs/builds/formal_dependency_environment.md`。

五个 CUDA profile 均由 repository 隔离执行路径选择并在各自子解释器中运行, Notebook 父解释器不直接导入科学实现。锁资格化只验证目标 CPython、Linux x86_64、直接与传递 wheel 闭包以及 PyTorch index 身份, 可以在无 GPU host 完成; 正式运行必须在 Colab GPU 中安装已提交锁并通过 `pip check`、torch/CUDA identity、CUDA 可用性与对应科学 smoke。任一缺失锁 profile 都保持 fail-closed, 不形成正式 GPU 结果。

## 方法配置语义

正式运行使用三个分支标识:

| 分支标识 | 数学角色 | 配置含义 |
| --- | --- | --- |
| `lf_content` | 空间低通 LF 主证据 | 密钥高斯模板经空间平均池化形成低通模板, 再投影到 LF 安全子空间。 |
| `tail_robust` | 高斯幅值尾部截断补充证据 | 按元素绝对幅值和展平索引稳定排序, 精确保留冻结比例的高斯分布尾部, 不定义空间频带。 |
| `attention_geometry` | Q/K 相对关系几何锚点 | 使用真实 attention 目标梯度、子空间投影与单调回溯更新。 |

`tail_fraction` 只定义高斯模板的幅值尾部保留比例。它不是频率截止值, 也不能解释为空间频带比例。正式运行记录统一使用 `tail_robust`、`tail_score` 和 `lambda_tail` 语义。

## prompt 划分口径

当前三类运行层级共用 3:33:34 的固定划分比例, 即 dev、calibration、test
分别约占 4.29%、47.14% 和 48.57%。对应数量为 3/33/34、30/330/340 和
300/3300/3400。该比例确保 calibration negative 足以冻结阈值, 同时让 test
集合在零误报时满足目标 FPR 的单侧 95% 二项分布上界。
