# Configs

此目录保存论文运行配置、prompt 配置和运行环境约束记录。

## 论文 prompt 配置

| 配置文件 | prompt 数量 | 目标 FPR | 支持主张 | 用途 |
| --- | ---: | ---: | --- | --- |
| `paper_main_probe_paper_prompts.txt` | 70 | 0.1 | `probe_claim` | 小规模正式结果包, test 包含34个独立 Prompt。 |
| `paper_main_pilot_paper_prompts.txt` | 700 | 0.01 | `pilot_claim` | 中等规模正式结果包, test 包含340个独立 Prompt。 |
| `paper_main_full_paper_prompts.txt` | 7000 | 0.001 | `full_claim` | 全规模正式结果包, test 包含3400个独立 Prompt。 |

Prompt bank 中受外部来源登记约束的子集来自 Google Research PartiPrompts。固定来源版本、许可证、文件摘要和分层选择摘要见 `configs/prompt_source_registry.json`。

三组配置只允许样本数量和 fixed-FPR 标准不同; 方法参数、攻击协议、baseline 入口、Wilson 单侧 FPR 上界、Hoeffding 结果区间、随机种子和结果闭合逻辑必须保持一致。共同协议结果记录必须拒绝 proxy、placeholder、fallback、synthetic 和 formal-null 证据。

当前统一方法参数位于 `configs/model_sd35.yaml`。正式数据集入口通过 `experiments/protocol/method_runtime_config.py` 直接解析该文件, 并拒绝环境变量改变已冻结的模型身份或方法超参数。环境变量只用于设备、数据访问凭据、输出位置和单次 session 调度等运行控制。正式 Jacobian 直接使用512维归一化 CLIP embedding 与204维手工结构统计向量。204维向量固定为 RGB 通道均值/标准差、水平/垂直绝对梯度均值和8x8 RGB 平均池化, 不单独表示一般感知质量。分支风险中的局部对比度固定为灰度相对反射填充5x5局部均值的绝对偏离；相邻步稳定度直接比较当前与紧邻上一 scheduler 步的解码 RGB, 因而注入时刻必须同时保留真实前后调度时刻。`jacobian_candidate_count=20` 定义可用于形成基底的密钥种子池, `null_space_rank=4` 定义必须得到的独立基底秩。每个种子通过无阻尼 PSD-CG 执行风险支持完整 Jacobian 约束投影; `null_space_cg_max_iterations=64`、`null_space_cg_relative_tolerance=0.000001`、`maximum_relative_response_residual=0.0001` 和 `minimum_projection_energy_retention=0.01` 共同构成 fail-closed 门禁。只有 CG 收敛、QR 后每列完整 Jacobian 残差、正交性和能量全部通过时, 运行记录才能声明 Null Space。三个分支合成后, 运行时按真实 latent dtype 完成加法, 从实际写回值恢复 `written_latent - latent`, 并重新执行完整特征精确 JVP；`maximum_quantized_write_relative_jacobian_response=0.0001` 约束该响应范数相对当前完整特征范数的比例。`keyed_prg_version=sha256_counter_box_muller_float32_v1` 冻结内容模板、Jacobian 候选方向和注意力关系符号共享的 SHA-256 计数器流、53位均匀映射、Box-Muller 变换和 CPU float32 规范化路径；CPU/CUDA 设备 RNG 不参与方法身份。尾部模板按绝对幅值和展平索引稳定排序，精确保留 `ceil(element_count * tail_fraction)` 个元素。注意力分支以 `attention_stable_token_fraction=0.50` 和 `attention_unstable_pair_weight=0.25` 共享 pair 权重构造规则。数据依赖的选择身份不跨生成端与检测端比较：一次注入内部冻结一个身份, 一次仅图像盲检在 raw、registration 与 aligned 路径内部冻结另一个身份。`minimum_semantic_preservation_cosine` 与 `maximum_handcrafted_structure_feature_relative_drift` 同时约束 clean 到完整方法、clean 到 carrier-only 及 carrier-only 到完整方法的最终成图保持性, 防止内容漂移伪造 attention 差异。`minimum_final_image_attention_score_gain=0.0001` 要求完整方法相对同 seed、同 scheduler、同 LF/tail 配置和算子且仅关闭 attention geometry 的 carrier-only 反事实, 在自身盲选择分数和冻结 carrier-only pair 权重分数上都取得严格超过0.0001的真实 Q/K 增益。该对照包含 attention 开关对后续 LF、tail 与轨迹的交互效应, 不冻结两侧后续 realized update。缺失首 latent 字节身份、完整调度、无 attention 原子 JSONL、CUDA 证据、三边最终内容保持证据或产物身份一致性时正式运行失败。

模型与外部数据资源统一登记在 `configs/model_source_registry.json`。每条 Hugging Face 资源记录同时给出仓库标识、资源类型和40位小写十六进制提交 revision。主方法的 SD3.5 Medium 与 CLIP 图像编码器必须使用 `configs/model_sd35.yaml` 中的精确 revision, 不允许使用 `main`、分支名或短提交。该约束用于固定实际下载内容, 与固定本项目 Git 代码版本共同构成可重建运行输入。

对于单一权重文件即可确定运行输入的资源, `required_files` 进一步登记相对路径、字节大小和 SHA-256。official-reference 的 `ViT-g-14` OpenCLIP checkpoint 采用该约束; 共享缓存中的额外文件、符号链接、大小漂移或摘要漂移都会阻断正式命令。

## 隔离依赖 profile

正式运行路径使用 `dependency_profile_registry.json` 登记一个通用 CPU 父编排环境与五个相互隔离的 CUDA 科学执行环境。每条记录固定 CPython patch 版本和 Linux 机器架构; 科学 profile 还固定 CUDA、PyTorch wheel index 以及匹配的 `torch` / `torchvision` local version。一次正式 session 必须先资格化父编排环境, 再准备当前 workflow 唯一选择的一个科学子环境。

| profile | Python | accelerator | torch / torchvision | 运行职责 |
| --- | --- | --- | --- | --- |
| `workflow_orchestrator` | `3.12.13` | `cpu` | 不适用 | 父进程编排、固定模型快照下载、CPU 结果闭合和隔离子环境调度。 |
| `sd35_method_runtime_gpu` | `3.12.13` | `12.8` | `2.11.0+cu128` / `0.26.0+cu128` | SLM-WM SD3.5 主方法、攻击与正式质量指标。 |
| `t2smark_sd35_gpu` | `3.12.13` | `12.4` | `2.5.0+cu124` / `0.20.0+cu124` | T2SMark SD3.5 官方入口。 |
| `tree_ring_official_py39_cu117` | `3.9.19` | `11.7` | `1.13.0+cu117` / `0.14.0+cu117` | Tree-Ring 官方原始机制参考。 |
| `gaussian_shading_official_py38_cu117` | `3.8.20` | `11.7` | `1.13.0+cu117` / `0.14.0+cu117` | Gaussian Shading 官方原始机制参考。 |
| `shallow_diffuse_official_py39_cu117` | `3.9.19` | `11.7` | `1.13.0+cu117` / `0.14.0+cu117` | Shallow Diffuse 官方原始机制参考。 |

`dependency_profiles/*_direct.txt` 是已经提交的精确直接依赖输入。当前六个文件共111个条目; 每个有效条目只能采用 `name==version` 形式, 不允许版本范围、可漂移标签、安装升级选项、远程 URL 或未版本化包。五个 CUDA profile 的 `torch` 和 `torchvision` 必须与登记的 CUDA local version 和 PyTorch index 完全一致。父编排 profile 不登记 `torch`、`torchvision`、CUDA 或 PyTorch index, 但固定 `uv==0.11.28` 与 `huggingface_hub==1.20.1`, 用于科学子解释器创建和固定 revision 模型快照下载。五个科学执行 profile 均固定 `pip==24.3.1`、`setuptools==75.3.0` 和 `wheel==0.45.1`。

直接输入不等于完整 Python wheel 闭包。每个 profile 必须在登记的 Linux x86_64 与完整 CPython patch 中解析并生成 `dependency_profiles/<profile>_lock.txt`; 五个科学 profile 还必须向各自登记的 PyTorch wheel index 解析固定 `torch` / `torchvision` identity。候选生成使用 `pip --dry-run` 且不导入 torch 或执行 CUDA, 因而可以在无 GPU Linux x86_64 host 完成。锁中的直接依赖和传递依赖都必须固定版本并携带实际候选 wheel 的 SHA-256。完整锁需要从实际 resolver report 重建、人工审查、通过回传接收器复验并提交, 不允许人工编造 wheel 摘要。只有锁文件存在、格式有效、逐项带 SHA-256 且覆盖全部直接输入时, 对应 profile 才能达到 `formal_ready=True`。

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
