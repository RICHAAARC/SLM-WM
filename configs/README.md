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

当前统一方法参数位于 `configs/model_sd35.yaml`。正式数据集入口通过 `experiments/protocol/method_runtime_config.py` 直接解析该文件, 并拒绝环境变量改变已冻结的模型身份或方法超参数。环境变量只用于设备、数据访问凭据、输出位置和单次 session 调度等运行控制。`jacobian_candidate_count=20`、`null_space_rank=4`、`maximum_relative_response_residual=0.0001` 和 `minimum_projection_energy_retention=0.01` 共同约束语义条件低响应子空间: 前两项定义候选与保留秩, 后两项阻止高响应方向或近零盲检投影进入正式记录。

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

直接输入不等于完整 Python wheel 闭包。每个 profile 必须在登记的 Linux x86_64 目标环境中解析并生成 `dependency_profiles/<profile>_lock.txt`: 父编排 profile 使用 CPU 环境, 五个科学 profile 使用匹配的 GPU 环境。目标环境可以位于 Colab 或满足同一 profile 的普通 Linux 服务器。锁中的直接依赖和传递依赖都必须固定版本并携带真实 wheel 的 SHA-256。完整锁需要从实际解析结果回填并提交, 不允许人工编造 wheel 摘要。只有锁文件存在、格式有效、逐项带 SHA-256 且覆盖全部直接输入时, 对应 profile 才能达到 `formal_ready=True`。

`workflow_orchestrator` inspection 只核验 CPU 平台与完整锁包集合, 不导入 torch, 也不执行 CUDA 门禁。`experiments.runtime.dependency_preparation` 对五个科学 profile 额外执行 `sys.executable -m pip check`; 父编排 profile 仍核验全部锁包, 但将该兼容性命令标记为不适用。`experiments.runtime.isolated_dependency_environment` 使用固定 `uv==0.11.28` 内置的冻结可下载 Python distribution 列表和完整 CPython patch 创建科学子环境, 并要求实际 `uv` executable 同时通过当前解释器 distribution `RECORD` 的路径与 SHA-256 核验, 防止 PATH 中同版本伪造文件。详细证据契约见 `docs/builds/formal_dependency_environment.md`。

五个 CUDA profile 均由 repository 隔离执行路径选择并在各自子解释器中运行, Notebook 父解释器不直接导入科学实现. 当前唯一外部阻断类别是完整哈希锁资格审查: CPU 父锁需要在匹配的 Linux x86_64 环境完成审查, 五个科学锁需要在匹配各自 CPython、CUDA 与 PyTorch index 的 Colab 或 Linux CUDA 环境完成审查并提交. 在此之前全部 profile 保持 fail-closed, 不形成正式 GPU 结果.

## 方法配置语义

正式运行使用三个分支标识:

| 分支标识 | 数学角色 | 配置含义 |
| --- | --- | --- |
| `lf_content` | 空间低通 LF 主证据 | 密钥高斯模板经空间平均池化形成低通模板, 再投影到 LF 安全子空间。 |
| `tail_robust` | 高斯幅值尾部截断补充证据 | 按元素绝对幅值分位点保留高斯分布尾部, 不定义空间频带。 |
| `attention_geometry` | Q/K 相对关系几何锚点 | 使用真实 attention 目标梯度、子空间投影与单调回溯更新。 |

`tail_fraction` 只定义高斯模板的幅值尾部保留比例。它不是频率截止值, 也不能解释为空间频带比例。正式运行记录统一使用 `tail_robust`、`tail_score` 和 `lambda_tail` 语义。

## prompt 划分口径

当前三类运行层级共用 3:33:34 的固定划分比例, 即 dev、calibration、test
分别约占 4.29%、47.14% 和 48.57%。对应数量为 3/33/34、30/330/340 和
300/3300/3400。该比例确保 calibration negative 足以冻结阈值, 同时让 test
集合在零误报时满足目标 FPR 的单侧 95% 二项分布上界。
