# SLM-WM

本仓库用于实现“语义条件化的潜空间流形水印-SLM”。项目目标是形成可审计、可重跑、可用于论文投稿材料的方法机制、真实模型运行链路、共同攻击协议、外部 baseline 对比、内部消融和结果图表重建流程。

方法名称中的“潜流形”严格指当前 latent 点处由716维完整特征局部线性化诱导的隐式水平集安全切空间解释。正式算子计算的是分支风险支持的 Jacobian Null Space 数值基底，不验证全局流形定理条件，也不构造全局非线性流形、坐标图、测地线或回缩。三个分支采用投影、强度缩放、Q/K 梯度回溯和实际 dtype 写回复验组成的构造式协议，不声明求解一个联合标量 `argmax`。

## 项目定位

核心方法的权威可证伪不变量定义在 `docs/builds/method_semantic_invariants.md`, `configs/method_semantic_registry.json` 只登记公式到实现、CPU 性质和 GPU 原子的追踪关系。登记完整或方法定义摘要一致不表示科学性质已经通过；当前能力必须以受治理构建状态记录为准。

SLM-WM 的核心思路是在扩散模型潜空间中构造三个受语义条件约束的互补分支: 空间低通 LF 主证据、高斯幅值尾部截断鲁棒补充证据和真实 Q/K Self-Attention 相对关系几何锚点。三个分支分别使用分支风险场, 并通过716维完整特征 Jacobian 的精确 JVP/VJP 与无阻尼 PSD-CG 约束投影求解 rank-4 Null Space。716维特征由512维归一化 CLIP embedding 与204维明确限定的 RGB 通道统计、梯度和8x8池化手工结构向量连接而成；后者不单独代表一般感知质量。注意力算子直接从冻结二维抽样图像 token 的真实 Q/K 构造中心化 logit、可微 rank、关系概率和距离调制中心化概率四分量图；第4分量是概率偏离与公开距离偏离的双中心交互, 均匀 attention 时严格为0。各分量逐行归一化后执行密钥投影, 再按冻结的非负归一化分量权重组合；完整方法使用四项0.25, 四个留一变体分别把一项置零并令其余三项各取 $1/3$。注意力嵌入与仅图像盲检共享四分量算子和稳定 token pair 权重构造规则, 但分别由各自可见的真实 Q/K 数据产生身份；一次注入内部冻结一个身份, 一次盲检则在 raw、registration 和 aligned 路径中冻结另一个身份。几何恢复使用与攻击配置无关的分层搜索。最终成图必须在 CUDA 上比较同 seed、同 scheduler、同 LF/tail 配置与算子且只关闭 attention geometry 的 carrier-only 反事实与完整方法。首个注入前 latent 由 dtype、shape 和全部原始字节 SHA-256 证明相同；carrier-only 的逐注入原子必须证明没有 attention 分数、更新、关系、pair 身份或 attention Null Space。该对照估计 attention 开关经后续 LF、tail 和生成轨迹交互传播后的总机制效应, 不假设两侧 realized carrier update 完全相等, 也不解释为纯直接效应。clean、carrier-only 与完整方法三对最终成图必须同时通过完整 CLIP 语义与手工结构统计保持门禁, 随后完整方法还必须通过自身盲选择和冻结 carrier-only pair 权重的双归因增益门禁。检测侧只读取待检图像、密钥和公开模型配置, 在包含几何救回的完整 fixed-FPR 协议下冻结阈值。正式 Q/K 算子精确绑定 `transformer_blocks.0.attn` 与 `transformer_blocks.23.attn`；token 坐标采用角点 token 中心分别落在 -1 与 1 的归一化 xy 网格, 图像仿射重采样统一使用 `align_corners=True`。

三个分支的风险预算必须同时进入 Null Space 和最终幅值构造。单位安全方向只能沿原方向执行满足逐位置预算包络的最大可行标量缩放；禁止在风险投影后无条件恢复固定二范数强度。三分支合成后的实际 dtype 增量还必须重新通过联合预算包络与完整特征 JVP。

每次注入以版本化 Tensor 内容协议绑定三个分支的完整风险、预算和资格 mask, Jacobian 候选矩阵、风险路由候选及其响应、投影方向及其响应、QR 基底及其响应、逐列 QR 参考响应, 三个实际分支更新以及真正量化写回增量。注意力链同时绑定各冻结层的抽样 Q、K、中心化 logit、关系概率和二维 token 索引, 并分别保存原 latent、内容基底、接受候选、实际写回、三张最终成图及盲检原图/对齐图的有序原子摘要。摘要包含 dtype、shape 和连续原始字节, 因而可区分数值、精度或形状漂移。

高斯幅值尾部截断分支的正式标识为 `tail_robust`。该分支按高斯模板元素的绝对幅值选择分布尾部, 不执行 FFT、DCT、带通滤波或空间频带选择, 因而不具有空间频率含义。

正式生成模型固定为 `stabilityai/stable-diffusion-3.5-medium@b940f670f0eda2d07fbb75229e779da1ad11eb80`, 语义条件编码器固定为 `openai/clip-vit-base-patch32@3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`。主方法、common-backbone baseline 和 T2SMark 的正式 loader 必须显式传入对应 revision。模型与数据来源登记在 `configs/model_source_registry.json`, 方法参数由运行时直接解析 `configs/model_sd35.yaml`; 分支名、`main`、短提交和改变正式方法参数的环境变量均不是正式运行输入。

正式公平比较采用3个生成种子与3个水印密钥组成的9个交叉重复。每个 Prompt 在同一重复中为 SLM-WM、Tree-Ring、Gaussian Shading、Shallow Diffuse 和 T2SMark 分配相同的公开生成 seed, 并由版本化 SHA-256 计数器流和 Box-Muller 变换构造同一个基础 latent Tensor。该 Tensor 在转换到目标 dtype 后以实际字节摘要绑定到全部方法记录；任一方法的 seed、密钥重复索引、基础 latent 内容摘要或身份摘要不同, 该样本不得进入配对统计。各水印方法继续使用自身编码和检测算子, 共同协议只控制生成随机性、重复身份和公平预算。`SLM_WM_RANDOMIZATION_REPEAT_ID` 每次只选择一个登记重复, Drive 输出按重复隔离；论文正式汇总必须覆盖全部9个重复, 单个重复不能替代多种子、多密钥证据。

正式 Prompt bank 由固定字节身份的 Microsoft COCO 2017 train captions 与 Google Research PartiPrompts 构造。选择协议先为每张 COCO 图像保留一条合格 caption, 再按来源记录 SHA-256 排序选择6000条 COCO caption 与1000条 PartiPrompt, 以 6:1 顺序交织。`probe_paper`、`pilot_paper` 与 `full_paper` 分别使用同一清单的前70、前700和前7000条。Prompt ID 不包含运行层级名称, split 在连续70条固定块内按风险类别分层分配3/33/34, 因而相同前缀的 Prompt ID 和 split 在三级运行中保持不变。`configs/prompt_selection_manifest.jsonl` 绑定每条 Prompt 的上游记录和原始 UTF-8 文本, 正式入口会联合 `configs/prompt_source_registry.json` 执行逐字节重建门禁。完整协议与复验命令见 `docs/builds/prompt_dataset_provenance.md`。

Tree-Ring、Gaussian Shading 与 Shallow Diffuse 的 official-reference 使用登记的 Stable Diffusion 2.1 镜像快照和本地 `ViT-g-14` OpenCLIP checkpoint。正式运行会核验 Git 源码身份、Prompt 数据 revision、扩散模型逐文件快照、OpenCLIP checkpoint 大小与 SHA-256, 并只接受当前成功命令显式产生的完整科学指标。

## 运行层级

项目使用同一批 repository modules、服务器命令和 Notebook 入口支撑三类论文运行层级。三类运行层级在方法参数、攻击协议、baseline 入口、Wilson 单侧 FPR 上界、Hoeffding 结果区间、随机种子、证据门禁和结果闭合逻辑上必须保持一致; 运行语义只允许 Prompt 数量与目标 FPR 不同。

| 运行层级 | prompt 数量 | 目标 FPR | 支持主张 | Google Drive 结果根目录 | 说明 |
| --- | ---: | ---: | --- | --- | --- |
| `probe_paper` | 70 | 0.1 | `probe_claim` | `/content/drive/MyDrive/SLM/probe_paper_results` | dev/calibration/test 为 3/33/34。 |
| `pilot_paper` | 700 | 0.01 | `pilot_claim` | `/content/drive/MyDrive/SLM/pilot_paper_results` | dev/calibration/test 为 30/330/340。 |
| `full_paper` | 7000 | 0.001 | `full_claim` | `/content/drive/MyDrive/SLM/full_paper_results` | dev/calibration/test 为 300/3300/3400。 |

fixed-FPR clean negative 门禁等于当前层级完整 test split: `probe_paper=34`、`pilot_paper=340`、`full_paper=3400`。该门禁不是独立配置分叉; 切换 `SLM_WM_PAPER_RUN_NAME` 后, Colab 入口和服务器入口自动得到对应值。

正式 FID/KID 的最小图像对数量等于当前运行层级的完整 Prompt 数量, 即 70/700/7000。质量评估不能只抽取前100个样本后代表完整运行层级。

三类正式结果包均拒绝 proxy、placeholder、fallback、synthetic 和 formal-null 证据进入共同协议结果记录。诊断入口可以产生环境检查报告, 但诊断报告不能支持三类论文主张。

单次主方法结果使用 Tensor 原始字节身份与规范 RGB uint8 像素身份形成总科学内容绑定。写入端和完成结果加载器都必须从磁盘重读更新 JSONL、检测 JSONL 和图像后重算风险、Null Space、实际量化 JVP Tensor、三分支更新、注入五角色 Q/K、最终三图与检测 Q/K、公开噪声及 carrier-only 反事实的有序联合身份。最终三图 Q/K 逐角色持久化并复验同一公开噪声 Tensor、PRG 身份及索引0、1、2, 像素-Q/K 绑定也包含噪声身份；检测共享该身份并从3继续。对齐摘要必须是规范 SHA-256。attention 开启时, 写后和打包前都以完整配置或持久化脱敏配置重建 carrier-only 产物, 任一残留 attention 字段都会拒绝。数据集汇总重算每个内嵌绑定记录；单元 manifest 必须自包含、路径无重复且配置与结果完全一致, 数据集 manifest 必须无重复并覆盖全部互斥单元叶子。打包器不接受仅摘要结果、遗漏叶子或被篡改叶子。SHA-256 只能证明同一结果包内的内容一致性, 不能替代模型来源证明、真实 GPU 运行或论文统计结论。

正式机制消融固定为完整方法加14个真实重运行变体。除风险、Null Space、内容载体、完整 attention geometry 和图像对齐对照外, 四个 Q/K 关系分量分别执行一次留一消融：被移除分量权重为0, 其余三个分量各为 $1/3$。每个变体都使用自身配置重新生成、攻击、仅图像检测和 calibration 阈值冻结, 不从完整方法分数推导结果。

在 Colab 中切换运行层级时, 只应修改 Notebook 顶部的入口变量:

```python
SLM_WM_PAPER_RUN_NAME = "pilot_paper"
```

运行层级、prompt 文件、样本数、目标 FPR、fixed-FPR 门禁、持久化目录和常用环境变量由 `scripts/formal_workflow_environment.py` 统一派生。`paper_workflow/colab_utils/paper_run_environment.py` 只增加 Notebook 会话起点记录并向内转发。Notebook 不维护重复配置表。

## 五层代码边界

```text
main/                       论文核心方法与最小数学工具
configs/                    prompt、模型来源与正式依赖 profile 登记
experiments/                SLM 主方法实验协议、主实验攻击、内部消融和服务器 runner
paper_experiments/          完整论文实验、外部 baseline 适配、受治理导入和公平对比
paper_workflow/             Colab Notebook 入口、Drive 包装和 session helper
paper_workflow/notebooks/   Colab Notebook 文件
scripts/                    服务器入口、结果重建、记录生成、检查和打包命令
external_baseline/          外部 baseline 源码缓存、来源登记和项目维护 adapter
docs/                       方法设计、构建流程、字段登记和治理说明
tools/                      可执行治理审计与仓库检查工具
tests/                      分层测试目录
.codex/                     Agent 协作契约与 skill 文件
outputs/                    统一本地持久化输出根目录, 默认不提交
outputs/audit_reports/      harness 审计输出, 默认不提交
```

依赖方向必须保持为 `paper_workflow/ -> scripts/ -> paper_experiments/ -> experiments/ -> main/`。`external_baseline/` 是外部源码缓存与 adapter 边界, 不进入最小方法发布包。

## 最小核心方法发布

`minimal_method_package` 是可以脱离开发仓库验证和安装的 clean detached Git 包。它使用 `docs/core_method_package_readme.md` 生成专用包根 README, 只包含 `main/`、两个方法身份配置、`configs/core_method_dependency_identity.json`、`pyproject.toml`、抽离 manifest 与根目录验证入口。标准 wheel 发现规则只允许 `main` 及其子包。

该最小包声明 `torch>=2.11,<2.12`, 与正式 SD3.5 GPU 依赖锁中的 PyTorch 2.11 系列一致; 具体 CPU 或 CUDA wheel 由安装环境选择。最小包不消费论文实验层的六个完整依赖锁。Diffusers、Transformers、模型权重、CUDA 运行资格、实验 runner、baseline 和论文证据闭合都属于外层运行包。正式抽离要求源工作树 clean; 抽离后应在包根执行 `python -I validate_core_method_package.py --root .`, 复验文件字节、Git 身份、依赖协议、构建元数据与全部核心模块独立导入。抽离通过只证明核心代码发布边界闭合, 不证明真实 GPU 方法运行或论文结果成立。详细契约见 `docs/extraction_profiles.md` 与 `docs/release_boundary.md`。

精确父解释器的受治理子入口固定为 `scripts/formal_workflow_entry.py`。`scripts/run_formal_workflow_host.py`、直接 GPU 服务器命令和 Colab Notebook 均进入该 scripts 层边界；`scripts/` 中不存在对 `paper_workflow/` 的运行时导入或动态执行路径。

## Notebook 与服务器入口

Notebook 只负责挂载 Google Drive、在 `/content` 检出精确提交、选择运行层级、调用 repository scripts 和显示结果路径。正式环境配置、records、thresholds、tables、figures、reports、manifests、checkpoint 发布和结果包镜像逻辑必须位于 `main/`、`experiments/`、`paper_experiments/` 或 `scripts/`。

- Colab 入口说明见 `paper_workflow/notebooks/README.md`。
- 服务器命令入口说明见 `scripts/README.md`。
- 完整论文实验层说明见 `paper_experiments/README.md`。

后续修复 bug 时, 优先修改脚本、协议模块、完整论文实验模块或 Colab helper, 不应把正式逻辑写回 Notebook cell。

## 论文产物治理

1. records 是论文结果事实来源。
2. tables、figures 和 reports 必须可由 records 与 manifests 重建。
3. supported claims 必须绑定到受治理 records、tables、figures、reports 或 manifests。
4. 本地持久化输出必须写入 `outputs/`。
5. Colab 结果包应写入当前运行层级对应的 Google Drive 目录。跨会话恢复只能消费执行锁与 SHA-256 绑定的受治理 checkpoint, 或精确覆盖本次全部请求角色的有效闭合包; 普通下载副本不得作为上游输入。
6. 三类论文运行层级均使用真实图像、真实攻击、真实检测重打分和受治理 baseline 导入; 非正式证据只能用于诊断, 不能进入 claim-ready 统计。

## 推荐运行入口

`probe_paper`、`pilot_paper` 与 `full_paper` 使用同一重跑路径:

1. 在 Colab GPU 中运行 `paper_workflow/notebooks/semantic_watermark_image_only_run.ipynb`。该入口调用 `scripts.semantic_watermark_scientific_workflow`, 后者在一个受验证 `sd35_method_runtime_gpu` 子解释器中执行分支风险、真实 Jacobian Null Space、空间 LF、高斯幅值尾部截断、真实 Q/K 几何嵌入、仅图像检测、完整 fixed-FPR 冻结、常规攻击、再扩散攻击和 torch-fidelity 正式 Inception FID/KID。
2. 在同一主方法 workflow 中按需启用正式消融。每个机制变体对完整 Prompt 集重新生成图像和重跑检测, 并使用自己的 calibration split 冻结阈值。
3. 当前通过各自 Colab Notebook 运行三个 common-backbone method-faithful、T2SMark 和三个 official-reference 原环境复现; 相同 repository runner 也可在具备条件时由 `scripts/run_gpu_server_workflow.py` 脱离 Notebook 调用。Common-backbone 归档必须包含与官方 commit 关键算子的数值忠实度报告, official-reference 结果只进入补充方法忠实度审计。
4. 每个 GPU seed-key 运行通过 `scripts/write_randomization_repeat_evidence_package.py` 将7类活动随机化 leaf ZIP 原字节封装为单 repeat 证据包。精确9个 repeat 全部存在并通过聚合重算后, 才能由 CPU 汇总入口执行结果记录、共同协议、统计分析、证据门禁和完整结果包重建。

正式检测接口只允许读取待检图像、密钥和公开模型配置, 不允许读取生成 Prompt、源 latent、生成轨迹或样本级 Null Space。attention capture、latent injection 和 aligned rescoring 入口只用于独立诊断, 不属于 SLM-WM 正式结果链。

各 Notebook 的前后依赖、并行要求和 GPU / CPU 需求见 `paper_workflow/notebooks/README.md`。

## 必需检查

在提交仓库修改前, 应运行以下命令:

```bash
pytest -q
python tools/harness/run_all_audits.py
```

`tools/harness/inspect_repository.py` 可用于额外检查仓库结构, 但不能替代完整 harness 审计。

## Git 与输出约束

- Git 提交信息必须使用中文。
- 生成的 `outputs/` 内容默认不提交。
- harness 审计报告必须写入 `outputs/audit_reports/`。
- 除 `docs/` 下的人类可读规划文件外, 路径、代码、配置、测试、脚本、skill 和根目录说明不得使用过程标记词, 必须使用表达职责、机制或协议角色的语义名称。
