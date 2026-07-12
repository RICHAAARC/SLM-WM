# 当前构建状态

## 文档职责

本文档只记录当前仓库的构建边界、可执行能力和尚未闭合的正式证据。论文结论必须来自 `outputs/` 下受治理的 records、tables、figures、reports 与 manifests, 不能由本文档替代。

## 当前构建单元

- 当前单元: `core_method_runtime_construction`。
- 下一单元: `experiment_protocol_validation`。
- 本地与 CPU 服务器负责代码检查、结果包物化、协议校验、统计重建和证据审计。
- SD3.5 生成、精确 Jacobian 响应、Q/K attention 梯度、真实图像攻击、再扩散攻击和 Inception FID/KID 必须在 Colab GPU 上运行。
- 只有 Colab GPU 产物通过同一正式协议门禁后, 才能进入论文结论闭合。

## 正式运行层级

三类运行层级执行同一方法实现、攻击矩阵、baseline 流程、消融流程、检测器和结果闭合命令。差异仅限 Prompt 数量、固定划分后的样本数量和目标 FPR。

| run_name | Prompt 数量 | dev | calibration | test | target FPR | claim scope |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `probe_paper` | 70 | 3 | 33 | 34 | 0.1 | `probe_claim` |
| `pilot_paper` | 700 | 30 | 330 | 340 | 0.01 | `pilot_claim` |
| `full_paper` | 7000 | 300 | 3300 | 3400 | 0.001 | `full_claim` |

每个运行层级必须完整使用自己的 test split。结果导入不得用低于34、340或3400的 positive / negative 统计替代对应层级的正式样本规模。

## 当前方法机制

1. **分支风险场**: 风险估计作用于各水印分支, 用于约束载体在语义敏感区域的写入强度。
2. **完整特征 Jacobian Null Space**: 对512维归一化 CLIP embedding 与204维 RGB 统计/梯度/8x8池化手工结构向量执行 JVP/VJP, 再通过无阻尼风险支持约束投影求解逐列通过残差门禁的方向；204维向量不单独表示一般感知质量。
3. **空间 LF 内容载体**: 使用具有明确低通含义的空间平滑结构承载内容码。
4. **高斯幅值尾部截断载体**: 按高斯幅值分布的尾部集合选择鲁棒补充分支, 不把该分支解释为空间高频。
5. **Q/K attention 几何**: 从精确冻结的 `transformer_blocks.0.attn` 与 `transformer_blocks.23.attn` 读取真实 Q/K 状态, 在角点中心映射到 -1 与 1 且图像重采样使用 `align_corners=True` 的统一坐标约定下构造图几何并驱动 attention-relative update。
6. **仅图像盲检**: 检测器只接收最终图像、方法密钥和公开模型, 不读取生成 latent 轨迹。
7. **同阈值几何救援**: 几何分支只能在冻结的同一检测阈值下修正证据, 不得重新调节 operating point。
8. **科学内容身份**: 版本化 Tensor 摘要绑定三个分支的风险、预算、资格 mask、Jacobian 候选/响应/基底、三分支更新和量化写回增量；Q/K 原子进一步绑定每个冻结层的抽样 Q、K、中心化 logit、关系概率和二维 token 索引。

## 当前代码分层

依赖方向固定为:

`paper_workflow/ -> scripts/ -> paper_experiments/ -> experiments/ -> main/`

- `main/`: 最小论文方法实现包。
- `experiments/`: 主方法数据协议、GPU runtime、攻击、正式消融和结果 schema。
- `paper_experiments/`: 外部 baseline、公平比较、官方复现、证据审计和投稿门禁。
- `scripts/`: 可脱离 Notebook 执行的 GPU / CPU 服务器命令。
- `paper_workflow/`: Colab 入口、session 配置和 Drive 打包包装。

Notebook 不包含方法算子、攻击实现、baseline 实现、统计逻辑或论文图表构造逻辑。

## 当前实验闭合要求

### 主方法

- 三分支写入必须来自真实 SD3.5 runtime。
- 科学算子门禁必须记录 Jacobian 投影残差、子空间秩、分支能量、attention 捕获真实性、完整关键 Tensor 内容摘要和图像证据摘要。
- clean negative、wrong-key negative、positive source 与 attacked image 必须进入同一仅图像检测协议。
- clean 与 attacked false positive 必须分别记录, 并使用冻结阈值评估。

### 外部 baseline

- Tree-Ring、Gaussian Shading 与 Shallow Diffuse 使用方法忠实 SD3.5 主表适配结果。
- 三种方法的官方原生环境复现用于方法忠实度核对和补充材料。
- T2SMark 使用其 SD3.5 原生正式复现记录进入主表, 不使用其他 observation 补齐缺失攻击项。
- baseline 候选只能绑定一个明确存在的证据文件或结果包。显式指定结果包后, 缺失规范条目必须停止导入。
- 所有方法共享相同 Prompt split、攻击矩阵、目标 FPR 和统计字段。

### 正式消融

- 每个消融变体必须重新运行相应方法路径, 生成独立 detection records 与 manifest。
- 消融表必须从变体 records 重建。
- 仅修改标签、复制 Full 结果或基于单个汇总分数推导变体结果均不属于正式消融。

### 论文证据

- 结果分析图必须引用真实攻击图像。失败案例记录缺少图像时直接停止构图。
- FID/KID 只接受正式 Inception 特征计算结果。
- 共同协议记录拒绝任何非正式证据标记。
- probe、pilot 和 full 使用相同闭合判断; 统计置信强度由各自样本规模和目标 FPR 决定。

## 当前状态表

| 能力 | 代码状态 | 正式 GPU 证据状态 |
| --- | --- | --- |
| 核心方法算子 | 已实现可执行路径与轻量门禁 | 需要 Colab GPU 结果包验证 |
| 仅图像盲检 | 已实现数据集运行与 fixed-FPR 记录路径 | 需要按34/340/3400完整运行 |
| 常规与几何攻击 | 已实现真实图像攻击协议 | 需要 Colab GPU 完整攻击矩阵 |
| 再扩散攻击 | 已实现真实后端编排 | 需要 Colab GPU 后端结果 |
| 正式消融 | 已实现独立变体运行与记录重建 | 需要各层级真实变体结果 |
| 数据集质量 | 已实现正式 FID/KID 导入门禁 | 需要 Inception 特征结果 |
| 外部 baseline | 已实现方法忠实主表与官方复现双链 | 需要完整 baseline 结果包 |
| 结果物化与审计 | 已实现 fail-closed 输入检查与证据重建 | 等待完整上游包后执行 |
| 投稿冻结 | 已实现投稿就绪门禁 | 当前不能声明正式论文结果已闭合 |

## 进入下一构建单元的条件

1. Colab GPU 为当前提交生成主方法、全部攻击、正式消融、FID/KID 和全部 baseline 结果包。
2. 每个结果包包含真实 records、环境报告、manifest、代码版本与配置摘要。
3. `probe_paper` 先以34个 test 样本形成 FPR=0.1 的完整论文闭合结论。
4. `pilot_paper` 与 `full_paper` 使用同一路径分别扩展到340和3400个 test 样本。
5. `pytest -q` 与 `python tools/harness/run_all_audits.py` 全部通过。
6. 论文表、图、报告可从受治理 records 与 manifests 重建, 且不存在未闭合的关键证据缺口。
