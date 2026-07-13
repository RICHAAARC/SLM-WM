# 当前构建状态

## 文档职责

本文档只记录当前仓库的构建边界、可执行能力和尚未闭合的正式证据。论文结论必须来自 `outputs/` 下受治理的 records、tables、figures、reports 与 manifests, 不能由本文档替代。

## 当前构建单元

- 当前单元: `experiment_protocol_validation`。
- 下一单元: `paper_artifact_rebuild_gate`。
- 当前方法语义层级: `cpu_verified`。权威不变量、追踪关系、核心实现、真实模型绑定和科学内容重建已经完成独立正反例审计；正式完整方法路径未发现剩余 P0 方法偏离或 P1 代理、占位和静默替代。审计证据见 `docs/builds/method_conformance_report.md`。
- 本地与 CPU 服务器负责代码检查、结果包物化、协议校验、统计重建和证据审计。
- SD3.5 生成、精确 Jacobian 响应、Q/K attention 梯度、真实图像攻击、再扩散攻击和 Inception FID/KID 必须在 Colab GPU 上运行。
- 只有 Colab GPU 产物通过同一正式协议门禁后, 才能进入论文结论闭合。

## 正式运行层级

三类运行层级执行同一方法实现、攻击矩阵、baseline 流程、消融流程、检测器和结果闭合命令。差异仅限 Prompt 数量、固定划分后的样本数量和目标 FPR。

三类运行层级还共享同一个3生成 seed × 3水印密钥交叉重复注册表。每次 GPU 执行选择一个重复并隔离输出, 主方法与4个主表 baseline 对同一 Prompt 必须使用相同生成 seed 和实际基础 latent Tensor；最终论文证据需要汇总全部9个重复。该协议已由代码和轻量公式测试冻结, 真实重复结果仍必须由后续 GPU 运行产生, 当前文档不把协议就绪表述为结果已经成立。

| run_name | Prompt 数量 | dev | calibration | test | target FPR | claim scope |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `probe_paper` | 70 | 3 | 33 | 34 | 0.1 | `probe_claim` |
| `pilot_paper` | 700 | 30 | 330 | 340 | 0.01 | `pilot_claim` |
| `full_paper` | 7000 | 300 | 3300 | 3400 | 0.001 | `full_claim` |

每个运行层级必须完整使用自己的 test split。结果导入不得用低于34、340或3400的 positive / negative 统计替代对应层级的正式样本规模。

## 当前方法机制

1. **分支风险场**: 五类信号使用冻结解析范围；三个分支显式消费唯一 YAML 配置, 以严格资格 mask 形成同一 NCHW 有效预算。该 Tensor 同时进入 Null Space 和实际分支硬包络。安全投影保持 float32, 零预算支持清理后对每个实际单位方向独立重跑完整特征精确 JVP；CPU 正反例覆盖两个 epsilon、缩放不变性、退化步长、预算单调性、零支持与批间隔离。
2. **局部安全切空间与 Jacobian Null Space**: “潜流形”严格限定为当前 latent 点处隐式完整特征水平集的局部切空间解释。对512维归一化 CLIP embedding 与204维 RGB 统计/梯度/8x8池化手工结构向量执行 JVP/VJP, 再通过无阻尼风险支持约束投影求解逐列通过残差门禁的方向；实现不构造全局非线性流形或验证常秩定理条件, 204维向量也不单独表示一般感知质量。
3. **空间 LF 内容载体**: 使用具有明确低通含义的空间平滑结构承载内容码。
4. **高斯幅值尾部截断载体**: 只按高斯幅值分布的尾部集合选择鲁棒补充分支, 截断后仅执行 L2 归一化并保持未选坐标精确0, 不把该分支解释为空间高频。
5. **Q/K attention 几何**: 从精确冻结的 `transformer_blocks.0.attn` 与 `transformer_blocks.23.attn` 读取真实 Q/K 状态, 在角点中心映射到 -1 与 1 且图像重采样使用 `align_corners=True` 的统一坐标约定下构造图几何并驱动 attention-relative update。
6. **仅图像盲检**: 检测器只接收最终图像、方法密钥和公开模型, 不读取生成 latent 轨迹。
7. **同阈值几何救援**: 几何分支只能在冻结的同一检测阈值下修正证据, 不得重新调节 operating point。
8. **科学内容身份**: 版本化 Tensor 身份与规范 RGB uint8 像素身份联合绑定风险、预算、资格 mask、Jacobian 八类候选/响应/基底、风险清理后逐分支 JVP、实际量化写回的完整716维参考特征和 JVP Tensor、三分支更新、注入五角色 Q/K、最终三图 Q/K、检测 raw/aligned Q/K、公开噪声和 carrier-only 反事实。最终三图 Q/K 逐角色绑定同一公开噪声 Tensor、PRG 身份及索引0、1、2, 像素-Q/K binding 同时包含噪声身份；检测共享该身份并从3继续。alignment 摘要必须规范。attention 开启时, 写后和打包前都用同一 validator 以完整配置或持久化脱敏配置复验 carrier-only 反事实并拒绝残留 attention 字段。写入端和完成结果加载器从磁盘重建；数据集汇总重算内嵌绑定记录。单元 manifest 必须自包含、路径无重复且配置与结果完全一致, 数据集 manifest 必须无重复并覆盖全部互斥单元叶子；打包器拒绝仅摘要结果、遗漏叶子和内容篡改。该机制只证明包内一致性, 不替代外部来源、GPU 执行和论文结果验证。
9. **构造式更新协议**: LF 与尾部模板分别执行安全投影和风险预算硬包络缩放, 注意力分支直接消费同一风险有界单位方向并从最大允许步长执行单调回溯。attention 候选与最终写回共同调用固定顺序 float32 合成原语并对 original latent 单次 cast, 外层以共同缩放候选复验实际 dtype 预算、JVP、有限特征与 Q/K。项目不声明求解联合标量 `argmax`。

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
- 正式集合固定为完整方法与14个变体, 其中四个变体分别移除中心化 Q/K logit、可微行内 rank、抽样关系概率和距离调制中心化概率；被移除分量权重为0, 其余三个分量各为 $1/3$。
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
| 核心方法算子 | 风险、载体、Q/K、Null Space、唯一合成与磁盘级科学内容重建已有真实实现, 独立方法审计未发现剩余 P0/P1 | 执行 Colab GPU 科学预检 |
| 仅图像盲检 | 已实现数据集运行与 fixed-FPR 记录路径 | 需要按34/340/3400完整运行 |
| 常规与几何攻击 | 已实现真实图像攻击协议 | 需要 Colab GPU 完整攻击矩阵 |
| 再扩散攻击 | 已实现真实后端编排 | 需要 Colab GPU 后端结果 |
| 正式消融 | 已实现独立变体运行与记录重建 | 需要各层级真实变体结果 |
| 数据集质量 | 已实现正式 FID/KID 导入门禁 | 需要 Inception 特征结果 |
| 外部 baseline | 已实现方法忠实主表与官方复现双链 | 需要完整 baseline 结果包 |
| 结果物化与审计 | 已实现单 repeat 的7类随机化包双来源身份核验、自包含原字节封装、精确9组件加3个跨 repeat 不变包的版本化聚合来源构造与独立复验, 并阻断未绑定聚合来源的正式 claim 入口；component 与 aggregate provenance 均固定不可直接支持论文 claim | 尚需由真实 GPU 结果物化9个组件, 再完成45阈值重算、跨重复统计、FID/KID 与消融聚合 |
| 投稿冻结 | 已实现投稿就绪门禁 | 当前不能声明正式论文结果已闭合 |

## 方法构建单元完成判定

1. `configs/method_semantic_registry.json` 登记的全部核心方法不变量具有独立 CPU 正例、反例和跨模块性质测试。
2. 风险硬包络、精确 Jacobian 低响应子空间、真实 Q/K 更新、实际 dtype 合成和仅图像检测形成同一可执行方法闭环。
3. 从冻结方法定义反向执行的独立审计确认不存在代理、占位、静默替代或文档与实现漂移。
4. `pytest -q` 与 `python tools/harness/run_all_audits.py` 全部通过。
5. 方法完成审计已经形成 `docs/builds/method_conformance_report.md`, 当前进入 `experiment_protocol_validation` 处理9重复、fixed-FPR、baseline 公平性和论文结果协议。
