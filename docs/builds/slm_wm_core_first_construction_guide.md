# SLM-WM 核心优先构建指南

## 1. 文档职责

本文档说明如何从最小论文方法实现逐层构建到完整论文实验。方法定义以同目录的算法原语、技术路线和方法章节文档为准; 本文只描述代码分层、执行顺序、正式证据要求和验收方式。

## 2. 构建原则

1. 先实现可独立测试的科学算子, 再接入 SD3.5 runtime。
2. `main/` 只保存论文核心方法, 不保存实验协议、baseline、CLI 或 Notebook 逻辑。
3. 正式论文运行只接受真实模型、真实图像、真实攻击和仅图像盲检结果。
4. probe、pilot 与 full 执行同一代码路径和 FPR=0.1 工作点, 仅统计规模与统计强度不同。
5. records 是结果事实来源, tables、figures 与 reports 必须可重建。
6. 缺少正式输入时立即停止当前闭合步骤, 不生成替代指标或空白论文图。

## 3. 五层代码结构

依赖方向固定为:

`paper_workflow/ -> scripts/ -> paper_experiments/ -> experiments/ -> main/`

### 3.1 核心方法层 `main/`

| 子目录 | 职责 |
| --- | --- |
| `main/core/` | JSON 稳定摘要、Tensor 内容摘要和密钥 PRG 等最小数学与数据工具 |
| `main/methods/semantic/` | 分支风险场与语义条件向量 |
| `main/methods/subspace/` | 完整特征 Jacobian Null Space |
| `main/methods/carrier/` | 空间 LF、高斯幅值尾部截断与密钥张量载体 |
| `main/methods/geometry/` | 真实 Q/K attention 几何和可微更新 |
| `main/methods/detection/` | 最终图像盲检与同阈值证据判定 |

该层不得依赖 `experiments/`、`paper_experiments/`、`scripts/`、`paper_workflow/` 或 `tools/`。

### 3.2 主方法实验层 `experiments/`

| 子目录 | 职责 |
| --- | --- |
| `experiments/protocol/` | Prompt、split、fixed-FPR、攻击配置与结果 schema |
| `experiments/runtime/` | SD3.5 pipeline、语义特征、真实图像攻击和运行环境 |
| `experiments/runners/` | 主方法生成、攻击和仅图像检测数据集运行 |
| `experiments/ablations/` | 改变真实机制后重新运行的正式消融 |
| `experiments/artifacts/` | manifest 与正式质量产物 |

### 3.3 完整论文实验层 `paper_experiments/`

该层负责外部 baseline 的方法忠实实现、官方复现、受治理导入、公平对比、论文证据审计和投稿门禁。外部 baseline 不得进入 `main/`。

### 3.4 独立执行层 `scripts/`

脚本必须能在 GPU 服务器或 CPU 汇总服务器直接运行。Notebook 只能调用这些脚本或其下层 runner, 不能成为唯一执行入口。

### 3.5 Colab 运行层 `paper_workflow/`

该层只负责 Colab session 配置、Drive 路径、进度显示、Notebook runtime 报告和打包入口。Notebook 不写正式 records、阈值、表格或论文图。

## 4. 核心方法实现顺序

### 4.1 分支风险场

1. 从语义条件响应计算 LF、`tail_robust` 与 attention 三个分支的风险。
2. 保留分支间独立风险, 不把单一位置风险复制到全部载体。
3. 输出归一化风险、分支权重和稳定摘要。
4. 为每个分支的完整风险值、连续预算和资格 mask 保存版本化 Tensor 内容 SHA-256。

### 4.2 完整特征 Jacobian Null Space

1. 在真实 SD3.5 / CLIP 语义响应上构造精确 JVP 或等价 Jacobian 线性算子。
2. 对候选方向执行奇异值分解或等价低响应求解。
3. 记录候选秩、保留秩、响应残差和投影能量保留率。
4. 保存实际候选矩阵、风险预算、响应矩阵和最终基底的版本化 Tensor 内容 SHA-256。
5. 残差或能量门禁失败时停止该样本写入。

### 4.3 内容载体

- 空间 LF 分支使用明确的空间低通构造。
- `tail_robust` 分支按高斯模板绝对幅值的尾部集合截断。
- 两个分支均按密钥生成符号与置换, 再投影到逐列通过完整 Jacobian 残差门禁的 Null Space。
- `tail_robust` 不执行 FFT、DCT、带通滤波或空间频带 mask。

### 4.4 Q/K attention 几何

1. 精确解析 `transformer_blocks.0.attn` 与 `transformer_blocks.23.attn`, 从模块 `to_q`、`to_k` 读取真实投影, 在角点 token 中心映射到 -1 与 1 的冻结二维抽样集合上同时保存中心化 logits 与关系概率；token 插值和图像仿射重采样统一使用 `align_corners=True`。
2. 构造中心化 logit、温度0.25的可微 rank、抽样关系概率和概率偏离与距离偏离的双中心交互四分量图, 各分量逐行加权归一化后通过密钥符号投影按冻结协议组合；完整方法为四项等权, 留一变体为一项0和其余三项各 $1/3$。
3. 由跨层稳定度和 attention 显著度选择 token, 构造可跨阶段核对身份的 pair 权重。
4. 对 latent 求真实目标梯度, 并投影到安全方向。
5. 使用单调回溯验证更新后 Q/K 目标确实提升。
6. 以同 seed、同 scheduler 和相同 LF/tail 配置及算子生成只关闭 attention geometry 的 carrier-only 成图；该比较测量包含后续轨迹交互的总机制效应, 不假设干预后 realized carrier 相等。
7. 核验首个注入前 latent 字节身份、完整注入顺序和 scheduler 轨迹；逐条拒绝含 attention 来源、分数、更新、关系、pair 身份或 attention Null Space 的 carrier-only 原子。
8. 持久化 carrier-only 更新原子 JSONL, 将路径、实际文件 SHA-256 和解析内容摘要绑定到结果、manifest 与缓存复验。
9. 验证 clean 到完整方法、clean 到 carrier-only 及 carrier-only 到完整方法三条最终 CLIP 语义和204维手工结构统计边。
10. 对 clean、carrier-only 与完整方法成图重新编码真实 Q/K, 验证自身盲选择归因增益和冻结 carrier-only pair 权重归因增益, 并记录全部四个 Q/K 依赖分量的配对增益。
11. 按原 latent、内容基底、接受候选和实际写回四个角色保存逐层 Q/K 原子；每层绑定抽样 Q、K、中心化 logit、关系概率和二维 token 索引。
12. 记录更新强度、梯度范数、回溯次数、前后目标值、直接 Q/K 来源、四分量与密钥投影身份、反事实身份和 pair 权重身份, 并把三分支更新、Q/K 原子、反事实原子、身份与图像摘要写入 manifest。

### 4.5 仅图像盲检

1. 检测输入只包括最终图像、方法密钥与公开模型。
2. 通过公开 VAE / 视觉编码器恢复检测特征。
3. 计算内容分支分数与 attention 几何证据。
4. 仿射注册使用攻击配置无关的分层搜索, 并把同一 pair 权重从观测网格传递到规范网格。
5. 对齐图像重新提取 Q/K 后使用传递权重计算同步分数, 不重新选择 token。
6. 保存原图与对齐图两个 Q/K 原子角色, 并复验冻结层顺序和联合摘要。
7. 阈值只在 calibration clean negative 上冻结。
8. test split 使用同一阈值评估 positive、clean negative、wrong-key negative 和 attacked image。
9. fixed-FPR 门禁使用95%单侧 Wilson 上界, 通用有界指标表使用 Hoeffding 区间。

## 5. 正式运行层级

| run_name | Prompt | dev | calibration | test | target FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| `probe_paper` | 70 | 3 | 33 | 34 | 0.1 |
| `pilot_paper` | 700 | 30 | 330 | 340 | 0.1 |
| `full_paper` | 7000 | 300 | 3300 | 3400 | 0.1 |

所有层级必须完整执行自己的 test split。probe 的职责是用最小正式规模验证完整论文链路, 不是执行不同机制或较弱检测制度。

## 6. Colab GPU 执行顺序

1. 配置当前 `SLM_WM_PAPER_RUN_NAME` 和一个权威 `SLM_WM_RANDOMIZATION_REPEAT_ID`。
2. 运行 `paper_workflow/notebooks/semantic_watermark_image_only_run.ipynb` 生成当前 repeat 的主方法、真实攻击、正式消融和数据集质量结果。
3. 运行 Tree-Ring、Gaussian Shading、Shallow Diffuse 三个 common-backbone Notebook 与 T2SMark 正式复现 Notebook, 全部使用同一 repeat ID。
4. 运行 `paper_workflow/notebooks/randomization_repeat_evidence_run.ipynb`, 把当前 repeat 的7类活动随机化 leaf 封装为自包含证据组件。
5. 对权威9个 repeat 分别重复第1至4步; 每个 repeat 使用独立 Drive 子目录。
6. Tree-Ring、Gaussian Shading 与 Shallow Diffuse 的跨 repeat 不变官方原环境复现各运行一次, 留待最终聚合层选择。
7. 下载9个单 repeat 组件和3个不变包到 CPU 汇总环境; 不在 Notebook 内执行跨重复统计或论文产物重建。

Colab 中断后允许从同一配置、同一代码版本和通过摘要核对的当前结果继续; 任何复用记录仍必须通过正式 schema 与证据路径校验。

## 7. CPU 汇总服务器执行顺序

1. 以规范顺序显式传入9个单 repeat 组件和3个不变包, 调用 `python -m paper_experiments.runners.randomization_aggregate_provenance` 构造自包含聚合来源包。
2. 使用 `paper_experiments.runners.randomization_aggregate_record_workspace` 从不可变来源对象建立临时读取边界, 重新复验全部12个输入的原始字节、9个组件内嵌的63个活动 leaf、3个不变包、代码版本、执行锁和随机化协议；不接受目录扫描推断、单 repeat 锁、外部成员路径或调用方 ready 字段。
3. 先由 `paper_experiments.runners.randomization_prompt_source_contract` 从9个主方法 leaf 内嵌的来源注册表、冻结选择清单和当前层级 Prompt 文件重建同一个 Prompt exact-set, 再将不可变聚合来源对象传给 `paper_experiments.runners.randomization_method_repeat_thresholds`。桥接层逐条核对9份 runtime records 与包内 Prompt 字节、精确配置 schema、完整正式方法配置、模型、seed 和运行前预注册 key plan, 在 CPU 上重新生成规范基础 latent, 最后调用 `paper_experiments.analysis.method_repeat_fixed_fpr` 从45组原始 observation 分别重算45个方法重复阈值。每个阈值只消费所属 repeat 的 calibration clean negatives, 并核对对应 producer 阈值声明、五方法共同模型、seed、key 与基础 latent 身份；不得先合并9个 repeat 后只计算5个阈值。
4. 将同一个不可变聚合来源对象传给 `paper_experiments.runners.randomization_paired_superiority`。该入口逐 repeat 使用对应5个阈值构造主方法与4个 baseline 的真实攻击后配对 outcome, 并核验仅图像盲检、真实攻击图像、攻击 seed、生成 seed、密钥和基础 latent 身份；同时从相同原始来源提取未攻击 clean-watermarked 实测 SSIM。对固定 baseline 和 Prompt, 只有9个注册 repeat 的绝对 SSIM 差都不超过0.02时, 才把完整 `9 x attack` 检测块纳入质量匹配子集。全样本与质量匹配均先在 Prompt 内平均完整攻击和9重复, 再以34 / 340 / 3400个 test Prompt 为独立单位执行 bootstrap 与 Hoeffding 推断；两类比较分别使用固定4项 Holm 家族, 不得把9重复展开为9倍独立样本。
5. 将同一个不可变聚合来源对象传给 `paper_experiments.runners.randomization_dataset_quality`, 从9份完整 Prompt 质量记录的原始 Inception 特征联合重算一次 FID/KID。正式样本对数量固定为 `9 x 70 / 700 / 7000`, 不得平均单 repeat FID/KID, 也不得把派生指标表当作原始统计输入。
6. 将同一个不可变聚合来源对象传给 `paper_experiments.runners.randomization_ablation_necessity`, 从9份真实重运行消融原始记录重建机制必要性统计；不得读取或平均单重复派生表。
7. 构造 baseline 正式候选并执行受治理导入, 同时传播唯一 `randomization_aggregate_digest`。
8. 写出主方法与 baseline 的共同协议结果记录, 重建 fixed-FPR 表、优势表与真实失败案例图。
9. 执行证据审计、投稿就绪审计、结果闭合语义门禁和完整结果打包。

CPU 服务器不承担 SD3.5 GPU 推理, 也不能以代码检查结果替代图像实验。

## 8. 正式消融要求

正式集合固定为完整方法和14个变体：共享全局风险、移除风险路由、移除 Jacobian Null Space、LF-only、Tail-only、移除 LF、移除尾部载体、移除幅值截断、四个 Q/K 分量逐项留一、移除完整 Q/K attention geometry 以及移除图像对齐。

每个变体必须重新生成图像、重新攻击、重新盲检并使用自己的 calibration split 冻结完整判定协议。四个分量留一变体必须在嵌入、注册与检测全链应用同一权重摘要；消融结果不能从完整方法分数复制或变换得到。

## 9. 外部 baseline 要求

- Tree-Ring、Gaussian Shading、Shallow Diffuse: 方法忠实 SD3.5 主表适配 + 官方原生环境核对。
- T2SMark: SD3.5 原生正式复现结果直接进入主表。
- 每个 baseline 必须覆盖完整 Prompt split 与攻击矩阵。
- 所有方法使用同一目标 FPR、攻击参数和统计字段。
- 缺失攻击项不能由另一种实现路径补齐。

## 10. 论文证据门禁

1. 每条 supported claim 必须映射到受治理记录。
2. 数据集质量只接受正式 Inception FID/KID。
3. 失败案例图必须渲染真实攻击图像。
4. manifest 必须记录输入、输出、配置摘要、代码版本和重建命令。
5. 投稿就绪要求完整方法 × 攻击模板覆盖, 不能只依赖部分成功记录。

## 11. 本地验收

```bash
pytest -q
python tools/harness/run_all_audits.py
```

本地验收只证明代码、轻量算法和治理边界一致。正式论文有效性仍由 Colab GPU 结果包决定。
