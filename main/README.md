# Main

`main/` 保存可独立复用的 SLM-WM 方法对象、科学算子、检测逻辑、核心记录结构和分析能力。该目录不负责 Notebook 编排、外部 baseline 运行或大规模结果打包。

## 子目录职责

- `methods/semantic/`: 构造 LF、`tail_robust` 与 attention geometry 三个分支的独立风险场、资格集合和承载预算。
- `methods/subspace/`: 使用精确 JVP、联合语义与视觉响应矩阵、SVD 和残差门禁求解语义条件低响应子空间。
- `methods/carrier/`: 构造空间低通 LF 模板、高斯幅值尾部截断模板及其安全子空间投影。
- `methods/geometry/`: 从真实 Transformer Q/K 计算 attention 关系、目标梯度、单调回溯更新和几何恢复。
- `methods/detection/`: 实现只读取待检图像、密钥和公开模型配置的盲检接口与内容分数。
- `core/`: 保存稳定摘要、manifest、records 和方法对象等通用结构。
- `analysis/`: 保存证据闭合、投稿就绪性和产物来源审计逻辑。
- `protocol/` 与 `cli/`: 保留核心协议和命令行边界。

## 正式方法边界

当前内容分支是空间低通 LF 与高斯幅值尾部截断 `tail_robust`。后者按高斯模板元素绝对幅值选择分布尾部, 不使用空间频率变换。张量正式运行调用 `methods/carrier/keyed_tensor.py`, 轻量原语与内容载体分别使用 `derive_tail_carrier(...)` 和 `methods/carrier/tail.py`。

正式检测不得接收 Prompt、初始噪声、生成轨迹、源 latent 或样本级 Null Space。需要模型运行和数据集协议时, 上层应调用 `experiments/runners/semantic_watermark_runtime.py` 与 `experiments/runners/image_only_dataset_runtime.py`。

## 可复用部分

精确 JVP/SVD 求解器、固定模板到安全子空间的投影、归一化相关分数、attention 关系图和完整 fixed-FPR 冻结原则属于可迁移的通用工程结构。分支风险定义、三分支组合和同阈值几何救回属于本项目的方法特定设计。
