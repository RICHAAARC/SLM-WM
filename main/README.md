# Main

`main/` 是 SLM-WM 的最小论文方法实现包。该目录只保存方法定义和方法运行所需的最小数学工具, 不保存实验编排、结果治理、baseline、命令行入口或 Notebook 逻辑。

## 目录职责

- `methods/semantic/`: 构造 LF、`tail_robust` 与 attention geometry 三个分支的独立风险场、资格集合和连续承载预算。
- `methods/subspace/`: 使用716维完整特征 JVP/VJP、显式风险算子、无阻尼 PSD-CG 和逐列残差门禁求解 Jacobian Null Space。
- `methods/carrier/`: 构造空间低通 LF 模板、高斯幅值尾部截断模板及其安全子空间投影。
- `methods/geometry/`: 从真实 Transformer Q/K 直接构造中心化 logit、可微 rank、抽样图像 token 关系概率和距离调制中心化概率四分量图, 计算目标梯度, 构造可核对身份的稳定 token pair 权重, 并通过攻击配置无关的分层搜索恢复二维参考系。
- `methods/detection/`: 实现只读取待检图像、密钥和公开检测配置的盲检接口, 注册前后使用同一 pair 权重身份且不在对齐后重新选择 token。
- `core/digest.py`: 提供方法记录所需的稳定摘要函数。

## 分层边界

- 数据划分、fixed-FPR、攻击、正式消融和主方法运行位于 `experiments/`。
- baseline、公平对比、论文证据审计和投稿就绪分析位于 `paper_experiments/`。
- 独立服务器命令位于 `scripts/`。
- Colab 包装和 Notebook 位于 `paper_workflow/`。

`main/` 不得导入上述任何外层目录。

## 方法语义

内容分支由空间低通 LF 与高斯幅值尾部截断 `tail_robust` 组成。`tail_robust` 按高斯模板元素绝对幅值选择分布尾部, 不使用 FFT、DCT、带通滤波或空间频带掩码。

正式检测不得接收 Prompt、生成种子、初始噪声、生成轨迹、源 latent 或样本级 Null Space。检测端可使用的全部公开随机量必须由密钥、模型标识和冻结检测协议确定。
