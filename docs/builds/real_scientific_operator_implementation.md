# 真实科学算子实现与论文证据闭合

## 一、文件级职责

| 机制 | 正式实现 | 作用 |
| --- | --- | --- |
| 分支风险场 | `main/methods/semantic/branch_risk.py` | 分别构造 LF、尾部截断和注意力几何风险与承载预算 |
| 真实 Jacobian Null Space | `main/methods/subspace/jacobian_nullspace.py` | 通过 autograd JVP、联合响应矩阵和 SVD 求解 latent 低响应基底 |
| LF 与尾部载体 | `main/methods/carrier/keyed_tensor.py` | 构造检测端可重建的固定模板, 并在嵌入端投影到安全子空间 |
| 真实注意力梯度 | `main/methods/geometry/differentiable_attention.py` | 从 Transformer `to_q`/`to_k` 得到真实 attention, 对 latent 求梯度 |
| 几何恢复 | `main/methods/geometry/attention_alignment.py` | 执行密钥关系匹配、三点 RANSAC 和仿射估计 |
| 仅图像检测 | `main/methods/detection/image_only.py` | 只接收图像、密钥和公开模型配置, 完成内容主判与同阈值救回 |
| 真实模型运行 | `experiments/runners/semantic_watermark_runtime.py` | 在 SD3/SD3.5 采样过程中执行全部真实嵌入算子 |
| 数据集协议 | `experiments/runners/image_only_dataset_runtime.py` | 运行 Prompt 数据集、冻结完整 evidence 协议并生成 test 记录 |
| 正式消融 | `experiments/ablations/runtime_rerun.py` | 对每个机制配置重新生成、重新攻击和重新检测 |
| 正式 FID/KID | `experiments/artifacts/dataset_level_quality_outputs.py` | 使用 torch-fidelity 0.4.0 的 TensorFlow 兼容 Inception v3 2048 维特征生成可审计质量记录 |
| Colab 续跑 | `paper_workflow/notebooks/semantic_watermark_image_only_run.ipynb` | 在 Drive 持久化工作区分批运行主方法、质量评估与正式消融 |

## 二、检测访问边界

正式检测接口为：

```python
detect_image_only_watermark(
    image,
    key_material,
    config,
    image_latent_encoder,
    image_attention_extractor,
    image_aligner,
)
```

该接口没有原始 latent、生成轨迹、原始图像、prompt 或样本级安全基底参数。
`experiments/runners/aligned_rescoring.py` 中依赖生成轨迹的旧路径只保留为历史诊断,
其结果不得进入论文主方法记录。

## 三、固定模板与安全投影的盲检闭合

检测模板由密钥、公开模型标识和 latent 形状确定。嵌入端求得安全基底 $B$ 后执行：

$$
\bar\nu=BB^\top\nu.
$$

检测端与原始固定模板 $\nu$ 计算相关性, 不需要恢复 $B$。该设计的主要考虑在于：

1. $B$ 只控制嵌入方向对语义和视觉特征的响应；
2. 固定模板提供图像盲检所需的可重建参考；
3. $\nu^\top BB^\top\nu\ge0$, 投影保留的模板能量可以作为运行记录审计；
4. 投影能量过低时运行必须失败, 不能退回 one-hot 或周期平铺代理。

## 四、完整实验链

1. `scripts/run_image_only_dataset_runtime.py` 读取当前 `paper_run` Prompt 文件；
2. 复用一次加载的 SD3/SD3.5、VAE 和 CLIP 运行时；
3. 对每个 Prompt 生成 clean 与 watermarked 图像；
4. 对选定 test Prompt 执行标准图像攻击和真实再扩散攻击；
5. 所有样本只从最终图像重新编码并检测；
6. calibration clean negative 冻结包含 rescue 的完整判定协议；
7. test split 只应用冻结协议并报告置信上界；
8. `scripts/run_runtime_rerun_ablations.py` 重新运行全部机制消融；
9. 结果 records 和 manifest 进入论文共同协议 builder；
10. 生成轨迹检测、proxy 分数和 counterfactual 分数变换不能支持论文主张。

## 五、主张边界

实现存在不等于论文结果成立。下列条件全部满足后, 结果记录才允许进入主张门禁：

1. 运行记录的 `jvp_mode` 为 `torch_func_linearize_exact_jvp`、`torch_autograd_exact_jvp` 或 `torch_autograd_exact_jvp_compatibility`, 且不得使用有限差分；
2. 基底记录包含响应残差、归一化相对响应残差和正交误差, 且相对响应不超过 0.75；
3. attention 来源为真实 Q/K 投影和 autograd；
4. 检测访问模式为 `image_key_public_model_only`；
5. test clean negative 的 95% 误报率上界不超过目标 FPR；
6. FID/KID 使用 torch-fidelity `inception-v3-compat` 的 2048 维特征, 配对质量指标来自真实图像集合；
7. 消融记录明确 `generation_rerun=true` 且未使用 counterfactual 分数变换；
8. 外部 baseline 使用相同 Prompt、攻击和固定 FPR 统计边界。

正式 FID/KID 的样本门禁分别为 70/700/7000 对 clean/watermarked 图像, 与三个运行层级的 Prompt 总数一致。该门禁属于项目特定的证据治理要求; 通用做法是明确记录特征提取器版本、输入图像摘要、特征维度和实际样本数。

质量后端固定为 [torch-fidelity v0.4.0](https://github.com/toshas/torch-fidelity/tree/v0.4.0), 提取器为 `inception-v3-compat`, 特征层为 `2048`。运行记录必须保存 `feature_extractor_id=torch_fidelity_0_4_0_inception_v3_compat_2048`; 只有 `canonical_formal_feature_extractor_ready=true` 的质量摘要才能通过论文记录门禁。普通 torchvision ImageNet 分类权重或像素直方图不能冒充该后端。

为降低 Colab 上精确 JVP 遇到 fused attention 不支持 forward AD 的风险, 正式运行固定 CLIP 视觉编码器使用 eager attention, VAE 使用 Diffusers `AttnProcessor`。该调整只改变等价注意力算子的运行实现, 不更改模型权重或方法目标; 实际配置写入 `scientific_autograd_compatibility` 环境记录。显存不足、形状错误或模型实现错误仍直接失败, 不能被兼容路径吞掉。
