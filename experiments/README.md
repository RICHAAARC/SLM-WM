# Experiments

`experiments/` 是不依赖 Notebook 的实验实现层, 依赖方向为 `experiments -> main`。

## 当前职责

- `protocol/`: 论文级 Prompt 分层、fixed-FPR、共同攻击、正式证据和运行规模配置。
- `runtime/`: SD3.5 模型加载、真实图像攻击、语义特征、进度与仓库环境工具。
- `runners/semantic_watermark_runtime.py`: 完整执行分支风险、16维语义与视觉条件 Jacobian、20候选方向的4维 Null Space、LF、Gaussian 幅值尾部截断、真实多层 Q/K 几何更新和仅图像盲检。
- `runners/image_only_dataset_runtime.py`: 运行完整 Prompt 集, 在 calibration split 独立冻结检测协议, 只在 test split 形成论文统计。
- `ablations/runtime_rerun.py`: 每个消融配置重新生成、攻击、检测并独立校准, 不读取或变换完整方法分数。
- `artifacts/dataset_level_quality_outputs.py`: 从真实图像对提取正式 Inception 特征并构建 FID / KID 质量证据。
- `artifacts/`: 只保留通用 manifest schema 和正式质量产物构建器。

## 正式方法边界

内容载体为 `lf_content` 与 `tail_robust`。`tail_robust` 仅按 Gaussian 元素绝对幅值执行分位点尾部截断, 不具有空间频带含义。注意力稳定度来自至少两个真实 Q/K 层对应关系行的余弦一致性。

所有持久化输出必须写入 `outputs/`。本层不得导入 `paper_experiments/`、`scripts/` 或 `paper_workflow/`。
