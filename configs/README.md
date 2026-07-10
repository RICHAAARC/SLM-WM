# Configs

此目录保存论文运行配置、prompt 配置和运行环境约束记录。

## 论文 prompt 配置

| 配置文件 | prompt 数量 | 目标 FPR | 支持主张 | 用途 |
| --- | ---: | ---: | --- | --- |
| `paper_main_probe_paper_prompts.txt` | 70 | 0.1 | `probe_claim` | 小规模正式结果包, test 包含34个独立 Prompt。 |
| `paper_main_pilot_paper_prompts.txt` | 700 | 0.01 | `pilot_claim` | 中等规模正式结果包, test 包含340个独立 Prompt。 |
| `paper_main_full_paper_prompts.txt` | 7000 | 0.001 | `full_claim` | 全规模正式结果包, test 包含3400个独立 Prompt。 |

新增 Prompt 来自 Google Research PartiPrompts。固定来源版本、许可证、文件摘要和
分层选择摘要见 `configs/prompt_source_registry.json`。

三组配置只允许样本数量和 fixed-FPR 标准不同; 方法参数、攻击协议、baseline 入口、bootstrap 设置、随机种子和结果闭合逻辑必须保持一致。共同协议结果记录必须拒绝 proxy、placeholder、fallback、synthetic 和 formal-null 证据。

当前统一方法参数位于 `configs/model_sd35.yaml`。其中 `jacobian_candidate_count=12`、`null_space_rank=4`、`maximum_relative_response_residual=0.75` 和 `minimum_projection_energy_retention=0.01` 共同约束语义条件低响应子空间: 前两项定义候选与保留秩, 后两项阻止高响应方向或近零盲检投影进入正式记录。

## prompt 划分口径

当前三类运行层级共用 3:33:34 的固定划分比例, 即 dev、calibration、test
分别约占 4.29%、47.14% 和 48.57%。对应数量为 3/33/34、30/330/340 和
300/3300/3400。该比例确保 calibration negative 足以冻结阈值, 同时让 test
集合在零误报时满足目标 FPR 的单侧 95% 二项分布上界。

## Colab 运行环境约束记录

`colab_sd35_runtime_constraints.txt` 记录一次已验证的 SD3.5 Medium Colab 运行依赖组合。该文件用于远程 Notebook 复现参考, 不属于本地默认安装依赖, 也不应被 `pytest -q` 路径自动安装。
