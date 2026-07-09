# Configs

此目录保存论文运行配置、prompt 配置和运行环境约束记录。

## 论文 prompt 配置

| 配置文件 | prompt 数量 | 目标 FPR | 支持主张 | 用途 |
| --- | ---: | ---: | --- | --- |
| `paper_main_probe_paper_prompts.txt` | 60 | 0.1 | `probe_claim` | 小规模正式结果包, 用于验证完整论文流程在真实环境中可闭合。 |
| `paper_main_pilot_paper_prompts.txt` | 600 | 0.01 | `pilot_claim` | 中等规模正式结果包, 用于 fixed-FPR=0.01 的论文证据判断。 |
| `paper_main_full_paper_prompts.txt` | 6000 | 0.001 | `full_claim` | 全规模正式结果包, 用于 fixed-FPR=0.001 的最终论文主张。 |

三组配置只允许样本数量和 fixed-FPR 标准不同; 方法参数、攻击协议、baseline 入口、bootstrap 设置、随机种子和结果闭合逻辑必须保持一致。共同协议结果记录必须拒绝 proxy、placeholder、fallback、synthetic 和 formal-null 证据。

## prompt 划分口径

当前三类运行层级共用同一划分比例: dev 5%、calibration 55%、test 40%。该比例用于降低 calibration negative 数量不足风险, 并保证 fixed-FPR 阈值校准与 test 统计互不混用。

## Colab 运行环境约束记录

`colab_sd35_runtime_constraints.txt` 记录一次已验证的 SD3.5 Medium Colab 运行依赖组合。该文件用于远程 Notebook 复现参考, 不属于本地默认安装依赖, 也不应被 `pytest -q` 路径自动安装。
