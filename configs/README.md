# Configs

此目录保存论文实验配置模板。

## 论文 prompt 配置

| 配置文件 | prompt 数量 | 用途 |
| --- | ---: | --- |
| `paper_main_probe_paper_prompts.txt` | 60 | 小规模正式论文流程结果包, 固定 FPR 为 0.1, 支持 `probe_claim`。 |
| `paper_main_pilot_paper_prompts.txt` | 600 | 中等规模正式论文流程结果包, 固定 FPR 为 0.01, 支持 `pilot_claim`。 |
| `paper_main_full_paper_prompts.txt` | 6000 | 全规模正式论文流程结果包, 固定 FPR 为 0.001, 支持 `full_claim`。 |

三组配置只允许样本数量和 fixed-FPR 标准不同; 共同协议结果记录必须拒绝 proxy、placeholder、fallback、synthetic 和 formal-null 证据。

## Colab 运行环境约束记录

`colab_sd35_runtime_constraints.txt` 记录一次已验证的 SD3.5 Medium Colab 运行依赖组合。该文件用于远程 Notebook 复现参考, 不属于本地默认安装依赖, 也不应被 `pytest -q` 路径自动安装。
