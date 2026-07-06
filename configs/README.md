# Configs

此目录保存论文实验配置模板。

## 论文 prompt 配置

| 配置文件 | prompt 数量 | 用途 |
| --- | ---: | --- |
| `paper_main_probe_prompts.txt` | 10 | 仅用于诊断入口, 不进入正式统计。 |
| `paper_main_probe_paper_prompts.txt` | 60 | 使用完整论文 workflow 的小规模服务器流程对齐验证。 |
| `paper_main_pilot_paper_prompts.txt` | 600 | pilot 论文结果运行。 |
| `paper_main_full_paper_prompts.txt` | 6000 | 完整论文结果运行。 |

## Colab 运行环境约束记录

`colab_sd35_runtime_constraints.txt` 记录一次已验证的 SD3.5 Medium Colab 运行依赖组合。该文件用于远程 Notebook 复现参考, 不属于本地默认安装依赖, 也不应被 `pytest -q` 路径自动安装。
