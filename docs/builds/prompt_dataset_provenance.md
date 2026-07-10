# Prompt 数据来源与 70/700/7000 划分

## 一、来源

新增 Prompt 来自 Google Research 发布的 PartiPrompts：

- 项目地址：`https://github.com/google-research/parti`；
- 固定 revision：`5a657978134374ce28973948331b319adef164bd`；
- 原始文件：`PartiPrompts.tsv`；
- 原始记录数：1632；
- SHA-256：`66b9d693fe1231fe2a9e7541c85a2a5814317a92f9fa183b770653b78e328f82`；
- 许可证：Apache-2.0。

机器可读登记见 `configs/prompt_source_registry.json`。

## 二、补充数量

| Prompt 集 | 原数量 | 新增 | 当前数量 |
| --- | ---: | ---: | ---: |
| probe_paper | 60 | 10 | 70 |
| pilot_paper | 600 | 100 | 700 |
| full_paper | 6000 | 1000 | 7000 |

选择过程先使用小写和空白归一化执行去重, 再按 PartiPrompts Category 比例分配
名额。每个 Category 内使用固定 SHA-256 排序选择。三个 Prompt 集新增部分互不
复用, 并且不与原有三个 Prompt 文件中的任何规范化文本重复。由于 Prompt 文件
属于 `configs/` 受治理输入, 选择前还会排除仓库命名规范保留的过程标记词; 该
排除规则已写入机器可读来源登记, 不会在生成结果后临时改写 Prompt 文本。

## 三、固定划分

三级规模共享 3:33:34 比例：

| Prompt 数量 | dev | calibration | test |
| ---: | ---: | ---: | ---: |
| 70 | 3 | 33 | 34 |
| 700 | 30 | 330 | 340 |
| 7000 | 300 | 3300 | 3400 |

Prompt 先按 Prompt 集与风险类别分层, 再按稳定摘要排序分配。calibration 与 test
的 Prompt ID 完全不相交。test 数量分别满足在零误报时验证 0.1、0.01 和 0.001
目标 FPR 所需的单侧 95% 二项分布上界。
