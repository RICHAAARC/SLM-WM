# T2SMark 外部 baseline

## 目录职责

- `source/`: 官方源码本地快照, 由 `external_baseline/.gitignore` 排除。
- `adapter/`: 本项目维护的协议 adapter, 用于产生统一 `baseline_observations.json`。

## SD3.5 Medium 适配判断

T2SMark 官方源码包含 SD3.5 Medium 入口。本项目 adapter 负责读取官方或适配运行输出, 转写为统一 observation、图像 digest、攻击矩阵记录和 受治理导入 候选记录。

T2SMark 与 SLM-WM 的空间 LF、高斯幅值尾部截断和 Q/K attention geometry 分支分别保留自身方法定义。adapter 只负责共同协议转写, 不把某个 baseline 的算子解释为 SLM-WM 内部机制。

## 正式运行入口

- `paper_workflow/notebooks/official_reference_t2smark_run.ipynb`: 运行 T2SMark 官方 SD3.5 主表正式路径, 对完整 Prompt split 执行 fixed-FPR 校准与完整攻击矩阵, 输出到 `outputs/t2smark_formal_reproduction/<paper_run_name>/`。

Notebook 只准备 CPU `workflow_orchestrator`; 完整 T2SMark runner 由 repository dispatch 在 `t2smark_sd35_gpu` 隔离子解释器中运行. 当前运行资格由该 profile 在匹配 Colab 或 Linux CUDA 环境完成目标完整哈希锁审查的门禁决定.

`t2smark_formal_reproduction` 是 T2SMark 主表结果的唯一生成路径。runner 按登记 commit 克隆源码, 要求工作树精确等于 `HEAD + formal_protocol_git_diff.txt`, 再验证仅图像 clean/positive 分数、成对质量图像和完整攻击对。已有官方结果只有在论文层级、完整 Prompt 摘要、模型、随机种子、20/20/4.5 预算、攻击集合、源码 commit、补丁摘要、工作树摘要和事实文件 SHA-256 全部匹配时才可被当前运行采用。最终结果必须通过 受治理导入校验器 与完整模板覆盖门禁。

## 当前可用入口

- adapter: `external_baseline/primary/t2smark/adapter/run_slm_eval.py`
- 输出边界: 只能写入 `outputs/` 下的 observation、manifest、候选记录和证据报告。
- 论文主张: 是否进入 `probe_claim`、`pilot_claim` 或 `full_claim` 由 prompt 数量、fixed-FPR 校准、共同攻击矩阵检测和 formal import validator 共同决定。
