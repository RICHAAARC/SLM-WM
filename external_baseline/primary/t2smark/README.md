# T2SMark 外部 baseline

## 目录职责

- `source/`: 官方源码本地快照, 由 `external_baseline/.gitignore` 排除。
- `adapter/`: 本项目维护的协议 adapter, 用于产生统一 `baseline_observations.json`。

## SD3.5 Medium 适配判断

T2SMark 官方源码包含 SD3.5 Medium 入口。本项目 adapter 负责读取官方或适配运行输出, 转写为统一 observation、图像 digest、攻击矩阵记录和 governed import 候选记录。

T2SMark 与 SLM-WM 的空间 LF、高斯幅值尾部截断和 Q/K attention geometry 分支分别保留自身方法定义。adapter 只负责共同协议转写, 不把某个 baseline 的算子解释为 SLM-WM 内部机制。

## method-faithful 与 official reference

- `paper_workflow/notebooks/official_reference_t2smark_run.ipynb`: 运行 T2SMark 官方 SD3.5 主表正式路径, 对完整 Prompt split 执行 fixed-FPR 校准与 `full_main` / `full_extra` 攻击矩阵, 输出到 `outputs/t2smark_formal_reproduction/`。
- `paper_workflow/notebooks/external_baseline_t2smark_run.ipynb`: 在多 baseline 批量调度中运行同一 T2SMark 官方 SD3.5 适配链路, 用于批量运行诊断与证据镜像。

T2SMark 原生支持 SD3.5 Medium, 因此 `t2smark_formal_reproduction` 是其主表结果的权威生成路径。批量调度输出不得覆盖或合并到该正式候选记录; 最终结果必须通过 governed import validator 与完整模板覆盖门禁。

## 当前可用入口

- adapter: `external_baseline/primary/t2smark/adapter/run_slm_eval.py`
- 输出边界: 只能写入 `outputs/` 下的 observation、manifest、候选记录和证据报告。
- 论文主张: 是否进入 `probe_claim`、`pilot_claim` 或 `full_claim` 由 prompt 数量、fixed-FPR 校准、共同攻击矩阵检测和 formal import validator 共同决定。
